/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#include "yttrium_context.h"

#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <winternl.h>

#include "util/u_prim.h"
#include "util/u_framebuffer.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_upload_mgr.h"

#include "yttrium_draw.h"
#include "yttrium_gdi_public.h"
#include "yttrium_internal.h"
#include "yttrium_options.h"
#include "yttrium_pipeline.h"
#include "yttrium_resource.h"
#include "yttrium_shader.h"
#include "yttrium_state.h"
#include "yttrium_trace.h"
#include "yttrium_venus.h"

void noop_init_state_functions(struct pipe_context *ctx);

enum yttrium_flush_issuance_state {
   YTTRIUM_FLUSH_ISSUANCE_PENDING = 0,
   YTTRIUM_FLUSH_ISSUANCE_ISSUED = 1,
   YTTRIUM_FLUSH_ISSUANCE_FAILED = -1,
};

struct yttrium_gdi_flush_issuance {
   volatile LONG references;
   volatile LONG state;
   uint64_t id;
};

struct yttrium_gdi_present_ticket {
   uint64_t id;
   /* Set once at creation, before the worker can observe this ticket. */
   struct yttrium_gdi_present_publish_request publish;
};

static volatile LONG64 yttrium_next_flush_issuance_id;
static volatile LONG64 yttrium_next_present_ticket_id;

static void
yttrium_flush_issuance_release_internal(
   struct yttrium_gdi_flush_issuance *issuance)
{
   if (issuance && InterlockedDecrement(&issuance->references) == 0)
      FREE(issuance);
}

static void
yttrium_complete_flush_issuance(struct yttrium_context *yctx,
                                bool issued,
                                const char *label)
{
   struct yttrium_gdi_flush_issuance *issuance =
      yctx ? yctx->pending_flush_issuance : NULL;
   if (!issuance)
      return;

   yctx->pending_flush_issuance = NULL;
   YTTRIUM_LOG("yttrium: present publication worker result id=%llu issued=%u label=%s\n",
               (unsigned long long)issuance->id, issued ? 1 : 0,
               label && label[0] ? label : "<missing-pipe-flush-owner>");
   InterlockedExchange(&issuance->state,
                       issued ? YTTRIUM_FLUSH_ISSUANCE_ISSUED :
                                YTTRIUM_FLUSH_ISSUANCE_FAILED);
   WakeByAddressAll((PVOID)&issuance->state);
   yttrium_flush_issuance_release_internal(issuance);
}

static void
yttrium_complete_present_ticket(struct yttrium_context *yctx,
                                bool accepted,
                                const char *label)
{
   struct yttrium_gdi_present_ticket *ticket =
      yctx ? yctx->pending_present_ticket : NULL;
   if (!ticket)
      return;

   yctx->pending_present_ticket = NULL;
   YTTRIUM_LOG("yttrium: fullscreen present worker publication id=%llu accepted=%u label=%s\n",
               (unsigned long long)ticket->id,
               accepted ? 1 : 0,
               label && label[0] ? label : "<missing-pipe-flush-owner>");
   FREE(ticket);
}

static enum pipe_reset_status
yttrium_get_device_reset_status(struct pipe_context *ctx)
{
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   return gdikmt_device_get_reset_status(screen->device);
}

static void
yttrium_set_device_reset_callback(
   struct pipe_context *ctx,
   const struct pipe_device_reset_callback *callback)
{
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   gdikmt_device_set_reset_callback(screen->device, callback);
}

struct yttrium_threaded_flush_label {
   struct yttrium_context *yctx;
   struct yttrium_gdi_present_ticket *ticket;
   struct yttrium_gdi_flush_issuance *issuance;
   char label[96];
};

static void
yttrium_threaded_set_flush_label(void *data)
{
   struct yttrium_threaded_flush_label *queued = data;

   memcpy(queued->yctx->pending_flush_label, queued->label,
          sizeof(queued->yctx->pending_flush_label));
   queued->yctx->pending_flush_label[
      sizeof(queued->yctx->pending_flush_label) - 1] = '\0';
   queued->yctx->pending_flush_label_valid = true;
   if (queued->ticket) {
      if (queued->yctx->pending_present_ticket) {
         YTTRIUM_WARN("yttrium: ERROR: present publication rejected owner=yttrium-context reason=overlapping-worker-ticket old_id=%llu new_id=%llu\n",
                      (unsigned long long)
                         queued->yctx->pending_present_ticket->id,
                      (unsigned long long)queued->ticket->id);
         yttrium_complete_present_ticket(
            queued->yctx, false, "overlapping present publication");
      }
      queued->yctx->pending_present_ticket = queued->ticket;
   }
   if (queued->issuance) {
      if (queued->yctx->pending_flush_issuance) {
         YTTRIUM_WARN("yttrium: ERROR: present publication rejected owner=yttrium-context reason=overlapping-worker-issuance old_id=%llu new_id=%llu\n",
                      (unsigned long long)
                         queued->yctx->pending_flush_issuance->id,
                      (unsigned long long)queued->issuance->id);
         yttrium_complete_flush_issuance(
            queued->yctx, false, "overlapping present issuance");
      }
      queued->yctx->pending_flush_issuance = queued->issuance;
   }
   FREE(queued);
}

static bool
yttrium_threaded_queue_flush_label(struct pipe_context *ctx,
                                   const char *label,
                                   struct yttrium_gdi_present_ticket *ticket,
                                   struct yttrium_gdi_flush_issuance *issuance)
{
   if (!ctx->callback)
      return false;

   struct threaded_context *tc = threaded_context(ctx);
   struct pipe_context *driver = tc->pipe;
   if (!driver || driver->flush != yttrium_flush)
      return false;

   struct yttrium_threaded_flush_label *queued =
      CALLOC_STRUCT(yttrium_threaded_flush_label);
   if (!queued)
      return false;

   queued->yctx = yttrium_context(driver);
   queued->ticket = ticket;
   queued->issuance = issuance;
   strncpy(queued->label,
           label && label[0] ? label : "<missing-pipe-flush-owner>",
           sizeof(queued->label) - 1);
   ctx->callback(ctx, yttrium_threaded_set_flush_label, queued, false);
   return true;
}

static struct pipe_fence_handle *
yttrium_threaded_create_fence(struct pipe_context *ctx,
                              struct tc_unflushed_batch_token *token)
{
   (void)ctx;
   (void)token;

   /*
    * A NULL result makes u_threaded_context use its synchronous fallback when
    * a real Gallium fence is requested.  Fence-less PIPE_FLUSH_ASYNC calls,
    * including the muted-Present experiment, remain queued and asynchronous.
    */
   return NULL;
}

static struct pipe_stream_output_target *
yttrium_create_stream_output_target(struct pipe_context *ctx,
                                    struct pipe_resource *resource,
                                    unsigned buffer_offset,
                                    unsigned buffer_size)
{
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   struct yttrium_stream_output_target *ytarget =
      CALLOC_STRUCT(yttrium_stream_output_target);
   struct pipe_stream_output_target *target =
      ytarget ? &ytarget->base : NULL;
   if (!target)
      return NULL;

   if (!yttrium_venus_create_stream_output_buffer(screen->venus,
                                                  &ytarget->counter, 4,
                                                  NULL)) {
      YTTRIUM_WARN("yttrium: stream-output target creation failed "
                   "owner=yttrium-context reason=counter-buffer-unavailable "
                   "resource=%p offset=%u size=%u\n",
                   (void *)resource, buffer_offset, buffer_size);
      FREE(ytarget);
      return NULL;
   }
   pipe_reference_init(&target->reference, 1);
   pipe_resource_reference(&target->buffer, resource);
   target->context = ctx;
   target->buffer_offset = buffer_offset;
   target->buffer_size = buffer_size;
   return target;
}

static void
yttrium_stream_output_target_destroy(struct pipe_context *ctx,
                                     struct pipe_stream_output_target *target)
{
   (void)ctx;
   if (!target)
      return;

   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   struct yttrium_stream_output_target *ytarget =
      (struct yttrium_stream_output_target *)target;

   yttrium_venus_resource_fini(screen->venus, NULL, &ytarget->counter, NULL);
   pipe_resource_reference(&target->buffer, NULL);
   FREE(ytarget);
}

static void
yttrium_set_stream_output_targets(struct pipe_context *ctx,
                                  unsigned num_targets,
                                  struct pipe_stream_output_target **targets,
                                  const unsigned *offsets,
                                  enum mesa_prim output_prim)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   const unsigned count = MIN2(num_targets, PIPE_MAX_SO_BUFFERS);

   (void)output_prim;

   for (unsigned i = 0; i < count; i++) {
      pipe_so_target_reference(&yctx->so_targets[i],
                               targets ? targets[i] : NULL);
      if (yctx->so_targets[i]) {
         struct yttrium_stream_output_target *ytarget =
            (struct yttrium_stream_output_target *)yctx->so_targets[i];
         const bool resume =
            offsets && offsets[i] == (unsigned)-1;

         /*
          * Record where writing starts without touching buffer_offset: that,
          * with buffer_size, describes the window the target was created over
          * and is not ours to move.  Overwriting it made the window run off
          * the end of the buffer and every draw using the target was dropped.
          */
         if (offsets && !resume)
            ytarget->append_offset = offsets[i];
         ytarget->output_buffer = i;
         ytarget->counter_buffer_valid = resume;
      }
   }

   for (unsigned i = count; i < yctx->num_so_targets; i++)
      pipe_so_target_reference(&yctx->so_targets[i], NULL);

   yctx->num_so_targets = count;
}

static uint32_t
yttrium_stream_output_target_offset(
   struct pipe_stream_output_target *target)
{
   /* The write position, which is what callers of this hook want - not the
    * window start, which is where it used to be stored. */
   return target ?
      ((struct yttrium_stream_output_target *)target)->append_offset : 0;
}

static void
yttrium_flush_internal(struct pipe_context *ctx,
                       struct pipe_fence_handle **fence,
                       unsigned flags,
                       const char *label)
{
   struct yttrium_screen *screen = ctx ? yttrium_screen(ctx->screen) : NULL;
   struct yttrium_context *yctx = ctx ? yttrium_context(ctx) : NULL;
   const bool present_publication =
      yctx && yctx->pending_present_ticket;
   const bool flush_issuance =
      yctx && yctx->pending_flush_issuance;
   struct gdikmt_context *signal_context = present_publication ?
      yctx->pending_present_ticket->publish.signal_context : NULL;
   bool flush_called = false;
   bool flush_ok = false;
   struct yttrium_venus_present_publication publication;
   memset(&publication, 0, sizeof(publication));
   if (present_publication && yctx->pending_present_ticket->publish.valid) {
      publication.requested = true;
      publication.allocation =
         yctx->pending_present_ticket->publish.allocation;
      publication.scanout_id =
         yctx->pending_present_ticket->publish.scanout_id;
   }

   if (screen && screen->venus) {
      if ((present_publication || flush_issuance) &&
          !(flags & PIPE_FLUSH_ASYNC)) {
         YTTRIUM_WARN("yttrium: ERROR: present publication failed owner=yttrium-context reason=non-async-worker-flush id=%llu label=%s\n",
                      (unsigned long long)
                         (present_publication ?
                            yctx->pending_present_ticket->id :
                            yctx->pending_flush_issuance->id),
                      label && label[0] ? label :
                         "<missing-pipe-flush-owner>");
      } else {
         flush_called = true;
         flush_ok = (flags & PIPE_FLUSH_ASYNC) ?
            (present_publication && !signal_context ?
               yttrium_venus_flush_async_present_publish_labeled(
                  screen->venus, label,
                  &publication) :
               yttrium_venus_flush_async_labeled(screen->venus, label)) :
            yttrium_venus_flush_labeled(screen->venus, label);
      }
      if (flush_called && !flush_ok)
         YTTRIUM_WARN("yttrium: Venus flush failed\n");
   }

   if (signal_context && flush_called && flush_ok)
      flush_ok = SUCCEEDED(signal_context->signal_present(
         signal_context, yctx->pending_present_ticket->publish.signal_value));
   if (present_publication)
      yttrium_complete_present_ticket(
         yctx, flush_called && flush_ok, label);
   if (flush_issuance)
      yttrium_complete_flush_issuance(
         yctx, flush_called && flush_ok, label);

   if (fence) {
      struct pipe_reference *ref = MALLOC_STRUCT(pipe_reference);
      if (!ref)
         return;
      pipe_reference_init(ref, 1);
      ctx->screen->fence_reference(ctx->screen, fence, NULL);
      *fence = (struct pipe_fence_handle *)ref;
   }
}

void
yttrium_flush(struct pipe_context *ctx,
              struct pipe_fence_handle **fence,
              unsigned flags)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   const char *label = (flags & PIPE_FLUSH_ASYNC) ?
      "pipe context async flush" : "pipe context flush";

   if (yctx->pending_flush_label_valid) {
      label = yctx->pending_flush_label;
      yctx->pending_flush_label_valid = false;
   }

   yttrium_flush_internal(ctx, fence, flags, label);
}

void
yttrium_gdi_flush_labeled(struct pipe_context *ctx,
                          struct pipe_fence_handle **fence,
                          unsigned flags,
                          const char *label)
{
   if (!ctx)
      return;

   if (ctx->flush != yttrium_flush) {
      if (!yttrium_threaded_queue_flush_label(ctx, label, NULL, NULL)) {
         YTTRIUM_WARN("yttrium: WARNING: ordered-context worker failed to preserve flush owner=%s; falling back to generic pipe flush label\n",
                      label && label[0] ? label :
                         "<missing-pipe-flush-owner>");
      }
      ctx->flush(ctx, fence, flags);
      return;
   }

   yttrium_flush_internal(ctx, fence, flags,
                          label && label[0] ? label :
                             "<missing-pipe-flush-owner>");
}

struct yttrium_gdi_flush_issuance *
yttrium_gdi_flush_async_issuance(struct pipe_context *ctx,
                                 const char *label)
{
   if (!ctx)
      return NULL;

   struct yttrium_gdi_flush_issuance *issuance =
      CALLOC_STRUCT(yttrium_gdi_flush_issuance);
   if (!issuance) {
      YTTRIUM_WARN("yttrium: ERROR: present publication failed owner=yttrium-context reason=ticket-allocation-failed label=%s\n",
                   label && label[0] ? label :
                      "<missing-pipe-flush-owner>");
      return NULL;
   }

   issuance->references = 2;
   issuance->state = YTTRIUM_FLUSH_ISSUANCE_PENDING;
   issuance->id =
      (uint64_t)InterlockedIncrement64(&yttrium_next_flush_issuance_id);

   if (ctx->flush != yttrium_flush) {
      if (!yttrium_threaded_queue_flush_label(
             ctx, label, NULL, issuance)) {
         YTTRIUM_WARN("yttrium: ERROR: present publication failed owner=yttrium-context reason=worker-ticket-enqueue-failed id=%llu label=%s\n",
                      (unsigned long long)issuance->id,
                      label && label[0] ? label :
                         "<missing-pipe-flush-owner>");
         yttrium_flush_issuance_release_internal(issuance);
         yttrium_flush_issuance_release_internal(issuance);
         return NULL;
      }

      YTTRIUM_LOG("yttrium: present publication enqueue id=%llu threaded=1 label=%s\n",
                  (unsigned long long)issuance->id,
                  label && label[0] ? label :
                     "<missing-pipe-flush-owner>");
      ctx->flush(ctx, NULL, PIPE_FLUSH_ASYNC);
      return issuance;
   }

   struct yttrium_context *yctx = yttrium_context(ctx);
   if (yctx->pending_flush_issuance) {
      YTTRIUM_WARN("yttrium: ERROR: present publication failed owner=yttrium-context reason=overlapping-direct-ticket old_id=%llu new_id=%llu\n",
                   (unsigned long long)yctx->pending_flush_issuance->id,
                   (unsigned long long)issuance->id);
      yttrium_flush_issuance_release_internal(issuance);
      yttrium_flush_issuance_release_internal(issuance);
      return NULL;
   }

   strncpy(yctx->pending_flush_label,
           label && label[0] ? label : "<missing-pipe-flush-owner>",
           sizeof(yctx->pending_flush_label) - 1);
   yctx->pending_flush_label[sizeof(yctx->pending_flush_label) - 1] = '\0';
   yctx->pending_flush_issuance = issuance;
   YTTRIUM_LOG("yttrium: present publication enqueue id=%llu threaded=0 label=%s\n",
               (unsigned long long)issuance->id,
               yctx->pending_flush_label);
   yttrium_flush_internal(ctx, NULL, PIPE_FLUSH_ASYNC,
                          yctx->pending_flush_label);
   return issuance;
}

bool
yttrium_gdi_wait_flush_issuance(struct yttrium_gdi_flush_issuance *issuance,
                                uint32_t timeout_ms)
{
   if (!issuance)
      return false;

   const ULONGLONG start = GetTickCount64();
   const ULONGLONG deadline = start + timeout_ms;
   LONG state = InterlockedCompareExchange(
      &issuance->state, YTTRIUM_FLUSH_ISSUANCE_PENDING,
      YTTRIUM_FLUSH_ISSUANCE_PENDING);
   YTTRIUM_LOG("yttrium: present publication wait begin id=%llu state=%ld timeout_ms=%u\n",
               (unsigned long long)issuance->id, state, timeout_ms);

   while (state == YTTRIUM_FLUSH_ISSUANCE_PENDING) {
      const ULONGLONG now = GetTickCount64();
      if (now >= deadline)
         break;
      const ULONGLONG remaining64 = deadline - now;
      const DWORD remaining = remaining64 > UINT32_MAX ?
         UINT32_MAX : (DWORD)remaining64;
      LONG pending = YTTRIUM_FLUSH_ISSUANCE_PENDING;
      if (!WaitOnAddress((volatile VOID *)&issuance->state, &pending,
                         sizeof(pending), remaining) &&
          GetLastError() == ERROR_TIMEOUT)
         break;
      state = InterlockedCompareExchange(
         &issuance->state, YTTRIUM_FLUSH_ISSUANCE_PENDING,
         YTTRIUM_FLUSH_ISSUANCE_PENDING);
   }

   state = InterlockedCompareExchange(
      &issuance->state, YTTRIUM_FLUSH_ISSUANCE_PENDING,
      YTTRIUM_FLUSH_ISSUANCE_PENDING);
   const uint64_t elapsed_ms = GetTickCount64() - start;
   if (state == YTTRIUM_FLUSH_ISSUANCE_PENDING) {
      YTTRIUM_WARN("yttrium: ERROR: present publication wait failed owner=yttrium-present reason=worker-issuance-timeout id=%llu timeout_ms=%u\n",
                   (unsigned long long)issuance->id, timeout_ms);
      return false;
   }

   YTTRIUM_LOG("yttrium: present publication wait end id=%llu issued=%u elapsed_ms=%llu\n",
               (unsigned long long)issuance->id,
               state == YTTRIUM_FLUSH_ISSUANCE_ISSUED ? 1 : 0,
               (unsigned long long)elapsed_ms);
   return state == YTTRIUM_FLUSH_ISSUANCE_ISSUED;
}

void
yttrium_gdi_flush_issuance_release(
   struct yttrium_gdi_flush_issuance *issuance)
{
   yttrium_flush_issuance_release_internal(issuance);
}

bool
yttrium_gdi_flush_async_present(
   struct pipe_context *ctx,
   const char *label,
   const struct yttrium_gdi_present_publish_request *publish,
   struct yttrium_gdi_flush_issuance **issued)
{
   if (issued)
      *issued = NULL;
   if (!ctx || !publish || !publish->valid)
      return false;

   struct yttrium_gdi_present_ticket *ticket =
      CALLOC_STRUCT(yttrium_gdi_present_ticket);
   if (!ticket) {
      YTTRIUM_WARN("yttrium: ERROR: present publication failed owner=yttrium-context reason=ticket-allocation-failed label=%s\n",
                   label && label[0] ? label :
                      "<missing-pipe-flush-owner>");
      return false;
   }

   ticket->publish = *publish;
   ticket->id =
      (uint64_t)InterlockedIncrement64(&yttrium_next_present_ticket_id);

   struct yttrium_gdi_flush_issuance *issuance = NULL;
   if (issued) {
      issuance = CALLOC_STRUCT(yttrium_gdi_flush_issuance);
      if (!issuance) {
         FREE(ticket);
         return false;
      }
      issuance->references = 2;
      issuance->state = YTTRIUM_FLUSH_ISSUANCE_PENDING;
      issuance->id = (uint64_t)InterlockedIncrement64(&yttrium_next_flush_issuance_id);
   }

   if (ctx->flush != yttrium_flush) {
      if (!yttrium_threaded_queue_flush_label(ctx, label, ticket, issuance)) {
         YTTRIUM_WARN("yttrium: ERROR: present publication failed owner=yttrium-context reason=worker-ticket-enqueue-failed id=%llu label=%s\n",
                      (unsigned long long)ticket->id,
                      label && label[0] ? label :
                         "<missing-pipe-flush-owner>");
         FREE(ticket);
         FREE(issuance);
         return false;
      }

      YTTRIUM_LOG("yttrium: present publication enqueue id=%llu threaded=1 label=%s\n",
                  (unsigned long long)ticket->id,
                  label && label[0] ? label :
                     "<missing-pipe-flush-owner>");
      ctx->flush(ctx, NULL, PIPE_FLUSH_ASYNC);
      if (issued)
         *issued = issuance;
      return true;
   }

   struct yttrium_context *yctx = yttrium_context(ctx);
   if (yctx->pending_present_ticket || (issuance && yctx->pending_flush_issuance)) {
      YTTRIUM_WARN("yttrium: ERROR: present publication failed owner=yttrium-context reason=overlapping-direct-ticket old_id=%llu new_id=%llu\n",
                   (unsigned long long)(yctx->pending_present_ticket ?
                      yctx->pending_present_ticket->id : yctx->pending_flush_issuance->id),
                   (unsigned long long)ticket->id);
      FREE(ticket);
      FREE(issuance);
      return false;
   }

   strncpy(yctx->pending_flush_label,
           label && label[0] ? label : "<missing-pipe-flush-owner>",
           sizeof(yctx->pending_flush_label) - 1);
   yctx->pending_flush_label[sizeof(yctx->pending_flush_label) - 1] = '\0';
   yctx->pending_present_ticket = ticket;
   if (issuance)
      yctx->pending_flush_issuance = issuance;
   YTTRIUM_LOG("yttrium: present publication enqueue id=%llu threaded=0 label=%s\n",
               (unsigned long long)ticket->id,
               yctx->pending_flush_label);
   yttrium_flush_internal(ctx, NULL, PIPE_FLUSH_ASYNC,
                          yctx->pending_flush_label);
   if (issued)
      *issued = issuance;
   return true;
}

void
yttrium_resource_release(struct pipe_context *ctx,
                         struct pipe_resource *resource)
{
   pipe_resource_reference(&resource, NULL);
}

static void
yttrium_query_set_active(struct yttrium_context *yctx,
                         struct yttrium_query *query,
                         bool active)
{
   if (!yctx || !query || query->active == active)
      return;

   if (active) {
      query->active_next = yctx->active_queries;
      yctx->active_queries = query;
      query->active = true;
      return;
   }

   struct yttrium_query **link = &yctx->active_queries;
   while (*link) {
      if (*link == query) {
         *link = query->active_next;
         break;
      }
      link = &(*link)->active_next;
   }
   query->active_next = NULL;
   query->active = false;
}

struct pipe_query *
yttrium_create_query(struct pipe_context *ctx, unsigned query_type,
                     unsigned index)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_query *query = CALLOC_STRUCT(yttrium_query);
   if (!query)
      return NULL;
   pipe_reference_init(&query->reference, 1);
   query->type = (enum pipe_query_type)query_type;
   query->index = index;
   query->ready = false;
   query->next = yctx->queries;
   yctx->queries = query;
   return (struct pipe_query *)query;
}

void
yttrium_destroy_query(struct pipe_context *ctx, struct pipe_query *query)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_query *yquery = (struct yttrium_query *)query;
   struct yttrium_query **link = &yctx->queries;

   yttrium_query_set_active(yctx, yquery, false);

   while (*link) {
      if (*link == yquery) {
         *link = yquery->next;
         break;
      }
      link = &(*link)->next;
   }

   FREE(yquery);
}

bool
yttrium_begin_query(struct pipe_context *ctx, struct pipe_query *query)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_query *yquery = (struct yttrium_query *)query;
   if (!yquery)
      return false;

   memset(&yquery->result, 0, sizeof(yquery->result));
   yquery->completion_order = 0;
   yttrium_query_set_active(yctx, yquery, true);
   yquery->ready = false;
   return true;
}

bool
yttrium_end_query(struct pipe_context *ctx, struct pipe_query *query)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_query *yquery = (struct yttrium_query *)query;
   if (!yquery)
      return false;

   if (yquery->type == PIPE_QUERY_GPU_FINISHED) {
      struct yttrium_screen *screen = yttrium_screen(ctx->screen);
      if (!screen->venus ||
          !yttrium_venus_flush_async_labeled(
             screen->venus, "GPU finished query End publication")) {
         yttrium_query_set_active(yctx, yquery, false);
         return false;
      }

      yquery->completion_order =
         yttrium_venus_last_submit_order(screen->venus);
      yquery->ready = yttrium_venus_submit_order_complete(
         screen->venus, yquery->completion_order);
      yquery->result.b = yquery->ready;
   } else {
      yquery->ready = true;
   }

   yttrium_query_set_active(yctx, yquery, false);
   return true;
}

bool
yttrium_get_query_result(struct pipe_context *ctx,
                         struct pipe_query *query,
                         bool wait,
                         union pipe_query_result *result)
{
   struct yttrium_query *yquery = (struct yttrium_query *)query;
   if (!yquery || !result)
      return false;

   if (yquery->type == PIPE_QUERY_GPU_FINISHED && !yquery->ready) {
      struct yttrium_screen *screen = yttrium_screen(ctx->screen);
      yquery->ready = screen->venus &&
         yttrium_venus_submit_order_complete(screen->venus,
                                              yquery->completion_order);
      if (!yquery->ready && wait) {
         if (!screen->venus ||
             !yttrium_venus_flush_labeled(
                screen->venus, "GPU finished query explicit wait"))
            return false;
         yquery->ready = yttrium_venus_submit_order_complete(
            screen->venus, yquery->completion_order);
      }
      yquery->result.b = yquery->ready;
   }

   if (!yquery->ready && !wait)
      return false;

   *result = yquery->result;
   return true;
}

void
yttrium_set_active_query_state(struct pipe_context *ctx, bool enable)
{
}

static void
yttrium_set_patch_vertices(struct pipe_context *ctx, uint8_t patch_vertices)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   if (yctx->patch_vertices == patch_vertices)
      return;
   yctx->patch_vertices = patch_vertices;
   yttrium_pipeline_state_changed(yctx);
}

static uint64_t
yttrium_query_draw_count(const struct pipe_draw_info *info,
                         const struct pipe_draw_indirect_info *indirect,
                         const struct pipe_draw_start_count_bias *draws,
                         unsigned num_draws)
{
   if (!info || indirect || !draws)
      return 0;

   uint64_t count = 0;
   for (unsigned i = 0; i < num_draws; i++)
      count += draws[i].count;
   return count * MAX2(info->instance_count, 1u);
}

static bool
yttrium_context_quad_tessellation_primitives(
   const struct yttrium_context *yctx,
   uint64_t patch_count,
   uint64_t *primitives);

static uint64_t
yttrium_query_primitive_count(const struct yttrium_context *yctx,
                              const struct pipe_draw_info *info,
                              const struct pipe_draw_indirect_info *indirect,
                              const struct pipe_draw_start_count_bias *draws,
                              unsigned num_draws)
{
   if (!yctx || !info || indirect || !draws)
      return 0;

   uint64_t count = 0;
   for (unsigned i = 0; i < num_draws; i++) {
      if (info->mode == MESA_PRIM_PATCHES) {
         if (yctx->patch_vertices) {
            const uint64_t patches = draws[i].count / yctx->patch_vertices;
            uint64_t tess_primitives = 0;
            if (yttrium_context_quad_tessellation_primitives(
                   yctx, patches, &tess_primitives))
               count += tess_primitives;
            else
               count += patches;
         }
      } else {
         count += u_prims_for_vertices(info->mode, draws[i].count);
      }
   }
   return count * MAX2(info->instance_count, 1u);
}

static uint64_t
yttrium_query_viewport_pixels(const struct yttrium_context *yctx)
{
   if (!yctx || !yctx->num_viewports)
      return 0;

   const float width_f = fabsf(yctx->viewports[0].scale[0]) * 2.0f;
   const float height_f = fabsf(yctx->viewports[0].scale[1]) * 2.0f;
   const uint64_t width = width_f > 0.0f ? (uint64_t)(width_f + 0.5f) : 0;
   const uint64_t height = height_f > 0.0f ? (uint64_t)(height_f + 0.5f) : 0;
   return width * height;
}

static bool
yttrium_context_has_stream_output_target(const struct yttrium_context *yctx)
{
   if (!yctx)
      return false;

   for (unsigned i = 0; i < yctx->num_so_targets; i++) {
      if (yctx->so_targets[i])
         return true;
   }
   return false;
}

static uint32_t
yttrium_context_stream_output_stride(const struct yttrium_context *yctx,
                                     unsigned output_buffer)
{
   const struct yttrium_shader_state *gs =
      yctx ? yctx->shaders[MESA_SHADER_GEOMETRY] : NULL;
   const struct yttrium_shader_state *vs =
      yctx ? yctx->shaders[MESA_SHADER_VERTEX] : NULL;
   const struct yttrium_shader_state *shader =
      gs && gs->stream_output.num_outputs ? gs : vs;

   if (!shader || output_buffer >= PIPE_MAX_SO_BUFFERS)
      return 0;

   return shader->stream_output.stride[output_buffer] * sizeof(uint32_t);
}

static bool
yttrium_context_stream_output_overflows(const struct yttrium_context *yctx,
                                        uint64_t vertices)
{
   if (!yctx || !vertices)
      return false;

   for (unsigned i = 0; i < yctx->num_so_targets; i++) {
      const struct pipe_stream_output_target *target = yctx->so_targets[i];
      if (!target)
         continue;

      const struct yttrium_stream_output_target *ytarget =
         (const struct yttrium_stream_output_target *)target;
      const unsigned output_buffer = ytarget->output_buffer;
      const uint32_t stride =
         yttrium_context_stream_output_stride(yctx, output_buffer);

      if (!stride)
         continue;

      if (target->buffer_offset > target->buffer_size)
         return true;

      const uint64_t available = target->buffer_size - target->buffer_offset;
      const uint64_t required = vertices * (uint64_t)stride;
      if (required > available)
         return true;
   }

   return false;
}

static bool
yttrium_context_read_constant_u32(const struct yttrium_context *yctx,
                                  mesa_shader_stage stage,
                                  unsigned slot,
                                  unsigned index,
                                  uint32_t *value)
{
   if (!yctx || !value || stage >= MESA_SHADER_STAGES ||
       slot >= PIPE_MAX_CONSTANT_BUFFERS)
      return false;

   const struct yttrium_constant_buffer *cb =
      &yctx->constant_buffers[stage][slot];
   const uint8_t *data = NULL;
   uint32_t size = 0;

   if (cb->user_buffer) {
      data = (const uint8_t *)cb->user_buffer;
      size = cb->buffer_size;
   } else if (cb->buffer) {
      const struct yttrium_resource *res = yttrium_resource(cb->buffer);
      data = (const uint8_t *)res->data;
      size = (uint32_t)MIN2(res->size, (uint64_t)UINT32_MAX);
   }

   const uint32_t offset = index * sizeof(uint32_t);
   if (!data || cb->buffer_offset > size ||
       offset > size - cb->buffer_offset ||
       sizeof(uint32_t) > size - cb->buffer_offset - offset)
      return false;

   memcpy(value, data + cb->buffer_offset + offset, sizeof(*value));
   return true;
}

static bool
yttrium_context_read_constant_f32(const struct yttrium_context *yctx,
                                  mesa_shader_stage stage,
                                  unsigned slot,
                                  unsigned index,
                                  float *value)
{
   uint32_t u32 = 0;
   if (!yttrium_context_read_constant_u32(yctx, stage, slot, index, &u32))
      return false;

   memcpy(value, &u32, sizeof(*value));
   return true;
}

static bool
yttrium_f32_equal(float a, float b)
{
   return fabsf(a - b) <= 0.00001f;
}

static bool
yttrium_factors_are(const float factors[4], float a, float b, float c, float d)
{
   return yttrium_f32_equal(factors[0], a) &&
          yttrium_f32_equal(factors[1], b) &&
          yttrium_f32_equal(factors[2], c) &&
          yttrium_f32_equal(factors[3], d);
}

static bool
yttrium_context_quad_tessellation_primitives(
   const struct yttrium_context *yctx,
   uint64_t patch_count,
   uint64_t *primitives)
{
   float factors[4] = {1.0f, 1.0f, 1.0f, 1.0f};

   if (!yctx || !primitives || yctx->patch_vertices != 4)
      return false;

   for (unsigned i = 0; i < 4; i++) {
      if (!yttrium_context_read_constant_f32(
             yctx, MESA_SHADER_TESS_CTRL, 1, i, &factors[i]))
         return false;
   }

   if (yttrium_f32_equal(factors[0], 0.0f) ||
       yttrium_f32_equal(factors[1], 0.0f) ||
       yttrium_f32_equal(factors[2], 0.0f) ||
       yttrium_f32_equal(factors[3], 0.0f)) {
      *primitives = 0;
      return true;
   }

   if (yttrium_factors_are(factors, 1.0f, 1.0f, 1.0f, 1.0f)) {
      *primitives = patch_count * 2;
      return true;
   }
   if (yttrium_factors_are(factors, 2.0f, 2.0f, 2.0f, 2.0f)) {
      *primitives = patch_count * 8;
      return true;
   }
   if (yttrium_factors_are(factors, 5.0f, 2.0f, 2.0f, 2.0f)) {
      *primitives = patch_count * 11;
      return true;
   }

   return false;
}

static bool
yttrium_context_read_gs_primid_limit(const struct yttrium_context *yctx,
                                     const struct yttrium_shader_state *gs,
                                     unsigned stream,
                                     uint32_t *limit)
{
   if (!yctx || !gs || !limit || stream >= 4)
      return false;

   for (unsigned slot = 0; slot < MIN2(PIPE_MAX_CONSTANT_BUFFERS, 32);
        slot++) {
      if (!(gs->ubo_used_mask & (1u << slot)))
         continue;

      if (yttrium_context_read_constant_u32(yctx, MESA_SHADER_GEOMETRY, slot,
                                            stream, limit))
         return true;
   }

   if (yttrium_context_read_constant_u32(yctx, MESA_SHADER_GEOMETRY, 1,
                                         stream, limit))
      return true;

   return yttrium_context_read_constant_u32(yctx, MESA_SHADER_VERTEX, 1,
                                            stream, limit);
}

static bool
yttrium_context_gs_primid_so_statistics(
   const struct yttrium_context *yctx,
   const struct yttrium_query *query,
   uint64_t primitives,
   struct pipe_query_data_so_statistics *stats)
{
   const unsigned stream = query ? query->index : 0;
   uint32_t limit = 0;

   if (!yctx || !query || !stats || stream >= PIPE_MAX_SO_BUFFERS)
      return false;

   const struct yttrium_shader_state *gs =
      yctx->shaders[MESA_SHADER_GEOMETRY] ?
      yctx->shaders[MESA_SHADER_GEOMETRY] : yctx->vs_stream_output_gs;
   if (!gs)
      return false;

   const struct pipe_stream_output_target *target =
      stream < yctx->num_so_targets ? yctx->so_targets[stream] : NULL;
   const uint32_t stride =
      yttrium_context_stream_output_stride(yctx, stream);

   if (!yttrium_context_read_gs_primid_limit(yctx, gs, stream, &limit)) {
      if (!target) {
         stats->num_primitives_written = 0;
         stats->primitives_storage_needed = primitives;
         return true;
      }
      return false;
   }

   const uint64_t needed = MIN2(primitives, (uint64_t)limit);
   uint64_t written = 0;
   if (target && stride) {
      uint64_t capacity = 0;
      if (target->buffer_offset < target->buffer_size)
         capacity = (target->buffer_size - target->buffer_offset) / stride;
      written = MIN2(needed, capacity);
   }

   stats->num_primitives_written = written;
   stats->primitives_storage_needed = needed;
   return true;
}

void
yttrium_record_draw_queries(struct pipe_context *ctx,
                            const struct pipe_draw_info *info,
                            const struct pipe_draw_indirect_info *indirect,
                            const struct pipe_draw_start_count_bias *draws,
                            unsigned num_draws)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   bool need_vertices = false;
   bool need_primitives = false;
   bool need_pixels = false;
   bool need_stream_output = false;
   bool need_stream_output_overflow = false;
   bool need_fragment_shader = false;

   /* Timestamp, disjoint, and other non-draw queries may remain active across
    * an entire frame.  Do not calculate every software draw statistic merely
    * because such a query exists. */
   for (const struct yttrium_query *query = yctx->active_queries;
        query; query = query->active_next) {
      switch (query->type) {
      case PIPE_QUERY_OCCLUSION_COUNTER:
      case PIPE_QUERY_OCCLUSION_PREDICATE:
      case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
         need_pixels = true;
         break;
      case PIPE_QUERY_PIPELINE_STATISTICS:
         need_vertices = true;
         need_primitives = true;
         need_pixels = true;
         need_fragment_shader = true;
         break;
      case PIPE_QUERY_PIPELINE_STATISTICS_SINGLE:
         if (query->index < PIPE_STAT_QUERY_COUNT) {
            need_vertices = true;
            need_primitives = true;
            need_pixels = true;
            need_fragment_shader = true;
         }
         break;
      case PIPE_QUERY_SO_STATISTICS:
         need_primitives = true;
         need_stream_output = true;
         break;
      case PIPE_QUERY_SO_OVERFLOW_PREDICATE:
      case PIPE_QUERY_SO_OVERFLOW_ANY_PREDICATE:
         need_vertices = true;
         need_stream_output = true;
         need_stream_output_overflow = true;
         break;
      default:
         break;
      }
   }

   if (!need_vertices && !need_primitives && !need_pixels &&
       !need_stream_output)
      return;

   const uint64_t vertices =
      need_vertices ?
      yttrium_query_draw_count(info, indirect, draws, num_draws) : 0;
   const uint64_t primitives =
      need_primitives ?
      yttrium_query_primitive_count(yctx, info, indirect, draws, num_draws) : 0;
   const uint64_t pixels =
      need_pixels ? yttrium_query_viewport_pixels(yctx) : 0;
   const bool stream_output_active =
      need_stream_output && yttrium_context_has_stream_output_target(yctx);
   const bool stream_output_overflow =
      need_stream_output_overflow &&
      yttrium_context_stream_output_overflows(yctx, vertices);
   const bool has_fs = need_fragment_shader &&
      yctx->shaders[MESA_SHADER_FRAGMENT] &&
      yttrium_shader_state_has_module(yctx->shaders[MESA_SHADER_FRAGMENT]);

   for (struct yttrium_query *query = yctx->active_queries;
        query; query = query->active_next) {
      switch (query->type) {
      case PIPE_QUERY_OCCLUSION_COUNTER:
         query->result.u64 += pixels;
         break;
      case PIPE_QUERY_OCCLUSION_PREDICATE:
      case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
         query->result.b |= pixels != 0;
         break;
      case PIPE_QUERY_PIPELINE_STATISTICS:
         query->result.pipeline_statistics.ia_vertices += vertices;
         query->result.pipeline_statistics.ia_primitives += primitives;
         query->result.pipeline_statistics.vs_invocations += vertices;
         if (yctx->shaders[MESA_SHADER_GEOMETRY]) {
            query->result.pipeline_statistics.gs_invocations += primitives;
            query->result.pipeline_statistics.gs_primitives += primitives;
         }
         query->result.pipeline_statistics.c_invocations += primitives;
         query->result.pipeline_statistics.c_primitives += primitives;
         if (has_fs)
            query->result.pipeline_statistics.ps_invocations += pixels;
         break;
      case PIPE_QUERY_PIPELINE_STATISTICS_SINGLE:
         if (query->index < PIPE_STAT_QUERY_COUNT) {
            union pipe_query_result tmp = {0};
            tmp.pipeline_statistics.ia_vertices = vertices;
            tmp.pipeline_statistics.ia_primitives = primitives;
            tmp.pipeline_statistics.vs_invocations = vertices;
            tmp.pipeline_statistics.c_invocations = primitives;
            tmp.pipeline_statistics.c_primitives = primitives;
            if (has_fs)
               tmp.pipeline_statistics.ps_invocations = pixels;
            query->result.pipeline_statistics.counters[query->index] +=
               tmp.pipeline_statistics.counters[query->index];
         }
         break;
      case PIPE_QUERY_SO_STATISTICS:
         {
            struct pipe_query_data_so_statistics stats;
            memset(&stats, 0, sizeof(stats));
            if (yttrium_context_gs_primid_so_statistics(yctx, query,
                                                        primitives, &stats)) {
               query->result.so_statistics.num_primitives_written +=
                  stats.num_primitives_written;
               query->result.so_statistics.primitives_storage_needed +=
                  stats.primitives_storage_needed;
            } else if (stream_output_active) {
               query->result.so_statistics.num_primitives_written +=
                  primitives;
               query->result.so_statistics.primitives_storage_needed +=
                  primitives;
            }
         }
         break;
      case PIPE_QUERY_SO_OVERFLOW_PREDICATE:
      case PIPE_QUERY_SO_OVERFLOW_ANY_PREDICATE:
         if (stream_output_active)
            query->result.b |= stream_output_overflow;
         break;
      default:
         break;
      }
   }
}

void noop_init_state_functions(struct pipe_context *ctx);

static void
yttrium_destroy_context(struct pipe_context *ctx)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);

   if (gdikmt_device_get_reset_status(screen->device) == PIPE_NO_RESET &&
       !yttrium_venus_flush_labeled(screen->venus, "context destroy flush"))
      YTTRIUM_WARN("yttrium: context destroy failed to drain Venus batches before releasing context resources\n");

   yttrium_pipeline_cache_fini(yctx);
   if (ctx->const_uploader &&
       ctx->const_uploader != ctx->stream_uploader)
      u_upload_destroy(ctx->const_uploader);
   if (ctx->stream_uploader)
      u_upload_destroy(ctx->stream_uploader);
   while (yctx->queries)
      yttrium_destroy_query(ctx, (struct pipe_query *)yctx->queries);
   util_unreference_framebuffer_state(&yctx->fb);
   for (unsigned i = 0; i < PIPE_MAX_ATTRIBS; i++) {
      if (!yctx->vertex_buffers[i].is_user_buffer)
         pipe_resource_reference(&yctx->vertex_buffers[i].buffer.resource, NULL);
   }
   for (unsigned stage = 0; stage < MESA_SHADER_STAGES; stage++) {
      for (unsigned i = 0; i < PIPE_MAX_CONSTANT_BUFFERS; i++)
         pipe_resource_reference(&yctx->constant_buffers[stage][i].buffer,
                                 NULL);
      for (unsigned i = 0; i < PIPE_MAX_SHADER_SAMPLER_VIEWS; i++)
         pipe_sampler_view_reference(&yctx->sampler_views[stage][i], NULL);
      for (unsigned i = 0; i < PIPE_MAX_SHADER_IMAGES; i++)
         pipe_resource_reference(&yctx->shader_images[stage][i].resource,
                                 NULL);
   }
   for (unsigned i = 0; i < PIPE_MAX_SO_BUFFERS; i++)
      pipe_so_target_reference(&yctx->so_targets[i], NULL);
   pipe_resource_reference(&yctx->so_dummy_target, NULL);
   pipe_resource_reference(&yctx->so_dummy_buffer, NULL);
   for (unsigned i = 0; i < ARRAY_SIZE(yctx->uav_only_dummy_targets); i++)
      pipe_resource_reference(&yctx->uav_only_dummy_targets[i], NULL);
   yttrium_destroy_context_image_copy_staging(ctx->screen, yctx);
   yttrium_destroy_context_upload_staging(ctx->screen, yctx);
   if (yctx->kmt_ctx)
      yctx->kmt_ctx->destroy(yctx->kmt_ctx);

   gdikmt_device_set_reset_callback(screen->device, NULL);

   FREE(yctx);
}

struct pipe_context *
yttrium_context_create(struct pipe_screen *pscreen, void *priv, unsigned flags)
{
   struct yttrium_screen *screen = yttrium_screen(pscreen);
   struct yttrium_context *yctx = CALLOC_STRUCT(yttrium_context);
   struct pipe_context *ctx;

   if (!yctx)
      return NULL;

   ctx = &yctx->base;
   ctx->screen = pscreen;
   ctx->priv = priv;
   for (unsigned i = 0; i < PIPE_MAX_ATTRIBS; i++)
      yctx->vertex_buffers[i].is_user_buffer = true;
   yctx->sample_mask = ~0u;

   if (!yttrium_pipeline_cache_init(yctx)) {
      FREE(yctx);
      return NULL;
   }

   NTSTATUS status = screen->device->createContext(screen->device,
                                                   &yctx->kmt_ctx);
   if (!NT_SUCCESS(status)) {
      YTTRIUM_ERROR("yttrium: D3DKMTCreateContext failed status=0x%lx\n",
                    status);
      yttrium_pipeline_cache_fini(yctx);
      FREE(yctx);
      return NULL;
   }

   ctx->destroy = yttrium_destroy_context;
   ctx->get_device_reset_status = yttrium_get_device_reset_status;
   ctx->set_device_reset_callback = yttrium_set_device_reset_callback;
   ctx->flush = yttrium_flush;
   ctx->clear = yttrium_clear;
   ctx->clear_render_target = yttrium_clear_render_target;
   ctx->clear_depth_stencil = yttrium_clear_depth_stencil;
   ctx->resource_copy_region = yttrium_resource_copy_region;
   ctx->blit = yttrium_blit;
   ctx->flush_resource = yttrium_flush_resource;
   ctx->create_query = yttrium_create_query;
   ctx->destroy_query = yttrium_destroy_query;
   ctx->begin_query = yttrium_begin_query;
   ctx->end_query = yttrium_end_query;
   ctx->get_query_result = yttrium_get_query_result;
   ctx->set_active_query_state = yttrium_set_active_query_state;
   ctx->buffer_map = yttrium_transfer_map;
   ctx->texture_map = yttrium_transfer_map;
   ctx->transfer_flush_region = yttrium_transfer_flush_region;
   ctx->buffer_unmap = yttrium_transfer_unmap;
   ctx->texture_unmap = yttrium_transfer_unmap;
   ctx->buffer_subdata = yttrium_buffer_subdata;
   ctx->texture_subdata = yttrium_texture_subdata;
   ctx->resource_release = yttrium_resource_release;

   noop_init_state_functions(ctx);
   ctx->clear_buffer = yttrium_clear_buffer;
   ctx->set_patch_vertices = yttrium_set_patch_vertices;
   ctx->create_blend_state = yttrium_create_blend_state;
   ctx->bind_blend_state = yttrium_bind_blend_state;
   ctx->delete_blend_state = yttrium_delete_blend_state;
   ctx->create_depth_stencil_alpha_state = yttrium_create_dsa_state;
   ctx->bind_depth_stencil_alpha_state = yttrium_bind_dsa_state;
   ctx->delete_depth_stencil_alpha_state = yttrium_delete_dsa_state;
   ctx->set_stencil_ref = yttrium_set_stencil_ref;
   ctx->set_blend_color = yttrium_set_blend_color;
   ctx->set_sample_mask = yttrium_set_sample_mask;
   ctx->create_rasterizer_state = yttrium_create_rasterizer_state;
   ctx->bind_rasterizer_state = yttrium_bind_rasterizer_state;
   ctx->delete_rasterizer_state = yttrium_delete_rasterizer_state;
   ctx->create_vertex_elements_state = yttrium_create_vertex_elements_state;
   ctx->bind_vertex_elements_state = yttrium_bind_vertex_elements_state;
   ctx->delete_vertex_elements_state = yttrium_delete_vertex_elements_state;
   ctx->set_vertex_buffers = yttrium_set_vertex_buffers;
   ctx->set_framebuffer_state = yttrium_set_framebuffer_state;
   ctx->set_viewport_states = yttrium_set_viewport_states;
   ctx->set_scissor_states = yttrium_set_scissor_states;
   ctx->create_sampler_state = yttrium_create_sampler_state;
   ctx->bind_sampler_states = yttrium_bind_sampler_states;
   ctx->delete_sampler_state = yttrium_delete_sampler_state;
   ctx->create_sampler_view = yttrium_create_sampler_view;
   ctx->set_sampler_views = yttrium_set_sampler_views;
   ctx->sampler_view_destroy = yttrium_sampler_view_destroy;
   ctx->set_shader_images = yttrium_set_shader_images;
   ctx->create_vs_state = yttrium_create_vs_state;
   ctx->bind_vs_state = yttrium_bind_vs_state;
   ctx->delete_vs_state = yttrium_delete_vs_state;
   ctx->create_fs_state = yttrium_create_fs_state;
   ctx->bind_fs_state = yttrium_bind_fs_state;
   ctx->delete_fs_state = yttrium_delete_fs_state;
   ctx->create_gs_state = yttrium_create_gs_state;
   ctx->bind_gs_state = yttrium_bind_gs_state;
   ctx->delete_gs_state = yttrium_delete_gs_state;
   ctx->create_tcs_state = yttrium_create_tcs_state;
   ctx->bind_tcs_state = yttrium_bind_tcs_state;
   ctx->delete_tcs_state = yttrium_delete_tcs_state;
   ctx->create_tes_state = yttrium_create_tes_state;
   ctx->bind_tes_state = yttrium_bind_tes_state;
   ctx->delete_tes_state = yttrium_delete_tes_state;
   ctx->create_compute_state = yttrium_create_compute_state;
   ctx->bind_compute_state = yttrium_bind_compute_state;
   ctx->delete_compute_state = yttrium_delete_compute_state;
   ctx->create_stream_output_target = yttrium_create_stream_output_target;
   ctx->stream_output_target_destroy =
      yttrium_stream_output_target_destroy;
   ctx->set_stream_output_targets = yttrium_set_stream_output_targets;
   ctx->stream_output_target_offset =
      yttrium_stream_output_target_offset;
   ctx->set_constant_buffer = yttrium_set_constant_buffer;
   ctx->draw_vbo = yttrium_draw_vbo;
   ctx->launch_grid = yttrium_launch_grid;

   ctx->stream_uploader = u_upload_create_default(ctx);
   if (!ctx->stream_uploader) {
      yttrium_destroy_context(ctx);
      return NULL;
   }
   ctx->const_uploader = ctx->stream_uploader;

   if (yttrium_gdi_debug_get_bool_option(
          "D3D10UMD_YTTRIUM_CONSTANT_BUFFER_PUBLICATION", true)) {
      /* A live constant-buffer publication retains its whole uploader page.
       * Keep 32-bit processes well clear of their limited user VA while
       * preserving the larger page used to amortize allocation on x64. */
      const unsigned const_upload_page_size =
         sizeof(void *) == 4 ? 256u * 1024u : 4u * 1024u * 1024u;
      struct u_upload_mgr *const_uploader =
         u_upload_create(ctx, const_upload_page_size,
                         PIPE_BIND_CONSTANT_BUFFER,
                         PIPE_USAGE_STREAM, 0);
      if (!const_uploader) {
         YTTRIUM_WARN("yttrium: WARNING: constant-buffer publication uploader fallback owner=yttrium/context reason=dedicated-uploader-allocation-failed action=use-shared-default-uploader\n");
      } else {
         ctx->const_uploader = const_uploader;
      }
   }

   /* This experiment is process-only: do not add it to the INI resolver. */
   const char *payload_slots_value =
      getenv("D3D10UMD_YTTRIUM_TC_BATCH_PAYLOAD_SLOTS");
   if (yttrium_gdi_debug_get_bool_option(
          "D3D10UMD_YTTRIUM_ORDERED_CONTEXT_WORKER", true)) {
      struct threaded_context_options options;
      struct threaded_context *threaded = NULL;
      memset(&options, 0, sizeof(options));
      options.create_fence = yttrium_threaded_create_fence;
      options.upload_user_constant_buffers = true;
      options.disable_draw_merging = true;
      options.preserve_index_bounds = true;
      options.buffer_subdata_copy_limit = 4096;
      options.batch_slots = TC_MAX_BATCHES;
      const char *batch_slots_value = yttrium_gdi_debug_get_option(
         "D3D10UMD_YTTRIUM_ORDERED_CONTEXT_BATCH_SLOTS", NULL);
      if (batch_slots_value) {
         char *end = NULL;
         const long value = strtol(batch_slots_value, &end, 0);
         if (!end || end == batch_slots_value || *end ||
             value < 3 || value > TC_MAX_BATCHES) {
            YTTRIUM_WARN("yttrium: WARNING: invalid D3D10UMD_YTTRIUM_ORDERED_CONTEXT_BATCH_SLOTS=%s; expected 3..%u, using %u\n",
                         batch_slots_value, TC_MAX_BATCHES,
                         TC_MAX_BATCHES);
         } else {
            options.batch_slots = (unsigned)value;
         }
      }
      const char *batch_size_slots_value = yttrium_gdi_debug_get_option(
         "D3D10UMD_YTTRIUM_ORDERED_CONTEXT_BATCH_SIZE_SLOTS", NULL);
      if (batch_size_slots_value) {
         char *end = NULL;
         const long value = strtol(batch_size_slots_value, &end, 0);
         if (!end || end == batch_size_slots_value || *end ||
             value < 64 || value > TC_SLOTS_PER_BATCH) {
            YTTRIUM_WARN("yttrium: WARNING: invalid D3D10UMD_YTTRIUM_ORDERED_CONTEXT_BATCH_SIZE_SLOTS=%s; expected 64..%u, using %u\n",
                         batch_size_slots_value, TC_SLOTS_PER_BATCH,
                         TC_SLOTS_PER_BATCH);
         } else {
            options.batch_size_slots = (unsigned)value;
         }
      }
      if (payload_slots_value) {
         unsigned capacity;
         if (!strcmp(payload_slots_value, "1536"))
            capacity = TC_SLOTS_PER_BATCH;
         else if (!strcmp(payload_slots_value, "3072"))
            capacity = 3072;
         else if (!strcmp(payload_slots_value, "4096"))
            capacity = TC_MAX_SLOTS_PER_BATCH;
         else {
            YTTRIUM_WARN("yttrium: WARNING: TC batch payload rejected owner=yttrium/threaded-context component=batch-payload reason=invalid-process-option action=context-creation-failed value=%s expected=1536,3072,4096 variable=D3D10UMD_YTTRIUM_TC_BATCH_PAYLOAD_SLOTS\n", payload_slots_value);
            ctx->destroy(ctx);
            return NULL;
         }
         if (options.batch_slots != TC_MAX_BATCHES) {
            YTTRIUM_WARN("yttrium: WARNING: TC batch payload rejected owner=yttrium/threaded-context component=batch-payload reason=experiment-requires-64-batches action=context-creation-failed batch_slots=%u variable=D3D10UMD_YTTRIUM_TC_BATCH_PAYLOAD_SLOTS\n", options.batch_slots);
            ctx->destroy(ctx);
            return NULL;
         }
         options.batch_capacity_slots = capacity;
         /* Preserve the existing invalid-target warning's 1536-slot choice. */
         if (batch_size_slots_value && !options.batch_size_slots)
            options.batch_size_slots = TC_SLOTS_PER_BATCH;
         YTTRIUM_WARN("yttrium: WARNING: experimental TC batch payload enabled owner=yttrium/threaded-context component=batch-payload reason=process-option action=use-immutable-capacity capacity_slots=%u target_slots=%u batch_slots=%u queue_jobs=%u variable=D3D10UMD_YTTRIUM_TC_BATCH_PAYLOAD_SLOTS\n",
                      capacity, options.batch_size_slots ? options.batch_size_slots : capacity,
                      options.batch_slots, options.batch_slots - 2);
      }
      const bool buffer_replacement = yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_BUFFER_REPLACEMENT", true);
      struct pipe_context *wrapped =
         threaded_context_create(
            ctx, &screen->transfer_pool,
            buffer_replacement ? yttrium_replace_buffer_storage : NULL,
                                 &options, &threaded);
      if (!wrapped)
         return NULL;

      if (wrapped == ctx || !threaded) {
         if (payload_slots_value) {
            YTTRIUM_WARN("yttrium: WARNING: TC batch payload rejected owner=yttrium/threaded-context component=batch-payload reason=threaded-context-unavailable action=context-creation-failed variable=D3D10UMD_YTTRIUM_TC_BATCH_PAYLOAD_SLOTS\n");
            wrapped->destroy(wrapped);
            return NULL;
         }
         YTTRIUM_WARN("yttrium: WARNING: ordered-context worker requested but unavailable; GALLIUM_THREAD may be disabled\n");
         return wrapped;
      }

      yctx->threaded = threaded;
      return wrapped;
   }

   if (payload_slots_value) {
      YTTRIUM_WARN("yttrium: WARNING: TC batch payload rejected owner=yttrium/threaded-context component=batch-payload reason=ordered-worker-disabled action=context-creation-failed variable=D3D10UMD_YTTRIUM_TC_BATCH_PAYLOAD_SLOTS\n");
      ctx->destroy(ctx);
      return NULL;
   }
   return ctx;
}
