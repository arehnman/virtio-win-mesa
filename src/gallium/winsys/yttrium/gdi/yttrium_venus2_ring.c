/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#include <string.h>
#include <windows.h>

#include "util/os_time.h"
#include "yttrium_trace.h"
#include "yttrium_venus2_private.h"
#include "yttrium_venus2_ring.h"

#include "venus-protocol/vn_protocol_driver_transport.h"

#define YTTRIUM_VENUS_RING_PAUSE_BACKOFF_ITERS 4096
#define YTTRIUM_VENUS_RING_SLEEP0_BACKOFF_ITERS 65536

static INIT_ONCE yttrium_venus_ring_global_lock_once =
   INIT_ONCE_STATIC_INIT;
static CRITICAL_SECTION yttrium_venus_ring_global_lock;

static BOOL CALLBACK
yttrium_venus_init_ring_global_lock(PINIT_ONCE init_once,
                                    PVOID parameter,
                                    PVOID *context)
{
   (void)init_once;
   (void)parameter;
   (void)context;
   InitializeCriticalSection(&yttrium_venus_ring_global_lock);
   return TRUE;
}

void
yttrium_venus2_ring_lock(void)
{
   InitOnceExecuteOnce(&yttrium_venus_ring_global_lock_once,
                       yttrium_venus_init_ring_global_lock, NULL, NULL);
   EnterCriticalSection(&yttrium_venus_ring_global_lock);
}

void
yttrium_venus2_ring_unlock(void)
{
   LeaveCriticalSection(&yttrium_venus_ring_global_lock);
}

static void
yttrium_venus_ring_lock(struct yttrium_venus *venus)
{
   (void)venus;
   yttrium_venus2_ring_lock();
}

static void
yttrium_venus_ring_unlock(struct yttrium_venus *venus)
{
   (void)venus;
   yttrium_venus2_ring_unlock();
}

static void
yttrium_venus_ring_wait_backoff(uint32_t *iters)
{
   const uint32_t iter = (*iters)++;

   if (iter < YTTRIUM_VENUS_RING_PAUSE_BACKOFF_ITERS)
      YieldProcessor();
   else if (iter < YTTRIUM_VENUS_RING_SLEEP0_BACKOFF_ITERS)
      Sleep(0);
   else
      Sleep(1);
}

static uint32_t
yttrium_venus_ring_command_type_from_submit(
   const struct vn_ring_submit_command *submit,
   uint32_t command_size)
{
   uint32_t command_type = UINT32_MAX;

   if (submit && submit->buffer.base && command_size >= sizeof(command_type))
      memcpy(&command_type, submit->buffer.base, sizeof(command_type));

   return command_type;
}

bool
yttrium_venus_ring_create(struct yttrium_venus *venus)
{
   struct yttrium_venus_ring *ring = &venus->ring;

   ring->head_offset = 0;
   ring->tail_offset = 64;
   ring->status_offset = 128;
   ring->buffer_offset = 192;
   ring->buffer_size = yttrium_venus_ring_buffer_size();
   ring->buffer_mask = ring->buffer_size - 1;
   ring->extra_offset = ring->buffer_offset + ring->buffer_size;
   ring->extra_size = sizeof(uint32_t);
   ring->id = yttrium_venus_next_id(venus);

   if (!yttrium_venus_bo_create(venus, &ring->bo,
                                ring->extra_offset + ring->extra_size))
      return false;

   ring->head = (volatile uint32_t *)((uint8_t *)ring->bo.map +
                                      ring->head_offset);
   ring->tail = (volatile uint32_t *)((uint8_t *)ring->bo.map +
                                      ring->tail_offset);
   ring->status = (volatile uint32_t *)((uint8_t *)ring->bo.map +
                                        ring->status_offset);
   ring->buffer = (uint8_t *)ring->bo.map + ring->buffer_offset;

   const VkRingCreateInfoMESA info = {
      .sType = VK_STRUCTURE_TYPE_RING_CREATE_INFO_MESA,
      .resourceId = ring->bo.res_id,
      .size = ring->bo.size,
      .idleTimeout = YTTRIUM_VENUS_RING_IDLE_TIMEOUT_NS,
      .headOffset = ring->head_offset,
      .tailOffset = ring->tail_offset,
      .statusOffset = ring->status_offset,
      .bufferOffset = ring->buffer_offset,
      .bufferSize = ring->buffer_size,
      .extraOffset = ring->extra_offset,
      .extraSize = ring->extra_size,
   };

   uint8_t data[256];
   struct vn_cs_encoder enc =
      VN_CS_ENCODER_INITIALIZER_LOCAL(data, sizeof(data));
   vn_encode_vkCreateRingMESA(&enc, 0, ring->id, &info);

   if (yttrium_venus_warn_encoder_overflow("create-ring", &enc, 0)) {
      yttrium_venus_bo_destroy(venus, &ring->bo);
      memset(ring, 0, sizeof(*ring));
      return false;
   }

   if (!yttrium_venus_raw_submit(venus, data, vn_cs_encoder_get_len(&enc))) {
      yttrium_venus_bo_destroy(venus, &ring->bo);
      memset(ring, 0, sizeof(*ring));
      return false;
   }

   YTTRIUM_LOG("yttrium: Venus ring id=%llu res_id=%u size=0x%llx\n",
                (unsigned long long)ring->id, ring->bo.res_id,
                (unsigned long long)ring->bo.size);
   return true;
}

void
yttrium_venus_ring_forget_at_device_teardown(struct yttrium_venus *venus)
{
   if (!venus)
      return;

   /* The runtime owns final allocation destruction in DestroyDevice, but the
    * UMD must release its process mapping before that ownership transfer. */
   yttrium_venus_bo_destroy(venus, &venus->ring.bo);
   memset(&venus->ring, 0, sizeof(venus->ring));
}

bool
yttrium_venus_ring_ready(const struct yttrium_venus_ring *ring)
{
   return ring && ring->head && ring->tail && ring->status &&
          ring->buffer && ring->buffer_size;
}

static struct yttrium_venus *
yttrium_venus_from_ring(struct yttrium_venus_ring *ring)
{
   return ring ? (struct yttrium_venus *)((uint8_t *)ring -
                                          offsetof(struct yttrium_venus,
                                                   ring)) :
                 NULL;
}

static void ATTRIBUTE_NOINLINE
yttrium_venus_mark_ring_failed(struct yttrium_venus *venus,
                               const char *reason)
{
   if (!venus)
      return;

   venus->failed = true;
   if (gdikmt_device_get_reset_status(venus->device) != PIPE_NO_RESET)
      return;

   YTTRIUM_WARN("yttrium: ERROR: Venus ring marked failed; reporting device removal owner=venus2-ring reason=%s runtime_destroying=%u callback=%u\n",
                 reason ? reason : "<unknown>",
                 venus->device ? venus->device->runtime_destroying : 0,
                 venus->device && venus->device->reset_callback.reset ? 1 : 0);

   gdikmt_device_report_reset(venus->device, PIPE_UNKNOWN_CONTEXT_RESET);
}

#if defined(YTTRIUM_VENUS_HAS_NATIVE_SEH)
static LONG
yttrium_venus_ring_exception_filter(DWORD code)
{
   return code == EXCEPTION_ACCESS_VIOLATION ||
          code == EXCEPTION_IN_PAGE_ERROR ?
             EXCEPTION_EXECUTE_HANDLER :
             EXCEPTION_CONTINUE_SEARCH;
}
#endif

static uint32_t ATTRIBUTE_NOINLINE
yttrium_venus_ring_faulting_load_dword(const volatile uint32_t *ptr)
{
   return *ptr;
}

static void ATTRIBUTE_NOINLINE
yttrium_venus_ring_faulting_store_dword(volatile uint32_t *ptr,
                                        uint32_t value)
{
   *ptr = value;
}

static void ATTRIBUTE_NOINLINE
yttrium_venus_ring_faulting_copy_bytes(void *dst,
                                       const void *src,
                                       size_t size)
{
   memcpy(dst, src, size);
}

static bool
yttrium_venus_ring_load_dword(const char *field,
                              const volatile uint32_t *ptr,
                              uint32_t *out)
{
   if (out)
      *out = 0;

   if (!out || !ptr) {
      YTTRIUM_WARN("yttrium: Venus ring %s pointer is not readable ptr=%p\n",
                   field ? field : "<unknown>", (const void *)ptr);
      return false;
   }

   YTTRIUM_VENUS_SEH_TRY {
      *out = yttrium_venus_ring_faulting_load_dword(ptr);
      return true;
   } YTTRIUM_VENUS_SEH_EXCEPT(
      yttrium_venus_ring_exception_filter(GetExceptionCode())) {
      YTTRIUM_WARN("yttrium: Venus ring %s pointer read fault ptr=%p\n",
                   field ? field : "<unknown>", (const void *)ptr);
      return false;
   }
}

static bool
yttrium_venus_ring_store_dword(const char *field,
                               volatile uint32_t *ptr,
                               uint32_t value)
{
   if (!ptr) {
      YTTRIUM_WARN("yttrium: Venus ring %s pointer is not writable ptr=%p\n",
                   field ? field : "<unknown>", (const void *)ptr);
      return false;
   }

   YTTRIUM_VENUS_SEH_TRY {
      yttrium_venus_ring_faulting_store_dword(ptr, value);
      return true;
   } YTTRIUM_VENUS_SEH_EXCEPT(
      yttrium_venus_ring_exception_filter(GetExceptionCode())) {
      YTTRIUM_WARN("yttrium: Venus ring %s pointer write fault ptr=%p\n",
                   field ? field : "<unknown>", (const void *)ptr);
      return false;
   }
}

static bool
yttrium_venus_ring_copy_bytes(const char *label,
                              void *dst,
                              const void *src,
                              size_t size)
{
   if (!size)
      return true;

   if (!dst || !src) {
      YTTRIUM_WARN("yttrium: Venus ring copy rejected label=%s dst=%p src=%p size=%llu\n",
                   label ? label : "<unknown>", dst, src,
                   (unsigned long long)size);
      return false;
   }

   YTTRIUM_VENUS_SEH_TRY {
      yttrium_venus_ring_faulting_copy_bytes(dst, src, size);
      return true;
   } YTTRIUM_VENUS_SEH_EXCEPT(
      yttrium_venus_ring_exception_filter(GetExceptionCode())) {
      YTTRIUM_WARN("yttrium: Venus ring copy fault label=%s dst=%p src=%p size=%llu\n",
                   label ? label : "<unknown>", dst, src,
                   (unsigned long long)size);
      return false;
   }
}

static bool
yttrium_venus_ring_snapshot(struct yttrium_venus_ring *ring,
                            uint32_t *head,
                            uint32_t *tail,
                            uint32_t *status)
{
   if (!yttrium_venus_ring_ready(ring))
      return false;

   if (head && !yttrium_venus_ring_load_dword("head", ring->head, head))
      return false;
   if (tail && !yttrium_venus_ring_load_dword("tail", ring->tail, tail))
      return false;
   if (status &&
       !yttrium_venus_ring_load_dword("status", ring->status, status))
      return false;

   return true;
}

static uint32_t
yttrium_venus_ring_head_or_zero(struct yttrium_venus *venus)
{
   uint32_t head = 0;

   if (venus && venus->ring.head)
      yttrium_venus_ring_load_dword("head", venus->ring.head, &head);

   return head;
}

bool
yttrium_venus_ring_probe(struct yttrium_venus *venus,
                         const char *failure_reason)
{
   uint32_t head;

   if (!venus || venus->failed ||
       gdikmt_device_get_reset_status(venus->device) != PIPE_NO_RESET)
      return false;

   if (!yttrium_venus_ring_ready(&venus->ring) ||
       !yttrium_venus_ring_load_dword("head", venus->ring.head, &head)) {
      yttrium_venus_mark_ring_failed(venus, failure_reason);
      return false;
   }

   return true;
}

bool
yttrium_venus_ring_seqno_complete(struct yttrium_venus *venus,
                                  const struct yttrium_venus_ring_dependency *dependency)
{
   uint32_t head = 0;

   if (!venus || !yttrium_venus_ring_dependency_valid(dependency) ||
       !yttrium_venus_ring_ready(&venus->ring))
      return false;
   if (!yttrium_venus_ring_load_dword("head", venus->ring.head, &head))
      return false;

   return yttrium_venus_ring_dependency_reached(head, dependency);
}

static bool
yttrium_venus_ring_flush_notify_if_idle(struct yttrium_venus *venus,
                                        const char *label,
                                        bool blocking);

static bool
yttrium_venus_ring_publish_tail(struct yttrium_venus *venus,
                                const char *label,
                                bool blocking)
{
   if (!venus || !yttrium_venus_ring_ready(&venus->ring))
      return false;

   if (venus->ring.published_cur == venus->ring.cur)
      return true;

   MemoryBarrier();
   if (!yttrium_venus_ring_store_dword("tail", venus->ring.tail,
                                       venus->ring.cur)) {
      yttrium_venus_mark_ring_failed(venus, "tail-publish");
      return false;
   }
   MemoryBarrier();

   venus->ring.published_cur = venus->ring.cur;
   venus->ring_notify_pending = true;
   venus->ring_notify_seqno = venus->ring.cur;
   return yttrium_venus_ring_flush_notify_if_idle(venus, label, blocking);
}

bool
yttrium_venus_ring_publish_for_submit(struct yttrium_venus *venus, uint32_t *seqno)
{
   /* The caller serializes recording and captures only a published cursor. */
   if (!yttrium_venus_ring_publish_tail(venus, "scheduled GPU submit", false))
      return false;
   *seqno = venus->ring.published_cur;
   return true;
}

static bool
yttrium_venus_ring_wait_space(struct yttrium_venus_ring *ring,
                              uint32_t size,
                              const char *label,
                              uint32_t command_type,
                              uint32_t command_size,
                              uint32_t reply_size)
{
   struct yttrium_venus *venus = yttrium_venus_from_ring(ring);
   const uint64_t start = GetTickCount64();
   const uint64_t start_us =
      yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;
   uint64_t wait_start_us = 0;
   uint32_t head = 0;
   uint32_t wait_backoff_iters = 0;

   if (!yttrium_venus_ring_ready(ring)) {
      YTTRIUM_WARN("yttrium: Venus ring wait-space rejected invalid ring ring=%p head=%p tail=%p status=%p buffer=%p size=%u write_size=%u\n",
                   ring,
                   ring ? (void *)ring->head : NULL,
                   ring ? (void *)ring->tail : NULL,
                   ring ? (void *)ring->status : NULL,
                   ring ? ring->buffer : NULL,
                   ring ? ring->buffer_size : 0,
                   size);
      yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_WAIT_SPACE,
                                 3, start_us, NULL, size,
                                 ring ? ring->buffer_size : 0,
                                 ring ? ring->cur : 0, 0);
      return false;
   }

   if (!yttrium_venus_ring_load_dword("head", ring->head, &head)) {
      yttrium_venus_mark_ring_failed(venus, "wait-space-head");
      yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_WAIT_SPACE,
                                 4, start_us, NULL, size, ring->buffer_size,
                                 ring->cur, 0);
      return false;
   }

   if (size > ring->buffer_size) {
      yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_WAIT_SPACE,
                                 1, start_us, NULL, size, ring->buffer_size,
                                 ring->cur, head);
      return false;
   }

   const uint32_t used_after_write = ring->cur + size - head;
   const uint32_t notify_threshold =
      ring->buffer_size - ring->buffer_size / 4;
   if (used_after_write > notify_threshold &&
       venus && venus->ring_notify_pending) {
      /* High occupancy is not evidence that the renderer is asleep.  If it
       * is already consuming the ring, another doorbell only adds a KMD
       * escape and can turn one outstanding wake into a kick per command.
       */
      if (!yttrium_venus_ring_flush_notify_if_idle(
             venus, "ring high-watermark", false))
         return false;
      if (!yttrium_venus_ring_load_dword("head", ring->head, &head)) {
         yttrium_venus_mark_ring_failed(venus, "wait-space-head");
         yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_WAIT_SPACE,
                                    4, start_us, NULL, size, ring->buffer_size,
                                    ring->cur, 0);
         return false;
      }
   }

   /* A transaction keeps its tail private until the batch boundary.  If that
    * private prefix fills the ring, publish it before waiting so the renderer
    * can consume space.  The command bytes are already complete and ordered.
    */
   if (ring->cur + size - head > ring->buffer_size && venus &&
       venus->ring_transaction_active &&
       ring->published_cur != ring->cur) {
      if (!yttrium_venus_ring_publish_tail(
             venus, "ring transaction partial", false))
         return false;
      if (!yttrium_venus_ring_load_dword("head", ring->head, &head)) {
         yttrium_venus_mark_ring_failed(venus, "wait-space-head");
         return false;
      }
   }

   while (ring->cur + size - head > ring->buffer_size) {
      if (!wait_start_us)
         wait_start_us = yttrium_trace_now_us();
      if (GetTickCount64() - start > YTTRIUM_VENUS_RING_WAIT_MS) {
         YTTRIUM_LOG("yttrium: Venus ring space wait timed out cur=%u head=%u size=%u capacity=%u\n",
                      ring->cur, head, size, ring->buffer_size);
         yttrium_venus_debug_sync_wait(
            venus, YTTRIUM_VENUS_SYNC_WAIT_RING_SPACE,
            yttrium_trace_now_us() - wait_start_us, 2, label, size, head,
            command_type, command_size, reply_size);
         yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_WAIT_SPACE,
                                    2, start_us, NULL, size, ring->buffer_size,
                                    ring->cur, head);
         yttrium_venus_mark_ring_failed(venus, "wait-space-timeout");
         return false;
      }
      if (!(wait_backoff_iters & 0xffu) && venus &&
          venus->ring_notify_pending &&
          !yttrium_venus_ring_flush_notify_if_idle(
             venus, "ring space idle transition", false))
         return false;
      yttrium_venus_ring_wait_backoff(&wait_backoff_iters);
      if (!yttrium_venus_ring_load_dword("head", ring->head, &head)) {
         yttrium_venus_mark_ring_failed(venus, "wait-space-head");
         yttrium_venus_debug_sync_wait(
            venus, YTTRIUM_VENUS_SYNC_WAIT_RING_SPACE,
            wait_start_us ? yttrium_trace_now_us() - wait_start_us : 0,
            4, label, size, ring->cur,
            command_type, command_size, reply_size);
         yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_WAIT_SPACE,
                                    4, start_us, NULL, size, ring->buffer_size,
                                    ring->cur, 0);
         return false;
      }
   }

   if (wait_start_us) {
      yttrium_venus_debug_sync_wait(
         venus, YTTRIUM_VENUS_SYNC_WAIT_RING_SPACE,
         yttrium_trace_now_us() - wait_start_us, 0, label, size, head,
         command_type, command_size, reply_size);
   }
   yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_WAIT_SPACE,
                              0, start_us, NULL, size, ring->buffer_size,
                              ring->cur, head);
   return true;
}

static bool
yttrium_venus_ring_write(struct yttrium_venus_ring *ring,
                         const void *data,
                         uint32_t size,
                         const char *label,
                         uint32_t command_type,
                         uint32_t command_size,
                         uint32_t reply_size)
{
   struct yttrium_venus *venus = yttrium_venus_from_ring(ring);

   if (!yttrium_venus_ring_wait_space(ring, size, label, command_type,
                                      command_size, reply_size))
      return false;

   const uint32_t offset = ring->cur & ring->buffer_mask;
   if (offset + size <= ring->buffer_size) {
      if (!yttrium_venus_ring_copy_bytes("ring-buffer-write",
                                         ring->buffer + offset, data, size)) {
         yttrium_venus_mark_ring_failed(venus, "ring-buffer-write");
         return false;
      }
   } else {
      const uint32_t first = ring->buffer_size - offset;
      if (!yttrium_venus_ring_copy_bytes("ring-buffer-wrap-first",
                                         ring->buffer + offset, data, first) ||
          !yttrium_venus_ring_copy_bytes(
             "ring-buffer-wrap-second", ring->buffer,
             (const uint8_t *)data + first, size - first)) {
         yttrium_venus_mark_ring_failed(venus, "ring-buffer-wrap-write");
         return false;
      }
   }

   ring->cur += size;
   return true;
}

static bool
yttrium_venus_ring_wait_seqno(struct yttrium_venus_ring *ring,
                              uint32_t seqno,
                              const char *label,
                              uint32_t command_type,
                              uint32_t command_size,
                              uint32_t reply_size)
{
   struct yttrium_venus *venus = yttrium_venus_from_ring(ring);
   const uint64_t start = GetTickCount64();
   const uint64_t start_us =
      yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;
   uint64_t wait_start_us = 0;
   uint32_t head = 0;
   uint32_t tail = 0;
   uint32_t status = 0;
   uint32_t wait_backoff_iters = 0;

   /* A ring timeout is terminal for this Venus instance.  Device removal has
    * already been reported by the first failing waiter, so later (or
    * concurrent) waits must fail immediately instead of each consuming the
    * full timeout again.
    */
   if (!venus || venus->failed)
      return false;

   ring->current_wait_command_count =
      ring->protocol_command_count - ring->wait_baseline_command_count;
   ring->current_wait_command_bytes =
      ring->protocol_command_bytes - ring->wait_baseline_command_bytes;
   ring->current_wait_queue_submit_count =
      ring->protocol_queue_submit_count -
      ring->wait_baseline_queue_submit_count;
   ring->wait_baseline_command_count = ring->protocol_command_count;
   ring->wait_baseline_command_bytes = ring->protocol_command_bytes;
   ring->wait_baseline_queue_submit_count =
      ring->protocol_queue_submit_count;

   if (!yttrium_venus_ring_ready(ring)) {
      YTTRIUM_WARN("yttrium: Venus ring wait-seqno rejected invalid ring ring=%p head=%p tail=%p status=%p buffer=%p size=%u seqno=%u\n",
                   ring,
                   ring ? (void *)ring->head : NULL,
                   ring ? (void *)ring->tail : NULL,
                   ring ? (void *)ring->status : NULL,
                   ring ? ring->buffer : NULL,
                   ring ? ring->buffer_size : 0,
                   seqno);
      yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_WAIT_SEQNO,
                                 2, start_us, NULL, seqno, 0,
                                 ring ? ring->cur : 0, 0);
      return false;
   }

   if (!yttrium_venus_ring_snapshot(ring, &head, &tail, &status)) {
      yttrium_venus_mark_ring_failed(venus, "wait-seqno-snapshot");
      yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_WAIT_SEQNO,
                                 3, start_us, NULL, seqno, 0, 0, 0);
      return false;
   }

   while (!yttrium_venus_ring_cursor_reached(head, seqno)) {
      if (venus->failed)
         return false;
      if (!wait_start_us)
         wait_start_us = yttrium_trace_now_us();
      if (GetTickCount64() - start > YTTRIUM_VENUS_RING_WAIT_MS) {
         YTTRIUM_WARN("yttrium: FATAL: Venus ring wait-seqno timeout wait_ms=%llu head=%u tail=%u seqno=%u status=0x%x command=%s(%u) command_size=%u reply_size=%u ring_cur=%u\n",
                      (unsigned long long)YTTRIUM_VENUS_RING_WAIT_MS,
                      head, tail, seqno, status,
                      yttrium_venus_command_type_name(command_type),
                      command_type, command_size, reply_size, ring->cur);
         yttrium_venus_debug_sync_wait(
            venus, YTTRIUM_VENUS_SYNC_WAIT_RING_SEQNO,
            yttrium_trace_now_us() - wait_start_us, 1, label, seqno, head,
            command_type, command_size, reply_size);
         yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_WAIT_SEQNO,
                                    1, start_us, NULL, seqno, status, head, tail);
         yttrium_venus_mark_ring_failed(venus, "wait-seqno-timeout");
         return false;
      }
      yttrium_venus_ring_wait_backoff(&wait_backoff_iters);
      if (!yttrium_venus_ring_snapshot(ring, &head, &tail, &status)) {
         yttrium_venus_mark_ring_failed(venus, "wait-seqno-snapshot");
         yttrium_venus_debug_sync_wait(
            venus, YTTRIUM_VENUS_SYNC_WAIT_RING_SEQNO,
            wait_start_us ? yttrium_trace_now_us() - wait_start_us : 0,
            3, label, seqno, head,
            command_type, command_size, reply_size);
         yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_WAIT_SEQNO,
                                    3, start_us, NULL, seqno, 0, 0, 0);
         return false;
      }
   }

   if (wait_start_us) {
      yttrium_venus_debug_sync_wait(
         venus, YTTRIUM_VENUS_SYNC_WAIT_RING_SEQNO,
         yttrium_trace_now_us() - wait_start_us, 0, label, seqno, head,
         command_type, command_size, reply_size);
   }
   yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_WAIT_SEQNO,
                              0, start_us, NULL, seqno, status, head, tail);
   return true;
}

bool
yttrium_venus_ring_wait_for_seqno(struct yttrium_venus *venus,
                                  const struct yttrium_venus_ring_dependency *dependency,
                                  const char *label)
{
   if (!venus || !yttrium_venus_ring_dependency_valid(dependency))
      return false;

   /* The command may have been published as a reply-free asynchronous
    * submit, which is allowed to leave an idle-host notification pending.
    * Once a caller transitions to an explicit wait, that deferred wake can
    * no longer rely on later ring traffic.
    */
   if (!yttrium_venus_ring_flush_notify_if_idle(venus, label, true))
      return false;

   return yttrium_venus_ring_wait_seqno(
      &venus->ring, dependency->seqno, label,
      VK_COMMAND_TYPE_vkWaitForFences_EXT, 0, 0);
}

static bool
yttrium_venus_ring_notify(struct yttrium_venus *venus,
                          uint32_t seqno,
                          const char *label,
                          bool blocking)
{
   uint8_t data[64];
   struct vn_cs_encoder enc =
      VN_CS_ENCODER_INITIALIZER_LOCAL(data, sizeof(data));
   vn_encode_vkNotifyRingMESA(&enc, 0, venus->ring.id, seqno, 0);

   if (yttrium_venus_warn_encoder_overflow("notify-ring", &enc, 0))
      return false;

   if (!yttrium_venus_raw_submit_ring_kick(
          venus, data, vn_cs_encoder_get_len(&enc), seqno, blocking, label)) {
      yttrium_venus_mark_ring_failed(venus, "notify-raw-submit");
      return false;
   }

   const int64_t now = os_time_get_nano();
   venus->ring_last_notify = now;
   venus->ring_next_notify = now + YTTRIUM_VENUS_RING_IDLE_TIMEOUT_NS;
   return true;
}

bool
yttrium_venus_ring_flush_notify(struct yttrium_venus *venus,
                                const char *label,
                                bool blocking)
{
   /*
    * This unlocked read is only an advisory fast path.  The locked re-check
    * below owns the pending-notify state, and callers may already hold this
    * recursive CRITICAL_SECTION when forcing a high-watermark flush.
    */
   if (!venus || !venus->ring_notify_pending)
      return true;

   yttrium_venus_ring_lock(venus);
   if (!venus->ring_notify_pending) {
      yttrium_venus_ring_unlock(venus);
      return true;
   }

   const uint32_t seqno = venus->ring_notify_seqno;
   if (!yttrium_venus_ring_notify(venus, seqno, label, blocking)) {
      const uint32_t head = yttrium_venus_ring_head_or_zero(venus);
      YTTRIUM_LOG("yttrium: Venus deferred ring notify failed label=%s seqno=%u head=%u cur=%u\n",
                  label ? label : "<unknown>", seqno, head,
                  venus->ring.cur);
      yttrium_venus_mark_ring_failed(venus, "flush-notify");
      yttrium_venus_ring_unlock(venus);
      return false;
   }

   venus->ring_notify_pending = false;
   yttrium_venus_ring_unlock(venus);
   return true;
}

/*
 * Ring the doorbell only if the host has actually parked.  A notify costs a
 * D3DKMTRender - a syscall, a WDDM scheduler enqueue and contention on the
 * scheduler's spinlock - while a host that is still polling will see the new
 * tail on its own.
 *
 * blocking says the caller is about to wait on this seqno.  A waiter cannot
 * afford the rate limit below: if the host is idle and we skip the notify, the
 * wait lasts until something else happens to wake it.  Non-blocking callers can
 * safely skip, because the pending notify survives and a later submit, the high
 * watermark, or the idle timeout will send it.
 */
static bool
yttrium_venus_ring_flush_notify_if_idle(struct yttrium_venus *venus,
                                        const char *label,
                                        bool blocking)
{
   if (!venus || !venus->ring_notify_pending)
      return true;

   uint32_t status = 0;
   if (!yttrium_venus_ring_load_dword("status", venus->ring.status, &status)) {
      yttrium_venus_mark_ring_failed(venus, "idle-status");
      return false;
   }

   if (!(status & VK_RING_STATUS_IDLE_BIT_MESA))
      return true;

   /*
    * The host marks the ring idle after the same timeout, so an idle bit
    * observed before this guest-side rate limit expires represents a recent
    * notify window, not a lost wakeup.
    */
   if (!blocking) {
      const int64_t now = os_time_get_nano();
      if (!os_time_timeout(venus->ring_last_notify,
                           venus->ring_next_notify, now))
         return true;
   }

   return yttrium_venus_ring_flush_notify(venus, label, blocking);
}

bool
yttrium_venus_warn_encoder_overflow(const char *label,
                                    const struct vn_cs_encoder *enc,
                                    size_t reply_size)
{
   if (!enc || !enc->fatal_error)
      return false;

   const size_t command_size = vn_cs_encoder_get_len(enc);
   const size_t command_capacity =
      enc->buffer_count && enc->buffers ?
      (size_t)((const uint8_t *)enc->end -
               (const uint8_t *)enc->buffers[0].base) : 0;

   YTTRIUM_WARN("yttrium: Venus command encoder overflow label=%s cmd_size=%llu capacity=%llu offset=%llu write_size=%llu available=%llu reply_size=%llu\n",
                label ? label : "<unknown>",
                (unsigned long long)command_size,
                (unsigned long long)command_capacity,
                (unsigned long long)enc->fatal_offset,
                (unsigned long long)enc->fatal_size,
                (unsigned long long)enc->fatal_available,
                (unsigned long long)reply_size);
   return true;
}

bool
yttrium_venus_async_submit_succeeded(
   struct yttrium_venus *venus,
   const struct vn_ring_submit_command *submit,
   const char *operation,
   uint64_t object_id)
{
   if (submit && submit->ring_seqno_valid)
      return true;

   const enum pipe_reset_status reset_status =
      venus && venus->device ? gdikmt_device_get_reset_status(venus->device) :
              PIPE_UNKNOWN_CONTEXT_RESET;
   const char *reason =
      !submit ? "missing-submit" :
      (venus && (venus->failed || reset_status != PIPE_NO_RESET)) ?
         "transport-failed" :
         "local-command-encoding-or-ring-enqueue-failed";

   YTTRIUM_WARN("yttrium: ERROR: asynchronous Venus object command rejected owner=venus2 operation=%s object_id=%llu reason=%s action=fail-object-create transport_failed=%u reset_status=%d\n",
                operation ? operation : "<unknown>",
                (unsigned long long)object_id,
                reason,
                venus ? venus->failed : 1,
                reset_status);
   return false;
}

static void
yttrium_venus_warn_ring_submit_failure(
   struct yttrium_venus *venus,
   const struct vn_ring_submit_command *submit,
   uint32_t status,
   const char *reason,
   uint32_t command_size,
   uint32_t seqno)
{
   uint32_t command_words[4] = { 0 };
   uint32_t head = 0;
   uint32_t tail = 0;
   uint32_t ring_status = 0;

   if (submit && submit->buffer.base && command_size) {
      const size_t copy_size =
         command_size < sizeof(command_words) ? command_size :
                                                sizeof(command_words);
      memcpy(command_words, submit->buffer.base, copy_size);
   }

   if (venus)
      yttrium_venus_ring_snapshot(&venus->ring, &head, &tail, &ring_status);

   YTTRIUM_WARN("yttrium: Venus ring submit failed reason=%s status=%u cmd_size=%u reply_size=%llu seqno=%u valid=%u cur=%u head=%u tail=%u ring_status=0x%x failed=%u reply_map=%p reply_size_bo=%llu reply_res_id=%u cmd0=0x%08x cmd1=0x%08x cmd2=0x%08x cmd3=0x%08x\n",
                reason ? reason : "<unknown>",
                status,
                command_size,
                (unsigned long long)(submit ? submit->reply_size : 0),
                seqno,
                submit ? submit->ring_seqno_valid : 0,
                venus ? venus->ring.cur : 0,
                head,
                tail,
                ring_status,
                venus ? venus->failed : 0,
                venus ? venus->reply_bo.map : NULL,
                (unsigned long long)(venus ? venus->reply_bo.size : 0),
                venus ? venus->reply_bo.res_id : 0,
                command_words[0],
                command_words[1],
                command_words[2],
                command_words[3]);
}

bool
yttrium_venus_drain_ring(struct yttrium_venus *venus, const char *label)
{
   bool ok = false;
   uint32_t head = 0;
   uint32_t tail = 0;
   uint32_t status = 0;

   if (!venus || venus->failed || !yttrium_venus_ring_ready(&venus->ring))
      return false;

   yttrium_venus_ring_lock(venus);

   if (!yttrium_venus_ring_flush_notify(venus, label, true))
      goto out;

   const uint32_t seqno = venus->ring.cur;
   if (!yttrium_venus_ring_snapshot(&venus->ring, &head, &tail, &status)) {
      yttrium_venus_mark_ring_failed(venus, "drain-snapshot");
      goto out;
   }

   if (yttrium_venus_ring_cursor_reached(head, seqno)) {
      ok = true;
      goto out;
   }

   MemoryBarrier();
   if (!yttrium_venus_ring_store_dword("tail", venus->ring.tail, seqno)) {
      yttrium_venus_mark_ring_failed(venus, "drain-tail");
      goto out;
   }
   venus->ring.published_cur = seqno;
   MemoryBarrier();

   if (!yttrium_venus_ring_notify(venus, seqno, label, true) ||
       !yttrium_venus_ring_wait_seqno(&venus->ring, seqno, label,
                                      UINT32_MAX, 0, 0)) {
      yttrium_venus_ring_snapshot(&venus->ring, &head, &tail, &status);
      YTTRIUM_LOG("yttrium: Venus ring drain failed label=%s head=%u tail=%u seqno=%u status=0x%x\n",
                   label ? label : "<unknown>",
                   head,
                   tail,
                   seqno,
                   status);
      goto out;
   }

   ok = true;

out:
   yttrium_venus_ring_unlock(venus);
   return ok;
}

static bool
yttrium_venus2_vn_ring_submit_commands_locked(
   struct vn_ring *vn_ring,
   struct vn_ring_submit_command *submit,
   uint32_t command_count)
{
   const uint64_t start_us =
      yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;
   struct yttrium_venus *venus =
      vn_ring ? (struct yttrium_venus *)vn_ring->driver : NULL;

   if (!venus) {
      yttrium_venus_warn_ring_submit_failure(venus, submit, 1,
                                             "missing-or-failed-venus",
                                             0, 0);
      yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_SUBMIT,
                                 1, start_us, NULL, 0, 0, 0, 0);
      return false;
   }
   if (venus->failed) {
      yttrium_venus_mark_ring_failed(venus, "submit-after-failed-ring");
      return false;
   }
   if (gdikmt_device_get_reset_status(venus->device) != PIPE_NO_RESET) {
      yttrium_venus_mark_ring_failed(venus, "submit-after-device-removal");
      return false;
   }

   if (!yttrium_venus_ring_ready(&venus->ring)) {
      yttrium_venus_warn_ring_submit_failure(venus, submit, 8,
                                             "invalid-ring",
                                             0, 0);
      yttrium_venus_mark_ring_failed(venus, "invalid-ring");
      yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_SUBMIT,
                                 8, start_us, NULL, 0, 0, 0, 0);
      return false;
   }

   const uint32_t command_size =
      (uint32_t)vn_cs_encoder_get_len(&submit->command);
   const uint32_t command_type =
      command_count == 1 ?
         yttrium_venus_ring_command_type_from_submit(submit, command_size) :
         UINT32_MAX;
   const char *wait_label = submit->reply_size ?
      yttrium_venus_command_type_name(command_type) :
      "Venus ring producer";
   if (yttrium_venus_warn_encoder_overflow("ring-submit",
                                           &submit->command,
                                           submit->reply_size)) {
      yttrium_venus_warn_ring_submit_failure(venus, submit, 10,
                                             "encoder-overflow",
                                             command_size, 0);
      return false;
   }
   if (!command_size) {
      yttrium_venus_warn_ring_submit_failure(venus, submit, 2,
                                             "empty-command",
                                             0, 0);
      yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_SUBMIT,
                                 2, start_us, NULL, 0, submit->reply_size, 0, 0);
      return false;
   }

   uint32_t reply_command_size = 0;
   if (submit->reply_size) {
      if (!venus->reply_bo.map || submit->reply_size > venus->reply_bo.size) {
         yttrium_venus_warn_ring_submit_failure(venus, submit, 3,
                                                "reply-buffer-too-small",
                                                command_size, 0);
         yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_SUBMIT,
                                    3, start_us, NULL, command_size,
                                    submit->reply_size, 0, 0);
         return false;
      }

      memset(venus->reply_bo.map, 0, submit->reply_size);

      const VkCommandStreamDescriptionMESA stream = {
         .resourceId = venus->reply_bo.res_id,
         .offset = 0,
         .size = submit->reply_size,
      };

      uint8_t reply_data[64];
      struct vn_cs_encoder reply_enc =
         VN_CS_ENCODER_INITIALIZER_LOCAL(reply_data, sizeof(reply_data));
      vn_encode_vkSetReplyCommandStreamMESA(&reply_enc, 0, &stream);
      reply_command_size = (uint32_t)vn_cs_encoder_get_len(&reply_enc);

      if (!yttrium_venus_ring_write(
             &venus->ring, reply_data, reply_command_size, wait_label,
             command_type, command_size,
             submit->reply_size > UINT32_MAX ? UINT32_MAX :
                                                (uint32_t)submit->reply_size)) {
         const uint32_t head = yttrium_venus_ring_head_or_zero(venus);
         yttrium_venus_warn_ring_submit_failure(venus, submit, 4,
                                                "reply-stream-write",
                                                command_size, 0);
         yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_SUBMIT,
                                    4, start_us, NULL, command_size,
                                    submit->reply_size, venus->ring.cur,
                                    head);
         return false;
      }
   }

   if (!yttrium_venus_ring_write(
          &venus->ring, submit->buffer.base, command_size, wait_label,
          command_type, command_size,
          submit->reply_size > UINT32_MAX ? UINT32_MAX :
                                             (uint32_t)submit->reply_size)) {
      const uint32_t head = yttrium_venus_ring_head_or_zero(venus);
      yttrium_venus_warn_ring_submit_failure(venus, submit, 5,
                                             "command-write",
                                             command_size, 0);
      yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_SUBMIT,
                                 5, start_us, NULL, command_size,
                                 submit->reply_size, venus->ring.cur,
                                 head);
      return false;
   }

   venus->ring.protocol_command_count +=
      command_count + (reply_command_size ? 1 : 0);
   venus->ring.protocol_command_bytes += command_size + reply_command_size;
   if (command_type == VK_COMMAND_TYPE_vkQueueSubmit_EXT)
      venus->ring.protocol_queue_submit_count++;

   const uint32_t seqno = venus->ring.cur;
   submit->ring_seqno = seqno;

   /*
    * A reply-bearing submit used to notify unconditionally, which meant every
    * synchronous command cost a doorbell whether or not the host needed waking
    * - and those are the same commands we then block on, so it was a syscall
    * and a scheduler round trip in front of every wait.  Take the same idle
    * check as the fire-and-forget path, flagged blocking so an idle host is
    * still woken immediately.
    */
   const bool defer_tail =
      venus->ring_transaction_active && !submit->reply_size &&
      venus->ring_transaction_thread_id == GetCurrentThreadId();
   if (!defer_tail &&
       !yttrium_venus_ring_publish_tail(venus,
                                       submit->reply_size ?
                                          "ring reply" : "ring idle",
                                       submit->reply_size != 0)) {
      const uint32_t head = yttrium_venus_ring_head_or_zero(venus);
      const bool tail_failed =
         venus->ring.published_cur != venus->ring.cur;
      yttrium_venus_warn_ring_submit_failure(venus, submit,
                                             tail_failed ? 9 : 6,
                                             tail_failed ?
                                                "tail-publish" : "notify",
                                             command_size, seqno);
      yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_SUBMIT,
                                 tail_failed ? 9 : 6, start_us, NULL,
                                 command_size,
                                 submit->reply_size, seqno, head);
      return false;
   }

   if (submit->reply_size) {
      if (!yttrium_venus_ring_wait_seqno(&venus->ring, seqno, wait_label,
                                         command_type,
                                         command_size,
                                         submit->reply_size > UINT32_MAX ?
                                            UINT32_MAX :
                                            (uint32_t)submit->reply_size)) {
         const uint32_t head = yttrium_venus_ring_head_or_zero(venus);
         yttrium_venus_warn_ring_submit_failure(venus, submit, 7,
                                                "wait-seqno",
                                                command_size, seqno);
         yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_SUBMIT,
                                    7, start_us, NULL, command_size,
                                    submit->reply_size, seqno,
                                    head);
         return false;
      }

      submit->reply =
         VN_CS_DECODER_INITIALIZER(venus->reply_bo.map, submit->reply_size);
   }

   /*
    * Name the command in the otherwise unused label field.  Ring submits are
    * the guest's dominant per-draw cost - ~23 per draw at ~2 us - so which
    * commands make up that 23 is the whole question, and it is not answerable
    * from a count alone.
    */
   const uint32_t head = yttrium_venus_ring_head_or_zero(venus);
   yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RING_SUBMIT,
                              0, start_us,
                              command_count == 1 ?
                                 yttrium_venus_command_type_name(command_type) :
                                 "Deferred draw command stream",
                              command_size, submit->reply_size, seqno, head);
   return true;
}

bool
yttrium_venus2_vn_ring_submit_command_locked(
   struct vn_ring *vn_ring,
   struct vn_ring_submit_command *submit)
{
   return yttrium_venus2_vn_ring_submit_commands_locked(vn_ring, submit, 1);
}

bool
yttrium_venus2_command_stream_flush(struct yttrium_venus_command_stream *stream)
{
   if (stream->failed)
      return false;
   if (!stream->size)
      return true;

   struct vn_ring_submit_command submit;
   struct vn_cs_encoder *enc = vn_ring_submit_command_init(
      &stream->venus->vn_ring, &submit, stream->data, stream->size, 0);
   enc->cur = stream->data + stream->size;
   stream->failed = !yttrium_venus2_vn_ring_submit_commands_locked(
      &stream->venus->vn_ring, &submit, stream->command_count);
   stream->size = 0;
   stream->command_count = 0;
   return !stream->failed;
}

static bool
yttrium_venus2_command_stream_submit(struct vn_ring *vn_ring,
                                     struct vn_ring_submit_command *submit)
{
   struct yttrium_venus_command_stream *stream = vn_ring->driver;
   if (stream->failed)
      return false;

   const size_t size = vn_cs_encoder_get_len(&submit->command);
   if (submit->reply_size || !size ||
       yttrium_venus_warn_encoder_overflow("deferred-command-stream",
                                           &submit->command,
                                           submit->reply_size)) {
      YTTRIUM_WARN("yttrium: ERROR: deferred command stream rejected command owner=venus2-ring size=%llu reply_size=%llu\n",
                   (unsigned long long)size,
                   (unsigned long long)submit->reply_size);
      stream->failed = true;
      return false;
   }

   /* Never split an encoded command. Larger commands use their own write;
    * the pending prefix must precede it. Both use the same locked transport.
    */
   if (size > sizeof(stream->data) - stream->size &&
       !yttrium_venus2_command_stream_flush(stream))
      return false;
   if (size > sizeof(stream->data)) {
      stream->failed = !yttrium_venus2_vn_ring_submit_command_locked(
         &stream->venus->vn_ring, submit);
      return !stream->failed;
   }

   memcpy(stream->data + stream->size, submit->buffer.base, size);
   stream->size += (uint32_t)size;
   stream->command_count++;
   /* The transaction owns the ring until the prefix has been flushed. No
    * caller of this async-only sink consumes the seqno before that boundary.
    */
   submit->ring_seqno = stream->venus->ring.cur + stream->size;
   return true;
}

void
yttrium_venus2_command_stream_init(struct yttrium_venus_command_stream *stream,
                                  struct yttrium_venus *venus)
{
   assert(venus->ring_transaction_active &&
          venus->ring_transaction_thread_id == GetCurrentThreadId());
   stream->vn_ring = (struct vn_ring) {
      .driver = stream,
      .submit_command = yttrium_venus2_command_stream_submit,
   };
   stream->venus = venus;
   stream->size = 0;
   stream->command_count = 0;
   stream->failed = false;
}

bool
yttrium_venus2_ring_transaction_begin(struct yttrium_venus *venus)
{
   if (!venus || venus->failed || !yttrium_venus_ring_ready(&venus->ring))
      return false;

   yttrium_venus2_ring_lock();
   if (venus->ring_transaction_active) {
      yttrium_venus2_ring_unlock();
      return false;
   }

   venus->ring_transaction_active = true;
   venus->ring_transaction_thread_id = GetCurrentThreadId();
   return true;
}

bool
yttrium_venus2_ring_transaction_end(struct yttrium_venus *venus,
                                     const char *label)
{
   if (!venus || !venus->ring_transaction_active ||
       venus->ring_transaction_thread_id != GetCurrentThreadId())
      return false;

   const bool ok = yttrium_venus_ring_publish_tail(venus, label, false);
   venus->ring_transaction_active = false;
   venus->ring_transaction_thread_id = 0;
   yttrium_venus2_ring_unlock();
   return ok;
}

bool
yttrium_venus2_vn_ring_submit_command(struct vn_ring *vn_ring,
                                      struct vn_ring_submit_command *submit)
{
   yttrium_venus2_ring_lock();
   const bool ok =
      yttrium_venus2_vn_ring_submit_command_locked(vn_ring, submit);
   /*
    * Reply commands use one shared reply_bo.  Keep the global ring lock held
    * across get_command_reply/free_command_reply so no later submit can
    * overwrite that buffer before Venus decodes the reply.
    */
   if (!ok || !submit || !submit->reply_size)
      yttrium_venus2_ring_unlock();
   return ok;
}

struct vn_cs_decoder *
yttrium_venus2_vn_ring_get_command_reply(
   struct vn_ring *vn_ring,
   struct vn_ring_submit_command *submit)
{
   if (!submit)
      return NULL;

   if (submit->reply_size && !submit->ring_seqno_valid) {
      yttrium_venus2_vn_ring_warn_reply_request_with_invalid_seqno(
         vn_ring, submit);
      return NULL;
   }

   return submit->reply_size ? &submit->reply : NULL;
}

void
yttrium_venus2_vn_ring_free_command_reply(
   struct vn_ring *vn_ring,
   struct vn_ring_submit_command *submit)
{
   (void)vn_ring;
   /* Balanced with the reply-path lock handoff in submit_command above. */
   if (submit && submit->reply_size && submit->ring_seqno_valid)
      yttrium_venus2_ring_unlock();
}

void
yttrium_venus2_vn_ring_warn_reply_request_with_invalid_seqno(
   struct vn_ring *vn_ring,
   struct vn_ring_submit_command *submit)
{
   struct yttrium_venus *venus =
      vn_ring ? (struct yttrium_venus *)vn_ring->driver : NULL;

   yttrium_venus_warn_ring_submit_failure(
      venus, submit, 8, "reply-request-with-invalid-seqno",
      submit ? (uint32_t)vn_cs_encoder_get_len(&submit->command) : 0,
      submit ? submit->ring_seqno : 0);
}
