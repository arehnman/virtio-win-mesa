/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#include "yttrium_present.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>
#include <winternl.h>
#include <d3dkmthk.h>

#include "gdikmt/gdikmt.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"

#include "yttrium_context.h"
#include "yttrium_draw.h"
#include "yttrium_gdi_public.h"
#include "yttrium_internal.h"
#include "yttrium_options.h"
#include "yttrium_resource.h"
#include "yttrium_trace.h"
#include "yttrium_venus.h"

bool
yttrium_ensure_display_mapped(struct pipe_context *ctx,
                              struct yttrium_resource *res);

static struct pipe_context *
yttrium_present_driver_context(struct pipe_context *ctx)
{
   if (!ctx || !ctx->callback)
      return ctx;

   struct pipe_context *driver = threaded_context(ctx)->pipe;
   return driver && driver->flush == yttrium_flush ? driver : ctx;
}

struct yttrium_present_publication {
   struct yttrium_gdi_flush_issuance *issuance;
   struct gdikmt_context *producer;
   uint64_t value;
};

static HRESULT
yttrium_prepare_present_publication(void *data)
{
   struct yttrium_present_publication *publication = data;

   if (!publication)
      return E_FAIL;
   if (publication->issuance &&
       !yttrium_gdi_wait_flush_issuance(publication->issuance, 5000))
      return E_FAIL;
   if (publication->producer)
      return publication->producer->wait_present(publication->producer,
                                                 publication->value);
   return publication->issuance ? S_OK : E_FAIL;
}

void
yttrium_flush_frontbuffer(struct pipe_screen *pscreen,
                          struct pipe_context *ctx,
                          struct pipe_resource *resource,
                          unsigned level, unsigned layer,
                          void *context_private,
                          unsigned nboxes, struct pipe_box *boxes)
{
   struct yttrium_screen *screen = yttrium_screen(pscreen);
   struct yttrium_resource *res = yttrium_resource(resource);
   struct pipe_context *frontend_ctx = ctx;
   const struct gdikmt_present_info *present_info =
      gdikmt_present_info_from_context(context_private);
   struct gdikmt_present_info *present_status =
      (struct gdikmt_present_info *)present_info;
   void *dxgi_context_private =
      present_info ? present_info->dxgi_context : context_private;
   const D3DKMT_PRESENT *present =
      (const D3DKMT_PRESENT *)dxgi_context_private;

   struct pipe_context *driver_ctx = yttrium_present_driver_context(ctx);
   struct yttrium_context *yctx = yttrium_context(driver_ctx);

   /* Before the early returns, so a failed Present still counts. */
   yttrium_rt_stats_frame_advance();

   if (!res || !res->hAllocation || !yctx || !yctx->kmt_ctx) {
      if (present_status)
         present_status->status = E_FAIL;
      YTTRIUM_WARN("yttrium: present failed, missing display allocation or context\n");
      return;
   }

   if (!context_private) {
      yttrium_gdi_flush_labeled(frontend_ctx, NULL, PIPE_FLUSH_ASYNC,
                                "display frontbuffer publication");
      return;
   }

   struct gdikmt_context *present_ctx = yctx->kmt_ctx;
   if (yttrium_resource_is_venus_backed_display(res)) {
      present_ctx = yttrium_venus_get_kmt_context(screen->venus);
      if (!present_ctx) {
         if (present_status)
            present_status->status = E_FAIL;
         YTTRIUM_WARN("yttrium: KMD present skipped, missing Venus render context hAllocation=0x%lx res_id=%u primary=%u\n",
                     (unsigned long)res->hAllocation,
                     res->venus_res_id,
                     res->primary_target);
         return;
      }
   }

   /* Direct application scanout updates must follow GPU completion, so hand
    * them to the ordered worker and let that worker issue the scanout escape.
    * Windowed and DWM presents retain the ordinary pfnPresentCb path. */
   const bool scheduled = present_ctx->scheduled_present_mode != 0;
   struct gdikmt_context *producer = present_ctx;
   const bool worker_publish =
      !scheduled && yttrium_present_timeline_sync_enabled() && present_info &&
      present_info->version >= 3 && present_info->application_scanout;

   if (worker_publish) {
      const struct yttrium_gdi_present_publish_request publish = {
         .allocation = (uint32_t)res->hAllocation,
         .scanout_id = 0,
         .valid = true,
      };

      if (!yttrium_gdi_flush_async_present(
             frontend_ctx, "display Present worker publication", &publish, NULL)) {
         if (present_status)
            present_status->status = E_FAIL;
         return;
      }

      /*
       * The worker owns the only publication path.  Calling pfnPresentCb here
       * would let the miniport race an unordered scanout update against the
       * completion-ordered escape.  Setting PRESENT_TIMELINE_SYNC=0 skips this
       * branch and deliberately restores pfnPresentCb for fullscreen testing.
       */
      if (present_status)
         present_status->status = S_OK;
      return;
   }

   struct yttrium_gdi_flush_issuance *issuance = NULL;
   const bool wait_for_issuance = yttrium_present_issuance_wait_enabled();
   struct yttrium_present_publication publication = {0};

   if (scheduled) {
      HRESULT hr = producer->reserve_present(producer, &present_ctx, &publication.value);
      if (FAILED(hr)) {
         YTTRIUM_WARN("yttrium: ERROR scheduled Present owner=yttrium-present operation=reserve hr=%lx\n", (unsigned long)hr);
         if (present_status) present_status->status = hr;
         return;
      }
      publication.producer = producer;
      const struct yttrium_gdi_present_publish_request publish = {
         .allocation = (uint32_t)res->hAllocation,
         .valid = true,
         .signal_context = producer,
         .signal_value = publication.value,
      };
      if (!yttrium_gdi_flush_async_present(frontend_ctx, "scheduled Present signal",
             &publish, &issuance)) {
         if (present_status) present_status->status = E_FAIL;
         return;
      }
   } else if (wait_for_issuance) {
      issuance = yttrium_gdi_flush_async_issuance(
         frontend_ctx, "display Present publication");
      if (!issuance) {
         if (present_status)
            present_status->status = E_FAIL;
         return;
      }
   } else {
      yttrium_gdi_flush_labeled(frontend_ctx, NULL, PIPE_FLUSH_ASYNC,
                                "display Present publication");
   }

   publication.issuance = issuance;

   struct gdikmt_present_info device_present_info;
   memset(&device_present_info, 0, sizeof(device_present_info));
   device_present_info.magic = GDIKMT_PRESENT_INFO_MAGIC;
   device_present_info.version = 2;
   device_present_info.dxgi_context = dxgi_context_private;
   device_present_info.hDstAllocation =
      present_info ? present_info->hDstAllocation : 0;
   device_present_info.status = S_OK;
   if (scheduled || wait_for_issuance) {
      device_present_info.prepare_present =
         yttrium_prepare_present_publication;
      device_present_info.prepare_present_data = &publication;
   }

   NTSTATUS status = screen->device->present(present_ctx, res->hAllocation,
                                             &device_present_info, boxes);
   yttrium_gdi_flush_issuance_release(issuance);
   if (present_status)
      present_status->status = status;
   if (!NT_SUCCESS(status)) {
      YTTRIUM_WARN("yttrium: present callback path failed status=0x%lx hAllocation=0x%lx res_id=%u dxgi_owner=0x%lx hWindow=%p dxgi_source=0x%lx dxgi_destination=0x%lx flags=0x%x flip_interval=%u present_count=%u\n",
                   status,
                   (unsigned long)res->hAllocation,
                   res->venus_res_id,
                   present ? (unsigned long)present->hContext : 0,
                   present ? present->hWindow : NULL,
                   present ? (unsigned long)present->hSource : 0,
                   present ? (unsigned long)present->hDestination : 0,
                   present ? present->Flags.Value : 0,
                   present ? present->FlipInterval : 0,
                   present ? present->PresentCount : 0);
   }
}

void
yttrium_format_indexed_path(char *dst, size_t dst_size,
                            const char *path, unsigned index)
{
   const char *marker;
   size_t prefix_len;

   if (!dst || !dst_size)
      return;

   dst[0] = '\0';
   if (!path)
      return;

   marker = strstr(path, "%u");
   if (!marker) {
      snprintf(dst, dst_size, "%s", path);
      return;
   }

   prefix_len = (size_t)(marker - path);
   if (prefix_len >= dst_size)
      prefix_len = dst_size - 1;

   memcpy(dst, path, prefix_len);
   dst[prefix_len] = '\0';
   snprintf(dst + prefix_len, dst_size - prefix_len, "%u%s",
            index, marker + 2);
}

bool
yttrium_ensure_display_mapped(struct pipe_context *ctx,
                              struct yttrium_resource *res)
{
   if (res->map)
      return true;

   return yttrium_map_display_allocation(ctx->screen, res);
}
