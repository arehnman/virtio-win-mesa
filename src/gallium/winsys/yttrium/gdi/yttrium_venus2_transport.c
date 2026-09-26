/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#include <stdint.h>
#include <string.h>
#include <windows.h>
#include <winternl.h>

#include <d3dkmthk.h>

#include "gdikmt/gdikmt.h"
#include "pipe/p_defines.h"
#include "util/u_math.h"
#include "virtio/wddm/viogpu_wddm_driver.h"

#include "yttrium_trace.h"
#include "yttrium_venus2_private.h"
#include "yttrium_venus2_ring.h"

enum yttrium_venus_submit_path {
   YTTRIUM_VENUS_SUBMIT_PATH_UNKNOWN = 0,
   YTTRIUM_VENUS_SUBMIT_PATH_ESCAPE = 1,
   YTTRIUM_VENUS_SUBMIT_PATH_RENDER_EVENT = 3,
};

static const char *
yttrium_venus_submit_path_name(uint32_t path)
{
   switch (path) {
   case YTTRIUM_VENUS_SUBMIT_PATH_ESCAPE:
      return "kmd_submit_escape";
   case YTTRIUM_VENUS_SUBMIT_PATH_RENDER_EVENT:
      return "render_event";
   default:
      return "unknown";
   }
}

static bool
yttrium_venus_resize_submit_buffers(struct yttrium_venus *venus,
                                    size_t command_size)
{
   struct gdikmt_context *ctx = venus->kmt_ctx;

   if (!ctx || !ctx->pCommandBuffer)
      return false;

   if (ctx->CommandBufferSize >= command_size)
      return true;

   VIOGPU_COMMAND_HDR *hdr = (VIOGPU_COMMAND_HDR *)ctx->pCommandBuffer;
   memset(hdr, 0, sizeof(*hdr));
   hdr->type = VIOGPU_CMD_NOP;
   hdr->size = 0;

   struct gdikmt_render render;
   memset(&render, 0, sizeof(render));
   render.CommandLength = sizeof(*hdr);
   render.ResizeCommandBuffer = true;
   render.NewCommandBufferSize =
      (UINT)MIN2((uint64_t)UINT_MAX, (uint64_t)command_size + 0x1000);

   NTSTATUS status = ctx->render(ctx, &render);
   if (!NT_SUCCESS(status)) {
      YTTRIUM_LOG("yttrium: Venus submit-buffer resize failed status=0x%lx need=%llu have=%u\n",
                   status, (unsigned long long)command_size,
                   ctx->CommandBufferSize);
      return false;
   }

   return ctx->CommandBufferSize >= command_size;
}

bool
yttrium_venus_submit_gpu(struct yttrium_venus *venus, const void *data, size_t size)
{
   const size_t overhead = sizeof(VIOGPU_COMMAND_HDR) + sizeof(VIOGPU_TIMELINE_SUBMIT);
   struct gdikmt_context *ctx = venus->kmt_ctx;
   if (size > UINT_MAX - overhead ||
       !yttrium_venus_resize_submit_buffers(venus, overhead + size))
      return false;
   VIOGPU_COMMAND_HDR *header = ctx->pCommandBuffer;
   VIOGPU_TIMELINE_SUBMIT *packet = (VIOGPU_TIMELINE_SUBMIT *)(header + 1);
   header->type = VIOGPU_CMD_SUBMIT_TIMELINE;
   header->size = (UINT)(sizeof(*packet) + size);
   packet->RingIndex = 1;
   packet->Reserved = 0;
   memcpy(packet + 1, data, size);
   struct gdikmt_render render = {0};
   render.CommandLength = (UINT)(overhead + size);
   /* Process-only diagnostic: delay worker submission to exercise the
    * Present issuance checkpoint. Never read on the normal path. */
   char delay_text[16] = {0};
   GetEnvironmentVariableA("D3D10UMD_YTTRIUM_SCHEDULED_SUBMIT_DELAY_MS",
                          delay_text, sizeof(delay_text));
   unsigned delay = (unsigned)strtoul(delay_text, NULL, 10);
   if (delay && delay <= 1000) {
      YTTRIUM_LOG("yttrium: scheduled GPU submit delay begin ms=%u\n", delay);
      Sleep(delay);
      YTTRIUM_LOG("yttrium: scheduled GPU submit delay end ms=%u\n", delay);
   }
   NTSTATUS status = ctx->render(ctx, &render);
   YTTRIUM_LOG("yttrium: scheduled GPU submit context=0x%x bytes=%llu timeline=1 hr=%lx\n",
               ctx->kmt_handle(ctx), (unsigned long long)size, (unsigned long)status);
   if (!NT_SUCCESS(status)) {
      YTTRIUM_WARN("yttrium: ERROR scheduled GPU submit owner=venus2 hr=%lx\n", (unsigned long)status);
      gdikmt_device_report_reset(venus->device, PIPE_UNKNOWN_CONTEXT_RESET);
      return false;
   }
   return true;
}

/*
 * Hand a command straight to the KMD's submit escape.
 *
 * The alternative is ctx->render(), which walks the whole WDDM submission
 * path - DxgkRender, DXGCONTEXT::Render, VidSchSubmitCommand and the GPU
 * scheduler's spinlock in VidSchiUpdatePriorityTables.  That is a lot of
 * machinery for what is usually a 64-byte ring doorbell, and it showed up as
 * the largest kernel cost in the driver.  The escape reaches the same
 * ctrlQueue.SubmitCommand() and is queued identically.
 *
 * It completes through a KMD-side callback with no user-mode event, so a
 * caller that needs one still has to go the long way round.
 */
static bool
yttrium_venus_escape_submit(struct yttrium_venus *venus,
                            const void *data,
                            size_t size)
{
   struct gdikmt_context *ctx = venus ? venus->kmt_ctx : NULL;
   struct gdikmt_device *device = ctx ? ctx->device : NULL;
   uint8_t buffer[4096];

   if (!device || !device->escape)
      return false;

   const size_t payload_size = sizeof(VIOGPU_SUBMIT_CMD_REQ) + size;
   const size_t total_size = sizeof(VIOGPU_ESCAPE) + payload_size;

   /* DataLength is a USHORT, and the scratch above is fixed. */
   if (payload_size > USHRT_MAX || total_size > sizeof(buffer))
      return false;

   VIOGPU_ESCAPE *escape = (VIOGPU_ESCAPE *)buffer;
   memset(escape, 0, sizeof(*escape));
   escape->Type = VIOGPU_SUBMIT_CMD;
   escape->DataLength = (USHORT)payload_size;

   VIOGPU_SUBMIT_CMD_REQ *req = (VIOGPU_SUBMIT_CMD_REQ *)(escape + 1);
   req->hContext = ctx->kmt_handle ? ctx->kmt_handle(ctx) : 0;
   req->CmdType = VIOGPU_CMD_SUBMIT;
   req->CmdSize = (UINT)size;
   memcpy(req + 1, data, size);

   return NT_SUCCESS(device->escape(device, buffer, (UINT)total_size));
}

/*
 * Publish a display allocation straight through the display miniport, which
 * sets the scanout and flushes it in one escape.  Width, height, format, stride
 * and offset are left zero so the miniport fills them from the allocation's own
 * options rather than this caller re-deriving them.
 *
 * This is the ordered worker's publication path.  It carries no DMA packet and
 * no fence, so it neither waits for nor delays the runtime's Present; the
 * worker calls it only once the frame's rendering has completed.
 */
NTSTATUS
yttrium_venus_escape_publish_display(struct yttrium_venus *venus,
                                     uint32_t allocation,
                                     uint32_t scanout_id)
{
   struct gdikmt_device *device = venus ? venus->device : NULL;

   if (!device || !device->escape || !allocation)
      return STATUS_INVALID_PARAMETER;

   VIOGPU_ESCAPE escape;
   memset(&escape, 0, sizeof(escape));
   escape.Type = VIOGPU_RES_SET_SCANOUT_BLOB;
   escape.DataLength = sizeof(VIOGPU_RES_SET_SCANOUT_BLOB_REQ);
   escape.ResourceSetScanoutBlob.ResHandle = allocation;
   escape.ResourceSetScanoutBlob.ScanoutId = scanout_id;

   return device->escape(device, &escape, sizeof(escape));
}

static bool
yttrium_venus_raw_submit_with_event(struct yttrium_venus *venus,
                                    const void *data,
                                    size_t size,
                                    HANDLE completion_event,
                                    uint32_t *submit_path)
{
   if (!venus ||
       gdikmt_device_get_reset_status(venus->device) != PIPE_NO_RESET) {
      if (venus)
         venus->failed = true;
      return false;
   }

   struct gdikmt_context *ctx = venus->kmt_ctx;
   const size_t command_size = sizeof(VIOGPU_COMMAND_HDR) + size;

   if (submit_path)
      *submit_path = YTTRIUM_VENUS_SUBMIT_PATH_UNKNOWN;

   if (!ctx || !data || !size)
      return false;

   if (!completion_event) {
      if (submit_path)
         *submit_path = YTTRIUM_VENUS_SUBMIT_PATH_ESCAPE;
      const uint64_t escape_start_us =
         yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;
      const bool escaped = yttrium_venus_escape_submit(venus, data, size);
      yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RAW_SUBMIT,
                                 escaped ? 0 : 1, escape_start_us, NULL, size,
                                 0, 0, 0);
      if (escaped) {
         return true;
      }
      if (gdikmt_device_get_reset_status(venus->device) != PIPE_NO_RESET) {
         venus->failed = true;
         return false;
      }
      return false;
   } else if (submit_path) {
      *submit_path = YTTRIUM_VENUS_SUBMIT_PATH_RENDER_EVENT;
   }

   if (size > UINT_MAX || command_size > UINT_MAX ||
       !yttrium_venus_resize_submit_buffers(venus, command_size))
      return false;

   uint8_t *command = (uint8_t *)ctx->pCommandBuffer;
   VIOGPU_COMMAND_HDR *hdr = (VIOGPU_COMMAND_HDR *)command;
   hdr->type = VIOGPU_CMD_SUBMIT;
   hdr->size = (UINT)size;
   memcpy(command + sizeof(*hdr), data, size);

   struct gdikmt_render render;
   memset(&render, 0, sizeof(render));
   render.CommandLength = (UINT)command_size;
   render.CompletionEvent = completion_event;

   const uint64_t start_us =
      yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;
   NTSTATUS status = ctx->render(ctx, &render);
   yttrium_venus_trace_timing(YTTRIUM_TRACE_TIMING_VENUS_RAW_SUBMIT,
                              (uint32_t)status, start_us, NULL, size,
                              command_size, 0, 0);
   if (!NT_SUCCESS(status)) {
      YTTRIUM_LOG("yttrium: Venus raw submit failed status=0x%lx bytes=%llu\n",
                   status, (unsigned long long)size);
      return false;
   }

   return true;
}

bool
yttrium_venus_raw_submit(struct yttrium_venus *venus,
                         const void *data,
                         size_t size)
{
   const bool submitted =
      yttrium_venus_raw_submit_with_event(venus, data, size, NULL, NULL);
   if (!submitted)
      YTTRIUM_WARN("yttrium: ERROR: Venus2 ring creation submit failed owner=venus2-transport component=ring-create reason=escape-submit-failed action=fail-ring-creation path=escape-only render_fallback=disabled bytes=%llu\n",
                   (unsigned long long)size);
   return submitted;
}

bool
yttrium_venus_raw_submit_ring_kick(struct yttrium_venus *venus,
                                   const void *data,
                                   size_t size,
                                   uint32_t seqno,
                                   bool blocking,
                                   const char *label)
{
   const bool trace = yttrium_trace_sync_wait_is_enabled();
   const uint64_t start_us = trace ? yttrium_trace_now_us() : 0;
   uint32_t path = YTTRIUM_VENUS_SUBMIT_PATH_UNKNOWN;
   /* Ring doorbells are deliberately escape-only.  Falling back to RenderCb
    * reintroduces the scheduler path whose cost this transport avoids and can
    * reorder a wake behind work that depends on the ring being drained. */
   const bool submitted =
      yttrium_venus_raw_submit_with_event(venus, data, size, NULL, &path);
   const uint64_t elapsed_us = trace ? yttrium_trace_now_us() - start_us : 0;

   if (!submitted)
      YTTRIUM_WARN("yttrium: ERROR: Venus2 ring doorbell escape submit failed owner=venus2 seqno=%u label=%s render_fallback=disabled\n",
                   seqno, label ? label : "<unknown>");

   yttrium_trace_ring_kick(
      elapsed_us, seqno, submitted ? 0 : 1, path,
      yttrium_venus_submit_path_name(path),
      size > UINT32_MAX ? UINT32_MAX : (uint32_t)size,
      blocking ? 1 : 0, label);
   return submitted;
}

bool
yttrium_venus_raw_submit_sync(struct yttrium_venus *venus,
                              const void *data,
                              size_t size,
                              const char *label)
{
   HANDLE event = CreateEventA(NULL, TRUE, FALSE, NULL);
   if (!event) {
      YTTRIUM_LOG("yttrium: Venus sync raw submit event creation failed label=%s err=%lu\n",
                  label ? label : "<unknown>", GetLastError());
      return false;
   }

   const bool submitted =
      yttrium_venus_raw_submit_with_event(venus, data, size, event, NULL);
   if (!submitted) {
      CloseHandle(event);
      return false;
   }

   const uint64_t wait_start_us = yttrium_trace_now_us();
   const DWORD wait = WaitForSingleObject(event, YTTRIUM_VENUS_RING_WAIT_MS);
   const uint64_t wait_elapsed_us = yttrium_trace_now_us() - wait_start_us;
   CloseHandle(event);

   yttrium_venus_debug_sync_wait(
      venus, YTTRIUM_VENUS_SYNC_WAIT_RAW_SUBMIT, wait_elapsed_us,
      wait == WAIT_OBJECT_0 ? 0 : wait,
      label,
      size > UINT32_MAX ? UINT32_MAX : (uint32_t)size, 0,
      UINT32_MAX, size > UINT32_MAX ? UINT32_MAX : (uint32_t)size, 0);

   if (wait != WAIT_OBJECT_0) {
      YTTRIUM_LOG("yttrium: Venus sync raw submit wait failed label=%s wait=%lu err=%lu timeout_ms=%u\n",
                  label ? label : "<unknown>", wait, GetLastError(),
                  YTTRIUM_VENUS_RING_WAIT_MS);
      return false;
   }

   return true;
}

bool
yttrium_venus_bo_create(struct yttrium_venus *venus,
                        struct yttrium_venus_bo *bo,
                        uint64_t size)
{
   VIOGPU_CREATE_ALLOCATION_EXCHANGE alloc_exchange;
   VIOGPU_CREATE_RESOURCE_EXCHANGE res_exchange;
   struct gdikmt_createallocation create;
   D3DDDI_ALLOCATIONINFO alloc_info;

   memset(bo, 0, sizeof(*bo));
   memset(&alloc_exchange, 0, sizeof(alloc_exchange));
   memset(&res_exchange, 0, sizeof(res_exchange));
   memset(&create, 0, sizeof(create));
   memset(&alloc_info, 0, sizeof(alloc_info));

   bo->size = align64(MAX2(size, 1), 4096);

   alloc_exchange.ResourceOptions.target = PIPE_BUFFER;
   alloc_exchange.ResourceOptions.width =
      (ULONG)MIN2(bo->size, (uint64_t)UINT_MAX);
   alloc_exchange.ResourceOptions.height = 1;
   alloc_exchange.ResourceOptions.depth = 1;
   alloc_exchange.ResourceOptions.array_size = 1;
   alloc_exchange.Size = bo->size;
   alloc_exchange.BlobMem = VIRTGPU_BLOB_MEM_HOST3D;
   alloc_exchange.BlobFlags = VIRTGPU_BLOB_FLAG_USE_MAPPABLE;

   create.NumAllocations = 1;
   create.pAllocationInfo = &alloc_info;
   create.pPrivateDriverData = &res_exchange;
   create.PrivateDriverDataSize = sizeof(res_exchange);
   create.force_allocation_handle = true;

   alloc_info.pPrivateDriverData = &alloc_exchange;
   alloc_info.PrivateDriverDataSize = sizeof(alloc_exchange);

   NTSTATUS status = venus->device->createAllocation(venus->device, &create);
   if (!NT_SUCCESS(status)) {
      YTTRIUM_WARN("yttrium: internal allocation create failed owner=venus2-transport-create status=0x%lx size=0x%llx\n",
                   (unsigned long)status, (unsigned long long)bo->size);
      return false;
   }

   bo->hResource = create.hResource;
   bo->hAllocation = alloc_info.hAllocation;

   VIOGPU_ESCAPE resinfo;
   memset(&resinfo, 0, sizeof(resinfo));
   resinfo.Type = VIOGPU_RES_INFO;
   resinfo.DataLength = sizeof(VIOGPU_RES_INFO_REQ);
   resinfo.ResourceInfo.ResHandle = bo->hAllocation;

   status = venus->device->escape(venus->device, &resinfo, sizeof(resinfo));
   if (!NT_SUCCESS(status)) {
      YTTRIUM_WARN("yttrium: internal allocation query failed owner=venus2-transport-res-info status=0x%lx hAllocation=0x%lx hResource=%p\n",
                   (unsigned long)status, (unsigned long)bo->hAllocation,
                   bo->hResource);
      status = venus->device->destroyAllocation(
         venus->device, bo->hResource, bo->hAllocation);
      if (!NT_SUCCESS(status)) {
         YTTRIUM_WARN("yttrium: internal allocation destroy failed owner=venus2-transport-res-info-cleanup status=0x%lx hAllocation=0x%lx hResource=%p\n",
                      (unsigned long)status,
                      (unsigned long)bo->hAllocation, bo->hResource);
      }
      memset(bo, 0, sizeof(*bo));
      return false;
   }

   bo->res_id = resinfo.ResourceInfo.Id;

   VIOGPU_ESCAPE map;
   memset(&map, 0, sizeof(map));
   map.Type = VIOGPU_RES_MAP_BLOB;
   map.DataLength = sizeof(VIOGPU_RES_MAP_BLOB_REQ);
   map.ResourceMapBlob.ResHandle = bo->hAllocation;
   map.ResourceMapBlob.Size = bo->size;

   status = venus->device->escape(venus->device, &map, sizeof(map));
   if (!NT_SUCCESS(status) || !map.ResourceMapBlob.UserVa) {
      YTTRIUM_WARN("yttrium: internal allocation map failed owner=venus2-transport-map status=0x%lx hAllocation=0x%lx hResource=%p size=0x%llx map=%p\n",
                   (unsigned long)status,
                   (unsigned long)bo->hAllocation, bo->hResource,
                   (unsigned long long)bo->size,
                   (void *)(uintptr_t)map.ResourceMapBlob.UserVa);
      status = venus->device->destroyAllocation(
         venus->device, bo->hResource, bo->hAllocation);
      if (!NT_SUCCESS(status)) {
         YTTRIUM_WARN("yttrium: internal allocation destroy failed owner=venus2-transport-map-cleanup status=0x%lx hAllocation=0x%lx hResource=%p\n",
                      (unsigned long)status,
                      (unsigned long)bo->hAllocation, bo->hResource);
      }
      memset(bo, 0, sizeof(*bo));
      return false;
   }

   bo->map = (void *)(uintptr_t)map.ResourceMapBlob.UserVa;
   bo->map_info = map.ResourceMapBlob.MapInfo;
   memset(bo->map, 0, bo->size);

   YTTRIUM_LOG("yttrium: Venus BO hAllocation=0x%lx res_id=%u size=0x%llx map=%p\n",
                (unsigned long)bo->hAllocation, bo->res_id,
                (unsigned long long)bo->size, bo->map);
   return true;
}

void
yttrium_venus_bo_destroy(struct yttrium_venus *venus,
                         struct yttrium_venus_bo *bo)
{
   if (!bo->hAllocation)
      return;

   if (bo->map) {
      VIOGPU_ESCAPE unmap;
      memset(&unmap, 0, sizeof(unmap));
      unmap.Type = VIOGPU_RES_UNMAP_BLOB;
      unmap.DataLength = sizeof(VIOGPU_RES_UNMAP_BLOB_REQ);
      unmap.ResourceUnmapBlob.ResHandle = bo->hAllocation;
      NTSTATUS unmap_status =
         venus->device->escape(venus->device, &unmap, sizeof(unmap));
      if (!NT_SUCCESS(unmap_status)) {
         YTTRIUM_WARN("yttrium: internal allocation unmap failed owner=venus2-transport-bo-unmap status=0x%lx hAllocation=0x%lx hResource=%p res_id=%u size=0x%llx\n",
                      (unsigned long)unmap_status,
                      (unsigned long)bo->hAllocation, bo->hResource,
                      bo->res_id, (unsigned long long)bo->size);
      }
   }

   NTSTATUS status = venus->device->destroyAllocation(
      venus->device, bo->hResource, bo->hAllocation);
   if (!NT_SUCCESS(status)) {
      YTTRIUM_WARN("yttrium: internal allocation destroy failed owner=venus2-transport-bo-destroy status=0x%lx hAllocation=0x%lx hResource=%p\n",
                   (unsigned long)status, (unsigned long)bo->hAllocation,
                   bo->hResource);
   }
   memset(bo, 0, sizeof(*bo));
}

void
yttrium_venus_bo_forget_at_device_teardown(const char *label,
                                           struct yttrium_venus_bo *bo)
{
   if (!bo || !bo->hAllocation)
      return;

   YTTRIUM_LOG("yttrium: Venus %s BO cleanup deferred to device teardown hAllocation=0x%lx hResource=%p res_id=%u size=0x%llx map=%p\n",
               label ? label : "<unknown>",
               (unsigned long)bo->hAllocation,
               bo->hResource,
               bo->res_id,
               (unsigned long long)bo->size,
               bo->map);
   memset(bo, 0, sizeof(*bo));
}
