/*
 * Copyright 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 *
 * Copyright 2020 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <sys/types.h>
#include <string.h>
#include <io.h>
#include <windows.h>
#include <winternl.h>

#include "virtio/wddm/viogpu_wddm_driver.h"

#include "virtio/virtio-gpu/venus_hw.h"

#include "vn_renderer_d3dkmt.h"
#include "vn_common.h"
#include "util/os_file.h"
#include "util/sparse_array.h"

#include "vn_renderer_internal.h"

#ifndef VIRTGPU_BLOB_MEM_GUEST_VRAM
#define VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#endif

#define VIRTGPU_PCI_VENDOR_ID 0x1af4
#define VIRTGPU_PCI_DEVICE_ID 0x1050

#ifndef VIRTGPU_USE_MONITORED_FENCE
#define VIRTGPU_USE_MONITORED_FENCE 0
#endif
#ifndef VIRTGPU_LOG_SHMEM
#define VIRTGPU_LOG_SHMEM 0
#endif
#ifndef VIRTGPU_LOG_SYNC
#define VIRTGPU_LOG_SYNC 0
#endif
#ifndef VIRTGPU_SUBMIT_VIA_ESCAPE
#define VIRTGPU_SUBMIT_VIA_ESCAPE 1
#endif

#ifndef NDEBUG
static unsigned int vn_large_submit_warn_count;
#endif

struct virtgpu;

struct virtgpu_shmem {
   struct vn_renderer_shmem base;
   uint32_t alloc_handle;
};

struct virtgpu_bo {
   struct vn_renderer_bo base;
   uint32_t alloc_handle;
   uint32_t blob_flags;
};

struct virtgpu_sync {
   struct vn_renderer_sync base;

   /*
    * syncobj is in one of these states
    *
    *  - value N:      syncobj has a signaled fence chain with seqno N
    *  - pending N->M: syncobj has an unsignaled fence chain with seqno M
    *                 (which may point to another unsignaled fence chain with
    *                  seqno between N and M, and so on)
    *
    * TODO Do we want to use binary syncobjs?  They would be
    *
    *  - value 0: syncobj has no fence
    *  - value 1: syncobj has a signaled fence with seqno 0
    *
    * They are cheaper but require special care.
    */
   uint32_t syncobj_handle;
   volatile uint64_t *fence_cpu_va;
   bool is_monitored;
   uint64_t last_signaled;
};

struct virtgpu_resource_info {
   uint32_t alloc_handle;
   uint32_t res_handle;
   uint32_t size;
   uint32_t blob_mem;
};

struct virtgpu {
   struct vn_renderer base;

   struct vn_instance *instance;

   int fd;

   bool has_primary;
   int primary_major;
   int primary_minor;
   int render_major;
   int render_minor;

   uint32_t max_timeline_count;

   struct {
      uint32_t id;
      uint32_t version;
      struct virgl_renderer_capset_venus data;
   } capset;

   uint32_t shmem_blob_mem;
   uint32_t bo_blob_mem;

   /* Use allocation handles for indexing because resource ids grow
    * monotonically by default (see virtio_gpu_resource_id_get).
    */
   struct util_sparse_array shmem_array;
   struct util_sparse_array bo_array;

   mtx_t import_mutex;

   struct vn_renderer_shmem_cache shmem_cache;

   bool supports_cross_device;

   HMODULE gdi32;
   PFND3DKMT_QUERYADAPTERINFO pfnQueryAdapterInfo;
   PFND3DKMT_ESCAPE pfnEscape;
   PFND3DKMT_CREATEDEVICE pfnCreateDevice;
   PFND3DKMT_CREATEALLOCATION pfnCreateAllocation;
   PFND3DKMT_DESTROYDEVICE pfnDestroyDevice;
   PFND3DKMT_CREATECONTEXT pfnCreateContext;
   PFND3DKMT_DESTROYCONTEXT pfnDestroyContext;
   PFND3DKMT_RENDER pfnRender;
   PFND3DKMT_CREATESYNCHRONIZATIONOBJECT2 pfnCreateSynchronizationObject2;
   PFND3DKMT_DESTROYSYNCHRONIZATIONOBJECT pfnDestroySynchronizationObject;
   PFND3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU pfnWaitForSynchronizationObjectFromCpu;
   PFND3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMCPU pfnSignalSynchronizationObjectFromCpu;
   PFND3DKMT_WAITFORSYNCHRONIZATIONOBJECT2 pfnWaitForSynchronizationObject2;
   PFND3DKMT_SIGNALSYNCHRONIZATIONOBJECT2 pfnSignalSynchronizationObject2;
   PFND3DKMT_LOCK pfnLock;
   PFND3DKMT_UNLOCK pfnUnlock;
   PFND3DKMT_DESTROYALLOCATION pfnDestroyAllocation;
   PFND3DKMT_SHAREOBJECTS pfnShareObjects;
   PFND3DKMT_OPENRESOURCEFROMNTHANDLE pfnOpenResourceFromNtHandle;
   PFND3DKMT_OPENADAPTERFROMHDC pfnOpenAdapterFromHdc;
   PFND3DKMT_CLOSEADAPTER pfnCloseAdapter;
   D3DKMT_HANDLE hAdapter;
   D3DKMT_HANDLE hDevice;
   D3DKMT_HANDLE hContext;
   void *command_buffer;
   D3DDDI_ALLOCATIONLIST *allocation_list;
   D3DDDI_PATCHLOCATIONLIST *patch_location_list;
   UINT command_buffer_size;
   UINT allocation_list_size;
   UINT patch_location_list_size;
   VIOGPU_ADAPTERINFO adapter_info;
};

static int
timeout_to_poll_timeout(uint64_t timeout)
{
   const uint64_t ns_per_ms = 1000000;
   const uint64_t ms = (timeout + ns_per_ms - 1) / ns_per_ms;
   if (!ms && timeout)
      return -1;
   return ms <= INT_MAX ? (int)ms : -1;
}

static bool
virtgpu_submit_ctx_init(struct virtgpu *gpu);

static void
virtgpu_submit_ctx_fini(struct virtgpu *gpu);

#if VIRTGPU_SUBMIT_VIA_ESCAPE
static int
virtgpu_submit_escape(struct virtgpu *gpu, uint32_t cmd_type,
                      const void *payload, uint32_t payload_size);
#endif

static int
d3dkmt_submit_signal_syncs(struct virtgpu *gpu,
                        struct vn_renderer_sync *const *syncs,
                        const uint64_t *sync_values,
                        uint32_t sync_count)
{
   for (uint32_t i = 0; i < sync_count; i++) {
      struct virtgpu_sync *sync = (struct virtgpu_sync *)syncs[i];
      const uint64_t fence_value = sync_values[i];

      D3DKMT_SIGNALSYNCHRONIZATIONOBJECT2 signal = {0};
      signal.hContext = gpu->hContext;
      signal.ObjectCount = 1;
      signal.ObjectHandleArray[0] = (D3DKMT_HANDLE)sync->syncobj_handle;
      signal.Flags.Value = 0;
      signal.Fence.FenceValue = fence_value;

      NTSTATUS status = gpu->pfnSignalSynchronizationObject2(&signal);
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance,
                "failed to signal syncobj handle=%u value=%" PRIu64 " status=%lx",
                sync->syncobj_handle, fence_value, status);
         return -1;
      }
#if VIRTGPU_LOG_SYNC
      vn_log(gpu->instance,
             "signal_syncobj ok handle=%u value=%" PRIu64,
             sync->syncobj_handle, fence_value);
#endif

      if (fence_value > sync->last_signaled)
         sync->last_signaled = fence_value;
   }

   return 0;
}

static uint32_t *
d3dkmt_submit_alloc_handles(struct vn_renderer_bo *const *bos, uint32_t bo_count)
{
   uint32_t *alloc_handles = malloc(sizeof(*alloc_handles) * bo_count);
   if (!alloc_handles)
      return NULL;

   for (uint32_t i = 0; i < bo_count; i++) {
      struct virtgpu_bo *bo = (struct virtgpu_bo *)bos[i];
      alloc_handles[i] = bo->alloc_handle;
   }

   return alloc_handles;
}

static int
d3dkmt_submit_resize(struct virtgpu *gpu, size_t cmd_size, uint32_t bo_count)
{
   const bool needs_resize =
      cmd_size > gpu->command_buffer_size ||
      bo_count > gpu->allocation_list_size ||
      bo_count > gpu->patch_location_list_size;
   if (!needs_resize)
      return 0;

   const UINT needed_cmd = (UINT)cmd_size;
   const UINT needed_list = bo_count;
   const UINT grow_cmd =
      gpu->command_buffer_size + (gpu->command_buffer_size >> 1);
   const UINT grow_list =
      gpu->allocation_list_size + (gpu->allocation_list_size >> 1);
   const UINT new_cmd_size = MAX2(needed_cmd + 0x100, grow_cmd);
   const UINT new_list_size = MAX2(needed_list + 16, grow_list);

   VIOGPU_COMMAND_HDR *hdr = (VIOGPU_COMMAND_HDR *)gpu->command_buffer;
   hdr->type = VIOGPU_CMD_NOP;
   hdr->size = 0;

   D3DKMT_RENDER resize = {0};
   resize.hContext = gpu->hContext;
   resize.CommandOffset = 0;
   resize.CommandLength = sizeof(*hdr);
   resize.Flags.ResizeCommandBuffer = TRUE;
   resize.Flags.ResizeAllocationList = TRUE;
   resize.Flags.ResizePatchLocationList = TRUE;
   resize.NewCommandBufferSize = new_cmd_size;
   resize.NewAllocationListSize = new_list_size;
   resize.NewPatchLocationListSize = new_list_size;

   NTSTATUS status = gpu->pfnRender(&resize);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to resize submit buffers: %lx", status);
      return -1;
   }

   if (resize.pNewCommandBuffer)
      gpu->command_buffer = resize.pNewCommandBuffer;
   if (resize.NewCommandBufferSize)
      gpu->command_buffer_size = resize.NewCommandBufferSize;
   if (resize.pNewAllocationList)
      gpu->allocation_list = resize.pNewAllocationList;
   if (resize.NewAllocationListSize)
      gpu->allocation_list_size = resize.NewAllocationListSize;
   if (resize.pNewPatchLocationList)
      gpu->patch_location_list = resize.pNewPatchLocationList;
   if (resize.NewPatchLocationListSize)
      gpu->patch_location_list_size = resize.NewPatchLocationListSize;

   return 0;
}

static int
d3dkmt_submit(struct virtgpu *gpu, const struct vn_renderer_submit *submit)
{
#if VIRTGPU_SUBMIT_VIA_ESCAPE
   if (!virtgpu_submit_ctx_init(gpu)) {
      vn_log(gpu->instance, "failed to init D3DKMT submit context");
      return -1;
   }
   int ret = 0;
   for (uint32_t i = 0; i < submit->batch_count; i++) {
      const struct vn_renderer_submit_batch *batch = &submit->batches[i];
#ifndef NDEBUG
      if (batch->cs_size > 256 && vn_large_submit_warn_count < 5) {
         vn_log(gpu->instance,
                "large submit (escape) cs_size=%zu batch=%u",
                (size_t)batch->cs_size, i);
         vn_large_submit_warn_count++;
      }
#endif
      if (virtgpu_submit_escape(gpu, VIOGPU_CMD_SUBMIT,
                                batch->cs_data, batch->cs_size)) {
         ret = -1;
         break;
      }
      if (batch->sync_count) {
         ret = d3dkmt_submit_signal_syncs(gpu, batch->syncs, batch->sync_values,
                                          batch->sync_count);
         if (ret)
            break;
      }
   }
   return ret;
#else
   if (!virtgpu_submit_ctx_init(gpu)) {
      vn_log(gpu->instance, "failed to init D3DKMT submit context");
      return -1;
   }

   /* TODO pass a prebuilt handle list to avoid malloc/loop */
   uint32_t *alloc_handles = NULL;
   if (submit->bo_count) {
      alloc_handles =
         d3dkmt_submit_alloc_handles(submit->bos, submit->bo_count);
      if (!alloc_handles)
         return -1;
   }

   assert(submit->batch_count);

   int ret = 0;
   for (uint32_t i = 0; i < submit->batch_count; i++) {
      const struct vn_renderer_submit_batch *batch = &submit->batches[i];

      const size_t cmd_size = sizeof(VIOGPU_COMMAND_HDR) + batch->cs_size;
      if (d3dkmt_submit_resize(gpu, cmd_size, submit->bo_count)) {
         ret = -1;
         break;
      }

      VIOGPU_COMMAND_HDR *hdr = (VIOGPU_COMMAND_HDR *)gpu->command_buffer;
      hdr->type = VIOGPU_CMD_SUBMIT;
      hdr->size = batch->cs_size;
      memcpy((uint8_t *)gpu->command_buffer + sizeof(*hdr), batch->cs_data,
             batch->cs_size);

      for (uint32_t j = 0; j < submit->bo_count; j++) {
         memset(&gpu->allocation_list[j], 0, sizeof(*gpu->allocation_list));
         gpu->allocation_list[j].hAllocation =
            (D3DKMT_HANDLE)alloc_handles[j];
         memset(&gpu->patch_location_list[j], 0,
                sizeof(*gpu->patch_location_list));
         gpu->patch_location_list[j].AllocationIndex = j;
      }

      D3DKMT_RENDER render = {0};
      render.hContext = gpu->hContext;
      render.CommandOffset = 0;
      render.CommandLength = (UINT)cmd_size;
      render.AllocationCount = submit->bo_count;
      render.PatchLocationCount = submit->bo_count;

      NTSTATUS status = gpu->pfnRender(&render);
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance, "failed to submit cmd: %lx", status);
         ret = -1;
         break;
      }

      if (render.pNewCommandBuffer)
         gpu->command_buffer = render.pNewCommandBuffer;
      if (render.NewCommandBufferSize)
         gpu->command_buffer_size = render.NewCommandBufferSize;
      if (render.pNewAllocationList)
         gpu->allocation_list = render.pNewAllocationList;
      if (render.NewAllocationListSize)
         gpu->allocation_list_size = render.NewAllocationListSize;
      if (render.pNewPatchLocationList)
         gpu->patch_location_list = render.pNewPatchLocationList;
      if (render.NewPatchLocationListSize)
         gpu->patch_location_list_size = render.NewPatchLocationListSize;

      if (batch->sync_count) {
         ret = d3dkmt_submit_signal_syncs(gpu, batch->syncs, batch->sync_values,
                                       batch->sync_count);
         if (ret)
            break;
      }
   }

   free(alloc_handles);
   return ret;
#endif
}

static bool
virtgpu_d3dkmt_escape(struct virtgpu *gpu, VIOGPU_ESCAPE *esc)
{
   if (!gpu->pfnEscape || !gpu->hAdapter || !gpu->hDevice)
      return false;

   D3DKMT_ESCAPE args = {0};
   args.hAdapter = gpu->hAdapter;
   args.hDevice = gpu->hDevice;
   args.pPrivateDriverData = esc;
   args.PrivateDriverDataSize = sizeof(*esc);

   NTSTATUS status = gpu->pfnEscape(&args);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "escape type=0x%x failed status=0x%lx", esc ? esc->Type : 0, status);
      return false;
   }
   return true;
}

#if VIRTGPU_SUBMIT_VIA_ESCAPE
static bool
virtgpu_d3dkmt_escape_data(struct virtgpu *gpu, VIOGPU_ESCAPE *esc, size_t size)
{
   if (!gpu->pfnEscape || !gpu->hAdapter || !gpu->hDevice)
      return false;

   D3DKMT_ESCAPE args = {0};
   args.hAdapter = gpu->hAdapter;
   args.hDevice = gpu->hDevice;
   args.pPrivateDriverData = esc;
   args.PrivateDriverDataSize = (UINT)size;

   NTSTATUS status = gpu->pfnEscape(&args);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "escape type=0x%x failed status=0x%lx", esc ? esc->Type : 0, status);
      return false;
   }
   return true;
}

static int
virtgpu_submit_escape(struct virtgpu *gpu, uint32_t cmd_type,
                      const void *payload, uint32_t payload_size)
{
   const size_t req_size = sizeof(VIOGPU_SUBMIT_CMD_REQ) + payload_size;
   if (req_size > UINT16_MAX) {
      vn_log(gpu->instance, "escape submit too large: %zu", req_size);
      return -1;
   }

   const size_t total = sizeof(VIOGPU_ESCAPE) + req_size;
   uint8_t *buf = calloc(1, total);
   if (!buf)
      return -1;

   VIOGPU_ESCAPE *esc = (VIOGPU_ESCAPE *)buf;
   esc->Type = VIOGPU_SUBMIT_CMD;
   esc->DataLength = (USHORT)req_size;

   VIOGPU_SUBMIT_CMD_REQ *req = (VIOGPU_SUBMIT_CMD_REQ *)(esc + 1);
   req->hContext = gpu->hContext;
   req->CmdType = cmd_type;
   req->CmdSize = payload_size;

   if (payload_size)
      memcpy(req + 1, payload, payload_size);

   const bool ok = virtgpu_d3dkmt_escape_data(gpu, esc, total);
   free(buf);
   return ok ? 0 : -1;
}
#endif

static bool
virtgpu_submit_ctx_init(struct virtgpu *gpu)
{
   if (gpu->hContext)
      return true;

   if (!gpu->pfnCreateContext || !gpu->pfnRender)
      return false;

   D3DKMT_CREATECONTEXT create = {0};
   create.hDevice = gpu->hDevice;
   NTSTATUS status = gpu->pfnCreateContext(&create);
   if (!NT_SUCCESS(status))
      return false;

   gpu->hContext = create.hContext;
   gpu->command_buffer = create.pCommandBuffer;
   gpu->command_buffer_size = create.CommandBufferSize;
   gpu->allocation_list = create.pAllocationList;
   gpu->allocation_list_size = create.AllocationListSize;
   gpu->patch_location_list = create.pPatchLocationList;
   gpu->patch_location_list_size = create.PatchLocationListSize;

   VIOGPU_COMMAND_HDR *hdr = (VIOGPU_COMMAND_HDR *)gpu->command_buffer;
   hdr->type = VIOGPU_CMD_NOP;
   hdr->size = 0;

   D3DKMT_RENDER render = {0};
   render.hContext = gpu->hContext;
   render.CommandOffset = 0;
   render.CommandLength = sizeof(*hdr);
   render.AllocationCount = 0;
   render.PatchLocationCount = 0;
   render.Flags.ResizeCommandBuffer = TRUE;
   render.Flags.ResizeAllocationList = TRUE;
   render.Flags.ResizePatchLocationList = TRUE;
   render.NewCommandBufferSize = 0x10000;
   render.NewAllocationListSize = 1024;
   render.NewPatchLocationListSize = 1024;

   status = gpu->pfnRender(&render);
   if (!NT_SUCCESS(status)) {
      D3DKMT_DESTROYCONTEXT destroy = {0};
      destroy.hContext = gpu->hContext;
      gpu->pfnDestroyContext(&destroy);
      gpu->hContext = 0;
      return false;
   }

   if (render.pNewCommandBuffer)
      gpu->command_buffer = render.pNewCommandBuffer;
   if (render.NewCommandBufferSize)
      gpu->command_buffer_size = render.NewCommandBufferSize;
   if (render.pNewAllocationList)
      gpu->allocation_list = render.pNewAllocationList;
   if (render.NewAllocationListSize)
      gpu->allocation_list_size = render.NewAllocationListSize;
   if (render.pNewPatchLocationList)
      gpu->patch_location_list = render.pNewPatchLocationList;
   if (render.NewPatchLocationListSize)
      gpu->patch_location_list_size = render.NewPatchLocationListSize;

   return true;
}

static void
virtgpu_submit_ctx_fini(struct virtgpu *gpu)
{
   if (gpu->pfnDestroyContext && gpu->hContext) {
      D3DKMT_DESTROYCONTEXT destroy = {0};
      destroy.hContext = gpu->hContext;
      gpu->pfnDestroyContext(&destroy);
   }
   gpu->hContext = 0;
   gpu->command_buffer = NULL;
   gpu->command_buffer_size = 0;
   gpu->allocation_list = NULL;
   gpu->allocation_list_size = 0;
   gpu->patch_location_list = NULL;
   gpu->patch_location_list_size = 0;
}

static uint64_t
virtgpu_getparam(struct virtgpu *gpu, uint64_t param)
{
   VIOGPU_ESCAPE esc = {0};
   esc.Type = VIOGPU_GET_CAPS;
   esc.DataLength = sizeof(VIOGPU_PARAM_REQ);
   esc.Parameter.ParamId = (ULONG)param;
   esc.Parameter.Value = 0;

   return virtgpu_d3dkmt_escape(gpu, &esc) ? esc.Parameter.Value : 0;
}

static int
virtgpu_get_caps(struct virtgpu *gpu,
                       uint32_t id,
                       uint32_t version,
                       void *capset,
                       size_t capset_size)
{
   VIOGPU_ESCAPE esc = {0};
   esc.Type = VIOGPU_GET_CAPS;
   esc.DataLength = sizeof(VIOGPU_CAPSET_REQ);
   esc.Capset.CapsetId = id;
   esc.Capset.Version = version;
   esc.Capset.Size = (ULONG)capset_size;
   esc.Capset.Capset = (ULONG64)(UINT_PTR)capset;

   return virtgpu_d3dkmt_escape(gpu, &esc) ? 0 : -1;
}

static int
virtgpu_context_init(struct virtgpu *gpu, uint32_t capset_id)
{
   VIOGPU_ESCAPE esc = {0};
   esc.Type = VIOGPU_CTX_INIT;
   esc.DataLength = sizeof(VIOGPU_CTX_INIT_REQ);
   esc.CtxInit.CapsetID = capset_id;
   return virtgpu_d3dkmt_escape(gpu, &esc) ? 0 : -1;
}

static void
virtgpu_d3dkmt_gem_close(struct virtgpu *gpu, uint32_t alloc_handle);

static uint32_t
virtgpu_d3dkmt_resource_create_blob(struct virtgpu *gpu,
                                   uint32_t blob_mem,
                                   uint32_t blob_flags,
                                   size_t blob_size,
                                   uint64_t blob_id,
                                   uint32_t *res_id)
{
#ifdef SIMULATE_BO_SIZE_FIX
   blob_size = align64(blob_size, 4096);
#endif
   (void)blob_mem;
   (void)blob_id;

   VIOGPU_CREATE_ALLOCATION_EXCHANGE alloc_exchange;
   VIOGPU_CREATE_RESOURCE_EXCHANGE res_exchange;
   memset(&alloc_exchange, 0, sizeof(alloc_exchange));
   memset(&res_exchange, 0, sizeof(res_exchange));

   alloc_exchange.ResourceOptions.target = 0; /* PIPE_BUFFER */
   alloc_exchange.ResourceOptions.format = 0;
   alloc_exchange.ResourceOptions.bind = 0;
   alloc_exchange.ResourceOptions.width = (ULONG)blob_size;
   alloc_exchange.ResourceOptions.height = 1;
   alloc_exchange.ResourceOptions.depth = 1;
   alloc_exchange.ResourceOptions.array_size = 1;
   alloc_exchange.ResourceOptions.last_level = 0;
   alloc_exchange.ResourceOptions.nr_samples = 0;
   alloc_exchange.ResourceOptions.flags = blob_flags;
   alloc_exchange.Size = blob_size;
   /* MESA-VIRTIO PATCH BEGIN: blob resource fields */
   alloc_exchange.BlobId = blob_id;
   alloc_exchange.BlobMem = blob_mem;
   alloc_exchange.BlobFlags = blob_flags;
   /* MESA-VIRTIO PATCH END */

   D3DDDI_ALLOCATIONINFO alloc_info;
   memset(&alloc_info, 0, sizeof(alloc_info));
   alloc_info.pPrivateDriverData = &alloc_exchange;
   alloc_info.PrivateDriverDataSize = sizeof(alloc_exchange);

   D3DKMT_CREATEALLOCATION create = {0};
   create.hDevice = gpu->hDevice;
   create.NumAllocations = 1;
   create.pAllocationInfo = &alloc_info;
   create.pPrivateDriverData = &res_exchange;
   create.PrivateDriverDataSize = sizeof(res_exchange);
   create.Flags.CreateResource = TRUE;

   NTSTATUS status = gpu->pfnCreateAllocation(&create);
   if (!NT_SUCCESS(status))
      vn_log(gpu->instance,
             "pfnCreateAllocation failed status=0x%lx tid=%lu blob_id=0x%llx blob_mem=0x%x blob_flags=0x%x blob_size=0x%zx",
             status, GetCurrentThreadId(), blob_id, blob_mem, blob_flags,
             blob_size);

   if (!NT_SUCCESS(status))
      return 0;

   VIOGPU_ESCAPE resinfo = {0};
   resinfo.Type = VIOGPU_RES_INFO;
   resinfo.DataLength = sizeof(VIOGPU_RES_INFO_REQ);
   resinfo.ResourceInfo.ResHandle = alloc_info.hAllocation;
   if (!virtgpu_d3dkmt_escape(gpu, &resinfo)) {
      vn_log(gpu->instance,"virtgpu_d3dkmt_escape failed");
      virtgpu_d3dkmt_gem_close(gpu, (uint32_t)alloc_info.hAllocation);
      return 0;
   }

   *res_id = resinfo.ResourceInfo.Id;
   return (uint32_t)alloc_info.hAllocation;
}

static int
virtgpu_d3dkmt_resource_info(struct virtgpu *gpu,
                            uint32_t alloc_handle,
                            struct virtgpu_resource_info *info)
{
   *info = (struct virtgpu_resource_info){
      .alloc_handle = alloc_handle,
      .res_handle = 0,
      .size = 0,
      .blob_mem = 0,
   };

   VIOGPU_ESCAPE resinfo = {0};
   resinfo.Type = VIOGPU_RES_INFO;
   resinfo.DataLength = sizeof(VIOGPU_RES_INFO_REQ);
   resinfo.ResourceInfo.ResHandle = (D3DKMT_HANDLE)alloc_handle;
   if (!virtgpu_d3dkmt_escape(gpu, &resinfo))
      return -1;

   info->res_handle = resinfo.ResourceInfo.Id;
   return 0;
}

static int
virtgpu_d3dkmt_resource_map_blob(struct virtgpu *gpu,
                                uint32_t alloc_handle,
                                uint64_t offset,
                                uint64_t size,
                                uint32_t *map_info,
                                void **user_va)
{
   VIOGPU_ESCAPE esc = {0};
   esc.Type = VIOGPU_RES_MAP_BLOB;
   esc.DataLength = sizeof(VIOGPU_RES_MAP_BLOB_REQ);
   esc.ResourceMapBlob.ResHandle = (D3DKMT_HANDLE)alloc_handle;
   esc.ResourceMapBlob.Offset = offset;
   esc.ResourceMapBlob.Size = size;
   esc.ResourceMapBlob.UserVa = 0;
   esc.ResourceMapBlob.MapInfo = 0;

   if (!virtgpu_d3dkmt_escape(gpu, &esc))
       return -1;

   if (map_info)
      *map_info = esc.ResourceMapBlob.MapInfo;
   if (user_va)
      *user_va = (void *)(uintptr_t)esc.ResourceMapBlob.UserVa;

   return 0;
}

static int
virtgpu_d3dkmt_resource_unmap_blob(struct virtgpu *gpu, uint32_t alloc_handle)
{
   VIOGPU_ESCAPE esc = {0};
   esc.Type = VIOGPU_RES_UNMAP_BLOB;
   esc.DataLength = sizeof(VIOGPU_RES_UNMAP_BLOB_REQ);
   esc.ResourceUnmapBlob.ResHandle = (D3DKMT_HANDLE)alloc_handle;

   return virtgpu_d3dkmt_escape(gpu, &esc) ? 0 : -1;
}

static void
virtgpu_d3dkmt_gem_close(struct virtgpu *gpu, uint32_t alloc_handle)
{
   if (!gpu->pfnDestroyAllocation || !alloc_handle)
      return;

   D3DKMT_HANDLE handle = (D3DKMT_HANDLE)alloc_handle;
   D3DKMT_DESTROYALLOCATION destroy = {0};
   destroy.hDevice = gpu->hDevice;
   destroy.hResource = 0;
   destroy.phAllocationList = &handle;
   destroy.AllocationCount = 1;

   ASSERTED NTSTATUS status = gpu->pfnDestroyAllocation(&destroy);
   assert(NT_SUCCESS(status));
}

static int
virtgpu_d3dkmt_prime_handle_to_fd(struct virtgpu *gpu,
                                 uint32_t alloc_handle,
                                 bool mappable)
{
   if (!gpu->pfnShareObjects || !alloc_handle)
      return -1;

   D3DKMT_HANDLE handle = (D3DKMT_HANDLE)alloc_handle;
   OBJECT_ATTRIBUTES attrs = {0};
   attrs.Length = sizeof(attrs);
   HANDLE shared = NULL;
   NTSTATUS status = gpu->pfnShareObjects(1, &handle, &attrs,
                                          SHARED_ALLOCATION_ALL_ACCESS,
                                          &shared);
   if (!NT_SUCCESS(status) || !shared)
      return -1;

   const intptr_t os_handle = (intptr_t)shared;
   int fd = _open_osfhandle(os_handle, 0);
   if (fd < 0)
      CloseHandle(shared);

   return fd;
}

static uint32_t
virtgpu_d3dkmt_prime_fd_to_handle(struct virtgpu *gpu, int fd)
{
   if (!gpu->pfnOpenResourceFromNtHandle)
      return 0;

   HANDLE shared = (HANDLE)_get_osfhandle(fd);
   if (shared == INVALID_HANDLE_VALUE)
      return 0;

   D3DDDI_OPENALLOCATIONINFO2 alloc_info = {0};
   D3DKMT_OPENRESOURCEFROMNTHANDLE open = {0};
   open.hDevice = gpu->hDevice;
   open.hNtHandle = shared;
   open.NumAllocations = 1;
   open.pOpenAllocationInfo2 = &alloc_info;

   NTSTATUS status = gpu->pfnOpenResourceFromNtHandle(&open);
   if (!NT_SUCCESS(status))
      return 0;

   return (uint32_t)alloc_info.hAllocation;
}

static bool
virtgpu_load_d3dkmt(struct virtgpu *gpu)
{
   gpu->gdi32 = LoadLibraryA("gdi32.dll");
   if (!gpu->gdi32)
      return false;

   gpu->pfnQueryAdapterInfo = (PFND3DKMT_QUERYADAPTERINFO)GetProcAddress(
      gpu->gdi32, "D3DKMTQueryAdapterInfo");
   gpu->pfnEscape = (PFND3DKMT_ESCAPE)GetProcAddress(gpu->gdi32, "D3DKMTEscape");
   gpu->pfnCreateDevice =
      (PFND3DKMT_CREATEDEVICE)GetProcAddress(gpu->gdi32, "D3DKMTCreateDevice");
   gpu->pfnCreateAllocation =
      (PFND3DKMT_CREATEALLOCATION)GetProcAddress(gpu->gdi32,
                                                 "D3DKMTCreateAllocation");
   gpu->pfnDestroyDevice = (PFND3DKMT_DESTROYDEVICE)GetProcAddress(
      gpu->gdi32, "D3DKMTDestroyDevice");
   gpu->pfnCreateContext =
      (PFND3DKMT_CREATECONTEXT)GetProcAddress(gpu->gdi32, "D3DKMTCreateContext");
   gpu->pfnDestroyContext = (PFND3DKMT_DESTROYCONTEXT)GetProcAddress(
      gpu->gdi32, "D3DKMTDestroyContext");
   gpu->pfnRender = (PFND3DKMT_RENDER)GetProcAddress(gpu->gdi32, "D3DKMTRender");
   gpu->pfnCreateSynchronizationObject2 =
      (PFND3DKMT_CREATESYNCHRONIZATIONOBJECT2)GetProcAddress(
         gpu->gdi32, "D3DKMTCreateSynchronizationObject2");
   gpu->pfnDestroySynchronizationObject =
      (PFND3DKMT_DESTROYSYNCHRONIZATIONOBJECT)GetProcAddress(
         gpu->gdi32, "D3DKMTDestroySynchronizationObject");
   gpu->pfnWaitForSynchronizationObjectFromCpu =
      (PFND3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU)GetProcAddress(
         gpu->gdi32, "D3DKMTWaitForSynchronizationObjectFromCpu");
   gpu->pfnSignalSynchronizationObjectFromCpu =
      (PFND3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMCPU)GetProcAddress(
         gpu->gdi32, "D3DKMTSignalSynchronizationObjectFromCpu");
   gpu->pfnWaitForSynchronizationObject2 =
      (PFND3DKMT_WAITFORSYNCHRONIZATIONOBJECT2)GetProcAddress(
         gpu->gdi32, "D3DKMTWaitForSynchronizationObject2");
   gpu->pfnSignalSynchronizationObject2 =
      (PFND3DKMT_SIGNALSYNCHRONIZATIONOBJECT2)GetProcAddress(
         gpu->gdi32, "D3DKMTSignalSynchronizationObject2");
   gpu->pfnLock = (PFND3DKMT_LOCK)GetProcAddress(gpu->gdi32, "D3DKMTLock");
   gpu->pfnUnlock =
      (PFND3DKMT_UNLOCK)GetProcAddress(gpu->gdi32, "D3DKMTUnlock");
   gpu->pfnDestroyAllocation = (PFND3DKMT_DESTROYALLOCATION)GetProcAddress(
      gpu->gdi32, "D3DKMTDestroyAllocation");
   gpu->pfnShareObjects = (PFND3DKMT_SHAREOBJECTS)GetProcAddress(
      gpu->gdi32, "D3DKMTShareObjects");
   gpu->pfnOpenResourceFromNtHandle =
      (PFND3DKMT_OPENRESOURCEFROMNTHANDLE)GetProcAddress(
         gpu->gdi32, "D3DKMTOpenResourceFromNtHandle");
   gpu->pfnOpenAdapterFromHdc =
      (PFND3DKMT_OPENADAPTERFROMHDC)GetProcAddress(gpu->gdi32,
                                                   "D3DKMTOpenAdapterFromHdc");
   gpu->pfnCloseAdapter = (PFND3DKMT_CLOSEADAPTER)GetProcAddress(
      gpu->gdi32, "D3DKMTCloseAdapter");

   if (!gpu->pfnQueryAdapterInfo || !gpu->pfnEscape || !gpu->pfnCreateDevice ||
       !gpu->pfnCreateAllocation || !gpu->pfnDestroyDevice ||
       !gpu->pfnCreateContext ||
       !gpu->pfnDestroyContext || !gpu->pfnRender ||
       !gpu->pfnCreateSynchronizationObject2 ||
       !gpu->pfnDestroySynchronizationObject ||
       !gpu->pfnWaitForSynchronizationObject2 ||
       !gpu->pfnSignalSynchronizationObject2 ||
       !gpu->pfnDestroyAllocation || !gpu->pfnShareObjects ||
       !gpu->pfnOpenResourceFromNtHandle ||
       !gpu->pfnOpenAdapterFromHdc ||
       !gpu->pfnCloseAdapter) {
      FreeLibrary(gpu->gdi32);
      gpu->gdi32 = NULL;
      return false;
   }

   return true;
}

static void
virtgpu_unload_d3dkmt(struct virtgpu *gpu)
{
   if (gpu->gdi32) {
      FreeLibrary(gpu->gdi32);
      gpu->gdi32 = NULL;
   }
   gpu->pfnQueryAdapterInfo = NULL;
   gpu->pfnEscape = NULL;
   gpu->pfnCreateDevice = NULL;
   gpu->pfnCreateAllocation = NULL;
   gpu->pfnDestroyDevice = NULL;
   gpu->pfnCreateContext = NULL;
   gpu->pfnDestroyContext = NULL;
   gpu->pfnRender = NULL;
   gpu->pfnCreateSynchronizationObject2 = NULL;
   gpu->pfnDestroySynchronizationObject = NULL;
   gpu->pfnWaitForSynchronizationObjectFromCpu = NULL;
   gpu->pfnSignalSynchronizationObjectFromCpu = NULL;
   gpu->pfnWaitForSynchronizationObject2 = NULL;
   gpu->pfnSignalSynchronizationObject2 = NULL;
   gpu->pfnLock = NULL;
   gpu->pfnUnlock = NULL;
   gpu->pfnDestroyAllocation = NULL;
   gpu->pfnShareObjects = NULL;
   gpu->pfnOpenResourceFromNtHandle = NULL;
   gpu->pfnOpenAdapterFromHdc = NULL;
   gpu->pfnCloseAdapter = NULL;
}

static void *
virtgpu_d3dkmt_map(struct virtgpu *gpu, uint32_t alloc_handle, size_t size)
{
   uint32_t map_info = 0;
   void *user_va = NULL;
   if (virtgpu_d3dkmt_resource_map_blob(gpu, alloc_handle, 0, size, &map_info,
                                       &user_va)) {
      vn_log(gpu->instance,"virtgpu_d3dkmt_resource_map_blob failed");
      return NULL;
   }

   if (!user_va) {
      vn_log(gpu->instance, "ResourceMapBlob returned null user_va");
      return NULL;
   }
   return user_va;
}

static uint32_t
virtgpu_syncobj_create(struct virtgpu *gpu,
                       volatile uint64_t **out_cpu_va,
                       bool *out_monitored,
                       uint64_t initial_val)
{
   if (!gpu->pfnCreateSynchronizationObject2 || !gpu->hDevice)
      return 0;

   D3DKMT_CREATESYNCHRONIZATIONOBJECT2 create = {0};
   create.hDevice = gpu->hDevice;
   create.Info.Flags.Value = 0;

   NTSTATUS status = STATUS_INVALID_PARAMETER;
#if VIRTGPU_USE_MONITORED_FENCE
   if (gpu->pfnSignalSynchronizationObjectFromCpu &&
       gpu->pfnWaitForSynchronizationObjectFromCpu) {
      create.Info.Type = D3DDDI_MONITORED_FENCE;
      create.Info.MonitoredFence.InitialFenceValue = initial_val;
      create.Info.MonitoredFence.EngineAffinity = 0;

      status = gpu->pfnCreateSynchronizationObject2(&create);
      if (NT_SUCCESS(status)) {
         if (out_cpu_va) {
            *out_cpu_va = (volatile uint64_t *)
               create.Info.MonitoredFence.FenceValueCPUVirtualAddress;
         }
         if (out_monitored)
            *out_monitored = true;
         return (uint32_t)create.hSyncObject;
      }
   }

   vn_log(gpu->instance, "syncobj_create monitored failed: %lx; trying fence",
          status);
#endif

   create = (D3DKMT_CREATESYNCHRONIZATIONOBJECT2){0};
   create.hDevice = gpu->hDevice;
   create.Info.Type = D3DDDI_FENCE;
   create.Info.Flags.Value = 0;
   create.Info.Fence.FenceValue = initial_val;

   status = gpu->pfnCreateSynchronizationObject2(&create);
   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "syncobj_create fence failed: %lx", status);
      return 0;
   }

   if (out_cpu_va)
      *out_cpu_va = NULL;
   if (out_monitored)
      *out_monitored = false;
   return (uint32_t)create.hSyncObject;
}

static void
virtgpu_syncobj_destroy(struct virtgpu *gpu, uint32_t syncobj_handle)
{
   if (!gpu->pfnDestroySynchronizationObject || !syncobj_handle)
      return;

   D3DKMT_DESTROYSYNCHRONIZATIONOBJECT destroy = {0};
   destroy.hSyncObject = (D3DKMT_HANDLE)syncobj_handle;
   gpu->pfnDestroySynchronizationObject(&destroy);
}

static int
virtgpu_syncobj_export(struct virtgpu *gpu,
                       uint32_t syncobj_handle,
                       bool sync_file)
{
   (void)gpu;
   (void)syncobj_handle;
   (void)sync_file;
   return -1;
}

static int
virtgpu_syncobj_signal(struct virtgpu *gpu,
                       struct virtgpu_sync *sync,
                       uint64_t point,
                       bool allow_rewind)
{
   if (!sync || !sync->syncobj_handle)
      return -1;

   if (sync->is_monitored) {
#if VIRTGPU_USE_MONITORED_FENCE
      if (!gpu->pfnSignalSynchronizationObjectFromCpu)
         return -1;

      D3DKMT_HANDLE handles[1] = {(D3DKMT_HANDLE)sync->syncobj_handle};
      const UINT64 values[1] = {point};

      D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMCPU signal = {0};
      signal.hDevice = gpu->hDevice;
      signal.ObjectCount = 1;
      signal.ObjectHandleArray = handles;
      signal.FenceValueArray = values;
      signal.Flags.Value = 0;
      if (allow_rewind)
         signal.Flags.AllowFenceRewind = 1;

      const NTSTATUS status = gpu->pfnSignalSynchronizationObjectFromCpu(&signal);
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance,
                "syncobj_signal failed handle=%u point=%" PRIu64 " status=%lx",
                sync->syncobj_handle, point, status);
         return -1;
      }
#if VIRTGPU_LOG_SYNC
      vn_log(gpu->instance,
             "syncobj_signal ok handle=%u point=%" PRIu64 " monitored=1",
             sync->syncobj_handle, point);
#endif
#else
      return -1;
#endif
   } else {
      if (!gpu->pfnSignalSynchronizationObject2)
         return -1;

      D3DKMT_SIGNALSYNCHRONIZATIONOBJECT2 signal = {0};
      signal.hContext = gpu->hContext;
      signal.ObjectCount = 1;
      signal.ObjectHandleArray[0] = (D3DKMT_HANDLE)sync->syncobj_handle;
      signal.Flags.Value = 0;
      signal.Flags.SignalAtSubmission = 0;
      if (allow_rewind)
         signal.Flags.AllowFenceRewind = 1;
      signal.Fence.FenceValue = point;

      const NTSTATUS status = gpu->pfnSignalSynchronizationObject2(&signal);
      if (!NT_SUCCESS(status)) {
         vn_log(gpu->instance,
                "syncobj_signal(fence) failed handle=%u point=%" PRIu64 " status=%lx",
                sync->syncobj_handle, point, status);
         return -1;
      }
#if VIRTGPU_LOG_SYNC
      vn_log(gpu->instance,
             "syncobj_signal(fence) ok handle=%u point=%" PRIu64 " allow_rewind=%u",
             sync->syncobj_handle, point, allow_rewind ? 1u : 0u);
#endif
   }

   if (point > sync->last_signaled)
      sync->last_signaled = point;

   return 0;
}

static int
virtgpu_syncobj_wait(struct virtgpu *gpu,
                     const struct vn_renderer_wait *wait,
                     bool wait_avail)
{
   if (wait_avail) {
      errno = EINVAL;
      return -1;
   }
   if (!wait->sync_count)
      return 0;
   if (!gpu->hDevice)
      return -1;

   const int poll_timeout = timeout_to_poll_timeout(wait->timeout);
   const bool wait_any = wait->wait_any;

   /* simple wait-all semantics */
   (void)wait_any;
#if VIRTGPU_LOG_SYNC
   vn_log(gpu->instance,
          "syncobj_wait: count=%u wait_any=%u timeout=%llu poll_timeout=%d",
          wait->sync_count, wait_any, (unsigned long long)wait->timeout,
          poll_timeout);
#endif

   for (uint32_t i = 0; i < wait->sync_count; i++) {
      struct virtgpu_sync *sync = (struct virtgpu_sync *)wait->syncs[i];
      const uint64_t point = wait->sync_values[i];
#if VIRTGPU_LOG_SYNC
      vn_log(gpu->instance,
             "syncobj_wait: idx=%u handle=%u point=%" PRIu64 " monitored=%u",
             i, sync->syncobj_handle, point, sync->is_monitored);
#endif

      if (sync->is_monitored) {
#if VIRTGPU_USE_MONITORED_FENCE
         if (!gpu->pfnWaitForSynchronizationObjectFromCpu) {
            errno = EINVAL;
            return -1;
         }

         D3DKMT_HANDLE handle = (D3DKMT_HANDLE)sync->syncobj_handle;
         UINT64 value = point;

         HANDLE event = NULL;
         if (poll_timeout >= 0) {
            event = CreateEventA(NULL, TRUE, FALSE, NULL);
            if (!event) {
               vn_log(gpu->instance,
                      "syncobj_wait: CreateEvent failed err=%lu",
                      GetLastError());
               errno = EIO;
               return -1;
            }
         }

         D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU args = {0};
         args.hDevice = gpu->hDevice;
         args.ObjectCount = 1;
         args.ObjectHandleArray = &handle;
         args.FenceValueArray = &value;
         args.hAsyncEvent = event;
         args.Flags.Value = 0;

         const NTSTATUS status =
            gpu->pfnWaitForSynchronizationObjectFromCpu(&args);
         if (!NT_SUCCESS(status)) {
            if (event)
               CloseHandle(event);
            vn_log(gpu->instance,
                   "syncobj_wait: WaitFromCpu failed handle=%u point=%" PRIu64
                   " status=%lx",
                   sync->syncobj_handle, point, status);
            errno = EIO;
            return -1;
         }

         if (event) {
            const DWORD wait_ms = (DWORD)poll_timeout;
            const DWORD w = WaitForSingleObject(event, wait_ms);
            CloseHandle(event);
            if (w == WAIT_TIMEOUT) {
               vn_log(gpu->instance,
                      "syncobj_wait: WaitFromCpu timeout handle=%u point=%" PRIu64,
                      sync->syncobj_handle, point);
               errno = ETIME;
               return -1;
            }
            if (w != WAIT_OBJECT_0) {
               vn_log(gpu->instance,
                      "syncobj_wait: WaitFromCpu failed handle=%u point=%" PRIu64
                      " w=%lu",
                      sync->syncobj_handle, point, w);
               errno = EIO;
               return -1;
            }
         }
#else
         errno = EINVAL;
         return -1;
#endif
      } else {
         if (!gpu->pfnWaitForSynchronizationObject2 || !gpu->hContext) {
            errno = EINVAL;
            return -1;
         }

         if (poll_timeout >= 0 && poll_timeout != 0) {
            vn_log(gpu->instance,
                   "sync wait timeout ignored for fence syncobj");
         }

         D3DKMT_WAITFORSYNCHRONIZATIONOBJECT2 args = {0};
         args.hContext = gpu->hContext;
         args.ObjectCount = 1;
         args.ObjectHandleArray[0] = (D3DKMT_HANDLE)sync->syncobj_handle;
         args.Fence.FenceValue = point;

         const NTSTATUS status =
            gpu->pfnWaitForSynchronizationObject2(&args);
         if (!NT_SUCCESS(status)) {
            vn_log(gpu->instance,
                   "syncobj_wait: Wait2 failed handle=%u point=%" PRIu64
                   " status=%lx",
                   sync->syncobj_handle, point, status);
            errno = EIO;
            return -1;
         }
#if VIRTGPU_LOG_SYNC
         vn_log(gpu->instance,
                "syncobj_wait: Wait2 ok handle=%u point=%" PRIu64,
                sync->syncobj_handle, point);
#endif
      }
   }

   return 0;
}

static VkResult
virtgpu_sync_write(struct vn_renderer *renderer,
                   struct vn_renderer_sync *_sync,
                   uint64_t val)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_sync *sync = (struct virtgpu_sync *)_sync;

   const int ret = virtgpu_syncobj_signal(gpu, sync, val, false);

   return ret ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
}

static VkResult
virtgpu_sync_read(struct vn_renderer *renderer,
                  struct vn_renderer_sync *_sync,
                  uint64_t *val)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_sync *sync = (struct virtgpu_sync *)_sync;

   (void)gpu;
   if (sync->fence_cpu_va)
      *val = *sync->fence_cpu_va;
   else
      *val = sync->last_signaled;
   return VK_SUCCESS;
}

static VkResult
virtgpu_sync_reset(struct vn_renderer *renderer,
                   struct vn_renderer_sync *_sync,
                   uint64_t initial_val)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_sync *sync = (struct virtgpu_sync *)_sync;

   int ret = virtgpu_syncobj_signal(gpu, sync, 0, true);
   if (!ret)
      ret = virtgpu_syncobj_signal(gpu, sync, initial_val, true);

   return ret ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
}

static int
virtgpu_sync_export_syncobj(struct vn_renderer *renderer,
                            struct vn_renderer_sync *_sync,
                            bool sync_file)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_sync *sync = (struct virtgpu_sync *)_sync;

   assert(!"D3DKMT backend does not support sync export");
   return virtgpu_syncobj_export(gpu, sync->syncobj_handle, sync_file);
}

static void
virtgpu_sync_destroy(struct vn_renderer *renderer,
                     struct vn_renderer_sync *_sync)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_sync *sync = (struct virtgpu_sync *)_sync;

   virtgpu_syncobj_destroy(gpu, sync->syncobj_handle);

   free(sync);
}

static VkResult
virtgpu_sync_create_from_syncobj(struct vn_renderer *renderer,
                                 int fd,
                                 bool sync_file,
                                 struct vn_renderer_sync **out_sync)
{
   assert(!"D3DKMT backend does not support sync import");
   (void)renderer;
   (void)fd;
   (void)sync_file;
   (void)out_sync;

   return VK_ERROR_INVALID_EXTERNAL_HANDLE;
}

static VkResult
virtgpu_sync_create(struct vn_renderer *renderer,
                    uint64_t initial_val,
                    uint32_t flags,
                    struct vn_renderer_sync **out_sync)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;

   /* TODO */
   if (flags & VN_RENDERER_SYNC_SHAREABLE)
      vn_log(gpu->instance, "virtgpu_sync_create: shareable not supported flags=0x%x", flags);
   if (flags & VN_RENDERER_SYNC_SHAREABLE)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   /* always false because we don't use binary syncobjs */
   volatile uint64_t *cpu_va = NULL;
   bool is_monitored = false;
   const uint32_t syncobj_handle =
      virtgpu_syncobj_create(gpu, &cpu_va, &is_monitored, initial_val);
   if (!syncobj_handle)
      vn_log(gpu->instance, "virtgpu_sync_create: syncobj_create failed");
   if (!syncobj_handle)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct virtgpu_sync *sync = calloc(1, sizeof(*sync));
   if (!sync) {
      virtgpu_syncobj_destroy(gpu, syncobj_handle);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   sync->syncobj_handle = syncobj_handle;
   sync->fence_cpu_va = cpu_va;
   sync->is_monitored = is_monitored;
   sync->last_signaled = initial_val;
   /* we will have a sync_id when shareable is true and virtio-gpu associates
    * a host sync object with guest syncobj
    */
   sync->base.sync_id = 0;

   *out_sync = &sync->base;

   return VK_SUCCESS;
}

static void
virtgpu_bo_invalidate(struct vn_renderer *renderer,
                      struct vn_renderer_bo *bo,
                      VkDeviceSize offset,
                      VkDeviceSize size)
{
   /* nop because kernel makes every mapping coherent */
}

static void
virtgpu_bo_flush(struct vn_renderer *renderer,
                 struct vn_renderer_bo *bo,
                 VkDeviceSize offset,
                 VkDeviceSize size)
{
   /* nop because kernel makes every mapping coherent */
}

static void *
virtgpu_bo_map(struct vn_renderer *renderer, struct vn_renderer_bo *_bo)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_bo *bo = (struct virtgpu_bo *)_bo;
   const bool mappable = bo->blob_flags & VIRTGPU_BLOB_FLAG_USE_MAPPABLE;

   /* not thread-safe but is fine */
   if (!bo->base.mmap_ptr && mappable) {
      bo->base.mmap_ptr =
         virtgpu_d3dkmt_map(gpu, bo->alloc_handle, bo->base.mmap_size);
   }

   return bo->base.mmap_ptr;
}

static int
virtgpu_bo_export_dma_buf(struct vn_renderer *renderer,
                          struct vn_renderer_bo *_bo)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_bo *bo = (struct virtgpu_bo *)_bo;
   const bool mappable = bo->blob_flags & VIRTGPU_BLOB_FLAG_USE_MAPPABLE;
   const bool shareable = bo->blob_flags & VIRTGPU_BLOB_FLAG_USE_SHAREABLE;

   return shareable
             ? virtgpu_d3dkmt_prime_handle_to_fd(gpu, bo->alloc_handle,
                                                mappable)
             : -1;
}

static bool
virtgpu_bo_destroy(struct vn_renderer *renderer, struct vn_renderer_bo *_bo)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_bo *bo = (struct virtgpu_bo *)_bo;

   mtx_lock(&gpu->import_mutex);

   /* Check the refcount again after the import lock is grabbed.  Yes, we use
    * the double-checked locking anti-pattern.
    */
   if (vn_refcount_is_valid(&bo->base.refcount)) {
      mtx_unlock(&gpu->import_mutex);
      return false;
   }

   if (bo->base.mmap_ptr && (bo->blob_flags & VIRTGPU_BLOB_FLAG_USE_MAPPABLE))
      virtgpu_d3dkmt_resource_unmap_blob(gpu, bo->alloc_handle);

   /* Clear the allocation handle to mark the bo invalid. Must be set
    * before closing the handle. Otherwise the same handle can be reused
    * by another newly created bo and unexpectedly zeroed in the tracker.
    */
   const uint32_t alloc_handle = bo->alloc_handle;
   bo->alloc_handle = 0;
   virtgpu_d3dkmt_gem_close(gpu, alloc_handle);

   mtx_unlock(&gpu->import_mutex);

   return true;
}

static uint32_t
virtgpu_bo_blob_flags(struct virtgpu *gpu,
                      VkMemoryPropertyFlags flags,
                      VkExternalMemoryHandleTypeFlags external_handles)
{
   uint32_t blob_flags = 0;
   if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
      blob_flags |= VIRTGPU_BLOB_FLAG_USE_MAPPABLE;
   if (external_handles)
      blob_flags |= VIRTGPU_BLOB_FLAG_USE_SHAREABLE;
   if (external_handles & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) {
      if (gpu->supports_cross_device)
         blob_flags |= VIRTGPU_BLOB_FLAG_USE_CROSS_DEVICE;
   }

   return blob_flags;
}

static VkResult
virtgpu_bo_create_from_dma_buf(struct vn_renderer *renderer,
                               VkDeviceSize size,
                               int fd,
                               VkMemoryPropertyFlags flags,
                               struct vn_renderer_bo **out_bo)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_resource_info info;
   uint32_t alloc_handle = 0;
   struct virtgpu_bo *bo = NULL;

   mtx_lock(&gpu->import_mutex);

   alloc_handle = virtgpu_d3dkmt_prime_fd_to_handle(gpu, fd);
   if (!alloc_handle)
      goto fail;
   bo = util_sparse_array_get(&gpu->bo_array, alloc_handle);

   if (virtgpu_d3dkmt_resource_info(gpu, alloc_handle, &info))
      goto fail;

   /* Upon import, blob_flags is not passed to the kernel and is only for
    * internal use. Set it to what works best for us.
    * - blob mem: SHAREABLE + conditional MAPPABLE per VkMemoryPropertyFlags
    * - classic 3d: SHAREABLE only for export and to fail the map
    */
   uint32_t blob_flags = VIRTGPU_BLOB_FLAG_USE_SHAREABLE;
   size_t mmap_size = 0;
   if (info.blob_mem) {
      /* must be VIRTGPU_BLOB_MEM_HOST3D or VIRTGPU_BLOB_MEM_GUEST_VRAM */
      if (info.blob_mem != gpu->bo_blob_mem)
         goto fail;

      blob_flags |= virtgpu_bo_blob_flags(gpu, flags, 0);

      /* mmap_size is only used when mappable */
      mmap_size = 0;
      if (blob_flags & VIRTGPU_BLOB_FLAG_USE_MAPPABLE) {
         /* If queried blob size is smaller than requested allocation size, we
          * drop the mappable flag to defer the mapping failure till the app's
          * vkMapMemory api call.
          *
          * Use size zero to request mapping the whole bo.
          */
         if (info.size < size)
            blob_flags &= ~VIRTGPU_BLOB_FLAG_USE_MAPPABLE;
         else
            mmap_size = size > 0 ? size : info.size;
      }
   }

   /* we check bo->alloc_handle instead of bo->refcount because bo->refcount
    * might only be memset to 0 and is not considered initialized in theory
    */
   if (bo->alloc_handle == alloc_handle) {
      if (bo->base.mmap_size < mmap_size)
         goto fail;
      if (blob_flags & ~bo->blob_flags)
         goto fail;

      /* we can't use vn_renderer_bo_ref as the refcount may drop to 0
       * temporarily before virtgpu_bo_destroy grabs the lock
       */
      vn_refcount_fetch_add_relaxed(&bo->base.refcount, 1);
   } else {
      *bo = (struct virtgpu_bo){
         .base = {
            .refcount = VN_REFCOUNT_INIT(1),
            .res_id = info.res_handle,
            .mmap_size = mmap_size,
         },
         .alloc_handle = alloc_handle,
         .blob_flags = blob_flags,
      };
   }

   mtx_unlock(&gpu->import_mutex);

   *out_bo = &bo->base;

   return VK_SUCCESS;

fail:
   if (alloc_handle && bo->alloc_handle != alloc_handle)
      virtgpu_d3dkmt_gem_close(gpu, alloc_handle);
   mtx_unlock(&gpu->import_mutex);
   return VK_ERROR_INVALID_EXTERNAL_HANDLE;
}

static VkResult
virtgpu_bo_create_from_device_memory(
   struct vn_renderer *renderer,
   VkDeviceSize size,
   vn_object_id mem_id,
   VkMemoryPropertyFlags flags,
   VkExternalMemoryHandleTypeFlags external_handles,
   struct vn_renderer_bo **out_bo)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   const uint32_t blob_flags =
      virtgpu_bo_blob_flags(gpu, flags, external_handles);

   uint32_t res_id;
#if VIRTGPU_LOG_SHMEM   
   vn_log(gpu->instance, "resource_create_blob path=bo_create size=0x%llx mem_id=0x%llx flags=0x%x ext=0x%x",
           (unsigned long long)size, (unsigned long long)mem_id, flags, external_handles);
#endif
   uint32_t alloc_handle = virtgpu_d3dkmt_resource_create_blob(
      gpu, gpu->bo_blob_mem, blob_flags, size, mem_id, &res_id);

   if (!alloc_handle)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct virtgpu_bo *bo =
      util_sparse_array_get(&gpu->bo_array, alloc_handle);
   *bo = (struct virtgpu_bo){
      .base = {
         .refcount = VN_REFCOUNT_INIT(1),
         .res_id = res_id,
         .mmap_size = size,
      },
      .alloc_handle = alloc_handle,
      .blob_flags = blob_flags,
   };

   *out_bo = &bo->base;

   return VK_SUCCESS;
}

static void
virtgpu_shmem_destroy_now(struct vn_renderer *renderer,
                          struct vn_renderer_shmem *_shmem)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   struct virtgpu_shmem *shmem = (struct virtgpu_shmem *)_shmem;

   if (shmem->base.mmap_ptr)
      virtgpu_d3dkmt_resource_unmap_blob(gpu, shmem->alloc_handle);
   virtgpu_d3dkmt_gem_close(gpu, shmem->alloc_handle);
}

static void
virtgpu_shmem_destroy(struct vn_renderer *renderer,
                      struct vn_renderer_shmem *shmem)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;

   if (vn_renderer_shmem_cache_add(&gpu->shmem_cache, shmem))
      return;

   virtgpu_shmem_destroy_now(&gpu->base, shmem);
}

static struct vn_renderer_shmem *
virtgpu_shmem_create(struct vn_renderer *renderer, size_t size)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   const size_t alloc_size = align64(size, 4096);

#if VIRTGPU_LOG_SHMEM
   vn_log(gpu->instance, "shmem_create: req size=0x%zx mask=0x%zx alloc=0x%zx",
          size, size & 0xfff, alloc_size);
#endif
   struct vn_renderer_shmem *cached_shmem =
      vn_renderer_shmem_cache_get(&gpu->shmem_cache, alloc_size);
   if (cached_shmem) {
      cached_shmem->refcount = VN_REFCOUNT_INIT(1);
#if VIRTGPU_LOG_SHMEM
      vn_log(gpu->instance,
             "shmem_create: cache hit size=0x%zx ptr=%p mask=0x%zx",
             cached_shmem->mmap_size, cached_shmem->mmap_ptr,
             (size_t)cached_shmem->mmap_ptr & 0xfff);
#endif
      return cached_shmem;
   }

   uint32_t res_id;
   uint32_t alloc_handle = virtgpu_d3dkmt_resource_create_blob(
      gpu, gpu->shmem_blob_mem, VIRTGPU_BLOB_FLAG_USE_MAPPABLE, alloc_size, 0,
      &res_id);

   if (!alloc_handle)
      return NULL;

   void *ptr = virtgpu_d3dkmt_map(gpu, alloc_handle, alloc_size);
   if (!ptr) {
      virtgpu_d3dkmt_gem_close(gpu, alloc_handle);
      return NULL;
   }
#if VIRTGPU_LOG_SHMEM
   vn_log(gpu->instance,
          "shmem_create: alloc=0x%x res_id=0x%x size=0x%zx ptr=%p mask=0x%zx",
          alloc_handle, res_id, alloc_size, ptr, (size_t)ptr & 0xfff);
#endif

   struct virtgpu_shmem *shmem =
      util_sparse_array_get(&gpu->shmem_array, alloc_handle);
   *shmem = (struct virtgpu_shmem){
      .base = {
         .refcount = VN_REFCOUNT_INIT(1),
         .res_id = res_id,
         .mmap_size = alloc_size,
         .mmap_ptr = ptr,
      },
      .alloc_handle = alloc_handle,
   };

   return &shmem->base;
}

static VkResult
virtgpu_wait(struct vn_renderer *renderer,
             const struct vn_renderer_wait *wait)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;

   const int ret = virtgpu_syncobj_wait(gpu, wait, false);
   if (ret && errno != ETIME)
      return VK_ERROR_DEVICE_LOST;

   return ret ? VK_TIMEOUT : VK_SUCCESS;
}

static VkResult
virtgpu_submit(struct vn_renderer *renderer,
               const struct vn_renderer_submit *submit)
{
   struct virtgpu *gpu = (struct virtgpu *)renderer;

   const int ret = d3dkmt_submit(gpu, submit);
   return ret ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
}

static void
virtgpu_init_renderer_info(struct virtgpu *gpu)
{
   struct vn_renderer_info *info = &gpu->base.info;

   info->drm.props = (VkPhysicalDeviceDrmPropertiesEXT){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT,
      .hasPrimary = gpu->has_primary,
      .hasRender = true,
      .primaryMajor = gpu->primary_major,
      .primaryMinor = gpu->primary_minor,
      .renderMajor = gpu->render_major,
      .renderMinor = gpu->render_minor,
   };

   info->pci.vendor_id = VIRTGPU_PCI_VENDOR_ID;
   info->pci.device_id = VIRTGPU_PCI_DEVICE_ID;

   info->has_dma_buf_import = true;
   /* TODO switch from emulation to native sync */
   info->has_external_sync = true;

   info->has_implicit_fencing = false;

   const struct virgl_renderer_capset_venus *capset = &gpu->capset.data;
   info->wire_format_version = capset->wire_format_version;
   info->vk_xml_version = capset->vk_xml_version;
   info->vk_ext_command_serialization_spec_version =
      capset->vk_ext_command_serialization_spec_version;
   info->vk_mesa_venus_protocol_spec_version =
      capset->vk_mesa_venus_protocol_spec_version;
   assert(capset->supports_blob_id_0);

   /* ensure vk_extension_mask is large enough to hold all capset masks */
   STATIC_ASSERT(sizeof(info->vk_extension_mask) >=
                 sizeof(capset->vk_extension_mask1));
   memcpy(info->vk_extension_mask, capset->vk_extension_mask1,
          sizeof(capset->vk_extension_mask1));

   assert(capset->allow_vk_wait_syncs);

   assert(capset->supports_multiple_timelines);
   info->max_timeline_count = gpu->max_timeline_count;

   if (gpu->bo_blob_mem == VIRTGPU_BLOB_MEM_GUEST_VRAM)
      info->has_guest_vram = true;

   /* Use guest blob allocations from dedicated heap (Host visible memory) */
   if (gpu->bo_blob_mem == VIRTGPU_BLOB_MEM_HOST3D && capset->use_guest_vram)
      info->has_guest_vram = true;
}

static void
virtgpu_destroy(struct vn_renderer *renderer,
                const VkAllocationCallbacks *alloc)
{  
   struct virtgpu *gpu = (struct virtgpu *)renderer;
   if (VN_DEBUG(INIT))
      vn_log(gpu->instance, "%s", __FUNCTION__);

   vn_renderer_shmem_cache_fini(&gpu->shmem_cache);

   virtgpu_submit_ctx_fini(gpu);

   if (gpu->pfnDestroyDevice && gpu->hDevice) {
      D3DKMT_DESTROYDEVICE destroy = {0};
      destroy.hDevice = gpu->hDevice;
      gpu->pfnDestroyDevice(&destroy);
      gpu->hDevice = 0;
   }
   if (gpu->pfnCloseAdapter && gpu->hAdapter) {
      D3DKMT_CLOSEADAPTER close = {0};
      close.hAdapter = gpu->hAdapter;
      gpu->pfnCloseAdapter(&close);
      gpu->hAdapter = 0;
   }
   virtgpu_unload_d3dkmt(gpu);

   mtx_destroy(&gpu->import_mutex);

   util_sparse_array_finish(&gpu->shmem_array);
   util_sparse_array_finish(&gpu->bo_array);

   vk_free(alloc, gpu);
}

static inline void
virtgpu_init_shmem_blob_mem(ASSERTED struct virtgpu *gpu)
{
   /* VIRTGPU_BLOB_MEM_GUEST allocates from the guest system memory.  It is
    * contiguous in the guest but becomes scatter-gather in the host, which
    * is slower to process.  With host process isolation, the host cannot
    * access those segments directly.
    *
    * To keep exported blobs usable, the easiest path is to reuse
    * VIRTGPU_BLOB_MEM_HOST3D.  That is, when the renderer sees a request to
    * export a blob where
    *
    *  - blob_mem is VIRTGPU_BLOB_MEM_HOST3D
    *  - blob_flags is VIRTGPU_BLOB_FLAG_USE_MAPPABLE
    *  - blob_id is 0
    *
    * it allocates a host shmem.
    *
    * supports_blob_id_0 has been enforced by mandated render server config.
    */
   assert(gpu->capset.data.supports_blob_id_0);
   gpu->shmem_blob_mem = VIRTGPU_BLOB_MEM_HOST3D;
}

static VkResult
virtgpu_init_context(struct virtgpu *gpu)
{
   assert(!gpu->capset.version);
   const int ret = virtgpu_context_init(gpu, gpu->capset.id);
   if (ret) {
      vn_log(gpu->instance, "failed to initialize context: %s",
               strerror(errno));
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   return VK_SUCCESS;
}

static VkResult
virtgpu_init_capset(struct virtgpu *gpu)
{
   gpu->capset.id = VIRTGPU_DRM_CAPSET_VENUS;
   gpu->capset.version = 0;

   const int ret =
      virtgpu_get_caps(gpu, gpu->capset.id, gpu->capset.version,
                             &gpu->capset.data, sizeof(gpu->capset.data));
   if (ret) {
      vn_log(gpu->instance, "failed to get venus v%d capset: %s",
                gpu->capset.version, strerror(errno));
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   if (gpu->capset.data.wire_format_version == 0) {
      vn_log(gpu->instance, "Unsupported wire format version %u",
            gpu->capset.data.wire_format_version);
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   return VK_SUCCESS;
}

static VkResult
virtgpu_init_params(struct virtgpu *gpu)
{
   NTSTATUS Status;
   VIOGPU_ADAPTERINFO info;
   memset(&info, 0, sizeof(info));
   D3DKMT_QUERYADAPTERINFO query = {0};
   query.hAdapter = gpu->hAdapter;
   query.Type = KMTQAITYPE_UMDRIVERPRIVATE;
   query.pPrivateDriverData = &info;
   query.PrivateDriverDataSize = sizeof(info);

   Status = gpu->pfnQueryAdapterInfo(&query);

   if (!NT_SUCCESS(Status)) 
   {
      vn_log(gpu->instance, "Failed to request adapter info(D3DKMTQueryAdapterInfo) with status code: %lx\n",
         Status);
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   char * param;
   param = "VIRTGPU_PARAM_3D_FEATURES";
   if (!info.Flags.Supports3d)
      goto err;

   param = "VIRTGPU_PARAM_CAPSET_QUERY_FIX";
   if (!info.Flags.has_capset_query_fix)
      goto err;

   param = "VIRTGPU_PARAM_RESOURCE_BLOB";
   if (!info.Flags.has_resource_blob)
      goto err;

   param = "VIRTGPU_PARAM_CONTEXT_INIT";
   if (!info.Flags.has_context_init)
      goto err;

   if (info.Flags.has_host_visible) {
      gpu->bo_blob_mem = VIRTGPU_BLOB_MEM_HOST3D;
   } else {
      gpu->bo_blob_mem = VIRTGPU_BLOB_MEM_GUEST_VRAM;
   }

   /* Cross-device feature is optional.  It enables sharing external
    * allocations with other virtio devices, like virtio-wl or virtio-video
    * used by ChromeOS VMs.  Qemu doesn't support cross-device sharing.
    */
   if (info.Flags.has_resource_assign_uuid)
      gpu->supports_cross_device = true;

   /* implied by CONTEXT_INIT uapi */
   gpu->max_timeline_count = 64;

   return VK_SUCCESS;

err:
   vn_log(gpu->instance, "required kernel %s param is missing", param);

   return VK_ERROR_INITIALIZATION_FAILED;
}

static VkResult
virtgpu_open(struct virtgpu *gpu)
{
   if (!virtgpu_load_d3dkmt(gpu)) {
      vn_log(gpu->instance, "failed to load D3DKMT entrypoints");
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   HDC hdc = GetDC(NULL);
   if (!hdc) {
      vn_log(gpu->instance, "failed to get primary HDC");
      virtgpu_unload_d3dkmt(gpu);
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   D3DKMT_OPENADAPTERFROMHDC open = {0};
   open.hDc = hdc;
   NTSTATUS status = gpu->pfnOpenAdapterFromHdc(&open);
   ReleaseDC(NULL, hdc);

   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to open primary adapter");
      virtgpu_unload_d3dkmt(gpu);
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   D3DKMT_CREATEDEVICE create = {0};
   create.hAdapter = open.hAdapter;
   status = gpu->pfnCreateDevice(&create);

   if (!NT_SUCCESS(status)) {
      vn_log(gpu->instance, "failed to create D3DKMT device");
      D3DKMT_CLOSEADAPTER close = {0};
      close.hAdapter = open.hAdapter;
      gpu->pfnCloseAdapter(&close);
      virtgpu_unload_d3dkmt(gpu);
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   VIOGPU_ADAPTERINFO info;
   memset(&info, 0, sizeof(info));
   D3DKMT_QUERYADAPTERINFO query = {0};
   query.hAdapter = open.hAdapter;
   query.Type = KMTQAITYPE_UMDRIVERPRIVATE;
   query.pPrivateDriverData = &info;
   query.PrivateDriverDataSize = sizeof(info);
   status = gpu->pfnQueryAdapterInfo(&query);

   if (!NT_SUCCESS(status) || info.IamVioGPU != VIOGPU_IAM) {
      vn_log(gpu->instance, "primary adapter is not virtio-win viogpu");
      D3DKMT_DESTROYDEVICE destroy = {0};
      destroy.hDevice = create.hDevice;
      gpu->pfnDestroyDevice(&destroy);
      D3DKMT_CLOSEADAPTER close = {0};
      close.hAdapter = open.hAdapter;
      gpu->pfnCloseAdapter(&close);
      virtgpu_unload_d3dkmt(gpu);
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   gpu->hAdapter = open.hAdapter;
   gpu->hDevice = create.hDevice;
   gpu->adapter_info = info;
   gpu->has_primary = true;
   gpu->primary_major = 0;
   gpu->primary_minor = 0;
   gpu->render_major = 0;
   gpu->render_minor = 0;
   if (VN_DEBUG(INIT))
      vn_log(gpu->instance, "using primary virtio-win adapter");

   return VK_SUCCESS;
}

static VkResult
virtgpu_init(struct virtgpu *gpu)
{
   if (VN_DEBUG(INIT))
      vn_log(gpu->instance, "%s", __FUNCTION__);

   util_sparse_array_init(&gpu->shmem_array, sizeof(struct virtgpu_shmem),
                          1024);
   util_sparse_array_init(&gpu->bo_array, sizeof(struct virtgpu_bo), 1024);

   mtx_init(&gpu->import_mutex, mtx_plain);

   VkResult result = virtgpu_open(gpu);
   if (result == VK_SUCCESS)
      result = virtgpu_init_params(gpu);
   if (result == VK_SUCCESS)
      result = virtgpu_init_capset(gpu);
   if (result == VK_SUCCESS)
      result = virtgpu_init_context(gpu);

   if (result)
      vn_log(gpu->instance,"virtgpu_init failed %x", result);

   if (result != VK_SUCCESS)
      return result;

   virtgpu_init_shmem_blob_mem(gpu);

   vn_renderer_shmem_cache_init(&gpu->shmem_cache, &gpu->base,
                                 virtgpu_shmem_destroy_now);

   virtgpu_init_renderer_info(gpu);

   gpu->base.ops.destroy = virtgpu_destroy;
   gpu->base.ops.submit = virtgpu_submit;
   gpu->base.ops.wait = virtgpu_wait;

   gpu->base.shmem_ops.create = virtgpu_shmem_create;
   gpu->base.shmem_ops.destroy = virtgpu_shmem_destroy;

   gpu->base.bo_ops.create_from_device_memory = virtgpu_bo_create_from_device_memory;
   gpu->base.bo_ops.create_from_dma_buf = virtgpu_bo_create_from_dma_buf;
   gpu->base.bo_ops.destroy = virtgpu_bo_destroy;
   gpu->base.bo_ops.export_dma_buf = virtgpu_bo_export_dma_buf;
   gpu->base.bo_ops.map = virtgpu_bo_map;
   gpu->base.bo_ops.flush = virtgpu_bo_flush;
   gpu->base.bo_ops.invalidate = virtgpu_bo_invalidate;

   gpu->base.sync_ops.create = virtgpu_sync_create;
   gpu->base.sync_ops.create_from_syncobj = virtgpu_sync_create_from_syncobj;
   gpu->base.sync_ops.destroy = virtgpu_sync_destroy;
   gpu->base.sync_ops.export_syncobj = virtgpu_sync_export_syncobj;
   gpu->base.sync_ops.reset = virtgpu_sync_reset;
   gpu->base.sync_ops.read = virtgpu_sync_read;
   gpu->base.sync_ops.write = virtgpu_sync_write;

   return VK_SUCCESS;
}

VkResult
vn_renderer_create_virtgpu(struct vn_instance *instance,
                           const VkAllocationCallbacks *alloc,
                           struct vn_renderer **renderer)
{
   if (VN_DEBUG(INIT))
      vn_log(NULL, "%s", __FUNCTION__);

   struct virtgpu *gpu = vk_zalloc(alloc, sizeof(*gpu), VN_DEFAULT_ALIGN,
                                   VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!gpu) {
      vn_log(gpu->instance, "vn_renderer_create_virtgpu VK_ERROR_OUT_OF_HOST_MEMORY");
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   gpu->instance = instance;
   gpu->fd = -1;

   VkResult result = virtgpu_init(gpu);
   if (result != VK_SUCCESS) {
      vn_log(gpu->instance, "virtgpu_init failed result=%x", result);
      virtgpu_destroy(&gpu->base, alloc);
      return result;
   }

   *renderer = &gpu->base;

   return VK_SUCCESS;
}
