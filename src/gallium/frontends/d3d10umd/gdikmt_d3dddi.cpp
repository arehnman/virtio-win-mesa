#include "gdikmt_d3dddi.h"
#include "Debug.h"
#include <string.h>
#include "gdikmt/gdikmt.h"
#include "pipe/p_state.h"
#include "util/u_debug.h"
#include "util/u_memory.h"
#include "virtio/wddm/viogpu_wddm_driver.h"
#include "gallium/winsys/yttrium/gdi/yttrium_gdi_public.h"

#include <d3dukmdt.h>
#include <d3dkmthk.h>
#include "winddk_compat.h"

struct gdikmt_d3dddi_residency {
   SRWLOCK lock;
   D3DDDICB_CREATEPAGINGQUEUE queue;
   PFND3DDDI_MAKERESIDENTCB make_resident;
   PFND3DDDI_EVICTCB evict;
   PFND3DDDI_WAITFORSYNCHRONIZATIONOBJECTFROMCPUCB wait;
   PFND3DDDI_DESTROYPAGINGQUEUECB destroy_queue;
   PFND3DDDI_CREATESYNCHRONIZATIONOBJECT2CB create_sync;
   PFND3DDDI_WAITFORSYNCHRONIZATIONOBJECTFROMGPUCB wait_gpu;
   PFND3DDDI_SIGNALSYNCHRONIZATIONOBJECTFROMGPUCB signal_gpu;
};

HRESULT
gdikmt_d3dddi_init_residency(struct gdikmt_device_d3dddi *device,
                            const void *runtime_callbacks, UINT runtime_version)
{
   VIOGPU_ADAPTERINFO info = {};
   D3DDDICB_QUERYADAPTERINFO query = {};
   query.pPrivateDriverData = &info;
   query.PrivateDriverDataSize = sizeof(info);
   HRESULT hr = device->pAdapterCallbacks->pfnQueryAdapterInfoCb(
      device->hRTAdapter, &query);
   if (FAILED(hr))
      return hr;
   if (info.IamVioGPU != VIOGPU_IAM)
      return E_INVALIDARG;
   if (!info.Flags.requires_explicit_residency)
      return S_OK;

   /* D3D9's local KTCallbacks ends at Win7. Only read the newer entries from
    * the original runtime table, and only when its reported version has them. */
   if (!runtime_callbacks || runtime_version < D3D_UMD_INTERFACE_VERSION_WDDM2_0) {
      yttrium_gdi_trace_warnf("yttrium: ERROR residency initialization owner=d3dddi reason=WDDM2-runtime-callbacks-required version=0x%x\n", runtime_version);
      return E_NOTIMPL;
   }
   const D3DDDI_DEVICECALLBACKS *callbacks =
      (const D3DDDI_DEVICECALLBACKS *)runtime_callbacks;
   if (!callbacks->pfnMakeResidentCb || !callbacks->pfnEvictCb ||
       !callbacks->pfnWaitForSynchronizationObjectFromCpuCb ||
       !callbacks->pfnCreatePagingQueueCb || !callbacks->pfnDestroyPagingQueueCb) {
      yttrium_gdi_trace_warnf("yttrium: ERROR residency initialization owner=d3dddi reason=missing-WDDM2-callback\n");
      return E_NOTIMPL;
   }

   struct gdikmt_d3dddi_residency *state = CALLOC_STRUCT(gdikmt_d3dddi_residency);
   if (!state)
      return E_OUTOFMEMORY;
   InitializeSRWLock(&state->lock);
   state->make_resident = callbacks->pfnMakeResidentCb;
   state->evict = callbacks->pfnEvictCb;
   state->wait = callbacks->pfnWaitForSynchronizationObjectFromCpuCb;
   state->destroy_queue = callbacks->pfnDestroyPagingQueueCb;
   /* This Win8 callback is also beyond D3D9's Win7-sized local table. */
   state->create_sync = callbacks->pfnCreateSynchronizationObject2Cb;
   state->wait_gpu = callbacks->pfnWaitForSynchronizationObjectFromGpuCb;
   state->signal_gpu = callbacks->pfnSignalSynchronizationObjectFromGpuCb;
   state->queue.Priority = D3DDDI_PAGINGQUEUE_PRIORITY_NORMAL;
   hr = callbacks->pfnCreatePagingQueueCb(device->hRTDevice, &state->queue);
   if (FAILED(hr)) {
      yttrium_gdi_trace_warnf("yttrium: ERROR residency initialization owner=d3dddi operation=CreatePagingQueue status=0x%lx\n", (unsigned long)hr);
      FREE(state);
      return hr;
   }
   device->residency = state;
   return S_OK;
}

static HRESULT
gdikmt_d3dddi_make_resident(struct gdikmt_device_d3dddi *device,
                           UINT count, const D3DKMT_HANDLE *allocations)
{
   struct gdikmt_d3dddi_residency *state = device->residency;
   if (!state || !count)
      return S_OK;

   AcquireSRWLockExclusive(&state->lock);
   D3DDDI_MAKERESIDENT resident = {};
   resident.hPagingQueue = state->queue.hPagingQueue;
   resident.NumAllocations = count;
   resident.AllocationList = allocations;
   /* Blob backing is retained for the allocation lifetime. There is no idle
    * backing to trim here. Permit the current budget to be exceeded, but keep
    * the OS maximum budget and normal OOM handling (never MustSucceed). */
   resident.Flags.CantTrimFurther = 1;
   HRESULT hr = state->make_resident(device->hRTDevice, &resident);
   /* MakeResident is atomic on failure: NumAllocations may still contain the
    * input count, but no residency references were acquired. Only a successful
    * request (including E_PENDING) can need rollback after the fence wait. */
   const UINT acquired = SUCCEEDED(hr) || hr == E_PENDING ? resident.NumAllocations : 0;
   if (hr == E_PENDING) {
      D3DDDICB_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {};
      wait.ObjectCount = 1;
      wait.ObjectHandleArray = &state->queue.hSyncObject;
      wait.FenceValueArray = &resident.PagingFenceValue;
      hr = state->wait(device->hRTDevice, &wait);
   }
   if (SUCCEEDED(hr) && resident.NumAllocations != count)
      hr = E_FAIL;
   if (FAILED(hr)) {
      yttrium_gdi_trace_warnf("yttrium: ERROR allocation residency owner=d3dddi status=0x%lx requested=%u acquired=%u bytes_to_trim=%llu\n", (unsigned long)hr, count, acquired, resident.NumBytesToTrim);
      /* Undo only references acquired by this request; opened allocations
       * remain owned by the runtime. */
      if (acquired && acquired <= count) {
         D3DDDICB_EVICT rollback = {};
         rollback.NumAllocations = acquired;
         rollback.AllocationList = allocations;
         HRESULT rollback_hr = state->evict(device->hRTDevice, &rollback);
         if (FAILED(rollback_hr))
            yttrium_gdi_trace_warnf("yttrium: ERROR residency rollback owner=d3dddi status=0x%lx count=%u\n", (unsigned long)rollback_hr, rollback.NumAllocations);
      }
   }
   ReleaseSRWLockExclusive(&state->lock);
   return hr;
}

static HRESULT
gdikmt_d3dddi_open_residency(struct gdikmt_device_d3dddi *device,
                            struct gdikmt_openallocation *options)
{
   if (!device->residency || !options->NumAllocations)
      return S_OK;
   D3DKMT_HANDLE *handles = (D3DKMT_HANDLE *)
      CALLOC(options->NumAllocations, sizeof(*handles));
   if (!handles)
      return E_OUTOFMEMORY;
   for (UINT i = 0; i < options->NumAllocations; ++i)
      handles[i] = options->pOpenAllocation[i].hAllocation;
   HRESULT hr = gdikmt_d3dddi_make_resident(device, options->NumAllocations, handles);
   FREE(handles);
   return hr;
}

/*
 * Measurement knob for the KMD present round trip.  Rendering is published
 * independently before this layer is reached.  This option controls only
 * whether pfnPresentCb is issued:
 *
 *   D3D10UMD_YTTRIUM_PRESENT_EVERY=0 skip every callback
 *   D3D10UMD_YTTRIUM_PRESENT_EVERY=N issue one callback in N
 *
 * Skipped presents still emit the BEFORE/AFTER trace pair so the frame count
 * and the present rate stay comparable against an unmodified run.
 */
static bool
yttrium_present_skip_this_one(void)
{
   static const int64_t every = []() {
      int64_t n = yttrium_gdi_debug_get_num_option(
                     "D3D10UMD_YTTRIUM_PRESENT_EVERY", 1);
      return n >= 0 ? n : 1;
   }();
   static volatile LONG64 counter;

   if (every == 0)
      return true;
   if (every == 1)
      return false;

   const uint64_t index =
      (uint64_t)(InterlockedIncrement64(&counter) - 1);
   return (index % (uint64_t)every) != 0;
}

static HRESULT
yttrium_prepare_present(const struct gdikmt_present_info *present_info)
{
   if (!present_info || present_info->version < 2 ||
       !present_info->prepare_present)
      return S_OK;

   return present_info->prepare_present(present_info->prepare_present_data);
}

static inline struct gdikmt_context_d3dddi *
gdikmt_context_d3dddi(struct gdikmt_context *iws)
{
   return (struct gdikmt_context_d3dddi *)iws;
};

static inline struct gdikmt_device_d3dddi *
gdikmt_device_d3dddi(struct gdikmt_device *iws)
{
   return (struct gdikmt_device_d3dddi *)iws;
}

static NTSTATUS
gdikmt_d3dddi_record_status(struct gdikmt_device *device,
                            NTSTATUS status,
                            const char *callback)
{
   if (status != (NTSTATUS)D3DDDIERR_DEVICEREMOVED &&
       status != STATUS_DEVICE_REMOVED)
      return status;

   if (gdikmt_device_get_reset_status(device) == PIPE_NO_RESET) {
      yttrium_gdi_trace_warnf(
         "yttrium: device removed by runtime callback "
         "owner=yttrium-runtime-callback "
         "callback=%s status=0x%lx pid=%lu tid=%lu\n",
         callback ? callback : "<unknown>", (unsigned long)status,
         GetCurrentProcessId(), GetCurrentThreadId());
   }
   gdikmt_device_report_reset(device, PIPE_UNKNOWN_CONTEXT_RESET);
   return status;
}

static bool
gdikmt_is_known_viogpu_escape_type(USHORT type)
{
   switch (type) {
   case VIOGPU_GET_DEVICE_ID:
   case VIOGPU_GET_CUSTOM_RESOLUTION:
   case VIOGPU_GET_CAPS:
   case VIOGPU_RES_INFO:
   case VIOGPU_RES_BUSY:
   case VIOGPU_RES_MAP_BLOB:
   case VIOGPU_RES_UNMAP_BLOB:
   case VIOGPU_RES_CREATE_BLOB:
   case VIOGPU_RES_SET_SCANOUT_BLOB:
   case VIOGPU_RES_ATTACH_WAIT:
   case VIOGPU_CTX_INIT:
   case VIOGPU_SUBMIT_CMD:
   case VIOGPU_QUERY_TIMELINE_SUBMIT:
      return true;
   default:
      return false;
   }
}

static void
gdikmt_warn_invalid_viogpu_escape(const char *source,
                                  HANDLE hRTAdapter,
                                  HANDLE hRTDevice,
                                  const void *private_data,
                                  UINT private_size)
{
   static volatile LONG warning_count;
   LONG count = InterlockedIncrement(&warning_count);
   if (count > 8)
      return;

   USHORT type = 0xffff;
   USHORT data_length = 0xffff;
   unsigned char first_bytes[16] = {0};

   if (private_data) {
      const unsigned char *bytes = (const unsigned char *)private_data;
      const UINT copy = private_size < sizeof(first_bytes) ?
         private_size : (UINT)sizeof(first_bytes);
      memcpy(first_bytes, bytes, copy);
      if (private_size >= sizeof(type))
         memcpy(&type, bytes, sizeof(type));
      if (private_size >= sizeof(type) + sizeof(data_length))
         memcpy(&data_length, bytes + sizeof(type), sizeof(data_length));
   }

   yttrium_gdi_trace_warnf(
      "yttrium: UMD invalid VIOGPU escape source=%s hRTAdapter=%p hRTDevice=%p private=%p private_size=%u type=0x%x data_length=%u pid=%lu tid=%lu bytes=%02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
      source, hRTAdapter, hRTDevice, private_data, private_size, type, data_length,
      GetCurrentProcessId(), GetCurrentThreadId(),
      first_bytes[0], first_bytes[1], first_bytes[2], first_bytes[3],
      first_bytes[4], first_bytes[5], first_bytes[6], first_bytes[7],
      first_bytes[8], first_bytes[9], first_bytes[10], first_bytes[11],
      first_bytes[12], first_bytes[13], first_bytes[14], first_bytes[15]);
}

static void
gdikmt_check_viogpu_escape(const char *source,
                           HANDLE hRTAdapter,
                           HANDLE hRTDevice,
                           const void *private_data,
                           UINT private_size)
{
   const UINT header_size = sizeof(USHORT) + sizeof(USHORT);
   if (!private_data || private_size < header_size) {
      gdikmt_warn_invalid_viogpu_escape(source, hRTAdapter, hRTDevice,
                                        private_data, private_size);
      return;
   }

   const VIOGPU_ESCAPE *escape = (const VIOGPU_ESCAPE *)private_data;
   if (!gdikmt_is_known_viogpu_escape_type(escape->Type))
      gdikmt_warn_invalid_viogpu_escape(source, hRTAdapter, hRTDevice,
                                        private_data, private_size);
}

NTSTATUS
gdikmt_d3dddi_queryadapterinfo(struct gdikmt_device *_device,
                               KMTQUERYADAPTERINFOTYPE Type,
                               VOID *pPrivateDriverData,
                               UINT PrivateDriverDataSize)
{
   struct gdikmt_device_d3dddi *device = gdikmt_device_d3dddi(_device);

   D3DDDICB_QUERYADAPTERINFO queryAdapterInfo;
   memset(&queryAdapterInfo, 0, sizeof(queryAdapterInfo));
   queryAdapterInfo.pPrivateDriverData = pPrivateDriverData;
   queryAdapterInfo.PrivateDriverDataSize = PrivateDriverDataSize;

   NTSTATUS status =
      device->pAdapterCallbacks->pfnQueryAdapterInfoCb(device->hRTAdapter,
                                                       &queryAdapterInfo);
   return gdikmt_d3dddi_record_status(_device, status,
                                      "pfnQueryAdapterInfoCb");
}

NTSTATUS
gdikmt_d3dddi_escape(struct gdikmt_device *_device, VOID *pPrivateDriverData,
                     UINT PrivateDriverDataSize)
{
   struct gdikmt_device_d3dddi *device = gdikmt_device_d3dddi(_device);
   gdikmt_check_viogpu_escape("d3dddi", device->hRTAdapter, device->hRTDevice,
                              pPrivateDriverData, PrivateDriverDataSize);

   D3DDDICB_ESCAPE escape;
   memset(&escape, 0, sizeof(escape));
   escape.hDevice = device->hRTDevice;
   escape.pPrivateDriverData = pPrivateDriverData;
   escape.PrivateDriverDataSize = PrivateDriverDataSize;

   NTSTATUS status =
      device->KTCallbacks.pfnEscapeCb(device->hRTAdapter, &escape);
   return gdikmt_d3dddi_record_status(_device, status, "pfnEscapeCb");
}

NTSTATUS
gdikmt_d3dddi_render(struct gdikmt_context *_ctx, struct gdikmt_render *options)
{
   struct gdikmt_context_d3dddi *ctx = gdikmt_context_d3dddi(_ctx);
   struct gdikmt_device_d3dddi *dev = gdikmt_device_d3dddi(ctx->base.device);

   D3DDDICB_RENDER render;
   memset(&render, 0, sizeof(render));
   render.hContext = ctx->hContext;

   render.CommandOffset = options->CommandOffset;
   render.CommandLength = options->CommandLength;
   render.NumAllocations = options->AllocationCount;
   render.NumPatchLocations = options->PatchLocationCount;

   render.NewCommandBufferSize = options->NewCommandBufferSize;
   render.NewAllocationListSize = options->NewAllocationListSize;
   render.NewPatchLocationListSize = options->NewPatchLocationListSize;

   render.Flags.ResizeCommandBuffer = options->ResizeCommandBuffer;
   render.Flags.ResizeAllocationList = options->ResizeAllocationList;
   render.Flags.ResizePatchLocationList = options->ResizePatchLocationList;

   yttrium_gdi_trace_debugf("yttrium: pfnRenderCb before hDevice=%p hContext=%p command_length=%u command_offset=%u allocations=%u patches=%u resize_cmd=%u resize_alloc=%u resize_patch=%u new_sizes=%u/%u/%u completion=%p\n",
                            dev->hRTDevice,
                            render.hContext,
                            render.CommandLength,
                            render.CommandOffset,
                            render.NumAllocations,
                            render.NumPatchLocations,
                            render.Flags.ResizeCommandBuffer ? 1 : 0,
                            render.Flags.ResizeAllocationList ? 1 : 0,
                            render.Flags.ResizePatchLocationList ? 1 : 0,
                            render.NewCommandBufferSize,
                            render.NewAllocationListSize,
                            render.NewPatchLocationListSize,
                            options->CompletionEvent);
   NTSTATUS Status = gdikmt_d3dddi_record_status(
      &dev->base,
      dev->KTCallbacks.pfnRenderCb(dev->hRTDevice, &render),
      "pfnRenderCb");
   yttrium_gdi_trace_debugf("yttrium: pfnRenderCb after status=0x%lx hContext=%p new_cmd=%p new_alloc=%p new_patch=%p new_sizes=%u/%u/%u\n",
                            (unsigned long)Status,
                            render.hContext,
                            render.pNewCommandBuffer,
                            render.pNewAllocationList,
                            render.pNewPatchLocationList,
                            render.NewCommandBufferSize,
                            render.NewAllocationListSize,
                            render.NewPatchLocationListSize);
   if (options->CompletionEvent) {
      NTSTATUS signal_status;
      if (dev->use_legacy_signal_sync) {
         static bool warned_legacy_completion_event;
         if (!warned_legacy_completion_event) {
            yttrium_gdi_trace_warnf("yttrium: WARNING: D3D9 render "
                                    "completion event is CPU-signaled, "
                                    "not GPU-complete\n");
            yttrium_gdi_user_logf("WARNING: yttrium: D3D9 render "
                                  "completion event is CPU-signaled, "
                                  "not GPU-complete\n");
            warned_legacy_completion_event = true;
         }
         signal_status = SetEvent(options->CompletionEvent) ? S_OK : E_FAIL;
      } else {
         D3DDDICB_SIGNALSYNCHRONIZATIONOBJECT2 signalEvent;
         memset(&signalEvent, 0, sizeof(signalEvent));
         signalEvent.hContext = ctx->hContext;
         signalEvent.ObjectCount = 0;
         signalEvent.BroadcastContextCount = 0;
         signalEvent.Flags.EnqueueCpuEvent = TRUE;
         signalEvent.CpuEventHandle = options->CompletionEvent;
         signal_status = gdikmt_d3dddi_record_status(
            &dev->base,
            dev->KTCallbacks.pfnSignalSynchronizationObject2Cb(
               dev->hRTDevice, &signalEvent),
            "pfnSignalSynchronizationObject2Cb");
      }
      if (!NT_SUCCESS(signal_status))
         return signal_status;
   }

   ctx->base.pCommandBuffer = render.pNewCommandBuffer;
   ctx->base.pAllocationList = render.pNewAllocationList;
   ctx->base.pPatchLocationList = render.pNewPatchLocationList;

   ctx->base.CommandBufferSize = render.NewCommandBufferSize;
   ctx->base.AllocationListSize = render.NewAllocationListSize;
   ctx->base.PatchLocationListSize = render.NewPatchLocationListSize;

   return Status;
}

static HRESULT
gdikmt_d3dddi_reserve_present(struct gdikmt_context *_ctx,
                             struct gdikmt_context **consumer, uint64_t *value)
{
   struct gdikmt_context_d3dddi *ctx = gdikmt_context_d3dddi(_ctx);
   struct gdikmt_device_d3dddi *dev = gdikmt_device_d3dddi(_ctx->device);
   if (!ctx->present_fence) {
      D3DDDICB_CREATESYNCHRONIZATIONOBJECT2 create = {};
      create.Info.Type = D3DDDI_MONITORED_FENCE;
      HRESULT hr = dev->residency->create_sync(dev->hRTDevice, &create);
      if (FAILED(hr)) return hr;
      ctx->present_fence = create.hSyncObject;
   }
   if (!ctx->present_context) {
      HRESULT hr = dev->base.createContext(&dev->base, &ctx->present_context);
      if (FAILED(hr)) return hr;
   }
   if (ctx->present_value == UINT64_MAX)
      return E_FAIL;
   *consumer = ctx->present_context;
   *value = ++ctx->present_value;
   return S_OK;
}

static HRESULT
gdikmt_d3dddi_signal_present(struct gdikmt_context *_ctx, uint64_t value)
{
   struct gdikmt_context_d3dddi *ctx = gdikmt_context_d3dddi(_ctx);
   struct gdikmt_device_d3dddi *dev = gdikmt_device_d3dddi(_ctx->device);
   D3DDDICB_SIGNALSYNCHRONIZATIONOBJECTFROMGPU signal = {};
   signal.hContext = ctx->hContext;
   signal.ObjectCount = 1;
   signal.ObjectHandleArray = &ctx->present_fence;
   signal.MonitoredFenceValueArray = &value;
   HRESULT hr = dev->residency->signal_gpu(dev->hRTDevice, &signal);
   yttrium_gdi_trace_debugf("yttrium: scheduled Present GPU signal producer=%p fence=%x value=%llu hr=%lx\n", ctx->hContext, ctx->present_fence, value, (unsigned long)hr);
   if (FAILED(hr)) {
      yttrium_gdi_trace_warnf("yttrium: ERROR scheduled Present owner=d3dddi operation=SignalFromGpu value=%llu hr=%lx\n", value, (unsigned long)hr);
      gdikmt_device_report_reset(&dev->base, PIPE_UNKNOWN_CONTEXT_RESET);
   }
   return hr;
}

static HRESULT
gdikmt_d3dddi_wait_present(struct gdikmt_context *_ctx, uint64_t value)
{
   struct gdikmt_context_d3dddi *ctx = gdikmt_context_d3dddi(_ctx);
   struct gdikmt_device_d3dddi *dev = gdikmt_device_d3dddi(_ctx->device);
   if (gdikmt_device_get_reset_status(&dev->base) != PIPE_NO_RESET)
      return D3DDDIERR_DEVICEREMOVED;
   D3DDDICB_WAITFORSYNCHRONIZATIONOBJECTFROMGPU wait = {};
   wait.hContext = gdikmt_context_d3dddi(ctx->present_context)->hContext;
   wait.ObjectCount = 1;
   wait.ObjectHandleArray = &ctx->present_fence;
   wait.MonitoredFenceValueArray = &value;
   HRESULT hr = dev->residency->wait_gpu(dev->hRTDevice, &wait);
   yttrium_gdi_trace_debugf("yttrium: scheduled Present GPU wait consumer=%p fence=%x value=%llu hr=%lx\n", wait.hContext, ctx->present_fence, value, (unsigned long)hr);
   if (FAILED(hr))
      yttrium_gdi_trace_warnf("yttrium: ERROR scheduled Present owner=d3dddi operation=WaitFromGpu value=%llu hr=%lx\n", value, (unsigned long)hr);
   return hr;
}

void
gdikmt_d3dddi_destroycontext(struct gdikmt_context *_ctx)
{
   struct gdikmt_context_d3dddi *ctx = gdikmt_context_d3dddi(_ctx);
   struct gdikmt_device_d3dddi *dev = gdikmt_device_d3dddi(ctx->base.device);

   if (ctx->present_context)
      ctx->present_context->destroy(ctx->present_context);
   if (ctx->present_fence && !dev->base.runtime_destroying) {
      D3DDDICB_DESTROYSYNCHRONIZATIONOBJECT destroy = {};
      destroy.hSyncObject = ctx->present_fence;
      HRESULT hr = dev->KTCallbacks.pfnDestroySynchronizationObjectCb(dev->hRTDevice, &destroy);
      if (FAILED(hr))
         yttrium_gdi_trace_warnf("yttrium: ERROR scheduled Present owner=d3dddi operation=DestroyFence hr=%lx\n", (unsigned long)hr);
   }

   D3DDDICB_DESTROYCONTEXT destroyContext;
   memset(&destroyContext, 0, sizeof(destroyContext));
   destroyContext.hContext = ctx->hContext;
   ResourceEvent(RESOURCE_EVENT_CB_BEFORE,
                 (uint64_t)(uintptr_t)dev->hRTDevice,
                 ctx, NULL, 0, RESOURCE_EVENT_CB_DESTROY_CONTEXT, 0,
                 (uint64_t)(uintptr_t)ctx->hContext);
   NTSTATUS status =
      dev->KTCallbacks.pfnDestroyContextCb(dev->hRTDevice, &destroyContext);
   status = gdikmt_d3dddi_record_status(&dev->base, status,
                                        "pfnDestroyContextCb");
   ResourceEvent(RESOURCE_EVENT_CB_AFTER,
                 (uint64_t)(uintptr_t)dev->hRTDevice,
                 ctx, NULL, 0, RESOURCE_EVENT_CB_DESTROY_CONTEXT,
                 (uint32_t)status,
                 (uint64_t)(uintptr_t)ctx->hContext);

   FREE(ctx);

   return;
}

static D3DKMT_HANDLE
gdikmt_d3dddi_context_kmt_handle(struct gdikmt_context *_ctx)
{
   struct gdikmt_context_d3dddi *ctx = gdikmt_context_d3dddi(_ctx);

   return (D3DKMT_HANDLE)(uintptr_t)ctx->hContext;
}

NTSTATUS
gdikmt_d3dddi_createcontext(struct gdikmt_device *_device,
                            struct gdikmt_context **out_ctx)
{
   struct gdikmt_device_d3dddi *device = gdikmt_device_d3dddi(_device);

   struct gdikmt_context_d3dddi *ctx = CALLOC_STRUCT(gdikmt_context_d3dddi);
   if (!ctx) {
      return STATUS_NO_MEMORY;
   }
   ctx->base.device = _device;

   D3DDDICB_CREATECONTEXT createContext;
   memset(&createContext, 0, sizeof(createContext));
   if (device->use_legacy_signal_sync)
      createContext.EngineAffinity = 1;
   NTSTATUS Status = gdikmt_d3dddi_record_status(
      _device,
      device->KTCallbacks.pfnCreateContextCb(device->hRTDevice,
                                             &createContext),
      "pfnCreateContextCb");
   yttrium_gdi_trace_debugf("yttrium: pfnCreateContextCb status=0x%lx hContext=%p cmd=%p cmd_size=%u alloc_list=%p alloc_list_size=%u patch_list=%p patch_list_size=%u\n",
                            (unsigned long)Status,
                            createContext.hContext,
                            createContext.pCommandBuffer,
                            createContext.CommandBufferSize,
                            createContext.pAllocationList,
                            createContext.AllocationListSize,
                            createContext.pPatchLocationList,
                            createContext.PatchLocationListSize);

   if (NT_SUCCESS(Status)) {
      ctx->hContext = createContext.hContext;

      ctx->base.pCommandBuffer = createContext.pCommandBuffer;
      ctx->base.pAllocationList = createContext.pAllocationList;
      ctx->base.pPatchLocationList = createContext.pPatchLocationList;

      ctx->base.CommandBufferSize = createContext.CommandBufferSize;
      ctx->base.AllocationListSize = createContext.AllocationListSize;
      ctx->base.PatchLocationListSize = createContext.PatchLocationListSize;

      ctx->base.destroy = gdikmt_d3dddi_destroycontext;
      ctx->base.render = gdikmt_d3dddi_render;
      ctx->base.kmt_handle = gdikmt_d3dddi_context_kmt_handle;

      /* Read once per process so producer and Present contexts use one mode.
       * Scheduled Present is the default; a process can opt out with 0. */
      static const int scheduled_mode = []() {
         char mode[8] = {};
         const DWORD length = GetEnvironmentVariableA(
            "D3D10UMD_YTTRIUM_SCHEDULED_PRESENT", mode, sizeof(mode));
         if (length >= sizeof(mode) ||
             (length && strcmp(mode, "0") && strcmp(mode, "1")))
            return -1;
         return !length || !strcmp(mode, "1") ? 1 : 0;
      }();
      if (scheduled_mode < 0) {
         yttrium_gdi_trace_warnf("yttrium: ERROR scheduled Present owner=d3dddi reason=invalid-mode expected=0-or-1\n");
         ctx->base.destroy(&ctx->base);
         return E_INVALIDARG;
      }
      if (scheduled_mode) {
         VIOGPU_ESCAPE query = {};
         query.Type = VIOGPU_QUERY_TIMELINE_SUBMIT;
         query.DataLength = sizeof(UINT);
         HRESULT hr = device->base.escape(&device->base, &query, sizeof(query));
         if (FAILED(hr) || query.Id != VIOGPU_TIMELINE_SUBMIT_VERSION ||
             !device->residency || !device->residency->wait_gpu ||
             !device->residency->signal_gpu ||
             !device->residency->create_sync ||
             !device->KTCallbacks.pfnDestroySynchronizationObjectCb) {
            yttrium_gdi_trace_warnf("yttrium: ERROR scheduled Present owner=d3dddi reason=unsupported-KMD-or-runtime hr=%lx version=%u\n", (unsigned long)hr, query.Id);
            ctx->base.destroy(&ctx->base);
            return E_NOTIMPL;
         }
         ctx->base.scheduled_present_mode = 1;
         ctx->base.reserve_present = gdikmt_d3dddi_reserve_present;
         ctx->base.signal_present = gdikmt_d3dddi_signal_present;
         ctx->base.wait_present = gdikmt_d3dddi_wait_present;
         yttrium_gdi_trace_debugf("yttrium: scheduled Present mode=%u context=%p\n", ctx->base.scheduled_present_mode, ctx->hContext);
      }

      *out_ctx = &ctx->base;
   }

   return Status;
}

NTSTATUS
gdikmt_d3dddi_createallocation(struct gdikmt_device *_device,
                               struct gdikmt_createallocation *options)
{
   struct gdikmt_device_d3dddi *device = gdikmt_device_d3dddi(_device);
   D3DKMT_HANDLE *resident_handles = NULL;
   if (device->residency && options->NumAllocations) {
      resident_handles = (D3DKMT_HANDLE *)
         CALLOC(options->NumAllocations, sizeof(*resident_handles));
      if (!resident_handles)
         return E_OUTOFMEMORY;
   }

   D3DDDICB_ALLOCATE createAllocation;
   memset(&createAllocation, 0, sizeof(createAllocation));
   createAllocation.NumAllocations = options->NumAllocations;
   createAllocation.pAllocationInfo = options->pAllocationInfo;

   createAllocation.pPrivateDriverData = options->pPrivateDriverData;
   createAllocation.PrivateDriverDataSize = options->PrivateDriverDataSize;

   HANDLE hRTResource =
      options->force_allocation_handle ? NULL : device->hRTResource;
   createAllocation.hResource = hRTResource;

   if (device->isPrimary) {
      options->pAllocationInfo->VidPnSourceId = device->allocationVidPn;
      options->pAllocationInfo->Flags.Primary = 1;
   }

   NTSTATUS Status = gdikmt_d3dddi_record_status(
      _device,
      device->KTCallbacks.pfnAllocateCb(device->hRTDevice,
                                        &createAllocation),
      "pfnAllocateCb");
   yttrium_gdi_trace_debugf("yttrium: pfnAllocateCb status=0x%lx hRTResource=%p hKMResource=0x%lx allocations=%u primary=%u vidpn=%u private_size=%u resource_private=%p\n",
                            (unsigned long)Status,
                            hRTResource,
                            (unsigned long)createAllocation.hKMResource,
                            createAllocation.NumAllocations,
                            device->isPrimary ? 1 : 0,
                            device->allocationVidPn,
                            createAllocation.PrivateDriverDataSize,
                            createAllocation.pPrivateDriverData);
   for (UINT i = 0; i < createAllocation.NumAllocations; i++) {
      const D3DDDI_ALLOCATIONINFO *info = &createAllocation.pAllocationInfo[i];
      yttrium_gdi_trace_debugf("yttrium: pfnAllocateCb allocation[%u] hAllocation=0x%lx pSystemMem=%p vidpn=%u flags=0x%x private_size=%u private=%p\n",
                               i,
                               (unsigned long)info->hAllocation,
                               info->pSystemMem,
                               info->VidPnSourceId,
                               info->Flags.Value,
                               info->PrivateDriverDataSize,
                               info->pPrivateDriverData);
      ResourceEvent(RESOURCE_EVENT_CB_AFTER,
                    (uint64_t)info->hAllocation,
                    hRTResource,
                    device->hRTResource,
                    (int32_t)Status,
                    RESOURCE_EVENT_CB_ALLOCATE,
                    options->force_allocation_handle ? 1 : 0,
                    (uint64_t)(uintptr_t)device->hRTDevice);
   }

   options->hAllocationResource = options->force_allocation_handle ? NULL :
      (HANDLE)(uintptr_t)createAllocation.hKMResource;
   options->hResource = options->force_allocation_handle ? NULL :
      (hRTResource ? hRTResource : options->hAllocationResource);
   options->hResourceIsD3D9Runtime =
      hRTResource && device->hRTResourceIsD3D9;

   if (NT_SUCCESS(Status) && resident_handles) {
      for (UINT i = 0; i < createAllocation.NumAllocations; ++i)
         resident_handles[i] = createAllocation.pAllocationInfo[i].hAllocation;
      Status = gdikmt_d3dddi_make_resident(device, createAllocation.NumAllocations,
                                          resident_handles);
      if (!NT_SUCCESS(Status)) {
         D3DDDICB_DEALLOCATE cleanup = {};
         cleanup.hResource = hRTResource;
         if (!cleanup.hResource) {
            cleanup.NumAllocations = createAllocation.NumAllocations;
            cleanup.HandleList = resident_handles;
         }
         HRESULT cleanup_hr = device->KTCallbacks.pfnDeallocateCb(device->hRTDevice, &cleanup);
         if (FAILED(cleanup_hr))
            yttrium_gdi_trace_warnf("yttrium: ERROR allocation residency cleanup owner=d3dddi status=0x%lx\n", (unsigned long)cleanup_hr);
         for (UINT i = 0; i < createAllocation.NumAllocations; ++i)
            createAllocation.pAllocationInfo[i].hAllocation = 0;
         options->hResource = options->hAllocationResource = NULL;
         options->hResourceIsD3D9Runtime = false;
      }
   }
   FREE(resident_handles);
   return Status;
}

NTSTATUS
gdikmt_d3dddi_destroyallocation(struct gdikmt_device *_device, HANDLE hResource,
                                D3DKMT_HANDLE hAllocation)
{
   struct gdikmt_device_d3dddi *device = gdikmt_device_d3dddi(_device);

   /* DestroyDevice is itself a runtime callback.  At this point the runtime
    * has destroyed all child resources and owns final cleanup of allocations
    * associated with the device.  Calling back into pfnDeallocateCb from the
    * Gallium screen/context teardown is rejected with E_INVALIDARG even
    * though the kernel allocation remains queryable until DestroyDevice
    * returns.  Do all driver-side unmapping before reaching here, then leave
    * the runtime-owned allocation for the enclosing device teardown. */
   if (_device->runtime_destroying) {
      yttrium_gdi_trace_debugf(
         "yttrium: allocation release delegated to runtime device teardown "
         "hAllocation=0x%lx hResource=%p hRTDevice=%p\n",
         (unsigned long)hAllocation, hResource, device->hRTDevice);
      return STATUS_SUCCESS;
   }

   D3DDDICB_DEALLOCATE destroyAllocation;
   /* HandleList is consumed by pfnDeallocateCb below, so its storage must
    * remain live after the branch which selects the deallocation path. */
   D3DKMT_HANDLE allocationHandle = hAllocation;
   memset(&destroyAllocation, 0, sizeof(destroyAllocation));

   if (hResource) {
      destroyAllocation.hResource = hResource;
   } else {
      destroyAllocation.NumAllocations = 1;
      destroyAllocation.HandleList = &allocationHandle;
   }

   ResourceEvent(RESOURCE_EVENT_CB_BEFORE,
                 (uint64_t)hAllocation,
                 destroyAllocation.hResource, NULL, 0,
                 RESOURCE_EVENT_CB_DEALLOCATE,
                 destroyAllocation.NumAllocations,
                 (uint64_t)(uintptr_t)device->hRTDevice);
   NTSTATUS status =
      device->KTCallbacks.pfnDeallocateCb(device->hRTDevice,
                                          &destroyAllocation);
   status = gdikmt_d3dddi_record_status(_device, status,
                                        "pfnDeallocateCb");
   if (!NT_SUCCESS(status)) {
      VIOGPU_ESCAPE resinfo;
      memset(&resinfo, 0, sizeof(resinfo));
      resinfo.Type = VIOGPU_RES_INFO;
      resinfo.DataLength = sizeof(VIOGPU_RES_INFO_REQ);
      resinfo.ResourceInfo.ResHandle = hAllocation;
      NTSTATUS probe_status = STATUS_DEVICE_REMOVED;
      if (gdikmt_device_get_reset_status(_device) == PIPE_NO_RESET)
         probe_status =
            gdikmt_d3dddi_escape(_device, &resinfo, sizeof(resinfo));

      yttrium_gdi_trace_warnf(
         "yttrium: allocation deallocate callback failed "
         "status=0x%lx hAllocation=0x%lx "
         "hResource=%p path=%s count=%u hRTDevice=%p "
         "callback=%s post_failure_res_info_status=0x%lx "
         "res_id=%u pid=%lu tid=%lu\n",
         (unsigned long)status, (unsigned long)hAllocation,
         destroyAllocation.hResource,
         destroyAllocation.hResource ? "resource" : "handle-list",
         destroyAllocation.NumAllocations, device->hRTDevice,
         "pfnDeallocateCb",
         (unsigned long)probe_status,
         NT_SUCCESS(probe_status) ? resinfo.ResourceInfo.Id : 0,
         GetCurrentProcessId(), GetCurrentThreadId());
   }
   ResourceEvent(RESOURCE_EVENT_CB_AFTER,
                 (uint64_t)hAllocation,
                 destroyAllocation.hResource, NULL, (int32_t)status,
                 RESOURCE_EVENT_CB_DEALLOCATE,
                 destroyAllocation.NumAllocations,
                 (uint64_t)(uintptr_t)device->hRTDevice);
   return status;
}

NTSTATUS
gdikmt_d3dddi_lockallocation(struct gdikmt_device *_device,
                             D3DKMT_HANDLE hAllocation,
                             D3DDDICB_LOCKFLAGS flags, void **out_ptr)
{
   struct gdikmt_device_d3dddi *device = gdikmt_device_d3dddi(_device);

   D3DDDICB_LOCK lock;
   memset(&lock, 0, sizeof(lock));
   lock.Flags = flags;
   lock.Flags.LockEntire = TRUE;
   lock.hAllocation = hAllocation;
   NTSTATUS Status = gdikmt_d3dddi_record_status(
      _device,
      device->KTCallbacks.pfnLockCb(device->hRTDevice, &lock),
      "pfnLockCb");

   *out_ptr = lock.pData;

   return Status;
}

NTSTATUS
gdikmt_d3dddi_unlockallocation(struct gdikmt_device *_device,
                               D3DKMT_HANDLE hAllocation)
{
   struct gdikmt_device_d3dddi *device = gdikmt_device_d3dddi(_device);

   D3DDDICB_UNLOCK unlock;
   memset(&unlock, 0, sizeof(unlock));
   unlock.NumAllocations = 1;
   unlock.phAllocations = &hAllocation;

   NTSTATUS status =
      device->KTCallbacks.pfnUnlockCb(device->hRTDevice, &unlock);
   return gdikmt_d3dddi_record_status(_device, status, "pfnUnlockCb");
}

NTSTATUS
gdikmt_d3dddi_queryallocation(struct gdikmt_device *_device,
                              struct gdikmt_openallocation *options)
{
   struct gdikmt_device_d3dddi *device = gdikmt_device_d3dddi(_device);

   if (device->pD3D9OpenResource) {
      options->hAllocationResource =
         (HANDLE)(uintptr_t)device->pD3D9OpenResource->hKMResource;
      options->hResource = device->hRTResource ?
         device->hRTResource :
         options->hAllocationResource;
      options->hResourceIsD3D9Runtime =
         device->hRTResource && device->hRTResourceIsD3D9;
      options->NumAllocations = device->pD3D9OpenResource->NumAllocations;
      options->PrivateDriverDataSize =
         device->pD3D9OpenResource->PrivateDriverDataSize;
      options->TotalBufferSize = 1;
      return STATUS_SUCCESS;
   }

   if (!device->pOpenResource)
      return STATUS_INVALID_PARAMETER;

   options->hAllocationResource =
      (HANDLE)(uintptr_t)device->pOpenResource->hKMResource.handle;
   options->hResource = device->hRTResource ?
      device->hRTResource : options->hAllocationResource;
   options->hResourceIsD3D9Runtime =
      device->hRTResource && device->hRTResourceIsD3D9;
   options->NumAllocations = device->pOpenResource->NumAllocations;
   options->PrivateDriverDataSize =
      device->pOpenResource->PrivateDriverDataSize;
   options->TotalBufferSize = 1;

   return STATUS_SUCCESS;
}

NTSTATUS
gdikmt_d3dddi_openallocation(struct gdikmt_device *_device,
                             struct gdikmt_openallocation *options)
{
   struct gdikmt_device_d3dddi *device = gdikmt_device_d3dddi(_device);

   if (device->pD3D9OpenResource) {
      for (UINT i = 0; i < options->NumAllocations; i++)
         options->pOpenAllocation[i] =
            device->pD3D9OpenResource->pOpenAllocationInfo[i];

      memcpy(options->pPrivateDriverData,
             device->pD3D9OpenResource->pPrivateDriverData,
             device->pD3D9OpenResource->PrivateDriverDataSize);

      options->hAllocationResource =
         (HANDLE)(uintptr_t)device->pD3D9OpenResource->hKMResource;
      options->hResource = device->hRTResource ?
         device->hRTResource : options->hAllocationResource;
      options->hResourceIsD3D9Runtime =
         device->hRTResource && device->hRTResourceIsD3D9;
      options->PrivateDriverDataSize =
         device->pD3D9OpenResource->PrivateDriverDataSize;
      return gdikmt_d3dddi_open_residency(device, options);
   }

   if (!device->pOpenResource)
      return STATUS_INVALID_PARAMETER;

   for (UINT i = 0; i < options->NumAllocations; i++) {
      options->pOpenAllocation[i] =
         device->pOpenResource->pOpenAllocationInfo[i];
   }

   memcpy(options->pPrivateDriverData,
          device->pOpenResource->pPrivateDriverData,
          device->pOpenResource->PrivateDriverDataSize);

   options->hAllocationResource =
      (HANDLE)(uintptr_t)device->pOpenResource->hKMResource.handle;
   options->hResource = device->hRTResource ?
      device->hRTResource : options->hAllocationResource;
   options->hResourceIsD3D9Runtime =
      device->hRTResource && device->hRTResourceIsD3D9;
   options->PrivateDriverDataSize =
      device->pOpenResource->PrivateDriverDataSize;

   return gdikmt_d3dddi_open_residency(device, options);
}

NTSTATUS
gdikmt_d3dddi_present(struct gdikmt_context *_ctx, D3DKMT_HANDLE hSrcAllocation,
                      void *winsys_drawable_handle, struct pipe_box *sub_box)
{
   struct gdikmt_context_d3dddi *ctx = gdikmt_context_d3dddi(_ctx);
   struct gdikmt_device_d3dddi *device = gdikmt_device_d3dddi(_ctx->device);
   const struct gdikmt_present_info *present_info =
      gdikmt_present_info_from_context(winsys_drawable_handle);

   if (!device->pDXGIBaseCallbacks && device->KTCallbacks.pfnPresentCb) {
      D3DDDICB_PRESENT kmPresent;
      memset(&kmPresent, 0, sizeof(kmPresent));
      kmPresent.hSrcAllocation = hSrcAllocation;
      kmPresent.hDstAllocation =
         present_info ? present_info->hDstAllocation : 0;
      kmPresent.hContext = ctx->hContext;
      kmPresent.BroadcastContextCount = 0;
      yttrium_gdi_trace_debugf("yttrium: d3d9 pfnPresentCb before hSrcAllocation=0x%lx hDstAllocation=0x%lx hContext=%p broadcast=%u sync_override_valid=%u sync_override=%u\n",
                               (unsigned long)kmPresent.hSrcAllocation,
                               (unsigned long)kmPresent.hDstAllocation,
                               kmPresent.hContext,
                               kmPresent.BroadcastContextCount,
                               kmPresent.SyncIntervalOverrideValid ? 1 : 0,
                               kmPresent.SyncIntervalOverride);
      HRESULT res = yttrium_prepare_present(present_info);
      if (FAILED(res)) {
         yttrium_gdi_trace_warnf("yttrium: ERROR: display callback suppressed owner=yttrium-runtime-callback reason=render-publication-not-issued callback=D3D9-pfnPresentCb status=0x%lx\n",
                                 (unsigned long)res);
         return res;
      }

      if (yttrium_present_skip_this_one()) {
         yttrium_gdi_trace_debugf("yttrium: d3d9 pfnPresentCb skipped by PRESENT_EVERY\n");
         return S_OK;
      }

      yttrium_gdi_trace_debugf("yttrium: d3d9 pfnPresentCb actual callback issue\n");
      res = device->KTCallbacks.pfnPresentCb(device->hRTDevice, &kmPresent);
      res = gdikmt_d3dddi_record_status(_ctx->device, res,
                                        "pfnPresentCb");
      yttrium_gdi_trace_debugf("yttrium: d3d9 pfnPresentCb after status=0x%lx hSrcAllocation=0x%lx hDstAllocation=0x%lx hContext=%p optimize=%u private_size=%u private=%p\n",
                               (unsigned long)res,
                               (unsigned long)kmPresent.hSrcAllocation,
                               (unsigned long)kmPresent.hDstAllocation,
                               kmPresent.hContext,
                               kmPresent.bOptimizeForComposition ? 1 : 0,
                               kmPresent.PrivateDriverDataSize,
                               kmPresent.pPrivateDriverData);
      return res;
   }

   DXGIDDICB_PRESENT kmPresent;
   memset(&kmPresent, 0, sizeof(kmPresent));
   kmPresent.hSrcAllocation = hSrcAllocation;
   kmPresent.hDstAllocation =
      present_info ? present_info->hDstAllocation : 0;
   kmPresent.pDXGIContext =
      present_info ? present_info->dxgi_context : winsys_drawable_handle;
   kmPresent.hContext = ctx->hContext;
   kmPresent.BroadcastContextCount = 0;
   yttrium_gdi_trace_present_callback(
      YTTRIUM_GDI_TRACE_PFN_PRESENT_BEFORE,
      0,
      (uint64_t)kmPresent.hSrcAllocation,
      (uint64_t)kmPresent.hDstAllocation,
      (uint64_t)(uintptr_t)kmPresent.hContext,
      (uint64_t)(uintptr_t)kmPresent.pDXGIContext,
      kmPresent.PrivateDriverDataSize,
      kmPresent.bOptimizeForComposition ? 1 : 0,
      kmPresent.BroadcastContextCount);
   yttrium_gdi_trace_debugf("yttrium: pfnPresentCb before hSrcAllocation=0x%lx hDstAllocation=0x%lx hContext=%p pDXGIContext=%p private_size=%u private=%p broadcast=%u sync_override_valid=%u sync_override=%u\n",
                            (unsigned long)kmPresent.hSrcAllocation,
                            (unsigned long)kmPresent.hDstAllocation,
                            kmPresent.hContext,
                            kmPresent.pDXGIContext,
                            kmPresent.PrivateDriverDataSize,
                            kmPresent.pPrivateDriverData,
                            kmPresent.BroadcastContextCount,
                            kmPresent.SyncIntervalOverrideValid ? 1 : 0,
                            kmPresent.SyncIntervalOverride);
   /*
    * Diagnostic: pfnPresentCb costs 19.4 ms per frame windowed, where it
    * blocks on DWM compositing the surface through the CPU, and 0.5 ms
    * fullscreen.  Skipping or thinning it measures what the frame costs
    * without the present round trip, which is the only way to separate the
    * render loop from the compositor.
    *
    * PRESENT_EVERY=0 skips the callback entirely.  PRESENT_EVERY=N issues one
    * callback in N and skips the rest, which keeps the window updating while
    * removing most of the KMD ResFlush round trips.  Rendering publication is
    * asynchronous and happens independently for every application Present.
    */
   HRESULT res = yttrium_prepare_present(present_info);
   if (FAILED(res)) {
      yttrium_gdi_trace_warnf("yttrium: ERROR: display callback suppressed owner=yttrium-runtime-callback reason=render-publication-not-issued callback=DXGI-pfnPresentCb status=0x%lx\n",
                              (unsigned long)res);
      yttrium_gdi_trace_present_callback(
         YTTRIUM_GDI_TRACE_PFN_PRESENT_AFTER, res,
         (uint64_t)kmPresent.hSrcAllocation,
         (uint64_t)kmPresent.hDstAllocation,
         (uint64_t)(uintptr_t)kmPresent.hContext,
         (uint64_t)(uintptr_t)kmPresent.pDXGIContext,
         kmPresent.PrivateDriverDataSize,
         kmPresent.bOptimizeForComposition ? 1 : 0,
         kmPresent.BroadcastContextCount);
      return res;
   }

   if (yttrium_present_skip_this_one()) {
      yttrium_gdi_trace_debugf("yttrium: pfnPresentCb skipped by PRESENT_EVERY\n");
      yttrium_gdi_trace_present_callback(
         YTTRIUM_GDI_TRACE_PFN_PRESENT_AFTER, 0,
         (uint64_t)kmPresent.hSrcAllocation,
         (uint64_t)kmPresent.hDstAllocation,
         (uint64_t)(uintptr_t)kmPresent.hContext,
         (uint64_t)(uintptr_t)kmPresent.pDXGIContext,
         kmPresent.PrivateDriverDataSize,
         kmPresent.bOptimizeForComposition ? 1 : 0,
         kmPresent.BroadcastContextCount);
      return S_OK;
   }

   yttrium_gdi_trace_debugf("yttrium: pfnPresentCb actual callback issue\n");
   res = device->pDXGIBaseCallbacks->pfnPresentCb(device->hRTDevice, &kmPresent);
   res = gdikmt_d3dddi_record_status(_ctx->device, res,
                                     "DXGI pfnPresentCb");
   yttrium_gdi_trace_present_callback(
      YTTRIUM_GDI_TRACE_PFN_PRESENT_AFTER,
      res,
      (uint64_t)kmPresent.hSrcAllocation,
      (uint64_t)kmPresent.hDstAllocation,
      (uint64_t)(uintptr_t)kmPresent.hContext,
      (uint64_t)(uintptr_t)kmPresent.pDXGIContext,
      kmPresent.PrivateDriverDataSize,
      kmPresent.bOptimizeForComposition ? 1 : 0,
      kmPresent.BroadcastContextCount);
   yttrium_gdi_trace_debugf("yttrium: pfnPresentCb after status=0x%lx hSrcAllocation=0x%lx hDstAllocation=0x%lx hContext=%p pDXGIContext=%p optimize=%u private_size=%u private=%p\n",
                            (unsigned long)res,
                            (unsigned long)kmPresent.hSrcAllocation,
                            (unsigned long)kmPresent.hDstAllocation,
                            kmPresent.hContext,
                            kmPresent.pDXGIContext,
                            kmPresent.bOptimizeForComposition ? 1 : 0,
                            kmPresent.PrivateDriverDataSize,
                            kmPresent.pPrivateDriverData);
   return res;
};

NTSTATUS
gdikmt_d3dddi_setdisplaymode(struct gdikmt_device *_device,
                             D3DKMT_HANDLE hSrcAllocation)
{
   struct gdikmt_device_d3dddi *device = gdikmt_device_d3dddi(_device);

   D3DDDICB_SETDISPLAYMODE setMode;
   memset(&setMode, 0, sizeof(setMode));
   setMode.hPrimaryAllocation = hSrcAllocation;
   setMode.PrivateDriverFormatAttribute = 0;

   NTSTATUS status =
      device->KTCallbacks.pfnSetDisplayModeCb(device->hRTDevice, &setMode);
   return gdikmt_d3dddi_record_status(_device, status,
                                      "pfnSetDisplayModeCb");
};

void
gdikmt_d3dddi_destroy(struct gdikmt_device *_device)
{
   struct gdikmt_device_d3dddi *device = gdikmt_device_d3dddi(_device);
   struct gdikmt_d3dddi_residency *state = device->residency;
   if (!state)
      return;
   /* Called after the Gallium workers and contexts have been destroyed. The
    * runtime releases remaining allocation residency references when it
    * closes the allocations/device; do not evict live GPU storage early. */
   D3DDDI_DESTROYPAGINGQUEUE destroy = {};
   destroy.hPagingQueue = state->queue.hPagingQueue;
   HRESULT hr = state->destroy_queue(device->hRTDevice, &destroy);
   if (FAILED(hr))
      yttrium_gdi_trace_warnf("yttrium: ERROR residency teardown owner=d3dddi operation=DestroyPagingQueue status=0x%lx\n", (unsigned long)hr);
   FREE(state);
   device->residency = NULL;
}

void
gdikmt_d3dddi_fill_basefuncs(struct gdikmt_device_d3dddi *device)
{
   device->base.destroy = gdikmt_d3dddi_destroy;
   device->base.queryAdapterInfo = gdikmt_d3dddi_queryadapterinfo;
   device->base.escape = gdikmt_d3dddi_escape;

   device->base.createContext = gdikmt_d3dddi_createcontext;

   device->base.createAllocation = gdikmt_d3dddi_createallocation;
   device->base.destroyAllocation = gdikmt_d3dddi_destroyallocation;
   device->base.lockAllocation = gdikmt_d3dddi_lockallocation;
   device->base.unlockAllocation = gdikmt_d3dddi_unlockallocation;
   device->base.queryAllocation = gdikmt_d3dddi_queryallocation;
   device->base.openAllocation = gdikmt_d3dddi_openallocation;

   device->base.present = gdikmt_d3dddi_present;
   device->base.setDisplayMode = gdikmt_d3dddi_setdisplaymode;
};
