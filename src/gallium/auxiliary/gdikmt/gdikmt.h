// Helper to iteract with windows kernel-mode drivers
// Required because of difference between D3DKMT**** functions and callbacks used in umd

#ifndef gdikmt_h
#define gdikmt_h

#include <windows.h>
#include <winternl.h>
#include <d3dkmthk.h>
#include <string.h>
#include "pipe/p_state.h"

typedef unsigned char boolean;

#define GDIKMT_PRESENT_INFO_MAGIC 0x544e5250u

typedef HRESULT (*gdikmt_prepare_present_func)(void *data);

struct gdikmt_present_info {
   UINT magic;
   UINT version;
   void *dxgi_context;
   D3DKMT_HANDLE hDstAllocation;
   HRESULT status;
   gdikmt_prepare_present_func prepare_present;
   void *prepare_present_data;
   boolean application_scanout;
};

static inline const struct gdikmt_present_info *
gdikmt_present_info_from_context(const void *context)
{
   const struct gdikmt_present_info *info =
      (const struct gdikmt_present_info *)context;

   return info && info->magic == GDIKMT_PRESENT_INFO_MAGIC ? info : NULL;
}

struct gdikmt_render {
   UINT CommandLength;
   UINT CommandOffset;
   UINT AllocationCount;
   UINT PatchLocationCount;
   
   UINT NewCommandBufferSize;
   UINT NewAllocationListSize;
   UINT NewPatchLocationListSize;
   
   boolean ResizeCommandBuffer;
   boolean ResizeAllocationList;
   boolean ResizePatchLocationList;

   HANDLE CompletionEvent;
};

struct gdikmt_createallocation {
  VOID *pPrivateDriverData;
  UINT PrivateDriverDataSize;
  HANDLE hResource;
  HANDLE hAllocationResource;
  boolean hResourceIsD3D9Runtime;
  UINT NumAllocations;
  D3DDDI_ALLOCATIONINFO  *pAllocationInfo;
  boolean force_allocation_handle;
};

struct gdikmt_openallocation {
   D3DKMT_HANDLE hGlobalHandle;
   HANDLE hResource;
   HANDLE hAllocationResource;
   boolean hResourceIsD3D9Runtime;
   UINT NumAllocations;
   
   VOID *pPrivateDriverData;
   UINT PrivateDriverDataSize;
   
   VOID *pTotalBuffer;
   UINT TotalBufferSize;

   D3DDDI_OPENALLOCATIONINFO *pOpenAllocation;

   UINT PrivateRuntimeSize;
};


struct gdikmt_context {
   struct gdikmt_device *device;
   
   void *pCommandBuffer;
   D3DDDI_ALLOCATIONLIST *pAllocationList;
   D3DDDI_PATCHLOCATIONLIST *pPatchLocationList;

   UINT CommandBufferSize;
   UINT AllocationListSize;
   UINT PatchLocationListSize;

   D3DKMT_HANDLE (*kmt_handle)(struct gdikmt_context* device);
   void (*destroy)(struct gdikmt_context* device);
   NTSTATUS (*render)(struct gdikmt_context* device, struct gdikmt_render *options);
   /* D3DDDI defaults to 1: wait for worker submit/signal issuance. */
   unsigned scheduled_present_mode;
   HRESULT (*reserve_present)(struct gdikmt_context *ctx,
                              struct gdikmt_context **consumer, uint64_t *value);
   HRESULT (*signal_present)(struct gdikmt_context *ctx, uint64_t value);
   HRESULT (*wait_present)(struct gdikmt_context *ctx, uint64_t value);
};

struct gdikmt_device {
   /* The D3D runtime owns device-associated allocations once its
    * DestroyDevice callback has begun.  Runtime callback backends must not
    * issue late pfnDeallocateCb calls from that phase. */
   boolean runtime_destroying;

   /* Device-loss reporting is shared by the Gallium context and the KMT
    * transport because either side can be the first to observe removal. */
   volatile LONG reset_status;
   volatile LONG reset_callback_notified;
   struct pipe_device_reset_callback reset_callback;

   void (*destroy)(struct gdikmt_device* device);

   NTSTATUS (*queryAdapterInfo)(
    struct gdikmt_device*   device,
    KMTQUERYADAPTERINFOTYPE Type,
    VOID*                   pPrivateDriverData,
    UINT                    PrivateDriverDataSize
   );

   NTSTATUS (*escape)(
    struct gdikmt_device*   device,
    VOID*                   pPrivateDriverData,
    UINT                    PrivateDriverDataSize
   );

   NTSTATUS (*createContext)(struct gdikmt_device* device, struct gdikmt_context** out_ctx);

   NTSTATUS (*createAllocation)(struct gdikmt_device* device, struct gdikmt_createallocation *options);
   NTSTATUS (*destroyAllocation)(struct gdikmt_device* device, HANDLE hResource,
                                 D3DKMT_HANDLE hAllocation);
   NTSTATUS (*lockAllocation)(struct gdikmt_device* device, D3DKMT_HANDLE hAllocation, D3DDDICB_LOCKFLAGS flags, void **out_ptr);
   NTSTATUS (*unlockAllocation)(struct gdikmt_device* device, D3DKMT_HANDLE hAllocation);
   NTSTATUS (*queryAllocation)(struct gdikmt_device* device, struct gdikmt_openallocation* options);
   NTSTATUS (*openAllocation)(struct gdikmt_device* device, struct gdikmt_openallocation* options);

   NTSTATUS (*present)(struct gdikmt_context* ctx, D3DKMT_HANDLE hSrcAllocation, void *winsys_drawable_handle, struct pipe_box *sub_box);
   NTSTATUS (*setDisplayMode)(struct gdikmt_device* device, D3DKMT_HANDLE hSrcAllocation);
};

static inline enum pipe_reset_status
gdikmt_device_get_reset_status(struct gdikmt_device *device)
{
   if (!device)
      return PIPE_NO_RESET;

   return (enum pipe_reset_status)InterlockedCompareExchange(
      &device->reset_status, PIPE_NO_RESET, PIPE_NO_RESET);
}

static inline void
gdikmt_device_notify_reset(struct gdikmt_device *device)
{
   enum pipe_reset_status status = gdikmt_device_get_reset_status(device);

   if (!device || status == PIPE_NO_RESET || device->runtime_destroying ||
       !device->reset_callback.reset)
      return;

   if (InterlockedCompareExchange(&device->reset_callback_notified, 1, 0) == 0)
      device->reset_callback.reset(device->reset_callback.data, status);
}

static inline void
gdikmt_device_set_reset_callback(
   struct gdikmt_device *device,
   const struct pipe_device_reset_callback *callback)
{
   if (!device)
      return;

   if (callback)
      device->reset_callback = *callback;
   else
      memset(&device->reset_callback, 0, sizeof(device->reset_callback));

   gdikmt_device_notify_reset(device);
}

static inline void
gdikmt_device_report_reset(struct gdikmt_device *device,
                           enum pipe_reset_status status)
{
   if (!device || status == PIPE_NO_RESET)
      return;

   InterlockedCompareExchange(&device->reset_status, (LONG)status,
                              PIPE_NO_RESET);
   gdikmt_device_notify_reset(device);
}


#ifdef __cplusplus
extern "C" {
#endif

struct gdikmt_device *gdikmt_create_from_hdc(HDC hDC);

#ifdef __cplusplus
}
#endif

#endif
