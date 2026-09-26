#include "gdikmt/gdikmt.h"
#include <d3d10umddi.h>

struct gdikmt_device_d3dddi {
   gdikmt_device base;

   HANDLE hRTAdapter;
   HANDLE hRTDevice;

   D3DDDI_ADAPTERCALLBACKS *pAdapterCallbacks;
   DXGI_DDI_BASE_CALLBACKS *pDXGIBaseCallbacks;

   UINT allocationVidPn;
   boolean isPrimary;
   HANDLE hRTResource;
   boolean hRTResourceIsD3D9;

   const D3D10DDIARG_OPENRESOURCE *pOpenResource;
   const D3DDDIARG_OPENRESOURCE *pD3D9OpenResource;
   boolean use_legacy_signal_sync;
   /* Keep this before the version-dependent callback table. D3D9 retains
    * its Win7 DDI layout; residency callbacks are copied from the runtime. */
   struct gdikmt_d3dddi_residency *residency;
   D3DDDI_DEVICECALLBACKS KTCallbacks;
};


struct gdikmt_context_d3dddi {
    struct gdikmt_context base;

    HANDLE hContext;
    struct gdikmt_context *present_context;
    D3DKMT_HANDLE present_fence;
    uint64_t present_value;
};

void gdikmt_d3dddi_fill_basefuncs(struct gdikmt_device_d3dddi *device);
HRESULT gdikmt_d3dddi_init_residency(struct gdikmt_device_d3dddi *device,
                                    const void *runtime_callbacks,
                                    UINT runtime_version);
