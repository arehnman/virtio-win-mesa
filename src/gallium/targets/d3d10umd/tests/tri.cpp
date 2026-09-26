/**************************************************************************
 *
 * Copyright 2012-2021 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 **************************************************************************/


#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <wctype.h>

#include <initguid.h>
#include <windows.h>
#include <intrin.h>

#include <d3d11.h>
#include <dxgi1_2.h>
#include <d3d9.h>

#include <wrl/client.h>

#include <vector>
#include <algorithm>
#include <math.h>

using Microsoft::WRL::ComPtr;

#include "tri_vs_4_0.h"
#include "tri_ps_4_0.h"
#include "tri_vertexid_texture_vs_4_0.h"
#include "tri_vertexid_cbuffer_vs_4_0.h"
#include "tri_texture_ps_4_0.h"

enum class BlendMode {
    None,
    Alpha,
    DstOnly,
};

enum class Api {
    D3D11,
    D3D9,
    D3D9Ex,
};

static void
usage(const char *argv0)
{
   fprintf(stderr,
             "Usage: %s [--hardware] [--api d3d11|d3d9|d3d9ex] [--bgra] [--srgb] [--fullscreen] [--width W] [--height H] [--frames N] [--flip-sequential] [--flip-discard] [--discard] [--clear-black] [--viewport-inset N] [--scissor-inset N] [--indexed] [--triangle-list] [--strip-quad] [--line-box] [--texture-probe] [--vertex-id-texture-probe] [--ubo-probe] [--cpu-vertex-probe] [--depth-state-probe] [--stencil-mask-probe] [--alpha-blend] [--dst-only-blend] [--write-mask-red] [--sample-mask-zero] [--cull-front] [--cull-back] [--front-cw] [--list-adapters] [--adapter N] [--adapter-name SUBSTR] [--output N] [--benchmark] [--present-only] [--animate] [--warmup N] [--sync-interval N] [--draw-batch-perf] [--draw-count N] [--retire-resource-probe] [--retire-count N] [--upload-perf] [--upload-bytes N] [--upload-iters N]\n",
            argv0);
}

static bool
parseUIntArg(int argc, char **argv, int *index, unsigned *value)
{
    if (*index + 1 >= argc) {
        return false;
    }

    char *end = nullptr;
    unsigned long parsed = strtoul(argv[*index + 1], &end, 0);
    if (!end || *end) {
        return false;
    }

    *value = (unsigned)parsed;
    (*index)++;
    return true;
}

static bool
wideContainsAsciiCaseInsensitive(const wchar_t *haystack, const char *needle)
{
    if (!needle || !*needle) {
        return true;
    }

    for (const wchar_t *h = haystack; h && *h; ++h) {
        size_t i = 0;
        while (needle[i] && h[i]) {
            wchar_t hc = towlower(h[i]);
            wchar_t nc = towlower((wchar_t)(unsigned char)needle[i]);
            if (hc != nc) {
                break;
            }
            ++i;
        }
        if (!needle[i]) {
            return true;
        }
    }

    return false;
}

static void
printAdapters(IDXGIFactory1 *factory)
{
    for (UINT adapterIndex = 0;; ++adapterIndex) {
        ComPtr<IDXGIAdapter1> adapter;
        HRESULT hr = factory->EnumAdapters1(adapterIndex, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        if (FAILED(hr)) {
            printf("adapter[%u]: EnumAdapters1 failed %lx\n", adapterIndex, hr);
            break;
        }

        DXGI_ADAPTER_DESC1 desc;
        ZeroMemory(&desc, sizeof desc);
        hr = adapter->GetDesc1(&desc);
        if (FAILED(hr)) {
            printf("adapter[%u]: GetDesc1 failed %lx\n", adapterIndex, hr);
            continue;
        }

        printf("adapter[%u]: %S vendor=0x%04x device=0x%04x subsys=0x%08x rev=0x%02x flags=0x%x\n",
               adapterIndex,
               desc.Description,
               desc.VendorId,
               desc.DeviceId,
               desc.SubSysId,
               desc.Revision,
               desc.Flags);

        for (UINT outputIndex = 0;; ++outputIndex) {
            ComPtr<IDXGIOutput> output;
            hr = adapter->EnumOutputs(outputIndex, &output);
            if (hr == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            if (FAILED(hr)) {
                printf("  output[%u]: EnumOutputs failed %lx\n", outputIndex, hr);
                break;
            }

            DXGI_OUTPUT_DESC outputDesc;
            ZeroMemory(&outputDesc, sizeof outputDesc);
            hr = output->GetDesc(&outputDesc);
            if (FAILED(hr)) {
                printf("  output[%u]: GetDesc failed %lx\n", outputIndex, hr);
                continue;
            }

            RECT r = outputDesc.DesktopCoordinates;
            printf("  output[%u]: %S attached=%u desktop={%ld,%ld,%ld,%ld}\n",
                   outputIndex,
                   outputDesc.DeviceName,
                   outputDesc.AttachedToDesktop ? 1 : 0,
                   r.left,
                   r.top,
                   r.right,
                   r.bottom);
        }
    }
}

static bool
selectAdapter(IDXGIFactory1 *factory,
              bool byIndex,
              UINT adapterIndex,
              const char *adapterName,
              IDXGIAdapter1 **outAdapter)
{
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        HRESULT hr = factory->EnumAdapters1(i, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        if (FAILED(hr)) {
            fprintf(stderr, "EnumAdapters1(%u) failed %lx\n", i, hr);
            return false;
        }

        DXGI_ADAPTER_DESC1 desc;
        ZeroMemory(&desc, sizeof desc);
        hr = adapter->GetDesc1(&desc);
        if (FAILED(hr)) {
            fprintf(stderr, "GetDesc1(%u) failed %lx\n", i, hr);
            return false;
        }

        if ((byIndex && i == adapterIndex) ||
            (adapterName &&
             wideContainsAsciiCaseInsensitive(desc.Description, adapterName))) {
            printf("selected adapter[%u]: %S\n", i, desc.Description);
            *outAdapter = adapter.Detach();
            return true;
        }
    }

    if (byIndex) {
        fprintf(stderr, "adapter[%u] not found\n", adapterIndex);
    } else {
        fprintf(stderr, "adapter matching \"%s\" not found\n", adapterName);
    }
    return false;
}

struct SampleStats {
    double mean;
    double min;
    double median;
    double p95;
    double p99;
    double max;
    double stddev;
    double sum;
};

/* reportPresentBenchmark() below computes the same statistics inline and is
 * deliberately left alone: its output is being compared across kernel and BAR
 * configurations, so it must not shift while an investigation is in flight. */
static SampleStats
computeStats(std::vector<double> &samples)
{
    SampleStats s = {};
    if (samples.empty()) {
        return s;
    }
    std::sort(samples.begin(), samples.end());
    const size_t n = samples.size();
    for (double v : samples) {
        s.sum += v;
    }
    s.mean = s.sum / (double)n;
    double var = 0.0;
    for (double v : samples) {
        var += (v - s.mean) * (v - s.mean);
    }
    s.stddev = sqrt(var / (double)n);
    auto pct = [&](double p) -> double {
        size_t idx = (size_t)(p / 100.0 * (double)(n - 1) + 0.5);
        if (idx >= n) {
            idx = n - 1;
        }
        return samples[idx];
    };
    s.min = samples.front();
    s.max = samples.back();
    s.median = pct(50);
    s.p95 = pct(95);
    s.p99 = pct(99);
    return s;
}

/* Guest CPU write bandwidth into a mapped dynamic buffer - the path every
 * vertex, index and constant upload takes.  Map cost and write cost are timed
 * separately because they answer different questions: where the mapping lives
 * shows up in the memcpy (cached system memory and write-combining device
 * memory differ by roughly an order of magnitude), while the map itself barely
 * moves. */
static void
reportUploadBenchmark(std::vector<double> &mapMs, std::vector<double> &writeMs,
                      std::vector<double> &unmapMs, unsigned bytes)
{
    if (writeMs.empty()) {
        return;
    }
    const SampleStats map = computeStats(mapMs);
    const SampleStats write = computeStats(writeMs);
    const SampleStats unmap = computeStats(unmapMs);
    const double mib = (double)bytes / (1024.0 * 1024.0);
    auto bandwidth = [&](double ms) -> double {
        return ms > 0.0 ? mib / 1024.0 / (ms / 1000.0) : 0.0;
    };
    printf("\n=== upload benchmark ===\n");
    printf("samples=%u bytes=%u (%.2f MiB per iteration)\n",
           (unsigned)writeMs.size(), bytes, mib);
    printf("map ms:   avg=%.4f min=%.4f median=%.4f p95=%.4f p99=%.4f max=%.4f stddev=%.4f\n",
           map.mean, map.min, map.median, map.p95, map.p99, map.max,
           map.stddev);
    printf("write ms: avg=%.4f min=%.4f median=%.4f p95=%.4f p99=%.4f max=%.4f stddev=%.4f\n",
           write.mean, write.min, write.median, write.p95, write.p99,
           write.max, write.stddev);
    printf("unmap ms: avg=%.4f min=%.4f median=%.4f p95=%.4f p99=%.4f max=%.4f stddev=%.4f\n",
           unmap.mean, unmap.min, unmap.median, unmap.p95, unmap.p99,
           unmap.max, unmap.stddev);
    printf("write bandwidth: avg=%.2f GiB/s median=%.2f GiB/s (best %.2f GiB/s)\n",
           bandwidth(write.mean), bandwidth(write.median),
           bandwidth(write.min));
    printf("upload rate: %.1f uploads/sec  (map+write+unmap avg=%.4f ms)\n",
           (map.mean + write.mean + unmap.mean) > 0.0 ?
              1000.0 / (map.mean + write.mean + unmap.mean) : 0.0,
           map.mean + write.mean + unmap.mean);
}

/* Shared by every API path so the "=== present benchmark ===" block is byte
 * identical and the numbers are directly comparable across d3d11/d3d9/d3d9ex. */
static void
reportPresentBenchmark(std::vector<double> &presentMs, bool presentOnly,
                       unsigned syncInterval, unsigned width, unsigned height,
                       const char *swapName, const char *formatName)
{
    if (presentMs.empty()) {
        return;
    }
    std::sort(presentMs.begin(), presentMs.end());
    const size_t n = presentMs.size();
    double sum = 0.0;
    for (double v : presentMs) {
        sum += v;
    }
    const double mean = sum / (double)n;
    double var = 0.0;
    for (double v : presentMs) {
        var += (v - mean) * (v - mean);
    }
    var /= (double)n;
    const double stddev = sqrt(var);
    auto pct = [&](double p) -> double {
        size_t idx = (size_t)(p / 100.0 * (double)(n - 1) + 0.5);
        if (idx >= n) {
            idx = n - 1;
        }
        return presentMs[idx];
    };
    printf("\n=== present benchmark ===\n");
    printf("samples=%u present-only=%u sync-interval=%u %ux%u swap=%s format=%s\n",
           (unsigned)n, presentOnly ? 1 : 0, syncInterval, width, height,
           swapName, formatName);
    printf("present ms: avg=%.4f min=%.4f median=%.4f p95=%.4f p99=%.4f max=%.4f stddev=%.4f\n",
           mean, presentMs.front(), pct(50), pct(95), pct(99),
           presentMs.back(), stddev);
    printf("present rate: %.1f presents/sec  (total %.2f ms over %u samples)\n",
           mean > 0.0 ? 1000.0 / mean : 0.0, sum, (unsigned)n);
}

/* Wall-clock throughput over the measured frames.  The present statistics above
 * bracket Present() only, so "present rate" is what presenting alone could
 * sustain - map, draw and submit costs all sit outside that bracket.  This is
 * the number to compare whenever a change touches anything but Present. */
static void
reportFrameRate(unsigned frames, double wallMs)
{
    if (!frames || wallMs <= 0.0) {
        return;
    }
    printf("frame rate: %.1f fps  (%u frames in %.2f ms, %.4f ms/frame)\n",
           1000.0 * (double)frames / wallMs, frames, wallMs,
           wallMs / (double)frames);
}

static bool
readD3D9RenderTargetPixel(IDirect3DDevice9 *dev, UINT x, UINT y,
                          D3DCOLOR *color)
{
    IDirect3DSurface9 *target = nullptr;
    IDirect3DSurface9 *readback = nullptr;
    D3DSURFACE_DESC desc = {};
    D3DLOCKED_RECT locked = {};
    bool ok = false;

    HRESULT hr = dev->GetRenderTarget(0, &target);
    if (FAILED(hr) || !target)
        goto out;
    hr = target->GetDesc(&desc);
    if (FAILED(hr) || x >= desc.Width || y >= desc.Height ||
        (desc.Format != D3DFMT_X8R8G8B8 &&
         desc.Format != D3DFMT_A8R8G8B8))
        goto out;
    hr = dev->CreateOffscreenPlainSurface(desc.Width, desc.Height, desc.Format,
                                          D3DPOOL_SYSTEMMEM, &readback,
                                          nullptr);
    if (FAILED(hr) || !readback)
        goto out;
    hr = dev->GetRenderTargetData(target, readback);
    if (FAILED(hr))
        goto out;
    hr = readback->LockRect(&locked, nullptr, D3DLOCK_READONLY);
    if (FAILED(hr))
        goto out;

    *color = *(const D3DCOLOR *)((const uint8_t *)locked.pBits +
                                y * locked.Pitch + x * sizeof(D3DCOLOR));
    readback->UnlockRect();
    ok = true;

out:
    if (readback)
        readback->Release();
    if (target)
        target->Release();
    return ok;
}

/* D3D9 / D3D9Ex path.  Legacy (Direct3DCreate9 + CreateDevice) uses the
 * blt-model D3DSWAPEFFECT_DISCARD; Ex (Direct3DCreate9Ex + CreateDeviceEx)
 * uses the flip-model D3DSWAPEFFECT_FLIPEX.  The animation and benchmark loop
 * mirror the D3D11 path (CPU-rotated triangle re-uploaded per frame, wall-clock
 * driven, present timed with QPC) so results line up with --api d3d11. */
static int
runD3D9(bool ex,
        unsigned windowWidth, unsigned windowHeight, bool fullscreen,
        unsigned frames, bool framesExplicitlySet, bool benchmark,
        bool presentOnly, bool animate, unsigned warmupFrames,
        unsigned syncInterval, bool clearBlack, bool writeMaskRed)
{
    HINSTANCE hInstance = GetModuleHandle(nullptr);

    WNDCLASSEX wc = {
        sizeof(WNDCLASSEX), CS_CLASSDC, DefWindowProc, 0, 0,
        hInstance, nullptr, nullptr, nullptr, nullptr, "tri_d3d9", nullptr
    };
    RegisterClassEx(&wc);

    DWORD dwStyle =
        WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_OVERLAPPEDWINDOW | WS_VISIBLE;
    RECT rect = {0, 0, (LONG)windowWidth, (LONG)windowHeight};
    AdjustWindowRect(&rect, dwStyle, false);

    HWND hWnd = CreateWindow(wc.lpszClassName,
                             ex ? "tri (d3d9ex)" : "tri (d3d9)",
                             dwStyle, CW_USEDEFAULT, CW_USEDEFAULT,
                             rect.right - rect.left, rect.bottom - rect.top,
                             nullptr, nullptr, hInstance, nullptr);
    if (!hWnd) {
        return EXIT_FAILURE;
    }

    HRESULT hr;
    IDirect3D9 *d3d = nullptr;
    IDirect3D9Ex *d3dEx = nullptr;
    if (ex) {
        hr = Direct3DCreate9Ex(D3D_SDK_VERSION, &d3dEx);
        if (FAILED(hr) || !d3dEx) {
            fprintf(stderr, "Direct3DCreate9Ex failed %lx\n", hr);
            return EXIT_FAILURE;
        }
        d3d = d3dEx;
    } else {
        d3d = Direct3DCreate9(D3D_SDK_VERSION);
        if (!d3d) {
            fprintf(stderr, "Direct3DCreate9 failed\n");
            return EXIT_FAILURE;
        }
    }

    const char *swapName = ex ? "d3d9ex-flipex" : "d3d9-discard";

    D3DPRESENT_PARAMETERS pp;
    ZeroMemory(&pp, sizeof pp);
    pp.Windowed = fullscreen ? FALSE : TRUE;
    pp.hDeviceWindow = hWnd;
    pp.BackBufferWidth = windowWidth;
    pp.BackBufferHeight = windowHeight;
    pp.BackBufferFormat = D3DFMT_X8R8G8B8;
    pp.BackBufferCount = 1;
    pp.SwapEffect = ex ? D3DSWAPEFFECT_FLIPEX : D3DSWAPEFFECT_DISCARD;
    pp.PresentationInterval = syncInterval ? D3DPRESENT_INTERVAL_ONE
                                           : D3DPRESENT_INTERVAL_IMMEDIATE;

    IDirect3DDevice9 *dev = nullptr;
    IDirect3DDevice9Ex *devEx = nullptr;
    DWORD behavior =
        D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE;
    if (ex) {
        hr = d3dEx->CreateDeviceEx(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hWnd,
                                   behavior, &pp, nullptr, &devEx);
        if (FAILED(hr)) {
            fprintf(stderr,
                    "CreateDeviceEx(hardware) failed %lx; retrying software\n",
                    hr);
            behavior =
                D3DCREATE_SOFTWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE;
            hr = d3dEx->CreateDeviceEx(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hWnd,
                                       behavior, &pp, nullptr, &devEx);
        }
        if (FAILED(hr)) {
            fprintf(stderr, "CreateDeviceEx failed %lx\n", hr);
            return EXIT_FAILURE;
        }
        dev = devEx;
    } else {
        hr = d3d->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hWnd,
                               behavior, &pp, &dev);
        if (FAILED(hr)) {
            fprintf(stderr,
                    "CreateDevice(hardware) failed %lx; retrying software\n",
                    hr);
            behavior =
                D3DCREATE_SOFTWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE;
            hr = d3d->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hWnd,
                                   behavior, &pp, &dev);
        }
        if (FAILED(hr)) {
            fprintf(stderr, "CreateDevice failed %lx\n", hr);
            return EXIT_FAILURE;
        }
    }

    printf("d3d9%s device created swap=%s %ux%u windowed=%u\n",
           ex ? "ex" : "", swapName, windowWidth, windowHeight,
           pp.Windowed ? 1 : 0);

    struct D3D9Vertex {
        float x, y, z;
        D3DCOLOR color;
    };
    const DWORD fvf = D3DFVF_XYZ | D3DFVF_DIFFUSE;
    /* Clip-space positions (identity fixed-function transforms) so the triangle
     * matches the D3D11 one; lighting off so DIFFUSE shows through. */
    const D3D9Vertex triangle[3] = {
        { -0.9f, -0.9f, 0.5f, D3DCOLOR_COLORVALUE(0.8f, 0.0f, 0.0f, 1.0f) },
        {  0.9f, -0.9f, 0.5f, D3DCOLOR_COLORVALUE(0.0f, 0.9f, 0.0f, 1.0f) },
        {  0.0f,  0.9f, 0.5f, D3DCOLOR_COLORVALUE(0.0f, 0.0f, 0.7f, 1.0f) },
    };

    IDirect3DVertexBuffer9 *vb = nullptr;
    hr = dev->CreateVertexBuffer(sizeof triangle,
                                 D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY, fvf,
                                 D3DPOOL_DEFAULT, &vb, nullptr);
    if (FAILED(hr)) {
        fprintf(stderr, "CreateVertexBuffer failed %lx\n", hr);
        return EXIT_FAILURE;
    }
    {
        void *p = nullptr;
        if (SUCCEEDED(vb->Lock(0, sizeof triangle, &p, D3DLOCK_DISCARD))) {
            memcpy(p, triangle, sizeof triangle);
            vb->Unlock();
        }
    }

    dev->SetRenderState(D3DRS_LIGHTING, FALSE);
    dev->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
    dev->SetRenderState(D3DRS_ZENABLE, FALSE);
    if (writeMaskRed) {
        hr = dev->SetRenderState(D3DRS_COLORWRITEENABLE,
                                 D3DCOLORWRITEENABLE_RED);
        if (FAILED(hr)) {
            fprintf(stderr, "SetRenderState(COLORWRITEENABLE) failed %lx\n",
                    hr);
            return EXIT_FAILURE;
        }
    }
    dev->SetFVF(fvf);
    dev->SetStreamSource(0, vb, 0, sizeof(D3D9Vertex));

    const float baseColor[3] = {
        clearBlack || writeMaskRed ? 0.0f : 0.3f,
        clearBlack || writeMaskRed ? 0.0f : 0.1f,
        clearBlack || writeMaskRed ? 0.0f : 0.3f,
    };

    std::vector<double> presentMs;
    double qpcToMs = 0.0;
    if (benchmark) {
        LARGE_INTEGER qpf;
        QueryPerformanceFrequency(&qpf);
        qpcToMs = 1000.0 / (double)qpf.QuadPart;
        presentMs.reserve(frames);
        printf("benchmark: frames=%u warmup=%u present-only=%u sync-interval=%u %ux%u swap=%s format=D3DFMT_X8R8G8B8\n",
               frames, warmupFrames, presentOnly ? 1 : 0, syncInterval,
               windowWidth, windowHeight, swapName);
    }

    LARGE_INTEGER animQpf = {};
    LARGE_INTEGER animStart = {};
    if (animate) {
        QueryPerformanceFrequency(&animQpf);
        QueryPerformanceCounter(&animStart);
    }

    const bool animateForever = animate && !benchmark && !framesExplicitlySet;
    const unsigned totalFrames = frames + (benchmark ? warmupFrames : 0);
    bool colorMaskProbeRan = false;

    for (unsigned frame = 0; animateForever || frame < totalFrames; ++frame) {
        if (benchmark || animate) {
            MSG msg;
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            if (animateForever && !IsWindow(hWnd)) {
                break;
            }
        }

        const bool drawThisFrame = !presentOnly || animate;
        if (drawThisFrame) {
            float r = baseColor[0], g = baseColor[1], b = baseColor[2];
            if (animate) {
                LARGE_INTEGER nowCounter;
                QueryPerformanceCounter(&nowCounter);
                const double tSec = (double)(nowCounter.QuadPart -
                                             animStart.QuadPart) /
                                    (double)animQpf.QuadPart;
                const float angle = (float)(tSec * 1.5707963);  /* 4 s / rev */
                const float c = cosf(angle);
                const float s = sinf(angle);

                D3D9Vertex rot[3];
                for (int v = 0; v < 3; ++v) {
                    rot[v] = triangle[v];
                    rot[v].x = triangle[v].x * c - triangle[v].y * s;
                    rot[v].y = triangle[v].x * s + triangle[v].y * c;
                }
                void *p = nullptr;
                if (SUCCEEDED(vb->Lock(0, sizeof rot, &p, D3DLOCK_DISCARD))) {
                    memcpy(p, rot, sizeof rot);
                    vb->Unlock();
                }

                const float colorPhase = (float)(tSec * 0.9);
                r = 0.5f + 0.5f * sinf(colorPhase);
                g = 0.5f + 0.5f * sinf(colorPhase + 2.0944f);
                b = 0.5f + 0.5f * sinf(colorPhase + 4.1888f);
            }
            if (writeMaskRed)
                r = g = b = 0.0f;

            /* Flip-model (FLIPEX) rotates backbuffers; make sure we render
             * into the current one.  Harmless no-op for DISCARD, but if the
             * driver doesn't keep the implicit render target pointed at the
             * current backbuffer this is what makes a flip present show
             * content instead of a stale/blank surface. */
            IDirect3DSurface9 *backBuffer = nullptr;
            if (SUCCEEDED(dev->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO,
                                             &backBuffer))) {
                dev->SetRenderTarget(0, backBuffer);
                backBuffer->Release();
            }

            dev->Clear(0, nullptr, D3DCLEAR_TARGET,
                       D3DCOLOR_COLORVALUE(r, g, b, 1.0f), 1.0f, 0);
            dev->BeginScene();
            hr = dev->DrawPrimitive(D3DPT_TRIANGLELIST, 0, 1);
            dev->EndScene();
            if (FAILED(hr)) {
                fprintf(stderr, "DrawPrimitive failed %lx\n", hr);
                return EXIT_FAILURE;
            }

            if (writeMaskRed && frame == 0) {
                D3DCOLOR pixel = 0;
                if (!readD3D9RenderTargetPixel(dev, windowWidth / 2,
                                               windowHeight / 2, &pixel)) {
                    fprintf(stderr, "D3D9 color write-mask readback failed\n");
                    return EXIT_FAILURE;
                }
                printf("d3d9 color write-mask probe pixel=0x%08lx\n", pixel);
                if (!(pixel & 0x00ff0000u) || (pixel & 0x0000ffffu)) {
                    fprintf(stderr,
                            "D3D9 red-only write mask mismatch pixel=0x%08lx\n",
                            pixel);
                    return EXIT_FAILURE;
                }
                colorMaskProbeRan = true;
            }
        }

        LARGE_INTEGER t0 = {}, t1 = {};
        if (benchmark) {
            QueryPerformanceCounter(&t0);
        }

        if (ex) {
            hr = devEx->PresentEx(nullptr, nullptr, nullptr, nullptr, 0);
        } else {
            hr = dev->Present(nullptr, nullptr, nullptr, nullptr);
        }

        if (benchmark) {
            QueryPerformanceCounter(&t1);
            if (frame >= warmupFrames) {
                presentMs.push_back((double)(t1.QuadPart - t0.QuadPart) *
                                    qpcToMs);
            }
        } else if (!animate) {
            printf("present frame=%u hr=%lx\n", frame, hr);
        }

        if (FAILED(hr)) {
            fprintf(stderr, "Present failed %lx\n", hr);
            break;
        }
    }

    if (writeMaskRed && !colorMaskProbeRan) {
        fprintf(stderr,
                "D3D9 color write-mask probe did not execute a draw/readback\n");
        return EXIT_FAILURE;
    }

    if (benchmark && !presentMs.empty()) {
        reportPresentBenchmark(presentMs, presentOnly, syncInterval,
                               windowWidth, windowHeight, swapName,
                               "D3DFMT_X8R8G8B8");
    } else if (!benchmark && !animate) {
        Sleep(5000);
    }

    dev->SetStreamSource(0, nullptr, 0, 0);
    if (vb) {
        vb->Release();
    }
    if (dev) {
        dev->Release();  /* devEx aliases dev */
    }
    if (d3d) {
        d3d->Release();  /* d3dEx aliases d3d */
    }
    DestroyWindow(hWnd);
    return EXIT_SUCCESS;
}

int
main(int argc, char *argv[])
{
    HRESULT hr;
    Api api = Api::D3D11;
    bool fullscreen = false;
    bool bgra = false;
    bool srgb = false;
    unsigned windowWidth = 250;
    unsigned windowHeight = 250;
    unsigned frames = 1;
    DXGI_SWAP_EFFECT swapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    bool clearBlack = false;
    unsigned viewportInset = 0;
    unsigned scissorInset = 0;
    bool useScissor = false;
    bool useIndexed = false;
    bool useTriangleList = false;
    bool useStripQuad = false;
    bool useLineBox = false;
    bool textureProbe = false;
    bool vertexIdTextureProbe = false;
    bool uboProbe = false;
    bool cpuVertexProbe = false;
    bool depthStateProbe = false;
    bool stencilMaskProbe = false;
    BlendMode blendMode = BlendMode::None;
    bool writeMaskRed = false;
    bool sampleMaskZero = false;
    D3D11_CULL_MODE cullMode = D3D11_CULL_NONE;
    bool frontCounterClockwise = true;
    bool listAdapters = false;
    bool selectAdapterByIndex = false;
    UINT selectedAdapterIndex = UINT_MAX;
    const char *selectedAdapterName = nullptr;
    bool selectOutput = false;
    UINT selectedOutputIndex = 0;
    bool benchmark = false;
    bool presentOnly = false;
    bool animate = false;
    bool framesExplicitlySet = false;
    unsigned warmupFrames = 100;
    unsigned syncInterval = 0;
    bool drawBatchPerf = false;
    unsigned drawCount = 50000;
    bool retireResourceProbe = false;
    unsigned retireCount = 256;
    bool uploadPerf = false;
    unsigned uploadBytes = 8u << 20;
    unsigned uploadIters = 200;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--hardware") == 0) {
            /* Hardware is the default and only supported path. */
        } else if (strcmp(argv[i], "--api") == 0) {
            if (++i >= argc) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            if (strcmp(argv[i], "d3d11") == 0) {
                api = Api::D3D11;
            } else if (strcmp(argv[i], "d3d9") == 0) {
                api = Api::D3D9;
            } else if (strcmp(argv[i], "d3d9ex") == 0) {
                api = Api::D3D9Ex;
            } else {
                fprintf(stderr, "Unknown --api: %s\n", argv[i]);
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strncmp(argv[i], "--api=", 6) == 0) {
            const char *v = argv[i] + 6;
            if (strcmp(v, "d3d11") == 0) {
                api = Api::D3D11;
            } else if (strcmp(v, "d3d9") == 0) {
                api = Api::D3D9;
            } else if (strcmp(v, "d3d9ex") == 0) {
                api = Api::D3D9Ex;
            } else {
                fprintf(stderr, "Unknown --api: %s\n", v);
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--bgra") == 0) {
            bgra = true;
        } else if (strcmp(argv[i], "--srgb") == 0) {
            srgb = true;
        } else if (strcmp(argv[i], "--fullscreen") == 0) {
            fullscreen = true;
        } else if (strcmp(argv[i], "--width") == 0) {
            if (!parseUIntArg(argc, argv, &i, &windowWidth)) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--height") == 0) {
            if (!parseUIntArg(argc, argv, &i, &windowHeight)) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--frames") == 0) {
            if (!parseUIntArg(argc, argv, &i, &frames) || frames == 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            framesExplicitlySet = true;
        } else if (strncmp(argv[i], "--frames=", 9) == 0) {
            char *end = nullptr;
            unsigned long value = strtoul(argv[i] + 9, &end, 0);
            if (!end || *end || value == 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            frames = (unsigned)value;
            framesExplicitlySet = true;
        } else if (strcmp(argv[i], "--flip-sequential") == 0) {
            swapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        } else if (strcmp(argv[i], "--flip-discard") == 0) {
            swapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        } else if (strcmp(argv[i], "--discard") == 0) {
            swapEffect = DXGI_SWAP_EFFECT_DISCARD;
        } else if (strcmp(argv[i], "--clear-black") == 0) {
            clearBlack = true;
        } else if (strcmp(argv[i], "--viewport-inset") == 0) {
            if (!parseUIntArg(argc, argv, &i, &viewportInset)) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strncmp(argv[i], "--viewport-inset=", 17) == 0) {
            char *end = nullptr;
            unsigned long value = strtoul(argv[i] + 17, &end, 0);
            if (!end || *end) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            viewportInset = (unsigned)value;
        } else if (strcmp(argv[i], "--scissor-inset") == 0) {
            if (!parseUIntArg(argc, argv, &i, &scissorInset)) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            useScissor = true;
        } else if (strncmp(argv[i], "--scissor-inset=", 17) == 0) {
            char *end = nullptr;
            unsigned long value = strtoul(argv[i] + 17, &end, 0);
            if (!end || *end) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            scissorInset = (unsigned)value;
            useScissor = true;
        } else if (strcmp(argv[i], "--indexed") == 0) {
            useIndexed = true;
        } else if (strcmp(argv[i], "--triangle-list") == 0) {
            useTriangleList = true;
        } else if (strcmp(argv[i], "--strip-quad") == 0) {
            useStripQuad = true;
        } else if (strcmp(argv[i], "--line-box") == 0) {
            useLineBox = true;
        } else if (strcmp(argv[i], "--texture-probe") == 0) {
            textureProbe = true;
        } else if (strcmp(argv[i], "--vertex-id-texture-probe") == 0) {
            vertexIdTextureProbe = true;
            textureProbe = true;
        } else if (strcmp(argv[i], "--ubo-probe") == 0) {
            uboProbe = true;
        } else if (strcmp(argv[i], "--cpu-vertex-probe") == 0) {
            cpuVertexProbe = true;
        } else if (strcmp(argv[i], "--depth-state-probe") == 0) {
            depthStateProbe = true;
        } else if (strcmp(argv[i], "--stencil-mask-probe") == 0) {
            depthStateProbe = true;
            stencilMaskProbe = true;
        } else if (strcmp(argv[i], "--alpha-blend") == 0) {
            blendMode = BlendMode::Alpha;
        } else if (strcmp(argv[i], "--dst-only-blend") == 0) {
            blendMode = BlendMode::DstOnly;
        } else if (strcmp(argv[i], "--write-mask-red") == 0) {
            writeMaskRed = true;
        } else if (strcmp(argv[i], "--sample-mask-zero") == 0) {
            sampleMaskZero = true;
        } else if (strcmp(argv[i], "--cull-front") == 0) {
            cullMode = D3D11_CULL_FRONT;
        } else if (strcmp(argv[i], "--cull-back") == 0) {
            cullMode = D3D11_CULL_BACK;
        } else if (strcmp(argv[i], "--front-cw") == 0) {
            frontCounterClockwise = false;
        } else if (strcmp(argv[i], "--list-adapters") == 0) {
            listAdapters = true;
        } else if (strcmp(argv[i], "--adapter") == 0) {
            unsigned value;
            if (!parseUIntArg(argc, argv, &i, &value)) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            selectAdapterByIndex = true;
            selectedAdapterIndex = value;
        } else if (strncmp(argv[i], "--adapter=", 10) == 0) {
            char *end = nullptr;
            unsigned long value = strtoul(argv[i] + 10, &end, 0);
            if (!end || *end) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            selectAdapterByIndex = true;
            selectedAdapterIndex = (UINT)value;
        } else if (strcmp(argv[i], "--adapter-name") == 0) {
            if (++i >= argc) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            selectedAdapterName = argv[i];
        } else if (strncmp(argv[i], "--adapter-name=", 15) == 0) {
            selectedAdapterName = argv[i] + 15;
        } else if (strcmp(argv[i], "--output") == 0) {
            unsigned value;
            if (!parseUIntArg(argc, argv, &i, &value)) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            selectOutput = true;
            selectedOutputIndex = value;
        } else if (strncmp(argv[i], "--output=", 9) == 0) {
            char *end = nullptr;
            unsigned long value = strtoul(argv[i] + 9, &end, 0);
            if (!end || *end) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            selectOutput = true;
            selectedOutputIndex = (UINT)value;
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            benchmark = true;
        } else if (strcmp(argv[i], "--present-only") == 0) {
            presentOnly = true;
        } else if (strcmp(argv[i], "--animate") == 0) {
            animate = true;
        } else if (strcmp(argv[i], "--warmup") == 0) {
            if (!parseUIntArg(argc, argv, &i, &warmupFrames)) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strncmp(argv[i], "--warmup=", 9) == 0) {
            char *end = nullptr;
            unsigned long value = strtoul(argv[i] + 9, &end, 0);
            if (!end || *end) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            warmupFrames = (unsigned)value;
        } else if (strcmp(argv[i], "--sync-interval") == 0) {
            if (!parseUIntArg(argc, argv, &i, &syncInterval)) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strncmp(argv[i], "--sync-interval=", 16) == 0) {
            char *end = nullptr;
            unsigned long value = strtoul(argv[i] + 16, &end, 0);
            if (!end || *end) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            syncInterval = (unsigned)value;
        } else if (strcmp(argv[i], "--draw-batch-perf") == 0) {
            drawBatchPerf = true;
        } else if (strcmp(argv[i], "--draw-count") == 0) {
            if (!parseUIntArg(argc, argv, &i, &drawCount) || drawCount == 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strncmp(argv[i], "--draw-count=", 13) == 0) {
            char *end = nullptr;
            unsigned long value = strtoul(argv[i] + 13, &end, 0);
            if (!end || *end || value == 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            drawCount = (unsigned)value;
        } else if (strcmp(argv[i], "--upload-perf") == 0) {
            uploadPerf = true;
        } else if (strcmp(argv[i], "--upload-bytes") == 0) {
            if (!parseUIntArg(argc, argv, &i, &uploadBytes) ||
                uploadBytes < 4096) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--upload-iters") == 0) {
            if (!parseUIntArg(argc, argv, &i, &uploadIters) ||
                uploadIters == 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--retire-resource-probe") == 0) {
            retireResourceProbe = true;
        } else if (strcmp(argv[i], "--retire-count") == 0) {
            if (!parseUIntArg(argc, argv, &i, &retireCount) ||
                retireCount == 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strncmp(argv[i], "--retire-count=", 15) == 0) {
            char *end = nullptr;
            unsigned long value = strtoul(argv[i] + 15, &end, 0);
            if (!end || *end || value == 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            retireCount = (unsigned)value;
        } else if (strcmp(argv[i], "--help") == 0 ||
                   strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (selectAdapterByIndex && selectedAdapterName) {
        fprintf(stderr, "--adapter and --adapter-name are mutually exclusive\n");
        return EXIT_FAILURE;
    }
    if (bgra && api != Api::D3D11) {
        fprintf(stderr, "--bgra is only supported with --api d3d11\n");
        return EXIT_FAILURE;
    }
    if (srgb && api != Api::D3D11) {
        fprintf(stderr, "--srgb is only supported with --api d3d11\n");
        return EXIT_FAILURE;
    }
    if (uploadPerf && api != Api::D3D11) {
        fprintf(stderr, "--upload-perf is only supported with --api d3d11\n");
        return EXIT_FAILURE;
    }
    if (uboProbe && vertexIdTextureProbe) {
        fprintf(stderr, "--ubo-probe and --vertex-id-texture-probe are mutually exclusive\n");
        return EXIT_FAILURE;
    }
    if (cpuVertexProbe && (vertexIdTextureProbe || uboProbe)) {
        fprintf(stderr, "--cpu-vertex-probe cannot be combined with --vertex-id-texture-probe or --ubo-probe\n");
        return EXIT_FAILURE;
    }
    if (drawBatchPerf) {
        useIndexed = true;
        useTriangleList = true;
        useStripQuad = false;
        useLineBox = false;
    }
    if (vertexIdTextureProbe) {
        useIndexed = false;
        useTriangleList = true;
        useStripQuad = false;
        useLineBox = false;
    }
    if (uboProbe) {
        useIndexed = false;
        useTriangleList = true;
        useStripQuad = false;
        useLineBox = false;
    }
    if (cpuVertexProbe) {
        useIndexed = false;
        useTriangleList = true;
        useStripQuad = false;
        useLineBox = false;
    }
    if (retireResourceProbe && drawBatchPerf) {
        fprintf(stderr, "--retire-resource-probe and --draw-batch-perf are mutually exclusive\n");
        return EXIT_FAILURE;
    }
    if ((useStripQuad ? 1 : 0) + (useTriangleList ? 1 : 0) +
        (useLineBox ? 1 : 0) > 1) {
        fprintf(stderr, "--strip-quad, --triangle-list, and --line-box are mutually exclusive\n");
        return EXIT_FAILURE;
    }
    if ((useStripQuad || useLineBox) && useIndexed) {
        fprintf(stderr, "--strip-quad and --line-box are currently non-indexed only\n");
        return EXIT_FAILURE;
    }

    if (benchmark && !framesExplicitlySet) {
        frames = 1000;
    }

    if (api != Api::D3D11) {
        return runD3D9(api == Api::D3D9Ex,
                       windowWidth, windowHeight, fullscreen,
                       frames, framesExplicitlySet, benchmark,
                       presentOnly, animate, warmupFrames,
                       syncInterval, clearBlack, writeMaskRed);
    }

    ComPtr<IDXGIFactory1> pFactory;
    hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)pFactory.GetAddressOf());
    if (FAILED(hr)) {
        fprintf(stderr, "CreateDXGIFactory1 failed %lx\n\r", hr);
        return EXIT_FAILURE;
    }

    if (listAdapters) {
        printAdapters(pFactory.Get());
        return EXIT_SUCCESS;
    }

    ComPtr<IDXGIAdapter1> pSelectedAdapter;
    if (selectAdapterByIndex || selectedAdapterName) {
        if (!selectAdapter(pFactory.Get(),
                           selectAdapterByIndex,
                           selectedAdapterIndex,
                           selectedAdapterName,
                           pSelectedAdapter.GetAddressOf())) {
            return EXIT_FAILURE;
        }
    }

    ComPtr<IDXGIOutput> pSelectedOutput;
    DXGI_OUTPUT_DESC selectedOutputDesc;
    bool haveSelectedOutputDesc = false;
    ZeroMemory(&selectedOutputDesc, sizeof selectedOutputDesc);
    if (selectOutput) {
        if (!pSelectedAdapter) {
            fprintf(stderr, "--output requires --adapter or --adapter-name\n");
            return EXIT_FAILURE;
        }

        hr = pSelectedAdapter->EnumOutputs(selectedOutputIndex, &pSelectedOutput);
        if (FAILED(hr)) {
            fprintf(stderr, "EnumOutputs(%u) failed %lx\n\r",
                    selectedOutputIndex,
                    hr);
            return EXIT_FAILURE;
        }

        hr = pSelectedOutput->GetDesc(&selectedOutputDesc);
        if (FAILED(hr)) {
            fprintf(stderr, "GetDesc(output %u) failed %lx\n\r",
                    selectedOutputIndex,
                    hr);
            return EXIT_FAILURE;
        }
        haveSelectedOutputDesc = true;

        RECT r = selectedOutputDesc.DesktopCoordinates;
        printf("selected output[%u]: %S attached=%u desktop={%ld,%ld,%ld,%ld}\n",
               selectedOutputIndex,
               selectedOutputDesc.DeviceName,
               selectedOutputDesc.AttachedToDesktop ? 1 : 0,
               r.left,
               r.top,
               r.right,
               r.bottom);
    }

    HINSTANCE hInstance = GetModuleHandle(nullptr);

    WNDCLASSEX wc = {
        sizeof(WNDCLASSEX),
        CS_CLASSDC,
        DefWindowProc,
        0,
        0,
        hInstance,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        "tri",
        nullptr
    };
    RegisterClassEx(&wc);

    DWORD dwStyle = WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_OVERLAPPEDWINDOW | WS_VISIBLE;

    RECT rect = {0, 0, (LONG)windowWidth, (LONG)windowHeight};
    AdjustWindowRect(&rect, dwStyle, false);

    int windowX = CW_USEDEFAULT;
    int windowY = CW_USEDEFAULT;
    if (haveSelectedOutputDesc) {
        const RECT outputRect = selectedOutputDesc.DesktopCoordinates;
        windowX = outputRect.left + 40;
        windowY = outputRect.top + 40;
        if (windowX + (rect.right - rect.left) > outputRect.right) {
            windowX = outputRect.left;
        }
        if (windowY + (rect.bottom - rect.top) > outputRect.bottom) {
            windowY = outputRect.top;
        }
    }

    HWND hWnd = CreateWindow(wc.lpszClassName,
                             "Simple example using DirectX10",
                             dwStyle,
                             windowX, windowY,
                             rect.right - rect.left,
                             rect.bottom - rect.top,
                             nullptr,
                             nullptr,
                             hInstance,
                             nullptr);
    if (!hWnd) {
        return EXIT_FAILURE;
    }

    UINT Flags = 0;
    hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_NULL, 0, D3D11_CREATE_DEVICE_DEBUG, nullptr, 0, D3D11_SDK_VERSION, nullptr, nullptr, nullptr);
    if (SUCCEEDED(hr)) {
        Flags |= D3D11_CREATE_DEVICE_DEBUG;
    }
    if (bgra) {
        Flags |= D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    }

    static const D3D_FEATURE_LEVEL FeatureLevels[] = {
        D3D_FEATURE_LEVEL_10_0
    };

    DXGI_SWAP_CHAIN_DESC SwapChainDesc;
    ZeroMemory(&SwapChainDesc, sizeof SwapChainDesc);
    SwapChainDesc.BufferDesc.Width = windowWidth;
    SwapChainDesc.BufferDesc.Height = windowHeight;
    SwapChainDesc.BufferDesc.Format =
        bgra ? (srgb ? DXGI_FORMAT_B8G8R8A8_UNORM_SRGB :
                       DXGI_FORMAT_B8G8R8A8_UNORM) :
               (srgb ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB :
                       DXGI_FORMAT_R8G8B8A8_UNORM);
    SwapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
    SwapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
    SwapChainDesc.SampleDesc.Quality = 0;
    SwapChainDesc.SampleDesc.Count = 1;
    SwapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    SwapChainDesc.BufferCount = 2;
    SwapChainDesc.OutputWindow = hWnd;
    SwapChainDesc.Windowed = !fullscreen;
    SwapChainDesc.SwapEffect = swapEffect;
    SwapChainDesc.Flags = fullscreen ? DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH : 0;

    ComPtr<ID3D11Device> pDevice;
    ComPtr<ID3D11DeviceContext> pDeviceContext;
    ComPtr<IDXGISwapChain> pSwapChain;
    IDXGIAdapter *deviceAdapter = pSelectedAdapter.Get();
    D3D_DRIVER_TYPE driverType = deviceAdapter ? D3D_DRIVER_TYPE_UNKNOWN :
                                                 D3D_DRIVER_TYPE_HARDWARE;
    hr = D3D11CreateDeviceAndSwapChain(deviceAdapter,
                                       driverType,
                                       NULL,
                                       Flags,
                                       FeatureLevels,
                                       _countof(FeatureLevels),
                                       D3D11_SDK_VERSION,
                                       &SwapChainDesc,
                                       &pSwapChain,
                                       &pDevice,
                                       nullptr, /* pFeatureLevel */
                                       &pDeviceContext);
    if (FAILED(hr)) {
        fprintf(stderr, "D3D11CreateDeviceAndSwapChain failed %lx\n\r", hr);
        return EXIT_FAILURE;
    }

    if (fullscreen) {
        DXGI_MODE_DESC mode = SwapChainDesc.BufferDesc;
        hr = pSwapChain->ResizeTarget(&mode);
        printf("ResizeTarget fullscreen mode returned %lx\n", hr);

        hr = pSwapChain->SetFullscreenState(TRUE, pSelectedOutput.Get());
        printf("SetFullscreenState(TRUE) returned %lx\n", hr);

        BOOL fullscreenState = FALSE;
        ComPtr<IDXGIOutput> fullscreenOutput;
        hr = pSwapChain->GetFullscreenState(&fullscreenState,
                                            fullscreenOutput.GetAddressOf());
        printf("GetFullscreenState returned %lx state=%u output=%p\n",
               hr, fullscreenState ? 1 : 0, fullscreenOutput.Get());
    }

    ComPtr<IDXGIDevice> pDXGIDevice;
    hr = pDevice->QueryInterface(IID_IDXGIDevice, (void **)&pDXGIDevice);
    if (FAILED(hr)) {
        fprintf(stderr, "QueryInterface failed %lx\n\r", hr);
        return EXIT_FAILURE;
    }

    ComPtr<IDXGIAdapter> pAdapter;
    hr = pDXGIDevice->GetAdapter(&pAdapter);
    if (FAILED(hr)) {
        fprintf(stderr, "GetAdapter failed %lx\n\r", hr);
        return EXIT_FAILURE;
    }

    DXGI_ADAPTER_DESC Desc;
    hr = pAdapter->GetDesc(&Desc);
    if (FAILED(hr)) {
        fprintf(stderr, "GetDesc failed %lx\n\r", hr);
        return EXIT_FAILURE;
    }

    printf("using %S\n", Desc.Description);
    const bool usesVertexIdShader = vertexIdTextureProbe || uboProbe;

    ComPtr<ID3D11Texture2D> pBackBuffer;
    hr = pSwapChain->GetBuffer(0, IID_ID3D11Texture2D, (void **)&pBackBuffer);
    if (FAILED(hr)) {
        fprintf(stderr, "GetBuffer failed %lx\n\r", hr);
        return EXIT_FAILURE;
    }

    D3D11_RENDER_TARGET_VIEW_DESC RenderTargetViewDesc;
    ZeroMemory(&RenderTargetViewDesc, sizeof RenderTargetViewDesc);
    RenderTargetViewDesc.Format = SwapChainDesc.BufferDesc.Format;
    RenderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    RenderTargetViewDesc.Texture2D.MipSlice = 0;

    ComPtr<ID3D11RenderTargetView> pRenderTargetView;
    hr = pDevice->CreateRenderTargetView(pBackBuffer.Get(), &RenderTargetViewDesc, &pRenderTargetView);
    if (FAILED(hr)) {
        fprintf(stderr, "CreateRenderTargetView failed %lx\n\r", hr);
        return EXIT_FAILURE;
    }

    pDeviceContext->OMSetRenderTargets(1, pRenderTargetView.GetAddressOf(), nullptr);


    const float clearColor[4] = {
        clearBlack ? 0.0f : 0.3f,
        clearBlack ? 0.0f : 0.1f,
        clearBlack ? 0.0f : 0.3f,
        1.0f
    };
    if (clearBlack) {
        printf("clear black enabled\n");
    }
    pDeviceContext->ClearRenderTargetView(pRenderTargetView.Get(), clearColor);

    ComPtr<ID3D11VertexShader> pVertexShader;
    const BYTE *vsBytecode =
        vertexIdTextureProbe ? g_VertexIdTextureVS :
        uboProbe ? g_VertexIdCBufferProbeVS : g_VS;
    const SIZE_T vsBytecodeSize =
        vertexIdTextureProbe ? sizeof g_VertexIdTextureVS :
        uboProbe ? sizeof g_VertexIdCBufferProbeVS : sizeof g_VS;
    hr = pDevice->CreateVertexShader(vsBytecode, vsBytecodeSize, nullptr,
                                     &pVertexShader);
    if (FAILED(hr)) {
        fprintf(stderr, "CreateVertexShader failed %lx\n\r", hr);
        return EXIT_FAILURE;
    }

    struct Vertex {
        float position[4];
        float color[4];
    };

    static const D3D11_INPUT_ELEMENT_DESC InputElementDescs[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, offsetof(Vertex, position), D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, offsetof(Vertex, color),    D3D11_INPUT_PER_VERTEX_DATA, 0 }
    };

    ComPtr<ID3D11InputLayout> pVertexLayout;
    if (!usesVertexIdShader) {
        hr = pDevice->CreateInputLayout(InputElementDescs,
                                        _countof(InputElementDescs),
                                        vsBytecode, vsBytecodeSize,
                                        &pVertexLayout);
        if (FAILED(hr)) {
            fprintf(stderr, "CreateInputLayout failed %lx\n\r", hr);
            return EXIT_FAILURE;
        }

        pDeviceContext->IASetInputLayout(pVertexLayout.Get());
    }

    ComPtr<ID3D11PixelShader> pPixelShader;
    const BYTE *psBytecode = textureProbe ? g_TextureProbePS : g_PS;
    const SIZE_T psBytecodeSize =
        textureProbe ? sizeof g_TextureProbePS : sizeof g_PS;
    hr = pDevice->CreatePixelShader(psBytecode, psBytecodeSize, nullptr,
                                    &pPixelShader);
    if (FAILED(hr)) {
        fprintf(stderr, "CreatePixelShader failed %lx\n\r", hr);
        return EXIT_FAILURE;
    }

    pDeviceContext->VSSetShader(pVertexShader.Get(), nullptr, 0);
    pDeviceContext->PSSetShader(pPixelShader.Get(), nullptr, 0);

    ComPtr<ID3D11Buffer> pUboProbeBuffer;
    if (uboProbe) {
        struct UboProbeConstants {
            float scaleOffset[4];
            float colorMul[4];
        };
        static const UboProbeConstants constants = {
            { 1.0f, 1.0f, 0.0f, 0.0f },
            { 1.0f, 1.0f, 1.0f, 1.0f },
        };

        D3D11_BUFFER_DESC UboDesc;
        ZeroMemory(&UboDesc, sizeof UboDesc);
        UboDesc.Usage = D3D11_USAGE_DEFAULT;
        UboDesc.ByteWidth = sizeof constants;
        UboDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        UboDesc.CPUAccessFlags = 0;
        UboDesc.MiscFlags = 0;

        D3D11_SUBRESOURCE_DATA UboData;
        UboData.pSysMem = &constants;
        UboData.SysMemPitch = 0;
        UboData.SysMemSlicePitch = 0;

        hr = pDevice->CreateBuffer(&UboDesc, &UboData, &pUboProbeBuffer);
        if (FAILED(hr)) {
            fprintf(stderr, "CreateBuffer(ubo-probe) failed %lx\n\r", hr);
            return EXIT_FAILURE;
        }

        ID3D11Buffer *constantBuffer = pUboProbeBuffer.Get();
        pDeviceContext->VSSetConstantBuffers(0, 1, &constantBuffer);
        printf("ubo probe bound vs_cb=0 bytes=%u\n", UboDesc.ByteWidth);
    }

    ComPtr<ID3D11Texture2D> pTextureProbe;
    ComPtr<ID3D11ShaderResourceView> pTextureProbeSRV;
    ComPtr<ID3D11SamplerState> pTextureProbeSampler;
    if (textureProbe) {
        static const unsigned char textureProbePixels[] = {
            255,   0,   0, 255,     0, 255,   0, 255,
              0,   0, 255, 255,   255, 255, 255, 255,
        };

        D3D11_TEXTURE2D_DESC TextureDesc;
        ZeroMemory(&TextureDesc, sizeof TextureDesc);
        TextureDesc.Width = 2;
        TextureDesc.Height = 2;
        TextureDesc.MipLevels = 1;
        TextureDesc.ArraySize = 1;
        TextureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        TextureDesc.SampleDesc.Count = 1;
        TextureDesc.Usage = D3D11_USAGE_IMMUTABLE;
        TextureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

        D3D11_SUBRESOURCE_DATA TextureData;
        ZeroMemory(&TextureData, sizeof TextureData);
        TextureData.pSysMem = textureProbePixels;
        TextureData.SysMemPitch = 2 * 4;

        hr = pDevice->CreateTexture2D(&TextureDesc, &TextureData,
                                      &pTextureProbe);
        if (FAILED(hr)) {
            fprintf(stderr, "CreateTexture2D(texture-probe) failed %lx\n\r",
                    hr);
            return EXIT_FAILURE;
        }

        D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc;
        ZeroMemory(&SRVDesc, sizeof SRVDesc);
        SRVDesc.Format = TextureDesc.Format;
        SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        SRVDesc.Texture2D.MostDetailedMip = 0;
        SRVDesc.Texture2D.MipLevels = 1;
        hr = pDevice->CreateShaderResourceView(pTextureProbe.Get(), &SRVDesc,
                                               &pTextureProbeSRV);
        if (FAILED(hr)) {
            fprintf(stderr,
                    "CreateShaderResourceView(texture-probe) failed %lx\n\r",
                    hr);
            return EXIT_FAILURE;
        }

        D3D11_SAMPLER_DESC SamplerDesc;
        ZeroMemory(&SamplerDesc, sizeof SamplerDesc);
        SamplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
        SamplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        SamplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        SamplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        SamplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
        SamplerDesc.MinLOD = 0.0f;
        SamplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
        hr = pDevice->CreateSamplerState(&SamplerDesc, &pTextureProbeSampler);
        if (FAILED(hr)) {
            fprintf(stderr, "CreateSamplerState(texture-probe) failed %lx\n\r",
                    hr);
            return EXIT_FAILURE;
        }

        ID3D11ShaderResourceView *srv = pTextureProbeSRV.Get();
        ID3D11SamplerState *sampler = pTextureProbeSampler.Get();
        pDeviceContext->PSSetShaderResources(0, 1, &srv);
        pDeviceContext->PSSetSamplers(0, 1, &sampler);
        printf("texture probe bound 2x2 rgba8 ps_srv=0 ps_sampler=0\n");
    }

    static const Vertex triangleVertices[] = {
        { { -0.9f, -0.9f, 0.5f, 1.0f}, { 0.8f, 0.0f, 0.0f, 0.1f } },
        { {  0.9f, -0.9f, 0.5f, 1.0f}, { 0.0f, 0.9f, 0.0f, 0.1f } },
        { {  0.0f,  0.9f, 0.5f, 1.0f}, { 0.0f, 0.0f, 0.7f, 0.1f } },
    };
    static const Vertex stripQuadVertices[] = {
        { { -0.9f, -0.9f, 0.5f, 1.0f}, { 0.8f, 0.0f, 0.0f, 0.1f } },
        { { -0.9f,  0.9f, 0.5f, 1.0f}, { 0.0f, 0.9f, 0.0f, 0.1f } },
        { {  0.9f, -0.9f, 0.5f, 1.0f}, { 0.0f, 0.0f, 0.7f, 0.1f } },
        { {  0.9f,  0.9f, 0.5f, 1.0f}, { 0.9f, 0.9f, 0.0f, 0.1f } },
    };
    static const Vertex lineBoxVertices[] = {
        { { -0.9f, -0.9f, 0.5f, 1.0f}, { 1.0f, 0.0f, 0.0f, 1.0f } },
        { {  0.9f, -0.9f, 0.5f, 1.0f}, { 0.0f, 1.0f, 0.0f, 1.0f } },
        { {  0.9f,  0.9f, 0.5f, 1.0f}, { 0.0f, 0.0f, 1.0f, 1.0f } },
        { { -0.9f,  0.9f, 0.5f, 1.0f}, { 1.0f, 1.0f, 0.0f, 1.0f } },
        { { -0.9f, -0.9f, 0.5f, 1.0f}, { 1.0f, 0.0f, 0.0f, 1.0f } },
    };
    const Vertex *vertices = useLineBox ? lineBoxVertices :
                             useStripQuad ? stripQuadVertices :
                                           triangleVertices;
    const UINT vertexCount =
        useLineBox ? (UINT)_countof(lineBoxVertices) :
        useStripQuad ? (UINT)_countof(stripQuadVertices) :
                       (UINT)_countof(triangleVertices);

    D3D11_BUFFER_DESC BufferDesc;
    ZeroMemory(&BufferDesc, sizeof BufferDesc);
    BufferDesc.Usage = drawBatchPerf ? D3D11_USAGE_DEFAULT :
                                     D3D11_USAGE_DYNAMIC;
    BufferDesc.ByteWidth = sizeof(Vertex) * vertexCount;
    BufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    BufferDesc.CPUAccessFlags = drawBatchPerf ? 0 :
                                                D3D11_CPU_ACCESS_WRITE;
    BufferDesc.MiscFlags = 0;

    D3D11_SUBRESOURCE_DATA BufferData;
    BufferData.pSysMem = vertices;
    BufferData.SysMemPitch = 0;
    BufferData.SysMemSlicePitch = 0;

    ComPtr<ID3D11Buffer> pVertexBuffer;
    UINT Stride = sizeof(Vertex);
    UINT Offset = 0;
    if (!usesVertexIdShader) {
        hr = pDevice->CreateBuffer(&BufferDesc, &BufferData, &pVertexBuffer);
        if (FAILED(hr)) {
            fprintf(stderr, "CreateBuffer failed %lx\n\r", hr);
            return EXIT_FAILURE;
        }

        pDeviceContext->IASetVertexBuffers(0, 1, pVertexBuffer.GetAddressOf(), &Stride, &Offset);
    } else if (vertexIdTextureProbe) {
        printf("vertex-id texture probe enabled\n");
    } else {
        printf("ubo vertex-id probe enabled\n");
    }

    static const unsigned short indices[] = {
        0, 1, 2,
    };

    ComPtr<ID3D11Buffer> pIndexBuffer;
    if (useIndexed) {
        D3D11_BUFFER_DESC IndexBufferDesc;
        ZeroMemory(&IndexBufferDesc, sizeof IndexBufferDesc);
        IndexBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
        IndexBufferDesc.ByteWidth = sizeof indices;
        IndexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
        IndexBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        IndexBufferDesc.MiscFlags = 0;

        D3D11_SUBRESOURCE_DATA IndexBufferData;
        IndexBufferData.pSysMem = indices;
        IndexBufferData.SysMemPitch = 0;
        IndexBufferData.SysMemSlicePitch = 0;

        hr = pDevice->CreateBuffer(&IndexBufferDesc, &IndexBufferData, &pIndexBuffer);
        if (FAILED(hr)) {
            fprintf(stderr, "CreateBuffer(index) failed %lx\n\r", hr);
            return EXIT_FAILURE;
        }

        printf("indexed draw enabled index_count=%u index_format=R16_UINT\n",
               (unsigned)_countof(indices));
        pDeviceContext->IASetIndexBuffer(pIndexBuffer.Get(), DXGI_FORMAT_R16_UINT, 0);
    }

    D3D11_VIEWPORT ViewPort;
    unsigned maxViewportInset = 0;
    if (windowWidth > 1 && windowHeight > 1) {
        maxViewportInset = (windowWidth < windowHeight ? windowWidth : windowHeight) / 2 - 1;
    }
    if (viewportInset > maxViewportInset) {
        viewportInset = maxViewportInset;
    }
    ViewPort.TopLeftX = (FLOAT)viewportInset;
    ViewPort.TopLeftY = (FLOAT)viewportInset;
    ViewPort.Width = (FLOAT)(windowWidth - 2 * viewportInset);
    ViewPort.Height = (FLOAT)(windowHeight - 2 * viewportInset);
    ViewPort.MinDepth = 0.0f;
    ViewPort.MaxDepth = 1.0f;
    printf("viewport x=%g y=%g w=%g h=%g\n",
           ViewPort.TopLeftX, ViewPort.TopLeftY,
           ViewPort.Width, ViewPort.Height);
    pDeviceContext->RSSetViewports(1, &ViewPort);

    if (useScissor) {
        unsigned maxScissorInset = 0;
        if (windowWidth > 1 && windowHeight > 1) {
            maxScissorInset = (windowWidth < windowHeight ? windowWidth : windowHeight) / 2 - 1;
        }
        if (scissorInset > maxScissorInset) {
            scissorInset = maxScissorInset;
        }

        D3D11_RECT ScissorRect;
        ScissorRect.left = (LONG)scissorInset;
        ScissorRect.top = (LONG)scissorInset;
        ScissorRect.right = (LONG)(windowWidth - scissorInset);
        ScissorRect.bottom = (LONG)(windowHeight - scissorInset);
        printf("scissor left=%ld top=%ld right=%ld bottom=%ld\n",
               ScissorRect.left, ScissorRect.top,
               ScissorRect.right, ScissorRect.bottom);
        pDeviceContext->RSSetScissorRects(1, &ScissorRect);
    }

    D3D11_RASTERIZER_DESC RasterizerDesc;
    ZeroMemory(&RasterizerDesc, sizeof RasterizerDesc);
    RasterizerDesc.CullMode = cullMode;
    RasterizerDesc.FillMode = D3D11_FILL_SOLID;
    RasterizerDesc.FrontCounterClockwise = frontCounterClockwise;
    RasterizerDesc.DepthClipEnable = true;
    RasterizerDesc.ScissorEnable = useScissor;
    printf("rasterizer cull=%u front_ccw=%u scissor=%u\n",
           (unsigned)cullMode,
           frontCounterClockwise ? 1 : 0,
           useScissor ? 1 : 0);
    ComPtr<ID3D11RasterizerState> pRasterizerState;
    hr = pDevice->CreateRasterizerState(&RasterizerDesc, &pRasterizerState);
    if (FAILED(hr)) {
        fprintf(stderr, "CreateRasterizerState failed %lx\n\r", hr);
        return EXIT_FAILURE;
    }
    pDeviceContext->RSSetState(pRasterizerState.Get());

    ComPtr<ID3D11BlendState> pBlendState;
    if (blendMode != BlendMode::None || writeMaskRed || sampleMaskZero) {
        D3D11_BLEND_DESC BlendDesc;
        ZeroMemory(&BlendDesc, sizeof BlendDesc);
        BlendDesc.RenderTarget[0].BlendEnable =
            blendMode == BlendMode::None ? FALSE : TRUE;
        BlendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
        BlendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_ZERO;
        BlendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
        BlendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
        BlendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
        BlendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
        BlendDesc.RenderTarget[0].RenderTargetWriteMask =
            writeMaskRed ? D3D11_COLOR_WRITE_ENABLE_RED :
                           D3D11_COLOR_WRITE_ENABLE_ALL;

        if (blendMode == BlendMode::DstOnly) {
            BlendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_ZERO;
            BlendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_ONE;
            BlendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ZERO;
            BlendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
        } else {
            BlendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
            BlendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
            BlendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
            BlendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
        }

        hr = pDevice->CreateBlendState(&BlendDesc, &pBlendState);
        if (FAILED(hr)) {
            fprintf(stderr, "CreateBlendState failed %lx\n\r", hr);
            return EXIT_FAILURE;
        }

        const float blendFactor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        const UINT sampleMask = sampleMaskZero ? 0 : 0xffffffff;
        if (blendMode == BlendMode::DstOnly) {
            printf("dst-only blend enabled src=ZERO dst=ONE\n");
        } else if (blendMode == BlendMode::Alpha) {
            printf("alpha blend enabled src=SRC_ALPHA dst=INV_SRC_ALPHA\n");
        }
        if (sampleMaskZero) {
            printf("sample mask zero enabled\n");
        }
        if (writeMaskRed) {
            printf("color write mask red enabled\n");
        }
        pDeviceContext->OMSetBlendState(pBlendState.Get(), blendFactor,
                                        sampleMask);
    }

    ComPtr<ID3D11DepthStencilState> pDepthStencilState;
    if (depthStateProbe) {
        D3D11_DEPTH_STENCIL_DESC DepthStencilDesc;
        ZeroMemory(&DepthStencilDesc, sizeof DepthStencilDesc);
        DepthStencilDesc.DepthEnable = TRUE;
        DepthStencilDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        DepthStencilDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
        DepthStencilDesc.StencilEnable = TRUE;
        DepthStencilDesc.StencilReadMask = stencilMaskProbe ? 0x0f : 0xff;
        DepthStencilDesc.StencilWriteMask = stencilMaskProbe ? 0xf0 : 0xff;
        DepthStencilDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
        DepthStencilDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
        DepthStencilDesc.FrontFace.StencilPassOp =
            stencilMaskProbe ? D3D11_STENCIL_OP_REPLACE : D3D11_STENCIL_OP_KEEP;
        DepthStencilDesc.FrontFace.StencilFunc =
            stencilMaskProbe ? D3D11_COMPARISON_EQUAL : D3D11_COMPARISON_ALWAYS;
        DepthStencilDesc.BackFace = DepthStencilDesc.FrontFace;

        hr = pDevice->CreateDepthStencilState(&DepthStencilDesc,
                                              &pDepthStencilState);
        if (FAILED(hr)) {
            fprintf(stderr,
                    "CreateDepthStencilState(depth-state-probe) failed %lx\n\r",
                    hr);
            return EXIT_FAILURE;
        }

        pDeviceContext->OMSetDepthStencilState(pDepthStencilState.Get(), 7);
        printf("depth/stencil state probe enabled depth=LEQUAL stencil_ref=7 read_mask=0x%02x write_mask=0x%02x func=%s pass=%s\n",
               (unsigned)DepthStencilDesc.StencilReadMask,
               (unsigned)DepthStencilDesc.StencilWriteMask,
               stencilMaskProbe ? "EQUAL" : "ALWAYS",
               stencilMaskProbe ? "REPLACE" : "KEEP");
    }

    D3D11_PRIMITIVE_TOPOLOGY primitiveTopology =
        useLineBox ? D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP :
        useTriangleList ? D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST :
                          D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
    printf("primitive topology=%s\n",
           useLineBox ? "line-box" :
           useTriangleList ? "triangle-list" :
           useStripQuad ? "triangle-strip-quad" : "triangle-strip");
    pDeviceContext->IASetPrimitiveTopology(primitiveTopology);

    if (benchmark && !framesExplicitlySet) {
        frames = 1000;
    }

    const char *swapEffectName =
        swapEffect == DXGI_SWAP_EFFECT_DISCARD ? "discard" :
        swapEffect == DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL ? "flip-sequential" :
        swapEffect == DXGI_SWAP_EFFECT_FLIP_DISCARD ? "flip-discard" :
                                                         "other";
    const char *swapFormatName =
        bgra ? (srgb ? "B8G8R8A8_UNORM_SRGB" : "B8G8R8A8_UNORM") :
               (srgb ? "R8G8B8A8_UNORM_SRGB" : "R8G8B8A8_UNORM");
    printf("swapchain format=%s\n", swapFormatName);

    /* Benchmark mode times only the Present() call (the KMD present/blt path),
     * excluding draw and message-pump overhead.  --present-only additionally
     * skips the per-frame clear+draw so the measurement is the present path in
     * isolation.  The first warmupFrames presents are discarded. */
    std::vector<double> presentMs;
    LARGE_INTEGER wallStart = {}, wallEnd = {};
    bool wallTiming = false;
    double qpcToMs = 0.0;
    if (benchmark) {
        LARGE_INTEGER qpf;
        QueryPerformanceFrequency(&qpf);
        qpcToMs = 1000.0 / (double)qpf.QuadPart;
        presentMs.reserve(frames);
        printf("benchmark: frames=%u warmup=%u present-only=%u sync-interval=%u %ux%u swap=%s format=%s\n",
               frames, warmupFrames, presentOnly ? 1 : 0, syncInterval,
               windowWidth, windowHeight, swapEffectName, swapFormatName);
    }

    LARGE_INTEGER animQpf = {};
    LARGE_INTEGER animStart = {};
    if (animate) {
        QueryPerformanceFrequency(&animQpf);
        QueryPerformanceCounter(&animStart);
    }

    /* --animate without an explicit --frames (and outside --benchmark) runs
     * until the window is closed so the motion can actually be watched. */
    const bool animateForever = animate && !benchmark && !framesExplicitlySet;
    const unsigned totalFrames = frames + (benchmark ? warmupFrames : 0);
    if (uploadPerf) {
        LARGE_INTEGER qpf = {}, t0 = {}, t1 = {}, t2 = {}, t3 = {};
        QueryPerformanceFrequency(&qpf);

        D3D11_BUFFER_DESC UploadDesc;
        ZeroMemory(&UploadDesc, sizeof UploadDesc);
        UploadDesc.Usage = D3D11_USAGE_DYNAMIC;
        UploadDesc.ByteWidth = uploadBytes;
        UploadDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        UploadDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        UploadDesc.MiscFlags = 0;

        ComPtr<ID3D11Buffer> pUploadBuffer;
        hr = pDevice->CreateBuffer(&UploadDesc, nullptr, &pUploadBuffer);
        if (FAILED(hr)) {
            fprintf(stderr, "CreateBuffer(upload, %u bytes) failed %lx\n",
                    uploadBytes, hr);
            return EXIT_FAILURE;
        }

        void *source = malloc(uploadBytes);
        if (!source) {
            fprintf(stderr, "out of memory staging %u bytes\n", uploadBytes);
            return EXIT_FAILURE;
        }
        memset(source, 0x5a, uploadBytes);

        printf("upload perf: iterations=%u warmup=%u bytes=%u\n",
               uploadIters, warmupFrames, uploadBytes);

        std::vector<double> mapMs, writeMs, unmapMs;
        mapMs.reserve(uploadIters);
        writeMs.reserve(uploadIters);
        unmapMs.reserve(uploadIters);

        const double scale = 1000.0 / (double)qpf.QuadPart;
        const uint64_t totalIters =
            (uint64_t)uploadIters + (uint64_t)warmupFrames;
        for (uint64_t i = 0; i < totalIters; ++i) {
            D3D11_MAPPED_SUBRESOURCE Mapped;
            ZeroMemory(&Mapped, sizeof Mapped);

            QueryPerformanceCounter(&t0);
            hr = pDeviceContext->Map(pUploadBuffer.Get(), 0,
                                     D3D11_MAP_WRITE_DISCARD, 0, &Mapped);
            QueryPerformanceCounter(&t1);
            if (FAILED(hr)) {
                fprintf(stderr, "Map(upload) failed %lx\n", hr);
                free(source);
                return EXIT_FAILURE;
            }

            memcpy(Mapped.pData, source, uploadBytes);
            /* Stores to write-combining memory sit in the CPU's write-combine
             * buffers until something flushes them, so without this fence the
             * timer stops before the bus has seen the data and WC memory looks
             * as fast as cached memory. */
            _mm_sfence();
            QueryPerformanceCounter(&t2);

            pDeviceContext->Unmap(pUploadBuffer.Get(), 0);
            QueryPerformanceCounter(&t3);

            if (i < warmupFrames) {
                continue;
            }
            mapMs.push_back((double)(t1.QuadPart - t0.QuadPart) * scale);
            writeMs.push_back((double)(t2.QuadPart - t1.QuadPart) * scale);
            unmapMs.push_back((double)(t3.QuadPart - t2.QuadPart) * scale);
        }

        free(source);
        reportUploadBenchmark(mapMs, writeMs, unmapMs, uploadBytes);
        return EXIT_SUCCESS;
    }

    if (retireResourceProbe) {
        LARGE_INTEGER qpf = {}, t0 = {}, t1 = {};
        QueryPerformanceFrequency(&qpf);

        printf("retire resource probe: draws=%u indexed=%u primitive=%s sync-interval=%u %ux%u swap=%s\n",
               retireCount, useIndexed ? 1 : 0,
               useLineBox ? "line-box" :
               useTriangleList ? "triangle-list" :
               useStripQuad ? "triangle-strip-quad" : "triangle-strip",
               syncInterval, windowWidth, windowHeight, swapEffectName);

        pDeviceContext->OMSetRenderTargets(1,
                                           pRenderTargetView.GetAddressOf(),
                                           nullptr);
        pDeviceContext->ClearRenderTargetView(pRenderTargetView.Get(),
                                              clearColor);

        D3D11_BUFFER_DESC RetireBufferDesc;
        ZeroMemory(&RetireBufferDesc, sizeof RetireBufferDesc);
        RetireBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        RetireBufferDesc.ByteWidth = sizeof(Vertex) * vertexCount;
        RetireBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        RetireBufferDesc.CPUAccessFlags = 0;
        RetireBufferDesc.MiscFlags = 0;

        ID3D11Buffer *nullBuffer = nullptr;
        UINT nullStride = 0;
        UINT nullOffset = 0;

        QueryPerformanceCounter(&t0);
        for (unsigned i = 0; i < retireCount; ++i) {
            std::vector<Vertex> tempVertices(vertexCount);
            for (UINT v = 0; v < vertexCount; ++v) {
                tempVertices[v] = vertices[v];
                const float phase = (float)((i + v) % 97) / 97.0f;
                tempVertices[v].color[0] =
                    0.25f + 0.75f * tempVertices[v].color[0];
                tempVertices[v].color[1] =
                    0.25f + 0.75f * tempVertices[v].color[1];
                tempVertices[v].color[2] =
                    0.25f + 0.75f * tempVertices[v].color[2];
                tempVertices[v].position[0] += (phase - 0.5f) * 0.001f;
            }

            D3D11_SUBRESOURCE_DATA RetireBufferData;
            RetireBufferData.pSysMem = tempVertices.data();
            RetireBufferData.SysMemPitch = 0;
            RetireBufferData.SysMemSlicePitch = 0;

            ComPtr<ID3D11Buffer> pTempVertexBuffer;
            hr = pDevice->CreateBuffer(&RetireBufferDesc,
                                       &RetireBufferData,
                                       &pTempVertexBuffer);
            if (FAILED(hr)) {
                fprintf(stderr,
                        "CreateBuffer(retire-resource-probe %u) failed %lx\n\r",
                        i, hr);
                return EXIT_FAILURE;
            }

            ID3D11Buffer *tempBuffer = pTempVertexBuffer.Get();
            pDeviceContext->IASetVertexBuffers(0, 1, &tempBuffer,
                                               &Stride, &Offset);
            if (useIndexed) {
                pDeviceContext->DrawIndexed(_countof(indices), 0, 0);
            } else {
                pDeviceContext->Draw(vertexCount, 0);
            }
            pDeviceContext->IASetVertexBuffers(0, 1, &nullBuffer,
                                               &nullStride, &nullOffset);
            pTempVertexBuffer.Reset();
        }
        pDeviceContext->Flush();
        QueryPerformanceCounter(&t1);

        hr = pSwapChain->Present(syncInterval, 0);

        const double elapsedMs =
            (double)(t1.QuadPart - t0.QuadPart) * 1000.0 /
            (double)qpf.QuadPart;
        const double drawsPerSec =
            elapsedMs > 0.0 ? (double)retireCount * 1000.0 / elapsedMs : 0.0;

        printf("\n=== retire resource probe ===\n");
        printf("draws=%u indexed=%u\n", retireCount, useIndexed ? 1 : 0);
        printf("create+draw+release+flush ms=%.4f draws/sec=%.1f\n",
               elapsedMs, drawsPerSec);
        printf("present hr=%lx\n", hr);

        if (FAILED(hr)) {
            fprintf(stderr,"Present failed %lx\n\r", hr);
            return EXIT_FAILURE;
        }
    } else if (drawBatchPerf) {
        LARGE_INTEGER qpf = {}, t0 = {}, t1 = {};
        QueryPerformanceFrequency(&qpf);

        const bool perfIndexed = useIndexed && !vertexIdTextureProbe;
        printf("draw batch perf: draws=%u warmup=%u indexed=%u primitive=triangle-list sync-interval=%u %ux%u swap=%s\n",
               drawCount, warmupFrames, perfIndexed ? 1 : 0, syncInterval,
               windowWidth, windowHeight, swapEffectName);

        pDeviceContext->OMSetRenderTargets(1,
                                           pRenderTargetView.GetAddressOf(),
                                           nullptr);
        pDeviceContext->ClearRenderTargetView(pRenderTargetView.Get(),
                                              clearColor);

        for (unsigned i = 0; i < warmupFrames; ++i) {
            if (perfIndexed) {
                pDeviceContext->DrawIndexed(_countof(indices), 0, 0);
            } else {
                pDeviceContext->Draw(vertexCount, 0);
            }
        }
        pDeviceContext->Flush();

        QueryPerformanceCounter(&t0);
        for (unsigned i = 0; i < drawCount; ++i) {
            if (perfIndexed) {
                pDeviceContext->DrawIndexed(_countof(indices), 0, 0);
            } else {
                pDeviceContext->Draw(vertexCount, 0);
            }
        }
        pDeviceContext->Flush();
        QueryPerformanceCounter(&t1);

        const double elapsedMs =
            (double)(t1.QuadPart - t0.QuadPart) * 1000.0 /
            (double)qpf.QuadPart;
        const double drawsPerSec =
            elapsedMs > 0.0 ? (double)drawCount * 1000.0 / elapsedMs : 0.0;

        hr = pSwapChain->Present(syncInterval, 0);

        printf("\n=== draw batch perf ===\n");
        printf("draws=%u warmup=%u indexed=%u primitive=triangle-list\n",
               drawCount, warmupFrames, perfIndexed ? 1 : 0);
        printf("draw+flush ms=%.4f draws/sec=%.1f\n",
               elapsedMs, drawsPerSec);
        printf("present hr=%lx\n", hr);

        if (FAILED(hr)) {
            fprintf(stderr,"Present failed %lx\n\r", hr);
            return EXIT_FAILURE;
        }
    } else for (unsigned frame = 0; animateForever || frame < totalFrames; ++frame) {
        if (benchmark && !wallTiming && frame == warmupFrames) {
            QueryPerformanceCounter(&wallStart);
            wallTiming = true;
        }
        if (benchmark || animate) {
            MSG msg;
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            if (animateForever && !IsWindow(hWnd)) {
                break;
            }
        }

        // --animate forces a draw (so the GPU writes fresh content every frame)
        // even under --present-only.  Rotating the triangle and cycling the
        // clear color makes a stale/incoherent present visibly freeze or stutter
        // -- the check for whether a cached source-blob mapping is coherent.
        const bool drawThisFrame = !presentOnly || animate;
        if (drawThisFrame) {
            const float *frameColor = clearColor;
            float animColor[4];
            if (animate) {
                /* Drive the animation from wall-clock time, not the frame
                 * index.  A fast flip path sustains thousands of presents/sec;
                 * a frame-indexed angle then spins tens of revs/sec and the
                 * 60 Hz display samples it as "random" triangles.  Time-based
                 * motion advances at a fixed rate (~0.25 rev/sec here) and
                 * reads as smooth regardless of the present rate. */
                LARGE_INTEGER nowCounter;
                QueryPerformanceCounter(&nowCounter);
                const double tSec = (double)(nowCounter.QuadPart -
                                             animStart.QuadPart) /
                                    (double)animQpf.QuadPart;
                const float angle = (float)(tSec * 1.5707963);  /* 4 s / rev */
                const float c = cosf(angle);
                const float s = sinf(angle);

                std::vector<Vertex> rot(vertexCount);
                for (UINT v = 0; v < vertexCount; ++v) {
                    rot[v] = vertices[v];
                    const float x = vertices[v].position[0];
                    const float y = vertices[v].position[1];
                    rot[v].position[0] = x * c - y * s;
                    rot[v].position[1] = x * s + y * c;
                }

                D3D11_MAPPED_SUBRESOURCE mapped;
                if (SUCCEEDED(pDeviceContext->Map(pVertexBuffer.Get(), 0,
                                                  D3D11_MAP_WRITE_DISCARD, 0,
                                                  &mapped))) {
                    memcpy(mapped.pData, rot.data(),
                           sizeof(Vertex) * vertexCount);
                    pDeviceContext->Unmap(pVertexBuffer.Get(), 0);
                }

                const float colorPhase = (float)(tSec * 0.9);
                animColor[0] = 0.5f + 0.5f * sinf(colorPhase);
                animColor[1] = 0.5f + 0.5f * sinf(colorPhase + 2.0944f);
                animColor[2] = 0.5f + 0.5f * sinf(colorPhase + 4.1888f);
                animColor[3] = 1.0f;
                frameColor = animColor;
            }

            pDeviceContext->OMSetRenderTargets(1,
                                               pRenderTargetView.GetAddressOf(),
                                               nullptr);
            pDeviceContext->ClearRenderTargetView(pRenderTargetView.Get(),
                                                  frameColor);

            if (useIndexed) {
                pDeviceContext->DrawIndexed(_countof(indices), 0, 0);
            } else {
                pDeviceContext->Draw(vertexCount, 0);
            }
        }

        LARGE_INTEGER t0 = {}, t1 = {};
        if (benchmark) {
            QueryPerformanceCounter(&t0);
        }

        hr = pSwapChain->Present(syncInterval, 0);

        if (benchmark) {
            QueryPerformanceCounter(&t1);
            if (frame >= warmupFrames) {
                presentMs.push_back((double)(t1.QuadPart - t0.QuadPart) *
                                    qpcToMs);
            }
        } else if (!animate) {
            printf("present frame=%u hr=%lx\n", frame, hr);
        }

        if (FAILED(hr)) {
            fprintf(stderr,"Present failed %lx\n\r", hr);
            return EXIT_FAILURE;
        }
    }

    if (wallTiming) {
        QueryPerformanceCounter(&wallEnd);
    }

    if (benchmark && !presentMs.empty()) {
        reportPresentBenchmark(presentMs, presentOnly, syncInterval,
                               windowWidth, windowHeight, swapEffectName,
                               swapFormatName);
        if (wallTiming) {
            reportFrameRate((unsigned)presentMs.size(),
                            (double)(wallEnd.QuadPart - wallStart.QuadPart) *
                            qpcToMs);
        }
    } else if (!drawBatchPerf && !retireResourceProbe) {
        Sleep(5000);
    }

    ID3D11Buffer *pNullBuffer = nullptr;
    UINT NullStride = 0;
    UINT NullOffset = 0;
    pDeviceContext->IASetVertexBuffers(0, 1, &pNullBuffer, &NullStride, &NullOffset);
    pDeviceContext->IASetIndexBuffer(nullptr, DXGI_FORMAT_R16_UINT, 0);

    pDeviceContext->OMSetRenderTargets(0, nullptr, nullptr);
    pDeviceContext->OMSetBlendState(nullptr, nullptr, 0xffffffff);
    pDeviceContext->OMSetDepthStencilState(nullptr, 0);

    pDeviceContext->IASetInputLayout(nullptr);

    pDeviceContext->VSSetShader(nullptr, nullptr, 0);

    if (textureProbe) {
        ID3D11ShaderResourceView *nullSRV = nullptr;
        ID3D11SamplerState *nullSampler = nullptr;
        pDeviceContext->PSSetShaderResources(0, 1, &nullSRV);
        pDeviceContext->PSSetSamplers(0, 1, &nullSampler);
    }

    pDeviceContext->PSSetShader(nullptr, nullptr, 0);

    pDeviceContext->RSSetState(nullptr);

    if (fullscreen) {
        pSwapChain->SetFullscreenState(FALSE, nullptr);
    }

    DestroyWindow(hWnd);

    return EXIT_SUCCESS;
}
