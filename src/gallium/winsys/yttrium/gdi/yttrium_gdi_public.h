/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */
#ifndef YTTRIUM_GDI_PUBLIC_H
#define YTTRIUM_GDI_PUBLIC_H

#include <stdbool.h>
#include <stdint.h>

#include "gallium/auxiliary/gdikmt/gdikmt.h"
#include "pipe/p_defines.h"

/* Yttrium-private pipe_resource flag: the KMD will CPU-read this display
 * image for a legacy blt Present, so its exported backing must be cached. */
#define YTTRIUM_GDI_RESOURCE_FLAG_CPU_READBACK PIPE_RESOURCE_FLAG_DRV_PRIV

/* Internal per-UAV counter image.  Its one R32_UINT texel may be copied to
 * an aligned byte offset in a buffer by CopyStructureCount. */
#define YTTRIUM_GDI_RESOURCE_FLAG_UAV_COUNTER (PIPE_RESOURCE_FLAG_DRV_PRIV << 1)

#ifdef __cplusplus
extern "C" {
#endif

struct pipe_resource;
struct pipe_context;
struct pipe_fence_handle;
struct yttrium_gdi_flush_issuance;
struct pipe_screen *
yttrium_gdi_screen_create(struct gdikmt_device *device);

void
yttrium_gdi_flush_labeled(struct pipe_context *ctx,
                          struct pipe_fence_handle **fence,
                          unsigned flags,
                          const char *label);

struct yttrium_gdi_flush_issuance *
yttrium_gdi_flush_async_issuance(struct pipe_context *ctx,
                                 const char *label);

bool
yttrium_gdi_wait_flush_issuance(
   struct yttrium_gdi_flush_issuance *issuance,
   uint32_t timeout_ms);

void
yttrium_gdi_flush_issuance_release(
   struct yttrium_gdi_flush_issuance *issuance);

/* The display allocation a Present published, handed to the ordered worker so
 * that it can issue the display flush after the frame completes. */
struct yttrium_gdi_present_publish_request {
   uint32_t allocation;
   uint32_t scanout_id;
   bool valid;
   struct gdikmt_context *signal_context;
   uint64_t signal_value;
};

bool
yttrium_gdi_flush_async_present(
   struct pipe_context *ctx,
   const char *label,
   const struct yttrium_gdi_present_publish_request *publish,
   struct yttrium_gdi_flush_issuance **issued);

bool
yttrium_gdi_screen_supports_logic_op(struct pipe_screen *screen);

/* Nonblocking handoff of an asynchronously detected native draw failure to
 * the D3D DDI entry point.  Reading the latch never drains the worker. */
bool
yttrium_gdi_take_draw_failure(struct pipe_context *ctx);

void
yttrium_gdi_resource_set_primary_target(struct pipe_resource *resource,
                                        bool primary_target);

void
yttrium_gdi_resource_set_allocation_ownership(struct pipe_resource *resource,
                                              bool owns_allocation);

bool
yttrium_gdi_resource_has_runtime_allocation(struct pipe_resource *resource);

bool
yttrium_gdi_resource_rotate_runtime_handles(
   struct pipe_resource *const *resources, unsigned count);

uint32_t
yttrium_gdi_pipeline_invalidate_resource(
   struct pipe_context *ctx, const struct pipe_resource *resource);

void
yttrium_gdi_resource_debug_log(struct pipe_resource *resource,
                               const char *label);

const char *
yttrium_gdi_debug_get_option(const char *name, const char *dfault);

bool
yttrium_gdi_debug_get_bool_option(const char *name, bool dfault);

int64_t
yttrium_gdi_debug_get_num_option(const char *name, int64_t dfault);

void
yttrium_gdi_debug_get_config_status(bool *loaded,
                                    bool *found,
                                    const char **path,
                                    unsigned *entry_count);

void
yttrium_gdi_trace_debugf(const char *format, ...);

void
yttrium_gdi_trace_warnf(const char *format, ...);

void
yttrium_gdi_trace_errorf(const char *format, ...);

bool
yttrium_gdi_trace_is_enabled(void);

/*
 * Resource lifetime and binding events.  Deliberately a single event with a
 * kind and generic slots rather than a formatted string: these fire on bind
 * paths, several times per draw, and a vsnprintf per event cost ~7% of the
 * process in a CPU profile of the draw path.  kind_name comes from a static
 * table, so nothing is formatted at emit time.  a, b and c carry whatever the
 * kind needs - see the kind enum for what each one means.
 */
void
yttrium_gdi_trace_resource_event(uint32_t kind,
                                 const char *kind_name,
                                 uint64_t handle,
                                 uint64_t object,
                                 uint64_t pipe_resource,
                                 int32_t refcount,
                                 uint32_t a,
                                 uint32_t b,
                                 uint64_t c);

enum yttrium_gdi_trace_dxgi_present_stage {
   YTTRIUM_GDI_TRACE_DXGI_PRESENT_BEGIN = 1,
   YTTRIUM_GDI_TRACE_DXGI_PRESENT_FLUSH_FRONTBUFFER = 2,
   YTTRIUM_GDI_TRACE_DXGI_PRESENT_END = 3,
};

void
yttrium_gdi_trace_dxgi_present(uint32_t stage,
                               uint64_t src_resource,
                               uint64_t dst_resource,
                               uint64_t src_allocation,
                               uint64_t dst_allocation,
                               uint64_t dxgi_context,
                               uint64_t hwnd,
                               uint64_t source,
                               uint64_t destination,
                               uint32_t src_subresource,
                               uint32_t dst_subresource,
                               uint32_t flags,
                               uint32_t flip_interval,
                               uint32_t present_count,
                               uint32_t gdi_readback,
                               uint32_t readback_presented);

enum yttrium_gdi_trace_pfn_present_stage {
   YTTRIUM_GDI_TRACE_PFN_PRESENT_BEFORE = 1,
   YTTRIUM_GDI_TRACE_PFN_PRESENT_AFTER = 2,
};

void
yttrium_gdi_trace_present_callback(uint32_t stage,
                                   long status,
                                   uint64_t src_allocation,
                                   uint64_t dst_allocation,
                                   uint64_t context,
                                   uint64_t dxgi_context,
                                   uint32_t private_size,
                                   uint32_t optimize_for_composition,
                                   uint32_t broadcast_count);

void
yttrium_gdi_user_logf(const char *format, ...);

#ifdef __cplusplus
}
#endif

#endif
