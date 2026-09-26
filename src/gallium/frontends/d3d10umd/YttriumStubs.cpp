/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "gallium/winsys/yttrium/gdi/yttrium_gdi_public.h"

#include "pipe/p_context.h"
#include "util/u_debug.h"

#include <stdarg.h>

extern "C" {

struct pipe_screen *
yttrium_gdi_screen_create(struct gdikmt_device *device)
{
   (void)device;
   return NULL;
}

void
yttrium_gdi_flush_labeled(struct pipe_context *ctx,
                          struct pipe_fence_handle **fence,
                          unsigned flags,
                          const char *label)
{
   (void)label;

   if (ctx && ctx->flush)
      ctx->flush(ctx, fence, flags);
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
   (void)ctx;
   (void)label;
   (void)publish;
   return false;
}

bool
yttrium_gdi_screen_supports_logic_op(struct pipe_screen *screen)
{
   (void)screen;
   /* Generic Gallium D3D10UMD targets advertised this capability before the
    * Yttrium-specific backend gate was introduced. */
   return true;
}

void
yttrium_gdi_resource_set_primary_target(struct pipe_resource *resource,
                                        bool primary_target)
{
   (void)resource;
   (void)primary_target;
}

void
yttrium_gdi_resource_set_allocation_ownership(struct pipe_resource *resource,
                                              bool owns_allocation)
{
   (void)resource;
   (void)owns_allocation;
}

bool
yttrium_gdi_resource_has_runtime_allocation(struct pipe_resource *resource)
{
   (void)resource;
   return false;
}

bool
yttrium_gdi_resource_rotate_runtime_handles(
   struct pipe_resource *const *resources, unsigned count)
{
   (void)resources;
   (void)count;
   return true;
}

uint32_t
yttrium_gdi_pipeline_invalidate_resource(
   struct pipe_context *ctx, const struct pipe_resource *resource)
{
   (void)ctx;
   (void)resource;
   return 0;
}

void
yttrium_gdi_resource_debug_log(struct pipe_resource *resource,
                               const char *label)
{
   (void)resource;
   (void)label;
}

const char *
yttrium_gdi_debug_get_option(const char *name, const char *dfault)
{
   return debug_get_option(name, dfault);
}

bool
yttrium_gdi_debug_get_bool_option(const char *name, bool dfault)
{
   return debug_get_bool_option(name, dfault);
}

int64_t
yttrium_gdi_debug_get_num_option(const char *name, int64_t dfault)
{
   return debug_get_num_option(name, dfault);
}

void
yttrium_gdi_debug_get_config_status(bool *loaded,
                                    bool *found,
                                    const char **path,
                                    unsigned *entry_count)
{
   if (loaded)
      *loaded = false;
   if (found)
      *found = false;
   if (path)
      *path = NULL;
   if (entry_count)
      *entry_count = 0;
}

void
yttrium_gdi_trace_debugf(const char *format, ...)
{
   (void)format;
}

void
yttrium_gdi_trace_warnf(const char *format, ...)
{
   va_list ap;

   _debug_printf("WARNING: ");
   va_start(ap, format);
   _debug_vprintf(format, ap);
   va_end(ap);
}

void
yttrium_gdi_trace_errorf(const char *format, ...)
{
   va_list ap;

   _debug_printf("ERROR: ");
   va_start(ap, format);
   _debug_vprintf(format, ap);
   va_end(ap);
}

bool
yttrium_gdi_trace_is_enabled(void)
{
   return false;
}

void
yttrium_gdi_trace_resource_event(uint32_t kind,
                                 const char *kind_name,
                                 uint64_t handle,
                                 uint64_t object,
                                 uint64_t pipe_resource,
                                 int32_t refcount,
                                 uint32_t a,
                                 uint32_t b,
                                 uint64_t c)
{
   (void)kind;
   (void)kind_name;
   (void)handle;
   (void)object;
   (void)pipe_resource;
   (void)refcount;
   (void)a;
   (void)b;
   (void)c;
}

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
                               uint32_t readback_presented)
{
   (void)stage;
   (void)src_resource;
   (void)dst_resource;
   (void)src_allocation;
   (void)dst_allocation;
   (void)dxgi_context;
   (void)hwnd;
   (void)source;
   (void)destination;
   (void)src_subresource;
   (void)dst_subresource;
   (void)flags;
   (void)flip_interval;
   (void)present_count;
   (void)gdi_readback;
   (void)readback_presented;
}

void
yttrium_gdi_trace_present_callback(uint32_t stage,
                                   long status,
                                   uint64_t src_allocation,
                                   uint64_t dst_allocation,
                                   uint64_t context,
                                   uint64_t dxgi_context,
                                   uint32_t private_size,
                                   uint32_t optimize_for_composition,
                                   uint32_t broadcast_count)
{
   (void)stage;
   (void)status;
   (void)src_allocation;
   (void)dst_allocation;
   (void)context;
   (void)dxgi_context;
   (void)private_size;
   (void)optimize_for_composition;
   (void)broadcast_count;
}

void
yttrium_gdi_user_logf(const char *format, ...)
{
   (void)format;
}

}
