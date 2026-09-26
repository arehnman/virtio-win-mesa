#include "Debug.h"

#include <stdarg.h>
#include <stdio.h>

#include "gallium/winsys/yttrium/gdi/yttrium_gdi_public.h"


#if MESA_DEBUG

unsigned st_debug = 0;

static const
struct debug_named_value st_debug_flags[] = {
   {"oldtexops", ST_DEBUG_OLD_TEX_OPS, "oldtexops"},
   {"tgsi", ST_DEBUG_TGSI, "tgsi"},
   DEBUG_NAMED_VALUE_END
};
void
st_debug_parse(void)
{
   st_debug = debug_get_flags_option("ST_DEBUG", st_debug_flags, st_debug);
}

#endif


void
DebugPrintf(const char *format, ...)
{
    char buf[4096];

    va_list ap;
    va_start(ap, format);
    vsnprintf(buf, sizeof buf, format, ap);
    va_end(ap);

    OutputDebugStringA(buf);
}


void
LogUnsupported(const char *file,
               unsigned line,
               const char *function,
               const char *expression)
{
    const char *safe_file = file ? file : "<unknown>";
    const char *safe_function = function ? function : "<unknown>";
    const char *safe_expression = expression ? expression : "<unknown>";

    DebugPrintf("%s:%u XXX %s\n", safe_function, line, safe_expression);
    yttrium_gdi_trace_warnf(
        "yttrium: WARNING: unsupported behavior owner=d3d10umd "
        "function=%s file=%s line=%u expression=%s\n",
        safe_function, safe_file, line, safe_expression);
}

static const char *
ResourceEventKindName(enum ResourceEventKind kind)
{
   switch (kind) {
   case RESOURCE_EVENT_REGISTER:             return "register";
   case RESOURCE_EVENT_UNREGISTER:           return "unregister";
   case RESOURCE_EVENT_CREATE:               return "create";
   case RESOURCE_EVENT_OPEN:                 return "open";
   case RESOURCE_EVENT_DESTROY_BEGIN:        return "destroy_begin";
   case RESOURCE_EVENT_DESTROY_END:          return "destroy_end";
   case RESOURCE_EVENT_DEVICE_DESTROY_BEGIN: return "device_destroy_begin";
   case RESOURCE_EVENT_DEVICE_DESTROY_PIPE:  return "device_destroy_pipe";
   case RESOURCE_EVENT_DEVICE_DESTROY_END:   return "device_destroy_end";
   case RESOURCE_EVENT_DEVICE_LIVE_RESOURCE: return "device_live_resource";
   case RESOURCE_EVENT_DEVICE_LIVE_COUNT:    return "device_live_count";
   case RESOURCE_EVENT_SET_VERTEX_BUFFER:    return "set_vertex_buffer";
   case RESOURCE_EVENT_SET_INDEX_BUFFER:     return "set_index_buffer";
   case RESOURCE_EVENT_SET_CONSTANT_BUFFER:  return "set_constant_buffer";
   case RESOURCE_EVENT_SET_SHADER_RESOURCES: return "set_shader_resources";
   case RESOURCE_EVENT_SET_RENDER_TARGET:    return "set_render_target";
   case RESOURCE_EVENT_SET_DEPTH_STENCIL:    return "set_depth_stencil";
   case RESOURCE_EVENT_RTV_CREATE:           return "rtv_create";
   case RESOURCE_EVENT_RTV_DESTROY:          return "rtv_destroy";
   case RESOURCE_EVENT_DSV_CREATE:           return "dsv_create";
   case RESOURCE_EVENT_DSV_DESTROY:          return "dsv_destroy";
   case RESOURCE_EVENT_SRV_CREATE:           return "srv_create";
   case RESOURCE_EVENT_SRV_DESTROY:          return "srv_destroy";
   case RESOURCE_EVENT_CB_BEFORE:            return "cb_before";
   case RESOURCE_EVENT_CB_AFTER:             return "cb_after";
   }

   return "unknown";
}

void
ResourceEvent(enum ResourceEventKind kind,
              uint64_t handle,
              const void *object,
              const void *pipe_resource,
              int32_t refcount,
              uint32_t a,
              uint32_t b,
              uint64_t c)
{
   if (!yttrium_gdi_trace_is_enabled())
      return;

   yttrium_gdi_trace_resource_event(kind,
                                    ResourceEventKindName(kind),
                                    handle,
                                    (uint64_t)(uintptr_t)object,
                                    (uint64_t)(uintptr_t)pipe_resource,
                                    refcount,
                                    a,
                                    b,
                                    c);
}


/**
 * Produce a human readable message from HRESULT.
 *
 * @sa http://msdn.microsoft.com/en-us/library/ms679351(VS.85).aspx
 */
void
CheckHResult(HRESULT hr, const char *function, unsigned line)
{
   if (FAILED(hr)) {
      LPSTR lpMessageBuffer = NULL;

      FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                     FORMAT_MESSAGE_FROM_SYSTEM,
                     NULL,
                     hr,
                     MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                     (LPSTR)&lpMessageBuffer,
                     0,
                     NULL);

      DebugPrintf("%s: %u: 0x%08lX: %s", function, line, hr, lpMessageBuffer);

      LocalFree(lpMessageBuffer);
   }
}


void
AssertFail(const char *expr,
           const char *file,
           unsigned line,
           const char *function)
{
   yttrium_gdi_trace_errorf("%s:%u:%s: Assertion `%s' failed.\n",
                            file, line, function, expr);
   DebugPrintf("%s:%u:%s: Assertion `%s' failed.\n", file, line, function, expr);
#if defined(__GNUC__)
   __asm("int3");
#elif defined(_MSC_VER)
   __debugbreak();
#else
   DebugBreak();
#endif
}
