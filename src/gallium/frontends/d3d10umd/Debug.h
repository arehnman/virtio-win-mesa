#pragma once

#ifdef __MINGW32__
#undef WIN32_LEAN_AND_MEAN /* for DEFINE_GUID macro */
#define _NO_OLDNAMES       /* avoid defining ssize_t */
#include <stdio.h>         /* for vsnprintf */
#undef fileno              /* we redefine this in vm_basic_defs.h */
#endif

#include <windows.h>
#include "util/u_debug.h"

#ifdef __cplusplus
extern "C" {
#endif


#define ST_DEBUG_OLD_TEX_OPS   (1 <<  0)
#define ST_DEBUG_TGSI          (1 <<  1)


#if MESA_DEBUG
extern unsigned st_debug;
#else
#define st_debug 0
#endif


#if MESA_DEBUG
void st_debug_parse(void);
#else
#define st_debug_parse() ((void)0)
#endif


void
DebugPrintf(const char *format, ...);

void
LogUnsupported(const char *file,
               unsigned line,
               const char *function,
               const char *expression);

/*
 * Resource lifetime and binding trace.  These fire on bind paths, several
 * times per draw, so they are emitted as a structured ETW event with generic
 * slots rather than a formatted string - a vsnprintf per event cost ~7% of the
 * process in a CPU profile of the draw path, and distorted the very captures
 * used to decide what to optimise.
 *
 * Add a kind here rather than reaching for a printf.  What a, b and c mean is
 * per-kind; keep the comment next to the enumerator honest.
 */
enum ResourceEventKind {
   RESOURCE_EVENT_REGISTER = 1,          /* a: primary, b: buffer, c: subresources */
   RESOURCE_EVENT_UNREGISTER,            /* a: primary, b: buffer, c: subresources */
   RESOURCE_EVENT_CREATE,                /* a: bind, b: ddi bind, c: misc flags */
   RESOURCE_EVENT_OPEN,                  /* a: bind, b: flags, c: hKMResource */
   RESOURCE_EVENT_DESTROY_BEGIN,         /* a: buffer, b: subresources */
   RESOURCE_EVENT_DESTROY_END,
   RESOURCE_EVENT_DEVICE_DESTROY_BEGIN,  /* object: pipe, c: screen */
   RESOURCE_EVENT_DEVICE_DESTROY_PIPE,   /* c: screen */
   RESOURCE_EVENT_DEVICE_DESTROY_END,
   RESOURCE_EVENT_DEVICE_LIVE_RESOURCE,  /* a: index, b: buffer */
   RESOURCE_EVENT_DEVICE_LIVE_COUNT,     /* a: count */
   RESOURCE_EVENT_SET_VERTEX_BUFFER,     /* a: slot, b: stride, c: offset */
   RESOURCE_EVENT_SET_INDEX_BUFFER,      /* a: format, c: offset */
   RESOURCE_EVENT_SET_CONSTANT_BUFFER,   /* a: stage, b: slot, c: size */
   RESOURCE_EVENT_SET_SHADER_RESOURCES,  /* a: stage, b: slot */
   RESOURCE_EVENT_SET_RENDER_TARGET,     /* a: slot */
   RESOURCE_EVENT_SET_DEPTH_STENCIL,
   RESOURCE_EVENT_RTV_CREATE,
   RESOURCE_EVENT_RTV_DESTROY,
   RESOURCE_EVENT_DSV_CREATE,
   RESOURCE_EVENT_DSV_DESTROY,
   RESOURCE_EVENT_SRV_CREATE,
   RESOURCE_EVENT_SRV_DESTROY,
   RESOURCE_EVENT_CB_BEFORE,             /* a: callback id, c: handle */
   RESOURCE_EVENT_CB_AFTER,              /* a: callback id, b: status */
};

/* Which runtime callback a RESOURCE_EVENT_CB_* event refers to, in slot a. */
enum ResourceEventCallback {
   RESOURCE_EVENT_CB_DESTROY_CONTEXT = 1,
   RESOURCE_EVENT_CB_DEALLOCATE,
   RESOURCE_EVENT_CB_ALLOCATE,
};

void
ResourceEvent(enum ResourceEventKind kind,
              uint64_t handle,
              const void *object,
              const void *pipe_resource,
              int32_t refcount,
              uint32_t a,
              uint32_t b,
              uint64_t c);


void
CheckHResult(HRESULT hr, const char *function, unsigned line);


#define CHECK_NTSTATUS(status) \
   CheckNTStatus(status, __func__, __LINE__)


#define CHECK_HRESULT(hr) \
   CheckHResult(hr, __func__, __LINE__)


void
AssertFail(const char *expr, const char *file, unsigned line, const char *function);

#ifndef NDEBUG
#undef assert
#define assert(expr) ((expr) ? (void)0 : AssertFail(#expr, __FILE__, __LINE__, __func__))
#endif


#ifndef NDEBUG
#define ASSERT(expr) ((expr) ? (void)0 : AssertFail(#expr, __FILE__, __LINE__, __func__))
#else
#define ASSERT(expr) do { } while (0 && (expr))
#endif


#if 0 && !defined(NDEBUG)
#define LOG_ENTRYPOINT() DebugPrintf("%s\n", __func__)
#else
#define LOG_ENTRYPOINT() (void)0
#endif

#define LOG_UNSUPPORTED_ENTRYPOINT() \
   LogUnsupported(__FILE__, __LINE__, __func__, "entrypoint")

#define LOG_UNSUPPORTED(expr) \
   do { \
      if (expr) \
         LogUnsupported(__FILE__, __LINE__, __func__, #expr); \
   } while (0)


#ifdef __cplusplus
}
#endif

