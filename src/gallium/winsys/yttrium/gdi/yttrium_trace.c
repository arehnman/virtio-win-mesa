/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#include "yttrium_trace.h"

/* The Windows SDK TraceLogging macros require the MSVC toolchain.  Other
 * builds retain text/error logging but compile ETW events as no-ops. */
#if defined(_MSC_VER)
#define YTTRIUM_HAVE_TRACELOGGING 1

#if defined(__clang__) && !defined(TLG_INLINE)
#define TLG_INLINE static inline
#endif
#if defined(__clang__) && !defined(TLG_PFORCEINLINE)
#define TLG_PFORCEINLINE static inline
#endif

#include <TraceLoggingProvider.h>
#else
#define YTTRIUM_HAVE_TRACELOGGING 0

/* Keep trace call-site arguments referenced without evaluating them. */
#define yttrium_tracelogging_provider 0
#define YTTRIUM_TRACELOGGING_UNUSED(value) ((int)sizeof(value))
#define YTTRIUM_TRACELOGGING_FIELD(value, name) \
   ((int)(sizeof(value) + sizeof(name)))
#define TraceLoggingLevel(value) YTTRIUM_TRACELOGGING_UNUSED(value)
#define TraceLoggingWideString(value, name) \
   YTTRIUM_TRACELOGGING_FIELD(value, name)
#define TraceLoggingString(value, name) \
   YTTRIUM_TRACELOGGING_FIELD(value, name)
#define TraceLoggingUInt32(value, name) \
   YTTRIUM_TRACELOGGING_FIELD(value, name)
#define TraceLoggingUInt64(value, name) \
   YTTRIUM_TRACELOGGING_FIELD(value, name)
#define TraceLoggingInt32(value, name) \
   YTTRIUM_TRACELOGGING_FIELD(value, name)
#define TraceLoggingFloat32(value, name) \
   YTTRIUM_TRACELOGGING_FIELD(value, name)
#define TraceLoggingWrite(provider, event_name, ...) \
   do { \
      (void)sizeof(provider); \
      (void)sizeof(event_name); \
      (void)sizeof((int[]){0, __VA_ARGS__}); \
   } while (0)
#endif

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "yttrium_options.h"

#if YTTRIUM_HAVE_TRACELOGGING && defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#endif
#if YTTRIUM_HAVE_TRACELOGGING
TRACELOGGING_DEFINE_PROVIDER(
   yttrium_tracelogging_provider,
   "Yttrium-Mesa",
   (0x6ad5f67a, 0x0f8b, 0x4b6c, 0xb0, 0x0e, 0x0d, 0x99, 0x28, 0x72,
    0xa0, 0x20));
#endif
#if YTTRIUM_HAVE_TRACELOGGING && defined(__clang__)
#pragma clang diagnostic pop
#endif

static volatile LONG yttrium_trace_refs;
#if YTTRIUM_HAVE_TRACELOGGING
static volatile LONG yttrium_tracelogging_registered;
#endif
static volatile LONG yttrium_trace_etw_enabled;
static volatile LONG yttrium_trace_wait_stats_only;
static volatile LONG yttrium_trace_text_log_enabled;
static volatile LONG yttrium_trace_verbose_etw_text;
static volatile LONG yttrium_trace_error_log_dir_ready;

#define YTTRIUM_TRACE_ERROR_LOG_DIR "C:\\ProgramData\\Yttrium"
#define YTTRIUM_TRACE_ERROR_LOG_PATH \
   YTTRIUM_TRACE_ERROR_LOG_DIR "\\yttrium-errors.log"

void
yttrium_trace_init(bool etw_enabled, bool text_enabled)
{
   yttrium_trace_text_log_enabled = text_enabled ? 1 : 0;
   yttrium_trace_verbose_etw_text =
      yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_VERBOSE_ETW_TEXT", false) ? 1 : 0;
   yttrium_trace_wait_stats_only =
      yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_WAIT_STATS_ONLY", false) ? 1 : 0;

   LONG refs = InterlockedIncrement(&yttrium_trace_refs);
#if YTTRIUM_HAVE_TRACELOGGING
   if (refs == 1) {
      yttrium_trace_etw_enabled = etw_enabled ? 1 : 0;
      if (etw_enabled &&
          TraceLoggingRegister(yttrium_tracelogging_provider) == ERROR_SUCCESS) {
         yttrium_tracelogging_registered = 1;
      }
   } else if (etw_enabled && !yttrium_tracelogging_registered) {
      yttrium_trace_etw_enabled = 1;
      if (!yttrium_tracelogging_registered &&
          InterlockedCompareExchange(&yttrium_tracelogging_registered,
                                     1, 0) == 0 &&
          TraceLoggingRegister(yttrium_tracelogging_provider) != ERROR_SUCCESS) {
         yttrium_tracelogging_registered = 0;
      }
   }
#else
   (void)refs;
   (void)etw_enabled;
   yttrium_trace_etw_enabled = 0;
#endif
}

void
yttrium_trace_shutdown(void)
{
   LONG refs = InterlockedDecrement(&yttrium_trace_refs);
   if (refs == 0) {
#if YTTRIUM_HAVE_TRACELOGGING
      if (InterlockedCompareExchange(&yttrium_tracelogging_registered,
                                     0, 1) == 1)
         TraceLoggingUnregister(yttrium_tracelogging_provider);
#endif
      yttrium_trace_etw_enabled = 0;
      yttrium_trace_wait_stats_only = 0;
      yttrium_trace_text_log_enabled = 0;
      yttrium_trace_verbose_etw_text = 0;
   } else if (refs < 0) {
      yttrium_trace_refs = 0;
   }
}

static bool
yttrium_trace_provider_is_enabled(void)
{
#if YTTRIUM_HAVE_TRACELOGGING
   return yttrium_trace_etw_enabled && yttrium_tracelogging_registered &&
          TraceLoggingProviderEnabled(yttrium_tracelogging_provider, 5, 0);
#else
   return false;
#endif
}

bool
yttrium_trace_is_enabled(void)
{
   return !yttrium_trace_wait_stats_only &&
          yttrium_trace_provider_is_enabled();
}

bool
yttrium_trace_sync_wait_is_enabled(void)
{
   return yttrium_trace_provider_is_enabled();
}

static bool
yttrium_tracelogging_is_enabled(void)
{
   return yttrium_trace_is_enabled();
}

bool
yttrium_trace_text_enabled(void)
{
   return yttrium_trace_text_log_enabled != 0;
}

bool
yttrium_trace_verbose_etw_text_enabled(void)
{
   return yttrium_trace_verbose_etw_text != 0;
}

static void
yttrium_trace_write_verbose_string(const WCHAR *message)
{
   if (!message || !yttrium_trace_verbose_etw_text_enabled() ||
       !yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "VerboseText",
      TraceLoggingLevel(5),
      TraceLoggingWideString(message, "message"));
}

const char *
yttrium_trace_process_name(void)
{
   static char name[64];
   static volatile LONG ready;

   if (!ready) {
      char path[MAX_PATH];
      const DWORD length = GetModuleFileNameA(NULL, path, ARRAY_SIZE(path));
      const char *base = path;

      if (length && length < ARRAY_SIZE(path)) {
         for (const char *p = path; *p; p++) {
            if (*p == '\\' || *p == '/')
               base = p + 1;
         }
      } else {
         base = "?";
      }

      snprintf(name, sizeof(name), "%s", base);
      InterlockedExchange(&ready, 1);
   }

   return name;
}

uint64_t
yttrium_trace_now_us(void)
{
   static LARGE_INTEGER freq;
   static volatile LONG freq_ready;
   LARGE_INTEGER now;

   if (!freq_ready) {
      LARGE_INTEGER local_freq;
      QueryPerformanceFrequency(&local_freq);
      freq = local_freq;
      InterlockedExchange(&freq_ready, 1);
   }

   QueryPerformanceCounter(&now);
   return (uint64_t)((now.QuadPart * 1000000ull) / freq.QuadPart);
}

static const char *
yttrium_trace_severity_name(uint32_t severity)
{
   switch (severity) {
   case YTTRIUM_TRACE_ERROR:
      return "error";
   case YTTRIUM_TRACE_WARNING:
      return "warning";
   default:
      return "debug";
   }
}

static void
yttrium_trace_write_error_log(uint32_t severity, const char *message)
{
   char record[1280];
   SYSTEMTIME time;
   HANDLE file;
   DWORD ignored;
   size_t len;

   if (severity < YTTRIUM_TRACE_WARNING || !message)
      return;

   if (InterlockedCompareExchange(&yttrium_trace_error_log_dir_ready, 1, 0) == 0)
      CreateDirectoryA(YTTRIUM_TRACE_ERROR_LOG_DIR, NULL);

   GetLocalTime(&time);
   snprintf(record, sizeof(record),
            "%04u-%02u-%02u %02u:%02u:%02u.%03u [%s pid=%lu process=%s] %s",
            (unsigned)time.wYear, (unsigned)time.wMonth,
            (unsigned)time.wDay, (unsigned)time.wHour,
            (unsigned)time.wMinute, (unsigned)time.wSecond,
            (unsigned)time.wMilliseconds,
            yttrium_trace_severity_name(severity),
            (unsigned long)GetCurrentProcessId(),
            yttrium_trace_process_name(), message);
   record[sizeof(record) - 1] = '\0';

   len = strlen(record);
   if (len && record[len - 1] != '\n' && len + 1 < sizeof(record)) {
      record[len++] = '\n';
      record[len] = '\0';
   }

   file = CreateFileA(YTTRIUM_TRACE_ERROR_LOG_PATH,
                      FILE_APPEND_DATA,
                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                      NULL,
                      OPEN_ALWAYS,
                      FILE_ATTRIBUTE_NORMAL,
                      NULL);
   if (file == INVALID_HANDLE_VALUE)
      return;

   WriteFile(file, record, (DWORD)strlen(record), &ignored, NULL);
   CloseHandle(file);
}

void
yttrium_trace_logf(uint32_t severity, const char *format, ...)
{
   char message[1024];
   va_list ap;
   int written;

   if (!format)
      return;

   if (severity == YTTRIUM_TRACE_DEBUG &&
       (!yttrium_trace_is_enabled() ||
        !yttrium_trace_verbose_etw_text_enabled()))
      return;

   va_start(ap, format);
   written = vsnprintf(message, sizeof(message), format, ap);
   va_end(ap);

   if (written < 0)
      return;

   message[sizeof(message) - 1] = '\0';

   if (severity >= YTTRIUM_TRACE_WARNING) {
      debug_printf("%s", message);
      yttrium_trace_write_error_log(severity, message);
   }

   if (yttrium_trace_is_enabled() && severity == YTTRIUM_TRACE_ERROR) {
      TraceLoggingWrite(
         yttrium_tracelogging_provider,
         "Message",
         TraceLoggingLevel(2),
         TraceLoggingUInt32(severity, "severity"),
         TraceLoggingString(yttrium_trace_severity_name(severity),
                            "severity_name"),
         TraceLoggingUInt32((uint32_t)(written >= (int)sizeof(message)),
                            "truncated"),
         TraceLoggingString(message, "message"));
   } else if (yttrium_trace_is_enabled() &&
              severity == YTTRIUM_TRACE_WARNING) {
      TraceLoggingWrite(
         yttrium_tracelogging_provider,
         "Message",
         TraceLoggingLevel(3),
         TraceLoggingUInt32(severity, "severity"),
         TraceLoggingString(yttrium_trace_severity_name(severity),
                            "severity_name"),
         TraceLoggingUInt32((uint32_t)(written >= (int)sizeof(message)),
                            "truncated"),
         TraceLoggingString(message, "message"));
   } else if (yttrium_trace_is_enabled()) {
      TraceLoggingWrite(
         yttrium_tracelogging_provider,
         "Message",
         TraceLoggingLevel(5),
         TraceLoggingUInt32(severity, "severity"),
         TraceLoggingString(yttrium_trace_severity_name(severity),
                            "severity_name"),
         TraceLoggingUInt32((uint32_t)(written >= (int)sizeof(message)),
                            "truncated"),
         TraceLoggingString(message, "message"));
   }
}

void
yttrium_trace_debug_stringf(const char *format, ...)
{
   char message[1024];
   wchar_t wide_message[1024];
   va_list ap;
   int written;

   if (!format || !yttrium_trace_is_enabled() ||
       !yttrium_trace_verbose_etw_text_enabled())
      return;

   va_start(ap, format);
   written = vsnprintf(message, sizeof(message), format, ap);
   va_end(ap);

   if (written < 0)
      return;

   message[sizeof(message) - 1] = '\0';
   MultiByteToWideChar(CP_UTF8, 0, message, -1, wide_message,
                       ARRAY_SIZE(wide_message));
   wide_message[ARRAY_SIZE(wide_message) - 1] = L'\0';
   yttrium_trace_write_verbose_string(wide_message);
}

void
yttrium_trace_resource_event(uint32_t kind,
                             const char *kind_name,
                             uint64_t handle,
                             uint64_t object,
                             uint64_t pipe_resource,
                             int32_t refcount,
                             uint32_t a,
                             uint32_t b,
                             uint64_t c)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "ResourceLifetime",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(kind, "kind"),
      TraceLoggingString(kind_name ? kind_name : "", "kind_name"),
      TraceLoggingUInt64(handle, "handle"),
      TraceLoggingUInt64(object, "object"),
      TraceLoggingUInt64(pipe_resource, "pipe_resource"),
      TraceLoggingInt32(refcount, "refcount"),
      TraceLoggingUInt32(a, "a"),
      TraceLoggingUInt32(b, "b"),
      TraceLoggingUInt64(c, "c"));
}

void
yttrium_trace_dxgi_present(uint32_t stage,
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
   if (!yttrium_trace_sync_wait_is_enabled())
      return;

   const uint32_t pid = GetCurrentProcessId();
   const uint32_t tid = GetCurrentThreadId();
   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "DxgiPresent",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(stage, "stage"),
      TraceLoggingUInt32(pid, "pid"),
      TraceLoggingUInt32(tid, "tid"),
      TraceLoggingUInt64(src_resource, "src_resource"),
      TraceLoggingUInt64(dst_resource, "dst_resource"),
      TraceLoggingUInt64(src_allocation, "src_allocation"),
      TraceLoggingUInt64(dst_allocation, "dst_allocation"),
      TraceLoggingUInt64(dxgi_context, "dxgi_context"),
      TraceLoggingUInt64(hwnd, "hwnd"),
      TraceLoggingUInt64(source, "source"),
      TraceLoggingUInt64(destination, "destination"),
      TraceLoggingUInt32(src_subresource, "src_subresource"),
      TraceLoggingUInt32(dst_subresource, "dst_subresource"),
      TraceLoggingUInt32(flags, "flags"),
      TraceLoggingUInt32(flip_interval, "flip_interval"),
      TraceLoggingUInt32(present_count, "present_count"),
      TraceLoggingUInt32(gdi_readback, "gdi_readback"),
      TraceLoggingUInt32(readback_presented, "readback_presented"));

   if (!yttrium_trace_verbose_etw_text_enabled())
      return;

   WCHAR message[512];
   swprintf(message, ARRAY_SIZE(message),
            L"yttrium dxgi_present stage=%u pid=%u tid=%u src_resource=0x%llx dst_resource=0x%llx src_alloc=0x%llx dst_alloc=0x%llx dxgi=0x%llx hwnd=0x%llx source=0x%llx destination=0x%llx src_sub=%u dst_sub=%u flags=0x%x flip=%u present_count=%u gdi_readback=%u readback_presented=%u",
            stage, pid, tid,
            (unsigned long long)src_resource,
            (unsigned long long)dst_resource,
            (unsigned long long)src_allocation,
            (unsigned long long)dst_allocation,
            (unsigned long long)dxgi_context,
            (unsigned long long)hwnd,
            (unsigned long long)source,
            (unsigned long long)destination,
            src_subresource, dst_subresource, flags, flip_interval,
            present_count, gdi_readback, readback_presented);
   message[ARRAY_SIZE(message) - 1] = L'\0';
   yttrium_trace_write_verbose_string(message);
}

void
yttrium_trace_present_callback(uint32_t stage,
                               long status,
                               uint64_t src_allocation,
                               uint64_t dst_allocation,
                               uint64_t context,
                               uint64_t dxgi_context,
                               uint32_t private_size,
                               uint32_t optimize_for_composition,
                               uint32_t broadcast_count)
{
   if (!yttrium_trace_sync_wait_is_enabled())
      return;

   const uint32_t pid = GetCurrentProcessId();
   const uint32_t tid = GetCurrentThreadId();
   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "PresentCallback",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(stage, "stage"),
      TraceLoggingUInt32(pid, "pid"),
      TraceLoggingUInt32(tid, "tid"),
      TraceLoggingUInt32((uint32_t)status, "status"),
      TraceLoggingUInt64(src_allocation, "src_allocation"),
      TraceLoggingUInt64(dst_allocation, "dst_allocation"),
      TraceLoggingUInt64(context, "context"),
      TraceLoggingUInt64(dxgi_context, "dxgi_context"),
      TraceLoggingUInt32(private_size, "private_size"),
      TraceLoggingUInt32(optimize_for_composition,
                         "optimize_for_composition"),
      TraceLoggingUInt32(broadcast_count, "broadcast_count"));

   if (!yttrium_trace_verbose_etw_text_enabled())
      return;

   WCHAR message[384];
   swprintf(message, ARRAY_SIZE(message),
            L"yttrium pfn_present_cb stage=%u pid=%u tid=%u status=0x%lx src_alloc=0x%llx dst_alloc=0x%llx context=0x%llx dxgi=0x%llx private_size=%u optimize=%u broadcast=%u",
            stage, pid, tid, (unsigned long)status,
            (unsigned long long)src_allocation,
            (unsigned long long)dst_allocation,
            (unsigned long long)context,
            (unsigned long long)dxgi_context,
            private_size, optimize_for_composition, broadcast_count);
   message[ARRAY_SIZE(message) - 1] = L'\0';
   yttrium_trace_write_verbose_string(message);
}

void
yttrium_trace_resource_import(uint32_t stage,
                              long status,
                              uint64_t global_handle,
                              uint64_t resource_handle,
                              uint64_t allocation,
                              uint32_t resource_id,
                              uint64_t memory_id,
                              uint32_t width,
                              uint32_t height,
                              uint32_t format,
                              uint32_t bind,
                              uint32_t flags,
                              uint32_t usage,
                              uint32_t display,
                              uint32_t classic,
                              uint32_t owns,
                              uint32_t venus_initialized,
                              uint32_t import_enabled,
                              uint64_t image_id,
                              uint64_t buffer_id,
                              uint64_t size,
                              uint32_t stride)
{
   if (!yttrium_trace_is_enabled())
      return;

   const uint32_t pid = GetCurrentProcessId();
   const uint32_t tid = GetCurrentThreadId();
   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "ResourceImport",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(stage, "stage"),
      TraceLoggingUInt32(pid, "pid"),
      TraceLoggingUInt32(tid, "tid"),
      TraceLoggingUInt32((uint32_t)status, "status"),
      TraceLoggingUInt64(global_handle, "global_handle"),
      TraceLoggingUInt64(resource_handle, "resource_handle"),
      TraceLoggingUInt64(allocation, "allocation"),
      TraceLoggingUInt32(resource_id, "resource_id"),
      TraceLoggingUInt64(memory_id, "memory_id"),
      TraceLoggingUInt32(width, "width"),
      TraceLoggingUInt32(height, "height"),
      TraceLoggingUInt32(format, "format"),
      TraceLoggingUInt32(bind, "bind"),
      TraceLoggingUInt32(flags, "flags"),
      TraceLoggingUInt32(usage, "usage"),
      TraceLoggingUInt32(display, "display"),
      TraceLoggingUInt32(classic, "classic"),
      TraceLoggingUInt32(owns, "owns"),
      TraceLoggingUInt32(venus_initialized, "venus_initialized"),
      TraceLoggingUInt32(import_enabled, "import_enabled"),
      TraceLoggingUInt64(image_id, "image_id"),
      TraceLoggingUInt64(buffer_id, "buffer_id"),
      TraceLoggingUInt64(size, "size"),
      TraceLoggingUInt32(stride, "stride"));

   if (!yttrium_trace_verbose_etw_text_enabled())
      return;

   WCHAR message[512];
   swprintf(message, ARRAY_SIZE(message),
            L"yttrium resource_import stage=%u pid=%u tid=%u status=0x%lx global=0x%llx hResource=0x%llx hAllocation=0x%llx res=%u mem=0x%llx size=%ux%u format=%u bind=0x%x flags=0x%x usage=%u display=%u classic=%u owns=%u venus=%u import=%u image=0x%llx buffer=0x%llx bytes=0x%llx stride=%u",
            stage, pid, tid, (unsigned long)status,
            (unsigned long long)global_handle,
            (unsigned long long)resource_handle,
            (unsigned long long)allocation, resource_id,
            (unsigned long long)memory_id, width, height, format, bind, flags,
            usage, display, classic, owns, venus_initialized, import_enabled,
            (unsigned long long)image_id, (unsigned long long)buffer_id,
            (unsigned long long)size, stride);
   message[ARRAY_SIZE(message) - 1] = L'\0';
   yttrium_trace_write_verbose_string(message);
}

void
yttrium_trace_resource_destroy(uint32_t stage,
                               long status,
                               uint64_t resource,
                               uint64_t resource_handle,
                               uint64_t allocation,
                               uint32_t resource_id,
                               uint64_t memory_id,
                               uint32_t display,
                               uint32_t primary,
                               uint32_t classic,
                               uint32_t owns,
                               uint32_t venus_initialized,
                               uint32_t venus_buffer_backed,
                               uint64_t image_id,
                               uint64_t buffer_id,
                               uint64_t venus_memory_id,
                               uint64_t scanout_allocation,
                               uint32_t scanout_resource_id,
                               uint32_t scanout_initialized,
                               uint64_t size,
                               uint32_t stride)
{
   if (!yttrium_trace_is_enabled())
      return;

   const uint32_t pid = GetCurrentProcessId();
   const uint32_t tid = GetCurrentThreadId();
   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "ResourceDestroy",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(stage, "stage"),
      TraceLoggingUInt32(pid, "pid"),
      TraceLoggingUInt32(tid, "tid"),
      TraceLoggingUInt32((uint32_t)status, "status"),
      TraceLoggingUInt64(resource, "resource"),
      TraceLoggingUInt64(resource_handle, "resource_handle"),
      TraceLoggingUInt64(allocation, "allocation"),
      TraceLoggingUInt32(resource_id, "resource_id"),
      TraceLoggingUInt64(memory_id, "memory_id"),
      TraceLoggingUInt32(display, "display"),
      TraceLoggingUInt32(primary, "primary"),
      TraceLoggingUInt32(classic, "classic"),
      TraceLoggingUInt32(owns, "owns"),
      TraceLoggingUInt32(venus_initialized, "venus_initialized"),
      TraceLoggingUInt32(venus_buffer_backed, "venus_buffer_backed"),
      TraceLoggingUInt64(image_id, "image_id"),
      TraceLoggingUInt64(buffer_id, "buffer_id"),
      TraceLoggingUInt64(venus_memory_id, "venus_memory_id"),
      TraceLoggingUInt64(scanout_allocation, "scanout_allocation"),
      TraceLoggingUInt32(scanout_resource_id, "scanout_resource_id"),
      TraceLoggingUInt32(scanout_initialized, "scanout_initialized"),
      TraceLoggingUInt64(size, "size"),
      TraceLoggingUInt32(stride, "stride"));

   if (!yttrium_trace_verbose_etw_text_enabled())
      return;

   WCHAR message[512];
   swprintf(message, ARRAY_SIZE(message),
            L"yttrium resource_destroy stage=%u pid=%u tid=%u status=0x%lx resource=0x%llx hResource=0x%llx hAllocation=0x%llx res=%u mem=0x%llx display=%u primary=%u classic=%u owns=%u venus=%u buffer_backed=%u image=0x%llx buffer=0x%llx venus_mem=0x%llx scanout_alloc=0x%llx scanout_res=%u scanout_init=%u bytes=0x%llx stride=%u",
            stage, pid, tid, (unsigned long)status,
            (unsigned long long)resource,
            (unsigned long long)resource_handle,
            (unsigned long long)allocation, resource_id,
            (unsigned long long)memory_id, display, primary, classic, owns,
            venus_initialized, venus_buffer_backed,
            (unsigned long long)image_id, (unsigned long long)buffer_id,
            (unsigned long long)venus_memory_id,
            (unsigned long long)scanout_allocation, scanout_resource_id,
            scanout_initialized, (unsigned long long)size, stride);
   message[ARRAY_SIZE(message) - 1] = L'\0';
   yttrium_trace_write_verbose_string(message);
}

void
yttrium_trace_timing(uint32_t point,
                     uint32_t status,
                     uint64_t elapsed_us,
                     const char *label,
                     uint64_t a,
                     uint64_t b,
                     uint32_t c,
                     uint32_t d)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "Timing",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(point, "point"),
      TraceLoggingUInt32(status, "status"),
      TraceLoggingUInt64(elapsed_us, "elapsed_us"),
      TraceLoggingString(label ? label : "", "label"),
      TraceLoggingUInt64(a, "a"),
      TraceLoggingUInt64(b, "b"),
      TraceLoggingUInt32(c, "c"),
      TraceLoggingUInt32(d, "d"));

   if (!yttrium_trace_verbose_etw_text_enabled())
      return;

   WCHAR message[256];
   swprintf(message, sizeof(message) / sizeof(message[0]),
            L"yttrium timing point=%u status=0x%x elapsed_us=%llu label=%hs a=%llu b=%llu c=%u d=%u",
            point, status, (unsigned long long)elapsed_us,
            label ? label : "", (unsigned long long)a,
            (unsigned long long)b, c, d);
   message[(sizeof(message) / sizeof(message[0])) - 1] = L'\0';
   yttrium_trace_write_verbose_string(message);
}

void
yttrium_trace_scanout_refresh(uint32_t reason,
                              uint32_t copy_id,
                              uint64_t allocation,
                              uint32_t resource_id,
                              uint32_t x,
                              uint32_t y,
                              uint32_t width,
                              uint32_t height,
                              uint32_t display,
                              uint32_t primary,
                              uint32_t classic,
                              uint32_t venus,
                              uint32_t scanout_present)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "ScanoutRefresh",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(reason, "reason"),
      TraceLoggingUInt32(copy_id, "copy_id"),
      TraceLoggingUInt64(allocation, "allocation"),
      TraceLoggingUInt32(resource_id, "resource_id"),
      TraceLoggingUInt32(x, "x"),
      TraceLoggingUInt32(y, "y"),
      TraceLoggingUInt32(width, "width"),
      TraceLoggingUInt32(height, "height"),
      TraceLoggingUInt32(display, "display"),
      TraceLoggingUInt32(primary, "primary"),
      TraceLoggingUInt32(classic, "classic"),
      TraceLoggingUInt32(venus, "venus"),
      TraceLoggingUInt32(scanout_present, "scanout_present"));

   if (!yttrium_trace_verbose_etw_text_enabled())
      return;

   WCHAR message[256];
   swprintf(message, sizeof(message) / sizeof(message[0]),
            L"yttrium scanout_refresh copy=%u reason=%u allocation=0x%llx res=%u box=%u,%u %ux%u display=%u primary=%u classic=%u venus=%u scanout_present=%u",
            copy_id, reason, (unsigned long long)allocation, resource_id, x, y,
            width, height, display, primary, classic, venus, scanout_present);
   message[(sizeof(message) / sizeof(message[0])) - 1] = L'\0';
   yttrium_trace_write_verbose_string(message);
}

void
yttrium_trace_resource_copy_target(uint32_t copy_id,
                                   uint32_t stage,
                                   uint32_t path,
                                   uint64_t src_allocation,
                                   uint32_t src_resource_id,
                                   uint64_t dst_allocation,
                                   uint32_t dst_resource_id,
                                   uint32_t dst_target,
                                   uint32_t dst_format,
                                   uint32_t dst_bind,
                                   uint32_t dst_width,
                                   uint32_t dst_height,
                                   uint32_t x,
                                   uint32_t y,
                                   uint32_t width,
                                   uint32_t height,
                                   uint32_t display,
                                   uint32_t primary,
                                   uint32_t classic,
                                   uint32_t venus,
                                   uint32_t buffer_backed,
                                   uint32_t image,
                                   uint32_t data,
                                   uint32_t scanout_present)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "ResourceCopyTarget",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(copy_id, "copy_id"),
      TraceLoggingUInt32(stage, "stage"),
      TraceLoggingUInt32(path, "path"),
      TraceLoggingUInt64(src_allocation, "src_allocation"),
      TraceLoggingUInt32(src_resource_id, "src_resource_id"),
      TraceLoggingUInt64(dst_allocation, "dst_allocation"),
      TraceLoggingUInt32(dst_resource_id, "dst_resource_id"),
      TraceLoggingUInt32(dst_target, "dst_target"),
      TraceLoggingUInt32(dst_format, "dst_format"),
      TraceLoggingUInt32(dst_bind, "dst_bind"),
      TraceLoggingUInt32(dst_width, "dst_width"),
      TraceLoggingUInt32(dst_height, "dst_height"),
      TraceLoggingUInt32(x, "x"),
      TraceLoggingUInt32(y, "y"),
      TraceLoggingUInt32(width, "width"),
      TraceLoggingUInt32(height, "height"),
      TraceLoggingUInt32(display, "display"),
      TraceLoggingUInt32(primary, "primary"),
      TraceLoggingUInt32(classic, "classic"),
      TraceLoggingUInt32(venus, "venus"),
      TraceLoggingUInt32(buffer_backed, "buffer_backed"),
      TraceLoggingUInt32(image, "image"),
      TraceLoggingUInt32(data, "data"),
      TraceLoggingUInt32(scanout_present, "scanout_present"));

   if (!yttrium_trace_verbose_etw_text_enabled())
      return;

   WCHAR message[384];
   swprintf(message, sizeof(message) / sizeof(message[0]),
            L"yttrium resource_copy_target copy=%u stage=%u path=%u src=0x%llx/%u dst=0x%llx/%u target=%u format=%u bind=0x%x size=%ux%u box=%u,%u %ux%u display=%u primary=%u classic=%u venus=%u buffer=%u image=%u data=%u scanout_present=%u",
            copy_id, stage, path, (unsigned long long)src_allocation,
            src_resource_id, (unsigned long long)dst_allocation,
            dst_resource_id, dst_target, dst_format, dst_bind, dst_width,
            dst_height, x, y, width, height, display, primary, classic, venus,
            buffer_backed, image, data, scanout_present);
   message[(sizeof(message) / sizeof(message[0])) - 1] = L'\0';
   yttrium_trace_write_verbose_string(message);
}

void
yttrium_trace_venus_activity(uint64_t window_us,
                             uint64_t total_bytes,
                             uint64_t max_elapsed_us,
                             uint32_t total_count,
                             uint32_t raw_count,
                             uint32_t ring_count,
                             uint32_t wait_space_count,
                             uint32_t wait_seqno_count,
                             uint32_t reset_count,
                             uint32_t queue_submit_count,
                             uint32_t wait_fences_count,
                             uint32_t submit_wait_count,
                             uint32_t failure_count,
                             uint32_t last_point,
                             uint32_t last_status,
                             uint32_t last_c,
                             uint32_t last_d)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "VenusActivity",
      TraceLoggingLevel(5),
      TraceLoggingUInt64(window_us, "window_us"),
      TraceLoggingUInt64(total_bytes, "total_bytes"),
      TraceLoggingUInt64(max_elapsed_us, "max_elapsed_us"),
      TraceLoggingUInt32(total_count, "total_count"),
      TraceLoggingUInt32(raw_count, "raw_count"),
      TraceLoggingUInt32(ring_count, "ring_count"),
      TraceLoggingUInt32(wait_space_count, "wait_space_count"),
      TraceLoggingUInt32(wait_seqno_count, "wait_seqno_count"),
      TraceLoggingUInt32(reset_count, "reset_count"),
      TraceLoggingUInt32(queue_submit_count, "queue_submit_count"),
      TraceLoggingUInt32(wait_fences_count, "wait_fences_count"),
      TraceLoggingUInt32(submit_wait_count, "submit_wait_count"),
      TraceLoggingUInt32(failure_count, "failure_count"),
      TraceLoggingUInt32(last_point, "last_point"),
      TraceLoggingUInt32(last_status, "last_status"),
      TraceLoggingUInt32(last_c, "last_c"),
      TraceLoggingUInt32(last_d, "last_d"));

   if (!yttrium_trace_verbose_etw_text_enabled())
      return;

   WCHAR message[384];
   swprintf(message, sizeof(message) / sizeof(message[0]),
            L"yttrium venus_activity window_us=%llu total=%u raw=%u ring=%u wait_space=%u wait_seqno=%u reset=%u queue=%u wait_fences=%u submit_wait=%u failures=%u bytes=%llu max_us=%llu last_point=%u last_status=0x%x last_c=%u last_d=%u",
            (unsigned long long)window_us, total_count, raw_count, ring_count,
            wait_space_count, wait_seqno_count, reset_count,
            queue_submit_count, wait_fences_count, submit_wait_count,
            failure_count, (unsigned long long)total_bytes,
            (unsigned long long)max_elapsed_us, last_point, last_status,
            last_c, last_d);
   message[(sizeof(message) / sizeof(message[0])) - 1] = L'\0';
   yttrium_trace_write_verbose_string(message);
}

void
yttrium_trace_sync_wait(uint32_t kind,
                        const char *kind_name,
                        uint32_t status,
                        uint64_t elapsed_us,
                        uint32_t command_type,
                        const char *command_name,
                        uint32_t command_size,
                        uint32_t reply_size,
                        uint64_t backlog_command_count,
                        uint64_t backlog_command_bytes,
                        uint64_t backlog_queue_submit_count,
                        const char *label,
                        uint32_t a,
                        uint32_t b)
{
   struct {
      uint64_t elapsed_us;
      uint32_t kind;
      uint32_t status;
      uint32_t command_type;
      uint32_t command_size;
      uint32_t reply_size;
      uint64_t backlog_command_count;
      uint64_t backlog_command_bytes;
      uint64_t backlog_queue_submit_count;
      uint32_t a;
      uint32_t b;
      uint32_t pid;
      uint32_t tid;
      char kind_name[32];
      char command_name[64];
   } payload = {
      elapsed_us, kind, status, command_type, command_size, reply_size,
      backlog_command_count, backlog_command_bytes,
      backlog_queue_submit_count, a, b,
      GetCurrentProcessId(), GetCurrentThreadId(),
   };

   if (!yttrium_trace_sync_wait_is_enabled())
      return;

   strncpy(payload.kind_name, kind_name ? kind_name : "unknown",
           sizeof(payload.kind_name) - 1);
   payload.kind_name[sizeof(payload.kind_name) - 1] = '\0';
   strncpy(payload.command_name, command_name ? command_name : "unknown",
           sizeof(payload.command_name) - 1);
   payload.command_name[sizeof(payload.command_name) - 1] = '\0';

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "SyncWait",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(kind, "kind"),
      TraceLoggingString(payload.kind_name, "kind_name"),
      TraceLoggingUInt32(status, "status"),
      TraceLoggingUInt64(elapsed_us, "elapsed_us"),
      TraceLoggingUInt32(command_type, "command_type"),
      TraceLoggingString(payload.command_name, "command_name"),
      TraceLoggingUInt32(command_size, "command_size"),
      TraceLoggingUInt32(reply_size, "reply_size"),
      TraceLoggingUInt64(backlog_command_count, "backlog_command_count"),
      TraceLoggingUInt64(backlog_command_bytes, "backlog_command_bytes"),
      TraceLoggingUInt64(backlog_queue_submit_count,
                         "backlog_queue_submit_count"),
      TraceLoggingString(label ? label : "", "label"),
      TraceLoggingUInt32(a, "a"),
      TraceLoggingUInt32(b, "b"),
      TraceLoggingUInt32(payload.pid, "pid"),
      TraceLoggingUInt32(payload.tid, "tid"));

   if (!yttrium_trace_verbose_etw_text_enabled())
      return;

   char message[384];
   wchar_t wide_message[384];
   snprintf(message, sizeof(message),
            "yttrium sync_wait kind=%s(%u) status=0x%x elapsed_us=%llu command=%s(%u) cmd_size=%u reply_size=%u label=%s a=%u b=%u pid=%u tid=%u",
            payload.kind_name, kind, status,
            (unsigned long long)elapsed_us,
            payload.command_name, command_type, command_size, reply_size,
            label ? label : "", a, b, payload.pid, payload.tid);
   message[sizeof(message) - 1] = '\0';

   MultiByteToWideChar(CP_UTF8, 0, message, -1, wide_message,
                       ARRAY_SIZE(wide_message));
   wide_message[ARRAY_SIZE(wide_message) - 1] = L'\0';
   yttrium_trace_write_verbose_string(wide_message);
}

void
yttrium_trace_ring_kick(uint64_t elapsed_us,
                        uint32_t seqno,
                        uint32_t status,
                        uint32_t path,
                        const char *path_name,
                        uint32_t command_size,
                        uint32_t blocking,
                        const char *label)
{
   struct {
      char path_name[32];
      uint32_t pid;
      uint32_t tid;
   } payload = {
      .pid = GetCurrentProcessId(),
      .tid = GetCurrentThreadId(),
   };

   if (!yttrium_trace_sync_wait_is_enabled())
      return;

   strncpy(payload.path_name, path_name ? path_name : "unknown",
           sizeof(payload.path_name) - 1);
   payload.path_name[sizeof(payload.path_name) - 1] = '\0';

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "RingKick",
      TraceLoggingLevel(5),
      TraceLoggingUInt64(elapsed_us, "elapsed_us"),
      TraceLoggingUInt32(seqno, "seqno"),
      TraceLoggingUInt32(status, "status"),
      TraceLoggingUInt32(path, "path"),
      TraceLoggingString(payload.path_name, "path_name"),
      TraceLoggingUInt32(command_size, "command_size"),
      TraceLoggingUInt32(blocking, "blocking"),
      TraceLoggingString(label ? label : "", "label"),
      TraceLoggingUInt32(payload.pid, "pid"),
      TraceLoggingUInt32(payload.tid, "tid"));
}

void
yttrium_trace_sync_wait_summary(const char *message)
{
   wchar_t wide_message[512];

   if (!message || !yttrium_trace_verbose_etw_text_enabled() ||
       !yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "SyncWaitSummary",
      TraceLoggingLevel(5),
      TraceLoggingString(message, "message"));

   MultiByteToWideChar(CP_UTF8, 0, message, -1, wide_message,
                       ARRAY_SIZE(wide_message));
   wide_message[ARRAY_SIZE(wide_message) - 1] = L'\0';
   yttrium_trace_write_verbose_string(wide_message);
}

void
yttrium_trace_screen_create(uint32_t supports_3d,
                            uint32_t has_resource_blob,
                            uint32_t has_host_visible,
                            uint32_t scanout_present,
                            uint32_t scanout_dry_run,
                            uint32_t mute_kmd_present,
                            uint32_t no_present,
                            uint32_t no_primary_present)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "ScreenCreate",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(supports_3d, "supports_3d"),
      TraceLoggingUInt32(has_resource_blob, "has_resource_blob"),
      TraceLoggingUInt32(has_host_visible, "has_host_visible"),
      TraceLoggingUInt32(scanout_present, "scanout_present"),
      TraceLoggingUInt32(scanout_dry_run, "scanout_dry_run"),
      TraceLoggingUInt32(mute_kmd_present, "mute_kmd_present"),
      TraceLoggingUInt32(no_present, "no_present"),
      TraceLoggingUInt32(no_primary_present, "no_primary_present"));
}

void
yttrium_trace_present(uint64_t allocation,
                      uint32_t resource_id,
                      uint64_t memory_id,
                      uint32_t display,
                      uint32_t primary,
                      uint32_t classic,
                      const void *context_private,
                      const void *window,
                      uint32_t boxes,
                      uint32_t scanout_present,
                      uint32_t scanout_dry_run,
                      uint32_t no_present,
                      uint32_t no_primary_present)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "Present",
      TraceLoggingLevel(5),
      TraceLoggingUInt64(allocation, "allocation"),
      TraceLoggingUInt32(resource_id, "resource_id"),
      TraceLoggingUInt64(memory_id, "memory_id"),
      TraceLoggingUInt32(display, "display"),
      TraceLoggingUInt32(primary, "primary"),
      TraceLoggingUInt32(classic, "classic"),
      TraceLoggingUInt64((uint64_t)(uintptr_t)context_private,
                         "context_private"),
      TraceLoggingUInt64((uint64_t)(uintptr_t)window, "window"),
      TraceLoggingUInt32(boxes, "boxes"),
      TraceLoggingUInt32(scanout_present, "scanout_present"),
      TraceLoggingUInt32(scanout_dry_run, "scanout_dry_run"),
      TraceLoggingUInt32(no_present, "no_present"),
      TraceLoggingUInt32(no_primary_present, "no_primary_present"));
}

void
yttrium_trace_present_state(uint16_t event_id,
                            uint64_t allocation,
                            uint32_t resource_id,
                            uint32_t a,
                            uint32_t b,
                            uint32_t c)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "PresentState",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(event_id, "event_id"),
      TraceLoggingUInt64(allocation, "allocation"),
      TraceLoggingUInt32(resource_id, "resource_id"),
      TraceLoggingUInt32(a, "a"),
      TraceLoggingUInt32(b, "b"),
      TraceLoggingUInt32(c, "c"));
}

void
yttrium_trace_present_submit(uint64_t allocation,
                             uint32_t resource_id,
                             const void *context_private,
                             long status,
                             uint64_t submitted_context,
                             uint64_t dxgi_context,
                             uint64_t dxgi_window,
                             uint64_t dxgi_source,
                             uint64_t dxgi_destination,
                             uint32_t dxgi_flags,
                             uint32_t dxgi_flip_interval,
                             uint32_t dxgi_present_count)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "PresentSubmit",
      TraceLoggingLevel(5),
      TraceLoggingUInt64(allocation, "allocation"),
      TraceLoggingUInt32(resource_id, "resource_id"),
      TraceLoggingUInt64((uint64_t)(uintptr_t)context_private,
                         "context_private"),
      TraceLoggingUInt32((uint32_t)status, "status"),
      TraceLoggingUInt64(submitted_context, "submitted_context"),
      TraceLoggingUInt64(dxgi_context, "dxgi_context"),
      TraceLoggingUInt64(dxgi_window, "dxgi_window"),
      TraceLoggingUInt64(dxgi_source, "dxgi_source"),
      TraceLoggingUInt64(dxgi_destination, "dxgi_destination"),
      TraceLoggingUInt32(dxgi_flags, "dxgi_flags"),
      TraceLoggingUInt32(dxgi_flip_interval, "dxgi_flip_interval"),
      TraceLoggingUInt32(dxgi_present_count, "dxgi_present_count"));

   if (yttrium_trace_verbose_etw_text_enabled()) {
      wchar_t message[512];
      swprintf(message, ARRAY_SIZE(message),
               L"yttrium present_submit allocation=0x%llx res=%u context=0x%llx submitted_context=0x%llx status=0x%lx dxgi_context=0x%llx hwnd=0x%llx source=0x%llx destination=0x%llx flags=0x%x flip_interval=%u present_count=%u",
               allocation, resource_id, (uint64_t)(uintptr_t)context_private,
               submitted_context, (unsigned long)status, dxgi_context,
               dxgi_window, dxgi_source, dxgi_destination, dxgi_flags,
               dxgi_flip_interval, dxgi_present_count);
      yttrium_trace_write_verbose_string(message);
   }
}

void
yttrium_trace_scanout_blob(uint64_t allocation,
                           uint32_t resource_id,
                           uint32_t scanout_id,
                           uint32_t width,
                           uint32_t height,
                           uint32_t x,
                           uint32_t y,
                           uint32_t format,
                           uint32_t stride,
                           uint32_t offset,
                           uint64_t size,
                           uint32_t classic,
                           long status)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "SetScanoutBlob",
      TraceLoggingLevel(5),
      TraceLoggingUInt64(allocation, "allocation"),
      TraceLoggingUInt32(resource_id, "resource_id"),
      TraceLoggingUInt32(scanout_id, "scanout_id"),
      TraceLoggingUInt32(width, "width"),
      TraceLoggingUInt32(height, "height"),
      TraceLoggingUInt32(x, "x"),
      TraceLoggingUInt32(y, "y"),
      TraceLoggingUInt32(format, "format"),
      TraceLoggingUInt32(stride, "stride"),
      TraceLoggingUInt32(offset, "offset"),
      TraceLoggingUInt64(size, "size"),
      TraceLoggingUInt32(classic, "classic"),
      TraceLoggingUInt32((uint32_t)status, "status"));
}

void
yttrium_trace_resource_copy(uint32_t path,
                            uint64_t src_allocation,
                            uint32_t src_resource_id,
                            uint64_t dst_allocation,
                            uint32_t dst_resource_id,
                            uint32_t src_x,
                            uint32_t src_y,
                            uint32_t src_layer,
                            uint32_t dst_x,
                            uint32_t dst_y,
                            uint32_t dst_layer,
                            uint32_t width,
                            uint32_t height,
                            uint32_t src_stride,
                            uint32_t dst_stride)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "ResourceCopy",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(path, "path"),
      TraceLoggingUInt64(src_allocation, "src_allocation"),
      TraceLoggingUInt32(src_resource_id, "src_resource_id"),
      TraceLoggingUInt64(dst_allocation, "dst_allocation"),
      TraceLoggingUInt32(dst_resource_id, "dst_resource_id"),
      TraceLoggingUInt32(src_x, "src_x"),
      TraceLoggingUInt32(src_y, "src_y"),
      TraceLoggingUInt32(src_layer, "src_layer"),
      TraceLoggingUInt32(dst_x, "dst_x"),
      TraceLoggingUInt32(dst_y, "dst_y"),
      TraceLoggingUInt32(dst_layer, "dst_layer"),
      TraceLoggingUInt32(width, "width"),
      TraceLoggingUInt32(height, "height"),
      TraceLoggingUInt32(src_stride, "src_stride"),
      TraceLoggingUInt32(dst_stride, "dst_stride"));
}

void
yttrium_trace_resource_copy_unsupported(uint64_t src_allocation,
                                        uint32_t src_resource_id,
                                        uint64_t dst_allocation,
                                        uint32_t dst_resource_id,
                                        int32_t width,
                                        int32_t height)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "ResourceCopyUnsupported",
      TraceLoggingLevel(5),
      TraceLoggingUInt64(src_allocation, "src_allocation"),
      TraceLoggingUInt32(src_resource_id, "src_resource_id"),
      TraceLoggingUInt64(dst_allocation, "dst_allocation"),
      TraceLoggingUInt32(dst_resource_id, "dst_resource_id"),
      TraceLoggingInt32(width, "width"),
      TraceLoggingInt32(height, "height"));
}

void
yttrium_trace_draw_vbo(uint32_t mode,
                       uint32_t num_draws,
                       uint32_t count,
                       uint32_t instances,
                       uint32_t start_instance,
                       const void *cbuf,
                       uint32_t have_vertices,
                       uint32_t vertex_count,
                       uint32_t vs_inputs,
                       uint32_t vs_outputs,
                       uint32_t fs_inputs,
                       uint32_t fs_outputs,
                       uint32_t fs_srvs,
                       uint32_t fs_samplers)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "DrawVbo",
      TraceLoggingLevel(5),
      /*
       * The provider is captured system-wide and DWM draws through this driver
       * too, so a draw count without a pid cannot be filtered to the workload -
       * it silently mixes processes while frame counts are pid-filtered, which
       * biases every per-draw figure derived from it.
       */
      TraceLoggingUInt32(GetCurrentProcessId(), "pid"),
      TraceLoggingUInt32(mode, "mode"),
      TraceLoggingUInt32(num_draws, "num_draws"),
      TraceLoggingUInt32(count, "count"),
      TraceLoggingUInt32(instances, "instances"),
      TraceLoggingUInt32(start_instance, "start_instance"),
      TraceLoggingUInt64((uint64_t)(uintptr_t)cbuf, "cbuf"),
      TraceLoggingUInt32(have_vertices, "have_vertices"),
      TraceLoggingUInt32(vertex_count, "vertex_count"),
      TraceLoggingUInt32(vs_inputs, "vs_inputs"),
      TraceLoggingUInt32(vs_outputs, "vs_outputs"),
      TraceLoggingUInt32(fs_inputs, "fs_inputs"),
      TraceLoggingUInt32(fs_outputs, "fs_outputs"),
      TraceLoggingUInt32(fs_srvs, "fs_srvs"),
      TraceLoggingUInt32(fs_samplers, "fs_samplers"));
}

void
yttrium_trace_draw_skip(uint32_t reason,
                        uint64_t allocation,
                        uint32_t resource_id,
                        uint32_t a,
                        uint32_t b,
                        uint32_t c)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "DrawSkip",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(reason, "reason"),
      TraceLoggingUInt64(allocation, "allocation"),
      TraceLoggingUInt32(resource_id, "resource_id"),
      TraceLoggingUInt32(a, "a"),
      TraceLoggingUInt32(b, "b"),
      TraceLoggingUInt32(c, "c"));
}

void
yttrium_trace_textured_sampler(uint32_t slot,
                               const void *texture,
                               uint64_t allocation,
                               uint32_t resource_id,
                               uint64_t image_id,
                               uint32_t width,
                               uint32_t height,
                               uint32_t format)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "TexturedSampler",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(slot, "slot"),
      TraceLoggingUInt64((uint64_t)(uintptr_t)texture, "texture"),
      TraceLoggingUInt64(allocation, "allocation"),
      TraceLoggingUInt32(resource_id, "resource_id"),
      TraceLoggingUInt64(image_id, "image_id"),
      TraceLoggingUInt32(width, "width"),
      TraceLoggingUInt32(height, "height"),
      TraceLoggingUInt32(format, "format"));
}

void
yttrium_trace_venus_draw(uint32_t event_id,
                         uint32_t dst_resource_id,
                         uint64_t dst_image_id,
                         uint32_t src_resource_id,
                         uint64_t src_image_id,
                         uint64_t pipeline_id,
                         uint32_t vertex_count,
                         float viewport_x,
                         float viewport_y,
                         float viewport_width,
                         float viewport_height,
                         int32_t scissor_x,
                         int32_t scissor_y,
                         uint32_t scissor_width,
                         uint32_t scissor_height)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "VenusDraw",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(event_id, "event_id"),
      TraceLoggingUInt32(dst_resource_id, "dst_resource_id"),
      TraceLoggingUInt64(dst_image_id, "dst_image_id"),
      TraceLoggingUInt32(src_resource_id, "src_resource_id"),
      TraceLoggingUInt64(src_image_id, "src_image_id"),
      TraceLoggingUInt64(pipeline_id, "pipeline_id"),
      TraceLoggingUInt32(vertex_count, "vertex_count"),
      TraceLoggingFloat32(viewport_x, "viewport_x"),
      TraceLoggingFloat32(viewport_y, "viewport_y"),
      TraceLoggingFloat32(viewport_width, "viewport_width"),
      TraceLoggingFloat32(viewport_height, "viewport_height"),
      TraceLoggingInt32(scissor_x, "scissor_x"),
      TraceLoggingInt32(scissor_y, "scissor_y"),
      TraceLoggingUInt32(scissor_width, "scissor_width"),
      TraceLoggingUInt32(scissor_height, "scissor_height"));
}

void
yttrium_trace_native_draw_batch_decision(uint32_t candidate,
                                         uint32_t reject_mask,
                                         uint32_t native_draw_batch_enabled,
                                         uint32_t cpu_vertex_batch_allowed,
                                         uint32_t cpu_vertex_batch_enabled,
                                         uint32_t has_cpu_vertex_upload,
                                         uint32_t has_sampled_descriptor,
                                         uint32_t has_ubo_descriptor,
                                         uint32_t push_descriptor_batch_enabled,
                                         uint32_t push_descriptors_available,
                                         uint32_t pipeline_has_push_layout,
                                         uint32_t pipeline_has_push_pipeline,
                                         uint32_t use_push_descriptors,
                                         uint32_t mode,
                                         uint32_t vertex_count,
                                         uint32_t index_count,
                                         uint64_t pipeline_id)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "NativeDrawBatchDecision",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(candidate, "candidate"),
      TraceLoggingUInt32(reject_mask, "reject_mask"),
      TraceLoggingUInt32(native_draw_batch_enabled,
                         "native_draw_batch_enabled"),
      TraceLoggingUInt32(cpu_vertex_batch_allowed,
                         "cpu_vertex_batch_allowed"),
      TraceLoggingUInt32(cpu_vertex_batch_enabled,
                         "cpu_vertex_batch_enabled"),
      TraceLoggingUInt32(has_cpu_vertex_upload, "has_cpu_vertex_upload"),
      TraceLoggingUInt32(has_sampled_descriptor, "has_sampled_descriptor"),
      TraceLoggingUInt32(has_ubo_descriptor, "has_ubo_descriptor"),
      TraceLoggingUInt32(push_descriptor_batch_enabled,
                         "push_descriptor_batch_enabled"),
      TraceLoggingUInt32(push_descriptors_available,
                         "push_descriptors_available"),
      TraceLoggingUInt32(pipeline_has_push_layout,
                         "pipeline_has_push_layout"),
      TraceLoggingUInt32(pipeline_has_push_pipeline,
                         "pipeline_has_push_pipeline"),
      TraceLoggingUInt32(use_push_descriptors, "use_push_descriptors"),
      TraceLoggingUInt32(mode, "mode"),
      TraceLoggingUInt32(vertex_count, "vertex_count"),
      TraceLoggingUInt32(index_count, "index_count"),
      TraceLoggingUInt64(pipeline_id, "pipeline_id"));
}

void
yttrium_trace_cmd_batch_submit(uint32_t async_submit,
                               uint32_t native_draw_only,
                               uint32_t op_count,
                               uint32_t resource_count,
                               uint32_t pipeline_count,
                               uint32_t transient_count,
                               uint32_t allocated_batch_count,
                               uint32_t live_batch_count,
                               uint32_t peak_live_batch_count,
                               uint32_t pending_submit_count,
                               uint32_t group_submit_size,
                               uint64_t batch_pool_bytes,
                               uint64_t ubo_arena_bytes,
                               uint64_t peak_ubo_arena_bytes,
                               uint64_t draw_backing_pool_bytes,
                               uint64_t peak_draw_backing_pool_bytes,
                               const char *label)
{
   if (!yttrium_trace_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "CmdBatchSubmit",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(async_submit, "async_submit"),
      TraceLoggingUInt32(native_draw_only, "native_draw_only"),
      TraceLoggingUInt32(op_count, "op_count"),
      TraceLoggingUInt32(resource_count, "resource_count"),
      TraceLoggingUInt32(pipeline_count, "pipeline_count"),
      TraceLoggingUInt32(transient_count, "transient_count"),
      TraceLoggingUInt32(allocated_batch_count, "allocated_batch_count"),
      TraceLoggingUInt32(live_batch_count, "live_batch_count"),
      TraceLoggingUInt32(peak_live_batch_count, "peak_live_batch_count"),
      TraceLoggingUInt32(pending_submit_count, "pending_submit_count"),
      TraceLoggingUInt32(group_submit_size, "group_submit_size"),
      TraceLoggingUInt64(batch_pool_bytes, "batch_pool_bytes"),
      TraceLoggingUInt64(ubo_arena_bytes, "ubo_arena_bytes"),
      TraceLoggingUInt64(peak_ubo_arena_bytes, "peak_ubo_arena_bytes"),
      TraceLoggingUInt64(draw_backing_pool_bytes, "draw_backing_pool_bytes"),
      TraceLoggingUInt64(peak_draw_backing_pool_bytes,
                         "peak_draw_backing_pool_bytes"),
      TraceLoggingString(label ? label : "", "label"));
}

void
yttrium_trace_deferred_draw_state(
   const struct yttrium_trace_deferred_draw_stats *stats)
{
   /* These batch totals are also useful with the low-volume wait profile. */
   if (!yttrium_trace_sync_wait_is_enabled())
      return;

   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "DeferredDrawState",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(stats->draws, "draws"),
      TraceLoggingUInt32(stats->render_passes, "render_passes"),
      TraceLoggingUInt32(stats->pipeline_binds, "pipeline_binds"),
      TraceLoggingUInt32(stats->pipeline_binds_skipped, "pipeline_binds_skipped"),
      TraceLoggingUInt32(stats->descriptor_binds, "descriptor_binds"),
      TraceLoggingUInt32(stats->descriptor_binds_skipped, "descriptor_binds_skipped"),
      TraceLoggingUInt32(stats->push_calls, "push_calls"),
      TraceLoggingUInt32(stats->push_writes, "push_writes"),
      TraceLoggingUInt32(stats->repeated_push_layouts, "repeated_push_layouts"),
      TraceLoggingUInt32(stats->requested_push_calls, "requested_push_calls"),
      TraceLoggingUInt32(stats->requested_push_writes, "requested_push_writes"),
      TraceLoggingUInt32(stats->push_comparisons, "push_comparisons"),
      TraceLoggingUInt32(stats->push_calls_skipped, "push_calls_skipped"),
      TraceLoggingUInt32(stats->push_writes_skipped, "push_writes_skipped"));
}

void
yttrium_trace_venus_upload(uint32_t kind,
                           uint32_t flags,
                           uint64_t bytes,
                           uint64_t src_object_id,
                           uint64_t dst_object_id,
                           uint32_t src_resource_id,
                           uint32_t dst_resource_id,
                           uint32_t width,
                           uint32_t height,
                           uint32_t depth,
                           uint32_t row_stride,
                           uint32_t layer_stride)
{
   if (!yttrium_trace_is_enabled())
      return;

   const uint32_t pid = GetCurrentProcessId();
   const uint32_t tid = GetCurrentThreadId();
   TraceLoggingWrite(
      yttrium_tracelogging_provider,
      "VenusUpload",
      TraceLoggingLevel(5),
      TraceLoggingUInt32(kind, "kind"),
      TraceLoggingUInt32(flags, "flags"),
      TraceLoggingUInt64(bytes, "bytes"),
      TraceLoggingUInt64(src_object_id, "src_object_id"),
      TraceLoggingUInt64(dst_object_id, "dst_object_id"),
      TraceLoggingUInt32(src_resource_id, "src_resource_id"),
      TraceLoggingUInt32(dst_resource_id, "dst_resource_id"),
      TraceLoggingUInt32(width, "width"),
      TraceLoggingUInt32(height, "height"),
      TraceLoggingUInt32(depth, "depth"),
      TraceLoggingUInt32(row_stride, "row_stride"),
      TraceLoggingUInt32(layer_stride, "layer_stride"),
      TraceLoggingUInt32(pid, "pid"),
      TraceLoggingUInt32(tid, "tid"));
}
