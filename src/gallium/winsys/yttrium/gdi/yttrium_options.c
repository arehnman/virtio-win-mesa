/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include "yttrium_gdi_public.h"
#include "yttrium_options.h"
#include "yttrium_trace.h"

#define YTTRIUM_CONFIG_MAX_ENTRIES 128
#define YTTRIUM_CONFIG_KEY_SIZE 96
#define YTTRIUM_CONFIG_VALUE_SIZE 512
#define YTTRIUM_USER_LOG_DEFAULT_DIR "C:\\ProgramData\\Yttrium"
#define YTTRIUM_USER_LOG_DEFAULT_PATH \
   YTTRIUM_USER_LOG_DEFAULT_DIR "\\yttrium-user.log"

struct yttrium_config_entry {
   char key[YTTRIUM_CONFIG_KEY_SIZE];
   char value[YTTRIUM_CONFIG_VALUE_SIZE];
};

static struct yttrium_config_entry yttrium_config_entries[YTTRIUM_CONFIG_MAX_ENTRIES];
static unsigned yttrium_config_entry_count;
static bool yttrium_config_loaded;
static bool yttrium_config_found;
static char yttrium_config_path[MAX_PATH];
static volatile LONG yttrium_user_log_dir_ready;

static char *
yttrium_trim_ascii(char *s)
{
   char *end;

   if (!s)
      return s;

   while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n')
      s++;

   end = s + strlen(s);
   while (end > s &&
          (end[-1] == ' ' || end[-1] == '\t' ||
           end[-1] == '\r' || end[-1] == '\n')) {
      *--end = '\0';
   }

   return s;
}

bool
yttrium_gdi_ascii_iequals(const char *a, const char *b)
{
   if (!a || !b)
      return false;

   while (*a && *b) {
      char ca = *a++;
      char cb = *b++;

      if (ca >= 'A' && ca <= 'Z')
         ca = (char)(ca - 'A' + 'a');
      if (cb >= 'A' && cb <= 'Z')
         cb = (char)(cb - 'A' + 'a');
      if (ca != cb)
         return false;
   }

   return *a == *b;
}

static bool
yttrium_config_load_path(const char *path)
{
   FILE *f;
   char line[1024];

   if (!path || !path[0])
      return false;

   f = fopen(path, "rb");
   if (!f)
      return false;

   yttrium_config_entry_count = 0;
   while (fgets(line, sizeof(line), f) &&
          yttrium_config_entry_count < YTTRIUM_CONFIG_MAX_ENTRIES) {
      char *trimmed = yttrium_trim_ascii(line);
      char *equals;
      char *key;
      char *value;
      size_t len;

      if (!trimmed[0] || trimmed[0] == '#' || trimmed[0] == ';' ||
          trimmed[0] == '[')
         continue;

      equals = strchr(trimmed, '=');
      if (!equals)
         continue;

      *equals = '\0';
      key = yttrium_trim_ascii(trimmed);
      value = yttrium_trim_ascii(equals + 1);
      if (!key[0])
         continue;

      len = strlen(value);
      if (len >= 2 &&
          ((value[0] == '"' && value[len - 1] == '"') ||
           (value[0] == '\'' && value[len - 1] == '\''))) {
         value[len - 1] = '\0';
         value++;
      }

      snprintf(yttrium_config_entries[yttrium_config_entry_count].key,
               sizeof(yttrium_config_entries[yttrium_config_entry_count].key),
               "%s", key);
      snprintf(yttrium_config_entries[yttrium_config_entry_count].value,
               sizeof(yttrium_config_entries[yttrium_config_entry_count].value),
               "%s", value);
      yttrium_config_entry_count++;
   }

   fclose(f);
   snprintf(yttrium_config_path, sizeof(yttrium_config_path), "%s", path);
   yttrium_config_found = true;
   return true;
}

static void
yttrium_config_load_once(void)
{
   static const char config_path[] = "C:/ProgramData/Yttrium/yttrium.ini";

   if (yttrium_config_loaded)
      return;

   yttrium_config_loaded = true;
   yttrium_config_load_path(config_path);
}

static const char *
yttrium_config_find(const char *name)
{
   const char *short_name = NULL;
   static const char prefix[] = "D3D10UMD_YTTRIUM_";

   yttrium_config_load_once();

   if (!name)
      return NULL;

   if (strncmp(name, prefix, strlen(prefix)) == 0)
      short_name = name + strlen(prefix);

   for (unsigned i = 0; i < yttrium_config_entry_count; i++) {
      const char *key = yttrium_config_entries[i].key;
      if (yttrium_gdi_ascii_iequals(key, name) ||
          (short_name && yttrium_gdi_ascii_iequals(key, short_name))) {
         return yttrium_config_entries[i].value;
      }
   }

   return NULL;
}

const char *
yttrium_gdi_debug_get_option(const char *name, const char *dfault)
{
   const char *env;
   const char *config;

   if (!name)
      return dfault;

   env = getenv(name);
   if (env)
      return env;

   config = yttrium_config_find(name);
   return config ? config : dfault;
}

bool
yttrium_gdi_debug_get_bool_option(const char *name, bool dfault)
{
   const char *value = yttrium_gdi_debug_get_option(name, NULL);

   if (!value)
      return dfault;

   if (yttrium_gdi_ascii_iequals(value, "true") ||
       yttrium_gdi_ascii_iequals(value, "yes") ||
       yttrium_gdi_ascii_iequals(value, "on"))
      return true;

   if (yttrium_gdi_ascii_iequals(value, "false") ||
       yttrium_gdi_ascii_iequals(value, "no") ||
       yttrium_gdi_ascii_iequals(value, "off"))
      return false;

   return strtol(value, NULL, 0) != 0;
}

bool
yttrium_gdi_static_ubo_sampled_cache_enabled(void)
{
   static volatile LONG cached = -1;
   LONG enabled = cached;

   if (enabled < 0) {
      const LONG resolved = yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_STATIC_UBO_SAMPLED_CACHE", true) ? 1 : 0;

      enabled = InterlockedCompareExchange(&cached, resolved, -1);
      if (enabled < 0)
         enabled = resolved;
   }

   return enabled != 0;
}

bool
yttrium_present_timeline_sync_enabled(void)
{
   static volatile LONG cached = -1;
   LONG enabled = cached;

   if (enabled < 0) {
      /* Used when scheduled Present is disabled. Zero restores the legacy
       * fullscreen path through pfnPresentCb; windowed/DWM uses that path. */
      const LONG resolved = yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_PRESENT_TIMELINE_SYNC", true) ? 1 : 0;

      enabled = InterlockedCompareExchange(&cached, resolved, -1);
      if (enabled < 0)
         enabled = resolved;
   }

   return enabled != 0;
}

bool
yttrium_present_issuance_wait_enabled(void)
{
   static volatile LONG cached = -1;
   LONG enabled = cached;

   if (enabled < 0) {
      const LONG resolved = yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_PRESENT_ISSUANCE_WAIT", true) ? 1 : 0;

      const LONG previous =
         InterlockedCompareExchange(&cached, resolved, -1);
      if (previous < 0) {
         enabled = resolved;
         if (!enabled) {
            YTTRIUM_WARN("yttrium: WARNING: runtime Present issuance ordering disabled owner=yttrium_present component=present-publication reason=process-option action=publish-without-worker-issuance-wait variable=D3D10UMD_YTTRIUM_PRESENT_ISSUANCE_WAIT risk=out-of-order-frames-and-cursor\n");
         }
      } else {
         enabled = previous;
      }
   }

   return enabled != 0;
}

bool
yttrium_gdi_debug_has_option(const char *name)
{
   return yttrium_gdi_debug_get_option(name, NULL) != NULL;
}

int64_t
yttrium_gdi_debug_get_num_option(const char *name, int64_t dfault)
{
   const char *value = yttrium_gdi_debug_get_option(name, NULL);
   char *end = NULL;
   int64_t number;

   if (!value)
      return dfault;

   number = strtoll(value, &end, 0);
   return end && end != value ? number : dfault;
}

void
yttrium_gdi_debug_get_config_status(bool *loaded,
                                    bool *found,
                                    const char **path,
                                    unsigned *entry_count)
{
   yttrium_config_load_once();

   if (loaded)
      *loaded = yttrium_config_loaded;
   if (found)
      *found = yttrium_config_found;
   if (path)
      *path = yttrium_config_path[0] ? yttrium_config_path : NULL;
   if (entry_count)
      *entry_count = yttrium_config_entry_count;
}

void
yttrium_gdi_trace_debugf(const char *format, ...)
{
   char message[1024];
   va_list ap;

   if (!format || !yttrium_trace_is_enabled())
      return;

   va_start(ap, format);
   vsnprintf(message, sizeof(message), format, ap);
   va_end(ap);

   message[sizeof(message) - 1] = '\0';
   yttrium_trace_debug_stringf("%s", message);
}

void
yttrium_gdi_trace_warnf(const char *format, ...)
{
   char message[1024];
   va_list ap;

   if (!format)
      return;

   va_start(ap, format);
   vsnprintf(message, sizeof(message), format, ap);
   va_end(ap);

   message[sizeof(message) - 1] = '\0';
   yttrium_trace_logf(YTTRIUM_TRACE_WARNING, "%s", message);
}

void
yttrium_gdi_trace_errorf(const char *format, ...)
{
   char message[1024];
   va_list ap;

   if (!format)
      return;

   va_start(ap, format);
   vsnprintf(message, sizeof(message), format, ap);
   va_end(ap);

   message[sizeof(message) - 1] = '\0';
   yttrium_trace_logf(YTTRIUM_TRACE_ERROR, "%s", message);
}

bool
yttrium_gdi_trace_is_enabled(void)
{
   return yttrium_trace_is_enabled();
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
   yttrium_trace_resource_event(kind, kind_name, handle, object,
                                pipe_resource, refcount, a, b, c);
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
   yttrium_trace_dxgi_present(stage, src_resource, dst_resource,
                              src_allocation, dst_allocation, dxgi_context,
                              hwnd, source, destination, src_subresource,
                              dst_subresource, flags, flip_interval,
                              present_count, gdi_readback,
                              readback_presented);
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
   yttrium_trace_present_callback(stage, status, src_allocation,
                                  dst_allocation, context, dxgi_context,
                                  private_size, optimize_for_composition,
                                  broadcast_count);
}

void
yttrium_gdi_user_logf(const char *format, ...)
{
   char message[1024];
   char record[1400];
   const char *path;
   SYSTEMTIME time;
   HANDLE file;
   DWORD ignored;
   size_t len;
   va_list ap;

   if (!format)
      return;

   path = yttrium_gdi_debug_get_option("D3D10UMD_YTTRIUM_USER_LOG_FILE",
                                       YTTRIUM_USER_LOG_DEFAULT_PATH);
   if (!path || !path[0])
      return;

   va_start(ap, format);
   vsnprintf(message, sizeof(message), format, ap);
   va_end(ap);
   message[sizeof(message) - 1] = '\0';

   if (InterlockedCompareExchange(&yttrium_user_log_dir_ready, 1, 0) == 0) {
      CreateDirectoryA("C:\\ProgramData\\Yttrium", NULL);
      CreateDirectoryA(YTTRIUM_USER_LOG_DEFAULT_DIR, NULL);
   }

   GetLocalTime(&time);
   snprintf(record, sizeof(record),
            "%04u-%02u-%02u %02u:%02u:%02u.%03u pid=%lu tid=%lu %s",
            (unsigned)time.wYear, (unsigned)time.wMonth,
            (unsigned)time.wDay, (unsigned)time.wHour,
            (unsigned)time.wMinute, (unsigned)time.wSecond,
            (unsigned)time.wMilliseconds,
            (unsigned long)GetCurrentProcessId(),
            (unsigned long)GetCurrentThreadId(),
            message);
   record[sizeof(record) - 1] = '\0';

   len = strlen(record);
   if (len && record[len - 1] != '\n' && len + 1 < sizeof(record)) {
      record[len++] = '\n';
      record[len] = '\0';
   }

   file = CreateFileA(path,
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
