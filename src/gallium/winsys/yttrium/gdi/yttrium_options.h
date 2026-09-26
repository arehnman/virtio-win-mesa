/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#ifndef YTTRIUM_OPTIONS_H
#define YTTRIUM_OPTIONS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool
yttrium_gdi_ascii_iequals(const char *a, const char *b);

const char *
yttrium_gdi_debug_get_option(const char *name, const char *dfault);

bool
yttrium_gdi_debug_get_bool_option(const char *name, bool dfault);

bool
yttrium_gdi_static_ubo_sampled_cache_enabled(void);

/*
 * When scheduled Present is disabled, this defaults on: application-owned
 * fullscreen scanouts are published by the ordered worker through the scanout
 * escape after GPU completion. Set
 * D3D10UMD_YTTRIUM_PRESENT_TIMELINE_SYNC=0 to force those Presents through the
 * legacy pfnPresentCb path.  Windowed/DWM Presents always use pfnPresentCb.
 */
bool
yttrium_present_timeline_sync_enabled(void);

/*
 * Enabled by default: ordinary runtime Presents enter pfnPresentCb only after
 * the ordered worker has issued their asynchronous render publication.  Set
 * D3D10UMD_YTTRIUM_PRESENT_ISSUANCE_WAIT=0 only for profiling; doing so can
 * publish rotating allocations out of order and resurface older frames or
 * cursor positions.
 */
bool
yttrium_present_issuance_wait_enabled(void);

bool
yttrium_gdi_debug_has_option(const char *name);

int64_t
yttrium_gdi_debug_get_num_option(const char *name, int64_t dfault);

void
yttrium_gdi_debug_get_config_status(bool *loaded,
                                    bool *found,
                                    const char **path,
                                    unsigned *entry_count);

#ifdef __cplusplus
}
#endif

#endif /* YTTRIUM_OPTIONS_H */
