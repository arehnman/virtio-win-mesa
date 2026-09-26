/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#ifndef YTTRIUM_VENUS2_RING_H
#define YTTRIUM_VENUS2_RING_H

#include "yttrium_venus2_private.h"

void
yttrium_venus2_ring_lock(void);

void
yttrium_venus2_ring_unlock(void);

/* Stack-local encoder sink for async vkCmd* calls inside a ring transaction.
 * It changes transport write granularity, not GPU submission boundaries.
 */
struct yttrium_venus_command_stream {
   struct vn_ring vn_ring;
   struct yttrium_venus *venus;
   uint32_t size;
   uint32_t command_count;
   bool failed;
   uint8_t data[16384];
};

void
yttrium_venus2_command_stream_init(struct yttrium_venus_command_stream *stream,
                                  struct yttrium_venus *venus);

bool
yttrium_venus2_command_stream_flush(struct yttrium_venus_command_stream *stream);

bool
yttrium_venus2_ring_transaction_begin(struct yttrium_venus *venus);

bool
yttrium_venus2_ring_transaction_end(struct yttrium_venus *venus,
                                     const char *label);

bool
yttrium_venus2_vn_ring_submit_command_locked(
   struct vn_ring *vn_ring,
   struct vn_ring_submit_command *submit);

void
yttrium_venus2_vn_ring_warn_reply_request_with_invalid_seqno(
   struct vn_ring *vn_ring,
   struct vn_ring_submit_command *submit);

bool
yttrium_venus_ring_create(struct yttrium_venus *venus);

bool
yttrium_venus_ring_publish_for_submit(struct yttrium_venus *venus, uint32_t *seqno);

bool
yttrium_venus_submit_gpu(struct yttrium_venus *venus, const void *data, size_t size);

void
yttrium_venus_ring_forget_at_device_teardown(struct yttrium_venus *venus);

bool
yttrium_venus_ring_ready(const struct yttrium_venus_ring *ring);

bool
yttrium_venus_ring_seqno_complete(struct yttrium_venus *venus,
                                  const struct yttrium_venus_ring_dependency *dependency);

bool
yttrium_venus_ring_probe(struct yttrium_venus *venus,
                         const char *failure_reason);

bool
yttrium_venus_ring_wait_for_seqno(struct yttrium_venus *venus,
                                  const struct yttrium_venus_ring_dependency *dependency,
                                  const char *label);

bool
yttrium_venus_ring_flush_notify(struct yttrium_venus *venus,
                                const char *label,
                                bool blocking);

bool
yttrium_venus_drain_ring(struct yttrium_venus *venus, const char *label);

#endif /* YTTRIUM_VENUS2_RING_H */
