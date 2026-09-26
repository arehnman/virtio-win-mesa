/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>
#include <winternl.h>

#include <d3dkmthk.h>

#include "gdikmt/gdikmt.h"
#include "compiler/shader_enums.h"
#include "pipe/p_defines.h"
#include "util/u_debug.h"
#include "util/format/u_format.h"
#include "util/format/u_formats.h"
#include "util/u_inlines.h"
#include "util/u_atomic.h"
#include "util/os_time.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "virtio/wddm/viogpu_wddm_driver.h"

#include "yttrium_trace.h"
#include "yttrium_options.h"
#include "yttrium_venus2.h"
#include "yttrium_venus2_ring.h"
#include "yttrium_pipeline.h"
#include "yttrium_shader.h"

#include "venus-protocol/vn_protocol_driver_buffer.h"
#include "venus-protocol/vn_protocol_driver_buffer_view.h"
#include "venus-protocol/vn_protocol_driver_command_buffer.h"
#include "venus-protocol/vn_protocol_driver_command_pool.h"
#include "venus-protocol/vn_protocol_driver_descriptor_pool.h"
#include "venus-protocol/vn_protocol_driver_descriptor_set.h"
#include "venus-protocol/vn_protocol_driver_descriptor_set_layout.h"
#include "venus-protocol/vn_protocol_driver_device.h"
#include "venus-protocol/vn_protocol_driver_device_memory.h"
#include "venus-protocol/vn_protocol_driver_fence.h"
#include "venus-protocol/vn_protocol_driver_framebuffer.h"
#include "venus-protocol/vn_protocol_driver_image.h"
#include "venus-protocol/vn_protocol_driver_image_view.h"
#include "venus-protocol/vn_protocol_driver_instance.h"
#include "venus-protocol/vn_protocol_driver_pipeline.h"
#include "venus-protocol/vn_protocol_driver_pipeline_layout.h"
#include "venus-protocol/vn_protocol_driver_queue.h"
#include "venus-protocol/vn_protocol_driver_render_pass.h"
#include "venus-protocol/vn_protocol_driver_sampler.h"
#include "venus-protocol/vn_protocol_driver_shader_module.h"
#include "venus-protocol/vn_protocol_driver_transport.h"

#include "yttrium_venus2_private.h"

#define YTTRIUM_VENUS_UBO_ARENA_PRE_SIZE_DEFAULT \
   (4ull * 1024ull * 1024ull)
#define YTTRIUM_VENUS_VERTEX_ARENA_PRE_SIZE_DEFAULT \
   (1ull * 1024ull * 1024ull)
#define YTTRIUM_VENUS_INDEX_ARENA_PRE_SIZE_DEFAULT \
   (4ull * 1024ull * 1024ull)
#define YTTRIUM_VENUS_UBO_ARENA_TOTAL_LIMIT \
   (256ull * 1024ull * 1024ull)

#define YTTRIUM_VENUS_SUBMIT_OBJECT_OR(_venus, _operation, _object, _handle, \
                                        _failure, ...)                       \
   do {                                                                  \
      struct vn_ring_submit_command submit;                              \
      vn_submit_##_operation(&(_venus)->vn_ring, 0, __VA_ARGS__,         \
                              &submit);                                   \
      if (!yttrium_venus_async_submit_succeeded(                         \
             (_venus), &submit, #_operation, (_object)->id)) {           \
         *(_handle) = VK_NULL_HANDLE;                                    \
         memset((_object), 0, sizeof(*(_object)));                       \
         _failure;                                                       \
      }                                                                  \
   } while (0)

#define YTTRIUM_VENUS_SUBMIT_COMMAND_OR(_venus, _operation, _object_id, \
                                         _failure, ...)                  \
   do {                                                                  \
      struct vn_ring_submit_command submit;                              \
      vn_submit_##_operation(&(_venus)->vn_ring, 0, __VA_ARGS__,         \
                              &submit);                                   \
      if (!yttrium_venus_async_submit_succeeded(                         \
             (_venus), &submit, #_operation, (_object_id))) {            \
         _failure;                                                       \
      }                                                                  \
   } while (0)

static VkDeviceSize
yttrium_venus_pre_size_option(const char *name, VkDeviceSize dfault)
{
   const int64_t value =
      yttrium_gdi_debug_get_num_option(name, (int64_t)dfault);

   return value > 0 ? (VkDeviceSize)value : 0;
}

static bool
yttrium_venus2_sample_count_flag(unsigned sample_count,
                                 VkSampleCountFlagBits *out_samples)
{
   switch (sample_count) {
   case 0:
   case 1:
      *out_samples = VK_SAMPLE_COUNT_1_BIT;
      return true;
   case 2:
      *out_samples = VK_SAMPLE_COUNT_2_BIT;
      return true;
   case 4:
      *out_samples = VK_SAMPLE_COUNT_4_BIT;
      return true;
   case 8:
      *out_samples = VK_SAMPLE_COUNT_8_BIT;
      return true;
   case 16:
      *out_samples = VK_SAMPLE_COUNT_16_BIT;
      return true;
   default:
      *out_samples = VK_SAMPLE_COUNT_1_BIT;
      return false;
   }
}

static VkDeviceSize
yttrium_venus_ubo_arena_pre_size(void)
{
   static int initialized;
   static VkDeviceSize size;

   if (!initialized) {
      size = yttrium_venus_pre_size_option(
         "D3D10UMD_YTTRIUM_UBO_ARENA_PRE_SIZE",
         YTTRIUM_VENUS_UBO_ARENA_PRE_SIZE_DEFAULT);
      initialized = 1;
   }
   return size;
}

static VkDeviceSize
yttrium_venus_vertex_arena_pre_size(void)
{
   static int initialized;
   static VkDeviceSize size;

   if (!initialized) {
      size = yttrium_venus_pre_size_option(
         "D3D10UMD_YTTRIUM_VERTEX_ARENA_PRE_SIZE",
         YTTRIUM_VENUS_VERTEX_ARENA_PRE_SIZE_DEFAULT);
      initialized = 1;
   }
   return size;
}

static VkDeviceSize
yttrium_venus_index_arena_pre_size(void)
{
   static int initialized;
   static VkDeviceSize size;

   if (!initialized) {
      size = yttrium_venus_pre_size_option(
         "D3D10UMD_YTTRIUM_INDEX_ARENA_PRE_SIZE",
         YTTRIUM_VENUS_INDEX_ARENA_PRE_SIZE_DEFAULT);
      initialized = 1;
   }
   return size;
}

static VkDeviceSize
yttrium_venus_pre_sized_allocation(VkDeviceSize requested,
                                   VkDeviceSize pre_size)
{
   return MAX2(requested, pre_size);
}


uint64_t
yttrium_venus_next_id(struct yttrium_venus *venus)
{
   return p_atomic_fetch_add(&venus->next_id, 1);
}

void
yttrium_venus_init_object(struct yttrium_venus *venus,
                          struct yttrium_venus_object *obj)
{
   obj->id = yttrium_venus_next_id(venus);
}

static const uint32_t yttrium_color_passthrough_fs_spv[] = {
   0x07230203, 0x00010000, 0x0008000b, 0x00000014,
   0x00000000, 0x00020011, 0x00000001, 0x0006000b,
   0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001,
   0x0007000f, 0x00000004, 0x00000004, 0x6e69616d,
   0x00000000, 0x00000009, 0x0000000b, 0x00030010,
   0x00000004, 0x00000007, 0x00030003, 0x00000002,
   0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
   0x00000000, 0x00050005, 0x00000009, 0x5f74756f,
   0x6f6c6f63, 0x00000072, 0x00050005, 0x0000000b,
   0x635f6e69, 0x726f6c6f, 0x00000000, 0x00040047,
   0x00000009, 0x0000001e, 0x00000000, 0x00040047,
   0x0000000b, 0x0000001e, 0x00000000, 0x00020013,
   0x00000002, 0x00030021, 0x00000003, 0x00000002,
   0x00030016, 0x00000006, 0x00000020, 0x00040017,
   0x00000007, 0x00000006, 0x00000004, 0x00040020,
   0x00000008, 0x00000003, 0x00000007, 0x0004003b,
   0x00000008, 0x00000009, 0x00000003, 0x00040020,
   0x0000000a, 0x00000001, 0x00000007, 0x0004003b,
   0x0000000a, 0x0000000b, 0x00000001, 0x00040017,
   0x0000000c, 0x00000006, 0x00000003, 0x0004002b,
   0x00000006, 0x0000000f, 0x3f800000, 0x00050036,
   0x00000002, 0x00000004, 0x00000000, 0x00000003,
   0x000200f8, 0x00000005, 0x0004003d, 0x00000007,
   0x0000000d, 0x0000000b, 0x0008004f, 0x0000000c,
   0x0000000e, 0x0000000d, 0x0000000d, 0x00000000,
   0x00000001, 0x00000002, 0x00050051, 0x00000006,
   0x00000010, 0x0000000e, 0x00000000, 0x00050051,
   0x00000006, 0x00000011, 0x0000000e, 0x00000001,
   0x00050051, 0x00000006, 0x00000012, 0x0000000e,
   0x00000002, 0x00070050, 0x00000007, 0x00000013,
   0x00000010, 0x00000011, 0x00000012, 0x0000000f,
   0x0003003e, 0x00000009, 0x00000013, 0x000100fd,
   0x00010038,
};

static const uint32_t yttrium_vertex_input_vs_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000018u,
   0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
   0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
   0x0009000fu, 0x00000000u, 0x00000004u, 0x6e69616du,
   0x00000000u, 0x0000000du, 0x00000011u, 0x00000015u,
   0x00000016u, 0x00030003u, 0x00000002u, 0x000001c2u,
   0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u,
   0x00060005u, 0x0000000bu, 0x505f6c67u, 0x65567265u,
   0x78657472u, 0x00000000u, 0x00060006u, 0x0000000bu,
   0x00000000u, 0x505f6c67u, 0x7469736fu, 0x006e6f69u,
   0x00070006u, 0x0000000bu, 0x00000001u, 0x505f6c67u,
   0x746e696fu, 0x657a6953u, 0x00000000u, 0x00070006u,
   0x0000000bu, 0x00000002u, 0x435f6c67u, 0x4470696cu,
   0x61747369u, 0x0065636eu, 0x00070006u, 0x0000000bu,
   0x00000003u, 0x435f6c67u, 0x446c6c75u, 0x61747369u,
   0x0065636eu, 0x00030005u, 0x0000000du, 0x00000000u,
   0x00050005u, 0x00000011u, 0x705f6e69u, 0x7469736fu,
   0x006e6f69u, 0x00050005u, 0x00000015u, 0x5f74756fu,
   0x6f6c6f63u, 0x00000072u, 0x00050005u, 0x00000016u,
   0x635f6e69u, 0x726f6c6fu, 0x00000000u, 0x00030047u,
   0x0000000bu, 0x00000002u, 0x00050048u, 0x0000000bu,
   0x00000000u, 0x0000000bu, 0x00000000u, 0x00050048u,
   0x0000000bu, 0x00000001u, 0x0000000bu, 0x00000001u,
   0x00050048u, 0x0000000bu, 0x00000002u, 0x0000000bu,
   0x00000003u, 0x00050048u, 0x0000000bu, 0x00000003u,
   0x0000000bu, 0x00000004u, 0x00040047u, 0x00000011u,
   0x0000001eu, 0x00000000u, 0x00040047u, 0x00000015u,
   0x0000001eu, 0x00000000u, 0x00040047u, 0x00000016u,
   0x0000001eu, 0x00000001u, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u,
   0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u,
   0x00000006u, 0x00000004u, 0x00040015u, 0x00000008u,
   0x00000020u, 0x00000000u, 0x0004002bu, 0x00000008u,
   0x00000009u, 0x00000001u, 0x0004001cu, 0x0000000au,
   0x00000006u, 0x00000009u, 0x0006001eu, 0x0000000bu,
   0x00000007u, 0x00000006u, 0x0000000au, 0x0000000au,
   0x00040020u, 0x0000000cu, 0x00000003u, 0x0000000bu,
   0x0004003bu, 0x0000000cu, 0x0000000du, 0x00000003u,
   0x00040015u, 0x0000000eu, 0x00000020u, 0x00000001u,
   0x0004002bu, 0x0000000eu, 0x0000000fu, 0x00000000u,
   0x00040020u, 0x00000010u, 0x00000001u, 0x00000007u,
   0x0004003bu, 0x00000010u, 0x00000011u, 0x00000001u,
   0x00040020u, 0x00000013u, 0x00000003u, 0x00000007u,
   0x0004003bu, 0x00000013u, 0x00000015u, 0x00000003u,
   0x0004003bu, 0x00000010u, 0x00000016u, 0x00000001u,
   0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u,
   0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003du,
   0x00000007u, 0x00000012u, 0x00000011u, 0x00050041u,
   0x00000013u, 0x00000014u, 0x0000000du, 0x0000000fu,
   0x0003003eu, 0x00000014u, 0x00000012u, 0x0004003du,
   0x00000007u, 0x00000017u, 0x00000016u, 0x0003003eu,
   0x00000015u, 0x00000017u, 0x000100fdu, 0x00010038u,
};

static const uint32_t yttrium_textured_vs_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x0000001eu,
   0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
   0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
   0x000b000fu, 0x00000000u, 0x00000004u, 0x6e69616du,
   0x00000000u, 0x0000000du, 0x00000011u, 0x00000015u,
   0x00000016u, 0x0000001au, 0x0000001cu, 0x00030003u,
   0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00060005u, 0x0000000bu,
   0x505f6c67u, 0x65567265u, 0x78657472u, 0x00000000u,
   0x00060006u, 0x0000000bu, 0x00000000u, 0x505f6c67u,
   0x7469736fu, 0x006e6f69u, 0x00070006u, 0x0000000bu,
   0x00000001u, 0x505f6c67u, 0x746e696fu, 0x657a6953u,
   0x00000000u, 0x00070006u, 0x0000000bu, 0x00000002u,
   0x435f6c67u, 0x4470696cu, 0x61747369u, 0x0065636eu,
   0x00070006u, 0x0000000bu, 0x00000003u, 0x435f6c67u,
   0x446c6c75u, 0x61747369u, 0x0065636eu, 0x00030005u,
   0x0000000du, 0x00000000u, 0x00040005u, 0x00000011u,
   0x705f6e69u, 0x0000736fu, 0x00040005u, 0x00000015u,
   0x6f635f76u, 0x00726f6cu, 0x00050005u, 0x00000016u,
   0x635f6e69u, 0x726f6c6fu, 0x00000000u, 0x00040005u,
   0x0000001au, 0x76755f76u, 0x00000000u, 0x00040005u,
   0x0000001cu, 0x755f6e69u, 0x00000076u, 0x00030047u,
   0x0000000bu, 0x00000002u, 0x00050048u, 0x0000000bu,
   0x00000000u, 0x0000000bu, 0x00000000u, 0x00050048u,
   0x0000000bu, 0x00000001u, 0x0000000bu, 0x00000001u,
   0x00050048u, 0x0000000bu, 0x00000002u, 0x0000000bu,
   0x00000003u, 0x00050048u, 0x0000000bu, 0x00000003u,
   0x0000000bu, 0x00000004u, 0x00040047u, 0x00000011u,
   0x0000001eu, 0x00000000u, 0x00040047u, 0x00000015u,
   0x0000001eu, 0x00000000u, 0x00040047u, 0x00000016u,
   0x0000001eu, 0x00000001u, 0x00040047u, 0x0000001au,
   0x0000001eu, 0x00000001u, 0x00040047u, 0x0000001cu,
   0x0000001eu, 0x00000002u, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u,
   0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u,
   0x00000006u, 0x00000004u, 0x00040015u, 0x00000008u,
   0x00000020u, 0x00000000u, 0x0004002bu, 0x00000008u,
   0x00000009u, 0x00000001u, 0x0004001cu, 0x0000000au,
   0x00000006u, 0x00000009u, 0x0006001eu, 0x0000000bu,
   0x00000007u, 0x00000006u, 0x0000000au, 0x0000000au,
   0x00040020u, 0x0000000cu, 0x00000003u, 0x0000000bu,
   0x0004003bu, 0x0000000cu, 0x0000000du, 0x00000003u,
   0x00040015u, 0x0000000eu, 0x00000020u, 0x00000001u,
   0x0004002bu, 0x0000000eu, 0x0000000fu, 0x00000000u,
   0x00040020u, 0x00000010u, 0x00000001u, 0x00000007u,
   0x0004003bu, 0x00000010u, 0x00000011u, 0x00000001u,
   0x00040020u, 0x00000013u, 0x00000003u, 0x00000007u,
   0x0004003bu, 0x00000013u, 0x00000015u, 0x00000003u,
   0x0004003bu, 0x00000010u, 0x00000016u, 0x00000001u,
   0x00040017u, 0x00000018u, 0x00000006u, 0x00000002u,
   0x00040020u, 0x00000019u, 0x00000003u, 0x00000018u,
   0x0004003bu, 0x00000019u, 0x0000001au, 0x00000003u,
   0x00040020u, 0x0000001bu, 0x00000001u, 0x00000018u,
   0x0004003bu, 0x0000001bu, 0x0000001cu, 0x00000001u,
   0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u,
   0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003du,
   0x00000007u, 0x00000012u, 0x00000011u, 0x00050041u,
   0x00000013u, 0x00000014u, 0x0000000du, 0x0000000fu,
   0x0003003eu, 0x00000014u, 0x00000012u, 0x0004003du,
   0x00000007u, 0x00000017u, 0x00000016u, 0x0003003eu,
   0x00000015u, 0x00000017u, 0x0004003du, 0x00000018u,
   0x0000001du, 0x0000001cu, 0x0003003eu, 0x0000001au,
   0x0000001du, 0x000100fdu, 0x00010038u,
};

static const uint32_t yttrium_textured_fs_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000018u,
   0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
   0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
   0x0008000fu, 0x00000004u, 0x00000004u, 0x6e69616du,
   0x00000000u, 0x00000009u, 0x00000011u, 0x00000015u,
   0x00030010u, 0x00000004u, 0x00000007u, 0x00030003u,
   0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00050005u, 0x00000009u,
   0x5f74756fu, 0x6f6c6f63u, 0x00000072u, 0x00040005u,
   0x0000000du, 0x30786574u, 0x00000000u, 0x00040005u,
   0x00000011u, 0x76755f76u, 0x00000000u, 0x00040005u,
   0x00000015u, 0x6f635f76u, 0x00726f6cu, 0x00040047u,
   0x00000009u, 0x0000001eu, 0x00000000u, 0x00040047u,
   0x0000000du, 0x00000021u, 0x00000000u, 0x00040047u,
   0x0000000du, 0x00000022u, 0x00000000u, 0x00040047u,
   0x00000011u, 0x0000001eu, 0x00000001u, 0x00040047u,
   0x00000015u, 0x0000001eu, 0x00000000u, 0x00020013u,
   0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u,
   0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u,
   0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u,
   0x00000008u, 0x00000003u, 0x00000007u, 0x0004003bu,
   0x00000008u, 0x00000009u, 0x00000003u, 0x00090019u,
   0x0000000au, 0x00000006u, 0x00000001u, 0x00000000u,
   0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u,
   0x0003001bu, 0x0000000bu, 0x0000000au, 0x00040020u,
   0x0000000cu, 0x00000000u, 0x0000000bu, 0x0004003bu,
   0x0000000cu, 0x0000000du, 0x00000000u, 0x00040017u,
   0x0000000fu, 0x00000006u, 0x00000002u, 0x00040020u,
   0x00000010u, 0x00000001u, 0x0000000fu, 0x0004003bu,
   0x00000010u, 0x00000011u, 0x00000001u, 0x00040020u,
   0x00000014u, 0x00000001u, 0x00000007u, 0x0004003bu,
   0x00000014u, 0x00000015u, 0x00000001u, 0x00050036u,
   0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u,
   0x000200f8u, 0x00000005u, 0x0004003du, 0x0000000bu,
   0x0000000eu, 0x0000000du, 0x0004003du, 0x0000000fu,
   0x00000012u, 0x00000011u, 0x00050057u, 0x00000007u,
   0x00000013u, 0x0000000eu, 0x00000012u, 0x0004003du,
   0x00000007u, 0x00000016u, 0x00000015u, 0x00050085u,
   0x00000007u, 0x00000017u, 0x00000013u, 0x00000016u,
   0x0003003eu, 0x00000009u, 0x00000017u, 0x000100fdu,
   0x00010038u,
};

static bool
yttrium_venus_find_queue_family(struct yttrium_venus *venus,
                                VkPhysicalDevice physical_device,
                                uint32_t *queue_family_index)
{
   uint32_t count = 0;
   vn_call_vkGetPhysicalDeviceQueueFamilyProperties(&venus->vn_ring,
                                                    physical_device,
                                                    &count, NULL);
   if (!count)
      return false;

   VkQueueFamilyProperties *props =
      CALLOC(count, sizeof(VkQueueFamilyProperties));
   if (!props)
      return false;

   vn_call_vkGetPhysicalDeviceQueueFamilyProperties(&venus->vn_ring,
                                                    physical_device,
                                                    &count, props);

   bool found = false;
   for (uint32_t i = 0; i < count; i++) {
      if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
         *queue_family_index = i;
         found = true;
         break;
      }
   }

   FREE(props);
   return found;
}

uint32_t
yttrium_venus_choose_memory_type(struct yttrium_venus *venus,
                                 uint32_t bits,
                                 VkMemoryPropertyFlags required,
                                 VkMemoryPropertyFlags preferred)
{
   /*
    * Required flags are hard requirements.  Only preferred flags may be
    * dropped, and only after trying the exact preferred match first.
    */
   for (uint32_t i = 0; i < venus->memory_props.memoryTypeCount; i++) {
      if (!(bits & (1u << i)))
         continue;

      const VkMemoryPropertyFlags flags =
         venus->memory_props.memoryTypes[i].propertyFlags;
      if ((flags & required) == required && (flags & preferred) == preferred)
         return i;
   }

   for (uint32_t i = 0; i < venus->memory_props.memoryTypeCount; i++) {
      if (!(bits & (1u << i)))
         continue;

      const VkMemoryPropertyFlags flags =
         venus->memory_props.memoryTypes[i].propertyFlags;
      if ((flags & required) == required)
         return i;
   }

   return UINT32_MAX;
}

static bool
yttrium_venus_memory_type_has_flags(struct yttrium_venus *venus,
                                    uint32_t index,
                                    VkMemoryPropertyFlags flags)
{
   if (!venus || index >= venus->memory_props.memoryTypeCount)
      return false;

   return (venus->memory_props.memoryTypes[index].propertyFlags & flags) ==
      flags;
}

void
yttrium_venus_unmap_memory(struct yttrium_venus *venus,
                           struct yttrium_venus_memory_mapping *mapping)
{
   if (!venus || !venus->device || !mapping ||
       !mapping->hAllocation)
      return;

   if (mapping->map && mapping->map_is_blob) {
      VIOGPU_ESCAPE unmap;
      memset(&unmap, 0, sizeof(unmap));
      unmap.Type = VIOGPU_RES_UNMAP_BLOB;
      unmap.DataLength = sizeof(VIOGPU_RES_UNMAP_BLOB_REQ);
      unmap.ResourceUnmapBlob.ResHandle =
         (D3DKMT_HANDLE)mapping->hAllocation;
      NTSTATUS unmap_status =
         venus->device->escape(venus->device, &unmap, sizeof(unmap));
      if (!NT_SUCCESS(unmap_status)) {
         YTTRIUM_WARN("yttrium: internal allocation unmap failed owner=venus2-memory-mapping-unmap status=0x%lx hAllocation=0x%lx hResource=%p size=0x%llx\n",
                      (unsigned long)unmap_status,
                      (unsigned long)mapping->hAllocation,
                      mapping->hResource,
                      (unsigned long long)mapping->size);
      }
   }

   NTSTATUS status = venus->device->destroyAllocation(
      venus->device, NULL,
      (D3DKMT_HANDLE)mapping->hAllocation);
   if (!NT_SUCCESS(status)) {
      YTTRIUM_WARN("yttrium: internal allocation destroy failed owner=venus2-memory-mapping-destroy status=0x%lx hAllocation=0x%lx hResource=%p size=0x%llx\n",
                   (unsigned long)status,
                   (unsigned long)mapping->hAllocation,
                   mapping->hResource,
                   (unsigned long long)mapping->size);
   }
   memset(mapping, 0, sizeof(*mapping));
}

bool
yttrium_venus_map_memory(struct yttrium_venus *venus,
                         uint64_t venus_memory_id,
                         uint64_t size,
                         struct yttrium_venus_memory_mapping *mapping)
{
   VIOGPU_CREATE_ALLOCATION_EXCHANGE alloc_exchange;
   VIOGPU_CREATE_RESOURCE_EXCHANGE res_exchange;
   struct gdikmt_createallocation create;
   D3DDDI_ALLOCATIONINFO alloc_info;

   if (!venus || !venus->device || !venus_memory_id || !size ||
       !mapping)
      return false;

   memset(mapping, 0, sizeof(*mapping));
   memset(&alloc_exchange, 0, sizeof(alloc_exchange));
   memset(&res_exchange, 0, sizeof(res_exchange));
   memset(&create, 0, sizeof(create));
   memset(&alloc_info, 0, sizeof(alloc_info));

   alloc_exchange.ResourceOptions.target = PIPE_BUFFER;
   alloc_exchange.ResourceOptions.width =
      (ULONG)MIN2(size, (uint64_t)UINT_MAX);
   alloc_exchange.ResourceOptions.height = 1;
   alloc_exchange.ResourceOptions.depth = 1;
   alloc_exchange.ResourceOptions.array_size = 1;
   alloc_exchange.Size = size;
   alloc_exchange.BlobId = venus_memory_id;
   alloc_exchange.BlobMem = VIRTGPU_BLOB_MEM_HOST3D;
   alloc_exchange.BlobFlags = VIRTGPU_BLOB_FLAG_USE_MAPPABLE |
                              VIRTGPU_BLOB_FLAG_USE_SHAREABLE;

   create.NumAllocations = 1;
   create.pAllocationInfo = &alloc_info;
   create.pPrivateDriverData = &res_exchange;
   create.PrivateDriverDataSize = sizeof(res_exchange);
   create.force_allocation_handle = true;

   alloc_info.pPrivateDriverData = &alloc_exchange;
   alloc_info.PrivateDriverDataSize = sizeof(alloc_exchange);

   NTSTATUS status = venus->device->createAllocation(venus->device, &create);
   if (!NT_SUCCESS(status)) {
      YTTRIUM_WARN("yttrium: internal allocation create failed owner=venus2-memory-mapping-create status=0x%lx mem_id=0x%llx size=0x%llx\n",
                   (unsigned long)status,
                   (unsigned long long)venus_memory_id,
                   (unsigned long long)size);
      return false;
   }

   mapping->hResource = create.hResource;
   mapping->hAllocation = alloc_info.hAllocation;
   mapping->size = size;

   VIOGPU_ESCAPE map;
   memset(&map, 0, sizeof(map));
   map.Type = VIOGPU_RES_MAP_BLOB;
   map.DataLength = sizeof(VIOGPU_RES_MAP_BLOB_REQ);
   map.ResourceMapBlob.ResHandle = (D3DKMT_HANDLE)mapping->hAllocation;
   map.ResourceMapBlob.Size = size;

   status = venus->device->escape(venus->device, &map, sizeof(map));
   if (NT_SUCCESS(status) && map.ResourceMapBlob.UserVa) {
      mapping->map = (void *)(uintptr_t)map.ResourceMapBlob.UserVa;
      mapping->map_is_blob = true;
      return true;
   }

   YTTRIUM_WARN("yttrium: internal allocation map failed owner=venus2-memory-mapping-map status=0x%lx hAllocation=0x%lx hResource=%p mem_id=0x%llx size=0x%llx user_va=0x%llx\n",
                (unsigned long)status,
                (unsigned long)mapping->hAllocation,
                mapping->hResource,
                (unsigned long long)venus_memory_id,
                (unsigned long long)size,
                (unsigned long long)map.ResourceMapBlob.UserVa);

   yttrium_venus_unmap_memory(venus, mapping);
   return false;
}

static bool
yttrium_venus_initialize_physical_device(struct yttrium_venus *venus)
{
   VkResult result;

   uint32_t instance_version = 0;
   result = vn_call_vkEnumerateInstanceVersion(&venus->vn_ring,
                                               &instance_version);
   if (result != VK_SUCCESS)
      YTTRIUM_LOG("yttrium: Venus vkEnumerateInstanceVersion returned %d\n",
                   result);
   yttrium_venus_init_object(venus, &venus->instance_obj);
   venus->instance = YTTRIUM_VENUS_HANDLE(VkInstance, &venus->instance_obj);

   const VkApplicationInfo app_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "yttrium",
      .applicationVersion = 1,
      .pEngineName = "yttrium",
      .engineVersion = 1,
      .apiVersion = VK_API_VERSION_1_1,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
   };

   result = vn_call_vkCreateInstance(&venus->vn_ring, &instance_info, NULL,
                                     &venus->instance);
   if (result != VK_SUCCESS) {
      YTTRIUM_LOG("yttrium: Venus vkCreateInstance failed result=%d\n",
                   result);
      return false;
   }
   venus->instance_initialized = true;

   uint32_t physical_device_count = 0;
   result = vn_call_vkEnumeratePhysicalDevices(&venus->vn_ring,
                                               venus->instance,
                                               &physical_device_count, NULL);
   if (result != VK_SUCCESS || !physical_device_count) {
      YTTRIUM_LOG("yttrium: Venus vkEnumeratePhysicalDevices count failed result=%d count=%u\n",
                   result, physical_device_count);
      return false;
   }

   if (physical_device_count > YTTRIUM_VENUS_MAX_PHYSICAL_DEVICES) {
      YTTRIUM_LOG("yttrium: Venus physical device count %u exceeds storage %u; ignoring later devices\n",
                  physical_device_count,
                  YTTRIUM_VENUS_MAX_PHYSICAL_DEVICES);
      physical_device_count = YTTRIUM_VENUS_MAX_PHYSICAL_DEVICES;
   }
   VkPhysicalDevice
      physical_devices[YTTRIUM_VENUS_MAX_PHYSICAL_DEVICES] = { 0 };
   for (uint32_t i = 0; i < physical_device_count; i++) {
      yttrium_venus_init_object(venus, &venus->physical_device_objs[i]);
      physical_devices[i] =
         YTTRIUM_VENUS_HANDLE(VkPhysicalDevice,
                              &venus->physical_device_objs[i]);
   }

   uint32_t request_count = physical_device_count;
   result = vn_call_vkEnumeratePhysicalDevices(&venus->vn_ring,
                                               venus->instance,
                                               &request_count,
                                               physical_devices);
   if ((result != VK_SUCCESS && result != VK_INCOMPLETE) || !request_count) {
      YTTRIUM_LOG("yttrium: Venus vkEnumeratePhysicalDevices failed result=%d count=%u\n",
                   result, request_count);
      return false;
   }

   uint32_t selected_index = UINT32_MAX;
   uint32_t selected_queue_family = 0;
   VkPhysicalDeviceProperties selected_props = { 0 };
   bool saw_cpu_physical_device = false;
   const int64_t index_override =
      yttrium_gdi_debug_get_num_option(
         "D3D10UMD_YTTRIUM_VK_PHYSICAL_DEVICE_INDEX", -1);
   const bool have_index_override = (index_override >= 0);

   const char *name_override =
      yttrium_gdi_debug_get_option(
         "D3D10UMD_YTTRIUM_VK_PHYSICAL_DEVICE_NAME", NULL);

   const bool have_name_override = name_override && name_override[0];
   const bool have_physical_device_override =
      have_index_override || have_name_override;
   for (uint32_t i = 0; i < request_count; i++) {
      uint32_t queue_family = 0;
      VkPhysicalDeviceProperties2 properties2 = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      };
      vn_call_vkGetPhysicalDeviceProperties2(&venus->vn_ring,
                                             physical_devices[i],
                                             &properties2);

      if (!yttrium_venus_find_queue_family(venus, physical_devices[i],
                                           &queue_family))
         continue;
      if (have_index_override && index_override != (int64_t)i)
         continue;
      if (!have_index_override && have_name_override &&
          _stricmp(properties2.properties.deviceName, name_override))
         continue;

      if (properties2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
         saw_cpu_physical_device = true;
         if (!have_physical_device_override)
            continue;
      }

      selected_index = i;
      selected_queue_family = queue_family;
      selected_props = properties2.properties;
      break;
   }

   if (selected_index == UINT32_MAX) {
      if (have_index_override) {
         YTTRIUM_LOG("yttrium: Venus no usable physical device option=D3D10UMD_YTTRIUM_VK_PHYSICAL_DEVICE_INDEX value=%lld count=%u\n",
                     (long long)index_override, request_count);
      } else if (have_name_override) {
         YTTRIUM_LOG("yttrium: Venus no usable physical device option=D3D10UMD_YTTRIUM_VK_PHYSICAL_DEVICE_NAME value=\"%s\" count=%u\n",
                     name_override, request_count);
      } else if (saw_cpu_physical_device) {
         YTTRIUM_LOG("yttrium: Venus no usable non-CPU physical device; set D3D10UMD_YTTRIUM_VK_PHYSICAL_DEVICE_INDEX or D3D10UMD_YTTRIUM_VK_PHYSICAL_DEVICE_NAME to select a CPU device explicitly\n");
      } else {
         YTTRIUM_LOG("yttrium: Venus no usable queue family count=%u\n",
                      request_count);
      }
      return false;
   }

   venus->physical_device = physical_devices[selected_index];
   venus->queue_family_index = selected_queue_family;
   const bool selected_cpu_device =
      selected_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
   if (have_physical_device_override) {
      YTTRIUM_LOG("yttrium: Venus Vulkan physical-device override reason=%s selected_index=%u count=%u name=\"%s\" type=%u vendor=0x%x device=0x%x queue_family=%u cpu=%u memory_requirements=strict\n",
                   have_index_override ?
                      "D3D10UMD_YTTRIUM_VK_PHYSICAL_DEVICE_INDEX" :
                      "D3D10UMD_YTTRIUM_VK_PHYSICAL_DEVICE_NAME",
                   selected_index, request_count, selected_props.deviceName,
                   selected_props.deviceType, selected_props.vendorID,
                   selected_props.deviceID, selected_queue_family,
                   selected_cpu_device);
   } else {
      YTTRIUM_LOG("yttrium: Venus selected physical device index=%u count=%u name=\"%s\" type=%u vendor=0x%x device=0x%x queue_family=%u cpu=%u memory_requirements=strict\n",
                  selected_index, request_count, selected_props.deviceName,
                  selected_props.deviceType, selected_props.vendorID,
                  selected_props.deviceID, selected_queue_family,
                  selected_cpu_device);
   }

   venus->instance_version = instance_version;
   venus->framebuffer_sample_counts =
      selected_props.limits.framebufferColorSampleCounts;
   venus->physical_device_initialized = true;
   return true;
}

static bool
yttrium_venus_create_device_objects(struct yttrium_venus *venus)
{
   VkResult result;

   if (!venus->physical_device_initialized &&
       !yttrium_venus_initialize_physical_device(venus))
      return false;

   memset(&venus->memory_props, 0, sizeof(venus->memory_props));
   vn_call_vkGetPhysicalDeviceMemoryProperties(&venus->vn_ring,
                                               venus->physical_device,
                                               &venus->memory_props);

   bool have_depth_bias_control_ext = false;
   bool have_transform_feedback_ext = false;
   bool have_vertex_attribute_divisor_ext = false;
   bool have_vertex_attribute_divisor_khr = false;
   bool have_push_descriptor_ext = false;
   bool have_create_renderpass2_khr = false;
   bool have_multisampled_render_to_single_sampled_ext = false;
   bool have_fragment_shader_interlock_ext = false;
   bool have_attachment_feedback_loop_layout_ext = false;
   bool have_load_store_op_none_khr = false;
   bool have_load_store_op_none_ext = false;
   bool have_vulkan_memory_model_khr = false;
   bool have_custom_border_color_ext = false;
   uint32_t extension_count = 0;
   result = vn_call_vkEnumerateDeviceExtensionProperties(
      &venus->vn_ring, venus->physical_device, NULL, &extension_count, NULL);
   if (result == VK_SUCCESS && extension_count) {
      VkExtensionProperties *extensions =
         CALLOC(extension_count, sizeof(*extensions));
      if (extensions) {
         uint32_t request_count = extension_count;
         result = vn_call_vkEnumerateDeviceExtensionProperties(
            &venus->vn_ring, venus->physical_device, NULL,
            &request_count, extensions);
         if (result == VK_SUCCESS || result == VK_INCOMPLETE) {
            for (uint32_t i = 0; i < request_count; i++) {
               if (!strcmp(extensions[i].extensionName,
                           VK_EXT_DEPTH_BIAS_CONTROL_EXTENSION_NAME)) {
                  have_depth_bias_control_ext = true;
               } else if (!strcmp(extensions[i].extensionName,
                                  VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME)) {
                  have_transform_feedback_ext = true;
               } else if (!strcmp(extensions[i].extensionName,
                                  VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME)) {
                  have_vertex_attribute_divisor_ext = true;
               } else if (!strcmp(extensions[i].extensionName,
                                  VK_KHR_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME)) {
                  have_vertex_attribute_divisor_khr = true;
               } else if (!strcmp(extensions[i].extensionName,
                                  VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) {
                  have_push_descriptor_ext = true;
               } else if (!strcmp(extensions[i].extensionName,
                                  VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME)) {
                  have_create_renderpass2_khr = true;
               } else if (!strcmp(extensions[i].extensionName,
                                  VK_EXT_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_EXTENSION_NAME)) {
                  have_multisampled_render_to_single_sampled_ext = true;
               } else if (!strcmp(extensions[i].extensionName,
                                  VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME)) {
                  have_fragment_shader_interlock_ext = true;
               } else if (!strcmp(extensions[i].extensionName,
                                  VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME)) {
                  have_attachment_feedback_loop_layout_ext = true;
               } else if (!strcmp(extensions[i].extensionName,
                                  VK_KHR_LOAD_STORE_OP_NONE_EXTENSION_NAME)) {
                  have_load_store_op_none_khr =
                     vn_cs_renderer_protocol_has_extension(
                        527 /* VK_KHR_load_store_op_none */);
               } else if (!strcmp(extensions[i].extensionName,
                                  VK_EXT_LOAD_STORE_OP_NONE_EXTENSION_NAME)) {
                  have_load_store_op_none_ext =
                     vn_cs_renderer_protocol_has_extension(
                        401 /* VK_EXT_load_store_op_none */);
               } else if (!strcmp(extensions[i].extensionName,
                                  VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME)) {
                  have_vulkan_memory_model_khr = true;
               } else if (!strcmp(extensions[i].extensionName,
                                  VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME)) {
                  have_custom_border_color_ext =
                     vn_cs_renderer_protocol_has_extension(
                        288 /* VK_EXT_custom_border_color */);
               }
            }
         }
         FREE(extensions);
      }
   }

   VkPhysicalDeviceDepthBiasControlFeaturesEXT depth_bias_control = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_BIAS_CONTROL_FEATURES_EXT,
   };
   VkPhysicalDeviceVulkan12Features features12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
   };
   VkPhysicalDeviceVulkan11Features features11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
   };
   VkPhysicalDeviceTransformFeedbackFeaturesEXT transform_feedback = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
   };
   VkPhysicalDeviceVertexAttributeDivisorFeatures vertex_divisor = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES,
   };
   VkPhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT
      multisampled_render_to_single_sampled = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_FEATURES_EXT,
   };
   VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT
      fragment_shader_interlock = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT,
   };
   VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT
      attachment_feedback_loop_layout = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_FEATURES_EXT,
   };
   VkPhysicalDeviceCustomBorderColorFeaturesEXT custom_border_color = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT,
   };
   const bool have_vertex_attribute_divisor =
      have_vertex_attribute_divisor_khr || have_vertex_attribute_divisor_ext;
   features12.pNext = &features11;
   void *features_pnext = &features12;
   if (have_custom_border_color_ext) {
      custom_border_color.pNext = features_pnext;
      features_pnext = &custom_border_color;
   }
   if (have_multisampled_render_to_single_sampled_ext) {
      multisampled_render_to_single_sampled.pNext = features_pnext;
      features_pnext = &multisampled_render_to_single_sampled;
   }
   if (have_fragment_shader_interlock_ext) {
      fragment_shader_interlock.pNext = features_pnext;
      features_pnext = &fragment_shader_interlock;
   }
   if (have_attachment_feedback_loop_layout_ext) {
      attachment_feedback_loop_layout.pNext = features_pnext;
      features_pnext = &attachment_feedback_loop_layout;
   }
   if (have_vertex_attribute_divisor) {
      vertex_divisor.pNext = features_pnext;
      features_pnext = &vertex_divisor;
   }
   if (have_depth_bias_control_ext) {
      depth_bias_control.pNext = features_pnext;
      features_pnext = &depth_bias_control;
   }
   if (have_transform_feedback_ext) {
      transform_feedback.pNext = features_pnext;
      features_pnext = &transform_feedback;
   }

   VkPhysicalDeviceFeatures2 features2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = features_pnext,
   };
   vn_call_vkGetPhysicalDeviceFeatures2(&venus->vn_ring,
                                        venus->physical_device,
                                        &features2);
   memset(&venus->transform_feedback_props, 0,
          sizeof(venus->transform_feedback_props));
   venus->transform_feedback_props.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT;
   VkPhysicalDeviceVertexAttributeDivisorProperties vertex_divisor_props = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES,
   };
   VkPhysicalDeviceVertexAttributeDivisorPropertiesEXT vertex_divisor_props_ext = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES_EXT,
   };
   void *properties_pnext = NULL;
   if (have_vertex_attribute_divisor_khr)
      properties_pnext = &vertex_divisor_props;
   else if (have_vertex_attribute_divisor_ext)
      properties_pnext = &vertex_divisor_props_ext;
   if (have_transform_feedback_ext) {
      venus->transform_feedback_props.pNext = properties_pnext;
      properties_pnext = &venus->transform_feedback_props;
   }
   VkPhysicalDevicePushDescriptorProperties push_descriptor_props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES,
   };
   if (have_push_descriptor_ext) {
      push_descriptor_props.pNext = properties_pnext;
      properties_pnext = &push_descriptor_props;
   }
   VkPhysicalDeviceCustomBorderColorPropertiesEXT custom_border_color_props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_PROPERTIES_EXT,
   };
   if (have_custom_border_color_ext) {
      custom_border_color_props.pNext = properties_pnext;
      properties_pnext = &custom_border_color_props;
   }
   VkPhysicalDeviceProperties2 properties2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = properties_pnext,
   };
   vn_call_vkGetPhysicalDeviceProperties2(&venus->vn_ring,
                                          venus->physical_device,
                                          &properties2);
   venus->uniform_buffer_offset_alignment =
      properties2.properties.limits.minUniformBufferOffsetAlignment;
   if (!util_is_power_of_two_nonzero64(
          venus->uniform_buffer_offset_alignment))
      venus->uniform_buffer_offset_alignment = 256;
   venus->depth_bias_control =
      have_depth_bias_control_ext && depth_bias_control.depthBiasControl;
   venus->transform_feedback =
      have_transform_feedback_ext && transform_feedback.transformFeedback;
   venus->vertex_attribute_instance_rate_divisor =
      have_vertex_attribute_divisor &&
      vertex_divisor.vertexAttributeInstanceRateDivisor;
   venus->vertex_attribute_instance_rate_zero_divisor =
      have_vertex_attribute_divisor &&
      vertex_divisor.vertexAttributeInstanceRateZeroDivisor;
   if (have_vertex_attribute_divisor_khr)
      venus->max_vertex_attrib_divisor =
         vertex_divisor_props.maxVertexAttribDivisor;
   else if (have_vertex_attribute_divisor_ext)
      venus->max_vertex_attrib_divisor =
         vertex_divisor_props_ext.maxVertexAttribDivisor;
   if (!venus->max_vertex_attrib_divisor)
      venus->max_vertex_attrib_divisor = 1;
   venus->vertex_divisor_nonzero_first_instance =
      have_vertex_attribute_divisor_khr &&
      vertex_divisor_props.supportsNonZeroFirstInstance;
   venus->draw_indirect_first_instance =
      features2.features.drawIndirectFirstInstance;
   venus->multi_viewport = features2.features.multiViewport &&
      properties2.properties.limits.maxViewports > 1;
   venus->shader_output_viewport_index =
      features12.shaderOutputViewportIndex;
   const bool shader_draw_parameters = features11.shaderDrawParameters;
   venus->depth_clamp = features2.features.depthClamp;
   venus->dual_src_blend = features2.features.dualSrcBlend;
   venus->independent_blend = features2.features.independentBlend;
   venus->logic_op = features2.features.logicOp;
   venus->sample_rate_shading = features2.features.sampleRateShading;
   const bool have_vulkan_memory_model =
      features12.vulkanMemoryModel &&
      features12.vulkanMemoryModelDeviceScope &&
      (properties2.properties.apiVersion >= VK_API_VERSION_1_2 ||
       have_vulkan_memory_model_khr);
   venus->tessellation_shader =
      features2.features.tessellationShader && have_vulkan_memory_model;
   venus->fragment_stores_and_atomics =
      features2.features.fragmentStoresAndAtomics;
   venus->storage_image_read_without_format =
      features2.features.shaderStorageImageReadWithoutFormat;
   venus->storage_image_write_without_format =
      features2.features.shaderStorageImageWriteWithoutFormat;
   venus->fragment_shader_pixel_interlock =
      have_fragment_shader_interlock_ext &&
      fragment_shader_interlock.fragmentShaderPixelInterlock;
   venus->attachment_feedback_loop_layout =
      have_attachment_feedback_loop_layout_ext &&
      attachment_feedback_loop_layout.attachmentFeedbackLoopLayout;
   venus->load_store_op_none =
      have_load_store_op_none_khr || have_load_store_op_none_ext;
   YTTRIUM_LOG("yttrium: Venus read-only depth capability load_store_op_none=%u khr=%u ext=%u\n",
               venus->load_store_op_none, have_load_store_op_none_khr,
               have_load_store_op_none_ext);
   venus->multisampled_render_to_single_sampled =
      have_multisampled_render_to_single_sampled_ext &&
      have_create_renderpass2_khr &&
      multisampled_render_to_single_sampled.multisampledRenderToSingleSampled;
   venus->push_descriptor =
      have_push_descriptor_ext && push_descriptor_props.maxPushDescriptors;
   venus->max_push_descriptors =
      venus->push_descriptor ? push_descriptor_props.maxPushDescriptors : 0;
   /* The pipeline key carries a typed color, not a concrete image format. */
   venus->custom_border_colors =
      have_custom_border_color_ext && custom_border_color.customBorderColors &&
      custom_border_color.customBorderColorWithoutFormat &&
      custom_border_color_props.maxCustomBorderColorSamplers;
   venus->custom_border_color_without_format = venus->custom_border_colors;
   venus->max_custom_border_color_samplers = venus->custom_border_colors ?
      custom_border_color_props.maxCustomBorderColorSamplers : 0;
   venus->max_dual_source_render_targets = venus->dual_src_blend ?
      properties2.properties.limits.maxFragmentDualSrcAttachments : 0;
   venus->max_sampler_anisotropy = features2.features.samplerAnisotropy ?
      MAX2(properties2.properties.limits.maxSamplerAnisotropy, 1.0f) : 1.0f;
   venus->max_sampler_lod_bias =
      MAX2(properties2.properties.limits.maxSamplerLodBias, 0.0f);
   venus->max_tessellation_patch_size =
      venus->tessellation_shader ?
      properties2.properties.limits.maxTessellationPatchSize : 0;
   venus->max_viewports =
      venus->multi_viewport && venus->shader_output_viewport_index ?
      MIN2(properties2.properties.limits.maxViewports,
           (uint32_t)PIPE_MAX_VIEWPORTS) : 1;
   venus->framebuffer_sample_counts =
      properties2.properties.limits.framebufferColorSampleCounts;
   venus->framebuffer_no_attachments_sample_counts =
      properties2.properties.limits.framebufferNoAttachmentsSampleCounts;
   venus->mipmap_precision_bits =
      properties2.properties.limits.mipmapPrecisionBits;
   YTTRIUM_LOG("yttrium: Venus device features depth_bias_ext=%u depth_bias=%u depth_clamp=%u dual_src_blend=%u independent_blend=%u sample_rate_shading=%u fragment_stores=%u interlock_ext=%u pixel_interlock=%u feedback_loop_ext=%u feedback_loop=%u create_renderpass2_khr=%u mrss_ext=%u mrss=%u transform_feedback_ext=%u transform_feedback=%u transform_feedback_draw=%u vertex_divisor_ext=%u vertex_divisor_khr=%u vertex_divisor=%u zero_divisor=%u push_descriptor_ext=%u push_descriptor=%u clip_distance=%u cull_distance=%u multi_viewport=%u shader_output_viewport_index=%u max_tf_stride=%u max_vertex_divisor=%u max_push_descriptors=%u extension_count=%u\n",
               have_depth_bias_control_ext, venus->depth_bias_control,
               venus->depth_clamp, venus->dual_src_blend,
               venus->independent_blend,
               venus->sample_rate_shading,
               venus->fragment_stores_and_atomics,
               have_fragment_shader_interlock_ext,
               venus->fragment_shader_pixel_interlock,
               have_attachment_feedback_loop_layout_ext,
               venus->attachment_feedback_loop_layout,
               have_create_renderpass2_khr,
               have_multisampled_render_to_single_sampled_ext,
               venus->multisampled_render_to_single_sampled,
               have_transform_feedback_ext, venus->transform_feedback,
               venus->transform_feedback_props.transformFeedbackDraw,
               have_vertex_attribute_divisor_ext,
               have_vertex_attribute_divisor_khr,
               venus->vertex_attribute_instance_rate_divisor,
               venus->vertex_attribute_instance_rate_zero_divisor,
               have_push_descriptor_ext, venus->push_descriptor,
               features2.features.shaderClipDistance,
               features2.features.shaderCullDistance,
               venus->multi_viewport,
               venus->shader_output_viewport_index,
               venus->transform_feedback_props.maxTransformFeedbackBufferDataStride,
               venus->max_vertex_attrib_divisor,
               venus->max_push_descriptors,
               extension_count);
   YTTRIUM_LOG("yttrium: Venus device limits max_clip=%u max_cull=%u max_combined_clip_cull=%u max_viewports=%u max_sampler_anisotropy=%.1f max_sampler_lod_bias=%.1f framebuffer_sample_counts=0x%x framebuffer_no_attachments_sample_counts=0x%x\n",
               properties2.properties.limits.maxClipDistances,
               properties2.properties.limits.maxCullDistances,
               properties2.properties.limits.maxCombinedClipAndCullDistances,
               venus->max_viewports,
               venus->max_sampler_anisotropy,
               venus->max_sampler_lod_bias,
               venus->framebuffer_sample_counts,
               venus->framebuffer_no_attachments_sample_counts);
   YTTRIUM_LOG("yttrium: Venus tessellation feature=%u max_patch_size=%u memory_model_ext=%u memory_model=%u device_scope=%u api_version=0x%x\n",
               venus->tessellation_shader,
               venus->max_tessellation_patch_size,
               have_vulkan_memory_model_khr,
               features12.vulkanMemoryModel,
               features12.vulkanMemoryModelDeviceScope,
               properties2.properties.apiVersion);
   YTTRIUM_LOG("yttrium: Venus storage image formatless features read=%u write=%u\n",
               venus->storage_image_read_without_format,
               venus->storage_image_write_without_format);
   YTTRIUM_LOG("yttrium: Venus custom border color ext=%u supported=%u without_format=%u enabled=%u max_samplers=%u\n",
               have_custom_border_color_ext,
               custom_border_color.customBorderColors,
               custom_border_color.customBorderColorWithoutFormat,
               venus->custom_border_colors,
               venus->max_custom_border_color_samplers);
   depth_bias_control.depthBiasControl = venus->depth_bias_control;
   depth_bias_control.depthBiasExact = VK_FALSE;
   transform_feedback.transformFeedback = venus->transform_feedback;
   transform_feedback.geometryStreams = VK_FALSE;
   vertex_divisor.vertexAttributeInstanceRateDivisor =
      venus->vertex_attribute_instance_rate_divisor;
   vertex_divisor.vertexAttributeInstanceRateZeroDivisor =
      venus->vertex_attribute_instance_rate_zero_divisor;
   features12.shaderOutputViewportIndex =
      venus->shader_output_viewport_index;
   features11 = (VkPhysicalDeviceVulkan11Features) {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .shaderDrawParameters = shader_draw_parameters,
   };
   multisampled_render_to_single_sampled.multisampledRenderToSingleSampled =
      venus->multisampled_render_to_single_sampled;
   fragment_shader_interlock.fragmentShaderPixelInterlock =
      venus->fragment_shader_pixel_interlock;
   fragment_shader_interlock.fragmentShaderSampleInterlock = VK_FALSE;
   fragment_shader_interlock.fragmentShaderShadingRateInterlock = VK_FALSE;
   attachment_feedback_loop_layout.attachmentFeedbackLoopLayout =
      venus->attachment_feedback_loop_layout;
   custom_border_color.customBorderColors = venus->custom_border_colors;
   custom_border_color.customBorderColorWithoutFormat =
      venus->custom_border_color_without_format;
   VkPhysicalDeviceFeatures enabled_features = {
      .drawIndirectFirstInstance = venus->draw_indirect_first_instance,
      .shaderClipDistance = features2.features.shaderClipDistance,
      .shaderCullDistance = features2.features.shaderCullDistance,
      /* Required by work the driver already submits: geometry-shader stages,
       * cube-array image views, and the dynamically indexed UBO array the
       * shader translation emits.  Leaving them off made those draws
       * undefined behaviour that happened to run on the host driver. */
      .geometryShader = features2.features.geometryShader,
      .imageCubeArray = features2.features.imageCubeArray,
      .shaderUniformBufferArrayDynamicIndexing =
         features2.features.shaderUniformBufferArrayDynamicIndexing,
      .depthClamp = venus->depth_clamp,
      .dualSrcBlend = venus->dual_src_blend,
      .independentBlend = venus->independent_blend,
      .logicOp = venus->logic_op,
      .samplerAnisotropy = features2.features.samplerAnisotropy,
      .sampleRateShading = venus->sample_rate_shading,
      .tessellationShader = venus->tessellation_shader,
      .fragmentStoresAndAtomics = venus->fragment_stores_and_atomics,
      .shaderStorageImageReadWithoutFormat =
         venus->storage_image_read_without_format,
      .shaderStorageImageWriteWithoutFormat =
         venus->storage_image_write_without_format,
      .multiViewport = venus->multi_viewport,
   };

   const char *device_extensions[12];
   uint32_t device_extension_count = 0;
   if (venus->depth_bias_control)
      device_extensions[device_extension_count++] =
         VK_EXT_DEPTH_BIAS_CONTROL_EXTENSION_NAME;
   if (venus->transform_feedback)
      device_extensions[device_extension_count++] =
         VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME;
   if (venus->vertex_attribute_instance_rate_divisor ||
       venus->vertex_attribute_instance_rate_zero_divisor) {
      device_extensions[device_extension_count++] =
         have_vertex_attribute_divisor_khr ?
         VK_KHR_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME :
         VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME;
   }
   if (venus->push_descriptor)
      device_extensions[device_extension_count++] =
         VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
   if (venus->multisampled_render_to_single_sampled) {
      device_extensions[device_extension_count++] =
         VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME;
      device_extensions[device_extension_count++] =
         VK_EXT_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_EXTENSION_NAME;
   }
   if (venus->fragment_shader_pixel_interlock)
      device_extensions[device_extension_count++] =
         VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME;
   if (venus->attachment_feedback_loop_layout)
      device_extensions[device_extension_count++] =
         VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME;
   if (venus->load_store_op_none)
      device_extensions[device_extension_count++] =
         have_load_store_op_none_khr ?
            VK_KHR_LOAD_STORE_OP_NONE_EXTENSION_NAME :
            VK_EXT_LOAD_STORE_OP_NONE_EXTENSION_NAME;
   if (venus->tessellation_shader && have_vulkan_memory_model_khr)
      device_extensions[device_extension_count++] =
         VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME;
   if (venus->custom_border_colors)
      device_extensions[device_extension_count++] =
         VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME;

   yttrium_venus_init_object(venus, &venus->device_obj);
   venus->device_handle = YTTRIUM_VENUS_HANDLE(VkDevice, &venus->device_obj);

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = venus->queue_family_index,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = features_pnext,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = device_extension_count,
      .ppEnabledExtensionNames = device_extension_count ? device_extensions :
                                 NULL,
      .pEnabledFeatures = &enabled_features,
   };

   result = vn_call_vkCreateDevice(&venus->vn_ring, venus->physical_device,
                                   &device_info, NULL,
                                   &venus->device_handle);
   if (result != VK_SUCCESS) {
      YTTRIUM_LOG("yttrium: Venus vkCreateDevice failed result=%d queue_family=%u extensions=%u mrss=%u pixel_interlock=%u\n",
                   result, venus->queue_family_index,
                   device_extension_count,
                   venus->multisampled_render_to_single_sampled,
                   venus->fragment_shader_pixel_interlock);
      return false;
   }

   yttrium_venus_init_object(venus, &venus->queue_obj);
   venus->queue = YTTRIUM_VENUS_HANDLE(VkQueue, &venus->queue_obj);

   const VkDeviceQueueTimelineInfoMESA timeline_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_TIMELINE_INFO_MESA,
      .ringIdx = 1,
   };
   const VkDeviceQueueInfo2 queue_info2 = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
      .pNext = &timeline_info,
      .queueFamilyIndex = venus->queue_family_index,
      .queueIndex = 0,
   };
   struct vn_ring_submit_command queue_submit;
   vn_submit_vkGetDeviceQueue2(
      &venus->vn_ring, VK_COMMAND_GENERATE_REPLY_BIT_EXT,
      venus->device_handle, &queue_info2, &venus->queue, &queue_submit);
   struct vn_cs_decoder *queue_reply =
      queue_submit.ring_seqno_valid ?
         vn_ring_get_command_reply(&venus->vn_ring, &queue_submit) : NULL;
   if (!queue_reply) {
      if (queue_submit.ring_seqno_valid)
         vn_ring_free_command_reply(&venus->vn_ring, &queue_submit);
      YTTRIUM_WARN("yttrium: ERROR: Venus queue acquisition failed owner=venus2 operation=vkGetDeviceQueue2 queue_id=%llu reason=%s action=fail-device-init\n",
                   (unsigned long long)venus->queue_obj.id,
                   queue_submit.ring_seqno_valid ? "missing-reply" :
                                                   "local-ring-enqueue-failed");
      venus->queue = VK_NULL_HANDLE;
      memset(&venus->queue_obj, 0, sizeof(venus->queue_obj));
      return false;
   }
   vn_decode_vkGetDeviceQueue2_reply(
      queue_reply, venus->device_handle, &queue_info2, &venus->queue);
   vn_ring_free_command_reply(&venus->vn_ring, &queue_submit);
   if (!venus->queue) {
      YTTRIUM_WARN("yttrium: ERROR: Venus queue acquisition failed owner=venus2 operation=vkGetDeviceQueue2 queue_id=%llu reason=null-queue action=fail-device-init\n",
                   (unsigned long long)venus->queue_obj.id);
      memset(&venus->queue_obj, 0, sizeof(venus->queue_obj));
      return false;
   }

   yttrium_venus_init_object(venus, &venus->command_pool_obj);
   venus->command_pool =
      YTTRIUM_VENUS_HANDLE(VkCommandPool, &venus->command_pool_obj);
   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = venus->queue_family_index,
   };
   result = vn_call_vkCreateCommandPool(&venus->vn_ring,
                                        venus->device_handle, &pool_info,
                                        NULL, &venus->command_pool);
   if (result != VK_SUCCESS) {
      YTTRIUM_LOG("yttrium: Venus vkCreateCommandPool failed result=%d queue_family=%u\n",
                   result, venus->queue_family_index);
      return false;
   }

   YTTRIUM_LOG("yttrium: Venus initialized api=%u.%u queue_family=%u instance_id=%llu device_id=%llu queue_id=%llu\n",
                VK_VERSION_MAJOR(venus->instance_version),
                VK_VERSION_MINOR(venus->instance_version),
                venus->queue_family_index,
                (unsigned long long)venus->instance_obj.id,
                (unsigned long long)venus->device_obj.id,
                (unsigned long long)venus->queue_obj.id);
   return true;
}

static bool
yttrium_venus_ensure_transport(struct yttrium_venus *venus)
{
   if (!venus || venus->failed)
      return false;
   if (venus->transport_initialized)
      return true;

   VIOGPU_ESCAPE ctx_init;
   memset(&ctx_init, 0, sizeof(ctx_init));
   ctx_init.Type = VIOGPU_CTX_INIT;
   ctx_init.DataLength = sizeof(VIOGPU_CTX_INIT_REQ);
   ctx_init.CtxInit.CapsetID = VIRTGPU_DRM_CAPSET_VENUS;

   NTSTATUS status = venus->device->escape(venus->device, &ctx_init,
                                           sizeof(ctx_init));
   if (!NT_SUCCESS(status)) {
      YTTRIUM_LOG("yttrium: Venus context init failed status=0x%lx\n",
                   status);
      venus->failed = true;
      return false;
   }

   status = venus->device->createContext(venus->device, &venus->kmt_ctx);
   if (!NT_SUCCESS(status) || !venus->kmt_ctx) {
      YTTRIUM_LOG("yttrium: Venus D3DKMTCreateContext failed status=0x%lx\n",
                   status);
      venus->failed = true;
      return false;
   }

   venus->vn_ring.driver = venus;
   venus->vn_ring.submit_command = yttrium_venus2_vn_ring_submit_command;
   venus->vn_ring.get_command_reply = yttrium_venus2_vn_ring_get_command_reply;
   venus->vn_ring.free_command_reply = yttrium_venus2_vn_ring_free_command_reply;

   if (!yttrium_venus_ring_create(venus)) {
      YTTRIUM_LOG("yttrium: Venus ring create failed\n");
      venus->failed = true;
      return false;
   }

   if (!yttrium_venus_bo_create(venus, &venus->reply_bo,
                                YTTRIUM_VENUS_REPLY_SIZE)) {
      YTTRIUM_LOG("yttrium: Venus reply BO create failed size=0x%x\n",
                   YTTRIUM_VENUS_REPLY_SIZE);
      venus->failed = true;
      return false;
   }

   venus->transport_initialized = true;
   return true;
}

bool
yttrium_venus_ensure_physical_device(struct yttrium_venus *venus)
{
   if (!yttrium_venus_ensure_transport(venus))
      return false;
   if (venus->physical_device_initialized)
      return true;

   if (!yttrium_venus_initialize_physical_device(venus)) {
      venus->failed = true;
      return false;
   }

   return true;
}

bool
yttrium_venus_ensure_initialized(struct yttrium_venus *venus)
{
   if (!venus || venus->failed)
      return false;
   if (venus->initialized)
      return true;

   if (!yttrium_venus_ensure_physical_device(venus) ||
       !yttrium_venus_create_device_objects(venus)) {
      YTTRIUM_LOG("yttrium: Venus device object creation failed\n");
      venus->failed = true;
      return false;
   }

   venus->initialized = true;
   return true;
}

bool
yttrium_venus2_storage_image_without_format_supported(
   struct yttrium_venus *venus, bool read, bool write)
{
   if (!yttrium_venus_ensure_initialized(venus))
      return false;

   return (!read || venus->storage_image_read_without_format) &&
          (!write || venus->storage_image_write_without_format);
}

static bool
yttrium_venus_storage_image_without_format_listed(VkFormat format)
{
   /* Vulkan "Formats Without Shader Storage Format": the device feature
    * guarantees apply only to this list, not every storage-capable format.
    * This deliberately uses the enabled core features, not the Vulkan 1.3
    * per-format feature-flags2 route.
    */
   switch (format) {
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SNORM:
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_R8G8B8A8_SINT:
   case VK_FORMAT_R32_UINT:
   case VK_FORMAT_R32_SINT:
   case VK_FORMAT_R32_SFLOAT:
   case VK_FORMAT_R32G32_UINT:
   case VK_FORMAT_R32G32_SINT:
   case VK_FORMAT_R32G32_SFLOAT:
   case VK_FORMAT_R32G32B32A32_UINT:
   case VK_FORMAT_R32G32B32A32_SINT:
   case VK_FORMAT_R32G32B32A32_SFLOAT:
   case VK_FORMAT_R16G16B16A16_UINT:
   case VK_FORMAT_R16G16B16A16_SINT:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
   case VK_FORMAT_R16G16_SFLOAT:
   case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R16G16B16A16_UNORM:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
   case VK_FORMAT_R16G16_UNORM:
   case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_R16_UNORM:
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R16G16B16A16_SNORM:
   case VK_FORMAT_R16G16_SNORM:
   case VK_FORMAT_R8G8_SNORM:
   case VK_FORMAT_R16_SNORM:
   case VK_FORMAT_R8_SNORM:
   case VK_FORMAT_R16G16_SINT:
   case VK_FORMAT_R8G8_SINT:
   case VK_FORMAT_R16_SINT:
   case VK_FORMAT_R8_SINT:
   case VK_FORMAT_A2B10G10R10_UINT_PACK32:
   case VK_FORMAT_R16G16_UINT:
   case VK_FORMAT_R8G8_UINT:
   case VK_FORMAT_R16_UINT:
   case VK_FORMAT_R8_UINT:
   case VK_FORMAT_A8_UNORM:
      return true;
   default:
      return false;
   }
}

bool
yttrium_venus2_storage_image_formatless_format_supported(
   struct yttrium_venus *venus, enum pipe_format format,
   bool read, bool write)
{
   if (format == PIPE_FORMAT_NONE || (unsigned)format >= PIPE_FORMAT_COUNT ||
       !yttrium_venus2_storage_image_without_format_supported(venus,
                                                               read, write))
      return false;

   uint8_t *cached = &venus->storage_image_formatless_formats[format];
   if (*cached)
      return *cached == 2;

   const VkFormat vk_format = yttrium_venus2_pipe_format(format);
   if (!yttrium_venus_storage_image_without_format_listed(vk_format)) {
      *cached = 1;
      return false;
   }

   VkFormatProperties2 properties = {
      .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
   };
   vn_call_vkGetPhysicalDeviceFormatProperties2(
      &venus->vn_ring, venus->physical_device, vk_format, &properties);
   const bool supported =
      properties.formatProperties.optimalTilingFeatures &
      VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
   *cached = supported ? 2 : 1;
   return supported;
}

bool
yttrium_venus2_supports_load_store_op_none(struct yttrium_venus *venus)
{
   return venus && venus->load_store_op_none;
}

static VkPipelineStageFlags
yttrium_venus_layout_stage(VkImageLayout layout)
{
   switch (layout) {
   case VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT:
      return VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
             VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
   case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
   case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
      return VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
             VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
   case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
      return VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
   case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
   case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      return VK_PIPELINE_STAGE_TRANSFER_BIT;
   case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      return VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
   case VK_IMAGE_LAYOUT_GENERAL:
      return VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
   case VK_IMAGE_LAYOUT_PREINITIALIZED:
      return VK_PIPELINE_STAGE_HOST_BIT;
   case VK_IMAGE_LAYOUT_UNDEFINED:
   default:
      return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
   }
}

static VkPipelineStageFlags
yttrium_venus_sampled_stage_flags(const struct yttrium_pipeline *pipeline)
{
   VkPipelineStageFlags flags = 0;
   const uint32_t stage_mask = pipeline ? pipeline->key.sampled_stage_mask : 0;

   if (stage_mask & (1u << MESA_SHADER_VERTEX))
      flags |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
   if (stage_mask & (1u << MESA_SHADER_TESS_CTRL))
      flags |= VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT;
   if (stage_mask & (1u << MESA_SHADER_TESS_EVAL))
      flags |= VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
   if (stage_mask & (1u << MESA_SHADER_GEOMETRY))
      flags |= VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
   if (stage_mask & (1u << MESA_SHADER_FRAGMENT))
      flags |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

   return flags ? flags : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
}

static VkAccessFlags
yttrium_venus_layout_access(VkImageLayout layout)
{
   switch (layout) {
   case VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT:
      return VK_ACCESS_SHADER_READ_BIT |
             VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
   case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      return VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
   case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
      return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
   case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
      return VK_ACCESS_SHADER_READ_BIT |
             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
   case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      return VK_ACCESS_TRANSFER_WRITE_BIT;
   case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      return VK_ACCESS_TRANSFER_READ_BIT;
   case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      return VK_ACCESS_SHADER_READ_BIT;
   case VK_IMAGE_LAYOUT_GENERAL:
      return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
   case VK_IMAGE_LAYOUT_PREINITIALIZED:
      return VK_ACCESS_HOST_WRITE_BIT;
   case VK_IMAGE_LAYOUT_UNDEFINED:
   default:
      return 0;
   }
}

static bool
yttrium_venus_image_subresource_range_equal(
   const VkImageSubresourceRange *a,
   const VkImageSubresourceRange *b)
{
   return a->aspectMask == b->aspectMask &&
          a->baseMipLevel == b->baseMipLevel &&
          a->levelCount == b->levelCount &&
          a->baseArrayLayer == b->baseArrayLayer &&
          a->layerCount == b->layerCount;
}

static bool
yttrium_venus_cmd_batch_try_fold_image_barrier(
   struct yttrium_venus *venus,
   VkPipelineStageFlags src_stages,
   VkPipelineStageFlags dst_stages,
   const VkImageMemoryBarrier *barrier)
{
   const VkAccessFlags attachment_write =
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

   if (!barrier || barrier->pNext ||
       barrier->oldLayout != barrier->newLayout ||
       !(barrier->dstAccessMask & attachment_write))
      return false;

   for (uint32_t i = venus->cmd_batch_image_barrier_count; i > 0; i--) {
      VkImageMemoryBarrier *existing =
         &venus->cmd_batch_image_barriers[i - 1];
      if (existing->image != barrier->image)
         continue;
      if (existing->pNext ||
          existing->srcQueueFamilyIndex != barrier->srcQueueFamilyIndex ||
          existing->dstQueueFamilyIndex != barrier->dstQueueFamilyIndex ||
          existing->newLayout != barrier->newLayout ||
          !(existing->dstAccessMask & attachment_write) ||
          !yttrium_venus_image_subresource_range_equal(
             &existing->subresourceRange, &barrier->subresourceRange))
         continue;

      /* Deferred barriers execute together before the first draw.  Preserve
       * the retained transition and range while merging the dependency masks
       * of a redundant same-layout attachment barrier.
       */
      existing->srcAccessMask |= barrier->srcAccessMask;
      existing->dstAccessMask |= barrier->dstAccessMask;
      venus->cmd_batch_image_src_stages |= src_stages;
      venus->cmd_batch_image_dst_stages |= dst_stages;
      return true;
   }

   return false;
}

static bool
yttrium_venus_cmd_batch_add_image_barrier(
   struct yttrium_venus *venus,
   VkPipelineStageFlags src_stages,
   VkPipelineStageFlags dst_stages,
   const VkImageMemoryBarrier *barrier);

static bool
yttrium_venus_build_image_transition(
   struct yttrium_venus_resource *resource,
   VkImageLayout new_layout,
   VkAccessFlags dst_access,
   const VkImageSubresourceRange *range,
   VkPipelineStageFlags *out_src_stage,
   VkImageMemoryBarrier *out_barrier)
{
   const VkImageLayout old_layout = resource->layout;
   VkImageSubresourceRange transition_range = *range;
   const VkImageAspectFlags resource_aspects =
      yttrium_venus_format_aspects(resource->vk_format);
   const VkImageAspectFlags depth_stencil_aspects =
      VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
   const VkAccessFlags write_access =
      VK_ACCESS_SHADER_WRITE_BIT |
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
      VK_ACCESS_TRANSFER_WRITE_BIT |
      VK_ACCESS_HOST_WRITE_BIT |
      VK_ACCESS_MEMORY_WRITE_BIT;
   VkPipelineStageFlags src_stage = yttrium_venus_layout_stage(old_layout);
   VkAccessFlags src_access = yttrium_venus_layout_access(old_layout);

   if (old_layout == new_layout &&
       !(src_access & write_access) &&
       !(dst_access & write_access))
      return false;

   if ((resource_aspects & depth_stencil_aspects) == depth_stencil_aspects &&
       (transition_range.aspectMask & depth_stencil_aspects) &&
       !(transition_range.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT)) {
      transition_range.aspectMask = resource_aspects;
   }

   if (old_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL &&
       new_layout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
      src_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      src_access |= VK_ACCESS_MEMORY_WRITE_BIT;
   }

   *out_src_stage = src_stage;
   *out_barrier = (VkImageMemoryBarrier) {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = src_access,
      .dstAccessMask = dst_access,
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = resource->image,
      .subresourceRange = transition_range,
   };
   return true;
}

static void
yttrium_venus_cmd_transition_image(struct yttrium_venus *venus,
                                   struct yttrium_venus_resource *resource,
                                   VkImageLayout new_layout,
                                   VkAccessFlags dst_access,
                                   VkPipelineStageFlags dst_stage,
                                   const VkImageSubresourceRange *range)
{
   VkPipelineStageFlags src_stage = 0;
   VkImageMemoryBarrier barrier;
   if (!yttrium_venus_build_image_transition(
          resource, new_layout, dst_access, range, &src_stage, &barrier))
      return;

   vn_async_vkCmdPipelineBarrier(&venus->vn_ring, venus->command_buffer,
                                 src_stage, dst_stage,
                                 barrier.oldLayout ==
                                       VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT ||
                                    barrier.newLayout ==
                                       VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT ?
                                    VK_DEPENDENCY_FEEDBACK_LOOP_BIT_EXT : 0,
                                 0, NULL, 0, NULL,
                                 1, &barrier);
   resource->layout = new_layout;
}

static bool
yttrium_venus_cmd_batch_transition_image(struct yttrium_venus *venus,
                                         struct yttrium_venus_resource *resource,
                                         VkImageLayout new_layout,
                                         VkAccessFlags dst_access,
                                         VkPipelineStageFlags dst_stage,
                                         const VkImageSubresourceRange *range)
{
   /* Both uses are reads, so the image-role index allows them to share a
    * command buffer.  Their layouts still differ: deferred barriers run
    * before all queued draws, so emit those draws before changing layout.
    * Keep the current batch and its transient allocations alive.
    */
   if (venus->cmd_batch_deferred_draw_count &&
       ((resource->layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
         new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) ||
        (resource->layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL &&
         new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)) &&
       !yttrium_venus_cmd_batch_emit_deferred_draws(
          venus, "read-only depth layout boundary"))
      return false;

   VkPipelineStageFlags src_stage = 0;
   VkImageMemoryBarrier barrier;
   if (!yttrium_venus_build_image_transition(
          resource, new_layout, dst_access, range, &src_stage, &barrier))
      return true;
   if (barrier.oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
       barrier.newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
      src_stage = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
   if (barrier.oldLayout ==
          VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT ||
       barrier.newLayout ==
          VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT)
      venus->cmd_batch_image_dependency_flags |=
         VK_DEPENDENCY_FEEDBACK_LOOP_BIT_EXT;
   if (!yttrium_venus_cmd_batch_add_image_barrier(
          venus, src_stage, dst_stage, &barrier))
      return false;

   resource->layout = new_layout;
   return true;
}

static bool
yttrium_venus_cmd_ensure_image_initialized(struct yttrium_venus *venus,
                                           struct yttrium_venus_resource *resource,
                                           uint32_t resource_id)
{
   if (resource->contents_initialized)
      return true;

   if (resource->owner &&
       util_format_is_compressed(resource->owner->format)) {
      /* Vulkan forbids vkCmdClearColorImage for compressed formats.  D3D
       * leaves any unwritten parts of a newly created resource undefined. */
      yttrium_venus_mark_aspects_initialized(resource,
                                             VK_IMAGE_ASPECT_COLOR_BIT);
      return true;
   }

   if (!(resource->image_usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
      YTTRIUM_LOG("yttrium: Venus image init rejected res_id=%u image_id=%llu usage=0x%x\n",
                   resource_id,
                   (unsigned long long)resource->image_obj.id,
                   resource->image_usage);
      return false;
   }

   const VkImageSubresourceRange range = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .baseMipLevel = 0,
      .levelCount = MAX2(resource->levels, 1),
      .baseArrayLayer = 0,
      .layerCount = MAX2(resource->layers, 1),
   };
   yttrium_venus_cmd_transition_image(venus, resource,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      VK_ACCESS_TRANSFER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      &range);

   const VkClearColorValue zero = {
      .float32 = { 0.0f, 0.0f, 0.0f, 0.0f },
   };
   vn_async_vkCmdClearColorImage(&venus->vn_ring, venus->command_buffer,
                                 resource->image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &zero, 1, &range);
   yttrium_venus_mark_aspects_initialized(resource,
                                          VK_IMAGE_ASPECT_COLOR_BIT);
   YTTRIUM_LOG("yttrium: Venus initialized image res_id=%u image_id=%llu before partial update\n",
                resource_id,
                (unsigned long long)resource->image_obj.id);
   return true;
}

bool
yttrium_venus_ensure_null_sampled_image(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource **out_resource,
   uint32_t *out_resource_id)
{
   uint64_t allocation_size = 0;

   if (!venus || !out_resource || !out_resource_id)
      return false;

   if (!venus->null_sampled_image.initialized) {
      if (!yttrium_venus2_create_sampled_texture_image(
             venus, &venus->null_sampled_image, PIPE_TEXTURE_2D,
             1, 1, 1, 1, 1, PIPE_FORMAT_R8G8B8A8_UNORM,
             &allocation_size)) {
         YTTRIUM_LOG("yttrium: Venus null sampled image create failed\n");
         return false;
      }
   }

   *out_resource = &venus->null_sampled_image;
   *out_resource_id = 0;
   return true;
}

static bool
yttrium_venus_cmd_ensure_depth_initialized(struct yttrium_venus *venus,
                                           struct yttrium_venus_resource *resource,
                                           uint32_t resource_id)
{
   if (resource->contents_initialized)
      return true;

   if (!(resource->image_usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
       !yttrium_venus_format_has_depth(resource->vk_format)) {
      YTTRIUM_LOG("yttrium: Venus depth image init rejected res_id=%u image_id=%llu usage=0x%x format=%u\n",
                  resource_id,
                  (unsigned long long)resource->image_obj.id,
                  resource->image_usage, resource->vk_format);
      return false;
   }

   const VkImageSubresourceRange range = {
      .aspectMask = yttrium_venus_format_aspects(resource->vk_format),
      .baseMipLevel = 0,
      .levelCount = 1,
      .baseArrayLayer = 0,
      .layerCount = 1,
   };
   yttrium_venus_cmd_transition_image(venus, resource,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      VK_ACCESS_TRANSFER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      &range);

   const VkClearDepthStencilValue clear_value = {
      .depth = 1.0f,
      .stencil = 0,
   };
   vn_async_vkCmdClearDepthStencilImage(&venus->vn_ring,
                                        venus->command_buffer,
                                        resource->image,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        &clear_value, 1, &range);
   yttrium_venus_mark_aspects_initialized(
      resource, yttrium_venus_format_aspects(resource->vk_format));
   YTTRIUM_LOG("yttrium: Venus initialized depth image res_id=%u image_id=%llu\n",
               resource_id,
               (unsigned long long)resource->image_obj.id);
   return true;
}

static void
yttrium_venus_destroy_graphics_objects(struct yttrium_venus *venus,
                                       struct yttrium_venus_resource *resource)
{
   if (!venus || !venus->initialized || !resource)
      return;

   const bool has_graphics_objects =
      resource->image_view || resource->render_pass || resource->framebuffer ||
      resource->pipeline_layout || resource->descriptor_set_layout ||
      resource->descriptor_pool || resource->sampler ||
      resource->vertex_shader || resource->fragment_shader ||
      resource->pipeline;
   bool current_batch_uses_resource = false;
   if (venus->display_copy_batch_recording) {
      for (uint32_t i = 0; i < venus->cmd_batch_pending_resource_count; i++) {
         if (venus->cmd_batch_pending_resources[i] == resource) {
            current_batch_uses_resource = true;
            break;
         }
      }
   }
   if (current_batch_uses_resource &&
       !yttrium_venus_flush_command_batch(
          venus, "graphics object generation retire")) {
      YTTRIUM_WARN("yttrium: ERROR: graphics object generation retirement failed owner=venus2 reason=batch_publish_failed resource=%p\n",
                   resource);
      return;
   }

   struct yttrium_venus_batch *batch =
      yttrium_venus_find_latest_resource_batch(venus, resource);
   struct yttrium_venus_retired_resource *retired =
      yttrium_venus_retired_graphics_objects_take(resource);
   const bool deferred = retired != NULL;
   if (retired) {
      if (batch)
         yttrium_venus_batch_retire_resource(batch, retired);
      else
         yttrium_venus_destroy_retired_resource(venus, retired);
   } else if (has_graphics_objects) {
      YTTRIUM_WARN("yttrium: WARNING: graphics object generation allocation failed owner=venus2 reason=bounded-memory-exhaustion action=synchronous-resource-retirement resource=%p\n",
                   resource);
      yttrium_venus_wait_resource_batches(
         venus, resource, "graphics object bounded-memory exhaustion");
   }

   if (!deferred) {
      if (resource->pipeline)
         vn_async_vkDestroyPipeline(&venus->vn_ring, venus->device_handle,
                                    resource->pipeline, NULL);
      if (resource->vertex_shader)
         vn_async_vkDestroyShaderModule(&venus->vn_ring,
                                        venus->device_handle,
                                        resource->vertex_shader, NULL);
      if (resource->fragment_shader)
         vn_async_vkDestroyShaderModule(&venus->vn_ring,
                                        venus->device_handle,
                                        resource->fragment_shader, NULL);
      if (resource->pipeline_layout)
         vn_async_vkDestroyPipelineLayout(&venus->vn_ring,
                                          venus->device_handle,
                                          resource->pipeline_layout, NULL);
      if (resource->descriptor_pool)
         vn_async_vkDestroyDescriptorPool(&venus->vn_ring,
                                          venus->device_handle,
                                          resource->descriptor_pool, NULL);
      if (resource->descriptor_set_layout)
         vn_async_vkDestroyDescriptorSetLayout(
            &venus->vn_ring, venus->device_handle,
            resource->descriptor_set_layout, NULL);
      if (resource->sampler)
         vn_async_vkDestroySampler(&venus->vn_ring, venus->device_handle,
                                   resource->sampler, NULL);
      if (resource->framebuffer)
         vn_async_vkDestroyFramebuffer(&venus->vn_ring,
                                       venus->device_handle,
                                       resource->framebuffer, NULL);
      if (resource->render_pass)
         vn_async_vkDestroyRenderPass(&venus->vn_ring, venus->device_handle,
                                      resource->render_pass, NULL);
      if (resource->image_view)
         vn_async_vkDestroyImageView(&venus->vn_ring, venus->device_handle,
                                     resource->image_view, NULL);
   }

   memset(&resource->image_view_obj, 0, sizeof(resource->image_view_obj));
   memset(&resource->render_pass_obj, 0, sizeof(resource->render_pass_obj));
   memset(&resource->framebuffer_obj, 0, sizeof(resource->framebuffer_obj));
   memset(&resource->pipeline_layout_obj, 0,
          sizeof(resource->pipeline_layout_obj));
   memset(&resource->descriptor_set_layout_obj, 0,
          sizeof(resource->descriptor_set_layout_obj));
   memset(&resource->descriptor_pool_obj, 0,
          sizeof(resource->descriptor_pool_obj));
   memset(&resource->descriptor_set_obj, 0,
          sizeof(resource->descriptor_set_obj));
   memset(&resource->sampler_obj, 0, sizeof(resource->sampler_obj));
   memset(&resource->vertex_shader_obj, 0, sizeof(resource->vertex_shader_obj));
   memset(&resource->fragment_shader_obj, 0,
          sizeof(resource->fragment_shader_obj));
   memset(&resource->pipeline_obj, 0, sizeof(resource->pipeline_obj));

   resource->image_view = VK_NULL_HANDLE;
   resource->render_pass = VK_NULL_HANDLE;
   resource->framebuffer = VK_NULL_HANDLE;
   resource->pipeline_layout = VK_NULL_HANDLE;
   resource->descriptor_set_layout = VK_NULL_HANDLE;
   resource->descriptor_pool = VK_NULL_HANDLE;
   resource->descriptor_set = VK_NULL_HANDLE;
   resource->sampler = VK_NULL_HANDLE;
   resource->vertex_shader = VK_NULL_HANDLE;
   resource->fragment_shader = VK_NULL_HANDLE;
   resource->pipeline = VK_NULL_HANDLE;
   resource->graphics_ready = false;
   resource->graphics_mode = YTTRIUM_VENUS_GRAPHICS_VERTEX_BUFFER;
   resource->graphics_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   resource->graphics_cull_mode = VK_CULL_MODE_NONE;
   resource->graphics_front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
   resource->graphics_blend_enable = VK_FALSE;
   resource->graphics_sample_mask = ~0u;
   resource->graphics_color_write_mask = VK_COLOR_COMPONENT_R_BIT |
                                         VK_COLOR_COMPONENT_G_BIT |
                                         VK_COLOR_COMPONENT_B_BIT |
                                         VK_COLOR_COMPONENT_A_BIT;
   resource->graphics_src_color_blend_factor = VK_BLEND_FACTOR_ONE;
   resource->graphics_dst_color_blend_factor = VK_BLEND_FACTOR_ZERO;
   resource->graphics_color_blend_op = VK_BLEND_OP_ADD;
   resource->graphics_src_alpha_blend_factor = VK_BLEND_FACTOR_ONE;
   resource->graphics_dst_alpha_blend_factor = VK_BLEND_FACTOR_ZERO;
   resource->graphics_alpha_blend_op = VK_BLEND_OP_ADD;
}

bool
yttrium_venus2_create_shader_module(struct yttrium_venus *venus,
                                   struct yttrium_venus_object *obj,
                                   VkShaderModule *shader,
                                   const uint32_t *code,
                                   size_t code_size,
                                   const char *label,
                                   VkResult *out_result)
{
   if (!yttrium_venus_ensure_initialized(venus)) {
      if (out_result)
         *out_result = VK_ERROR_INITIALIZATION_FAILED;
      if (shader)
         *shader = VK_NULL_HANDLE;
      if (obj)
         memset(obj, 0, sizeof(*obj));
      YTTRIUM_WARN("yttrium: Venus vkCreateShaderModule %s rejected before Venus init code_size=%llu\n",
                   label ? label : "unknown",
                   (unsigned long long)code_size);
      return false;
   }

   yttrium_venus_init_object(venus, obj);
   *shader = YTTRIUM_VENUS_HANDLE(VkShaderModule, obj);

   const VkShaderModuleCreateInfo shader_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = code_size,
      .pCode = code,
   };
   /*
    * Synchronous on purpose.  This used to be vn_async_ with the result
    * hard-coded to VK_SUCCESS, so a create the host never received - a ring
    * write that did not fit, or SPIR-V the host rejected - left the driver
    * believing the module existed.  The damage surfaced thousands of commands
    * later as vkCreateGraphicsPipelines failing to look up the object, then a
    * fifteen second wait-seqno timeout, with nothing in either log naming the
    * shader.
    *
    * Shader modules are created once per shader, not per draw, so the round
    * trip does not show up in a frame.
    */
   const VkResult result =
      vn_call_vkCreateShaderModule(&venus->vn_ring, venus->device_handle,
                                   &shader_info, NULL, shader);
   if (out_result)
      *out_result = result;

   if (result != VK_SUCCESS) {
      YTTRIUM_WARN("yttrium: Venus vkCreateShaderModule %s failed result=%d code_size=%llu id=%llu\n",
                   label ? label : "unknown", result,
                   (unsigned long long)code_size,
                   (unsigned long long)obj->id);
      *shader = VK_NULL_HANDLE;
      memset(obj, 0, sizeof(*obj));
      return false;
   }

   return true;
}

void
yttrium_venus2_destroy_shader_module(struct yttrium_venus *venus,
                                    struct yttrium_venus_object *obj,
                                    VkShaderModule shader)
{
   if (!venus || !venus->initialized || !shader)
      return;

   vn_async_vkDestroyShaderModule(&venus->vn_ring, venus->device_handle,
                                  shader, NULL);
   if (obj)
      memset(obj, 0, sizeof(*obj));
}

void
yttrium_venus2_pipeline_fini(struct yttrium_venus *venus,
                            struct yttrium_pipeline *pipeline)
{
   if (!venus || !venus->initialized || !pipeline)
      return;

   struct yttrium_venus_render_target *render_target_cache_entry =
      pipeline->render_target_cache_entry;
   bool deferred = false;
   struct yttrium_venus_batch *batch = NULL;
   if (venus->display_copy_batch_recording) {
      for (uint32_t i = 0; i < venus->cmd_batch_pending_pipeline_count; i++) {
         if (venus->cmd_batch_pending_pipelines[i] == pipeline) {
            batch = venus->cmd_batch;
            if (!batch) {
               yttrium_venus_flush_command_batch(
                  venus, "pipeline destroy current batch fallback");
               batch = yttrium_venus_find_latest_pipeline_batch(venus,
                                                                 pipeline);
            }
            break;
         }
      }
   }

   if (!batch)
      batch = yttrium_venus_find_latest_pipeline_batch(venus, pipeline);

   if (batch) {
      struct yttrium_venus_retired_resource *retired =
         yttrium_venus_retired_pipeline_create(pipeline);
      if (retired) {
         yttrium_venus_batch_retire_resource(batch, retired);
         deferred = true;
      } else {
         YTTRIUM_WARN("yttrium: Venus pipeline destroy fallback owner=venus2 reason=retired_pipeline_alloc_failed action=synchronous_wait\n");
         yttrium_venus_flush_command_batch(venus,
                                           "pipeline destroy oom fallback");
         yttrium_venus_wait_pipeline_batches(venus, pipeline,
                                             "pipeline destroy oom fallback");
      }
   }

   if (!deferred) {
      if (pipeline->pipeline)
         vn_async_vkDestroyPipeline(&venus->vn_ring, venus->device_handle,
                                    pipeline->pipeline, NULL);
      if (pipeline->push_pipeline)
         vn_async_vkDestroyPipeline(&venus->vn_ring, venus->device_handle,
                                    pipeline->push_pipeline, NULL);
      if (pipeline->pipeline_layout)
         vn_async_vkDestroyPipelineLayout(&venus->vn_ring, venus->device_handle,
                                          pipeline->pipeline_layout, NULL);
      if (pipeline->push_pipeline_layout)
         vn_async_vkDestroyPipelineLayout(&venus->vn_ring,
                                          venus->device_handle,
                                          pipeline->push_pipeline_layout,
                                          NULL);
      if (pipeline->push_pipeline_layout_alt)
         vn_async_vkDestroyPipelineLayout(
            &venus->vn_ring, venus->device_handle,
            pipeline->push_pipeline_layout_alt, NULL);
      if (pipeline->descriptor_pool)
         vn_async_vkDestroyDescriptorPool(&venus->vn_ring, venus->device_handle,
                                          pipeline->descriptor_pool, NULL);
      for (uint32_t i = 0; i < YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES; i++) {
         if (pipeline->samplers[i])
            vn_async_vkDestroySampler(&venus->vn_ring, venus->device_handle,
                                      pipeline->samplers[i], NULL);
      }
      assert(venus->custom_border_color_sampler_count >=
             pipeline->custom_border_color_sampler_count);
      venus->custom_border_color_sampler_count -=
         pipeline->custom_border_color_sampler_count;
      if (pipeline->descriptor_set_layout)
         vn_async_vkDestroyDescriptorSetLayout(&venus->vn_ring,
                                               venus->device_handle,
                                               pipeline->descriptor_set_layout,
                                               NULL);
      if (pipeline->push_descriptor_set_layout)
         vn_async_vkDestroyDescriptorSetLayout(
            &venus->vn_ring, venus->device_handle,
            pipeline->push_descriptor_set_layout, NULL);
      if (pipeline->push_descriptor_set_layout_alt)
         vn_async_vkDestroyDescriptorSetLayout(
            &venus->vn_ring, venus->device_handle,
            pipeline->push_descriptor_set_layout_alt, NULL);
      if (!render_target_cache_entry && pipeline->framebuffer)
         vn_async_vkDestroyFramebuffer(&venus->vn_ring, venus->device_handle,
                                       pipeline->framebuffer, NULL);
      if (!render_target_cache_entry && pipeline->render_pass)
         vn_async_vkDestroyRenderPass(&venus->vn_ring, venus->device_handle,
                                      pipeline->render_pass, NULL);
      if (!render_target_cache_entry && pipeline->depth_image_view)
         vn_async_vkDestroyImageView(&venus->vn_ring, venus->device_handle,
                                     pipeline->depth_image_view, NULL);
      for (uint32_t i = 0; !render_target_cache_entry &&
                           i < PIPE_MAX_COLOR_BUFS; i++) {
         if (pipeline->image_views[i])
            vn_async_vkDestroyImageView(&venus->vn_ring, venus->device_handle,
                                        pipeline->image_views[i], NULL);
      }
   }

   if (render_target_cache_entry && !deferred)
      yttrium_venus_render_target_release(venus,
                                          render_target_cache_entry);

   memset(pipeline->image_view_objs, 0, sizeof(pipeline->image_view_objs));
   memset(&pipeline->depth_image_view_obj, 0,
          sizeof(pipeline->depth_image_view_obj));
   memset(&pipeline->render_pass_obj, 0, sizeof(pipeline->render_pass_obj));
   memset(&pipeline->framebuffer_obj, 0, sizeof(pipeline->framebuffer_obj));
   memset(&pipeline->descriptor_set_layout_obj, 0,
          sizeof(pipeline->descriptor_set_layout_obj));
   memset(&pipeline->descriptor_pool_obj, 0,
          sizeof(pipeline->descriptor_pool_obj));
   memset(&pipeline->descriptor_set_obj, 0,
          sizeof(pipeline->descriptor_set_obj));
   memset(pipeline->sampler_objs, 0, sizeof(pipeline->sampler_objs));
   memset(&pipeline->pipeline_layout_obj, 0,
          sizeof(pipeline->pipeline_layout_obj));
   memset(&pipeline->pipeline_obj, 0, sizeof(pipeline->pipeline_obj));
   memset(&pipeline->push_descriptor_set_layout_obj, 0,
          sizeof(pipeline->push_descriptor_set_layout_obj));
   memset(&pipeline->push_descriptor_set_layout_alt_obj, 0,
          sizeof(pipeline->push_descriptor_set_layout_alt_obj));
   memset(&pipeline->push_pipeline_layout_obj, 0,
          sizeof(pipeline->push_pipeline_layout_obj));
   memset(&pipeline->push_pipeline_layout_alt_obj, 0,
          sizeof(pipeline->push_pipeline_layout_alt_obj));
   memset(&pipeline->push_pipeline_obj, 0,
          sizeof(pipeline->push_pipeline_obj));
   for (uint32_t i = 0; i < pipeline->ubo_count; i++) {
      memset(&pipeline->ubos[i], 0, sizeof(pipeline->ubos[i]));
   }
   memset(pipeline->image_views, 0, sizeof(pipeline->image_views));
   pipeline->color_attachment_count = 0;
   pipeline->render_pass = VK_NULL_HANDLE;
   pipeline->framebuffer = VK_NULL_HANDLE;
   pipeline->descriptor_set_layout = VK_NULL_HANDLE;
   pipeline->descriptor_pool = VK_NULL_HANDLE;
   pipeline->descriptor_set = VK_NULL_HANDLE;
   memset(pipeline->samplers, 0, sizeof(pipeline->samplers));
   pipeline->custom_border_color_sampler_count = 0;
   pipeline->pipeline_layout = VK_NULL_HANDLE;
   pipeline->pipeline = VK_NULL_HANDLE;
   pipeline->push_descriptor_set_layout = VK_NULL_HANDLE;
   pipeline->push_descriptor_set_layout_alt = VK_NULL_HANDLE;
   pipeline->push_pipeline_layout = VK_NULL_HANDLE;
   pipeline->push_pipeline_layout_alt = VK_NULL_HANDLE;
   pipeline->push_pipeline = VK_NULL_HANDLE;
   pipeline->depth_image_view = VK_NULL_HANDLE;
   pipeline->render_target_cache_entry = NULL;
   pipeline->ubo_count = 0;
   pipeline->ubo_descriptor_count = 0;
   pipeline->sampled_image_descriptor_count = 0;
   pipeline->sampled_buffer_descriptor_count = 0;
   pipeline->storage_image_descriptor_count = 0;
   pipeline->storage_buffer_descriptor_count = 0;
   pipeline->storage_image_mask = 0;
   pipeline->storage_buffer_mask = 0;
   pipeline->has_sampled_image = false;
   pipeline->has_sampled_buffer = false;
   pipeline->has_storage_image = false;
   pipeline->has_storage_buffer = false;
}

static bool
yttrium_venus_pipeline_prepare_ubo_descriptors(
   struct yttrium_venus *venus,
   struct yttrium_pipeline *pipeline,
   const struct yttrium_venus_ubo_binding_layout *ubo_bindings,
   uint32_t ubo_binding_count,
   const struct yttrium_venus_sampled_binding_layout *sampled_bindings,
   uint32_t sampled_binding_count,
   uint64_t storage_image_mask,
   uint64_t storage_buffer_mask,
   VkShaderStageFlags storage_stage_flags,
   bool allow_push_layout_rotation)
{
   const uint64_t storage_mask = storage_image_mask | storage_buffer_mask;
   if (!storage_stage_flags)
      storage_stage_flags = VK_SHADER_STAGE_FRAGMENT_BIT;

   if (!ubo_binding_count && !sampled_binding_count && !storage_mask)
      return true;

   if (!pipeline || (ubo_binding_count && !ubo_bindings) ||
       ubo_binding_count > YTTRIUM_VENUS_MAX_PIPELINE_UBO_BINDINGS ||
       (sampled_binding_count && !sampled_bindings) ||
       sampled_binding_count > YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES)
      return false;
   if ((storage_mask & ~YTTRIUM_VENUS_PIPELINE_STORAGE_IMAGE_MASK) ||
       (storage_image_mask & storage_buffer_mask)) {
      YTTRIUM_LOG("yttrium: Venus native descriptors rejected storage masks image=0x%llx buffer=0x%llx supported=0x%llx\n",
                  (unsigned long long)storage_image_mask,
                  (unsigned long long)storage_buffer_mask,
                  (unsigned long long)YTTRIUM_VENUS_PIPELINE_STORAGE_IMAGE_MASK);
      return false;
   }

   VkDescriptorSetLayoutBinding bindings
      [YTTRIUM_VENUS_MAX_PIPELINE_UBO_BINDINGS +
       YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES +
       YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES];
   uint32_t binding_count = 0;
   uint32_t ubo_descriptor_count = 0;
   uint32_t sampled_image_descriptor_count = 0;
   uint32_t sampled_buffer_descriptor_count = 0;
   uint32_t storage_image_descriptor_count = 0;
   uint32_t storage_buffer_descriptor_count = 0;
   memset(bindings, 0, sizeof(bindings));
   pipeline->ubo_count = 0;
   pipeline->ubo_descriptor_count = 0;
   pipeline->sampled_image_descriptor_count = 0;
   pipeline->sampled_buffer_descriptor_count = 0;
   pipeline->storage_image_descriptor_count = 0;
   pipeline->storage_buffer_descriptor_count = 0;
   pipeline->storage_image_mask = 0;
   pipeline->storage_buffer_mask = 0;
   pipeline->has_sampled_image = false;
   pipeline->has_sampled_buffer = false;
   pipeline->has_storage_image = false;
   pipeline->has_storage_buffer = false;

   for (uint32_t i = 0; i < ubo_binding_count; i++) {
      if (!ubo_bindings[i].descriptor_count)
         return false;
      if (pipeline->ubo_count + ubo_bindings[i].descriptor_count >
          YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS)
         return false;
      for (uint32_t sampled_index = 0;
           sampled_index < sampled_binding_count; sampled_index++) {
         if (ubo_bindings[i].binding ==
             sampled_bindings[sampled_index].binding) {
            YTTRIUM_LOG("yttrium: Venus native descriptors rejected sampled binding collision ubo_binding=%u sampled_index=%u sampled_binding=%u ubos=%u\n",
                        ubo_bindings[i].binding, sampled_index,
                        sampled_bindings[sampled_index].binding,
                        ubo_binding_count);
            return false;
         }
      }
      for (uint32_t slot = 0;
           slot < YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES; slot++) {
         if (!(storage_mask & (1ull << slot)))
            continue;
         const uint32_t image_binding =
            yttrium_shader_storage_image_binding(slot);
         if (image_binding == UINT32_MAX)
            return false;
         if (ubo_bindings[i].binding == image_binding) {
            YTTRIUM_LOG("yttrium: Venus native descriptors rejected storage binding collision ubo_binding=%u image_slot=%u ubos=%u image_mask=0x%llx buffer_mask=0x%llx\n",
                        ubo_bindings[i].binding, slot, ubo_binding_count,
                        (unsigned long long)storage_image_mask,
                        (unsigned long long)storage_buffer_mask);
            return false;
         }
      }

      bindings[binding_count++] = (VkDescriptorSetLayoutBinding) {
         .binding = ubo_bindings[i].binding,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = ubo_bindings[i].descriptor_count,
         .stageFlags = ubo_bindings[i].stage_flags,
      };
      ubo_descriptor_count += ubo_bindings[i].descriptor_count;

      for (uint32_t j = 0; j < ubo_bindings[i].descriptor_count; j++) {
         struct yttrium_venus_ubo_slot *slot =
            &pipeline->ubos[pipeline->ubo_count++];
         slot->binding = ubo_bindings[i].binding;
         slot->array_element = j;
      }
   }

   for (uint32_t sampled_index = 0;
        sampled_index < sampled_binding_count; sampled_index++) {
      const struct yttrium_venus_sampled_binding_layout *sampled =
         &sampled_bindings[sampled_index];
      const mesa_shader_stage stage = (mesa_shader_stage)sampled->stage;
      const uint32_t sampled_binding =
         yttrium_shader_sampler_binding(stage, sampled->raw_slot);
      if (stage < MESA_SHADER_VERTEX || stage > MESA_SHADER_COMPUTE ||
          !sampled->stage_flags || sampled_binding == UINT32_MAX ||
          sampled_binding != sampled->binding)
         return false;
      for (uint32_t prior = 0; prior < sampled_index; prior++) {
         if (sampled_bindings[prior].binding == sampled->binding) {
            YTTRIUM_LOG("yttrium: Venus native descriptors rejected duplicate sampled binding index=%u prior=%u binding=%u\n",
                        sampled_index, prior, sampled->binding);
            return false;
         }
      }

      bindings[binding_count++] = (VkDescriptorSetLayoutBinding) {
         .binding = sampled_binding,
         .descriptorType =
            sampled->buffer ?
               VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER :
               VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = sampled->stage_flags,
      };
      if (sampled->buffer)
         sampled_buffer_descriptor_count++;
      else
         sampled_image_descriptor_count++;
   }
   pipeline->has_sampled_image = sampled_image_descriptor_count != 0;
   pipeline->has_sampled_buffer = sampled_buffer_descriptor_count != 0;
   for (uint32_t slot = 0;
        slot < YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES; slot++) {
      if (!(storage_mask & (1ull << slot)))
         continue;
      const uint32_t image_binding =
         yttrium_shader_storage_image_binding(slot);
      if (image_binding == UINT32_MAX)
         return false;

      bindings[binding_count++] = (VkDescriptorSetLayoutBinding) {
         .binding = image_binding,
         .descriptorType =
            (storage_buffer_mask & (1ull << slot)) ?
               VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER :
               VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .descriptorCount = 1,
         .stageFlags = storage_stage_flags,
      };
      if (storage_buffer_mask & (1ull << slot))
         storage_buffer_descriptor_count++;
      else
         storage_image_descriptor_count++;
   }
   pipeline->storage_image_mask = storage_image_mask;
   pipeline->storage_buffer_mask = storage_buffer_mask;
   pipeline->has_storage_image = storage_image_mask != 0;
   pipeline->has_storage_buffer = storage_buffer_mask != 0;

   yttrium_venus_init_object(venus, &pipeline->descriptor_set_layout_obj);
   pipeline->descriptor_set_layout =
      YTTRIUM_VENUS_HANDLE(VkDescriptorSetLayout,
                            &pipeline->descriptor_set_layout_obj);
   const VkDescriptorSetLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = binding_count,
      .pBindings = bindings,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateDescriptorSetLayout,
      &pipeline->descriptor_set_layout_obj, &pipeline->descriptor_set_layout,
      return false,
      venus->device_handle, &layout_info, NULL,
      &pipeline->descriptor_set_layout);
   const uint32_t descriptor_count =
      ubo_descriptor_count + sampled_image_descriptor_count +
      sampled_buffer_descriptor_count + storage_image_descriptor_count +
      storage_buffer_descriptor_count;
   const bool push_descriptor_supported =
      venus->push_descriptor && descriptor_count &&
      descriptor_count <= venus->max_push_descriptors;
   if (push_descriptor_supported) {
      yttrium_venus_init_object(
         venus, &pipeline->push_descriptor_set_layout_obj);
      pipeline->push_descriptor_set_layout =
         YTTRIUM_VENUS_HANDLE(
            VkDescriptorSetLayout,
            &pipeline->push_descriptor_set_layout_obj);
      const VkDescriptorSetLayoutCreateInfo push_layout_info = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
         .bindingCount = binding_count,
         .pBindings = bindings,
      };
      YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
         venus, vkCreateDescriptorSetLayout,
         &pipeline->push_descriptor_set_layout_obj,
         &pipeline->push_descriptor_set_layout, return false,
         venus->device_handle, &push_layout_info, NULL,
         &pipeline->push_descriptor_set_layout);
      if (allow_push_layout_rotation && descriptor_count > 1 &&
          yttrium_venus_push_descriptor_layout_rotation_enabled()) {
         yttrium_venus_init_object(
            venus, &pipeline->push_descriptor_set_layout_alt_obj);
         pipeline->push_descriptor_set_layout_alt =
            YTTRIUM_VENUS_HANDLE(
               VkDescriptorSetLayout,
               &pipeline->push_descriptor_set_layout_alt_obj);
         YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
            venus, vkCreateDescriptorSetLayout,
            &pipeline->push_descriptor_set_layout_alt_obj,
            &pipeline->push_descriptor_set_layout_alt, return false,
            venus->device_handle, &push_layout_info, NULL,
            &pipeline->push_descriptor_set_layout_alt);
      }
   }

   yttrium_venus_init_object(venus, &pipeline->descriptor_pool_obj);
   pipeline->descriptor_pool =
      YTTRIUM_VENUS_HANDLE(VkDescriptorPool,
                            &pipeline->descriptor_pool_obj);
   VkDescriptorPoolSize pool_sizes[5];
   uint32_t pool_size_count = 0;
   memset(pool_sizes, 0, sizeof(pool_sizes));
   if (ubo_descriptor_count) {
      pool_sizes[pool_size_count++] = (VkDescriptorPoolSize) {
         .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = ubo_descriptor_count,
      };
   }
   if (sampled_image_descriptor_count) {
      pool_sizes[pool_size_count++] = (VkDescriptorPoolSize) {
         .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = sampled_image_descriptor_count,
      };
   }
   if (sampled_buffer_descriptor_count) {
      pool_sizes[pool_size_count++] = (VkDescriptorPoolSize) {
         .type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
         .descriptorCount = sampled_buffer_descriptor_count,
      };
   }
   if (storage_image_descriptor_count) {
      pool_sizes[pool_size_count++] = (VkDescriptorPoolSize) {
         .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .descriptorCount = storage_image_descriptor_count,
      };
   }
   if (storage_buffer_descriptor_count) {
      pool_sizes[pool_size_count++] = (VkDescriptorPoolSize) {
         .type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
         .descriptorCount = storage_buffer_descriptor_count,
      };
   }
   const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = pool_size_count,
      .pPoolSizes = pool_sizes,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateDescriptorPool, &pipeline->descriptor_pool_obj,
      &pipeline->descriptor_pool, return false, venus->device_handle,
      &pool_info, NULL,
      &pipeline->descriptor_pool);

   yttrium_venus_init_object(venus, &pipeline->descriptor_set_obj);
   pipeline->descriptor_set =
      YTTRIUM_VENUS_HANDLE(VkDescriptorSet,
                            &pipeline->descriptor_set_obj);
   const VkDescriptorSetAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pipeline->descriptor_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &pipeline->descriptor_set_layout,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkAllocateDescriptorSets, &pipeline->descriptor_set_obj,
      &pipeline->descriptor_set, return false, venus->device_handle, &alloc_info,
      &pipeline->descriptor_set);
   pipeline->ubo_descriptor_count = ubo_descriptor_count;
   pipeline->sampled_image_descriptor_count =
      sampled_image_descriptor_count;
   pipeline->sampled_buffer_descriptor_count =
      sampled_buffer_descriptor_count;
   pipeline->storage_image_descriptor_count =
      storage_image_descriptor_count;
   pipeline->storage_buffer_descriptor_count =
      storage_buffer_descriptor_count;

   if (sampled_image_descriptor_count) {
      for (uint32_t sampled_index = 0;
           sampled_index < sampled_binding_count; sampled_index++) {
         const struct yttrium_venus_sampled_binding_layout *layout =
            &sampled_bindings[sampled_index];
         if (layout->buffer)
            continue;

         const struct yttrium_venus_sampler_state *sampler =
            &layout->sampler;
         const VkBorderColor border_color = sampler->border_color;
         const bool custom_border =
            border_color == VK_BORDER_COLOR_FLOAT_CUSTOM_EXT ||
            border_color == VK_BORDER_COLOR_INT_CUSTOM_EXT;
         if (custom_border &&
             (!venus->custom_border_colors ||
              !venus->custom_border_color_without_format)) {
            YTTRIUM_WARN("yttrium: Venus sampler rejected owner=venus2-sampler reason=custom-border-color-without-format-unavailable stage=%u raw_slot=%u border_color=%u\n",
                         layout->stage, layout->raw_slot, border_color);
            return false;
         }
         if (custom_border && venus->custom_border_color_sampler_count >=
                              venus->max_custom_border_color_samplers) {
            YTTRIUM_WARN("yttrium: Venus sampler rejected owner=venus2-sampler reason=max-custom-border-color-samplers stage=%u raw_slot=%u live=%u limit=%u\n",
                         layout->stage, layout->raw_slot,
                         venus->custom_border_color_sampler_count,
                         venus->max_custom_border_color_samplers);
            return false;
         }
         const VkSamplerCustomBorderColorCreateInfoEXT custom_border_info = {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT,
            .customBorderColor = sampler->custom_border_color,
            .format = VK_FORMAT_UNDEFINED,
         };
         if (sampler && sampler->anisotropy_enable &&
             sampler->max_anisotropy > 1.0f &&
             venus->max_sampler_anisotropy <= 1.0f) {
            static volatile LONG warned;

            if (InterlockedCompareExchange(&warned, 1, 0) == 0) {
               YTTRIUM_WARN(
                  "yttrium: WARNING: anisotropic sampler fallback "
                  "owner=venus2-sampler "
                  "reason=host-samplerAnisotropy-feature-unavailable "
                  "fallback=linear-filtering requested=%.1f supported=%.1f\n",
                  sampler->max_anisotropy,
                  venus->max_sampler_anisotropy);
            }
         }
         const bool anisotropy_enabled = sampler &&
            venus->max_sampler_anisotropy > 1.0f &&
            sampler->anisotropy_enable && sampler->max_anisotropy > 1.0f;
         const float requested_mip_lod_bias =
            sampler ? sampler->mip_lod_bias : 0.0f;
         float effective_mip_lod_bias = requested_mip_lod_bias;
         if (!(effective_mip_lod_bias == effective_mip_lod_bias))
            effective_mip_lod_bias = 0.0f;
         else
            effective_mip_lod_bias =
               CLAMP(effective_mip_lod_bias,
                     -venus->max_sampler_lod_bias,
                     venus->max_sampler_lod_bias);
         if (!(effective_mip_lod_bias == requested_mip_lod_bias)) {
            static volatile LONG warned;

            if (InterlockedCompareExchange(&warned, 1, 0) == 0) {
               YTTRIUM_WARN(
                  "yttrium: WARNING: sampler LOD bias clamped "
                  "owner=venus2-sampler "
                  "reason=request-outside-host-limits "
                  "fallback=clamp requested=%g effective=%g limit=%g\n",
                  requested_mip_lod_bias,
                  effective_mip_lod_bias,
                  venus->max_sampler_lod_bias);
            }
         }
         const VkSamplerCreateInfo sampler_info = {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = custom_border ? &custom_border_info : NULL,
            .magFilter = sampler ? sampler->mag_filter : VK_FILTER_NEAREST,
            .minFilter = sampler ? sampler->min_filter : VK_FILTER_NEAREST,
            .mipmapMode = sampler ? sampler->mipmap_mode :
               VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = sampler ? sampler->address_mode_u :
               VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = sampler ? sampler->address_mode_v :
               VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = sampler ? sampler->address_mode_w :
               VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .mipLodBias = effective_mip_lod_bias,
            .anisotropyEnable = anisotropy_enabled,
            .maxAnisotropy = anisotropy_enabled ?
               MIN2(sampler->max_anisotropy,
                    venus->max_sampler_anisotropy) : 1.0f,
            .compareEnable = sampler ? sampler->compare_enable : VK_FALSE,
            .compareOp = sampler ? sampler->compare_op :
               VK_COMPARE_OP_ALWAYS,
            .minLod = sampler ? sampler->min_lod : 0.0f,
            .maxLod = sampler ? sampler->max_lod : 0.0f,
            .borderColor = border_color,
         };

         yttrium_venus_init_object(
            venus, &pipeline->sampler_objs[sampled_index]);
         pipeline->samplers[sampled_index] =
            YTTRIUM_VENUS_HANDLE(VkSampler,
                                  &pipeline->sampler_objs[sampled_index]);
         YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
            venus, vkCreateSampler,
            &pipeline->sampler_objs[sampled_index],
            &pipeline->samplers[sampled_index], return false,
            venus->device_handle,
            &sampler_info, NULL,
            &pipeline->samplers[sampled_index]);
         if (custom_border) {
            venus->custom_border_color_sampler_count++;
            pipeline->custom_border_color_sampler_count++;
         }
      }
   }

   YTTRIUM_LOG("yttrium: Venus native descriptors set_layout_id=%llu push_set_layout_id=%llu pool_id=%llu set_id=%llu bindings=%u descriptors=%u ubos=%u sampled_bindings=%u storage_image_mask=0x%llx storage_buffer_mask=0x%llx sampled_images=%u sampled_buffers=%u storage_images=%u storage_buffers=%u push_supported=%u max_push=%u\n",
                (unsigned long long)pipeline->descriptor_set_layout_obj.id,
                (unsigned long long)
                   pipeline->push_descriptor_set_layout_obj.id,
                (unsigned long long)pipeline->descriptor_pool_obj.id,
                (unsigned long long)pipeline->descriptor_set_obj.id,
                binding_count, descriptor_count, ubo_descriptor_count,
                sampled_binding_count,
                (unsigned long long)storage_image_mask,
                (unsigned long long)storage_buffer_mask,
                sampled_image_descriptor_count,
                sampled_buffer_descriptor_count,
                storage_image_descriptor_count,
                storage_buffer_descriptor_count,
                push_descriptor_supported,
                venus->max_push_descriptors);
   return true;
}

static uint32_t
yttrium_venus_render_target_cache_configured_capacity(void)
{
   static bool initialized;
   static uint32_t capacity;

   if (!initialized) {
      int64_t configured = yttrium_gdi_debug_get_num_option(
         "D3D10UMD_YTTRIUM_RENDER_TARGET_CACHE_SIZE", 256);
      if (configured < 0)
         configured = 0;
      if (configured > YTTRIUM_VENUS_RENDER_TARGET_CACHE_MAX)
         configured = YTTRIUM_VENUS_RENDER_TARGET_CACHE_MAX;
      capacity = (uint32_t)configured;
      initialized = true;
   }

   return capacity;
}

static void
yttrium_venus_render_target_destroy(
   struct yttrium_venus *venus,
   struct yttrium_venus_render_target *target)
{
   if (!target)
      return;

   if (venus && venus->initialized) {
      if (target->framebuffer)
         vn_async_vkDestroyFramebuffer(&venus->vn_ring, venus->device_handle,
                                       target->framebuffer, NULL);
      if (target->render_pass)
         vn_async_vkDestroyRenderPass(&venus->vn_ring, venus->device_handle,
                                      target->render_pass, NULL);
      if (target->depth_image_view)
         vn_async_vkDestroyImageView(&venus->vn_ring, venus->device_handle,
                                     target->depth_image_view, NULL);
      for (uint32_t i = 0; i < PIPE_MAX_COLOR_BUFS; i++) {
         if (target->image_views[i])
            vn_async_vkDestroyImageView(&venus->vn_ring,
                                        venus->device_handle,
                                        target->image_views[i], NULL);
      }
   }

   FREE(target);
}

void
yttrium_venus_render_target_release(
   struct yttrium_venus *venus,
   struct yttrium_venus_render_target *target)
{
   if (!target)
      return;

   if (!target->refcount) {
      YTTRIUM_WARN("yttrium: Venus render-target cache refcount underflow owner=venus2 framebuffer_id=%llu\n",
                   (unsigned long long)target->framebuffer_obj.id);
      return;
   }

   target->refcount--;
   if (!target->refcount && !target->cached)
      yttrium_venus_render_target_destroy(venus, target);
}

static struct yttrium_venus_render_target *
yttrium_venus_render_target_lookup(
   struct yttrium_venus *venus,
   const struct yttrium_venus_render_target_key *key)
{
   if (!venus || !venus->render_target_cache ||
       !venus->render_target_cache_capacity)
      return NULL;

   for (uint32_t i = 0; i < venus->render_target_cache_capacity; i++) {
      struct yttrium_venus_render_target *target =
         venus->render_target_cache[i];
      if (!target || memcmp(&target->key, key, sizeof(*key)))
         continue;

      target->refcount++;
      return target;
   }

   return NULL;
}

static void
yttrium_venus_render_target_assign(
   struct yttrium_pipeline *pipeline,
   struct yttrium_venus_render_target *target)
{
   pipeline->render_target_cache_entry = target;
   memcpy(pipeline->image_view_objs, target->image_view_objs,
          sizeof(pipeline->image_view_objs));
   pipeline->depth_image_view_obj = target->depth_image_view_obj;
   pipeline->render_pass_obj = target->render_pass_obj;
   pipeline->framebuffer_obj = target->framebuffer_obj;
   memcpy(pipeline->image_views, target->image_views,
          sizeof(pipeline->image_views));
   pipeline->depth_image_view = target->depth_image_view;
   pipeline->render_pass = target->render_pass;
   pipeline->framebuffer = target->framebuffer;
   pipeline->color_attachment_count = target->key.color_attachment_count;
}

static struct yttrium_venus_render_target *
yttrium_venus_render_target_adopt(
   struct yttrium_venus *venus,
   const struct yttrium_venus_render_target_key *key,
   struct yttrium_pipeline *pipeline)
{
   if (!venus || !pipeline || !venus->render_target_cache ||
       !venus->render_target_cache_capacity)
      return NULL;

   uint32_t slot = venus->render_target_cache_capacity;
   for (uint32_t i = 0; i < venus->render_target_cache_capacity; i++) {
      if (!venus->render_target_cache[i]) {
         slot = i;
         break;
      }
   }

   /*
    * Do not evict a cached render pass during the device lifetime.  Graphics
    * pipeline creation is encoded asynchronously, so a pipeline object can
    * have released its CPU-side target reference while the host decoder still
    * needs the render-pass ID from vkCreateGraphicsPipelines.  The first LRU
    * implementation destroyed that ID at the capacity boundary and the host
    * failed object lookup.  A full cache falls back to ordinary pipeline-owned
    * target objects; image destruction and device teardown remain the only
    * cache-removal points.
    */
   if (slot == venus->render_target_cache_capacity)
      return NULL;

   struct yttrium_venus_render_target *target =
      CALLOC_STRUCT(yttrium_venus_render_target);
   if (!target)
      return NULL;

   target->key = *key;
   memcpy(target->image_view_objs, pipeline->image_view_objs,
          sizeof(target->image_view_objs));
   target->depth_image_view_obj = pipeline->depth_image_view_obj;
   target->render_pass_obj = pipeline->render_pass_obj;
   target->framebuffer_obj = pipeline->framebuffer_obj;
   for (uint32_t i = 0; i < PIPE_MAX_COLOR_BUFS; i++) {
      target->image_views[i] = pipeline->image_views[i] ?
         YTTRIUM_VENUS_HANDLE(VkImageView, &target->image_view_objs[i]) :
         VK_NULL_HANDLE;
   }
   target->depth_image_view = pipeline->depth_image_view ?
      YTTRIUM_VENUS_HANDLE(VkImageView, &target->depth_image_view_obj) :
      VK_NULL_HANDLE;
   target->render_pass = pipeline->render_pass ?
      YTTRIUM_VENUS_HANDLE(VkRenderPass, &target->render_pass_obj) :
      VK_NULL_HANDLE;
   target->framebuffer = pipeline->framebuffer ?
      YTTRIUM_VENUS_HANDLE(VkFramebuffer, &target->framebuffer_obj) :
      VK_NULL_HANDLE;
   target->refcount = 1;
   target->cached = true;
   venus->render_target_cache[slot] = target;
   yttrium_venus_render_target_assign(pipeline, target);
   return target;
}

static void
yttrium_venus_render_target_cache_invalidate_image(
   struct yttrium_venus *venus,
   uint64_t image_id)
{
   if (!venus || !venus->render_target_cache ||
       !venus->render_target_cache_capacity || !image_id)
      return;

   for (uint32_t i = 0; i < venus->render_target_cache_capacity; i++) {
      struct yttrium_venus_render_target *target =
         venus->render_target_cache[i];
      if (!target)
         continue;

      bool matches = target->key.depth_image_id == image_id;
      for (uint32_t c = 0; !matches && c < PIPE_MAX_COLOR_BUFS; c++)
         matches = target->key.color_image_ids[c] == image_id;
      if (!matches)
         continue;

      venus->render_target_cache[i] = NULL;
      target->cached = false;
      if (!target->refcount)
         yttrium_venus_render_target_destroy(venus, target);
   }
}

static void
yttrium_venus_render_target_cache_destroy(struct yttrium_venus *venus)
{
   if (!venus || !venus->render_target_cache ||
       !venus->render_target_cache_capacity)
      return;

   for (uint32_t i = 0; i < venus->render_target_cache_capacity; i++) {
      struct yttrium_venus_render_target *target =
         venus->render_target_cache[i];
      venus->render_target_cache[i] = NULL;
      if (!target)
         continue;
      if (target->refcount)
         YTTRIUM_WARN("yttrium: Venus render-target cache live entry at device destroy owner=venus2 framebuffer_id=%llu refs=%u action=device_teardown\n",
                      (unsigned long long)target->framebuffer_obj.id,
                      target->refcount);
      target->cached = false;
      yttrium_venus_render_target_destroy(venus, target);
   }

   FREE(venus->render_target_cache);
   venus->render_target_cache = NULL;
   venus->render_target_cache_capacity = 0;
}

bool
yttrium_venus2_pipeline_init(struct yttrium_venus *venus,
                            struct yttrium_pipeline *pipeline,
                            struct yttrium_venus_resource *resource,
                            uint32_t resource_id,
                            struct yttrium_venus_resource **color_resources,
                            const uint32_t *color_resource_ids,
                            uint32_t color_resource_count,
                            struct yttrium_venus_resource *depth_resource,
                            uint32_t depth_resource_id,
                            VkShaderModule vertex_shader,
                            VkShaderModule tess_ctrl_shader,
                            VkShaderModule tess_eval_shader,
                            VkShaderModule geometry_shader,
                            VkShaderModule fragment_shader,
                            const VkVertexInputBindingDescription *bindings,
                            uint32_t binding_count,
                            const uint32_t *binding_divisors,
                            const VkVertexInputAttributeDescription *attribs,
                            uint32_t attrib_count,
                            const struct yttrium_venus_ubo_binding_layout *ubo_bindings,
                            uint32_t ubo_binding_count,
                            const struct yttrium_venus_sampled_binding_layout *sampled_bindings,
                            uint32_t sampled_binding_count,
                            uint64_t storage_image_mask,
                            uint64_t storage_buffer_mask,
                            VkShaderStageFlags storage_stage_flags,
                            const struct yttrium_venus_draw_state *draw_state)
{
   const bool has_depth = depth_resource != NULL;

   if (!yttrium_venus_ensure_initialized(venus))
      return false;
   if (color_resource_count > PIPE_MAX_COLOR_BUFS)
      return false;

   bool has_color = false;
   for (uint32_t i = 0; i < color_resource_count; i++) {
      if (color_resources && color_resources[i]) {
         has_color = true;
         break;
      }
   }
   /* The actual target buffers are validated at draw time.  This exception
    * only permits the pipeline shape explicitly tagged by the frontend. */
   const bool targetless_stream_output =
      draw_state && draw_state->targetless_stream_output &&
      pipeline && pipeline->key.targetless_stream_output &&
      !color_resource_count && !has_color && !has_depth &&
      venus->transform_feedback;
   const uint32_t color_feedback_loop_mask =
      pipeline ? pipeline->key.color_feedback_loop_mask : 0;
   const bool depth_feedback_loop =
      pipeline && pipeline->key.depth_feedback_loop;
   const bool depth_read_only = pipeline && pipeline->key.depth_read_only;
   if (depth_read_only &&
       (!venus->load_store_op_none || !has_depth || depth_feedback_loop)) {
      YTTRIUM_WARN("yttrium: Venus native pipeline rejected owner=venus2 reason=invalid_read_only_depth store_none=%u has_depth=%u depth_feedback=%u\n",
                   venus->load_store_op_none, has_depth, depth_feedback_loop);
      return false;
   }
   if ((color_feedback_loop_mask || depth_feedback_loop) &&
       (!venus->attachment_feedback_loop_layout ||
        (color_feedback_loop_mask &
         ~BITFIELD_MASK(color_resource_count)) ||
        (depth_feedback_loop && !has_depth))) {
      YTTRIUM_WARN("yttrium: Venus native pipeline rejected owner=venus2 reason=attachment_feedback_loop_unavailable feature=%u color_mask=0x%x color_count=%u depth=%u has_depth=%u\n",
                   venus->attachment_feedback_loop_layout,
                   color_feedback_loop_mask, color_resource_count,
                   depth_feedback_loop, has_depth);
      return false;
   }
   const bool storage_buffer_target =
      resource && !has_color && !has_depth && storage_buffer_mask;
   const bool render_image_target =
      resource && resource->initialized && !resource->buffer_backed &&
      resource->image && resource->vk_format != VK_FORMAT_UNDEFINED;

   const uint32_t render_level = draw_state ? draw_state->render_level : 0;
   const uint32_t render_layer = draw_state ? draw_state->render_layer : 0;
   const uint32_t render_layers =
      draw_state ? MAX2(draw_state->render_layers, 1) : 1;
   const uint32_t depth_level = draw_state ? draw_state->depth_level : 0;
   const uint32_t depth_layer = draw_state ? draw_state->depth_layer : 0;
   const uint32_t depth_layers =
      draw_state ? MAX2(draw_state->depth_layers, 1) : 1;
   const uint32_t render_width =
      draw_state && draw_state->render_width ?
      draw_state->render_width :
      yttrium_venus_subresource_width(resource, render_level);
   const uint32_t render_height =
      draw_state && draw_state->render_height ?
      draw_state->render_height :
      yttrium_venus_subresource_height(resource, render_level);

   if (!pipeline || !resource ||
       (!storage_buffer_target && !render_image_target) ||
       (!has_color && !has_depth && !storage_image_mask &&
        !storage_buffer_mask && !targetless_stream_output) ||
       (has_color && (!color_resources || !color_resource_ids)) ||
       !render_width || !render_height ||
       (render_image_target &&
        !yttrium_venus_valid_render_subresource(resource, render_level,
                                                render_layer, render_layers)) ||
       !vertex_shader ||
       (binding_count && !bindings) || (attrib_count && !attribs) ||
       !draw_state ||
       ubo_binding_count > YTTRIUM_VENUS_MAX_PIPELINE_UBO_BINDINGS) {
      YTTRIUM_WARN("yttrium: Venus native pipeline rejected res_id=%u initialized=%u buffer_backed=%u usage=0x%x format=%u extent=%ux%u subresource=%u/%u+%u sub_extent=%ux%u levels=%u layers=%u vs=0x%llx fs=0x%llx bindings=%u attribs=%u ubos=%u sampled_bindings=%u storage_image_mask=0x%llx storage_buffer_mask=0x%llx targetless_so=%u tf=%u\n",
                   resource_id,
                   resource ? resource->initialized : 0,
                   resource ? resource->buffer_backed : 0,
                   resource ? resource->image_usage : 0,
                   resource ? resource->vk_format : VK_FORMAT_UNDEFINED,
                   resource ? resource->width : 0,
                   resource ? resource->height : 0,
                   render_level, render_layer, render_layers,
                   render_width, render_height,
                   resource ? resource->levels : 0,
                   resource ? resource->layers : 0,
                   (unsigned long long)YTTRIUM_VENUS_HANDLE_TO_U64(
                      vertex_shader),
                   (unsigned long long)YTTRIUM_VENUS_HANDLE_TO_U64(
                      fragment_shader),
                   binding_count, attrib_count,
                   ubo_binding_count, sampled_binding_count,
                   (unsigned long long)storage_image_mask,
                   (unsigned long long)storage_buffer_mask,
                   draw_state ? draw_state->targetless_stream_output : 0,
                   venus->transform_feedback);
      return false;
   }

   if (draw_state->targetless_stream_output &&
       !targetless_stream_output) {
      YTTRIUM_WARN("yttrium: Venus native pipeline rejected owner=venus2 reason=invalid_targetless_stream_output color_count=%u has_color=%u has_depth=%u tf=%u\n",
                   color_resource_count, has_color, has_depth,
                   venus->transform_feedback);
      return false;
   }

   const bool has_tess_ctrl = tess_ctrl_shader != VK_NULL_HANDLE;
   const bool has_tess_eval = tess_eval_shader != VK_NULL_HANDLE;
   const bool patch_topology =
      draw_state->topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
   const uint32_t patch_vertices = pipeline->key.patch_vertices;
   const bool tessellation =
      has_tess_ctrl || has_tess_eval || patch_topology || patch_vertices;
   const bool valid_patch_vertices =
      patch_vertices && patch_vertices <= venus->max_tessellation_patch_size;
   if (tessellation &&
       (!venus->tessellation_shader || !has_tess_ctrl ||
        !has_tess_eval || !patch_topology ||
        !valid_patch_vertices)) {
      YTTRIUM_WARN("yttrium: tessellation pipeline rejected feature=%u topology=%u tcs=%u tes=%u patch_vertices=%u max_patch_vertices=%u\n",
                   venus->tessellation_shader,
                   draw_state->topology, has_tess_ctrl, has_tess_eval,
                   patch_vertices, venus->max_tessellation_patch_size);
      return false;
   }

   VkSampleCountFlagBits attachment_samples =
      resource->samples ? resource->samples : VK_SAMPLE_COUNT_1_BIT;
   if (has_color) {
      for (uint32_t i = 0; i < color_resource_count; i++) {
         if (color_resources[i]) {
            attachment_samples = color_resources[i]->samples ?
               color_resources[i]->samples : VK_SAMPLE_COUNT_1_BIT;
            break;
         }
      }
   } else if (has_depth) {
      attachment_samples = depth_resource->samples ?
         depth_resource->samples : VK_SAMPLE_COUNT_1_BIT;
   }
   VkSampleCountFlagBits render_samples = attachment_samples;
   VkSampleCountFlagBits forced_samples = VK_SAMPLE_COUNT_1_BIT;
   const bool use_forced_sample_interlock =
      draw_state->forced_sample_interlock &&
      !has_color && !has_depth &&
      draw_state->forced_sample_count > 1 &&
      venus->fragment_shader_pixel_interlock &&
      venus->fragment_stores_and_atomics &&
      yttrium_venus2_sample_count_flag(draw_state->forced_sample_count,
                                       &forced_samples) &&
      (venus->framebuffer_no_attachments_sample_counts & forced_samples);
   if (draw_state->forced_sample_interlock &&
       !use_forced_sample_interlock) {
      YTTRIUM_WARN("yttrium: Venus native pipeline rejected forced-sample interlock samples=%u color=%u depth=%u pixel_interlock=%u fragment_stores=%u no_attachment_counts=0x%x\n",
                  draw_state->forced_sample_count, has_color, has_depth,
                  venus->fragment_shader_pixel_interlock,
                  venus->fragment_stores_and_atomics,
                  venus->framebuffer_no_attachments_sample_counts);
      return false;
   }
   const bool use_mrss =
      !use_forced_sample_interlock &&
      draw_state->forced_sample_count > 1 &&
      attachment_samples == VK_SAMPLE_COUNT_1_BIT &&
      venus->multisampled_render_to_single_sampled &&
      yttrium_venus2_sample_count_flag(draw_state->forced_sample_count,
                                       &forced_samples) &&
      (venus->framebuffer_sample_counts & forced_samples);
   if (use_mrss || use_forced_sample_interlock)
      render_samples = forced_samples;
   /* The deferred draw grouper begins one pipeline-owned render pass and may
    * bind later pipelines inside it.  Preserve the exact compatibility state
    * used to create this render pass; attachment sample counts alone do not
    * distinguish an ordinary single-sampled pass from MRSS, or two MRSS
    * rasterization sample counts. */
   pipeline->use_mrss = use_mrss;
   pipeline->render_samples = render_samples;

   for (uint32_t i = 0; i < color_resource_count; i++) {
      struct yttrium_venus_resource *color = color_resources[i];
      const uint32_t color_id = color_resource_ids[i];
      const uint32_t color_level =
         draw_state && i < ARRAY_SIZE(draw_state->rt_level) ?
            draw_state->rt_level[i] : render_level;
      const uint32_t color_layer =
         draw_state && i < ARRAY_SIZE(draw_state->rt_layer) ?
            draw_state->rt_layer[i] : render_layer;

      if (!color)
         continue;

      if (!color->initialized || color->buffer_backed ||
          !color->image ||
          !(color->image_usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ||
          ((color_feedback_loop_mask & (1u << i)) &&
           !(color->image_usage &
             VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT)) ||
          color->vk_format == VK_FORMAT_UNDEFINED ||
          (color->samples ? color->samples : VK_SAMPLE_COUNT_1_BIT) !=
             attachment_samples ||
          /* Validate the subresource this target actually names. */
          !yttrium_venus_valid_render_subresource(
             color, color_level, color_layer, render_layers) ||
          yttrium_venus_subresource_width(color, color_level) !=
             render_width ||
          yttrium_venus_subresource_height(color, color_level) !=
             render_height) {
         YTTRIUM_WARN("yttrium: Venus native pipeline rejected color rt%u res_id=%u initialized=%u buffer_backed=%u usage=0x%x format=%u extent=%ux%u subresource=%u/%u+%u render=%ux%u\n",
                     i, color_id,
                     color ? color->initialized : 0,
                     color ? color->buffer_backed : 0,
                     color ? color->image_usage : 0,
                     color ? color->vk_format : VK_FORMAT_UNDEFINED,
                     color ? color->width : 0,
                     color ? color->height : 0,
                     color_level, color_layer, render_layers,
                     render_width, render_height);
         return false;
      }
   }

   if (has_depth &&
       (!depth_resource->initialized || depth_resource->buffer_backed ||
        !depth_resource->image ||
        !(depth_resource->image_usage &
          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ||
        (depth_feedback_loop &&
         !(depth_resource->image_usage &
           VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT)) ||
        !yttrium_venus_format_has_depth(depth_resource->vk_format) ||
        (depth_resource->samples ? depth_resource->samples :
                                   VK_SAMPLE_COUNT_1_BIT) !=
           attachment_samples ||
        !yttrium_venus_valid_render_subresource(
           depth_resource, depth_level, depth_layer, depth_layers) ||
        depth_layers < render_layers ||
        /* Vulkan framebuffers require attachments to cover the framebuffer
         * extent; they do not require the attachment extents to be equal.
         * D3D9 relies on this when a smaller render target retains the
         * automatic depth surface created for the backbuffer. */
        yttrium_venus_subresource_width(depth_resource, depth_level) <
           render_width ||
        yttrium_venus_subresource_height(depth_resource, depth_level) <
           render_height)) {
      YTTRIUM_WARN("yttrium: Venus native pipeline rejected depth res_id=%u initialized=%u buffer_backed=%u usage=0x%x format=%u extent=%ux%u subresource=%u/%u+%u render=%ux%u render_layers=%u\n",
                  depth_resource_id,
                  depth_resource ? depth_resource->initialized : 0,
                  depth_resource ? depth_resource->buffer_backed : 0,
                  depth_resource ? depth_resource->image_usage : 0,
                  depth_resource ? depth_resource->vk_format :
                     VK_FORMAT_UNDEFINED,
                  depth_resource ? depth_resource->width : 0,
                  depth_resource ? depth_resource->height : 0,
                  depth_level, depth_layer, depth_layers,
                  render_width, render_height, render_layers);
      return false;
   }

   const bool render_target_cache_enabled =
      venus->render_target_cache && venus->render_target_cache_capacity;
   struct yttrium_venus_render_target_key render_target_key;
   struct yttrium_venus_render_target *cached_render_target = NULL;
   if (render_target_cache_enabled) {
      memset(&render_target_key, 0, sizeof(render_target_key));
      render_target_key.depth_image_id =
         has_depth ? depth_resource->image_obj.id : 0;
      render_target_key.depth_format =
         has_depth ? depth_resource->vk_format : VK_FORMAT_UNDEFINED;
      render_target_key.depth_view_type = has_depth ?
         yttrium_venus_render_view_type(depth_resource, depth_layers) : 0;
      render_target_key.color_attachment_count = color_resource_count;
      render_target_key.width = render_width;
      render_target_key.height = render_height;
      render_target_key.layers = render_layers;
      render_target_key.depth_level = depth_level;
      render_target_key.depth_layer = depth_layer;
      render_target_key.depth_layers = depth_layers;
      render_target_key.attachment_samples = attachment_samples;
      render_target_key.render_samples = render_samples;
      render_target_key.use_mrss = use_mrss;
      render_target_key.color_feedback_loop_mask =
         color_feedback_loop_mask;
      render_target_key.depth_feedback_loop = depth_feedback_loop;
      render_target_key.depth_read_only = depth_read_only;
      for (uint32_t i = 0; i < color_resource_count; i++) {
         struct yttrium_venus_resource *color = color_resources[i];
         if (!color)
            continue;
         render_target_key.color_image_ids[i] = color->image_obj.id;
         render_target_key.color_formats[i] =
            pipeline->key.rt_format[i] != VK_FORMAT_UNDEFINED ?
               pipeline->key.rt_format[i] : color->vk_format;
         render_target_key.color_view_types[i] =
            yttrium_venus_render_view_type(color, render_layers);
         render_target_key.color_levels[i] = draw_state->rt_level[i];
         render_target_key.color_layers[i] = draw_state->rt_layer[i];
      }

      cached_render_target =
         yttrium_venus_render_target_lookup(venus, &render_target_key);
   }

   if (cached_render_target) {
      yttrium_venus_render_target_assign(pipeline, cached_render_target);
   } else {

   for (uint32_t i = 0; i < color_resource_count; i++) {
      struct yttrium_venus_resource *color = color_resources[i];

      if (!color)
         continue;

      const VkFormat attachment_format =
         pipeline->key.rt_format[i] != VK_FORMAT_UNDEFINED ?
         pipeline->key.rt_format[i] : color->vk_format;
      const uint32_t color_level =
         draw_state && i < ARRAY_SIZE(draw_state->rt_level) ?
            draw_state->rt_level[i] : render_level;

      yttrium_venus_init_object(venus, &pipeline->image_view_objs[i]);
      pipeline->image_views[i] =
         YTTRIUM_VENUS_HANDLE(VkImageView, &pipeline->image_view_objs[i]);
      /*
       * This target's own slice, not the first target's.  Getting it wrong
       * renders into the wrong layer of the array and reports nothing.
       */
      const uint32_t color_layer =
         draw_state && i < ARRAY_SIZE(draw_state->rt_layer) ?
            draw_state->rt_layer[i] : render_layer;
      const VkImageViewCreateInfo view_info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = color->image,
         .viewType = yttrium_venus_render_view_type(color, render_layers),
         .format = attachment_format,
         .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = color_level,
            .levelCount = 1,
            .baseArrayLayer = color_layer,
            .layerCount = render_layers,
         },
      };
      YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
         venus, vkCreateImageView, &pipeline->image_view_objs[i],
         &pipeline->image_views[i],
         goto pipeline_submit_failed, venus->device_handle, &view_info, NULL,
         &pipeline->image_views[i]);
   }
   pipeline->color_attachment_count = color_resource_count;

   if (has_depth) {
      yttrium_venus_init_object(venus, &pipeline->depth_image_view_obj);
      pipeline->depth_image_view =
         YTTRIUM_VENUS_HANDLE(VkImageView,
                               &pipeline->depth_image_view_obj);
      const VkImageViewCreateInfo depth_view_info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = depth_resource->image,
         .viewType =
            yttrium_venus_render_view_type(depth_resource, depth_layers),
         .format = depth_resource->vk_format,
         .subresourceRange = {
            .aspectMask = yttrium_venus_format_aspects(
               depth_resource->vk_format),
            .baseMipLevel = depth_level,
            .levelCount = 1,
            .baseArrayLayer = depth_layer,
            .layerCount = depth_layers,
         },
      };
      YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
         venus, vkCreateImageView, &pipeline->depth_image_view_obj,
         &pipeline->depth_image_view,
         goto pipeline_submit_failed, venus->device_handle,
         &depth_view_info, NULL, &pipeline->depth_image_view);
   }

   yttrium_venus_init_object(venus, &pipeline->render_pass_obj);
   pipeline->render_pass =
      YTTRIUM_VENUS_HANDLE(VkRenderPass, &pipeline->render_pass_obj);
   VkAttachmentDescription attachments[PIPE_MAX_COLOR_BUFS + 1];
   VkAttachmentDescription2 attachments2[PIPE_MAX_COLOR_BUFS + 1];
   uint32_t attachment_count = 0;
   uint32_t color_attachment_indices[PIPE_MAX_COLOR_BUFS];
   for (uint32_t i = 0; i < color_resource_count; i++) {
      struct yttrium_venus_resource *color = color_resources[i];

      if (!color) {
         color_attachment_indices[i] = VK_ATTACHMENT_UNUSED;
         continue;
      }

      const VkFormat attachment_format =
         pipeline->key.rt_format[i] != VK_FORMAT_UNDEFINED ?
         pipeline->key.rt_format[i] : color->vk_format;
      const VkImageLayout attachment_layout =
         color_feedback_loop_mask & (1u << i) ?
            VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT :
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

      color_attachment_indices[i] = attachment_count;
      attachments[attachment_count++] = (VkAttachmentDescription) {
         .format = attachment_format,
         .samples = color->samples ? color->samples : VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = attachment_layout,
         .finalLayout = attachment_layout,
      };
      attachments2[color_attachment_indices[i]] =
         (VkAttachmentDescription2) {
            .sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
            .format = attachment_format,
            .samples = color->samples ? color->samples :
               VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = attachment_layout,
            .finalLayout = attachment_layout,
         };
   }
   const uint32_t depth_attachment = attachment_count;
   if (has_depth) {
      const bool has_stencil =
         yttrium_venus_format_has_stencil(depth_resource->vk_format);
      const VkImageLayout attachment_layout = depth_feedback_loop ?
         VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT :
         depth_read_only ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL :
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      attachments[attachment_count++] = (VkAttachmentDescription) {
         .format = depth_resource->vk_format,
         .samples = depth_resource->samples ?
            depth_resource->samples : VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = depth_read_only ? VK_ATTACHMENT_STORE_OP_NONE :
                    VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = has_stencil ? VK_ATTACHMENT_LOAD_OP_LOAD :
                          VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = has_stencil ? VK_ATTACHMENT_STORE_OP_STORE :
                           VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = attachment_layout,
         .finalLayout = attachment_layout,
      };
      attachments2[depth_attachment] = (VkAttachmentDescription2) {
         .sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
         .format = depth_resource->vk_format,
         .samples = depth_resource->samples ?
            depth_resource->samples : VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = depth_read_only ? VK_ATTACHMENT_STORE_OP_NONE :
                    VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = has_stencil ? VK_ATTACHMENT_LOAD_OP_LOAD :
                          VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = has_stencil ? VK_ATTACHMENT_STORE_OP_STORE :
                           VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = attachment_layout,
         .finalLayout = attachment_layout,
      };
   }
   VkAttachmentReference color_refs[PIPE_MAX_COLOR_BUFS];
   VkAttachmentReference2 color_refs2[PIPE_MAX_COLOR_BUFS];
   for (uint32_t i = 0; i < color_resource_count; i++) {
      const VkImageLayout attachment_layout =
         color_feedback_loop_mask & (1u << i) ?
            VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT :
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      color_refs[i] = (VkAttachmentReference) {
         .attachment = color_attachment_indices[i],
         .layout = attachment_layout,
      };
      color_refs2[i] = (VkAttachmentReference2) {
         .sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
         .attachment = color_attachment_indices[i],
         .layout = attachment_layout,
      };
   }
   const VkImageLayout depth_attachment_layout = depth_feedback_loop ?
      VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT :
      depth_read_only ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL :
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
   const VkAttachmentReference depth_ref = {
      .attachment = depth_attachment,
      .layout = depth_attachment_layout,
   };
   const VkAttachmentReference2 depth_ref2 = {
      .sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
      .attachment = depth_attachment,
      .layout = depth_attachment_layout,
   };
   if (use_mrss) {
      const VkMultisampledRenderToSingleSampledInfoEXT mrss_info = {
         .sType =
            VK_STRUCTURE_TYPE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_INFO_EXT,
         .multisampledRenderToSingleSampledEnable = VK_TRUE,
         .rasterizationSamples = render_samples,
      };
      const VkSubpassDescription2 subpass2 = {
         .sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2,
         .pNext = &mrss_info,
         .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
         .colorAttachmentCount = color_resource_count,
         .pColorAttachments = has_color ? color_refs2 : NULL,
         .pDepthStencilAttachment = has_depth ? &depth_ref2 : NULL,
      };
      const VkRenderPassCreateInfo2 render_pass_info2 = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2,
         .attachmentCount = attachment_count,
         .pAttachments = attachments2,
         .subpassCount = 1,
         .pSubpasses = &subpass2,
      };
      YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
         venus, vkCreateRenderPass2, &pipeline->render_pass_obj,
         &pipeline->render_pass,
         goto pipeline_submit_failed, venus->device_handle,
         &render_pass_info2, NULL, &pipeline->render_pass);
   } else {
      const VkSubpassDescription subpass = {
         .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
         .colorAttachmentCount = color_resource_count,
         .pColorAttachments = has_color ? color_refs : NULL,
         .pDepthStencilAttachment = has_depth ? &depth_ref : NULL,
      };
      const VkRenderPassCreateInfo render_pass_info = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
         .attachmentCount = attachment_count,
         .pAttachments = attachments,
         .subpassCount = 1,
         .pSubpasses = &subpass,
      };
      YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
         venus, vkCreateRenderPass, &pipeline->render_pass_obj,
         &pipeline->render_pass,
         goto pipeline_submit_failed, venus->device_handle,
         &render_pass_info, NULL, &pipeline->render_pass);
   }

   yttrium_venus_init_object(venus, &pipeline->framebuffer_obj);
   pipeline->framebuffer =
      YTTRIUM_VENUS_HANDLE(VkFramebuffer, &pipeline->framebuffer_obj);
   VkImageView framebuffer_attachments[PIPE_MAX_COLOR_BUFS + 1];
   uint32_t framebuffer_attachment_count = 0;
   for (uint32_t i = 0; i < color_resource_count; i++) {
      if (!color_resources[i])
         continue;

      framebuffer_attachments[framebuffer_attachment_count++] =
         pipeline->image_views[i];
   }
   if (has_depth)
      framebuffer_attachments[framebuffer_attachment_count++] =
         pipeline->depth_image_view;
   const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = pipeline->render_pass,
      .attachmentCount = framebuffer_attachment_count,
      .pAttachments = framebuffer_attachments,
      .width = render_width,
      .height = render_height,
      .layers = render_layers,
   };
   const VkResult framebuffer_result =
      vn_call_vkCreateFramebuffer(&venus->vn_ring,
                                  venus->device_handle,
                                  &framebuffer_info, NULL,
                                  &pipeline->framebuffer);
   if (framebuffer_result != VK_SUCCESS) {
      YTTRIUM_WARN("yttrium: Venus pipeline framebuffer create failed owner=venus2 result=%d framebuffer_id=%llu render_pass_id=%llu attachments=%u extent=%ux%u layers=%u\n",
                   framebuffer_result,
                   (unsigned long long)pipeline->framebuffer_obj.id,
                   (unsigned long long)pipeline->render_pass_obj.id,
                   framebuffer_attachment_count, render_width,
                   render_height, render_layers);
      pipeline->framebuffer = VK_NULL_HANDLE;
      memset(&pipeline->framebuffer_obj, 0,
             sizeof(pipeline->framebuffer_obj));
      yttrium_venus2_pipeline_fini(venus, pipeline);
      return false;
   }

   if (render_target_cache_enabled)
      (void)yttrium_venus_render_target_adopt(
         venus, &render_target_key, pipeline);
   }

   if (!yttrium_venus_pipeline_prepare_ubo_descriptors(
          venus, pipeline, ubo_bindings, ubo_binding_count,
          sampled_bindings, sampled_binding_count,
          storage_image_mask, storage_buffer_mask, storage_stage_flags,
          true)) {
      yttrium_venus2_pipeline_fini(venus, pipeline);
      return false;
   }

   yttrium_venus_init_object(venus, &pipeline->pipeline_layout_obj);
   pipeline->pipeline_layout =
      YTTRIUM_VENUS_HANDLE(VkPipelineLayout,
                            &pipeline->pipeline_layout_obj);
   const VkPushConstantRange push_constant_ranges[2] = {
      {
         .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
         .offset = YTTRIUM_SHADER_VS_PUSH_CONSTANT_OFFSET,
         .size = YTTRIUM_SHADER_VS_PUSH_CONSTANT_BYTES,
      },
      {
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
         .offset = YTTRIUM_SHADER_FS_PUSH_CONSTANT_OFFSET,
         .size = YTTRIUM_SHADER_FS_PUSH_CONSTANT_BYTES,
      },
   };
   const uint32_t push_constant_range_count =
      yttrium_gdi_static_ubo_sampled_cache_enabled() ?
      ARRAY_SIZE(push_constant_ranges) : 0;
   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = pipeline->descriptor_set_layout ? 1 : 0,
      .pSetLayouts = pipeline->descriptor_set_layout ?
                     &pipeline->descriptor_set_layout : NULL,
      .pushConstantRangeCount = push_constant_range_count,
      .pPushConstantRanges = push_constant_range_count ?
                             push_constant_ranges : NULL,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreatePipelineLayout, &pipeline->pipeline_layout_obj,
      &pipeline->pipeline_layout,
      goto pipeline_submit_failed, venus->device_handle, &layout_info, NULL,
      &pipeline->pipeline_layout);
   if (pipeline->push_descriptor_set_layout) {
      yttrium_venus_init_object(venus, &pipeline->push_pipeline_layout_obj);
      pipeline->push_pipeline_layout =
         YTTRIUM_VENUS_HANDLE(VkPipelineLayout,
                               &pipeline->push_pipeline_layout_obj);
      const VkPipelineLayoutCreateInfo push_layout_info = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
         .setLayoutCount = 1,
         .pSetLayouts = &pipeline->push_descriptor_set_layout,
         .pushConstantRangeCount = push_constant_range_count,
         .pPushConstantRanges = push_constant_range_count ?
                                push_constant_ranges : NULL,
      };
      YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
         venus, vkCreatePipelineLayout,
         &pipeline->push_pipeline_layout_obj, &pipeline->push_pipeline_layout,
         goto pipeline_submit_failed,
         venus->device_handle, &push_layout_info, NULL,
         &pipeline->push_pipeline_layout);
      if (pipeline->push_descriptor_set_layout_alt) {
         yttrium_venus_init_object(
            venus, &pipeline->push_pipeline_layout_alt_obj);
         pipeline->push_pipeline_layout_alt =
            YTTRIUM_VENUS_HANDLE(
               VkPipelineLayout, &pipeline->push_pipeline_layout_alt_obj);
         VkPipelineLayoutCreateInfo push_layout_alt_info = push_layout_info;
         push_layout_alt_info.pSetLayouts =
            &pipeline->push_descriptor_set_layout_alt;
         YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
            venus, vkCreatePipelineLayout,
            &pipeline->push_pipeline_layout_alt_obj,
            &pipeline->push_pipeline_layout_alt,
            goto pipeline_submit_failed, venus->device_handle,
            &push_layout_alt_info, NULL,
            &pipeline->push_pipeline_layout_alt);
      }
   }

   VkPipelineShaderStageCreateInfo stages[5];
   uint32_t stage_count = 0;
   stages[stage_count++] = (VkPipelineShaderStageCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = vertex_shader,
      .pName = "main",
   };
   if (has_tess_ctrl) {
      stages[stage_count++] = (VkPipelineShaderStageCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
         .module = tess_ctrl_shader,
         .pName = "main",
      };
   }
   if (has_tess_eval) {
      stages[stage_count++] = (VkPipelineShaderStageCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
         .module = tess_eval_shader,
         .pName = "main",
      };
   }
   if (geometry_shader) {
      stages[stage_count++] = (VkPipelineShaderStageCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_GEOMETRY_BIT,
         .module = geometry_shader,
         .pName = "main",
      };
   }
   if (fragment_shader) {
      stages[stage_count++] = (VkPipelineShaderStageCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fragment_shader,
         .pName = "main",
      };
   }
   VkVertexInputBindingDivisorDescription vertex_divisors
      [YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS];
   uint32_t vertex_divisor_count = 0;
   for (uint32_t i = 0; i < binding_count; i++) {
      const uint32_t divisor = binding_divisors ? binding_divisors[i] : 1;
      if (divisor == 1)
         continue;

      vertex_divisors[vertex_divisor_count++] =
         (VkVertexInputBindingDivisorDescription) {
            .binding = bindings[i].binding,
            .divisor = divisor == UINT32_MAX ? 0 : divisor,
         };
   }
   const VkPipelineVertexInputDivisorStateCreateInfo vertex_divisor_state = {
      .sType =
         VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO,
      .vertexBindingDivisorCount = vertex_divisor_count,
      .pVertexBindingDivisors = vertex_divisor_count ? vertex_divisors : NULL,
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .pNext = vertex_divisor_count ? &vertex_divisor_state : NULL,
      .vertexBindingDescriptionCount = binding_count,
      .pVertexBindingDescriptions = binding_count ? bindings : NULL,
      .vertexAttributeDescriptionCount = attrib_count,
      .pVertexAttributeDescriptions = attrib_count ? attribs : NULL,
   };
   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = draw_state->topology,
      .primitiveRestartEnable = draw_state->primitive_restart_enable,
   };
   const VkPipelineTessellationStateCreateInfo tessellation_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
      .patchControlPoints = patch_vertices,
   };
   const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = MAX2(draw_state->viewport_count, 1),
      .scissorCount = MAX2(draw_state->viewport_count, 1),
   };
   const bool force_unorm_depth_bias =
      venus->depth_bias_control && has_depth &&
      (depth_resource->vk_format == VK_FORMAT_D24_UNORM_S8_UINT ||
       depth_resource->vk_format == VK_FORMAT_X8_D24_UNORM_PACK32);
   const VkDepthBiasRepresentationInfoEXT depth_bias_representation = {
      .sType = VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT,
      .depthBiasRepresentation =
         VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT,
      .depthBiasExact = VK_FALSE,
   };
   const VkPipelineRasterizationStateCreateInfo raster = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .pNext = force_unorm_depth_bias ? &depth_bias_representation : NULL,
      .depthClampEnable = draw_state->depth_clamp_enable,
      .rasterizerDiscardEnable = draw_state->rasterizer_discard_enable,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = draw_state->cull_mode,
      .frontFace = draw_state->front_face,
      .depthBiasEnable = draw_state->depth_bias_enable,
      .depthBiasConstantFactor = draw_state->depth_bias_constant_factor,
      .depthBiasClamp = draw_state->depth_bias_clamp,
      .depthBiasSlopeFactor = draw_state->depth_bias_slope_factor,
      .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = render_samples,
      .sampleShadingEnable =
         venus->sample_rate_shading &&
         pipeline->key.sample_shading_enable &&
         render_samples != VK_SAMPLE_COUNT_1_BIT,
      .minSampleShading = 1.0f,
      .alphaToCoverageEnable = draw_state->alpha_to_coverage_enable,
      .pSampleMask = &draw_state->sample_mask,
   };
   VkPipelineColorBlendAttachmentState color_blend_attachments
      [PIPE_MAX_COLOR_BUFS];
   for (uint32_t i = 0; i < color_resource_count; i++) {
      color_blend_attachments[i] = (VkPipelineColorBlendAttachmentState) {
         .blendEnable = draw_state->rt_blend_enable[i],
         .srcColorBlendFactor = draw_state->rt_src_color_blend_factor[i],
         .dstColorBlendFactor = draw_state->rt_dst_color_blend_factor[i],
         .colorBlendOp = draw_state->rt_color_blend_op[i],
         .srcAlphaBlendFactor = draw_state->rt_src_alpha_blend_factor[i],
         .dstAlphaBlendFactor = draw_state->rt_dst_alpha_blend_factor[i],
         .alphaBlendOp = draw_state->rt_alpha_blend_op[i],
         .colorWriteMask = draw_state->rt_color_write_mask[i],
      };
   }
   const VkPipelineColorBlendStateCreateInfo color_blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .logicOpEnable = draw_state->logic_op_enable,
      .logicOp = draw_state->logic_op,
      .attachmentCount = color_resource_count,
      .pAttachments = has_color ? color_blend_attachments : NULL,
   };
   const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = has_depth ? draw_state->depth_test_enable : VK_FALSE,
      .depthWriteEnable = has_depth ? draw_state->depth_write_enable :
                          VK_FALSE,
      .depthCompareOp = draw_state->depth_compare_op,
      .depthBoundsTestEnable = VK_FALSE,
      .stencilTestEnable = has_depth ? draw_state->stencil_test_enable :
                           VK_FALSE,
      .front = draw_state->stencil_front,
      .back = draw_state->stencil_back,
      .minDepthBounds = 0.0f,
      .maxDepthBounds = 1.0f,
   };
   const VkDynamicState dynamic_states[3] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
      VK_DYNAMIC_STATE_BLEND_CONSTANTS,
   };
   const VkPipelineDynamicStateCreateInfo dynamic = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 3,
      .pDynamicStates = dynamic_states,
   };

   yttrium_venus_init_object(venus, &pipeline->pipeline_obj);
   pipeline->pipeline =
      YTTRIUM_VENUS_HANDLE(VkPipeline, &pipeline->pipeline_obj);
   const VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .flags =
         (color_feedback_loop_mask ?
            VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT : 0) |
         (depth_feedback_loop ?
            VK_PIPELINE_CREATE_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT :
            0),
      .stageCount = stage_count,
      .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &input_assembly,
      .pTessellationState = tessellation ? &tessellation_state : NULL,
      .pViewportState = &viewport_state,
      .pRasterizationState = &raster,
      .pMultisampleState = &multisample,
      .pDepthStencilState = has_depth ? &depth_stencil : NULL,
      .pColorBlendState = &color_blend,
      .pDynamicState = &dynamic,
      .layout = pipeline->pipeline_layout,
      .renderPass = pipeline->render_pass,
   };
   const bool sync_tessellation_pipeline =
      tessellation && yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_SYNC_TESSELLATION_PIPELINES", false);
   VkResult pipeline_result = VK_SUCCESS;
   if (sync_tessellation_pipeline) {
      pipeline_result = vn_call_vkCreateGraphicsPipelines(
         &venus->vn_ring, venus->device_handle, VK_NULL_HANDLE, 1,
         &pipeline_info, NULL, &pipeline->pipeline);
   } else {
      YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
         venus, vkCreateGraphicsPipelines, &pipeline->pipeline_obj,
         &pipeline->pipeline,
         goto pipeline_submit_failed, venus->device_handle, VK_NULL_HANDLE, 1,
         &pipeline_info, NULL, &pipeline->pipeline);
   }
   if (pipeline_result != VK_SUCCESS) {
      YTTRIUM_WARN("yttrium: tessellation pipeline create failed result=%d topology=%u patch_vertices=%u\n",
                   pipeline_result, draw_state->topology, patch_vertices);
      yttrium_venus2_pipeline_fini(venus, pipeline);
      return false;
   }
   if (pipeline->push_pipeline_layout) {
      yttrium_venus_init_object(venus, &pipeline->push_pipeline_obj);
      pipeline->push_pipeline =
         YTTRIUM_VENUS_HANDLE(VkPipeline, &pipeline->push_pipeline_obj);
      VkGraphicsPipelineCreateInfo push_pipeline_info = pipeline_info;
      push_pipeline_info.layout = pipeline->push_pipeline_layout;
      if (sync_tessellation_pipeline) {
         pipeline_result = vn_call_vkCreateGraphicsPipelines(
            &venus->vn_ring, venus->device_handle, VK_NULL_HANDLE, 1,
            &push_pipeline_info, NULL, &pipeline->push_pipeline);
      } else {
         YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
            venus, vkCreateGraphicsPipelines,
            &pipeline->push_pipeline_obj, &pipeline->push_pipeline,
            goto pipeline_submit_failed,
            venus->device_handle, VK_NULL_HANDLE, 1, &push_pipeline_info,
            NULL, &pipeline->push_pipeline);
      }
      if (pipeline_result != VK_SUCCESS) {
         YTTRIUM_WARN("yttrium: tessellation push pipeline create failed result=%d topology=%u patch_vertices=%u\n",
                      pipeline_result, draw_state->topology, patch_vertices);
         yttrium_venus2_pipeline_fini(venus, pipeline);
         return false;
      }
   }

   YTTRIUM_LOG("yttrium: Venus native pipeline setup res_id=%u image_id=%llu depth_res_id=%u depth_image_id=%llu pipeline_id=%llu push_pipeline_id=%llu push_layout_id=%llu extent=%ux%u topology=%u restart=%u cull=0x%x front=%u blend=%u color_mask=0x%x sample_mask=0x%x bindings=%u attribs=%u ubo_bindings=%u ubo_slots=%u sampled_bindings=%u depth_test=%u depth_write=%u depth_compare=%u\n",
                resource_id,
                (unsigned long long)resource->image_obj.id,
                has_depth ? depth_resource_id : 0,
                has_depth ?
                   (unsigned long long)depth_resource->image_obj.id : 0,
                (unsigned long long)pipeline->pipeline_obj.id,
                (unsigned long long)pipeline->push_pipeline_obj.id,
                (unsigned long long)pipeline->push_pipeline_layout_obj.id,
                resource->width, resource->height,
                draw_state->topology,
                draw_state->primitive_restart_enable,
                draw_state->cull_mode,
                draw_state->front_face,
                draw_state->blend_enable,
                 draw_state->color_write_mask,
                 draw_state->sample_mask,
                 binding_count, attrib_count,
                 ubo_binding_count, pipeline->ubo_count,
                 sampled_binding_count,
                 draw_state->depth_test_enable,
                 draw_state->depth_write_enable,
                 draw_state->depth_compare_op);
   return true;

pipeline_submit_failed:
   yttrium_venus2_pipeline_fini(venus, pipeline);
   return false;
}

bool
yttrium_venus2_compute_pipeline_init(
   struct yttrium_venus *venus,
   struct yttrium_pipeline *pipeline,
   VkShaderModule compute_shader,
   const struct yttrium_venus_ubo_binding_layout *ubo_bindings,
   uint32_t ubo_binding_count,
   const struct yttrium_venus_sampled_binding_layout *sampled_bindings,
   uint32_t sampled_binding_count,
   uint64_t storage_image_mask,
   uint64_t storage_buffer_mask)
{
   if (!yttrium_venus_ensure_initialized(venus))
      return false;
   if (!pipeline || !compute_shader ||
       (!storage_image_mask && !storage_buffer_mask) ||
       ubo_binding_count > YTTRIUM_VENUS_MAX_PIPELINE_UBO_BINDINGS ||
       sampled_binding_count > YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES)
      return false;

   if (!yttrium_venus_pipeline_prepare_ubo_descriptors(
          venus, pipeline, ubo_bindings, ubo_binding_count,
          sampled_bindings, sampled_binding_count,
          storage_image_mask, storage_buffer_mask,
          VK_SHADER_STAGE_COMPUTE_BIT, true)) {
      yttrium_venus2_pipeline_fini(venus, pipeline);
      return false;
   }

   yttrium_venus_init_object(venus, &pipeline->pipeline_layout_obj);
   pipeline->pipeline_layout =
      YTTRIUM_VENUS_HANDLE(VkPipelineLayout,
                            &pipeline->pipeline_layout_obj);
   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = pipeline->descriptor_set_layout ? 1 : 0,
      .pSetLayouts = pipeline->descriptor_set_layout ?
                     &pipeline->descriptor_set_layout : NULL,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreatePipelineLayout, &pipeline->pipeline_layout_obj,
      &pipeline->pipeline_layout,
      goto compute_submit_failed, venus->device_handle, &layout_info, NULL,
      &pipeline->pipeline_layout);
   if (pipeline->push_descriptor_set_layout) {
      yttrium_venus_init_object(venus, &pipeline->push_pipeline_layout_obj);
      pipeline->push_pipeline_layout =
         YTTRIUM_VENUS_HANDLE(VkPipelineLayout,
                               &pipeline->push_pipeline_layout_obj);
      const VkPipelineLayoutCreateInfo push_layout_info = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
         .setLayoutCount = 1,
         .pSetLayouts = &pipeline->push_descriptor_set_layout,
      };
      YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
         venus, vkCreatePipelineLayout,
         &pipeline->push_pipeline_layout_obj, &pipeline->push_pipeline_layout,
         goto compute_submit_failed,
         venus->device_handle, &push_layout_info, NULL,
         &pipeline->push_pipeline_layout);
      if (pipeline->push_descriptor_set_layout_alt) {
         yttrium_venus_init_object(
            venus, &pipeline->push_pipeline_layout_alt_obj);
         pipeline->push_pipeline_layout_alt =
            YTTRIUM_VENUS_HANDLE(
               VkPipelineLayout, &pipeline->push_pipeline_layout_alt_obj);
         VkPipelineLayoutCreateInfo push_layout_alt_info = push_layout_info;
         push_layout_alt_info.pSetLayouts =
            &pipeline->push_descriptor_set_layout_alt;
         YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
            venus, vkCreatePipelineLayout,
            &pipeline->push_pipeline_layout_alt_obj,
            &pipeline->push_pipeline_layout_alt,
            goto compute_submit_failed, venus->device_handle,
            &push_layout_alt_info, NULL,
            &pipeline->push_pipeline_layout_alt);
      }
   }

   yttrium_venus_init_object(venus, &pipeline->pipeline_obj);
   pipeline->pipeline =
      YTTRIUM_VENUS_HANDLE(VkPipeline, &pipeline->pipeline_obj);
   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = compute_shader,
         .pName = "main",
      },
      .layout = pipeline->pipeline_layout,
   };
   VkResult result =
      vn_call_vkCreateComputePipelines(&venus->vn_ring,
                                       venus->device_handle,
                                       VK_NULL_HANDLE, 1,
                                       &pipeline_info, NULL,
                                       &pipeline->pipeline);
   if (result != VK_SUCCESS) {
      YTTRIUM_WARN("yttrium: Venus native compute pipeline create failed result=%d pipeline_id=%llu ubo_bindings=%u storage_image_mask=0x%llx storage_buffer_mask=0x%llx\n",
                   result,
                   (unsigned long long)pipeline->pipeline_obj.id,
                   ubo_binding_count,
                   (unsigned long long)storage_image_mask,
                   (unsigned long long)storage_buffer_mask);
      yttrium_venus2_pipeline_fini(venus, pipeline);
      return false;
   }
   if (pipeline->push_pipeline_layout) {
      yttrium_venus_init_object(venus, &pipeline->push_pipeline_obj);
      pipeline->push_pipeline =
         YTTRIUM_VENUS_HANDLE(VkPipeline, &pipeline->push_pipeline_obj);
      VkComputePipelineCreateInfo push_pipeline_info = pipeline_info;
      push_pipeline_info.layout = pipeline->push_pipeline_layout;
      YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
         venus, vkCreateComputePipelines, &pipeline->push_pipeline_obj,
         &pipeline->push_pipeline,
         goto compute_submit_failed, venus->device_handle, VK_NULL_HANDLE, 1,
         &push_pipeline_info, NULL, &pipeline->push_pipeline);
   }

   YTTRIUM_LOG("yttrium: Venus native compute pipeline setup pipeline_id=%llu push_pipeline_id=%llu push_layout_id=%llu ubo_bindings=%u ubo_slots=%u storage_image_mask=0x%llx storage_buffer_mask=0x%llx\n",
               (unsigned long long)pipeline->pipeline_obj.id,
               (unsigned long long)pipeline->push_pipeline_obj.id,
               (unsigned long long)pipeline->push_pipeline_layout_obj.id,
               ubo_binding_count, pipeline->ubo_count,
               (unsigned long long)storage_image_mask,
               (unsigned long long)storage_buffer_mask);
   return true;

compute_submit_failed:
   yttrium_venus2_pipeline_fini(venus, pipeline);
   return false;
}

static VkComponentSwizzle
yttrium_venus_component_swizzle(enum pipe_swizzle swizzle,
                                VkComponentSwizzle fallback)
{
   switch (swizzle) {
   case PIPE_SWIZZLE_X:
      return VK_COMPONENT_SWIZZLE_R;
   case PIPE_SWIZZLE_Y:
      return VK_COMPONENT_SWIZZLE_G;
   case PIPE_SWIZZLE_Z:
      return VK_COMPONENT_SWIZZLE_B;
   case PIPE_SWIZZLE_W:
      return VK_COMPONENT_SWIZZLE_A;
   case PIPE_SWIZZLE_0:
      return VK_COMPONENT_SWIZZLE_ZERO;
   case PIPE_SWIZZLE_1:
      return VK_COMPONENT_SWIZZLE_ONE;
   default:
      return fallback;
   }
}

static enum pipe_swizzle
yttrium_venus_swizzle_key_channel(uint32_t swizzle_key,
                                  unsigned channel,
                                  enum pipe_swizzle fallback)
{
   return yttrium_venus_sample_swizzle_normalize(
      (swizzle_key >> (channel * YTTRIUM_VENUS_SAMPLE_SWIZZLE_BITS)) &
         YTTRIUM_VENUS_SAMPLE_SWIZZLE_MASK,
      fallback);
}

static VkComponentMapping
yttrium_venus_component_mapping_from_swizzle_key(uint32_t swizzle_key)
{
   const enum pipe_swizzle r =
      yttrium_venus_swizzle_key_channel(swizzle_key, 0, PIPE_SWIZZLE_X);
   const enum pipe_swizzle g =
      yttrium_venus_swizzle_key_channel(swizzle_key, 1, PIPE_SWIZZLE_Y);
   const enum pipe_swizzle b =
      yttrium_venus_swizzle_key_channel(swizzle_key, 2, PIPE_SWIZZLE_Z);
   const enum pipe_swizzle a =
      yttrium_venus_swizzle_key_channel(swizzle_key, 3, PIPE_SWIZZLE_W);

   return (VkComponentMapping) {
      .r = yttrium_venus_component_swizzle(r, VK_COMPONENT_SWIZZLE_R),
      .g = yttrium_venus_component_swizzle(g, VK_COMPONENT_SWIZZLE_G),
      .b = yttrium_venus_component_swizzle(b, VK_COMPONENT_SWIZZLE_B),
      .a = yttrium_venus_component_swizzle(a, VK_COMPONENT_SWIZZLE_A),
   };
}

static VkComponentSwizzle
yttrium_venus_remap_component_swizzle(VkComponentSwizzle swizzle,
                                      VkComponentSwizzle from,
                                      VkComponentSwizzle to)
{
   return swizzle == from ? to : swizzle;
}

static VkComponentMapping
yttrium_venus_sample_view_components(VkImageAspectFlags aspect_mask,
                                     uint32_t swizzle_key)
{
   VkComponentMapping components =
      yttrium_venus_component_mapping_from_swizzle_key(swizzle_key);

   if (aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) {
      components.r =
         yttrium_venus_remap_component_swizzle(components.r,
                                               VK_COMPONENT_SWIZZLE_G,
                                               VK_COMPONENT_SWIZZLE_R);
      components.g =
         yttrium_venus_remap_component_swizzle(components.g,
                                               VK_COMPONENT_SWIZZLE_G,
                                               VK_COMPONENT_SWIZZLE_R);
      components.b =
         yttrium_venus_remap_component_swizzle(components.b,
                                               VK_COMPONENT_SWIZZLE_G,
                                               VK_COMPONENT_SWIZZLE_R);
      components.a =
         yttrium_venus_remap_component_swizzle(components.a,
                                               VK_COMPONENT_SWIZZLE_G,
                                               VK_COMPONENT_SWIZZLE_R);
   }

   return components;
}

static VkFormat
yttrium_venus_sample_view_format(VkFormat resource_format,
                                 VkFormat view_format,
                                 VkImageAspectFlags aspect_mask)
{
   if (aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) {
      if (resource_format == VK_FORMAT_D16_UNORM_S8_UINT) {
         if (view_format == VK_FORMAT_D16_UNORM_S8_UINT ||
             view_format == VK_FORMAT_D16_UNORM)
            return VK_FORMAT_D16_UNORM;
      } else if (resource_format == VK_FORMAT_D24_UNORM_S8_UINT) {
         if (view_format == VK_FORMAT_X8_D24_UNORM_PACK32 ||
             view_format == VK_FORMAT_D24_UNORM_S8_UINT)
            return VK_FORMAT_X8_D24_UNORM_PACK32;
      } else if (resource_format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
         if (view_format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
             view_format == VK_FORMAT_D32_SFLOAT ||
             view_format == VK_FORMAT_X8_D24_UNORM_PACK32 ||
             view_format == VK_FORMAT_D24_UNORM_S8_UINT)
            return VK_FORMAT_D32_SFLOAT_S8_UINT;
      } else if (resource_format == VK_FORMAT_D32_SFLOAT) {
         if (view_format == VK_FORMAT_X8_D24_UNORM_PACK32 ||
             view_format == VK_FORMAT_D32_SFLOAT)
            return VK_FORMAT_D32_SFLOAT;
      }
   }

   if ((aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) &&
       (resource_format == VK_FORMAT_D16_UNORM_S8_UINT ||
        resource_format == VK_FORMAT_D24_UNORM_S8_UINT ||
        resource_format == VK_FORMAT_D32_SFLOAT_S8_UINT)) {
      if (view_format == VK_FORMAT_D16_UNORM_S8_UINT ||
          view_format == VK_FORMAT_D24_UNORM_S8_UINT ||
          view_format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
          view_format == VK_FORMAT_X8_D24_UNORM_PACK32 ||
          view_format == VK_FORMAT_S8_UINT)
         return resource_format;
   }

   return view_format;
}

/*
 * Per-resource image views are keyed by the exact view description.  A mip
 * chain that is sampled or stored one level at a time (an HZB build, a
 * downsample pyramid) needs one view per level, so the list grows on demand
 * instead of rejecting the draw or dispatch at a fixed bound.
 *
 * Each entry is allocated on its own and never moves: a Venus object handle
 * IS the address of its yttrium_venus_object (YTTRIUM_VENUS_HANDLE), and
 * handed-out VkImageViews stay live in descriptor writes and recorded
 * batches.  Relocating an entry dangles every handle already issued from it,
 * which the host reports as a failed object lookup.
 */
static struct yttrium_venus_sample_image_view *
yttrium_venus_find_image_view(struct yttrium_venus_resource *resource,
                              VkFormat vk_format,
                              uint32_t swizzle_key,
                              VkImageViewType view_type,
                              VkImageAspectFlags aspect_mask,
                              uint32_t first_level,
                              uint32_t level_count,
                              uint32_t first_layer,
                              uint32_t layer_count)
{
   for (struct yttrium_venus_sample_image_view *entry =
           resource->sample_image_views;
        entry; entry = entry->next) {
      if (entry->view && entry->vk_format == vk_format &&
          entry->swizzle_key == swizzle_key &&
          entry->view_type == view_type &&
          entry->aspect_mask == aspect_mask &&
          entry->first_level == first_level &&
          entry->level_count == level_count &&
          entry->first_layer == first_layer &&
          entry->layer_count == layer_count)
         return entry;
   }
   return NULL;
}

/*
 * Link a zeroed entry into the resource.  The caller fills the view key in
 * with yttrium_venus_commit_image_view() once the view exists, and unlinks it
 * with yttrium_venus_discard_image_view() if the create fails.
 */
static struct yttrium_venus_sample_image_view *
yttrium_venus_add_image_view(struct yttrium_venus_resource *resource,
                             uint32_t resource_id)
{
   struct yttrium_venus_sample_image_view *entry =
      CALLOC_STRUCT(yttrium_venus_sample_image_view);
   if (!entry) {
      YTTRIUM_WARN("yttrium: Venus image view allocation failed owner=venus2 res_id=%u image_id=%llu\n",
                   resource_id,
                   (unsigned long long)resource->image_obj.id);
      return NULL;
   }

   entry->next = resource->sample_image_views;
   resource->sample_image_views = entry;
   return entry;
}

static void
yttrium_venus_discard_image_view(struct yttrium_venus_resource *resource,
                                 struct yttrium_venus_sample_image_view *entry)
{
   assert(resource->sample_image_views == entry);
   resource->sample_image_views = entry->next;
   FREE(entry);
}

static void
yttrium_venus_commit_image_view(struct yttrium_venus_sample_image_view *entry,
                                VkFormat vk_format,
                                uint32_t swizzle_key,
                                VkImageViewType view_type,
                                VkImageAspectFlags aspect_mask,
                                uint32_t first_level,
                                uint32_t level_count,
                                uint32_t first_layer,
                                uint32_t layer_count)
{
   entry->vk_format = vk_format;
   entry->swizzle_key = swizzle_key;
   entry->view_type = view_type;
   entry->aspect_mask = aspect_mask;
   entry->first_level = first_level;
   entry->level_count = level_count;
   entry->first_layer = first_layer;
   entry->layer_count = layer_count;
}

static bool
yttrium_venus_ensure_sample_image_view(struct yttrium_venus *venus,
                                       struct yttrium_venus_resource *resource,
                                       uint32_t resource_id,
                                       VkFormat vk_format,
                                       uint32_t swizzle_key,
                                       VkImageViewType view_type,
                                       uint32_t first_level,
                                       uint32_t level_count,
                                       uint32_t first_layer,
                                       uint32_t layer_count,
                                       VkImageAspectFlags aspect_mask,
                                       VkImageView *out_view)
{
   if (!resource || !resource->initialized || resource->buffer_backed ||
       !resource->image || !out_view)
      return false;
   *out_view = VK_NULL_HANDLE;
   if (vk_format == VK_FORMAT_UNDEFINED)
      vk_format = resource->vk_format;
   const VkImageAspectFlags resource_aspects =
      yttrium_venus_format_aspects(resource->vk_format);
   const bool depth_stencil_image =
      (resource_aspects &
       (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0;
   if (!aspect_mask)
      aspect_mask = depth_stencil_image ? VK_IMAGE_ASPECT_DEPTH_BIT :
                                          VK_IMAGE_ASPECT_COLOR_BIT;
   if (depth_stencil_image) {
      const VkImageAspectFlags ds_aspects =
         aspect_mask &
         (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
      if ((ds_aspects != VK_IMAGE_ASPECT_DEPTH_BIT &&
           ds_aspects != VK_IMAGE_ASPECT_STENCIL_BIT) ||
          (aspect_mask & VK_IMAGE_ASPECT_COLOR_BIT) ||
          !(resource_aspects & ds_aspects)) {
         YTTRIUM_WARN("yttrium: Venus sampled image rejected bad aspect res_id=%u image_id=%llu resource_format=%u aspects=0x%x requested=0x%x\n",
                      resource_id,
                      (unsigned long long)resource->image_obj.id,
                      resource->vk_format, resource_aspects, aspect_mask);
         return false;
      }
   } else if (aspect_mask != VK_IMAGE_ASPECT_COLOR_BIT) {
      YTTRIUM_WARN("yttrium: Venus sampled image rejected non-color aspect res_id=%u image_id=%llu resource_format=%u requested=0x%x\n",
                   resource_id,
                   (unsigned long long)resource->image_obj.id,
                   resource->vk_format, aspect_mask);
      return false;
   }
   vk_format =
      yttrium_venus_sample_view_format(resource->vk_format, vk_format,
                                       aspect_mask);
   if (!(resource->image_usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
      YTTRIUM_WARN("yttrium: Venus sampled image rejected res_id=%u image_id=%llu usage=0x%x\n",
                   resource_id,
                   (unsigned long long)resource->image_obj.id,
                   resource->image_usage);
      return false;
   }
   if (!yttrium_venus_valid_image_subresource(resource, first_level,
                                              first_layer, layer_count) ||
       !level_count ||
       level_count > MAX2(resource->levels, 1) - first_level) {
      YTTRIUM_WARN("yttrium: Venus sampled image rejected bad view range res_id=%u image_id=%llu view_type=%u levels=%u+%u layers=%u+%u resource_levels=%u resource_layers=%u\n",
                   resource_id,
                   (unsigned long long)resource->image_obj.id,
                   view_type, first_level, level_count, first_layer,
                   layer_count, resource->levels, resource->layers);
      return false;
   }

   struct yttrium_venus_sample_image_view *entry =
      yttrium_venus_find_image_view(resource, vk_format, swizzle_key,
                                    view_type, aspect_mask, first_level,
                                    level_count, first_layer, layer_count);
   if (entry) {
      *out_view = entry->view;
      return true;
   }

   entry = yttrium_venus_add_image_view(resource, resource_id);
   if (!entry)
      return false;

   yttrium_venus_init_object(venus, &entry->obj);
   entry->view = YTTRIUM_VENUS_HANDLE(VkImageView, &entry->obj);
   const VkComponentMapping components =
      yttrium_venus_sample_view_components(aspect_mask, swizzle_key);
   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = resource->image,
      .viewType = view_type,
      .format = vk_format,
      .components = components,
      .subresourceRange = {
         .aspectMask = aspect_mask,
         .baseMipLevel = first_level,
         .levelCount = level_count,
         .baseArrayLayer = first_layer,
         .layerCount = layer_count,
      },
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateImageView, &entry->obj, &entry->view,
      goto sample_image_view_submit_failed, venus->device_handle,
      &view_info, NULL, &entry->view);
   yttrium_venus_commit_image_view(entry, vk_format, swizzle_key,
                                   view_type, aspect_mask, first_level,
                                   level_count, first_layer, layer_count);
   *out_view = entry->view;
   return true;

sample_image_view_submit_failed:
   yttrium_venus_discard_image_view(resource, entry);
   return false;
}

static bool
yttrium_venus_ensure_storage_image_view(struct yttrium_venus *venus,
                                        struct yttrium_venus_resource *resource,
                                        uint32_t resource_id,
                                        VkFormat vk_format,
                                        VkImageViewType view_type,
                                        uint32_t first_level,
                                        uint32_t level_count,
                                        uint32_t first_layer,
                                        uint32_t layer_count,
                                        VkImageAspectFlags aspect_mask,
                                        VkImageView *out_view)
{
   if (!resource || !resource->initialized || resource->buffer_backed ||
       !resource->image || !out_view)
      return false;
   *out_view = VK_NULL_HANDLE;
   if (vk_format == VK_FORMAT_UNDEFINED)
      vk_format = resource->vk_format;
   if (vk_format == VK_FORMAT_UNDEFINED)
      return false;
   if (!aspect_mask)
      aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT;
   if (aspect_mask != VK_IMAGE_ASPECT_COLOR_BIT)
      return false;
   if (!(resource->image_usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
      YTTRIUM_WARN("yttrium: Venus storage image rejected res_id=%u image_id=%llu usage=0x%x\n",
                   resource_id,
                   (unsigned long long)resource->image_obj.id,
                   resource->image_usage);
      return false;
   }
   if (!yttrium_venus_valid_image_subresource(resource, first_level,
                                              first_layer, layer_count) ||
       !level_count ||
       level_count > MAX2(resource->levels, 1) - first_level)
      return false;

   const uint32_t swizzle_key = YTTRIUM_VENUS_SAMPLE_SWIZZLE_IDENTITY;
   struct yttrium_venus_sample_image_view *entry =
      yttrium_venus_find_image_view(resource, vk_format, swizzle_key,
                                    view_type, aspect_mask, first_level,
                                    level_count, first_layer, layer_count);
   if (entry) {
      *out_view = entry->view;
      return true;
   }

   entry = yttrium_venus_add_image_view(resource, resource_id);
   if (!entry)
      return false;

   yttrium_venus_init_object(venus, &entry->obj);
   entry->view = YTTRIUM_VENUS_HANDLE(VkImageView, &entry->obj);
   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = resource->image,
      .viewType = view_type,
      .format = vk_format,
      .subresourceRange = {
         .aspectMask = aspect_mask,
         .baseMipLevel = first_level,
         .levelCount = level_count,
         .baseArrayLayer = first_layer,
         .layerCount = layer_count,
      },
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateImageView, &entry->obj, &entry->view,
      goto storage_image_view_submit_failed, venus->device_handle,
      &view_info, NULL, &entry->view);

   yttrium_venus_commit_image_view(entry, vk_format, swizzle_key,
                                   view_type, aspect_mask, first_level,
                                   level_count, first_layer, layer_count);
   *out_view = entry->view;
   return true;

storage_image_view_submit_failed:
   yttrium_venus_discard_image_view(resource, entry);
   return false;
}

static bool
yttrium_venus_ensure_sample_buffer_view(struct yttrium_venus *venus,
                                        struct yttrium_venus_resource *resource,
                                        uint32_t resource_id,
                                        enum pipe_format pipe_format,
                                        VkDeviceSize offset,
                                        VkDeviceSize range,
                                        VkBufferView *out_view)
{
   const VkFormat vk_format = yttrium_venus2_pipe_format(pipe_format);
   const unsigned blocksize = util_format_get_blocksize(pipe_format);

   if (out_view)
      *out_view = VK_NULL_HANDLE;
   if (!resource || !resource->initialized || !resource->buffer_backed ||
       !resource->buffer || vk_format == VK_FORMAT_UNDEFINED || !blocksize ||
       !out_view)
      return false;
   if (!(resource->buffer_usage & VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT)) {
      YTTRIUM_LOG("yttrium: Venus sampled buffer rejected res_id=%u buffer_id=%llu usage=0x%x format=%u vk_format=%u offset=0x%llx range=0x%llx blocksize=%u\n",
                  resource_id,
                  (unsigned long long)resource->buffer_obj.id,
                  resource->buffer_usage, pipe_format, vk_format,
                  (unsigned long long)offset,
                  (unsigned long long)range,
                  blocksize);
      return false;
   }

   if (!resource->image_size || offset > resource->image_size) {
      YTTRIUM_LOG("yttrium: Venus sampled buffer view rejected range res_id=%u buffer_id=%llu size=0x%llx offset=0x%llx range=0x%llx format=%u vk_format=%u\n",
                  resource_id,
                  (unsigned long long)resource->buffer_obj.id,
                  (unsigned long long)resource->image_size,
                  (unsigned long long)offset,
                  (unsigned long long)range,
                  pipe_format, vk_format);
      return false;
   }
   if (!range)
      range = resource->image_size - offset;
   if (range > resource->image_size - offset)
      range = resource->image_size - offset;
   range -= range % blocksize;
   if (!range || offset % blocksize) {
      YTTRIUM_LOG("yttrium: Venus sampled buffer view rejected alignment res_id=%u buffer_id=%llu size=0x%llx offset=0x%llx range=0x%llx format=%u vk_format=%u blocksize=%u\n",
                  resource_id,
                  (unsigned long long)resource->buffer_obj.id,
                  (unsigned long long)resource->image_size,
                  (unsigned long long)offset,
                  (unsigned long long)range,
                  pipe_format, vk_format, blocksize);
      return false;
   }

   const VkDeviceSize view_range =
      (!offset && range == resource->image_size) ||
      offset + range >= resource->image_size ?
      VK_WHOLE_SIZE : range;

   /* Keep typed buffer views alive for the resource lifetime; one draw can
    * bind several SRVs with different formats over the same buffer.
    */
   for (struct yttrium_venus_sample_buffer_view *entry =
           resource->sample_buffer_views;
        entry; entry = entry->next) {
      if (entry->view && entry->vk_format == vk_format &&
          entry->offset == offset && entry->range == view_range) {
         *out_view = entry->view;
         return true;
      }
   }

   struct yttrium_venus_sample_buffer_view *entry =
      CALLOC_STRUCT(yttrium_venus_sample_buffer_view);
   if (!entry) {
      YTTRIUM_WARN("yttrium: Venus sampled buffer view allocation failed res_id=%u buffer_id=%llu format=%u vk_format=%u size=0x%llx offset=0x%llx range=0x%llx view_range=0x%llx\n",
                   resource_id,
                   (unsigned long long)resource->buffer_obj.id,
                   pipe_format, vk_format,
                   (unsigned long long)resource->image_size,
                   (unsigned long long)offset,
                   (unsigned long long)range,
                   (unsigned long long)view_range);
      return false;
   }

   yttrium_venus_init_object(venus, &entry->obj);
   entry->view = YTTRIUM_VENUS_HANDLE(VkBufferView, &entry->obj);
   entry->next = resource->sample_buffer_views;
   resource->sample_buffer_views = entry;
   const VkBufferViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
      .buffer = resource->buffer,
      .format = vk_format,
      .offset = offset,
      .range = view_range,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateBufferView, &entry->obj, &entry->view,
      goto sample_buffer_view_submit_failed, venus->device_handle,
      &view_info, NULL, &entry->view);

   entry->vk_format = vk_format;
   entry->offset = offset;
   entry->range = view_range;
   *out_view = entry->view;
   return true;

sample_buffer_view_submit_failed:
   resource->sample_buffer_views = entry->next;
   FREE(entry);
   return false;
}

static bool
yttrium_venus_ensure_storage_buffer_view(struct yttrium_venus *venus,
                                         struct yttrium_venus_resource *resource,
                                         uint32_t resource_id,
                                         enum pipe_format pipe_format,
                                         VkDeviceSize offset,
                                         VkDeviceSize range,
                                         VkBufferView *out_view)
{
   const VkFormat vk_format = yttrium_venus2_pipe_format(pipe_format);
   const unsigned blocksize = util_format_get_blocksize(pipe_format);

   if (out_view)
      *out_view = VK_NULL_HANDLE;
   if (!resource || !resource->initialized || !resource->buffer_backed ||
       !resource->buffer || vk_format == VK_FORMAT_UNDEFINED || !blocksize ||
       !out_view)
      return false;
   if (!(resource->buffer_usage & VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT)) {
      YTTRIUM_LOG("yttrium: Venus storage buffer rejected res_id=%u buffer_id=%llu usage=0x%x format=%u vk_format=%u offset=0x%llx range=0x%llx blocksize=%u\n",
                  resource_id,
                  (unsigned long long)resource->buffer_obj.id,
                  resource->buffer_usage, pipe_format, vk_format,
                  (unsigned long long)offset,
                  (unsigned long long)range,
                  blocksize);
      return false;
   }

   if (!resource->image_size || offset > resource->image_size) {
      YTTRIUM_LOG("yttrium: Venus storage buffer view rejected range res_id=%u buffer_id=%llu size=0x%llx offset=0x%llx range=0x%llx format=%u vk_format=%u\n",
                  resource_id,
                  (unsigned long long)resource->buffer_obj.id,
                  (unsigned long long)resource->image_size,
                  (unsigned long long)offset,
                  (unsigned long long)range,
                  pipe_format, vk_format);
      return false;
   }
   if (!range)
      range = resource->image_size - offset;
   if (range > resource->image_size - offset)
      range = resource->image_size - offset;
   range -= range % blocksize;
   if (!range || offset % blocksize) {
      YTTRIUM_LOG("yttrium: Venus storage buffer view rejected alignment res_id=%u buffer_id=%llu size=0x%llx offset=0x%llx range=0x%llx format=%u vk_format=%u blocksize=%u\n",
                  resource_id,
                  (unsigned long long)resource->buffer_obj.id,
                  (unsigned long long)resource->image_size,
                  (unsigned long long)offset,
                  (unsigned long long)range,
                  pipe_format, vk_format, blocksize);
      return false;
   }

   const VkDeviceSize view_range =
      (!offset && range == resource->image_size) ||
      offset + range >= resource->image_size ?
      VK_WHOLE_SIZE : range;

   for (struct yttrium_venus_sample_buffer_view *entry =
           resource->sample_buffer_views;
        entry; entry = entry->next) {
      if (entry->view && entry->vk_format == vk_format &&
          entry->offset == offset && entry->range == view_range) {
         *out_view = entry->view;
         return true;
      }
   }

   struct yttrium_venus_sample_buffer_view *entry =
      CALLOC_STRUCT(yttrium_venus_sample_buffer_view);
   if (!entry) {
      YTTRIUM_WARN("yttrium: Venus storage buffer view allocation failed res_id=%u buffer_id=%llu format=%u vk_format=%u size=0x%llx offset=0x%llx range=0x%llx view_range=0x%llx\n",
                   resource_id,
                   (unsigned long long)resource->buffer_obj.id,
                   pipe_format, vk_format,
                   (unsigned long long)resource->image_size,
                   (unsigned long long)offset,
                   (unsigned long long)range,
                   (unsigned long long)view_range);
      return false;
   }

   yttrium_venus_init_object(venus, &entry->obj);
   entry->view = YTTRIUM_VENUS_HANDLE(VkBufferView, &entry->obj);
   entry->next = resource->sample_buffer_views;
   resource->sample_buffer_views = entry;
   const VkBufferViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
      .buffer = resource->buffer,
      .format = vk_format,
      .offset = offset,
      .range = view_range,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateBufferView, &entry->obj, &entry->view,
      goto storage_buffer_view_submit_failed, venus->device_handle,
      &view_info, NULL, &entry->view);

   entry->vk_format = vk_format;
   entry->offset = offset;
   entry->range = view_range;
   *out_view = entry->view;
   return true;

storage_buffer_view_submit_failed:
   resource->sample_buffer_views = entry->next;
   FREE(entry);
   return false;
}

static bool
yttrium_venus_prepare_textured_descriptors(struct yttrium_venus *venus,
                                           struct yttrium_venus_resource *resource,
                                           uint32_t resource_id)
{
   if (resource->descriptor_set && resource->sampler)
      return true;

   yttrium_venus_init_object(venus, &resource->descriptor_set_layout_obj);
   resource->descriptor_set_layout =
      YTTRIUM_VENUS_HANDLE(VkDescriptorSetLayout,
                            &resource->descriptor_set_layout_obj);
   const VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };
   const VkDescriptorSetLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &binding,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateDescriptorSetLayout,
      &resource->descriptor_set_layout_obj, &resource->descriptor_set_layout,
      return false,
      venus->device_handle, &layout_info, NULL,
      &resource->descriptor_set_layout);

   yttrium_venus_init_object(venus, &resource->descriptor_pool_obj);
   resource->descriptor_pool =
      YTTRIUM_VENUS_HANDLE(VkDescriptorPool,
                            &resource->descriptor_pool_obj);
   const VkDescriptorPoolSize pool_size = {
      .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
   };
   const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateDescriptorPool, &resource->descriptor_pool_obj,
      &resource->descriptor_pool, return false, venus->device_handle,
      &pool_info, NULL,
      &resource->descriptor_pool);

   yttrium_venus_init_object(venus, &resource->descriptor_set_obj);
   resource->descriptor_set =
      YTTRIUM_VENUS_HANDLE(VkDescriptorSet,
                            &resource->descriptor_set_obj);
   const VkDescriptorSetAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = resource->descriptor_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &resource->descriptor_set_layout,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkAllocateDescriptorSets, &resource->descriptor_set_obj,
      &resource->descriptor_set, return false, venus->device_handle, &alloc_info,
      &resource->descriptor_set);

   yttrium_venus_init_object(venus, &resource->sampler_obj);
   resource->sampler =
      YTTRIUM_VENUS_HANDLE(VkSampler, &resource->sampler_obj);
   const VkSamplerCreateInfo sampler_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .minLod = 0.0f,
      .maxLod = 0.0f,
      .maxAnisotropy = 1.0f,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateSampler, &resource->sampler_obj, &resource->sampler,
      return false,
      venus->device_handle, &sampler_info, NULL, &resource->sampler);

   YTTRIUM_LOG("yttrium: Venus textured descriptors res_id=%u set_layout_id=%llu pool_id=%llu set_id=%llu sampler_id=%llu\n",
                resource_id,
                (unsigned long long)resource->descriptor_set_layout_obj.id,
                (unsigned long long)resource->descriptor_pool_obj.id,
                (unsigned long long)resource->descriptor_set_obj.id,
                (unsigned long long)resource->sampler_obj.id);
   return true;
}

static bool
yttrium_venus_prepare_graphics(struct yttrium_venus *venus,
                               struct yttrium_venus_resource *resource,
                               uint32_t resource_id,
                               enum yttrium_venus_graphics_mode mode,
                               const struct yttrium_venus_draw_state *draw_state)
{
   const VkCullModeFlags cull_mode =
      draw_state ? draw_state->cull_mode : VK_CULL_MODE_NONE;
   const VkFrontFace front_face =
      draw_state ? draw_state->front_face : VK_FRONT_FACE_COUNTER_CLOCKWISE;
   const VkPrimitiveTopology topology =
      draw_state ? draw_state->topology : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   const VkBool32 blend_enable =
      draw_state ? draw_state->blend_enable : VK_FALSE;
   const VkColorComponentFlags color_write_mask =
      draw_state ? draw_state->color_write_mask :
                   VK_COLOR_COMPONENT_R_BIT |
                   VK_COLOR_COMPONENT_G_BIT |
                   VK_COLOR_COMPONENT_B_BIT |
                   VK_COLOR_COMPONENT_A_BIT;
   const VkBlendFactor src_color_blend_factor =
      draw_state ? draw_state->src_color_blend_factor : VK_BLEND_FACTOR_ONE;
   const VkBlendFactor dst_color_blend_factor =
      draw_state ? draw_state->dst_color_blend_factor : VK_BLEND_FACTOR_ZERO;
   const VkBlendOp color_blend_op =
      draw_state ? draw_state->color_blend_op : VK_BLEND_OP_ADD;
   const VkBlendFactor src_alpha_blend_factor =
      draw_state ? draw_state->src_alpha_blend_factor : VK_BLEND_FACTOR_ONE;
   const VkBlendFactor dst_alpha_blend_factor =
      draw_state ? draw_state->dst_alpha_blend_factor : VK_BLEND_FACTOR_ZERO;
   const VkBlendOp alpha_blend_op =
      draw_state ? draw_state->alpha_blend_op : VK_BLEND_OP_ADD;
   const VkSampleMask sample_mask =
      draw_state ? draw_state->sample_mask : ~0u;
   const VkBool32 alpha_to_coverage_enable =
      draw_state ? draw_state->alpha_to_coverage_enable : VK_FALSE;
   const uint32_t render_level = draw_state ? draw_state->render_level : 0;
   const uint32_t render_layer = draw_state ? draw_state->render_layer : 0;
   const uint32_t render_layers =
      draw_state ? MAX2(draw_state->render_layers, 1) : 1;
   const uint32_t render_width =
      draw_state && draw_state->render_width ?
      draw_state->render_width :
      yttrium_venus_subresource_width(resource, render_level);
   const uint32_t render_height =
      draw_state && draw_state->render_height ?
      draw_state->render_height :
      yttrium_venus_subresource_height(resource, render_level);

   if (!resource)
      return false;

   const VkSampleCountFlagBits render_samples =
      resource->samples ? resource->samples : VK_SAMPLE_COUNT_1_BIT;

   if (resource->graphics_ready) {
      if (resource->graphics_mode == mode &&
          resource->graphics_level == render_level &&
          resource->graphics_layer == render_layer &&
          resource->graphics_layers == render_layers &&
          resource->graphics_topology == topology &&
          resource->graphics_cull_mode == cull_mode &&
          resource->graphics_front_face == front_face &&
          resource->graphics_blend_enable == blend_enable &&
          resource->graphics_sample_mask == sample_mask &&
          resource->graphics_alpha_to_coverage_enable ==
             alpha_to_coverage_enable &&
          resource->graphics_color_write_mask == color_write_mask &&
          resource->graphics_src_color_blend_factor == src_color_blend_factor &&
          resource->graphics_dst_color_blend_factor == dst_color_blend_factor &&
          resource->graphics_color_blend_op == color_blend_op &&
          resource->graphics_src_alpha_blend_factor == src_alpha_blend_factor &&
          resource->graphics_dst_alpha_blend_factor == dst_alpha_blend_factor &&
          resource->graphics_alpha_blend_op == alpha_blend_op)
         return true;
      yttrium_venus_destroy_graphics_objects(venus, resource);
   }

   if (!resource->initialized || resource->buffer_backed || !resource->image ||
       !(resource->image_usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ||
       resource->vk_format == VK_FORMAT_UNDEFINED || !render_width ||
       !render_height ||
       !yttrium_venus_valid_render_subresource(resource, render_level,
                                               render_layer, render_layers)) {
      YTTRIUM_LOG("yttrium: Venus draw setup rejected res_id=%u initialized=%u buffer_backed=%u usage=0x%x format=%u extent=%ux%u subresource=%u/%u+%u sub_extent=%ux%u\n",
                   resource_id, resource->initialized, resource->buffer_backed,
                   resource->image_usage, resource->vk_format,
                   resource->width, resource->height,
                   render_level, render_layer, render_layers,
                   render_width, render_height);
      return false;
   }

   yttrium_venus_init_object(venus, &resource->image_view_obj);
   resource->image_view =
      YTTRIUM_VENUS_HANDLE(VkImageView, &resource->image_view_obj);
   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = resource->image,
      .viewType = yttrium_venus_render_view_type(resource, render_layers),
      .format = resource->vk_format,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .baseMipLevel = render_level,
         .levelCount = 1,
         .baseArrayLayer = render_layer,
         .layerCount = render_layers,
      },
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateImageView, &resource->image_view_obj,
      &resource->image_view,
      goto graphics_submit_failed, venus->device_handle, &view_info, NULL,
      &resource->image_view);

   yttrium_venus_init_object(venus, &resource->render_pass_obj);
   resource->render_pass =
      YTTRIUM_VENUS_HANDLE(VkRenderPass, &resource->render_pass_obj);
   const VkAttachmentDescription color_attachment = {
      .format = resource->vk_format,
      .samples = render_samples,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkAttachmentReference color_ref = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_ref,
   };
   const VkRenderPassCreateInfo render_pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &color_attachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateRenderPass, &resource->render_pass_obj,
      &resource->render_pass,
      goto graphics_submit_failed, venus->device_handle, &render_pass_info,
      NULL, &resource->render_pass);

   yttrium_venus_init_object(venus, &resource->framebuffer_obj);
   resource->framebuffer =
      YTTRIUM_VENUS_HANDLE(VkFramebuffer, &resource->framebuffer_obj);
   const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = resource->render_pass,
      .attachmentCount = 1,
      .pAttachments = &resource->image_view,
      .width = render_width,
      .height = render_height,
      .layers = render_layers,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateFramebuffer, &resource->framebuffer_obj,
      &resource->framebuffer,
      goto graphics_submit_failed, venus->device_handle, &framebuffer_info,
      NULL, &resource->framebuffer);

   const bool textured =
      mode == YTTRIUM_VENUS_GRAPHICS_TEXTURED_VERTEX_BUFFER;
   if (textured &&
       !yttrium_venus_prepare_textured_descriptors(venus, resource,
                                                   resource_id)) {
      yttrium_venus_destroy_graphics_objects(venus, resource);
      return false;
   }

   yttrium_venus_init_object(venus, &resource->pipeline_layout_obj);
   resource->pipeline_layout =
      YTTRIUM_VENUS_HANDLE(VkPipelineLayout,
                            &resource->pipeline_layout_obj);
   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = textured ? 1 : 0,
      .pSetLayouts = textured ? &resource->descriptor_set_layout : NULL,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreatePipelineLayout, &resource->pipeline_layout_obj,
      &resource->pipeline_layout,
      goto graphics_submit_failed, venus->device_handle, &layout_info, NULL,
      &resource->pipeline_layout);

   const uint32_t *vs_code = textured ?
      yttrium_textured_vs_spv : yttrium_vertex_input_vs_spv;
   const size_t vs_size = textured ?
      sizeof(yttrium_textured_vs_spv) : sizeof(yttrium_vertex_input_vs_spv);
   const uint32_t *fs_code = textured ?
      yttrium_textured_fs_spv : yttrium_color_passthrough_fs_spv;
   const size_t fs_size = textured ?
      sizeof(yttrium_textured_fs_spv) :
      sizeof(yttrium_color_passthrough_fs_spv);

   if (!yttrium_venus2_create_shader_module(venus, &resource->vertex_shader_obj,
                                           &resource->vertex_shader,
                                           vs_code, vs_size, "vertex",
                                           NULL) ||
       !yttrium_venus2_create_shader_module(venus,
                                           &resource->fragment_shader_obj,
                                           &resource->fragment_shader,
                                           fs_code, fs_size, "fragment",
                                           NULL)) {
      yttrium_venus_destroy_graphics_objects(venus, resource);
      return false;
   }

   const VkPipelineShaderStageCreateInfo stages[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = resource->vertex_shader,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = resource->fragment_shader,
         .pName = "main",
      },
   };
   const VkVertexInputBindingDescription vertex_binding = {
      .binding = 0,
      .stride = mode == YTTRIUM_VENUS_GRAPHICS_TEXTURED_VERTEX_BUFFER ?
         sizeof(struct yttrium_venus_textured_vertex) :
         sizeof(struct yttrium_venus_triangle_vertex),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription vertex_attributes[2] = {
      {
         .location = 0,
         .binding = 0,
         .format = VK_FORMAT_R32G32B32A32_SFLOAT,
         .offset = offsetof(struct yttrium_venus_triangle_vertex, position),
      },
      {
         .location = 1,
         .binding = 0,
         .format = VK_FORMAT_R32G32B32A32_SFLOAT,
         .offset = offsetof(struct yttrium_venus_triangle_vertex, color),
      },
   };
   const VkVertexInputAttributeDescription textured_vertex_attributes[3] = {
      {
         .location = 0,
         .binding = 0,
         .format = VK_FORMAT_R32G32B32A32_SFLOAT,
         .offset = offsetof(struct yttrium_venus_textured_vertex, position),
      },
      {
         .location = 1,
         .binding = 0,
         .format = VK_FORMAT_R32G32B32A32_SFLOAT,
         .offset = offsetof(struct yttrium_venus_textured_vertex, color),
      },
      {
         .location = 2,
         .binding = 0,
         .format = VK_FORMAT_R32G32_SFLOAT,
         .offset = offsetof(struct yttrium_venus_textured_vertex, texcoord),
      },
   };
   const uint32_t vertex_attribute_count = textured ? 3 :
      2;
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding,
      .vertexAttributeDescriptionCount = vertex_attribute_count,
      .pVertexAttributeDescriptions = textured ?
         textured_vertex_attributes : vertex_attributes,
   };
   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = topology,
   };
   const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
   };
   const VkPipelineRasterizationStateCreateInfo raster = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = cull_mode,
      .frontFace = front_face,
      .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = render_samples,
      .alphaToCoverageEnable = alpha_to_coverage_enable,
      .pSampleMask = &sample_mask,
   };
   const VkPipelineColorBlendAttachmentState color_blend_attachment = {
      .blendEnable = blend_enable,
      .srcColorBlendFactor = src_color_blend_factor,
      .dstColorBlendFactor = dst_color_blend_factor,
      .colorBlendOp = color_blend_op,
      .srcAlphaBlendFactor = src_alpha_blend_factor,
      .dstAlphaBlendFactor = dst_alpha_blend_factor,
      .alphaBlendOp = alpha_blend_op,
      .colorWriteMask = color_write_mask,
   };
   const VkPipelineColorBlendStateCreateInfo color_blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &color_blend_attachment,
   };
   const VkDynamicState dynamic_states[3] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
      VK_DYNAMIC_STATE_BLEND_CONSTANTS,
   };
   const VkPipelineDynamicStateCreateInfo dynamic = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 3,
      .pDynamicStates = dynamic_states,
   };

   yttrium_venus_init_object(venus, &resource->pipeline_obj);
   resource->pipeline = YTTRIUM_VENUS_HANDLE(VkPipeline,
                                             &resource->pipeline_obj);
   const VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &input_assembly,
      .pViewportState = &viewport_state,
      .pRasterizationState = &raster,
      .pMultisampleState = &multisample,
      .pColorBlendState = &color_blend,
      .pDynamicState = &dynamic,
      .layout = resource->pipeline_layout,
      .renderPass = resource->render_pass,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateGraphicsPipelines, &resource->pipeline_obj,
      &resource->pipeline,
      goto graphics_submit_failed, venus->device_handle, VK_NULL_HANDLE, 1,
      &pipeline_info, NULL, &resource->pipeline);

   resource->graphics_ready = true;
   resource->graphics_mode = mode;
   resource->graphics_level = render_level;
   resource->graphics_layer = render_layer;
   resource->graphics_layers = render_layers;
   resource->graphics_topology = topology;
   resource->graphics_cull_mode = cull_mode;
   resource->graphics_front_face = front_face;
   resource->graphics_blend_enable = blend_enable;
   resource->graphics_sample_mask = sample_mask;
   resource->graphics_alpha_to_coverage_enable =
      alpha_to_coverage_enable;
   resource->graphics_color_write_mask = color_write_mask;
   resource->graphics_src_color_blend_factor = src_color_blend_factor;
   resource->graphics_dst_color_blend_factor = dst_color_blend_factor;
   resource->graphics_color_blend_op = color_blend_op;
   resource->graphics_src_alpha_blend_factor = src_alpha_blend_factor;
   resource->graphics_dst_alpha_blend_factor = dst_alpha_blend_factor;
   resource->graphics_alpha_blend_op = alpha_blend_op;
   const char *mode_name =
      mode == YTTRIUM_VENUS_GRAPHICS_TEXTURED_VERTEX_BUFFER ? "textured" :
      "vertex-buffer";
   YTTRIUM_LOG("yttrium: Venus graphics setup res_id=%u image_id=%llu pipeline_id=%llu extent=%ux%u mode=%s topology=%u cull=0x%x front=%u blend=%u color_mask=0x%x sample_mask=0x%x rgb=(%u,%u,%u) alpha=(%u,%u,%u)\n",
                resource_id, (unsigned long long)resource->image_obj.id,
                (unsigned long long)resource->pipeline_obj.id,
                resource->width, resource->height, mode_name,
                topology, cull_mode, front_face,
                blend_enable, color_write_mask,
                sample_mask,
                src_color_blend_factor, dst_color_blend_factor,
                color_blend_op, src_alpha_blend_factor,
                dst_alpha_blend_factor, alpha_blend_op);
   return true;

graphics_submit_failed:
   yttrium_venus_destroy_graphics_objects(venus, resource);
   return false;
}

static bool
yttrium_venus_ensure_draw_vertex_buffer(struct yttrium_venus *venus,
                                        struct yttrium_venus_resource *resource,
                                        uint32_t resource_id,
                                        VkDeviceSize size,
                                        bool force_new_generation)
{
   if (!force_new_generation &&
       resource->draw_vertex_buffer && resource->draw_vertex_memory &&
       resource->draw_vertex_buffer_size >= size)
      return true;

   const VkDeviceSize minimum_size =
      MAX2(size, resource->draw_vertex_buffer_size);

   if (resource->draw_vertex_buffer || resource->draw_vertex_memory) {
      if (!yttrium_venus_flush_command_batch(venus,
                                             "vertex backing rotate"))
         return false;

      struct yttrium_venus_batch *batch =
         yttrium_venus_find_latest_resource_batch(venus, resource);
      struct yttrium_venus_retired_resource *retired =
         yttrium_venus_retired_draw_vertex_backing_take(resource);
      if (!retired) {
         YTTRIUM_WARN("yttrium: WARNING: vertex backing generation allocation failed owner=venus2 reason=bounded-memory-exhaustion action=synchronous-resource-retirement resource=%u\n",
                      resource_id);
         if (!yttrium_venus_wait_resource_batches(
                venus, resource,
                "vertex backing bounded-memory exhaustion"))
            return false;
      } else if (batch) {
         yttrium_venus_batch_retire_resource(batch, retired);
      } else {
         yttrium_venus_recycle_draw_backing(venus, retired);
      }
   }

   yttrium_venus_unmap_memory(venus, &resource->draw_vertex_mapping);
   if (resource->draw_vertex_buffer)
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->draw_vertex_buffer, NULL);
   if (resource->draw_vertex_memory)
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            resource->draw_vertex_memory, NULL);

   memset(&resource->draw_vertex_buffer_obj, 0,
          sizeof(resource->draw_vertex_buffer_obj));
   memset(&resource->draw_vertex_memory_obj, 0,
          sizeof(resource->draw_vertex_memory_obj));
   resource->draw_vertex_buffer = VK_NULL_HANDLE;
   resource->draw_vertex_memory = VK_NULL_HANDLE;
   resource->draw_vertex_buffer_size = 0;
   /* Fresh arena - the roll has to start over with it. */
   resource->draw_vertex_roll_base = 0;
   memset(&resource->draw_vertex_mapping, 0,
          sizeof(resource->draw_vertex_mapping));

   const VkDeviceSize allocation_size =
      yttrium_venus_pre_sized_allocation(
         minimum_size, yttrium_venus_vertex_arena_pre_size());
   if (yttrium_venus_take_draw_backing(
          venus, resource, YTTRIUM_VENUS_DRAW_BACKING_VERTEX,
          allocation_size))
      return true;

   yttrium_venus_init_object(venus, &resource->draw_vertex_buffer_obj);
   resource->draw_vertex_buffer =
      YTTRIUM_VENUS_HANDLE(VkBuffer, &resource->draw_vertex_buffer_obj);

   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = allocation_size,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateBuffer, &resource->draw_vertex_buffer_obj,
      &resource->draw_vertex_buffer,
      goto vertex_submit_failed, venus->device_handle, &buffer_info, NULL,
      &resource->draw_vertex_buffer);

   VkMemoryRequirements reqs;
   memset(&reqs, 0, sizeof(reqs));
   vn_call_vkGetBufferMemoryRequirements(&venus->vn_ring,
                                         venus->device_handle,
                                         resource->draw_vertex_buffer, &reqs);

   /*
    * These persistently mapped arenas are streaming uploads.  Prefer cached
    * system memory by default, but allow a process-local A/B test of a
    * host-visible device-local/BAR heap.
    */
   const VkMemoryPropertyFlags preferred_memory_properties =
      venus->draw_arena_bar ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                            : VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
   uint32_t memory_type_index =
      yttrium_venus_choose_memory_type(
         venus, reqs.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
         preferred_memory_properties);
   const bool direct_map_candidate = memory_type_index != UINT32_MAX;
   if (memory_type_index == UINT32_MAX) {
      memory_type_index =
         yttrium_venus_choose_memory_type(
            venus, reqs.memoryTypeBits, 0,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   }
   if (memory_type_index == UINT32_MAX) {
      YTTRIUM_LOG("yttrium: Venus no memory type for vertex buffer bits=0x%x size=0x%llx\n",
                   reqs.memoryTypeBits,
                   (unsigned long long)allocation_size);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->draw_vertex_buffer, NULL);
      memset(&resource->draw_vertex_buffer_obj, 0,
             sizeof(resource->draw_vertex_buffer_obj));
      resource->draw_vertex_buffer = VK_NULL_HANDLE;
      return false;
   }

   yttrium_venus_init_object(venus, &resource->draw_vertex_memory_obj);
   resource->draw_vertex_memory =
      YTTRIUM_VENUS_HANDLE(VkDeviceMemory,
                            &resource->draw_vertex_memory_obj);

   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = MAX2(reqs.size, allocation_size),
      .memoryTypeIndex = memory_type_index,
   };
   VkResult result =
      direct_map_candidate ?
      vn_call_vkAllocateMemory(&venus->vn_ring, venus->device_handle,
                               &memory_info, NULL,
                               &resource->draw_vertex_memory) :
      VK_SUCCESS;
   if (!direct_map_candidate) {
      YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
         venus, vkAllocateMemory, &resource->draw_vertex_memory_obj,
         &resource->draw_vertex_memory,
         goto vertex_submit_failed, venus->device_handle, &memory_info, NULL,
         &resource->draw_vertex_memory);
   } else if (result != VK_SUCCESS) {
      YTTRIUM_WARN("yttrium: ERROR: vertex backing allocation failed owner=venus2 result=%d type=%u size=0x%llx bits=0x%x\n",
                   result, memory_type_index,
                   (unsigned long long)memory_info.allocationSize,
                   reqs.memoryTypeBits);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->draw_vertex_buffer, NULL);
      memset(&resource->draw_vertex_buffer_obj, 0,
             sizeof(resource->draw_vertex_buffer_obj));
      memset(&resource->draw_vertex_memory_obj, 0,
             sizeof(resource->draw_vertex_memory_obj));
      resource->draw_vertex_buffer = VK_NULL_HANDLE;
      resource->draw_vertex_memory = VK_NULL_HANDLE;
      return false;
   }

   result = direct_map_candidate ?
      vn_call_vkBindBufferMemory(&venus->vn_ring, venus->device_handle,
                                 resource->draw_vertex_buffer,
                                 resource->draw_vertex_memory, 0) :
      VK_SUCCESS;
   if (!direct_map_candidate) {
      YTTRIUM_VENUS_SUBMIT_COMMAND_OR(
         venus, vkBindBufferMemory, resource->draw_vertex_buffer_obj.id,
         goto vertex_submit_failed, venus->device_handle,
         resource->draw_vertex_buffer, resource->draw_vertex_memory, 0);
   } else if (result != VK_SUCCESS) {
      YTTRIUM_LOG("yttrium: Venus vertex buffer bind failed result=%d memory_id=%llu\n",
                  result,
                  (unsigned long long)resource->draw_vertex_memory_obj.id);
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            resource->draw_vertex_memory, NULL);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->draw_vertex_buffer, NULL);
      memset(&resource->draw_vertex_buffer_obj, 0,
             sizeof(resource->draw_vertex_buffer_obj));
      memset(&resource->draw_vertex_memory_obj, 0,
             sizeof(resource->draw_vertex_memory_obj));
      resource->draw_vertex_buffer = VK_NULL_HANDLE;
      resource->draw_vertex_memory = VK_NULL_HANDLE;
      return false;
   }

   if (direct_map_candidate &&
       yttrium_venus_memory_type_has_flags(
          venus, memory_type_index,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      if (!yttrium_venus_map_memory(
             venus, resource->draw_vertex_memory_obj.id,
             memory_info.allocationSize, &resource->draw_vertex_mapping)) {
         YTTRIUM_WARN("yttrium: WARNING: Venus direct CPU vertex upload map failed; falling back to vkCmdUpdateBuffer resource=%u memory_id=%llu size=0x%llx\n",
                      resource_id,
                      (unsigned long long)resource->draw_vertex_memory_obj.id,
                      (unsigned long long)memory_info.allocationSize);
      }
   } else {
      YTTRIUM_WARN("yttrium: WARNING: Venus direct CPU vertex upload unavailable; falling back to vkCmdUpdateBuffer resource=%u type=%u bits=0x%x size=0x%llx\n",
                   resource_id, memory_type_index, reqs.memoryTypeBits,
                   (unsigned long long)memory_info.allocationSize);
   }

   resource->draw_vertex_buffer_size = memory_info.allocationSize;
   resource->draw_vertex_buffer_generation++;
   YTTRIUM_LOG("yttrium: Venus vertex buffer setup res_id=%u buffer_id=%llu memory_id=%llu size=0x%llx req_size=0x%llx type=%u bits=0x%x\n",
                resource_id,
                (unsigned long long)resource->draw_vertex_buffer_obj.id,
                (unsigned long long)resource->draw_vertex_memory_obj.id,
                (unsigned long long)memory_info.allocationSize,
                (unsigned long long)reqs.size,
                memory_type_index, reqs.memoryTypeBits);
   return true;

vertex_submit_failed:
   if (resource->draw_vertex_memory)
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            resource->draw_vertex_memory, NULL);
   if (resource->draw_vertex_buffer)
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->draw_vertex_buffer, NULL);
   memset(&resource->draw_vertex_buffer_obj, 0,
          sizeof(resource->draw_vertex_buffer_obj));
   memset(&resource->draw_vertex_memory_obj, 0,
          sizeof(resource->draw_vertex_memory_obj));
   resource->draw_vertex_buffer = VK_NULL_HANDLE;
   resource->draw_vertex_memory = VK_NULL_HANDLE;
   resource->draw_vertex_buffer_size = 0;
   memset(&resource->draw_vertex_mapping, 0,
          sizeof(resource->draw_vertex_mapping));
   return false;
}

static bool
yttrium_venus_ensure_draw_index_buffer(struct yttrium_venus *venus,
                                       struct yttrium_venus_resource *resource,
                                       uint32_t resource_id,
                                       VkDeviceSize size,
                                       bool force_new_generation)
{
   if (!force_new_generation &&
       resource->draw_index_buffer && resource->draw_index_memory &&
       resource->draw_index_buffer_size >= size)
      return true;

   const VkDeviceSize minimum_size =
      MAX2(size, resource->draw_index_buffer_size);

   if (resource->draw_index_buffer || resource->draw_index_memory) {
      if (!yttrium_venus_flush_command_batch(venus,
                                             "index backing rotate"))
         return false;

      struct yttrium_venus_batch *batch =
         yttrium_venus_find_latest_resource_batch(venus, resource);
      struct yttrium_venus_retired_resource *retired =
         yttrium_venus_retired_draw_index_backing_take(resource);
      if (!retired) {
         YTTRIUM_WARN("yttrium: WARNING: index backing generation allocation failed owner=venus2 reason=bounded-memory-exhaustion action=synchronous-resource-retirement resource=%u\n",
                      resource_id);
         if (!yttrium_venus_wait_resource_batches(
                venus, resource,
                "index backing bounded-memory exhaustion"))
            return false;
      } else if (batch) {
         yttrium_venus_batch_retire_resource(batch, retired);
      } else {
         yttrium_venus_recycle_draw_backing(venus, retired);
      }
   }

   yttrium_venus_unmap_memory(venus, &resource->draw_index_mapping);
   if (resource->draw_index_buffer)
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->draw_index_buffer, NULL);
   if (resource->draw_index_memory)
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            resource->draw_index_memory, NULL);

   memset(&resource->draw_index_buffer_obj, 0,
          sizeof(resource->draw_index_buffer_obj));
   memset(&resource->draw_index_memory_obj, 0,
          sizeof(resource->draw_index_memory_obj));
   resource->draw_index_buffer = VK_NULL_HANDLE;
   resource->draw_index_memory = VK_NULL_HANDLE;
   resource->draw_index_buffer_size = 0;
   /* Fresh arena - the roll has to start over with it. */
   resource->draw_index_roll_base = 0;
   memset(&resource->draw_index_mapping, 0,
          sizeof(resource->draw_index_mapping));

   const VkDeviceSize allocation_size =
      yttrium_venus_pre_sized_allocation(
         minimum_size, yttrium_venus_index_arena_pre_size());
   if (yttrium_venus_take_draw_backing(
          venus, resource, YTTRIUM_VENUS_DRAW_BACKING_INDEX,
          allocation_size))
      return true;

   yttrium_venus_init_object(venus, &resource->draw_index_buffer_obj);
   resource->draw_index_buffer =
      YTTRIUM_VENUS_HANDLE(VkBuffer, &resource->draw_index_buffer_obj);

   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = allocation_size,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateBuffer, &resource->draw_index_buffer_obj,
      &resource->draw_index_buffer,
      goto index_submit_failed, venus->device_handle, &buffer_info, NULL,
      &resource->draw_index_buffer);

   VkMemoryRequirements reqs;
   memset(&reqs, 0, sizeof(reqs));
   vn_call_vkGetBufferMemoryRequirements(&venus->vn_ring,
                                         venus->device_handle,
                                         resource->draw_index_buffer, &reqs);

   /* Match the process-selected vertex-arena heap preference. */
   const VkMemoryPropertyFlags preferred_memory_properties =
      venus->draw_arena_bar ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                            : VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
   uint32_t memory_type_index =
      yttrium_venus_choose_memory_type(
         venus, reqs.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
         preferred_memory_properties);
   const bool direct_map_candidate = memory_type_index != UINT32_MAX;
   if (memory_type_index == UINT32_MAX) {
      memory_type_index =
         yttrium_venus_choose_memory_type(
            venus, reqs.memoryTypeBits, 0,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   }
   if (memory_type_index == UINT32_MAX) {
      YTTRIUM_LOG("yttrium: Venus no memory type for index buffer bits=0x%x size=0x%llx\n",
                  reqs.memoryTypeBits,
                  (unsigned long long)allocation_size);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->draw_index_buffer, NULL);
      memset(&resource->draw_index_buffer_obj, 0,
             sizeof(resource->draw_index_buffer_obj));
      resource->draw_index_buffer = VK_NULL_HANDLE;
      return false;
   }

   yttrium_venus_init_object(venus, &resource->draw_index_memory_obj);
   resource->draw_index_memory =
      YTTRIUM_VENUS_HANDLE(VkDeviceMemory,
                           &resource->draw_index_memory_obj);

   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = MAX2(reqs.size, allocation_size),
      .memoryTypeIndex = memory_type_index,
   };
   VkResult result =
      direct_map_candidate ?
      vn_call_vkAllocateMemory(&venus->vn_ring, venus->device_handle,
                               &memory_info, NULL,
                               &resource->draw_index_memory) :
      VK_SUCCESS;
   if (!direct_map_candidate) {
      YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
         venus, vkAllocateMemory, &resource->draw_index_memory_obj,
         &resource->draw_index_memory,
         goto index_submit_failed, venus->device_handle, &memory_info, NULL,
         &resource->draw_index_memory);
   } else if (result != VK_SUCCESS) {
      YTTRIUM_WARN("yttrium: ERROR: index backing allocation failed owner=venus2 result=%d type=%u size=0x%llx bits=0x%x\n",
                   result, memory_type_index,
                   (unsigned long long)memory_info.allocationSize,
                   reqs.memoryTypeBits);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->draw_index_buffer, NULL);
      memset(&resource->draw_index_buffer_obj, 0,
             sizeof(resource->draw_index_buffer_obj));
      memset(&resource->draw_index_memory_obj, 0,
             sizeof(resource->draw_index_memory_obj));
      resource->draw_index_buffer = VK_NULL_HANDLE;
      resource->draw_index_memory = VK_NULL_HANDLE;
      return false;
   }

   result = direct_map_candidate ?
      vn_call_vkBindBufferMemory(&venus->vn_ring, venus->device_handle,
                                 resource->draw_index_buffer,
                                 resource->draw_index_memory, 0) :
      VK_SUCCESS;
   if (!direct_map_candidate) {
      YTTRIUM_VENUS_SUBMIT_COMMAND_OR(
         venus, vkBindBufferMemory, resource->draw_index_buffer_obj.id,
         goto index_submit_failed, venus->device_handle,
         resource->draw_index_buffer, resource->draw_index_memory, 0);
   } else if (result != VK_SUCCESS) {
      YTTRIUM_LOG("yttrium: Venus index buffer bind failed result=%d memory_id=%llu\n",
                  result,
                  (unsigned long long)resource->draw_index_memory_obj.id);
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            resource->draw_index_memory, NULL);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->draw_index_buffer, NULL);
      memset(&resource->draw_index_buffer_obj, 0,
             sizeof(resource->draw_index_buffer_obj));
      memset(&resource->draw_index_memory_obj, 0,
             sizeof(resource->draw_index_memory_obj));
      resource->draw_index_buffer = VK_NULL_HANDLE;
      resource->draw_index_memory = VK_NULL_HANDLE;
      return false;
   }

   if (direct_map_candidate &&
       yttrium_venus_memory_type_has_flags(
          venus, memory_type_index,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      if (!yttrium_venus_map_memory(
             venus, resource->draw_index_memory_obj.id,
             memory_info.allocationSize, &resource->draw_index_mapping)) {
         YTTRIUM_WARN("yttrium: WARNING: Venus direct CPU index upload map failed; falling back to vkCmdUpdateBuffer resource=%u memory_id=%llu size=0x%llx\n",
                      resource_id,
                      (unsigned long long)resource->draw_index_memory_obj.id,
                      (unsigned long long)memory_info.allocationSize);
      }
   } else {
      YTTRIUM_WARN("yttrium: WARNING: Venus direct CPU index upload unavailable; falling back to vkCmdUpdateBuffer resource=%u type=%u bits=0x%x size=0x%llx\n",
                   resource_id, memory_type_index, reqs.memoryTypeBits,
                   (unsigned long long)memory_info.allocationSize);
   }

   resource->draw_index_buffer_size = memory_info.allocationSize;
   resource->draw_index_buffer_generation++;
   YTTRIUM_LOG("yttrium: Venus index buffer setup res_id=%u buffer_id=%llu memory_id=%llu size=0x%llx req_size=0x%llx type=%u bits=0x%x\n",
               resource_id,
               (unsigned long long)resource->draw_index_buffer_obj.id,
               (unsigned long long)resource->draw_index_memory_obj.id,
               (unsigned long long)memory_info.allocationSize,
               (unsigned long long)reqs.size,
               memory_type_index, reqs.memoryTypeBits);
   return true;

index_submit_failed:
   if (resource->draw_index_memory)
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            resource->draw_index_memory, NULL);
   if (resource->draw_index_buffer)
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               resource->draw_index_buffer, NULL);
   memset(&resource->draw_index_buffer_obj, 0,
          sizeof(resource->draw_index_buffer_obj));
   memset(&resource->draw_index_memory_obj, 0,
          sizeof(resource->draw_index_memory_obj));
   resource->draw_index_buffer = VK_NULL_HANDLE;
   resource->draw_index_memory = VK_NULL_HANDLE;
   resource->draw_index_buffer_size = 0;
   memset(&resource->draw_index_mapping, 0,
          sizeof(resource->draw_index_mapping));
   return false;
}

static void
yttrium_venus_cmd_update_buffer_chunks(struct yttrium_venus *venus,
                                       VkCommandBuffer command_buffer,
                                       VkBuffer buffer,
                                       VkDeviceSize offset,
                                       VkDeviceSize size,
                                       const void *data)
{
   const VkDeviceSize max_update = 65536;
   const uint8_t *bytes = (const uint8_t *)data;

   while (size) {
      VkDeviceSize chunk = MIN2(size, max_update);

      assert((offset & 3) == 0);
      assert((chunk & 3) == 0);
      vn_async_vkCmdUpdateBuffer(&venus->vn_ring, command_buffer, buffer,
                                 offset, chunk, bytes);

      offset += chunk;
      bytes += chunk;
      size -= chunk;
   }
}

static bool
yttrium_venus_cmd_update_buffer_padded(struct yttrium_venus *venus,
                                       VkCommandBuffer command_buffer,
                                       VkBuffer buffer,
                                       VkDeviceSize offset,
                                       VkDeviceSize size,
                                       const void *data,
                                       VkDeviceSize *out_update_size)
{
   if (out_update_size)
      *out_update_size = 0;

   if (!size || !data)
      return false;

   const VkDeviceSize update_size = align64(size, 4);
   if (out_update_size)
      *out_update_size = update_size;

   if (update_size == size) {
      yttrium_venus_cmd_update_buffer_chunks(venus, command_buffer, buffer,
                                             offset, size, data);
      return true;
   }

   uint8_t stack_data[256];
   uint8_t *upload = update_size <= sizeof(stack_data) ?
      stack_data : MALLOC(update_size);
   if (!upload)
      return false;

   memcpy(upload, data, size);
   memset(upload + size, 0, update_size - size);
   yttrium_venus_cmd_update_buffer_chunks(venus, command_buffer, buffer,
                                          offset, update_size, upload);
   if (upload != stack_data)
      FREE(upload);
   return true;
}

static bool
yttrium_venus_copy_mapped_padded(void *map,
                                 VkDeviceSize map_size,
                                 VkDeviceSize offset,
                                 VkDeviceSize size,
                                 const void *data,
                                 VkDeviceSize *out_update_size)
{
   if (out_update_size)
      *out_update_size = 0;

   if (!map || !data || !size || offset > map_size)
      return false;

   const VkDeviceSize update_size = align64(size, 4);
   if (update_size < size || update_size > map_size - offset)
      return false;

   uint8_t *dst = (uint8_t *)map + offset;
   memcpy(dst, data, size);
   if (update_size != size)
      memset(dst + size, 0, update_size - size);

   if (out_update_size)
      *out_update_size = update_size;
   return true;
}

bool
yttrium_venus2_update_buffer(struct yttrium_venus *venus,
                            struct yttrium_venus_resource *resource,
                            uint64_t offset,
                            uint64_t size,
                            const void *data)
{
   const uint64_t padded_size = align64(size, 4);
   VkDeviceSize update_size;

   if (!venus || !resource || !resource->initialized ||
       !resource->buffer_backed || !resource->buffer || !data || !size ||
       (offset & 3) ||
       offset > resource->allocation_size ||
       size > resource->allocation_size - offset ||
       padded_size > resource->allocation_size - offset)
      return false;

   if (!yttrium_venus_begin_transfer_batch(venus))
      return false;

   if (!yttrium_venus_cmd_batch_track_resource(venus, resource)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "buffer update tracking failure");
      return false;
   }

   /* A partial argument update can follow an indirect draw using the same
    * bytes.  Queue order alone does not provide that read-to-write hazard. */
   const VkBufferMemoryBarrier overwrite_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                       VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = resource->buffer,
      .offset = offset,
      .size = padded_size,
   };
   vn_async_vkCmdPipelineBarrier(
      &venus->vn_ring, venus->command_buffer,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
      0, NULL, 1, &overwrite_barrier, 0, NULL);

   if (!yttrium_venus_cmd_update_buffer_padded(venus, venus->command_buffer,
                                               resource->buffer, offset, size,
                                               data, &update_size)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "buffer update setup failure");
      return false;
   }
   yttrium_trace_venus_upload(YTTRIUM_TRACE_VENUS_UPLOAD_UPDATE_BUFFER,
                              0, update_size, 0, resource->buffer_obj.id,
                              0, 0, 0, 0, 0, 0, 0);

   const VkBufferMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT |
                       VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = resource->buffer,
      .offset = offset,
      .size = update_size,
   };
   vn_async_vkCmdPipelineBarrier(&venus->vn_ring, venus->command_buffer,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT, 0,
                                 0, NULL, 1, &barrier, 0, NULL);

   return yttrium_venus_cmd_batch_after_record(venus, "buffer update");
}

bool
yttrium_venus2_clear_buffer(struct yttrium_venus *venus,
                            struct yttrium_venus_resource *resource,
                            uint64_t offset,
                            uint64_t size,
                            uint32_t value)
{
   if (!venus || !resource || !resource->initialized ||
       !resource->buffer_backed || !resource->buffer || !size ||
       (offset & 3) || (size & 3) ||
       offset > resource->allocation_size ||
       size > resource->allocation_size - offset)
      return false;

   if (!yttrium_venus_begin_transfer_batch(venus))
      return false;

   if (!yttrium_venus_cmd_batch_track_resource(venus, resource)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "buffer clear tracking failure");
      return false;
   }

   const VkBufferMemoryBarrier overwrite = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                       VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = resource->buffer,
      .offset = offset,
      .size = size,
   };
   vn_async_vkCmdPipelineBarrier(
      &venus->vn_ring, venus->command_buffer,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
      0, NULL, 1, &overwrite, 0, NULL);

   vn_async_vkCmdFillBuffer(&venus->vn_ring, venus->command_buffer,
                            resource->buffer, offset, size, value);

   const VkBufferMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT |
                       VK_ACCESS_MEMORY_WRITE_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = resource->buffer,
      .offset = offset,
      .size = size,
   };
   vn_async_vkCmdPipelineBarrier(&venus->vn_ring, venus->command_buffer,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                                 0, NULL, 1, &barrier, 0, NULL);

   return yttrium_venus_cmd_batch_after_record(venus, "buffer clear");
}

static struct yttrium_venus_ubo_arena *
yttrium_venus_take_free_ubo_arena(struct yttrium_venus *venus,
                                  VkDeviceSize minimum_size)
{
   struct yttrium_venus_ubo_arena **best_link = NULL;

   for (struct yttrium_venus_ubo_arena **link = &venus->free_ubo_arenas;
        *link; link = &(*link)->next) {
      if ((*link)->size < minimum_size)
         continue;
      if (!best_link || (*link)->size < (*best_link)->size)
         best_link = link;
   }
   if (!best_link)
      return NULL;

   struct yttrium_venus_ubo_arena *arena = *best_link;
   *best_link = arena->next;
   arena->next = NULL;
   arena->retired = false;
   arena->roll_base = 0;
   return arena;
}

static void
yttrium_venus_activate_ubo_arena(struct yttrium_venus *venus,
                                 struct yttrium_venus_ubo_arena *arena)
{
   arena->generation = ++venus->ubo_upload_buffer_generation;
   venus->ubo_upload_arena = arena;
   venus->ubo_upload_buffer = arena->buffer;
   venus->ubo_upload_memory = arena->memory;
   venus->ubo_upload_buffer_size = arena->size;
   venus->ubo_upload_roll_base = 0;
}

static void
yttrium_venus_discard_free_ubo_arenas(struct yttrium_venus *venus)
{
   struct yttrium_venus_ubo_arena *arena = venus->free_ubo_arenas;
   venus->free_ubo_arenas = NULL;
   while (arena) {
      struct yttrium_venus_ubo_arena *next = arena->next;
      yttrium_venus_unmap_memory(venus, &arena->mapping);
      if (arena->buffer)
         vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                                  arena->buffer, NULL);
      if (arena->memory)
         vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                               arena->memory, NULL);
      venus->ubo_arena_bytes =
         venus->ubo_arena_bytes >= arena->size ?
         venus->ubo_arena_bytes - arena->size : 0;
      FREE(arena);
      arena = next;
   }
}

static bool
yttrium_venus_ensure_ubo_upload_buffer(struct yttrium_venus *venus,
                                       VkDeviceSize size,
                                       bool force_new_generation)
{
   if (!force_new_generation &&
       venus->ubo_upload_buffer && venus->ubo_upload_memory &&
       venus->ubo_upload_buffer_size >= size)
      return true;

   /*
    * A cursor wrap needs another generation of the existing arena, not an
    * arena large enough to contain the consumed prefix plus the next upload.
    * Preserve the current generation size unless this individual layout is
    * genuinely larger.
    */
   const VkDeviceSize minimum_size =
      MAX2(size, venus->ubo_upload_buffer_size);
   const VkDeviceSize requested_size =
      yttrium_venus_pre_sized_allocation(minimum_size,
                                         yttrium_venus_ubo_arena_pre_size());
   const VkDeviceSize allocation_size =
      util_next_power_of_two64(MAX2(requested_size, (VkDeviceSize)4096));
   if (!allocation_size || allocation_size >
       YTTRIUM_VENUS_UBO_ARENA_TOTAL_LIMIT)
      return false;

   /*
    * A command batch can reference only one arena generation.  Publish that
    * batch before rotating, then retain its old backing on the submitted batch
    * until mapped completion retires the batch.  No GPU completion is needed.
    */
   if (venus->display_copy_batch_recording &&
       venus->cmd_batch_ubo_arena &&
       !yttrium_venus_flush_command_batch(venus,
                                          "ubo backing generation rotate"))
      return false;

   if (venus->ubo_upload_arena)
      yttrium_venus_retire_ubo_arena(venus,
                                     venus->ubo_upload_arena);

   struct yttrium_venus_ubo_arena *arena =
      yttrium_venus_take_free_ubo_arena(venus, allocation_size);
   if (arena) {
      yttrium_venus_activate_ubo_arena(venus, arena);
      return true;
   }

   if (venus->ubo_arena_bytes + allocation_size >
       YTTRIUM_VENUS_UBO_ARENA_TOTAL_LIMIT) {
      YTTRIUM_WARN("yttrium: WARNING: UBO backing generation budget exhausted owner=venus2 action=synchronous-bounded-memory-retirement live_bytes=%llu requested_bytes=%llu limit_bytes=%llu\n",
                   (unsigned long long)venus->ubo_arena_bytes,
                   (unsigned long long)allocation_size,
                   (unsigned long long)YTTRIUM_VENUS_UBO_ARENA_TOTAL_LIMIT);
      if (!yttrium_venus_drain_batches(
             venus, "ubo backing bounded-memory exhaustion"))
         return false;
      arena = yttrium_venus_take_free_ubo_arena(venus, allocation_size);
      if (arena) {
         yttrium_venus_activate_ubo_arena(venus, arena);
         return true;
      }
      yttrium_venus_discard_free_ubo_arenas(venus);
      if (venus->ubo_arena_bytes + allocation_size >
          YTTRIUM_VENUS_UBO_ARENA_TOTAL_LIMIT)
         return false;
   }

   arena = CALLOC_STRUCT(yttrium_venus_ubo_arena);
   if (!arena) {
      YTTRIUM_WARN("yttrium: WARNING: UBO backing generation allocation failed owner=venus2 reason=bounded-memory-exhaustion action=allocation-failed bytes=%llu\n",
                   (unsigned long long)allocation_size);
      return false;
   }

   yttrium_venus_init_object(venus, &arena->buffer_obj);
   arena->buffer = YTTRIUM_VENUS_HANDLE(VkBuffer, &arena->buffer_obj);
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = allocation_size,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateBuffer, &arena->buffer_obj, &arena->buffer,
      goto ubo_submit_failed,
      venus->device_handle, &buffer_info, NULL, &arena->buffer);

   VkMemoryRequirements reqs;
   memset(&reqs, 0, sizeof(reqs));
   vn_call_vkGetBufferMemoryRequirements(&venus->vn_ring,
                                         venus->device_handle,
                                         arena->buffer, &reqs);

   const bool direct_ubo_upload_enabled =
      yttrium_venus_direct_ubo_upload_enabled();
   uint32_t memory_type_index = UINT32_MAX;
   if (direct_ubo_upload_enabled) {
      /*
       * UBO generations are large, persistently mapped streaming uploads.
       * Prefer cached system memory for them rather than consuming a
       * discrete GPU's small host-visible device-local/BAR heap.  The latter
       * can be only 256 MiB, so a few 64 MiB generations exhaust it even when
       * the device has several GiB of non-host-visible local memory.
       */
      memory_type_index =
         yttrium_venus_choose_memory_type(
            venus, reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
   }
   const bool direct_map_candidate = memory_type_index != UINT32_MAX;
   if (memory_type_index == UINT32_MAX)
      memory_type_index =
         yttrium_venus_choose_memory_type(
            venus, reqs.memoryTypeBits, 0,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

   if (memory_type_index == UINT32_MAX) {
      YTTRIUM_LOG("yttrium: Venus no memory type for shared ubo buffer bits=0x%x size=0x%llx\n",
                  reqs.memoryTypeBits, (unsigned long long)allocation_size);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               arena->buffer, NULL);
      FREE(arena);
      return false;
   }

   yttrium_venus_init_object(venus, &arena->memory_obj);
   arena->memory =
      YTTRIUM_VENUS_HANDLE(VkDeviceMemory, &arena->memory_obj);
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = MAX2(reqs.size, allocation_size),
      .memoryTypeIndex = memory_type_index,
   };
   VkResult result =
      direct_map_candidate ?
      vn_call_vkAllocateMemory(&venus->vn_ring, venus->device_handle,
                               &memory_info, NULL,
                               &arena->memory) :
      VK_SUCCESS;
   if (!direct_map_candidate) {
      YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
         venus, vkAllocateMemory, &arena->memory_obj, &arena->memory,
         goto ubo_submit_failed, venus->device_handle, &memory_info, NULL,
         &arena->memory);
   } else if (result != VK_SUCCESS) {
      YTTRIUM_WARN("yttrium: WARNING: Venus UBO backing allocation exhausted owner=venus2 reason=vkAllocateMemory result=%d action=synchronous-bounded-memory-retirement type=%u requested_bytes=%llu live_bytes=%llu\n",
                   result, memory_type_index,
                   (unsigned long long)memory_info.allocationSize,
                   (unsigned long long)venus->ubo_arena_bytes);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               arena->buffer, NULL);
      FREE(arena);

      /*
       * Some host-visible heaps cannot hold two pre-sized generations at
       * once.  The old generation is already retained by its submitted
       * batch, so an allocation failure is genuine bounded-memory
       * exhaustion: wait only here, recycle the completed generation, and
       * let the caller retry the current layout at offset zero.
       */
      if (!yttrium_venus_drain_batches(
             venus, "ubo backing allocation exhaustion"))
         return false;
      arena = yttrium_venus_take_free_ubo_arena(venus, allocation_size);
      if (!arena) {
         YTTRIUM_WARN("yttrium: ERROR: Venus UBO backing retirement produced no reusable arena owner=venus2 requested_bytes=%llu live_bytes=%llu\n",
                      (unsigned long long)allocation_size,
                      (unsigned long long)venus->ubo_arena_bytes);
         return false;
      }
      yttrium_venus_activate_ubo_arena(venus, arena);
      return true;
   }

   result = direct_map_candidate ?
      vn_call_vkBindBufferMemory(&venus->vn_ring, venus->device_handle,
                                 arena->buffer, arena->memory, 0) :
      VK_SUCCESS;
   if (!direct_map_candidate) {
      YTTRIUM_VENUS_SUBMIT_COMMAND_OR(
         venus, vkBindBufferMemory, arena->buffer_obj.id,
         goto ubo_submit_failed, venus->device_handle, arena->buffer,
         arena->memory, 0);
   } else if (result != VK_SUCCESS) {
      YTTRIUM_LOG("yttrium: Venus shared ubo bind failed result=%d memory_id=%llu\n",
                  result,
                  (unsigned long long)arena->memory_obj.id);
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            arena->memory, NULL);
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               arena->buffer, NULL);
      FREE(arena);
      return false;
   }

   if (direct_map_candidate &&
       yttrium_venus_memory_type_has_flags(
          venus, memory_type_index,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      if (!yttrium_venus_map_memory(
             venus, arena->memory_obj.id,
             memory_info.allocationSize, &arena->mapping)) {
         YTTRIUM_WARN("yttrium: WARNING: Venus direct UBO upload unavailable owner=venus2 reason=map-memory-failed action=vkCmdUpdateBuffer-fallback memory_id=%llu size=0x%llx\n",
                      (unsigned long long)arena->memory_obj.id,
                      (unsigned long long)memory_info.allocationSize);
      }
   } else if (direct_ubo_upload_enabled) {
      YTTRIUM_WARN("yttrium: WARNING: Venus direct UBO upload unavailable owner=venus2 reason=no-host-visible-coherent-memory action=vkCmdUpdateBuffer-fallback type=%u bits=0x%x size=0x%llx\n",
                   memory_type_index, reqs.memoryTypeBits,
                   (unsigned long long)memory_info.allocationSize);
   }

   arena->size = memory_info.allocationSize;
   venus->ubo_arena_bytes += arena->size;
   venus->peak_ubo_arena_bytes =
      MAX2(venus->peak_ubo_arena_bytes, venus->ubo_arena_bytes);
   yttrium_venus_activate_ubo_arena(venus, arena);
   YTTRIUM_LOG("yttrium: Venus shared ubo buffer setup buffer_id=%llu memory_id=%llu size=0x%llx req_size=0x%llx requested=0x%llx type=%u flags=0x%x bits=0x%x\n",
               (unsigned long long)arena->buffer_obj.id,
               (unsigned long long)arena->memory_obj.id,
               (unsigned long long)memory_info.allocationSize,
               (unsigned long long)reqs.size,
               (unsigned long long)size,
               memory_type_index,
               venus->memory_props.memoryTypes[memory_type_index].propertyFlags,
               reqs.memoryTypeBits);
   return true;

ubo_submit_failed:
   if (arena->memory)
      vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                            arena->memory, NULL);
   if (arena->buffer)
      vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                               arena->buffer, NULL);
   FREE(arena);
   return false;
}

static struct yttrium_venus_ubo_slot *
yttrium_venus_pipeline_find_ubo_slot(struct yttrium_pipeline *pipeline,
                                     uint32_t binding,
                                     uint32_t array_element)
{
   if (!pipeline)
      return NULL;

   for (uint32_t i = 0; i < pipeline->ubo_count; i++) {
      if (pipeline->ubos[i].binding == binding &&
          pipeline->ubos[i].array_element == array_element)
         return &pipeline->ubos[i];
   }

   return NULL;
}

static const struct yttrium_venus_ubo_version *
yttrium_venus_find_resource_ubo_version(
   const struct yttrium_venus_ubo_version_cache *cache,
   uint64_t arena_generation,
   uint32_t contents_serial,
   uint32_t source_offset,
   uint32_t source_size)
{
   if (!cache || !arena_generation)
      return NULL;

   for (uint32_t i = 0; i < ARRAY_SIZE(cache->entries); i++) {
      const struct yttrium_venus_ubo_version *entry = &cache->entries[i];
      if (entry->valid &&
          entry->arena_generation == arena_generation &&
          entry->contents_serial == contents_serial &&
          entry->source_offset == source_offset &&
          entry->source_size == source_size)
         return entry;
   }

   return NULL;
}

static void
yttrium_venus_store_resource_ubo_version(
   struct yttrium_venus_ubo_version_cache *cache,
   uint64_t arena_generation,
   uint32_t contents_serial,
   uint32_t source_offset,
   uint32_t source_size,
   VkDeviceSize arena_offset)
{
   if (!cache || !arena_generation)
      return;

   uint32_t index = ARRAY_SIZE(cache->entries);
   for (uint32_t i = 0; i < ARRAY_SIZE(cache->entries); i++) {
      if (!cache->entries[i].valid ||
          cache->entries[i].arena_generation != arena_generation) {
         index = i;
         break;
      }
   }
   if (index == ARRAY_SIZE(cache->entries)) {
      index = cache->next % ARRAY_SIZE(cache->entries);
      cache->next++;
   }

   cache->entries[index] = (struct yttrium_venus_ubo_version) {
      .arena_generation = arena_generation,
      .arena_offset = arena_offset,
      .contents_serial = contents_serial,
      .source_offset = source_offset,
      .source_size = source_size,
      .valid = true,
   };
}

static bool
yttrium_venus_cmd_batch_alloc_descriptor_set(
   struct yttrium_venus *venus,
   struct yttrium_pipeline *pipeline,
   VkDescriptorSet *out_set)
{
   if (out_set)
      *out_set = VK_NULL_HANDLE;

   if (!venus || !pipeline || !out_set)
      return false;

   const uint32_t descriptor_count =
      pipeline->ubo_descriptor_count +
      pipeline->sampled_image_descriptor_count +
      pipeline->sampled_buffer_descriptor_count +
      pipeline->storage_image_descriptor_count +
      pipeline->storage_buffer_descriptor_count;
   if (!descriptor_count)
      return true;
   if (!pipeline->descriptor_set_layout)
      return false;

   struct yttrium_venus_cmd_batch_descriptor_pool *pool =
      &venus->cmd_batch_descriptor_pool;
   if (pool->pool_created &&
       (pool->ubo_descriptor_count != pipeline->ubo_descriptor_count ||
        pool->sampled_image_descriptor_count !=
           pipeline->sampled_image_descriptor_count ||
        pool->sampled_buffer_descriptor_count !=
           pipeline->sampled_buffer_descriptor_count ||
        pool->storage_image_descriptor_count !=
           pipeline->storage_image_descriptor_count ||
        pool->storage_buffer_descriptor_count !=
           pipeline->storage_buffer_descriptor_count)) {
      if (!yttrium_venus_flush_command_batch(
             venus, "native draw descriptor pool shape change"))
         return false;
      pool = &venus->cmd_batch_descriptor_pool;
   }

   const uint32_t limit = yttrium_venus_native_draw_batch_limit();
   if (pool->pool_created && pool->set_count >= pool->set_capacity) {
      if (!yttrium_venus_flush_command_batch(
             venus, "native draw descriptor pool full"))
         return false;
      pool = &venus->cmd_batch_descriptor_pool;
   }

   if (!pool->pool_created) {
      const bool checked_alloc =
         yttrium_venus_checked_descriptor_alloc_enabled();
      VkDescriptorPoolSize pool_sizes[5];
      uint32_t pool_size_count = 0;
      memset(pool_sizes, 0, sizeof(pool_sizes));
      if (pipeline->ubo_descriptor_count) {
         pool_sizes[pool_size_count++] = (VkDescriptorPoolSize) {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = pipeline->ubo_descriptor_count * limit,
         };
      }
      if (pipeline->sampled_image_descriptor_count) {
         pool_sizes[pool_size_count++] = (VkDescriptorPoolSize) {
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount =
               pipeline->sampled_image_descriptor_count * limit,
         };
      }
      if (pipeline->sampled_buffer_descriptor_count) {
         pool_sizes[pool_size_count++] = (VkDescriptorPoolSize) {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
            .descriptorCount =
               pipeline->sampled_buffer_descriptor_count * limit,
         };
      }
      if (pipeline->storage_image_descriptor_count) {
         pool_sizes[pool_size_count++] = (VkDescriptorPoolSize) {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount =
               pipeline->storage_image_descriptor_count * limit,
         };
      }
      if (pipeline->storage_buffer_descriptor_count) {
         pool_sizes[pool_size_count++] = (VkDescriptorPoolSize) {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
            .descriptorCount =
               pipeline->storage_buffer_descriptor_count * limit,
         };
      }

      pool->pool_obj = CALLOC_STRUCT(yttrium_venus_object);
      pool->set_objs = CALLOC(limit, sizeof(*pool->set_objs));
      pool->sets = CALLOC(limit, sizeof(*pool->sets));
      if (!pool->pool_obj || !pool->set_objs || !pool->sets) {
         yttrium_venus_cmd_batch_destroy_descriptor_pool(venus, pool);
         return false;
      }

      yttrium_venus_init_object(venus, pool->pool_obj);
      pool->pool = YTTRIUM_VENUS_HANDLE(VkDescriptorPool, pool->pool_obj);
      const VkDescriptorPoolCreateInfo pool_info = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets = limit,
         .poolSizeCount = pool_size_count,
         .pPoolSizes = pool_sizes,
      };
      if (checked_alloc) {
         VkResult result =
            vn_call_vkCreateDescriptorPool(&venus->vn_ring,
                                           venus->device_handle,
                                           &pool_info, NULL, &pool->pool);
         if (result != VK_SUCCESS) {
            YTTRIUM_WARN("yttrium: Venus checked batch descriptor pool create failed result=%d max_sets=%u ubos=%u sampled_images=%u sampled_buffers=%u storage_images=%u storage_buffers=%u\n",
                         result, limit,
                         pipeline->ubo_descriptor_count,
                         pipeline->sampled_image_descriptor_count,
                         pipeline->sampled_buffer_descriptor_count,
                         pipeline->storage_image_descriptor_count,
                         pipeline->storage_buffer_descriptor_count);
            yttrium_venus_cmd_batch_destroy_descriptor_pool(venus, pool);
            return false;
         }
      } else {
         YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
            venus, vkCreateDescriptorPool, pool->pool_obj, &pool->pool,
            goto batch_descriptor_pool_submit_failed, venus->device_handle,
            &pool_info, NULL, &pool->pool);
      }
      pool->set_capacity = limit;
      pool->ubo_descriptor_count = pipeline->ubo_descriptor_count;
      pool->sampled_image_descriptor_count =
         pipeline->sampled_image_descriptor_count;
      pool->sampled_buffer_descriptor_count =
         pipeline->sampled_buffer_descriptor_count;
      pool->storage_image_descriptor_count =
         pipeline->storage_image_descriptor_count;
      pool->storage_buffer_descriptor_count =
         pipeline->storage_buffer_descriptor_count;
      pool->pool_created = true;
   }

   if (pool->set_count >= pool->set_capacity)
      return false;

   const uint32_t index = pool->set_count++;
   yttrium_venus_init_object(venus, &pool->set_objs[index]);
   pool->sets[index] =
      YTTRIUM_VENUS_HANDLE(VkDescriptorSet, &pool->set_objs[index]);
   const VkDescriptorSetAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool->pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &pipeline->descriptor_set_layout,
   };
   if (yttrium_venus_checked_descriptor_alloc_enabled()) {
      VkResult result =
         vn_call_vkAllocateDescriptorSets(&venus->vn_ring,
                                          venus->device_handle,
                                          &alloc_info,
                                          &pool->sets[index]);
      if (result != VK_SUCCESS) {
         YTTRIUM_WARN("yttrium: Venus checked batch descriptor set alloc failed result=%d pool_id=%llu set_index=%u capacity=%u ubos=%u sampled_images=%u sampled_buffers=%u storage_images=%u storage_buffers=%u\n",
                      result,
                      (unsigned long long)pool->pool_obj->id,
                      index, pool->set_capacity,
                      pipeline->ubo_descriptor_count,
                      pipeline->sampled_image_descriptor_count,
                      pipeline->sampled_buffer_descriptor_count,
                      pipeline->storage_image_descriptor_count,
                      pipeline->storage_buffer_descriptor_count);
         pool->set_count--;
         memset(&pool->set_objs[index], 0, sizeof(pool->set_objs[index]));
         pool->sets[index] = VK_NULL_HANDLE;
         return false;
      }
   } else {
      YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
         venus, vkAllocateDescriptorSets, &pool->set_objs[index],
         &pool->sets[index], goto batch_descriptor_set_submit_failed,
         venus->device_handle, &alloc_info, &pool->sets[index]);
   }
   *out_set = pool->sets[index];
   return true;

batch_descriptor_set_submit_failed:
   pool->set_count--;
   return false;

batch_descriptor_pool_submit_failed:
   yttrium_venus_cmd_batch_destroy_descriptor_pool(venus, pool);
   return false;
}

static bool
yttrium_venus_layout_pipeline_ubo_uploads(
   struct yttrium_venus *venus,
   struct yttrium_pipeline *pipeline,
   const struct yttrium_venus_ubo_upload *uploads,
   uint32_t upload_count,
   bool batch_ubo,
   bool allow_resource_versions)
{
   if (!pipeline->ubo_count)
      return upload_count == 0;

   if (!uploads || upload_count != pipeline->ubo_count)
      return false;

   struct yttrium_venus_ubo_slot *slots[YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS];
   memset(slots, 0, sizeof(slots));

   VkDeviceSize alignment =
      MAX2(venus->uniform_buffer_offset_alignment, (VkDeviceSize)4);
   if (!util_is_power_of_two_nonzero64(alignment))
      alignment = 256;

retry_layout:
   /*
    * The watermark resets on batch flush, so on its own it would put the first
    * upload of every batch back at offset 0 - on top of bytes the batches
    * still in flight are reading.  Draining for that was the largest stall in
    * the driver: 754 drains over a twelve second capture, 1336 ms, 8.3 ms per
    * frame, every one of them labelled here.
    *
    * The arena is pre-sized well beyond one batch, so roll forward instead and
    * spend the headroom.  The watermark keeps an upload clear of others in the
    * same batch and the roll base keeps it clear of earlier batches, so taking
    * the larger of the two satisfies both.  A wrap rotates generations.
    */
   VkDeviceSize total_size =
      batch_ubo ? align64(MAX2(venus->cmd_batch_ubo_watermark,
                               venus->ubo_upload_roll_base), alignment)
                : 0;
   const VkDeviceSize layout_base = total_size;
   const bool resource_ubo_versions =
      batch_ubo && allow_resource_versions &&
      yttrium_venus_direct_ubo_upload_enabled() &&
      venus->ubo_upload_arena &&
      venus->ubo_upload_arena->mapping.map;
   bool has_arena_upload = false;
   for (uint32_t i = 0; i < upload_count; i++) {
      const struct yttrium_venus_ubo_upload *upload = &uploads[i];
      if (!upload->data || !upload->size || (upload->size & 3) ||
          upload->size > YTTRIUM_VENUS_MAX_PIPELINE_UBO_BYTES)
         return false;

      struct yttrium_venus_ubo_slot *slot =
         yttrium_venus_pipeline_find_ubo_slot(pipeline, upload->binding,
                                              upload->array_element);
      if (!slot)
         return false;

      slot->upload_reused = false;
      slot->resource_version_cacheable = false;

      if (upload->direct_resource) {
         struct yttrium_venus_resource *resource = upload->direct_resource;
         if (!resource->initialized || !resource->buffer_backed ||
             !resource->buffer || !resource->memory ||
             !(resource->buffer_usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
             (upload->direct_offset & (alignment - 1)) ||
             upload->direct_offset > resource->image_size ||
             upload->size > resource->image_size -
                                upload->direct_offset)
            return false;

         slot->buffer = resource->buffer;
         slot->memory = resource->memory;
         slot->offset = upload->direct_offset;
         slot->size = upload->size;
         /* The immutable bytes already reside in the descriptor's buffer. */
         slot->upload_reused = true;
         slots[i] = slot;
         continue;
      }

      has_arena_upload = true;

      if (resource_ubo_versions && upload->source_version_cache) {
         slot->resource_version_cacheable = true;
         const struct yttrium_venus_ubo_version *version =
            yttrium_venus_find_resource_ubo_version(
               upload->source_version_cache,
               venus->ubo_upload_buffer_generation,
               upload->source_contents_serial,
               upload->source_offset, (uint32_t)upload->size);
         if (version) {
            slot->offset = version->arena_offset;
            slot->size = upload->size;
            slot->upload_reused = true;
         } else {
            for (uint32_t j = 0; j < i; j++) {
               const struct yttrium_venus_ubo_upload *previous = &uploads[j];
               if (previous->source_version_cache !=
                      upload->source_version_cache ||
                   previous->source_contents_serial !=
                      upload->source_contents_serial ||
                   previous->source_offset != upload->source_offset ||
                   previous->size != upload->size)
                  continue;

               slot->offset = slots[j]->offset;
               slot->size = upload->size;
               slot->upload_reused = true;
               break;
            }
         }
         slots[i] = slot;
         if (slot->upload_reused)
            continue;
      }

      total_size = align64(total_size, alignment);
      const VkDeviceSize update_size = align64((VkDeviceSize)upload->size, 4);
      if (total_size + update_size < total_size)
         return false;
      slot->offset = total_size;
      slot->size = upload->size;
      total_size += update_size;
      slots[i] = slot;
   }

   if (!has_arena_upload)
      return true;

   /*
    * The rolling cursor ran off the end of this generation.  Publish the
    * current batch, retain its generation on that batch, and lay out only the
    * current uploads in a fresh generation.  Growing to total_size here would
    * include the already-consumed prefix and double the arena on every wrap.
    */
   if (batch_ubo &&
       venus->ubo_upload_buffer && venus->ubo_upload_memory &&
       venus->ubo_upload_buffer_size < total_size) {
      const VkDeviceSize layout_size = total_size - layout_base;
      if (!yttrium_venus_ensure_ubo_upload_buffer(venus, layout_size, true))
         return false;
      goto retry_layout;
   }

   const uint64_t layout_generation =
      venus->ubo_upload_buffer_generation;
   if (!yttrium_venus_ensure_ubo_upload_buffer(venus, total_size, false))
      return false;
   if (venus->ubo_upload_buffer_generation != layout_generation)
      goto retry_layout;

   for (uint32_t i = 0; i < upload_count; i++) {
      struct yttrium_venus_ubo_slot *slot = slots[i];
      if (uploads[i].direct_resource)
         continue;
      slot->buffer = venus->ubo_upload_buffer;
      slot->memory = venus->ubo_upload_memory;
   }

   if (batch_ubo) {
      if (!venus->ubo_upload_arena ||
          (venus->cmd_batch_ubo_arena &&
           venus->cmd_batch_ubo_arena != venus->ubo_upload_arena))
         return false;
      venus->cmd_batch_ubo_arena = venus->ubo_upload_arena;
      venus->cmd_batch_ubo_watermark = total_size;
      venus->ubo_upload_roll_base = total_size;
      venus->ubo_upload_arena->roll_base = total_size;
   }
   return true;
}

static bool
yttrium_venus_layout_draw_vertex_uploads(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   uint32_t resource_id,
   const struct yttrium_venus_vertex_upload *uploads,
   uint32_t upload_count,
   bool batch_vertex,
   VkDeviceSize *draw_offsets,
   VkDeviceSize *out_buffer_size,
   VkDeviceSize *out_range_offset,
   VkDeviceSize *out_range_size,
   bool *out_has_cpu_upload)
{
   bool has_cpu_upload = false;

   if (!venus || !resource || !draw_offsets || !out_buffer_size ||
       !out_range_offset || !out_range_size || !out_has_cpu_upload)
      return false;

   *out_buffer_size = 0;
   *out_range_offset = 0;
   *out_range_size = 0;
   *out_has_cpu_upload = false;

   if (!uploads && upload_count)
      return false;

   for (uint32_t i = 0; i < upload_count; i++) {
      const struct yttrium_venus_vertex_upload *upload = &uploads[i];

      draw_offsets[i] = upload->buffer_offset;
      if (upload->resource) {
         if (!upload->resource->initialized ||
             !upload->resource->buffer_backed ||
             !upload->resource->buffer ||
             !(upload->resource->buffer_usage &
               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) ||
             !upload->size)
            return false;
         continue;
      }

      if (!upload->data || !upload->size || (upload->buffer_offset & 3))
         return false;

      has_cpu_upload = true;
   }

   if (!has_cpu_upload)
      return true;

retry_layout:
   {
      /*
       * Roll forward from where the last batch left off rather than restarting
       * at 0, so this batch writes bytes no in-flight batch is reading.  The
       * watermark keeps this upload clear of others in the same batch and the
       * resource's roll base keeps it clear of its own earlier batches, so the
       * larger of the two satisfies both - and because the roll base is per
       * resource, resources sharing the global watermark cannot rewind each
       * other.  Only a wrap waits - see below.
       */
      const VkDeviceSize base =
         batch_vertex ? align64(MAX2(venus->cmd_batch_vertex_watermark,
                                     resource->draw_vertex_roll_base), 4)
                      : 0;
      VkDeviceSize range_begin = UINT64_MAX;
      VkDeviceSize range_end = 0;
      VkDeviceSize buffer_size = 0;

      for (uint32_t i = 0; i < upload_count; i++) {
         const struct yttrium_venus_vertex_upload *upload = &uploads[i];
         if (upload->resource) {
            draw_offsets[i] = upload->buffer_offset;
            continue;
         }

         const VkDeviceSize offset = base + upload->buffer_offset;
         const VkDeviceSize update_size =
            align64((VkDeviceSize)upload->size, 4);
         if (offset < base || offset + update_size < offset)
            return false;

         draw_offsets[i] = offset;
         const VkDeviceSize end = offset + update_size;
         range_begin = MIN2(range_begin, offset);
         range_end = MAX2(range_end, end);
         buffer_size = MAX2(buffer_size, end);
      }

      if (range_begin == UINT64_MAX || range_end <= range_begin)
         return false;

      /* Rotate backing and retry at offset zero when the rolling cursor wraps. */
      if (batch_vertex &&
          resource->draw_vertex_buffer && resource->draw_vertex_memory &&
          resource->draw_vertex_buffer_size < buffer_size) {
         if (!yttrium_venus_ensure_draw_vertex_buffer(
                venus, resource, resource_id, buffer_size - base, true))
            return false;
         goto retry_layout;
      }

      if (!yttrium_venus_ensure_draw_vertex_buffer(venus, resource,
                                                   resource_id,
                                                   buffer_size, false))
         return false;

      if (batch_vertex) {
         venus->cmd_batch_vertex_watermark = buffer_size;
         resource->draw_vertex_roll_base = buffer_size;
      }
      *out_buffer_size = buffer_size;
      *out_range_offset = range_begin;
      *out_range_size = range_end - range_begin;
      *out_has_cpu_upload = true;
      return true;
   }
}

static bool
yttrium_venus_layout_draw_index_upload(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   uint32_t resource_id,
   const void *index_data,
   VkDeviceSize index_data_size,
   bool batch_index,
   VkDeviceSize *out_index_offset,
   VkDeviceSize *out_index_size)
{
   if (!venus || !resource || !out_index_offset || !out_index_size)
      return false;

   *out_index_offset = 0;
   *out_index_size = 0;

   if (!index_data_size)
      return true;
   if (!index_data)
      return false;

   const VkDeviceSize upload_size = align64(index_data_size, 4);
   if (upload_size < index_data_size)
      return false;

retry_layout:
   {
      /*
       * Roll forward from where the last batch left off rather than restarting
       * at 0, so this batch writes bytes no in-flight batch is reading.  The
       * watermark keeps this upload clear of others in the same batch and the
       * resource's roll base keeps it clear of its own earlier batches, so the
       * larger of the two satisfies both.  Only a wrap waits - see below.
       */
      const VkDeviceSize offset =
         batch_index ? align64(MAX2(venus->cmd_batch_index_watermark,
                                    resource->draw_index_roll_base), 4)
                     : 0;
      const VkDeviceSize buffer_size = offset + upload_size;
      if (buffer_size < offset)
         return false;

      /* Rotate backing and retry at offset zero when the rolling cursor wraps. */
      if (batch_index &&
          resource->draw_index_buffer && resource->draw_index_memory &&
          resource->draw_index_buffer_size < buffer_size) {
         if (!yttrium_venus_ensure_draw_index_buffer(
                venus, resource, resource_id, upload_size, true))
            return false;
         goto retry_layout;
      }

      if (!yttrium_venus_ensure_draw_index_buffer(venus, resource,
                                                  resource_id, buffer_size,
                                                  false))
         return false;

      if (batch_index) {
         venus->cmd_batch_index_watermark = buffer_size;
         resource->draw_index_roll_base = buffer_size;
      }
      *out_index_offset = offset;
      *out_index_size = upload_size;
      return true;
   }
}

static bool
yttrium_venus_update_pipeline_ubo_descriptors(
   struct yttrium_venus *venus,
   struct yttrium_pipeline *pipeline,
   VkDescriptorSet descriptor_set,
   const struct yttrium_venus_ubo_upload *uploads,
   uint32_t upload_count,
   bool push_descriptors,
   VkDescriptorBufferInfo *push_infos,
   VkWriteDescriptorSet *push_writes,
   uint32_t *push_write_count)
{
   if (push_write_count)
      *push_write_count = 0;

   if (!pipeline->ubo_count)
      return upload_count == 0;

   if ((!push_descriptors && !descriptor_set) || !uploads ||
       upload_count != pipeline->ubo_count ||
       (push_descriptors &&
        (!push_infos || !push_writes || !push_write_count)))
      return false;

   VkWriteDescriptorSet local_writes[YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS];
   VkDescriptorBufferInfo local_infos[YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS];
   VkWriteDescriptorSet *writes =
      push_descriptors ? push_writes : local_writes;
   VkDescriptorBufferInfo *infos =
      push_descriptors ? push_infos : local_infos;

   for (uint32_t i = 0; i < upload_count; i++) {
      const struct yttrium_venus_ubo_upload *upload = &uploads[i];
      struct yttrium_venus_ubo_slot *slot =
         yttrium_venus_pipeline_find_ubo_slot(pipeline, upload->binding,
                                              upload->array_element);
      if (!slot || !slot->buffer || !slot->size)
         return false;

      infos[i] = (VkDescriptorBufferInfo) {
         .buffer = slot->buffer,
         .offset = slot->offset,
         .range = upload->size,
      };
      writes[i] = (VkWriteDescriptorSet) {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = push_descriptors ? VK_NULL_HANDLE : descriptor_set,
         .dstBinding = upload->binding,
         .dstArrayElement = upload->array_element,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .pBufferInfo = &infos[i],
      };
   }

   if (push_descriptors) {
      *push_write_count = upload_count;
      return true;
   }

   vn_async_vkUpdateDescriptorSets(&venus->vn_ring, venus->device_handle,
                                   upload_count, writes, 0, NULL);
   return true;
}

static bool
yttrium_venus_draw_graphics(struct yttrium_venus *venus,
                            struct yttrium_venus_resource *resource,
                            uint32_t resource_id,
                            enum yttrium_venus_graphics_mode mode,
                            const struct yttrium_venus_triangle_vertex *vertices,
                            uint32_t vertex_count,
                            const struct yttrium_venus_draw_state *draw_state)
{
   const bool vertex_buffer = mode == YTTRIUM_VENUS_GRAPHICS_VERTEX_BUFFER;
   const VkDeviceSize vertex_data_size =
      sizeof(struct yttrium_venus_triangle_vertex) * vertex_count;

   if (!vertex_buffer)
      return false;
   if (!vertices)
      return false;
   if (!vertex_count)
      return false;

   const VkViewport full_viewport = {
      .x = 0.0f,
      .y = 0.0f,
      .width = (float)resource->width,
      .height = (float)resource->height,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
   };
   const VkRect2D full_scissor = {
      .offset = { 0, 0 },
      .extent = { resource->width, resource->height },
   };
   const VkViewport *viewport =
      draw_state ? &draw_state->viewports[0] : &full_viewport;
   const VkRect2D *scissor =
      draw_state ? &draw_state->scissors[0] : &full_scissor;
   const float default_blend_constants[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
   const float *blend_constants =
      draw_state ? draw_state->blend_constants : default_blend_constants;

   if (!yttrium_venus_prepare_graphics(venus, resource, resource_id,
                                       mode, draw_state))
      return false;

   if (!yttrium_venus_ensure_draw_vertex_buffer(venus, resource, resource_id,
                                                vertex_data_size, false))
      return false;

   if (!yttrium_venus_begin_command_batch(venus, "vertex draw", false, true))
      return false;

   if (!yttrium_venus_cmd_batch_track_resource(venus, resource)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "vertex draw tracking failure");
      return false;
   }

   if (vertex_buffer) {
      yttrium_venus_cmd_update_buffer_chunks(venus, venus->command_buffer,
                                             resource->draw_vertex_buffer, 0,
                                             vertex_data_size, vertices);
      const VkBufferMemoryBarrier vertex_barrier = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .buffer = resource->draw_vertex_buffer,
         .offset = 0,
         .size = vertex_data_size,
      };
      vn_async_vkCmdPipelineBarrier(&venus->vn_ring, venus->command_buffer,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0,
                                    0, NULL, 1, &vertex_barrier, 0, NULL);
   }

   const uint32_t render_level = draw_state ? draw_state->render_level : 0;
   const uint32_t render_layer = draw_state ? draw_state->render_layer : 0;
   const uint32_t render_layers =
      draw_state ? MAX2(draw_state->render_layers, 1) : 1;
   const uint32_t render_width =
      draw_state && draw_state->render_width ?
      draw_state->render_width :
      yttrium_venus_subresource_width(resource, render_level);
   const uint32_t render_height =
      draw_state && draw_state->render_height ?
      draw_state->render_height :
      yttrium_venus_subresource_height(resource, render_level);

   const VkImageSubresourceRange range =
      yttrium_venus_render_barrier_range(resource,
                                         VK_IMAGE_ASPECT_COLOR_BIT,
                                         render_level, render_layer,
                                         render_layers);
   if (!yttrium_venus_cmd_ensure_image_initialized(venus, resource,
                                                   resource_id))
      goto fail_command_batch;
   yttrium_venus_cmd_transition_image(venus, resource,
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                      VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                      &range);

   const VkRenderPassBeginInfo render_pass_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = resource->render_pass,
      .framebuffer = resource->framebuffer,
      .renderArea = {
         .offset = { 0, 0 },
         .extent = { render_width, render_height },
      },
   };
   vn_async_vkCmdBeginRenderPass(&venus->vn_ring, venus->command_buffer,
                                 &render_pass_begin,
                                 VK_SUBPASS_CONTENTS_INLINE);

   vn_async_vkCmdSetViewport(&venus->vn_ring, venus->command_buffer, 0, 1,
                             viewport);
   vn_async_vkCmdSetScissor(&venus->vn_ring, venus->command_buffer, 0, 1,
                            scissor);
   vn_async_vkCmdBindPipeline(&venus->vn_ring, venus->command_buffer,
                              VK_PIPELINE_BIND_POINT_GRAPHICS,
                              resource->pipeline);
   vn_async_vkCmdSetBlendConstants(&venus->vn_ring, venus->command_buffer,
                                   blend_constants);
   const VkBuffer vertex_buffers[1] = { resource->draw_vertex_buffer };
   const VkDeviceSize vertex_offsets[1] = { 0 };
   vn_async_vkCmdBindVertexBuffers(&venus->vn_ring, venus->command_buffer,
                                   0, 1, vertex_buffers, vertex_offsets);
   vn_async_vkCmdDraw(&venus->vn_ring, venus->command_buffer,
                      vertex_count, 1, 0, 0);
   vn_async_vkCmdEndRenderPass(&venus->vn_ring, venus->command_buffer);

   resource->layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   yttrium_trace_venus_draw(YTTRIUM_EVENT_VENUS_DRAW,
                            resource_id, resource->image_obj.id,
                            0, 0, resource->pipeline_obj.id,
                            vertex_count,
                            viewport->x, viewport->y,
                            viewport->width, viewport->height,
                            scissor->offset.x, scissor->offset.y,
                            scissor->extent.width, scissor->extent.height);
   YTTRIUM_LOG("yttrium: Venus draw vertices res_id=%u image_id=%llu pipeline_id=%llu count=%u viewport=%f,%f %fx%f scissor=%d,%d %ux%u topology=%u cull=0x%x front=%u blend=%u color_mask=0x%x sample_mask=0x%x v0=(%f,%f,%f,%f) c0=(%f,%f,%f,%f) vlast=(%f,%f,%f,%f) clast=(%f,%f,%f,%f)\n",
               resource_id, (unsigned long long)resource->image_obj.id,
               (unsigned long long)resource->pipeline_obj.id,
               vertex_count,
               viewport->x, viewport->y, viewport->width,
               viewport->height, scissor->offset.x, scissor->offset.y,
               scissor->extent.width, scissor->extent.height,
               resource->graphics_topology,
               resource->graphics_cull_mode,
               resource->graphics_front_face,
               resource->graphics_blend_enable,
               resource->graphics_color_write_mask,
               resource->graphics_sample_mask,
               vertices[0].position[0], vertices[0].position[1],
               vertices[0].position[2], vertices[0].position[3],
               vertices[0].color[0], vertices[0].color[1],
               vertices[0].color[2], vertices[0].color[3],
               vertices[vertex_count - 1].position[0],
               vertices[vertex_count - 1].position[1],
               vertices[vertex_count - 1].position[2],
               vertices[vertex_count - 1].position[3],
               vertices[vertex_count - 1].color[0],
               vertices[vertex_count - 1].color[1],
               vertices[vertex_count - 1].color[2],
               vertices[vertex_count - 1].color[3]);
   return yttrium_venus_cmd_batch_after_record(venus, "vertex draw");

fail_command_batch:
   yttrium_venus_cancel_command_batch_setup_failure(
      venus, "vertex draw failure");
   return false;
}

bool
yttrium_venus2_draw_textured_vertices(struct yttrium_venus *venus,
                                     struct yttrium_venus_resource *resource,
                                     uint32_t resource_id,
                                     struct yttrium_venus_resource *sampled,
                                     uint32_t sampled_resource_id,
                                     const struct yttrium_venus_textured_vertex *vertices,
                                     uint32_t vertex_count,
                                     const struct yttrium_venus_draw_state *draw_state)
{
   if (!resource || !sampled || resource == sampled || !vertices ||
       !vertex_count || vertex_count > YTTRIUM_VENUS_MAX_DRAW_VERTICES)
      return false;
   VkImageView sampled_view = VK_NULL_HANDLE;
   if (!yttrium_venus_ensure_sample_image_view(venus, sampled,
                                               sampled_resource_id,
                                               sampled->vk_format,
                                               YTTRIUM_VENUS_SAMPLE_SWIZZLE_IDENTITY,
                                               VK_IMAGE_VIEW_TYPE_2D,
                                               0, 1, 0, 1,
                                               VK_IMAGE_ASPECT_COLOR_BIT,
                                               &sampled_view))
      return false;
   if (!yttrium_venus_prepare_graphics(
          venus, resource, resource_id,
          YTTRIUM_VENUS_GRAPHICS_TEXTURED_VERTEX_BUFFER, draw_state))
      return false;

   const VkDeviceSize vertex_data_size =
      sizeof(struct yttrium_venus_textured_vertex) * vertex_count;
   if (!yttrium_venus_ensure_draw_vertex_buffer(venus, resource, resource_id,
                                                vertex_data_size, false))
      return false;

   /*
    * Allocate descriptor state owned by this asynchronous command batch.
    * Updating resource->descriptor_set in place required waiting for every
    * older draw that referenced it.  The batch descriptor pool is retained
    * with the submitted slot and destroyed only after mapped completion
    * feedback retires that slot.
    */
   struct yttrium_pipeline descriptor_shape;
   memset(&descriptor_shape, 0, sizeof(descriptor_shape));
   descriptor_shape.descriptor_set_layout = resource->descriptor_set_layout;
   descriptor_shape.sampled_image_descriptor_count = 1;
   VkDescriptorSet draw_descriptor_set = VK_NULL_HANDLE;
   if (!yttrium_venus_cmd_batch_alloc_descriptor_set(
          venus, &descriptor_shape, &draw_descriptor_set))
      return false;

   const VkDescriptorImageInfo image_info = {
      .sampler = resource->sampler,
      .imageView = sampled_view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   const VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = draw_descriptor_set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &image_info,
   };
   vn_async_vkUpdateDescriptorSets(&venus->vn_ring, venus->device_handle,
                                   1, &write, 0, NULL);

   const VkViewport full_viewport = {
      .x = 0.0f,
      .y = 0.0f,
      .width = (float)resource->width,
      .height = (float)resource->height,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
   };
   const VkRect2D full_scissor = {
      .offset = { 0, 0 },
      .extent = { resource->width, resource->height },
   };
   const VkViewport *viewport =
      draw_state ? &draw_state->viewports[0] : &full_viewport;
   const VkRect2D *scissor =
      draw_state ? &draw_state->scissors[0] : &full_scissor;
   const float default_blend_constants[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
   const float *blend_constants =
      draw_state ? draw_state->blend_constants : default_blend_constants;

   if (!yttrium_venus_begin_command_batch(venus, "textured draw", false, true))
      return false;

   if (!yttrium_venus_cmd_batch_track_resource(venus, resource) ||
       !yttrium_venus_cmd_batch_track_resource(venus, sampled)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "textured draw tracking failure");
      return false;
   }

   yttrium_venus_cmd_update_buffer_chunks(venus, venus->command_buffer,
                                          resource->draw_vertex_buffer, 0,
                                          vertex_data_size, vertices);
   const VkBufferMemoryBarrier vertex_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = resource->draw_vertex_buffer,
      .offset = 0,
      .size = vertex_data_size,
   };
   vn_async_vkCmdPipelineBarrier(&venus->vn_ring, venus->command_buffer,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0,
                                 0, NULL, 1, &vertex_barrier, 0, NULL);

   const uint32_t render_level = draw_state ? draw_state->render_level : 0;
   const uint32_t render_layer = draw_state ? draw_state->render_layer : 0;
   const uint32_t render_layers =
      draw_state ? MAX2(draw_state->render_layers, 1) : 1;
   const uint32_t render_width =
      draw_state && draw_state->render_width ?
      draw_state->render_width :
      yttrium_venus_subresource_width(resource, render_level);
   const uint32_t render_height =
      draw_state && draw_state->render_height ?
      draw_state->render_height :
      yttrium_venus_subresource_height(resource, render_level);
   const VkImageSubresourceRange range =
      yttrium_venus_render_barrier_range(resource,
                                         VK_IMAGE_ASPECT_COLOR_BIT,
                                         render_level, render_layer,
                                         render_layers);
   const VkImageSubresourceRange sampled_range = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .baseMipLevel = 0,
      .levelCount = 1,
      .baseArrayLayer = 0,
      .layerCount = 1,
   };
   if (!yttrium_venus_cmd_ensure_image_initialized(venus, sampled,
                                                   sampled_resource_id))
      goto fail_command_batch;
   yttrium_venus_cmd_transition_image(venus, sampled,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      VK_ACCESS_SHADER_READ_BIT,
                                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                      &sampled_range);
   if (!yttrium_venus_cmd_ensure_image_initialized(venus, resource,
                                                   resource_id))
      goto fail_command_batch;
   yttrium_venus_cmd_transition_image(venus, resource,
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                      VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                      &range);

   const VkRenderPassBeginInfo render_pass_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = resource->render_pass,
      .framebuffer = resource->framebuffer,
      .renderArea = {
         .offset = { 0, 0 },
         .extent = { render_width, render_height },
      },
   };
   vn_async_vkCmdBeginRenderPass(&venus->vn_ring, venus->command_buffer,
                                 &render_pass_begin,
                                 VK_SUBPASS_CONTENTS_INLINE);
   vn_async_vkCmdSetViewport(&venus->vn_ring, venus->command_buffer, 0, 1,
                             viewport);
   vn_async_vkCmdSetScissor(&venus->vn_ring, venus->command_buffer, 0, 1,
                            scissor);
   vn_async_vkCmdBindPipeline(&venus->vn_ring, venus->command_buffer,
                              VK_PIPELINE_BIND_POINT_GRAPHICS,
                              resource->pipeline);
   vn_async_vkCmdBindDescriptorSets(&venus->vn_ring, venus->command_buffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    resource->pipeline_layout, 0, 1,
                                    &draw_descriptor_set, 0, NULL);
   vn_async_vkCmdSetBlendConstants(&venus->vn_ring, venus->command_buffer,
                                   blend_constants);
   const VkBuffer vertex_buffers[1] = { resource->draw_vertex_buffer };
   const VkDeviceSize vertex_offsets[1] = { 0 };
   vn_async_vkCmdBindVertexBuffers(&venus->vn_ring, venus->command_buffer,
                                   0, 1, vertex_buffers, vertex_offsets);
   vn_async_vkCmdDraw(&venus->vn_ring, venus->command_buffer,
                      vertex_count, 1, 0, 0);
   vn_async_vkCmdEndRenderPass(&venus->vn_ring, venus->command_buffer);

   resource->layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   yttrium_trace_venus_draw(YTTRIUM_EVENT_VENUS_TEXTURED_DRAW,
                            resource_id, resource->image_obj.id,
                            sampled_resource_id, sampled->image_obj.id,
                            resource->pipeline_obj.id, vertex_count,
                            viewport->x, viewport->y,
                            viewport->width, viewport->height,
                            scissor->offset.x, scissor->offset.y,
                            scissor->extent.width, scissor->extent.height);
   YTTRIUM_LOG("yttrium: Venus textured draw dst_res_id=%u dst_image_id=%llu src_res_id=%u src_image_id=%llu pipeline_id=%llu count=%u viewport=%f,%f %fx%f scissor=%d,%d %ux%u uv0=(%f,%f) uvlast=(%f,%f)\n",
               resource_id,
               (unsigned long long)resource->image_obj.id,
               sampled_resource_id,
               (unsigned long long)sampled->image_obj.id,
               (unsigned long long)resource->pipeline_obj.id,
               vertex_count,
               viewport->x, viewport->y, viewport->width,
               viewport->height, scissor->offset.x, scissor->offset.y,
               scissor->extent.width, scissor->extent.height,
               vertices[0].texcoord[0], vertices[0].texcoord[1],
               vertices[vertex_count - 1].texcoord[0],
               vertices[vertex_count - 1].texcoord[1]);
   return yttrium_venus_cmd_batch_after_record(venus, "textured draw");

fail_command_batch:
   yttrium_venus_cancel_command_batch_setup_failure(
      venus, "textured draw failure");
   return false;
}

static bool
yttrium_venus_deferred_draw_object_id_valid(uint64_t id)
{
   switch (id) {
   case 0:
   case UINT64_C(0xdddddddddddddddd):
   case UINT64_C(0xcdcdcdcdcdcdcdcd):
   case UINT64_C(0xcccccccccccccccc):
   case UINT64_C(0xfeeefeeefeeefeee):
      return false;
   default:
      return true;
   }
}

static bool
yttrium_venus_deferred_draw_validate_handles(
   const struct yttrium_venus_deferred_draw *draw,
   uint32_t draw_index)
{
   if (!draw)
      return false;

   if (!yttrium_venus_deferred_draw_object_id_valid(
          draw->render_pass_obj.id) ||
       !yttrium_venus_deferred_draw_object_id_valid(
          draw->framebuffer_obj.id) ||
       !yttrium_venus_deferred_draw_object_id_valid(
          draw->pipeline_obj.id) ||
       !yttrium_venus_deferred_draw_object_id_valid(
          draw->pipeline_layout_obj.id)) {
      YTTRIUM_WARN("yttrium: Venus deferred render-pass draw rejected invalid handles draw=%u render_pass=%llu framebuffer=%llu pipeline=%llu layout=%llu\n",
                   draw_index,
                   (unsigned long long)draw->render_pass_obj.id,
                   (unsigned long long)draw->framebuffer_obj.id,
                   (unsigned long long)draw->pipeline_obj.id,
                   (unsigned long long)draw->pipeline_layout_obj.id);
      return false;
   }

   if (!draw->use_push_descriptors && draw->descriptor_set_obj.id &&
       !yttrium_venus_deferred_draw_object_id_valid(
          draw->descriptor_set_obj.id)) {
      YTTRIUM_WARN("yttrium: Venus deferred render-pass draw rejected invalid descriptor set draw=%u descriptor_set=%llu\n",
                   draw_index,
                   (unsigned long long)draw->descriptor_set_obj.id);
      return false;
   }

   return true;
}

static void
yttrium_venus_deferred_draw_fixup_handles(
   struct yttrium_venus_deferred_draw *draw)
{
   if (!draw)
      return;

   draw->render_pass =
      YTTRIUM_VENUS_HANDLE(VkRenderPass, &draw->render_pass_obj);
   draw->framebuffer =
      YTTRIUM_VENUS_HANDLE(VkFramebuffer, &draw->framebuffer_obj);
   draw->pipeline =
      YTTRIUM_VENUS_HANDLE(VkPipeline, &draw->pipeline_obj);
   draw->pipeline_layout =
      YTTRIUM_VENUS_HANDLE(VkPipelineLayout,
                            &draw->pipeline_layout_obj);
   if (draw->descriptor_set_obj.id) {
      draw->descriptor_set =
         YTTRIUM_VENUS_HANDLE(VkDescriptorSet,
                               &draw->descriptor_set_obj);
   }

   for (uint32_t i = 0; i < draw->ubo_push_write_count; i++)
      draw->push_writes[i].pBufferInfo = &draw->ubo_infos[i];

   for (uint32_t i = 0; i < draw->sampled_push_write_count; i++) {
      const uint32_t write_index = draw->ubo_push_write_count + i;
      if (draw->sampled_sampler_objs[i].id) {
         draw->sampled_image_infos[i].sampler =
            YTTRIUM_VENUS_HANDLE(VkSampler,
                                  &draw->sampled_sampler_objs[i]);
      }
      if (draw->push_writes[write_index].pImageInfo)
         draw->push_writes[write_index].pImageInfo =
            &draw->sampled_image_infos[i];
      if (draw->push_writes[write_index].pTexelBufferView)
         draw->push_writes[write_index].pTexelBufferView =
            &draw->sampled_buffer_views[i];
   }

}

static bool
yttrium_venus_deferred_draw_ensure_capacity(struct yttrium_venus *venus)
{
   if (venus->cmd_batch_deferred_draw_count <
       venus->cmd_batch_deferred_draw_capacity)
      return true;

   uint32_t new_capacity = venus->cmd_batch_deferred_draw_capacity ?
      venus->cmd_batch_deferred_draw_capacity * 2 : 64;
   const uint32_t limit = yttrium_venus_native_draw_batch_limit();
   if (new_capacity < venus->cmd_batch_deferred_draw_count + 1)
      new_capacity = venus->cmd_batch_deferred_draw_count + 1;
   if (new_capacity > limit)
      new_capacity = limit;
   if (new_capacity <= venus->cmd_batch_deferred_draw_capacity) {
      YTTRIUM_LOG("yttrium: Venus deferred draw overflow count=%u limit=%u\n",
                  venus->cmd_batch_deferred_draw_count, limit);
      return false;
   }

   struct yttrium_venus_deferred_draw *draws =
      realloc(venus->cmd_batch_deferred_draws,
              new_capacity * sizeof(*venus->cmd_batch_deferred_draws));
   if (!draws) {
      YTTRIUM_LOG("yttrium: Venus deferred draw allocation failed count=%u capacity=%u\n",
                  venus->cmd_batch_deferred_draw_count, new_capacity);
      return false;
   }

   memset(draws + venus->cmd_batch_deferred_draw_capacity, 0,
          (new_capacity - venus->cmd_batch_deferred_draw_capacity) *
          sizeof(*draws));
   venus->cmd_batch_deferred_draws = draws;
   venus->cmd_batch_deferred_draw_capacity = new_capacity;
   for (uint32_t i = 0; i < venus->cmd_batch_deferred_draw_count; i++)
      yttrium_venus_deferred_draw_fixup_handles(
         &venus->cmd_batch_deferred_draws[i]);
   return true;
}

void
yttrium_venus_cmd_batch_clear_upload_barriers(struct yttrium_venus *venus)
{
   if (!venus)
      return;

   venus->cmd_batch_upload_barrier_count = 0;
   venus->cmd_batch_upload_src_stages = 0;
   venus->cmd_batch_upload_dst_stages = 0;
}

void
yttrium_venus_cmd_batch_clear_image_barriers(struct yttrium_venus *venus)
{
   if (!venus)
      return;

   venus->cmd_batch_image_barrier_count = 0;
   venus->cmd_batch_image_src_stages = 0;
   venus->cmd_batch_image_dst_stages = 0;
   venus->cmd_batch_image_dependency_flags = 0;
}

void
yttrium_venus_cmd_batch_clear_uploads(struct yttrium_venus *venus)
{
   if (!venus)
      return;

   for (uint32_t i = 0; i < venus->cmd_batch_upload_count; i++)
      FREE(venus->cmd_batch_uploads[i].data);
   venus->cmd_batch_upload_count = 0;
}

static bool
yttrium_venus_cmd_batch_ensure_upload_capacity(struct yttrium_venus *venus)
{
   if (venus->cmd_batch_upload_count < venus->cmd_batch_upload_capacity)
      return true;

   uint32_t new_capacity = venus->cmd_batch_upload_capacity ?
      venus->cmd_batch_upload_capacity * 2 : 64;
   if (new_capacity < venus->cmd_batch_upload_count + 1)
      new_capacity = venus->cmd_batch_upload_count + 1;

   struct yttrium_venus_cmd_batch_upload *uploads =
      realloc(venus->cmd_batch_uploads,
              new_capacity * sizeof(*venus->cmd_batch_uploads));
   if (!uploads)
      return false;

   memset(uploads + venus->cmd_batch_upload_capacity, 0,
          (new_capacity - venus->cmd_batch_upload_capacity) *
          sizeof(*uploads));
   venus->cmd_batch_uploads = uploads;
   venus->cmd_batch_upload_capacity = new_capacity;
   return true;
}

static bool
yttrium_venus_cmd_batch_try_merge_upload(
   struct yttrium_venus_cmd_batch_upload *record,
   VkBuffer buffer,
   VkDeviceSize offset,
   VkDeviceSize size,
   const uint8_t *data)
{
   if (!record || record->buffer != buffer || !record->data)
      return false;

   const VkDeviceSize record_end = record->offset + record->size;
   const VkDeviceSize upload_end = offset + size;
   if (record_end < record->offset || upload_end < offset)
      return false;

   if (offset >= record->offset && upload_end <= record_end) {
      memcpy(record->data + (offset - record->offset), data, size);
      return true;
   }

   if (offset < record->offset || offset > record_end)
      return false;

   const VkDeviceSize gap = offset - record_end;
   const VkDeviceSize new_size = upload_end - record->offset;
   if (new_size < record->size || gap > 64 * 1024)
      return false;

   if (new_size > record->capacity) {
      VkDeviceSize new_capacity = record->capacity ? record->capacity : 256;
      while (new_capacity < new_size) {
         const VkDeviceSize grown = new_capacity * 2;
         if (grown < new_capacity) {
            new_capacity = new_size;
            break;
         }
         new_capacity = grown;
      }
      if (new_capacity > SIZE_MAX)
         return false;

      uint8_t *new_data = realloc(record->data, (size_t)new_capacity);
      if (!new_data)
         return false;
      record->data = new_data;
      record->capacity = new_capacity;
   }

   if (gap)
      memset(record->data + record->size, 0, gap);
   memcpy(record->data + (offset - record->offset), data, size);
   record->size = new_size;
   return true;
}

static bool
yttrium_venus_cmd_batch_add_upload(
   struct yttrium_venus *venus,
   VkBuffer buffer,
   VkDeviceSize offset,
   VkDeviceSize size,
   const void *data,
   VkDeviceSize *out_update_size)
{
   if (out_update_size)
      *out_update_size = 0;

   if (!venus || !buffer || !data || !size || (offset & 3))
      return false;

   const VkDeviceSize update_size = align64(size, 4);
   if (update_size < size)
      return false;
   if (out_update_size)
      *out_update_size = update_size;

   uint8_t stack_data[256];
   const uint8_t *bytes = data;
   uint8_t *upload = NULL;
   if (update_size != size) {
      upload = update_size <= sizeof(stack_data) ? stack_data :
         MALLOC(update_size);
      if (!upload)
         return false;
      memcpy(upload, data, size);
      memset(upload + size, 0, update_size - size);
      bytes = upload;
   }

   for (uint32_t i = 0; i < venus->cmd_batch_upload_count; i++) {
      if (yttrium_venus_cmd_batch_try_merge_upload(
             &venus->cmd_batch_uploads[i], buffer, offset, update_size,
             bytes)) {
         if (upload && upload != stack_data)
            FREE(upload);
         return true;
      }
   }

   if (!yttrium_venus_cmd_batch_ensure_upload_capacity(venus)) {
      if (upload && upload != stack_data)
         FREE(upload);
      return false;
   }

   uint8_t *copy = MALLOC(update_size);
   if (!copy) {
      if (upload && upload != stack_data)
         FREE(upload);
      return false;
   }
   memcpy(copy, bytes, update_size);
   if (upload && upload != stack_data)
      FREE(upload);

   struct yttrium_venus_cmd_batch_upload *record =
      &venus->cmd_batch_uploads[venus->cmd_batch_upload_count++];
   record->buffer = buffer;
   record->offset = offset;
   record->size = update_size;
   record->capacity = update_size;
   record->data = copy;
   return true;
}

static void
yttrium_venus_cmd_batch_emit_uploads(struct yttrium_venus *venus)
{
   if (!venus || !venus->cmd_batch_upload_count)
      return;

   for (uint32_t i = 0; i < venus->cmd_batch_upload_count; i++) {
      const struct yttrium_venus_cmd_batch_upload *upload =
         &venus->cmd_batch_uploads[i];
      yttrium_trace_venus_upload(
         YTTRIUM_TRACE_VENUS_UPLOAD_BATCH_UPDATE_BUFFER,
         0, upload->size, 0, YTTRIUM_VENUS_HANDLE_TO_U64(upload->buffer),
         0, 0, 0, 0, 0, 0, 0);
      yttrium_venus_cmd_update_buffer_chunks(
         venus, venus->command_buffer, upload->buffer, upload->offset,
         upload->size, upload->data);
   }
   yttrium_venus_cmd_batch_clear_uploads(venus);
}

static bool
yttrium_venus_cmd_batch_add_image_barrier(
   struct yttrium_venus *venus,
   VkPipelineStageFlags src_stages,
   VkPipelineStageFlags dst_stages,
   const VkImageMemoryBarrier *barrier)
{
   if (!venus || !barrier)
      return false;

   if (venus->compact_image_barriers &&
       yttrium_venus_cmd_batch_try_fold_image_barrier(
          venus, src_stages, dst_stages, barrier))
      return true;

   if (venus->cmd_batch_image_barrier_count >=
       venus->cmd_batch_image_barrier_capacity) {
      uint32_t new_capacity = venus->cmd_batch_image_barrier_capacity ?
         venus->cmd_batch_image_barrier_capacity * 2 : 64;
      if (new_capacity < venus->cmd_batch_image_barrier_count + 1)
         new_capacity = venus->cmd_batch_image_barrier_count + 1;

      VkImageMemoryBarrier *barriers =
         realloc(venus->cmd_batch_image_barriers,
                 new_capacity * sizeof(*venus->cmd_batch_image_barriers));
      if (!barriers)
         return false;

      venus->cmd_batch_image_barriers = barriers;
      venus->cmd_batch_image_barrier_capacity = new_capacity;
   }

   venus->cmd_batch_image_barriers
      [venus->cmd_batch_image_barrier_count++] = *barrier;
   venus->cmd_batch_image_src_stages |= src_stages;
   venus->cmd_batch_image_dst_stages |= dst_stages;
   return true;
}

static bool
yttrium_venus_cmd_batch_add_upload_barrier(
   struct yttrium_venus *venus,
   VkPipelineStageFlags src_stages,
   VkPipelineStageFlags dst_stages,
   const VkBufferMemoryBarrier *barrier)
{
   if (!venus || !barrier)
      return false;

   const uint64_t buffer_key = YTTRIUM_VENUS_HANDLE_TO_U64(barrier->buffer);
   const uint32_t bucket = (uint32_t)((buffer_key >> 4) ^ (buffer_key >> 10)) &
      (ARRAY_SIZE(venus->cmd_batch_upload_barrier_candidates) - 1);
   uint32_t existing_index =
      venus->cmd_batch_upload_barrier_candidates[bucket] - 1;
   /* A clear can leave an old index behind, and growth can move the array.
    * Bounds and the full merge key make both cases safe without an epoch. */
   if (existing_index >= venus->cmd_batch_upload_barrier_count ||
       venus->cmd_batch_upload_barriers[existing_index].buffer != barrier->buffer ||
       venus->cmd_batch_upload_barriers[existing_index].srcQueueFamilyIndex !=
          barrier->srcQueueFamilyIndex ||
       venus->cmd_batch_upload_barriers[existing_index].dstQueueFamilyIndex !=
          barrier->dstQueueFamilyIndex ||
       venus->cmd_batch_upload_barriers[existing_index].pNext || barrier->pNext) {
      for (existing_index = 0;
           existing_index < venus->cmd_batch_upload_barrier_count;
           existing_index++) {
         const VkBufferMemoryBarrier *existing =
            &venus->cmd_batch_upload_barriers[existing_index];
         if (existing->buffer == barrier->buffer &&
             existing->srcQueueFamilyIndex == barrier->srcQueueFamilyIndex &&
             existing->dstQueueFamilyIndex == barrier->dstQueueFamilyIndex &&
             !existing->pNext && !barrier->pNext)
            break;
      }
   }

   if (existing_index < venus->cmd_batch_upload_barrier_count) {
      VkBufferMemoryBarrier *existing =
         &venus->cmd_batch_upload_barriers[existing_index];

      const VkDeviceSize start = MIN2(existing->offset, barrier->offset);
      existing->srcAccessMask |= barrier->srcAccessMask;
      existing->dstAccessMask |= barrier->dstAccessMask;
      if (existing->size == VK_WHOLE_SIZE ||
          barrier->size == VK_WHOLE_SIZE) {
         existing->offset = start;
         existing->size = VK_WHOLE_SIZE;
      } else {
         const VkDeviceSize existing_end =
            existing->offset + existing->size;
         const VkDeviceSize barrier_end = barrier->offset + barrier->size;
         if (existing_end < existing->offset ||
             barrier_end < barrier->offset)
            return false;
         existing->offset = start;
         existing->size = MAX2(existing_end, barrier_end) - start;
      }
      venus->cmd_batch_upload_barrier_candidates[bucket] = existing_index + 1;
      venus->cmd_batch_upload_src_stages |= src_stages;
      venus->cmd_batch_upload_dst_stages |= dst_stages;
      return true;
   }

   if (venus->cmd_batch_upload_barrier_count >=
       venus->cmd_batch_upload_barrier_capacity) {
      uint32_t new_capacity = venus->cmd_batch_upload_barrier_capacity ?
         venus->cmd_batch_upload_barrier_capacity * 2 : 64;
      if (new_capacity < venus->cmd_batch_upload_barrier_count + 1)
         new_capacity = venus->cmd_batch_upload_barrier_count + 1;

      VkBufferMemoryBarrier *barriers =
         realloc(venus->cmd_batch_upload_barriers,
                 new_capacity * sizeof(*venus->cmd_batch_upload_barriers));
      if (!barriers)
         return false;

      venus->cmd_batch_upload_barriers = barriers;
      venus->cmd_batch_upload_barrier_capacity = new_capacity;
   }

   venus->cmd_batch_upload_barriers
      [venus->cmd_batch_upload_barrier_count++] = *barrier;
   venus->cmd_batch_upload_barrier_candidates[bucket] =
      venus->cmd_batch_upload_barrier_count;
   venus->cmd_batch_upload_src_stages |= src_stages;
   venus->cmd_batch_upload_dst_stages |= dst_stages;
   return true;
}

static void
yttrium_venus_cmd_batch_emit_deferred_barriers(struct yttrium_venus *venus)
{
   if (!venus ||
       (!venus->cmd_batch_upload_barrier_count &&
        !venus->cmd_batch_image_barrier_count))
      return;

   vn_async_vkCmdPipelineBarrier(
      &venus->vn_ring, venus->command_buffer,
      venus->cmd_batch_upload_src_stages |
      venus->cmd_batch_image_src_stages,
      venus->cmd_batch_upload_dst_stages |
      venus->cmd_batch_image_dst_stages,
      venus->cmd_batch_image_dependency_flags, 0, NULL,
      venus->cmd_batch_upload_barrier_count,
      venus->cmd_batch_upload_barriers,
      venus->cmd_batch_image_barrier_count,
      venus->cmd_batch_image_barriers);
   yttrium_venus_cmd_batch_clear_upload_barriers(venus);
   yttrium_venus_cmd_batch_clear_image_barriers(venus);
}

static bool
yttrium_venus_deferred_draw_make_render_key(
   struct yttrium_pipeline *pipeline,
   struct yttrium_venus_resource **color_resources,
   uint32_t color_resource_count,
   struct yttrium_venus_resource *depth_resource,
   const struct yttrium_venus_draw_state *draw_state,
   uint32_t render_width,
   uint32_t render_height,
   struct yttrium_venus_render_pass_group_key *key)
{
   if (!pipeline || !draw_state || !key ||
       color_resource_count > PIPE_MAX_COLOR_BUFS)
      return false;

   memset(key, 0, sizeof(*key));
   key->color_attachment_count = color_resource_count;
   key->render_width = render_width;
   key->render_height = render_height;
   key->render_layers = MAX2(draw_state->render_layers, 1);
   key->use_mrss = pipeline->use_mrss;
   key->render_samples = pipeline->render_samples;
   key->color_feedback_loop_mask =
      pipeline->key.color_feedback_loop_mask;
   key->depth_feedback_loop = pipeline->key.depth_feedback_loop;
   key->depth_read_only = pipeline->key.depth_read_only;

   for (uint32_t i = 0; i < color_resource_count; i++) {
      struct yttrium_venus_resource *color =
         color_resources ? color_resources[i] : NULL;
      if (!color)
         continue;

      key->color_image_ids[i] = color->image_obj.id;
      key->color_formats[i] =
         pipeline->key.rt_format[i] != VK_FORMAT_UNDEFINED ?
         pipeline->key.rt_format[i] : color->vk_format;
      key->color_samples[i] =
         color->samples ? color->samples : VK_SAMPLE_COUNT_1_BIT;
      /* Per target, because the view is created from these - see the key. */
      key->color_levels[i] = draw_state->rt_level[i];
      key->color_layers[i] = draw_state->rt_layer[i];
   }

   if (depth_resource) {
      key->depth_image_id = depth_resource->image_obj.id;
      key->depth_format = depth_resource->vk_format;
      key->depth_level = draw_state->depth_level;
      key->depth_layer = draw_state->depth_layer;
      key->depth_layers = MAX2(draw_state->depth_layers, 1);
      key->depth_samples =
         depth_resource->samples ? depth_resource->samples :
                                   VK_SAMPLE_COUNT_1_BIT;
   }

   return true;
}

static bool
yttrium_venus_resource_in_list(struct yttrium_venus_resource *resource,
                               struct yttrium_venus_resource **resources,
                               uint32_t resource_count)
{
   if (!resource)
      return false;

   for (uint32_t i = 0; i < resource_count; i++) {
      if (resources[i] == resource)
         return true;
   }
   return false;
}

static bool
yttrium_venus_sampled_attachment_feedback_loop(
   const struct yttrium_pipeline *pipeline,
   struct yttrium_venus_resource *sampled_resource,
   struct yttrium_venus_resource **color_resources,
   uint32_t color_resource_count,
   struct yttrium_venus_resource *depth_resource,
   bool *color_feedback,
   bool *depth_feedback)
{
   bool color = false;
   bool depth = false;

   if (pipeline && sampled_resource) {
      for (uint32_t i = 0; i < color_resource_count; i++) {
         if (color_resources && color_resources[i] == sampled_resource &&
             (pipeline->key.color_feedback_loop_mask & (1u << i))) {
            color = true;
            break;
         }
      }
      depth = pipeline->key.depth_feedback_loop &&
              sampled_resource == depth_resource;
   }

   if (color_feedback)
      *color_feedback = color;
   if (depth_feedback)
      *depth_feedback = depth;
   return color || depth;
}

static bool
yttrium_venus_deferred_draw_sampled_target_overlap(
   const struct yttrium_venus_sampled_image *sampled_images,
   uint32_t sampled_image_count,
   struct yttrium_venus_resource **color_resources,
   uint32_t color_resource_count,
   struct yttrium_venus_resource *depth_resource,
   bool depth_read_only)
{
   if (!sampled_images)
      return false;

   for (uint32_t i = 0; i < sampled_image_count; i++) {
      struct yttrium_venus_resource *sampled = sampled_images[i].resource;

      if (!sampled || sampled_images[i].buffer)
         continue;
      if (yttrium_venus_resource_in_list(sampled, color_resources,
                                         color_resource_count) ||
          (sampled == depth_resource && !depth_read_only))
         return true;
   }

   return false;
}

static bool
yttrium_venus_deferred_draw_allowed(
   struct yttrium_pipeline *pipeline,
   const struct yttrium_venus_sampled_image *sampled_images,
   uint32_t sampled_image_count,
   struct yttrium_venus_resource **color_resources,
   uint32_t color_resource_count,
   struct yttrium_venus_resource *depth_resource,
   uint32_t storage_image_count,
   uint32_t so_target_count,
   bool draw_auto)
{
   if (!yttrium_venus_render_pass_batch_enabled())
      return false;
   if (!pipeline || !pipeline->render_pass || !pipeline->framebuffer)
      return false;
   /* Sampled buffers are repopulated with vkCmdUpdateBuffer for each draw.
    * Deferred render-pass emission would record every update before any of
    * the queued draws, so later contents could replace the data intended for
    * earlier draws that reference the same buffer.  Keep those draws ordered
    * with their uploads until sampled-buffer data has per-draw backing. */
   if (pipeline->has_sampled_buffer)
      return false;
   if (pipeline->has_storage_image || pipeline->has_storage_buffer ||
       storage_image_count || so_target_count || draw_auto)
      return false;
   if (yttrium_venus_deferred_draw_sampled_target_overlap(
          sampled_images, sampled_image_count, color_resources,
          color_resource_count, depth_resource, pipeline->key.depth_read_only))
      return false;

   return true;
}

static bool
yttrium_venus_compact_draw_ensure_capacity(struct yttrium_venus *venus,
                                           size_t packet_size);

static bool
yttrium_venus_cmd_batch_append_compact_draw(
   struct yttrium_venus *venus,
   struct yttrium_pipeline *pipeline,
   const struct yttrium_venus_object *pipeline_layout_obj,
   uint32_t render_width,
   uint32_t render_height,
   const struct yttrium_venus_render_pass_group_key *render_key,
   bool use_push_descriptors,
   VkDescriptorSet descriptor_set,
   const VkWriteDescriptorSet *ubo_writes,
   const VkDescriptorBufferInfo *ubo_infos,
   uint32_t ubo_update_count,
   const VkWriteDescriptorSet *sampled_writes,
   const VkDescriptorImageInfo *sampled_image_infos,
   const VkBufferView *sampled_buffer_views,
   uint32_t sampled_update_count,
   const struct yttrium_venus_draw_state *draw_state,
   const VkBuffer *vertex_buffers,
   const VkDeviceSize *vertex_offsets,
   uint32_t vertex_buffer_count,
   bool indexed,
   VkBuffer index_buffer,
   VkDeviceSize index_offset,
   VkIndexType index_type,
   uint32_t vertex_count,
   uint32_t index_count,
   uint32_t instance_count,
   int32_t vertex_offset)
{
   const uint32_t viewport_count = MAX2(draw_state->viewport_count, 1);
   const uint32_t push_write_count = use_push_descriptors ?
      ubo_update_count + sampled_update_count : 0;
   const uint16_t push_constant_vs_size = draw_state->push_constant_vs_size;
   const uint16_t push_constant_fs_size = draw_state->push_constant_fs_size;

   if (viewport_count > PIPE_MAX_VIEWPORTS ||
       vertex_buffer_count > YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS ||
       push_write_count > YTTRIUM_VENUS_DEFERRED_DRAW_PUSH_WRITE_LIMIT ||
       push_constant_vs_size > YTTRIUM_SHADER_VS_PUSH_CONSTANT_BYTES ||
       push_constant_fs_size > YTTRIUM_SHADER_FS_PUSH_CONSTANT_BYTES ||
       (push_constant_vs_size & 3) || (push_constant_fs_size & 3))
      return false;

   if (use_push_descriptors) {
      for (uint32_t i = 0; i < ubo_update_count; i++) {
         if (ubo_writes[i].pNext || ubo_writes[i].descriptorCount != 1 ||
             !ubo_writes[i].pBufferInfo)
            return false;
      }
      for (uint32_t i = 0; i < sampled_update_count; i++) {
         const bool image = sampled_writes[i].pImageInfo != NULL;
         const bool buffer_view =
            sampled_writes[i].pTexelBufferView != NULL;
         if (sampled_writes[i].pNext ||
             sampled_writes[i].descriptorCount != 1 ||
             image == buffer_view)
            return false;
      }
   }

   size_t packet_size = sizeof(struct yttrium_venus_compact_draw_packet);
   packet_size += viewport_count * sizeof(VkViewport);
   packet_size += viewport_count * sizeof(VkRect2D);
   packet_size += vertex_buffer_count * sizeof(VkBuffer);
   packet_size += vertex_buffer_count * sizeof(VkDeviceSize);
   packet_size += push_constant_vs_size + push_constant_fs_size;
   packet_size += push_write_count *
      sizeof(struct yttrium_venus_compact_descriptor_write);
   packet_size = (size_t)align64(packet_size, sizeof(uint64_t));
   if (!yttrium_venus_compact_draw_ensure_capacity(venus, packet_size))
      return false;

   uint8_t *packet_base = venus->cmd_batch_compact_draw_packets +
      venus->cmd_batch_compact_draw_packet_size;
   struct yttrium_venus_compact_draw_packet *draw =
      (struct yttrium_venus_compact_draw_packet *)packet_base;
   memset(draw, 0, sizeof(*draw));
   draw->packet_size = (uint32_t)packet_size;
   draw->viewport_count = viewport_count;
   draw->vertex_buffer_count = vertex_buffer_count;
   draw->push_write_count = push_write_count;
   draw->declared_push_descriptor_count =
      pipeline->ubo_descriptor_count +
      pipeline->sampled_image_descriptor_count +
      pipeline->sampled_buffer_descriptor_count +
      pipeline->storage_image_descriptor_count +
      pipeline->storage_buffer_descriptor_count;
   draw->render_pass_obj = pipeline->render_pass_obj;
   draw->framebuffer_obj = pipeline->framebuffer_obj;
   draw->pipeline_obj = use_push_descriptors ?
      pipeline->push_pipeline_obj : pipeline->pipeline_obj;
   draw->pipeline_layout_obj = *pipeline_layout_obj;
   if (descriptor_set && descriptor_set == pipeline->descriptor_set)
      draw->descriptor_set_obj = pipeline->descriptor_set_obj;
   else
      draw->descriptor_set = descriptor_set;
   draw->render_key = *render_key;
   draw->index_buffer = index_buffer;
   draw->index_offset = index_offset;
   draw->index_type = index_type;
   draw->render_width = render_width;
   draw->render_height = render_height;
   draw->vertex_count = vertex_count;
   draw->index_count = index_count;
   draw->instance_count = instance_count;
   draw->vertex_offset = vertex_offset;
   draw->push_constant_vs_size = push_constant_vs_size;
   draw->push_constant_fs_size = push_constant_fs_size;
   memcpy(draw->blend_constants, draw_state->blend_constants,
          sizeof(draw->blend_constants));
   draw->indexed = indexed;
   draw->use_push_descriptors = use_push_descriptors;

   uint8_t *cursor = (uint8_t *)(draw + 1);
   memcpy(cursor, draw_state->viewports,
          viewport_count * sizeof(VkViewport));
   cursor += viewport_count * sizeof(VkViewport);
   memcpy(cursor, draw_state->scissors,
          viewport_count * sizeof(VkRect2D));
   cursor += viewport_count * sizeof(VkRect2D);
   memcpy(cursor, vertex_buffers, vertex_buffer_count * sizeof(VkBuffer));
   cursor += vertex_buffer_count * sizeof(VkBuffer);
   memcpy(cursor, vertex_offsets,
          vertex_buffer_count * sizeof(VkDeviceSize));
   cursor += vertex_buffer_count * sizeof(VkDeviceSize);
   memcpy(cursor,
          draw_state->push_constant_data +
             YTTRIUM_SHADER_VS_PUSH_CONSTANT_OFFSET,
          push_constant_vs_size);
   cursor += push_constant_vs_size;
   memcpy(cursor,
          draw_state->push_constant_data +
             YTTRIUM_SHADER_FS_PUSH_CONSTANT_OFFSET,
          push_constant_fs_size);
   cursor += push_constant_fs_size;

   struct yttrium_venus_compact_descriptor_write *compact_writes =
      (struct yttrium_venus_compact_descriptor_write *)cursor;
   memset(compact_writes, 0,
          push_write_count * sizeof(*compact_writes));
   uint32_t write_index = 0;
   for (uint32_t i = 0; i < ubo_update_count && use_push_descriptors; i++) {
      struct yttrium_venus_compact_descriptor_write *write =
         &compact_writes[write_index++];
      write->dst_binding = ubo_writes[i].dstBinding;
      write->dst_array_element = ubo_writes[i].dstArrayElement;
      write->descriptor_count = ubo_writes[i].descriptorCount;
      write->descriptor_type = ubo_writes[i].descriptorType;
      write->kind = YTTRIUM_VENUS_COMPACT_DESCRIPTOR_BUFFER;
      write->info.buffer = ubo_infos[i];
   }
   for (uint32_t i = 0;
        i < sampled_update_count && use_push_descriptors; i++) {
      struct yttrium_venus_compact_descriptor_write *write =
         &compact_writes[write_index++];
      write->dst_binding = sampled_writes[i].dstBinding;
      write->dst_array_element = sampled_writes[i].dstArrayElement;
      write->descriptor_count = sampled_writes[i].descriptorCount;
      write->descriptor_type = sampled_writes[i].descriptorType;
      if (sampled_writes[i].pImageInfo) {
         write->kind = YTTRIUM_VENUS_COMPACT_DESCRIPTOR_IMAGE;
         write->info.image = sampled_image_infos[i];
         if (write->info.image.sampler) {
            for (uint32_t j = 0;
                 j < YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES; j++) {
               if (write->info.image.sampler == pipeline->samplers[j]) {
                  write->sampler_obj = pipeline->sampler_objs[j];
                  break;
               }
            }
         }
      } else {
         write->kind = YTTRIUM_VENUS_COMPACT_DESCRIPTOR_BUFFER_VIEW;
         write->info.buffer_view = sampled_buffer_views[i];
      }
   }

   venus->cmd_batch_compact_draw_packet_size += packet_size;
   venus->cmd_batch_deferred_draw_count++;
   return true;
}

static bool
yttrium_venus_cmd_batch_append_deferred_draw(
   struct yttrium_venus *venus,
   struct yttrium_pipeline *pipeline,
   const struct yttrium_venus_object *pipeline_layout_obj,
   uint32_t render_width,
   uint32_t render_height,
   const struct yttrium_venus_render_pass_group_key *render_key,
   bool use_push_descriptors,
   VkDescriptorSet descriptor_set,
   const VkWriteDescriptorSet *ubo_writes,
   const VkDescriptorBufferInfo *ubo_infos,
   uint32_t ubo_update_count,
   const VkWriteDescriptorSet *sampled_writes,
   const VkDescriptorImageInfo *sampled_image_infos,
   const VkBufferView *sampled_buffer_views,
   uint32_t sampled_update_count,
   const struct yttrium_venus_draw_state *draw_state,
   const VkBuffer *vertex_buffers,
   const VkDeviceSize *vertex_offsets,
   uint32_t vertex_buffer_count,
   bool indexed,
   VkBuffer index_buffer,
   VkDeviceSize index_offset,
   VkIndexType index_type,
   uint32_t vertex_count,
   uint32_t index_count,
   uint32_t instance_count,
   int32_t vertex_offset)
{
   if (use_push_descriptors &&
       ubo_update_count + sampled_update_count >
       YTTRIUM_VENUS_DEFERRED_DRAW_PUSH_WRITE_LIMIT)
      return false;
   if (!draw_state ||
       draw_state->push_constant_vs_size >
          YTTRIUM_SHADER_VS_PUSH_CONSTANT_BYTES ||
       draw_state->push_constant_fs_size >
          YTTRIUM_SHADER_FS_PUSH_CONSTANT_BYTES ||
       (draw_state->push_constant_vs_size & 3) ||
       (draw_state->push_constant_fs_size & 3))
      return false;
   if (!pipeline)
      return false;
   if (!render_key)
      return false;

   if (yttrium_venus_compact_draw_packets_enabled()) {
      return yttrium_venus_cmd_batch_append_compact_draw(
         venus, pipeline, pipeline_layout_obj, render_width, render_height,
         render_key,
         use_push_descriptors, descriptor_set, ubo_writes, ubo_infos,
         ubo_update_count, sampled_writes, sampled_image_infos,
         sampled_buffer_views, sampled_update_count, draw_state,
         vertex_buffers, vertex_offsets, vertex_buffer_count, indexed,
         index_buffer, index_offset, index_type, vertex_count, index_count,
         instance_count, vertex_offset);
   }

   if (!yttrium_venus_deferred_draw_ensure_capacity(venus))
      return false;

   struct yttrium_venus_deferred_draw *draw =
      &venus->cmd_batch_deferred_draws
         [venus->cmd_batch_deferred_draw_count++];
   memset(draw, 0, sizeof(*draw));

   draw->render_pass_obj = pipeline->render_pass_obj;
   draw->framebuffer_obj = pipeline->framebuffer_obj;
   draw->pipeline_obj = use_push_descriptors ?
      pipeline->push_pipeline_obj : pipeline->pipeline_obj;
   draw->pipeline_layout_obj = *pipeline_layout_obj;
   if (descriptor_set && descriptor_set == pipeline->descriptor_set)
      draw->descriptor_set_obj = pipeline->descriptor_set_obj;
   else
      draw->descriptor_set = descriptor_set;
   draw->render_key = *render_key;
   draw->render_width = render_width;
   draw->render_height = render_height;
   draw->viewport_count = MAX2(draw_state->viewport_count, 1);
   draw->vertex_buffer_count = vertex_buffer_count;
   draw->indexed = indexed;
   draw->index_buffer = index_buffer;
   draw->index_offset = index_offset;
   draw->index_type = index_type;
   draw->vertex_count = vertex_count;
   draw->index_count = index_count;
   draw->instance_count = instance_count;
   draw->vertex_offset = vertex_offset;
   draw->push_constant_vs_size = draw_state->push_constant_vs_size;
   draw->push_constant_fs_size = draw_state->push_constant_fs_size;
   draw->use_push_descriptors = use_push_descriptors;
   memcpy(draw->viewports, draw_state->viewports,
          draw->viewport_count * sizeof(draw->viewports[0]));
   memcpy(draw->scissors, draw_state->scissors,
          draw->viewport_count * sizeof(draw->scissors[0]));
   memcpy(draw->blend_constants, draw_state->blend_constants,
          sizeof(draw->blend_constants));
   memcpy(draw->push_constant_data, draw_state->push_constant_data,
          sizeof(draw->push_constant_data));
   memcpy(draw->vertex_buffers, vertex_buffers,
          vertex_buffer_count * sizeof(draw->vertex_buffers[0]));
   memcpy(draw->vertex_offsets, vertex_offsets,
          vertex_buffer_count * sizeof(draw->vertex_offsets[0]));

   if (use_push_descriptors) {
      uint32_t push_write_count = 0;

      for (uint32_t i = 0; i < ubo_update_count; i++) {
         draw->ubo_infos[i] = ubo_infos[i];
         draw->push_writes[push_write_count] = ubo_writes[i];
         draw->push_writes[push_write_count].pBufferInfo =
            &draw->ubo_infos[i];
         push_write_count++;
      }
      draw->ubo_push_write_count = ubo_update_count;
      for (uint32_t i = 0; i < sampled_update_count; i++) {
         draw->sampled_image_infos[i] = sampled_image_infos[i];
         draw->sampled_buffer_views[i] = sampled_buffer_views[i];
         if (sampled_writes[i].pImageInfo &&
             draw->sampled_image_infos[i].sampler) {
            for (uint32_t j = 0;
                 j < YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES; j++) {
               if (draw->sampled_image_infos[i].sampler ==
                   pipeline->samplers[j]) {
                  draw->sampled_sampler_objs[i] =
                     pipeline->sampler_objs[j];
                  draw->sampled_image_infos[i].sampler =
                     YTTRIUM_VENUS_HANDLE(
                        VkSampler, &draw->sampled_sampler_objs[i]);
                  break;
               }
            }
         }
         draw->push_writes[push_write_count] = sampled_writes[i];
         if (sampled_writes[i].pImageInfo)
            draw->push_writes[push_write_count].pImageInfo =
               &draw->sampled_image_infos[i];
         if (sampled_writes[i].pTexelBufferView)
            draw->push_writes[push_write_count].pTexelBufferView =
               &draw->sampled_buffer_views[i];
         push_write_count++;
      }
      draw->sampled_push_write_count = sampled_update_count;
      draw->push_write_count = push_write_count;
      draw->declared_push_descriptor_count =
         pipeline->ubo_descriptor_count +
         pipeline->sampled_image_descriptor_count +
         pipeline->sampled_buffer_descriptor_count +
         pipeline->storage_image_descriptor_count +
         pipeline->storage_buffer_descriptor_count;
   }

   yttrium_venus_deferred_draw_fixup_handles(draw);

   return true;
}

static bool
yttrium_venus_compact_draw_validate_handles(
   const struct yttrium_venus_compact_draw_packet *draw,
   uint32_t draw_index)
{
   if (!draw)
      return false;

   if (!yttrium_venus_deferred_draw_object_id_valid(
          draw->render_pass_obj.id) ||
       !yttrium_venus_deferred_draw_object_id_valid(
          draw->framebuffer_obj.id) ||
       !yttrium_venus_deferred_draw_object_id_valid(
          draw->pipeline_obj.id) ||
       !yttrium_venus_deferred_draw_object_id_valid(
          draw->pipeline_layout_obj.id)) {
      YTTRIUM_WARN("yttrium: Venus compact draw rejected invalid handles draw=%u render_pass=%llu framebuffer=%llu pipeline=%llu layout=%llu\n",
                   draw_index,
                   (unsigned long long)draw->render_pass_obj.id,
                   (unsigned long long)draw->framebuffer_obj.id,
                   (unsigned long long)draw->pipeline_obj.id,
                   (unsigned long long)draw->pipeline_layout_obj.id);
      return false;
   }

   if (!draw->use_push_descriptors && draw->descriptor_set_obj.id &&
       !yttrium_venus_deferred_draw_object_id_valid(
          draw->descriptor_set_obj.id)) {
      YTTRIUM_WARN("yttrium: Venus compact draw rejected invalid descriptor set draw=%u descriptor_set=%llu\n",
                   draw_index,
                   (unsigned long long)draw->descriptor_set_obj.id);
      return false;
   }

   return true;
}

static bool
yttrium_venus_compact_draw_get_payload(
   const struct yttrium_venus_compact_draw_packet *draw,
   const VkViewport **viewports,
   const VkRect2D **scissors,
   const VkBuffer **vertex_buffers,
   const VkDeviceSize **vertex_offsets,
   const uint8_t **push_constant_vs,
   const uint8_t **push_constant_fs,
   const struct yttrium_venus_compact_descriptor_write **writes)
{
   if (!draw || !viewports || !scissors || !vertex_buffers ||
       !vertex_offsets || !push_constant_vs || !push_constant_fs || !writes ||
       !draw->viewport_count || draw->viewport_count > PIPE_MAX_VIEWPORTS ||
       draw->vertex_buffer_count >
          YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS ||
       draw->push_write_count >
          YTTRIUM_VENUS_DEFERRED_DRAW_PUSH_WRITE_LIMIT ||
       draw->push_constant_vs_size >
          YTTRIUM_SHADER_VS_PUSH_CONSTANT_BYTES ||
       draw->push_constant_fs_size >
          YTTRIUM_SHADER_FS_PUSH_CONSTANT_BYTES ||
       (draw->push_constant_vs_size & 3) ||
       (draw->push_constant_fs_size & 3))
      return false;

   size_t expected = sizeof(*draw);
   expected += draw->viewport_count * sizeof(**viewports);
   expected += draw->viewport_count * sizeof(**scissors);
   expected += draw->vertex_buffer_count * sizeof(**vertex_buffers);
   expected += draw->vertex_buffer_count * sizeof(**vertex_offsets);
   expected += draw->push_constant_vs_size + draw->push_constant_fs_size;
   expected += draw->push_write_count * sizeof(**writes);
   expected = (size_t)align64(expected, sizeof(uint64_t));
   if (expected != draw->packet_size)
      return false;

   const uint8_t *cursor = (const uint8_t *)(draw + 1);
   *viewports = (const VkViewport *)cursor;
   cursor += draw->viewport_count * sizeof(**viewports);
   *scissors = (const VkRect2D *)cursor;
   cursor += draw->viewport_count * sizeof(**scissors);
   *vertex_buffers = (const VkBuffer *)cursor;
   cursor += draw->vertex_buffer_count * sizeof(**vertex_buffers);
   *vertex_offsets = (const VkDeviceSize *)cursor;
   cursor += draw->vertex_buffer_count * sizeof(**vertex_offsets);
   *push_constant_vs = cursor;
   cursor += draw->push_constant_vs_size;
   *push_constant_fs = cursor;
   cursor += draw->push_constant_fs_size;
   *writes = (const struct yttrium_venus_compact_descriptor_write *)cursor;
   return true;
}

static bool
yttrium_venus_compact_draw_ensure_capacity(struct yttrium_venus *venus,
                                           size_t packet_size)
{
   const uint32_t limit = yttrium_venus_native_draw_batch_limit();
   if (!venus || venus->cmd_batch_deferred_draw_count >= limit ||
       packet_size > UINT32_MAX ||
       packet_size > SIZE_MAX - venus->cmd_batch_compact_draw_packet_size)
      return false;

   const size_t required =
      venus->cmd_batch_compact_draw_packet_size + packet_size;
   if (required <= venus->cmd_batch_compact_draw_packet_capacity)
      return true;

   size_t new_capacity = venus->cmd_batch_compact_draw_packet_capacity ?
      venus->cmd_batch_compact_draw_packet_capacity * 2 : 64 * 1024;
   if (new_capacity < venus->cmd_batch_compact_draw_packet_capacity)
      return false;
   if (new_capacity < required)
      new_capacity = required;

   uint8_t *packets =
      realloc(venus->cmd_batch_compact_draw_packets, new_capacity);
   if (!packets)
      return false;

   venus->cmd_batch_compact_draw_packets = packets;
   venus->cmd_batch_compact_draw_packet_capacity = new_capacity;
   return true;
}

/* Descriptor contents, not C wrapper/payload addresses. The zero-filled scalar
 * representation keeps comparisons independent of compact/legacy packet
 * storage and of copied sampler-object wrappers.
 */
struct yttrium_venus_single_push_descriptor {
   uint64_t object_id;
   uint64_t sampler_id;
   VkDeviceSize offset;
   VkDeviceSize range;
   uint32_t binding;
   uint32_t array_element;
   VkDescriptorType type;
   VkImageLayout image_layout;
};

static uint64_t
yttrium_venus_push_descriptor_object_id(uint64_t handle)
{
   const struct yttrium_venus_object *object =
      (const struct yttrium_venus_object *)(uintptr_t)handle;
   return object ? object->id : 0;
}

static bool
yttrium_venus_normalize_single_push_descriptor(
   const VkWriteDescriptorSet *write,
   struct yttrium_venus_single_push_descriptor *out)
{
   if (!write || write->sType != VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET ||
       write->pNext || write->descriptorCount != 1)
      return false;

   *out = (struct yttrium_venus_single_push_descriptor) {
      .binding = write->dstBinding,
      .array_element = write->dstArrayElement,
      .type = write->descriptorType,
   };
   switch (write->descriptorType) {
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      if (!write->pBufferInfo)
         return false;
      out->object_id = yttrium_venus_push_descriptor_object_id(
         YTTRIUM_VENUS_HANDLE_TO_U64(write->pBufferInfo->buffer));
      out->offset = write->pBufferInfo->offset;
      out->range = write->pBufferInfo->range;
      return true;
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      if (!write->pImageInfo)
         return false;
      out->object_id = yttrium_venus_push_descriptor_object_id(
         YTTRIUM_VENUS_HANDLE_TO_U64(write->pImageInfo->imageView));
      out->sampler_id = yttrium_venus_push_descriptor_object_id(
         YTTRIUM_VENUS_HANDLE_TO_U64(write->pImageInfo->sampler));
      out->image_layout = write->pImageInfo->imageLayout;
      return true;
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      if (!write->pTexelBufferView)
         return false;
      out->object_id = yttrium_venus_push_descriptor_object_id(
         YTTRIUM_VENUS_HANDLE_TO_U64(*write->pTexelBufferView));
      return true;
   default:
      return false;
   }
}

static void
yttrium_venus_cmd_push_static_ubo_constants(
   struct yttrium_venus *venus,
   struct vn_ring *command_ring,
   VkPipelineLayout pipeline_layout,
   const uint8_t *vs_data,
   uint16_t vs_size,
   const uint8_t *fs_data,
   uint16_t fs_size)
{
   if (vs_size) {
      vn_async_vkCmdPushConstants(
         command_ring, venus->command_buffer, pipeline_layout,
         VK_SHADER_STAGE_VERTEX_BIT,
         YTTRIUM_SHADER_VS_PUSH_CONSTANT_OFFSET, vs_size, vs_data);
   }
   if (fs_size) {
      vn_async_vkCmdPushConstants(
         command_ring, venus->command_buffer, pipeline_layout,
         VK_SHADER_STAGE_FRAGMENT_BIT,
         YTTRIUM_SHADER_FS_PUSH_CONSTANT_OFFSET, fs_size, fs_data);
   }
}

static bool
yttrium_venus_cmd_batch_emit_deferred_draw_commands(
   struct yttrium_venus *venus,
   struct vn_ring *command_ring,
   const char *label)
{
   if (!venus || !venus->cmd_batch_deferred_draw_count)
      return true;

   (void)label;
   struct yttrium_trace_deferred_draw_stats trace_stats = { 0 };
   uint64_t last_push_layout_id = 0;
   struct yttrium_venus_render_pass_group_key current_key;
   bool render_pass_open = false;
   bool viewport_valid = false;
   bool scissor_valid = false;
   bool blend_constants_valid = false;
   bool pipeline_valid = false;
   bool descriptor_set_valid = false;
   bool single_push_valid = false;
   uint64_t single_push_pipeline_id = 0;
   uint64_t single_push_layout_id = 0;
   struct yttrium_venus_single_push_descriptor last_single_push;
   bool vertex_buffers_valid = false;
   bool index_buffer_valid = false;
   /* Every packet producer uses yttrium_venus2_pipeline_init, whose
    * ordinary/push/alternate layouts all have the same VS/FS push ranges
    * while this process-immutable option is enabled. No compute packet or
    * untracked command can enter this emitter's local cache lifetime. */
   const bool canonical_push_ranges =
      yttrium_gdi_static_ubo_sampled_cache_enabled();
   bool push_constant_vs_valid = false;
   bool push_constant_fs_valid = false;
   uint16_t last_push_constant_vs_size = 0;
   uint16_t last_push_constant_fs_size = 0;
   uint8_t last_push_constant_vs[YTTRIUM_SHADER_VS_PUSH_CONSTANT_BYTES];
   uint8_t last_push_constant_fs[YTTRIUM_SHADER_FS_PUSH_CONSTANT_BYTES];
   uint32_t last_viewport_count = 0;
   uint32_t last_scissor_count = 0;
   uint32_t last_vertex_buffer_count = 0;
   uint64_t last_pipeline_id = 0;
   uint64_t last_descriptor_layout_id = 0;
   uint64_t last_descriptor_set_id = 0;
   VkDescriptorSet last_descriptor_set = VK_NULL_HANDLE;
   VkBuffer last_vertex_buffers[YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS];
   VkDeviceSize
      last_vertex_offsets[YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS];
   VkBuffer last_index_buffer = VK_NULL_HANDLE;
   VkDeviceSize last_index_offset = 0;
   VkIndexType last_index_type = VK_INDEX_TYPE_UINT16;
   VkViewport last_viewports[PIPE_MAX_VIEWPORTS];
   VkRect2D last_scissors[PIPE_MAX_VIEWPORTS];
   float last_blend_constants[4];
   VkWriteDescriptorSet compact_push_writes
      [YTTRIUM_VENUS_DEFERRED_DRAW_PUSH_WRITE_LIMIT];
   VkDescriptorBufferInfo compact_buffer_infos
      [YTTRIUM_VENUS_DEFERRED_DRAW_PUSH_WRITE_LIMIT];
   VkDescriptorImageInfo compact_image_infos
      [YTTRIUM_VENUS_DEFERRED_DRAW_PUSH_WRITE_LIMIT];
   VkBufferView compact_buffer_views
      [YTTRIUM_VENUS_DEFERRED_DRAW_PUSH_WRITE_LIMIT];
   const bool compact_packets =
      yttrium_venus_compact_draw_packets_enabled();
   size_t compact_offset = 0;
   memset(&current_key, 0, sizeof(current_key));
   memset(last_vertex_buffers, 0, sizeof(last_vertex_buffers));
   memset(last_vertex_offsets, 0, sizeof(last_vertex_offsets));
   memset(last_viewports, 0, sizeof(last_viewports));
   memset(last_scissors, 0, sizeof(last_scissors));
   memset(last_blend_constants, 0, sizeof(last_blend_constants));

   yttrium_venus_cmd_batch_emit_uploads(venus);
   yttrium_venus_cmd_batch_emit_deferred_barriers(venus);

   for (uint32_t i = 0; i < venus->cmd_batch_deferred_draw_count; i++) {
      const struct yttrium_venus_render_pass_group_key *render_key;
      VkRenderPass render_pass;
      VkFramebuffer framebuffer;
      VkPipeline pipeline;
      VkPipelineLayout pipeline_layout;
      VkDescriptorSet descriptor_set;
      uint64_t pipeline_id;
      uint64_t pipeline_layout_id;
      uint64_t descriptor_set_id;
      const VkViewport *viewports;
      const VkRect2D *scissors;
      const float *blend_constants;
      const VkBuffer *vertex_buffers;
      const VkDeviceSize *vertex_offsets;
      const VkWriteDescriptorSet *push_writes;
      const uint8_t *push_constant_vs;
      const uint8_t *push_constant_fs;
      VkBuffer index_buffer;
      VkDeviceSize index_offset;
      VkIndexType index_type;
      uint32_t render_width;
      uint32_t render_height;
      uint32_t viewport_count;
      uint32_t vertex_buffer_count;
      uint32_t push_write_count;
      uint32_t declared_push_descriptor_count;
      uint32_t vertex_count;
      uint32_t index_count;
      uint32_t instance_count;
      uint16_t push_constant_vs_size;
      uint16_t push_constant_fs_size;
      int32_t vertex_offset;
      bool indexed;
      bool use_push_descriptors;

      if (compact_packets) {
         if (compact_offset > venus->cmd_batch_compact_draw_packet_size ||
             sizeof(struct yttrium_venus_compact_draw_packet) >
                venus->cmd_batch_compact_draw_packet_size - compact_offset) {
            venus->cmd_batch_deferred_draw_count = 0;
            venus->cmd_batch_compact_draw_packet_size = 0;
            return false;
         }

         const struct yttrium_venus_compact_draw_packet *draw =
            (const struct yttrium_venus_compact_draw_packet *)
               (venus->cmd_batch_compact_draw_packets + compact_offset);
         if (!draw->packet_size ||
             draw->packet_size >
                venus->cmd_batch_compact_draw_packet_size - compact_offset ||
             !yttrium_venus_compact_draw_validate_handles(draw, i)) {
            venus->cmd_batch_deferred_draw_count = 0;
            venus->cmd_batch_compact_draw_packet_size = 0;
            return false;
         }

         const struct yttrium_venus_compact_descriptor_write
            *compact_writes;
         if (!yttrium_venus_compact_draw_get_payload(
                draw, &viewports, &scissors, &vertex_buffers,
                &vertex_offsets, &push_constant_vs, &push_constant_fs,
                &compact_writes)) {
            venus->cmd_batch_deferred_draw_count = 0;
            venus->cmd_batch_compact_draw_packet_size = 0;
            return false;
         }

         for (uint32_t j = 0; j < draw->push_write_count; j++) {
            const struct yttrium_venus_compact_descriptor_write *compact =
               &compact_writes[j];
            compact_push_writes[j] = (VkWriteDescriptorSet) {
               .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
               .dstBinding = compact->dst_binding,
               .dstArrayElement = compact->dst_array_element,
               .descriptorCount = compact->descriptor_count,
               .descriptorType = compact->descriptor_type,
            };
            switch (compact->kind) {
            case YTTRIUM_VENUS_COMPACT_DESCRIPTOR_BUFFER:
               compact_buffer_infos[j] = compact->info.buffer;
               compact_push_writes[j].pBufferInfo =
                  &compact_buffer_infos[j];
               break;
            case YTTRIUM_VENUS_COMPACT_DESCRIPTOR_IMAGE:
               compact_image_infos[j] = compact->info.image;
               if (compact->sampler_obj.id) {
                  if (!yttrium_venus_deferred_draw_object_id_valid(
                         compact->sampler_obj.id)) {
                     venus->cmd_batch_deferred_draw_count = 0;
                     venus->cmd_batch_compact_draw_packet_size = 0;
                     return false;
                  }
                  compact_image_infos[j].sampler =
                     YTTRIUM_VENUS_HANDLE(
                        VkSampler,
                        &compact_writes[j].sampler_obj);
               }
               compact_push_writes[j].pImageInfo =
                  &compact_image_infos[j];
               break;
            case YTTRIUM_VENUS_COMPACT_DESCRIPTOR_BUFFER_VIEW:
               compact_buffer_views[j] = compact->info.buffer_view;
               compact_push_writes[j].pTexelBufferView =
                  &compact_buffer_views[j];
               break;
            default:
               venus->cmd_batch_deferred_draw_count = 0;
               venus->cmd_batch_compact_draw_packet_size = 0;
               return false;
            }
         }

         render_key = &draw->render_key;
         render_pass = YTTRIUM_VENUS_HANDLE(
            VkRenderPass, &draw->render_pass_obj);
         framebuffer = YTTRIUM_VENUS_HANDLE(
            VkFramebuffer, &draw->framebuffer_obj);
         pipeline = YTTRIUM_VENUS_HANDLE(
            VkPipeline, &draw->pipeline_obj);
         pipeline_layout = YTTRIUM_VENUS_HANDLE(
            VkPipelineLayout, &draw->pipeline_layout_obj);
         descriptor_set = draw->descriptor_set_obj.id ?
            YTTRIUM_VENUS_HANDLE(
               VkDescriptorSet, &draw->descriptor_set_obj) :
            draw->descriptor_set;
         pipeline_id = draw->pipeline_obj.id;
         pipeline_layout_id = draw->pipeline_layout_obj.id;
         descriptor_set_id = draw->descriptor_set_obj.id;
         blend_constants = draw->blend_constants;
         push_writes = compact_push_writes;
         index_buffer = draw->index_buffer;
         index_offset = draw->index_offset;
         index_type = draw->index_type;
         render_width = draw->render_width;
         render_height = draw->render_height;
         viewport_count = draw->viewport_count;
         vertex_buffer_count = draw->vertex_buffer_count;
         push_write_count = draw->push_write_count;
         declared_push_descriptor_count = draw->declared_push_descriptor_count;
         vertex_count = draw->vertex_count;
         index_count = draw->index_count;
         instance_count = draw->instance_count;
         push_constant_vs_size = draw->push_constant_vs_size;
         push_constant_fs_size = draw->push_constant_fs_size;
         vertex_offset = draw->vertex_offset;
         indexed = draw->indexed;
         use_push_descriptors = draw->use_push_descriptors;
         compact_offset += draw->packet_size;
      } else {
         const struct yttrium_venus_deferred_draw *draw =
            &venus->cmd_batch_deferred_draws[i];
         if (!yttrium_venus_deferred_draw_validate_handles(draw, i)) {
            venus->cmd_batch_deferred_draw_count = 0;
            return false;
         }

         render_key = &draw->render_key;
         render_pass = draw->render_pass;
         framebuffer = draw->framebuffer;
         pipeline = draw->pipeline;
         pipeline_layout = draw->pipeline_layout;
         descriptor_set = draw->descriptor_set;
         pipeline_id = draw->pipeline_obj.id;
         pipeline_layout_id = draw->pipeline_layout_obj.id;
         descriptor_set_id = draw->descriptor_set_obj.id;
         viewports = draw->viewports;
         scissors = draw->scissors;
         blend_constants = draw->blend_constants;
         vertex_buffers = draw->vertex_buffers;
         vertex_offsets = draw->vertex_offsets;
         push_writes = draw->push_writes;
         push_constant_vs = draw->push_constant_data +
            YTTRIUM_SHADER_VS_PUSH_CONSTANT_OFFSET;
         push_constant_fs = draw->push_constant_data +
            YTTRIUM_SHADER_FS_PUSH_CONSTANT_OFFSET;
         index_buffer = draw->index_buffer;
         index_offset = draw->index_offset;
         index_type = draw->index_type;
         render_width = draw->render_width;
         render_height = draw->render_height;
         viewport_count = draw->viewport_count;
         vertex_buffer_count = draw->vertex_buffer_count;
         push_write_count = draw->push_write_count;
         declared_push_descriptor_count = draw->declared_push_descriptor_count;
         vertex_count = draw->vertex_count;
         index_count = draw->index_count;
         instance_count = draw->instance_count;
         push_constant_vs_size = draw->push_constant_vs_size;
         push_constant_fs_size = draw->push_constant_fs_size;
         vertex_offset = draw->vertex_offset;
         indexed = draw->indexed;
         use_push_descriptors = draw->use_push_descriptors;
      }

      const bool compatible =
         render_pass_open &&
         memcmp(&current_key, render_key, sizeof(current_key)) == 0;

      if (!compatible) {
         if (render_pass_open)
            vn_async_vkCmdEndRenderPass(command_ring,
                                        venus->command_buffer);

         const VkRenderPassBeginInfo render_pass_begin = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = render_pass,
            .framebuffer = framebuffer,
            .renderArea = {
               .offset = { 0, 0 },
               .extent = { render_width, render_height },
            },
         };
         vn_async_vkCmdBeginRenderPass(command_ring,
                                       venus->command_buffer,
                                       &render_pass_begin,
                                       VK_SUBPASS_CONTENTS_INLINE);
         trace_stats.render_passes++;
         current_key = *render_key;
         render_pass_open = true;
         viewport_valid = false;
         scissor_valid = false;
         blend_constants_valid = false;
         pipeline_valid = false;
         descriptor_set_valid = false;
         single_push_valid = false;
         vertex_buffers_valid = false;
         index_buffer_valid = false;
         push_constant_vs_valid = false;
         push_constant_fs_valid = false;
      }

      if (!viewport_valid || last_viewport_count != viewport_count ||
          memcmp(last_viewports, viewports,
                 viewport_count * sizeof(viewports[0])) != 0) {
         vn_async_vkCmdSetViewport(command_ring, venus->command_buffer, 0,
                                   viewport_count, viewports);
         memcpy(last_viewports, viewports,
                viewport_count * sizeof(viewports[0]));
         last_viewport_count = viewport_count;
         viewport_valid = true;
      }

      if (!scissor_valid || last_scissor_count != viewport_count ||
          memcmp(last_scissors, scissors,
                 viewport_count * sizeof(scissors[0])) != 0) {
         vn_async_vkCmdSetScissor(command_ring, venus->command_buffer, 0,
                                  viewport_count, scissors);
         memcpy(last_scissors, scissors,
                viewport_count * sizeof(scissors[0]));
         last_scissor_count = viewport_count;
         scissor_valid = true;
      }

      /* Packet-local wrappers have different addresses even when they name
       * the same Venus object.  Compare the copied IDs, not those handles. */
      if (!pipeline_valid || last_pipeline_id != pipeline_id) {
         vn_async_vkCmdBindPipeline(command_ring, venus->command_buffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline);
         last_pipeline_id = pipeline_id;
         pipeline_valid = true;
         trace_stats.pipeline_binds++;
      } else {
         trace_stats.pipeline_binds_skipped++;
      }

      if (use_push_descriptors && push_write_count) {
         trace_stats.requested_push_calls++;
         trace_stats.requested_push_writes += push_write_count;
         /* Restrict this first cache to a complete one-descriptor layout.
          * Such layouts have no alternate layout/rotation. Multi-descriptor
          * pushes always execute unchanged, so skipping cannot consume their
          * parity and accidentally restore ANV's preservation-copy path.
          * Producer uploads, barriers, role checks and held references have
          * already run for every draw; only the redundant command is omitted.
          */
         struct yttrium_venus_single_push_descriptor current_single_push;
         bool same_push = false;
         if (declared_push_descriptor_count == 1 && push_write_count == 1 &&
             yttrium_venus_normalize_single_push_descriptor(
                push_writes, &current_single_push)) {
            if (single_push_valid && single_push_pipeline_id == pipeline_id &&
                single_push_layout_id == pipeline_layout_id) {
               trace_stats.push_comparisons++;
               same_push = memcmp(&last_single_push, &current_single_push,
                                  sizeof(current_single_push)) == 0;
            }
            last_single_push = current_single_push;
            single_push_pipeline_id = pipeline_id;
            single_push_layout_id = pipeline_layout_id;
            single_push_valid = true;
         } else {
            single_push_valid = false;
         }
         if (!same_push) {
            vn_async_vkCmdPushDescriptorSet(
               command_ring, venus->command_buffer,
               VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0,
               push_write_count, push_writes);
            trace_stats.push_calls++;
            trace_stats.push_writes += push_write_count;
            if (last_push_layout_id == pipeline_layout_id)
               trace_stats.repeated_push_layouts++;
            last_push_layout_id = pipeline_layout_id;
         } else {
            trace_stats.push_calls_skipped++;
            trace_stats.push_writes_skipped += push_write_count;
         }
         descriptor_set_valid = false;
      } else if (descriptor_set) {
         single_push_valid = false;
         if (!descriptor_set_valid ||
             last_descriptor_layout_id != pipeline_layout_id ||
             last_descriptor_set_id != descriptor_set_id ||
             (!descriptor_set_id && last_descriptor_set != descriptor_set)) {
            vn_async_vkCmdBindDescriptorSets(command_ring,
                                             venus->command_buffer,
                                             VK_PIPELINE_BIND_POINT_GRAPHICS,
                                             pipeline_layout, 0, 1,
                                             &descriptor_set, 0, NULL);
            last_descriptor_layout_id = pipeline_layout_id;
            last_descriptor_set_id = descriptor_set_id;
            /* Batch-owned sets keep their original stable handles instead
             * of embedding a copied descriptor_set_obj in the packet. */
            last_descriptor_set = descriptor_set;
            descriptor_set_valid = true;
            trace_stats.descriptor_binds++;
            last_push_layout_id = 0;
         } else {
            trace_stats.descriptor_binds_skipped++;
         }
      } else {
         descriptor_set_valid = false;
         single_push_valid = false;
      }

      /* Push-constant compatibility depends on the complete range
       * definitions, not descriptor-layout identity. Preserve the shadow
       * across the existing compatible descriptor-layout rotation. */
      const bool update_push_constant_vs = push_constant_vs_size &&
         (!canonical_push_ranges || !push_constant_vs_valid ||
          last_push_constant_vs_size != push_constant_vs_size ||
          memcmp(last_push_constant_vs, push_constant_vs,
                 push_constant_vs_size) != 0);
      const bool update_push_constant_fs = push_constant_fs_size &&
         (!canonical_push_ranges || !push_constant_fs_valid ||
          last_push_constant_fs_size != push_constant_fs_size ||
          memcmp(last_push_constant_fs, push_constant_fs,
                 push_constant_fs_size) != 0);
      yttrium_venus_cmd_push_static_ubo_constants(
         venus, command_ring, pipeline_layout,
         push_constant_vs, update_push_constant_vs ? push_constant_vs_size : 0,
         push_constant_fs, update_push_constant_fs ? push_constant_fs_size : 0);
      if (update_push_constant_vs) {
         memcpy(last_push_constant_vs, push_constant_vs,
                push_constant_vs_size);
         last_push_constant_vs_size = push_constant_vs_size;
      }
      if (update_push_constant_fs) {
         memcpy(last_push_constant_fs, push_constant_fs,
                push_constant_fs_size);
         last_push_constant_fs_size = push_constant_fs_size;
      }
      push_constant_vs_valid =
         canonical_push_ranges && push_constant_vs_size != 0;
      push_constant_fs_valid =
         canonical_push_ranges && push_constant_fs_size != 0;

      if (!blend_constants_valid ||
          memcmp(last_blend_constants, blend_constants,
                 sizeof(last_blend_constants)) != 0) {
         vn_async_vkCmdSetBlendConstants(command_ring,
                                         venus->command_buffer,
                                         blend_constants);
         memcpy(last_blend_constants, blend_constants,
                sizeof(last_blend_constants));
         blend_constants_valid = true;
      }

      if (vertex_buffer_count) {
         if (!vertex_buffers_valid ||
             last_vertex_buffer_count != vertex_buffer_count ||
             memcmp(last_vertex_buffers, vertex_buffers,
                    vertex_buffer_count * sizeof(vertex_buffers[0])) != 0 ||
             memcmp(last_vertex_offsets, vertex_offsets,
                    vertex_buffer_count * sizeof(vertex_offsets[0])) != 0) {
            vn_async_vkCmdBindVertexBuffers(
               command_ring, venus->command_buffer, 0,
               vertex_buffer_count, vertex_buffers, vertex_offsets);
            memcpy(last_vertex_buffers, vertex_buffers,
                   vertex_buffer_count * sizeof(vertex_buffers[0]));
            memcpy(last_vertex_offsets, vertex_offsets,
                   vertex_buffer_count * sizeof(vertex_offsets[0]));
            last_vertex_buffer_count = vertex_buffer_count;
            vertex_buffers_valid = true;
         }
      } else {
         vertex_buffers_valid = false;
      }

      if (indexed) {
         if (!index_buffer_valid ||
             last_index_buffer != index_buffer ||
             last_index_offset != index_offset ||
             last_index_type != index_type) {
            vn_async_vkCmdBindIndexBuffer(command_ring,
                                          venus->command_buffer,
                                          index_buffer, index_offset,
                                          index_type);
            last_index_buffer = index_buffer;
            last_index_offset = index_offset;
            last_index_type = index_type;
            index_buffer_valid = true;
         }
         vn_async_vkCmdDrawIndexed(command_ring, venus->command_buffer,
                                   index_count, instance_count, 0,
                                   vertex_offset, 0);
      } else {
         vn_async_vkCmdDraw(command_ring, venus->command_buffer,
                            vertex_count, instance_count, 0, 0);
      }
      trace_stats.draws++;
   }

   if (compact_packets &&
       compact_offset != venus->cmd_batch_compact_draw_packet_size) {
      venus->cmd_batch_deferred_draw_count = 0;
      venus->cmd_batch_compact_draw_packet_size = 0;
      return false;
   }

   if (render_pass_open)
      vn_async_vkCmdEndRenderPass(command_ring, venus->command_buffer);

   yttrium_trace_deferred_draw_state(&trace_stats);
   venus->cmd_batch_deferred_draw_count = 0;
   venus->cmd_batch_compact_draw_packet_size = 0;
   return true;
}

bool
yttrium_venus_cmd_batch_emit_deferred_draws(struct yttrium_venus *venus,
                                            const char *label)
{
   if (!venus || !venus->cmd_batch_deferred_draw_count)
      return true;

   if (!yttrium_venus2_ring_transaction_begin(venus))
      return false;

   struct yttrium_venus_command_stream stream;
   yttrium_venus2_command_stream_init(&stream, venus);
   const bool emit_ok = yttrium_venus_cmd_batch_emit_deferred_draw_commands(
      venus, &stream.vn_ring, label);
   const bool flush_ok = yttrium_venus2_command_stream_flush(&stream);
   const bool end_ok =
      yttrium_venus2_ring_transaction_end(venus, label);
   return emit_ok && flush_ok && end_ok;
}

static uint32_t
yttrium_venus_device_local_draw_source_serial(
   const struct yttrium_venus_resource *resource)
{
   return resource ? resource->draw_source_contents_serial : 0;
}

static bool
yttrium_venus_device_local_draw_mirror_has_contents(
   const struct yttrium_venus_resource *resource)
{
   return resource &&
      (resource->device_local_draw_pending_valid ||
       resource->device_local_draw_contents_valid);
}

static uint32_t
yttrium_venus_device_local_draw_mirror_serial(
   const struct yttrium_venus_resource *resource)
{
   if (!resource)
      return 0;
   return resource->device_local_draw_pending_valid ?
      resource->device_local_draw_pending_serial :
      resource->device_local_draw_contents_serial;
}

static bool
yttrium_venus_device_local_draw_mirror_needs_copy(
   const struct yttrium_venus_resource *resource)
{
   if (!resource || !resource->device_local_draw_buffer ||
       !resource->device_local_draw_memory)
      return false;

   return !yttrium_venus_device_local_draw_mirror_has_contents(resource) ||
      yttrium_venus_device_local_draw_mirror_serial(resource) !=
         yttrium_venus_device_local_draw_source_serial(resource);
}

static VkBuffer
yttrium_venus_prepare_device_local_draw_mirror(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   bool *out_mirrored)
{
   if (out_mirrored)
      *out_mirrored = false;
   if (!venus || !resource || !resource->buffer ||
       !resource->device_local_draw_buffer ||
       !resource->device_local_draw_memory)
      return resource ? resource->buffer : VK_NULL_HANDLE;

   if (yttrium_venus_device_local_draw_mirror_needs_copy(resource)) {
      const VkDeviceSize size = resource->device_local_draw_buffer_size;
      const uint32_t source_serial =
         yttrium_venus_device_local_draw_source_serial(resource);
      const bool mirror_had_contents =
         yttrium_venus_device_local_draw_mirror_has_contents(resource);
      if (!yttrium_venus_cmd_batch_record_draw_mirror_update(
             venus, resource, source_serial))
         return VK_NULL_HANDLE;

      VkBufferMemoryBarrier barriers[2];
      uint32_t barrier_count = 0;

      barriers[barrier_count++] = (VkBufferMemoryBarrier) {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT |
                          VK_ACCESS_MEMORY_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .buffer = resource->buffer,
         .offset = 0,
         .size = size,
      };
      if (mirror_had_contents) {
         barriers[barrier_count++] = (VkBufferMemoryBarrier) {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = resource->device_local_draw_buffer,
            .offset = 0,
            .size = size,
         };
      }
      vn_async_vkCmdPipelineBarrier(
         &venus->vn_ring, venus->command_buffer,
         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL,
         barrier_count, barriers, 0, NULL);

      const VkBufferCopy copy = {
         .srcOffset = 0,
         .dstOffset = 0,
         .size = size,
      };
      vn_async_vkCmdCopyBuffer(&venus->vn_ring, venus->command_buffer,
                               resource->buffer,
                               resource->device_local_draw_buffer,
                               1, &copy);

      VkAccessFlags draw_access = 0;
      if (resource->buffer_usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
         draw_access |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
      if (resource->buffer_usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT)
         draw_access |= VK_ACCESS_INDEX_READ_BIT;
      const VkBufferMemoryBarrier draw_barrier = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = draw_access,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .buffer = resource->device_local_draw_buffer,
         .offset = 0,
         .size = size,
      };
      vn_async_vkCmdPipelineBarrier(
         &venus->vn_ring, venus->command_buffer,
         VK_PIPELINE_STAGE_TRANSFER_BIT,
         VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, NULL,
         1, &draw_barrier, 0, NULL);

   }

   if (out_mirrored)
      *out_mirrored = true;
   return resource->device_local_draw_buffer;
}

static uint64_t
yttrium_venus_sampled_descriptor_object_id(
   const struct yttrium_venus_sampled_image *sampled)
{
   if (!sampled || !sampled->resource)
      return 0;
   return sampled->buffer ? sampled->resource->buffer_obj.id :
                            sampled->resource->image_obj.id;
}

static bool
yttrium_venus_sampled_descriptor_equal(
   const struct yttrium_venus_sampled_image *a,
   const struct yttrium_venus_sampled_image *b)
{
   return a && b && a->resource == b->resource &&
      a->resource_id == b->resource_id && a->binding == b->binding &&
      a->buffer_offset == b->buffer_offset &&
      a->buffer_range == b->buffer_range &&
      a->swizzle_key == b->swizzle_key && a->view_type == b->view_type &&
      a->aspect_mask == b->aspect_mask &&
      a->first_level == b->first_level &&
      a->level_count == b->level_count &&
      a->first_layer == b->first_layer &&
      a->layer_count == b->layer_count && a->format == b->format &&
      a->buffer == b->buffer;
}

static bool
yttrium_venus_pipeline_sampled_cache_matches(
   const struct yttrium_pipeline *pipeline,
   const struct yttrium_venus_sampled_image *sampled_images,
   uint32_t sampled_image_count)
{
   if (!pipeline || !pipeline->sampled_descriptor_cache_initialized ||
       !sampled_images ||
       pipeline->sampled_descriptor_cache_count != sampled_image_count)
      return false;

   for (uint32_t i = 0; i < sampled_image_count; i++) {
      if (!yttrium_venus_sampled_descriptor_equal(
             &pipeline->sampled_descriptor_cache[i], &sampled_images[i]) ||
          pipeline->sampled_descriptor_cache_object_ids[i] !=
             yttrium_venus_sampled_descriptor_object_id(&sampled_images[i]))
         return false;
   }
   return true;
}

static void
yttrium_venus_pipeline_sampled_cache_initialize(
   struct yttrium_pipeline *pipeline,
   const struct yttrium_venus_sampled_image *sampled_images,
   uint32_t sampled_image_count)
{
   if (!pipeline || !sampled_images ||
       sampled_image_count > YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES)
      return;

   pipeline->sampled_descriptor_cache_count = sampled_image_count;
   for (uint32_t i = 0; i < sampled_image_count; i++) {
      pipeline->sampled_descriptor_cache[i] = sampled_images[i];
      pipeline->sampled_descriptor_cache[i].buffer_data = NULL;
      pipeline->sampled_descriptor_cache[i].buffer_size = 0;
      pipeline->sampled_descriptor_cache_object_ids[i] =
         yttrium_venus_sampled_descriptor_object_id(&sampled_images[i]);
      pipe_resource_reference(
         &pipeline->sampled_descriptor_cache_resources[i],
         sampled_images[i].pipe_resource);
   }
   pipeline->sampled_descriptor_cache_initialized = true;
}

static const struct yttrium_venus_object *
yttrium_venus_select_pipeline_layout(
   uint32_t *layout_index,
   struct yttrium_pipeline *pipeline,
   bool use_push_descriptors,
   uint32_t push_descriptor_count)
{
   if (!use_push_descriptors)
      return &pipeline->pipeline_layout_obj;

   const uint32_t expected_descriptor_count =
      pipeline->ubo_descriptor_count +
      pipeline->sampled_image_descriptor_count +
      pipeline->sampled_buffer_descriptor_count +
      pipeline->storage_image_descriptor_count +
      pipeline->storage_buffer_descriptor_count;

   /*
    * ANV preserves push descriptors when a layout object is reused, which can
    * copy the old host backing on every draw or dispatch. Alternating compatible
    * layouts makes that preservation unnecessary. This is valid only while the
    * command publishes every descriptor in the selected layout. Graphics and
    * compute must use separate indices: interleaving bind points must not consume
    * the other bind point's parity and repeat its previous layout.
    */
   if (pipeline->push_pipeline_layout_alt &&
       push_descriptor_count == expected_descriptor_count &&
       yttrium_venus_push_descriptor_layout_rotation_enabled()) {
      const uint32_t index = (*layout_index)++;
      if (index & 1)
         return &pipeline->push_pipeline_layout_alt_obj;
   }

   return &pipeline->push_pipeline_layout_obj;
}

bool
yttrium_venus2_draw_pipeline(struct yttrium_venus *venus,
                            struct yttrium_venus_resource *resource,
                             uint32_t resource_id,
                             struct yttrium_venus_resource **color_resources,
                             const uint32_t *color_resource_ids,
                             uint32_t color_resource_count,
                             struct yttrium_venus_resource *depth_resource,
                             uint32_t depth_resource_id,
                            struct yttrium_pipeline *pipeline,
                            const struct yttrium_venus_sampled_image *sampled_images,
                            uint32_t sampled_image_count,
                            const struct yttrium_venus_storage_image *storage_images,
                            uint32_t storage_image_count,
                            const struct yttrium_venus_vertex_upload *vertex_uploads,
                            uint32_t vertex_upload_count,
                            uint32_t vertex_count,
                            uint32_t instance_count,
                            const void *index_data,
                            size_t index_data_size,
                            struct yttrium_venus_resource *index_resource,
                            uint32_t index_resource_id,
                            VkDeviceSize index_buffer_offset,
                            uint32_t index_count,
                            VkIndexType index_type,
                            bool index_host_write_pending,
                            int32_t vertex_offset,
                            const struct yttrium_venus_ubo_upload *ubo_uploads,
                            uint32_t ubo_upload_count,
                            struct yttrium_venus_stream_output_target *so_targets,
                            uint32_t so_target_count,
                            const struct yttrium_venus_stream_output_target *draw_auto_target,
                            uint32_t draw_auto_stride,
                            const struct yttrium_venus_draw_state *draw_state)
{
   const bool draw_auto = draw_auto_target != NULL;
   struct yttrium_venus_resource *indirect_resource =
      draw_state ? draw_state->indirect_resource : NULL;
   const bool draw_indirect = indirect_resource != NULL;
   const VkDeviceSize indirect_offset = draw_indirect ?
      draw_state->indirect_offset : 0;
   const VkDeviceSize indirect_size = index_resource ?
      sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand);
   const uint64_t total_start_us =
      yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;
   uint64_t stage_start_us = total_start_us;
   bool retried_transient_footprint_conflict = false;

   if (!venus || !resource || !pipeline || !pipeline->pipeline ||
       (!vertex_uploads && vertex_upload_count) ||
       (!draw_auto && !draw_indirect && !vertex_count) ||
       (!draw_indirect && !instance_count) || !draw_state ||
       vertex_upload_count > YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS)
      return false;

   if (draw_indirect) {
      if (draw_auto || !venus->draw_indirect_first_instance ||
          !indirect_resource->initialized || !indirect_resource->buffer_backed ||
          !indirect_resource->buffer ||
          !(indirect_resource->buffer_usage & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) ||
          (indirect_offset & 3) ||
          indirect_offset > indirect_resource->allocation_size ||
          indirect_size > indirect_resource->allocation_size - indirect_offset) {
         YTTRIUM_WARN("yttrium: native indirect draw rejected owner=venus2 reason=invalid_argument_buffer_or_missing_first_instance_feature first_instance=%u offset=%llu\n",
                      venus->draw_indirect_first_instance,
                      (unsigned long long)indirect_offset);
         return false;
      }
      /* firstInstance is GPU-owned; it cannot be inspected to decide whether
       * an optional vertex-divisor restriction applies to this command. */
      for (uint32_t i = 0; i < pipeline->key.num_bindings; i++) {
         if (pipeline->key.bindings[i].inputRate == VK_VERTEX_INPUT_RATE_INSTANCE &&
             pipeline->key.binding_divisors[i] != 1 &&
             !venus->vertex_divisor_nonzero_first_instance) {
            YTTRIUM_WARN("yttrium: native indirect draw rejected owner=venus2 reason=divisor_nonzero_first_instance_unsupported binding=%u divisor=%u\n",
                         i, pipeline->key.binding_divisors[i]);
            return false;
         }
      }
   }

   if (pipeline->key.targetless_stream_output !=
          draw_state->targetless_stream_output ||
       (draw_state->targetless_stream_output &&
        (pipeline->color_attachment_count || pipeline->depth_image_view ||
         color_resource_count || depth_resource || !so_target_count ||
         !so_targets || !venus->transform_feedback))) {
      YTTRIUM_WARN("yttrium: Venus native draw rejected owner=venus2 reason=invalid_targetless_stream_output pipeline_id=%llu key_targetless_so=%u draw_targetless_so=%u pipeline_color_count=%u pipeline_depth=%u color_count=%u depth=%u so_targets=%u tf=%u\n",
                   (unsigned long long)pipeline->pipeline_obj.id,
                   pipeline->key.targetless_stream_output,
                   draw_state->targetless_stream_output,
                   pipeline->color_attachment_count,
                   pipeline->depth_image_view != VK_NULL_HANDLE,
                   color_resource_count, depth_resource != NULL,
                   so_target_count, venus->transform_feedback);
      return false;
   }

   if (draw_auto) {
      if (!yttrium_venus2_transform_feedback_draw_enabled(venus) ||
          !draw_auto_stride || index_count || index_data ||
          index_data_size || index_resource) {
         YTTRIUM_LOG("yttrium: Venus native draw rejected DrawAuto tf_draw=%u stride=%u index_count=%u index_data=%p index_size=0x%llx index_resource=%p\n",
                     yttrium_venus2_transform_feedback_draw_enabled(venus),
                     draw_auto_stride, index_count, index_data,
                     (unsigned long long)index_data_size, index_resource);
         return false;
      }
      if (!draw_auto_target->counter_buffer_valid ||
          !draw_auto_target->counter_resource ||
          !draw_auto_target->counter_resource->initialized ||
          !draw_auto_target->counter_resource->buffer_backed ||
          !draw_auto_target->counter_resource->buffer) {
         YTTRIUM_LOG("yttrium: Venus native draw rejected DrawAuto counter target=%p valid=%u counter=%p\n",
                     draw_auto_target,
                     draw_auto_target->counter_buffer_valid,
                     draw_auto_target->counter_resource);
         return false;
      }
      const uint32_t max_stride =
         yttrium_venus2_max_transform_feedback_stride(venus);
      if (max_stride && draw_auto_stride > max_stride) {
         YTTRIUM_LOG("yttrium: Venus native draw rejected DrawAuto stride=%u max_stride=%u\n",
                     draw_auto_stride, max_stride);
         return false;
      }
   }

   if (so_target_count) {
      if (!venus->transform_feedback ||
          so_target_count > YTTRIUM_VENUS_MAX_STREAM_OUTPUT_TARGETS ||
          !so_targets) {
         YTTRIUM_LOG("yttrium: Venus native draw rejected stream output unavailable pipeline_id=%llu tf=%u targets=%u\n",
                     (unsigned long long)pipeline->pipeline_obj.id,
                     venus->transform_feedback, so_target_count);
         return false;
      }

      for (uint32_t i = 0; i < so_target_count; i++) {
         struct yttrium_venus_stream_output_target *target =
            &so_targets[i];
         struct yttrium_venus_resource *so_resource = target->resource;
         struct yttrium_venus_resource *counter =
            target->counter_resource;

         if (!so_resource || !so_resource->initialized ||
             !so_resource->buffer_backed || !so_resource->buffer ||
             !(so_resource->buffer_usage &
               VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT) ||
             !target->buffer_size ||
             (target->counter_buffer_valid && !counter) ||
             (counter &&
              (!counter->initialized || !counter->buffer_backed ||
               !counter->buffer ||
               !(counter->buffer_usage &
                 VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT)))) {
            YTTRIUM_LOG("yttrium: Venus native draw rejected stream output target pipeline_id=%llu slot=%u res=%p res_id=%u init=%u buffer_backed=%u buffer=0x%llx usage=0x%x offset=0x%llx size=0x%llx counter=%p counter_init=%u counter_buffer=0x%llx counter_usage=0x%x\n",
                        (unsigned long long)pipeline->pipeline_obj.id,
                        i, so_resource, target->resource_id,
                        so_resource ? so_resource->initialized : 0,
                        so_resource ? so_resource->buffer_backed : 0,
                        (unsigned long long)(so_resource ?
                           YTTRIUM_VENUS_HANDLE_TO_U64(so_resource->buffer) :
                           0),
                        so_resource ? so_resource->buffer_usage : 0,
                        (unsigned long long)target->buffer_offset,
                        (unsigned long long)target->buffer_size,
                        counter,
                        counter ? counter->initialized : 0,
                        (unsigned long long)(counter ?
                           YTTRIUM_VENUS_HANDLE_TO_U64(counter->buffer) : 0),
                        counter ? counter->buffer_usage : 0);
            return false;
         }
      }
   }

   if (color_resource_count > PIPE_MAX_COLOR_BUFS)
      return false;

   const bool has_color = pipeline->color_attachment_count != 0;
   const bool has_depth = pipeline->depth_image_view != VK_NULL_HANDLE;
   if (has_color &&
       (!color_resources || !color_resource_ids ||
        color_resource_count != pipeline->color_attachment_count)) {
      YTTRIUM_LOG("yttrium: Venus native draw rejected color target mismatch pipeline_id=%llu expected=%u got=%u\n",
                  (unsigned long long)pipeline->pipeline_obj.id,
                  pipeline->color_attachment_count, color_resource_count);
      return false;
   }
   for (uint32_t i = 0; i < color_resource_count; i++) {
      struct yttrium_venus_resource *color = color_resources[i];

      if (!color)
         continue;

      if (!color->initialized || color->buffer_backed ||
          !color->image ||
          !(color->image_usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) {
         YTTRIUM_LOG("yttrium: Venus native draw rejected color resource pipeline_id=%llu rt=%u res=%p res_id=%u initialized=%u buffer_backed=%u image=0x%llx usage=0x%x\n",
                     (unsigned long long)pipeline->pipeline_obj.id,
                     i, color, color_resource_ids[i],
                     color ? color->initialized : 0,
                     color ? color->buffer_backed : 0,
                     (unsigned long long)(color ?
                        YTTRIUM_VENUS_HANDLE_TO_U64(color->image) : 0),
                     color ? color->image_usage : 0);
         return false;
      }
   }
   if (has_depth &&
       (!depth_resource || !depth_resource->initialized ||
        depth_resource->buffer_backed || !depth_resource->image ||
        !yttrium_venus_format_has_depth(depth_resource->vk_format))) {
      YTTRIUM_LOG("yttrium: Venus native draw rejected depth resource pipeline_id=%llu depth=%p depth_id=%u initialized=%u buffer_backed=%u image=0x%llx format=%u\n",
                  (unsigned long long)pipeline->pipeline_obj.id,
                  depth_resource, depth_resource_id,
                  depth_resource ? depth_resource->initialized : 0,
                  depth_resource ? depth_resource->buffer_backed : 0,
                  (unsigned long long)(depth_resource ?
                     YTTRIUM_VENUS_HANDLE_TO_U64(depth_resource->image) : 0),
                  depth_resource ? depth_resource->vk_format :
                     VK_FORMAT_UNDEFINED);
      return false;
   }

   const uint32_t render_level = draw_state ? draw_state->render_level : 0;
   const uint32_t render_layer = draw_state ? draw_state->render_layer : 0;
   const uint32_t render_layers =
      draw_state ? MAX2(draw_state->render_layers, 1) : 1;
   const uint32_t render_width =
      draw_state && draw_state->render_width ?
      draw_state->render_width :
      yttrium_venus_subresource_width(resource, render_level);
   const uint32_t render_height =
      draw_state && draw_state->render_height ?
      draw_state->render_height :
      yttrium_venus_subresource_height(resource, render_level);

   const bool has_index_resource =
      index_resource && index_resource->initialized &&
      index_resource->buffer_backed && index_resource->buffer &&
      (index_resource->buffer_usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
   const bool indexed =
      index_data || index_data_size || index_count || index_resource;
   if (indexed &&
       ((!index_data && !has_index_resource) || !index_data_size ||
        (!draw_indirect && !index_count) ||
        (index_resource && !has_index_resource) ||
        (index_type != VK_INDEX_TYPE_UINT16 &&
         index_type != VK_INDEX_TYPE_UINT32))) {
      YTTRIUM_LOG("yttrium: Venus native draw rejected invalid index args data=%p size=0x%llx count=%u type=%u resource=%p resource_id=%u initialized=%u buffer_backed=%u buffer=0x%llx usage=0x%x\n",
                  index_data, (unsigned long long)index_data_size,
                  index_count, index_type, index_resource,
                  index_resource_id,
                  index_resource ? index_resource->initialized : 0,
                  index_resource ? index_resource->buffer_backed : 0,
                  (unsigned long long)(index_resource ?
                     YTTRIUM_VENUS_HANDLE_TO_U64(index_resource->buffer) : 0),
                  index_resource ? index_resource->buffer_usage : 0);
      return false;
   }

retry_batch_layout:
   const VkDeviceSize index_upload_size =
      indexed && !has_index_resource ?
      align64((VkDeviceSize)index_data_size, 4) : 0;
   bool native_draw_batch_candidate = false;

   VkDeviceSize draw_vertex_offsets
      [YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS];
   VkDeviceSize vertex_data_size = 0;
   VkDeviceSize vertex_range_offset = 0;
   VkDeviceSize vertex_range_size = 0;
   bool has_cpu_vertex_upload = false;
   memset(draw_vertex_offsets, 0, sizeof(draw_vertex_offsets));
   if (!yttrium_venus_layout_draw_vertex_uploads(
           venus, resource, resource_id, vertex_uploads, vertex_upload_count,
           false, draw_vertex_offsets,
           &vertex_data_size, &vertex_range_offset, &vertex_range_size,
           &has_cpu_vertex_upload)) {
      YTTRIUM_LOG("yttrium: Venus native draw rejected vertex upload layout pipeline_id=%llu vertex_bindings=%u\n",
                  (unsigned long long)pipeline->pipeline_obj.id,
                  vertex_upload_count);
      if (!venus->display_copy_batch_recording)
         yttrium_venus_cmd_batch_clear_transient_upload_state(venus);
      return false;
   }
   const bool has_sampled_descriptor =
      pipeline->has_sampled_image || pipeline->has_sampled_buffer;
   const bool has_ubo_descriptor = pipeline->ubo_descriptor_count != 0;
   const struct yttrium_venus_native_draw_batch_state batch_state =
      yttrium_venus_get_native_draw_batch_state(
         pipeline, has_cpu_vertex_upload, has_sampled_descriptor,
         has_ubo_descriptor);
   native_draw_batch_candidate = batch_state.candidate;
   const bool async_native_draw = yttrium_venus_async_batch_enabled();
   bool use_push_descriptors = batch_state.use_push_descriptors;
   const bool static_cache_eligible =
      yttrium_gdi_static_ubo_sampled_cache_enabled() &&
      async_native_draw && use_push_descriptors && has_sampled_descriptor &&
      !has_ubo_descriptor && !pipeline->has_storage_image &&
      !pipeline->has_storage_buffer && pipeline->descriptor_set &&
      pipeline->pipeline_layout && pipeline->pipeline && sampled_images &&
      sampled_image_count == pipeline->sampled_image_descriptor_count +
                                pipeline->sampled_buffer_descriptor_count;
   bool use_static_sampled_cache = false;
   bool initialize_static_sampled_cache = false;
   if (static_cache_eligible) {
      if (!pipeline->sampled_descriptor_cache_initialized) {
         use_static_sampled_cache = true;
         initialize_static_sampled_cache = true;
      } else if (yttrium_venus_pipeline_sampled_cache_matches(
                    pipeline, sampled_images, sampled_image_count)) {
         use_static_sampled_cache = true;
      }
      if (use_static_sampled_cache)
         use_push_descriptors = false;
   }
   yttrium_trace_native_draw_batch_decision(
      native_draw_batch_candidate ? 1 : 0,
      batch_state.reject_mask,
      batch_state.native_draw_batch_enabled ? 1 : 0,
      batch_state.cpu_vertex_batch_allowed ? 1 : 0,
      batch_state.cpu_vertex_batch_enabled ? 1 : 0,
      has_cpu_vertex_upload ? 1 : 0,
      has_sampled_descriptor ? 1 : 0,
      has_ubo_descriptor ? 1 : 0,
      batch_state.push_descriptor_batch_enabled ? 1 : 0,
      batch_state.push_descriptors_available ? 1 : 0,
      batch_state.pipeline_has_push_layout ? 1 : 0,
      batch_state.pipeline_has_push_pipeline ? 1 : 0,
      use_push_descriptors ? 1 : 0,
      draw_state->topology,
      vertex_count,
      index_count,
      pipeline->pipeline_obj.id);
   struct yttrium_venus_render_pass_group_key render_pass_key;
   bool render_pass_key_valid = false;
   memset(&render_pass_key, 0, sizeof(render_pass_key));
   const bool render_pass_batch_allowed =
      native_draw_batch_candidate && !draw_indirect &&
      (!has_cpu_vertex_upload || !has_sampled_descriptor ||
       yttrium_venus_sampled_cpu_vertex_render_pass_batch_enabled()) &&
      yttrium_venus_deferred_draw_allowed(
         pipeline, sampled_images, sampled_image_count, color_resources,
         color_resource_count, depth_resource, storage_image_count,
         so_target_count, draw_auto);
   if (render_pass_batch_allowed) {
      render_pass_key_valid =
         yttrium_venus_deferred_draw_make_render_key(
            pipeline, color_resources, color_resource_count, depth_resource,
            draw_state, render_width, render_height, &render_pass_key);
   }
   const bool render_pass_batch_candidate =
      render_pass_batch_allowed && render_pass_key_valid;
   const bool defer_render_pass =
      native_draw_batch_candidate && render_pass_batch_candidate;
   if (defer_render_pass && venus->display_copy_batch_recording &&
       venus->cmd_batch_deferred_draw_count &&
       yttrium_venus_cmd_batch_deferred_image_role_conflict(
          venus, sampled_images, sampled_image_count, color_resources,
          color_resource_count, depth_resource, pipeline->key.depth_read_only)) {
      if (!yttrium_venus_flush_command_batch(
             venus, "native draw deferred image role boundary")) {
         if (!venus->display_copy_batch_recording)
            yttrium_venus_cmd_batch_clear_transient_upload_state(venus);
         return false;
      }
   }
   if (venus->display_copy_batch_recording &&
       venus->cmd_batch_deferred_draw_count && !native_draw_batch_candidate) {
      if (!yttrium_venus_flush_command_batch(
             venus, "native draw after render-pass batch"))
         return false;
   }
   if (native_draw_batch_candidate && venus->display_copy_batch_recording &&
       venus->cmd_batch_deferred_draw_count && !render_pass_batch_candidate) {
      if (!yttrium_venus_flush_command_batch(
             venus, "native draw after render-pass batch"))
         return false;
   }
   VkDeviceSize expected_vertex_watermark =
      venus->cmd_batch_vertex_watermark;
   if (async_native_draw && has_cpu_vertex_upload &&
       !yttrium_venus_layout_draw_vertex_uploads(
          venus, resource, resource_id, vertex_uploads,
          vertex_upload_count, true, draw_vertex_offsets, &vertex_data_size,
          &vertex_range_offset, &vertex_range_size,
          &has_cpu_vertex_upload)) {
      YTTRIUM_LOG("yttrium: Venus native draw rejected batched vertex upload layout pipeline_id=%llu vertex_bindings=%u\n",
                  (unsigned long long)pipeline->pipeline_obj.id,
                  vertex_upload_count);
      if (!venus->display_copy_batch_recording)
         yttrium_venus_cmd_batch_clear_transient_upload_state(venus);
      return false;
   }
   expected_vertex_watermark = venus->cmd_batch_vertex_watermark;

   if (!yttrium_venus_layout_pipeline_ubo_uploads(
           venus, pipeline, ubo_uploads, ubo_upload_count,
           async_native_draw, async_native_draw)) {
      YTTRIUM_LOG("yttrium: Venus native draw rejected ubo layout pipeline_id=%llu expected=%u got=%u\n",
                   (unsigned long long)pipeline->pipeline_obj.id,
                   pipeline->ubo_count, ubo_upload_count);
      if (!venus->display_copy_batch_recording)
         yttrium_venus_cmd_batch_clear_transient_upload_state(venus);
      return false;
   }
   const VkDeviceSize expected_ubo_watermark =
      venus->cmd_batch_ubo_watermark;
   if (has_cpu_vertex_upload &&
       venus->cmd_batch_vertex_watermark != expected_vertex_watermark &&
       !yttrium_venus_layout_draw_vertex_uploads(
          venus, resource, resource_id, vertex_uploads,
          vertex_upload_count, async_native_draw,
          draw_vertex_offsets, &vertex_data_size, &vertex_range_offset,
          &vertex_range_size, &has_cpu_vertex_upload)) {
      YTTRIUM_LOG("yttrium: Venus native draw rejected vertex upload relayout after ubo layout pipeline_id=%llu vertex_bindings=%u\n",
                  (unsigned long long)pipeline->pipeline_obj.id,
                  vertex_upload_count);
      if (!venus->display_copy_batch_recording)
         yttrium_venus_cmd_batch_clear_transient_upload_state(venus);
      return false;
   }
   expected_vertex_watermark = venus->cmd_batch_vertex_watermark;

   VkDeviceSize draw_index_offset = 0;
   VkDeviceSize draw_index_upload_size = 0;
   if (!yttrium_venus_layout_draw_index_upload(
           venus, resource, resource_id, index_data, index_upload_size,
           async_native_draw, &draw_index_offset,
           &draw_index_upload_size)) {
      YTTRIUM_LOG("yttrium: Venus native draw rejected index upload layout pipeline_id=%llu index_bytes=0x%llx\n",
                  (unsigned long long)pipeline->pipeline_obj.id,
                  (unsigned long long)index_data_size);
      if (!venus->display_copy_batch_recording)
         yttrium_venus_cmd_batch_clear_transient_upload_state(venus);
      return false;
   }
   if (has_index_resource) {
      draw_index_offset = index_buffer_offset;
      draw_index_upload_size = (VkDeviceSize)index_data_size;
   }
   const VkDeviceSize expected_index_watermark =
      venus->cmd_batch_index_watermark;
   if (has_cpu_vertex_upload &&
       venus->cmd_batch_vertex_watermark != expected_vertex_watermark &&
       !yttrium_venus_layout_draw_vertex_uploads(
          venus, resource, resource_id, vertex_uploads,
          vertex_upload_count, async_native_draw,
          draw_vertex_offsets, &vertex_data_size, &vertex_range_offset,
          &vertex_range_size, &has_cpu_vertex_upload)) {
      YTTRIUM_LOG("yttrium: Venus native draw rejected vertex upload relayout after index layout pipeline_id=%llu vertex_bindings=%u\n",
                  (unsigned long long)pipeline->pipeline_obj.id,
                  vertex_upload_count);
      if (!venus->display_copy_batch_recording)
         yttrium_venus_cmd_batch_clear_transient_upload_state(venus);
      return false;
   }
   expected_vertex_watermark = venus->cmd_batch_vertex_watermark;

   VkDescriptorSet draw_descriptor_set =
      use_push_descriptors ? VK_NULL_HANDLE : pipeline->descriptor_set;
   if (async_native_draw &&
       !use_push_descriptors &&
       !use_static_sampled_cache &&
       (pipeline->ubo_descriptor_count ||
        pipeline->sampled_image_descriptor_count ||
        pipeline->sampled_buffer_descriptor_count ||
        pipeline->storage_image_descriptor_count ||
        pipeline->storage_buffer_descriptor_count)) {
      if (!yttrium_venus_cmd_batch_alloc_descriptor_set(
             venus, pipeline, &draw_descriptor_set)) {
         YTTRIUM_LOG("yttrium: Venus native draw rejected descriptor set allocation pipeline_id=%llu\n",
                     (unsigned long long)pipeline->pipeline_obj.id);
         if (!venus->display_copy_batch_recording)
            yttrium_venus_cmd_batch_clear_transient_upload_state(venus);
         return false;
      }
      if (ubo_upload_count &&
          venus->cmd_batch_ubo_watermark != expected_ubo_watermark &&
          !yttrium_venus_layout_pipeline_ubo_uploads(
             venus, pipeline, ubo_uploads, ubo_upload_count, true, true)) {
         YTTRIUM_LOG("yttrium: Venus native draw rejected ubo relayout pipeline_id=%llu expected=%u got=%u\n",
                     (unsigned long long)pipeline->pipeline_obj.id,
                     pipeline->ubo_count, ubo_upload_count);
         if (!venus->display_copy_batch_recording) {
            yttrium_venus_cmd_batch_clear_transient_upload_state(venus);
            yttrium_venus_cmd_batch_destroy_descriptor_pool(
               venus, &venus->cmd_batch_descriptor_pool);
         }
         return false;
      }
      if (has_cpu_vertex_upload &&
          venus->cmd_batch_vertex_watermark != expected_vertex_watermark &&
          !yttrium_venus_layout_draw_vertex_uploads(
             venus, resource, resource_id, vertex_uploads,
             vertex_upload_count, true, draw_vertex_offsets,
             &vertex_data_size, &vertex_range_offset, &vertex_range_size,
             &has_cpu_vertex_upload)) {
         YTTRIUM_LOG("yttrium: Venus native draw rejected vertex upload relayout pipeline_id=%llu vertex_bindings=%u\n",
                     (unsigned long long)pipeline->pipeline_obj.id,
                     vertex_upload_count);
         if (!venus->display_copy_batch_recording) {
            yttrium_venus_cmd_batch_clear_transient_upload_state(venus);
            yttrium_venus_cmd_batch_destroy_descriptor_pool(
               venus, &venus->cmd_batch_descriptor_pool);
         }
         return false;
      }
      if (indexed &&
          venus->cmd_batch_index_watermark != expected_index_watermark &&
          !yttrium_venus_layout_draw_index_upload(
             venus, resource, resource_id, index_data, index_upload_size,
             true, &draw_index_offset, &draw_index_upload_size)) {
         YTTRIUM_LOG("yttrium: Venus native draw rejected index upload relayout pipeline_id=%llu index_bytes=0x%llx\n",
                     (unsigned long long)pipeline->pipeline_obj.id,
                     (unsigned long long)index_data_size);
         if (!venus->display_copy_batch_recording) {
            yttrium_venus_cmd_batch_clear_transient_upload_state(venus);
            yttrium_venus_cmd_batch_destroy_descriptor_pool(
               venus, &venus->cmd_batch_descriptor_pool);
         }
         return false;
      }
      if (has_index_resource) {
         draw_index_offset = index_buffer_offset;
         draw_index_upload_size = (VkDeviceSize)index_data_size;
      }
   }

   VkDescriptorImageInfo sampled_image_infos
      [YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   VkBufferView sampled_buffer_views[YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   VkWriteDescriptorSet sampled_writes
      [YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   VkDescriptorImageInfo storage_image_infos
      [YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES];
   VkBufferView storage_buffer_views
      [YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES];
   VkWriteDescriptorSet storage_writes
      [YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES];
   VkDescriptorBufferInfo ubo_infos[YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS];
   VkWriteDescriptorSet ubo_writes[YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS];
   uint32_t sampled_update_count = 0;
   uint32_t storage_update_count = 0;
   uint32_t ubo_update_count = 0;
   bool sampled_push_writes_ready = false;
   bool storage_push_writes_ready = false;
   /*
    * These eight arrays are sized for the maximum binding count and used to be
    * cleared in full on every draw - 12 kB of zeroing per draw, which made
    * memset the second most expensive function in the driver.
    *
    * None of it was load-bearing.  Every element that gets used is assigned a
    * complete compound literal, which zero-fills the members it does not name,
    * and every consumer is bounded by the matching *_update_count: the
    * vkUpdateDescriptorSets calls below, the deferred-draw recorder, and the
    * pImageInfo/pTexelBufferView pointers, which each address one element with
    * descriptorCount 1.  Elements past the count are never read.
    */

   if (!yttrium_venus_update_pipeline_ubo_descriptors(
           venus, pipeline, draw_descriptor_set,
           ubo_uploads, ubo_upload_count, use_push_descriptors,
           ubo_infos, ubo_writes, &ubo_update_count)) {
      YTTRIUM_LOG("yttrium: Venus native draw rejected ubo uploads pipeline_id=%llu expected=%u got=%u\n",
                   (unsigned long long)pipeline->pipeline_obj.id,
                   pipeline->ubo_count, ubo_upload_count);
      if (!venus->display_copy_batch_recording) {
         yttrium_venus_cmd_batch_destroy_descriptor_pool(
            venus, &venus->cmd_batch_descriptor_pool);
         yttrium_venus_cmd_batch_clear_transient_upload_state(venus);
      }
      return false;
   }

   if (has_sampled_descriptor) {
      const uint32_t expected_sampled_count =
         pipeline->key.sampled_binding_count;
      if (!sampled_images || !sampled_image_count ||
          sampled_image_count != expected_sampled_count ||
          sampled_image_count > YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES)
         return false;

      if (!use_push_descriptors && !draw_descriptor_set) {
         YTTRIUM_LOG("yttrium: Venus native draw rejected sampled descriptors pipeline_id=%llu descriptor_set=0x%llx count=%u expected=%u stages=0x%x\n",
                     (unsigned long long)pipeline->pipeline_obj.id,
                     (unsigned long long)YTTRIUM_VENUS_HANDLE_TO_U64(
                        draw_descriptor_set),
                     sampled_image_count,
                     expected_sampled_count,
                     pipeline->key.sampled_stage_mask);
         return false;
      }

      for (uint32_t i = 0; i < sampled_image_count; i++) {
         const struct yttrium_venus_sampled_image *sampled =
            &sampled_images[i];
         const struct yttrium_venus_sampled_binding_layout *layout =
            &pipeline->key.sampled_bindings[i];
         struct yttrium_venus_resource *sampled_resource = sampled->resource;
         uint32_t sampled_resource_id = sampled->resource_id;
         const uint32_t raw_slot = layout->raw_slot;

         if (sampled->binding != layout->binding ||
             sampled->buffer != (layout->buffer != VK_FALSE)) {
            YTTRIUM_LOG("yttrium: Venus native draw rejected sampled binding pipeline_id=%llu index=%u binding=%u expected_binding=%u stage=%u raw_slot=%u buffer=%u expected_buffer=%u\n",
                        (unsigned long long)pipeline->pipeline_obj.id,
                        i, sampled->binding, layout->binding,
                        layout->stage, raw_slot, sampled->buffer,
                        layout->buffer);
            return false;
         }

         const bool expect_buffer = layout->buffer != VK_FALSE;

         if (expect_buffer) {
            if (!sampled_resource || sampled_resource == resource ||
                !sampled_resource->initialized ||
                !sampled_resource->buffer_backed ||
                !sampled_resource->buffer) {
               YTTRIUM_LOG("yttrium: Venus native draw rejected sampled buffer pipeline_id=%llu sampled=%p sampled_id=%u initialized=%u buffer_backed=%u buffer=0x%llx binding=%u raw_slot=%u format=%u\n",
                           (unsigned long long)pipeline->pipeline_obj.id,
                           sampled_resource,
                           sampled_resource_id,
                           sampled_resource ?
                              sampled_resource->initialized : 0,
                           sampled_resource ?
                              sampled_resource->buffer_backed : 0,
                           (unsigned long long)(sampled_resource ?
                              YTTRIUM_VENUS_HANDLE_TO_U64(
                                 sampled_resource->buffer) : 0),
                           sampled->binding, raw_slot, sampled->format);
               return false;
            }
            VkBufferView buffer_view = VK_NULL_HANDLE;
            if (!yttrium_venus_ensure_sample_buffer_view(
                   venus, sampled_resource, sampled_resource_id,
                   sampled->format, sampled->buffer_offset,
                   sampled->buffer_range, &buffer_view))
               return false;

            sampled_buffer_views[sampled_update_count] = buffer_view;
            sampled_writes[sampled_update_count] = (VkWriteDescriptorSet) {
               .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
               .dstSet = use_push_descriptors ? VK_NULL_HANDLE :
                         draw_descriptor_set,
               .dstBinding = sampled->binding,
               .descriptorCount = 1,
               .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
               .pTexelBufferView =
                  &sampled_buffer_views[sampled_update_count],
            };
         } else {
            if (!pipeline->samplers[i]) {
               YTTRIUM_LOG("yttrium: Venus native draw rejected sampled image missing sampler pipeline_id=%llu index=%u stage=%u raw_slot=%u binding=%u\n",
                           (unsigned long long)pipeline->pipeline_obj.id,
                           i, layout->stage, raw_slot, layout->binding);
               return false;
            }

            if (!sampled_resource &&
                !yttrium_venus_ensure_null_sampled_image(
                   venus, &sampled_resource, &sampled_resource_id))
               return false;

            const bool sampled_attachment =
               yttrium_venus_resource_in_list(
                  sampled_resource, color_resources,
                  color_resource_count) ||
               sampled_resource == depth_resource;
            const bool feedback_loop =
               yttrium_venus_sampled_attachment_feedback_loop(
                  pipeline, sampled_resource, color_resources,
                  color_resource_count, depth_resource, NULL, NULL);
            const bool depth_read_only = pipeline->key.depth_read_only &&
                                         sampled_resource == depth_resource;

            if (!sampled_resource ||
                ((sampled_resource == resource || sampled_attachment) &&
                 !feedback_loop && !depth_read_only) ||
                !sampled_resource->initialized ||
                sampled_resource->buffer_backed ||
                !sampled_resource->image) {
               YTTRIUM_LOG("yttrium: Venus native draw rejected sampled image pipeline_id=%llu sampled=%p sampled_id=%u initialized=%u buffer_backed=%u image=0x%llx binding=%u raw_slot=%u attachment=%u feedback_loop=%u color_feedback_mask=0x%x depth_feedback=%u\n",
                           (unsigned long long)pipeline->pipeline_obj.id,
                           sampled_resource,
                           sampled_resource_id,
                           sampled_resource ?
                              sampled_resource->initialized : 0,
                           sampled_resource ?
                              sampled_resource->buffer_backed : 0,
                           (unsigned long long)(sampled_resource ?
                              YTTRIUM_VENUS_HANDLE_TO_U64(
                                 sampled_resource->image) : 0),
                           sampled->binding, raw_slot,
                           sampled_attachment, feedback_loop,
                           pipeline->key.color_feedback_loop_mask,
                           pipeline->key.depth_feedback_loop);
               return false;
            }

            VkImageView sampled_view = VK_NULL_HANDLE;
            if (!yttrium_venus_ensure_sample_image_view(
                   venus, sampled_resource, sampled_resource_id,
                   yttrium_venus2_pipe_format(sampled->format),
                   sampled->swizzle_key, sampled->view_type,
                   sampled->first_level, sampled->level_count,
                   sampled->first_layer, sampled->layer_count,
                   sampled->aspect_mask,
                   &sampled_view))
               return false;

            sampled_image_infos[sampled_update_count] =
               (VkDescriptorImageInfo) {
               .sampler = pipeline->samplers[i],
               .imageView = sampled_view,
               .imageLayout = feedback_loop ?
                  VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT :
                  depth_read_only ?
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL :
                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
            sampled_writes[sampled_update_count] = (VkWriteDescriptorSet) {
               .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
               .dstSet = use_push_descriptors ? VK_NULL_HANDLE :
                         draw_descriptor_set,
               .dstBinding = sampled->binding,
               .descriptorCount = 1,
               .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
               .pImageInfo = &sampled_image_infos[sampled_update_count],
            };
         }
         sampled_update_count++;
      }

      if (use_push_descriptors) {
         sampled_push_writes_ready = sampled_update_count != 0;
      } else if (!use_static_sampled_cache ||
                 initialize_static_sampled_cache) {
         vn_async_vkUpdateDescriptorSets(&venus->vn_ring,
                                         venus->device_handle,
                                         sampled_update_count,
                                         sampled_writes, 0, NULL);
         if (initialize_static_sampled_cache) {
            yttrium_venus_pipeline_sampled_cache_initialize(
               pipeline, sampled_images, sampled_image_count);
         }
      }
   }

   if (use_push_descriptors && !ubo_update_count &&
       !sampled_push_writes_ready &&
       !(pipeline->has_storage_image || pipeline->has_storage_buffer))
      return false;

   if (pipeline->has_storage_image || pipeline->has_storage_buffer) {
      const uint64_t storage_mask =
         pipeline->storage_image_mask | pipeline->storage_buffer_mask;
      if (!storage_images || !storage_image_count ||
          storage_image_count > YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES ||
          (!use_push_descriptors && !draw_descriptor_set))
         return false;

      uint64_t seen_mask = 0;
      for (uint32_t i = 0; i < storage_image_count; i++) {
         const struct yttrium_venus_storage_image *image =
            &storage_images[i];
         struct yttrium_venus_resource *image_resource = image->resource;
         const uint32_t raw_slot =
            image->binding >= YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE ?
            image->binding - YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE :
            UINT32_MAX;
         const uint64_t raw_mask =
            raw_slot < YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES ?
            (1ull << raw_slot) : 0;

         const bool storage_buffer =
            raw_mask && (pipeline->storage_buffer_mask & raw_mask);

         if (raw_slot >= YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES ||
             !(storage_mask & raw_mask) ||
             (seen_mask & raw_mask) ||
             image->binding != yttrium_shader_storage_image_binding(raw_slot))
            return false;

         if (storage_buffer) {
            if (!image_resource || !image_resource->initialized ||
                !image_resource->buffer_backed || !image_resource->buffer)
               return false;

            VkBufferView buffer_view = VK_NULL_HANDLE;
            if (!yttrium_venus_ensure_storage_buffer_view(
                   venus, image_resource, image->resource_id, image->format,
                   image->buffer_offset, image->buffer_range, &buffer_view))
               return false;

            storage_buffer_views[storage_update_count] = buffer_view;
            storage_writes[storage_update_count] = (VkWriteDescriptorSet) {
               .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
               .dstSet = use_push_descriptors ? VK_NULL_HANDLE :
                         draw_descriptor_set,
               .dstBinding = image->binding,
               .descriptorCount = 1,
               .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
               .pTexelBufferView =
                  &storage_buffer_views[storage_update_count],
            };
         } else {
            if (!image_resource || !image_resource->initialized ||
                image_resource->buffer_backed || !image_resource->image)
               return false;

            VkImageView image_view = VK_NULL_HANDLE;
            if (!yttrium_venus_ensure_storage_image_view(
                   venus, image_resource, image->resource_id,
                   yttrium_venus2_pipe_format(image->format),
                   image->view_type, image->first_level, image->level_count,
                   image->first_layer, image->layer_count, image->aspect_mask,
                   &image_view))
               return false;

            storage_image_infos[storage_update_count] =
               (VkDescriptorImageInfo) {
               .imageView = image_view,
               .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };
            storage_writes[storage_update_count] = (VkWriteDescriptorSet) {
               .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
               .dstSet = use_push_descriptors ? VK_NULL_HANDLE :
                         draw_descriptor_set,
               .dstBinding = image->binding,
               .descriptorCount = 1,
               .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
               .pImageInfo = &storage_image_infos[storage_update_count],
            };
         }
         storage_update_count++;
         seen_mask |= raw_mask;
      }

      if (seen_mask != storage_mask)
         return false;

      if (use_push_descriptors) {
         storage_push_writes_ready = storage_update_count != 0;
      } else {
         vn_async_vkUpdateDescriptorSets(&venus->vn_ring,
                                         venus->device_handle,
                                         storage_update_count,
                                         storage_writes, 0, NULL);
      }
   }

   if (use_push_descriptors && !ubo_update_count &&
       !sampled_push_writes_ready && !storage_push_writes_ready)
      return false;

   yttrium_venus_trace_timing(
      YTTRIUM_TRACE_TIMING_VENUS_NATIVE_DRAW_PREP,
      0, stage_start_us, NULL, resource_id,
      pipeline->pipeline_obj.id, sampled_image_count, ubo_upload_count);
   stage_start_us =
      yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;

   const bool native_draw_batch = async_native_draw;
   const VkPipeline draw_pipeline =
      use_push_descriptors ? pipeline->push_pipeline : pipeline->pipeline;
   const struct yttrium_venus_object *draw_pipeline_layout_obj =
      yttrium_venus_select_pipeline_layout(
         &venus->push_descriptor_layout_index, pipeline, use_push_descriptors,
         ubo_update_count + sampled_update_count + storage_update_count);
   const VkPipelineLayout draw_pipeline_layout =
      YTTRIUM_VENUS_HANDLE(
         VkPipelineLayout, draw_pipeline_layout_obj);
   VkCommandBuffer saved_command_buffer = venus->command_buffer;
   VkBuffer draw_index_buffer =
      has_index_resource ? index_resource->buffer : resource->draw_index_buffer;

   if (native_draw_batch) {
      if (!yttrium_venus_begin_command_batch(venus, "native draw batch",
                                             true, true)) {
         if (!venus->display_copy_batch_recording) {
            yttrium_venus_cmd_batch_clear_transient_upload_state(venus);
            yttrium_venus_cmd_batch_destroy_descriptor_pool(
               venus, &venus->cmd_batch_descriptor_pool);
         }
         return false;
      }
      if (!yttrium_venus_cmd_batch_track_draw_refs(
             venus, resource, vertex_uploads, vertex_upload_count,
             color_resources, color_resource_count,
             depth_resource, pipeline)) {
         yttrium_venus_cancel_command_batch_setup_failure(
            venus, "native draw batch pending tracking failure");
         return false;
      }
      for (uint32_t i = 0; i < ubo_upload_count; i++) {
         if (ubo_uploads[i].direct_resource &&
             !yttrium_venus_cmd_batch_track_resource(
                venus, ubo_uploads[i].direct_resource)) {
            yttrium_venus_cancel_command_batch_setup_failure(
               venus, "native draw batch direct ubo tracking failure");
            return false;
         }
      }
      if (has_index_resource &&
          !yttrium_venus_cmd_batch_track_resource(venus, index_resource)) {
         yttrium_venus_cancel_command_batch_setup_failure(
            venus, "native draw batch index tracking failure");
         return false;
      }
      if (draw_indirect &&
          !yttrium_venus_cmd_batch_track_resource(venus, indirect_resource)) {
         yttrium_venus_cancel_command_batch_setup_failure(
            venus, "native draw batch indirect argument tracking failure");
         return false;
      }
      if (!yttrium_venus_cmd_batch_track_sampled_refs(
             venus, sampled_images, sampled_image_count)) {
         yttrium_venus_cancel_command_batch_setup_failure(
            venus, "native draw batch sampled tracking failure");
         return false;
      }
      if (!yttrium_venus_cmd_batch_track_storage_refs(
             venus, storage_images, storage_image_count)) {
         yttrium_venus_cancel_command_batch_setup_failure(
            venus, "native draw batch storage tracking failure");
         return false;
      }
      if (!yttrium_venus_cmd_batch_track_stream_output_refs(
             venus, so_targets, so_target_count, draw_auto_target)) {
         yttrium_venus_cancel_command_batch_setup_failure(
            venus, "native draw batch stream-output tracking failure");
         return false;
      }
      struct yttrium_venus_cmd_batch_footprint footprint;
      memset(&footprint, 0, sizeof(footprint));
      /*
       * Batch descriptor sets are updated per draw and therefore must not be
       * reused while earlier commands are pending.  The pipeline-owned static
       * sampled set is different: it is initialized once, retained by the
       * tracked pipeline, and never updated when the cached bindings match.
       * Treating that immutable set as transient forces a false conflict on
       * every repeated draw that uses it.
       */
      footprint.descriptor_set =
         use_static_sampled_cache ? VK_NULL_HANDLE : draw_descriptor_set;
      footprint.pipeline_id = pipeline->pipeline_obj.id;
      footprint.resource_id = resource_id;
      yttrium_venus_cmd_batch_collect_image_roles(
         sampled_images, sampled_image_count, color_resources,
         color_resource_count, depth_resource, pipeline->key.depth_read_only,
         &footprint);
      if (!yttrium_venus_pipeline_ubo_footprint(
             venus, pipeline, ubo_uploads, ubo_upload_count,
             &footprint.ubo)) {
         yttrium_venus_cancel_command_batch_setup_failure(
            venus, "native draw batch ubo footprint failure");
         return false;
      }
      if (has_cpu_vertex_upload) {
         footprint.vertex.buffer = resource->draw_vertex_buffer;
         footprint.vertex.offset = vertex_range_offset;
         footprint.vertex.size = vertex_range_size;
         footprint.vertex.generation =
            resource->draw_vertex_buffer_generation;
      }
      if (indexed && !has_index_resource) {
         footprint.index.buffer = draw_index_buffer;
         footprint.index.offset = draw_index_offset;
         footprint.index.size = draw_index_upload_size;
         footprint.index.generation = resource->draw_index_buffer_generation;
      }
      if (!yttrium_venus_cmd_batch_record_footprint(venus, &footprint)) {
         if (!retried_transient_footprint_conflict) {
            retried_transient_footprint_conflict = true;
            if (yttrium_venus_flush_command_batch(
                   venus, "native draw transient footprint boundary"))
               goto retry_batch_layout;
         }
         YTTRIUM_WARN("yttrium: Venus native draw emit failed owner=venus2 reason=transient_footprint_boundary resource_id=%u pipeline_id=%llu vertex_buffer=0x%llx vertex_offset=0x%llx vertex_size=0x%llx index_buffer=0x%llx index_offset=0x%llx index_size=0x%llx\n",
                      resource_id,
                      (unsigned long long)pipeline->pipeline_obj.id,
                      (unsigned long long)YTTRIUM_VENUS_HANDLE_TO_U64(
                         footprint.vertex.buffer),
                      (unsigned long long)footprint.vertex.offset,
                      (unsigned long long)footprint.vertex.size,
                      (unsigned long long)YTTRIUM_VENUS_HANDLE_TO_U64(
                         footprint.index.buffer),
                      (unsigned long long)footprint.index.offset,
                      (unsigned long long)footprint.index.size);
         yttrium_venus_cancel_command_batch_setup_failure(
            venus, "native draw transient footprint boundary failure");
         return false;
      }
   }
   if (indexed)
      draw_index_buffer =
         has_index_resource ? index_resource->buffer : resource->draw_index_buffer;

   bool mirror_copy_pending = false;
   for (uint32_t i = 0; !draw_indirect && i < vertex_upload_count; i++) {
      if (vertex_uploads[i].resource &&
          yttrium_venus_device_local_draw_mirror_needs_copy(
             vertex_uploads[i].resource)) {
         mirror_copy_pending = true;
         break;
      }
   }
   if (!draw_indirect && !mirror_copy_pending && has_index_resource) {
      mirror_copy_pending =
         yttrium_venus_device_local_draw_mirror_needs_copy(index_resource);
   }
   if (mirror_copy_pending && venus->cmd_batch_deferred_draw_count &&
       !yttrium_venus_cmd_batch_emit_deferred_draws(
          venus, "device-local draw mirror update"))
      goto fail_restore_command_buffer;

   VkBuffer direct_vertex_buffers
      [YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS];
   bool direct_vertex_mirrored
      [YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS];
   memset(direct_vertex_buffers, 0, sizeof(direct_vertex_buffers));
   memset(direct_vertex_mirrored, 0, sizeof(direct_vertex_mirrored));
   for (uint32_t i = 0; i < vertex_upload_count; i++) {
      if (!vertex_uploads[i].resource)
         continue;
      direct_vertex_buffers[i] = draw_indirect ?
         vertex_uploads[i].resource->buffer :
         yttrium_venus_prepare_device_local_draw_mirror(
            venus, vertex_uploads[i].resource, &direct_vertex_mirrored[i]);
      if (!direct_vertex_buffers[i])
         goto fail_restore_command_buffer;
   }
   bool direct_index_mirrored = false;
   if (has_index_resource && !draw_indirect) {
      draw_index_buffer = yttrium_venus_prepare_device_local_draw_mirror(
         venus, index_resource, &direct_index_mirrored);
      if (!draw_index_buffer)
         goto fail_restore_command_buffer;
   }

   if (mirror_copy_pending &&
       venus->test_fail_after_draw_mirror_copy_once &&
       !venus->test_fail_after_draw_mirror_copy_consumed) {
      venus->test_fail_after_draw_mirror_copy_consumed = true;
      YTTRIUM_WARN("yttrium: TEST: injecting native draw failure after device-local mirror copy owner=venus2 action=abort_command_batch\n");
      goto fail_restore_command_buffer;
   }

   const bool direct_cpu_vertex_upload =
      yttrium_venus_direct_cpu_vertex_upload_enabled();
   const bool direct_vertex_upload =
      native_draw_batch && has_cpu_vertex_upload &&
      direct_cpu_vertex_upload && resource->draw_vertex_mapping.map;
   const bool direct_index_upload =
      native_draw_batch && indexed && !has_index_resource &&
      direct_cpu_vertex_upload &&
      resource->draw_index_mapping.map;

   if (native_draw_batch && has_cpu_vertex_upload && !defer_render_pass &&
       !direct_vertex_upload) {
      const VkBufferMemoryBarrier vertex_reuse_barrier = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .buffer = resource->draw_vertex_buffer,
         .offset = vertex_range_offset,
         .size = vertex_range_size,
      };
      vn_async_vkCmdPipelineBarrier(&venus->vn_ring, venus->command_buffer,
                                    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                    0, NULL, 1, &vertex_reuse_barrier,
                                    0, NULL);
   }

   for (uint32_t i = 0; i < vertex_upload_count; i++) {
      VkDeviceSize written_vertex_size = 0;
      const struct yttrium_venus_vertex_upload *upload = &vertex_uploads[i];
      if (upload->resource)
         continue;
      if (direct_vertex_upload) {
         if (!yttrium_venus_copy_mapped_padded(
                resource->draw_vertex_mapping.map,
                resource->draw_vertex_mapping.size, draw_vertex_offsets[i],
                upload->size, upload->data, &written_vertex_size))
            goto fail_restore_command_buffer;
         yttrium_trace_venus_upload(
            YTTRIUM_TRACE_VENUS_UPLOAD_DIRECT_VERTEX, 0,
            written_vertex_size, 0, resource->draw_vertex_buffer_obj.id,
            0, resource_id, 0, 0, 0, 0, 0);
      } else if (defer_render_pass) {
         if (!yttrium_venus_cmd_batch_add_upload(
                venus, resource->draw_vertex_buffer, draw_vertex_offsets[i],
                upload->size, upload->data, &written_vertex_size))
            goto fail_restore_command_buffer;
      } else {
         if (!yttrium_venus_cmd_update_buffer_padded(
                venus, venus->command_buffer, resource->draw_vertex_buffer,
                draw_vertex_offsets[i], upload->size, upload->data,
                &written_vertex_size))
            goto fail_restore_command_buffer;
      }
   }
   VkBufferMemoryBarrier buffer_barriers
      [YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS + 1];
   uint32_t buffer_barrier_count = 0;
   VkPipelineStageFlags vertex_src_stages = 0;
   if (has_cpu_vertex_upload) {
      const VkAccessFlags src_access =
         direct_vertex_upload ? VK_ACCESS_HOST_WRITE_BIT :
         VK_ACCESS_TRANSFER_WRITE_BIT;
      const VkPipelineStageFlags src_stage =
         direct_vertex_upload ? VK_PIPELINE_STAGE_HOST_BIT :
         VK_PIPELINE_STAGE_TRANSFER_BIT;
      buffer_barriers[buffer_barrier_count++] = (VkBufferMemoryBarrier) {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
         .srcAccessMask = src_access,
         .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .buffer = resource->draw_vertex_buffer,
         .offset = vertex_range_offset,
         .size = vertex_range_size,
      };
      vertex_src_stages |= src_stage;
   }
   for (uint32_t i = 0; i < vertex_upload_count; i++) {
      const struct yttrium_venus_vertex_upload *upload = &vertex_uploads[i];
      if (!upload->resource)
         continue;
      if (direct_vertex_mirrored[i])
         continue;
      const VkAccessFlags src_access =
         VK_ACCESS_TRANSFER_WRITE_BIT |
         VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT |
         (draw_indirect ? VK_ACCESS_SHADER_WRITE_BIT : 0) |
         (upload->host_write_pending ? VK_ACCESS_HOST_WRITE_BIT : 0);
      const VkPipelineStageFlags src_stage =
         VK_PIPELINE_STAGE_TRANSFER_BIT |
         VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT |
         (draw_indirect ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : 0) |
         (upload->host_write_pending ? VK_PIPELINE_STAGE_HOST_BIT : 0);
      buffer_barriers[buffer_barrier_count++] = (VkBufferMemoryBarrier) {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
         .srcAccessMask = src_access,
         .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .buffer = upload->resource->buffer,
         .offset = upload->buffer_offset,
         .size = upload->size,
      };
      vertex_src_stages |= src_stage;
   }
   if (indexed && !has_index_resource) {
      VkDeviceSize written_index_size = 0;
      if (direct_index_upload) {
         if (!yttrium_venus_copy_mapped_padded(
                resource->draw_index_mapping.map,
                resource->draw_index_mapping.size, draw_index_offset,
                index_data_size, index_data, &written_index_size))
            goto fail_restore_command_buffer;
         yttrium_trace_venus_upload(
            YTTRIUM_TRACE_VENUS_UPLOAD_DIRECT_INDEX, 0,
            written_index_size, 0, resource->draw_index_buffer_obj.id,
            0, resource_id, 0, 0, 0, 0, 0);
      } else if (defer_render_pass) {
         if (!yttrium_venus_cmd_batch_add_upload(
                venus, draw_index_buffer, draw_index_offset, index_data_size,
                index_data, &written_index_size))
            goto fail_restore_command_buffer;
      } else {
         if (!yttrium_venus_cmd_update_buffer_padded(
                venus, venus->command_buffer, draw_index_buffer,
                draw_index_offset, index_data_size, index_data,
                &written_index_size))
            goto fail_restore_command_buffer;
      }

      const VkAccessFlags src_access =
         direct_index_upload ? VK_ACCESS_HOST_WRITE_BIT :
         VK_ACCESS_TRANSFER_WRITE_BIT;
      const VkPipelineStageFlags src_stage =
         direct_index_upload ? VK_PIPELINE_STAGE_HOST_BIT :
         VK_PIPELINE_STAGE_TRANSFER_BIT;
      buffer_barriers[buffer_barrier_count++] =
         (VkBufferMemoryBarrier) {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = src_access,
            .dstAccessMask = VK_ACCESS_INDEX_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = draw_index_buffer,
            .offset = draw_index_offset,
            .size = written_index_size,
      };
      vertex_src_stages |= src_stage;
   } else if (indexed && has_index_resource && !direct_index_mirrored) {
      const VkAccessFlags src_access =
         VK_ACCESS_TRANSFER_WRITE_BIT |
         VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT |
         (draw_indirect ? VK_ACCESS_SHADER_WRITE_BIT : 0) |
         (index_host_write_pending ? VK_ACCESS_HOST_WRITE_BIT : 0);
      const VkPipelineStageFlags src_stage =
         VK_PIPELINE_STAGE_TRANSFER_BIT |
         VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT |
         (draw_indirect ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : 0) |
         (index_host_write_pending ? VK_PIPELINE_STAGE_HOST_BIT : 0);
      buffer_barriers[buffer_barrier_count++] =
         (VkBufferMemoryBarrier) {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = src_access,
            .dstAccessMask = VK_ACCESS_INDEX_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = draw_index_buffer,
            .offset = draw_index_offset,
            .size = index_data_size,
         };
      vertex_src_stages |= src_stage;
   }
   if (buffer_barrier_count && defer_render_pass) {
      for (uint32_t i = 0; i < buffer_barrier_count; i++) {
         if (!yttrium_venus_cmd_batch_add_upload_barrier(
                venus, vertex_src_stages,
                VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                &buffer_barriers[i]))
            goto fail_restore_command_buffer;
      }
   } else if (buffer_barrier_count) {
      vn_async_vkCmdPipelineBarrier(&venus->vn_ring, venus->command_buffer,
                                     vertex_src_stages,
                                     VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0,
                                     0, NULL, buffer_barrier_count,
                                     buffer_barriers, 0, NULL);
   }

   if (draw_indirect) {
      /* CopyStructureCount, UAV stores and CPU-authored initial arguments all
       * converge here.  Consume the original buffer, not a CPU shadow or a
       * vertex/index mirror.  This is outside the render pass. */
      const VkBufferMemoryBarrier barrier = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .buffer = indirect_resource->buffer,
         .offset = indirect_offset,
         .size = indirect_size,
      };
      vn_async_vkCmdPipelineBarrier(
         &venus->vn_ring, venus->command_buffer,
         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
         VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0,
         0, NULL, 1, &barrier, 0, NULL);
   }

   const VkPipelineStageFlags sampled_stage_flags =
      yttrium_venus_sampled_stage_flags(pipeline);

   if (pipeline->has_sampled_buffer) {
      VkBufferMemoryBarrier sampled_buffer_barriers
         [YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
      uint32_t sampled_buffer_barrier_count = 0;
      memset(sampled_buffer_barriers, 0, sizeof(sampled_buffer_barriers));

      for (uint32_t i = 0; i < sampled_image_count; i++) {
         const struct yttrium_venus_sampled_image *sampled =
            &sampled_images[i];
         struct yttrium_venus_resource *sampled_resource =
            sampled->resource;

         if (!sampled->buffer)
            continue;
         if (!sampled_resource || !sampled_resource->buffer ||
             !sampled->buffer_size ||
             (!sampled->buffer_data && !sampled_resource->contents_initialized) ||
             sampled_buffer_barrier_count >=
                YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES)
            goto fail_restore_command_buffer;

         VkDeviceSize written_size = 0;
         if (sampled->buffer_data) {
            const VkBufferMemoryBarrier overwrite = {
               .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
               .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT |
                                VK_ACCESS_MEMORY_WRITE_BIT,
               .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
               .buffer = sampled_resource->buffer,
               .offset = 0,
               .size = sampled->buffer_size,
            };
            vn_async_vkCmdPipelineBarrier(
               &venus->vn_ring, venus->command_buffer,
               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
               0, NULL, 1, &overwrite, 0, NULL);
            if (!yttrium_venus_cmd_update_buffer_padded(
                   venus, venus->command_buffer, sampled_resource->buffer, 0,
                   sampled->buffer_size, sampled->buffer_data, &written_size))
               goto fail_restore_command_buffer;
         }
         YTTRIUM_LOG("yttrium: Venus sampled buffer publish res_id=%u buffer_id=%llu view_offset=0x%llx view_range=0x%llx upload_offset=0x0 upload_size=0x%llx written_size=0x%llx\n",
                     sampled->resource_id,
                     (unsigned long long)sampled_resource->buffer_obj.id,
                     (unsigned long long)sampled->buffer_offset,
                     (unsigned long long)sampled->buffer_range,
                     (unsigned long long)sampled->buffer_size,
                     (unsigned long long)written_size);

         sampled_buffer_barriers[sampled_buffer_barrier_count++] =
            (VkBufferMemoryBarrier) {
               .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
               .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT |
                                VK_ACCESS_HOST_WRITE_BIT,
               .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
               .buffer = sampled_resource->buffer,
               .offset = sampled->buffer_offset,
               .size = sampled->buffer_range,
            };
         sampled_resource->contents_initialized = true;
      }

      if (sampled_buffer_barrier_count) {
         vn_async_vkCmdPipelineBarrier(&venus->vn_ring,
                                        venus->command_buffer,
                                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT |
                                        VK_PIPELINE_STAGE_HOST_BIT,
                                        sampled_stage_flags,
                                        0, 0, NULL,
                                        sampled_buffer_barrier_count,
                                        sampled_buffer_barriers, 0, NULL);
      }
   }

   if (pipeline->has_storage_buffer) {
      VkBufferMemoryBarrier storage_buffer_barriers
         [YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES];
      uint32_t storage_buffer_barrier_count = 0;
      memset(storage_buffer_barriers, 0, sizeof(storage_buffer_barriers));

      for (uint32_t i = 0; i < storage_image_count; i++) {
         const struct yttrium_venus_storage_image *storage =
            &storage_images[i];
         struct yttrium_venus_resource *storage_resource =
            storage->resource;

         if (!storage->buffer)
            continue;
         if (!storage_resource || !storage_resource->buffer ||
             storage_buffer_barrier_count >=
                YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES)
            goto fail_restore_command_buffer;
         VkDeviceSize written_size = 0;
         if (storage->buffer_data && storage->buffer_size) {
            const VkBufferMemoryBarrier overwrite = {
               .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
               .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT |
                                VK_ACCESS_MEMORY_WRITE_BIT,
               .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
               .buffer = storage_resource->buffer,
               .offset = 0,
               .size = storage->buffer_size,
            };
            vn_async_vkCmdPipelineBarrier(
               &venus->vn_ring, venus->command_buffer,
               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
               0, NULL, 1, &overwrite, 0, NULL);
            if (!yttrium_venus_cmd_update_buffer_padded(
                   venus, venus->command_buffer, storage_resource->buffer, 0,
                   storage->buffer_size, storage->buffer_data, &written_size))
               goto fail_restore_command_buffer;
         }

         storage_buffer_barriers[storage_buffer_barrier_count++] =
            (VkBufferMemoryBarrier) {
               .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
               .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT |
                                VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
               .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                VK_ACCESS_SHADER_WRITE_BIT,
               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
               .buffer = storage_resource->buffer,
               .offset = storage->buffer_offset,
               .size = storage->buffer_range,
            };
         storage_resource->contents_initialized = true;
      }

      if (storage_buffer_barrier_count) {
         vn_async_vkCmdPipelineBarrier(&venus->vn_ring,
                                        venus->command_buffer,
                                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT |
                                        VK_PIPELINE_STAGE_HOST_BIT,
                                        VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                                        0, 0, NULL,
                                        storage_buffer_barrier_count,
                                        storage_buffer_barriers, 0, NULL);
      }
   }

   VkBufferMemoryBarrier ubo_barriers[YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS];
   const bool direct_ubo_upload =
      native_draw_batch && yttrium_venus_direct_ubo_upload_enabled() &&
      venus->ubo_upload_arena && venus->ubo_upload_arena->mapping.map;
   const VkPipelineStageFlags ubo_src_stage =
      VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
   uint32_t ubo_barrier_count = 0;
   for (uint32_t i = 0; i < ubo_upload_count; i++) {
      const struct yttrium_venus_ubo_upload *upload = &ubo_uploads[i];
      struct yttrium_venus_ubo_slot *slot =
         yttrium_venus_pipeline_find_ubo_slot(pipeline, upload->binding,
                                              upload->array_element);
      if (!slot || !slot->buffer)
         goto fail_restore_command_buffer;

      const bool direct_resource = upload->direct_resource != NULL;
      if (direct_resource) {
         yttrium_trace_venus_upload(
            YTTRIUM_TRACE_VENUS_UPLOAD_DIRECT_UBO, 3, 0,
            0, upload->direct_resource->buffer_obj.id,
            upload->direct_offset, upload->size, 0, 0, 0, 0, 0);
      } else if (direct_ubo_upload) {
         if (slot->upload_reused) {
            yttrium_trace_venus_upload(
               YTTRIUM_TRACE_VENUS_UPLOAD_DIRECT_UBO, 1, 0,
               0, venus->ubo_upload_arena->buffer_obj.id,
               0, 0, 0, 0, 0, 0, 0);
         } else {
            VkDeviceSize written_ubo_size = 0;
            if (!yttrium_venus_copy_mapped_padded(
                   venus->ubo_upload_arena->mapping.map,
                   venus->ubo_upload_arena->mapping.size, slot->offset,
                   upload->size, upload->data, &written_ubo_size))
               goto fail_restore_command_buffer;
            if (slot->resource_version_cacheable) {
               yttrium_venus_store_resource_ubo_version(
                  upload->source_version_cache,
                  venus->ubo_upload_buffer_generation,
                  upload->source_contents_serial,
                  upload->source_offset, (uint32_t)upload->size,
                  slot->offset);
            }
            yttrium_trace_venus_upload(
               YTTRIUM_TRACE_VENUS_UPLOAD_DIRECT_UBO, 0,
               written_ubo_size, 0,
               venus->ubo_upload_arena->buffer_obj.id,
               0, 0, 0, 0, 0, 0, 0);
         }
      } else if (defer_render_pass) {
         if (!yttrium_venus_cmd_batch_add_upload(venus, slot->buffer,
                                                 slot->offset, upload->size,
                                                 upload->data, NULL))
            goto fail_restore_command_buffer;
      } else {
         yttrium_venus_cmd_update_buffer_chunks(venus, venus->command_buffer,
                                                slot->buffer, slot->offset,
                                                upload->size, upload->data);
      }
      ubo_barriers[ubo_barrier_count++] = (VkBufferMemoryBarrier) {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
         .srcAccessMask = direct_resource || direct_ubo_upload ?
                             VK_ACCESS_HOST_WRITE_BIT :
                             VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .buffer = slot->buffer,
         .offset = slot->offset,
         .size = upload->size,
      };
   }
   if (ubo_barrier_count && defer_render_pass) {
      for (uint32_t i = 0; i < ubo_barrier_count; i++) {
         if (!yttrium_venus_cmd_batch_add_upload_barrier(
                venus, ubo_src_stage,
                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                &ubo_barriers[i]))
            goto fail_restore_command_buffer;
      }
   } else if (ubo_barrier_count) {
      vn_async_vkCmdPipelineBarrier(
         &venus->vn_ring, venus->command_buffer, ubo_src_stage,
         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
         0, 0, NULL, ubo_barrier_count, ubo_barriers, 0, NULL);
   }

   if (pipeline->has_sampled_image) {
      for (uint32_t i = 0; i < sampled_image_count; i++) {
         if (sampled_images[i].buffer)
            continue;

         struct yttrium_venus_resource *sampled_resource =
            sampled_images[i].resource;
         uint32_t sampled_resource_id = sampled_images[i].resource_id;
         VkImageAspectFlags sample_aspects = sampled_images[i].aspect_mask;
         bool duplicate_resource = false;
         if (sampled_resource) {
            for (uint32_t j = 0; j < i; j++) {
               if (!sampled_images[j].buffer &&
                   sampled_images[j].resource == sampled_resource) {
                  duplicate_resource = true;
                  break;
               }
            }
            if (duplicate_resource)
               continue;

            for (uint32_t j = i + 1; j < sampled_image_count; j++) {
               if (!sampled_images[j].buffer &&
                   sampled_images[j].resource == sampled_resource)
                  sample_aspects |= sampled_images[j].aspect_mask;
            }
         }
         if (!sample_aspects)
            sample_aspects = VK_IMAGE_ASPECT_COLOR_BIT;
         const VkImageSubresourceRange sample_range = {
            .aspectMask = sample_aspects,
            .baseMipLevel = sampled_images[i].first_level,
            .levelCount = sampled_images[i].level_count,
            .baseArrayLayer = sampled_images[i].first_layer,
            .layerCount = sampled_images[i].layer_count,
         };

         if (!sampled_resource &&
             !yttrium_venus_ensure_null_sampled_image(
                venus, &sampled_resource, &sampled_resource_id))
            goto fail_restore_command_buffer;

         if (!yttrium_venus_cmd_ensure_image_initialized(venus,
                                                          sampled_resource,
                                                          sampled_resource_id))
            goto fail_restore_command_buffer;
         bool color_feedback = false;
         bool depth_feedback = false;
         const bool feedback_loop =
            yttrium_venus_sampled_attachment_feedback_loop(
               pipeline, sampled_resource, color_resources,
               color_resource_count, depth_resource,
               &color_feedback, &depth_feedback);
         const bool depth_read_only = pipeline->key.depth_read_only &&
                                      sampled_resource == depth_resource;
         const VkImageLayout sampled_layout = feedback_loop ?
            VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT :
            depth_read_only ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL :
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
         VkAccessFlags sampled_access = VK_ACCESS_SHADER_READ_BIT;
         VkPipelineStageFlags sampled_stages = sampled_stage_flags;
         if (color_feedback) {
            sampled_access |= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                              VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            sampled_stages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
         }
         if (depth_feedback || depth_read_only) {
            sampled_access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                              (depth_feedback ?
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : 0);
            sampled_stages |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                              VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
         }
         if (defer_render_pass) {
            if (!yttrium_venus_cmd_batch_transition_image(
                   venus, sampled_resource, sampled_layout,
                   sampled_access, sampled_stages,
                   &sample_range))
               goto fail_restore_command_buffer;
         } else {
            yttrium_venus_cmd_transition_image(
               venus, sampled_resource, sampled_layout,
               sampled_access, sampled_stages, &sample_range);
         }
      }
   }
   if (pipeline->has_storage_image) {
      const VkPipelineStageFlags storage_stage_flags =
         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      for (uint32_t i = 0; i < storage_image_count; i++) {
         struct yttrium_venus_resource *image_resource =
            storage_images[i].resource;
         if (storage_images[i].buffer)
            continue;
         if (!image_resource)
            goto fail_restore_command_buffer;

         const VkImageSubresourceRange range = {
            .aspectMask = storage_images[i].aspect_mask ?
               storage_images[i].aspect_mask : VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = storage_images[i].first_level,
            .levelCount = storage_images[i].level_count,
            .baseArrayLayer = storage_images[i].first_layer,
            .layerCount = storage_images[i].layer_count,
         };

         if (!yttrium_venus_cmd_ensure_image_initialized(
                venus, image_resource, storage_images[i].resource_id))
            goto fail_restore_command_buffer;
         yttrium_venus_cmd_transition_image(venus, image_resource,
                                            VK_IMAGE_LAYOUT_GENERAL,
                                            VK_ACCESS_SHADER_READ_BIT |
                                            VK_ACCESS_SHADER_WRITE_BIT,
                                            storage_stage_flags,
                                            &range);
      }
   }
   for (uint32_t i = 0; i < color_resource_count; i++) {
      struct yttrium_venus_resource *color = color_resources[i];
      const uint32_t color_id = color_resource_ids[i];
      const uint32_t color_level =
         draw_state && i < ARRAY_SIZE(draw_state->rt_level) ?
            draw_state->rt_level[i] : render_level;
      const uint32_t color_layer =
         draw_state && i < ARRAY_SIZE(draw_state->rt_layer) ?
            draw_state->rt_layer[i] : render_layer;

      if (!color)
         continue;

      if (!yttrium_venus_cmd_ensure_image_initialized(venus, color,
                                                      color_id))
         goto fail_restore_command_buffer;
      const VkImageSubresourceRange range =
         yttrium_venus_render_barrier_range(color,
                                            VK_IMAGE_ASPECT_COLOR_BIT,
                                            color_level, color_layer,
                                            render_layers);
      const bool feedback_loop =
         (pipeline->key.color_feedback_loop_mask & (1u << i)) != 0;
      const VkImageLayout color_layout = feedback_loop ?
         VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT :
         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      const VkAccessFlags color_access =
         VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
         (feedback_loop ? VK_ACCESS_SHADER_READ_BIT : 0);
      const VkPipelineStageFlags color_stages =
         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
         (feedback_loop ? sampled_stage_flags : 0);
      if (defer_render_pass) {
         if (!yttrium_venus_cmd_batch_transition_image(
                venus, color, color_layout, color_access,
                color_stages, &range))
            goto fail_restore_command_buffer;
      } else {
         yttrium_venus_cmd_transition_image(
            venus, color, color_layout, color_access,
            color_stages, &range);
      }
   }
   if (has_depth) {
      if (!yttrium_venus_cmd_ensure_depth_initialized(venus, depth_resource,
                                                      depth_resource_id))
         goto fail_restore_command_buffer;

      const VkImageSubresourceRange depth_range =
         yttrium_venus_render_barrier_range(
            depth_resource,
            yttrium_venus_format_aspects(depth_resource->vk_format),
            draw_state->depth_level, draw_state->depth_layer,
            MAX2(draw_state->depth_layers, 1));
      const bool feedback_loop = pipeline->key.depth_feedback_loop;
      const bool depth_read_only = pipeline->key.depth_read_only;
      const VkImageLayout depth_layout = feedback_loop ?
         VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT :
         depth_read_only ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL :
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      const VkAccessFlags depth_access =
         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
         (depth_read_only ? 0 : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT) |
         ((feedback_loop || depth_read_only) ? VK_ACCESS_SHADER_READ_BIT : 0);
      const VkPipelineStageFlags depth_stages =
         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
         VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
         ((feedback_loop || depth_read_only) ? sampled_stage_flags : 0);
      if (defer_render_pass) {
         if (!yttrium_venus_cmd_batch_transition_image(
                venus, depth_resource, depth_layout,
                depth_access, depth_stages, &depth_range))
            goto fail_restore_command_buffer;
      } else {
         yttrium_venus_cmd_transition_image(
            venus, depth_resource, depth_layout,
            depth_access, depth_stages, &depth_range);
      }
   }

   if (so_target_count || draw_auto) {
      VkBufferMemoryBarrier so_barriers
         [YTTRIUM_VENUS_MAX_STREAM_OUTPUT_TARGETS * 2 + 1];
      uint32_t so_barrier_count = 0;
      VkPipelineStageFlags src_stages = 0;
      VkPipelineStageFlags dst_stages = 0;

      memset(so_barriers, 0, sizeof(so_barriers));
      for (uint32_t i = 0; i < so_target_count; i++) {
         const struct yttrium_venus_stream_output_target *target =
            &so_targets[i];

         so_barriers[so_barrier_count++] = (VkBufferMemoryBarrier) {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                             VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
            .dstAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = target->resource->buffer,
            .offset = target->buffer_offset,
            .size = target->buffer_size,
         };
         src_stages |= VK_PIPELINE_STAGE_TRANSFER_BIT |
                       VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
         dst_stages |= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;

         if (target->counter_buffer_valid) {
            so_barriers[so_barrier_count++] = (VkBufferMemoryBarrier) {
               .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
               .srcAccessMask =
                  VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
               .dstAccessMask =
                  VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT,
               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
               .buffer = target->counter_resource->buffer,
               .offset = 0,
               .size = sizeof(uint32_t),
            };
            src_stages |= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
            dst_stages |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
         }
      }

      if (draw_auto) {
         so_barriers[so_barrier_count++] = (VkBufferMemoryBarrier) {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask =
               VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
            .dstAccessMask =
               VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = draw_auto_target->counter_resource->buffer,
            .offset = 0,
            .size = sizeof(uint32_t),
         };
         src_stages |= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
         dst_stages |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
      }

      vn_async_vkCmdPipelineBarrier(&venus->vn_ring, venus->command_buffer,
                                    src_stages, dst_stages, 0,
                                    0, NULL, so_barrier_count, so_barriers,
                                    0, NULL);
   }

   const uint32_t viewport_count = MAX2(draw_state->viewport_count, 1);
   VkBuffer vertex_buffers[YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS];
   VkDeviceSize vertex_offsets[YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS];
   for (uint32_t i = 0; i < vertex_upload_count; i++) {
      vertex_buffers[i] = vertex_uploads[i].resource ?
         direct_vertex_buffers[i] : resource->draw_vertex_buffer;
      vertex_offsets[i] = vertex_uploads[i].resource ?
         vertex_uploads[i].buffer_offset : draw_vertex_offsets[i];
   }
   VkBuffer so_buffers[YTTRIUM_VENUS_MAX_STREAM_OUTPUT_TARGETS];
   VkDeviceSize so_offsets[YTTRIUM_VENUS_MAX_STREAM_OUTPUT_TARGETS];
   VkDeviceSize so_sizes[YTTRIUM_VENUS_MAX_STREAM_OUTPUT_TARGETS];
   VkBuffer so_counter_buffers[YTTRIUM_VENUS_MAX_STREAM_OUTPUT_TARGETS];
   VkDeviceSize so_counter_offsets[YTTRIUM_VENUS_MAX_STREAM_OUTPUT_TARGETS];

   if (defer_render_pass) {
      if (!yttrium_venus_cmd_batch_append_deferred_draw(
             venus, pipeline, draw_pipeline_layout_obj,
             render_width, render_height,
             &render_pass_key, use_push_descriptors, draw_descriptor_set,
             ubo_writes, ubo_infos, ubo_update_count,
             sampled_writes, sampled_image_infos, sampled_buffer_views,
             sampled_update_count, draw_state, vertex_buffers,
             vertex_offsets, vertex_upload_count, indexed, draw_index_buffer,
             draw_index_offset, index_type, vertex_count, index_count,
             instance_count, vertex_offset))
         goto fail_restore_command_buffer;
   } else {
      const VkRenderPassBeginInfo render_pass_begin = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
         .renderPass = pipeline->render_pass,
         .framebuffer = pipeline->framebuffer,
         .renderArea = {
            .offset = { 0, 0 },
            .extent = { render_width, render_height },
         },
      };
      vn_async_vkCmdBeginRenderPass(&venus->vn_ring, venus->command_buffer,
                                    &render_pass_begin,
                                    VK_SUBPASS_CONTENTS_INLINE);

      vn_async_vkCmdSetViewport(&venus->vn_ring, venus->command_buffer, 0,
                                viewport_count, draw_state->viewports);
      vn_async_vkCmdSetScissor(&venus->vn_ring, venus->command_buffer, 0,
                               viewport_count, draw_state->scissors);
      vn_async_vkCmdBindPipeline(&venus->vn_ring, venus->command_buffer,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 draw_pipeline);
      if (use_push_descriptors) {
         VkWriteDescriptorSet push_writes
            [YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS +
             YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES +
             YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES];
         uint32_t push_write_count = 0;

         for (uint32_t i = 0; i < ubo_update_count; i++)
            push_writes[push_write_count++] = ubo_writes[i];
         for (uint32_t i = 0; i < sampled_update_count; i++)
            push_writes[push_write_count++] = sampled_writes[i];
         for (uint32_t i = 0; i < storage_update_count; i++)
            push_writes[push_write_count++] = storage_writes[i];

         if (push_write_count) {
            vn_async_vkCmdPushDescriptorSet(&venus->vn_ring,
                                            venus->command_buffer,
                                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            draw_pipeline_layout, 0,
                                            push_write_count,
                                            push_writes);
         }
      } else if (draw_descriptor_set) {
         vn_async_vkCmdBindDescriptorSets(&venus->vn_ring,
                                          venus->command_buffer,
                                          VK_PIPELINE_BIND_POINT_GRAPHICS,
                                          draw_pipeline_layout, 0, 1,
                                          &draw_descriptor_set, 0, NULL);
      }
      yttrium_venus_cmd_push_static_ubo_constants(
         venus, &venus->vn_ring, draw_pipeline_layout,
         draw_state->push_constant_data +
            YTTRIUM_SHADER_VS_PUSH_CONSTANT_OFFSET,
         draw_state->push_constant_vs_size,
         draw_state->push_constant_data +
            YTTRIUM_SHADER_FS_PUSH_CONSTANT_OFFSET,
         draw_state->push_constant_fs_size);
      vn_async_vkCmdSetBlendConstants(&venus->vn_ring,
                                      venus->command_buffer,
                                      draw_state->blend_constants);
      if (vertex_upload_count) {
         vn_async_vkCmdBindVertexBuffers(&venus->vn_ring,
                                         venus->command_buffer, 0,
                                         vertex_upload_count, vertex_buffers,
                                         vertex_offsets);
      }
      if (so_target_count) {
         memset(so_buffers, 0, sizeof(so_buffers));
         memset(so_offsets, 0, sizeof(so_offsets));
         memset(so_sizes, 0, sizeof(so_sizes));
         memset(so_counter_buffers, 0, sizeof(so_counter_buffers));
         memset(so_counter_offsets, 0, sizeof(so_counter_offsets));

         for (uint32_t i = 0; i < so_target_count; i++) {
            struct yttrium_venus_stream_output_target *target =
               &so_targets[i];

            so_buffers[i] = target->resource->buffer;
            so_offsets[i] = target->buffer_offset;
            so_sizes[i] = target->buffer_size;
            if (target->counter_buffer_valid && target->counter_resource)
               so_counter_buffers[i] = target->counter_resource->buffer;
         }

         vn_async_vkCmdBindTransformFeedbackBuffersEXT(
            &venus->vn_ring, venus->command_buffer, 0, so_target_count,
            so_buffers, so_offsets, so_sizes);
         vn_async_vkCmdBeginTransformFeedbackEXT(
            &venus->vn_ring, venus->command_buffer, 0, so_target_count,
            so_counter_buffers, so_counter_offsets);
      }

      if (draw_auto) {
         vn_async_vkCmdDrawIndirectByteCountEXT(
            &venus->vn_ring, venus->command_buffer,
            instance_count, 0,
            draw_auto_target->counter_resource->buffer, 0, 0,
            draw_auto_stride);
      } else if (indexed) {
         vn_async_vkCmdBindIndexBuffer(&venus->vn_ring, venus->command_buffer,
                                       draw_index_buffer, draw_index_offset,
                                       index_type);
         if (draw_indirect)
            vn_async_vkCmdDrawIndexedIndirect(
               &venus->vn_ring, venus->command_buffer,
               indirect_resource->buffer, indirect_offset, 1,
               sizeof(VkDrawIndexedIndirectCommand));
         else
            vn_async_vkCmdDrawIndexed(&venus->vn_ring, venus->command_buffer,
                                      index_count, instance_count, 0,
                                      vertex_offset, 0);
      } else if (draw_indirect) {
         vn_async_vkCmdDrawIndirect(
            &venus->vn_ring, venus->command_buffer,
            indirect_resource->buffer, indirect_offset, 1,
            sizeof(VkDrawIndirectCommand));
      } else {
         vn_async_vkCmdDraw(&venus->vn_ring, venus->command_buffer,
                            vertex_count, instance_count, 0, 0);
      }
      if (so_target_count) {
         for (uint32_t i = 0; i < so_target_count; i++) {
            so_counter_buffers[i] = so_targets[i].counter_resource ?
               so_targets[i].counter_resource->buffer : VK_NULL_HANDLE;
            so_counter_offsets[i] = 0;
         }
         vn_async_vkCmdEndTransformFeedbackEXT(
            &venus->vn_ring, venus->command_buffer, 0, so_target_count,
            so_counter_buffers, so_counter_offsets);
      }
      vn_async_vkCmdEndRenderPass(&venus->vn_ring, venus->command_buffer);
   }

   yttrium_venus_trace_timing(
      YTTRIUM_TRACE_TIMING_VENUS_NATIVE_DRAW_RECORD,
      0, stage_start_us, NULL, resource_id,
      pipeline->pipeline_obj.id, vertex_count, index_count);

   bool submitted = false;
   if (native_draw_batch_candidate) {
      submitted = yttrium_venus_cmd_batch_after_native_draw_record(
         venus, "native pipeline draw batch");
   } else {
      submitted = yttrium_venus_cmd_batch_submit_after_record(
         venus, "native pipeline draw async");
   }
   venus->command_buffer = saved_command_buffer;
   if (!submitted) {
      yttrium_venus_trace_timing(
         YTTRIUM_TRACE_TIMING_VENUS_NATIVE_DRAW_TOTAL,
         1, total_start_us, NULL, resource_id,
         pipeline->pipeline_obj.id, vertex_count, index_count);
      return false;
   }
   yttrium_venus_trace_timing(
      YTTRIUM_TRACE_TIMING_VENUS_NATIVE_DRAW_TOTAL,
      0, total_start_us, NULL, resource_id,
      pipeline->pipeline_obj.id, vertex_count, index_count);

   for (uint32_t i = 0; i < so_target_count; i++)
      so_targets[i].counter_buffer_valid =
         so_targets[i].counter_resource != NULL;

   for (uint32_t i = 0; i < storage_image_count; i++) {
      if (storage_images[i].resource)
         storage_images[i].resource->contents_initialized = true;
   }

   for (uint32_t i = 0; i < color_resource_count; i++) {
      struct yttrium_venus_resource *color = color_resources[i];
      if (!color || color->buffer_backed)
         continue;

      yttrium_venus_mark_aspects_initialized(color,
                                             VK_IMAGE_ASPECT_COLOR_BIT);
      color->layout =
         pipeline->key.color_feedback_loop_mask & (1u << i) ?
            VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT :
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   }
   const struct yttrium_venus_sampled_image *first_sampled =
      sampled_image_count ? &sampled_images[0] : NULL;
   yttrium_trace_venus_draw(YTTRIUM_EVENT_VENUS_DRAW,
                             resource_id, resource->image_obj.id,
                             first_sampled ? first_sampled->resource_id : 0,
                             first_sampled && first_sampled->resource ?
                                (first_sampled->buffer ?
                                    first_sampled->resource->buffer_obj.id :
                                    first_sampled->resource->image_obj.id) :
                                0,
                             pipeline->pipeline_obj.id,
                             indexed ? index_count : vertex_count,
                            draw_state->viewports[0].x,
                            draw_state->viewports[0].y,
                            draw_state->viewports[0].width,
                            draw_state->viewports[0].height,
                            draw_state->scissors[0].offset.x,
                            draw_state->scissors[0].offset.y,
                            draw_state->scissors[0].extent.width,
                            draw_state->scissors[0].extent.height);
   YTTRIUM_LOG("yttrium: Venus native draw res_id=%u image_id=%llu depth_res_id=%u depth_image_id=%llu sampled_res_id=%u sampled_object_id=%llu sampled_buffer=%u pipeline_id=%llu count=%u instances=%u vertex_bytes=0x%llx vertex_bindings=%u indexed=%u index_count=%u index_bytes=0x%llx index_type=%u vertex_offset=%d ubos=%u sampled_count=%u sampled_expected=%u sampled_stages=0x%x viewports=%u viewport0=%f,%f %fx%f scissor0=%d,%d %ux%u topology=%u depth_test=%u depth_write=%u depth_compare=%u\n",
                 resource_id,
                 (unsigned long long)resource->image_obj.id,
                 has_depth ? depth_resource_id : 0,
                 has_depth && depth_resource ?
                    (unsigned long long)depth_resource->image_obj.id : 0,
                 first_sampled ? first_sampled->resource_id : 0,
                 first_sampled && first_sampled->resource ?
                    (unsigned long long)(first_sampled->buffer ?
                       first_sampled->resource->buffer_obj.id :
                       first_sampled->resource->image_obj.id) : 0,
                 first_sampled ? first_sampled->buffer : 0,
                 (unsigned long long)pipeline->pipeline_obj.id,
                 indexed ? index_count : vertex_count,
                 instance_count,
                 (unsigned long long)vertex_data_size,
                 vertex_upload_count,
                 indexed,
                 indexed ? index_count : 0,
                 indexed ? (unsigned long long)index_data_size : 0,
                 indexed ? index_type : 0,
                 indexed ? vertex_offset : 0,
                 ubo_upload_count,
                 sampled_image_count,
                 pipeline->key.sampled_binding_count,
                 pipeline->key.sampled_stage_mask,
                 draw_state->viewport_count,
                 draw_state->viewports[0].x,
                draw_state->viewports[0].y,
                draw_state->viewports[0].width,
                draw_state->viewports[0].height,
                draw_state->scissors[0].offset.x,
                draw_state->scissors[0].offset.y,
                draw_state->scissors[0].extent.width,
                draw_state->scissors[0].extent.height,
                draw_state->topology,
                draw_state->depth_test_enable,
                draw_state->depth_write_enable,
                draw_state->depth_compare_op);
   if (has_depth && depth_resource &&
       (draw_state->depth_write_enable || draw_state->stencil_test_enable))
      yttrium_venus_ds_clear_history_invalidate(depth_resource);
   return true;

fail_restore_command_buffer:
   yttrium_venus_abort_command_batch(venus, "native draw record failure");
   venus->command_buffer = saved_command_buffer;
   return false;
}

bool
yttrium_venus2_draw_vertex_buffer_vertices(struct yttrium_venus *venus,
                                          struct yttrium_venus_resource *resource,
                                          uint32_t resource_id,
                                          const struct yttrium_venus_triangle_vertex *vertices,
                                          uint32_t vertex_count,
                                          const struct yttrium_venus_draw_state *draw_state)
{
   if (vertex_count > YTTRIUM_VENUS_MAX_DRAW_VERTICES) {
      YTTRIUM_LOG("yttrium: Venus vertex-buffer draw vertex count %u exceeds limit %u\n",
                   vertex_count, YTTRIUM_VENUS_MAX_DRAW_VERTICES);
      return false;
   }

   return yttrium_venus_draw_graphics(venus, resource, resource_id,
                                      YTTRIUM_VENUS_GRAPHICS_VERTEX_BUFFER,
                                      vertices, vertex_count, draw_state);
}

static bool
yttrium_venus_clear_display_buffer(struct yttrium_venus *venus,
                                   struct yttrium_venus_resource *resource,
                                   uint32_t resource_id,
                                   enum pipe_format pipe_format,
                                   uint64_t allocation_size,
                                   const union pipe_color_union *color)
{
   uint32_t pattern = 0;
   if (!resource || !resource->initialized || !resource->buffer_backed ||
       !resource->buffer)
      return false;

   if (!yttrium_venus_clear_pattern(pipe_format, color, &pattern))
      return false;

   if (!yttrium_venus_begin_transfer_batch(venus))
      return false;

   if (!yttrium_venus_cmd_batch_track_resource(venus, resource)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "buffer clear tracking failure");
      return false;
   }

   vn_async_vkCmdFillBuffer(&venus->vn_ring, venus->command_buffer,
                            resource->buffer, 0, allocation_size, pattern);

   YTTRIUM_LOG("yttrium: Venus fill display buffer res_id=%u buffer_id=%llu pattern=0x%08x size=0x%llx\n",
                resource_id, (unsigned long long)resource->buffer_obj.id,
                pattern, (unsigned long long)allocation_size);
   return yttrium_venus_cmd_batch_after_record(venus, "buffer clear");
}

static bool
yttrium_venus_compute_push_descriptors_complete(
   struct yttrium_pipeline *pipeline,
   const VkWriteDescriptorSet *ubo_writes,
   uint32_t ubo_update_count,
   uint32_t sampled_image_count,
   uint32_t storage_update_count)
{
   if (ubo_update_count != pipeline->ubo_descriptor_count ||
       ubo_update_count != pipeline->ubo_count ||
       ubo_update_count > YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS ||
       sampled_image_count != pipeline->sampled_image_descriptor_count ||
       pipeline->sampled_buffer_descriptor_count ||
       storage_update_count != pipeline->storage_image_descriptor_count +
                                  pipeline->storage_buffer_descriptor_count)
      return false;

   /* Sampled bindings are checked in layout order and storage bindings have an
    * exact seen mask below. Counts alone do not exclude duplicate UBO writes;
    * require one write for each declared (binding, array element) as well. */
   uint32_t seen_ubos = 0;
   for (uint32_t i = 0; i < ubo_update_count; i++) {
      const VkWriteDescriptorSet *write = &ubo_writes[i];
      struct yttrium_venus_ubo_slot *slot =
         yttrium_venus_pipeline_find_ubo_slot(
            pipeline, write->dstBinding, write->dstArrayElement);
      if (!slot || write->descriptorCount != 1 ||
          write->descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
         return false;
      const uint32_t bit = 1u << (slot - pipeline->ubos);
      if (seen_ubos & bit)
         return false;
      seen_ubos |= bit;
   }
   return true;
}

bool
yttrium_venus2_dispatch_compute(
   struct yttrium_venus *venus,
   struct yttrium_pipeline *pipeline,
   const struct yttrium_venus_sampled_image *sampled_images,
   uint32_t sampled_image_count,
   const struct yttrium_venus_storage_image *storage_images,
   uint32_t storage_image_count,
   const struct yttrium_venus_ubo_upload *ubo_uploads,
   uint32_t ubo_upload_count,
   uint32_t grid_x,
   uint32_t grid_y,
   uint32_t grid_z)
{
   if (!venus || !pipeline || !pipeline->pipeline || !grid_x || !grid_y ||
       !grid_z)
      return false;

   if (sampled_image_count != pipeline->key.sampled_binding_count ||
       sampled_image_count > YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES ||
       (sampled_image_count && !sampled_images) || pipeline->has_sampled_buffer)
      return false;

   const bool has_descriptors =
      pipeline->ubo_descriptor_count ||
      pipeline->sampled_image_descriptor_count ||
      pipeline->storage_image_descriptor_count ||
      pipeline->storage_buffer_descriptor_count;
   const bool use_push_descriptors =
      has_descriptors &&
      yttrium_venus_push_descriptor_batch_enabled() &&
      pipeline->push_descriptor_set_layout &&
      pipeline->push_pipeline_layout &&
      pipeline->push_pipeline;

   if (!yttrium_venus_layout_pipeline_ubo_uploads(
          venus, pipeline, ubo_uploads, ubo_upload_count,
          true, false))
      return false;
   const VkDeviceSize expected_ubo_watermark =
      venus->cmd_batch_ubo_watermark;

   VkDescriptorSet dispatch_descriptor_set =
      use_push_descriptors ? VK_NULL_HANDLE : pipeline->descriptor_set;
   if (has_descriptors && !use_push_descriptors) {
      if (!yttrium_venus_cmd_batch_alloc_descriptor_set(
             venus, pipeline, &dispatch_descriptor_set))
         return false;
      if (ubo_upload_count &&
          venus->cmd_batch_ubo_watermark != expected_ubo_watermark &&
          !yttrium_venus_layout_pipeline_ubo_uploads(
             venus, pipeline, ubo_uploads, ubo_upload_count, true, false))
         return false;
   }

   VkDescriptorBufferInfo ubo_infos[YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS];
   VkWriteDescriptorSet ubo_writes[YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS];
   uint32_t ubo_update_count = 0;
   memset(ubo_infos, 0, sizeof(ubo_infos));
   memset(ubo_writes, 0, sizeof(ubo_writes));
   if (!yttrium_venus_update_pipeline_ubo_descriptors(
          venus, pipeline,
          dispatch_descriptor_set,
          ubo_uploads, ubo_upload_count, use_push_descriptors, ubo_infos,
          ubo_writes, &ubo_update_count))
      return false;

   struct yttrium_venus_sampled_image resolved_samples
      [YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   VkDescriptorImageInfo sampled_infos
      [YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   VkWriteDescriptorSet sampled_writes
      [YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   memset(resolved_samples, 0, sizeof(resolved_samples));
   memset(sampled_infos, 0, sizeof(sampled_infos));
   memset(sampled_writes, 0, sizeof(sampled_writes));
   for (uint32_t i = 0; i < sampled_image_count; i++) {
      const struct yttrium_venus_sampled_binding_layout *layout =
         &pipeline->key.sampled_bindings[i];
      struct yttrium_venus_sampled_image *sampled = &resolved_samples[i];
      *sampled = sampled_images[i];
      if (layout->stage != MESA_SHADER_COMPUTE || layout->buffer ||
          sampled->buffer || sampled->binding != layout->binding ||
          !pipeline->samplers[i])
         return false;
      if (!sampled->resource &&
          !yttrium_venus_ensure_null_sampled_image(
             venus, &sampled->resource, &sampled->resource_id))
         return false;

      const struct yttrium_venus_resource *resource = sampled->resource;
      if (!resource || !resource->initialized || resource->buffer_backed ||
          !resource->image ||
          !(resource->image_usage & VK_IMAGE_USAGE_SAMPLED_BIT))
         return false;

      /* The existing depth initializer covers only mip 0 / layer 0, but
       * layout and initialization state are tracked for the whole image.
       * Do not claim the remaining subresources were initialized by it.
       */
      if (yttrium_venus_format_has_depth(resource->vk_format) &&
          !resource->contents_initialized &&
          (resource->levels > 1 || resource->layers > 1)) {
         YTTRIUM_WARN("yttrium: compute dispatch rejected owner=yttrium_venus reason=uninitialized_depth_subresources_unsupported res_id=%u levels=%u layers=%u\n",
                      sampled->resource_id, resource->levels, resource->layers);
         return false;
      }

      VkImageView view = VK_NULL_HANDLE;
      if (!yttrium_venus_ensure_sample_image_view(
             venus, sampled->resource, sampled->resource_id,
             yttrium_venus2_pipe_format(sampled->format),
             sampled->swizzle_key, sampled->view_type,
             sampled->first_level, sampled->level_count,
             sampled->first_layer, sampled->layer_count,
             sampled->aspect_mask, &view))
         return false;

      /* GENERAL also permits non-overlapping SRV/UAV views of one image.
       * Keep their descriptors and the resource's tracked layout consistent.
       */
      sampled_infos[i] = (VkDescriptorImageInfo) {
         .sampler = pipeline->samplers[i],
         .imageView = view,
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      };
      sampled_writes[i] = (VkWriteDescriptorSet) {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = use_push_descriptors ? VK_NULL_HANDLE :
                      dispatch_descriptor_set,
         .dstBinding = sampled->binding,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &sampled_infos[i],
      };
   }
   if (sampled_image_count && !use_push_descriptors) {
      vn_async_vkUpdateDescriptorSets(&venus->vn_ring, venus->device_handle,
                                      sampled_image_count, sampled_writes,
                                      0, NULL);
   }

   VkDescriptorImageInfo storage_image_infos
      [YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES];
   VkBufferView storage_buffer_views[YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES];
   VkWriteDescriptorSet storage_writes[YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES];
   uint32_t storage_update_count = 0;
   memset(storage_image_infos, 0, sizeof(storage_image_infos));
   memset(storage_buffer_views, 0, sizeof(storage_buffer_views));
   memset(storage_writes, 0, sizeof(storage_writes));

   if (pipeline->has_storage_image || pipeline->has_storage_buffer) {
      const uint64_t storage_mask =
         pipeline->storage_image_mask | pipeline->storage_buffer_mask;
      if (!storage_images ||
          storage_image_count > YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES ||
          (!use_push_descriptors && !dispatch_descriptor_set))
         return false;

      uint64_t seen_mask = 0;
      for (uint32_t i = 0; i < storage_image_count; i++) {
         const struct yttrium_venus_storage_image *image =
            &storage_images[i];
         struct yttrium_venus_resource *image_resource = image->resource;
         const uint32_t raw_slot =
            image->binding >= YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE ?
            image->binding - YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE :
            UINT32_MAX;
         const uint64_t raw_mask =
            raw_slot < YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES ?
            (1ull << raw_slot) : 0;
         const bool storage_buffer =
            raw_mask && (pipeline->storage_buffer_mask & raw_mask);

         if (raw_slot >= YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES ||
             !(storage_mask & raw_mask) || (seen_mask & raw_mask) ||
             image->binding != yttrium_shader_storage_image_binding(raw_slot))
            return false;

         if (storage_buffer) {
            if (!image->buffer || !image_resource ||
                !image_resource->initialized ||
                !image_resource->buffer_backed || !image_resource->buffer)
               return false;

            VkBufferView buffer_view = VK_NULL_HANDLE;
            if (!yttrium_venus_ensure_storage_buffer_view(
                   venus, image_resource, image->resource_id, image->format,
                   image->buffer_offset, image->buffer_range, &buffer_view))
               return false;

            storage_buffer_views[storage_update_count] = buffer_view;
            storage_writes[storage_update_count] = (VkWriteDescriptorSet) {
               .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
               .dstSet = use_push_descriptors ? VK_NULL_HANDLE :
                         dispatch_descriptor_set,
               .dstBinding = image->binding,
               .descriptorCount = 1,
               .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
               .pTexelBufferView =
                  &storage_buffer_views[storage_update_count],
            };
         } else {
            if (!image_resource || !image_resource->initialized ||
                image_resource->buffer_backed || !image_resource->image)
               return false;

            VkImageView image_view = VK_NULL_HANDLE;
            if (!yttrium_venus_ensure_storage_image_view(
                   venus, image_resource, image->resource_id,
                   yttrium_venus2_pipe_format(image->format),
                   image->view_type, image->first_level, image->level_count,
                   image->first_layer, image->layer_count, image->aspect_mask,
                   &image_view))
               return false;

            storage_image_infos[storage_update_count] =
               (VkDescriptorImageInfo) {
               .imageView = image_view,
               .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };
            storage_writes[storage_update_count] = (VkWriteDescriptorSet) {
               .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
               .dstSet = use_push_descriptors ? VK_NULL_HANDLE :
                         dispatch_descriptor_set,
               .dstBinding = image->binding,
               .descriptorCount = 1,
               .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
               .pImageInfo = &storage_image_infos[storage_update_count],
            };
         }
         storage_update_count++;
         seen_mask |= raw_mask;
      }

      if (seen_mask != storage_mask)
         return false;

      if (!use_push_descriptors) {
         vn_async_vkUpdateDescriptorSets(&venus->vn_ring,
                                         venus->device_handle,
                                         storage_update_count,
                                         storage_writes, 0, NULL);
      }
   }

   /* Rotate only after proving a complete write of the declared layout.
    * Existing producers supply complete state; incomplete input retains the
    * original layout selection. */
   const bool complete_push_descriptors =
      use_push_descriptors && pipeline->push_pipeline_layout_alt &&
      yttrium_venus_push_descriptor_layout_rotation_enabled() &&
      yttrium_venus_compute_push_descriptors_complete(
         pipeline, ubo_writes, ubo_update_count,
         sampled_image_count, storage_update_count);

   if (!yttrium_venus_begin_command_batch(venus, "native compute dispatch",
                                          false, true))
      return false;

   if (!yttrium_venus_cmd_batch_track_pipeline(venus, pipeline)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "native compute dispatch pipeline tracking failure");
      return false;
   }
   for (uint32_t i = 0; i < sampled_image_count; i++) {
      if (!yttrium_venus_cmd_batch_track_resource(
             venus, resolved_samples[i].resource)) {
         yttrium_venus_cancel_command_batch_setup_failure(
            venus, "native compute dispatch sampled resource tracking failure");
         return false;
      }
   }
   for (uint32_t i = 0; i < ubo_upload_count; i++) {
      if (ubo_uploads[i].direct_resource &&
          !yttrium_venus_cmd_batch_track_resource(
             venus, ubo_uploads[i].direct_resource)) {
         yttrium_venus_cancel_command_batch_setup_failure(
            venus, "native compute dispatch direct ubo tracking failure");
         return false;
      }
   }
   for (uint32_t i = 0; i < storage_image_count; i++) {
      if (!yttrium_venus_cmd_batch_track_resource(
             venus, storage_images[i].resource)) {
         yttrium_venus_cancel_command_batch_setup_failure(
            venus, "native compute dispatch resource tracking failure");
         return false;
      }
   }

   if (pipeline->has_storage_buffer) {
      VkBufferMemoryBarrier storage_barriers
         [YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES];
      uint32_t storage_barrier_count = 0;
      memset(storage_barriers, 0, sizeof(storage_barriers));

      for (uint32_t i = 0; i < storage_image_count; i++) {
         const struct yttrium_venus_storage_image *storage =
            &storage_images[i];
         struct yttrium_venus_resource *storage_resource =
            storage->resource;

         if (!storage->buffer)
            continue;
         if (!storage_resource || !storage_resource->buffer ||
             storage_barrier_count >=
                YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES)
            goto fail_command_batch;

         if (storage->buffer_data && storage->buffer_size) {
            VkDeviceSize written_size = 0;
            const VkBufferMemoryBarrier overwrite = {
               .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
               .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT |
                                VK_ACCESS_MEMORY_WRITE_BIT,
               .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
               .buffer = storage_resource->buffer,
               .offset = 0,
               .size = storage->buffer_size,
            };
            vn_async_vkCmdPipelineBarrier(
               &venus->vn_ring, venus->command_buffer,
               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
               0, NULL, 1, &overwrite, 0, NULL);
            if (!yttrium_venus_cmd_update_buffer_padded(
                   venus, venus->command_buffer, storage_resource->buffer, 0,
                   storage->buffer_size, storage->buffer_data,
                   &written_size))
               goto fail_command_batch;
         }
         storage_barriers[storage_barrier_count++] =
               (VkBufferMemoryBarrier) {
                  .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                  .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT |
                                   VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
                  .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                   VK_ACCESS_SHADER_WRITE_BIT,
                  .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                  .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                  .buffer = storage_resource->buffer,
                  .offset = storage->buffer_offset,
                  .size = storage->buffer_range,
               };
         storage_resource->contents_initialized = true;
      }

      if (storage_barrier_count) {
         vn_async_vkCmdPipelineBarrier(&venus->vn_ring,
                                       venus->command_buffer,
                                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT |
                                       VK_PIPELINE_STAGE_HOST_BIT,
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       0, 0, NULL,
                                       storage_barrier_count,
                                       storage_barriers, 0, NULL);
      }
   }

   VkBufferMemoryBarrier ubo_barriers[YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS];
   memset(ubo_barriers, 0, sizeof(ubo_barriers));
   uint32_t ubo_barrier_count = 0;
   for (uint32_t i = 0; i < ubo_upload_count; i++) {
      const struct yttrium_venus_ubo_upload *upload = &ubo_uploads[i];
      struct yttrium_venus_ubo_slot *slot =
         yttrium_venus_pipeline_find_ubo_slot(pipeline, upload->binding,
                                              upload->array_element);
      if (!slot || !slot->buffer)
         goto fail_command_batch;

      const bool direct_resource = upload->direct_resource != NULL;
      if (!direct_resource && !slot->upload_reused) {
         yttrium_venus_cmd_update_buffer_chunks(venus, venus->command_buffer,
                                                slot->buffer, slot->offset,
                                                upload->size, upload->data);
         if (slot->resource_version_cacheable) {
            yttrium_venus_store_resource_ubo_version(
               upload->source_version_cache,
               venus->ubo_upload_buffer_generation,
               upload->source_contents_serial,
               upload->source_offset, (uint32_t)upload->size,
               slot->offset);
         }
      }
      ubo_barriers[ubo_barrier_count++] = (VkBufferMemoryBarrier) {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
         .srcAccessMask = direct_resource ? VK_ACCESS_HOST_WRITE_BIT :
                                            VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .buffer = slot->buffer,
         .offset = slot->offset,
         .size = upload->size,
      };
   }
   if (ubo_barrier_count) {
      vn_async_vkCmdPipelineBarrier(&venus->vn_ring, venus->command_buffer,
                                    VK_PIPELINE_STAGE_HOST_BIT |
                                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    0, 0, NULL,
                                    ubo_barrier_count, ubo_barriers,
                                    0, NULL);
   }

   for (uint32_t i = 0; i < sampled_image_count; i++) {
      struct yttrium_venus_sampled_image *sampled = &resolved_samples[i];
      struct yttrium_venus_resource *resource = sampled->resource;
      const bool depth = yttrium_venus_format_has_depth(resource->vk_format);
      if (depth ?
          !yttrium_venus_cmd_ensure_depth_initialized(
             venus, resource, sampled->resource_id) :
          !yttrium_venus_cmd_ensure_image_initialized(
             venus, resource, sampled->resource_id))
         goto fail_command_batch;

      /* Layout state is per resource, not per view. Transition all mips and
       * layers; the descriptor still exposes only the requested SRV range.
       */
      const VkImageSubresourceRange range = {
         .aspectMask = yttrium_venus_format_aspects(resource->vk_format),
         .baseMipLevel = 0,
         .levelCount = MAX2(resource->levels, 1),
         .baseArrayLayer = 0,
         .layerCount = MAX2(resource->layers, 1),
      };
      yttrium_venus_cmd_transition_image(venus, resource,
                                         VK_IMAGE_LAYOUT_GENERAL,
                                         VK_ACCESS_SHADER_READ_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         &range);
   }

   if (pipeline->has_storage_image) {
      for (uint32_t i = 0; i < storage_image_count; i++) {
         struct yttrium_venus_resource *image_resource =
            storage_images[i].resource;

         if (storage_images[i].buffer)
            continue;
         if (!image_resource)
            goto fail_command_batch;

         const VkImageSubresourceRange range = {
            .aspectMask = storage_images[i].aspect_mask ?
               storage_images[i].aspect_mask : VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = MAX2(image_resource->levels, 1),
            .baseArrayLayer = 0,
            .layerCount = MAX2(image_resource->layers, 1),
         };

         if (!yttrium_venus_cmd_ensure_image_initialized(
                venus, image_resource, storage_images[i].resource_id))
            goto fail_command_batch;
         yttrium_venus_cmd_transition_image(venus, image_resource,
                                            VK_IMAGE_LAYOUT_GENERAL,
                                            VK_ACCESS_SHADER_READ_BIT |
                                            VK_ACCESS_SHADER_WRITE_BIT,
                                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                            &range);
      }
   }

   const VkPipeline compute_pipeline =
      use_push_descriptors ? pipeline->push_pipeline : pipeline->pipeline;
   const struct yttrium_venus_object *compute_pipeline_layout_obj =
      yttrium_venus_select_pipeline_layout(
         &venus->compute_push_descriptor_layout_index, pipeline,
         use_push_descriptors,
         complete_push_descriptors ?
            ubo_update_count + sampled_image_count + storage_update_count : 0);
   const VkPipelineLayout compute_pipeline_layout =
      YTTRIUM_VENUS_HANDLE(VkPipelineLayout, compute_pipeline_layout_obj);
   vn_async_vkCmdBindPipeline(&venus->vn_ring, venus->command_buffer,
                              VK_PIPELINE_BIND_POINT_COMPUTE,
                              compute_pipeline);
   if (use_push_descriptors) {
      VkWriteDescriptorSet push_writes
         [YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS +
          YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES +
          YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES];
      uint32_t push_write_count = 0;

      for (uint32_t i = 0; i < ubo_update_count; i++)
         push_writes[push_write_count++] = ubo_writes[i];
      for (uint32_t i = 0; i < sampled_image_count; i++)
         push_writes[push_write_count++] = sampled_writes[i];
      for (uint32_t i = 0; i < storage_update_count; i++)
         push_writes[push_write_count++] = storage_writes[i];

      if (push_write_count) {
         vn_async_vkCmdPushDescriptorSet(&venus->vn_ring,
                                         venus->command_buffer,
                                         VK_PIPELINE_BIND_POINT_COMPUTE,
                                         compute_pipeline_layout, 0,
                                         push_write_count,
                                         push_writes);
      }
   } else if (dispatch_descriptor_set) {
      vn_async_vkCmdBindDescriptorSets(&venus->vn_ring,
                                       venus->command_buffer,
                                       VK_PIPELINE_BIND_POINT_COMPUTE,
                                       compute_pipeline_layout, 0, 1,
                                       &dispatch_descriptor_set, 0, NULL);
   }
   vn_async_vkCmdDispatch(&venus->vn_ring, venus->command_buffer,
                          grid_x, grid_y, grid_z);

   for (uint32_t i = 0; i < storage_image_count; i++) {
      if (storage_images[i].resource)
         storage_images[i].resource->contents_initialized = true;
   }

   YTTRIUM_LOG("yttrium: Venus native compute dispatch pipeline_id=%llu grid=%ux%ux%u storage_count=%u ubo_count=%u\n",
               (unsigned long long)pipeline->pipeline_obj.id,
               grid_x, grid_y, grid_z, storage_image_count, ubo_upload_count);
   return yttrium_venus_cmd_batch_after_record(venus,
                                               "native compute dispatch");

fail_command_batch:
   yttrium_venus_cancel_command_batch_setup_failure(
      venus, "native compute dispatch failure");
   return false;
}


static bool
yttrium_venus_cmd_clear_depth_stencil_attachment(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   uint32_t resource_id,
   VkImageAspectFlags aspects,
   double depth,
   unsigned stencil,
   uint32_t level,
   uint32_t layer,
   uint32_t dstx,
   uint32_t dsty,
   uint32_t width,
   uint32_t height);

static bool
yttrium_venus_cmd_clear_color_attachment(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   uint32_t resource_id,
   enum pipe_format pipe_format,
   const union pipe_color_union *color,
   uint32_t level,
   uint32_t layer,
   uint32_t dstx,
   uint32_t dsty,
   uint32_t width,
   uint32_t height);

static VkClearColorValue
yttrium_venus_clear_color_value(enum pipe_format pipe_format,
                                const union pipe_color_union *color)
{
   VkClearColorValue clear;
   memset(&clear, 0, sizeof(clear));

   if (util_format_is_pure_sint(pipe_format)) {
      memcpy(clear.int32, color->i, sizeof(clear.int32));
   } else if (util_format_is_pure_uint(pipe_format)) {
      memcpy(clear.uint32, color->ui, sizeof(clear.uint32));
   } else {
      memcpy(clear.float32, color->f, sizeof(clear.float32));
   }

   return clear;
}

static bool
yttrium_venus_cmd_clear_depth_stencil_image(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   uint32_t resource_id,
   VkImageAspectFlags aspects,
   double depth,
   unsigned stencil,
   uint32_t level,
   uint32_t first_layer,
   uint32_t layer_count);

bool
yttrium_venus2_clear_depth_stencil(struct yttrium_venus *venus,
                                  struct yttrium_venus_resource *resource,
                                  uint32_t resource_id,
                                  unsigned clear_flags,
                                  double depth,
                                  unsigned stencil,
                                  uint32_t level,
                                  uint32_t first_layer,
                                  uint32_t layer_count,
                                  uint32_t dstx,
                                  uint32_t dsty,
                                  uint32_t width,
                                  uint32_t height)
{
   if (!yttrium_venus_ensure_initialized(venus))
      return false;

   if (!resource || !resource->initialized || resource->buffer_backed ||
       !resource->image ||
       !(resource->image_usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
       !yttrium_venus_format_has_depth(resource->vk_format) ||
       !yttrium_venus_valid_image_subresource(resource, level, first_layer,
                                              layer_count)) {
      YTTRIUM_LOG("yttrium: Venus depth clear rejected res_id=%u initialized=%u buffer_backed=%u image=0x%llx usage=0x%x format=%u flags=0x%x level=%u first_layer=%u layer_count=%u levels=%u layers=%u\n",
                  resource_id,
                  resource ? resource->initialized : 0,
                  resource ? resource->buffer_backed : 0,
                  (unsigned long long)(resource ?
                     YTTRIUM_VENUS_HANDLE_TO_U64(resource->image) : 0),
                  resource ? resource->image_usage : 0,
                  resource ? resource->vk_format : VK_FORMAT_UNDEFINED,
                  clear_flags,
                  level, first_layer, layer_count,
                  resource ? resource->levels : 0,
                  resource ? resource->layers : 0);
      return false;
   }

   VkImageAspectFlags aspects = 0;
   if (clear_flags & PIPE_CLEAR_DEPTH)
      aspects |= VK_IMAGE_ASPECT_DEPTH_BIT;
   if ((clear_flags & PIPE_CLEAR_STENCIL) &&
       yttrium_venus_format_has_stencil(resource->vk_format))
      aspects |= VK_IMAGE_ASPECT_STENCIL_BIT;
   if (!aspects)
      return true;

   const uint32_t level_width = yttrium_venus_subresource_width(resource,
                                                               level);
   const uint32_t level_height = yttrium_venus_subresource_height(resource,
                                                                 level);
   if (!width || !height || dstx >= level_width || dsty >= level_height ||
       width > level_width - dstx || height > level_height - dsty)
      return false;

   const bool full_clear =
      dstx == 0 && dsty == 0 && width == level_width &&
      height == level_height;

   if (!yttrium_venus_begin_transfer_batch(venus))
      return false;

   if (!yttrium_venus_cmd_batch_track_resource(venus, resource)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "depth clear tracking failure");
      return false;
   }

   if (!resource->contents_initialized && !full_clear &&
       !yttrium_venus_cmd_ensure_depth_initialized(venus, resource,
                                                   resource_id)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "depth clear init failure");
      return false;
   }

   if (full_clear) {
      if (!yttrium_venus_cmd_clear_depth_stencil_image(
             venus, resource, resource_id, aspects, depth, stencil,
             level, first_layer, layer_count)) {
         yttrium_venus_cancel_command_batch_setup_failure(
            venus, "depth clear image setup failure");
         return false;
      }
   } else {
      for (uint32_t i = 0; i < layer_count; i++) {
         if (!yttrium_venus_cmd_clear_depth_stencil_attachment(
                venus, resource, resource_id, aspects, depth, stencil,
                level, first_layer + i, dstx, dsty, width, height)) {
            yttrium_venus_cancel_command_batch_setup_failure(
               venus, "depth clear attachment setup failure");
            return false;
         }
      }
   }

   yttrium_venus_mark_aspects_initialized(resource, aspects);
   for (uint32_t i = 0; i < layer_count; i++) {
      yttrium_venus_ds_clear_history_note(resource, aspects, depth, stencil,
                                          level, first_layer + i, dstx, dsty,
                                          width, height);
   }
   YTTRIUM_LOG("yttrium: Venus clear depth/stencil res_id=%u image_id=%llu aspects=0x%x depth=%f stencil=%u level=%u layers=%u+%u\n",
               resource_id,
               (unsigned long long)resource->image_obj.id,
               aspects, (float)depth, stencil,
               level, first_layer, layer_count);
   return yttrium_venus_cmd_batch_after_record(venus, "depth clear");
}

bool
yttrium_venus2_clear_display(struct yttrium_venus *venus,
                            struct gdikmt_context *ctx,
                            struct yttrium_venus_resource *resource,
                            uint32_t resource_id,
                            uint32_t width,
                            uint32_t height,
                            enum pipe_format pipe_format,
                            uint64_t allocation_size,
                            const union pipe_color_union *color,
                            uint32_t level,
                            uint32_t first_layer,
                            uint32_t layer_count)
{
   (void)ctx;
   (void)width;
   (void)height;

   if (!yttrium_venus_ensure_initialized(venus))
      return false;

   if (yttrium_venus_clear_display_buffer(venus, resource, resource_id,
                                          pipe_format, allocation_size,
                                          color))
      return true;

   if (!resource || !resource->initialized || resource->buffer_backed ||
       !resource->image ||
       !yttrium_venus_valid_render_subresource(resource, level, first_layer,
                                               layer_count)) {
      YTTRIUM_LOG("yttrium: Venus display resource is not image-backed or subresource invalid res_id=%u initialized=%u buffer_backed=%u level=%u first_layer=%u layer_count=%u levels=%u layers=%u\n",
                   resource_id, resource ? resource->initialized : 0,
                   resource ? resource->buffer_backed : 0,
                   level, first_layer, layer_count,
                   resource ? resource->levels : 0,
                   resource ? resource->layers : 0);
      return false;
   }

   const bool is_3d = yttrium_venus_resource_is_3d(resource);
   const uint32_t full_layer_count = is_3d ?
      yttrium_venus_subresource_depth(resource, level) :
      MAX2(resource->layers, 1);
   if (is_3d && (first_layer || layer_count != full_layer_count))
      return false;

   if (!yttrium_venus_begin_transfer_batch(venus))
      return false;

   if (!yttrium_venus_cmd_batch_track_resource(venus, resource)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "image clear tracking failure");
      return false;
   }

   const VkImageSubresourceRange range =
      yttrium_venus_render_barrier_range(resource, VK_IMAGE_ASPECT_COLOR_BIT,
                                         level, first_layer, layer_count);

   if (!resource->contents_initialized &&
       !(level == 0 && first_layer == 0 &&
         layer_count == full_layer_count &&
         MAX2(resource->levels, 1) == 1)) {
      if (!yttrium_venus_cmd_ensure_image_initialized(venus, resource,
                                                      resource_id)) {
         yttrium_venus_cancel_command_batch_setup_failure(
            venus, "image clear init failure");
         return false;
      }
   }

   yttrium_venus_cmd_transition_image(venus, resource,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      VK_ACCESS_TRANSFER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      &range);

   const VkClearColorValue clear =
      yttrium_venus_clear_color_value(pipe_format, color);
   vn_async_vkCmdClearColorImage(&venus->vn_ring, venus->command_buffer,
                                 resource->image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &clear, 1, &range);

   resource->contents_initialized = true;
   YTTRIUM_LOG("yttrium: Venus clear display res_id=%u image_id=%llu level=%u layers=%u+%u color_f=%f,%f,%f,%f color_ui=%u,%u,%u,%u color_i=%d,%d,%d,%d\n",
                resource_id, (unsigned long long)resource->image_obj.id,
                level, first_layer, layer_count,
                color->f[0], color->f[1], color->f[2], color->f[3],
                color->ui[0], color->ui[1], color->ui[2], color->ui[3],
                color->i[0], color->i[1], color->i[2], color->i[3]);
   return yttrium_venus_cmd_batch_after_record(venus, "image clear");
}

bool
yttrium_venus2_clear_display_rect(struct yttrium_venus *venus,
                                  struct gdikmt_context *ctx,
                                  struct yttrium_venus_resource *resource,
                                  uint32_t resource_id,
                                  enum pipe_format pipe_format,
                                  const union pipe_color_union *color,
                                  uint32_t level,
                                  uint32_t first_layer,
                                  uint32_t layer_count,
                                  uint32_t dstx,
                                  uint32_t dsty,
                                  uint32_t width,
                                  uint32_t height)
{
   (void)ctx;

   if (!yttrium_venus_ensure_initialized(venus))
      return false;

   if (!resource || !resource->initialized || resource->buffer_backed ||
       !resource->image ||
       !(resource->image_usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ||
       !yttrium_venus_valid_render_subresource(resource, level, first_layer,
                                               layer_count)) {
      YTTRIUM_LOG("yttrium: Venus color rect clear rejected res_id=%u initialized=%u buffer_backed=%u image=0x%llx usage=0x%x level=%u layers=%u+%u\n",
                  resource_id,
                  resource ? resource->initialized : 0,
                  resource ? resource->buffer_backed : 0,
                  resource ? (unsigned long long)resource->image_obj.id : 0,
                  resource ? resource->image_usage : 0,
                  level, first_layer, layer_count);
      return false;
   }

   const uint32_t level_width =
      yttrium_venus_subresource_width(resource, level);
   const uint32_t level_height =
      yttrium_venus_subresource_height(resource, level);
   if (!level_width || !level_height ||
       dstx >= level_width || dsty >= level_height)
      return false;

   width = MIN2(width, level_width - dstx);
   height = MIN2(height, level_height - dsty);
   if (!width || !height)
      return false;

   if (!yttrium_venus_begin_transfer_batch(venus))
      return false;

   if (!yttrium_venus_cmd_batch_track_resource(venus, resource)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "color rect clear tracking failure");
      return false;
   }

   const bool full_clear =
      dstx == 0 && dsty == 0 &&
      width == level_width && height == level_height;
   if (!resource->contents_initialized && !full_clear &&
       !yttrium_venus_cmd_ensure_image_initialized(venus, resource,
                                                   resource_id)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "color rect clear init failure");
      return false;
   }

   for (uint32_t i = 0; i < layer_count; i++) {
      if (!yttrium_venus_cmd_clear_color_attachment(
             venus, resource, resource_id, pipe_format, color, level,
             first_layer + i, dstx, dsty, width, height)) {
         yttrium_venus_cancel_command_batch_setup_failure(
            venus, "color rect clear attachment setup failure");
         return false;
      }
   }

   resource->contents_initialized = true;
   YTTRIUM_LOG("yttrium: Venus color rect clear res_id=%u image_id=%llu level=%u layers=%u+%u rect=%u,%u %ux%u color_f=%f,%f,%f,%f color_ui=%u,%u,%u,%u color_i=%d,%d,%d,%d\n",
               resource_id,
               (unsigned long long)resource->image_obj.id,
               level, first_layer, layer_count,
               dstx, dsty, width, height,
               color->f[0], color->f[1], color->f[2], color->f[3],
               color->ui[0], color->ui[1], color->ui[2], color->ui[3],
               color->i[0], color->i[1], color->i[2], color->i[3]);
   return yttrium_venus_cmd_batch_after_record(venus, "color rect clear");
}

bool
yttrium_venus2_copy_image_to_display_buffer(struct yttrium_venus *venus,
                                           struct yttrium_venus_resource *render,
                                           struct yttrium_venus_resource *scanout,
                                           uint32_t render_resource_id,
                                           uint32_t scanout_resource_id,
                                           uint32_t width,
                                           uint32_t height,
                                           enum pipe_format pipe_format,
                                           uint32_t scanout_stride)
{
   return yttrium_venus2_copy_image_region_to_display_buffer(
      venus, render, scanout, render_resource_id, scanout_resource_id,
      0, 0, 0, 0, 0, 0, width, height, pipe_format, scanout_stride);
}

bool
yttrium_venus2_copy_image_region_to_display_buffer(struct yttrium_venus *venus,
                                                  struct yttrium_venus_resource *render,
                                                  struct yttrium_venus_resource *scanout,
                                                  uint32_t render_resource_id,
                                                  uint32_t scanout_resource_id,
                                                  uint32_t src_level,
                                                  uint32_t src_layer,
                                                  uint32_t src_x,
                                                  uint32_t src_y,
                                                  uint32_t dst_x,
                                                  uint32_t dst_y,
                                                  uint32_t width,
                                                  uint32_t height,
                                                  enum pipe_format pipe_format,
                                                  uint32_t scanout_stride)
{
   return yttrium_venus2_copy_image_region_aspect_to_display_buffer(
      venus, render, scanout, render_resource_id, scanout_resource_id,
      src_level, src_layer, src_x, src_y, dst_x, dst_y, width, height,
      pipe_format, scanout_stride, 0);
}

bool
yttrium_venus2_copy_image_region_aspect_to_display_buffer(struct yttrium_venus *venus,
                                                         struct yttrium_venus_resource *render,
                                                         struct yttrium_venus_resource *scanout,
                                                         uint32_t render_resource_id,
                                                         uint32_t scanout_resource_id,
                                                         uint32_t src_level,
                                                         uint32_t src_layer,
                                                         uint32_t src_x,
                                                         uint32_t src_y,
                                                         uint32_t dst_x,
                                                         uint32_t dst_y,
                                                         uint32_t width,
                                                         uint32_t height,
                                                         enum pipe_format pipe_format,
                                                         uint32_t scanout_stride,
                                                         VkImageAspectFlags copy_aspect)
{
   if (!render || !render->initialized || render->buffer_backed ||
       !render->image || !scanout || !scanout->initialized ||
       !scanout->buffer_backed || !scanout->buffer)
      return false;
   if (!(scanout->buffer_usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT))
      return false;
   if (!(render->image_usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
      return false;
   const bool src_is_3d = render->layers == 1 && render->depth > 1;
   const uint32_t src_base_layer = src_is_3d ? 0 : src_layer;
   const uint32_t src_z = src_is_3d ? src_layer : 0;
   if (!yttrium_venus_valid_image_subresource(render, src_level,
                                              src_base_layer, 1))
      return false;

   const unsigned cpp = util_format_get_blocksize(pipe_format);
   if (!cpp || !width || !height || scanout_stride < width * cpp ||
       scanout_stride % cpp)
      return false;
   const uint32_t src_level_width =
      yttrium_venus_subresource_width(render, src_level);
   const uint32_t src_level_height =
      yttrium_venus_subresource_height(render, src_level);
   const uint32_t src_level_depth =
      yttrium_venus_subresource_depth(render, src_level);
   if (src_x > src_level_width || src_y > src_level_height ||
       width > src_level_width - src_x ||
       height > src_level_height - src_y)
      return false;
   if (src_is_3d && src_z >= src_level_depth)
      return false;

   const VkDeviceSize buffer_offset =
      (VkDeviceSize)dst_y * scanout_stride + (VkDeviceSize)dst_x * cpp;
   const VkDeviceSize last_byte =
      buffer_offset + (VkDeviceSize)(height - 1) * scanout_stride +
      (VkDeviceSize)width * cpp;
   if (last_byte > scanout->allocation_size)
      return false;

   const VkImageAspectFlags aspects = yttrium_venus_format_aspects(
      render->vk_format);
   if (!copy_aspect) {
      if (!(aspects & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT)))
         return false;
      copy_aspect =
         (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT :
                                                 VK_IMAGE_ASPECT_COLOR_BIT;
   } else if ((copy_aspect & ~aspects) ||
              (copy_aspect & (copy_aspect - 1))) {
      return false;
   }

   if (!yttrium_venus_begin_transfer_batch(venus))
      return false;

   if (!yttrium_venus_cmd_batch_track_resource(venus, render) ||
       !yttrium_venus_cmd_batch_track_resource(venus, scanout)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "image-to-buffer tracking failure");
      return false;
   }

   const VkImageSubresourceRange range = {
      .aspectMask = copy_aspect,
      .baseMipLevel = src_level,
      .levelCount = 1,
      .baseArrayLayer = src_base_layer,
      .layerCount = 1,
   };
   if (!yttrium_venus_cmd_ensure_image_initialized(venus, render,
                                                   render_resource_id)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "image-to-buffer source init failure");
      return false;
   }
   yttrium_venus_cmd_transition_image(venus, render,
                                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                      VK_ACCESS_TRANSFER_READ_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      &range);

   /* The destination can also be a GPU indirect/counter buffer, not only a
    * readback staging buffer.  Order the overwritten range after shader,
    * indirect and host accesses as well as preceding transfers. */
   const VkBufferMemoryBarrier overwrite_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                       VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = scanout->buffer,
      .offset = buffer_offset,
      .size = last_byte - buffer_offset,
   };
   vn_async_vkCmdPipelineBarrier(&venus->vn_ring, venus->command_buffer,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                 0, NULL, 1, &overwrite_barrier, 0, NULL);

   const VkBufferImageCopy copy_region = {
      .bufferOffset = buffer_offset,
      .bufferRowLength = scanout_stride / cpp,
      .bufferImageHeight = 0,
      .imageSubresource = {
         .aspectMask = copy_aspect,
         .mipLevel = src_level,
         .baseArrayLayer = src_base_layer,
         .layerCount = 1,
      },
      .imageOffset = { (int32_t)src_x, (int32_t)src_y, (int32_t)src_z },
      .imageExtent = { width, height, 1 },
   };
   vn_async_vkCmdCopyImageToBuffer(&venus->vn_ring, venus->command_buffer,
                                   render->image,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   scanout->buffer, 1, &copy_region);
   {
      const VkBufferMemoryBarrier host_read_barrier = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                          VK_ACCESS_HOST_READ_BIT,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .buffer = scanout->buffer,
         .offset = buffer_offset,
         .size = last_byte - buffer_offset,
      };
      vn_async_vkCmdPipelineBarrier(
         &venus->vn_ring, venus->command_buffer,
         VK_PIPELINE_STAGE_TRANSFER_BIT,
         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0,
         0, NULL, 1, &host_read_barrier, 0, NULL);
   }
   yttrium_trace_venus_upload(
      YTTRIUM_TRACE_VENUS_UPLOAD_IMAGE_TO_BUFFER, 0,
      (uint64_t)scanout_stride * height, render->image_obj.id,
      scanout->buffer_obj.id, render_resource_id, scanout_resource_id,
      width, height, 1, scanout_stride, 0);

   render->layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   YTTRIUM_LOG("yttrium: Venus copied render image res_id=%u image_id=%llu to scanout buffer res_id=%u buffer_id=%llu src_level=%u src_layer=%u src=%u,%u dst=%u,%u %ux%u stride=%u row_length=%u offset=0x%llx\n",
                render_resource_id,
                (unsigned long long)render->image_obj.id,
                scanout_resource_id,
                (unsigned long long)scanout->buffer_obj.id,
                src_level, src_layer, src_x, src_y, dst_x, dst_y,
                width, height, scanout_stride, scanout_stride / cpp,
                (unsigned long long)buffer_offset);
   return yttrium_venus_cmd_batch_after_record(venus,
                                               "image-to-buffer copy");
}

bool
yttrium_venus2_copy_buffer_to_buffer(struct yttrium_venus *venus,
                                    struct yttrium_venus_resource *src,
                                    struct yttrium_venus_resource *dst,
                                    VkDeviceSize src_offset,
                                    VkDeviceSize dst_offset,
                                    VkDeviceSize size)
{
   if (!venus || !src || !dst || !size ||
       !src->initialized || !src->buffer_backed || !src->buffer ||
       !dst->initialized || !dst->buffer_backed || !dst->buffer)
      return false;
   if (!(src->buffer_usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) ||
       !(dst->buffer_usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT))
      return false;
   if (src_offset > src->allocation_size ||
       size > src->allocation_size - src_offset ||
       dst_offset > dst->allocation_size ||
       size > dst->allocation_size - dst_offset)
      return false;

   if (!yttrium_venus_begin_transfer_batch(venus))
      return false;

   if (!yttrium_venus_cmd_batch_track_resource(venus, src) ||
       !yttrium_venus_cmd_batch_track_resource(venus, dst)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "buffer copy tracking failure");
      return false;
   }

   const VkBufferMemoryBarrier src_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = src->buffer,
      .offset = src_offset,
      .size = size,
   };
   vn_async_vkCmdPipelineBarrier(
      &venus->vn_ring, venus->command_buffer,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
      0, NULL, 1, &src_barrier, 0, NULL);

   const VkBufferMemoryBarrier overwrite_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                       VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = dst->buffer,
      .offset = dst_offset,
      .size = size,
   };
   vn_async_vkCmdPipelineBarrier(
      &venus->vn_ring, venus->command_buffer,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
      0, NULL, 1, &overwrite_barrier, 0, NULL);

   const VkBufferCopy copy_region = {
      .srcOffset = src_offset,
      .dstOffset = dst_offset,
      .size = size,
   };
   vn_async_vkCmdCopyBuffer(&venus->vn_ring, venus->command_buffer,
                            src->buffer, dst->buffer, 1, &copy_region);
   yttrium_trace_venus_upload(
      YTTRIUM_TRACE_VENUS_UPLOAD_BUFFER_COPY, 0, size,
      src->buffer_obj.id, dst->buffer_obj.id, 0, 0, 0, 0, 0, 0, 0);

   const VkBufferMemoryBarrier dst_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                       VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = dst->buffer,
      .offset = dst_offset,
      .size = size,
   };
   vn_async_vkCmdPipelineBarrier(&venus->vn_ring, venus->command_buffer,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT, 0,
                                 0, NULL, 1, &dst_barrier, 0, NULL);

   return yttrium_venus_cmd_batch_after_record(venus, "buffer copy");
}

static bool
yttrium_venus_cmd_clear_depth_stencil_image(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   uint32_t resource_id,
   VkImageAspectFlags aspects,
   double depth,
   unsigned stencil,
   uint32_t level,
   uint32_t first_layer,
   uint32_t layer_count)
{
   const VkImageAspectFlags format_aspects =
      yttrium_venus_format_aspects(resource->vk_format) &
      (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
   aspects &= format_aspects;
   if (!aspects || !layer_count)
      return false;

   const VkImageSubresourceRange range = {
      .aspectMask = aspects,
      .baseMipLevel = level,
      .levelCount = 1,
      .baseArrayLayer = first_layer,
      .layerCount = layer_count,
   };

   yttrium_venus_cmd_transition_image(venus, resource,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      VK_ACCESS_TRANSFER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      &range);

   const VkClearDepthStencilValue clear_value = {
      .depth = (float)depth,
      .stencil = stencil,
   };
   vn_async_vkCmdClearDepthStencilImage(&venus->vn_ring,
                                        venus->command_buffer,
                                        resource->image,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        &clear_value, 1, &range);

   YTTRIUM_LOG("yttrium: Venus image-cleared depth/stencil res_id=%u image_id=%llu aspects=0x%x depth=%f stencil=%u level=%u layers=%u+%u\n",
               resource_id,
               (unsigned long long)resource->image_obj.id,
               aspects, (float)depth, stencil,
               level, first_layer, layer_count);
   return true;
}

static bool
yttrium_venus_cmd_clear_depth_stencil_attachment(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   uint32_t resource_id,
   VkImageAspectFlags aspects,
   double depth,
   unsigned stencil,
   uint32_t level,
   uint32_t layer,
   uint32_t dstx,
   uint32_t dsty,
   uint32_t width,
   uint32_t height)
{
   const uint32_t level_width =
      yttrium_venus_subresource_width(resource, level);
   const uint32_t level_height =
      yttrium_venus_subresource_height(resource, level);
   const VkSampleCountFlagBits samples =
      resource->samples ? resource->samples : VK_SAMPLE_COUNT_1_BIT;
   const bool clear_depth = (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
   const bool clear_stencil = (aspects & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
   const bool has_stencil =
      yttrium_venus_format_has_stencil(resource->vk_format);
   const bool full_clear =
      dstx == 0 && dsty == 0 && width == level_width &&
      height == level_height;

   if (!level_width || !level_height || !aspects)
      return false;

   uint32_t transient = UINT32_MAX;
   if (!yttrium_venus_cmd_batch_alloc_transient(venus, &transient))
      return false;

   VkImageView view = venus->cmd_batch_transients[transient].view;
   VkRenderPass render_pass =
      venus->cmd_batch_transients[transient].render_pass;
   VkFramebuffer framebuffer =
      venus->cmd_batch_transients[transient].framebuffer;

   const VkImageSubresourceRange range = {
      .aspectMask = yttrium_venus_format_aspects(resource->vk_format) &
                    (VK_IMAGE_ASPECT_DEPTH_BIT |
                     VK_IMAGE_ASPECT_STENCIL_BIT),
      .baseMipLevel = level,
      .levelCount = 1,
      .baseArrayLayer = layer,
      .layerCount = 1,
   };
   yttrium_venus_cmd_transition_image(
      venus, resource, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
      &range);

   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = resource->image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = resource->vk_format,
      .subresourceRange = range,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateImageView,
      &venus->cmd_batch_transients[transient].view_obj, &view,
      goto depth_clear_submit_failed, venus->device_handle, &view_info, NULL,
      &view);
   venus->cmd_batch_transients[transient].view_created = true;

   const VkAttachmentDescription attachment = {
      .format = resource->vk_format,
      .samples = samples,
      .loadOp = (full_clear && clear_depth) ?
         VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = has_stencil ?
         ((full_clear && clear_stencil) ?
             VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_LOAD) :
         VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = has_stencil ? VK_ATTACHMENT_STORE_OP_STORE :
                        VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
   };
   const VkAttachmentReference depth_ref = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
   };
   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .pDepthStencilAttachment = &depth_ref,
   };
   const VkRenderPassCreateInfo render_pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &attachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateRenderPass,
      &venus->cmd_batch_transients[transient].render_pass_obj, &render_pass,
      goto depth_clear_submit_failed, venus->device_handle,
      &render_pass_info, NULL, &render_pass);
   venus->cmd_batch_transients[transient].render_pass_created = true;

   const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount = 1,
      .pAttachments = &view,
      .width = level_width,
      .height = level_height,
      .layers = 1,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateFramebuffer,
      &venus->cmd_batch_transients[transient].framebuffer_obj, &framebuffer,
      goto depth_clear_submit_failed, venus->device_handle,
      &framebuffer_info, NULL, &framebuffer);
   venus->cmd_batch_transients[transient].framebuffer_created = true;

   const VkRenderPassBeginInfo render_pass_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = {
         .offset = { 0, 0 },
         .extent = { level_width, level_height },
      },
   };
   vn_async_vkCmdBeginRenderPass(&venus->vn_ring, venus->command_buffer,
                                 &render_pass_begin,
                                 VK_SUBPASS_CONTENTS_INLINE);
   const VkClearAttachment clear_attachment = {
      .aspectMask = aspects,
      .clearValue = {
         .depthStencil = {
            .depth = (float)depth,
            .stencil = stencil,
         },
      },
   };
   const VkClearRect clear_rect = {
      .rect = {
         .offset = { (int32_t)dstx, (int32_t)dsty },
         .extent = { width, height },
      },
      .baseArrayLayer = 0,
      .layerCount = 1,
   };
   vn_async_vkCmdClearAttachments(&venus->vn_ring, venus->command_buffer,
                                  1, &clear_attachment, 1, &clear_rect);
   vn_async_vkCmdEndRenderPass(&venus->vn_ring, venus->command_buffer);

   YTTRIUM_LOG("yttrium: Venus attachment-cleared depth/stencil res_id=%u image_id=%llu aspects=0x%x depth=%f stencil=%u level=%u layer=%u rect=%u,%u %ux%u\n",
               resource_id,
               (unsigned long long)resource->image_obj.id,
               aspects, (float)depth, stencil,
               level, layer, dstx, dsty, width, height);
   return true;

depth_clear_submit_failed:
   yttrium_venus_cmd_batch_release_transient(venus, transient);
   return false;
}

static bool
yttrium_venus_cmd_clear_color_attachment(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *resource,
   uint32_t resource_id,
   enum pipe_format pipe_format,
   const union pipe_color_union *color,
   uint32_t level,
   uint32_t layer,
   uint32_t dstx,
   uint32_t dsty,
   uint32_t width,
   uint32_t height)
{
   const uint32_t level_width =
      yttrium_venus_subresource_width(resource, level);
   const uint32_t level_height =
      yttrium_venus_subresource_height(resource, level);
   const VkSampleCountFlagBits samples =
      resource->samples ? resource->samples : VK_SAMPLE_COUNT_1_BIT;
   const bool full_clear =
      dstx == 0 && dsty == 0 && width == level_width &&
      height == level_height;
   VkFormat view_format = yttrium_venus2_pipe_format(pipe_format);
   if (view_format == VK_FORMAT_UNDEFINED)
      view_format = resource->vk_format;

   if (!level_width || !level_height || view_format == VK_FORMAT_UNDEFINED)
      return false;

   uint32_t transient = UINT32_MAX;
   if (!yttrium_venus_cmd_batch_alloc_transient(venus, &transient))
      return false;

   VkImageView view = venus->cmd_batch_transients[transient].view;
   VkRenderPass render_pass =
      venus->cmd_batch_transients[transient].render_pass;
   VkFramebuffer framebuffer =
      venus->cmd_batch_transients[transient].framebuffer;

   const VkImageSubresourceRange range =
      yttrium_venus_render_barrier_range(resource, VK_IMAGE_ASPECT_COLOR_BIT,
                                         level, layer, 1);
   yttrium_venus_cmd_transition_image(
      venus, resource, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      &range);

   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = resource->image,
      .viewType = yttrium_venus_render_view_type(resource, 1),
      .format = view_format,
      .subresourceRange = range,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateImageView,
      &venus->cmd_batch_transients[transient].view_obj, &view,
      goto color_clear_submit_failed, venus->device_handle, &view_info, NULL,
      &view);
   venus->cmd_batch_transients[transient].view_created = true;

   const VkAttachmentDescription attachment = {
      .format = view_format,
      .samples = samples,
      .loadOp = full_clear ? VK_ATTACHMENT_LOAD_OP_DONT_CARE :
                             VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkAttachmentReference color_ref = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_ref,
   };
   const VkRenderPassCreateInfo render_pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &attachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateRenderPass,
      &venus->cmd_batch_transients[transient].render_pass_obj, &render_pass,
      goto color_clear_submit_failed, venus->device_handle,
      &render_pass_info, NULL, &render_pass);
   venus->cmd_batch_transients[transient].render_pass_created = true;

   const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount = 1,
      .pAttachments = &view,
      .width = level_width,
      .height = level_height,
      .layers = 1,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateFramebuffer,
      &venus->cmd_batch_transients[transient].framebuffer_obj, &framebuffer,
      goto color_clear_submit_failed, venus->device_handle,
      &framebuffer_info, NULL, &framebuffer);
   venus->cmd_batch_transients[transient].framebuffer_created = true;

   const VkRenderPassBeginInfo render_pass_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = {
         .offset = { 0, 0 },
         .extent = { level_width, level_height },
      },
   };
   vn_async_vkCmdBeginRenderPass(&venus->vn_ring, venus->command_buffer,
                                 &render_pass_begin,
                                 VK_SUBPASS_CONTENTS_INLINE);
   const VkClearColorValue clear =
      yttrium_venus_clear_color_value(pipe_format, color);
   const VkClearAttachment clear_attachment = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .colorAttachment = 0,
      .clearValue = {
         .color = clear,
      },
   };
   const VkClearRect clear_rect = {
      .rect = {
         .offset = { (int32_t)dstx, (int32_t)dsty },
         .extent = { width, height },
      },
      .baseArrayLayer = 0,
      .layerCount = 1,
   };
   vn_async_vkCmdClearAttachments(&venus->vn_ring, venus->command_buffer,
                                  1, &clear_attachment, 1, &clear_rect);
   vn_async_vkCmdEndRenderPass(&venus->vn_ring, venus->command_buffer);

   YTTRIUM_LOG("yttrium: Venus attachment-cleared color res_id=%u image_id=%llu level=%u layer=%u rect=%u,%u %ux%u\n",
               resource_id,
               (unsigned long long)resource->image_obj.id,
               level, layer, dstx, dsty, width, height);
   return true;

color_clear_submit_failed:
   yttrium_venus_cmd_batch_release_transient(venus, transient);
   return false;
}

bool
yttrium_venus2_copy_display_image(struct yttrium_venus *venus,
                                 struct yttrium_venus_resource *src,
                                 struct yttrium_venus_resource *dst,
                                 uint32_t src_resource_id,
                                 uint32_t dst_resource_id,
                                 uint32_t src_level,
                                 uint32_t src_x,
                                 uint32_t src_y,
                                 uint32_t src_layer,
                                 uint32_t dst_level,
                                 uint32_t dst_x,
                                 uint32_t dst_y,
                                 uint32_t dst_layer,
                                 uint32_t width,
                                 uint32_t height)
{
   const uint32_t requested_src_x = src_x;
   const uint32_t requested_src_y = src_y;
   const uint32_t requested_src_layer = src_layer;
   const uint32_t requested_src_level = src_level;
   const uint32_t requested_dst_x = dst_x;
   const uint32_t requested_dst_y = dst_y;
   const uint32_t requested_dst_layer = dst_layer;
   const uint32_t requested_dst_level = dst_level;
   const uint32_t requested_width = width;
   const uint32_t requested_height = height;
   bool promoted_full_copy = false;

   if (!src || !src->initialized || src->buffer_backed || !src->image ||
       !dst || !dst->initialized || dst->buffer_backed || !dst->image)
      return false;
   if (!(src->image_usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ||
       !(dst->image_usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
      return false;
   if (!yttrium_venus_valid_image_subresource(src, src_level, src_layer, 1) ||
       !yttrium_venus_valid_image_subresource(dst, dst_level, dst_layer, 1))
      return false;
   const uint32_t src_level_width =
      yttrium_venus_subresource_width(src, src_level);
   const uint32_t src_level_height =
      yttrium_venus_subresource_height(src, src_level);
   const uint32_t dst_level_width =
      yttrium_venus_subresource_width(dst, dst_level);
   const uint32_t dst_level_height =
      yttrium_venus_subresource_height(dst, dst_level);
   if (src_x > src_level_width || src_y > src_level_height ||
       dst_x > dst_level_width || dst_y > dst_level_height ||
       width > src_level_width - src_x ||
       height > src_level_height - src_y ||
       width > dst_level_width - dst_x ||
       height > dst_level_height - dst_y)
      return false;

   bool full_dst_copy =
      dst_x == 0 && dst_y == 0 && dst_level == 0 && dst_layer == 0 &&
      width == dst->width && height == dst->height &&
      MAX2(dst->levels, 1) == 1 && MAX2(dst->layers, 1) == 1;

   if (!dst->contents_initialized && !full_dst_copy &&
       src_level == dst_level && src_layer == dst_layer &&
       src->width == dst->width && src->height == dst->height &&
       src->vk_format == dst->vk_format &&
       MAX2(src->levels, 1) == 1 && MAX2(dst->levels, 1) == 1 &&
       MAX2(src->layers, 1) == 1 && MAX2(dst->layers, 1) == 1) {
      YTTRIUM_LOG("yttrium: Venus promoted partial display image copy to full copy src_res_id=%u dst_res_id=%u dirty=%u,%u->%u,%u %ux%u\n",
                  src_resource_id, dst_resource_id,
                  src_x, src_y, dst_x, dst_y, width, height);
      src_x = 0;
      src_y = 0;
      dst_x = 0;
      dst_y = 0;
      width = dst->width;
      height = dst->height;
      full_dst_copy = true;
      promoted_full_copy = true;
   }

   if (!src->contents_initialized &&
       !(src->image_usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
      YTTRIUM_LOG("yttrium: Venus display image copy rejected uninitialized source without transfer-dst usage src_res_id=%u dst_res_id=%u\n",
                  src_resource_id, dst_resource_id);
      return false;
   }
   if (!dst->contents_initialized && !full_dst_copy &&
       !(dst->image_usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
      YTTRIUM_LOG("yttrium: Venus display image copy rejected uninitialized partial destination without transfer-dst usage src_res_id=%u dst_res_id=%u\n",
                  src_resource_id, dst_resource_id);
      return false;
   }

   /*
    * Copy whichever aspects the two formats share.  This was COLOR-only, so a
    * depth-to-depth copy fell through to "unsupported": Superposition copies
    * its depth buffer into a sampled Z32_FLOAT texture and reconstructs world
    * position from it in the light pass, and with the copy refused that texture
    * stayed zero, every light contributed nothing, and the scene rendered black.
    *
    * The depth-stencil path next door is not a substitute - it replays clear
    * history onto the destination and only applies when the source holds
    * nothing but clears, which a G-buffer depth buffer never does.
    */
   const VkImageAspectFlags shared_aspects =
      yttrium_venus_format_aspects(src->vk_format) &
      yttrium_venus_format_aspects(dst->vk_format);
   VkImageAspectFlags copy_aspects =
      shared_aspects & VK_IMAGE_ASPECT_COLOR_BIT;

   if (!copy_aspects)
      copy_aspects = shared_aspects & (VK_IMAGE_ASPECT_DEPTH_BIT |
                                       VK_IMAGE_ASPECT_STENCIL_BIT);
   if (!copy_aspects)
      return false;

   if (!yttrium_venus_begin_display_copy_batch(venus))
      return false;

   if (!yttrium_venus_cmd_batch_track_resource(venus, src) ||
       !yttrium_venus_cmd_batch_track_resource(venus, dst)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "display image copy tracking failure");
      return false;
   }

   const VkImageSubresourceRange src_range = {
      .aspectMask = copy_aspects,
      .baseMipLevel = src_level,
      .levelCount = 1,
      .baseArrayLayer = src_layer,
      .layerCount = 1,
   };
   const VkImageSubresourceRange dst_range = {
      .aspectMask = copy_aspects,
      .baseMipLevel = dst_level,
      .levelCount = 1,
      .baseArrayLayer = dst_layer,
      .layerCount = 1,
   };
   if (!yttrium_venus_cmd_ensure_image_initialized(venus, src,
                                                   src_resource_id))
      goto fail_batch;
   yttrium_venus_cmd_transition_image(venus, src,
                                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                      VK_ACCESS_TRANSFER_READ_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      &src_range);

   if (!dst->contents_initialized && !full_dst_copy &&
       !yttrium_venus_cmd_ensure_image_initialized(venus, dst,
                                                   dst_resource_id))
      goto fail_batch;
   yttrium_venus_cmd_transition_image(venus, dst,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      VK_ACCESS_TRANSFER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      &dst_range);

   const VkImageCopy copy_region = {
      .srcSubresource = {
         .aspectMask = copy_aspects,
         .mipLevel = src_level,
         .baseArrayLayer = src_layer,
         .layerCount = 1,
      },
      .srcOffset = { (int32_t)src_x, (int32_t)src_y, 0 },
      .dstSubresource = {
         .aspectMask = copy_aspects,
         .mipLevel = dst_level,
         .baseArrayLayer = dst_layer,
         .layerCount = 1,
      },
      .dstOffset = { (int32_t)dst_x, (int32_t)dst_y, 0 },
      .extent = { width, height, 1 },
   };
   yttrium_trace_debug_stringf(
      "yttrium: Venus display image copy cmd src_res=%u dst_res=%u src_image=%llu dst_image=%llu src_mem=%llu dst_mem=%llu req_src_level=%u req_src=%u,%u,%u req_dst_level=%u req_dst=%u,%u,%u req_size=%ux%u eff_src_level=%u eff_src=%u,%u,%u eff_dst_level=%u eff_dst=%u,%u,%u eff_size=%ux%u src_extent=%ux%u dst_extent=%ux%u src_fmt=%u dst_fmt=%u src_usage=0x%x dst_usage=0x%x src_layout=%u dst_layout=%u src_init=%u dst_init_before=%u full=%u promoted=%u same=%u src_off=0x%llx dst_off=0x%llx src_size=0x%llx dst_size=0x%llx src_row=%llu dst_row=%llu src_array=%llu dst_array=%llu",
      src_resource_id, dst_resource_id,
      (unsigned long long)src->image_obj.id,
      (unsigned long long)dst->image_obj.id,
      (unsigned long long)src->memory_obj.id,
      (unsigned long long)dst->memory_obj.id,
      requested_src_level, requested_src_x, requested_src_y,
      requested_src_layer,
      requested_dst_level, requested_dst_x, requested_dst_y,
      requested_dst_layer,
      requested_width, requested_height,
      src_level, src_x, src_y, src_layer,
      dst_level, dst_x, dst_y, dst_layer, width, height,
      src->width, src->height, dst->width, dst->height,
      src->vk_format, dst->vk_format,
      src->image_usage, dst->image_usage,
      src->layout, dst->layout,
      src->contents_initialized, dst->contents_initialized,
      full_dst_copy, promoted_full_copy, src == dst,
      (unsigned long long)src->image_offset,
      (unsigned long long)dst->image_offset,
      (unsigned long long)src->image_size,
      (unsigned long long)dst->image_size,
      (unsigned long long)src->image_row_pitch,
      (unsigned long long)dst->image_row_pitch,
      (unsigned long long)src->image_array_pitch,
      (unsigned long long)dst->image_array_pitch);
   vn_async_vkCmdCopyImage(&venus->vn_ring, venus->command_buffer,
                           src->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           dst->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &copy_region);
   yttrium_trace_venus_upload(
      YTTRIUM_TRACE_VENUS_UPLOAD_IMAGE_COPY, promoted_full_copy ? 1 : 0,
      (uint64_t)width * height * 4, src->image_obj.id, dst->image_obj.id,
      src_resource_id, dst_resource_id, width, height, 1,
      (uint32_t)MIN2(dst->image_row_pitch, (uint64_t)UINT32_MAX), 0);

   src->layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   dst->layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   yttrium_venus_mark_aspects_initialized(dst, copy_aspects);

   YTTRIUM_LOG("yttrium: Venus copied display image src_res_id=%u image_id=%llu level=%u %u,%u layer=%u dst_res_id=%u image_id=%llu level=%u %u,%u layer=%u %ux%u\n",
                src_resource_id,
                (unsigned long long)src->image_obj.id,
                src_level, src_x, src_y, src_layer,
                dst_resource_id,
                (unsigned long long)dst->image_obj.id,
                dst_level, dst_x, dst_y, dst_layer,
                width, height);
   return yttrium_venus_cmd_batch_submit_after_record(
      venus, "display image copy");

fail_batch:
   yttrium_venus_flush_command_batch(venus,
                                          "display image copy batch abort");
   return false;
}

bool
yttrium_venus2_copy_depth_stencil_image(struct yttrium_venus *venus,
                                       struct yttrium_venus_resource *src,
                                       struct yttrium_venus_resource *dst,
                                       uint32_t src_resource_id,
                                       uint32_t dst_resource_id,
                                       uint32_t src_level,
                                       uint32_t src_x,
                                       uint32_t src_y,
                                       uint32_t src_layer,
                                       uint32_t dst_level,
                                       uint32_t dst_x,
                                       uint32_t dst_y,
                                       uint32_t dst_layer,
                                       uint32_t width,
                                       uint32_t height,
                                       VkImageAspectFlags aspect_mask)
{
   if (!src || !src->initialized || src->buffer_backed || !src->image ||
       !dst || !dst->initialized || dst->buffer_backed || !dst->image)
      return false;
   if (!(src->image_usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ||
       !(dst->image_usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
      return false;
   if (src->vk_format != dst->vk_format ||
       src->vk_format == VK_FORMAT_UNDEFINED)
      return false;
   if (!yttrium_venus_valid_image_subresource(src, src_level, src_layer, 1) ||
       !yttrium_venus_valid_image_subresource(dst, dst_level, dst_layer, 1))
      return false;

   const VkImageAspectFlags format_aspects =
      yttrium_venus_format_aspects(src->vk_format) &
      (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
   const VkImageAspectFlags requested_aspects = aspect_mask ?
      (aspect_mask & format_aspects) : format_aspects;
   const VkImageAspectFlags aspects =
      requested_aspects & yttrium_venus_initialized_aspects(src);
   if (!aspects)
      return true;

   const uint32_t src_level_width =
      yttrium_venus_subresource_width(src, src_level);
   const uint32_t src_level_height =
      yttrium_venus_subresource_height(src, src_level);
   const uint32_t dst_level_width =
      yttrium_venus_subresource_width(dst, dst_level);
   const uint32_t dst_level_height =
      yttrium_venus_subresource_height(dst, dst_level);
   if (!width || !height ||
       src_x > src_level_width || src_y > src_level_height ||
       dst_x > dst_level_width || dst_y > dst_level_height ||
       width > src_level_width - src_x ||
       height > src_level_height - src_y ||
       width > dst_level_width - dst_x ||
       height > dst_level_height - dst_y)
      return false;

   const bool full_src_copy =
      src_x == 0 && src_y == 0 && src_level == 0 && src_layer == 0 &&
      width == src->width && height == src->height &&
      MAX2(src->levels, 1) == 1 && MAX2(src->layers, 1) == 1;
   const bool full_dst_copy =
      dst_x == 0 && dst_y == 0 && dst_level == 0 && dst_layer == 0 &&
      width == dst->width && height == dst->height &&
      MAX2(dst->levels, 1) == 1 && MAX2(dst->layers, 1) == 1;
   if (!full_src_copy || !full_dst_copy ||
       src_level_width != dst_level_width ||
       src_level_height != dst_level_height) {
      YTTRIUM_WARN("yttrium: Venus depth/stencil copy rejected non-full clear-history copy src_res_id=%u dst_res_id=%u src_level=%u src=%u,%u src_extent=%ux%u dst_level=%u dst=%u,%u dst_extent=%ux%u copy=%ux%u aspects=0x%x\n",
                   src_resource_id, dst_resource_id,
                   src_level, src_x, src_y, src_level_width, src_level_height,
                   dst_level, dst_x, dst_y, dst_level_width, dst_level_height,
                   width, height, aspects);
      return false;
   }
   if (!src->ds_clear_history_valid) {
      YTTRIUM_WARN("yttrium: Venus depth/stencil copy rejected source with non-clear history src_res_id=%u dst_res_id=%u aspects=0x%x\n",
                   src_resource_id, dst_resource_id, aspects);
      return false;
   }

   uint32_t replay_count = 0;
   VkImageAspectFlags covered_aspects = 0;
   for (uint32_t i = 0; i < src->ds_clear_history_count; i++) {
      const struct yttrium_venus_ds_clear_record *record =
         &src->ds_clear_history[i];
      const VkImageAspectFlags record_aspects = record->aspects & aspects;
      if (!record_aspects || record->level != src_level ||
          record->layer != src_layer)
         continue;

      if (record->x >= src_level_width || record->y >= src_level_height ||
          record->width > src_level_width - record->x ||
          record->height > src_level_height - record->y) {
         YTTRIUM_WARN("yttrium: Venus depth/stencil copy rejected invalid clear-history rect src_res_id=%u dst_res_id=%u rect=%u,%u %ux%u extent=%ux%u aspects=0x%x\n",
                      src_resource_id, dst_resource_id,
                      record->x, record->y, record->width, record->height,
                      src_level_width, src_level_height, record_aspects);
         return false;
      }
      const bool full_record =
         record->x == 0 && record->y == 0 &&
         record->width == src_level_width &&
         record->height == src_level_height;
      const VkImageAspectFlags new_aspects = record_aspects & ~covered_aspects;
      if (new_aspects && !full_record) {
         YTTRIUM_WARN("yttrium: Venus depth/stencil copy rejected clear history without full base src_res_id=%u dst_res_id=%u aspects=0x%x covered=0x%x record=0x%x rect=%u,%u %ux%u extent=%ux%u\n",
                      src_resource_id, dst_resource_id, aspects,
                      covered_aspects, record_aspects,
                      record->x, record->y, record->width, record->height,
                      src_level_width, src_level_height);
         return false;
      }
      if (full_record)
         covered_aspects |= record_aspects;
      replay_count++;
   }
   if (!replay_count || (covered_aspects & aspects) != aspects) {
      YTTRIUM_WARN("yttrium: Venus depth/stencil copy rejected incomplete clear history src_res_id=%u dst_res_id=%u aspects=0x%x history=%u replay=%u full_base=%u\n",
                   src_resource_id, dst_resource_id, aspects,
                   src->ds_clear_history_count, replay_count,
                   (covered_aspects & aspects) == aspects);
      return false;
   }

   if (!yttrium_venus_begin_display_copy_batch(venus))
      return false;

   if (!yttrium_venus_cmd_batch_track_resource(venus, src) ||
       !yttrium_venus_cmd_batch_track_resource(venus, dst)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "depth/stencil image copy tracking failure");
      return false;
   }

   if (!dst->ds_clear_history_valid)
      yttrium_venus_ds_clear_history_reset(dst);
   else
      yttrium_venus_ds_clear_history_drop_aspects(dst, aspects);
   for (uint32_t i = 0; i < src->ds_clear_history_count; i++) {
      const struct yttrium_venus_ds_clear_record *record =
         &src->ds_clear_history[i];
      const VkImageAspectFlags record_aspects = record->aspects & aspects;
      if (!record_aspects || record->level != src_level ||
          record->layer != src_layer)
         continue;

      if (!yttrium_venus_cmd_clear_depth_stencil_attachment(
             venus, dst, dst_resource_id, record_aspects, record->depth,
             record->stencil, dst_level, dst_layer, record->x, record->y,
             record->width, record->height)) {
         yttrium_venus_ds_clear_history_invalidate(dst);
         yttrium_venus_cancel_command_batch_setup_failure(
            venus, "depth/stencil clear-history copy setup failure");
         return false;
      }
      yttrium_venus_ds_clear_history_note(
         dst, record_aspects, record->depth, record->stencil, dst_level,
         dst_layer, record->x, record->y, record->width, record->height);
   }

   yttrium_venus_mark_aspects_initialized(dst, aspects);
   YTTRIUM_LOG("yttrium: Venus replay-copied depth/stencil clear history src_res_id=%u image_id=%llu dst_res_id=%u image_id=%llu level=%u layer=%u -> level=%u layer=%u %ux%u aspects=0x%x records=%u\n",
               src_resource_id,
               (unsigned long long)src->image_obj.id,
               dst_resource_id,
               (unsigned long long)dst->image_obj.id,
               src_level, src_layer, dst_level, dst_layer,
               width, height, aspects, replay_count);
   return yttrium_venus_cmd_batch_submit_after_record(
      venus, "depth/stencil image copy");
}

static VkFormat
yttrium_venus2_resolve_linear_format(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_B8G8R8A8_SRGB:
      return VK_FORMAT_B8G8R8A8_UNORM;
   case VK_FORMAT_R8G8B8A8_SRGB:
      return VK_FORMAT_R8G8B8A8_UNORM;
   default:
      return format;
   }
}

static bool
yttrium_venus2_resolve_formats_compatible(VkFormat image_format,
                                          VkFormat view_format)
{
   return image_format == view_format ||
          yttrium_venus2_resolve_linear_format(image_format) ==
             yttrium_venus2_resolve_linear_format(view_format);
}

static bool
yttrium_venus_cmd_resolve_color_attachment(
   struct yttrium_venus *venus,
   struct yttrium_venus_resource *src,
   struct yttrium_venus_resource *dst,
   VkFormat resolve_vk_format,
   uint32_t src_level,
   uint32_t src_x,
   uint32_t src_y,
   uint32_t src_layer,
   uint32_t dst_level,
   uint32_t dst_x,
   uint32_t dst_y,
   uint32_t dst_layer,
   uint32_t width,
   uint32_t height)
{
   if (!(src->image_usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ||
       !(dst->image_usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT))
      return false;
   if (!yttrium_venus2_resolve_formats_compatible(src->vk_format,
                                                  resolve_vk_format) ||
       !yttrium_venus2_resolve_formats_compatible(dst->vk_format,
                                                  resolve_vk_format))
      return false;
   if (src_x != dst_x || src_y != dst_y)
      return false;

   const uint32_t framebuffer_width =
      MIN2(yttrium_venus_subresource_width(src, src_level),
           yttrium_venus_subresource_width(dst, dst_level));
   const uint32_t framebuffer_height =
      MIN2(yttrium_venus_subresource_height(src, src_level),
           yttrium_venus_subresource_height(dst, dst_level));
   if (src_x > framebuffer_width || src_y > framebuffer_height ||
       width > framebuffer_width - src_x ||
       height > framebuffer_height - src_y)
      return false;

   uint32_t src_transient = UINT32_MAX;
   uint32_t dst_transient = UINT32_MAX;
   if (!yttrium_venus_cmd_batch_alloc_transient(venus, &src_transient))
      return false;
   if (!yttrium_venus_cmd_batch_alloc_transient(venus, &dst_transient)) {
      yttrium_venus_cmd_batch_release_transient(venus, src_transient);
      return false;
   }

   VkImageView src_view = venus->cmd_batch_transients[src_transient].view;
   VkImageView dst_view = venus->cmd_batch_transients[dst_transient].view;
   VkRenderPass render_pass =
      venus->cmd_batch_transients[src_transient].render_pass;
   VkFramebuffer framebuffer =
      venus->cmd_batch_transients[src_transient].framebuffer;

   const VkImageSubresourceRange src_range = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .baseMipLevel = src_level,
      .levelCount = 1,
      .baseArrayLayer = src_layer,
      .layerCount = 1,
   };
   const VkImageSubresourceRange dst_range = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .baseMipLevel = dst_level,
      .levelCount = 1,
      .baseArrayLayer = dst_layer,
      .layerCount = 1,
   };
   yttrium_venus_cmd_transition_image(
      venus, src, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, &src_range);
   yttrium_venus_cmd_transition_image(
      venus, dst, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, &dst_range);

   const VkImageViewCreateInfo src_view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = src->image,
      .viewType = yttrium_venus_render_view_type(src, 1),
      .format = resolve_vk_format,
      .subresourceRange = src_range,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateImageView,
      &venus->cmd_batch_transients[src_transient].view_obj, &src_view,
      goto resolve_submit_failed, venus->device_handle, &src_view_info, NULL,
      &src_view);
   venus->cmd_batch_transients[src_transient].view_created = true;

   const VkImageViewCreateInfo dst_view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = dst->image,
      .viewType = yttrium_venus_render_view_type(dst, 1),
      .format = resolve_vk_format,
      .subresourceRange = dst_range,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateImageView,
      &venus->cmd_batch_transients[dst_transient].view_obj, &dst_view,
      goto resolve_submit_failed, venus->device_handle, &dst_view_info, NULL,
      &dst_view);
   venus->cmd_batch_transients[dst_transient].view_created = true;

   const VkAttachmentDescription attachments[2] = {
      {
         .format = resolve_vk_format,
         .samples = src->samples ? src->samples : VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      },
      {
         .format = resolve_vk_format,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      },
   };
   const VkAttachmentReference color_ref = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkAttachmentReference resolve_ref = {
      .attachment = 1,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_ref,
      .pResolveAttachments = &resolve_ref,
   };
   const VkRenderPassCreateInfo render_pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = ARRAY_SIZE(attachments),
      .pAttachments = attachments,
      .subpassCount = 1,
      .pSubpasses = &subpass,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateRenderPass,
      &venus->cmd_batch_transients[src_transient].render_pass_obj,
      &render_pass, goto resolve_submit_failed, venus->device_handle,
      &render_pass_info, NULL, &render_pass);
   venus->cmd_batch_transients[src_transient].render_pass_created = true;

   const VkImageView framebuffer_attachments[2] = {
      src_view,
      dst_view,
   };
   const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount = ARRAY_SIZE(framebuffer_attachments),
      .pAttachments = framebuffer_attachments,
      .width = framebuffer_width,
      .height = framebuffer_height,
      .layers = 1,
   };
   YTTRIUM_VENUS_SUBMIT_OBJECT_OR(
      venus, vkCreateFramebuffer,
      &venus->cmd_batch_transients[src_transient].framebuffer_obj,
      &framebuffer, goto resolve_submit_failed, venus->device_handle,
      &framebuffer_info, NULL, &framebuffer);
   venus->cmd_batch_transients[src_transient].framebuffer_created = true;

   const VkRenderPassBeginInfo render_pass_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = {
         .offset = { (int32_t)src_x, (int32_t)src_y },
         .extent = { width, height },
      },
   };
   vn_async_vkCmdBeginRenderPass(&venus->vn_ring, venus->command_buffer,
                                 &render_pass_begin,
                                 VK_SUBPASS_CONTENTS_INLINE);
   vn_async_vkCmdEndRenderPass(&venus->vn_ring, venus->command_buffer);
   return true;

resolve_submit_failed:
   yttrium_venus_cmd_batch_release_transient(venus, dst_transient);
   yttrium_venus_cmd_batch_release_transient(venus, src_transient);
   return false;
}

bool
yttrium_venus2_resolve_display_image(struct yttrium_venus *venus,
                                    struct yttrium_venus_resource *src,
                                    struct yttrium_venus_resource *dst,
                                    uint32_t src_resource_id,
                                    uint32_t dst_resource_id,
                                    uint32_t src_level,
                                    uint32_t src_x,
                                    uint32_t src_y,
                                    uint32_t src_layer,
                                    uint32_t dst_level,
                                    uint32_t dst_x,
                                    uint32_t dst_y,
                                    uint32_t dst_layer,
                                    enum pipe_format resolve_format,
                                    uint32_t width,
                                    uint32_t height)
{
   if (!src || !src->initialized || src->buffer_backed || !src->image ||
       !dst || !dst->initialized || dst->buffer_backed || !dst->image)
      return false;
   if (!(src->image_usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ||
       !(dst->image_usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
      return false;
   if ((src->samples ? src->samples : VK_SAMPLE_COUNT_1_BIT) ==
       VK_SAMPLE_COUNT_1_BIT)
      return false;
   if ((dst->samples ? dst->samples : VK_SAMPLE_COUNT_1_BIT) !=
       VK_SAMPLE_COUNT_1_BIT)
      return false;

   const VkFormat resolve_vk_format =
      yttrium_venus2_pipe_format(resolve_format);
   if (src->vk_format == VK_FORMAT_UNDEFINED ||
       dst->vk_format == VK_FORMAT_UNDEFINED ||
       resolve_vk_format == VK_FORMAT_UNDEFINED)
      return false;
   if (!(yttrium_venus_format_aspects(src->vk_format) &
         yttrium_venus_format_aspects(dst->vk_format) &
         yttrium_venus_format_aspects(resolve_vk_format) &
         VK_IMAGE_ASPECT_COLOR_BIT))
      return false;

   if (!yttrium_venus_valid_image_subresource(src, src_level, src_layer, 1) ||
       !yttrium_venus_valid_image_subresource(dst, dst_level, dst_layer, 1))
      return false;
   if (src == dst)
      return false;

   const uint32_t src_level_width =
      yttrium_venus_subresource_width(src, src_level);
   const uint32_t src_level_height =
      yttrium_venus_subresource_height(src, src_level);
   const uint32_t dst_level_width =
      yttrium_venus_subresource_width(dst, dst_level);
   const uint32_t dst_level_height =
      yttrium_venus_subresource_height(dst, dst_level);
   if (!width || !height ||
       src_x > src_level_width || src_y > src_level_height ||
       dst_x > dst_level_width || dst_y > dst_level_height ||
       width > src_level_width - src_x ||
       height > src_level_height - src_y ||
       width > dst_level_width - dst_x ||
       height > dst_level_height - dst_y)
      return false;

   if (!yttrium_venus_begin_transfer_batch(venus))
      return false;

   if (!yttrium_venus_cmd_batch_track_resource(venus, src) ||
       !yttrium_venus_cmd_batch_track_resource(venus, dst)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "display image resolve tracking failure");
      return false;
   }

   if (!yttrium_venus_cmd_ensure_image_initialized(venus, src,
                                                   src_resource_id)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "display image resolve source init failure");
      return false;
   }
   if (!dst->contents_initialized &&
       !yttrium_venus_cmd_ensure_image_initialized(venus, dst,
                                                   dst_resource_id)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "display image resolve destination init failure");
      return false;
   }

   const bool direct_resolve =
      src->vk_format == dst->vk_format &&
      src->vk_format == resolve_vk_format;
   if (direct_resolve) {
      const VkImageSubresourceRange src_range = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .baseMipLevel = src_level,
         .levelCount = 1,
         .baseArrayLayer = src_layer,
         .layerCount = 1,
      };
      const VkImageSubresourceRange dst_range = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .baseMipLevel = dst_level,
         .levelCount = 1,
         .baseArrayLayer = dst_layer,
         .layerCount = 1,
      };
      yttrium_venus_cmd_transition_image(venus, src,
                                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                         VK_ACCESS_TRANSFER_READ_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         &src_range);
      yttrium_venus_cmd_transition_image(venus, dst,
                                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                         VK_ACCESS_TRANSFER_WRITE_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         &dst_range);

      const VkImageResolve region = {
         .srcSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = src_level,
            .baseArrayLayer = src_layer,
            .layerCount = 1,
         },
         .srcOffset = { (int32_t)src_x, (int32_t)src_y, 0 },
         .dstSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = dst_level,
            .baseArrayLayer = dst_layer,
            .layerCount = 1,
         },
         .dstOffset = { (int32_t)dst_x, (int32_t)dst_y, 0 },
         .extent = { width, height, 1 },
      };
      vn_async_vkCmdResolveImage(&venus->vn_ring, venus->command_buffer,
                                 src->image,
                                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 dst->image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 1, &region);
   } else if (!yttrium_venus_cmd_resolve_color_attachment(
                 venus, src, dst, resolve_vk_format, src_level, src_x, src_y,
                 src_layer, dst_level, dst_x, dst_y, dst_layer, width,
                 height)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "display image typed resolve setup failure");
      return false;
   }

   yttrium_trace_venus_upload(
      YTTRIUM_TRACE_VENUS_UPLOAD_IMAGE_RESOLVE, 0,
      (uint64_t)width * height * 4, src->image_obj.id, dst->image_obj.id,
      src_resource_id, dst_resource_id, width, height, 1,
      (uint32_t)MIN2(dst->image_row_pitch, (uint64_t)UINT32_MAX), 0);

   yttrium_venus_mark_aspects_initialized(dst, VK_IMAGE_ASPECT_COLOR_BIT);
   return yttrium_venus_cmd_batch_after_record(venus,
                                               "display image resolve");
}

bool
yttrium_venus2_blit_display_image(struct yttrium_venus *venus,
                                 struct yttrium_venus_resource *src,
                                 struct yttrium_venus_resource *dst,
                                 uint32_t src_resource_id,
                                 uint32_t dst_resource_id,
                                 uint32_t src_level,
                                 uint32_t src_x,
                                 uint32_t src_y,
                                 uint32_t src_layer,
                                 uint32_t src_width,
                                 uint32_t src_height,
                                 uint32_t src_depth,
                                 uint32_t dst_level,
                                 uint32_t dst_x,
                                 uint32_t dst_y,
                                 uint32_t dst_layer,
                                 uint32_t dst_width,
                                 uint32_t dst_height,
                                 uint32_t dst_depth,
                                 bool linear_filter)
{
   if (!src || !src->initialized || src->buffer_backed || !src->image ||
       !dst || !dst->initialized || dst->buffer_backed || !dst->image)
      return false;
   if (!(src->image_usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ||
       !(dst->image_usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
      return false;

   const bool volume_blit =
      src->layers == 1 && dst->layers == 1 &&
      (src->depth > 1 || dst->depth > 1);
   const uint32_t src_base_layer = volume_blit ? 0 : src_layer;
   const uint32_t dst_base_layer = volume_blit ? 0 : dst_layer;
   if (!yttrium_venus_valid_image_subresource(src, src_level, src_base_layer, 1) ||
       !yttrium_venus_valid_image_subresource(dst, dst_level, dst_base_layer, 1))
      return false;
   if (src == dst && src_level == dst_level &&
       src_base_layer == dst_base_layer)
      return false;
   if (!(yttrium_venus_format_aspects(src->vk_format) &
         yttrium_venus_format_aspects(dst->vk_format) &
         VK_IMAGE_ASPECT_COLOR_BIT))
      return false;

   const uint32_t src_level_width =
      yttrium_venus_subresource_width(src, src_level);
   const uint32_t src_level_height =
      yttrium_venus_subresource_height(src, src_level);
   const uint32_t src_level_depth =
      yttrium_venus_subresource_depth(src, src_level);
   const uint32_t dst_level_width =
      yttrium_venus_subresource_width(dst, dst_level);
   const uint32_t dst_level_height =
      yttrium_venus_subresource_height(dst, dst_level);
   const uint32_t dst_level_depth =
      yttrium_venus_subresource_depth(dst, dst_level);
   if (!src_width || !src_height || !src_depth ||
       !dst_width || !dst_height || !dst_depth ||
       src_x > src_level_width || src_y > src_level_height ||
       dst_x > dst_level_width || dst_y > dst_level_height ||
       src_width > src_level_width - src_x ||
       src_height > src_level_height - src_y ||
       dst_width > dst_level_width - dst_x ||
       dst_height > dst_level_height - dst_y)
      return false;
   if (volume_blit) {
      if (src_layer > src_level_depth || dst_layer > dst_level_depth ||
          src_depth > src_level_depth - src_layer ||
          dst_depth > dst_level_depth - dst_layer)
         return false;
   } else if (src_depth != 1 || dst_depth != 1) {
      return false;
   }

   if (!yttrium_venus_begin_transfer_batch(venus))
      return false;

   if (!yttrium_venus_cmd_batch_track_resource(venus, src) ||
       !yttrium_venus_cmd_batch_track_resource(venus, dst)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "display image blit tracking failure");
      return false;
   }

   if (!yttrium_venus_cmd_ensure_image_initialized(venus, src,
                                                   src_resource_id)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "display image blit source init failure");
      return false;
   }
   if (src != dst && !dst->contents_initialized &&
       !yttrium_venus_cmd_ensure_image_initialized(venus, dst,
                                                   dst_resource_id)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "display image blit destination init failure");
      return false;
   }

   const VkImageSubresourceRange src_range = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .baseMipLevel = src_level,
      .levelCount = 1,
      .baseArrayLayer = src_base_layer,
      .layerCount = 1,
   };
   const VkImageSubresourceRange dst_range = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .baseMipLevel = dst_level,
      .levelCount = 1,
      .baseArrayLayer = dst_base_layer,
      .layerCount = 1,
   };

   if (src == dst) {
      const VkImageLayout old_layout = src->layout;
      const VkImageMemoryBarrier barriers[2] = {
         {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = yttrium_venus_layout_access(old_layout),
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = old_layout,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = src->image,
            .subresourceRange = src_range,
         },
         {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = yttrium_venus_layout_access(old_layout),
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = old_layout,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = dst->image,
            .subresourceRange = dst_range,
         },
      };
      vn_async_vkCmdPipelineBarrier(&venus->vn_ring, venus->command_buffer,
                                    yttrium_venus_layout_stage(old_layout),
                                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    0, 0, NULL, 0, NULL, 2, barriers);
      src->layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   } else {
      yttrium_venus_cmd_transition_image(venus, src,
                                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                         VK_ACCESS_TRANSFER_READ_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         &src_range);
      yttrium_venus_cmd_transition_image(venus, dst,
                                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                         VK_ACCESS_TRANSFER_WRITE_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         &dst_range);
   }

   const VkImageBlit blit_region = {
      .srcSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .mipLevel = src_level,
         .baseArrayLayer = src_base_layer,
         .layerCount = 1,
      },
      .srcOffsets = {
         { (int32_t)src_x, (int32_t)src_y,
           volume_blit ? (int32_t)src_layer : 0 },
         { (int32_t)(src_x + src_width), (int32_t)(src_y + src_height),
           volume_blit ? (int32_t)(src_layer + src_depth) : 1 },
      },
      .dstSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .mipLevel = dst_level,
         .baseArrayLayer = dst_base_layer,
         .layerCount = 1,
      },
      .dstOffsets = {
         { (int32_t)dst_x, (int32_t)dst_y,
           volume_blit ? (int32_t)dst_layer : 0 },
         { (int32_t)(dst_x + dst_width), (int32_t)(dst_y + dst_height),
           volume_blit ? (int32_t)(dst_layer + dst_depth) : 1 },
      },
   };
   vn_async_vkCmdBlitImage(&venus->vn_ring, venus->command_buffer,
                           src->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           dst->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &blit_region,
                           linear_filter ? VK_FILTER_LINEAR :
                                           VK_FILTER_NEAREST);
   yttrium_trace_venus_upload(
      YTTRIUM_TRACE_VENUS_UPLOAD_IMAGE_BLIT,
      linear_filter ? 1 : 0,
      (uint64_t)dst_width * dst_height * MAX2(dst_depth, 1) * 4,
      src->image_obj.id, dst->image_obj.id, src_resource_id, dst_resource_id,
      dst_width, dst_height, MAX2(dst_depth, 1),
      (uint32_t)MIN2(dst->image_row_pitch, (uint64_t)UINT32_MAX),
      (uint32_t)MIN2(dst->image_array_pitch, (uint64_t)UINT32_MAX));

   if (src == dst) {
      const VkImageMemoryBarrier restore_src = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = src->image,
         .subresourceRange = src_range,
      };
      vn_async_vkCmdPipelineBarrier(&venus->vn_ring, venus->command_buffer,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    0, 0, NULL, 0, NULL, 1,
                                    &restore_src);
      src->layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   }

   dst->contents_initialized = true;
   return yttrium_venus_cmd_batch_after_record(venus, "display image blit");
}

bool
yttrium_venus2_copy_buffer_to_display_image(struct yttrium_venus *venus,
                                           struct yttrium_venus_resource *src,
                                           struct yttrium_venus_resource *dst,
                                           uint32_t src_resource_id,
                                           uint32_t dst_resource_id,
                                           VkDeviceSize src_offset,
                                           uint32_t src_stride,
                                           uint32_t src_layer_stride,
                                           uint32_t dst_level,
                                           uint32_t dst_x,
                                           uint32_t dst_y,
                                           uint32_t dst_layer,
                                           uint32_t width,
                                           uint32_t height,
                                           uint32_t depth,
                                           enum pipe_format pipe_format)
{
   if (!src || !src->initialized || !src->buffer_backed || !src->buffer ||
       !dst || !dst->initialized || dst->buffer_backed || !dst->image)
      return false;
   if (!(dst->image_usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
      return false;
   const bool dst_is_3d = MAX2(dst->depth, 1) > 1;
   const uint32_t dst_array_layer = dst_is_3d ? 0 : dst_layer;
   const uint32_t dst_z = dst_is_3d ? dst_layer : 0;
   if (!yttrium_venus_valid_image_subresource(dst, dst_level,
                                              dst_array_layer, 1))
      return false;

   const unsigned cpp = util_format_get_blocksize(pipe_format);
   const unsigned block_width = util_format_get_blockwidth(pipe_format);
   const uint32_t min_stride = util_format_get_stride(pipe_format, width);
   if (!cpp || !block_width || !depth || src_stride < min_stride ||
       src_stride % cpp)
      return false;
   const uint32_t buffer_row_length =
      util_format_is_compressed(pipe_format) ?
      (src_stride / cpp) * block_width : src_stride / cpp;
   uint32_t buffer_image_height = 0;
   if (depth > 1 && src_layer_stride) {
      if (src_layer_stride < src_stride || src_layer_stride % src_stride)
         return false;
      buffer_image_height =
         util_format_is_compressed(pipe_format) ?
         (src_layer_stride / src_stride) *
            util_format_get_blockheight(pipe_format) :
         src_layer_stride / src_stride;
      if (buffer_image_height < height)
         return false;
   }
   const uint32_t dst_level_width =
      yttrium_venus_subresource_width(dst, dst_level);
   const uint32_t dst_level_height =
      yttrium_venus_subresource_height(dst, dst_level);
   const uint32_t dst_level_depth =
      yttrium_venus_subresource_depth(dst, dst_level);
   if (dst_x > dst_level_width || dst_y > dst_level_height ||
       width > dst_level_width - dst_x ||
       height > dst_level_height - dst_y)
      return false;
   if (dst_is_3d) {
      if (dst_z > dst_level_depth || depth > dst_level_depth - dst_z)
         return false;
   } else if (depth != 1) {
      return false;
   }

   if (!yttrium_venus_begin_transfer_batch(venus))
      return false;

   if (!yttrium_venus_cmd_batch_track_resource(venus, src) ||
       !yttrium_venus_cmd_batch_track_resource(venus, dst)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "buffer-to-image tracking failure");
      return false;
   }

   const VkBufferMemoryBarrier buffer_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = src->buffer,
      .offset = src_offset,
      .size = VK_WHOLE_SIZE,
   };
   vn_async_vkCmdPipelineBarrier(&venus->vn_ring, venus->command_buffer,
                                 VK_PIPELINE_STAGE_HOST_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                 0, NULL, 1, &buffer_barrier, 0, NULL);

   const VkImageSubresourceRange dst_range = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .baseMipLevel = dst_level,
      .levelCount = 1,
      .baseArrayLayer = dst_array_layer,
      .layerCount = 1,
   };
   const bool full_dst_copy =
      dst_x == 0 && dst_y == 0 && dst_z == 0 && dst_level == 0 &&
      dst_array_layer == 0 &&
      width == dst->width && height == dst->height &&
      MAX2(dst->depth, 1) == 1 && MAX2(dst->levels, 1) == 1 &&
      MAX2(dst->layers, 1) == 1;
   if (!dst->contents_initialized && !full_dst_copy &&
       !yttrium_venus_cmd_ensure_image_initialized(venus, dst,
                                                   dst_resource_id)) {
      yttrium_venus_cancel_command_batch_setup_failure(
         venus, "buffer-to-image destination init failure");
      return false;
   }
   yttrium_venus_cmd_transition_image(venus, dst,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      VK_ACCESS_TRANSFER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      &dst_range);

   const VkBufferImageCopy copy_region = {
      .bufferOffset = src_offset,
      .bufferRowLength = buffer_row_length,
      .bufferImageHeight = buffer_image_height,
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .mipLevel = dst_level,
         .baseArrayLayer = dst_array_layer,
         .layerCount = 1,
      },
      .imageOffset = { (int32_t)dst_x, (int32_t)dst_y, (int32_t)dst_z },
      .imageExtent = { width, height, depth },
   };
   vn_async_vkCmdCopyBufferToImage(&venus->vn_ring, venus->command_buffer,
                                   src->buffer, dst->image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   1, &copy_region);
   const uint64_t copy_bytes =
      src_layer_stride ? (uint64_t)src_layer_stride * depth :
      (uint64_t)src_stride * height * depth;
   yttrium_trace_venus_upload(
      YTTRIUM_TRACE_VENUS_UPLOAD_BUFFER_TO_IMAGE, 0, copy_bytes,
      src->buffer_obj.id, dst->image_obj.id, src_resource_id,
      dst_resource_id, width, height, depth, src_stride, src_layer_stride);

   dst->layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   dst->contents_initialized = true;
   YTTRIUM_LOG("yttrium: Venus copied buffer to display image src_res_id=%u buffer_id=%llu dst_res_id=%u image_id=%llu src_offset=0x%llx dst_level=%u dst=%u,%u layer=%u %ux%ux%u stride=%u layer_stride=%u row_length=%u image_height=%u\n",
                src_resource_id,
                (unsigned long long)src->buffer_obj.id,
                dst_resource_id,
                (unsigned long long)dst->image_obj.id,
                (unsigned long long)src_offset,
                dst_level, dst_x, dst_y, dst_layer,
                width, height, depth, src_stride, src_layer_stride,
                buffer_row_length, buffer_image_height);
   return yttrium_venus_cmd_batch_after_record(venus,
                                               "buffer-to-image copy");
}

bool
yttrium_venus2_resource_fini(
   struct yttrium_venus *venus,
   struct gdikmt_context *ctx,
   struct yttrium_venus_resource *resource,
   const struct yttrium_venus_allocation_snapshot *allocation)
{
   (void)ctx;

   if (!venus || !venus->initialized || !resource || !resource->initialized)
      return false;

   bool allocation_adopted = false;

   yttrium_venus_flush_command_batch(venus,
                                          "resource destroy flush");
   yttrium_venus_render_target_cache_invalidate_image(
      venus, resource->image_obj.id);

   /*
    * A resource-backed allocation must be deallocated through the runtime
    * hResource while that object is still alive.  hAllocationResource is the
    * KM resource handle and is not a valid D3DDDICB_DEALLOCATE::hResource;
    * using it crashes the D3D10/11 runtime, while a standalone HandleList
    * deallocation is rejected with E_INVALIDARG.  Drain this explicit resource
    * lifetime edge, retire the Venus objects, and leave the allocation for the
    * caller to deallocate synchronously through its live runtime handle.
    */
   const bool runtime_resource_backed =
      allocation && allocation->hAllocation && allocation->hResource &&
      allocation->hAllocationResource;
   if (runtime_resource_backed) {
      if (!yttrium_venus_wait_resource_batches(
             venus, resource,
             "runtime resource-backed allocation destroy")) {
         YTTRIUM_WARN("yttrium: Venus resource-backed destroy could not drain batches owner=venus2 hAllocation=0x%lx hResource=%p\n",
                      (unsigned long)allocation->hAllocation,
                      allocation->hResource);
      }
   }

   const bool runtime_handle_allocation =
      allocation && allocation->hAllocation && allocation->hResource &&
      !allocation->hAllocationResource;
   if (runtime_handle_allocation) {
      /*
       * D3D runtime-resource allocations must be deallocated through
       * hResource.  If no stable allocation-resource handle was captured, do
       * not stash the runtime pointer in a deferred batch-retire list where
       * teardown can outlive the runtime resource object.
       */
      if (!yttrium_venus_wait_resource_batches(
             venus, resource, "resource destroy runtime allocation")) {
         YTTRIUM_WARN("yttrium: Venus runtime allocation destroy could not drain resource batches owner=venus2 action=immediate_destroy hAllocation=0x%lx hResource=%p\n",
                      (unsigned long)allocation->hAllocation,
                      allocation->hResource);
      }
   }

   struct yttrium_venus_retired_resource *retired =
      yttrium_venus_retired_resource_create(resource);
   if (!retired) {
      YTTRIUM_LOG("yttrium: Venus resource retire allocation failed owner=venus2 reason=out-of-memory; falling back to synchronous resource destroy\n");
      yttrium_venus_wait_resource_batches(venus, resource,
                                          "resource destroy oom fallback");
      yttrium_venus_destroy_graphics_objects(venus, resource);
      retired = yttrium_venus_retired_resource_create(resource);
   }

   if (retired) {
      if (!runtime_resource_backed)
         yttrium_venus_retired_resource_adopt_allocation(retired, allocation);
      allocation_adopted = retired->hAllocation != 0;

      struct yttrium_venus_batch *batch =
         (runtime_resource_backed || runtime_handle_allocation) ?
         NULL : yttrium_venus_find_latest_resource_batch(venus, resource);
      if (batch) {
         yttrium_venus_batch_retire_resource(batch, retired);
      } else {
         yttrium_venus_destroy_retired_resource(venus, retired);
      }
   } else {
      struct yttrium_venus_sample_image_view *image_view =
         resource->sample_image_views;
      while (image_view) {
         struct yttrium_venus_sample_image_view *next = image_view->next;
         if (image_view->view)
            vn_async_vkDestroyImageView(&venus->vn_ring, venus->device_handle,
                                        image_view->view, NULL);
         FREE(image_view);
         image_view = next;
      }
      resource->sample_image_views = NULL;
      struct yttrium_venus_sample_buffer_view *buffer_view =
         resource->sample_buffer_views;
      while (buffer_view) {
         struct yttrium_venus_sample_buffer_view *next = buffer_view->next;
         if (buffer_view->view)
            vn_async_vkDestroyBufferView(&venus->vn_ring,
                                         venus->device_handle,
                                         buffer_view->view, NULL);
         FREE(buffer_view);
         buffer_view = next;
      }
      yttrium_venus_unmap_memory(venus, &resource->draw_vertex_mapping);
      yttrium_venus_unmap_memory(venus, &resource->draw_index_mapping);
      if (resource->draw_vertex_buffer)
         vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                                  resource->draw_vertex_buffer, NULL);
      if (resource->draw_vertex_memory)
         vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                               resource->draw_vertex_memory, NULL);
      if (resource->draw_index_buffer)
         vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                                  resource->draw_index_buffer, NULL);
      if (resource->draw_index_memory)
         vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                               resource->draw_index_memory, NULL);
      if (resource->device_local_draw_buffer)
         vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                                  resource->device_local_draw_buffer, NULL);
      if (resource->device_local_draw_memory)
         vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                               resource->device_local_draw_memory, NULL);
      if (resource->buffer)
         vn_async_vkDestroyBuffer(&venus->vn_ring, venus->device_handle,
                                  resource->buffer, NULL);
      if (resource->image)
         vn_async_vkDestroyImage(&venus->vn_ring, venus->device_handle,
                                 resource->image, NULL);
      if (resource->memory)
         vn_async_vkFreeMemory(&venus->vn_ring, venus->device_handle,
                               resource->memory, NULL);
   }

   if (runtime_resource_backed)
      yttrium_venus_drain_ring(venus,
                               "runtime resource-backed allocation destroy");

   memset(resource, 0, sizeof(*resource));
   return allocation_adopted;
}

struct yttrium_venus *
yttrium_venus2_create(struct gdikmt_device *device)
{
   struct yttrium_venus *venus = CALLOC_STRUCT(yttrium_venus);
   if (!venus)
      return NULL;

   const uint32_t initial_batch_count = yttrium_venus_batch_count();
   venus->batches =
      CALLOC(YTTRIUM_VENUS_BATCH_COUNT_MAX, sizeof(*venus->batches));
   if (!venus->batches) {
      YTTRIUM_WARN("yttrium: ERROR: Venus2 batch table allocation failed owner=venus2 slots=%u bytes=%llu\n",
                   YTTRIUM_VENUS_BATCH_COUNT_MAX,
                   (unsigned long long)
                      (YTTRIUM_VENUS_BATCH_COUNT_MAX *
                       sizeof(*venus->batches)));
      FREE(venus);
      return NULL;
   }
   for (uint32_t i = 0; i < initial_batch_count; i++) {
      venus->batches[i] = CALLOC_STRUCT(yttrium_venus_batch);
      if (!venus->batches[i]) {
         YTTRIUM_WARN("yttrium: ERROR: Venus2 initial batch allocation failed owner=venus2 slot=%u initial_slots=%u bytes_per_slot=%llu\n",
                      i, initial_batch_count,
                      (unsigned long long)sizeof(*venus->batches[i]));
         for (uint32_t j = 0; j < i; j++)
            FREE(venus->batches[j]);
         FREE(venus->batches);
         FREE(venus);
         return NULL;
      }
   }
   venus->batch_count = initial_batch_count;
   venus->batch_capacity = YTTRIUM_VENUS_BATCH_COUNT_MAX;
   venus->group_queue_submit_size =
      MIN2((uint32_t)YTTRIUM_VENUS_GROUP_QUEUE_SUBMIT_DEFAULT,
           MIN2((uint32_t)YTTRIUM_VENUS_GROUP_QUEUE_SUBMIT_MAX,
                (uint32_t)YTTRIUM_VENUS_BATCH_COUNT_MAX));

   const bool group_queue_submits_requested =
      yttrium_gdi_debug_get_bool_option(
         YTTRIUM_VENUS_GROUP_QUEUE_SUBMITS_ENV, true);
   venus->group_queue_submits =
      group_queue_submits_requested &&
      yttrium_venus_batch_epoch_wait_enabled();
   if (group_queue_submits_requested && !venus->group_queue_submits) {
      YTTRIUM_WARN("yttrium: ERROR: Venus2 grouped queue submits disabled owner=venus2 reason=batch_epoch_wait_required %s=1 D3D10UMD_YTTRIUM_BATCH_EPOCH_WAIT=0\n",
                   YTTRIUM_VENUS_GROUP_QUEUE_SUBMITS_ENV);
   }

   const int64_t group_queue_submit_op_threshold =
      yttrium_gdi_debug_get_num_option(
         YTTRIUM_VENUS_GROUP_QUEUE_SUBMIT_OP_THRESHOLD_ENV, 0);
   if (group_queue_submit_op_threshold > 0) {
      if (venus->group_queue_submits) {
         venus->group_queue_submit_op_threshold =
            (uint64_t)group_queue_submit_op_threshold;
         YTTRIUM_WARN("yttrium: WARNING: experimental grouped queue submit operation threshold enabled owner=venus2 component=grouped-queue-submit reason=process-option action=flush-at-operation-threshold threshold=%llu variable=%s\n",
                      (unsigned long long)
                         venus->group_queue_submit_op_threshold,
                      YTTRIUM_VENUS_GROUP_QUEUE_SUBMIT_OP_THRESHOLD_ENV);
      } else {
         YTTRIUM_WARN("yttrium: WARNING: grouped queue submit operation threshold ignored owner=venus2 component=grouped-queue-submit reason=grouped-submits-disabled action=leave-threshold-disabled requested=%lld variable=%s\n",
                      (long long)group_queue_submit_op_threshold,
                      YTTRIUM_VENUS_GROUP_QUEUE_SUBMIT_OP_THRESHOLD_ENV);
      }
   }

   if (venus->group_queue_submits) {
      venus->pending_submit_batches =
         CALLOC(venus->group_queue_submit_size,
                sizeof(*venus->pending_submit_batches));
      if (!venus->pending_submit_batches) {
         YTTRIUM_WARN("yttrium: ERROR: Venus2 grouped queue submit storage allocation failed owner=venus2 reason=out_of_memory slots=%u bytes=%llu\n",
                      venus->group_queue_submit_size,
                      (unsigned long long)
                         (venus->group_queue_submit_size *
                          sizeof(*venus->pending_submit_batches)));
         for (uint32_t i = 0; i < venus->batch_count; i++)
            FREE(venus->batches[i]);
         FREE(venus->batches);
         FREE(venus);
         return NULL;
      }
   }

   venus->device = device;
   venus->next_id = 1;
   venus->vn_ring.driver = venus;
   venus->vn_ring.submit_command = yttrium_venus2_vn_ring_submit_command;
   venus->vn_ring.get_command_reply = yttrium_venus2_vn_ring_get_command_reply;
   venus->vn_ring.free_command_reply = yttrium_venus2_vn_ring_free_command_reply;
   venus->compact_image_barriers =
      yttrium_gdi_debug_get_bool_option(
         YTTRIUM_VENUS_COMPACT_IMAGE_BARRIERS_ENV, true);
   venus->device_local_static_draw_buffers =
      yttrium_gdi_debug_get_bool_option(
         YTTRIUM_VENUS_DEVICE_LOCAL_STATIC_DRAW_BUFFERS_ENV, true);
   const char *draw_mirror_fault =
      getenv(YTTRIUM_VENUS_TEST_FAIL_AFTER_DRAW_MIRROR_COPY_ONCE_ENV);
   venus->test_fail_after_draw_mirror_copy_once =
      draw_mirror_fault && draw_mirror_fault[0] &&
      strcmp(draw_mirror_fault, "0") != 0;
   if (venus->test_fail_after_draw_mirror_copy_once) {
      YTTRIUM_WARN("yttrium: TEST: device-local draw-mirror fault injection enabled owner=venus2 source=process_environment variable=%s\n",
                   YTTRIUM_VENUS_TEST_FAIL_AFTER_DRAW_MIRROR_COPY_ONCE_ENV);
   }
   venus->draw_arena_bar =
      yttrium_gdi_debug_get_bool_option(
         YTTRIUM_VENUS_DRAW_ARENA_BAR_ENV, false);

   const uint32_t render_target_cache_capacity =
      yttrium_venus_render_target_cache_configured_capacity();
   if (render_target_cache_capacity) {
      venus->render_target_cache =
         CALLOC(render_target_cache_capacity,
                sizeof(*venus->render_target_cache));
      if (venus->render_target_cache) {
         venus->render_target_cache_capacity =
            render_target_cache_capacity;
      } else {
         YTTRIUM_WARN("yttrium: Venus render-target cache allocation failed owner=venus2 capacity=%u action=cache_disabled\n",
                      render_target_cache_capacity);
      }
   }
   return venus;
}

struct gdikmt_context *
yttrium_venus2_get_kmt_context(struct yttrium_venus *venus)
{
   if (!yttrium_venus_ensure_initialized(venus))
      return NULL;

   return venus->kmt_ctx;
}

bool
yttrium_venus2_wait_resource(struct yttrium_venus *venus,
                             struct yttrium_venus_resource *resource,
                             const char *label)
{
   return yttrium_venus_wait_resource_batches(venus, resource, label);
}

void
yttrium_venus2_destroy(struct yttrium_venus *venus)
{
   if (!venus)
      return;

   if (venus->initialized) {
      yttrium_venus_flush_command_batch(venus,
                                             "Venus destroy flush");
      yttrium_venus_cmd_batch_destroy_descriptor_pool(
         venus, &venus->cmd_batch_descriptor_pool);

      yttrium_venus2_resource_fini(venus, venus->kmt_ctx,
                                  &venus->null_sampled_image, NULL);
      yttrium_venus2_resource_fini(venus, venus->kmt_ctx,
                                  &venus->null_sampled_buffer, NULL);

      yttrium_venus_destroy_batches(venus);
      yttrium_venus_render_target_cache_destroy(venus);

      yttrium_venus_destroy_ubo_arenas(venus);
      if (venus->command_pool)
         vn_async_vkDestroyCommandPool(&venus->vn_ring, venus->device_handle,
                                       venus->command_pool, NULL);

      yttrium_venus_drain_ring(venus, "device object destroy");

      if (venus->device_handle)
         vn_async_vkDestroyDevice(&venus->vn_ring, venus->device_handle, NULL);

      yttrium_venus_drain_ring(venus, "device destroy");

   }

   if (venus->instance_initialized && venus->instance) {
      vn_async_vkDestroyInstance(&venus->vn_ring, venus->instance, NULL);
      yttrium_venus_drain_ring(venus, "instance destroy");
   }

   if (venus->ring.id) {
      uint8_t data[32];
      struct vn_cs_encoder enc =
         VN_CS_ENCODER_INITIALIZER_LOCAL(data, sizeof(data));
      vn_encode_vkDestroyRingMESA(&enc, 0, venus->ring.id);
      if (yttrium_venus_warn_encoder_overflow("destroy-ring", &enc, 0) ||
          !yttrium_venus_raw_submit_sync(venus, data,
                                         vn_cs_encoder_get_len(&enc),
                                         "destroy ring"))
         YTTRIUM_LOG("yttrium: Venus destroy ring sync submit failed ring_id=%llu\n",
                     (unsigned long long)venus->ring.id);
   }

   venus->failed = true;
   yttrium_venus_bo_destroy(venus, &venus->reply_bo);
   yttrium_venus_ring_forget_at_device_teardown(venus);

   if (venus->kmt_ctx)
      venus->kmt_ctx->destroy(venus->kmt_ctx);

   yttrium_venus_cmd_batch_destroy_footprint_index(venus);
   free(venus->cmd_batch_footprints);
   free(venus->cmd_batch_deferred_draws);
   free(venus->cmd_batch_compact_draw_packets);
   yttrium_venus_cmd_batch_clear_uploads(venus);
   free(venus->cmd_batch_uploads);
   free(venus->cmd_batch_upload_barriers);
   free(venus->cmd_batch_image_barriers);
   free(venus->cmd_batch_draw_mirror_updates);
   FREE(venus->render_target_cache);
   FREE(venus->pending_submit_batches);
   for (uint32_t i = 0; i < venus->batch_count; i++) {
      if (venus->batches[i])
         free(venus->batches[i]->draw_mirror_updates);
      FREE(venus->batches[i]);
   }
   FREE(venus->batches);
   FREE(venus);
}
