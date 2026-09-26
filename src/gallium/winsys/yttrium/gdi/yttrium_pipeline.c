/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#include "yttrium_pipeline.h"

#include <stdint.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "pipe/p_shader_tokens.h"
#include "util/bitscan.h"
#include "util/format/u_format.h"
#include "util/u_math.h"
#include "util/u_memory.h"

#include "yttrium_internal.h"
#include "yttrium_gdi_public.h"
#include "yttrium_options.h"
#include "yttrium_resource.h"
#include "yttrium_shader.h"
#include "yttrium_trace.h"

#ifndef VK_LOD_CLAMP_NONE
#define VK_LOD_CLAMP_NONE 1000.0f
#endif

#define YTTRIUM_PIPELINE_TIMING_SLOW_US 1000
#define YTTRIUM_FORCED_SAMPLE_INTERLOCK_IMAGE_SLOT 0
#define YTTRIUM_PIPELINE_CACHE_DEFAULT_SIZE 512
#define YTTRIUM_PIPELINE_CACHE_MAX_SIZE 4096

struct yttrium_compute_pipeline_cache_entry {
   struct yttrium_pipeline *pipeline;
   uint32_t shader_id;
   uint64_t module_id;
   uint32_t ubo_binding_count;
   struct yttrium_venus_ubo_binding_layout
      ubo_bindings[YTTRIUM_VENUS_MAX_PIPELINE_UBO_BINDINGS];
};

static uint64_t
yttrium_pipeline_vk_image_to_u64(VkImage image)
{
#if VK_USE_64_BIT_PTR_DEFINES
   return (uint64_t)(uintptr_t)image;
#else
   return (uint64_t)image;
#endif
}

static void
yttrium_pipeline_trace_timing(uint32_t point,
                              uint32_t status,
                              uint64_t start_us,
                              const char *label,
                              uint64_t a,
                              uint64_t b,
                              uint32_t c,
                              uint32_t d)
{
   if (!start_us)
      return;

   const uint64_t elapsed_us = yttrium_trace_now_us() - start_us;
   if (status || elapsed_us >= YTTRIUM_PIPELINE_TIMING_SLOW_US)
      yttrium_trace_timing(point, status, elapsed_us, label, a, b, c, d);
}

static bool
yttrium_pipeline_verbose_trace_enabled(void)
{
   static int enabled = -1;

   if (enabled < 0) {
      enabled = yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_VERBOSE_SAMPLER_DIAGNOSTIC", false) ? 1 : 0;
   }

   return enabled && yttrium_trace_is_enabled();
}

static bool
yttrium_pipeline_key_equal(const struct yttrium_pipeline_key *a,
                           const struct yttrium_pipeline_key *b)
{
   return memcmp(a, b, sizeof(*a)) == 0;
}

static uint32_t
yttrium_pipeline_hash_mix(uint32_t hash, uint64_t value)
{
   value ^= value >> 33;
   value *= UINT64_C(0xff51afd7ed558ccd);
   value ^= value >> 33;
   hash ^= (uint32_t)value ^ (uint32_t)(value >> 32);
   return hash * 16777619u;
}

static uint32_t
yttrium_pipeline_hash_bytes(uint32_t hash, const void *data, size_t size)
{
   const uint8_t *bytes = data;

   for (size_t i = 0; i < size; i++)
      hash = (hash ^ bytes[i]) * 16777619u;
   return hash;
}

static uint32_t
yttrium_pipeline_key_hash(const struct yttrium_pipeline_key *key)
{
   uint32_t hash = 2166136261u;

   hash = yttrium_pipeline_hash_mix(hash,
      (uint64_t)key->vs_id | ((uint64_t)key->tcs_id << 32));
   hash = yttrium_pipeline_hash_mix(hash,
      (uint64_t)key->tes_id | ((uint64_t)key->gs_id << 32));
   hash = yttrium_pipeline_hash_mix(hash,
      (uint64_t)key->fs_id | ((uint64_t)key->rt_count << 32));
   for (uint32_t i = 0; i < key->rt_count; i++) {
      hash = yttrium_pipeline_hash_mix(hash, key->dst_image_id[i]);
      hash = yttrium_pipeline_hash_mix(hash,
         (uint64_t)key->rt_format[i] |
         ((uint64_t)key->rt_level[i] << 32) |
         ((uint64_t)key->rt_layer[i] << 48));
   }
   hash = yttrium_pipeline_hash_mix(hash, key->zs_image_id);
   hash = yttrium_pipeline_hash_mix(hash,
      (uint64_t)key->zs_format | ((uint64_t)key->zs_level << 32) |
      ((uint64_t)key->zs_layer << 48));
   hash = yttrium_pipeline_hash_mix(hash,
      (uint64_t)key->topology | ((uint64_t)key->patch_vertices << 32) |
      ((uint64_t)key->rasterization_samples << 40));
   hash = yttrium_pipeline_hash_mix(hash,
      (uint64_t)key->primitive_restart_enable |
      ((uint64_t)key->rasterizer_discard_enable << 1) |
      ((uint64_t)key->targetless_stream_output << 2));
   hash = yttrium_pipeline_hash_mix(hash,
      (uint64_t)key->cull_mode | ((uint64_t)key->front_face << 32) |
      ((uint64_t)key->depth_test_enable << 40) |
      ((uint64_t)key->depth_write_enable << 41) |
      ((uint64_t)key->depth_read_only << 42) |
      ((uint64_t)key->depth_compare_op << 48));
   hash = yttrium_pipeline_hash_mix(hash,
      (uint64_t)key->sampled_stage_mask |
      ((uint64_t)key->sampled_binding_count << 32));
   hash = yttrium_pipeline_hash_bytes(
      hash, key->sampled_bindings,
      key->sampled_binding_count * sizeof(key->sampled_bindings[0]));
   hash = yttrium_pipeline_hash_mix(hash,
      key->storage_image_mask ^ key->storage_buffer_mask);
   hash = yttrium_pipeline_hash_mix(hash,
      (uint64_t)key->num_bindings | ((uint64_t)key->num_attribs << 32));
   for (uint32_t i = 0; i < key->num_attribs; i++) {
      hash = yttrium_pipeline_hash_mix(hash,
         (uint64_t)key->attribs[i].location |
         ((uint64_t)key->attribs[i].binding << 16) |
         ((uint64_t)key->attribs[i].format << 32));
   }

   return hash;
}

static uint32_t
yttrium_pipeline_cache_configured_size(void)
{
   int64_t size = yttrium_gdi_debug_get_num_option(
      "D3D10UMD_YTTRIUM_PIPELINE_CACHE_SIZE",
      YTTRIUM_PIPELINE_CACHE_DEFAULT_SIZE);

   size = CLAMP(size, 1, YTTRIUM_PIPELINE_CACHE_MAX_SIZE);
   return (uint32_t)size;
}

bool
yttrium_pipeline_cache_init(struct yttrium_context *yctx)
{
   if (!yctx)
      return false;

   yctx->pipeline_cache_size = yttrium_pipeline_cache_configured_size();
   yctx->pipeline_cache =
      CALLOC(yctx->pipeline_cache_size, sizeof(*yctx->pipeline_cache));
   yctx->pipeline_cache_hash_size = 1;
   while (yctx->pipeline_cache_hash_size < yctx->pipeline_cache_size * 2)
      yctx->pipeline_cache_hash_size <<= 1;
   yctx->pipeline_cache_hash_heads =
      MALLOC(yctx->pipeline_cache_hash_size *
             sizeof(*yctx->pipeline_cache_hash_heads));
   yctx->pipeline_cache_hash_next =
      MALLOC(yctx->pipeline_cache_size *
             sizeof(*yctx->pipeline_cache_hash_next));
   if (!yctx->pipeline_cache || !yctx->pipeline_cache_hash_heads ||
       !yctx->pipeline_cache_hash_next) {
      YTTRIUM_ERROR("yttrium: failed to allocate graphics pipeline cache with %u entries\n",
                    yctx->pipeline_cache_size);
      FREE(yctx->pipeline_cache);
      FREE(yctx->pipeline_cache_hash_heads);
      FREE(yctx->pipeline_cache_hash_next);
      yctx->pipeline_cache = NULL;
      yctx->pipeline_cache_hash_heads = NULL;
      yctx->pipeline_cache_hash_next = NULL;
      yctx->pipeline_cache_hash_size = 0;
      yctx->pipeline_cache_size = 0;
      return false;
   }

   for (uint32_t i = 0; i < yctx->pipeline_cache_hash_size; i++)
      yctx->pipeline_cache_hash_heads[i] = UINT32_MAX;
   for (uint32_t i = 0; i < yctx->pipeline_cache_size; i++)
      yctx->pipeline_cache_hash_next[i] = UINT32_MAX;

   return true;
}

static bool
yttrium_pipeline_nir_uses_sample_pos(const nir_shader *nir)
{
   if (!nir || nir->info.stage != MESA_SHADER_FRAGMENT)
      return false;

   nir_foreach_function_impl(impl, (nir_shader *)nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            /* Explicit interpolation selects an evaluation location; it does
             * not change a pixel-frequency shader to sample-frequency.
             */
            if (intr->intrinsic == nir_intrinsic_load_sample_id ||
                intr->intrinsic == nir_intrinsic_load_sample_pos ||
                intr->intrinsic == nir_intrinsic_load_sample_pos_or_center)
               return true;
         }
      }
   }

   return false;
}

static bool
yttrium_pipeline_fs_uses_sample_shading_uncached(
   const struct yttrium_shader_state *fs)
{
   if (fs->nir && (fs->nir->info.fs.uses_sample_shading ||
                   yttrium_pipeline_nir_uses_sample_pos(fs->nir)))
      return true;

   if (fs->info.opcode_count[TGSI_OPCODE_SAMPLE_POS])
      return true;

   for (uint8_t i = 0; i < fs->info.num_inputs; i++) {
      if (fs->info.input_interpolate_loc[i] == TGSI_INTERPOLATE_LOC_SAMPLE)
         return true;
   }

   for (uint8_t i = 0; i < fs->info.num_system_values; i++) {
      if (fs->info.system_value_semantic_name[i] == TGSI_SEMANTIC_SAMPLEID ||
          fs->info.system_value_semantic_name[i] == TGSI_SEMANTIC_SAMPLEPOS)
         return true;
   }

   return false;
}

/*
 * pipeline_build_key() asks this on every draw, and answering it walks every
 * instruction of the fragment shader's NIR - which made exec_node_is_tail_
 * sentinel one of the hottest functions in the driver.  The answer is a
 * property of the shader, so compute it once and keep it.
 */
static bool
yttrium_pipeline_fs_uses_sample_shading(struct yttrium_shader_state *fs)
{
   if (!fs || fs->stage != MESA_SHADER_FRAGMENT)
      return false;

   if (fs->uses_sample_shading < 0) {
      fs->uses_sample_shading =
         yttrium_pipeline_fs_uses_sample_shading_uncached(fs) ? 1 : 0;
   }

   return fs->uses_sample_shading != 0;
}

static uint64_t
yttrium_pipeline_a8_rt_fs_hash(uint64_t fs_hash, uint32_t a8_rt_mask)
{
   return fs_hash ^ 0xa8a8a8a800000000ull ^ (uint64_t)a8_rt_mask;
}

static uint64_t
yttrium_pipeline_dual_source_fs_hash(uint64_t fs_hash)
{
   return fs_hash ^ 0xd51d51d500000000ull;
}

static uint64_t
yttrium_pipeline_alpha_test_fs_hash(uint64_t fs_hash, uint32_t func,
                                    float ref)
{
   union {
      float f;
      uint32_t u;
   } ref_bits = { ref };

   return fs_hash ^ 0xa17e570000000000ull ^
          ((uint64_t)func << 32) ^ (uint64_t)ref_bits.u;
}

static uint64_t
yttrium_pipeline_forced_sample_mask_fs_hash(uint64_t fs_hash,
                                            uint32_t sample_count)
{
   return fs_hash ^ 0xf5a6c00000000000ull ^ (uint64_t)sample_count;
}

static uint64_t
yttrium_pipeline_sample_mask_expand_fs_hash(uint64_t fs_hash,
                                            uint32_t hw_sample_count,
                                            uint32_t app_sample_count)
{
   return fs_hash ^ 0x5e6a9d0000000000ull ^
          ((uint64_t)hw_sample_count << 32) ^ (uint64_t)app_sample_count;
}

static uint64_t
yttrium_pipeline_forced_sample_interlock_fs_hash(uint64_t fs_hash)
{
   return fs_hash ^ 0x1e7e12c000000000ull ^
          YTTRIUM_FORCED_SAMPLE_INTERLOCK_IMAGE_SLOT;
}

/* Highest forced sample count the app is allowed to reach.  D2D/XAML asks for
 * 16x target-independent rasterization, which no current Venus backend can
 * rasterize at; the cap exists so the limit can be moved for bring-up without
 * a rebuild.
 */
static uint32_t
yttrium_pipeline_forced_sample_count_limit(void)
{
   static int limit = -1;

   if (limit < 0) {
      limit = (int)yttrium_gdi_debug_get_num_option(
         "D3D10UMD_YTTRIUM_FORCED_SAMPLE_COUNT_MAX", 0);
      if (limit < 0)
         limit = 0;
   }

   return (uint32_t)limit;
}

/* Largest sample count <= `requested` that the device can rasterize at with
 * VK_EXT_multisampled_render_to_single_sampled.  Returns 0 when there is none.
 */
static uint32_t
yttrium_pipeline_supported_forced_sample_count(struct yttrium_context *yctx,
                                               uint32_t requested)
{
   struct yttrium_venus *venus = yttrium_screen(yctx->base.screen)->venus;
   const uint32_t limit = yttrium_pipeline_forced_sample_count_limit();

   if (requested <= 1 || !util_is_power_of_two_nonzero(requested))
      return 0;

   for (uint32_t samples = requested; samples > 1; samples >>= 1) {
      if (limit && samples > limit)
         continue;
      if (yttrium_venus_supports_multisampled_render_to_single_sampled(
             venus, samples))
         return samples;
   }

   return 0;
}

static uint32_t
yttrium_pipeline_supported_forced_sample_interlock_count(
   struct yttrium_context *yctx,
   uint32_t requested)
{
   struct yttrium_venus *venus = yttrium_screen(yctx->base.screen)->venus;
   const uint32_t limit = yttrium_pipeline_forced_sample_count_limit();

   if (requested <= 1 || !util_is_power_of_two_nonzero(requested))
      return 0;

   for (uint32_t samples = requested; samples > 1; samples >>= 1) {
      if (limit && samples > limit)
         continue;
      if (yttrium_venus_supports_forced_sample_interlock(venus, samples))
         return samples;
   }

   return 0;
}

static bool
yttrium_pipeline_can_use_forced_sample_interlock(
   struct yttrium_context *yctx,
   const struct yttrium_resource *dst,
   const struct yttrium_resource *zs,
   const struct yttrium_venus_draw_state *draw_state)
{
   struct yttrium_shader_state *fs =
      yctx ? yctx->shaders[MESA_SHADER_FRAGMENT] : NULL;
   const struct pipe_surface *cbuf =
      yctx && yctx->fb.nr_cbufs == 1 ? &yctx->fb.cbufs[0] : NULL;
   const struct yttrium_resource *rt =
      cbuf && cbuf->texture ? yttrium_resource(cbuf->texture) : NULL;
   struct yttrium_screen *screen =
      yctx ? yttrium_screen(yctx->base.screen) : NULL;

   return screen && dst && dst == rt && !zs && draw_state && fs && fs->nir &&
      draw_state->rt_count == 1 &&
      draw_state->forced_sample_count > 1 &&
      draw_state->logic_op_enable &&
      draw_state->logic_op == VK_LOGIC_OP_XOR &&
      !draw_state->rt_blend_enable[0] &&
      (draw_state->rt_color_write_mask[0] & VK_COLOR_COMPONENT_R_BIT) &&
      !draw_state->alpha_test_enable &&
      !draw_state->alpha_to_coverage_enable &&
      cbuf->format == PIPE_FORMAT_R16_UINT &&
      cbuf->level == 0 &&
      cbuf->first_layer == 0 && cbuf->last_layer == 0 &&
      dst->base.target == PIPE_TEXTURE_2D &&
      MAX2(dst->base.nr_samples, 1) == 1 &&
      dst->venus.vk_format == VK_FORMAT_R16_UINT &&
      (dst->venus.image_usage & VK_IMAGE_USAGE_STORAGE_BIT) &&
      !fs->nir->info.fs.uses_discard &&
      yttrium_shader_state_uses_sample_mask_in(fs) &&
      !yttrium_pipeline_fs_uses_sample_shading(fs) &&
      !yttrium_shader_state_image_used_mask(fs) &&
      !yctx->num_shader_images[MESA_SHADER_FRAGMENT] &&
      yttrium_pipeline_supported_forced_sample_interlock_count(
         yctx, draw_state->forced_sample_count) > 1;
}

static VkBlendFactor
yttrium_pipeline_x8_color_blend_factor(VkBlendFactor factor)
{
   switch (factor) {
   case VK_BLEND_FACTOR_DST_ALPHA:
      return VK_BLEND_FACTOR_ONE;
   case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA:
      return VK_BLEND_FACTOR_ZERO;
   default:
      return factor;
   }
}

static void
yttrium_pipeline_apply_x8_rt_state(
   const struct yttrium_pipeline_key *key,
   const struct yttrium_venus_draw_state *draw_state,
   struct yttrium_venus_draw_state *out_state)
{
   *out_state = *draw_state;

   for (uint32_t i = 0; i < key->rt_count; i++) {
      if (!(key->x8_rt_mask & (1u << i)))
         continue;

      out_state->rt_color_write_mask[i] &= ~VK_COLOR_COMPONENT_A_BIT;
      out_state->rt_src_color_blend_factor[i] =
         yttrium_pipeline_x8_color_blend_factor(
            out_state->rt_src_color_blend_factor[i]);
      out_state->rt_dst_color_blend_factor[i] =
         yttrium_pipeline_x8_color_blend_factor(
            out_state->rt_dst_color_blend_factor[i]);

      if (i == 0) {
         out_state->color_write_mask = out_state->rt_color_write_mask[i];
         out_state->src_color_blend_factor =
            out_state->rt_src_color_blend_factor[i];
         out_state->dst_color_blend_factor =
            out_state->rt_dst_color_blend_factor[i];
      }
   }
}

static bool
yttrium_pipeline_blend_factor_uses_src1(VkBlendFactor factor)
{
   return factor == VK_BLEND_FACTOR_SRC1_COLOR ||
          factor == VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR ||
          factor == VK_BLEND_FACTOR_SRC1_ALPHA ||
          factor == VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
}

static enum mesa_prim
yttrium_pipeline_prim_from_topology(VkPrimitiveTopology topology)
{
   switch (topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      return MESA_PRIM_POINTS;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
      return MESA_PRIM_LINES;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
      return MESA_PRIM_LINE_STRIP;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      return MESA_PRIM_TRIANGLES;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
      return MESA_PRIM_TRIANGLE_STRIP;
   default:
      return MESA_PRIM_COUNT;
   }
}

static bool
yttrium_pipeline_needs_cull_distance_gs(
   const struct yttrium_shader_state *vs,
   const struct yttrium_shader_state *gs,
   VkPrimitiveTopology topology)
{
   return !gs && vs && vs->nir &&
          vs->nir->info.cull_distance_array_size &&
          yttrium_pipeline_prim_from_topology(topology) != MESA_PRIM_COUNT;
}

static uint64_t
yttrium_pipeline_generated_cull_gs_hash(uint64_t vs_hash,
                                        VkPrimitiveTopology topology)
{
   return vs_hash ^ 0xc311d157accd1575ull ^
          ((uint64_t)topology << 32) ^ (uint64_t)topology;
}

static uint32_t
yttrium_pipeline_peek_u32(const void *data, size_t size, unsigned index)
{
   uint32_t value = 0;
   const size_t offset = (size_t)index * sizeof(value);

   if (data && offset <= size && sizeof(value) <= size - offset)
      memcpy(&value, (const uint8_t *)data + offset, sizeof(value));

   return value;
}

static uint32_t
yttrium_pipeline_peek_u32_at(const void *data, size_t size, uint64_t offset)
{
   uint32_t value = 0;

   if (data && offset <= size && sizeof(value) <= size - (size_t)offset)
      memcpy(&value, (const uint8_t *)data + offset, sizeof(value));

   return value;
}

static float
yttrium_pipeline_peek_f32(const void *data, size_t size, unsigned index)
{
   float value = 0.0f;
   const size_t offset = (size_t)index * sizeof(value);

   if (data && offset <= size && sizeof(value) <= size - offset)
      memcpy(&value, (const uint8_t *)data + offset, sizeof(value));

   return value;
}

struct yttrium_pipeline_data_summary {
   uint64_t inspected;
   uint64_t nonzero_bytes;
   uint64_t first_nonzero;
   uint64_t last_nonzero;
   uint32_t first_nonzero_u32;
   uint32_t last_nonzero_u32;
   uint32_t hash;
};

static struct yttrium_pipeline_data_summary
yttrium_pipeline_summarize_data(const void *data, size_t size)
{
   struct yttrium_pipeline_data_summary summary = {
      .first_nonzero = UINT64_MAX,
      .last_nonzero = UINT64_MAX,
      .hash = 2166136261u,
   };
   const uint8_t *bytes = data;
   const size_t inspected = MIN2(size, (size_t)4 * 1024 * 1024);

   summary.inspected = inspected;
   if (!bytes)
      return summary;

   for (size_t i = 0; i < inspected; i++) {
      const uint8_t value = bytes[i];
      summary.hash ^= value;
      summary.hash *= 16777619u;
      if (value) {
         summary.nonzero_bytes++;
         if (summary.first_nonzero == UINT64_MAX)
            summary.first_nonzero = i;
         summary.last_nonzero = i;
      }
   }

   if (summary.first_nonzero != UINT64_MAX)
      summary.first_nonzero_u32 =
         yttrium_pipeline_peek_u32_at(data, size, summary.first_nonzero);
   if (summary.last_nonzero != UINT64_MAX)
      summary.last_nonzero_u32 =
         yttrium_pipeline_peek_u32_at(data, size, summary.last_nonzero);

   return summary;
}

static uint32_t
yttrium_pipeline_peek_index(const void *data, size_t size, unsigned index,
                            VkIndexType type)
{
   if (!data)
      return 0;

   switch (type) {
   case VK_INDEX_TYPE_UINT16: {
      uint16_t value = 0;
      const size_t offset = (size_t)index * sizeof(value);
      if (offset <= size && sizeof(value) <= size - offset)
         memcpy(&value, (const uint8_t *)data + offset, sizeof(value));
      return value;
   }
   case VK_INDEX_TYPE_UINT32:
      return yttrium_pipeline_peek_u32(data, size, index);
   default:
      return 0;
   }
}

static enum pipe_format
yttrium_pipeline_format_as_unorm(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8_UINT:
      return PIPE_FORMAT_R8_UNORM;
   case PIPE_FORMAT_R8G8_UINT:
      return PIPE_FORMAT_R8G8_UNORM;
   case PIPE_FORMAT_R8G8B8_UINT:
      return PIPE_FORMAT_R8G8B8_UNORM;
   case PIPE_FORMAT_R8G8B8A8_UINT:
      return PIPE_FORMAT_R8G8B8A8_UNORM;
   default:
      return format;
   }
}

static enum pipe_format
yttrium_pipeline_format_as_snorm(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8_SINT:
      return PIPE_FORMAT_R8_SNORM;
   case PIPE_FORMAT_R8G8_SINT:
      return PIPE_FORMAT_R8G8_SNORM;
   case PIPE_FORMAT_R8G8B8_SINT:
      return PIPE_FORMAT_R8G8B8_SNORM;
   case PIPE_FORMAT_R8G8B8A8_SINT:
      return PIPE_FORMAT_R8G8B8A8_SNORM;
   default:
      return format;
   }
}

static bool
yttrium_pipeline_sampled_buffer_uses_r8_bitcast_coords(
   const struct yttrium_shader_state *shader,
   uint32_t slot,
   enum pipe_format view_format)
{
   if (!shader || slot >= ARRAY_SIZE(shader->info.sampler_type) ||
       slot >= ARRAY_SIZE(shader->info.sampler_targets))
      return false;

   return shader->info.sampler_targets[slot] == TGSI_TEXTURE_BUFFER &&
          shader->info.sampler_type[slot] == TGSI_RETURN_TYPE_UNORM &&
          view_format == PIPE_FORMAT_R8_UINT;
}

static enum pipe_format
yttrium_pipeline_sampled_buffer_format(
   const struct yttrium_shader_state *shader,
   uint32_t slot,
   enum pipe_format view_format)
{
   if (!shader || slot >= ARRAY_SIZE(shader->info.sampler_type))
      return view_format;

   const bool mixed_return =
      slot < 32 && (shader->sampler_mixed_return_mask & (1u << slot)) &&
      slot < ARRAY_SIZE(shader->info.sampler_targets) &&
      shader->info.sampler_targets[slot] == TGSI_TEXTURE_BUFFER &&
      shader->info.sampler_type[slot] == TGSI_RETURN_TYPE_FLOAT;
   if (mixed_return) {
      switch (view_format) {
      case PIPE_FORMAT_R32G32B32A32_SINT:
      case PIPE_FORMAT_R32G32B32A32_UINT:
         return PIPE_FORMAT_R32G32B32A32_FLOAT;
      default:
         break;
      }
   }

   if (yttrium_pipeline_sampled_buffer_uses_r8_bitcast_coords(
          shader, slot, view_format))
      return PIPE_FORMAT_R32_FLOAT;

   switch (shader->info.sampler_type[slot]) {
   case TGSI_RETURN_TYPE_UNORM:
      return yttrium_pipeline_format_as_unorm(view_format);
   case TGSI_RETURN_TYPE_SNORM:
      return yttrium_pipeline_format_as_snorm(view_format);
   default:
      return view_format;
   }
}

static enum pipe_swizzle
yttrium_pipeline_sample_swizzle_for_format(enum pipe_format format,
                                           unsigned swizzle)
{
   const enum pipe_swizzle normalized =
      yttrium_venus_sample_swizzle_normalize(swizzle, PIPE_SWIZZLE_X);
   const struct util_format_description *desc =
      util_format_description(format);

   if (format == PIPE_FORMAT_A8_UNORM) {
      if (normalized == PIPE_SWIZZLE_W)
         return PIPE_SWIZZLE_X;
      if (normalized == PIPE_SWIZZLE_X ||
          normalized == PIPE_SWIZZLE_Y ||
          normalized == PIPE_SWIZZLE_Z)
         return PIPE_SWIZZLE_0;
   }

   if (normalized == PIPE_SWIZZLE_W && desc &&
       desc->swizzle[3] == PIPE_SWIZZLE_1)
      return PIPE_SWIZZLE_1;

   return normalized;
}

static uint32_t
yttrium_pipeline_sample_swizzle_key(enum pipe_format format,
                                    const struct pipe_sampler_view *view)
{
   if (!view)
      return YTTRIUM_VENUS_SAMPLE_SWIZZLE_IDENTITY;

   return yttrium_venus_sample_swizzle_key(
      yttrium_pipeline_sample_swizzle_for_format(format, view->swizzle_r),
      yttrium_pipeline_sample_swizzle_for_format(format, view->swizzle_g),
      yttrium_pipeline_sample_swizzle_for_format(format, view->swizzle_b),
      yttrium_pipeline_sample_swizzle_for_format(format, view->swizzle_a));
}

static VkFilter
yttrium_pipeline_sampler_filter(unsigned filter)
{
   return filter == PIPE_TEX_FILTER_LINEAR ? VK_FILTER_LINEAR :
                                             VK_FILTER_NEAREST;
}

static VkSamplerMipmapMode
yttrium_pipeline_sampler_mipmap_mode(unsigned filter)
{
   return filter == PIPE_TEX_MIPFILTER_LINEAR ?
      VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
}

static VkSamplerAddressMode
yttrium_pipeline_sampler_address_mode(unsigned mode)
{
   switch (mode) {
   case PIPE_TEX_WRAP_REPEAT:
      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
   case PIPE_TEX_WRAP_MIRROR_REPEAT:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
   case PIPE_TEX_WRAP_CLAMP_TO_BORDER:
   case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_BORDER:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
   case PIPE_TEX_WRAP_MIRROR_CLAMP:
   case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE:
      return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
   case PIPE_TEX_WRAP_CLAMP:
   case PIPE_TEX_WRAP_CLAMP_TO_EDGE:
   default:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
   }
}

static VkCompareOp
yttrium_pipeline_sampler_compare_op(unsigned func)
{
   switch (func) {
   case PIPE_FUNC_NEVER:
      return VK_COMPARE_OP_NEVER;
   case PIPE_FUNC_LESS:
      return VK_COMPARE_OP_LESS;
   case PIPE_FUNC_EQUAL:
      return VK_COMPARE_OP_EQUAL;
   case PIPE_FUNC_LEQUAL:
      return VK_COMPARE_OP_LESS_OR_EQUAL;
   case PIPE_FUNC_GREATER:
      return VK_COMPARE_OP_GREATER;
   case PIPE_FUNC_NOTEQUAL:
      return VK_COMPARE_OP_NOT_EQUAL;
   case PIPE_FUNC_GEQUAL:
      return VK_COMPARE_OP_GREATER_OR_EQUAL;
   case PIPE_FUNC_ALWAYS:
   default:
      return VK_COMPARE_OP_ALWAYS;
   }
}

static float
yttrium_pipeline_sampler_lod(float lod)
{
   if (lod > VK_LOD_CLAMP_NONE)
      return VK_LOD_CLAMP_NONE;
   if (lod < -VK_LOD_CLAMP_NONE)
      return -VK_LOD_CLAMP_NONE;
   return lod;
}

static void
yttrium_pipeline_default_sampler_state(
   struct yttrium_venus_sampler_state *out)
{
   *out = (struct yttrium_venus_sampler_state) {
      .min_filter = VK_FILTER_NEAREST,
      .mag_filter = VK_FILTER_NEAREST,
      .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .compare_op = VK_COMPARE_OP_ALWAYS,
      .border_color = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
      .min_lod = 0.0f,
      .max_lod = 0.0f,
      .max_anisotropy = 1.0f,
   };
}

static void
yttrium_pipeline_sampler_state_from_pipe(
   const struct pipe_sampler_state *state,
   struct yttrium_venus_sampler_state *out)
{
   yttrium_pipeline_default_sampler_state(out);
   if (!state)
      return;

   out->min_filter =
      yttrium_pipeline_sampler_filter(state->min_img_filter);
   out->mag_filter =
      yttrium_pipeline_sampler_filter(state->mag_img_filter);
   out->mipmap_mode =
      yttrium_pipeline_sampler_mipmap_mode(state->min_mip_filter);
   out->address_mode_u =
      yttrium_pipeline_sampler_address_mode(state->wrap_s);
   out->address_mode_v =
      yttrium_pipeline_sampler_address_mode(state->wrap_t);
   out->address_mode_w =
      yttrium_pipeline_sampler_address_mode(state->wrap_r);
   out->mip_lod_bias = state->lod_bias;
   out->min_lod = yttrium_pipeline_sampler_lod(state->min_lod);
   out->max_lod = yttrium_pipeline_sampler_lod(state->max_lod);
   if (out->max_lod < out->min_lod)
      out->max_lod = out->min_lod;
   out->max_anisotropy = state->max_anisotropy ?
      (float)state->max_anisotropy : 1.0f;
   out->anisotropy_enable = state->max_anisotropy > 1;
   out->compare_enable =
      state->compare_mode == PIPE_TEX_COMPARE_R_TO_TEXTURE;
   out->compare_op =
      yttrium_pipeline_sampler_compare_op(state->compare_func);
}

static void
yttrium_pipeline_sampler_state_force_integer_fetch(
   struct yttrium_venus_sampler_state *state)
{
   state->min_filter = VK_FILTER_NEAREST;
   state->mag_filter = VK_FILTER_NEAREST;
   state->mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
   state->anisotropy_enable = VK_FALSE;
   state->max_anisotropy = 1.0f;
}

static bool
yttrium_pipeline_sampler_border_color(
   const struct pipe_sampler_state *state,
   const struct pipe_sampler_view *view,
   struct yttrium_venus_sampler_state *out)
{
   const bool integer = view && util_format_is_pure_integer(view->format);
   out->border_color = integer ? VK_BORDER_COLOR_INT_TRANSPARENT_BLACK :
                                VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;

   /* Border values do not affect non-border addressing.  Keep their cache
    * keys canonical and do not require a custom-border capability for them.
    * Null views retain the existing zero-result resource handling as well.
    */
   if (!state || !view ||
       (out->address_mode_u != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER &&
        out->address_mode_v != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER &&
        out->address_mode_w != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER))
      return true;

   const union pipe_color_union *color = &state->border_color;
   const bool zero_bits = !color->ui[0] && !color->ui[1] &&
                          !color->ui[2] && !color->ui[3];
   if (state->border_color_is_integer != integer && !zero_bits) {
      YTTRIUM_WARN("yttrium: sampler border rejected owner=yttrium-pipeline "
                   "reason=border-numeric-type-mismatch format=%u "
                   "border_integer=%u view_integer=%u action=reject-draw\n",
                   view->format, state->border_color_is_integer, integer);
      return false;
   }

   /* Comparison sampling only observes red.  In particular, (1, 0, 0, 0)
    * is the same white depth border as (1, 1, 1, 1), without a custom sampler.
    */
   if (out->compare_enable && !integer &&
       yttrium_pipeline_sample_swizzle_for_format(view->format,
                                                  view->swizzle_r) ==
          PIPE_SWIZZLE_X) {
      if (color->f[0] == 0.0f)
         return true;
      if (color->f[0] == 1.0f) {
         out->border_color = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
         return true;
      }
   }

   const bool black_rgb = integer ?
      (!color->ui[0] && !color->ui[1] && !color->ui[2]) :
      (color->f[0] == 0.0f && color->f[1] == 0.0f && color->f[2] == 0.0f);
   const bool zero_alpha = integer ? color->ui[3] == 0 : color->f[3] == 0.0f;
   const bool one_alpha = integer ? color->ui[3] == 1 : color->f[3] == 1.0f;
   const bool white_rgb = integer ?
      (color->ui[0] == 1 && color->ui[1] == 1 && color->ui[2] == 1) :
      (color->f[0] == 1.0f && color->f[1] == 1.0f && color->f[2] == 1.0f);

   if (black_rgb && zero_alpha)
      return true;

   if (black_rgb && one_alpha) {
      out->border_color = integer ? VK_BORDER_COLOR_INT_OPAQUE_BLACK :
                                   VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
   } else if (white_rgb && one_alpha) {
      out->border_color = integer ? VK_BORDER_COLOR_INT_OPAQUE_WHITE :
                                   VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
   } else {
      out->border_color = integer ? VK_BORDER_COLOR_INT_CUSTOM_EXT :
                                   VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
      memcpy(&out->custom_border_color, color, sizeof(out->custom_border_color));
   }

   /* Opaque black/custom borders with a nonidentity image-view mapping need
    * VK_EXT_border_color_swizzle.  A8 additionally needs its alpha packed
    * into the R8 backing image's border red channel.  Do not silently sample
    * the wrong components until those paths have their own coverage.
    */
   const bool white = white_rgb && one_alpha;
   if (view->format == PIPE_FORMAT_A8_UNORM ||
       (!white && yttrium_pipeline_sample_swizzle_key(view->format, view) !=
                     YTTRIUM_VENUS_SAMPLE_SWIZZLE_IDENTITY)) {
      YTTRIUM_WARN("yttrium: sampler border rejected owner=yttrium-pipeline "
                   "reason=border-view-swizzle-unsupported format=%u "
                   "border=%u action=reject-draw\n",
                   view->format, out->border_color);
      return false;
   }

   /* Vulkan forbids formatless custom borders on these packed formats. */
   const bool custom = out->border_color == VK_BORDER_COLOR_INT_CUSTOM_EXT ||
                       out->border_color == VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
   if (custom &&
       (view->format == PIPE_FORMAT_B5G6R5_UNORM ||
        view->format == PIPE_FORMAT_B5G5R5A1_UNORM)) {
      YTTRIUM_WARN("yttrium: sampler border rejected owner=yttrium-pipeline "
                   "reason=formatless-border-packed-format format=%u "
                   "action=reject-draw\n", view->format);
      return false;
   }
   return true;
}

static void *
yttrium_pipeline_expand_r8_to_r32_float_bits(const void *data,
                                             uint64_t byte_count,
                                             size_t *out_size)
{
   if (out_size)
      *out_size = 0;

   if (!data || byte_count > SIZE_MAX / sizeof(uint32_t))
      return NULL;

   const size_t src_size = (size_t)byte_count;
   uint32_t *expanded = MALLOC(src_size * sizeof(uint32_t));
   if (!expanded)
      return NULL;

   const uint8_t *src = (const uint8_t *)data;
   for (size_t i = 0; i < src_size; i++)
      expanded[i] = src[i];

   if (out_size)
      *out_size = src_size * sizeof(uint32_t);
   return expanded;
}

static void
yttrium_pipeline_trace_sampled_texture_fail(
   const char *reason,
   uint32_t slot,
   uint32_t binding,
   uint32_t sampler_mask,
   const struct yttrium_resource *dst,
   const struct pipe_sampler_view *view,
   const struct yttrium_resource *src)
{
   yttrium_trace_debug_stringf(
      "yttrium: shader_draw_probe sampled texture collect fail reason=%s slot=%u binding=%u sampler_mask=0x%x view=%p texture=%p view_format=%u view_target=%u view_buf_offset=%u view_buf_size=%u dst_res_id=%u dst_image_id=%llu dst_display=%u dst_primary=%u dst_classic=%u dst_init=%u dst_buffer=%u dst_image=0x%llx dst_layout=%u src_res_id=%u src_image_id=%llu src_display=%u src_primary=%u src_classic=%u src_owned=%u src_imported=%u src_init=%u src_contents=%u src_buffer=%u src_image=0x%llx src_layout=%u src_usage=0x%x src_vk_format=%u src_alloc=0x%lx src_resource=%p src_mem=0x%llx src_data=%p src_map=%p src_map_blob=%u src_size=0x%llx src_stride=%u src_extent=%ux%u src_target=%u src_format=%u",
      reason ? reason : "<null>",
      slot,
      binding,
      sampler_mask,
      (const void *)view,
      view ? (const void *)view->texture : NULL,
      view ? view->format : PIPE_FORMAT_NONE,
      view ? view->target : 0,
      view ? view->u.buf.offset : 0,
      view ? view->u.buf.size : 0,
      dst ? dst->venus_res_id : 0,
      (unsigned long long)(dst ? dst->venus.image_obj.id : 0),
      dst ? dst->display_target : 0,
      dst ? dst->primary_target : 0,
      dst ? dst->classic_display : 0,
      dst ? dst->venus.initialized : 0,
      dst ? dst->venus.buffer_backed : 0,
      (unsigned long long)(dst ?
         yttrium_pipeline_vk_image_to_u64(dst->venus.image) : 0),
      dst ? dst->venus.layout : VK_IMAGE_LAYOUT_UNDEFINED,
      src ? src->venus_res_id : 0,
      (unsigned long long)(src ? src->venus.image_obj.id : 0),
      src ? src->display_target : 0,
      src ? src->primary_target : 0,
      src ? src->classic_display : 0,
      src ? src->owns_allocation : 0,
      src ? !src->owns_allocation : 0,
      src ? src->venus.initialized : 0,
      src ? src->venus.contents_initialized : 0,
      src ? src->venus.buffer_backed : 0,
      (unsigned long long)(src ?
         yttrium_pipeline_vk_image_to_u64(src->venus.image) : 0),
      src ? src->venus.layout : VK_IMAGE_LAYOUT_UNDEFINED,
      src ? src->venus.image_usage : 0,
      src ? src->venus.vk_format : VK_FORMAT_UNDEFINED,
      (unsigned long)(src ? src->hAllocation : 0),
      src ? src->hResource : NULL,
      (unsigned long long)(src ? src->venus_mem_id : 0),
      src ? src->data : NULL,
      src ? src->map : NULL,
      src ? src->map_is_blob : 0,
      (unsigned long long)(src ? src->size : 0),
      src ? src->stride : 0,
      src ? src->base.width0 : 0,
      src ? src->base.height0 : 0,
      src ? src->base.target : 0,
      src ? src->base.format : PIPE_FORMAT_NONE);
}

static void
yttrium_pipeline_trace_draw_state(const struct yttrium_resource *dst,
                                  const struct yttrium_resource *zs,
                                  const struct yttrium_venus_draw_state *state)
{
   if (!state)
      return;

   yttrium_trace_debug_stringf(
      "yttrium: shader_draw_probe draw state dst_res_id=%u zs_res_id=%u viewports=%u viewport0=%f,%f %fx%f depth=%f..%f scissor0=%d,%d %ux%u topology=%u restart=%u cull=0x%x front=%u blend=%u color_mask=0x%x sample_mask=0x%x rgb=(%u,%u,%u) alpha=(%u,%u,%u) depth_test=%u depth_write=%u depth_compare=%u alpha_test=%u alpha_func=%u alpha_ref=%f stencil=%u",
      dst ? dst->venus_res_id : 0,
      zs ? zs->venus_res_id : 0,
      state->viewport_count,
      state->viewports[0].x,
      state->viewports[0].y,
      state->viewports[0].width,
      state->viewports[0].height,
      state->viewports[0].minDepth,
      state->viewports[0].maxDepth,
      state->scissors[0].offset.x,
      state->scissors[0].offset.y,
      state->scissors[0].extent.width,
      state->scissors[0].extent.height,
      state->topology,
      state->primitive_restart_enable,
      state->cull_mode,
      state->front_face,
      state->blend_enable,
      state->color_write_mask,
      state->sample_mask,
      state->src_color_blend_factor,
      state->dst_color_blend_factor,
      state->color_blend_op,
      state->src_alpha_blend_factor,
      state->dst_alpha_blend_factor,
      state->alpha_blend_op,
      state->depth_test_enable,
      state->depth_write_enable,
      state->depth_compare_op,
      state->alpha_test_enable,
      state->alpha_func,
      state->alpha_ref_value,
      state->stencil_test_enable);
}

void
yttrium_pipeline_destroy(struct yttrium_venus *venus,
                         struct yttrium_pipeline *pipeline)
{
   if (!pipeline)
      return;

   yttrium_venus_pipeline_fini(venus, pipeline);
   if (pipeline->generated_vs) {
      if (pipeline->generated_vs->module)
         yttrium_venus_destroy_shader_module(
            venus, &pipeline->generated_vs->module_obj,
            pipeline->generated_vs->module);
      pipeline->generated_vs->module = VK_NULL_HANDLE;
      yttrium_shader_state_destroy(NULL, pipeline->generated_vs);
   }
   if (pipeline->generated_gs) {
      if (pipeline->generated_gs->module)
         yttrium_venus_destroy_shader_module(
            venus, &pipeline->generated_gs->module_obj,
            pipeline->generated_gs->module);
      pipeline->generated_gs->module = VK_NULL_HANDLE;
      yttrium_shader_state_destroy(NULL, pipeline->generated_gs);
   }
   if (pipeline->generated_fs) {
      if (pipeline->generated_fs->module)
         yttrium_venus_destroy_shader_module(
            venus, &pipeline->generated_fs->module_obj,
            pipeline->generated_fs->module);
      pipeline->generated_fs->module = VK_NULL_HANDLE;
      yttrium_shader_state_destroy(NULL, pipeline->generated_fs);
   }
   for (uint32_t i = 0; i < PIPE_MAX_COLOR_BUFS; i++)
      pipe_resource_reference(&pipeline->rt_resources[i], NULL);
   pipe_resource_reference(&pipeline->zs_resource, NULL);
   for (uint32_t i = 0;
        i < YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES; i++)
      pipe_resource_reference(
         &pipeline->sampled_descriptor_cache_resources[i], NULL);
   FREE(pipeline);
}

static void
yttrium_pipeline_cache_hash_remove(struct yttrium_context *yctx,
                                   uint32_t cache_slot)
{
   struct yttrium_pipeline *pipeline = yctx->pipeline_cache[cache_slot];
   if (!pipeline || !yctx->pipeline_cache_hash_size)
      return;

   const uint32_t bucket =
      pipeline->key_hash & (yctx->pipeline_cache_hash_size - 1);
   uint32_t *link = &yctx->pipeline_cache_hash_heads[bucket];
   while (*link != UINT32_MAX) {
      if (*link == cache_slot) {
         *link = yctx->pipeline_cache_hash_next[cache_slot];
         yctx->pipeline_cache_hash_next[cache_slot] = UINT32_MAX;
         return;
      }
      link = &yctx->pipeline_cache_hash_next[*link];
   }
}

static void
yttrium_pipeline_cache_hash_insert(struct yttrium_context *yctx,
                                   uint32_t cache_slot)
{
   struct yttrium_pipeline *pipeline = yctx->pipeline_cache[cache_slot];
   const uint32_t bucket =
      pipeline->key_hash & (yctx->pipeline_cache_hash_size - 1);
   yctx->pipeline_cache_hash_next[cache_slot] =
      yctx->pipeline_cache_hash_heads[bucket];
   yctx->pipeline_cache_hash_heads[bucket] = cache_slot;
}

static uint32_t
yttrium_pipeline_resource_reference_count(
   const struct yttrium_pipeline *pipeline,
   const struct pipe_resource *resource)
{
   if (!pipeline || !resource)
      return 0;

   uint32_t count = pipeline->zs_resource == resource ? 1 : 0;
   for (uint32_t i = 0; i < PIPE_MAX_COLOR_BUFS; i++) {
      if (pipeline->rt_resources[i] == resource)
         count++;
   }
   for (uint32_t i = 0;
        i < YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES; i++) {
      if (pipeline->sampled_descriptor_cache_resources[i] == resource)
         count++;
   }
   return count;
}

uint32_t
yttrium_gdi_pipeline_invalidate_resource(
   struct pipe_context *ctx, const struct pipe_resource *resource)
{
   if (!ctx || !resource)
      return 0;

   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   uint32_t released_refs = 0;
   uint32_t write_slot = 0;
   bool current_was_cached = false;

   for (uint32_t i = 0; i < yctx->pipeline_cache_count; i++) {
      struct yttrium_pipeline *pipeline = yctx->pipeline_cache[i];
      const uint32_t pipeline_refs =
         yttrium_pipeline_resource_reference_count(pipeline, resource);
      current_was_cached |= pipeline == yctx->pipeline;
      if (pipeline_refs) {
         if (pipeline == yctx->pipeline)
            yctx->pipeline = NULL;
         released_refs += pipeline_refs;
         yttrium_pipeline_destroy(screen->venus, pipeline);
      } else {
         yctx->pipeline_cache[write_slot++] = pipeline;
      }
   }

   if (!current_was_cached) {
      const uint32_t current_refs =
         yttrium_pipeline_resource_reference_count(yctx->pipeline, resource);
      if (current_refs) {
         released_refs += current_refs;
         yttrium_pipeline_destroy(screen->venus, yctx->pipeline);
         yctx->pipeline = NULL;
      }
   }

   if (!released_refs)
      return 0;

   for (uint32_t i = write_slot; i < yctx->pipeline_cache_count; i++)
      yctx->pipeline_cache[i] = NULL;
   yctx->pipeline_cache_count = write_slot;
   yctx->pipeline_cache_next = 0;

   for (uint32_t i = 0; i < yctx->pipeline_cache_hash_size; i++)
      yctx->pipeline_cache_hash_heads[i] = UINT32_MAX;
   for (uint32_t i = 0; i < yctx->pipeline_cache_size; i++)
      yctx->pipeline_cache_hash_next[i] = UINT32_MAX;
   for (uint32_t i = 0; i < yctx->pipeline_cache_count; i++)
      yttrium_pipeline_cache_hash_insert(yctx, i);

   return released_refs;
}

static void
yttrium_pipeline_invalidate(struct yttrium_context *yctx)
{
   if (!yctx)
      return;

   struct yttrium_screen *screen = yttrium_screen(yctx->base.screen);
   for (uint32_t i = 0; i < yctx->pipeline_cache_count; i++) {
      if (yctx->pipeline_cache[i] == yctx->pipeline)
         yctx->pipeline = NULL;
      yttrium_pipeline_destroy(screen->venus, yctx->pipeline_cache[i]);
      yctx->pipeline_cache[i] = NULL;
   }

   if (yctx->pipeline)
      yttrium_pipeline_destroy(screen->venus, yctx->pipeline);
   yctx->pipeline = NULL;
   yctx->pipeline_cache_count = 0;
   yctx->pipeline_cache_next = 0;
   for (uint32_t i = 0; i < yctx->pipeline_cache_hash_size; i++)
      yctx->pipeline_cache_hash_heads[i] = UINT32_MAX;
}

void
yttrium_pipeline_cache_fini(struct yttrium_context *yctx)
{
   if (!yctx)
      return;

   yttrium_compute_pipeline_cache_invalidate(yctx, NULL);
   yttrium_pipeline_invalidate(yctx);
   FREE(yctx->pipeline_cache);
   FREE(yctx->pipeline_cache_hash_heads);
   FREE(yctx->pipeline_cache_hash_next);
   yctx->pipeline_cache = NULL;
   yctx->pipeline_cache_hash_heads = NULL;
   yctx->pipeline_cache_hash_next = NULL;
   yctx->pipeline_cache_hash_size = 0;
   yctx->pipeline_cache_size = 0;
}

void
yttrium_pipeline_fast_state_changed(struct yttrium_context *yctx)
{
   if (!yctx)
      return;

   yctx->pipeline_state_serial++;
   if (!yctx->pipeline_state_serial)
      yctx->pipeline_state_serial++;
}

void
yttrium_pipeline_state_changed(struct yttrium_context *yctx)
{
   if (!yctx)
      return;

   yttrium_pipeline_fast_state_changed(yctx);

   /* Every immutable graphics-pipeline input is represented in the key.
    * Dynamic draw state is emitted separately.  Clear only the current fast
    * pointer so the next draw performs a complete-key cache lookup.
    */
   yctx->pipeline = NULL;
}

static VkShaderStageFlags
yttrium_pipeline_vk_shader_stage(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return VK_SHADER_STAGE_VERTEX_BIT;
   case MESA_SHADER_TESS_CTRL:
      return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
   case MESA_SHADER_TESS_EVAL:
      return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
   case MESA_SHADER_GEOMETRY:
      return VK_SHADER_STAGE_GEOMETRY_BIT;
   case MESA_SHADER_FRAGMENT:
      return VK_SHADER_STAGE_FRAGMENT_BIT;
   case MESA_SHADER_COMPUTE:
      return VK_SHADER_STAGE_COMPUTE_BIT;
   default:
      return 0;
   }
}

static VkShaderStageFlags
yttrium_pipeline_sampled_stage_flags(uint32_t sampled_stage_mask)
{
   VkShaderStageFlags flags = 0;
   const mesa_shader_stage stages[] = {
      MESA_SHADER_VERTEX,
      MESA_SHADER_TESS_CTRL,
      MESA_SHADER_TESS_EVAL,
      MESA_SHADER_GEOMETRY,
      MESA_SHADER_FRAGMENT,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(stages); i++) {
      const mesa_shader_stage stage = stages[i];
      if (sampled_stage_mask & (1u << stage))
         flags |= yttrium_pipeline_vk_shader_stage(stage);
   }

   return flags;
}

static mesa_shader_stage
yttrium_pipeline_first_graphics_stage(uint32_t stage_mask)
{
   const mesa_shader_stage stages[] = {
      MESA_SHADER_VERTEX,
      MESA_SHADER_TESS_CTRL,
      MESA_SHADER_TESS_EVAL,
      MESA_SHADER_GEOMETRY,
      MESA_SHADER_FRAGMENT,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(stages); i++) {
      const mesa_shader_stage stage = stages[i];
      if (stage_mask & (1u << stage))
         return stage;
   }

   return MESA_SHADER_NONE;
}

static bool
yttrium_pipeline_shader_supported(const struct yttrium_shader_state *shader)
{
   if (!shader || !yttrium_shader_state_has_module(shader))
      return false;

   if (yttrium_shader_state_is_resource_free(shader))
      return true;

   if (shader->ubo_used_mask && !yttrium_shader_pipeline_enabled()) {
      YTTRIUM_WARN("yttrium: shader_draw_probe pipeline shader skipped reason=ubo_pipeline_disabled stage=%s id=%u ubo_mask=0x%x\n",
                   yttrium_shader_stage_name(shader->stage),
                   shader->id,
                   shader->ubo_used_mask);
      return false;
   }

   if (yttrium_shader_state_is_uniform_buffer_only(shader)) {
      return true;
   }

   if (yttrium_shader_state_is_sampled_texture_only(shader)) {
      const uint32_t sampler_mask =
         yttrium_shader_state_sampler_used_mask(shader);
      if (yttrium_shader_pipeline_enabled() &&
          yttrium_pipeline_vk_shader_stage(shader->stage) &&
          sampler_mask &&
          !(sampler_mask & ~YTTRIUM_VENUS_PIPELINE_SAMPLED_IMAGE_MASK))
         return true;

      YTTRIUM_WARN("yttrium: shader_draw_probe pipeline shader skipped reason=texture_pipeline_unsupported stage=%s id=%u enabled=%u sampler_mask=0x%x supported=0x%x\n",
                   yttrium_shader_stage_name(shader->stage),
                   shader->id,
                   yttrium_shader_pipeline_enabled(),
                   sampler_mask,
                   YTTRIUM_VENUS_PIPELINE_SAMPLED_IMAGE_MASK);
   }

   if (yttrium_shader_state_is_storage_image_only(shader)) {
      const uint64_t image_mask =
         yttrium_shader_state_image_used_mask(shader);
      if (yttrium_pipeline_vk_shader_stage(shader->stage) && image_mask &&
          !(image_mask & ~YTTRIUM_VENUS_PIPELINE_STORAGE_IMAGE_MASK))
         return true;
   }

   if (yttrium_shader_state_is_sampled_storage_image_only(shader)) {
      const uint32_t sampler_mask =
         yttrium_shader_state_sampler_used_mask(shader);
      const uint64_t image_mask =
         yttrium_shader_state_image_used_mask(shader);
      if (yttrium_shader_pipeline_enabled() &&
          yttrium_pipeline_vk_shader_stage(shader->stage) &&
          sampler_mask && image_mask &&
          !(sampler_mask & ~YTTRIUM_VENUS_PIPELINE_SAMPLED_IMAGE_MASK) &&
          !(image_mask & ~YTTRIUM_VENUS_PIPELINE_STORAGE_IMAGE_MASK))
         return true;
   }

   return false;
}

static bool
yttrium_pipeline_fragment_shader_absent(
   const struct yttrium_shader_state *fs)
{
   return !fs ||
          (fs->stage == MESA_SHADER_FRAGMENT &&
           !yttrium_shader_state_has_module(fs) &&
           yttrium_shader_state_is_resource_free(fs) &&
           !fs->ubo_used_mask &&
           !yttrium_shader_state_sampler_used_mask(fs) &&
           !fs->info.num_inputs &&
           !fs->info.num_outputs);
}

static uint8_t
yttrium_pipeline_unshadow_sampler_target(uint8_t target)
{
   switch (target) {
   case TGSI_TEXTURE_SHADOW1D:
      return TGSI_TEXTURE_1D;
   case TGSI_TEXTURE_SHADOW1D_ARRAY:
      return TGSI_TEXTURE_1D_ARRAY;
   case TGSI_TEXTURE_SHADOW2D:
      return TGSI_TEXTURE_2D;
   case TGSI_TEXTURE_SHADOW2D_ARRAY:
      return TGSI_TEXTURE_2D_ARRAY;
   case TGSI_TEXTURE_SHADOWCUBE:
      return TGSI_TEXTURE_CUBE;
   case TGSI_TEXTURE_SHADOWCUBE_ARRAY:
      return TGSI_TEXTURE_CUBE_ARRAY;
   default:
      return target;
   }
}

static bool
yttrium_pipeline_2d_array_view_can_be_cube(
   const struct pipe_sampler_view *view,
   const struct yttrium_resource *src,
   uint32_t *layer_count)
{
   if (!view || !src || src->base.target != PIPE_TEXTURE_2D_ARRAY ||
       view->target != PIPE_TEXTURE_2D_ARRAY ||
       src->base.width0 != src->base.height0 ||
       view->u.tex.last_layer < view->u.tex.first_layer)
      return false;

   const uint32_t layers =
      view->u.tex.last_layer - view->u.tex.first_layer + 1;
   if (layers < 6 || layers % 6)
      return false;

   if (layer_count)
      *layer_count = layers;
   return true;
}

static bool
yttrium_pipeline_sampler_slot_is_buffer(
   const struct yttrium_shader_state *shader,
   uint32_t slot,
   const struct pipe_sampler_view *view,
   const struct yttrium_resource *src,
   bool *out_buffer)
{
   if (!shader || !view || !src || !out_buffer)
      return false;

   const uint8_t shader_target =
      slot < ARRAY_SIZE(shader->info.sampler_targets) ?
      shader->info.sampler_targets[slot] : TGSI_TEXTURE_UNKNOWN;
   const uint8_t texture_target =
      yttrium_pipeline_unshadow_sampler_target(shader_target);
   const bool view_buffer =
      view->target == PIPE_BUFFER || src->base.target == PIPE_BUFFER;

   if (texture_target == TGSI_TEXTURE_BUFFER) {
      *out_buffer = true;
      return view_buffer;
   }

   if (texture_target != TGSI_TEXTURE_UNKNOWN) {
      *out_buffer = false;
      if (view_buffer)
         return false;

      switch (texture_target) {
      case TGSI_TEXTURE_1D:
         return src->base.target == PIPE_TEXTURE_1D &&
                view->target == PIPE_TEXTURE_1D;
      case TGSI_TEXTURE_1D_ARRAY:
         return src->base.target == PIPE_TEXTURE_1D_ARRAY &&
                view->target == PIPE_TEXTURE_1D_ARRAY;
      case TGSI_TEXTURE_2D:
         if (view->target != PIPE_TEXTURE_2D)
            return false;
         if (src->base.target == PIPE_TEXTURE_2D)
            return true;
         return src->base.target == PIPE_TEXTURE_2D_ARRAY &&
                view->u.tex.first_layer == view->u.tex.last_layer;
      case TGSI_TEXTURE_2D_ARRAY:
         /* D3D represents a one-element Texture2DArray with the same
          * resource shape as a non-array Texture2D.  Preserve the shader's
          * array interpretation by creating a one-layer Vulkan array view. */
         if (src->base.target != PIPE_TEXTURE_2D_ARRAY &&
             src->base.target != PIPE_TEXTURE_2D)
            return false;
         if (view->target == PIPE_TEXTURE_2D_ARRAY)
            return true;
         return view->target == PIPE_TEXTURE_2D &&
                view->u.tex.first_layer == view->u.tex.last_layer;
      case TGSI_TEXTURE_2D_MSAA:
         return src->base.target == PIPE_TEXTURE_2D &&
                src->base.nr_samples > 1 &&
                view->target == PIPE_TEXTURE_2D;
      case TGSI_TEXTURE_2D_ARRAY_MSAA:
         return (src->base.target == PIPE_TEXTURE_2D_ARRAY ||
                 src->base.target == PIPE_TEXTURE_2D) &&
                src->base.nr_samples > 1 &&
                (view->target == PIPE_TEXTURE_2D_ARRAY ||
                 (view->target == PIPE_TEXTURE_2D &&
                  view->u.tex.first_layer == view->u.tex.last_layer));
      case TGSI_TEXTURE_3D:
         return src->base.target == PIPE_TEXTURE_3D &&
                view->target == PIPE_TEXTURE_3D;
      case TGSI_TEXTURE_CUBE:
         if (src->base.target == PIPE_TEXTURE_CUBE &&
             view->target == PIPE_TEXTURE_CUBE)
            return true;
         uint32_t cube_layers;
         return yttrium_pipeline_2d_array_view_can_be_cube(view, src,
                                                           &cube_layers) &&
                cube_layers == 6;
      case TGSI_TEXTURE_CUBE_ARRAY:
         /* D3D represents a one-element TextureCubeArray with the same
          * resource and view shape as a plain cube, so accept that and let
          * the view type follow the shader's array interpretation. */
         if (src->base.target == PIPE_TEXTURE_CUBE &&
             view->target == PIPE_TEXTURE_CUBE)
            return true;
         return (src->base.target == PIPE_TEXTURE_CUBE_ARRAY &&
                 view->target == PIPE_TEXTURE_CUBE_ARRAY) ||
                yttrium_pipeline_2d_array_view_can_be_cube(view, src, NULL);
      default:
         return false;
      }
   }

   *out_buffer = view_buffer;
   if (view_buffer)
      return true;

   switch (view->target) {
   case PIPE_TEXTURE_1D:
   case PIPE_TEXTURE_1D_ARRAY:
   case PIPE_TEXTURE_2D:
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_3D:
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY:
      return src->base.target == view->target;
   default:
      return false;
   }
}

static bool
yttrium_pipeline_sampler_slot_targets_buffer(
   const struct yttrium_shader_state *shader,
   uint32_t slot)
{
   return shader && slot < ARRAY_SIZE(shader->info.sampler_targets) &&
          shader->info.sampler_targets[slot] == TGSI_TEXTURE_BUFFER;
}

static bool
yttrium_pipeline_resolve_sampler_view_slot(
   const struct yttrium_context *yctx,
   const struct yttrium_shader_state *shader,
   mesa_shader_stage stage,
   uint32_t sampler_mask,
   uint32_t sampler_slot,
   unsigned *view_slot_out)
{
   if (!yctx || !shader || stage >= MESA_SHADER_STAGES ||
       sampler_slot >= PIPE_MAX_SAMPLERS || !view_slot_out)
      return false;

   unsigned view_slot =
      yttrium_shader_state_sampler_view_index(shader, sampler_slot);
   if (view_slot < PIPE_MAX_SHADER_SAMPLER_VIEWS &&
       yctx->sampler_views[stage][view_slot]) {
      *view_slot_out = view_slot;
      return true;
   }

   if (yttrium_shader_state_has_explicit_sampler_binding(shader,
                                                          sampler_slot))
      return false;

   if (util_bitcount(sampler_mask) != 1)
      return false;

   unsigned single_bound_slot = PIPE_MAX_SHADER_SAMPLER_VIEWS;
   for (unsigned slot = 0; slot < PIPE_MAX_SHADER_SAMPLER_VIEWS; slot++) {
      if (!yctx->sampler_views[stage][slot])
         continue;

      if (single_bound_slot != PIPE_MAX_SHADER_SAMPLER_VIEWS)
         return false;

      single_bound_slot = slot;
   }

   if (single_bound_slot == PIPE_MAX_SHADER_SAMPLER_VIEWS)
      return false;

   *view_slot_out = single_bound_slot;
   return true;
}

static bool
yttrium_pipeline_sampled_image_view_desc(
   const struct pipe_sampler_view *view,
   const struct yttrium_resource *src,
   uint8_t shader_target,
   VkImageViewType *view_type,
   uint32_t *first_level,
   uint32_t *level_count,
   uint32_t *first_layer,
   uint32_t *layer_count)
{
   if (!view || !src || !view_type || !first_level || !level_count ||
       !first_layer || !layer_count)
      return false;

   switch (view->target) {
   case PIPE_TEXTURE_1D:
      *view_type = VK_IMAGE_VIEW_TYPE_1D;
      break;
   case PIPE_TEXTURE_1D_ARRAY:
      *view_type = VK_IMAGE_VIEW_TYPE_1D_ARRAY;
      break;
   case PIPE_TEXTURE_2D:
      if ((yttrium_pipeline_unshadow_sampler_target(shader_target) ==
              TGSI_TEXTURE_2D_ARRAY ||
           shader_target == TGSI_TEXTURE_2D_ARRAY_MSAA) &&
          view->u.tex.first_layer == view->u.tex.last_layer)
         *view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      else
         *view_type = VK_IMAGE_VIEW_TYPE_2D;
      break;
   case PIPE_TEXTURE_2D_ARRAY:
   {
      const uint8_t sampler_target =
         yttrium_pipeline_unshadow_sampler_target(shader_target);
      if (sampler_target == TGSI_TEXTURE_CUBE ||
          sampler_target == TGSI_TEXTURE_CUBE_ARRAY) {
         uint32_t cube_layers;
         if (!yttrium_pipeline_2d_array_view_can_be_cube(view, src,
                                                         &cube_layers))
            return false;

         if (sampler_target == TGSI_TEXTURE_CUBE) {
            if (cube_layers != 6)
               return false;
            *view_type = VK_IMAGE_VIEW_TYPE_CUBE;
         } else {
            *view_type = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
         }
      } else {
         *view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      }
      break;
   }
   case PIPE_TEXTURE_3D:
      *view_type = VK_IMAGE_VIEW_TYPE_3D;
      break;
   case PIPE_TEXTURE_CUBE:
      /* A shader that declares a cube array over D3D's one-element cube needs
       * a cube-array view, the same way a one-layer Texture2DArray does. */
      *view_type = yttrium_pipeline_unshadow_sampler_target(shader_target) ==
                      TGSI_TEXTURE_CUBE_ARRAY ?
                   VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
      break;
   case PIPE_TEXTURE_CUBE_ARRAY:
      *view_type = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
      break;
   default:
      return false;
   }

   if (view->u.tex.last_level < view->u.tex.first_level ||
       view->u.tex.last_layer < view->u.tex.first_layer)
      return false;

   *first_level = view->u.tex.first_level;
   *level_count = view->u.tex.last_level - view->u.tex.first_level + 1;
   if (view->target == PIPE_TEXTURE_3D) {
      *first_layer = 0;
      *layer_count = 1;
   } else {
      *first_layer = view->u.tex.first_layer;
      *layer_count = view->u.tex.last_layer - view->u.tex.first_layer + 1;
   }

   if (*first_level >= MAX2(src->venus.levels, 1) ||
       *level_count > MAX2(src->venus.levels, 1) - *first_level ||
       *first_layer >= MAX2(src->venus.layers, 1) ||
       *layer_count > MAX2(src->venus.layers, 1) - *first_layer)
      return false;

   return true;
}

static bool
yttrium_pipeline_build_sampled_resource_masks(
   const struct yttrium_context *yctx,
   const struct yttrium_shader_state *shader,
   mesa_shader_stage stage,
   const struct yttrium_resource *dst,
   uint32_t *sampled_image_mask,
   uint32_t *sampled_buffer_mask)
{
   if (!sampled_image_mask || !sampled_buffer_mask)
      return false;

   *sampled_image_mask = 0;
   *sampled_buffer_mask = 0;

   if (!yttrium_shader_state_is_sampled_texture_only(shader) &&
       !yttrium_shader_state_is_sampled_storage_image_only(shader))
      return true;

   const uint32_t sampler_mask =
      yttrium_shader_state_sampler_used_mask(shader);
   if (!sampler_mask)
      return true;

   if (sampler_mask & ~YTTRIUM_VENUS_PIPELINE_SAMPLED_IMAGE_MASK) {
      yttrium_pipeline_trace_sampled_texture_fail(
         "key_unsupported_sampler_mask", UINT32_MAX, UINT32_MAX,
         sampler_mask, dst, NULL, NULL);
      return false;
   }

   for (uint32_t slot = 0;
        slot < YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES; slot++) {
      if (!(sampler_mask & (1u << slot)))
         continue;

      const uint32_t binding =
         yttrium_shader_sampler_binding(stage, slot);
      unsigned view_slot = slot;
      if (!yttrium_pipeline_resolve_sampler_view_slot(
             yctx, shader, stage, sampler_mask, slot, &view_slot)) {
         if (yttrium_pipeline_sampler_slot_targets_buffer(shader, slot))
            *sampled_buffer_mask |= 1u << slot;
         else
            *sampled_image_mask |= 1u << slot;
         continue;
      }
      struct pipe_sampler_view *view =
         yctx->sampler_views[stage][view_slot];
      struct yttrium_resource *src =
         view && view->texture ? yttrium_resource(view->texture) : NULL;
      bool buffer = false;

      if (binding == UINT32_MAX) {
         yttrium_pipeline_trace_sampled_texture_fail(
            "key_bad_sampler_binding", slot, binding, sampler_mask, dst,
            view, src);
         return false;
      }

      if (!src) {
         if (yttrium_pipeline_sampler_slot_targets_buffer(shader, slot))
            *sampled_buffer_mask |= 1u << slot;
         else
            *sampled_image_mask |= 1u << slot;
         continue;
      }

      if (!yttrium_pipeline_sampler_slot_is_buffer(shader, slot, view, src,
                                                   &buffer)) {
         yttrium_pipeline_trace_sampled_texture_fail(
            "key_unsupported_sampler_target", slot, binding, sampler_mask,
            dst, view, src);
         YTTRIUM_WARN("yttrium: shader_draw_probe pipeline key skipped unsupported sampled target slot=%u shader_target=%u view_target=%u res_target=%u format=%u\n",
                      slot,
                      slot < ARRAY_SIZE(shader->info.sampler_targets) ?
                         shader->info.sampler_targets[slot] :
                         TGSI_TEXTURE_UNKNOWN,
                      view ? view->target : PIPE_MAX_TEXTURE_TYPES,
                      src->base.target, view ? view->format :
                         PIPE_FORMAT_NONE);
         return false;
      }

      if (buffer)
         *sampled_buffer_mask |= 1u << slot;
      else
         *sampled_image_mask |= 1u << slot;
   }

   return ((*sampled_image_mask | *sampled_buffer_mask) == sampler_mask);
}

static void
yttrium_pipeline_build_feedback_loop_masks(
   const struct yttrium_context *yctx,
   const struct yttrium_shader_state *shader,
   mesa_shader_stage stage,
   const struct yttrium_resource *dst,
   const struct yttrium_resource *zs,
   uint32_t sampled_image_mask,
   struct yttrium_pipeline_key *key)
{
   if (!shader || stage == MESA_SHADER_NONE || !sampled_image_mask)
      return;

   for (uint32_t slot = 0;
        slot < YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES; slot++) {
      if (!(sampled_image_mask & (1u << slot)))
         continue;

      unsigned view_slot = slot;
      if (!yttrium_pipeline_resolve_sampler_view_slot(
             yctx, shader, stage,
             yttrium_shader_state_sampler_used_mask(shader), slot,
             &view_slot))
         continue;

      const struct pipe_sampler_view *view =
         yctx->sampler_views[stage][view_slot];
      const struct yttrium_resource *sampled =
         view && view->texture ? yttrium_resource(view->texture) : NULL;
      if (!sampled || sampled->base.target == PIPE_BUFFER)
         continue;

      for (uint32_t rt_index = 0; rt_index < key->rt_count; rt_index++) {
         const struct yttrium_resource *rt = NULL;
         if (yctx->fb.nr_cbufs) {
            if (rt_index < yctx->fb.nr_cbufs &&
                yctx->fb.cbufs[rt_index].texture)
               rt = yttrium_resource(
                  yctx->fb.cbufs[rt_index].texture);
         } else if (!rt_index) {
            rt = dst;
         }

         if (sampled == rt)
            key->color_feedback_loop_mask |= 1u << rt_index;
      }

      if (sampled == zs)
         key->depth_feedback_loop = VK_TRUE;
   }
}

static bool
yttrium_pipeline_add_sampled_layouts(
   const struct yttrium_context *yctx,
   const struct yttrium_shader_state *sampled_shader,
   uint32_t sampled_image_mask,
   uint32_t sampled_buffer_mask,
   struct yttrium_pipeline_key *key)
{
   const mesa_shader_stage sampled_stage = sampled_shader->stage;
   const uint32_t sampler_mask =
      yttrium_shader_state_sampler_used_mask(sampled_shader);

   for (uint32_t slot = 0; slot < PIPE_MAX_SAMPLERS; slot++) {
      const uint32_t slot_mask = 1u << slot;
      if (!(sampler_mask & slot_mask))
         continue;
      if (key->sampled_binding_count >=
          YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES) {
         YTTRIUM_WARN("yttrium: shader_draw_probe pipeline key skipped sampled descriptor limit stages=0x%x count=%u max=%u stage=%u sampler=0x%x\n",
                      key->sampled_stage_mask,
                      key->sampled_binding_count + 1,
                      YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES,
                      sampled_stage, sampler_mask);
         return false;
      }

      const uint32_t binding =
         yttrium_shader_sampler_binding(sampled_stage, slot);
      if (binding == UINT32_MAX)
         return false;

      struct yttrium_venus_sampled_binding_layout *layout =
         &key->sampled_bindings[key->sampled_binding_count++];
      *layout = (struct yttrium_venus_sampled_binding_layout) {
         .binding = binding,
         .stage = sampled_stage,
         .raw_slot = slot,
         .stage_flags = yttrium_pipeline_vk_shader_stage(sampled_stage),
         .buffer = (sampled_buffer_mask & slot_mask) != 0,
      };

      if (sampled_image_mask & slot_mask) {
         const unsigned state_slot =
            yttrium_shader_state_sampler_state_index(sampled_shader, slot);
         const struct yttrium_sampler_state *sampler =
            state_slot < PIPE_MAX_SAMPLERS ?
               yctx->sampler_states[sampled_stage][state_slot] : NULL;
         yttrium_pipeline_sampler_state_from_pipe(
            sampler ? &sampler->state : NULL, &layout->sampler);

         unsigned view_slot = slot;
         const struct pipe_sampler_view *view = NULL;
         if (yttrium_pipeline_resolve_sampler_view_slot(
                yctx, sampled_shader, sampled_stage, sampler_mask, slot,
                &view_slot)) {
            view = yctx->sampler_views[sampled_stage][view_slot];
            if (view && util_format_is_pure_integer(view->format)) {
               yttrium_pipeline_sampler_state_force_integer_fetch(
                  &layout->sampler);
               if (yttrium_pipeline_verbose_trace_enabled()) {
                  yttrium_trace_debug_stringf(
                     "yttrium: integer_sampler_state stage=%u sampler_slot=%u view_slot=%u format=%u min_filter=%u mag_filter=%u mipmap_mode=%u",
                     sampled_stage, slot, view_slot, view->format,
                     layout->sampler.min_filter,
                     layout->sampler.mag_filter,
                     layout->sampler.mipmap_mode);
               }
            }
         }
         if (!yttrium_pipeline_sampler_border_color(
                sampler ? &sampler->state : NULL, view, &layout->sampler))
            return false;
      }
   }
   return true;
}

static bool
yttrium_pipeline_read_only_depth_view(
   const struct pipe_sampler_view *view,
   const struct yttrium_resource *zs)
{
   return view && zs && view->texture == &zs->base &&
          view->target == PIPE_TEXTURE_2D &&
          (view->format == PIPE_FORMAT_R32_FLOAT ||
           view->format == PIPE_FORMAT_Z32_FLOAT) &&
          view->u.tex.first_level == 0 && view->u.tex.last_level == 0 &&
          view->u.tex.first_layer == 0 && view->u.tex.last_layer == 0;
}

static bool
yttrium_pipeline_sampled_depth_read_only(
   const struct yttrium_context *yctx,
   const struct yttrium_resource *zs,
   const struct yttrium_shader_state *fs,
   const struct yttrium_pipeline_key *key)
{
   /* Resource layout tracking is currently whole-image. Start with the
    * single-subresource depth-only case, so a read-only attachment never
    * changes the layout of an independently writable mip, layer or stencil
    * aspect. Keep shader-written depth and storage paths on the existing
    * feedback-loop path even when their fixed-function write mask is zero.
    */
   if (!zs || !zs->venus.contents_initialized ||
       !key->depth_feedback_loop || !key->depth_test_enable ||
       key->depth_write_enable || key->stencil_test_enable ||
       key->storage_image_mask || key->storage_buffer_mask ||
       key->rasterization_samples != 1 || key->forced_sample_count ||
       key->forced_sample_interlock ||
       zs->base.target != PIPE_TEXTURE_2D || zs->base.last_level != 0 ||
       zs->base.array_size != 1 || zs->base.depth0 != 1 ||
       zs->base.nr_samples > 1 || zs->base.nr_storage_samples > 1 ||
       zs->venus.vk_format != VK_FORMAT_D32_SFLOAT ||
       key->zs_level != 0 || key->zs_layer != 0 || key->zs_layers != 1 ||
       !fs || fs->info.writes_z || fs->info.writes_stencil ||
       fs->info.writes_memory)
      return false;

   /* STORE is itself a depth write. The read-only path must be able to
    * preserve prior contents with STORE_OP_NONE, not STORE or DONT_CARE.
    */
   if (!yttrium_venus_supports_load_store_op_none(
          yttrium_screen(yctx->base.screen)->venus))
      return false;

   bool sampled_depth = false;
   for (uint32_t i = 0; i < key->sampled_binding_count; i++) {
      const struct yttrium_venus_sampled_binding_layout *layout =
         &key->sampled_bindings[i];
      const mesa_shader_stage stage = (mesa_shader_stage)layout->stage;
      if (stage < MESA_SHADER_VERTEX || stage > MESA_SHADER_FRAGMENT ||
          layout->raw_slot >= PIPE_MAX_SAMPLERS)
         return false;

      const struct yttrium_shader_state *shader = yctx->shaders[stage];
      unsigned view_slot = layout->raw_slot;
      if (!yttrium_pipeline_resolve_sampler_view_slot(
             yctx, shader, stage,
             yttrium_shader_state_sampler_used_mask(shader),
             layout->raw_slot, &view_slot))
         continue;

      const struct pipe_sampler_view *view =
         yctx->sampler_views[stage][view_slot];
      if (!view || view->texture != &zs->base)
         continue;
      if (layout->buffer || !yttrium_pipeline_read_only_depth_view(view, zs))
         return false;
      sampled_depth = true;
   }
   return sampled_depth;
}

static bool
yttrium_pipeline_build_key(struct yttrium_context *yctx,
                           const struct yttrium_resource *dst,
                           const struct yttrium_resource *zs,
                           const struct yttrium_venus_draw_state *draw_state,
                           struct yttrium_pipeline_key *key)
{
   const struct yttrium_shader_state *vs =
      yctx->shaders[MESA_SHADER_VERTEX];
   const struct yttrium_shader_state *tcs =
      yctx->shaders[MESA_SHADER_TESS_CTRL];
   const struct yttrium_shader_state *tes =
      yctx->shaders[MESA_SHADER_TESS_EVAL];
   const struct yttrium_shader_state *gs =
      yctx->shaders[MESA_SHADER_GEOMETRY];
   struct yttrium_shader_state *fs_raw =
      yctx->shaders[MESA_SHADER_FRAGMENT];
   struct yttrium_shader_state *fs =
      yttrium_pipeline_fragment_shader_absent(fs_raw) ? NULL : fs_raw;

   if (!yttrium_pipeline_shader_supported(vs) ||
       (tcs && !yttrium_pipeline_shader_supported(tcs)) ||
       (tes && !yttrium_pipeline_shader_supported(tes)) ||
       (gs && !yttrium_pipeline_shader_supported(gs)) ||
       (fs && !yttrium_pipeline_shader_supported(fs))) {
      YTTRIUM_WARN("yttrium: shader_draw_probe pipeline key skipped shader unsupported vs=%p vs_module=%u vs_resource_free=%u vs_ubo_only=%u vs_texture_only=%u tcs=%p tcs_module=%u tcs_resource_free=%u tcs_ubo_only=%u tcs_texture_only=%u tes=%p tes_module=%u tes_resource_free=%u tes_ubo_only=%u tes_texture_only=%u gs=%p gs_module=%u gs_resource_free=%u gs_ubo_only=%u gs_texture_only=%u fs=%p fs_module=%u fs_resource_free=%u fs_ubo_only=%u fs_texture_only=%u fs_sampler=0x%x\n",
                   vs,
                   vs ? yttrium_shader_state_has_module(vs) : 0,
                   vs ? yttrium_shader_state_is_resource_free(vs) : 0,
                   vs ? yttrium_shader_state_is_uniform_buffer_only(vs) : 0,
                   vs ? yttrium_shader_state_is_sampled_texture_only(vs) : 0,
                   tcs,
                   tcs ? yttrium_shader_state_has_module(tcs) : 0,
                   tcs ? yttrium_shader_state_is_resource_free(tcs) : 0,
                   tcs ? yttrium_shader_state_is_uniform_buffer_only(tcs) : 0,
                   tcs ? yttrium_shader_state_is_sampled_texture_only(tcs) : 0,
                   tes,
                   tes ? yttrium_shader_state_has_module(tes) : 0,
                   tes ? yttrium_shader_state_is_resource_free(tes) : 0,
                   tes ? yttrium_shader_state_is_uniform_buffer_only(tes) : 0,
                   tes ? yttrium_shader_state_is_sampled_texture_only(tes) : 0,
                   gs,
                   gs ? yttrium_shader_state_has_module(gs) : 0,
                   gs ? yttrium_shader_state_is_resource_free(gs) : 0,
                   gs ? yttrium_shader_state_is_uniform_buffer_only(gs) : 0,
                   gs ? yttrium_shader_state_is_sampled_texture_only(gs) : 0,
                   fs_raw,
                   fs_raw ? yttrium_shader_state_has_module(fs_raw) : 0,
                   fs_raw ? yttrium_shader_state_is_resource_free(fs_raw) : 0,
                   fs_raw ? yttrium_shader_state_is_uniform_buffer_only(fs_raw) : 0,
                   fs_raw ? yttrium_shader_state_is_sampled_texture_only(fs_raw) : 0,
                   fs_raw ? yttrium_shader_state_sampler_used_mask(fs_raw) : 0);
      return false;
   }

   const bool dst_storage_buffer =
      dst && dst->base.target == PIPE_BUFFER;
   const bool dst_render_image =
      dst && dst->venus.initialized && !dst->venus.buffer_backed &&
      dst->venus.image && dst->venus.vk_format != VK_FORMAT_UNDEFINED;
   if (!dst_storage_buffer && !dst_render_image) {
      YTTRIUM_WARN("yttrium: shader_draw_probe pipeline key skipped dst unsupported dst=%p initialized=%u buffer_backed=%u image=0x%llx format=%u res_id=%u\n",
                   dst,
                   dst ? dst->venus.initialized : 0,
                   dst ? dst->venus.buffer_backed : 0,
                   (unsigned long long)(dst ?
                      yttrium_pipeline_vk_image_to_u64(dst->venus.image) : 0),
                   dst ? dst->venus.vk_format : VK_FORMAT_UNDEFINED,
                   dst ? dst->venus_res_id : 0);
      return false;
   }

   if (yctx->vertex_elements &&
       !yctx->vertex_elements->vk_vertex_input_valid) {
      YTTRIUM_WARN("yttrium: shader_draw_probe pipeline key skipped vertex input unsupported ve=%p valid=%u bindings=%u attribs=%u\n",
                   yctx->vertex_elements,
                   yctx->vertex_elements ?
                      yctx->vertex_elements->vk_vertex_input_valid : 0,
                   yctx->vertex_elements ?
                      yctx->vertex_elements->num_bindings : 0,
                   yctx->vertex_elements ?
                      yctx->vertex_elements->num_elements : 0);
      return false;
   }

   if (!draw_state) {
      YTTRIUM_WARN("yttrium: shader_draw_probe pipeline key skipped missing draw_state\n");
      return false;
   }

   const uint32_t explicit_rt_count =
      MIN2(draw_state->rt_count, (uint32_t)PIPE_MAX_COLOR_BUFS);
   const bool implicit_dst_color =
      dst_render_image && explicit_rt_count == 0 &&
      !draw_state->targetless_stream_output &&
      !draw_state->forced_sample_interlock && dst != zs &&
      (dst->display_target ||
       (dst->venus.image_usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0);
   const uint32_t rt_count = implicit_dst_color ? 1 : explicit_rt_count;

   for (uint32_t i = 0; i < explicit_rt_count; i++) {
      const struct pipe_surface *cbuf = &yctx->fb.cbufs[i];
      const struct yttrium_resource *rt =
         cbuf->texture ? yttrium_resource(cbuf->texture) : NULL;
      if (!rt)
         continue;

      const uint32_t rt_width =
         u_minify(rt->base.width0, cbuf->level);
      const uint32_t rt_height =
         u_minify(rt->base.height0, cbuf->level);
      const uint32_t rt_layers =
         cbuf->last_layer >= cbuf->first_layer ?
         cbuf->last_layer - cbuf->first_layer + 1 : 1;

      if (!rt->venus.initialized || rt->venus.buffer_backed ||
          !rt->venus.image ||
          !(rt->venus.image_usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ||
          rt->venus.vk_format == VK_FORMAT_UNDEFINED ||
          (i < ARRAY_SIZE(draw_state->rt_level) &&
           cbuf->level != draw_state->rt_level[i]) ||
          /* Each target's own subresource has to agree, not the first
           * target's. */
          (i < ARRAY_SIZE(draw_state->rt_layer) &&
           cbuf->first_layer != draw_state->rt_layer[i]) ||
          rt_layers != MAX2(draw_state->render_layers, 1) ||
          rt_width != draw_state->render_width ||
          rt_height != draw_state->render_height) {
         YTTRIUM_WARN("yttrium: shader_draw_probe pipeline key skipped rt%u unsupported rt=%p initialized=%u buffer_backed=%u image=0x%llx usage=0x%x format=%u view=%u/%u+%u extent=%ux%u expected=%u/%u+%u %ux%u\n",
                      i, rt,
                      rt ? rt->venus.initialized : 0,
                      rt ? rt->venus.buffer_backed : 0,
                      (unsigned long long)(rt ?
                         yttrium_pipeline_vk_image_to_u64(rt->venus.image) : 0),
                      rt ? rt->venus.image_usage : 0,
                      rt ? rt->venus.vk_format : VK_FORMAT_UNDEFINED,
                      cbuf->level, cbuf->first_layer, rt_layers,
                      rt_width, rt_height,
                      i < ARRAY_SIZE(draw_state->rt_level) ?
                         draw_state->rt_level[i] : draw_state->render_level,
                      i < ARRAY_SIZE(draw_state->rt_layer) ?
                         draw_state->rt_layer[i] : draw_state->render_layer,
                      MAX2(draw_state->render_layers, 1),
                      draw_state->render_width, draw_state->render_height);
         return false;
      }
   }

   const struct pipe_surface *zsbuf =
      zs && yctx->fb.zsbuf.texture ? &yctx->fb.zsbuf : NULL;
   const uint32_t zs_width = zsbuf ?
      u_minify(zs->base.width0, zsbuf->level) : 0;
   const uint32_t zs_height = zsbuf ?
      u_minify(zs->base.height0, zsbuf->level) : 0;
   const uint32_t zs_layers = zsbuf &&
      zsbuf->last_layer >= zsbuf->first_layer ?
      zsbuf->last_layer - zsbuf->first_layer + 1 : 1;
   if (zs &&
       (!zsbuf || !zs->venus.initialized || zs->venus.buffer_backed ||
        !zs->venus.image ||
        !(zs->venus.image_usage &
          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ||
        zs->venus.vk_format == VK_FORMAT_UNDEFINED ||
        zsbuf->level != draw_state->depth_level ||
        zsbuf->first_layer != draw_state->depth_layer ||
        zs_layers != MAX2(draw_state->depth_layers, 1) ||
        zs_layers < MAX2(draw_state->render_layers, 1) ||
        zs_width < draw_state->render_width ||
        zs_height < draw_state->render_height)) {
      YTTRIUM_WARN("yttrium: shader_draw_probe pipeline key skipped zs unsupported zs=%p initialized=%u buffer_backed=%u image=0x%llx usage=0x%x format=%u view=%u/%u+%u extent=%ux%u expected=%u/%u+%u %ux%u render_layers=%u\n",
                   zs,
                   zs ? zs->venus.initialized : 0,
                   zs ? zs->venus.buffer_backed : 0,
                   (unsigned long long)(zs ?
                      yttrium_pipeline_vk_image_to_u64(zs->venus.image) : 0),
                   zs ? zs->venus.image_usage : 0,
                   zs ? zs->venus.vk_format : VK_FORMAT_UNDEFINED,
                   zsbuf ? zsbuf->level : 0,
                   zsbuf ? zsbuf->first_layer : 0, zs_layers,
                   zs_width, zs_height,
                   draw_state->depth_level, draw_state->depth_layer,
                   MAX2(draw_state->depth_layers, 1),
                   draw_state->render_width, draw_state->render_height,
                   MAX2(draw_state->render_layers, 1));
      return false;
   }

   memset(key, 0, sizeof(*key));
   key->vs_id = vs->id;
   key->tcs_id = tcs ? tcs->id : 0;
   key->tes_id = tes ? tes->id : 0;
   key->gs_id = gs ? gs->id :
      (yttrium_pipeline_needs_cull_distance_gs(vs, gs,
                                               draw_state->topology) ?
       UINT32_MAX : 0);
   key->fs_id = fs ? fs->id : 0;
   key->vs_hash = vs->spirv_hash;
   key->tcs_hash = tcs ? tcs->spirv_hash : 0;
   key->tes_hash = tes ? tes->spirv_hash : 0;
   key->gs_hash = gs ? gs->spirv_hash :
      (key->gs_id == UINT32_MAX ?
       yttrium_pipeline_generated_cull_gs_hash(vs->spirv_hash,
                                               draw_state->topology) : 0);
   key->fs_hash = fs ? fs->spirv_hash : 0;
   key->rt_count = rt_count;
   for (uint32_t i = 0; i < rt_count; i++) {
      const struct yttrium_resource *rt = implicit_dst_color ?
         dst : (yctx->fb.cbufs[i].texture ?
                yttrium_resource(yctx->fb.cbufs[i].texture) : NULL);

      key->dst_image_id[i] = rt ? rt->venus.image_obj.id : 0;
      const enum pipe_format rt_format = implicit_dst_color ?
         dst->base.format : yctx->fb.cbufs[i].format;
      key->rt_format[i] = rt_format != PIPE_FORMAT_NONE ?
         yttrium_venus_pipe_format(rt_format) :
         (rt ? rt->venus.vk_format : VK_FORMAT_UNDEFINED);
      if (rt_format == PIPE_FORMAT_A8_UNORM)
         key->a8_rt_mask |= 1u << i;
      if (rt_format == PIPE_FORMAT_B8G8R8X8_UNORM ||
          rt_format == PIPE_FORMAT_R8G8B8X8_UNORM)
         key->x8_rt_mask |= 1u << i;
      key->blend_enable[i] = draw_state->rt_blend_enable[i];
      key->color_write_mask[i] = draw_state->rt_color_write_mask[i];
      key->src_color_blend_factor[i] =
         draw_state->rt_src_color_blend_factor[i];
      key->dst_color_blend_factor[i] =
         draw_state->rt_dst_color_blend_factor[i];
      key->color_blend_op[i] = draw_state->rt_color_blend_op[i];
      key->src_alpha_blend_factor[i] =
         draw_state->rt_src_alpha_blend_factor[i];
      key->dst_alpha_blend_factor[i] =
         draw_state->rt_dst_alpha_blend_factor[i];
      key->alpha_blend_op[i] = draw_state->rt_alpha_blend_op[i];
      if (i == 0 && key->blend_enable[i] &&
          (yttrium_pipeline_blend_factor_uses_src1(
              key->src_color_blend_factor[i]) ||
           yttrium_pipeline_blend_factor_uses_src1(
              key->dst_color_blend_factor[i]) ||
           yttrium_pipeline_blend_factor_uses_src1(
              key->src_alpha_blend_factor[i]) ||
           yttrium_pipeline_blend_factor_uses_src1(
              key->dst_alpha_blend_factor[i])))
         key->dual_source_blend = 1;
   }
   key->zs_image_id = zs ? zs->venus.image_obj.id : 0;
   key->logic_op_enable = draw_state->logic_op_enable;
   key->logic_op = draw_state->logic_op;
   if (key->dual_source_blend && fs)
      key->fs_hash = yttrium_pipeline_dual_source_fs_hash(key->fs_hash);
   if (key->a8_rt_mask && fs)
      key->fs_hash =
         yttrium_pipeline_a8_rt_fs_hash(key->fs_hash, key->a8_rt_mask);
   key->zs_format = zs ? zs->venus.vk_format : VK_FORMAT_UNDEFINED;
   key->width = draw_state->render_width ? draw_state->render_width :
      dst->venus.width;
   key->height = draw_state->render_height ? draw_state->render_height :
      dst->venus.height;
   /* Per target: the pipeline owns the views, so two draws differing only in
    * which mip or slice a target names must not share one. */
   for (uint32_t i = 0; i < ARRAY_SIZE(key->rt_level); i++) {
      key->rt_level[i] = draw_state->rt_level[i];
      key->rt_layer[i] = draw_state->rt_layer[i];
   }
   key->rt_layers = MAX2(draw_state->render_layers, 1);
   key->zs_level = draw_state->depth_level;
   key->zs_layer = draw_state->depth_layer;
   key->zs_layers = MAX2(draw_state->depth_layers, 1);
   key->viewport_count = MAX2(draw_state->viewport_count, 1);
   key->topology = draw_state->topology;
   key->patch_vertices =
      draw_state->topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST ?
      yctx->patch_vertices : 0;
   key->primitive_restart_enable = draw_state->primitive_restart_enable;
   key->rasterizer_discard_enable =
      draw_state->rasterizer_discard_enable;
   key->targetless_stream_output =
      draw_state->targetless_stream_output;
   key->cull_mode = draw_state->cull_mode;
   key->front_face = draw_state->front_face;
   key->depth_bias_enable = draw_state->depth_bias_enable;
   key->depth_clamp_enable = draw_state->depth_clamp_enable;
   key->depth_bias_constant_factor =
      draw_state->depth_bias_constant_factor;
   key->depth_bias_clamp = draw_state->depth_bias_clamp;
   key->depth_bias_slope_factor = draw_state->depth_bias_slope_factor;
   key->sample_mask = draw_state->sample_mask;
   key->rasterization_samples = MAX2(draw_state->rasterization_samples, 1);
   key->forced_sample_count = draw_state->forced_sample_count;
   key->forced_sample_interlock = draw_state->forced_sample_interlock;
   const bool forced_sample_state = key->forced_sample_count > 1;
   bool forced_sample_blend_sample_mask = false;
   if (forced_sample_state && fs &&
       yttrium_shader_state_uses_sample_mask_in(fs)) {
      for (uint32_t i = 0; i < key->rt_count; i++) {
         if (key->blend_enable[i]) {
            forced_sample_blend_sample_mask = true;
            break;
         }
      }
   }
   key->forced_sample_expand = 0;
   const uint32_t requested_forced_samples = key->forced_sample_count;
   const uint32_t hw_forced_samples = key->forced_sample_interlock ?
      yttrium_pipeline_supported_forced_sample_interlock_count(
         yctx, requested_forced_samples) :
      forced_sample_blend_sample_mask ? 0 :
      yttrium_pipeline_supported_forced_sample_count(yctx,
                                                     requested_forced_samples);
   if (hw_forced_samples) {
      key->forced_sample_count = hw_forced_samples;
      key->rasterization_samples = hw_forced_samples;
      /* The device cannot rasterize at the count the app asked for, so widen
       * the coverage mask the FS reads back to the requested width.  Without
       * this the shader divides an N-sample popcount by a larger sample count
       * and every edge loses its partial coverage.
       */
      if (hw_forced_samples != requested_forced_samples && fs &&
          yttrium_shader_state_uses_sample_mask_in(fs)) {
         key->forced_sample_expand = requested_forced_samples;
         key->fs_hash =
            yttrium_pipeline_sample_mask_expand_fs_hash(
               key->fs_hash, hw_forced_samples, requested_forced_samples);
      }
   } else {
      key->forced_sample_count = 0;
   }
   /* Only stub the coverage mask out when the app asked for a forced sample
    * count we could not honor at all.  A genuinely multisampled attachment
    * gets a real per-sample coverage mask from the hardware.
    */
   if (!key->forced_sample_count && requested_forced_samples > 1 &&
       key->rasterization_samples > 1 && fs &&
       yttrium_shader_state_uses_sample_mask_in(fs))
      key->fs_hash =
         yttrium_pipeline_forced_sample_mask_fs_hash(
            key->fs_hash, key->rasterization_samples);
   if (key->forced_sample_interlock && fs) {
      key->fs_hash =
         yttrium_pipeline_forced_sample_interlock_fs_hash(key->fs_hash);
      key->logic_op_enable = VK_FALSE;
   }
   key->sample_shading_enable =
      yttrium_pipeline_fs_uses_sample_shading(fs);
   key->alpha_to_coverage_enable =
      forced_sample_state ? VK_FALSE :
                                 draw_state->alpha_to_coverage_enable;
   key->depth_test_enable = zs ? draw_state->depth_test_enable : VK_FALSE;
   key->depth_write_enable = zs ? draw_state->depth_write_enable : VK_FALSE;
   key->depth_compare_op = zs ? draw_state->depth_compare_op :
      VK_COMPARE_OP_ALWAYS;
   key->alpha_test_enable =
      draw_state->alpha_test_enable && fs ? VK_TRUE : VK_FALSE;
   key->alpha_func = key->alpha_test_enable ? draw_state->alpha_func :
      PIPE_FUNC_ALWAYS;
   key->alpha_ref_value = key->alpha_test_enable ?
      draw_state->alpha_ref_value : 0.0f;
   if (key->alpha_test_enable)
      key->fs_hash =
         yttrium_pipeline_alpha_test_fs_hash(key->fs_hash, key->alpha_func,
                                             key->alpha_ref_value);
   key->stencil_test_enable = zs ? draw_state->stencil_test_enable :
      VK_FALSE;
   key->stencil_front = zs ? draw_state->stencil_front :
      (VkStencilOpState) { 0 };
   key->stencil_back = zs ? draw_state->stencil_back :
      (VkStencilOpState) { 0 };
   key->vs_ubo_used_mask = vs->ubo_used_mask;
   key->tcs_ubo_used_mask = tcs ? tcs->ubo_used_mask : 0;
   key->tes_ubo_used_mask = tes ? tes->ubo_used_mask : 0;
   key->fs_ubo_used_mask = fs ? fs->ubo_used_mask : 0;
   key->gs_ubo_used_mask = gs ? gs->ubo_used_mask : 0;
   const struct yttrium_shader_state *graphics_shaders[] = {
      vs,
      tcs,
      tes,
      gs,
      fs,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(graphics_shaders); i++) {
      const struct yttrium_shader_state *sampled_shader =
         graphics_shaders[i];
      if ((!yttrium_shader_state_is_sampled_texture_only(sampled_shader) &&
           !yttrium_shader_state_is_sampled_storage_image_only(
              sampled_shader)) ||
          !yttrium_shader_state_sampler_used_mask(sampled_shader))
         continue;

      const mesa_shader_stage sampled_stage = sampled_shader->stage;
      const uint32_t sampler_mask =
         yttrium_shader_state_sampler_used_mask(sampled_shader);
      uint32_t sampled_image_mask = 0;
      uint32_t sampled_buffer_mask = 0;
      if (sampled_stage < MESA_SHADER_VERTEX ||
          sampled_stage > MESA_SHADER_FRAGMENT ||
          !yttrium_pipeline_build_sampled_resource_masks(
             yctx, sampled_shader, sampled_stage, dst, &sampled_image_mask,
             &sampled_buffer_mask)) {
         YTTRIUM_WARN("yttrium: shader_draw_probe pipeline key skipped sampled resource masks stage=%u sampler=0x%x image_mask=0x%x buffer_mask=0x%x\n",
                      sampled_stage, sampler_mask, sampled_image_mask,
                      sampled_buffer_mask);
         return false;
      }

      key->sampled_stage_mask |= 1u << sampled_stage;
      yttrium_pipeline_build_feedback_loop_masks(
         yctx, sampled_shader, sampled_stage, dst, zs,
         sampled_image_mask, key);

      if (!yttrium_pipeline_add_sampled_layouts(
             yctx, sampled_shader, sampled_image_mask, sampled_buffer_mask, key))
         return false;
   }

   for (unsigned i = 0; i < ARRAY_SIZE(graphics_shaders); i++) {
      const struct yttrium_shader_state *storage_shader = graphics_shaders[i];
      if (!yttrium_shader_state_is_storage_image_only(storage_shader) &&
          !yttrium_shader_state_is_sampled_storage_image_only(storage_shader))
         continue;

      const uint64_t storage_mask =
         yttrium_shader_state_image_used_mask(storage_shader);
      if (!storage_mask)
         continue;

      for (uint32_t slot = 0;
           slot < YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES; slot++) {
         const uint64_t slot_mask = 1ull << slot;
         if (!(storage_mask & slot_mask))
            continue;

         const struct pipe_image_view *view =
            &yctx->shader_images[storage_shader->stage][slot];
         struct yttrium_resource *src =
            view->resource ? yttrium_resource(view->resource) : NULL;
         if (src && src->base.target == PIPE_BUFFER)
            key->storage_buffer_mask |= slot_mask;
         else
            key->storage_image_mask |= slot_mask;
      }
      key->storage_stage_mask |= 1u << storage_shader->stage;
   }
   if (key->forced_sample_interlock) {
      key->storage_image_mask |=
         1ull << YTTRIUM_FORCED_SAMPLE_INTERLOCK_IMAGE_SLOT;
      key->storage_stage_mask |= 1u << MESA_SHADER_FRAGMENT;
   }
   if (yttrium_pipeline_sampled_depth_read_only(yctx, zs, fs, key)) {
      key->depth_read_only = VK_TRUE;
      key->depth_feedback_loop = VK_FALSE;
   }
   key->vs_ubo_default = vs->ubo_default;
   key->tcs_ubo_default = tcs ? tcs->ubo_default : 0;
   key->tes_ubo_default = tes ? tes->ubo_default : 0;
   key->fs_ubo_default = fs ? fs->ubo_default : 0;
   key->gs_ubo_default = gs ? gs->ubo_default : 0;
   key->vs_ubo_first = vs->ubo_first;
   key->tcs_ubo_first = tcs ? tcs->ubo_first : 0;
   key->tes_ubo_first = tes ? tes->ubo_first : 0;
   key->fs_ubo_first = fs ? fs->ubo_first : 0;
   key->gs_ubo_first = gs ? gs->ubo_first : 0;
   key->vs_ubo_count = vs->ubo_count;
   key->tcs_ubo_count = tcs ? tcs->ubo_count : 0;
   key->tes_ubo_count = tes ? tes->ubo_count : 0;
   key->fs_ubo_count = fs ? fs->ubo_count : 0;
   key->gs_ubo_count = gs ? gs->ubo_count : 0;
   if (yctx->vertex_elements) {
      key->num_bindings = yctx->vertex_elements->num_bindings;
      key->num_attribs = yctx->vertex_elements->num_attribs;
      memcpy(key->bindings, yctx->vertex_elements->bindings,
             key->num_bindings * sizeof(key->bindings[0]));
      for (uint32_t i = 0; i < key->num_bindings; i++) {
         const uint32_t divisor = yctx->vertex_elements->binding_divisor[i];
         key->binding_divisors[i] =
            key->bindings[i].inputRate == VK_VERTEX_INPUT_RATE_INSTANCE &&
            yttrium_venus_vertex_attribute_divisor_supported(
               yttrium_screen(yctx->base.screen)->venus, divisor) ?
            divisor : 1;
      }
      memcpy(key->attribs, yctx->vertex_elements->attribs,
             key->num_attribs * sizeof(key->attribs[0]));
   }
   return true;
}

static bool
yttrium_pipeline_build_ubo_layouts(
   const struct yttrium_shader_state *vs,
   const struct yttrium_shader_state *tcs,
   const struct yttrium_shader_state *tes,
   const struct yttrium_shader_state *gs,
   const struct yttrium_shader_state *fs,
   struct yttrium_venus_ubo_binding_layout *layouts,
   uint32_t *layout_count);

static bool
yttrium_pipeline_collect_ubo_uploads(
   const struct yttrium_context *yctx,
   struct yttrium_venus_ubo_upload *uploads,
   uint32_t *upload_count);

static struct yttrium_resource *
yttrium_pipeline_get_stream_output_dummy_buffer(struct pipe_context *ctx)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct pipe_resource templ;

   if (yctx->so_dummy_buffer)
      return yttrium_resource(yctx->so_dummy_buffer);

   memset(&templ, 0, sizeof(templ));
   templ.target = PIPE_BUFFER;
   templ.format = PIPE_FORMAT_R8_UNORM;
   templ.width0 = 1;
   templ.height0 = 1;
   templ.depth0 = 1;
   templ.array_size = 1;
   templ.last_level = 0;
   templ.nr_samples = 1;
   templ.nr_storage_samples = 1;
   templ.usage = PIPE_USAGE_DEFAULT;
   templ.bind = PIPE_BIND_STREAM_OUTPUT;

   yctx->so_dummy_buffer = ctx->screen->resource_create(ctx->screen, &templ);
   if (!yctx->so_dummy_buffer) {
      YTTRIUM_WARN("yttrium: stream-output dummy buffer allocation failed\n");
      return NULL;
   }

   return yttrium_resource(yctx->so_dummy_buffer);
}

static bool
yttrium_pipeline_collect_stream_output_target(
   struct pipe_context *ctx,
   struct pipe_stream_output_target *target,
   unsigned slot,
   struct yttrium_venus_stream_output_target *targets,
   uint32_t *target_count)
{
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);

   if (!targets || !target_count)
      return false;

   if (*target_count >= YTTRIUM_VENUS_MAX_STREAM_OUTPUT_TARGETS) {
      YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped stream output target count unsupported slot=%u count=%u max=%u\n",
                   slot, *target_count,
                   YTTRIUM_VENUS_MAX_STREAM_OUTPUT_TARGETS);
      return false;
   }

   if (!target) {
      struct yttrium_resource *dummy =
         yttrium_pipeline_get_stream_output_dummy_buffer(ctx);
      if (!dummy) {
         YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped stream output null target slot=%u dummy unavailable\n",
                      slot);
         return false;
      }

      if (!yttrium_venus_create_stream_output_buffer(screen->venus,
                                                     &dummy->venus,
                                                     dummy->size,
                                                     &dummy->venus_mem_id)) {
         YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped stream output null target slot=%u dummy buffer unavailable res=%p size=0x%llx\n",
                      slot, dummy, (unsigned long long)dummy->size);
         return false;
      }
      if (!dummy->venus_res_id)
         dummy->venus_res_id = (uint32_t)dummy->venus.memory_obj.id;

      targets[*target_count] = (struct yttrium_venus_stream_output_target) {
         .resource = &dummy->venus,
         .resource_id = dummy->venus_res_id,
         .buffer_offset = 0,
         .buffer_size = 1,
         .counter_resource = NULL,
         .counter_buffer_valid = false,
      };
      (*target_count)++;
      return true;
   }

   struct yttrium_stream_output_target *ytarget =
      (struct yttrium_stream_output_target *)target;
   struct yttrium_resource *res =
      target->buffer ? yttrium_resource(target->buffer) : NULL;

   if (!res || !target->buffer_size ||
       target->buffer_offset > res->size ||
       target->buffer_size > res->size - target->buffer_offset ||
       ytarget->append_offset >= target->buffer_size) {
      YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped stream output bad target slot=%u target=%p res=%p offset=0x%x size=0x%x append=0x%x res_size=0x%llx\n",
                   slot, target, res, target->buffer_offset,
                   target->buffer_size, ytarget->append_offset,
                   res ? (unsigned long long)res->size : 0);
      return false;
   }

   if (!yttrium_venus_create_stream_output_buffer(screen->venus,
                                                  &res->venus,
                                                  res->size,
                                                  &res->venus_mem_id)) {
      YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped stream output buffer unavailable slot=%u res=%p res_id=%u size=0x%llx bind=0x%x initialized=%u usage=0x%x\n",
                   slot, res, res->venus_res_id,
                   (unsigned long long)res->size,
                   res->base.bind, res->venus.initialized,
                   res->venus.buffer_usage);
      return false;
   }
   if (!res->venus_res_id)
      res->venus_res_id = (uint32_t)res->venus.memory_obj.id;

   /* SO may overwrite only part of the target. Publish initial CPU bytes
    * first, but never replay a detached shadow after a GPU write. Later CPU
    * updates to GPU-owned buffers publish their exact range immediately.
    */
   if (res->data_dirty) {
      if (res->gpu_buffer_written || !res->data ||
          res->size > res->data_capacity ||
          !yttrium_venus_update_buffer(screen->venus, &res->venus, 0,
                                        res->size, res->data)) {
         YTTRIUM_WARN("yttrium: stream output publication failed owner=yttrium-pipeline reason=initial-buffer-publication-failed slot=%u gpu_written=%u action=fail-draw\n",
                      slot, res->gpu_buffer_written);
         return false;
      }
      res->data_dirty = false;
      res->venus.contents_initialized = true;
   }

   if (!yttrium_venus_create_stream_output_buffer(screen->venus,
                                                  &ytarget->counter,
                                                  4, NULL)) {
      YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped stream output counter unavailable slot=%u target=%p\n",
                   slot, target);
      return false;
   }

   /*
    * Bind from where writing starts.  The window is (buffer_offset,
    * buffer_size); the append offset moves the start within it, so the size
    * has to come down by the same amount or the binding runs off the end.
    * On a resume the counter buffer carries the position and append_offset
    * stays where the last explicit offset put it, so the binding is stable
    * across begin/end.
    */
   targets[*target_count] = (struct yttrium_venus_stream_output_target) {
      .resource = &res->venus,
      .resource_id = res->venus_res_id,
      .buffer_offset = target->buffer_offset + ytarget->append_offset,
      .buffer_size = target->buffer_size - ytarget->append_offset,
      .counter_resource = &ytarget->counter,
      .counter_buffer_valid = ytarget->counter_buffer_valid,
   };
   (*target_count)++;
   return true;
}

static bool
yttrium_pipeline_collect_stream_output_targets(
   struct pipe_context *ctx,
   struct yttrium_context *yctx,
   struct yttrium_venus_stream_output_target *targets,
   uint32_t *target_count)
{
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);

   if (!targets || !target_count)
      return false;

   *target_count = 0;
   if (!yctx->num_so_targets)
      return true;

   /*
    * Keep num_so_targets as the frontend-requested slot count: draw_vbo uses
    * it to route SO-only draws through the dummy render target.  An all-NULL
    * binding has no transform-feedback destination, though, so do not turn it
    * into dummy Venus buffers or begin transform feedback.  Mixed bindings
    * still need their NULL placeholders to preserve output-buffer slots.
    */
   bool has_live_target = false;
   for (unsigned i = 0; i < yctx->num_so_targets; i++) {
      if (yctx->so_targets[i]) {
         has_live_target = true;
         break;
      }
   }
   if (!has_live_target)
      return true;

   if (!yttrium_venus_transform_feedback_enabled(screen->venus)) {
      YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped stream output transform feedback unavailable targets=%u\n",
                   yctx->num_so_targets);
      return false;
   }

   if (yctx->num_so_targets > YTTRIUM_VENUS_MAX_STREAM_OUTPUT_TARGETS) {
      YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped stream output target count unsupported targets=%u max=%u\n",
                   yctx->num_so_targets,
                   YTTRIUM_VENUS_MAX_STREAM_OUTPUT_TARGETS);
      return false;
   }

   for (unsigned i = 0; i < yctx->num_so_targets; i++) {
      if (!yttrium_pipeline_collect_stream_output_target(
             ctx, yctx->so_targets[i], i, targets, target_count))
         return false;
   }

   return true;
}

static uint32_t
yttrium_pipeline_stream_output_stride(const struct yttrium_context *yctx,
                                      unsigned output_buffer)
{
   const struct yttrium_shader_state *gs =
      yctx ? yctx->shaders[MESA_SHADER_GEOMETRY] : NULL;
   const struct yttrium_shader_state *vs =
      yctx ? yctx->shaders[MESA_SHADER_VERTEX] : NULL;
   const struct yttrium_shader_state *shader =
      gs && gs->stream_output.num_outputs ? gs : vs;

   if (!shader || output_buffer >= PIPE_MAX_SO_BUFFERS)
      return 0;

   return shader->stream_output.stride[output_buffer] * sizeof(uint32_t);
}

static void
yttrium_pipeline_record_current_identity(
   struct yttrium_context *yctx,
   const struct yttrium_resource *dst,
   const struct yttrium_resource *zs,
   const struct yttrium_venus_draw_state *draw_state)
{
   if (!yctx || !draw_state)
      return;

   yctx->current_pipeline_state_serial = yctx->pipeline_state_serial;
   yctx->current_pipeline_dst_cache_id = dst ? dst->cache_id : 0;
   yctx->current_pipeline_dst_image_id =
      dst ? dst->venus.image_obj.id : 0;
   yctx->current_pipeline_zs_cache_id = zs ? zs->cache_id : 0;
   yctx->current_pipeline_zs_image_id =
      zs ? zs->venus.image_obj.id : 0;
   yctx->current_pipeline_topology = draw_state->topology;
   yctx->current_pipeline_primitive_restart_enable =
      draw_state->primitive_restart_enable;
   yctx->current_pipeline_targetless_stream_output =
      draw_state->targetless_stream_output;
   yctx->current_pipeline_viewport_count = draw_state->viewport_count;
   yctx->current_pipeline_rasterization_samples =
      draw_state->rasterization_samples;
   yctx->current_pipeline_forced_sample_count =
      draw_state->forced_sample_count;
   yctx->current_pipeline_forced_sample_interlock =
      draw_state->forced_sample_interlock;
   yctx->current_pipeline_rt_count = draw_state->rt_count;
   yctx->current_pipeline_render_width = draw_state->render_width;
   yctx->current_pipeline_render_height = draw_state->render_height;
   yctx->current_pipeline_render_layers = draw_state->render_layers;
   yctx->current_pipeline_depth_level = draw_state->depth_level;
   yctx->current_pipeline_depth_layer = draw_state->depth_layer;
   yctx->current_pipeline_depth_layers = draw_state->depth_layers;
}

static bool
yttrium_pipeline_current_identity_equal(
   const struct yttrium_context *yctx,
   const struct yttrium_resource *dst,
   const struct yttrium_resource *zs,
   const struct yttrium_venus_draw_state *draw_state)
{
   if (!yctx || !yctx->pipeline || !draw_state ||
       yctx->current_pipeline_state_serial != yctx->pipeline_state_serial)
      return false;

   return yctx->current_pipeline_dst_cache_id ==
             (dst ? dst->cache_id : 0) &&
          yctx->current_pipeline_dst_image_id ==
             (dst ? dst->venus.image_obj.id : 0) &&
          yctx->current_pipeline_zs_cache_id ==
             (zs ? zs->cache_id : 0) &&
          yctx->current_pipeline_zs_image_id ==
             (zs ? zs->venus.image_obj.id : 0) &&
          yctx->current_pipeline_topology == draw_state->topology &&
          yctx->current_pipeline_primitive_restart_enable ==
             draw_state->primitive_restart_enable &&
          yctx->current_pipeline_targetless_stream_output ==
             draw_state->targetless_stream_output &&
          yctx->current_pipeline_viewport_count ==
             draw_state->viewport_count &&
          yctx->current_pipeline_rasterization_samples ==
             draw_state->rasterization_samples &&
          yctx->current_pipeline_forced_sample_count ==
             draw_state->forced_sample_count &&
          yctx->current_pipeline_forced_sample_interlock ==
             draw_state->forced_sample_interlock &&
          yctx->current_pipeline_rt_count == draw_state->rt_count &&
          yctx->current_pipeline_render_width == draw_state->render_width &&
          yctx->current_pipeline_render_height == draw_state->render_height &&
          yctx->current_pipeline_render_layers == draw_state->render_layers &&
          yctx->current_pipeline_depth_level == draw_state->depth_level &&
          yctx->current_pipeline_depth_layer == draw_state->depth_layer &&
          yctx->current_pipeline_depth_layers == draw_state->depth_layers;
}

static struct yttrium_pipeline *
yttrium_pipeline_get(struct yttrium_context *yctx,
                     struct yttrium_resource *dst,
                     struct yttrium_resource *zs,
                     const struct yttrium_venus_draw_state *draw_state)
{
   struct yttrium_screen *screen = yttrium_screen(yctx->base.screen);
   const struct yttrium_shader_state *vs =
      yctx->shaders[MESA_SHADER_VERTEX];
   const struct yttrium_shader_state *tcs =
      yctx->shaders[MESA_SHADER_TESS_CTRL];
   const struct yttrium_shader_state *tes =
      yctx->shaders[MESA_SHADER_TESS_EVAL];
   const struct yttrium_shader_state *gs =
      yctx->shaders[MESA_SHADER_GEOMETRY];
   const struct yttrium_shader_state *fs_raw =
      yctx->shaders[MESA_SHADER_FRAGMENT];
   const struct yttrium_shader_state *fs =
      yttrium_pipeline_fragment_shader_absent(fs_raw) ? NULL : fs_raw;
   struct yttrium_pipeline_key key;

   if (yttrium_pipeline_current_identity_equal(yctx, dst, zs, draw_state))
      return yctx->pipeline;

   if (!yttrium_pipeline_build_key(yctx, dst, zs, draw_state, &key))
      return NULL;
   const uint32_t key_hash = yttrium_pipeline_key_hash(&key);

   if (yctx->pipeline && yttrium_pipeline_key_equal(&yctx->pipeline->key,
                                                    &key)) {
      yttrium_pipeline_record_current_identity(yctx, dst, zs, draw_state);
      return yctx->pipeline;
   }

   const uint32_t bucket =
      key_hash & (yctx->pipeline_cache_hash_size - 1);
   for (uint32_t i = yctx->pipeline_cache_hash_heads[bucket];
        i != UINT32_MAX; i = yctx->pipeline_cache_hash_next[i]) {
      if (yctx->pipeline_cache[i]->key_hash == key_hash &&
          yttrium_pipeline_key_equal(&yctx->pipeline_cache[i]->key, &key)) {
         yctx->pipeline = yctx->pipeline_cache[i];
         yttrium_pipeline_record_current_identity(yctx, dst, zs, draw_state);
         return yctx->pipeline;
      }
   }

   struct yttrium_pipeline *pipeline = CALLOC_STRUCT(yttrium_pipeline);
   if (!pipeline)
      return NULL;

   struct yttrium_shader_state *generated_vs = NULL;
   struct yttrium_shader_state *generated_gs = NULL;
   struct yttrium_shader_state *generated_fs = NULL;
   const struct yttrium_shader_state *pipeline_vs = vs;
   const struct yttrium_shader_state *pipeline_gs = gs;
   const struct yttrium_shader_state *pipeline_fs = fs;
   bool emulate_forced_sample_mask =
      !key.forced_sample_count && draw_state->forced_sample_count > 1 &&
      key.rasterization_samples > 1 && pipeline_fs &&
      yttrium_shader_state_uses_sample_mask_in(pipeline_fs);
   if (draw_state->forced_sample_count > 1) {
      /* stub: SV_Coverage is faked as fully covered, so the app sees binary
       * coverage instead of antialiased edges.
       */
      YTTRIUM_WARN("yttrium: forced-sample draw %s fs=%u requested=%u hw=%u expand=%u coverage=%s rt_format=%u %ux%u\n",
                  yttrium_trace_process_name(),
                  key.fs_id, draw_state->forced_sample_count,
                  key.forced_sample_count, key.forced_sample_expand,
                  emulate_forced_sample_mask ? "stub" :
                  key.forced_sample_expand ? "expanded" :
                  key.forced_sample_count ? "native" : "none",
                  key.rt_count ? key.rt_format[0] : VK_FORMAT_UNDEFINED,
                  key.width, key.height);
   }

   if (!gs && key.gs_id == UINT32_MAX) {
      unsigned cull_distance_slot = 0;
      generated_vs =
         yttrium_shader_create_cull_distance_vs(&yctx->base, vs,
                                                &cull_distance_slot);
      if (!generated_vs) {
         YTTRIUM_WARN("yttrium: shader_draw_probe pipeline create skipped generated cull-distance VS failed vs=%u topology=%u\n",
                      key.vs_id, key.topology);
         FREE(pipeline);
         return NULL;
      }
      pipeline_vs = generated_vs;

      const bool passthrough_prim_id =
         fs && fs->nir &&
         (fs->nir->info.inputs_read & VARYING_BIT_PRIMITIVE_ID);
      generated_gs =
         yttrium_shader_create_cull_distance_gs(
            &yctx->base, generated_vs,
            yttrium_pipeline_prim_from_topology(key.topology),
            passthrough_prim_id, cull_distance_slot);
      if (!generated_gs) {
         YTTRIUM_WARN("yttrium: shader_draw_probe pipeline create skipped generated cull-distance GS failed vs=%u topology=%u\n",
                      key.vs_id, key.topology);
         yttrium_shader_state_destroy(screen, generated_vs);
         FREE(pipeline);
         return NULL;
      }
      pipeline_gs = generated_gs;
   }

   if (key.dual_source_blend && fs) {
      generated_fs =
         yttrium_shader_create_dual_source_fs(&yctx->base, fs);
      if (!generated_fs) {
         YTTRIUM_WARN("yttrium: shader_draw_probe pipeline create skipped generated dual-source FS failed fs=%u\n",
                      key.fs_id);
         if (generated_gs)
            yttrium_shader_state_destroy(screen, generated_gs);
         if (generated_vs)
            yttrium_shader_state_destroy(screen, generated_vs);
         FREE(pipeline);
         return NULL;
      }
      pipeline_fs = generated_fs;
   }

   if (key.a8_rt_mask && pipeline_fs) {
      struct yttrium_shader_state *a8_fs =
         yttrium_shader_create_a8_rt_fs(&yctx->base, pipeline_fs,
                                        key.a8_rt_mask);
      if (!a8_fs) {
         YTTRIUM_WARN("yttrium: shader_draw_probe pipeline create skipped generated A8 RT FS failed fs=%u mask=0x%x\n",
                      key.fs_id, key.a8_rt_mask);
         if (generated_fs)
            yttrium_shader_state_destroy(screen, generated_fs);
         if (generated_gs)
            yttrium_shader_state_destroy(screen, generated_gs);
         if (generated_vs)
            yttrium_shader_state_destroy(screen, generated_vs);
         FREE(pipeline);
         return NULL;
      }
      if (generated_fs)
         yttrium_shader_state_destroy(screen, generated_fs);
      generated_fs = a8_fs;
      pipeline_fs = generated_fs;
   }

   if (key.alpha_test_enable && pipeline_fs) {
      struct yttrium_shader_state *alpha_test_fs =
         yttrium_shader_create_alpha_test_fs(&yctx->base, pipeline_fs,
                                             key.alpha_func,
                                             key.alpha_ref_value);
      if (!alpha_test_fs) {
         YTTRIUM_WARN("yttrium: shader_draw_probe pipeline create skipped generated alpha-test FS failed fs=%u func=%u ref=%f\n",
                      key.fs_id, key.alpha_func, key.alpha_ref_value);
         if (generated_fs)
            yttrium_shader_state_destroy(screen, generated_fs);
         if (generated_gs)
            yttrium_shader_state_destroy(screen, generated_gs);
         if (generated_vs)
            yttrium_shader_state_destroy(screen, generated_vs);
         FREE(pipeline);
         return NULL;
      }
      if (generated_fs)
         yttrium_shader_state_destroy(screen, generated_fs);
      generated_fs = alpha_test_fs;
      pipeline_fs = generated_fs;
   }

   if (key.forced_sample_interlock && pipeline_fs) {
      struct yttrium_shader_state *interlock_fs =
         yttrium_shader_create_forced_sample_interlock_fs(
            &yctx->base, pipeline_fs,
            YTTRIUM_FORCED_SAMPLE_INTERLOCK_IMAGE_SLOT);
      if (!interlock_fs) {
         YTTRIUM_WARN("yttrium: forced-sample-interlock pipeline create failed fs=%u samples=%u\n",
                      key.fs_id, key.forced_sample_count);
         if (generated_fs)
            yttrium_shader_state_destroy(screen, generated_fs);
         if (generated_gs)
            yttrium_shader_state_destroy(screen, generated_gs);
         if (generated_vs)
            yttrium_shader_state_destroy(screen, generated_vs);
         FREE(pipeline);
         return NULL;
      }
      if (generated_fs)
         yttrium_shader_state_destroy(screen, generated_fs);
      generated_fs = interlock_fs;
      pipeline_fs = generated_fs;
   }

   if (emulate_forced_sample_mask && pipeline_fs) {
      struct yttrium_shader_state *forced_sample_fs =
         yttrium_shader_create_forced_sample_mask_fs(
            &yctx->base, pipeline_fs, key.rasterization_samples);
      if (!forced_sample_fs) {
         YTTRIUM_WARN("yttrium: shader_draw_probe pipeline create skipped generated forced-sample-mask FS failed fs=%u samples=%u\n",
                      key.fs_id, key.rasterization_samples);
         if (generated_fs)
            yttrium_shader_state_destroy(screen, generated_fs);
         if (generated_gs)
            yttrium_shader_state_destroy(screen, generated_gs);
         if (generated_vs)
            yttrium_shader_state_destroy(screen, generated_vs);
         FREE(pipeline);
         return NULL;
      }
      if (generated_fs)
         yttrium_shader_state_destroy(screen, generated_fs);
      generated_fs = forced_sample_fs;
      pipeline_fs = generated_fs;
   }

   if (key.forced_sample_expand && pipeline_fs) {
      struct yttrium_shader_state *expand_fs =
         yttrium_shader_create_sample_mask_expand_fs(
            &yctx->base, pipeline_fs, key.forced_sample_count,
            key.forced_sample_expand);
      if (!expand_fs) {
         YTTRIUM_WARN("yttrium: shader_draw_probe pipeline create skipped generated sample-mask-expand FS failed fs=%u hw_samples=%u app_samples=%u\n",
                      key.fs_id, key.forced_sample_count,
                      key.forced_sample_expand);
         if (generated_fs)
            yttrium_shader_state_destroy(screen, generated_fs);
         if (generated_gs)
            yttrium_shader_state_destroy(screen, generated_gs);
         if (generated_vs)
            yttrium_shader_state_destroy(screen, generated_vs);
         FREE(pipeline);
         return NULL;
      }
      if (generated_fs)
         yttrium_shader_state_destroy(screen, generated_fs);
      generated_fs = expand_fs;
      pipeline_fs = generated_fs;
   }

   struct yttrium_venus_ubo_binding_layout ubo_layouts
      [YTTRIUM_VENUS_MAX_PIPELINE_UBO_BINDINGS];
   uint32_t ubo_layout_count = 0;
   if (!yttrium_pipeline_build_ubo_layouts(pipeline_vs, tcs, tes,
                                           pipeline_gs, pipeline_fs,
                                           ubo_layouts,
                                           &ubo_layout_count)) {
      YTTRIUM_WARN("yttrium: shader_draw_probe pipeline create skipped ubo layout vs_mask=0x%x tcs_mask=0x%x tes_mask=0x%x gs_mask=0x%x fs_mask=0x%x\n",
                   key.vs_ubo_used_mask, key.tcs_ubo_used_mask,
                   key.tes_ubo_used_mask, key.gs_ubo_used_mask,
                   key.fs_ubo_used_mask);
      if (generated_fs)
         yttrium_shader_state_destroy(screen, generated_fs);
      if (generated_gs)
         yttrium_shader_state_destroy(screen, generated_gs);
      if (generated_vs)
         yttrium_shader_state_destroy(screen, generated_vs);
      FREE(pipeline);
      return NULL;
   }

   const uint32_t sampled_binding_count = key.sampled_binding_count;
   const uint64_t storage_image_mask = key.storage_image_mask;
   const uint64_t storage_buffer_mask = key.storage_buffer_mask;
   struct yttrium_venus_resource *color_resources[PIPE_MAX_COLOR_BUFS];
   uint32_t color_resource_ids[PIPE_MAX_COLOR_BUFS];
   memset(color_resources, 0, sizeof(color_resources));
   memset(color_resource_ids, 0, sizeof(color_resource_ids));
   for (uint32_t i = 0; i < key.rt_count; i++) {
      struct yttrium_resource *rt = NULL;
      if (yctx->fb.nr_cbufs) {
         if (i < yctx->fb.nr_cbufs && yctx->fb.cbufs[i].texture)
            rt = yttrium_resource(yctx->fb.cbufs[i].texture);
      } else {
         rt = dst;
      }

      color_resources[i] = rt ? &rt->venus : NULL;
      color_resource_ids[i] = rt ? rt->venus_res_id : 0;
   }

   pipeline->key = key;
   pipeline->key_hash = key_hash;
   struct yttrium_venus_draw_state x8_draw_state;
   const struct yttrium_venus_draw_state *pipeline_draw_state = draw_state;
   if (key.x8_rt_mask) {
      yttrium_pipeline_apply_x8_rt_state(&key, draw_state, &x8_draw_state);
      pipeline_draw_state = &x8_draw_state;
   }
   struct yttrium_venus_draw_state forced_sample_draw_state;
   if (emulate_forced_sample_mask || key.forced_sample_count ||
       draw_state->forced_sample_count > 1) {
      forced_sample_draw_state = *pipeline_draw_state;
      if (emulate_forced_sample_mask) {
         forced_sample_draw_state.rasterization_samples = 1;
         forced_sample_draw_state.forced_sample_count = 0;
      } else if (key.forced_sample_count) {
         /* Possibly downgraded from what the app asked for; the backend keys
          * its MSRTSS decision off these, so hand it the count we can run.
          */
         forced_sample_draw_state.forced_sample_count =
            key.forced_sample_count;
         forced_sample_draw_state.rasterization_samples =
            key.forced_sample_count;
      }
      if (draw_state->forced_sample_count > 1)
         forced_sample_draw_state.alpha_to_coverage_enable = VK_FALSE;
      pipeline_draw_state = &forced_sample_draw_state;
   }

   if (!yttrium_venus_pipeline_init(screen->venus, pipeline,
                                     &dst->venus, dst->venus_res_id,
                                     color_resources, color_resource_ids,
                                     key.rt_count,
                                     zs ? &zs->venus : NULL,
                                     zs ? zs->venus_res_id : 0,
                                     pipeline_vs->module,
                                     tcs ? tcs->module : VK_NULL_HANDLE,
                                     tes ? tes->module : VK_NULL_HANDLE,
                                     pipeline_gs ? pipeline_gs->module :
                                        VK_NULL_HANDLE,
                                     pipeline_fs ? pipeline_fs->module :
                                        VK_NULL_HANDLE,
                                     key.bindings, key.num_bindings,
                                     key.binding_divisors,
                                     key.attribs, key.num_attribs,
                                     ubo_layouts, ubo_layout_count,
                                     sampled_binding_count ?
                                        key.sampled_bindings : NULL,
                                     sampled_binding_count,
                                     storage_image_mask,
                                     storage_buffer_mask,
                                     yttrium_pipeline_sampled_stage_flags(
                                        key.storage_stage_mask),
                                     pipeline_draw_state)) {
      YTTRIUM_WARN("yttrium: shader_draw_probe pipeline create failed res_id=%u zs_res_id=%u vs=%u tcs=%u tes=%u gs=%u fs=%u vs_hash=0x%llx tcs_hash=0x%llx tes_hash=0x%llx gs_hash=0x%llx fs_hash=0x%llx rt_count=%u format0=%u zs_format=%u topology=%u patch_vertices=%u bindings=%u attribs=%u targetless_so=%u discard=%u\n",
                   dst->venus_res_id, zs ? zs->venus_res_id : 0,
                   key.vs_id, key.tcs_id, key.tes_id, key.gs_id, key.fs_id,
                   (unsigned long long)key.vs_hash,
                   (unsigned long long)key.tcs_hash,
                   (unsigned long long)key.tes_hash,
                   (unsigned long long)key.gs_hash,
                   (unsigned long long)key.fs_hash,
                   key.rt_count, key.rt_format[0], key.zs_format,
                   key.topology, key.patch_vertices,
                   key.num_bindings, key.num_attribs,
                   pipeline_draw_state->targetless_stream_output,
                   pipeline_draw_state->rasterizer_discard_enable);
      if (generated_gs)
         yttrium_shader_state_destroy(screen, generated_gs);
      if (generated_fs)
         yttrium_shader_state_destroy(screen, generated_fs);
      if (generated_vs)
         yttrium_shader_state_destroy(screen, generated_vs);
      FREE(pipeline);
      return NULL;
   }

   pipeline->generated_vs = generated_vs;
   pipeline->generated_gs = generated_gs;
   pipeline->generated_fs = generated_fs;

   for (uint32_t i = 0; i < key.rt_count; i++) {
      struct yttrium_resource *rt = NULL;
      if (yctx->fb.nr_cbufs) {
         if (i < yctx->fb.nr_cbufs && yctx->fb.cbufs[i].texture)
            rt = yttrium_resource(yctx->fb.cbufs[i].texture);
      } else {
         rt = dst;
      }

      if (rt)
         pipe_resource_reference(&pipeline->rt_resources[i], &rt->base);
   }
   if (zs)
      pipe_resource_reference(&pipeline->zs_resource, &zs->base);

   YTTRIUM_LOG("yttrium: shader_draw_probe pipeline create ok pipeline_id=%llu res_id=%u zs_res_id=%u vs=%u tcs=%u tes=%u gs=%u fs=%u rt_count=%u format0=%u zs_format=%u topology=%u patch_vertices=%u bindings=%u attribs=%u ubo_layouts=%u sampled_bindings=%u sampled_stage=0x%x vs_ubo=0x%x tcs_ubo=0x%x tes_ubo=0x%x gs_ubo=0x%x fs_ubo=0x%x depth_test=%u depth_write=%u depth_compare=%u\n",
               (unsigned long long)pipeline->pipeline_obj.id,
               dst->venus_res_id, zs ? zs->venus_res_id : 0,
               key.vs_id, key.tcs_id, key.tes_id, key.gs_id, key.fs_id,
               key.rt_count, key.rt_format[0], key.zs_format,
               key.topology, key.patch_vertices,
               key.num_bindings, key.num_attribs,
               ubo_layout_count, sampled_binding_count,
               key.sampled_stage_mask,
               key.vs_ubo_used_mask, key.tcs_ubo_used_mask,
               key.tes_ubo_used_mask, key.gs_ubo_used_mask,
               key.fs_ubo_used_mask,
               key.depth_test_enable, key.depth_write_enable,
               key.depth_compare_op);
   yttrium_trace_debug_stringf(
      "yttrium: shader_draw_probe pipeline create ok pipeline_id=%llu res_id=%u zs_res_id=%u vs=%u tcs=%u tes=%u gs=%u fs=%u rt_count=%u format0=%u zs_format=%u topology=%u patch_vertices=%u bindings=%u attribs=%u ubo_layouts=%u sampled_bindings=%u sampled_stage=0x%x vs_ubo=0x%x tcs_ubo=0x%x tes_ubo=0x%x gs_ubo=0x%x fs_ubo=0x%x depth_test=%u depth_write=%u depth_compare=%u",
      (unsigned long long)pipeline->pipeline_obj.id,
      dst->venus_res_id, zs ? zs->venus_res_id : 0,
      key.vs_id, key.tcs_id, key.tes_id, key.gs_id, key.fs_id,
      key.rt_count, key.rt_format[0], key.zs_format, key.topology,
      key.patch_vertices,
      key.num_bindings, key.num_attribs,
      ubo_layout_count, sampled_binding_count,
      key.sampled_stage_mask,
      key.vs_ubo_used_mask, key.tcs_ubo_used_mask,
      key.tes_ubo_used_mask, key.gs_ubo_used_mask,
      key.fs_ubo_used_mask,
      key.depth_test_enable, key.depth_write_enable,
      key.depth_compare_op);
   YTTRIUM_LOG("yttrium: pipeline_create native pipeline=%p pipeline_id=%llu vs=%u tcs=%u tes=%u gs=%u fs=%u vs_hash=0x%llx tcs_hash=0x%llx tes_hash=0x%llx gs_hash=0x%llx fs_hash=0x%llx rt_count=%u rt_format0=%u zs_format=%u image_id0=%llu zs_image_id=%llu topology=%u patch_vertices=%u bindings=%u attribs=%u ubo_layouts=%u sampled_bindings=%u sampled_stage=0x%x vs_ubo=0x%x tcs_ubo=0x%x tes_ubo=0x%x gs_ubo=0x%x fs_ubo=0x%x depth_test=%u depth_write=%u depth_compare=%u\n",
                pipeline,
                (unsigned long long)pipeline->pipeline_obj.id,
                key.vs_id, key.tcs_id, key.tes_id, key.gs_id, key.fs_id,
                (unsigned long long)key.vs_hash,
                (unsigned long long)key.tcs_hash,
                (unsigned long long)key.tes_hash,
                (unsigned long long)key.gs_hash,
                (unsigned long long)key.fs_hash,
                key.rt_count,
                key.rt_format[0],
                key.zs_format,
                (unsigned long long)key.dst_image_id[0],
                (unsigned long long)key.zs_image_id,
                key.topology, key.patch_vertices,
                key.num_bindings, key.num_attribs,
                ubo_layout_count, sampled_binding_count,
                key.sampled_stage_mask,
                key.vs_ubo_used_mask, key.tcs_ubo_used_mask,
                key.tes_ubo_used_mask, key.gs_ubo_used_mask,
                key.fs_ubo_used_mask,
                key.depth_test_enable, key.depth_write_enable,
                key.depth_compare_op);
   uint32_t cache_slot;
   if (yctx->pipeline_cache_count < yctx->pipeline_cache_size) {
      cache_slot = yctx->pipeline_cache_count++;
   } else {
      cache_slot = yctx->pipeline_cache_next;
      if (yctx->pipeline_cache[cache_slot] == yctx->pipeline)
         yctx->pipeline = NULL;
      yttrium_pipeline_cache_hash_remove(yctx, cache_slot);
      yttrium_pipeline_destroy(screen->venus,
                               yctx->pipeline_cache[cache_slot]);
   }

   yctx->pipeline_cache[cache_slot] = pipeline;
   yttrium_pipeline_cache_hash_insert(yctx, cache_slot);
   yctx->pipeline_cache_next =
      (cache_slot + 1) % yctx->pipeline_cache_size;
   yctx->pipeline = pipeline;
   yttrium_pipeline_record_current_identity(yctx, dst, zs, draw_state);
   return pipeline;
}

struct yttrium_pipeline_draw_upload {
   struct yttrium_venus_vertex_upload vertex_uploads
      [YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS];
   void *owned_vertex_data[YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS];
   uint32_t vertex_upload_count;
   VkDeviceSize vertex_data_size;
   uint32_t first_vertex;
   uint32_t vertex_count;
   uint32_t instance_count;
   const void *index_data;
   size_t index_data_size;
   struct yttrium_venus_resource *index_resource;
   uint32_t index_resource_id;
   VkDeviceSize index_buffer_offset;
   bool index_host_write_pending;
   uint32_t index_count;
   VkIndexType index_type;
   int32_t vertex_offset;
};

static void
yttrium_pipeline_draw_upload_cleanup(struct yttrium_pipeline_draw_upload *upload)
{
   if (!upload)
      return;

   for (uint32_t i = 0; i < upload->vertex_upload_count; i++) {
      FREE(upload->owned_vertex_data[i]);
      upload->owned_vertex_data[i] = NULL;
   }
}

static void
yttrium_pipeline_trace_draw_upload(
   const struct yttrium_context *yctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_start_count_bias *draw,
   const struct yttrium_pipeline_draw_upload *upload)
{
   if (!yttrium_pipeline_verbose_trace_enabled() || !info || !draw || !upload)
      return;

   const struct yttrium_venus_vertex_upload *vb0 =
      upload->vertex_upload_count ? &upload->vertex_uploads[0] : NULL;
   const struct yttrium_pipeline_data_summary vb0_summary =
      vb0 ? yttrium_pipeline_summarize_data(vb0->data, vb0->size) :
            (struct yttrium_pipeline_data_summary) { 0 };

   yttrium_trace_debug_stringf(
      "yttrium: shader_draw_probe draw upload mode=%u index_size=%u restart=%u instances=%u start_instance=%u draw_count=%u start=%u bias=%d bounds=%u min=%u max=%u first_vertex=%u vertex_count=%u upload_instances=%u vertex_uploads=%u vertex_bytes=0x%llx indexed=%u index_count=%u index_type=%u index_bytes=0x%llx first_indices=%u,%u,%u,%u vb0_size=0x%llx vb0_first=%08x,%08x,%08x,%08x vb0_float=%f,%f,%f,%f,%f,%f,%f,%f vb0_inspected=0x%llx vb0_nonzero=0x%llx vb0_first_nz=0x%llx:%08x vb0_last_nz=0x%llx:%08x vb0_hash=0x%x num_vb=%u",
      info->mode,
      info->index_size,
      info->primitive_restart,
      info->instance_count,
      info->start_instance,
      draw->count,
      draw->start,
      draw->index_bias,
      info->index_bounds_valid,
      info->min_index,
      info->max_index,
      upload->first_vertex,
      upload->vertex_count,
      upload->instance_count,
      upload->vertex_upload_count,
      (unsigned long long)upload->vertex_data_size,
      upload->index_count != 0,
      upload->index_count,
      upload->index_type,
      (unsigned long long)upload->index_data_size,
      yttrium_pipeline_peek_index(upload->index_data,
                                  upload->index_data_size, 0,
                                  upload->index_type),
      yttrium_pipeline_peek_index(upload->index_data,
                                  upload->index_data_size, 1,
                                  upload->index_type),
      yttrium_pipeline_peek_index(upload->index_data,
                                  upload->index_data_size, 2,
                                  upload->index_type),
      yttrium_pipeline_peek_index(upload->index_data,
                                  upload->index_data_size, 3,
                                  upload->index_type),
      vb0 ? (unsigned long long)vb0->size : 0,
      vb0 ? yttrium_pipeline_peek_u32(vb0->data, vb0->size, 0) : 0,
      vb0 ? yttrium_pipeline_peek_u32(vb0->data, vb0->size, 1) : 0,
      vb0 ? yttrium_pipeline_peek_u32(vb0->data, vb0->size, 2) : 0,
      vb0 ? yttrium_pipeline_peek_u32(vb0->data, vb0->size, 3) : 0,
      vb0 ? yttrium_pipeline_peek_f32(vb0->data, vb0->size, 0) : 0.0f,
      vb0 ? yttrium_pipeline_peek_f32(vb0->data, vb0->size, 1) : 0.0f,
      vb0 ? yttrium_pipeline_peek_f32(vb0->data, vb0->size, 2) : 0.0f,
      vb0 ? yttrium_pipeline_peek_f32(vb0->data, vb0->size, 3) : 0.0f,
      vb0 ? yttrium_pipeline_peek_f32(vb0->data, vb0->size, 4) : 0.0f,
      vb0 ? yttrium_pipeline_peek_f32(vb0->data, vb0->size, 5) : 0.0f,
      vb0 ? yttrium_pipeline_peek_f32(vb0->data, vb0->size, 6) : 0.0f,
      vb0 ? yttrium_pipeline_peek_f32(vb0->data, vb0->size, 7) : 0.0f,
      (unsigned long long)vb0_summary.inspected,
      (unsigned long long)vb0_summary.nonzero_bytes,
      (unsigned long long)vb0_summary.first_nonzero,
      vb0_summary.first_nonzero_u32,
      (unsigned long long)vb0_summary.last_nonzero,
      vb0_summary.last_nonzero_u32,
      vb0_summary.hash,
      yctx ? yctx->num_vertex_buffers : 0);

   const struct yttrium_shader_state *fs =
      yctx ? yctx->shaders[MESA_SHADER_FRAGMENT] : NULL;
   if (!yctx || !yctx->vertex_elements ||
       !yttrium_shader_state_is_sampled_texture_only(fs))
      return;

   for (unsigned i = 0; i < yctx->vertex_elements->num_bindings; i++) {
      const VkVertexInputBindingDescription *binding =
         &yctx->vertex_elements->bindings[i];
      yttrium_trace_debug_stringf(
         "yttrium: shader_draw_probe vi binding[%u] source_vb=%u stride=%u rate=%u",
         i,
         yctx->vertex_elements->binding_map[i],
         binding->stride,
         binding->inputRate);
   }

   for (unsigned i = 0;
        i < MIN2(yctx->vertex_elements->num_attribs, 8); i++) {
      const VkVertexInputAttributeDescription *attrib =
         &yctx->vertex_elements->attribs[i];
      const struct pipe_vertex_element *ve =
         &yctx->vertex_elements->elements[attrib->location];
      yttrium_trace_debug_stringf(
         "yttrium: shader_draw_probe vi attrib[%u] location=%u binding=%u vk_format=%u offset=%u source_vb=%u src_offset=%u src_stride=%u src_format=%u divisor=%u",
         i,
         attrib->location,
         attrib->binding,
         attrib->format,
         attrib->offset,
         ve->vertex_buffer_index,
         ve->src_offset,
         ve->src_stride,
         ve->src_format,
         ve->instance_divisor);
   }
}

static bool
yttrium_pipeline_index_type(uint16_t index_size, VkIndexType *out_type)
{
   if (!out_type)
      return false;

   switch (index_size) {
   case 2:
      *out_type = VK_INDEX_TYPE_UINT16;
      return true;
   case 4:
      *out_type = VK_INDEX_TYPE_UINT32;
      return true;
   default:
      return false;
   }
}

static bool
yttrium_pipeline_index_restart_value(VkIndexType type, uint32_t *out_value)
{
   if (!out_value)
      return false;

   switch (type) {
   case VK_INDEX_TYPE_UINT16:
      *out_value = UINT16_MAX;
      return true;
   case VK_INDEX_TYPE_UINT32:
      *out_value = UINT32_MAX;
      return true;
   default:
      return false;
   }
}

static bool
yttrium_pipeline_scan_index_bounds(const void *data, size_t size,
                                   uint32_t count, VkIndexType type,
                                   bool primitive_restart,
                                   uint32_t *out_min,
                                   uint32_t *out_max)
{
   if (!data || !count || !out_min || !out_max)
      return false;

   uint32_t restart_value = 0;
   if (primitive_restart &&
       !yttrium_pipeline_index_restart_value(type, &restart_value))
      return false;

   uint32_t min_index = UINT32_MAX;
   uint32_t max_index = 0;
   bool found = false;

   /*
    * This runs once per index of every indexed draw, so it is worth
    * specialising: peek_index() reads a single element through memcpy() behind
    * a type switch and a bounds check, which made this scan the single hottest
    * function in the driver under D3D10/11 (D3D9 supplies the bounds with the
    * draw and never gets here).  Hoisting the type and the bounds out of the
    * loop leaves a typed load and two compares per index.
    */
   size_t stride;
   switch (type) {
   case VK_INDEX_TYPE_UINT16:
      stride = sizeof(uint16_t);
      break;
   case VK_INDEX_TYPE_UINT32:
      stride = sizeof(uint32_t);
      break;
   default:
      stride = 0;
      break;
   }

   /* Elements wholly inside the buffer; peek_index() reads the rest as zero. */
   const uint32_t avail =
      stride ? (uint32_t)MIN2((uint64_t)count, (uint64_t)(size / stride)) : 0;

   if (!stride) {
      /* Unknown index type: every element reads as zero, as peek_index does. */
      for (uint32_t i = 0; i < count; i++) {
         const uint32_t index =
            yttrium_pipeline_peek_index(data, size, i, type);

         if (primitive_restart && index == restart_value)
            continue;

         min_index = MIN2(min_index, index);
         max_index = MAX2(max_index, index);
         found = true;
      }
   } else if ((uintptr_t)data & (uintptr_t)(stride - 1)) {
      /* A base address the typed loads below cannot use. */
      for (uint32_t i = 0; i < count; i++) {
         const uint32_t index =
            yttrium_pipeline_peek_index(data, size, i, type);

         if (primitive_restart && index == restart_value)
            continue;

         min_index = MIN2(min_index, index);
         max_index = MAX2(max_index, index);
         found = true;
      }
   } else {
      /*
       * Scan the mapping directly.  Buffers are mapped write-back cached (the
       * bind-buffer path asks for a HOST_CACHED memory type), so there is no
       * longer anything to gain from staging the range into scratch first -
       * that only made sense while these reads were uncached.
       *
       * Without primitive restart this is a plain min/max reduction, so keep
       * it free of anything the vectoriser would choke on: no stores in the
       * loop body, and `found` decided once at the end.
       */
      if (type == VK_INDEX_TYPE_UINT16) {
         const uint16_t *indices = (const uint16_t *)data;

         if (primitive_restart) {
            for (uint32_t i = 0; i < avail; i++) {
               const uint32_t index = indices[i];

               if (index == restart_value)
                  continue;

               min_index = MIN2(min_index, index);
               max_index = MAX2(max_index, index);
               found = true;
            }
         } else {
            for (uint32_t i = 0; i < avail; i++) {
               const uint32_t index = indices[i];

               min_index = MIN2(min_index, index);
               max_index = MAX2(max_index, index);
            }
            found = avail != 0;
         }
      } else {
         const uint32_t *indices = (const uint32_t *)data;

         if (primitive_restart) {
            for (uint32_t i = 0; i < avail; i++) {
               const uint32_t index = indices[i];

               if (index == restart_value)
                  continue;

               min_index = MIN2(min_index, index);
               max_index = MAX2(max_index, index);
               found = true;
            }
         } else {
            for (uint32_t i = 0; i < avail; i++) {
               const uint32_t index = indices[i];

               min_index = MIN2(min_index, index);
               max_index = MAX2(max_index, index);
            }
            found = avail != 0;
         }
      }

      /* A short buffer leaves a tail of zero-reading elements. */
      if (avail < count && !(primitive_restart && restart_value == 0)) {
         min_index = 0;
         found = true;
      }
   }

   if (!found)
      return false;

   *out_min = min_index;
   *out_max = max_index;
   return true;
}

/*
 * The resource whose cache may answer this draw's index bounds, or NULL when
 * the result must be recomputed.  User indices have no resource to hang a
 * cache on, and anything the GPU can write behind our back - stream output, a
 * shader buffer or image - would not bump contents_serial, so those are never
 * cached.
 */
static struct yttrium_resource *
yttrium_pipeline_index_bounds_cache_resource(
   const struct pipe_draw_info *info)
{
   if (info->has_user_indices || !info->index.resource)
      return NULL;

   struct yttrium_resource *res = yttrium_resource(info->index.resource);

   if (res->base.bind & (PIPE_BIND_STREAM_OUTPUT |
                         PIPE_BIND_SHADER_BUFFER |
                         PIPE_BIND_SHADER_IMAGE))
      return NULL;

   return res;
}

/*
 * Index bounds for a draw, reusing the previous answer when nothing that feeds
 * it has moved.  Scanning is bound by cold cache-line fetches out of
 * host-visible memory rather than by the loop, so the only real saving is not
 * reading at all - and an app that redraws static geometry asks the same
 * question about the same bytes every frame.
 *
 * res is NULL when the result must not be cached, which the caller decides.
 */
static uint32_t
yttrium_pipeline_index_bounds_set(const struct yttrium_context *yctx,
                                  const struct yttrium_resource *res,
                                  uint64_t offset,
                                  uint32_t count)
{
   if (!yctx || !res)
      return UINT32_MAX;

   uint64_t hash = res->cache_id * 0x9e3779b97f4a7c15ull;

   hash ^= offset + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
   hash ^= (uint64_t)count + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
   hash ^= hash >> 29;

   return hash % YTTRIUM_INDEX_BOUNDS_CACHE_SET_COUNT;
}

static bool
yttrium_pipeline_index_bounds(struct yttrium_context *yctx,
                              struct yttrium_resource *res,
                              const void *data,
                              size_t size,
                              uint64_t offset,
                              uint32_t count,
                              VkIndexType type,
                              bool primitive_restart,
                              uint32_t *out_min,
                              uint32_t *out_max)
{
   const uint32_t set_index =
      yttrium_pipeline_index_bounds_set(yctx, res, offset, count);
   struct yttrium_index_bounds_entry *set = set_index != UINT32_MAX ?
      yctx->index_bounds_cache[set_index] : NULL;
   struct yttrium_index_bounds_entry *cache = NULL;
   struct yttrium_index_bounds_entry *empty = NULL;

   for (uint32_t way = 0;
        set && way < YTTRIUM_INDEX_BOUNDS_CACHE_WAYS;
        way++) {
      struct yttrium_index_bounds_entry *entry = &set[way];

      if (!entry->valid) {
         if (!empty)
            empty = entry;
         continue;
      }

      if (entry->cache_id == res->cache_id &&
          entry->offset == offset &&
          entry->count == count &&
          entry->type == (uint32_t)type &&
          entry->restart == primitive_restart) {
         cache = entry;
         break;
      }
   }

   if (cache && cache->serial == res->contents_serial) {
      *out_min = cache->min;
      *out_max = cache->max;
      return true;
   }

   if (!yttrium_pipeline_scan_index_bounds(data, size, count, type,
                                           primitive_restart,
                                           out_min, out_max))
      return false;

   if (set) {
      if (!cache) {
         if (empty) {
            cache = empty;
         } else {
            const uint32_t way =
               yctx->index_bounds_cache_next[set_index]++ %
               YTTRIUM_INDEX_BOUNDS_CACHE_WAYS;
            cache = &set[way];
         }
      }

      cache->cache_id = res->cache_id;
      cache->serial = res->contents_serial;
      cache->offset = offset;
      cache->count = count;
      cache->type = (uint32_t)type;
      cache->restart = primitive_restart;
      cache->min = *out_min;
      cache->max = *out_max;
      cache->valid = true;
   }

   return true;
}

static bool
yttrium_pipeline_add_vertex_upload(
   struct yttrium_pipeline_draw_upload *out,
   const void *data,
   uint64_t size)
{
   if (!out || !data || !size ||
       out->vertex_upload_count >=
          YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS ||
       size > SIZE_MAX)
      return false;

   const VkDeviceSize offset = align64(out->vertex_data_size, 4);
   const VkDeviceSize update_size = align64((VkDeviceSize)size, 4);
   if (offset + update_size < offset)
      return false;

   out->vertex_uploads[out->vertex_upload_count++] =
      (struct yttrium_venus_vertex_upload) {
         .data = data,
         .size = (size_t)size,
         .buffer_offset = offset,
      };
   out->vertex_data_size = offset + update_size;
   return true;
}

static bool
yttrium_pipeline_add_instance_divisor_upload(
   struct yttrium_pipeline_draw_upload *out,
   const uint8_t *base,
   uint64_t available,
   bool available_bounded,
   uint64_t buffer_offset,
   uint32_t stride,
   uint32_t start_instance,
   uint32_t instance_count,
   uint32_t divisor)
{
   if (!out || !base || !stride || !instance_count || divisor == 1)
      return false;

   const uint64_t last_source_instance =
      divisor == UINT32_MAX ? start_instance :
      (uint64_t)start_instance + (instance_count - 1) / divisor;
   const uint64_t last_offset =
      buffer_offset + last_source_instance * stride;
   const uint64_t size = (uint64_t)instance_count * stride;

   if (last_offset < buffer_offset || size > SIZE_MAX)
      return false;
   if (available_bounded &&
       (last_offset > available || stride > available - last_offset))
      return false;

   uint8_t *expanded = MALLOC(size);
   if (!expanded)
      return false;

   for (uint32_t i = 0; i < instance_count; i++) {
      const uint64_t source_instance =
         divisor == UINT32_MAX ? start_instance :
         (uint64_t)start_instance + i / divisor;
      const uint64_t source_offset =
         buffer_offset + source_instance * stride;
      memcpy(expanded + (uint64_t)i * stride, base + source_offset, stride);
   }

   const uint32_t upload_index = out->vertex_upload_count;
   if (!yttrium_pipeline_add_vertex_upload(out, expanded, size)) {
      FREE(expanded);
      return false;
   }

   out->owned_vertex_data[upload_index] = expanded;
   return true;
}

static uint64_t
yttrium_pipeline_instance_source_count(uint32_t instance_count,
                                       uint32_t divisor)
{
   if (!instance_count)
      return 0;
   if (divisor == UINT32_MAX)
      return 1;
   if (divisor <= 1)
      return instance_count;
   return ((uint64_t)instance_count + divisor - 1) / divisor;
}

static bool
yttrium_pipeline_add_vertex_resource(
   struct yttrium_pipeline_draw_upload *out,
   struct yttrium_resource *res,
   uint32_t resource_id,
   uint64_t offset,
   uint64_t size)
{
   if (!out || !res || !res->venus.initialized ||
       !res->venus.buffer_backed || !res->venus.buffer ||
       !(res->venus.buffer_usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) ||
       res->direct_bind_unsafe || !size || size > SIZE_MAX ||
       out->vertex_upload_count >=
          YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS)
      return false;

   const uint64_t available =
      res->venus.allocation_size ? res->venus.allocation_size : res->size;
   if (offset > available || size > available - offset)
      return false;

   out->vertex_uploads[out->vertex_upload_count++] =
      (struct yttrium_venus_vertex_upload) {
         .resource = &res->venus,
         .resource_id = resource_id,
         .size = (size_t)size,
         .buffer_offset = offset,
         .host_write_pending = res->data_dirty,
      };
   res->venus.draw_source_contents_serial = res->contents_serial;
   return true;
}

static void
yttrium_pipeline_mark_vertex_uploads_clean(
   struct yttrium_pipeline_draw_upload *upload)
{
   if (!upload)
      return;

   for (uint32_t i = 0; i < upload->vertex_upload_count; i++) {
      struct yttrium_venus_vertex_upload *vertex = &upload->vertex_uploads[i];
      if (!vertex->resource || !vertex->host_write_pending ||
          !vertex->resource->owner)
         continue;

      struct yttrium_resource *res =
         yttrium_resource(vertex->resource->owner);
      res->data_dirty = false;
      vertex->host_write_pending = false;
   }
}

static void
yttrium_pipeline_mark_index_upload_clean(
   struct yttrium_pipeline_draw_upload *upload)
{
   if (!upload || !upload->index_resource || !upload->index_host_write_pending ||
       !upload->index_resource->owner)
      return;

   struct yttrium_resource *res =
      yttrium_resource(upload->index_resource->owner);
   res->data_dirty = false;
   upload->index_host_write_pending = false;
}

static bool
yttrium_pipeline_add_zero_vertex_upload(
   struct yttrium_pipeline_draw_upload *out,
   uint64_t size)
{
   if (!out || !size || size > SIZE_MAX ||
       out->vertex_upload_count >=
          YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS)
      return false;

   void *data = CALLOC(1, (size_t)size);
   if (!data)
      return false;

   const uint32_t upload_index = out->vertex_upload_count;
   if (!yttrium_pipeline_add_vertex_upload(out, data, size)) {
      FREE(data);
      return false;
   }

   out->owned_vertex_data[upload_index] = data;
   return true;
}

static uint32_t
yttrium_pipeline_vertex_binding_span(
   const struct yttrium_vertex_elements_state *state,
   uint32_t binding_index)
{
   uint32_t span = 0;

   if (!state)
      return 0;

   for (unsigned i = 0; i < state->num_attribs; i++) {
      if (state->attribs[i].binding != binding_index)
         continue;

      const struct pipe_vertex_element *elem =
         &state->elements[state->attribs[i].location];
      const unsigned size = util_format_get_blocksize(elem->src_format);
      if (!size)
         return 0;
      span = MAX2(span, elem->src_offset + size);
   }

   return span;
}

static bool
yttrium_pipeline_draw_upload_fail(
   const char *reason,
   const struct yttrium_context *yctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_start_count_bias *draw,
   const struct yttrium_pipeline_draw_upload *out,
   uint32_t binding_index,
   uint32_t source_vb,
   uint64_t extra0,
   uint64_t extra1)
{
   const struct pipe_vertex_buffer *vb =
      yctx && source_vb < yctx->num_vertex_buffers ?
      &yctx->vertex_buffers[source_vb] : NULL;
   const VkVertexInputBindingDescription *binding =
      yctx && yctx->vertex_elements &&
      binding_index < yctx->vertex_elements->num_bindings ?
      &yctx->vertex_elements->bindings[binding_index] : NULL;
   const struct yttrium_resource *vb_res =
      vb && !vb->is_user_buffer && vb->buffer.resource ?
      yttrium_resource(vb->buffer.resource) : NULL;
   const struct yttrium_resource *index_res =
      info && info->index_size && !info->has_user_indices &&
      info->index.resource ? yttrium_resource(info->index.resource) : NULL;

   yttrium_trace_debug_stringf(
      "yttrium: shader_draw_probe draw upload fail reason=%s mode=%u index_size=%u restart=%u instances=%u start_instance=%u draw_count=%u start=%u bias=%d bounds=%u min=%u max=%u num_bindings=%u binding_index=%u source_vb=%u num_vb=%u stride=%u input_rate=%u vb_user=%u vb_res=%p vb_data=%p vb_size=0x%llx vb_offset=%u index_user=%u index_res=%p index_data=%p index_size_bytes=0x%llx first_vertex=%u vertex_count=%u upload_instances=%u vertex_uploads=%u vertex_bytes=0x%llx extra0=0x%llx extra1=0x%llx",
      reason ? reason : "unknown",
      info ? info->mode : 0,
      info ? info->index_size : 0,
      info ? info->primitive_restart : 0,
      info ? info->instance_count : 0,
      info ? info->start_instance : 0,
      draw ? draw->count : 0,
      draw ? draw->start : 0,
      draw ? draw->index_bias : 0,
      info ? info->index_bounds_valid : 0,
      info ? info->min_index : 0,
      info ? info->max_index : 0,
      yctx && yctx->vertex_elements ? yctx->vertex_elements->num_bindings : 0,
      binding_index,
      source_vb,
      yctx ? yctx->num_vertex_buffers : 0,
      binding ? binding->stride : 0,
      binding ? binding->inputRate : 0,
      vb ? vb->is_user_buffer : 0,
      vb_res,
      vb_res ? vb_res->data : NULL,
      (unsigned long long)(vb_res ? vb_res->size : 0),
      vb ? vb->buffer_offset : 0,
      info ? info->has_user_indices : 0,
      index_res,
      index_res ? index_res->data : NULL,
      (unsigned long long)(index_res ? index_res->size : 0),
      out ? out->first_vertex : 0,
      out ? out->vertex_count : 0,
      out ? out->instance_count : 0,
      out ? out->vertex_upload_count : 0,
      (unsigned long long)(out ? out->vertex_data_size : 0),
      (unsigned long long)extra0,
      (unsigned long long)extra1);
   return false;
}

static bool
yttrium_pipeline_get_draw_upload(struct yttrium_context *yctx,
                                 const struct pipe_draw_info *info,
                                 const struct pipe_draw_start_count_bias *draw,
                                 struct yttrium_pipeline_draw_upload *out)
{
   const uint32_t num_bindings =
      yctx && yctx->vertex_elements ? yctx->vertex_elements->num_bindings : 0;

   if (!yctx || num_bindings > YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS ||
       !info || !draw || !draw->count || !out)
      return yttrium_pipeline_draw_upload_fail(
         "bad_args", yctx, info, draw, out, UINT32_MAX, UINT32_MAX,
         num_bindings, draw ? draw->count : 0);

   memset(out, 0, sizeof(*out));

   const uint32_t instance_count =
      info->instance_count ? info->instance_count : 1;

   VkIndexType index_type;
   const uint8_t *index_base = NULL;
   uint64_t index_available = 0;
   bool index_available_bounded = false;

   uint64_t vertex_count = draw->count;
   uint64_t first_vertex = draw->start;
   if (info->index_size) {
      if (!yttrium_pipeline_index_type(info->index_size, &index_type))
         return yttrium_pipeline_draw_upload_fail(
            "bad_index_type", yctx, info, draw, out, UINT32_MAX,
            UINT32_MAX, info->index_size, 0);

      if (info->has_user_indices) {
         index_base = (const uint8_t *)info->index.user;
      } else if (info->index.resource) {
         struct yttrium_resource *index_res =
            yttrium_resource(info->index.resource);
         index_base = (const uint8_t *)index_res->data;
         index_available = index_res->size;
         index_available_bounded = true;
      }

      if (!index_base)
         return yttrium_pipeline_draw_upload_fail(
            "missing_index_base", yctx, info, draw, out, UINT32_MAX,
            UINT32_MAX, info->has_user_indices,
            (uint64_t)(uintptr_t)info->index.resource);

      const uint64_t index_offset =
         (uint64_t)draw->start * info->index_size;
      const uint64_t index_size =
         (uint64_t)draw->count * info->index_size;
      if (index_available_bounded &&
          (index_offset > index_available ||
           index_size > index_available - index_offset))
         return yttrium_pipeline_draw_upload_fail(
            "index_range_oob", yctx, info, draw, out, UINT32_MAX,
            UINT32_MAX, index_offset, index_available);
      if (index_size > SIZE_MAX)
         return yttrium_pipeline_draw_upload_fail(
            "index_size_too_large", yctx, info, draw, out, UINT32_MAX,
            UINT32_MAX, index_size, 0);

      out->index_data = index_base + index_offset;
      out->index_data_size = (size_t)index_size;
      out->index_count = draw->count;
      out->index_type = index_type;
      out->vertex_offset = draw->index_bias;
      if (!info->has_user_indices && info->index.resource) {
         struct yttrium_resource *index_res =
            yttrium_resource(info->index.resource);
         if (!yttrium_resource_publish_private_draw_buffer(
                &yctx->base, index_res))
            return false;
         if (index_res->venus.initialized &&
             index_res->venus.buffer_backed &&
             index_res->venus.buffer &&
             !index_res->direct_bind_unsafe &&
             (index_res->venus.buffer_usage &
              VK_BUFFER_USAGE_INDEX_BUFFER_BIT)) {
            out->index_resource = &index_res->venus;
            out->index_resource_id = index_res->venus_res_id;
            out->index_buffer_offset = (VkDeviceSize)index_offset;
            out->index_host_write_pending = index_res->data_dirty;
            index_res->venus.draw_source_contents_serial =
               index_res->contents_serial;
         }
      }

      uint32_t min_index = 0;
      uint32_t max_index = 0;
      if (info->index_bounds_valid) {
         min_index = info->min_index;
         max_index = info->max_index;
         if (max_index < min_index)
            return yttrium_pipeline_draw_upload_fail(
               "indexed_bounds_range", yctx, info, draw, out, UINT32_MAX,
               UINT32_MAX, (uint64_t)min_index, (uint64_t)max_index);
      } else if (!yttrium_pipeline_index_bounds(
                    yctx,
                    yttrium_pipeline_index_bounds_cache_resource(info),
                    out->index_data,
                    out->index_data_size,
                    index_offset,
                    out->index_count,
                    out->index_type,
                    info->primitive_restart,
                    &min_index, &max_index)) {
         return yttrium_pipeline_draw_upload_fail(
            "scan_index_bounds_failed", yctx, info, draw, out, UINT32_MAX,
            UINT32_MAX, out->index_data_size, out->index_count);
      }

      const int64_t first =
         (int64_t)min_index + (int64_t)draw->index_bias;
      const int64_t last =
         (int64_t)max_index + (int64_t)draw->index_bias;
      if (first < 0 || last < first || last > UINT32_MAX)
         return yttrium_pipeline_draw_upload_fail(
            "index_bias_range", yctx, info, draw, out, UINT32_MAX,
            UINT32_MAX, (uint64_t)min_index, (uint64_t)max_index);

      first_vertex = (uint64_t)first;
      vertex_count = (uint64_t)last - (uint64_t)first + 1;
      const int64_t vertex_offset =
         (int64_t)draw->index_bias - (int64_t)first_vertex;
      if (vertex_offset < INT32_MIN || vertex_offset > INT32_MAX)
         return yttrium_pipeline_draw_upload_fail(
            "vertex_offset_range", yctx, info, draw, out, UINT32_MAX,
            UINT32_MAX, (uint64_t)first_vertex, (uint64_t)vertex_offset);
      out->vertex_offset = (int32_t)vertex_offset;
   } else if (info->index_bounds_valid) {
      const int64_t min_index =
         (int64_t)info->min_index + (int64_t)draw->index_bias;
      const int64_t max_index =
         (int64_t)info->max_index + (int64_t)draw->index_bias;
      if (min_index < 0 || max_index < min_index ||
          max_index > UINT32_MAX)
         return yttrium_pipeline_draw_upload_fail(
            "nonindexed_bounds_range", yctx, info, draw, out,
            UINT32_MAX, UINT32_MAX, (uint64_t)info->min_index,
            (uint64_t)info->max_index);
      first_vertex = (uint64_t)min_index;
      vertex_count = (uint64_t)max_index - (uint64_t)min_index + 1;
   }

   if (!vertex_count || vertex_count > UINT32_MAX)
      return yttrium_pipeline_draw_upload_fail(
         "vertex_count_range", yctx, info, draw, out, UINT32_MAX,
         UINT32_MAX, vertex_count, 0);

   out->first_vertex = (uint32_t)first_vertex;
   out->vertex_count = (uint32_t)vertex_count;
   out->instance_count = instance_count;
   struct yttrium_screen *screen = yttrium_screen(yctx->base.screen);

   for (uint32_t binding_index = 0; binding_index < num_bindings;
        binding_index++) {
      const unsigned source_vb =
         yctx->vertex_elements->binding_map[binding_index];

      const VkVertexInputBindingDescription *binding =
         &yctx->vertex_elements->bindings[binding_index];
      const uint32_t binding_span =
         yttrium_pipeline_vertex_binding_span(yctx->vertex_elements,
                                              binding_index);
      const uint32_t fetch_stride =
         binding->stride ? binding->stride : binding_span;
      if (!fetch_stride)
         return yttrium_pipeline_draw_upload_fail(
            "zero_stride", yctx, info, draw, out, binding_index,
            source_vb, 0, 0);

      const bool instance_rate =
         binding->inputRate == VK_VERTEX_INPUT_RATE_INSTANCE;
      const uint32_t instance_divisor =
         yctx->vertex_elements->binding_divisor[binding_index];
      const bool native_instance_divisor =
         instance_rate &&
         yttrium_venus_vertex_attribute_divisor_supported(
            screen->venus, instance_divisor);
      const bool emulate_instance_divisor =
         instance_rate && instance_divisor != 1 &&
         !native_instance_divisor;
      const struct pipe_vertex_buffer *vb =
         source_vb < yctx->num_vertex_buffers ?
         &yctx->vertex_buffers[source_vb] : NULL;
      const uint8_t *base = NULL;
      uint64_t available = 0;
      bool available_bounded = false;

      if (vb && vb->is_user_buffer) {
         base = (const uint8_t *)vb->buffer.user;
      } else if (vb && vb->buffer.resource) {
         struct yttrium_resource *res =
            yttrium_resource(vb->buffer.resource);
         if (!yttrium_resource_publish_private_draw_buffer(
                &yctx->base, res))
            return false;
         if (res->venus.initialized && res->venus.buffer_backed &&
             res->venus.buffer && !res->direct_bind_unsafe &&
             (res->venus.buffer_usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
            available = emulate_instance_divisor ? res->size :
               (res->venus.allocation_size ?
                res->venus.allocation_size : res->size);
            available_bounded = true;
            base = emulate_instance_divisor ?
               (const uint8_t *)res->data : NULL;
         } else {
            base = (const uint8_t *)res->data;
            available = res->size;
            available_bounded = true;
         }
      }

      const uint64_t first =
         instance_rate ? info->start_instance :
                         out->first_vertex;
      const uint64_t count = instance_rate ?
         (native_instance_divisor ?
          yttrium_pipeline_instance_source_count(instance_count,
                                                instance_divisor) :
          instance_count) : vertex_count;
      const uint64_t offset =
         (uint64_t)(vb ? vb->buffer_offset : 0) +
         first * binding->stride;
      const uint64_t size = binding->stride ? count * binding->stride :
                                             fetch_stride;

      if (!emulate_instance_divisor && available_bounded &&
          (offset > available || size > available - offset))
         return yttrium_pipeline_draw_upload_fail(
            "vertex_range_oob", yctx, info, draw, out, binding_index,
            source_vb, offset, available);

      if (emulate_instance_divisor) {
         YTTRIUM_WARN("yttrium: software fallback emulating vertex instance divisor binding=%u source_vb=%u divisor=%u start_instance=%u instances=%u stride=%u size=0x%llx base=%p\n",
                      binding_index, source_vb, instance_divisor,
                      info->start_instance, instance_count,
                      binding->stride,
                      (unsigned long long)size, base);
         if (!base) {
            if (!yttrium_pipeline_add_zero_vertex_upload(out, size))
               return yttrium_pipeline_draw_upload_fail(
                  "add_zero_instance_upload_failed", yctx, info, draw,
                  out, binding_index, source_vb, size,
                  yctx->num_vertex_buffers);
            continue;
         }

         if (!yttrium_pipeline_add_instance_divisor_upload(
                out, base, available, available_bounded,
                (uint64_t)vb->buffer_offset, binding->stride,
                info->start_instance, instance_count,
                instance_divisor))
            return yttrium_pipeline_draw_upload_fail(
               "add_instance_divisor_upload_failed", yctx, info, draw, out,
               binding_index, source_vb, instance_divisor,
               instance_count);
         continue;
      }

      if (vb && !vb->is_user_buffer && vb->buffer.resource) {
         struct yttrium_resource *res =
            yttrium_resource(vb->buffer.resource);
         if (res->venus.initialized && res->venus.buffer_backed &&
             res->venus.buffer && !res->direct_bind_unsafe &&
             (res->venus.buffer_usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
            if (!yttrium_pipeline_add_vertex_resource(
                   out, res, res->venus_res_id, offset, size))
               return yttrium_pipeline_draw_upload_fail(
                  "add_vertex_resource_failed", yctx, info, draw, out,
                  binding_index, source_vb, offset, size);
            continue;
         }
      }

      if (!base) {
         if (!yttrium_pipeline_add_zero_vertex_upload(out, size))
            return yttrium_pipeline_draw_upload_fail(
               "add_zero_vertex_upload_failed", yctx, info, draw, out,
               binding_index, source_vb, size,
               yctx->num_vertex_buffers);
         continue;
      }

      if (!yttrium_pipeline_add_vertex_upload(out, base + offset, size))
         return yttrium_pipeline_draw_upload_fail(
            "add_vertex_upload_failed", yctx, info, draw, out,
            binding_index, source_vb, offset, size);
   }

   if (out->vertex_upload_count != num_bindings)
      return yttrium_pipeline_draw_upload_fail(
         "vertex_upload_count_mismatch", yctx, info, draw, out,
         UINT32_MAX, UINT32_MAX, out->vertex_upload_count,
         num_bindings);

   return true;
}

static bool
yttrium_pipeline_get_indirect_draw_upload(
   struct yttrium_context *yctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_start_count_bias *draw,
   struct yttrium_pipeline_draw_upload *out)
{
   const uint32_t num_bindings = yctx && yctx->vertex_elements ?
      yctx->vertex_elements->num_bindings : 0;
   if (!yctx || !info || !out ||
       num_bindings > YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS)
      return false;

   memset(out, 0, sizeof(*out));
   /* Counts and base vertex/instance belong exclusively to the GPU argument
    * buffer.  Bind the original complete resources, without scanning indices,
    * rebasing attributes, or estimating any count from their CPU mappings. */
   struct yttrium_screen *screen = yttrium_screen(yctx->base.screen);
   for (uint32_t i = 0; i < num_bindings; i++) {
      const unsigned source_vb = yctx->vertex_elements->binding_map[i];
      const struct pipe_vertex_buffer *vb = source_vb < yctx->num_vertex_buffers ?
         &yctx->vertex_buffers[source_vb] : NULL;
      const VkVertexInputBindingDescription *binding =
         &yctx->vertex_elements->bindings[i];
      if (!vb || vb->is_user_buffer || !vb->buffer.resource ||
          (binding->inputRate == VK_VERTEX_INPUT_RATE_INSTANCE &&
           !yttrium_venus_vertex_attribute_divisor_supported(
              screen->venus, yctx->vertex_elements->binding_divisor[i])))
         return yttrium_pipeline_draw_upload_fail(
            "indirect_non_native_vertex_binding", yctx, info, NULL, out,
            i, source_vb, 0, 0);

      struct yttrium_resource *res = yttrium_resource(vb->buffer.resource);
      if (!yttrium_resource_publish_private_draw_buffer(&yctx->base, res))
         return false;
      if (vb->buffer_offset >= res->size ||
          !yttrium_pipeline_add_vertex_resource(
             out, res, res->venus_res_id, vb->buffer_offset,
             res->size - vb->buffer_offset))
         return yttrium_pipeline_draw_upload_fail(
            "indirect_vertex_resource_unavailable", yctx, info, NULL, out,
            i, source_vb, vb->buffer_offset, res->size);
   }

   if (info->index_size) {
      if (info->has_user_indices || !info->index.resource ||
          !yttrium_pipeline_index_type(info->index_size, &out->index_type))
         return false;
      struct yttrium_resource *res = yttrium_resource(info->index.resource);
      const uint64_t offset = (uint64_t)(draw ? draw->start : 0) * info->index_size;
      if (!yttrium_resource_publish_private_draw_buffer(&yctx->base, res))
         return false;
      if (!res->venus.initialized || !res->venus.buffer_backed ||
          !res->venus.buffer || res->direct_bind_unsafe ||
          !(res->venus.buffer_usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) ||
          offset >= res->size || res->size - offset > SIZE_MAX)
         return yttrium_pipeline_draw_upload_fail(
            "indirect_index_resource_unavailable", yctx, info, NULL, out,
            UINT32_MAX, UINT32_MAX, offset, res->size);
      out->index_resource = &res->venus;
      out->index_resource_id = res->venus_res_id;
      out->index_buffer_offset = offset;
      out->index_data_size = (size_t)(res->size - offset);
      out->index_host_write_pending = res->data_dirty;
      res->venus.draw_source_contents_serial = res->contents_serial;
   }
   return true;
}

static bool
yttrium_pipeline_get_draw_auto_upload(
   struct yttrium_context *yctx,
   const struct pipe_draw_info *info,
   struct yttrium_pipeline_draw_upload *out)
{
   if (!yctx || !yctx->vertex_elements ||
       !yctx->vertex_elements->num_bindings ||
       yctx->vertex_elements->num_bindings >
          YTTRIUM_VENUS_MAX_PIPELINE_VERTEX_BINDINGS ||
       !info || info->index_size || !out)
      return yttrium_pipeline_draw_upload_fail(
         "draw_auto_bad_args", yctx, info, NULL, out,
         UINT32_MAX, UINT32_MAX,
         yctx && yctx->vertex_elements ?
            yctx->vertex_elements->num_bindings : 0,
         info ? info->index_size : 0);

   memset(out, 0, sizeof(*out));
   out->first_vertex = 0;
   out->vertex_count = 1;
   out->instance_count = info->instance_count ? info->instance_count : 1;

   for (uint32_t binding_index = 0;
        binding_index < yctx->vertex_elements->num_bindings;
        binding_index++) {
      const unsigned source_vb =
         yctx->vertex_elements->binding_map[binding_index];
      if (source_vb >= yctx->num_vertex_buffers)
         return yttrium_pipeline_draw_upload_fail(
            "draw_auto_source_vb_oob", yctx, info, NULL, out,
            binding_index, source_vb, yctx->num_vertex_buffers, 0);

      const VkVertexInputBindingDescription *binding =
         &yctx->vertex_elements->bindings[binding_index];
      if (!binding->stride)
         return yttrium_pipeline_draw_upload_fail(
            "draw_auto_zero_stride", yctx, info, NULL, out,
            binding_index, source_vb, 0, 0);

      const struct pipe_vertex_buffer *vb =
         &yctx->vertex_buffers[source_vb];
      if (vb->is_user_buffer || !vb->buffer.resource)
         return yttrium_pipeline_draw_upload_fail(
            "draw_auto_non_resource_vb", yctx, info, NULL, out,
            binding_index, source_vb, vb->is_user_buffer,
            (uint64_t)(uintptr_t)vb->buffer.resource);

      struct yttrium_resource *res =
         yttrium_resource(vb->buffer.resource);
      const bool instance_rate =
         binding->inputRate == VK_VERTEX_INPUT_RATE_INSTANCE;
      const uint64_t first =
         instance_rate ? info->start_instance : 0;
      const uint64_t offset =
         (uint64_t)vb->buffer_offset + first * binding->stride;
      const uint64_t available =
         res->venus.allocation_size ? res->venus.allocation_size : res->size;
      const uint64_t size =
         instance_rate ? (uint64_t)out->instance_count * binding->stride :
                         available > offset ? available - offset : 0;

      if (!yttrium_pipeline_add_vertex_resource(
             out, res, res->venus_res_id, offset, size)) {
         if (!res->data || offset > res->size || size > res->size - offset ||
             !yttrium_pipeline_add_vertex_upload(
                out, (const uint8_t *)res->data + offset, size))
            return yttrium_pipeline_draw_upload_fail(
               "draw_auto_add_vertex_resource_failed", yctx, info, NULL, out,
               binding_index, source_vb, offset, size);
      }
   }

   return out->vertex_upload_count == yctx->vertex_elements->num_bindings;
}

static bool
yttrium_pipeline_add_ubo_layout(
   const struct yttrium_shader_state *shader,
   VkShaderStageFlags stage_flags,
   struct yttrium_venus_ubo_binding_layout *layouts,
   uint32_t *layout_count)
{
   if (!layouts || !layout_count)
      return false;
   if (!shader)
      return true;

   /* Slots lowered to push constants intentionally contribute no UBO bindings.
    * Keep warning when other declared slots have no bindings: a lost UBO mask
    * can leave the SPIR-V using descriptors absent from the pipeline layout.
    */
   if (!shader->ubo_used_mask &&
       (shader->info.const_buffers_declared & ~shader->push_ubo_mask))
      YTTRIUM_WARN("yttrium: shader contributes no UBO bindings owner=yttrium-pipeline stage=%s id=%u declared=0x%x push_ubos=0x%x nir_ubos=%u token_hash=0x%llx\n",
                   yttrium_shader_stage_name(shader->stage), shader->id,
                   shader->info.const_buffers_declared,
                   shader->push_ubo_mask,
                   shader->nir ? shader->nir->info.num_ubos : 0,
                   (unsigned long long)shader->token_hash);

   for (unsigned raw_index = 0; raw_index < PIPE_MAX_CONSTANT_BUFFERS;
        raw_index++) {
      if (!(shader->ubo_used_mask & (1u << raw_index)))
         continue;

      const uint32_t binding =
         yttrium_shader_ubo_binding(shader->stage, raw_index);
      if (binding == UINT32_MAX)
         return false;
      if (*layout_count >= YTTRIUM_VENUS_MAX_PIPELINE_UBO_BINDINGS)
         return false;

      layouts[*layout_count] =
         (struct yttrium_venus_ubo_binding_layout) {
            .binding = binding,
            .descriptor_count = 1,
            .stage_flags = stage_flags,
         };
      (*layout_count)++;
   }

   return true;
}

static bool
yttrium_pipeline_build_ubo_layouts(
   const struct yttrium_shader_state *vs,
   const struct yttrium_shader_state *tcs,
   const struct yttrium_shader_state *tes,
   const struct yttrium_shader_state *gs,
   const struct yttrium_shader_state *fs,
   struct yttrium_venus_ubo_binding_layout *layouts,
   uint32_t *layout_count)
{
   if (!layouts || !layout_count)
      return false;

   *layout_count = 0;
   return yttrium_pipeline_add_ubo_layout(vs, VK_SHADER_STAGE_VERTEX_BIT,
                                          layouts, layout_count) &&
          yttrium_pipeline_add_ubo_layout(
             tcs, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
             layouts, layout_count) &&
          yttrium_pipeline_add_ubo_layout(
             tes, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
             layouts, layout_count) &&
          yttrium_pipeline_add_ubo_layout(gs, VK_SHADER_STAGE_GEOMETRY_BIT,
                                          layouts, layout_count) &&
          yttrium_pipeline_add_ubo_layout(fs, VK_SHADER_STAGE_FRAGMENT_BIT,
                                          layouts, layout_count);
}

static const uint8_t *
yttrium_pipeline_constant_buffer_bytes(const struct yttrium_constant_buffer *cb,
                                       uint64_t *available)
{
   if (available)
      *available = 0;

   if (!cb || !cb->buffer_size)
      return NULL;

   if (cb->user_buffer) {
      if (available)
         *available = cb->buffer_size;
      return cb->user_buffer;
   }

   if (cb->buffer) {
      const struct yttrium_resource *res = yttrium_resource(cb->buffer);
      if (res && res->data && cb->buffer_offset < res->size) {
         const uint64_t size = MIN2((uint64_t)cb->buffer_size,
                                    res->size - cb->buffer_offset);
         if (available)
            *available = size;
         return (const uint8_t *)res->data + cb->buffer_offset;
      }
   }

   return NULL;
}

static bool
yttrium_pipeline_add_shader_push_constants(
   const struct yttrium_context *yctx,
   const struct yttrium_shader_state *shader,
   struct yttrium_venus_draw_state *draw_state)
{
   if (!shader || !shader->push_constant_word_count)
      return true;
   if (!yctx || !draw_state ||
       shader->push_constant_offset > YTTRIUM_SHADER_PUSH_CONSTANT_BYTES ||
       shader->push_constant_word_count >
          YTTRIUM_SHADER_MAX_PUSH_CONSTANT_WORDS ||
       (uint32_t)shader->push_constant_word_count * sizeof(uint32_t) >
          YTTRIUM_SHADER_PUSH_CONSTANT_BYTES - shader->push_constant_offset)
      return false;

   const uint8_t *slot_data[PIPE_MAX_CONSTANT_BUFFERS] = { 0 };
   uint64_t slot_available[PIPE_MAX_CONSTANT_BUFFERS] = { 0 };
   for (unsigned raw_index = 0; raw_index < PIPE_MAX_CONSTANT_BUFFERS;
        raw_index++) {
      if (!(shader->push_ubo_mask & BITFIELD_BIT(raw_index)))
         continue;

      const struct yttrium_constant_buffer *cb =
         &yctx->constant_buffers[shader->stage][raw_index];
      slot_data[raw_index] =
         yttrium_pipeline_constant_buffer_bytes(
            cb, &slot_available[raw_index]);
      if (!slot_data[raw_index] || !slot_available[raw_index] ||
          slot_available[raw_index] > YTTRIUM_SHADER_MAX_UBO_BYTES)
         return false;
   }

   for (unsigned i = 0; i < shader->push_constant_word_count; i++) {
      const unsigned raw_index = shader->push_constant_source_slots[i];
      const uint64_t source_offset =
         (uint64_t)shader->push_constant_source_words[i] * sizeof(uint32_t);
      if (raw_index >= PIPE_MAX_CONSTANT_BUFFERS ||
          !slot_data[raw_index] ||
          source_offset > slot_available[raw_index] ||
          sizeof(uint32_t) > slot_available[raw_index] - source_offset)
         return false;

      memcpy(draw_state->push_constant_data +
                shader->push_constant_offset + i * sizeof(uint32_t),
             slot_data[raw_index] + source_offset, sizeof(uint32_t));
   }

   const uint16_t size =
      (uint16_t)(shader->push_constant_word_count * sizeof(uint32_t));
   if (shader->stage == MESA_SHADER_VERTEX)
      draw_state->push_constant_vs_size = size;
   else if (shader->stage == MESA_SHADER_FRAGMENT)
      draw_state->push_constant_fs_size = size;
   else
      return false;
   return true;
}

static bool
yttrium_pipeline_collect_push_constants(
   const struct yttrium_context *yctx,
   struct yttrium_venus_draw_state *draw_state)
{
   if (!yctx || !draw_state)
      return false;

   return yttrium_pipeline_add_shader_push_constants(
             yctx, yctx->shaders[MESA_SHADER_VERTEX], draw_state) &&
          yttrium_pipeline_add_shader_push_constants(
             yctx, yctx->shaders[MESA_SHADER_FRAGMENT], draw_state);
}

static bool
yttrium_pipeline_add_ubo_uploads(
   const struct yttrium_context *yctx,
   const struct yttrium_shader_state *shader,
   struct yttrium_venus_ubo_upload *uploads,
   uint32_t *upload_count)
{
   if (!yctx || !uploads || !upload_count)
      return false;
   if (!shader)
      return true;

   for (unsigned raw_index = 0; raw_index < PIPE_MAX_CONSTANT_BUFFERS;
        raw_index++) {
      if (!(shader->ubo_used_mask & (1u << raw_index)))
         continue;

      if (*upload_count >= YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS)
         return false;

      const uint32_t binding =
         yttrium_shader_ubo_binding(shader->stage, raw_index);
      if (binding == UINT32_MAX)
         return false;

      const struct yttrium_constant_buffer *cb =
         &yctx->constant_buffers[shader->stage][raw_index];
      uint64_t available = 0;
      const uint8_t *data =
         yttrium_pipeline_constant_buffer_bytes(cb, &available);
      if (!data || !available || available > YTTRIUM_SHADER_MAX_UBO_BYTES ||
         (available & 3)) {
         YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped ubo missing stage=%s raw=%u binding=%u elem=%u size=0x%llx data=%p\n",
                      yttrium_shader_stage_name(shader->stage),
                      raw_index,
                      binding,
                      0,
                      (unsigned long long)available, data);
         return false;
      }

      struct yttrium_resource *cb_res =
         cb->buffer ? yttrium_resource(cb->buffer) : NULL;
      struct yttrium_venus_resource *direct_resource =
         cb_res && cb_res->ordered_worker_upload_direct_backing ?
            &cb_res->venus : NULL;
      const bool version_cache_safe =
         cb_res && !direct_resource &&
         !(cb_res->base.bind & (PIPE_BIND_STREAM_OUTPUT |
                                PIPE_BIND_SHADER_BUFFER |
                                PIPE_BIND_SHADER_IMAGE));
      uploads[*upload_count] = (struct yttrium_venus_ubo_upload) {
         .binding = binding,
         .array_element = 0,
         .data = data,
         .size = (size_t)available,
         .direct_resource = direct_resource,
         .direct_offset = cb->buffer_offset,
         .source_contents_serial =
            cb_res ? cb_res->contents_serial : 0,
         .source_offset = cb->buffer_offset,
         .source_version_cache =
            version_cache_safe ? &cb_res->ubo_version_cache : NULL,
      };
      yttrium_trace_debug_stringf(
         "yttrium: shader_draw_probe ubo upload stage=%s raw=%u binding=%u size=0x%llx user=%u cb_res_id=%u cb_alloc=0x%lx cb_offset=%u cb_size=%u first=%08x,%08x,%08x,%08x",
         yttrium_shader_stage_name(shader->stage),
         raw_index,
         binding,
         (unsigned long long)available,
         cb->user_buffer != NULL,
         cb_res ? cb_res->venus_res_id : 0,
         (unsigned long)(cb_res ? cb_res->hAllocation : 0),
         cb->buffer_offset,
         cb->buffer_size,
         yttrium_pipeline_peek_u32(data, (size_t)available, 0),
         yttrium_pipeline_peek_u32(data, (size_t)available, 1),
         yttrium_pipeline_peek_u32(data, (size_t)available, 2),
         yttrium_pipeline_peek_u32(data, (size_t)available, 3));
      (*upload_count)++;
   }

   return true;
}

static bool
yttrium_pipeline_collect_ubo_uploads(
   const struct yttrium_context *yctx,
   struct yttrium_venus_ubo_upload *uploads,
   uint32_t *upload_count)
{
   const struct yttrium_shader_state *vs =
      yctx ? yctx->shaders[MESA_SHADER_VERTEX] : NULL;
   const struct yttrium_shader_state *tcs =
      yctx ? yctx->shaders[MESA_SHADER_TESS_CTRL] : NULL;
   const struct yttrium_shader_state *tes =
      yctx ? yctx->shaders[MESA_SHADER_TESS_EVAL] : NULL;
   const struct yttrium_shader_state *gs =
      yctx ? yctx->shaders[MESA_SHADER_GEOMETRY] : NULL;
   const struct yttrium_shader_state *fs =
      yctx ? yctx->shaders[MESA_SHADER_FRAGMENT] : NULL;

   if (!yctx || !uploads || !upload_count)
      return false;

   *upload_count = 0;
   return yttrium_pipeline_add_ubo_uploads(yctx, vs, uploads,
                                           upload_count) &&
          yttrium_pipeline_add_ubo_uploads(yctx, tcs, uploads,
                                           upload_count) &&
          yttrium_pipeline_add_ubo_uploads(yctx, tes, uploads,
                                           upload_count) &&
          yttrium_pipeline_add_ubo_uploads(yctx, gs, uploads,
                                           upload_count) &&
          yttrium_pipeline_add_ubo_uploads(yctx, fs, uploads,
                                            upload_count);
}

static bool
yttrium_pipeline_retire_sampled_texture_cpu_shadow(
   struct yttrium_resource *res)
{
   /*
    * A plain DEFAULT sampler texture starts with heap storage because its
    * Venus image is created lazily.  Once every subresource has been uploaded,
    * that image is authoritative: read maps already use image readback and
    * write maps/texture_subdata already stage directly into the image.  Keeping
    * the original heap allocation after that point only duplicates the texture
    * and, in a 32-bit application, can exhaust the process address space.
    *
    * Limit retirement to the unambiguous driver-owned case.  Shared, display,
    * mappable-allocation, mixed-bind, and non-DEFAULT resources retain their
    * existing ownership model.
    */
   if (!res || res->base.target == PIPE_BUFFER ||
       res->base.usage != PIPE_USAGE_DEFAULT ||
       res->base.bind != PIPE_BIND_SAMPLER_VIEW ||
       res->base.nr_samples > 1 || res->base.nr_storage_samples > 1 ||
       !res->data || res->data_capacity < res->size || !res->owns_data ||
       res->display_target || res->primary_target || res->classic_display ||
       res->hAllocation || res->hResource || res->hAllocationResource ||
       res->hResourceIsD3D9Runtime || res->map || res->owns_allocation ||
       res->replacement_storage || res->replacement_owner ||
       !res->venus.initialized || res->venus.buffer_backed ||
       res->venus.samples != VK_SAMPLE_COUNT_1_BIT ||
       !res->venus.image || !res->venus.contents_initialized ||
       res->data_dirty)
      return false;

   void *data = res->data;
   const uint64_t data_capacity = res->data_capacity;
   YTTRIUM_LOG("yttrium: sampled texture retired authoritative CPU shadow resource=%p res_id=%u data=%p capacity=0x%llx image_id=%llu\n",
               (void *)res, res->venus_res_id, data,
               (unsigned long long)data_capacity,
               (unsigned long long)res->venus.image_obj.id);
   res->data = NULL;
   res->data_capacity = 0;
   res->owns_data = false;
   FREE(data);
   return true;
}

static bool
yttrium_pipeline_ensure_resource_venus_texture(struct pipe_context *ctx,
                                               struct yttrium_resource *res)
{
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);

   if (!res ||
       (res->base.target != PIPE_TEXTURE_1D &&
        res->base.target != PIPE_TEXTURE_1D_ARRAY &&
        res->base.target != PIPE_TEXTURE_2D &&
        res->base.target != PIPE_TEXTURE_2D_ARRAY &&
        res->base.target != PIPE_TEXTURE_3D &&
        res->base.target != PIPE_TEXTURE_CUBE &&
        res->base.target != PIPE_TEXTURE_CUBE_ARRAY) ||
       (res->base.target == PIPE_TEXTURE_3D ?
          (res->base.depth0 < 1 || res->base.array_size != 1) :
          (res->base.depth0 != 1 || res->base.array_size < 1)) ||
       res->base.last_level >= 16)
      return false;

   if (!res->venus.initialized) {
      const bool needs_color_attachment =
         (res->base.bind &
          (PIPE_BIND_RENDER_TARGET | PIPE_BIND_DISPLAY_TARGET)) != 0;
      if (res->base.target == PIPE_TEXTURE_2D &&
          needs_color_attachment &&
          !util_format_is_compressed(res->base.format) &&
          res->base.last_level == 0 && res->base.array_size == 1) {
         uint64_t memory_id = 0;
         uint64_t allocation_size = 0;
         if (!yttrium_venus_create_display_image(screen->venus, &res->venus,
                                                 res->base.width0,
                                                 res->base.height0,
                                                 res->base.format,
                                                 res->size,
                                                 &memory_id,
                                                 &allocation_size)) {
            YTTRIUM_LOG("yttrium: shader_draw_probe sampled texture Venus image create failed texture=%p %ux%u format=%u\n",
                        (void *)&res->base,
                        res->base.width0, res->base.height0,
                        res->base.format);
            return false;
         }
         res->venus_mem_id = memory_id;
         res->venus_res_id = (uint32_t)memory_id;
      } else {
         uint64_t allocation_size = 0;
         const bool sampled_only =
            !needs_color_attachment ||
            res->base.target == PIPE_TEXTURE_1D ||
            res->base.target == PIPE_TEXTURE_1D_ARRAY ||
            res->base.target == PIPE_TEXTURE_3D ||
            res->base.target == PIPE_TEXTURE_CUBE ||
            res->base.target == PIPE_TEXTURE_CUBE_ARRAY ||
            util_format_is_compressed(res->base.format);
         const bool created =
            sampled_only ?
            yttrium_venus_create_sampled_texture_image(
               screen->venus, &res->venus, res->base.target,
               res->base.width0, res->base.height0,
               res->base.depth0,
               res->base.last_level + 1, res->base.array_size,
               res->base.format, &allocation_size) :
            yttrium_venus_create_color_attachment_image(
               screen->venus, &res->venus, res->base.width0,
               res->base.height0, res->base.last_level + 1,
               res->base.array_size, res->base.format,
               &allocation_size);
         if (!created) {
            YTTRIUM_LOG("yttrium: shader_draw_probe sampled texture Venus array image create failed texture=%p target=%u %ux%u levels=%u layers=%u format=%u\n",
                        (void *)&res->base, res->base.target,
                        res->base.width0, res->base.height0,
                        res->base.last_level + 1, res->base.array_size,
                        res->base.format);
            return false;
         }
         (void)allocation_size;
         res->venus_mem_id = res->venus.memory_obj.id;
         res->venus_res_id = (uint32_t)res->venus.memory_obj.id;
      }

      YTTRIUM_LOG("yttrium: shader_draw_probe sampled texture Venus image texture=%p image_id=%llu memory_id=%llu target=%u %ux%u levels=%u layers=%u format=%u\n",
                  (void *)&res->base,
                  (unsigned long long)res->venus.image_obj.id,
                  (unsigned long long)res->venus_mem_id,
                  res->base.target,
                  res->base.width0, res->base.height0,
                  res->base.last_level + 1, res->base.array_size,
                  res->base.format);
   }

   if (res->data &&
       (!res->venus.contents_initialized || res->data_dirty)) {
      if (yttrium_pipeline_verbose_trace_enabled()) {
         const struct yttrium_pipeline_data_summary data_summary =
            yttrium_pipeline_summarize_data(res->data, (size_t)res->size);
         yttrium_trace_debug_stringf(
            "yttrium: shader_draw_probe sampled texture cpu upload begin res_id=%u image_id=%llu display=%u primary=%u classic=%u initialized=%u contents=%u layout=%u data=%p size=0x%llx stride=%u first=%08x,%08x,%08x,%08x inspected=0x%llx nonzero=0x%llx first_nz=0x%llx:%08x last_nz=0x%llx:%08x hash=0x%x",
            res->venus_res_id,
            (unsigned long long)res->venus.image_obj.id,
            res->display_target,
            res->primary_target,
            res->classic_display,
            res->venus.initialized,
            res->venus.contents_initialized,
            res->venus.layout,
            res->data,
            (unsigned long long)res->size,
            res->stride,
            yttrium_pipeline_peek_u32(res->data, (size_t)res->size, 0),
            yttrium_pipeline_peek_u32(res->data, (size_t)res->size, 1),
            yttrium_pipeline_peek_u32(res->data, (size_t)res->size, 2),
            yttrium_pipeline_peek_u32(res->data, (size_t)res->size, 3),
            (unsigned long long)data_summary.inspected,
            (unsigned long long)data_summary.nonzero_bytes,
            (unsigned long long)data_summary.first_nonzero,
            data_summary.first_nonzero_u32,
            (unsigned long long)data_summary.last_nonzero,
            data_summary.last_nonzero_u32,
            data_summary.hash);
      }

      bool upload_ok = true;
      for (unsigned level = 0; level <= res->base.last_level; level++) {
         const unsigned width = u_minify(res->base.width0, level);
         const unsigned height = u_minify(res->base.height0, level);
         if (res->base.target == PIPE_TEXTURE_3D) {
            const unsigned depth = u_minify(res->base.depth0, level);
            upload_ok =
               yttrium_copy_cpu_to_venus_image(ctx, res, res, level,
                                               level, 0, 0, 0, 0, 0, 0,
                                               width, height, depth) &&
               upload_ok;
         } else {
            for (unsigned layer = 0; layer < res->base.array_size; layer++) {
               upload_ok =
                  yttrium_copy_cpu_to_venus_image(ctx, res, res, level,
                                                  level, 0, 0, layer, 0, 0,
                                                  layer, width, height, 1) &&
                  upload_ok;
            }
         }
      }

      if (!upload_ok) {
         YTTRIUM_LOG("yttrium: shader_draw_probe sampled texture CPU upload failed texture=%p image_id=%llu %ux%u format=%u\n",
                     (void *)&res->base,
                     (unsigned long long)res->venus.image_obj.id,
                     res->base.width0, res->base.height0,
                     res->base.format);
         yttrium_trace_debug_stringf(
            "yttrium: shader_draw_probe sampled texture cpu upload result=0 res_id=%u image_id=%llu",
            res->venus_res_id,
            (unsigned long long)res->venus.image_obj.id);
         return false;
      }

      yttrium_trace_debug_stringf(
         "yttrium: shader_draw_probe sampled texture cpu upload result=1 res_id=%u image_id=%llu contents=%u layout=%u",
         res->venus_res_id,
         (unsigned long long)res->venus.image_obj.id,
         res->venus.contents_initialized,
         res->venus.layout);
      res->data_dirty = false;
   }

   yttrium_pipeline_retire_sampled_texture_cpu_shadow(res);

   return res->venus.initialized && !res->venus.buffer_backed &&
          res->venus.image;
}

static bool
yttrium_pipeline_ensure_resource_venus_sampled_buffer(
   struct pipe_context *ctx,
   struct yttrium_resource *res,
   enum pipe_format format,
   uint64_t allocation_size)
{
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);

   if (!res || res->base.target != PIPE_BUFFER || !res->size ||
       !allocation_size || util_format_get_blocksize(format) == 0)
      return false;

   if (!res->venus.initialized) {
      uint64_t memory_id = 0;
      if (!yttrium_venus_create_sampled_buffer(screen->venus, &res->venus,
                                               allocation_size, format,
                                               &memory_id)) {
         YTTRIUM_LOG("yttrium: shader_draw_probe sampled buffer Venus create failed texture=%p size=0x%llx alloc_size=0x%llx format=%u\n",
                     (void *)&res->base,
                     (unsigned long long)res->size,
                     (unsigned long long)allocation_size,
                     format);
         return false;
      }

      YTTRIUM_LOG("yttrium: shader_draw_probe sampled buffer Venus buffer texture=%p buffer_id=%llu memory_id=%llu size=0x%llx alloc_size=0x%llx format=%u\n",
                  (void *)&res->base,
                  (unsigned long long)res->venus.buffer_obj.id,
                  (unsigned long long)memory_id,
                  (unsigned long long)res->size,
                  (unsigned long long)allocation_size,
                  format);
   }

   return res->venus.initialized && res->venus.buffer_backed &&
          res->venus.buffer &&
          (res->venus.buffer_usage &
           VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT) != 0;
}

static bool
yttrium_pipeline_ensure_resource_venus_storage_buffer(
   struct pipe_context *ctx,
   struct yttrium_resource *res,
   enum pipe_format format,
   uint64_t allocation_size)
{
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);

   if (!res || res->base.target != PIPE_BUFFER || !res->size ||
       !allocation_size || util_format_get_blocksize(format) == 0)
      return false;

   if (!res->venus.initialized) {
      uint64_t memory_id = 0;
      if (!yttrium_venus_create_sampled_buffer(screen->venus, &res->venus,
                                               allocation_size, format,
                                               &memory_id)) {
         YTTRIUM_LOG("yttrium: shader_draw_probe storage buffer Venus create failed image=%p size=0x%llx alloc_size=0x%llx format=%u\n",
                     (void *)&res->base,
                     (unsigned long long)res->size,
                     (unsigned long long)allocation_size,
                     format);
         return false;
      }
      if (!res->venus_res_id)
         res->venus_res_id = (uint32_t)memory_id;
   }

   return res->venus.initialized && res->venus.buffer_backed &&
          res->venus.buffer &&
          (res->venus.buffer_usage &
           VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT) != 0;
}

static bool
yttrium_pipeline_append_null_sampled_buffer(
   struct pipe_context *ctx,
   uint32_t binding,
   struct yttrium_venus_sampled_image *sampled_images,
   uint32_t *sampled_image_count)
{
   static const uint8_t zeros[4096] = { 0 };
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   struct yttrium_venus_resource *null_resource = NULL;
   uint32_t null_resource_id = 0;
   const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;

   if (!sampled_images || !sampled_image_count)
      return false;

   if (!yttrium_venus_ensure_null_sampled_buffer(
          screen->venus, format, sizeof(zeros), &null_resource,
          &null_resource_id))
      return false;

   sampled_images[*sampled_image_count] =
      (struct yttrium_venus_sampled_image) {
         .resource = null_resource,
         .buffer_data = zeros,
         .buffer_size = sizeof(zeros),
         .buffer_offset = 0,
         .buffer_range = sizeof(zeros),
         .resource_id = null_resource_id,
         .binding = binding,
         .format = format,
         .buffer = true,
      };
   (*sampled_image_count)++;
   return true;
}

static bool
yttrium_pipeline_format_has_stencil(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_S8_UINT:
   case VK_FORMAT_D16_UNORM_S8_UINT:
   case VK_FORMAT_D24_UNORM_S8_UINT:
   case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return true;
   default:
      return false;
   }
}

static VkImageAspectFlags
yttrium_pipeline_sampled_image_aspect(enum pipe_format format,
                                      const struct yttrium_resource *src)
{
   if (src && !src->venus.buffer_backed) {
      const VkFormat resource_format = src->venus.vk_format;
      const bool backing_has_depth =
         resource_format == VK_FORMAT_D16_UNORM ||
         resource_format == VK_FORMAT_D16_UNORM_S8_UINT ||
         resource_format == VK_FORMAT_D24_UNORM_S8_UINT ||
         resource_format == VK_FORMAT_D32_SFLOAT ||
         resource_format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
         resource_format == VK_FORMAT_X8_D24_UNORM_PACK32;
      const bool backing_has_stencil =
         yttrium_pipeline_format_has_stencil(resource_format);

      if (backing_has_depth || backing_has_stencil) {
         switch (format) {
         case PIPE_FORMAT_S8_UINT:
         case PIPE_FORMAT_R8_UINT:
         case PIPE_FORMAT_X24S8_UINT:
         case PIPE_FORMAT_S8X24_UINT:
         case PIPE_FORMAT_X32_S8X24_UINT:
            return backing_has_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT :
                                         VK_IMAGE_ASPECT_DEPTH_BIT;
         default:
            return backing_has_depth ? VK_IMAGE_ASPECT_DEPTH_BIT :
                                       VK_IMAGE_ASPECT_STENCIL_BIT;
         }
      }
   }

   switch (format) {
   case PIPE_FORMAT_Z16_UNORM:
   case PIPE_FORMAT_Z24X8_UNORM:
   case PIPE_FORMAT_X8Z24_UNORM:
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
   case PIPE_FORMAT_S8_UINT_Z24_UNORM:
   case PIPE_FORMAT_Z32_FLOAT:
   case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT:
   case PIPE_FORMAT_Z16_UNORM_S8_UINT:
      return VK_IMAGE_ASPECT_DEPTH_BIT;
   case PIPE_FORMAT_S8_UINT:
   case PIPE_FORMAT_X24S8_UINT:
   case PIPE_FORMAT_S8X24_UINT:
   case PIPE_FORMAT_X32_S8X24_UINT:
      return VK_IMAGE_ASPECT_STENCIL_BIT;
   default:
      return VK_IMAGE_ASPECT_COLOR_BIT;
   }
}

static bool
yttrium_pipeline_compute_null_sampled_image_supported(
   const struct yttrium_shader_state *shader,
   const struct yttrium_venus_sampled_binding_layout *layout)
{
   if (shader->stage != MESA_SHADER_COMPUTE)
      return true;

   const uint32_t slot = layout->raw_slot;
   const uint8_t target = slot < ARRAY_SIZE(shader->info.sampler_targets) ?
      shader->info.sampler_targets[slot] : TGSI_TEXTURE_UNKNOWN;
   const uint8_t type = slot < ARRAY_SIZE(shader->info.sampler_type) ?
      shader->info.sampler_type[slot] : TGSI_RETURN_TYPE_COUNT;
   /* The existing zero image is non-array, single-sample 2D UNORM. It can
    * serve floating-point fetches, but cannot represent integer, comparison,
    * array, multisample, cube or 3D shader image declarations. */
   const bool float_return = type == TGSI_RETURN_TYPE_FLOAT ||
      type == TGSI_RETURN_TYPE_UNORM || type == TGSI_RETURN_TYPE_SNORM;
   if (target == TGSI_TEXTURE_2D && float_return &&
       !layout->sampler.compare_enable)
      return true;

   YTTRIUM_WARN("yttrium: compute dispatch skipped owner=yttrium-pipeline reason=null_sampled_image_type_unsupported slot=%u binding=%u target=%u return_type=%u compare=%u action=reject-dispatch\n",
                slot, layout->binding, target, type,
                layout->sampler.compare_enable);
   return false;
}

struct yttrium_pipeline_owned_sampled_buffers {
   void *buffers[YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   uint32_t count;
};

static void
yttrium_pipeline_cleanup_sampled_buffers(
   struct yttrium_pipeline_owned_sampled_buffers *owned)
{
   for (uint32_t i = 0; i < owned->count; i++)
      FREE(owned->buffers[i]);
   owned->count = 0;
}

static bool
yttrium_pipeline_collect_sampled_textures(
   struct pipe_context *ctx,
   struct yttrium_context *yctx,
   struct yttrium_resource *dst,
   const struct yttrium_pipeline *pipeline,
   struct yttrium_venus_sampled_image *sampled_images,
   uint32_t *sampled_image_count,
   struct yttrium_pipeline_owned_sampled_buffers *sampled_owned_buffers)
{
   if (!sampled_images || !sampled_image_count) {
      yttrium_pipeline_trace_sampled_texture_fail(
         "bad_output_args", UINT32_MAX, UINT32_MAX, 0, dst, NULL, NULL);
      return false;
   }
   *sampled_image_count = 0;

   if (!pipeline || !pipeline->key.sampled_binding_count)
      return true;
   if (pipeline->key.sampled_binding_count >
       YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES) {
      YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped sampled descriptor count=%u max=%u\n",
                   pipeline->key.sampled_binding_count,
                   YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES);
      return false;
   }

   for (uint32_t descriptor_index = 0;
        descriptor_index < pipeline->key.sampled_binding_count;
        descriptor_index++) {
      const struct yttrium_venus_sampled_binding_layout *layout =
         &pipeline->key.sampled_bindings[descriptor_index];
      const mesa_shader_stage sampled_stage =
         (mesa_shader_stage)layout->stage;
      const uint32_t slot = layout->raw_slot;
      const uint32_t binding = layout->binding;
      const bool sampled_buffer = layout->buffer != VK_FALSE;
      const struct yttrium_shader_state *shader =
         sampled_stage >= MESA_SHADER_VERTEX &&
         sampled_stage <= MESA_SHADER_COMPUTE ?
         yctx->shaders[sampled_stage] : NULL;
      const uint32_t sampler_mask =
         yttrium_shader_state_sampler_used_mask(shader);

      if (!shader || slot >= PIPE_MAX_SAMPLERS ||
          !(sampler_mask & (1u << slot)) ||
          binding != yttrium_shader_sampler_binding(sampled_stage, slot)) {
         yttrium_pipeline_trace_sampled_texture_fail(
            "bad_sampler_binding", slot, binding, sampler_mask, dst, NULL,
            NULL);
         YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped sampled binding mismatch descriptor=%u stage=%u slot=%u binding=%u sampler_mask=0x%x\n",
                      descriptor_index, sampled_stage, slot, binding,
                      sampler_mask);
         return false;
      }

      unsigned view_slot = slot;
      if (!yttrium_pipeline_resolve_sampler_view_slot(
             yctx, shader, sampled_stage, sampler_mask, slot, &view_slot)) {
         if (sampled_buffer) {
            if (!yttrium_pipeline_append_null_sampled_buffer(
                   ctx, binding, sampled_images, sampled_image_count)) {
               yttrium_pipeline_trace_sampled_texture_fail(
                  "null_sampled_buffer_unavailable", slot, binding,
                  sampler_mask, dst, NULL, NULL);
               return false;
            }
            continue;
         }

         if (!yttrium_pipeline_compute_null_sampled_image_supported(shader,
                                                                    layout))
            return false;
         sampled_images[*sampled_image_count] =
            (struct yttrium_venus_sampled_image) {
               .resource = NULL,
               .resource_id = 0,
               .binding = binding,
               .swizzle_key = YTTRIUM_VENUS_SAMPLE_SWIZZLE_IDENTITY,
               .view_type = VK_IMAGE_VIEW_TYPE_2D,
               .aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT,
               .first_level = 0,
               .level_count = 1,
               .first_layer = 0,
               .layer_count = 1,
               .format = PIPE_FORMAT_R8G8B8A8_UNORM,
               .buffer = false,
            };
         (*sampled_image_count)++;
         continue;
      }
      struct pipe_sampler_view *view =
         yctx->sampler_views[sampled_stage][view_slot];
      struct yttrium_resource *src =
         view && view->texture ? yttrium_resource(view->texture) : NULL;
      if (!src) {
         if (sampled_buffer) {
            if (!yttrium_pipeline_append_null_sampled_buffer(
                   ctx, binding, sampled_images, sampled_image_count)) {
               yttrium_pipeline_trace_sampled_texture_fail(
                  "null_sampled_buffer_unavailable", slot, binding,
                  sampler_mask, dst, view, NULL);
               return false;
            }
            continue;
         }

         if (!yttrium_pipeline_compute_null_sampled_image_supported(shader,
                                                                    layout))
            return false;
         sampled_images[*sampled_image_count] =
            (struct yttrium_venus_sampled_image) {
               .resource = NULL,
               .resource_id = 0,
               .binding = binding,
               .swizzle_key = YTTRIUM_VENUS_SAMPLE_SWIZZLE_IDENTITY,
               .view_type = VK_IMAGE_VIEW_TYPE_2D,
               .aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT,
               .first_level = 0,
               .level_count = 1,
               .first_layer = 0,
               .layer_count = 1,
               .format = PIPE_FORMAT_R8G8B8A8_UNORM,
               .buffer = false,
            };
         (*sampled_image_count)++;
         continue;
      }

      bool unsupported_attachment_alias = false;
      for (uint32_t rt_index = 0;
           rt_index < pipeline->key.rt_count; rt_index++) {
         if (pipeline->rt_resources[rt_index] != &src->base)
            continue;
         unsupported_attachment_alias |=
            (pipeline->key.color_feedback_loop_mask &
             (1u << rt_index)) == 0;
      }
      if (pipeline->zs_resource == &src->base) {
         const bool depth_read_only = pipeline->key.depth_read_only &&
            yttrium_pipeline_read_only_depth_view(view, src);
         unsupported_attachment_alias |=
            !pipeline->key.depth_feedback_loop && !depth_read_only;
      }
      if (unsupported_attachment_alias) {
         yttrium_pipeline_trace_sampled_texture_fail(
            "feedback_loop_key_missing", slot, binding, sampler_mask, dst,
            view, src);
         YTTRIUM_WARN("yttrium: shader_draw_probe native draw skipped owner=yttrium reason=attachment_feedback_loop_key_missing slot=%u res_id=%u\n",
                      slot, src->venus_res_id);
         return false;
      }

      const uint8_t sampler_return =
         slot < ARRAY_SIZE(shader->info.sampler_type) ?
         shader->info.sampler_type[slot] : TGSI_RETURN_TYPE_COUNT;
      const uint8_t shader_target =
         slot < ARRAY_SIZE(shader->info.sampler_targets) ?
         shader->info.sampler_targets[slot] : TGSI_TEXTURE_UNKNOWN;
      const enum pipe_format sampled_format =
         sampled_buffer ?
         yttrium_pipeline_sampled_buffer_format(shader, slot, view->format) :
         view->format;
      const bool sampled_mixed_return =
         sampled_buffer && slot < 32 &&
         (shader->sampler_mixed_return_mask & (1u << slot));
      if (sampled_mixed_return &&
          util_format_is_pure_integer(view->format) &&
          sampled_format == view->format) {
         yttrium_pipeline_trace_sampled_texture_fail(
            "mixed_buffer_integer_format_unsupported", slot, binding,
            sampler_mask, dst, view, src);
         YTTRIUM_WARN("yttrium: shader_draw_probe native draw skipped "
                      "owner=yttrium-pipeline "
                      "reason=mixed_buffer_integer_format_unsupported "
                      "stage=%s slot=%u binding=%u view_format=%u "
                      "action=reject-draw\n",
                      yttrium_shader_stage_name(shader->stage), slot,
                      binding, view->format);
         return false;
      }
      const bool sampled_bitcast_upload =
         sampled_buffer &&
         yttrium_pipeline_sampled_buffer_uses_r8_bitcast_coords(
            shader, slot, view->format);
      if (sampled_buffer && sampled_format != view->format) {
         yttrium_trace_debug_stringf(
            "yttrium: shader_draw_probe sampled buffer format override slot=%u binding=%u sampler_return=%u view_format=%u sampled_format=%u bitcast_upload=%u",
            slot, binding, sampler_return, view->format, sampled_format,
            sampled_bitcast_upload);
      }

      if (sampled_buffer) {
         const uint64_t src_size =
            MIN2((uint64_t)src->base.width0, src->size);
         if (!src_size) {
            yttrium_pipeline_trace_sampled_texture_fail(
               "sampled_buffer_empty_source", slot, binding, sampler_mask,
               dst, view, src);
            return false;
         }
         if (sampled_bitcast_upload && src_size > UINT64_MAX / 4) {
            yttrium_pipeline_trace_sampled_texture_fail(
               "sampled_buffer_bitcast_too_large", slot, binding,
               sampler_mask, dst, view, src);
            return false;
         }
         const uint64_t sampled_allocation_size =
            sampled_bitcast_upload ? src_size * 4 : src->size;

         if (src->base.target != PIPE_BUFFER) {
            yttrium_pipeline_trace_sampled_texture_fail(
               "buffer_target_mismatch", slot, binding, sampler_mask, dst,
               view, src);
            YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped sampled buffer target mismatch slot=%u texture=%p view_target=%u target=%u format=%u\n",
                         slot, (void *)view->texture, view->target,
                         src->base.target, view->format);
            return false;
         }
         if (!src->data && !src->venus.contents_initialized) {
            yttrium_pipeline_trace_sampled_texture_fail(
               "sampled_buffer_no_data", slot, binding, sampler_mask, dst,
               view, src);
            YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped sampled buffer no data slot=%u texture=%p target=%u format=%u res_id=%u\n",
                         slot, (void *)view->texture, src->base.target,
                         view->format, src->venus_res_id);
            return false;
         }
         if (!yttrium_pipeline_ensure_resource_venus_sampled_buffer(
                ctx, src, sampled_format, sampled_allocation_size)) {
            yttrium_pipeline_trace_sampled_texture_fail(
               "not_venus_sampleable_buffer", slot, binding, sampler_mask,
               dst, view, src);
            YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped sampled buffer not Venus-sampleable slot=%u texture=%p target=%u format=%u sampled_format=%u alloc_size=0x%llx data=%p hAllocation=0x%lx\n",
                         slot, (void *)view->texture, src->base.target,
                         view->format, sampled_format,
                         (unsigned long long)sampled_allocation_size, src->data,
                         (unsigned long)src->hAllocation);
            return false;
         }

         if (!src->venus.initialized || !src->venus.buffer_backed ||
             !src->venus.buffer) {
            yttrium_pipeline_trace_sampled_texture_fail(
               "post_ensure_invalid_buffer", slot, binding, sampler_mask,
               dst, view, src);
            YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped sampled buffer post-ensure invalid slot=%u res_id=%u initialized=%u buffer_backed=%u buffer=%p\n",
                         slot, src->venus_res_id, src->venus.initialized,
                         src->venus.buffer_backed, src->venus.buffer);
            return false;
         }
      } else {
         if (!yttrium_pipeline_ensure_resource_venus_texture(ctx, src)) {
            yttrium_pipeline_trace_sampled_texture_fail(
               "not_venus_sampleable", slot, binding, sampler_mask, dst,
               view, src);
            YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped sampled texture not Venus-sampleable slot=%u texture=%p target=%u format=%u data=%p hAllocation=0x%lx\n",
                         slot, (void *)view->texture, src->base.target,
                         src->base.format, src->data,
                         (unsigned long)src->hAllocation);
            return false;
         }

         if (!src->venus.initialized || src->venus.buffer_backed ||
             !src->venus.image) {
            yttrium_pipeline_trace_sampled_texture_fail(
               "post_ensure_invalid", slot, binding, sampler_mask, dst,
               view, src);
            YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped sampled texture post-ensure invalid slot=%u res_id=%u initialized=%u buffer_backed=%u image=0x%llx\n",
                         slot, src->venus_res_id, src->venus.initialized,
                         src->venus.buffer_backed,
                         (unsigned long long)yttrium_pipeline_vk_image_to_u64(
                            src->venus.image));
            return false;
         }

      }

      uint64_t buffer_upload_size = 0;
      uint64_t buffer_offset = 0;
      uint64_t buffer_range = 0;
      unsigned buffer_blocksize = 0;
      const void *buffer_data = NULL;
      VkImageViewType image_view_type = VK_IMAGE_VIEW_TYPE_2D;
      uint32_t image_first_level = 0;
      uint32_t image_level_count = 1;
      uint32_t image_first_layer = 0;
      uint32_t image_layer_count = 1;
      if (sampled_buffer) {
         const uint64_t src_size =
            MIN2((uint64_t)src->base.width0, src->size);
         buffer_blocksize = util_format_get_blocksize(sampled_format);
         buffer_upload_size = src_size;
         buffer_offset = view->u.buf.offset;
         buffer_range = view->u.buf.size;

         if (!buffer_blocksize) {
            yttrium_pipeline_trace_sampled_texture_fail(
               "sampled_buffer_bad_format", slot, binding, sampler_mask,
               dst, view, src);
            return false;
         }
         if (buffer_offset > src_size) {
            yttrium_pipeline_trace_sampled_texture_fail(
               "sampled_buffer_offset_oob", slot, binding, sampler_mask,
               dst, view, src);
            return false;
         }
         if (!buffer_range)
            buffer_range = src_size - buffer_offset;
         if (buffer_range > src_size - buffer_offset)
            buffer_range = src_size - buffer_offset;
         buffer_range -= buffer_range % buffer_blocksize;
         if (!buffer_range) {
            yttrium_pipeline_trace_sampled_texture_fail(
               "sampled_buffer_empty_range", slot, binding, sampler_mask,
               dst, view, src);
            return false;
         }
         if (buffer_offset % buffer_blocksize) {
            yttrium_pipeline_trace_sampled_texture_fail(
               "sampled_buffer_unaligned_offset", slot, binding,
               sampler_mask, dst, view, src);
            return false;
         }
         if (buffer_upload_size > SIZE_MAX) {
            yttrium_pipeline_trace_sampled_texture_fail(
               "sampled_buffer_too_large", slot, binding, sampler_mask,
               dst, view, src);
            return false;
         }
         /* An SRV can consume a buffer produced by a preceding UAV draw.
          * A non-NULL detached CPU shadow is not evidence of ownership. */
         buffer_data = src->data_dirty ? src->data : NULL;

         if (sampled_bitcast_upload) {
            if (src->gpu_buffer_written ||
                ((src->base.bind & PIPE_BIND_STREAM_OUTPUT) &&
                 !src->data_dirty)) {
               YTTRIUM_WARN("yttrium: sampled buffer rejected owner=yttrium-pipeline reason=gpu-buffer-needs-cpu-bitcast-expansion slot=%u action=reject-draw\n", slot);
               return false;
            }
            size_t expanded_size = 0;
            void *expanded =
               yttrium_pipeline_expand_r8_to_r32_float_bits(
                  src->data, src_size, &expanded_size);
            if (!expanded || !expanded_size) {
               FREE(expanded);
               yttrium_pipeline_trace_sampled_texture_fail(
                  "sampled_buffer_bitcast_expand_failed", slot, binding,
                  sampler_mask, dst, view, src);
               return false;
            }
            if (sampled_owned_buffers)
               sampled_owned_buffers->buffers[sampled_owned_buffers->count++] =
                  expanded;
            else {
               FREE(expanded);
               yttrium_pipeline_trace_sampled_texture_fail(
                  "sampled_buffer_bitcast_no_owner", slot, binding,
                  sampler_mask, dst, view, src);
               return false;
            }

            buffer_data = expanded;
            buffer_upload_size = expanded_size;
            buffer_offset *= 4;
            buffer_range *= 4;
            buffer_blocksize = util_format_get_blocksize(sampled_format);
            yttrium_trace_debug_stringf(
               "yttrium: shader_draw_probe sampled buffer bitcast upload slot=%u binding=%u src_size=0x%llx expanded_size=0x%llx view_offset=0x%llx view_size=0x%llx scaled_offset=0x%llx scaled_range=0x%llx first=%08x,%08x,%08x,%08x",
               slot, binding,
               (unsigned long long)src_size,
               (unsigned long long)buffer_upload_size,
               (unsigned long long)view->u.buf.offset,
               (unsigned long long)view->u.buf.size,
               (unsigned long long)buffer_offset,
               (unsigned long long)buffer_range,
               yttrium_pipeline_peek_u32(expanded, expanded_size, 0),
               yttrium_pipeline_peek_u32(expanded, expanded_size, 1),
               yttrium_pipeline_peek_u32(expanded, expanded_size, 2),
               yttrium_pipeline_peek_u32(expanded, expanded_size, 3));
         }
      } else if (!yttrium_pipeline_sampled_image_view_desc(
                    view, src, shader_target, &image_view_type, &image_first_level,
                    &image_level_count, &image_first_layer,
                    &image_layer_count)) {
         yttrium_pipeline_trace_sampled_texture_fail(
            "sampled_image_bad_view", slot, binding, sampler_mask, dst, view,
            src);
         return false;
      }

      sampled_images[*sampled_image_count] =
         (struct yttrium_venus_sampled_image) {
            .resource = &src->venus,
            .pipe_resource = &src->base,
            .buffer_data = sampled_buffer ? buffer_data : NULL,
            .buffer_size = sampled_buffer ? (size_t)buffer_upload_size : 0,
            .buffer_offset = sampled_buffer ? buffer_offset : 0,
            .buffer_range = sampled_buffer ? buffer_range : 0,
            .resource_id = src->venus_res_id,
            .binding = binding,
            .swizzle_key =
               yttrium_pipeline_sample_swizzle_key(sampled_format, view),
            .view_type = image_view_type,
            .aspect_mask =
               sampled_buffer ? VK_IMAGE_ASPECT_COLOR_BIT :
                                yttrium_pipeline_sampled_image_aspect(sampled_format,
                                                                      src),
            .first_level = image_first_level,
            .level_count = image_level_count,
            .first_layer = image_first_layer,
            .layer_count = image_layer_count,
            .format = sampled_format,
            .buffer = sampled_buffer,
         };
      (*sampled_image_count)++;

      if (yttrium_pipeline_verbose_trace_enabled()) {
         const struct yttrium_pipeline_data_summary src_summary =
            yttrium_pipeline_summarize_data(src->data, (size_t)src->size);
         const struct yttrium_pipeline_data_summary upload_summary =
            sampled_buffer ?
               yttrium_pipeline_summarize_data(buffer_data,
                                               (size_t)buffer_upload_size) :
               (struct yttrium_pipeline_data_summary) { 0 };

         yttrium_trace_debug_stringf(
            "yttrium: shader_draw_probe sampled texture selected slot=%u binding=%u buffer=%u bitcast_upload=%u texture=%p alloc=0x%lx res_id=%u image_id=%llu buffer_id=%llu display=%u primary=%u classic=%u initialized=%u contents=%u layout=%u data=%p upload_data=%p size=0x%llx view_offset=0x%llx view_size=0x%llx buffer_offset=0x%llx buffer_range=0x%llx upload_size=0x%llx blocksize=%u %ux%u format=%u view_format=%u sampled_format=%u sampler_return=%u first=%08x,%08x,%08x,%08x view_first=%08x,%08x,%08x,%08x src_inspected=0x%llx src_nonzero=0x%llx src_first_nz=0x%llx:%08x src_last_nz=0x%llx:%08x src_hash=0x%x upload_inspected=0x%llx upload_nonzero=0x%llx upload_first_nz=0x%llx:%08x upload_last_nz=0x%llx:%08x upload_hash=0x%x",
            slot,
            binding,
            sampled_buffer,
            sampled_bitcast_upload,
            (void *)view->texture,
            (unsigned long)src->hAllocation,
            src->venus_res_id,
            (unsigned long long)src->venus.image_obj.id,
            (unsigned long long)src->venus.buffer_obj.id,
            src->display_target,
            src->primary_target,
            src->classic_display,
            src->venus.initialized,
            src->venus.contents_initialized,
            src->venus.layout,
            src->data,
            buffer_data,
            (unsigned long long)src->size,
            (unsigned long long)(sampled_buffer ? view->u.buf.offset : 0),
            (unsigned long long)(sampled_buffer ? view->u.buf.size : 0),
            (unsigned long long)buffer_offset,
            (unsigned long long)buffer_range,
            (unsigned long long)buffer_upload_size,
            buffer_blocksize,
            src->base.width0,
            src->base.height0,
            src->base.format,
            view->format,
            sampled_format,
            sampler_return,
            yttrium_pipeline_peek_u32(src->data, (size_t)src->size, 0),
            yttrium_pipeline_peek_u32(src->data, (size_t)src->size, 1),
            yttrium_pipeline_peek_u32(src->data, (size_t)src->size, 2),
            yttrium_pipeline_peek_u32(src->data, (size_t)src->size, 3),
            yttrium_pipeline_peek_u32(
               src->data ? (const uint8_t *)src->data + buffer_offset : NULL,
               sampled_buffer && src->size >= buffer_offset ?
                  (size_t)(src->size - buffer_offset) : 0, 0),
            yttrium_pipeline_peek_u32(
               src->data ? (const uint8_t *)src->data + buffer_offset : NULL,
               sampled_buffer && src->size >= buffer_offset ?
                  (size_t)(src->size - buffer_offset) : 0, 1),
            yttrium_pipeline_peek_u32(
               src->data ? (const uint8_t *)src->data + buffer_offset : NULL,
               sampled_buffer && src->size >= buffer_offset ?
                  (size_t)(src->size - buffer_offset) : 0, 2),
            yttrium_pipeline_peek_u32(
               src->data ? (const uint8_t *)src->data + buffer_offset : NULL,
               sampled_buffer && src->size >= buffer_offset ?
                  (size_t)(src->size - buffer_offset) : 0, 3),
            (unsigned long long)src_summary.inspected,
            (unsigned long long)src_summary.nonzero_bytes,
            (unsigned long long)src_summary.first_nonzero,
            src_summary.first_nonzero_u32,
            (unsigned long long)src_summary.last_nonzero,
            src_summary.last_nonzero_u32,
            src_summary.hash,
            (unsigned long long)upload_summary.inspected,
            (unsigned long long)upload_summary.nonzero_bytes,
            (unsigned long long)upload_summary.first_nonzero,
            upload_summary.first_nonzero_u32,
            (unsigned long long)upload_summary.last_nonzero,
            upload_summary.last_nonzero_u32,
            upload_summary.hash);
      }
   }

   if (*sampled_image_count != pipeline->key.sampled_binding_count) {
      YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped sampled descriptor collection incomplete expected=%u got=%u\n",
                   pipeline->key.sampled_binding_count,
                   *sampled_image_count);
      return false;
   }

   return true;
}

static bool
yttrium_pipeline_storage_image_view_desc(
   const struct pipe_image_view *view,
   const struct yttrium_resource *src,
   bool shader_arrayed,
   VkImageViewType *view_type,
   uint32_t *first_level,
   uint32_t *level_count,
   uint32_t *first_layer,
   uint32_t *layer_count)
{
   if (!view || !src || !view_type || !first_level || !level_count ||
       !first_layer || !layer_count)
      return false;

   /* D3D represents a one-element Texture1D/2DArray with the same resource
    * shape as a non-array texture, so the resource alone cannot say which
    * the shader meant.  The view type has to match the OpTypeImage Arrayed
    * flag, so follow the shader's declaration - the sampled path already
    * preserves it the same way.
    */
   switch (src->base.target) {
   case PIPE_TEXTURE_1D:
      *view_type = shader_arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY :
                                    VK_IMAGE_VIEW_TYPE_1D;
      break;
   case PIPE_TEXTURE_1D_ARRAY:
      *view_type = VK_IMAGE_VIEW_TYPE_1D_ARRAY;
      break;
   case PIPE_TEXTURE_2D:
      *view_type = shader_arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY :
                                    VK_IMAGE_VIEW_TYPE_2D;
      break;
   case PIPE_TEXTURE_2D_ARRAY:
      *view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      break;
   case PIPE_TEXTURE_3D:
      *view_type = VK_IMAGE_VIEW_TYPE_3D;
      break;
   default:
      return false;
   }

   if (view->u.tex.last_layer < view->u.tex.first_layer)
      return false;

   *first_level = view->u.tex.level;
   *level_count = 1;
   if (src->base.target == PIPE_TEXTURE_3D) {
      *first_layer = 0;
      *layer_count = 1;
   } else {
      *first_layer = view->u.tex.first_layer;
      *layer_count = view->u.tex.last_layer - view->u.tex.first_layer + 1;
   }

   if (*first_level >= MAX2(src->venus.levels, 1) ||
       *first_layer >= MAX2(src->venus.layers, 1) ||
       *layer_count > MAX2(src->venus.layers, 1) - *first_layer)
      return false;

   return true;
}

static bool
yttrium_pipeline_storage_views_match(const struct pipe_image_view *a,
                                     const struct pipe_image_view *b)
{
   if (a->resource != b->resource || !a->resource)
      return false;
   const enum pipe_format a_format = a->format != PIPE_FORMAT_NONE ?
      a->format : a->resource->format;
   const enum pipe_format b_format = b->format != PIPE_FORMAT_NONE ?
      b->format : b->resource->format;
   if (a_format != b_format)
      return false;
   if (a->resource->target == PIPE_BUFFER) {
      const uint32_t size = a->resource->width0;
      if (a->u.buf.offset != b->u.buf.offset || a->u.buf.offset > size)
         return false;
      const uint32_t remaining = size - a->u.buf.offset;
      return (a->u.buf.size ? MIN2(a->u.buf.size, remaining) : remaining) ==
             (b->u.buf.size ? MIN2(b->u.buf.size, remaining) : remaining);
   }
   return a->u.tex.level == b->u.tex.level &&
          a->u.tex.first_layer == b->u.tex.first_layer &&
          a->u.tex.last_layer == b->u.tex.last_layer &&
          a->u.tex.single_layer_view == b->u.tex.single_layer_view &&
          a->u.tex.is_2d_view_of_3d == b->u.tex.is_2d_view_of_3d;
}

static bool
yttrium_pipeline_collect_storage_images(
   struct pipe_context *ctx,
   const struct yttrium_context *yctx,
   const struct yttrium_pipeline *pipeline,
   mesa_shader_stage stage,
   struct yttrium_venus_storage_image *storage_images,
   uint32_t *storage_image_count)
{
   if (!ctx || !yctx || !pipeline || !storage_images || !storage_image_count)
      return false;

   *storage_image_count = 0;
   const uint64_t storage_mask =
      pipeline->storage_image_mask | pipeline->storage_buffer_mask;
   if (!storage_mask)
      return true;

   for (uint32_t slot = 0;
        slot < YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES; slot++) {
      const uint64_t slot_mask = 1ull << slot;
      if (!(storage_mask & slot_mask))
         continue;

      const struct pipe_image_view *view = NULL;
      struct yttrium_resource *src = NULL;
      mesa_shader_stage view_stage = MESA_SHADER_NONE;
      uint32_t stages = stage == MESA_SHADER_COMPUTE ?
         1u << stage : pipeline->key.storage_stage_mask;
      /* Graphics shares one Vulkan descriptor per slot, not one per stage.
       * Select a stage that actually declares this slot, and reject differing
       * views rather than accidentally using another stage's hidden counter. */
      while (stages) {
         const mesa_shader_stage candidate = (mesa_shader_stage)u_bit_scan(&stages);
         const struct yttrium_shader_state *shader =
            candidate == MESA_SHADER_FRAGMENT && pipeline->generated_fs ?
               pipeline->generated_fs :
            candidate == MESA_SHADER_VERTEX && pipeline->generated_vs ?
               pipeline->generated_vs : yctx->shaders[candidate];
         if (!(yttrium_shader_state_image_used_mask(shader) & slot_mask))
            continue;
         const struct pipe_image_view *candidate_view =
            &yctx->shader_images[candidate][slot];
         if (!candidate_view->resource ||
             (view && !yttrium_pipeline_storage_views_match(view, candidate_view))) {
            YTTRIUM_WARN("yttrium: storage image rejected owner=yttrium-pipeline reason=missing_or_conflicting_stage_view slot=%u first_stage=%u stage=%u first_resource=%p resource=%p\n",
                         slot, view_stage, candidate,
                         view ? view->resource : NULL, candidate_view->resource);
            return false;
         }
         view = candidate_view;
         view_stage = candidate;
         src = yttrium_resource(view->resource);
         if (shader->formatless_image_mask & slot_mask) {
            const enum pipe_format format = view->format != PIPE_FORMAT_NONE ?
               view->format : src->base.format;
            const bool read = shader->formatless_image_read_mask & slot_mask;
            const bool write = shader->formatless_image_write_mask & slot_mask;
            const bool uint_type = shader->formatless_image_uint_mask & slot_mask;
            const bool sint_type = shader->formatless_image_sint_mask & slot_mask;
            if (src->base.target == PIPE_BUFFER ||
                util_format_is_pure_uint(format) != uint_type ||
                util_format_is_pure_sint(format) != sint_type ||
                !yttrium_venus_storage_image_formatless_format_supported(
                   yttrium_screen(ctx->screen)->venus, format, read, write)) {
               YTTRIUM_WARN("yttrium: storage image rejected owner=yttrium-pipeline stage=%s slot=%u format=%s read=%u write=%u uint=%u sint=%u reason=incompatible-formatless-view\n",
                            yttrium_shader_stage_name(candidate), slot,
                            util_format_name(format), read, write,
                            uint_type, sint_type);
               return false;
            }
         }
      }
      if (!src)
         return false;

      if (pipeline->storage_buffer_mask & slot_mask) {
         if (src->base.target != PIPE_BUFFER)
            return false;

         const uint64_t src_size =
            MIN2((uint64_t)src->base.width0, src->size);
         const enum pipe_format format =
            view->format != PIPE_FORMAT_NONE ? view->format : src->base.format;
         const unsigned blocksize = util_format_get_blocksize(format);
         uint64_t buffer_offset = view->u.buf.offset;
         uint64_t buffer_range = view->u.buf.size;
         if (!src_size || !blocksize || buffer_offset > src_size)
            return false;
         if (!buffer_range)
            buffer_range = src_size - buffer_offset;
         if (buffer_range > src_size - buffer_offset)
            buffer_range = src_size - buffer_offset;
         buffer_range -= buffer_range % blocksize;
         if (!buffer_range || buffer_offset % blocksize)
            return false;

         if (!yttrium_pipeline_ensure_resource_venus_storage_buffer(
                ctx, src, format, src->size))
            return false;

         storage_images[*storage_image_count] =
            (struct yttrium_venus_storage_image) {
               .resource = &src->venus,
               .buffer_data = src->data_dirty ? src->data : NULL,
               .buffer_size = src->data_dirty && src->data ?
                  (size_t)src_size : 0,
               .buffer_offset = buffer_offset,
               .buffer_range = buffer_range,
               .resource_id = src->venus_res_id,
               .binding = yttrium_shader_storage_image_binding(slot),
               .format = format,
               .buffer = true,
            };
         (*storage_image_count)++;
         if (src->data_dirty && src->data)
            src->data_dirty = false;
         continue;
      }

      if (src->base.target == PIPE_BUFFER ||
          !src->venus.initialized || src->venus.buffer_backed ||
          !src->venus.image ||
          !(src->venus.image_usage & VK_IMAGE_USAGE_STORAGE_BIT))
         return false;

      /* Native UAV textures publish CPU writes directly through image staging.
       * A detached shadow has different ownership: a partial GPU write would
       * leave its untouched CPU bytes and GPU-produced bytes out of sync. */
      if (stage == MESA_SHADER_COMPUTE && src->data) {
         YTTRIUM_WARN("yttrium: compute dispatch skipped owner=yttrium-pipeline reason=shadow_backed_image_uav_unsupported slot=%u res_id=%u data=%p dirty=%u initialized=%u action=reject-dispatch\n",
                      slot, src->venus_res_id, src->data, src->data_dirty,
                      src->venus.contents_initialized);
         return false;
      }

      VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D;
      uint32_t first_level = 0;
      uint32_t level_count = 1;
      uint32_t first_layer = 0;
      uint32_t layer_count = 1;
      const struct yttrium_shader_state *storage_shader =
         stage < MESA_SHADER_STAGES ? yctx->shaders[stage] : NULL;
      const bool shader_arrayed =
         storage_shader && slot < 64 &&
         (storage_shader->image_array_mask & (1ull << slot)) != 0;
      if (!yttrium_pipeline_storage_image_view_desc(
             view, src, shader_arrayed, &view_type, &first_level,
             &level_count, &first_layer, &layer_count))
         return false;

      storage_images[*storage_image_count] =
         (struct yttrium_venus_storage_image) {
            .resource = &src->venus,
            .resource_id = src->venus_res_id,
            .binding = yttrium_shader_storage_image_binding(slot),
            .view_type = view_type,
            .aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT,
            .first_level = first_level,
            .level_count = level_count,
            .first_layer = first_layer,
            .layer_count = layer_count,
            .format = view->format != PIPE_FORMAT_NONE ?
               view->format : src->base.format,
         };
      (*storage_image_count)++;
   }

   return *storage_image_count == util_bitcount64(storage_mask);
}

/* Compute has no framebuffer dependency. Keep its bounded cache separate from
 * graphics state invalidation, and retire evictions through the same in-flight
 * batch tracking as graphics pipelines. A NULL shader releases the whole cache.
 */
void
yttrium_compute_pipeline_cache_invalidate(
   struct yttrium_context *yctx, const struct yttrium_shader_state *shader)
{
   struct yttrium_screen *screen = yttrium_screen(yctx->base.screen);
   for (uint32_t i = 0; i < YTTRIUM_COMPUTE_PIPELINE_CACHE_SIZE; i++) {
      struct yttrium_compute_pipeline_cache_entry *entry =
         yctx->compute_pipeline_cache[i];
      if (entry && (!shader || entry->shader_id == shader->id)) {
         yttrium_pipeline_destroy(screen->venus, entry->pipeline);
         FREE(entry);
         yctx->compute_pipeline_cache[i] = NULL;
      }
   }
}

static struct yttrium_pipeline *
yttrium_compute_pipeline_get(
   struct yttrium_context *yctx, const struct yttrium_shader_state *cs,
   const struct yttrium_pipeline *description,
   const struct yttrium_venus_ubo_binding_layout *ubo_bindings,
   uint32_t ubo_binding_count)
{
   struct yttrium_screen *screen = yttrium_screen(yctx->base.screen);
   uint32_t empty_slot = UINT32_MAX;
   for (uint32_t i = 0; i < YTTRIUM_COMPUTE_PIPELINE_CACHE_SIZE; i++) {
      const struct yttrium_compute_pipeline_cache_entry *entry =
         yctx->compute_pipeline_cache[i];
      if (!entry) {
         if (empty_slot == UINT32_MAX)
            empty_slot = i;
         continue;
      }
      struct yttrium_pipeline *pipeline = entry->pipeline;
      if (entry->shader_id == cs->id && entry->module_id == cs->module_obj.id &&
          entry->ubo_binding_count == ubo_binding_count &&
          pipeline->storage_image_mask == description->storage_image_mask &&
          pipeline->storage_buffer_mask == description->storage_buffer_mask &&
          pipeline->key.sampled_binding_count ==
             description->key.sampled_binding_count &&
          !memcmp(entry->ubo_bindings, ubo_bindings,
                  ubo_binding_count * sizeof(*ubo_bindings)) &&
          !memcmp(pipeline->key.sampled_bindings,
                  description->key.sampled_bindings,
                  description->key.sampled_binding_count *
                     sizeof(description->key.sampled_bindings[0])))
         return pipeline;
   }

   struct yttrium_compute_pipeline_cache_entry *entry =
      CALLOC_STRUCT(yttrium_compute_pipeline_cache_entry);
   struct yttrium_pipeline *pipeline = CALLOC_STRUCT(yttrium_pipeline);
   if (!entry || !pipeline) {
      YTTRIUM_WARN("yttrium: compute pipeline cache allocation failed owner=yttrium-pipeline action=reject-dispatch\n");
      FREE(entry);
      FREE(pipeline);
      return NULL;
   }

   /* Sampled layouts include immutable sampler state. Resource identities,
    * views and UBO contents are deliberately not cached: dispatch_compute
    * refreshes them in a fresh batch-owned descriptor set (or push commands).
    */
   pipeline->key = description->key;
   if (!yttrium_venus_compute_pipeline_init(
          screen->venus, pipeline, cs->module, ubo_bindings, ubo_binding_count,
          pipeline->key.sampled_bindings, pipeline->key.sampled_binding_count,
          description->storage_image_mask, description->storage_buffer_mask)) {
      yttrium_pipeline_destroy(screen->venus, pipeline);
      FREE(entry);
      return NULL;
   }
   entry->pipeline = pipeline;
   entry->shader_id = cs->id;
   entry->module_id = cs->module_obj.id;
   entry->ubo_binding_count = ubo_binding_count;
   memcpy(entry->ubo_bindings, ubo_bindings,
          ubo_binding_count * sizeof(*ubo_bindings));

   const uint32_t slot = empty_slot != UINT32_MAX ? empty_slot :
                           yctx->compute_pipeline_cache_next;
   struct yttrium_compute_pipeline_cache_entry *old =
      yctx->compute_pipeline_cache[slot];
   if (old) {
      yttrium_pipeline_destroy(screen->venus, old->pipeline);
      FREE(old);
   }
   yctx->compute_pipeline_cache[slot] = entry;
   yctx->compute_pipeline_cache_next =
      (slot + 1) % YTTRIUM_COMPUTE_PIPELINE_CACHE_SIZE;
   return pipeline;
}

void
yttrium_launch_grid(struct pipe_context *ctx,
                    const struct pipe_grid_info *info)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   struct yttrium_shader_state *cs =
      yctx ? yctx->shaders[MESA_SHADER_COMPUTE] : NULL;

   if (!ctx || !info || !cs || !yttrium_shader_state_has_module(cs))
      return;
   if (!info->grid[0] || !info->grid[1] || !info->grid[2])
      return;
   if (cs->info.shader_buffers_declared) {
      YTTRIUM_WARN("yttrium: compute dispatch skipped unsupported resources shader=%p samplers=0x%x ssbos=0x%x\n",
                   cs, cs->sampler_used_mask,
                   cs->info.shader_buffers_declared);
      return;
   }

   uint64_t image_mask = yttrium_shader_state_image_used_mask(cs);
   if (!image_mask && yctx->num_shader_images[MESA_SHADER_COMPUTE]) {
      for (uint32_t slot = 0;
           slot < MIN2(yctx->num_shader_images[MESA_SHADER_COMPUTE],
                       YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES); slot++) {
         if (yctx->shader_images[MESA_SHADER_COMPUTE][slot].resource)
            image_mask |= 1ull << slot;
      }
   }
   image_mask &= YTTRIUM_VENUS_PIPELINE_STORAGE_IMAGE_MASK;
   if (!image_mask)
      return;

   uint64_t storage_image_mask = 0;
   uint64_t storage_buffer_mask = 0;
   for (uint32_t slot = 0;
        slot < YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES; slot++) {
      const uint64_t slot_mask = 1ull << slot;
      if (!(image_mask & slot_mask))
         continue;

      const struct pipe_image_view *view =
         &yctx->shader_images[MESA_SHADER_COMPUTE][slot];
      struct yttrium_resource *res =
         view->resource ? yttrium_resource(view->resource) : NULL;
      if (!res) {
         YTTRIUM_WARN("yttrium: compute dispatch skipped missing image slot=%u shader=%p image_mask=0x%llx\n",
                      slot, cs, (unsigned long long)image_mask);
         return;
      }
      if (res->base.target == PIPE_BUFFER)
         storage_buffer_mask |= slot_mask;
      else
         storage_image_mask |= slot_mask;
   }
   struct yttrium_pipeline pipeline;
   memset(&pipeline, 0, sizeof(pipeline));
   pipeline.storage_image_mask = storage_image_mask;
   pipeline.storage_buffer_mask = storage_buffer_mask;
   pipeline.has_storage_image = storage_image_mask != 0;
   pipeline.has_storage_buffer = storage_buffer_mask != 0;

   uint32_t sampled_image_mask = 0;
   uint32_t sampled_buffer_mask = 0;
   if (!yttrium_pipeline_build_sampled_resource_masks(
          yctx, cs, MESA_SHADER_COMPUTE, NULL, &sampled_image_mask,
          &sampled_buffer_mask) ||
       (sampled_image_mask | sampled_buffer_mask) != cs->sampler_used_mask) {
      YTTRIUM_WARN("yttrium: compute dispatch skipped owner=yttrium-pipeline reason=sampled_resource_masks_unsupported shader=%p samplers=0x%x images=0x%x buffers=0x%x\n",
                   cs, cs->sampler_used_mask, sampled_image_mask,
                   sampled_buffer_mask);
      return;
   }
   /* Compute sampling is limited to native images. In particular, do not enter
    * the graphics sampled-buffer expansion/upload path with GPU-written data. */
   if (sampled_buffer_mask) {
      YTTRIUM_WARN("yttrium: compute dispatch skipped owner=yttrium-pipeline reason=sampled_buffers_unsupported shader=%p samplers=0x%x buffers=0x%x action=reject-dispatch\n",
                   cs, cs->sampler_used_mask, sampled_buffer_mask);
      return;
   }
   if (sampled_image_mask)
      pipeline.key.sampled_stage_mask = 1u << MESA_SHADER_COMPUTE;
   if (!yttrium_pipeline_add_sampled_layouts(
          yctx, cs, sampled_image_mask, 0, &pipeline.key)) {
      YTTRIUM_WARN("yttrium: compute dispatch skipped sampled layout failed shader=%p samplers=0x%x\n",
                   cs, cs->sampler_used_mask);
      return;
   }

   struct yttrium_venus_sampled_image sampled_images
      [YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   uint32_t sampled_image_count = 0;
   memset(sampled_images, 0, sizeof(sampled_images));
   if (!yttrium_pipeline_collect_sampled_textures(
          ctx, yctx, NULL, &pipeline, sampled_images, &sampled_image_count,
          NULL)) {
      YTTRIUM_WARN("yttrium: compute dispatch skipped sampled collect failed shader=%p samplers=0x%x\n",
                   cs, cs->sampler_used_mask);
      return;
   }

   struct yttrium_venus_storage_image storage_images
      [YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES];
   uint32_t storage_image_count = 0;
   memset(storage_images, 0, sizeof(storage_images));
   if (!yttrium_pipeline_collect_storage_images(ctx, yctx, &pipeline,
                                                MESA_SHADER_COMPUTE,
                                                storage_images,
                                                &storage_image_count)) {
      YTTRIUM_WARN("yttrium: compute dispatch skipped storage collect failed shader=%p image_mask=0x%llx buffer_mask=0x%llx\n",
                   cs, (unsigned long long)storage_image_mask,
                   (unsigned long long)storage_buffer_mask);
      return;
   }

   struct yttrium_venus_ubo_binding_layout ubo_layouts
      [YTTRIUM_VENUS_MAX_PIPELINE_UBO_BINDINGS];
   uint32_t ubo_layout_count = 0;
   memset(ubo_layouts, 0, sizeof(ubo_layouts));
   if (!yttrium_pipeline_add_ubo_layout(cs, VK_SHADER_STAGE_COMPUTE_BIT,
                                        ubo_layouts, &ubo_layout_count)) {
      YTTRIUM_WARN("yttrium: compute dispatch skipped ubo layout failed shader=%p ubo_mask=0x%x\n",
                   cs, cs->ubo_used_mask);
      return;
   }

   struct yttrium_pipeline *cached =
      yttrium_compute_pipeline_get(yctx, cs, &pipeline, ubo_layouts,
                                  ubo_layout_count);
   if (!cached) {
      YTTRIUM_WARN("yttrium: compute dispatch skipped pipeline init failed shader=%p storage_mask=0x%llx buffer_mask=0x%llx ubos=%u\n",
                   cs, (unsigned long long)storage_image_mask,
                   (unsigned long long)storage_buffer_mask,
                   ubo_layout_count);
      return;
   }

   struct yttrium_venus_ubo_upload ubo_uploads
      [YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS];
   uint32_t ubo_upload_count = 0;
   memset(ubo_uploads, 0, sizeof(ubo_uploads));
   if (!yttrium_pipeline_add_ubo_uploads(yctx, cs, ubo_uploads,
                                         &ubo_upload_count)) {
      YTTRIUM_WARN("yttrium: compute dispatch skipped ubo upload collect failed shader=%p ubo_mask=0x%x\n",
                   cs, cs->ubo_used_mask);
      return;
   }

   if (!yttrium_venus_dispatch_compute(screen->venus, cached,
                                       sampled_images, sampled_image_count,
                                       storage_images, storage_image_count,
                                       ubo_uploads, ubo_upload_count,
                                       info->grid[0], info->grid[1],
                                       info->grid[2])) {
      YTTRIUM_WARN("yttrium: compute dispatch failed shader=%p pipeline_id=%llu grid=%ux%ux%u storage_count=%u ubos=%u\n",
                   cs, (unsigned long long)cached->pipeline_obj.id,
                   info->grid[0], info->grid[1], info->grid[2],
                   storage_image_count, ubo_upload_count);
   } else {
      for (uint32_t i = 0; i < storage_image_count; i++) {
         struct yttrium_venus_resource *native = storage_images[i].resource;
         if (!storage_images[i].buffer || !native || !native->owner)
            continue;
         struct yttrium_resource *res = yttrium_resource(native->owner);
         res->gpu_buffer_written = true;
         res->data_dirty = false;
         native->contents_initialized = true;
         res->contents_serial++;
      }
   }

}

enum yttrium_pipeline_draw_result
yttrium_pipeline_try_draw(struct pipe_context *ctx,
                          struct yttrium_resource *dst,
                          const struct pipe_draw_info *info,
                          const struct pipe_draw_indirect_info *indirect,
                          const struct pipe_draw_start_count_bias *draws,
                          unsigned num_draws,
                          const struct yttrium_venus_draw_state *draw_state)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   struct yttrium_pipeline_draw_upload upload;
   memset(&upload, 0, sizeof(upload));
   const uint64_t total_start_us =
      yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;

   const bool draw_auto =
      indirect && indirect->count_from_stream_output && !indirect->buffer &&
      !indirect->indirect_draw_count;
   const bool draw_indirect =
      indirect && indirect->buffer && !indirect->count_from_stream_output &&
      !indirect->indirect_draw_count && indirect->draw_count == 1;

   if (!info || !dst || dst->classic_display ||
       (indirect && !draw_auto && !draw_indirect) ||
       (!draw_auto && (!draws || num_draws != 1)) ||
       (draw_auto && num_draws != 1)) {
      YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped args dst=%p classic=%u indirect=%p draw_auto=%u draws=%p num_draws=%u\n",
                   dst, dst ? dst->classic_display : 0,
                   indirect, draw_auto, draws, num_draws);
      return false;
   }
   if (!draw_auto && !draw_indirect && !draws[0].count) {
      yttrium_trace_debug_stringf(
         "yttrium: shader_draw_probe try_draw no-op draw res_id=%u mode=%u index_size=%u instances=%u start=%u bias=%d",
         dst->venus_res_id,
         info ? info->mode : 0,
         info ? info->index_size : 0,
         info ? info->instance_count : 0,
         draws[0].start,
         draws[0].index_bias);
      return true;
   }

   struct yttrium_resource *args = draw_indirect ?
      yttrium_resource(indirect->buffer) : NULL;
   if (draw_indirect) {
      const uint32_t args_size = info->index_size ?
         sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand);
      if (args->base.target != PIPE_BUFFER || !args->venus.initialized ||
          !args->venus.buffer_backed || !args->venus.buffer ||
          args->direct_bind_unsafe ||
          !(args->venus.buffer_usage & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) ||
          (indirect->offset & 3) || indirect->offset > args->size ||
          args_size > args->size - indirect->offset) {
         YTTRIUM_WARN("yttrium: native indirect draw rejected owner=yttrium_pipeline reason=argument_buffer_not_native_or_invalid res_id=%u offset=%u bytes=%u usage=0x%x\n",
                      args->venus_res_id, indirect->offset, args_size,
                      args->venus.buffer_usage);
         return false;
      }
   }

   const bool depth_enabled =
      yctx->dsa && yctx->dsa->state.depth_enabled;
   const bool depth_bounds_enabled =
      yctx->dsa && yctx->dsa->state.depth_bounds_test;
   const bool stencil_enabled =
      yctx->dsa &&
      (yctx->dsa->state.stencil[0].enabled ||
       yctx->dsa->state.stencil[1].enabled);
   struct yttrium_resource *zs =
      yctx->fb.zsbuf.texture ? yttrium_resource(yctx->fb.zsbuf.texture) :
      NULL;

   if (depth_bounds_enabled && zs) {
      YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped depth bounds unsupported zsbuf=%p\n",
                   yctx->fb.zsbuf.texture);
      return false;
   }
   if (stencil_enabled && zs &&
       !yttrium_pipeline_format_has_stencil(zs->venus.vk_format)) {
      YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped stencil attachment format unsupported stencil0=%u stencil1=%u zsbuf=%p format=%u\n",
                   yctx->dsa ? yctx->dsa->state.stencil[0].enabled : 0,
                   yctx->dsa ? yctx->dsa->state.stencil[1].enabled : 0,
                   yctx->fb.zsbuf.texture,
                   zs->venus.vk_format);
      return false;
   }
   if (!depth_enabled && !stencil_enabled)
      zs = NULL;
   if (zs &&
       (!zs->venus.initialized || zs->venus.buffer_backed ||
        !zs->venus.image ||
        !(zs->venus.image_usage &
          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))) {
      YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped depth attachment not Venus-backed zs=%p initialized=%u buffer_backed=%u image=0x%llx usage=0x%x format=%u\n",
                   zs,
                   zs ? zs->venus.initialized : 0,
                   zs ? zs->venus.buffer_backed : 0,
                   (unsigned long long)(zs ?
                      yttrium_pipeline_vk_image_to_u64(zs->venus.image) : 0),
                   zs ? zs->venus.image_usage : 0,
                   zs ? zs->venus.vk_format : VK_FORMAT_UNDEFINED);
      return false;
   }

   struct yttrium_venus_draw_state interlock_draw_state;
   const struct yttrium_venus_draw_state *native_draw_state = draw_state;
   if (yttrium_pipeline_can_use_forced_sample_interlock(
          yctx, dst, zs, draw_state)) {
      interlock_draw_state = *draw_state;
      interlock_draw_state.forced_sample_interlock = VK_TRUE;
      interlock_draw_state.rt_count = 0;
      interlock_draw_state.blend_enable = VK_FALSE;
      interlock_draw_state.logic_op_enable = VK_FALSE;
      interlock_draw_state.color_write_mask = 0;
      for (uint32_t i = 0; i < PIPE_MAX_COLOR_BUFS; i++) {
         interlock_draw_state.rt_blend_enable[i] = VK_FALSE;
         interlock_draw_state.rt_color_write_mask[i] = 0;
      }
      native_draw_state = &interlock_draw_state;
   }

   uint64_t stage_start_us =
      yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;
   struct yttrium_pipeline *pipeline =
      yttrium_pipeline_get(yctx, dst, zs, native_draw_state);
   yttrium_pipeline_trace_timing(
      YTTRIUM_TRACE_TIMING_PIPELINE_GET,
      pipeline ? 0 : 1, stage_start_us, NULL,
      dst->venus_res_id,
      pipeline ? pipeline->pipeline_obj.id : 0,
      draw_auto, num_draws);
   if (!pipeline) {
      YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped no pipeline res_id=%u\n",
                   dst->venus_res_id);
      return false;
   }

   stage_start_us =
      yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;
   if (draw_indirect) {
      if (!yttrium_pipeline_get_indirect_draw_upload(yctx, info, draws, &upload)) {
         YTTRIUM_WARN("yttrium: native indirect draw rejected owner=yttrium_pipeline reason=non_native_geometry\n");
         return false;
      }
   } else if (draw_auto) {
      if (!yttrium_venus_transform_feedback_draw_enabled(screen->venus)) {
         YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped DrawAuto transform feedback draw unavailable\n");
         yttrium_pipeline_trace_timing(
            YTTRIUM_TRACE_TIMING_PIPELINE_DRAW_UPLOAD,
            1, stage_start_us, NULL, dst->venus_res_id,
            pipeline->pipeline_obj.id, draw_auto, num_draws);
         return false;
      }
      if (!yttrium_pipeline_get_draw_auto_upload(yctx, info, &upload)) {
         YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped DrawAuto vertex source mode=%u index_size=%u instances=%u bindings=%u\n",
                      info ? info->mode : 0,
                      info ? info->index_size : 0,
                      info ? info->instance_count : 0,
                      yctx->vertex_elements ?
                         yctx->vertex_elements->num_bindings : 0);
         yttrium_pipeline_trace_timing(
            YTTRIUM_TRACE_TIMING_PIPELINE_DRAW_UPLOAD,
            1, stage_start_us, NULL, dst->venus_res_id,
            pipeline->pipeline_obj.id, draw_auto, num_draws);
         return false;
      }
   } else if (!yttrium_pipeline_get_draw_upload(yctx, info, &draws[0],
                                                &upload)) {
      const unsigned source_vb =
         yctx->vertex_elements && yctx->vertex_elements->num_bindings ?
         yctx->vertex_elements->binding_map[0] : 0xffffffffu;
      const VkVertexInputBindingDescription *binding =
         yctx->vertex_elements && yctx->vertex_elements->num_bindings ?
         &yctx->vertex_elements->bindings[0] : NULL;
      const struct pipe_vertex_buffer *vb =
         source_vb < yctx->num_vertex_buffers ?
         &yctx->vertex_buffers[source_vb] : NULL;
      const struct yttrium_resource *vb_res =
         vb && !vb->is_user_buffer && vb->buffer.resource ?
         yttrium_resource(vb->buffer.resource) : NULL;
      const struct yttrium_resource *index_res =
         info && info->index_size && !info->has_user_indices &&
         info->index.resource ? yttrium_resource(info->index.resource) : NULL;
      YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped vertex source mode=%u index_size=%u restart=%u instances=%u draw_count=%u start=%u bias=%d bounds=%u min=%u max=%u num_draws=%u bindings=%u source_vb=%u num_vb=%u stride=%u vb_user=%u vb_res=%p vb_data=%p vb_offset=%u index_user=%u index_res=%p index_data=%p\n",
                   info ? info->mode : 0,
                   info ? info->index_size : 0,
                   info ? info->primitive_restart : 0,
                   info ? info->instance_count : 0,
                   draws[0].count, draws[0].start, draws[0].index_bias,
                   info ? info->index_bounds_valid : 0,
                   info ? info->min_index : 0,
                   info ? info->max_index : 0,
                   num_draws,
                   yctx->vertex_elements ?
                      yctx->vertex_elements->num_bindings : 0,
                   source_vb, yctx->num_vertex_buffers,
                   binding ? binding->stride : 0,
                   vb ? vb->is_user_buffer : 0,
                   vb_res, vb_res ? vb_res->data : NULL,
                   vb ? vb->buffer_offset : 0,
                   info ? info->has_user_indices : 0,
                   index_res, index_res ? index_res->data : NULL);
      yttrium_pipeline_trace_timing(
         YTTRIUM_TRACE_TIMING_PIPELINE_DRAW_UPLOAD,
         1, stage_start_us, NULL, dst->venus_res_id,
         pipeline->pipeline_obj.id, draw_auto, num_draws);
      return false;
   }
   yttrium_pipeline_trace_timing(
      YTTRIUM_TRACE_TIMING_PIPELINE_DRAW_UPLOAD,
      0, stage_start_us, NULL, dst->venus_res_id,
      pipeline->pipeline_obj.id, upload.vertex_upload_count,
      upload.index_count);

   if (draw_auto) {
      yttrium_trace_debug_stringf(
         "yttrium: shader_draw_probe draw upload DrawAuto mode=%u instances=%u vertex_uploads=%u",
         info ? info->mode : 0,
         upload.instance_count,
         upload.vertex_upload_count);
   } else if (!draw_indirect) {
      yttrium_pipeline_trace_draw_upload(yctx, info, &draws[0], &upload);
   }

   struct yttrium_venus_ubo_upload ubo_uploads
      [YTTRIUM_VENUS_MAX_PIPELINE_UBO_SLOTS];
   uint32_t ubo_upload_count = 0;
   stage_start_us =
      yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;
   if (!yttrium_pipeline_collect_ubo_uploads(yctx, ubo_uploads,
                                              &ubo_upload_count)) {
      YTTRIUM_WARN("yttrium: shader_draw_probe try_draw skipped ubo upload collect vs_ubo=0x%x tcs_ubo=0x%x tes_ubo=0x%x gs_ubo=0x%x fs_ubo=0x%x\n",
                   pipeline->key.vs_ubo_used_mask,
                   pipeline->key.tcs_ubo_used_mask,
                   pipeline->key.tes_ubo_used_mask,
                   pipeline->key.gs_ubo_used_mask,
                   pipeline->key.fs_ubo_used_mask);
      yttrium_pipeline_trace_timing(
         YTTRIUM_TRACE_TIMING_PIPELINE_UBO_UPLOADS,
         1, stage_start_us, NULL, dst->venus_res_id,
         pipeline->pipeline_obj.id,
         pipeline->key.vs_ubo_used_mask,
         pipeline->key.tcs_ubo_used_mask |
            pipeline->key.tes_ubo_used_mask |
            pipeline->key.gs_ubo_used_mask |
            pipeline->key.fs_ubo_used_mask);
      yttrium_pipeline_draw_upload_cleanup(&upload);
      return false;
   }
   yttrium_pipeline_trace_timing(
      YTTRIUM_TRACE_TIMING_PIPELINE_UBO_UPLOADS,
      0, stage_start_us, NULL, dst->venus_res_id,
      pipeline->pipeline_obj.id, ubo_upload_count,
      pipeline->ubo_count);

   struct yttrium_venus_sampled_image sampled_images
      [YTTRIUM_VENUS_MAX_PIPELINE_SAMPLED_IMAGES];
   struct yttrium_pipeline_owned_sampled_buffers sampled_owned_buffers;
   sampled_owned_buffers.count = 0;
   uint32_t sampled_image_count = 0;
   stage_start_us =
      yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;
   if (!yttrium_pipeline_collect_sampled_textures(ctx, yctx, dst,
                                                  pipeline,
                                                  sampled_images,
                                                  &sampled_image_count,
                                                  &sampled_owned_buffers)) {
      yttrium_pipeline_cleanup_sampled_buffers(&sampled_owned_buffers);
      yttrium_pipeline_draw_upload_cleanup(&upload);
      yttrium_pipeline_trace_timing(
         YTTRIUM_TRACE_TIMING_PIPELINE_SAMPLED_TEXTURES,
         1, stage_start_us, NULL, dst->venus_res_id,
         pipeline->pipeline_obj.id,
         pipeline->key.sampled_binding_count,
         pipeline->key.sampled_stage_mask);
      return false;
   }
   yttrium_pipeline_trace_timing(
      YTTRIUM_TRACE_TIMING_PIPELINE_SAMPLED_TEXTURES,
      0, stage_start_us, NULL, dst->venus_res_id,
      pipeline->pipeline_obj.id, sampled_image_count,
      pipeline->key.sampled_stage_mask);

   struct yttrium_venus_storage_image storage_images
      [YTTRIUM_VENUS_MAX_PIPELINE_STORAGE_IMAGES];
   uint32_t storage_image_count = 0;
   mesa_shader_stage storage_stage =
      yttrium_pipeline_first_graphics_stage(pipeline->key.storage_stage_mask);
   if (storage_stage == MESA_SHADER_NONE)
      storage_stage = MESA_SHADER_FRAGMENT;
   bool storage_images_valid = false;
   if (pipeline->key.forced_sample_interlock) {
      if (dst->venus.initialized && !dst->venus.buffer_backed &&
          dst->venus.image &&
          (dst->venus.image_usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
         storage_images[0] = (struct yttrium_venus_storage_image) {
            .resource = &dst->venus,
            .resource_id = dst->venus_res_id,
            .binding = yttrium_shader_storage_image_binding(
               YTTRIUM_FORCED_SAMPLE_INTERLOCK_IMAGE_SLOT),
            .view_type = VK_IMAGE_VIEW_TYPE_2D,
            .aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT,
            .first_level = pipeline->key.rt_level[0],
            .level_count = 1,
            .first_layer = pipeline->key.rt_layer[0],
            .layer_count = pipeline->key.rt_layers,
            .format = PIPE_FORMAT_R16_UINT,
         };
         storage_image_count = 1;
         storage_images_valid = true;
      }
   } else {
      storage_images_valid =
         yttrium_pipeline_collect_storage_images(ctx, yctx, pipeline,
                                                 storage_stage,
                                                 storage_images,
                                                 &storage_image_count);
   }
   if (!storage_images_valid) {
      yttrium_pipeline_cleanup_sampled_buffers(&sampled_owned_buffers);
      yttrium_pipeline_draw_upload_cleanup(&upload);
      return false;
   }

   struct yttrium_venus_stream_output_target so_targets
      [YTTRIUM_VENUS_MAX_STREAM_OUTPUT_TARGETS];
   uint32_t so_target_count = 0;
   memset(so_targets, 0, sizeof(so_targets));
   stage_start_us =
      yttrium_trace_is_enabled() ? yttrium_trace_now_us() : 0;
   if (!yttrium_pipeline_collect_stream_output_targets(ctx, yctx,
                                                       so_targets,
                                                       &so_target_count)) {
      yttrium_pipeline_cleanup_sampled_buffers(&sampled_owned_buffers);
      yttrium_pipeline_draw_upload_cleanup(&upload);
      yttrium_pipeline_trace_timing(
         YTTRIUM_TRACE_TIMING_PIPELINE_SO_TARGETS,
         1, stage_start_us, NULL, dst->venus_res_id,
         pipeline->pipeline_obj.id, so_target_count, draw_auto);
      return false;
   }
   yttrium_pipeline_trace_timing(
      YTTRIUM_TRACE_TIMING_PIPELINE_SO_TARGETS,
      0, stage_start_us, NULL, dst->venus_res_id,
      pipeline->pipeline_obj.id, so_target_count, draw_auto);

   struct yttrium_venus_stream_output_target draw_auto_target;
   struct yttrium_venus_stream_output_target *draw_auto_target_ptr = NULL;
   uint32_t draw_auto_target_count = 0;
   uint32_t draw_auto_stride = 0;
   memset(&draw_auto_target, 0, sizeof(draw_auto_target));
   if (draw_auto) {
      struct yttrium_stream_output_target *ytarget =
         (struct yttrium_stream_output_target *)
            indirect->count_from_stream_output;

      if (!ytarget || !ytarget->counter_buffer_valid) {
         yttrium_trace_debug_stringf(
            "yttrium: shader_draw_probe DrawAuto no-op invalid counter target=%p valid=%u",
            ytarget, ytarget ? ytarget->counter_buffer_valid : 0);
         yttrium_pipeline_cleanup_sampled_buffers(&sampled_owned_buffers);
         yttrium_pipeline_draw_upload_cleanup(&upload);
         return true;
      }

      if (!yttrium_pipeline_collect_stream_output_target(
             ctx, indirect->count_from_stream_output,
             ytarget->output_buffer, &draw_auto_target,
             &draw_auto_target_count) || draw_auto_target_count != 1) {
         yttrium_pipeline_cleanup_sampled_buffers(&sampled_owned_buffers);
         yttrium_pipeline_draw_upload_cleanup(&upload);
         return false;
      }

      draw_auto_stride =
         yttrium_pipeline_stream_output_stride(yctx, ytarget->output_buffer);
      if (!draw_auto_stride) {
         YTTRIUM_WARN("yttrium: shader_draw_probe DrawAuto skipped missing stream-output stride buffer=%u\n",
                      ytarget->output_buffer);
         yttrium_pipeline_cleanup_sampled_buffers(&sampled_owned_buffers);
         yttrium_pipeline_draw_upload_cleanup(&upload);
         return false;
      }
      draw_auto_target_ptr = &draw_auto_target;
   }

   struct yttrium_venus_draw_state push_draw_state = *native_draw_state;
   push_draw_state.indirect_resource = args ? &args->venus : NULL;
   push_draw_state.indirect_offset = draw_indirect ? indirect->offset : 0;
   push_draw_state.push_constant_vs_size = 0;
   push_draw_state.push_constant_fs_size = 0;
   memset(push_draw_state.push_constant_data, 0,
          sizeof(push_draw_state.push_constant_data));
   if (yttrium_gdi_static_ubo_sampled_cache_enabled() &&
       !yttrium_pipeline_collect_push_constants(yctx, &push_draw_state)) {
      yttrium_pipeline_cleanup_sampled_buffers(&sampled_owned_buffers);
      yttrium_pipeline_draw_upload_cleanup(&upload);
      return false;
   }
   native_draw_state = &push_draw_state;

   yttrium_pipeline_trace_draw_state(dst, zs, native_draw_state);

   struct yttrium_venus_resource *color_resources[PIPE_MAX_COLOR_BUFS];
   uint32_t color_resource_ids[PIPE_MAX_COLOR_BUFS];
   memset(color_resources, 0, sizeof(color_resources));
   memset(color_resource_ids, 0, sizeof(color_resource_ids));
   for (uint32_t i = 0; i < pipeline->key.rt_count; i++) {
      struct yttrium_resource *rt =
         pipeline->rt_resources[i] ?
         yttrium_resource(pipeline->rt_resources[i]) : NULL;

      color_resources[i] = rt ? &rt->venus : NULL;
      color_resource_ids[i] = rt ? rt->venus_res_id : 0;
   }

   const bool emitted =
      yttrium_venus_draw_pipeline(screen->venus, &dst->venus,
                                   dst->venus_res_id,
                                   color_resources, color_resource_ids,
                                   pipeline->key.rt_count,
                                   zs ? &zs->venus : NULL,
                                   zs ? zs->venus_res_id : 0,
                                   pipeline,
                                   sampled_image_count ? sampled_images : NULL,
                                   sampled_image_count,
                                   storage_image_count ? storage_images : NULL,
                                   storage_image_count,
                                   upload.vertex_uploads,
                                   upload.vertex_upload_count,
                                   upload.vertex_count,
                                   upload.instance_count,
                                   upload.index_data,
                                   upload.index_data_size,
                                   upload.index_resource,
                                   upload.index_resource_id,
                                   upload.index_buffer_offset,
                                   upload.index_count,
                                   upload.index_type,
                                   upload.index_host_write_pending,
                                   upload.vertex_offset,
                                   ubo_uploads,
                                   ubo_upload_count,
                                   so_target_count ? so_targets : NULL,
                                   so_target_count,
                                   draw_auto_target_ptr,
                                   draw_auto_stride, native_draw_state);
   yttrium_pipeline_cleanup_sampled_buffers(&sampled_owned_buffers);

   if (emitted) {
      for (uint32_t i = 0; i < sampled_image_count; i++) {
         if (sampled_images[i].buffer && sampled_images[i].buffer_data &&
             sampled_images[i].pipe_resource)
            yttrium_resource(sampled_images[i].pipe_resource)->data_dirty = false;
      }
      for (uint32_t i = 0; i < storage_image_count; i++) {
         struct yttrium_venus_resource *native = storage_images[i].resource;
         if (!storage_images[i].buffer || !native || !native->owner)
            continue;
         struct yttrium_resource *res = yttrium_resource(native->owner);
         res->gpu_buffer_written = true;
         res->data_dirty = false;
         native->contents_initialized = true;
         res->contents_serial++;
      }
      if (args) {
         args->data_dirty = false;
         yttrium_trace_debug_stringf(
            "yttrium: native indirect draw emitted res_id=%u args_res_id=%u args_buffer_id=%llu args_offset=%u indexed=%u count_source=gpu",
            dst->venus_res_id, args->venus_res_id,
            (unsigned long long)args->venus.buffer_obj.id,
            indirect->offset, info->index_size != 0);
      }
      yttrium_pipeline_mark_vertex_uploads_clean(&upload);
      yttrium_pipeline_mark_index_upload_clean(&upload);
      for (uint32_t i = 0; i < so_target_count; i++) {
         struct yttrium_stream_output_target *ytarget =
            (struct yttrium_stream_output_target *)yctx->so_targets[i];
         if (ytarget) {
            ytarget->counter_buffer_valid =
               so_targets[i].counter_buffer_valid;
            struct yttrium_resource *res =
               yttrium_resource(ytarget->base.buffer);
            /* A later readback must not republish stale CPU bytes over SO. */
            res->gpu_buffer_written = true;
            res->data_dirty = false;
            res->venus.contents_initialized = true;
            res->contents_serial++;
         }
      }
      yttrium_trace_debug_stringf(
         "yttrium: shader_draw_probe try_draw native emit result=%u res_id=%u sampled_count=%u so_targets=%u pipeline_id=%llu vertex_count=%u instances=%u vertex_bytes=0x%llx vertex_bindings=%u indexed=%u index_count=%u index_bytes=0x%llx index_type=%u vertex_offset=%d ubos=%u",
         emitted, dst->venus_res_id,
         sampled_image_count,
         so_target_count,
         (unsigned long long)pipeline->pipeline_obj.id,
         upload.vertex_count,
         upload.instance_count,
         (unsigned long long)upload.vertex_data_size,
         upload.vertex_upload_count,
         upload.index_count != 0,
         upload.index_count,
         (unsigned long long)upload.index_data_size,
         upload.index_type,
         upload.vertex_offset,
         ubo_upload_count);
   } else {
      yttrium_trace_debug_stringf(
         "yttrium: shader_draw_probe try_draw native emit result=%u res_id=%u sampled_count=%u so_targets=%u pipeline_id=%llu vertex_count=%u instances=%u vertex_bytes=0x%llx vertex_bindings=%u indexed=%u index_count=%u index_bytes=0x%llx index_type=%u vertex_offset=%d ubos=%u",
         emitted, dst->venus_res_id,
         sampled_image_count,
         so_target_count,
         (unsigned long long)pipeline->pipeline_obj.id,
         upload.vertex_count,
         upload.instance_count,
         (unsigned long long)upload.vertex_data_size,
         upload.vertex_upload_count,
         upload.index_count != 0,
         upload.index_count,
         (unsigned long long)upload.index_data_size,
         upload.index_type,
         upload.vertex_offset,
         ubo_upload_count);
   }
   yttrium_pipeline_draw_upload_cleanup(&upload);
   yttrium_pipeline_trace_timing(
      YTTRIUM_TRACE_TIMING_PIPELINE_DRAW_TOTAL,
      emitted ? 0 : 1, total_start_us, NULL, dst->venus_res_id,
      pipeline->pipeline_obj.id, upload.vertex_count,
      sampled_image_count);
   return emitted ? YTTRIUM_PIPELINE_DRAW_EMITTED :
                    YTTRIUM_PIPELINE_DRAW_EMIT_FAILED;
}
