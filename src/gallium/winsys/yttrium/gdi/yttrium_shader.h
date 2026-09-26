/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#ifndef YTTRIUM_SHADER_H
#define YTTRIUM_SHADER_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "compiler/shader_enums.h"
#include "pipe/p_context.h"
#include "pipe/p_state.h"
#include "tgsi/tgsi_scan.h"
#include <vulkan/vulkan.h>

#include "yttrium_venus.h"

#ifdef __cplusplus
extern "C" {
#endif

struct nir_shader;
struct spirv_shader;
struct yttrium_screen;

#define YTTRIUM_SHADER_UBO_SET 0
#define YTTRIUM_SHADER_SAMPLED_IMAGE_BINDING_BASE \
   (PIPE_MAX_CONSTANT_BUFFERS * 6)
#define YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE \
   (YTTRIUM_SHADER_SAMPLED_IMAGE_BINDING_BASE + \
    YTTRIUM_VENUS_SAMPLED_STAGE_COUNT * PIPE_MAX_SAMPLERS)
#define YTTRIUM_SHADER_MAX_UBO_BYTES (64 * 1024)
#define YTTRIUM_SHADER_MAX_UBO_DWORDS (YTTRIUM_SHADER_MAX_UBO_BYTES / 4)
#define YTTRIUM_SHADER_MAX_PUSH_CONSTANT_WORDS \
   (YTTRIUM_SHADER_FS_PUSH_CONSTANT_BYTES / sizeof(uint32_t))

struct yttrium_shader_state {
   uint32_t id;
   mesa_shader_stage stage;
   enum pipe_shader_ir type;
   struct tgsi_token *tokens;
   unsigned token_count;
   uint64_t token_hash;
   struct nir_shader *nir;
   unsigned static_shared_mem;
   struct spirv_shader *spirv;
   unsigned spirv_word_count;
   uint64_t spirv_hash;
   struct yttrium_venus_object module_obj;
   VkShaderModule module;
   /*
    * Whether the fragment shader forces sample-rate shading.  Answering it
    * means walking every instruction of the NIR, and the pipeline key needs it
    * on every draw, so memoise it: -1 not yet computed, 0 no, 1 yes.  A
    * shader's code does not change once it exists.
    */
   int8_t uses_sample_shading;

   bool resource_free;
   bool uniform_buffer_only;
   bool sampled_texture_only;
   bool storage_image_only;
   bool sampled_storage_image_only;
   bool vs_stream_output_gs;
   struct yttrium_shader_state *vs_stream_output_variant;
   uint32_t vs_stream_output_base_id;
   bool ubo_default;
   uint8_t ubo_first;
   uint8_t ubo_count;
   uint32_t ubo_used_mask;
   uint32_t push_ubo_mask;
   uint16_t push_constant_offset;
   uint8_t push_constant_word_count;
   uint8_t push_constant_source_slots[YTTRIUM_SHADER_MAX_PUSH_CONSTANT_WORDS];
   uint16_t push_constant_source_words[YTTRIUM_SHADER_MAX_PUSH_CONSTANT_WORDS];
   uint32_t sampler_used_mask;
   uint64_t image_used_mask;
   uint32_t sampler_binding_valid_mask;
   uint32_t sampler_mixed_return_mask;
   uint16_t sampler_view_index[PIPE_MAX_SAMPLERS];
   uint8_t sampler_state_index[PIPE_MAX_SAMPLERS];
   struct pipe_stream_output_info stream_output;
   struct tgsi_shader_info info;
   /* Typed texture UAVs take their channel layout from the bound view. */
   /* Storage-image slots the shader declares as arrayed.  The Vulkan view
    * type has to agree with the SPIR-V OpTypeImage Arrayed flag, and D3D
    * gives a one-element array the same resource shape as a non-array, so
    * the resource alone cannot tell them apart. */
   uint64_t image_array_mask;
   uint64_t formatless_image_mask;
   uint64_t formatless_image_read_mask;
   uint64_t formatless_image_write_mask;
   uint64_t formatless_image_uint_mask;
   uint64_t formatless_image_sint_mask;
};

const char *
yttrium_shader_stage_name(mesa_shader_stage stage);

bool
yttrium_shader_pipeline_enabled(void);

bool
yttrium_shader_caps_enabled(void);

bool
yttrium_shader_compile_enabled(void);

bool
yttrium_shader_module_enabled(void);

bool
yttrium_shader_pipeline_draw_enabled(void);

struct yttrium_shader_state *
yttrium_shader_state_create(struct pipe_context *ctx,
                            const struct pipe_shader_state *state,
                            mesa_shader_stage stage);

void
yttrium_shader_state_log_bind(const struct yttrium_shader_state *shader,
                              mesa_shader_stage stage);

void
yttrium_shader_state_destroy(struct yttrium_screen *screen,
                             struct yttrium_shader_state *shader);

bool
yttrium_shader_state_has_module(const struct yttrium_shader_state *shader);

bool
yttrium_shader_state_is_resource_free(const struct yttrium_shader_state *shader);

bool
yttrium_shader_state_is_uniform_buffer_only(
   const struct yttrium_shader_state *shader);

bool
yttrium_shader_state_is_sampled_texture_only(
   const struct yttrium_shader_state *shader);

bool
yttrium_shader_state_is_storage_image_only(
   const struct yttrium_shader_state *shader);

bool
yttrium_shader_state_is_sampled_storage_image_only(
   const struct yttrium_shader_state *shader);

struct yttrium_shader_state *
yttrium_shader_create_cull_distance_vs(struct pipe_context *ctx,
                                       const struct yttrium_shader_state *vs,
                                       unsigned *cull_distance_slot);

struct yttrium_shader_state *
yttrium_shader_create_cull_distance_gs(struct pipe_context *ctx,
                                       const struct yttrium_shader_state *vs,
                                       enum mesa_prim primitive_type,
                                       bool passthrough_prim_id,
                                       unsigned cull_distance_slot);

struct yttrium_shader_state *
yttrium_shader_create_a8_rt_fs(struct pipe_context *ctx,
                               const struct yttrium_shader_state *fs,
                               uint32_t a8_rt_mask);

struct yttrium_shader_state *
yttrium_shader_create_alpha_test_fs(struct pipe_context *ctx,
                                    const struct yttrium_shader_state *fs,
                                    uint32_t alpha_func,
                                    float alpha_ref_value);

struct yttrium_shader_state *
yttrium_shader_create_dual_source_fs(struct pipe_context *ctx,
                                     const struct yttrium_shader_state *fs);

bool
yttrium_shader_state_uses_sample_mask_in(
   const struct yttrium_shader_state *shader);

struct yttrium_shader_state *
yttrium_shader_create_forced_sample_mask_fs(
   struct pipe_context *ctx,
   const struct yttrium_shader_state *fs,
   unsigned sample_count);

struct yttrium_shader_state *
yttrium_shader_create_sample_mask_expand_fs(
   struct pipe_context *ctx,
   const struct yttrium_shader_state *fs,
   unsigned hw_sample_count,
   unsigned app_sample_count);

struct yttrium_shader_state *
yttrium_shader_create_forced_sample_interlock_fs(
   struct pipe_context *ctx,
   const struct yttrium_shader_state *fs,
   unsigned image_slot);

uint32_t
yttrium_shader_state_sampler_used_mask(
   const struct yttrium_shader_state *shader);

uint64_t
yttrium_shader_state_image_used_mask(
   const struct yttrium_shader_state *shader);

unsigned
yttrium_shader_state_sampler_view_index(
   const struct yttrium_shader_state *shader,
   unsigned sampler_slot);

unsigned
yttrium_shader_state_sampler_state_index(
   const struct yttrium_shader_state *shader,
   unsigned sampler_slot);

bool
yttrium_shader_state_has_explicit_sampler_binding(
   const struct yttrium_shader_state *shader,
   unsigned sampler_slot);

uint32_t
yttrium_shader_ubo_default_binding(mesa_shader_stage stage);

uint32_t
yttrium_shader_ubo_array_binding(mesa_shader_stage stage);

uint32_t
yttrium_shader_ubo_binding(mesa_shader_stage stage, unsigned raw_index);

uint32_t
yttrium_shader_sampler_binding(mesa_shader_stage stage, unsigned raw_index);

bool
yttrium_shader_sampler_binding_decode(uint32_t binding,
                                      mesa_shader_stage *stage,
                                      unsigned *raw_index);

uint32_t
yttrium_shader_storage_image_binding(unsigned raw_index);

void
yttrium_screen_init_shader_compiler(struct yttrium_screen *screen);

void
yttrium_finalize_nir(struct pipe_screen *screen,
                     struct nir_shader *nir,
                     bool optimize);

void *
yttrium_create_vs_state(struct pipe_context *ctx,
                        const struct pipe_shader_state *state);

void *
yttrium_create_fs_state(struct pipe_context *ctx,
                        const struct pipe_shader_state *state);

void *
yttrium_create_gs_state(struct pipe_context *ctx,
                        const struct pipe_shader_state *state);

void *
yttrium_create_tcs_state(struct pipe_context *ctx,
                         const struct pipe_shader_state *state);

void *
yttrium_create_tes_state(struct pipe_context *ctx,
                         const struct pipe_shader_state *state);

void *
yttrium_create_compute_state(struct pipe_context *ctx,
                             const struct pipe_compute_state *state);

void
yttrium_bind_vs_state(struct pipe_context *ctx, void *state);

void
yttrium_bind_fs_state(struct pipe_context *ctx, void *state);

void
yttrium_bind_gs_state(struct pipe_context *ctx, void *state);

void
yttrium_bind_tcs_state(struct pipe_context *ctx, void *state);

void
yttrium_bind_tes_state(struct pipe_context *ctx, void *state);

void
yttrium_bind_compute_state(struct pipe_context *ctx, void *state);

void
yttrium_delete_vs_state(struct pipe_context *ctx, void *state);

void
yttrium_delete_fs_state(struct pipe_context *ctx, void *state);

void
yttrium_delete_gs_state(struct pipe_context *ctx, void *state);

void
yttrium_delete_tcs_state(struct pipe_context *ctx, void *state);

void
yttrium_delete_tes_state(struct pipe_context *ctx, void *state);

void
yttrium_delete_compute_state(struct pipe_context *ctx, void *state);

void
yttrium_dump_capture_shader(const struct yttrium_shader_state *shader,
                            const char *path);

void
yttrium_write_shader_capture_metadata(FILE *file,
                                      const struct yttrium_shader_state *shader);

#ifdef __cplusplus
}
#endif

#endif /* YTTRIUM_SHADER_H */
