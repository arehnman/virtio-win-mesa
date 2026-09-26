/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#include <windows.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "compiler/spirv/spirv.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "compiler/nir/nir_deref.h"
#include "compiler/nir/nir_xfb_info.h"
#include "nir/nir_to_tgsi_info.h"
#include "nir/tgsi_to_nir.h"
#include "tgsi/tgsi_dump.h"
#include "tgsi/tgsi_info.h"
#include "tgsi/tgsi_parse.h"
#include "tgsi/tgsi_strings.h"
#include "tgsi/tgsi_transform.h"
#include "util/bitset.h"
#include "util/u_memory.h"
#include "util/u_math.h"

#include "nir_to_spirv.h"

#include "yttrium_shader.h"
#include "yttrium_internal.h"
#include "yttrium_options.h"
#include "yttrium_pipeline.h"
#include "yttrium_trace.h"

static volatile LONG yttrium_shader_sequence;

#define YTTRIUM_SHADER_DUMP_DIR "C:\\ProgramData\\Yttrium\\Shaders"

struct yttrium_spirv_probe {
   bool well_formed;
   bool has_memory_model;
   bool has_stage_entry;
   const char *failure;
   uint32_t version;
   uint32_t generator;
   uint32_t bound;
   uint32_t reserved;
   uint32_t instructions;
   uint32_t capabilities;
   uint32_t entry_points;
   uint32_t execution_modes;
   uint32_t location_decorations;
   uint32_t memory_addressing_model;
   uint32_t memory_model;
   uint32_t first_execution_model;
   uint32_t expected_execution_model;
   uint32_t input_variables;
   uint32_t output_variables;
   uint32_t uniform_variables;
   uint32_t uniform_constant_variables;
   uint32_t push_constant_variables;
   uint32_t private_variables;
   uint32_t bad_word_offset;
   uint32_t bad_word_count;
   uint32_t bad_opcode;
};

const char *
yttrium_shader_stage_name(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return "vs";
   case MESA_SHADER_FRAGMENT:
      return "fs";
   case MESA_SHADER_GEOMETRY:
      return "gs";
   case MESA_SHADER_TESS_CTRL:
      return "tcs";
   case MESA_SHADER_TESS_EVAL:
      return "tes";
   case MESA_SHADER_COMPUTE:
      return "cs";
   default:
      return "unknown";
   }
}

uint32_t
yttrium_shader_ubo_binding(mesa_shader_stage stage, unsigned raw_index)
{
   if (raw_index >= PIPE_MAX_CONSTANT_BUFFERS)
      return UINT32_MAX;

   switch (stage) {
   case MESA_SHADER_VERTEX:
      return raw_index;
   case MESA_SHADER_FRAGMENT:
      return PIPE_MAX_CONSTANT_BUFFERS + raw_index;
   case MESA_SHADER_GEOMETRY:
      return PIPE_MAX_CONSTANT_BUFFERS * 2 + raw_index;
   case MESA_SHADER_TESS_CTRL:
      return PIPE_MAX_CONSTANT_BUFFERS * 4 + raw_index;
   case MESA_SHADER_TESS_EVAL:
      return PIPE_MAX_CONSTANT_BUFFERS * 5 + raw_index;
   case MESA_SHADER_COMPUTE:
      return PIPE_MAX_CONSTANT_BUFFERS * 3 + raw_index;
   default:
      return UINT32_MAX;
   }
}

uint32_t
yttrium_shader_sampler_binding(mesa_shader_stage stage, unsigned raw_index)
{
   if (stage < MESA_SHADER_VERTEX ||
       stage >= YTTRIUM_VENUS_SAMPLED_STAGE_COUNT ||
       raw_index >= PIPE_MAX_SAMPLERS)
      return UINT32_MAX;

   return YTTRIUM_SHADER_SAMPLED_IMAGE_BINDING_BASE +
          (uint32_t)stage * PIPE_MAX_SAMPLERS + raw_index;
}

bool
yttrium_shader_sampler_binding_decode(uint32_t binding,
                                      mesa_shader_stage *stage,
                                      unsigned *raw_index)
{
   if (binding < YTTRIUM_SHADER_SAMPLED_IMAGE_BINDING_BASE ||
       binding >= YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE)
      return false;

   const uint32_t offset =
      binding - YTTRIUM_SHADER_SAMPLED_IMAGE_BINDING_BASE;
   const uint32_t decoded_stage = offset / PIPE_MAX_SAMPLERS;
   if (decoded_stage >= YTTRIUM_VENUS_SAMPLED_STAGE_COUNT)
      return false;

   if (stage)
      *stage = (mesa_shader_stage)decoded_stage;
   if (raw_index)
      *raw_index = offset % PIPE_MAX_SAMPLERS;
   return true;
}

uint32_t
yttrium_shader_storage_image_binding(unsigned raw_index)
{
   if (raw_index >= PIPE_MAX_SHADER_IMAGES)
      return UINT32_MAX;

   return YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE + raw_index;
}

uint32_t
yttrium_shader_ubo_default_binding(mesa_shader_stage stage)
{
   return yttrium_shader_ubo_binding(stage, 0);
}

uint32_t
yttrium_shader_ubo_array_binding(mesa_shader_stage stage)
{
   return yttrium_shader_ubo_binding(stage, 1);
}

static uint32_t
yttrium_shader_info_sampled_texture_mask(const struct tgsi_shader_info *info)
{
   if (!info)
      return 0;

   return info->samplers_declared |
          info->file_mask[TGSI_FILE_SAMPLER_VIEW];
}

static uint32_t
yttrium_nir_sampled_texture_mask32(const struct nir_shader *nir)
{
   uint32_t mask = 0;

   if (!nir)
      return 0;

   for (unsigned i = 0; i < 32; i++) {
      if (BITSET_TEST(nir->info.textures_used, i))
         mask |= 1u << i;
   }

   return mask;
}

static uint32_t
yttrium_shader_tgsi_sampled_texture_mask(
   const struct yttrium_shader_state *shader)
{
   struct tgsi_parse_context parse;
   uint32_t mask = 0;

   if (!shader || !shader->tokens)
      return 0;

   if (tgsi_parse_init(&parse, shader->tokens) != TGSI_PARSE_OK)
      return 0;

   while (!tgsi_parse_end_of_tokens(&parse)) {
      tgsi_parse_token(&parse);
      if (parse.FullToken.Token.Type != TGSI_TOKEN_TYPE_INSTRUCTION)
         continue;

      const struct tgsi_full_instruction *inst =
         &parse.FullToken.FullInstruction;
      for (unsigned i = 0; i < inst->Instruction.NumSrcRegs; i++) {
         const struct tgsi_full_src_register *src = &inst->Src[i];

         if (src->Register.File != TGSI_FILE_SAMPLER &&
             src->Register.File != TGSI_FILE_SAMPLER_VIEW)
            continue;

         if (src->Register.Indirect) {
            mask |= shader->info.samplers_declared |
                    shader->info.file_mask[TGSI_FILE_SAMPLER] |
                    shader->info.file_mask[TGSI_FILE_SAMPLER_VIEW];
         } else if (src->Register.Index < 32) {
            mask |= 1u << src->Register.Index;
         }
      }
   }

   tgsi_parse_free(&parse);
   return mask;
}

static uint32_t
yttrium_shader_sampled_texture_mask(
   const struct yttrium_shader_state *shader)
{
   uint32_t used_mask;

   if (!shader)
      return 0;

   /* Sampler-view declarations describe resource metadata and may use D3D
    * resource slots that differ from the sampler index carried by legacy TGSI
    * texture instructions.  Pipeline descriptors must follow the sampler and
    * sampler-view operands used by texture instructions, not every declared
    * resource slot.  If frontend folding removes every such operand, retain
    * the declarations so resource-query shaders still take the resource-aware
    * pipeline path.
    */
   used_mask = yttrium_shader_tgsi_sampled_texture_mask(shader) |
               yttrium_nir_sampled_texture_mask32(shader->nir);
   return used_mask ? used_mask :
                      yttrium_shader_info_sampled_texture_mask(&shader->info);
}

static void
yttrium_shader_trace_sampler_info(const struct yttrium_shader_state *shader,
                                  const char *phase)
{
   if (!shader)
      return;

   const uint32_t declared =
      yttrium_shader_info_sampled_texture_mask(&shader->info);
   const uint32_t used = shader->sampler_used_mask;

   for (unsigned i = 0; i < MIN2(PIPE_MAX_SHADER_SAMPLER_VIEWS, 32); i++) {
      if (!(declared & (1u << i)) && !(used & (1u << i)))
         continue;

      yttrium_trace_debug_stringf(
         "yttrium: shader_sampler_info phase=%s stage=%s id=%u slot=%u declared=%u used=%u sampler_mask=0x%x tgsi_target=%u tgsi_return=%u",
         phase ? phase : "unknown",
         yttrium_shader_stage_name(shader->stage),
         shader->id,
         i,
         (declared & (1u << i)) != 0,
         (used & (1u << i)) != 0,
         declared | used,
         shader->info.sampler_targets[i],
         shader->info.sampler_type[i]);
   }
}

static void
yttrium_shader_init_sampler_binding_map(struct yttrium_shader_state *shader)
{
   if (!shader)
      return;

   shader->sampler_binding_valid_mask = 0;
   shader->sampler_mixed_return_mask = 0;
   for (unsigned i = 0; i < PIPE_MAX_SAMPLERS; i++) {
      shader->sampler_view_index[i] = i;
      shader->sampler_state_index[i] = i;
   }
}

static bool
yttrium_shader_apply_sampler_binding_map(
   struct yttrium_shader_state *shader,
   const struct pipe_shader_sampler_binding_map *map)
{
   if (!shader || !map)
      return true;

   const uint32_t declared =
      yttrium_shader_info_sampled_texture_mask(&shader->info);
   if (map->mixed_return_mask & ~map->valid_mask) {
      YTTRIUM_WARN("yttrium: shader state creation failed "
                   "owner=yttrium-shader component=sampler-binding-map "
                   "reason=mixed-mask-outside-valid-map stage=%s id=%u "
                   "mixed=0x%x mapped=0x%x action=reject-shader\n",
                   yttrium_shader_stage_name(shader->stage), shader->id,
                   map->mixed_return_mask, map->valid_mask);
      return false;
   }
   if (declared & ~map->valid_mask) {
      YTTRIUM_WARN("yttrium: shader state creation failed "
                   "owner=yttrium-shader component=sampler-binding-map "
                   "reason=unmapped-compact-slot stage=%s id=%u "
                   "declared=0x%x mapped=0x%x action=reject-shader\n",
                   yttrium_shader_stage_name(shader->stage), shader->id,
                   declared, map->valid_mask);
      return false;
   }

   uint32_t remaining = map->valid_mask;
   while (remaining) {
      const unsigned slot = u_bit_scan(&remaining);
      const unsigned view = map->sampler_view_index[slot];
      const unsigned state = map->sampler_state_index[slot];
      if (view >= PIPE_MAX_SHADER_SAMPLER_VIEWS ||
          state > PIPE_MAX_SAMPLERS) {
         YTTRIUM_WARN("yttrium: shader state creation failed "
                      "owner=yttrium-shader component=sampler-binding-map "
                      "reason=invalid-map-entry stage=%s id=%u slot=%u "
                      "view=%u state=%u action=reject-shader\n",
                      yttrium_shader_stage_name(shader->stage), shader->id,
                      slot, view, state);
         return false;
      }

      if ((map->mixed_return_mask & (1u << slot)) &&
          (slot >= ARRAY_SIZE(shader->info.sampler_targets) ||
           slot >= ARRAY_SIZE(shader->info.sampler_type) ||
           shader->info.sampler_targets[slot] != TGSI_TEXTURE_BUFFER ||
           shader->info.sampler_type[slot] != TGSI_RETURN_TYPE_FLOAT)) {
         YTTRIUM_WARN("yttrium: shader state creation failed "
                      "owner=yttrium-shader component=sampler-binding-map "
                      "reason=invalid-mixed-slot stage=%s id=%u slot=%u "
                      "target=%u return=%u action=reject-shader\n",
                      yttrium_shader_stage_name(shader->stage), shader->id,
                      slot,
                      slot < ARRAY_SIZE(shader->info.sampler_targets) ?
                         shader->info.sampler_targets[slot] :
                         TGSI_TEXTURE_UNKNOWN,
                      slot < ARRAY_SIZE(shader->info.sampler_type) ?
                         shader->info.sampler_type[slot] :
                         TGSI_RETURN_TYPE_COUNT);
         return false;
      }

      shader->sampler_view_index[slot] = (uint16_t)view;
      shader->sampler_state_index[slot] = (uint8_t)state;
   }
   shader->sampler_binding_valid_mask = map->valid_mask;
   shader->sampler_mixed_return_mask = map->mixed_return_mask;
   return true;
}

static void
yttrium_shader_copy_sampler_binding_map(
   struct yttrium_shader_state *dst,
   const struct yttrium_shader_state *src)
{
   if (!dst || !src)
      return;

   dst->sampler_binding_valid_mask = src->sampler_binding_valid_mask;
   dst->sampler_mixed_return_mask = src->sampler_mixed_return_mask;
   memcpy(dst->sampler_view_index, src->sampler_view_index,
          sizeof(dst->sampler_view_index));
   memcpy(dst->sampler_state_index, src->sampler_state_index,
          sizeof(dst->sampler_state_index));
}

static void
yttrium_shader_scan_sampler_view_map(struct yttrium_shader_state *shader)
{
   struct tgsi_parse_context parse;

   if (!shader || !shader->tokens)
      return;

   if (tgsi_parse_init(&parse, shader->tokens) != TGSI_PARSE_OK)
      return;

   while (!tgsi_parse_end_of_tokens(&parse)) {
      int sampler_slot = -1;
      int view_slot = -1;

      tgsi_parse_token(&parse);
      if (parse.FullToken.Token.Type != TGSI_TOKEN_TYPE_INSTRUCTION)
         continue;

      const struct tgsi_full_instruction *inst =
         &parse.FullToken.FullInstruction;
      for (unsigned i = 0; i < inst->Instruction.NumSrcRegs; i++) {
         const struct tgsi_full_src_register *src = &inst->Src[i];

         if (src->Register.Indirect)
            continue;

         if (src->Register.File == TGSI_FILE_SAMPLER &&
             src->Register.Index < PIPE_MAX_SAMPLERS)
            sampler_slot = src->Register.Index;
         else if (src->Register.File == TGSI_FILE_SAMPLER_VIEW &&
                  src->Register.Index < PIPE_MAX_SHADER_SAMPLER_VIEWS)
            view_slot = src->Register.Index;
      }

      if (sampler_slot >= 0 && view_slot >= 0) {
         shader->sampler_view_index[sampler_slot] = (uint16_t)view_slot;
         yttrium_trace_debug_stringf(
            "yttrium: shader_sampler_view_map stage=%s id=%u sampler=%u view=%u",
            yttrium_shader_stage_name(shader->stage), shader->id,
            (unsigned)sampler_slot, (unsigned)view_slot);
      }
   }

   tgsi_parse_free(&parse);
}

static void
yttrium_shader_trace_io_info(const struct yttrium_shader_state *shader,
                             const char *phase)
{
   if (!shader)
      return;

   for (unsigned i = 0; i < shader->info.num_inputs; i++) {
      yttrium_trace_debug_stringf(
         "yttrium: shader_io phase=%s stage=%s id=%u input=%u semantic=%u index=%u usage=0x%x interp=%u loc=%u",
         phase ? phase : "<none>",
         yttrium_shader_stage_name(shader->stage),
         shader->id,
         i,
         shader->info.input_semantic_name[i],
         shader->info.input_semantic_index[i],
         shader->info.input_usage_mask[i],
         shader->info.input_interpolate[i],
         shader->info.input_interpolate_loc[i]);
   }

   for (unsigned i = 0; i < shader->info.num_outputs; i++) {
      yttrium_trace_debug_stringf(
         "yttrium: shader_io phase=%s stage=%s id=%u output=%u semantic=%u index=%u usage=0x%x streams=0x%x",
         phase ? phase : "<none>",
         yttrium_shader_stage_name(shader->stage),
         shader->id,
         i,
         shader->info.output_semantic_name[i],
         shader->info.output_semantic_index[i],
         shader->info.output_usagemask[i],
         shader->info.output_streams[i]);
   }
}

static void
yttrium_shader_trace_nir_samplers(const struct yttrium_shader_state *shader,
                                  const char *phase)
{
   if (!shader || !shader->nir)
      return;

   nir_foreach_variable_with_modes(var, shader->nir, nir_var_uniform) {
      const struct glsl_type *type = glsl_without_array(var->type);
      if (!glsl_type_is_sampler(type))
         continue;

      const uint32_t binding = var->data.binding;
      unsigned raw_slot = binding;
      (void)yttrium_shader_sampler_binding_decode(binding, NULL, &raw_slot);

      yttrium_trace_debug_stringf(
         "yttrium: shader_nir_sampler phase=%s stage=%s id=%u name=%s binding=%u raw_slot=%u descriptor_set=%u explicit=%u dim=%u result=%u array=%u shadow=%u type=%p",
         phase ? phase : "unknown",
         yttrium_shader_stage_name(shader->stage),
         shader->id,
         var->name ? var->name : "",
         binding,
         raw_slot,
         var->data.descriptor_set,
         var->data.explicit_binding,
         glsl_get_sampler_dim(type),
         glsl_get_sampler_result_type(type),
         glsl_sampler_type_is_array(type),
         glsl_sampler_type_is_shadow(type),
         type);
   }
}

static uint32_t
yttrium_shader_spirv_execution_model(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return SpvExecutionModelVertex;
   case MESA_SHADER_FRAGMENT:
      return SpvExecutionModelFragment;
   case MESA_SHADER_GEOMETRY:
      return SpvExecutionModelGeometry;
   case MESA_SHADER_TESS_CTRL:
      return SpvExecutionModelTessellationControl;
   case MESA_SHADER_TESS_EVAL:
      return SpvExecutionModelTessellationEvaluation;
   case MESA_SHADER_COMPUTE:
      return SpvExecutionModelGLCompute;
   default:
      return UINT32_MAX;
   }
}

static const char *
yttrium_spirv_probe_fail(struct yttrium_spirv_probe *probe,
                         const char *failure,
                         uint32_t offset,
                         uint32_t word_count,
                         uint32_t opcode)
{
   probe->well_formed = false;
   probe->failure = failure;
   probe->bad_word_offset = offset;
   probe->bad_word_count = word_count;
   probe->bad_opcode = opcode;
   return failure;
}

static bool
yttrium_shader_probe_spirv(const struct yttrium_shader_state *shader,
                           struct yttrium_spirv_probe *probe)
{
   const uint32_t *words;
   size_t word_count;
   size_t offset;

   if (!probe)
      return false;

   memset(probe, 0, sizeof(*probe));
   probe->well_formed = true;
   probe->expected_execution_model =
      shader ? yttrium_shader_spirv_execution_model(shader->stage) :
               UINT32_MAX;

   if (!shader || !shader->spirv || !shader->spirv->words ||
       shader->spirv->num_words < 5) {
      yttrium_spirv_probe_fail(probe, "short_header", 0, 0, 0);
      return false;
   }

   words = shader->spirv->words;
   word_count = shader->spirv->num_words;
   probe->version = words[1];
   probe->generator = words[2];
   probe->bound = words[3];
   probe->reserved = words[4];

   if (words[0] != SpvMagicNumber) {
      yttrium_spirv_probe_fail(probe, "bad_magic", 0, 0, words[0]);
      return false;
   }

   if (!probe->bound) {
      yttrium_spirv_probe_fail(probe, "zero_bound", 3, 0, 0);
      return false;
   }

   if (probe->reserved) {
      yttrium_spirv_probe_fail(probe, "nonzero_reserved", 4, 0,
                               probe->reserved);
      return false;
   }

   for (offset = 5; offset < word_count; ) {
      const uint32_t op_word = words[offset];
      const uint32_t inst_word_count = op_word >> 16;
      const uint32_t opcode = op_word & 0xffff;

      if (!inst_word_count) {
         yttrium_spirv_probe_fail(probe, "zero_instruction_words",
                                  (uint32_t)offset, inst_word_count, opcode);
         return false;
      }

      if (offset + inst_word_count > word_count) {
         yttrium_spirv_probe_fail(probe, "truncated_instruction",
                                  (uint32_t)offset, inst_word_count, opcode);
         return false;
      }

      probe->instructions++;
      switch (opcode) {
      case SpvOpCapability:
         probe->capabilities++;
         break;
      case SpvOpMemoryModel:
         if (inst_word_count >= 3) {
            probe->has_memory_model = true;
            probe->memory_addressing_model = words[offset + 1];
            probe->memory_model = words[offset + 2];
         }
         break;
      case SpvOpEntryPoint:
         if (inst_word_count >= 3) {
            const uint32_t execution_model = words[offset + 1];

            if (!probe->entry_points)
               probe->first_execution_model = execution_model;
            probe->entry_points++;
            if (execution_model == probe->expected_execution_model)
               probe->has_stage_entry = true;
         }
         break;
      case SpvOpExecutionMode:
         probe->execution_modes++;
         break;
      case SpvOpDecorate:
         if (inst_word_count >= 3 && words[offset + 2] == SpvDecorationLocation)
            probe->location_decorations++;
         break;
      case SpvOpVariable:
         if (inst_word_count >= 4) {
            switch (words[offset + 3]) {
            case SpvStorageClassInput:
               probe->input_variables++;
               break;
            case SpvStorageClassOutput:
               probe->output_variables++;
               break;
            case SpvStorageClassUniform:
               probe->uniform_variables++;
               break;
            case SpvStorageClassUniformConstant:
               probe->uniform_constant_variables++;
               break;
            case SpvStorageClassPushConstant:
               probe->push_constant_variables++;
               break;
            case SpvStorageClassPrivate:
               probe->private_variables++;
               break;
            default:
               break;
            }
         }
         break;
      default:
         break;
      }

      offset += inst_word_count;
   }

   return true;
}

static void
yttrium_shader_log_spirv_probe(const struct yttrium_shader_state *shader,
                               const struct yttrium_spirv_probe *probe)
{
   if (!shader || !probe)
      return;

   YTTRIUM_LOG("yttrium: shader_spirv_probe stage=%s id=%u valid=%u failure=%s words=%u version=0x%x generator=0x%x bound=%u instructions=%u caps=%u memory_model=%u/%u has_memory_model=%u entries=%u first_model=%u expected_model=%u has_stage_entry=%u exec_modes=%u locations=%u vars(input=%u output=%u uniform=%u uniform_constant=%u push=%u private=%u) bad(offset=%u words=%u opcode=%u) spirv_hash=0x%llx token_hash=0x%llx\n",
               yttrium_shader_stage_name(shader->stage),
               shader->id,
               probe->well_formed,
               probe->failure ? probe->failure : "none",
               shader->spirv_word_count,
               probe->version,
               probe->generator,
               probe->bound,
               probe->instructions,
               probe->capabilities,
               probe->memory_addressing_model,
               probe->memory_model,
               probe->has_memory_model,
               probe->entry_points,
               probe->first_execution_model,
               probe->expected_execution_model,
               probe->has_stage_entry,
               probe->execution_modes,
               probe->location_decorations,
               probe->input_variables,
               probe->output_variables,
               probe->uniform_variables,
               probe->uniform_constant_variables,
               probe->push_constant_variables,
               probe->private_variables,
               probe->bad_word_offset,
               probe->bad_word_count,
               probe->bad_opcode,
               (unsigned long long)shader->spirv_hash,
               (unsigned long long)shader->token_hash);
}

static void
yttrium_shader_dump_spirv_failure(const struct yttrium_shader_state *shader,
                                  const char *reason,
                                  int result)
{
   char path[MAX_PATH];
   FILE *file;

   if (!shader || !shader->spirv || !shader->spirv->words ||
       !shader->spirv->num_words)
      return;

   CreateDirectoryA("C:\\ProgramData\\Yttrium", NULL);
   CreateDirectoryA(YTTRIUM_SHADER_DUMP_DIR, NULL);

   snprintf(path, sizeof(path),
            YTTRIUM_SHADER_DUMP_DIR
            "\\yttrium_%s_id%u_result%d_%016llx.spv",
            yttrium_shader_stage_name(shader->stage),
            shader->id,
            result,
            (unsigned long long)shader->spirv_hash);
   path[sizeof(path) - 1] = '\0';

   file = fopen(path, "wb");
   if (!file) {
      YTTRIUM_WARN("yttrium: shader_spirv_dump failed stage=%s id=%u reason=%s path=%s\n",
                   yttrium_shader_stage_name(shader->stage),
                   shader->id,
                   reason ? reason : "unknown",
                   path);
      return;
   }

   fwrite(shader->spirv->words, sizeof(uint32_t), shader->spirv->num_words,
          file);
   fclose(file);

   YTTRIUM_WARN("yttrium: shader_spirv_dump wrote stage=%s id=%u reason=%s result=%d path=%s words=%u spirv_hash=0x%llx\n",
                yttrium_shader_stage_name(shader->stage),
                shader->id,
                reason ? reason : "unknown",
                result,
                path,
                shader->spirv_word_count,
                (unsigned long long)shader->spirv_hash);
}

static void
yttrium_shader_dump_sampled_success(const struct yttrium_shader_state *shader)
{
   char path[MAX_PATH];
   FILE *file;
   const uint32_t declared_mask =
      shader ? yttrium_shader_info_sampled_texture_mask(&shader->info) : 0;
   const uint32_t used_mask =
      shader ? shader->sampler_used_mask : 0;
   const uint32_t sampler_mask = declared_mask | used_mask;
   bool has_buffer_sampler = false;
   const DWORD pid = GetCurrentProcessId();

   if (!yttrium_gdi_debug_get_bool_option(
          "D3D10UMD_YTTRIUM_SHADER_DUMP_SAMPLED_SUCCESS", false))
      return;

   if (shader) {
      for (unsigned i = 0; i < MIN2(PIPE_MAX_SHADER_SAMPLER_VIEWS, 32); i++) {
         if (!(sampler_mask & (1u << i)))
            continue;
         if (shader->info.sampler_targets[i] == TGSI_TEXTURE_BUFFER) {
            has_buffer_sampler = true;
            break;
         }
      }
   }

   if (!shader ||
       (!has_buffer_sampler && !shader->sampled_texture_only) ||
       !shader->spirv || !shader->spirv->words || !shader->spirv->num_words)
      return;

   CreateDirectoryA("C:\\ProgramData\\Yttrium", NULL);
   CreateDirectoryA(YTTRIUM_SHADER_DUMP_DIR, NULL);

   snprintf(path, sizeof(path),
            YTTRIUM_SHADER_DUMP_DIR
            "\\yttrium_%s_pid%lu_id%u_sampled_%016llx.spv",
            yttrium_shader_stage_name(shader->stage),
            (unsigned long)pid,
            shader->id,
            (unsigned long long)shader->spirv_hash);
   path[sizeof(path) - 1] = '\0';
   file = fopen(path, "wb");
   if (file) {
      fwrite(shader->spirv->words, sizeof(uint32_t), shader->spirv->num_words,
             file);
      fclose(file);
      yttrium_trace_debug_stringf(
         "yttrium: shader_sampled_dump wrote stage=%s pid=%lu id=%u kind=spirv path=%s words=%u hash=0x%llx declared_mask=0x%x used_mask=0x%x buffer_sampler=%u",
         yttrium_shader_stage_name(shader->stage),
         (unsigned long)pid,
         shader->id,
         path,
         shader->spirv_word_count,
         (unsigned long long)shader->spirv_hash,
         declared_mask,
         used_mask,
         has_buffer_sampler);
   }

   if (shader->tokens) {
      snprintf(path, sizeof(path),
               YTTRIUM_SHADER_DUMP_DIR
               "\\yttrium_%s_pid%lu_id%u_sampled_%016llx.tgsi",
               yttrium_shader_stage_name(shader->stage),
               (unsigned long)pid,
               shader->id,
               (unsigned long long)shader->spirv_hash);
      path[sizeof(path) - 1] = '\0';
      file = fopen(path, "w");
      if (file) {
         yttrium_write_shader_capture_metadata(file, shader);
         tgsi_dump_to_file(shader->tokens, 0, file);
         fclose(file);
         yttrium_trace_debug_stringf(
            "yttrium: shader_sampled_dump wrote stage=%s pid=%lu id=%u kind=tgsi path=%s hash=0x%llx declared_mask=0x%x used_mask=0x%x buffer_sampler=%u",
            yttrium_shader_stage_name(shader->stage),
            (unsigned long)pid,
            shader->id,
            path,
            (unsigned long long)shader->spirv_hash,
            declared_mask,
            used_mask,
            has_buffer_sampler);
      }
   }

   if (shader->nir) {
      snprintf(path, sizeof(path),
               YTTRIUM_SHADER_DUMP_DIR
               "\\yttrium_%s_pid%lu_id%u_sampled_%016llx.nir",
               yttrium_shader_stage_name(shader->stage),
               (unsigned long)pid,
               shader->id,
               (unsigned long long)shader->spirv_hash);
      path[sizeof(path) - 1] = '\0';
      file = fopen(path, "w");
      if (file) {
         yttrium_write_shader_capture_metadata(file, shader);
         nir_print_shader(shader->nir, file);
         fclose(file);
         yttrium_trace_debug_stringf(
            "yttrium: shader_sampled_dump wrote stage=%s pid=%lu id=%u kind=nir path=%s hash=0x%llx declared_mask=0x%x used_mask=0x%x buffer_sampler=%u",
            yttrium_shader_stage_name(shader->stage),
            (unsigned long)pid,
            shader->id,
            path,
            (unsigned long long)shader->spirv_hash,
            declared_mask,
            used_mask,
            has_buffer_sampler);
      }
   }
}

bool
yttrium_shader_pipeline_enabled(void)
{
   /* Consulted per draw; the config does not change after start-up. */
   static int enabled = -1;

   if (enabled < 0) {
      enabled = yttrium_gdi_debug_get_bool_option(
         "D3D10UMD_YTTRIUM_SHADER_PIPELINE", true) ? 1 : 0;
   }

   return enabled != 0;
}

/*
 * Each of these adds one knob on top of the previous one, so the chain reads
 * up to four options per call - and yttrium_shader_pipeline_draw_enabled() is
 * consulted per draw.  The || short-circuits when the base knob is on, which
 * is why this only bites when it is off, and hid the cost during measurement.
 * Memoize each level the same way yttrium_shader_pipeline_enabled() does.
 */
bool
yttrium_shader_pipeline_draw_enabled(void)
{
   static int enabled = -1;

   if (enabled < 0) {
      enabled = (yttrium_shader_pipeline_enabled() ||
                 yttrium_gdi_debug_get_bool_option(
                    "D3D10UMD_YTTRIUM_SHADER_PIPELINE_DRAW", false)) ? 1 : 0;
   }

   return enabled != 0;
}

bool
yttrium_shader_module_enabled(void)
{
   static int enabled = -1;

   if (enabled < 0) {
      enabled = (yttrium_shader_pipeline_draw_enabled() ||
                 yttrium_gdi_debug_get_bool_option(
                    "D3D10UMD_YTTRIUM_SHADER_MODULE", false)) ? 1 : 0;
   }

   return enabled != 0;
}

bool
yttrium_shader_compile_enabled(void)
{
   static int enabled = -1;

   if (enabled < 0) {
      enabled = (yttrium_shader_module_enabled() ||
                 yttrium_gdi_debug_get_bool_option(
                    "D3D10UMD_YTTRIUM_SHADER_COMPILE", false)) ? 1 : 0;
   }

   return enabled != 0;
}

bool
yttrium_shader_caps_enabled(void)
{
   static int enabled = -1;

   if (enabled < 0) {
      enabled = (yttrium_shader_compile_enabled() ||
                 yttrium_gdi_debug_get_bool_option(
                    "D3D10UMD_YTTRIUM_SHADER_CAPS", false)) ? 1 : 0;
   }

   return enabled != 0;
}

static uint64_t
yttrium_fnv1a_bytes(const void *data, size_t size)
{
   const uint8_t *bytes = data;
   uint64_t hash = 1469598103934665603ull;

   if (!bytes)
      return 0;

   for (size_t i = 0; i < size; i++) {
      hash ^= bytes[i];
      hash *= 1099511628211ull;
   }

   return hash;
}

static uint64_t
yttrium_shader_token_hash(const struct yttrium_shader_state *shader)
{
   if (!shader || !shader->tokens || !shader->token_count)
      return 0;

   return yttrium_fnv1a_bytes(shader->tokens,
                              (size_t)shader->token_count *
                              sizeof(struct tgsi_token));
}

static uint64_t
yttrium_shader_spirv_hash(const struct spirv_shader *spirv)
{
   if (!spirv || !spirv->words || !spirv->num_words)
      return 0;

   return yttrium_fnv1a_bytes(spirv->words,
                              spirv->num_words * sizeof(uint32_t));
}

static bool
yttrium_shader_info_resource_free(const struct tgsi_shader_info *info)
{
   return info &&
          !info->const_buffers_declared &&
          !yttrium_shader_info_sampled_texture_mask(info) &&
          !info->images_declared &&
          !info->shader_buffers_declared &&
          !info->hw_atomic_declared &&
          !info->writes_memory &&
          !info->uses_fbfetch;
}

static bool
yttrium_shader_info_uniform_buffer_only(const struct tgsi_shader_info *info)
{
   return info &&
          !yttrium_shader_info_sampled_texture_mask(info) &&
          !info->images_declared &&
          !info->shader_buffers_declared &&
          !info->hw_atomic_declared &&
          !info->writes_memory &&
          !info->uses_fbfetch;
}

static bool
yttrium_shader_info_sampled_texture_only(const struct tgsi_shader_info *info)
{
   return info &&
          yttrium_shader_info_sampled_texture_mask(info) &&
          !info->images_declared &&
          !info->shader_buffers_declared &&
          !info->hw_atomic_declared &&
          !info->writes_memory &&
          !info->uses_fbfetch;
}

static bool
yttrium_shader_info_storage_image_only(const struct tgsi_shader_info *info)
{
   return info &&
          info->images_declared &&
          !yttrium_shader_info_sampled_texture_mask(info) &&
          !info->shader_buffers_declared &&
          !info->uses_fbfetch;
}

static bool
yttrium_shader_info_sampled_storage_image_only(
   const struct tgsi_shader_info *info)
{
   return info &&
          yttrium_shader_info_sampled_texture_mask(info) &&
          info->images_declared &&
          !info->shader_buffers_declared &&
          !info->uses_fbfetch;
}

static bool
yttrium_nir_resource_free(const struct nir_shader *nir)
{
   return nir &&
          !nir->info.num_textures &&
          !nir->info.num_ubos &&
          !nir->info.num_abos &&
          !nir->info.num_ssbos &&
          !nir->info.num_images &&
          !nir->info.writes_memory;
}

static bool
yttrium_nir_sampled_texture_only(const struct nir_shader *nir)
{
   return nir &&
          nir->info.num_textures &&
          !nir->info.num_abos &&
          !nir->info.num_ssbos &&
          !nir->info.num_images &&
          !nir->info.writes_memory;
}

static bool
yttrium_nir_storage_image_only(const struct nir_shader *nir)
{
   return nir &&
          nir->info.num_images &&
          !nir->info.num_textures &&
          !nir->info.num_abos &&
          !nir->info.num_ssbos;
}

static bool
yttrium_nir_sampled_storage_image_only(const struct nir_shader *nir)
{
   return nir &&
          nir->info.num_textures &&
          nir->info.num_images &&
          !nir->info.num_abos &&
          !nir->info.num_ssbos;
}

static bool
yttrium_nir_uniform_buffer_only(const struct nir_shader *nir)
{
   return nir &&
          !nir->info.num_textures &&
          !nir->info.num_abos &&
          !nir->info.num_ssbos &&
          !nir->info.num_images &&
          !nir->info.writes_memory;
}

static bool
yttrium_shader_state_is_placeholder_module(
   const struct yttrium_shader_state *shader)
{
   if (!shader)
      return true;

   if (!yttrium_shader_info_resource_free(&shader->info))
      return false;

   if (shader->nir) {
      if (!yttrium_nir_resource_free(shader->nir))
         return false;

      switch (shader->stage) {
      case MESA_SHADER_VERTEX:
      case MESA_SHADER_TESS_CTRL:
      case MESA_SHADER_TESS_EVAL:
         return shader->nir->info.outputs_written == 0;
      case MESA_SHADER_FRAGMENT:
         return shader->nir->info.outputs_written == 0 &&
                !shader->nir->info.fs.uses_discard;
      case MESA_SHADER_GEOMETRY:
         return shader->nir->info.outputs_written == 0 &&
                shader->stream_output.num_outputs == 0;
      default:
         return true;
      }
   }

   switch (shader->stage) {
   case MESA_SHADER_VERTEX:
      return shader->info.num_outputs == 0;
   case MESA_SHADER_FRAGMENT:
      return shader->info.num_outputs == 0 &&
             !shader->info.writes_z &&
             !shader->info.uses_kill;
   case MESA_SHADER_GEOMETRY:
      return shader->info.num_outputs == 0 &&
             shader->stream_output.num_outputs == 0;
   default:
      return true;
   }
}

static bool
yttrium_tgsi_src_file_supported_for_ttn(unsigned file)
{
   switch (file) {
   case TGSI_FILE_TEMPORARY:
   case TGSI_FILE_ADDRESS:
   case TGSI_FILE_IMMEDIATE:
   case TGSI_FILE_SYSTEM_VALUE:
   case TGSI_FILE_INPUT:
   case TGSI_FILE_OUTPUT:
   case TGSI_FILE_CONSTANT:
   case TGSI_FILE_MEMORY:
      return true;
   default:
      return false;
   }
}

static bool
yttrium_tgsi_direct_src_file_supported_for_ttn(unsigned file,
                                               bool indirect)
{
   switch (file) {
   case TGSI_FILE_NULL:
      return true;
   case TGSI_FILE_SAMPLER:
   case TGSI_FILE_IMAGE:
   case TGSI_FILE_BUFFER:
   case TGSI_FILE_MEMORY:
      return !indirect;
   default:
      return yttrium_tgsi_src_file_supported_for_ttn(file);
   }
}

static bool
yttrium_tgsi_system_value_supported_for_ttn(unsigned semantic)
{
   switch (semantic) {
   case TGSI_SEMANTIC_VERTEXID_NOBASE:
   case TGSI_SEMANTIC_VERTEXID:
   case TGSI_SEMANTIC_BASEVERTEX:
   case TGSI_SEMANTIC_INSTANCEID:
   case TGSI_SEMANTIC_INVOCATIONID:
   case TGSI_SEMANTIC_TESSCOORD:
   case TGSI_SEMANTIC_PRIMID:
   case TGSI_SEMANTIC_FACE:
   case TGSI_SEMANTIC_POSITION:
   case TGSI_SEMANTIC_PCOORD:
   case TGSI_SEMANTIC_THREAD_ID:
   case TGSI_SEMANTIC_BLOCK_ID:
   case TGSI_SEMANTIC_BLOCK_SIZE:
   case TGSI_SEMANTIC_CS_USER_DATA_AMD:
   case TGSI_SEMANTIC_SAMPLEID:
   case TGSI_SEMANTIC_SAMPLEMASK:
      return true;
   default:
      return false;
   }
}

static bool
yttrium_tgsi_log_unsupported_src_file(
   const struct yttrium_shader_state *shader,
   unsigned inst_no,
   enum tgsi_opcode opcode,
   unsigned src_idx,
   const char *part,
   unsigned file,
   int index)
{
   YTTRIUM_WARN("yttrium: shader_compile_nir skipped stage=%s id=%u reason=unsupported_tgsi_src_file inst=%u opcode=%s src=%u part=%s file=%s(%u) index=%d token_hash=0x%llx inputs=%u outputs=%u constbufs=0x%x samplers=0x%x images=0x%x buffers=0x%x\n",
                yttrium_shader_stage_name(shader->stage),
                shader->id,
                inst_no,
                tgsi_get_opcode_name(opcode),
                src_idx,
                part,
                tgsi_file_name(file),
                file,
                index,
                (unsigned long long)shader->token_hash,
                shader->info.num_inputs,
                shader->info.num_outputs,
                shader->info.const_buffers_declared,
                yttrium_shader_info_sampled_texture_mask(&shader->info),
                shader->info.images_declared,
                shader->info.shader_buffers_declared);
   return false;
}

static bool
yttrium_tgsi_log_unsupported_use(
   const struct yttrium_shader_state *shader,
   unsigned inst_no,
   enum tgsi_opcode opcode,
   unsigned slot,
   const char *reason,
   const char *part,
   unsigned file,
   int index)
{
   YTTRIUM_WARN("yttrium: shader_compile_nir skipped stage=%s id=%u reason=%s inst=%u opcode=%s slot=%u part=%s file=%s(%u) index=%d token_hash=0x%llx inputs=%u outputs=%u indirect_files=0x%x dim_indirect_files=0x%x\n",
                yttrium_shader_stage_name(shader->stage),
                shader->id,
                reason,
                inst_no,
                tgsi_get_opcode_name(opcode),
                slot,
                part,
                tgsi_file_name(file),
                file,
                index,
                (unsigned long long)shader->token_hash,
                shader->info.num_inputs,
                shader->info.num_outputs,
                shader->info.indirect_files,
                shader->info.dim_indirect_files);
   return false;
}

static bool
yttrium_tgsi_decl_file_supported_for_ttn(unsigned file)
{
   switch (file) {
   case TGSI_FILE_TEMPORARY:
   case TGSI_FILE_ADDRESS:
   case TGSI_FILE_SYSTEM_VALUE:
   case TGSI_FILE_BUFFER:
   case TGSI_FILE_IMAGE:
   case TGSI_FILE_MEMORY:
   case TGSI_FILE_SAMPLER:
   case TGSI_FILE_SAMPLER_VIEW:
   case TGSI_FILE_INPUT:
   case TGSI_FILE_OUTPUT:
   case TGSI_FILE_CONSTANT:
      return true;
   default:
      return false;
   }
}

static bool
yttrium_tgsi_address_index_supported(
   const struct yttrium_shader_state *shader,
   int index)
{
   return shader &&
          index >= 0 &&
          shader->info.file_max[TGSI_FILE_ADDRESS] >= 0 &&
          index <= shader->info.file_max[TGSI_FILE_ADDRESS];
}

static bool
yttrium_tgsi_dimensioned_src_supported_for_ttn(
   const struct yttrium_shader_state *shader,
   const struct tgsi_full_src_register *src)
{
   if (!shader || !src || !src->Register.Dimension)
      return true;

   if (src->Register.File == TGSI_FILE_CONSTANT)
      return true;

   return (shader->stage == MESA_SHADER_GEOMETRY ||
           shader->stage == MESA_SHADER_TESS_CTRL ||
           shader->stage == MESA_SHADER_TESS_EVAL) &&
          src->Register.File == TGSI_FILE_INPUT;
}

static bool
yttrium_tgsi_texture_sampler_src_index(enum tgsi_opcode opcode,
                                       unsigned *src_idx)
{
   switch (opcode) {
   case TGSI_OPCODE_TEX:
   case TGSI_OPCODE_TXP:
   case TGSI_OPCODE_TXB:
   case TGSI_OPCODE_TXL:
   case TGSI_OPCODE_TEX_LZ:
   case TGSI_OPCODE_TXF:
   case TGSI_OPCODE_TXF_LZ:
   case TGSI_OPCODE_SAMPLE_I:
   case TGSI_OPCODE_LODQ:
   case TGSI_OPCODE_SAMPLE_INFO:
      *src_idx = 1;
      return true;
   case TGSI_OPCODE_TEX2:
   case TGSI_OPCODE_TXB2:
   case TGSI_OPCODE_TXL2:
   case TGSI_OPCODE_TG4:
      *src_idx = 2;
      return true;
   case TGSI_OPCODE_SAMPLE_C:
   case TGSI_OPCODE_SAMPLE_C_LZ:
      *src_idx = 2;
      return true;
   case TGSI_OPCODE_TXD:
      *src_idx = 3;
      return true;
   default:
      return false;
   }
}

static bool
yttrium_tgsi_texture_sampler_src_supported_for_ttn(
   enum tgsi_opcode opcode,
   const struct tgsi_full_src_register *src)
{
   if (!src || src->Register.Indirect)
      return false;

   if (opcode == TGSI_OPCODE_LODQ ||
       opcode == TGSI_OPCODE_SAMPLE_INFO)
      return src->Register.File == TGSI_FILE_SAMPLER_VIEW;

   if (opcode == TGSI_OPCODE_SAMPLE_I ||
       opcode == TGSI_OPCODE_TG4 ||
       opcode == TGSI_OPCODE_SAMPLE_C ||
       opcode == TGSI_OPCODE_SAMPLE_C_LZ)
      return src->Register.File == TGSI_FILE_SAMPLER ||
             src->Register.File == TGSI_FILE_SAMPLER_VIEW;

   return src->Register.File == TGSI_FILE_SAMPLER;
}

static bool
yttrium_tgsi_explicit_sview_src_supported_for_ttn(
   enum tgsi_opcode opcode,
   const struct tgsi_full_src_register *src)
{
   if (!src || src->Register.File != TGSI_FILE_SAMPLER_VIEW ||
       src->Register.Indirect)
      return false;

   return opcode == TGSI_OPCODE_LODQ ||
          opcode == TGSI_OPCODE_SAMPLE_INFO ||
          opcode == TGSI_OPCODE_SAMPLE_I ||
          opcode == TGSI_OPCODE_TG4 ||
          opcode == TGSI_OPCODE_SAMPLE_C ||
          opcode == TGSI_OPCODE_SAMPLE_C_LZ;
}

static bool
yttrium_shader_preflight_tgsi_for_nir(
   const struct yttrium_shader_state *shader,
   const struct tgsi_token *tokens)
{
   struct tgsi_parse_context parse;
   unsigned inst_no = 0;
   bool ok = true;
   bool *temp_array = NULL;
   const int max_temp = shader ? shader->info.file_max[TGSI_FILE_TEMPORARY] : -1;

   if (!shader || !tokens)
      return false;

   if (max_temp >= 0) {
      temp_array = CALLOC(max_temp + 1, sizeof(bool));
      if (!temp_array)
         return false;
   }

   if (tgsi_parse_init(&parse, tokens) != TGSI_PARSE_OK) {
      YTTRIUM_WARN("yttrium: shader_compile_nir skipped stage=%s id=%u reason=tgsi_parse_failed token_hash=0x%llx\n",
                   yttrium_shader_stage_name(shader->stage),
                   shader->id,
                   (unsigned long long)shader->token_hash);
      FREE(temp_array);
      return false;
   }

   while (!tgsi_parse_end_of_tokens(&parse)) {
      tgsi_parse_token(&parse);
      if (parse.FullToken.Token.Type == TGSI_TOKEN_TYPE_DECLARATION) {
         const struct tgsi_full_declaration *decl =
            &parse.FullToken.FullDeclaration;

         if (!yttrium_tgsi_decl_file_supported_for_ttn(
                decl->Declaration.File)) {
            YTTRIUM_WARN("yttrium: shader_compile_nir skipped stage=%s id=%u reason=unsupported_tgsi_decl_file file=%s(%u) first=%u last=%u token_hash=0x%llx\n",
                         yttrium_shader_stage_name(shader->stage),
                         shader->id,
                         tgsi_file_name(decl->Declaration.File),
                         decl->Declaration.File,
                         decl->Range.First,
                         decl->Range.Last,
                         (unsigned long long)shader->token_hash);
            ok = false;
            goto out;
         }

         if (decl->Declaration.File == TGSI_FILE_TEMPORARY &&
             decl->Declaration.Array && temp_array) {
            for (unsigned i = decl->Range.First; i <= decl->Range.Last &&
                 i <= (unsigned)max_temp; i++)
               temp_array[i] = true;
         }
         continue;
      }

      if (parse.FullToken.Token.Type == TGSI_TOKEN_TYPE_PROPERTY)
         continue;

      if (parse.FullToken.Token.Type != TGSI_TOKEN_TYPE_INSTRUCTION)
         continue;

      const struct tgsi_full_instruction *inst =
         &parse.FullToken.FullInstruction;
      enum tgsi_opcode opcode = inst->Instruction.Opcode;
      unsigned sampler_src_idx;

      if (yttrium_tgsi_texture_sampler_src_index(opcode, &sampler_src_idx) &&
          (inst->Instruction.NumSrcRegs <= sampler_src_idx ||
           !yttrium_tgsi_texture_sampler_src_supported_for_ttn(
              opcode, inst->Instruction.NumSrcRegs > sampler_src_idx ?
                         &inst->Src[sampler_src_idx] : NULL))) {
         const struct tgsi_full_src_register *src =
            inst->Instruction.NumSrcRegs > sampler_src_idx ?
               &inst->Src[sampler_src_idx] : NULL;
         ok = yttrium_tgsi_log_unsupported_use(
            shader, inst_no, opcode, sampler_src_idx,
            "unsupported_tgsi_texture_sampler_source",
            "src",
            src ? src->Register.File : TGSI_FILE_NULL,
            src ? src->Register.Index : -1);
         goto out;
      }

      for (unsigned i = 0; i < inst->Instruction.NumSrcRegs; i++) {
         const struct tgsi_full_src_register *src = &inst->Src[i];

         if (!yttrium_tgsi_explicit_sview_src_supported_for_ttn(opcode, src) &&
             !yttrium_tgsi_direct_src_file_supported_for_ttn(
                src->Register.File, src->Register.Indirect)) {
            ok = yttrium_tgsi_log_unsupported_src_file(
               shader, inst_no, opcode, i, "src",
               src->Register.File, src->Register.Index);
            goto out;
         }

         if (src->Register.File == TGSI_FILE_ADDRESS &&
             !yttrium_tgsi_address_index_supported(
                shader, src->Register.Index)) {
            ok = yttrium_tgsi_log_unsupported_use(
               shader, inst_no, opcode, i, "unsupported_tgsi_address_index",
               "src", src->Register.File, src->Register.Index);
            goto out;
         }

         if (src->Register.File == TGSI_FILE_SYSTEM_VALUE &&
             src->Register.Indirect) {
            ok = yttrium_tgsi_log_unsupported_use(
               shader, inst_no, opcode, i, "unsupported_tgsi_sysval_indirect",
               "src", src->Register.File, src->Register.Index);
            goto out;
         }

         if (!yttrium_tgsi_dimensioned_src_supported_for_ttn(shader, src)) {
            ok = yttrium_tgsi_log_unsupported_use(
               shader, inst_no, opcode, i, "unsupported_tgsi_src_dimension",
               "src", src->Register.File, src->Register.Index);
            goto out;
         }

         if (src->Register.File == TGSI_FILE_OUTPUT &&
             shader->stage != MESA_SHADER_FRAGMENT) {
            ok = yttrium_tgsi_log_unsupported_use(
               shader, inst_no, opcode, i, "unsupported_tgsi_output_read",
               "src", src->Register.File, src->Register.Index);
            goto out;
         }

         if (src->Register.File == TGSI_FILE_SYSTEM_VALUE &&
             (src->Register.Index >= PIPE_MAX_SHADER_INPUTS ||
              !yttrium_tgsi_system_value_supported_for_ttn(
                 shader->info.system_value_semantic_name[src->Register.Index]))) {
            ok = yttrium_tgsi_log_unsupported_use(
               shader, inst_no, opcode, i, "unsupported_tgsi_system_value",
               "src", src->Register.File, src->Register.Index);
            goto out;
         }

         if (src->Register.Indirect &&
             !yttrium_tgsi_src_file_supported_for_ttn(src->Indirect.File)) {
            ok = yttrium_tgsi_log_unsupported_src_file(
               shader, inst_no, opcode, i, "indirect",
               src->Indirect.File, src->Indirect.Index);
            goto out;
         }

         if (src->Register.Indirect &&
             src->Indirect.File == TGSI_FILE_ADDRESS &&
             !yttrium_tgsi_address_index_supported(
                shader, src->Indirect.Index)) {
            ok = yttrium_tgsi_log_unsupported_use(
               shader, inst_no, opcode, i, "unsupported_tgsi_address_index",
               "indirect", src->Indirect.File, src->Indirect.Index);
            goto out;
         }

         if (src->Register.Dimension && src->Dimension.Indirect &&
             !yttrium_tgsi_src_file_supported_for_ttn(src->DimIndirect.File)) {
            ok = yttrium_tgsi_log_unsupported_src_file(
               shader, inst_no, opcode, i, "dim_indirect",
               src->DimIndirect.File, src->DimIndirect.Index);
            goto out;
         }

         if (src->Register.Dimension && src->Dimension.Indirect &&
             src->DimIndirect.File == TGSI_FILE_ADDRESS &&
             !yttrium_tgsi_address_index_supported(
                shader, src->DimIndirect.Index)) {
            ok = yttrium_tgsi_log_unsupported_use(
               shader, inst_no, opcode, i, "unsupported_tgsi_address_index",
               "dim_indirect", src->DimIndirect.File,
               src->DimIndirect.Index);
            goto out;
         }
      }

      for (unsigned i = 0; i < inst->Instruction.NumDstRegs; i++) {
         const struct tgsi_full_dst_register *dst = &inst->Dst[i];

         if (dst->Register.File == TGSI_FILE_TEMPORARY &&
             dst->Register.Indirect &&
             (max_temp < 0 ||
              dst->Register.Index > (unsigned)max_temp ||
              !temp_array || !temp_array[dst->Register.Index])) {
            ok = yttrium_tgsi_log_unsupported_use(
               shader, inst_no, opcode, i, "unsupported_tgsi_temp_indirect",
               "dst", dst->Register.File, dst->Register.Index);
            goto out;
         }

         if (dst->Register.File == TGSI_FILE_ADDRESS &&
             !yttrium_tgsi_address_index_supported(
                shader, dst->Register.Index)) {
            ok = yttrium_tgsi_log_unsupported_use(
               shader, inst_no, opcode, i, "unsupported_tgsi_address_index",
               "dst", dst->Register.File, dst->Register.Index);
            goto out;
         }

         if (dst->Register.Indirect &&
             dst->Indirect.File == TGSI_FILE_ADDRESS &&
             !yttrium_tgsi_address_index_supported(
                shader, dst->Indirect.Index)) {
            ok = yttrium_tgsi_log_unsupported_use(
               shader, inst_no, opcode, i, "unsupported_tgsi_address_index",
               "dst_indirect", dst->Indirect.File, dst->Indirect.Index);
            goto out;
         }

         if (dst->Register.Indirect &&
             !yttrium_tgsi_src_file_supported_for_ttn(dst->Indirect.File)) {
            ok = yttrium_tgsi_log_unsupported_use(
               shader, inst_no, opcode, i, "unsupported_tgsi_dst_indirect_file",
               "dst_indirect", dst->Indirect.File, dst->Indirect.Index);
            goto out;
         }
      }

      inst_no++;
   }

out:
   tgsi_parse_free(&parse);
   FREE(temp_array);
   return ok;
}

struct yttrium_tgsi_indirect_array_transform {
   struct tgsi_transform_context base;
   unsigned original_max_temp;
   unsigned temp_array_last;
   unsigned imm_temp_base;
   unsigned imm_count;
   bool temp_decl_emitted;
   bool need_imm_loads;
};

static struct yttrium_tgsi_indirect_array_transform *
yttrium_tgsi_indirect_array_transform(struct tgsi_transform_context *ctx)
{
   return (struct yttrium_tgsi_indirect_array_transform *)ctx;
}

static void
yttrium_tgsi_emit_indirect_temp_decl(
   struct yttrium_tgsi_indirect_array_transform *tr)
{
   struct tgsi_full_declaration decl;

   if (tr->temp_decl_emitted)
      return;

   decl = tgsi_default_full_declaration();
   decl.Declaration.File = TGSI_FILE_TEMPORARY;
   decl.Declaration.Array = 1;
   decl.Range.First = 0;
   decl.Range.Last = tr->temp_array_last;
   decl.Array.ArrayID = 1;
   tr->base.emit_declaration(&tr->base, &decl);
   tr->temp_decl_emitted = true;
}

static void
yttrium_tgsi_indirect_array_decl(struct tgsi_transform_context *ctx,
                                 struct tgsi_full_declaration *decl)
{
   struct yttrium_tgsi_indirect_array_transform *tr =
      yttrium_tgsi_indirect_array_transform(ctx);

   if (!tr->temp_decl_emitted)
      yttrium_tgsi_emit_indirect_temp_decl(tr);

   if (decl->Declaration.File == TGSI_FILE_TEMPORARY)
      return;

   ctx->emit_declaration(ctx, decl);
}

static void
yttrium_tgsi_indirect_array_prolog(struct tgsi_transform_context *ctx)
{
   struct yttrium_tgsi_indirect_array_transform *tr =
      yttrium_tgsi_indirect_array_transform(ctx);
   struct tgsi_full_instruction inst;

   if (!tr->temp_decl_emitted)
      yttrium_tgsi_emit_indirect_temp_decl(tr);

   if (!tr->need_imm_loads)
      return;

   for (unsigned i = 0; i < tr->imm_count; i++) {
      inst = tgsi_default_full_instruction();
      inst.Instruction.Opcode = TGSI_OPCODE_MOV;
      inst.Instruction.NumDstRegs = 1;
      tgsi_transform_dst_reg(&inst.Dst[0], TGSI_FILE_TEMPORARY,
                             tr->imm_temp_base + i, TGSI_WRITEMASK_XYZW);
      inst.Instruction.NumSrcRegs = 1;
      tgsi_transform_src_reg(&inst.Src[0], TGSI_FILE_IMMEDIATE, i,
                             TGSI_SWIZZLE_X, TGSI_SWIZZLE_Y,
                             TGSI_SWIZZLE_Z, TGSI_SWIZZLE_W);
      ctx->emit_instruction(ctx, &inst);
   }
}

static void
yttrium_tgsi_indirect_array_inst(struct tgsi_transform_context *ctx,
                                 struct tgsi_full_instruction *inst)
{
   struct yttrium_tgsi_indirect_array_transform *tr =
      yttrium_tgsi_indirect_array_transform(ctx);

   if (tr->need_imm_loads) {
      for (unsigned i = 0; i < inst->Instruction.NumSrcRegs; i++) {
         if (inst->Src[i].Register.File == TGSI_FILE_IMMEDIATE &&
             inst->Src[i].Register.Indirect) {
            inst->Src[i].Register.File = TGSI_FILE_TEMPORARY;
            inst->Src[i].Register.Index =
               tr->imm_temp_base + inst->Src[i].Register.Index;
         }
      }
   }

   ctx->emit_instruction(ctx, inst);
}

static struct tgsi_token *
yttrium_shader_lower_tgsi_indirect_arrays(
   const struct yttrium_shader_state *shader)
{
   struct yttrium_tgsi_indirect_array_transform tr;
   const bool temp_indirect =
      shader->info.indirect_files & (1u << TGSI_FILE_TEMPORARY);
   const bool imm_indirect =
      shader->info.indirect_files & (1u << TGSI_FILE_IMMEDIATE);

   if (!temp_indirect && !imm_indirect)
      return NULL;

   memset(&tr, 0, sizeof(tr));
   tr.base.transform_declaration = yttrium_tgsi_indirect_array_decl;
   tr.base.transform_instruction = yttrium_tgsi_indirect_array_inst;
   tr.base.prolog = yttrium_tgsi_indirect_array_prolog;
   tr.original_max_temp =
      shader->info.file_max[TGSI_FILE_TEMPORARY] >= 0 ?
      (unsigned)shader->info.file_max[TGSI_FILE_TEMPORARY] : 0;
   tr.imm_count = shader->info.immediate_count;
   tr.need_imm_loads = imm_indirect && tr.imm_count > 0;
   tr.imm_temp_base = tr.original_max_temp + 1;
   tr.temp_array_last = tr.original_max_temp;
   if (tr.need_imm_loads)
      tr.temp_array_last = tr.imm_temp_base + tr.imm_count - 1;

   return tgsi_transform_shader(shader->tokens,
                                tgsi_num_tokens(shader->tokens) +
                                32 + tr.imm_count * 2,
                                &tr.base);
}

struct yttrium_tgsi_tg4_rewrite_context {
   struct tgsi_transform_context base;
   unsigned component_imm;
};

static void
yttrium_tgsi_rewrite_tg4_prolog(struct tgsi_transform_context *ctx)
{
   tgsi_transform_immediate_int_decl(ctx, 0, 1, 2, 3);
}

static void
yttrium_tgsi_rewrite_tg4_sampler_view_inst(
   struct tgsi_transform_context *ctx,
   struct tgsi_full_instruction *inst)
{
   struct yttrium_tgsi_tg4_rewrite_context *tr =
      (struct yttrium_tgsi_tg4_rewrite_context *)ctx;

   if (inst->Instruction.Opcode == TGSI_OPCODE_TG4) {
      unsigned component = TGSI_SWIZZLE_X;

      if (inst->Instruction.NumSrcRegs > 2 &&
          (inst->Src[2].Register.File == TGSI_FILE_SAMPLER ||
           inst->Src[2].Register.File == TGSI_FILE_SAMPLER_VIEW) &&
          !inst->Src[2].Register.Indirect) {
         component = inst->Src[2].Register.SwizzleX;
         if (component > TGSI_SWIZZLE_W)
            component = TGSI_SWIZZLE_X;
      }

      if (inst->Instruction.NumSrcRegs > 1 &&
          inst->Src[1].Register.File == TGSI_FILE_SAMPLER_VIEW &&
          !inst->Src[1].Register.Indirect) {
         inst->Src[1].Register.File = TGSI_FILE_IMMEDIATE;
         inst->Src[1].Register.Index = tr->component_imm;
         inst->Src[1].Register.SwizzleX = component;
         inst->Src[1].Register.SwizzleY = component;
         inst->Src[1].Register.SwizzleZ = component;
         inst->Src[1].Register.SwizzleW = component;
         inst->Src[1].Register.Dimension = 0;
         inst->Src[1].Register.Absolute = 0;
         inst->Src[1].Register.Negate = 0;
      }
   }

   ctx->emit_instruction(ctx, inst);
}

static struct tgsi_token *
yttrium_shader_rewrite_tgsi_tg4_sampler_view_sources(
   const struct yttrium_shader_state *shader,
   const struct tgsi_token *tokens)
{
   struct yttrium_tgsi_tg4_rewrite_context tr;

   if (!shader || !tokens || !shader->info.opcode_count[TGSI_OPCODE_TG4])
      return NULL;

   memset(&tr, 0, sizeof(tr));
   tr.component_imm = shader->info.immediate_count;
   tr.base.prolog = yttrium_tgsi_rewrite_tg4_prolog;
   tr.base.transform_instruction = yttrium_tgsi_rewrite_tg4_sampler_view_inst;
   return tgsi_transform_shader(tokens, tgsi_num_tokens(tokens), &tr.base);
}

static bool
yttrium_shader_scalarize_spirv_alu(const nir_instr *instr, const void *data)
{
   (void)data;

   if (instr->type != nir_instr_type_alu)
      return false;

   const nir_alu_instr *alu = nir_instr_as_alu((nir_instr *)instr);
   switch (alu->op) {
   case nir_op_ball_fequal2:
   case nir_op_ball_fequal3:
   case nir_op_ball_fequal4:
   case nir_op_bany_fnequal2:
   case nir_op_bany_fnequal3:
   case nir_op_bany_fnequal4:
   case nir_op_ball_iequal2:
   case nir_op_ball_iequal3:
   case nir_op_ball_iequal4:
   case nir_op_bany_inequal2:
   case nir_op_bany_inequal3:
   case nir_op_bany_inequal4:
      return true;
   /* SPIR-V requires the offset and count operands of OpBitFieldInsert and
    * OpBitFieldUExtract/SExtract to be integer scalars, while NIR carries
    * every source of these at the result's component count.  A vector form
    * therefore produces a module spirv-val rejects ("Expected Offset Type to
    * be int scalar").  Both HLSL and GLSL define those operands as scalars,
    * so scalarising the op costs nothing the source did not already imply.
    */
   case nir_op_bitfield_insert:
   case nir_op_ubitfield_extract:
   case nir_op_ibitfield_extract:
      return true;
   default:
      return false;
   }
}

static bool
yttrium_ntv_supports_intrinsic(nir_intrinsic_op intrinsic)
{
   switch (intrinsic) {
   case nir_intrinsic_decl_reg:
   case nir_intrinsic_load_reg:
   case nir_intrinsic_store_reg:
   case nir_intrinsic_vulkan_resource_index:
   case nir_intrinsic_load_vulkan_descriptor:
   case nir_intrinsic_terminate:
   case nir_intrinsic_demote:
   case nir_intrinsic_load_deref:
   case nir_intrinsic_store_deref:
   case nir_intrinsic_load_push_constant_zink:
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant:
   case nir_intrinsic_store_global:
   case nir_intrinsic_load_front_face:
   case nir_intrinsic_load_view_index:
   case nir_intrinsic_load_base_instance:
   case nir_intrinsic_load_instance_id:
   case nir_intrinsic_load_base_vertex:
   case nir_intrinsic_load_first_vertex:
   case nir_intrinsic_load_draw_id:
   case nir_intrinsic_load_vertex_id:
   case nir_intrinsic_load_primitive_id:
   case nir_intrinsic_load_invocation_id:
   case nir_intrinsic_load_sample_id:
   case nir_intrinsic_load_point_coord_maybe_flipped:
   case nir_intrinsic_load_point_coord:
   case nir_intrinsic_load_sample_pos:
   case nir_intrinsic_load_frag_coord:
   case nir_intrinsic_load_layer_id:
   case nir_intrinsic_load_sample_mask_in:
   case nir_intrinsic_emit_vertex:
   case nir_intrinsic_end_primitive:
   case nir_intrinsic_is_helper_invocation:
   case nir_intrinsic_load_helper_invocation:
   case nir_intrinsic_load_patch_vertices_in:
   case nir_intrinsic_load_tess_coord:
   case nir_intrinsic_barrier:
   case nir_intrinsic_interp_deref_at_centroid:
   case nir_intrinsic_interp_deref_at_sample:
   case nir_intrinsic_interp_deref_at_offset:
   case nir_intrinsic_deref_atomic:
   case nir_intrinsic_deref_atomic_swap:
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap:
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap:
   case nir_intrinsic_begin_invocation_interlock:
   case nir_intrinsic_end_invocation_interlock:
   case nir_intrinsic_get_ssbo_size:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_load_workgroup_id:
   case nir_intrinsic_load_num_workgroups:
   case nir_intrinsic_load_local_invocation_id:
   case nir_intrinsic_load_global_invocation_id:
   case nir_intrinsic_load_local_invocation_index:
   case nir_intrinsic_load_subgroup_id:
   case nir_intrinsic_load_subgroup_eq_mask:
   case nir_intrinsic_load_subgroup_ge_mask:
   case nir_intrinsic_load_subgroup_invocation:
   case nir_intrinsic_load_subgroup_le_mask:
   case nir_intrinsic_load_subgroup_lt_mask:
   case nir_intrinsic_load_subgroup_size:
   case nir_intrinsic_load_num_subgroups:
   case nir_intrinsic_ballot:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_load_workgroup_size:
   case nir_intrinsic_load_shared:
   case nir_intrinsic_store_shared:
   case nir_intrinsic_load_scratch:
   case nir_intrinsic_store_scratch:
   case nir_intrinsic_shader_clock:
   case nir_intrinsic_vote_all:
   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_vote_feq:
   case nir_intrinsic_is_sparse_resident_zink:
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddy_fine:
   case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddy_coarse:
   case nir_intrinsic_reduce:
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan:
   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
   case nir_intrinsic_rotate:
   case nir_intrinsic_shuffle:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_elect:
   case nir_intrinsic_set_vertex_and_primitive_count:
   case nir_intrinsic_store_task_payload:
   case nir_intrinsic_load_task_payload:
   case nir_intrinsic_launch_mesh_workgroups:
      return true;
   default:
      return false;
   }
}

static bool
yttrium_shader_preflight_spirv_intrinsics(
   const struct yttrium_shader_state *shader)
{
   unsigned unsupported_count = 0;

   if (!shader || !shader->nir)
      return false;

   nir_foreach_function_impl(impl, shader->nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (yttrium_ntv_supports_intrinsic(intr->intrinsic))
               continue;

            if (unsupported_count < 8) {
               YTTRIUM_WARN("yttrium: shader_compile_spirv skipped stage=%s id=%u reason=unsupported_intrinsic intrinsic=%s op=%u token_hash=0x%llx inputs=%u outputs=%u textures=%u ubos=%u ssbos=%u images=%u writes_memory=%u\n",
                            yttrium_shader_stage_name(shader->stage),
                            shader->id,
                            nir_intrinsic_infos[intr->intrinsic].name,
                            (unsigned)intr->intrinsic,
                            (unsigned long long)shader->token_hash,
                            shader->info.num_inputs,
                            shader->info.num_outputs,
                            shader->nir->info.num_textures,
                            shader->nir->info.num_ubos,
                            shader->nir->info.num_ssbos,
                            shader->nir->info.num_images,
                            shader->nir->info.writes_memory);
            }
            unsupported_count++;
         }
      }
   }

   if (unsupported_count) {
      YTTRIUM_WARN("yttrium: shader_compile_spirv skipped stage=%s id=%u reason=unsupported_intrinsics count=%u token_hash=0x%llx\n",
                   yttrium_shader_stage_name(shader->stage),
                   shader->id,
                   unsupported_count,
                   (unsigned long long)shader->token_hash);
      return false;
   }

   return true;
}

static bool
yttrium_shader_preflight_spirv_distance_io(
   const struct yttrium_shader_state *shader)
{
   if (!shader || !shader->nir)
      return false;

   /*
    * Legacy VS/GS distance I/O can still be represented as vec4 here; NTV
    * performs its established compact-array lowering while emitting SPIR-V.
    * The tessellation interfaces are the ones that must already carry the
    * nested per-control-point array shape, and are the only interfaces this
    * preflight is intended to protect.
    */
   if (shader->nir->info.stage != MESA_SHADER_TESS_CTRL &&
       shader->nir->info.stage != MESA_SHADER_TESS_EVAL)
      return true;

   nir_foreach_variable_with_modes(var, shader->nir,
                                   nir_var_shader_in |
                                   nir_var_shader_out) {
      if (var->data.location != VARYING_SLOT_CLIP_DIST0 &&
          var->data.location != VARYING_SLOT_CULL_DIST0)
         continue;

      const struct glsl_type *component_type =
         glsl_without_array(var->type);
      const bool valid =
         glsl_type_is_array(var->type) &&
         glsl_type_is_scalar(component_type) &&
         glsl_get_base_type(component_type) == GLSL_TYPE_FLOAT &&
         glsl_get_bit_size(component_type) == 32;
      if (valid)
         continue;

      YTTRIUM_WARN("yttrium: shader_compile_spirv skipped stage=%s id=%u reason=invalid_distance_io location=%u mode=0x%x array=%u component_base=%u component_bits=%u component_width=%u token_hash=0x%llx\n",
                   yttrium_shader_stage_name(shader->stage),
                   shader->id, var->data.location, var->data.mode,
                   glsl_type_is_array(var->type),
                   (unsigned)glsl_get_base_type(component_type),
                   glsl_get_bit_size(component_type),
                   glsl_get_vector_elements(component_type),
                   (unsigned long long)shader->token_hash);
      return false;
   }

   return true;
}

static bool
yttrium_shader_lower_draw_id_bases_instr(nir_builder *b,
                                               nir_intrinsic_instr *instr,
                                               void *data)
{
   (void)data;

   if (instr->intrinsic != nir_intrinsic_load_vertex_id_zero_base &&
       instr->intrinsic != nir_intrinsic_load_instance_id)
      return false;

   b->cursor = nir_after_instr(&instr->instr);
   nir_def *zero_base;
   if (instr->intrinsic == nir_intrinsic_load_instance_id) {
      /* SPIR-V InstanceIndex includes firstInstance; D3D/Gallium INSTANCEID
       * does not.  Direct draws currently bind rebased VBs with firstInstance
       * zero, while native indirect draws carry the original GPU arguments. */
      zero_base = nir_isub(b, &instr->def, nir_load_base_instance(b));
   } else {
      nir_def *vertex_index = nir_load_vertex_id(b);
      nir_def *first_vertex = nir_load_first_vertex(b);
      zero_base = nir_isub(b, vertex_index, first_vertex);
   }

   nir_def_rewrite_uses_after(&instr->def, zero_base);
   if (instr->intrinsic != nir_intrinsic_load_instance_id)
      nir_instr_remove(&instr->instr);
   return true;
}

static bool
yttrium_shader_lower_draw_id_bases(struct yttrium_shader_state *shader)
{
   if (!shader || !shader->nir ||
       shader->nir->info.stage != MESA_SHADER_VERTEX)
      return false;

   return nir_shader_intrinsics_pass(
      shader->nir, yttrium_shader_lower_draw_id_bases_instr,
      nir_metadata_control_flow, NULL);
}

static bool
yttrium_shader_lower_explicit_lod_tie_break_tex(nir_builder *b,
                                                nir_tex_instr *tex,
                                                void *data)
{
   const float bias = *(const float *)data;

   if (tex->op != nir_texop_txl)
      return false;

   const int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   if (lod_idx < 0)
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *lod = tex->src[lod_idx].src.ssa;
   nir_def *positive_infinity =
      nir_imm_floatN_t(b, INFINITY, lod->bit_size);
   if (bias > 0.0f) {
      nir_def *zero = nir_imm_floatN_t(b, 0.0, lod->bit_size);
      nir_def *biased_lod = nir_fadd_imm(b, lod, bias);
      lod = nir_bcsel(b, nir_flt(b, zero, lod), biased_lod, lod);
   }
   nir_src_rewrite(&tex->src[lod_idx].src,
                   nir_nextafter(b, lod, positive_infinity));
   return true;
}

static bool
yttrium_shader_lower_explicit_lod_tie_break(
   struct yttrium_shader_state *shader,
   uint32_t mipmap_precision_bits)
{
   if (!shader || !shader->nir)
      return false;

   float bias = 0.0f;
   if (mipmap_precision_bits > 0 && mipmap_precision_bits <= 23) {
      /* For positive LODs, cross half of the host's fixed-point quantum, then
       * nextafter() below selects the positive side of the tie.  A single
       * float ULP is otherwise discarded before sampling on limited-precision
       * hardware.  Keep LOD zero in the base mip: a finite positive bias can
       * introduce mip-1 filtering on implementations that round it upward.
       */
      bias = ldexpf(0.5f, -(int)mipmap_precision_bits);
   }

   return nir_shader_tex_pass(shader->nir,
                              yttrium_shader_lower_explicit_lod_tie_break_tex,
                              nir_metadata_control_flow, (void *)&bias);
}

static bool
yttrium_shader_sampler_var_raw_slot(mesa_shader_stage shader_stage,
                                    const nir_variable *var,
                                    uint32_t *raw_slot)
{
   if (shader_stage < MESA_SHADER_VERTEX ||
       shader_stage >= YTTRIUM_VENUS_SAMPLED_STAGE_COUNT ||
       !var || !raw_slot)
      return false;

   uint32_t binding = var->data.binding;
   if (binding >= YTTRIUM_SHADER_SAMPLED_IMAGE_BINDING_BASE) {
      mesa_shader_stage stage = MESA_SHADER_NONE;
      unsigned decoded_slot = 0;
      if (!yttrium_shader_sampler_binding_decode(binding, &stage,
                                                 &decoded_slot) ||
          stage != shader_stage)
         return false;
      binding = decoded_slot;
   }

   if (binding >= PIPE_MAX_SAMPLERS)
      return false;

   *raw_slot = binding;
   return true;
}

static bool
yttrium_shader_sampler_slot_is_unorm_buffer(
   const struct yttrium_shader_state *shader,
   uint32_t slot)
{
   return shader &&
          slot < ARRAY_SIZE(shader->info.sampler_targets) &&
          slot < ARRAY_SIZE(shader->info.sampler_type) &&
          shader->info.sampler_targets[slot] == TGSI_TEXTURE_BUFFER &&
          shader->info.sampler_type[slot] == TGSI_RETURN_TYPE_UNORM;
}

struct yttrium_unorm_buffer_lower_state {
   uint32_t slot_mask;
   mesa_shader_stage stage;
};

static bool
yttrium_shader_lower_unorm_buffer_sampler_tex(nir_builder *b,
                                              nir_tex_instr *tex,
                                              void *data)
{
   const struct yttrium_unorm_buffer_lower_state *state = data;

   if (!state || !tex || tex->op != nir_texop_txf ||
       tex->sampler_dim != GLSL_SAMPLER_DIM_BUF)
      return false;

   const int texture_idx =
      nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (texture_idx < 0)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(tex->src[texture_idx].src);
   nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : NULL;
   uint32_t raw_slot = 0;
   if (!yttrium_shader_sampler_var_raw_slot(state->stage, var, &raw_slot) ||
       raw_slot >= 32 || !(state->slot_mask & BITFIELD_BIT(raw_slot)))
      return false;

   tex->dest_type = nir_type_uint32;

   b->cursor = nir_after_instr(&tex->instr);
   nir_def *as_float_bits = nir_bitcast_vector(b, &tex->def, 32);
   nir_def_rewrite_uses_after(&tex->def, as_float_bits);
   return true;
}

static bool
yttrium_shader_lower_unorm_buffer_samplers(
   struct yttrium_shader_state *shader)
{
   if (!shader || !shader->nir)
      return false;

   struct yttrium_unorm_buffer_lower_state state = {
      .slot_mask = 0,
      .stage = shader->stage,
   };

   for (uint32_t slot = 0;
        slot < MIN2(PIPE_MAX_SHADER_SAMPLER_VIEWS,
                    ARRAY_SIZE(shader->info.sampler_targets));
        slot++) {
      if (yttrium_shader_sampler_slot_is_unorm_buffer(shader, slot) &&
          slot < 32)
         state.slot_mask |= BITFIELD_BIT(slot);
   }
   if (!state.slot_mask)
      return false;

   nir_foreach_variable_with_modes(var, shader->nir, nir_var_uniform) {
      const struct glsl_type *type = glsl_without_array(var->type);
      if (!glsl_type_is_sampler(type) ||
          glsl_get_sampler_dim(type) != GLSL_SAMPLER_DIM_BUF)
         continue;

      uint32_t raw_slot = 0;
      if (!yttrium_shader_sampler_var_raw_slot(shader->stage, var,
                                               &raw_slot) ||
          raw_slot >= 32 || !(state.slot_mask & BITFIELD_BIT(raw_slot)))
         continue;

      var->type = glsl_sampler_type(GLSL_SAMPLER_DIM_BUF, false, false,
                                    GLSL_TYPE_UINT);
   }

   const bool progress = nir_shader_tex_pass(
      shader->nir, yttrium_shader_lower_unorm_buffer_sampler_tex,
      nir_metadata_control_flow, &state);
   if (!progress)
      return false;

   nir_fixup_deref_types(shader->nir);
   nir_shader_gather_info(shader->nir,
                          nir_shader_get_entrypoint(shader->nir));
   nir_tgsi_scan_shader(shader->nir, &shader->info, true);

   for (uint32_t slot = 0;
        slot < MIN2(PIPE_MAX_SHADER_SAMPLER_VIEWS,
                    ARRAY_SIZE(shader->info.sampler_targets));
        slot++) {
      if (slot < 32 && (state.slot_mask & BITFIELD_BIT(slot)))
         shader->info.sampler_type[slot] = TGSI_RETURN_TYPE_UINT;
   }

   yttrium_trace_debug_stringf(
      "yttrium: shader_compile_nir lowered unorm buffer samplers stage=%s id=%u token_hash=0x%llx",
      yttrium_shader_stage_name(shader->stage), shader->id,
      (unsigned long long)shader->token_hash);
   return true;
}

struct yttrium_ubo_lower_state {
   struct yttrium_shader_state *shader;
   uint32_t used_mask;
   bool failed;
};

static int
yttrium_first_set_bit(uint32_t mask)
{
   for (unsigned i = 0; i < 32; i++) {
      if (mask & (1u << i))
         return i;
   }
   return -1;
}

static unsigned
yttrium_bit_count32(uint32_t mask)
{
   unsigned count = 0;
   while (mask) {
      mask &= mask - 1;
      count++;
   }
   return count;
}

static bool
yttrium_shader_collect_ubo_loads(struct yttrium_shader_state *shader,
                                 uint32_t *used_mask)
{
   if (used_mask)
      *used_mask = 0;
   if (!shader || !shader->nir || !used_mask)
      return false;

   nir_foreach_function_impl(impl, shader->nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_ubo)
               continue;

            if (!nir_src_is_const(intr->src[0])) {
               YTTRIUM_WARN("yttrium: shader_compile_spirv skipped stage=%s id=%u reason=unsupported_ubo_indirect_index token_hash=0x%llx ubos=%u\n",
                            yttrium_shader_stage_name(shader->stage),
                            shader->id,
                            (unsigned long long)shader->token_hash,
                            shader->nir->info.num_ubos);
               return false;
            }

            const unsigned index = nir_src_as_uint(intr->src[0]);
            if (index >= PIPE_MAX_CONSTANT_BUFFERS || index >= 32) {
               YTTRIUM_WARN("yttrium: shader_compile_spirv skipped stage=%s id=%u reason=unsupported_ubo_index index=%u token_hash=0x%llx ubos=%u\n",
                            yttrium_shader_stage_name(shader->stage),
                            shader->id,
                            index,
                            (unsigned long long)shader->token_hash,
                            shader->nir->info.num_ubos);
               return false;
            }

            if (intr->def.bit_size != 32) {
               YTTRIUM_WARN("yttrium: shader_compile_spirv skipped stage=%s id=%u reason=unsupported_ubo_bits bits=%u token_hash=0x%llx ubos=%u\n",
                            yttrium_shader_stage_name(shader->stage),
                            shader->id,
                            intr->def.bit_size,
                            (unsigned long long)shader->token_hash,
                            shader->nir->info.num_ubos);
               return false;
            }

            *used_mask |= 1u << index;
         }
      }
   }

   return true;
}

static unsigned
yttrium_shader_push_constant_capacity_words(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return YTTRIUM_SHADER_VS_PUSH_CONSTANT_BYTES / sizeof(uint32_t);
   case MESA_SHADER_FRAGMENT:
      return YTTRIUM_SHADER_FS_PUSH_CONSTANT_BYTES / sizeof(uint32_t);
   default:
      return 0;
   }
}

static uint16_t
yttrium_shader_push_constant_offset(mesa_shader_stage stage)
{
   return stage == MESA_SHADER_FRAGMENT ?
      YTTRIUM_SHADER_FS_PUSH_CONSTANT_OFFSET :
      YTTRIUM_SHADER_VS_PUSH_CONSTANT_OFFSET;
}

static bool
yttrium_shader_select_push_ubos(struct yttrium_shader_state *shader,
                                uint32_t *used_mask)
{
   const unsigned capacity_words = shader ?
      yttrium_shader_push_constant_capacity_words(shader->stage) : 0;
   const unsigned words_per_slot =
      BITSET_WORDS(YTTRIUM_SHADER_MAX_UBO_DWORDS);
   BITSET_WORD *accessed = NULL;
   uint16_t word_counts[PIPE_MAX_CONSTANT_BUFFERS] = { 0 };
   bool fixed[PIPE_MAX_CONSTANT_BUFFERS] = { 0 };
   bool consumed[PIPE_MAX_CONSTANT_BUFFERS] = { 0 };
   unsigned selected_words = 0;

   if (!shader || !shader->nir || !used_mask || !*used_mask ||
       !capacity_words ||
       !yttrium_gdi_static_ubo_sampled_cache_enabled())
      return true;

   accessed = calloc(PIPE_MAX_CONSTANT_BUFFERS * words_per_slot,
                     sizeof(*accessed));
   if (!accessed) {
      YTTRIUM_WARN("yttrium: static UBO sampled cache selection skipped stage=%s id=%u reason=out_of_memory\n",
                   yttrium_shader_stage_name(shader->stage), shader->id);
      return true;
   }

   for (unsigned raw_index = 0; raw_index < PIPE_MAX_CONSTANT_BUFFERS;
        raw_index++)
      fixed[raw_index] = (*used_mask & BITFIELD_BIT(raw_index)) != 0;

   nir_foreach_function_impl(impl, shader->nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_ubo ||
                !nir_src_is_const(intr->src[0]))
               continue;

            const unsigned raw_index = nir_src_as_uint(intr->src[0]);
            if (raw_index >= PIPE_MAX_CONSTANT_BUFFERS)
               continue;
            if (!nir_src_is_const(intr->src[1])) {
               fixed[raw_index] = false;
               continue;
            }

            const uint64_t byte_offset = nir_src_as_uint(intr->src[1]);
            const uint64_t byte_size =
               (uint64_t)intr->num_components * sizeof(uint32_t);
            if ((byte_offset & (sizeof(uint32_t) - 1)) ||
                byte_offset > YTTRIUM_SHADER_MAX_UBO_BYTES ||
                byte_size > YTTRIUM_SHADER_MAX_UBO_BYTES - byte_offset) {
               fixed[raw_index] = false;
               continue;
            }

            BITSET_WORD *slot_words =
               accessed + raw_index * words_per_slot;
            const unsigned first_word =
               (unsigned)(byte_offset / sizeof(uint32_t));
            for (unsigned component = 0;
                 component < intr->num_components; component++) {
               const unsigned word = first_word + component;
               if (!BITSET_TEST(slot_words, word)) {
                  BITSET_SET(slot_words, word);
                  word_counts[raw_index]++;
               }
            }
         }
      }
   }

   while (selected_words < capacity_words) {
      unsigned best = PIPE_MAX_CONSTANT_BUFFERS;
      for (unsigned raw_index = 0;
           raw_index < PIPE_MAX_CONSTANT_BUFFERS; raw_index++) {
         if (consumed[raw_index] || !fixed[raw_index] ||
             !word_counts[raw_index] ||
             word_counts[raw_index] > capacity_words - selected_words)
            continue;
         if (best == PIPE_MAX_CONSTANT_BUFFERS ||
             word_counts[raw_index] < word_counts[best] ||
             (word_counts[raw_index] == word_counts[best] &&
              raw_index < best))
            best = raw_index;
      }
      if (best == PIPE_MAX_CONSTANT_BUFFERS)
         break;

      consumed[best] = true;
      shader->push_ubo_mask |= BITFIELD_BIT(best);
      BITSET_WORD *slot_words = accessed + best * words_per_slot;
      for (unsigned word = 0; word < YTTRIUM_SHADER_MAX_UBO_DWORDS;
           word++) {
         if (!BITSET_TEST(slot_words, word))
            continue;
         shader->push_constant_source_slots[selected_words] = (uint8_t)best;
         shader->push_constant_source_words[selected_words] = (uint16_t)word;
         selected_words++;
      }
   }

   free(accessed);
   shader->push_constant_offset =
      yttrium_shader_push_constant_offset(shader->stage);
   shader->push_constant_word_count = (uint8_t)selected_words;
   *used_mask &= ~shader->push_ubo_mask;
   return true;
}

static nir_variable *
yttrium_shader_create_push_constant_var(struct yttrium_shader_state *shader)
{
   if (!shader || !shader->nir || !shader->push_constant_word_count)
      return NULL;

   struct glsl_struct_field field = { 0 };
   field.name = "words";
   field.type = glsl_array_type(glsl_uint_type(),
                                shader->push_constant_word_count,
                                sizeof(uint32_t));
   field.offset = shader->push_constant_offset;
   const struct glsl_type *struct_type =
      glsl_struct_type(&field, 1, "yttrium_push_constants", false);
   nir_variable *var =
      nir_variable_create(shader->nir, nir_var_mem_push_const,
                          struct_type, "yttrium_push_constants");
   if (var)
      var->data.location = INT_MAX;
   return var;
}

static nir_variable *
yttrium_shader_create_ubo_var(struct nir_shader *nir,
                              const char *name,
                              unsigned driver_location,
                              uint32_t binding)
{
   struct glsl_struct_field field = { 0 };
   field.name = "base";
   field.type =
      glsl_array_type(glsl_uvec4_type(),
                      YTTRIUM_SHADER_MAX_UBO_DWORDS / 4, 16);

   const struct glsl_type *struct_type =
      glsl_interface_type(&field, 1, GLSL_INTERFACE_PACKING_STD140,
                          false, name);

   nir_variable *var =
      nir_variable_create(nir, nir_var_mem_ubo, struct_type, name);
   if (!var)
      return NULL;

   var->interface_type = var->type;
   var->data.mode = nir_var_mem_ubo;
   var->data.descriptor_set = YTTRIUM_SHADER_UBO_SET;
   var->data.binding = binding;
   var->data.explicit_binding = 1;
   var->data.driver_location = driver_location;
   return var;
}

struct yttrium_remove_ubo_state {
   struct yttrium_shader_state *shader;
   nir_variable *push_constant_var;
   nir_variable *vars[PIPE_MAX_CONSTANT_BUFFERS];
   bool failed;
};

static int
yttrium_shader_find_push_constant_word(
   const struct yttrium_shader_state *shader,
   unsigned raw_index,
   unsigned source_word)
{
   if (!shader)
      return -1;

   for (unsigned i = 0; i < shader->push_constant_word_count; i++) {
      if (shader->push_constant_source_slots[i] == raw_index &&
          shader->push_constant_source_words[i] == source_word)
         return (int)i;
   }
   return -1;
}

static bool
yttrium_shader_remove_ubo_load_instr(nir_builder *b,
                                     nir_instr *instr,
                                     void *data)
{
   struct yttrium_remove_ubo_state *state = data;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_load_ubo)
      return false;

   if (!nir_src_is_const(intr->src[0])) {
      state->failed = true;
      return false;
   }

   const unsigned raw_index = nir_src_as_uint(intr->src[0]);
   if (raw_index < PIPE_MAX_CONSTANT_BUFFERS &&
       (state->shader->push_ubo_mask & BITFIELD_BIT(raw_index))) {
      if (!state->push_constant_var || !nir_src_is_const(intr->src[1])) {
         state->failed = true;
         return false;
      }

      const uint64_t byte_offset = nir_src_as_uint(intr->src[1]);
      if (byte_offset & (sizeof(uint32_t) - 1)) {
         state->failed = true;
         return false;
      }

      b->cursor = nir_before_instr(instr);
      nir_def *loads[4] = { 0 };
      nir_deref_instr *deref_var =
         nir_build_deref_var(b, state->push_constant_var);
      nir_deref_instr *deref_struct =
         nir_build_deref_struct(b, deref_var, 0);
      for (unsigned i = 0; i < intr->num_components; i++) {
         const unsigned source_word =
            (unsigned)(byte_offset / sizeof(uint32_t)) + i;
         const int push_word = yttrium_shader_find_push_constant_word(
            state->shader, raw_index, source_word);
         if (push_word < 0) {
            state->failed = true;
            return false;
         }

         nir_deref_instr *deref =
            nir_build_deref_array_imm(b, deref_struct, (unsigned)push_word);
         loads[i] = nir_load_deref(b, deref);
      }

      nir_def *load = intr->num_components == 1 ?
         loads[0] : nir_vec(b, loads, intr->num_components);
      nir_def_rewrite_uses(&intr->def, load);
      nir_instr_remove(instr);
      return true;
   }

   nir_variable *var =
      raw_index < PIPE_MAX_CONSTANT_BUFFERS ? state->vars[raw_index] : NULL;

   if (!var) {
      state->failed = true;
      return false;
   }

   b->cursor = nir_before_instr(instr);
   nir_deref_instr *deref_var = nir_build_deref_var(b, var);
   nir_deref_instr *deref_struct = nir_build_deref_struct(b, deref_var, 0);
   nir_def *loads[4] = { 0 };

   for (unsigned i = 0; i < intr->num_components; i++) {
      nir_def *dword_index;
      nir_def *vec_index;
      nir_def *component;
      if (nir_src_is_const(intr->src[1])) {
         const uint64_t byte_offset = nir_src_as_uint(intr->src[1]);
         const uint64_t dword_offset = byte_offset / sizeof(uint32_t) + i;
         vec_index = nir_imm_int(b, (int)(dword_offset / 4));
         component = nir_imm_int(b, (int)(dword_offset & 3));
      } else {
         dword_index =
            nir_iadd_imm(b, nir_udiv_imm(b, intr->src[1].ssa,
                                         sizeof(uint32_t)), i);
         vec_index = nir_udiv_imm(b, dword_index, 4);
         component = nir_iand_imm(b, dword_index, 3);
      }

      nir_deref_instr *deref = nir_build_deref_array(b, deref_struct,
                                                     vec_index);
      loads[i] =
         nir_vector_extract(b, nir_load_deref(b, deref), component);
   }

   nir_def *load = nir_vec(b, loads, intr->num_components);
   nir_def_rewrite_uses(&intr->def, load);
   nir_instr_remove(instr);
   return true;
}

static bool
yttrium_shader_lower_ubos(struct yttrium_shader_state *shader)
{
   uint32_t used_mask = 0;

   if (!shader || !shader->nir)
      return false;

   nir_opt_constant_folding(shader->nir);

   if (!yttrium_shader_collect_ubo_loads(shader, &used_mask))
      return false;

   if (!yttrium_shader_select_push_ubos(shader, &used_mask))
      return false;

   shader->ubo_used_mask = used_mask;
   shader->ubo_default = (used_mask & 1u) != 0;
   shader->ubo_first = 0;
   shader->ubo_count = 0;

   if (!used_mask && !shader->push_constant_word_count)
      return true;

   nir_shader *nir = shader->nir;

   nir_foreach_variable_with_modes_safe(var, nir, nir_var_mem_ubo)
      var->data.mode = nir_var_shader_temp;
   nir_fixup_deref_modes(nir);
   nir_remove_dead_variables(nir, nir_var_shader_temp, NULL);

   struct yttrium_remove_ubo_state remove_state = {
      .shader = shader,
   };
   if (shader->push_constant_word_count) {
      remove_state.push_constant_var =
         yttrium_shader_create_push_constant_var(shader);
      if (!remove_state.push_constant_var)
         return false;
   }

   const uint32_t explicit_mask = used_mask & ~1u;
   if (explicit_mask) {
      const int first = yttrium_first_set_bit(explicit_mask);
      if (first < 0)
         return false;
      shader->ubo_first = (uint8_t)first;
      shader->ubo_count = (uint8_t)yttrium_bit_count32(explicit_mask);
   }

   for (unsigned raw_index = 0; raw_index < PIPE_MAX_CONSTANT_BUFFERS;
        raw_index++) {
      if (!(used_mask & (1u << raw_index)))
         continue;

      const uint32_t binding =
         yttrium_shader_ubo_binding(shader->stage, raw_index);
      if (binding == UINT32_MAX)
         return false;

      char name[64];
      snprintf(name, sizeof(name), raw_index == 0 ?
               "yttrium_uniform_0" : "yttrium_ubo_%u", raw_index);
      remove_state.vars[raw_index] =
         yttrium_shader_create_ubo_var(nir, name, raw_index, binding);
      if (!remove_state.vars[raw_index])
         return false;
   }

   nir_shader_instructions_pass(nir, yttrium_shader_remove_ubo_load_instr,
                                nir_metadata_control_flow, &remove_state);
   if (remove_state.failed) {
      YTTRIUM_WARN("yttrium: shader_compile_spirv skipped stage=%s id=%u reason=ubo_lower_failed token_hash=0x%llx used_mask=0x%x\n",
                   yttrium_shader_stage_name(shader->stage),
                   shader->id,
                   (unsigned long long)shader->token_hash,
                   used_mask);
      return false;
   }

   nir_fixup_deref_modes(nir);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   return true;
}

static bool
yttrium_shader_make_nir(struct pipe_context *ctx,
                        const struct pipe_shader_state *state,
                        struct yttrium_shader_state *shader)
{
   const struct pipe_screen *screen = ctx ? ctx->screen : NULL;
   const unsigned token_processor =
      shader && shader->tokens ? tgsi_get_processor_type(shader->tokens) :
      MESA_SHADER_NONE;
   const bool token_processor_valid =
      token_processor < MESA_SHADER_MESH_STAGES;
   const bool stage_valid = shader && shader->stage < MESA_SHADER_MESH_STAGES;
   const bool stage_options =
      screen && stage_valid && screen->nir_options[shader->stage];
   const bool token_options =
      screen && token_processor_valid && screen->nir_options[token_processor];

   if (!ctx || !state || !shader) {
      YTTRIUM_WARN("yttrium: shader_compile_nir skipped reason=bad_args ctx=%p state=%p shader=%p\n",
                  ctx, state, shader);
      return false;
   }

   YTTRIUM_LOG("yttrium: shader_compile_nir begin stage=%s id=%u ir=%u tokens=%p token_count=%u token_processor=%u screen=%p stage_options=%u token_options=%u finalize_nir=%u\n",
               yttrium_shader_stage_name(shader->stage),
               shader->id,
               state->type,
               shader->tokens,
               shader->token_count,
               token_processor,
               screen,
               stage_options,
               token_options,
               screen && screen->finalize_nir);

   if (state->type == PIPE_SHADER_IR_TGSI) {
      const struct tgsi_token *tokens;
      struct tgsi_token *lowered_tokens;
      struct tgsi_token *stripped_tokens;
      const bool needs_indirect_lowering =
         shader->info.indirect_files &
         ((1u << TGSI_FILE_TEMPORARY) | (1u << TGSI_FILE_IMMEDIATE));

      if (!shader->tokens) {
         YTTRIUM_WARN("yttrium: shader_compile_nir failed stage=%s id=%u reason=no_tgsi_tokens state_tokens=%p token_count=%u\n",
                     yttrium_shader_stage_name(shader->stage),
                     shader->id,
                     state->tokens,
                     shader->token_count);
         return false;
      }

      lowered_tokens = yttrium_shader_lower_tgsi_indirect_arrays(shader);
      if (needs_indirect_lowering && !lowered_tokens) {
         YTTRIUM_WARN("yttrium: shader_compile_nir skipped stage=%s id=%u reason=tgsi_indirect_array_lowering_failed token_hash=0x%llx indirect_files=0x%x immediates=%u max_temp=%d\n",
                      yttrium_shader_stage_name(shader->stage),
                      shader->id,
                      (unsigned long long)shader->token_hash,
                      shader->info.indirect_files,
                      shader->info.immediate_count,
                      shader->info.file_max[TGSI_FILE_TEMPORARY]);
         return false;
      }
      tokens = lowered_tokens ? lowered_tokens : shader->tokens;
      stripped_tokens =
         yttrium_shader_rewrite_tgsi_tg4_sampler_view_sources(shader, tokens);
      if (shader->info.opcode_count[TGSI_OPCODE_TG4] &&
          !stripped_tokens) {
         YTTRIUM_WARN("yttrium: shader_compile_nir skipped stage=%s id=%u reason=tgsi_tg4_sampler_view_rewrite_failed token_hash=0x%llx sampler_views=%d\n",
                      yttrium_shader_stage_name(shader->stage),
                      shader->id,
                      (unsigned long long)shader->token_hash,
                      shader->info.file_max[TGSI_FILE_SAMPLER_VIEW] + 1);
         if (lowered_tokens)
            tgsi_free_tokens(lowered_tokens);
         return false;
      }
      tokens = stripped_tokens ? stripped_tokens : tokens;

      if (!yttrium_shader_preflight_tgsi_for_nir(shader, tokens)) {
         if (stripped_tokens)
            tgsi_free_tokens(stripped_tokens);
         if (lowered_tokens)
            tgsi_free_tokens(lowered_tokens);
         return false;
      }
      shader->nir = tgsi_to_nir(tokens, ctx->screen, false);
      if (stripped_tokens)
         tgsi_free_tokens(stripped_tokens);
      if (lowered_tokens)
         tgsi_free_tokens(lowered_tokens);
   } else if (state->type == PIPE_SHADER_IR_NIR) {
      shader->nir = state->ir.nir;
   } else {
      YTTRIUM_WARN("yttrium: shader_compile_nir failed stage=%s id=%u reason=unsupported_ir ir=%u\n",
                  yttrium_shader_stage_name(shader->stage),
                  shader->id,
                  state->type);
      return false;
   }

   if (!shader->nir) {
      YTTRIUM_WARN("yttrium: shader_compile_nir failed stage=%s id=%u reason=null_nir ir=%u token_processor=%u screen=%p stage_options=%u token_options=%u finalize_nir=%u\n",
                  yttrium_shader_stage_name(shader->stage),
                  shader->id,
                  state->type,
                  token_processor,
                  screen,
                  stage_options,
                  token_options,
                  screen && screen->finalize_nir);
      return false;
   }

   shader->nir->info.stage = shader->stage;
   if (state->type == PIPE_SHADER_IR_TGSI) {
      const struct yttrium_screen *yscreen = yttrium_screen(ctx->screen);
      yttrium_shader_lower_explicit_lod_tie_break(
         shader,
         yttrium_venus_mipmap_precision_bits(yscreen->venus));
   }
   nir_shader_gather_info(shader->nir,
                          nir_shader_get_entrypoint(shader->nir));
   nir_tgsi_scan_shader(shader->nir, &shader->info, true);
   yttrium_shader_trace_sampler_info(shader, "post_nir_scan");
   yttrium_shader_trace_nir_samplers(shader, "post_nir_scan");
   YTTRIUM_LOG("yttrium: shader_compile_nir success stage=%s id=%u nir=%p nir_stage=%u inputs=%u outputs=%u textures=%u ubos=%u ssbos=%u images=%u writes_memory=%u\n",
               yttrium_shader_stage_name(shader->stage),
               shader->id,
               shader->nir,
               shader->nir->info.stage,
               shader->info.num_inputs,
               shader->info.num_outputs,
               shader->nir->info.num_textures,
               shader->nir->info.num_ubos,
               shader->nir->info.num_ssbos,
               shader->nir->info.num_images,
               shader->nir->info.writes_memory);
   return true;
}

static bool
yttrium_shader_rebind_samplers(struct yttrium_shader_state *shader)
{
   if (!shader || !shader->nir)
      return false;

   nir_shader *nir = shader->nir;
   yttrium_shader_trace_nir_samplers(shader, "pre_sampler_rebind");
   nir_foreach_variable_with_modes(var, nir, nir_var_uniform) {
      const struct glsl_type *type = glsl_without_array(var->type);
      if (!glsl_type_is_sampler(type))
         continue;

      uint32_t binding = var->data.binding;
      if (binding >= YTTRIUM_SHADER_SAMPLED_IMAGE_BINDING_BASE) {
         mesa_shader_stage bound_stage = MESA_SHADER_NONE;
         if (!yttrium_shader_sampler_binding_decode(binding, &bound_stage,
                                                    NULL) ||
             bound_stage != shader->stage)
            return false;
      } else {
         binding = yttrium_shader_sampler_binding(shader->stage, binding);
      }
      if (binding == UINT32_MAX)
         return false;

      var->data.descriptor_set = YTTRIUM_SHADER_UBO_SET;
      var->data.binding = binding;
      var->data.explicit_binding = true;
   }

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   yttrium_shader_trace_nir_samplers(shader, "post_sampler_rebind");
   return true;
}

static bool
yttrium_shader_rebind_storage_images(struct yttrium_shader_state *shader)
{
   if (!shader || !shader->nir)
      return false;

   nir_shader *nir = shader->nir;
   uint64_t used_mask = 0;
   nir_foreach_variable_with_modes(var, nir, nir_var_image) {
      const struct glsl_type *type = glsl_without_array(var->type);
      if (!glsl_type_is_image(type))
         continue;

      uint32_t binding = var->data.binding;
      if (binding >= YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE) {
         if (binding >=
             YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE + PIPE_MAX_SHADER_IMAGES)
            return false;
         binding -= YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE;
      }
      if (binding >= PIPE_MAX_SHADER_IMAGES)
         return false;

      used_mask |= 1ull << binding;
      var->data.descriptor_set = YTTRIUM_SHADER_UBO_SET;
      var->data.binding = yttrium_shader_storage_image_binding(binding);
      var->data.explicit_binding = true;
   }

   shader->image_used_mask = used_mask;
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   return true;
}

/* ntv_shader_prepare() assigns descriptor bindings to driver_location for
 * non-I/O variables.  NTV then uses driver_location as an index into compact
 * sampler_var[] and image_var[] arrays.  Yttrium's descriptor bindings are
 * global across shader stages, so they must not be used as those indexes.
 * Preserve the global Vulkan binding while restoring NTV's compact raw slot.
 */
static bool
yttrium_shader_compact_ntv_descriptor_locations(
   struct yttrium_shader_state *shader)
{
   if (!shader || !shader->nir)
      return false;

   nir_foreach_variable_with_modes(var, shader->nir, nir_var_uniform) {
      const struct glsl_type *type = glsl_without_array(var->type);
      if (!glsl_type_is_sampler(type))
         continue;

      mesa_shader_stage stage = MESA_SHADER_NONE;
      unsigned raw_slot = 0;
      if (yttrium_shader_sampler_binding_decode(var->data.binding, &stage,
                                                &raw_slot)) {
         if (stage != shader->stage)
            return false;
      } else {
         if (var->data.binding >= PIPE_MAX_SHADER_SAMPLER_VIEWS)
            return false;
         raw_slot = var->data.binding;
      }

      var->data.driver_location = raw_slot;
   }

   nir_foreach_variable_with_modes(var, shader->nir, nir_var_image) {
      const struct glsl_type *type = glsl_without_array(var->type);
      if (!glsl_type_is_image(type))
         continue;

      unsigned raw_slot = var->data.binding;
      if (raw_slot >= YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE) {
         raw_slot -= YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE;
      }
      if (raw_slot >= PIPE_MAX_SHADER_IMAGES)
         return false;

      var->data.driver_location = raw_slot;
   }

   return true;
}

static void
yttrium_shader_mark_stream_output_vars(struct yttrium_shader_state *shader)
{
   nir_shader *nir = shader ? shader->nir : NULL;
   nir_xfb_info *xfb = nir ? nir->xfb_info : NULL;

   if (!shader || !nir || !xfb)
      return;

   for (unsigned i = 0; i < xfb->output_count; i++) {
      const nir_xfb_output_info *output = &xfb->outputs[i];

      nir_foreach_shader_out_variable(var, nir) {
         if (var->data.location != output->location)
            continue;

         const struct glsl_type *type = glsl_without_array(var->type);
         if (!glsl_type_is_vector_or_scalar(type))
            continue;

         const unsigned components = glsl_get_vector_elements(type);
         const unsigned first = var->data.location_frac;
         const unsigned var_mask = BITFIELD_RANGE(first, components);
         const unsigned xfb_mask = output->component_mask & var_mask;
         unsigned offset_component = first;

         if (!xfb_mask)
            continue;

         if (components > 1) {
            if (xfb_mask != var_mask)
               continue;
         } else {
            unsigned tmp = xfb_mask;
            offset_component = u_bit_scan(&tmp);
         }

         if (offset_component < output->component_offset)
            continue;

         var->data.explicit_xfb_buffer = 1;
         var->data.explicit_xfb_stride = 1;
         var->data.xfb.buffer = output->buffer;
         var->data.xfb.stride = xfb->buffers[output->buffer].stride;
         var->data.offset = output->offset +
            (offset_component - output->component_offset) * sizeof(uint32_t);
         var->data.stream = xfb->buffer_to_stream[output->buffer];
      }
   }
}

enum {
   YTTRIUM_SHADER_MAX_INTERSTAGE_LOCATIONS = 32,

   /* Keep generated cull-distance handoff varyings in the low generic range.
    * That range remains stable across separately compiled stages and leaves
    * the sparse D3D9 declaration usages above GENERIC[19] untouched. */
   YTTRIUM_SHADER_GENERATED_GENERIC_VARYING_COUNT = 20,

   /* Nine has no GENERIC[26] spelling for a COLOR0 declaration. */
   YTTRIUM_SHADER_LEGACY_COLOR0_LOCATION = 26,
   YTTRIUM_SHADER_LEGACY_BFC0_LOCATION = 27,
   YTTRIUM_SHADER_LEGACY_BFC1_LOCATION = 28,
};

static bool
yttrium_shader_interstage_location(unsigned slot, unsigned *location)
{
   if (!location)
      return false;

   if (slot >= VARYING_SLOT_VAR0 && slot < VARYING_SLOT_MAX) {
      const unsigned generic = slot - VARYING_SLOT_VAR0;
      if (generic >= YTTRIUM_SHADER_MAX_INTERSTAGE_LOCATIONS)
         return false;

      *location = generic;
      return true;
   }

   if (slot >= VARYING_SLOT_PATCH0 && slot < VARYING_SLOT_TESS_MAX) {
      *location = slot - VARYING_SLOT_PATCH0;
      return true;
   }

   /* Use the same location for legacy and GENERIC spellings of the same D3D9
    * semantic.  This keeps locations stable when old and new shader models
    * are mixed without reserving 12 locations ahead of every generic. */
   if (slot >= VARYING_SLOT_TEX0 && slot <= VARYING_SLOT_TEX7) {
      *location = slot - VARYING_SLOT_TEX0;
      return true;
   }

   switch (slot) {
   case VARYING_SLOT_COL0:
      *location = YTTRIUM_SHADER_LEGACY_COLOR0_LOCATION;
      return true;
   case VARYING_SLOT_COL1:
      *location = 8; /* Nine's GENERIC spelling of COLOR1. */
      return true;
   case VARYING_SLOT_FOGC:
      *location = 16; /* Nine's GENERIC spelling of FOG. */
      return true;
   case VARYING_SLOT_BFC0:
      *location = YTTRIUM_SHADER_LEGACY_BFC0_LOCATION;
      return true;
   case VARYING_SLOT_BFC1:
      *location = YTTRIUM_SHADER_LEGACY_BFC1_LOCATION;
      return true;
   default:
      break;
   }

   if (slot > VARYING_SLOT_POS && slot < VARYING_SLOT_VAR0 &&
       slot < YTTRIUM_SHADER_MAX_INTERSTAGE_LOCATIONS) {
      *location = slot;
      return true;
   }

   return false;
}

static unsigned
yttrium_shader_find_free_varying_slots(const nir_shader *nir,
                                       unsigned slot_count)
{
   if (!slot_count ||
       slot_count > YTTRIUM_SHADER_GENERATED_GENERIC_VARYING_COUNT)
      return VARYING_SLOT_MAX;

   const unsigned max_slot =
      VARYING_SLOT_VAR0 + YTTRIUM_SHADER_GENERATED_GENERIC_VARYING_COUNT -
      slot_count;

   uint32_t used_locations = 0;
   u_foreach_bit64(slot, nir->info.outputs_written) {
      unsigned location = 0;
      if (yttrium_shader_interstage_location(slot, &location))
         used_locations |= BITFIELD_BIT(location);
   }

   for (unsigned slot = max_slot + 1;
        slot-- > VARYING_SLOT_VAR0;) {
      bool free = true;
      for (unsigned i = 0; i < slot_count; i++) {
         unsigned location = 0;
         if ((nir->info.outputs_written & BITFIELD64_BIT(slot + i)) ||
             !yttrium_shader_interstage_location(slot + i, &location) ||
             (used_locations & BITFIELD_BIT(location))) {
            free = false;
            break;
         }
      }
      if (free)
         return slot;
   }

   return VARYING_SLOT_MAX;
}

static bool
yttrium_shader_cull_distance_store_base_component(nir_deref_instr *deref,
                                                  nir_variable *cull,
                                                  unsigned *base_component)
{
   if (!deref || !cull || !base_component ||
       nir_deref_instr_get_variable(deref) != cull)
      return false;

   nir_deref_path path;
   nir_deref_path_init(&path, deref, NULL);

   bool ok = true;
   unsigned component = 0;
   nir_deref_instr **p = &path.path[1];

   if (*p) {
      if ((*p)->deref_type != nir_deref_type_array ||
          !nir_src_is_const((*p)->arr.index)) {
         ok = false;
         goto out;
      }

      component = nir_src_as_uint((*p)->arr.index);
      p++;
   }

   if (*p)
      ok = false;

out:
   nir_deref_path_finish(&path);
   if (!ok)
      return false;

   *base_component = component;
   return true;
}

static bool
yttrium_shader_copy_cull_distance_store(nir_builder *b,
                                        nir_intrinsic_instr *intrin,
                                        nir_variable *cull,
                                        nir_variable **generic,
                                        unsigned cull_size)
{
   if (intrin->intrinsic != nir_intrinsic_store_deref)
      return false;

   unsigned base_component = 0;
   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
   if (!yttrium_shader_cull_distance_store_base_component(
          deref, cull, &base_component))
      return false;

   nir_def *value = intrin->src[1].ssa;
   unsigned write_mask = nir_intrinsic_write_mask(intrin);
   if (!write_mask)
      write_mask = BITFIELD_MASK(value->num_components);

   bool copied = false;
   for (unsigned i = 0; i < value->num_components; i++) {
      if (!(write_mask & BITFIELD_BIT(i)))
         continue;

      const unsigned component = base_component + i;
      if (component >= cull_size)
         continue;

      const unsigned slot = component / 4;
      const unsigned channel = component % 4;
      nir_def *scalar = value->num_components == 1 ?
         value : nir_channel(b, value, i);

      nir_build_write_masked_store(b, nir_build_deref_var(b, generic[slot]),
                                   scalar, channel);
      copied = true;
   }

   return copied;
}

static bool
yttrium_shader_add_cull_distance_generic_outputs(nir_shader *nir,
                                                 unsigned cull_slot)
{
   if (!nir || nir->info.stage != MESA_SHADER_VERTEX ||
       !nir->info.cull_distance_array_size)
      return false;

   nir_variable *cull =
      nir_find_variable_with_location(nir, nir_var_shader_out,
                                      VARYING_SLOT_CULL_DIST0);
   if (!cull)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   if (!impl)
      return false;

   const unsigned cull_size = nir->info.cull_distance_array_size;
   const unsigned slot_count = DIV_ROUND_UP(cull_size, 4);
   nir_variable *generic[2] = { NULL };
   if (slot_count > ARRAY_SIZE(generic))
      return false;

   for (unsigned slot = 0; slot < slot_count; slot++) {
      generic[slot] =
         nir_create_variable_with_location(nir, nir_var_shader_out,
                                           cull_slot + slot,
                                           glsl_vec4_type());
      if (!generic[slot])
         return false;
      generic[slot]->data.interpolation = INTERP_MODE_NONE;
      generic[slot]->data.how_declared = nir_var_hidden;
      nir->info.outputs_written |= BITFIELD64_BIT(cull_slot + slot);
   }

   bool inserted = false;
   nir_foreach_block_safe(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

         nir_builder b = nir_builder_create(impl);
         b.cursor = nir_after_instr(instr);
         inserted |= yttrium_shader_copy_cull_distance_store(
            &b, intrin, cull, generic, cull_size);
      }
   }

   if (!inserted)
      return false;

   nir_shader_gather_info(nir, impl);
   return true;
}

struct yttrium_shader_state *
yttrium_shader_create_cull_distance_vs(struct pipe_context *ctx,
                                       const struct yttrium_shader_state *vs,
                                       unsigned *cull_distance_slot)
{
   struct yttrium_screen *screen = ctx ? yttrium_screen(ctx->screen) : NULL;

   if (!ctx || !screen || !vs || !vs->nir || !cull_distance_slot ||
       !vs->nir->info.cull_distance_array_size)
      return NULL;

   const unsigned slot_count =
      DIV_ROUND_UP(vs->nir->info.cull_distance_array_size, 4);
   const unsigned cull_slot =
      yttrium_shader_find_free_varying_slots(vs->nir, slot_count);
   if (cull_slot == VARYING_SLOT_MAX)
      return NULL;

   nir_shader *nir = nir_shader_clone(NULL, vs->nir);
   if (!nir)
      return NULL;

   if (!yttrium_shader_add_cull_distance_generic_outputs(nir, cull_slot)) {
      ralloc_free(nir);
      return NULL;
   }

   struct pipe_shader_state state;
   memset(&state, 0, sizeof(state));
   state.type = PIPE_SHADER_IR_NIR;
   state.ir.nir = nir;

   struct yttrium_shader_state *shader =
      yttrium_shader_state_create(ctx, &state, MESA_SHADER_VERTEX);
   if (!shader)
      return NULL;

   yttrium_shader_copy_sampler_binding_map(shader, vs);

   if (!yttrium_shader_state_has_module(shader)) {
      yttrium_shader_state_destroy(screen, shader);
      return NULL;
   }

   *cull_distance_slot = cull_slot;
   yttrium_trace_debug_stringf(
      "yttrium: shader_create_cull_distance_vs state=%p id=%u base_vs=%u cull=%u slot=%u",
      shader, shader->id, vs->id, vs->nir->info.cull_distance_array_size,
      cull_slot);
   return shader;
}

static bool
yttrium_shader_insert_cull_distance_gs_return(nir_shader *nir,
                                              unsigned cull_distance_slot,
                                              unsigned cull_distance_size)
{
   if (!nir) {
      YTTRIUM_WARN("yttrium: cull-distance GS insert failed no nir\n");
      return false;
   }
   if (nir->info.stage != MESA_SHADER_GEOMETRY) {
      YTTRIUM_WARN("yttrium: cull-distance GS insert failed stage=%u\n",
                   nir->info.stage);
      return false;
   }
   if (!nir->info.cull_distance_array_size) {
      YTTRIUM_WARN("yttrium: cull-distance GS insert failed cull_size=0 inputs=0x%llx outputs=0x%llx\n",
                   (unsigned long long)nir->info.inputs_read,
                   (unsigned long long)nir->info.outputs_written);
      return false;
   }

   if (!cull_distance_size || cull_distance_size > 8) {
      YTTRIUM_WARN("yttrium: cull-distance GS insert failed bad generic cull size=%u\n",
                   cull_distance_size);
      return false;
   }

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   if (!impl) {
      YTTRIUM_WARN("yttrium: cull-distance GS insert failed missing entrypoint\n");
      return false;
   }

   nir_builder b = nir_builder_create(impl);
   b.cursor = nir_before_impl(impl);

   nir_def *component_culled[8] = { 0 };
   for (unsigned vertex = 0; vertex < nir->info.gs.vertices_in; vertex++) {
      for (unsigned component = 0; component < cull_distance_size;
           component++) {
         const unsigned slot = cull_distance_slot + component / 4;
         const unsigned channel = component % 4;
         nir_variable *cull =
            nir_find_variable_with_location(nir, nir_var_shader_in, slot);
         if (!cull) {
            YTTRIUM_WARN("yttrium: cull-distance GS insert failed missing generic input slot=%u cull=%u inputs=0x%llx outputs=0x%llx\n",
                         slot, cull_distance_size,
                         (unsigned long long)nir->info.inputs_read,
                         (unsigned long long)nir->info.outputs_written);
            return false;
         }

         nir_deref_instr *vertex_deref =
            nir_build_deref_array_imm(&b, nir_build_deref_var(&b, cull),
                                      vertex);
         nir_def *value = nir_load_deref(&b, vertex_deref);
         nir_def *negative =
            nir_flt_imm(&b, nir_channel(&b, value, channel), 0.0f);
         if (vertex == 0)
            component_culled[component] = negative;
         else
            component_culled[component] =
               nir_iand(&b, component_culled[component], negative);
      }
   }

   nir_def *culled = component_culled[0];
   for (unsigned i = 1; i < cull_distance_size; i++)
      culled = nir_ior(&b, culled, component_culled[i]);

   nir_push_if(&b, culled);
   nir_jump(&b, nir_jump_return);
   nir_pop_if(&b, NULL);

   nir_shader_gather_info(nir, impl);
   return true;
}

struct yttrium_shader_state *
yttrium_shader_create_cull_distance_gs(struct pipe_context *ctx,
                                       const struct yttrium_shader_state *vs,
                                       enum mesa_prim primitive_type,
                                       bool passthrough_prim_id,
                                       unsigned cull_distance_slot)
{
   struct yttrium_screen *screen = ctx ? yttrium_screen(ctx->screen) : NULL;

   if (!ctx || !screen || !vs || !vs->nir ||
       !vs->nir->info.cull_distance_array_size)
      return NULL;

   nir_shader *nir =
      nir_create_passthrough_gs(&screen->nir_options, vs->nir,
                                primitive_type, primitive_type,
                                false, false, passthrough_prim_id);
   if (!nir) {
      YTTRIUM_WARN("yttrium: cull-distance GS create failed passthrough vs=%u primitive=%u\n",
                   vs->id, primitive_type);
      return NULL;
   }
   nir->info.clip_distance_array_size = vs->nir->info.clip_distance_array_size;
   nir->info.cull_distance_array_size = vs->nir->info.cull_distance_array_size;

   if (!yttrium_shader_insert_cull_distance_gs_return(
          nir, cull_distance_slot, vs->nir->info.cull_distance_array_size)) {
      ralloc_free(nir);
      return NULL;
   }

   struct pipe_shader_state state;
   memset(&state, 0, sizeof(state));
   state.type = PIPE_SHADER_IR_NIR;
   state.ir.nir = nir;

   struct yttrium_shader_state *shader =
      yttrium_shader_state_create(ctx, &state, MESA_SHADER_GEOMETRY);
   if (!shader) {
      YTTRIUM_WARN("yttrium: cull-distance GS create failed shader state vs=%u primitive=%u\n",
                   vs->id, primitive_type);
      return NULL;
   }

   if (!yttrium_shader_state_has_module(shader)) {
      YTTRIUM_WARN("yttrium: cull-distance GS create failed shader module id=%u vs=%u primitive=%u\n",
                   shader->id, vs->id, primitive_type);
      yttrium_shader_state_destroy(screen, shader);
      return NULL;
   }

   yttrium_trace_debug_stringf(
      "yttrium: shader_create_cull_distance_gs state=%p id=%u base_vs=%u primitive=%u cull=%u prim_id=%u",
      shader, shader->id, vs->id, primitive_type,
      vs->nir->info.cull_distance_array_size, passthrough_prim_id);
   return shader;
}

static bool
yttrium_shader_a8_rt_location_matches(unsigned location,
                                      uint32_t a8_rt_mask)
{
   if (location == FRAG_RESULT_COLOR)
      return (a8_rt_mask & 1u) != 0;

   if (location < FRAG_RESULT_DATA0 || location > FRAG_RESULT_DATA7)
      return false;

   return (a8_rt_mask & (1u << (location - FRAG_RESULT_DATA0))) != 0;
}

static bool
yttrium_shader_rewrite_a8_rt_store(nir_builder *b,
                                   nir_intrinsic_instr *intrin,
                                   uint32_t a8_rt_mask)
{
   if (intrin->intrinsic != nir_intrinsic_store_deref)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
   nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : NULL;
   if (!var || var->data.mode != nir_var_shader_out ||
       !yttrium_shader_a8_rt_location_matches(var->data.location,
                                              a8_rt_mask))
      return false;

   nir_def *value = intrin->src[1].ssa;
   if (!value || value->num_components != 4 || var->data.location_frac)
      return false;

   unsigned write_mask = nir_intrinsic_write_mask(intrin);
   if (!write_mask)
      write_mask = BITFIELD_MASK(value->num_components);
   if (!(write_mask & BITFIELD_BIT(3)))
      return false;

   nir_def *components[4];
   components[0] = nir_channel(b, value, 3);
   for (unsigned i = 1; i < 4; i++)
      components[i] = nir_channel(b, value, i);

   nir_src_rewrite(&intrin->src[1],
                   nir_vec(b, components, value->num_components));
   nir_intrinsic_set_write_mask(intrin, write_mask | BITFIELD_BIT(0));
   return true;
}

static bool
yttrium_shader_rewrite_a8_rt_outputs(nir_shader *nir,
                                     uint32_t a8_rt_mask)
{
   if (!nir || nir->info.stage != MESA_SHADER_FRAGMENT || !a8_rt_mask)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   if (!impl)
      return false;

   bool progress = false;
   nir_foreach_block_safe(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         nir_builder b = nir_builder_create(impl);
         b.cursor = nir_before_instr(instr);
         progress |=
            yttrium_shader_rewrite_a8_rt_store(&b, intrin, a8_rt_mask);
      }
   }

   if (progress)
      nir_shader_gather_info(nir, impl);
   return progress;
}

struct yttrium_shader_state *
yttrium_shader_create_a8_rt_fs(struct pipe_context *ctx,
                               const struct yttrium_shader_state *fs,
                               uint32_t a8_rt_mask)
{
   struct yttrium_screen *screen = ctx ? yttrium_screen(ctx->screen) : NULL;

   if (!ctx || !screen || !fs || !fs->nir ||
       fs->stage != MESA_SHADER_FRAGMENT || !a8_rt_mask)
      return NULL;

   nir_shader *nir = nir_shader_clone(NULL, fs->nir);
   if (!nir)
      return NULL;

   if (!yttrium_shader_rewrite_a8_rt_outputs(nir, a8_rt_mask)) {
      ralloc_free(nir);
      return NULL;
   }

   struct pipe_shader_state state;
   memset(&state, 0, sizeof(state));
   state.type = PIPE_SHADER_IR_NIR;
   state.ir.nir = nir;

   struct yttrium_shader_state *shader =
      yttrium_shader_state_create(ctx, &state, MESA_SHADER_FRAGMENT);
   if (!shader)
      return NULL;

   yttrium_shader_copy_sampler_binding_map(shader, fs);
   shader->token_hash =
      fs->token_hash ^ 0xa800000000000000ull ^ (uint64_t)a8_rt_mask;

   if (!yttrium_shader_state_has_module(shader)) {
      YTTRIUM_WARN("yttrium: A8 RT FS create failed shader module id=%u base_fs=%u mask=0x%x\n",
                   shader->id, fs->id, a8_rt_mask);
      yttrium_shader_state_destroy(screen, shader);
      return NULL;
   }

   yttrium_trace_debug_stringf(
      "yttrium: shader_create_a8_rt_fs state=%p id=%u base_fs=%u mask=0x%x",
      shader, shader->id, fs->id, a8_rt_mask);
   return shader;
}

static bool
yttrium_shader_alpha_test_location_matches(const nir_variable *var)
{
   return var &&
          var->data.mode == nir_var_shader_out &&
          var->data.index == 0 &&
          var->data.location_frac == 0 &&
          (var->data.location == FRAG_RESULT_COLOR ||
           var->data.location == FRAG_RESULT_DATA0);
}

static nir_def *
yttrium_shader_alpha_test_fail(nir_builder *b, nir_def *alpha,
                               uint32_t alpha_func, float alpha_ref_value)
{
   nir_def *ref =
      nir_imm_floatN_t(b, alpha_ref_value, alpha->bit_size);

   switch (alpha_func) {
   case PIPE_FUNC_NEVER:
      return nir_imm_true(b);
   case PIPE_FUNC_LESS:
      return nir_fge(b, alpha, ref);
   case PIPE_FUNC_EQUAL:
      return nir_fneu(b, alpha, ref);
   case PIPE_FUNC_LEQUAL:
      return nir_flt(b, ref, alpha);
   case PIPE_FUNC_GREATER:
      return nir_fge(b, ref, alpha);
   case PIPE_FUNC_NOTEQUAL:
      return nir_feq(b, alpha, ref);
   case PIPE_FUNC_GEQUAL:
      return nir_flt(b, alpha, ref);
   case PIPE_FUNC_ALWAYS:
   default:
      return NULL;
   }
}

static bool
yttrium_shader_rewrite_alpha_test_outputs(nir_shader *nir,
                                          uint32_t alpha_func,
                                          float alpha_ref_value)
{
   if (!nir || nir->info.stage != MESA_SHADER_FRAGMENT ||
       alpha_func == PIPE_FUNC_ALWAYS)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   if (!impl)
      return false;

   if (alpha_func == PIPE_FUNC_NEVER) {
      nir_builder b = nir_builder_create(impl);
      b.cursor = nir_before_impl(impl);
      nir_discard(&b);
      nir_shader_gather_info(nir, impl);
      return true;
   }

   bool progress = false;
   nir_foreach_block_safe(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (intrin->intrinsic != nir_intrinsic_store_deref)
            continue;

         nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
         nir_variable *var = deref ? nir_deref_instr_get_variable(deref) :
                                     NULL;
         if (!yttrium_shader_alpha_test_location_matches(var))
            continue;

         nir_def *value = intrin->src[1].ssa;
         if (!value || value->num_components < 4)
            continue;

         unsigned write_mask = nir_intrinsic_write_mask(intrin);
         if (!write_mask)
            write_mask = BITFIELD_MASK(value->num_components);
         if (!(write_mask & BITFIELD_BIT(3)))
            continue;

         nir_builder b = nir_builder_create(impl);
         b.cursor = nir_before_instr(instr);
         nir_def *fail =
            yttrium_shader_alpha_test_fail(&b, nir_channel(&b, value, 3),
                                           alpha_func, alpha_ref_value);
         if (!fail)
            continue;

         nir_push_if(&b, fail);
         nir_discard(&b);
         nir_pop_if(&b, NULL);
         progress = true;
      }
   }

   if (progress)
      nir_shader_gather_info(nir, impl);
   return progress;
}

struct yttrium_shader_state *
yttrium_shader_create_alpha_test_fs(struct pipe_context *ctx,
                                    const struct yttrium_shader_state *fs,
                                    uint32_t alpha_func,
                                    float alpha_ref_value)
{
   struct yttrium_screen *screen = ctx ? yttrium_screen(ctx->screen) : NULL;

   if (!ctx || !screen || !fs || !fs->nir ||
       fs->stage != MESA_SHADER_FRAGMENT ||
       alpha_func == PIPE_FUNC_ALWAYS)
      return NULL;

   nir_shader *nir = nir_shader_clone(NULL, fs->nir);
   if (!nir)
      return NULL;

   if (!yttrium_shader_rewrite_alpha_test_outputs(nir, alpha_func,
                                                  alpha_ref_value)) {
      ralloc_free(nir);
      return NULL;
   }

   struct pipe_shader_state state;
   memset(&state, 0, sizeof(state));
   state.type = PIPE_SHADER_IR_NIR;
   state.ir.nir = nir;

   struct yttrium_shader_state *shader =
      yttrium_shader_state_create(ctx, &state, MESA_SHADER_FRAGMENT);
   if (!shader)
      return NULL;

   yttrium_shader_copy_sampler_binding_map(shader, fs);
   union {
      float f;
      uint32_t u;
   } ref_bits = { alpha_ref_value };
   shader->token_hash =
      fs->token_hash ^ 0xa17e570000000000ull ^
      ((uint64_t)alpha_func << 32) ^ (uint64_t)ref_bits.u;

   if (!yttrium_shader_state_has_module(shader)) {
      YTTRIUM_WARN("yttrium: alpha-test FS create failed shader module id=%u base_fs=%u func=%u ref=%f\n",
                   shader->id, fs->id, alpha_func, alpha_ref_value);
      yttrium_shader_state_destroy(screen, shader);
      return NULL;
   }

   yttrium_trace_debug_stringf(
      "yttrium: shader_create_alpha_test_fs state=%p id=%u base_fs=%u func=%u ref=%f",
      shader, shader->id, fs->id, alpha_func, alpha_ref_value);
   return shader;
}

static bool
yttrium_shader_rewrite_dual_source_outputs(nir_shader *nir)
{
   if (!nir || nir->info.stage != MESA_SHADER_FRAGMENT)
      return false;

   bool progress = false;
   nir_foreach_shader_out_variable(var, nir) {
      if (var->data.location == FRAG_RESULT_DATA1 && var->data.index == 0) {
         var->data.location = FRAG_RESULT_DATA0;
         var->data.index = 1;
         progress = true;
      }
   }

   if (progress)
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   return progress;
}

struct yttrium_shader_state *
yttrium_shader_create_dual_source_fs(struct pipe_context *ctx,
                                     const struct yttrium_shader_state *fs)
{
   struct yttrium_screen *screen = ctx ? yttrium_screen(ctx->screen) : NULL;

   if (!ctx || !screen || !fs || !fs->nir ||
       fs->stage != MESA_SHADER_FRAGMENT)
      return NULL;

   nir_shader *nir = nir_shader_clone(NULL, fs->nir);
   if (!nir)
      return NULL;

   if (!yttrium_shader_rewrite_dual_source_outputs(nir)) {
      ralloc_free(nir);
      return NULL;
   }

   struct pipe_shader_state state;
   memset(&state, 0, sizeof(state));
   state.type = PIPE_SHADER_IR_NIR;
   state.ir.nir = nir;

   struct yttrium_shader_state *shader =
      yttrium_shader_state_create(ctx, &state, MESA_SHADER_FRAGMENT);
   if (!shader)
      return NULL;

   yttrium_shader_copy_sampler_binding_map(shader, fs);
   shader->token_hash = fs->token_hash ^ 0xd51d000000000000ull;

   if (!yttrium_shader_state_has_module(shader)) {
      YTTRIUM_WARN("yttrium: dual-source FS create failed shader module id=%u base_fs=%u\n",
                   shader->id, fs->id);
      yttrium_shader_state_destroy(screen, shader);
      return NULL;
   }

   yttrium_trace_debug_stringf(
      "yttrium: shader_create_dual_source_fs state=%p id=%u base_fs=%u",
      shader, shader->id, fs->id);
   return shader;
}

bool
yttrium_shader_state_uses_sample_mask_in(
   const struct yttrium_shader_state *shader)
{
   if (!shader || shader->stage != MESA_SHADER_FRAGMENT)
      return false;

   if (shader->nir) {
      if (BITSET_TEST(shader->nir->info.system_values_read,
                      SYSTEM_VALUE_SAMPLE_MASK_IN))
         return true;

      nir_foreach_function_impl(impl, shader->nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;

               const nir_intrinsic_instr *intr =
                  nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_sample_mask_in)
                  return true;
            }
         }
      }
   }

   for (uint8_t i = 0; i < shader->info.num_system_values; i++) {
      if (shader->info.system_value_semantic_name[i] ==
          TGSI_SEMANTIC_SAMPLEMASK)
         return true;
   }

   return false;
}

static bool
yttrium_shader_forced_sample_interlock_output(const nir_variable *var)
{
   return var &&
          var->data.mode == nir_var_shader_out &&
          var->data.index == 0 &&
          var->data.location_frac == 0 &&
          (var->data.location == FRAG_RESULT_COLOR ||
           var->data.location == FRAG_RESULT_DATA0);
}

static bool
yttrium_shader_rewrite_forced_sample_interlock(nir_shader *nir,
                                                unsigned image_slot)
{
   if (!nir || nir->info.stage != MESA_SHADER_FRAGMENT ||
       image_slot >= PIPE_MAX_SHADER_IMAGES || nir->info.fs.uses_discard)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   if (!impl)
      return false;

   nir_intrinsic_instr *output_store = NULL;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (intrin->intrinsic != nir_intrinsic_store_deref)
            continue;

         nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
         nir_variable *var =
            deref ? nir_deref_instr_get_variable(deref) : NULL;
         if (!var || var->data.mode != nir_var_shader_out)
            continue;
         if (!yttrium_shader_forced_sample_interlock_output(var) ||
             output_store)
            return false;

         nir_def *value = intrin->src[1].ssa;
         unsigned write_mask = nir_intrinsic_write_mask(intrin);
         if (!value || !value->num_components ||
             (write_mask && !(write_mask & BITFIELD_BIT(0))))
            return false;
         output_store = intrin;
      }
   }
   if (!output_store)
      return false;

   const struct glsl_type *image_type =
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT);
   nir_variable *image =
      nir_variable_create(nir, nir_var_image, image_type,
                          "yttrium_forced_sample_r16");
   if (!image)
      return false;
   image->data.descriptor_set = YTTRIUM_SHADER_UBO_SET;
   image->data.binding = yttrium_shader_storage_image_binding(image_slot);
   image->data.explicit_binding = true;
   image->data.image.format = PIPE_FORMAT_R16_UINT;

   nir_builder b = nir_builder_create(impl);
   b.cursor = nir_before_instr(&output_store->instr);

   nir_def *value = output_store->src[1].ssa;
   nir_def *src = nir_iand_imm(&b, nir_channel(&b, value, 0), 0xffff);
   nir_def *coord = nir_pad_vec4(
      &b, nir_f2i32(&b, nir_trim_vector(&b, nir_load_frag_coord(&b), 2)));
   nir_def *image_deref = &nir_build_deref_var(&b, image)->def;
   nir_def *sample = nir_imm_int(&b, 0);
   nir_def *lod = nir_imm_int(&b, 0);

   nir_begin_invocation_interlock(&b);
   nir_def *old =
      nir_image_deref_load(&b, 4, 32, image_deref, coord, sample, lod,
                           .image_dim = GLSL_SAMPLER_DIM_2D,
                           .image_array = false,
                           .format = PIPE_FORMAT_R16_UINT,
                           .access = 0,
                           .dest_type = nir_type_uint32);
   nir_def *result = nir_ixor(&b, nir_channel(&b, old, 0), src);
   nir_def *stored =
      nir_vec4(&b, result, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
               nir_imm_int(&b, 0));
   nir_image_deref_store(&b, image_deref, coord, sample, stored, lod,
                         .image_dim = GLSL_SAMPLER_DIM_2D,
                         .image_array = false,
                         .format = PIPE_FORMAT_R16_UINT,
                         .access = 0,
                         .src_type = nir_type_uint32);
   nir_end_invocation_interlock(&b);
   nir_instr_remove(&output_store->instr);

   nir->info.fs.pixel_interlock_ordered = false;
   nir->info.fs.pixel_interlock_unordered = true;
   nir_remove_dead_variables(nir, nir_var_shader_out, NULL);
   nir_shader_gather_info(nir, impl);
   return true;
}

struct yttrium_shader_state *
yttrium_shader_create_forced_sample_interlock_fs(
   struct pipe_context *ctx,
   const struct yttrium_shader_state *fs,
   unsigned image_slot)
{
   struct yttrium_screen *screen = ctx ? yttrium_screen(ctx->screen) : NULL;

   if (!ctx || !screen || !fs || !fs->nir ||
       fs->stage != MESA_SHADER_FRAGMENT)
      return NULL;

   nir_shader *nir = nir_shader_clone(NULL, fs->nir);
   if (!nir)
      return NULL;

   if (!yttrium_shader_rewrite_forced_sample_interlock(nir, image_slot)) {
      ralloc_free(nir);
      return NULL;
   }

   struct pipe_shader_state state;
   memset(&state, 0, sizeof(state));
   state.type = PIPE_SHADER_IR_NIR;
   state.ir.nir = nir;

   struct yttrium_shader_state *shader =
      yttrium_shader_state_create(ctx, &state, MESA_SHADER_FRAGMENT);
   if (!shader)
      return NULL;

   yttrium_shader_copy_sampler_binding_map(shader, fs);
   shader->token_hash =
      fs->token_hash ^ 0x1e7e12c000000000ull ^ (uint64_t)image_slot;

   if (!yttrium_shader_state_has_module(shader)) {
      YTTRIUM_WARN("yttrium: forced-sample-interlock FS create failed shader module id=%u base_fs=%u slot=%u\n",
                   shader->id, fs->id, image_slot);
      yttrium_shader_state_destroy(screen, shader);
      return NULL;
   }

   return shader;
}

static bool
yttrium_shader_rewrite_forced_sample_mask_instr(
   nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   const unsigned sample_count = *(const unsigned *)data;

   if (intrin->intrinsic != nir_intrinsic_load_sample_mask_in ||
       sample_count <= 1)
      return false;

   const uint32_t mask =
      sample_count >= 32 ? UINT32_MAX : BITFIELD_MASK(sample_count);

   b->cursor = nir_before_instr(&intrin->instr);
   nir_def_replace(&intrin->def,
                   nir_imm_intN_t(b, mask, intrin->def.bit_size));
   BITSET_CLEAR(b->shader->info.system_values_read,
                SYSTEM_VALUE_SAMPLE_MASK_IN);
   return true;
}

static bool
yttrium_shader_rewrite_forced_sample_mask(nir_shader *nir,
                                          unsigned sample_count)
{
   if (!nir || nir->info.stage != MESA_SHADER_FRAGMENT ||
       sample_count <= 1)
      return false;

   const bool progress =
      nir_shader_intrinsics_pass(
         nir, yttrium_shader_rewrite_forced_sample_mask_instr,
         nir_metadata_control_flow, &sample_count);
   if (progress)
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   return progress;
}


struct yttrium_shader_state *
yttrium_shader_create_forced_sample_mask_fs(
   struct pipe_context *ctx,
   const struct yttrium_shader_state *fs,
   unsigned sample_count)
{
   struct yttrium_screen *screen = ctx ? yttrium_screen(ctx->screen) : NULL;

   if (!ctx || !screen || !fs || !fs->nir ||
       fs->stage != MESA_SHADER_FRAGMENT || sample_count <= 1 ||
       !yttrium_shader_state_uses_sample_mask_in(fs))
      return NULL;

   nir_shader *nir = nir_shader_clone(NULL, fs->nir);
   if (!nir)
      return NULL;

   if (!yttrium_shader_rewrite_forced_sample_mask(nir, sample_count)) {
      ralloc_free(nir);
      return NULL;
   }

   struct pipe_shader_state state;
   memset(&state, 0, sizeof(state));
   state.type = PIPE_SHADER_IR_NIR;
   state.ir.nir = nir;

   struct yttrium_shader_state *shader =
      yttrium_shader_state_create(ctx, &state, MESA_SHADER_FRAGMENT);
   if (!shader)
      return NULL;

   yttrium_shader_copy_sampler_binding_map(shader, fs);
   shader->token_hash =
      fs->token_hash ^ 0xf5a6c00000000000ull ^ (uint64_t)sample_count;

   if (!yttrium_shader_state_has_module(shader)) {
      YTTRIUM_WARN("yttrium: forced-sample-mask FS create failed shader module id=%u base_fs=%u samples=%u\n",
                   shader->id, fs->id, sample_count);
      yttrium_shader_state_destroy(screen, shader);
      return NULL;
   }

   return shader;
}

struct yttrium_sample_mask_expand {
   unsigned hw_sample_count;
   unsigned app_sample_count;
};

static bool
yttrium_shader_expand_sample_mask_instr(nir_builder *b,
                                        nir_intrinsic_instr *intrin,
                                        void *data)
{
   const struct yttrium_sample_mask_expand *expand = data;
   const unsigned factor = expand->app_sample_count / expand->hw_sample_count;

   if (intrin->intrinsic != nir_intrinsic_load_sample_mask_in)
      return false;

   b->cursor = nir_after_instr(&intrin->instr);

   /* The rasterizer only runs at hw_sample_count samples, so SV_Coverage has
    * that many valid bits.  Shaders that turn coverage into an alpha value
    * divide by the sample count the app asked for, so replicate every
    * hardware bit `factor` times to keep popcount()/app_sample_count equal to
    * the fraction the hardware actually measured.
    */
   nir_def *hw_mask = &intrin->def;
   nir_def *expanded = nir_imm_intN_t(b, 0, hw_mask->bit_size);
   for (unsigned i = 0; i < expand->hw_sample_count; i++) {
      nir_def *bit = nir_iand_imm(b, nir_ushr_imm(b, hw_mask, i), 1);
      const uint64_t group = (uint64_t)BITFIELD_MASK(factor) << (i * factor);

      expanded = nir_ior(b, expanded, nir_imul_imm(b, bit, group));
   }

   nir_def_rewrite_uses_after(hw_mask, expanded);
   return true;
}

struct yttrium_shader_state *
yttrium_shader_create_sample_mask_expand_fs(
   struct pipe_context *ctx,
   const struct yttrium_shader_state *fs,
   unsigned hw_sample_count,
   unsigned app_sample_count)
{
   struct yttrium_screen *screen = ctx ? yttrium_screen(ctx->screen) : NULL;
   const struct yttrium_sample_mask_expand expand = {
      .hw_sample_count = hw_sample_count,
      .app_sample_count = app_sample_count,
   };

   if (!ctx || !screen || !fs || !fs->nir ||
       fs->stage != MESA_SHADER_FRAGMENT ||
       hw_sample_count <= 1 || app_sample_count <= hw_sample_count ||
       app_sample_count > 32 || (app_sample_count % hw_sample_count) != 0 ||
       !yttrium_shader_state_uses_sample_mask_in(fs))
      return NULL;

   nir_shader *nir = nir_shader_clone(NULL, fs->nir);
   if (!nir)
      return NULL;

   if (!nir_shader_intrinsics_pass(nir,
                                   yttrium_shader_expand_sample_mask_instr,
                                   nir_metadata_control_flow,
                                   (void *)&expand)) {
      ralloc_free(nir);
      return NULL;
   }
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   struct pipe_shader_state state;
   memset(&state, 0, sizeof(state));
   state.type = PIPE_SHADER_IR_NIR;
   state.ir.nir = nir;

   struct yttrium_shader_state *shader =
      yttrium_shader_state_create(ctx, &state, MESA_SHADER_FRAGMENT);
   if (!shader)
      return NULL;

   yttrium_shader_copy_sampler_binding_map(shader, fs);
   shader->token_hash =
      fs->token_hash ^ 0x5e6a9d0000000000ull ^
      ((uint64_t)hw_sample_count << 32) ^ (uint64_t)app_sample_count;

   if (!yttrium_shader_state_has_module(shader)) {
      YTTRIUM_WARN("yttrium: sample-mask-expand FS create failed shader module id=%u base_fs=%u hw_samples=%u app_samples=%u\n",
                   shader->id, fs->id, hw_sample_count, app_sample_count);
      yttrium_shader_state_destroy(screen, shader);
      return NULL;
   }

   return shader;
}

static void
yttrium_shader_prepare_stream_output_vars(struct yttrium_shader_state *shader)
{
   nir_shader *nir = shader ? shader->nir : NULL;
   nir_xfb_info *xfb = nir ? nir->xfb_info : NULL;
   bool scalarize_outputs = false;

   if (!nir || !xfb || !xfb->output_count)
      return;

   for (unsigned i = 0; i < xfb->output_count; i++) {
      const nir_xfb_output_info *output = &xfb->outputs[i];
      const unsigned xfb_components = util_bitcount(output->component_mask);
      nir_variable *var =
         nir_find_variable_with_location(nir, nir_var_shader_out,
                                         output->location);
      const struct glsl_type *type =
         var ? glsl_without_array(var->type) : NULL;

      if (type && glsl_type_is_vector_or_scalar(type) &&
          glsl_get_vector_elements(type) != xfb_components)
         scalarize_outputs = true;
   }

   if (scalarize_outputs)
      nir_lower_io_vars_to_scalar(nir, nir_var_shader_out);

   yttrium_shader_mark_stream_output_vars(shader);
}

static bool
yttrium_shader_stream_output_location(nir_shader *nir,
                                      unsigned register_index,
                                      unsigned *location)
{
   if (!nir || !location)
      return false;

   bool found = false;
   unsigned mapped_location = 0;

   nir_foreach_shader_out_variable(var, nir) {
      if (var->data.driver_location != register_index)
         continue;

      if (var->data.patch || var->data.location < 0 ||
          (unsigned)var->data.location >= VARYING_SLOT_MAX ||
          (var->data.per_view && nir->options &&
           nir->options->per_view_unique_driver_locations))
         return false;

      const unsigned candidate = (unsigned)var->data.location;
      if (found && candidate != mapped_location)
         return false;
      mapped_location = candidate;
      found = true;
   }

   if (!found)
      return false;

   *location = mapped_location;
   return true;
}

static void
yttrium_shader_dump_stream_output_mapping_failure(
   const struct yttrium_shader_state *shader)
{
   static volatile LONG dumped;
   char path[MAX_PATH];
   FILE *file;

   if (!shader || InterlockedCompareExchange(&dumped, 1, 0) != 0)
      return;

   CreateDirectoryA("C:\\ProgramData\\Yttrium", NULL);
   CreateDirectoryA(YTTRIUM_SHADER_DUMP_DIR, NULL);

   if (shader->tokens) {
      snprintf(path, sizeof(path),
               YTTRIUM_SHADER_DUMP_DIR
               "\\yttrium_%s_pid%lu_so_unmapped_%016llx.tgsi",
               yttrium_shader_stage_name(shader->stage),
               (unsigned long)GetCurrentProcessId(),
               (unsigned long long)shader->token_hash);
      path[sizeof(path) - 1] = '\0';
      file = fopen(path, "w");
      if (file) {
         yttrium_write_shader_capture_metadata(file, shader);
         for (unsigned i = 0; i < shader->stream_output.num_outputs; i++) {
            const struct pipe_stream_output *output =
               &shader->stream_output.output[i];
            fprintf(file,
                    "stream_output[%u] register=%u start=%u count=%u "
                    "buffer=%u offset=%u stream=%u\n",
                    i, output->register_index, output->start_component,
                    output->num_components, output->output_buffer,
                    output->dst_offset, output->stream);
         }
         tgsi_dump_to_file(shader->tokens, 0, file);
         fclose(file);
         YTTRIUM_WARN("yttrium: stream-output mapping failure TGSI wrote path=%s\n",
                      path);
      }
   }

   if (shader->nir) {
      snprintf(path, sizeof(path),
               YTTRIUM_SHADER_DUMP_DIR
               "\\yttrium_%s_pid%lu_so_unmapped_%016llx.nir",
               yttrium_shader_stage_name(shader->stage),
               (unsigned long)GetCurrentProcessId(),
               (unsigned long long)shader->token_hash);
      path[sizeof(path) - 1] = '\0';
      file = fopen(path, "w");
      if (file) {
         yttrium_write_shader_capture_metadata(file, shader);
         nir_print_shader(shader->nir, file);
         fclose(file);
         YTTRIUM_WARN("yttrium: stream-output mapping failure NIR wrote path=%s\n",
                      path);
      }
   }
}

static void
yttrium_shader_write_stream_output_metadata(
   FILE *file, const struct yttrium_shader_state *shader)
{
   if (!file || !shader)
      return;

   for (unsigned i = 0; i < PIPE_MAX_SO_BUFFERS; i++) {
      fprintf(file, "stream_output_stride[%u]=%u\n", i,
              shader->stream_output.stride[i]);
   }

   for (unsigned i = 0; i < shader->stream_output.num_outputs; i++) {
      const struct pipe_stream_output *output =
         &shader->stream_output.output[i];
      fprintf(file,
              "stream_output[%u] register=%u start=%u count=%u "
              "buffer=%u offset=%u stream=%u\n",
              i, output->register_index, output->start_component,
              output->num_components, output->output_buffer,
              output->dst_offset, output->stream);
   }
}

static void
yttrium_shader_dump_stream_output_success(
   const struct yttrium_shader_state *shader)
{
   char path[MAX_PATH];
   FILE *file;
   const DWORD pid = GetCurrentProcessId();

   if (!yttrium_gdi_debug_get_bool_option(
          "D3D10UMD_YTTRIUM_SHADER_DUMP_STREAM_OUTPUT_SUCCESS", false))
      return;

   if (!shader || !shader->stream_output.num_outputs ||
       (shader->stage != MESA_SHADER_VERTEX &&
        shader->stage != MESA_SHADER_GEOMETRY) ||
       !shader->spirv || !shader->spirv->words ||
       !shader->spirv->num_words)
      return;

   CreateDirectoryA("C:\\ProgramData\\Yttrium", NULL);
   CreateDirectoryA(YTTRIUM_SHADER_DUMP_DIR, NULL);

   snprintf(path, sizeof(path),
            YTTRIUM_SHADER_DUMP_DIR
            "\\yttrium_%s_pid%lu_id%u_so_%016llx.spv",
            yttrium_shader_stage_name(shader->stage),
            (unsigned long)pid, shader->id,
            (unsigned long long)shader->spirv_hash);
   path[sizeof(path) - 1] = '\0';
   file = fopen(path, "wb");
   if (file) {
      fwrite(shader->spirv->words, sizeof(uint32_t),
             shader->spirv->num_words, file);
      fclose(file);
      yttrium_trace_debug_stringf(
         "yttrium: shader_stream_output_dump wrote stage=%s pid=%lu id=%u "
         "kind=spirv module_id=%llu outputs=%u path=%s token_hash=0x%llx "
         "spirv_hash=0x%llx",
         yttrium_shader_stage_name(shader->stage), (unsigned long)pid,
         shader->id, (unsigned long long)shader->module_obj.id,
         shader->stream_output.num_outputs, path,
         (unsigned long long)shader->token_hash,
         (unsigned long long)shader->spirv_hash);
   }

   if (shader->tokens) {
      snprintf(path, sizeof(path),
               YTTRIUM_SHADER_DUMP_DIR
               "\\yttrium_%s_pid%lu_id%u_so_%016llx.tgsi",
               yttrium_shader_stage_name(shader->stage),
               (unsigned long)pid, shader->id,
               (unsigned long long)shader->spirv_hash);
      path[sizeof(path) - 1] = '\0';
      file = fopen(path, "w");
      if (file) {
         yttrium_write_shader_capture_metadata(file, shader);
         yttrium_shader_write_stream_output_metadata(file, shader);
         tgsi_dump_to_file(shader->tokens, 0, file);
         fclose(file);
         yttrium_trace_debug_stringf(
            "yttrium: shader_stream_output_dump wrote stage=%s pid=%lu "
            "id=%u kind=tgsi module_id=%llu outputs=%u path=%s "
            "token_hash=0x%llx spirv_hash=0x%llx",
            yttrium_shader_stage_name(shader->stage), (unsigned long)pid,
            shader->id, (unsigned long long)shader->module_obj.id,
            shader->stream_output.num_outputs, path,
            (unsigned long long)shader->token_hash,
            (unsigned long long)shader->spirv_hash);
      }
   }

   if (shader->nir) {
      snprintf(path, sizeof(path),
               YTTRIUM_SHADER_DUMP_DIR
               "\\yttrium_%s_pid%lu_id%u_so_%016llx.nir",
               yttrium_shader_stage_name(shader->stage),
               (unsigned long)pid, shader->id,
               (unsigned long long)shader->spirv_hash);
      path[sizeof(path) - 1] = '\0';
      file = fopen(path, "w");
      if (file) {
         yttrium_write_shader_capture_metadata(file, shader);
         yttrium_shader_write_stream_output_metadata(file, shader);
         nir_print_shader(shader->nir, file);
         fclose(file);
         yttrium_trace_debug_stringf(
            "yttrium: shader_stream_output_dump wrote stage=%s pid=%lu "
            "id=%u kind=nir module_id=%llu outputs=%u path=%s "
            "token_hash=0x%llx spirv_hash=0x%llx",
            yttrium_shader_stage_name(shader->stage), (unsigned long)pid,
            shader->id, (unsigned long long)shader->module_obj.id,
            shader->stream_output.num_outputs, path,
            (unsigned long long)shader->token_hash,
            (unsigned long long)shader->spirv_hash);
      }
   }
}

static bool
yttrium_shader_apply_stream_output(struct yttrium_shader_state *shader)
{
   if (!shader || !shader->nir || !shader->stream_output.num_outputs)
      return true;

   if (shader->stage == MESA_SHADER_FRAGMENT)
      return false;

   const struct pipe_stream_output_info *so = &shader->stream_output;
   if (so->num_outputs > PIPE_MAX_SO_OUTPUTS)
      return false;

   const uint64_t nir_outputs_written = shader->nir->info.outputs_written;

   nir_xfb_info *xfb =
      rzalloc_size(shader->nir, nir_xfb_info_size(so->num_outputs));
   if (!xfb)
      return false;

   for (unsigned i = 0; i < so->num_outputs; i++) {
      const struct pipe_stream_output *output = &so->output[i];

      if (output->output_buffer >= NIR_MAX_XFB_BUFFERS ||
          output->num_components == 0 ||
          output->num_components > 4 ||
          output->start_component + output->num_components > 4 ||
          output->stream >= NIR_MAX_XFB_STREAMS) {
         YTTRIUM_WARN("yttrium: stream-output apply rejected stage=%s id=%u "
                      "reason=invalid_declaration output=%u buffer=%u "
                      "start_component=%u "
                      "num_components=%u stream=%u register_index=%u "
                      "nir_outputs_written=0x%llx\n",
                      yttrium_shader_stage_name(shader->stage), shader->id,
                      i, output->output_buffer, output->start_component,
                      output->num_components, output->stream,
                      output->register_index,
                      (unsigned long long)nir_outputs_written);
         ralloc_free(xfb);
         return false;
      }

      unsigned location = 0;
      if (!yttrium_shader_stream_output_location(shader->nir,
                                                 output->register_index,
                                                 &location)) {
         YTTRIUM_WARN("yttrium: stream-output apply rejected stage=%s id=%u "
                      "reason=tgsi_output_register_unmapped output=%u "
                      "buffer=%u start_component=%u num_components=%u "
                      "stream=%u register_index=%u tgsi_output_count=%u "
                      "nir_outputs_written=0x%llx\n",
                      yttrium_shader_stage_name(shader->stage), shader->id,
                      i, output->output_buffer, output->start_component,
                      output->num_components, output->stream,
                      output->register_index, shader->info.num_outputs,
                      (unsigned long long)nir_outputs_written);
         yttrium_shader_dump_stream_output_mapping_failure(shader);
         ralloc_free(xfb);
         return false;
      }

      const uint8_t component_mask =
         ((1u << output->num_components) - 1) <<
         output->start_component;
      nir_xfb_output_info *xfb_output =
         &xfb->outputs[xfb->output_count++];

      xfb_output->buffer = output->output_buffer;
      xfb_output->offset = output->dst_offset * sizeof(uint32_t);
      xfb_output->location = location;
      xfb_output->component_mask = component_mask;
      xfb_output->component_offset = output->start_component;

      xfb->buffers_written |= 1u << output->output_buffer;
      xfb->streams_written |= 1u << output->stream;
      xfb->buffer_to_stream[output->output_buffer] = output->stream;
      xfb->buffers[output->output_buffer].stride =
         so->stride[output->output_buffer] * sizeof(uint32_t);
      shader->nir->info.xfb_stride[output->output_buffer] =
         so->stride[output->output_buffer];

   }

   if (!xfb->buffers_written || !xfb->streams_written ||
       !xfb->output_count) {
      ralloc_free(xfb);
      return false;
   }

   ralloc_free(shader->nir->xfb_info);
   shader->nir->xfb_info = xfb;
   shader->nir->info.has_transform_feedback_varyings = true;
   nir_io_add_intrinsic_xfb_info(shader->nir);
   yttrium_shader_prepare_stream_output_vars(shader);
   return true;
}

static void
yttrium_shader_restore_tgsi_geometry_info(struct yttrium_shader_state *shader)
{
   if (!shader || !shader->nir || !shader->tokens ||
       shader->stage != MESA_SHADER_GEOMETRY)
      return;

   const enum mesa_prim input_prim =
      (enum mesa_prim)shader->info.properties[TGSI_PROPERTY_GS_INPUT_PRIM];
   const enum mesa_prim output_prim =
      (enum mesa_prim)shader->info.properties[TGSI_PROPERTY_GS_OUTPUT_PRIM];

   shader->nir->info.gs.input_primitive = input_prim;
   shader->nir->info.gs.output_primitive = output_prim;
   shader->nir->info.gs.vertices_in = mesa_vertices_per_prim(input_prim);
   shader->nir->info.gs.vertices_out =
      shader->info.properties[TGSI_PROPERTY_GS_MAX_OUTPUT_VERTICES];
   shader->nir->info.gs.invocations =
      MAX2(shader->info.properties[TGSI_PROPERTY_GS_INVOCATIONS], 1);
}

static bool
yttrium_shader_prepare_shared_access_instr(nir_builder *b,
                                           nir_instr *instr,
                                           void *data)
{
   nir_intrinsic_instr *intrin;
   unsigned src_index;
   unsigned bit_size;
   unsigned divisor;

   (void)data;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   intrin = nir_instr_as_intrinsic(instr);
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_shared:
      src_index = 0;
      bit_size = intrin->def.bit_size;
      break;
   case nir_intrinsic_store_shared:
      src_index = 1;
      bit_size = nir_src_bit_size(intrin->src[0]);
      break;
   default:
      return false;
   }

   divisor = bit_size / 8;
   if (divisor <= 1)
      return false;

   b->cursor = nir_before_instr(instr);
   nir_src_rewrite(&intrin->src[src_index],
                   nir_udiv_imm(b, intrin->src[src_index].ssa, divisor));
   return true;
}

static bool
yttrium_shader_prepare_shared_access_for_spirv(nir_shader *nir)
{
   return nir_shader_instructions_pass(nir,
                                       yttrium_shader_prepare_shared_access_instr,
                                       nir_metadata_control_flow,
                                       NULL);
}

/* SPIR-V depth-reference samples have scalar results.  TGSI's legacy shadow
 * operations expose that scalar replicated across a vector, so preserve the
 * old interface explicitly before handing NIR to the zink SPIR-V backend.
 * Zink normally performs the equivalent lowering in lower_zs_swizzle_tex(),
 * but Yttrium invokes nir_to_spirv() directly and does not run that pass.
 */
static bool
yttrium_shader_lower_legacy_shadow_instr(nir_builder *b,
                                         nir_instr *instr,
                                         void *data)
{
   (void)data;

   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (!tex->is_shadow || tex->is_new_style_shadow ||
       tex->def.num_components <= 1 || tex->is_sparse ||
       tex->op == nir_texop_tg4)
      return false;

   const unsigned num_components = tex->def.num_components;
   tex->def.num_components = 1;
   tex->is_new_style_shadow = true;

   b->cursor = nir_after_instr(instr);
   nir_def *components[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < num_components; i++)
      components[i] = &tex->def;
   nir_def *splat = nir_vec(b, components, num_components);
   nir_def_rewrite_uses_after(&tex->def, splat);
   return true;
}

static bool
yttrium_shader_lower_legacy_shadow_for_spirv(nir_shader *nir)
{
   return nir_shader_instructions_pass(
      nir, yttrium_shader_lower_legacy_shadow_instr,
      nir_metadata_control_flow, NULL);
}

/* DXBC typed UAV declarations describe the numeric type, not the bound
 * view's channel layout.  The TGSI frontend uses R32 as a placeholder for
 * float/int/uint declarations; emitting that as a Vulkan image format loses
 * the other channels of RG/RGBA views.  Keep the image's GLSL numeric type,
 * but use Unknown for non-atomic texture images.  Buffers retain their typed
 * texel addressing, and native NIR helper images retain their exact formats.
 */
static bool
yttrium_shader_image_access(nir_intrinsic_instr *intr,
                            bool *read, bool *write, bool *atomic)
{
   *read = *write = *atomic = false;
   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
      *read = true;
      *atomic = nir_intrinsic_access(intr) & ACCESS_ATOMIC;
      return true;
   case nir_intrinsic_image_deref_store:
      *write = true;
      *atomic = nir_intrinsic_access(intr) & ACCESS_ATOMIC;
      return true;
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
      *read = *write = *atomic = true;
      return true;
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
      return true;
   default:
      return false;
   }
}

/* Record which storage-image slots the shader declares as arrayed, so the
 * pipeline can pick a view type that matches the SPIR-V.  Run after the
 * formatless-image pass, where bindings are final.
 */
static void
yttrium_shader_scan_image_arrays(struct yttrium_shader_state *shader)
{
   shader->image_array_mask = 0;

   nir_foreach_variable_with_modes(var, shader->nir, nir_var_image) {
      const struct glsl_type *type = glsl_without_array(var->type);
      if (!glsl_type_is_image(type) ||
          glsl_get_sampler_dim(type) == GLSL_SAMPLER_DIM_BUF ||
          !glsl_sampler_type_is_array(type))
         continue;

      const unsigned binding = var->data.binding;
      if (binding < YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE)
         continue;

      const unsigned slot = binding - YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE;
      if (slot >= 64)
         continue;

      shader->image_array_mask |= 1ull << slot;
   }
}

static bool
yttrium_shader_lower_formatless_images(struct pipe_context *ctx,
                                       struct yttrium_shader_state *shader)
{
   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   shader->formatless_image_mask = 0;
   shader->formatless_image_read_mask = 0;
   shader->formatless_image_write_mask = 0;
   shader->formatless_image_uint_mask = 0;
   shader->formatless_image_sint_mask = 0;

   nir_foreach_variable_with_modes(var, shader->nir, nir_var_image) {
      const struct glsl_type *type = glsl_without_array(var->type);
      if (!glsl_type_is_image(type) ||
          glsl_get_sampler_dim(type) == GLSL_SAMPLER_DIM_BUF)
         continue;
      /* Generated graphics variants clone the already lowered NIR.  Scan
       * their Unknown images again, without changing new typed helpers. */
      if (!shader->tokens && var->data.image.format != PIPE_FORMAT_NONE)
         continue;

      const unsigned binding = var->data.binding;
      if (glsl_type_is_array(var->type) ||
          binding < YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE ||
          binding >= YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE +
                     PIPE_MAX_SHADER_IMAGES) {
         YTTRIUM_WARN("yttrium: shader formatless image rejected owner=yttrium-shader stage=%s id=%u binding=%u reason=unsupported-image-binding-or-array\n",
                      yttrium_shader_stage_name(shader->stage), shader->id,
                      binding);
         return false;
      }

      bool read = false, write = false, atomic = false;
      nir_foreach_function_impl(impl, shader->nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               bool intr_read, intr_write, intr_atomic;
               if (!yttrium_shader_image_access(intr, &intr_read,
                                                &intr_write, &intr_atomic))
                  continue;
               nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
               if (nir_deref_instr_get_variable(deref) != var)
                  continue;
               if (deref->deref_type != nir_deref_type_var) {
                  YTTRIUM_WARN("yttrium: shader formatless image rejected owner=yttrium-shader stage=%s id=%u binding=%u reason=unsupported-image-dereference\n",
                               yttrium_shader_stage_name(shader->stage),
                               shader->id, binding);
                  return false;
               }
               read |= intr_read;
               write |= intr_write;
               atomic |= intr_atomic;
            }
         }
      }
      if (atomic) {
         if (var->data.image.format != PIPE_FORMAT_NONE)
            continue;
         YTTRIUM_WARN("yttrium: shader formatless image rejected owner=yttrium-shader stage=%s id=%u binding=%u reason=atomic-requires-typed-image\n",
                      yttrium_shader_stage_name(shader->stage), shader->id,
                      binding);
         return false;
      }
      if (!yttrium_venus_storage_image_without_format_supported(
             screen->venus, read, write)) {
         YTTRIUM_WARN("yttrium: shader formatless image rejected owner=yttrium-shader stage=%s id=%u binding=%u read=%u write=%u reason=missing-vulkan-feature\n",
                      yttrium_shader_stage_name(shader->stage), shader->id,
                      binding, read, write);
         return false;
      }
      const enum glsl_base_type numeric_type =
         glsl_get_sampler_result_type(type);
      if (numeric_type != GLSL_TYPE_FLOAT && numeric_type != GLSL_TYPE_UINT &&
          numeric_type != GLSL_TYPE_INT) {
         YTTRIUM_WARN("yttrium: shader formatless image rejected owner=yttrium-shader stage=%s id=%u binding=%u reason=unsupported-numeric-type type=%u\n",
                      yttrium_shader_stage_name(shader->stage), shader->id,
                      binding, numeric_type);
         return false;
      }
      const uint64_t bit =
         1ull << (binding - YTTRIUM_SHADER_STORAGE_IMAGE_BINDING_BASE);
      shader->formatless_image_mask |= bit;
      if (read)
         shader->formatless_image_read_mask |= bit;
      if (write)
         shader->formatless_image_write_mask |= bit;
      if (numeric_type == GLSL_TYPE_UINT)
         shader->formatless_image_uint_mask |= bit;
      if (numeric_type == GLSL_TYPE_INT)
         shader->formatless_image_sint_mask |= bit;
      var->data.image.format = PIPE_FORMAT_NONE;
      var->data.access &= ~(ACCESS_NON_READABLE | ACCESS_NON_WRITEABLE);
      if (!read)
         var->data.access |= ACCESS_NON_READABLE;
      if (!write)
         var->data.access |= ACCESS_NON_WRITEABLE;
   }

   nir_foreach_function_impl(impl, shader->nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            bool read, write, atomic;
            if (!yttrium_shader_image_access(intr, &read, &write, &atomic))
               continue;
            nir_variable *var =
               nir_deref_instr_get_variable(nir_src_as_deref(intr->src[0]));
            if (var && var->data.image.format == PIPE_FORMAT_NONE &&
                glsl_get_sampler_dim(glsl_without_array(var->type)) !=
                   GLSL_SAMPLER_DIM_BUF)
               nir_intrinsic_set_format(intr, PIPE_FORMAT_NONE);
         }
      }
   }
   return true;
}

/*
 * Recover the UBO mask from variables an earlier lowering pass left in the
 * NIR.  A generated shader (dual-source, A8 target, alpha test, interlock,
 * forced-sample, sample-mask expand) is cloned from NIR that has already been
 * through yttrium_shader_lower_ubos, so its UBO accesses are derefs on the
 * yttrium_ubo_N variables rather than load_ubo intrinsics.  The lowering pass
 * is guarded and does not run again for those, which left the mask at zero
 * while the variables - and their bindings - stayed in the shader.  The
 * pipeline layout is built from the mask, so the module ended up declaring a
 * descriptor the layout never did, and the Intel host driver divides by the
 * zero array_size it keeps for undeclared bindings and dies with SIGFPE.
 *
 * Key off the binding rather than the driver location: the descriptor
 * compaction run before nir_to_spirv rewrites driver locations, but the
 * bindings are what both the SPIR-V and the layout are built from.
 */
static uint32_t
yttrium_shader_lowered_ubo_mask(const struct yttrium_shader_state *shader)
{
   if (!shader || !shader->nir)
      return 0;

   /* TGSI default uniforms also declare a UBO at binding 0, which overlaps
    * the vertex-stage descriptor range.  That declaration alone does not
    * mean our lowering pass has run: load_ubo still needs conversion to
    * descriptor derefs or push constants.  Only recover bindings once all
    * of those raw loads have been lowered.
    */
   nir_foreach_function_impl(impl, shader->nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_ubo)
               return 0;
         }
      }
   }

   const uint32_t base = yttrium_shader_ubo_binding(shader->stage, 0);
   if (base == UINT32_MAX)
      return 0;

   uint32_t mask = 0;
   nir_foreach_variable_with_modes(var, shader->nir, nir_var_mem_ubo) {
      if (var->data.binding < base ||
          var->data.binding >= base + PIPE_MAX_CONSTANT_BUFFERS)
         continue;
      mask |= 1u << (var->data.binding - base);
   }

   return mask;
}

static bool
yttrium_shader_make_spirv(struct pipe_context *ctx,
                          struct yttrium_shader_state *shader)
{
   struct ntv_info ntv_info = {
      .is_native_vulkan = true,
      .spirv_version = SPIRV_VERSION(1, 0),
   };

   if (!shader || !shader->nir)
      return false;

   if (shader->nir->info.stage == MESA_SHADER_TESS_CTRL) {
      ntv_info.spirv_version = SPIRV_VERSION(1, 3);
      ntv_info.have_vulkan_memory_model = true;
   }

   /* Carry over what a previous lowering pass already put in this NIR; see
    * yttrium_shader_lowered_ubo_mask().  Zero here would tell the pipeline
    * layout the shader uses no constant buffers while the SPIR-V still
    * declares them. */
   const uint32_t lowered_ubo_mask = yttrium_shader_lowered_ubo_mask(shader);
   const uint32_t lowered_explicit_mask = lowered_ubo_mask & ~1u;

   shader->ubo_used_mask = lowered_ubo_mask;
   shader->ubo_default = (lowered_ubo_mask & 1u) != 0;
   shader->ubo_first = lowered_explicit_mask ?
      (uint8_t)yttrium_first_set_bit(lowered_explicit_mask) : 0;
   shader->ubo_count = lowered_explicit_mask ?
      (uint8_t)yttrium_bit_count32(lowered_explicit_mask) : 0;
   shader->sampler_used_mask = yttrium_shader_sampled_texture_mask(shader);
   shader->image_used_mask = shader->info.images_declared;

   shader->resource_free =
      yttrium_shader_info_resource_free(&shader->info) &&
      yttrium_nir_resource_free(shader->nir);
   shader->uniform_buffer_only =
      yttrium_nir_uniform_buffer_only(shader->nir) &&
      (yttrium_shader_info_uniform_buffer_only(&shader->info) ||
       shader->nir->info.num_ubos);
   shader->sampled_texture_only =
      yttrium_shader_info_sampled_texture_only(&shader->info) &&
      yttrium_nir_sampled_texture_only(shader->nir);
   shader->storage_image_only =
      yttrium_shader_info_storage_image_only(&shader->info) &&
      yttrium_nir_storage_image_only(shader->nir);
   shader->sampled_storage_image_only =
      yttrium_shader_info_sampled_storage_image_only(&shader->info) &&
      yttrium_nir_sampled_storage_image_only(shader->nir);

   if ((shader->sampled_texture_only || shader->sampled_storage_image_only) &&
       !yttrium_shader_rebind_samplers(shader)) {
      YTTRIUM_WARN("yttrium: shader_compile_spirv skipped stage=%s id=%u reason=sampler_rebind_failed token_hash=0x%llx samplers=0x%x\n",
                   yttrium_shader_stage_name(shader->stage),
                   shader->id,
                   (unsigned long long)shader->token_hash,
                   shader->sampler_used_mask);
      return false;
   }
   if (shader->sampled_texture_only || shader->sampled_storage_image_only)
      yttrium_shader_lower_unorm_buffer_samplers(shader);

   if ((shader->storage_image_only || shader->sampled_storage_image_only) &&
       !yttrium_shader_rebind_storage_images(shader)) {
      YTTRIUM_WARN("yttrium: shader_compile_spirv skipped stage=%s id=%u reason=image_rebind_failed token_hash=0x%llx images=0x%x\n",
                   yttrium_shader_stage_name(shader->stage),
                   shader->id,
                   (unsigned long long)shader->token_hash,
                   shader->info.images_declared);
      return false;
   }

   /* Already-lowered NIR keeps the mask recovered above; running the pass a
    * second time would rebuild variables the derefs no longer point at. */
   if (!lowered_ubo_mask && !shader->resource_free &&
       (shader->uniform_buffer_only || shader->sampled_texture_only ||
        shader->storage_image_only || shader->sampled_storage_image_only) &&
       shader->nir->info.num_ubos) {
      if (!yttrium_shader_lower_ubos(shader)) {
         YTTRIUM_WARN("yttrium: shader_compile_spirv skipped stage=%s id=%u reason=ubo_lower_failed_outer token_hash=0x%llx info_constbufs=0x%x nir_ubos=%u ubo_only=%u texture_only=%u\n",
                      yttrium_shader_stage_name(shader->stage),
                      shader->id,
                      (unsigned long long)shader->token_hash,
                      shader->info.const_buffers_declared,
                      shader->nir->info.num_ubos,
                      shader->uniform_buffer_only,
                      shader->sampled_texture_only);
         return false;
      }
   }

   if (!yttrium_shader_apply_stream_output(shader)) {
      YTTRIUM_WARN("yttrium: shader_compile_spirv skipped stage=%s id=%u reason=stream_output_apply_failed token_hash=0x%llx so_outputs=%u\n",
                   yttrium_shader_stage_name(shader->stage),
                   shader->id,
                   (unsigned long long)shader->token_hash,
                   shader->stream_output.num_outputs);
      return false;
   }

   /* The TGSI compatibility transform groups every temporary with indirect
    * temporary/immediate data in one function-local array.  A single dynamic
    * access otherwise prevents ntv_shader_prepare() from promoting any of the
    * ordinary temporaries to SSA, turning the whole shader into load/store
    * traffic.  Resolve small TGSI-generated arrays before ntv optimization;
    * leave native NIR and large arrays alone to avoid unbounded code growth.
    */
   const unsigned tgsi_indirect_files =
      (1u << TGSI_FILE_TEMPORARY) | (1u << TGSI_FILE_IMMEDIATE);
   if (shader->tokens &&
       (shader->info.indirect_files & tgsi_indirect_files) &&
       nir_lower_indirect_derefs_to_if_else_trees(
          shader->nir, nir_var_function_temp, 64)) {
      nir_shader_gather_info(shader->nir,
                             nir_shader_get_entrypoint(shader->nir));
   }

   if (nir_lower_alu_to_scalar(shader->nir,
                               yttrium_shader_scalarize_spirv_alu, NULL))
      nir_shader_gather_info(shader->nir,
                             nir_shader_get_entrypoint(shader->nir));

   if (yttrium_shader_lower_legacy_shadow_for_spirv(shader->nir))
      nir_shader_gather_info(shader->nir,
                             nir_shader_get_entrypoint(shader->nir));

   ntv_shader_prepare(shader->nir);
   if (!yttrium_shader_compact_ntv_descriptor_locations(shader)) {
      YTTRIUM_WARN("yttrium: shader_compile_spirv skipped stage=%s id=%u reason=descriptor_driver_location_failed token_hash=0x%llx\n",
                   yttrium_shader_stage_name(shader->stage), shader->id,
                   (unsigned long long)shader->token_hash);
      return false;
   }
   yttrium_shader_prepare_shared_access_for_spirv(shader->nir);
   if (shader->nir->info.stage == MESA_SHADER_FRAGMENT &&
       shader->nir->info.fs.uses_discard &&
       nir_lower_discard_if(shader->nir, nir_lower_terminate_if_to_cf))
      nir_shader_gather_info(shader->nir,
                             nir_shader_get_entrypoint(shader->nir));
   if (yttrium_shader_lower_draw_id_bases(shader))
      nir_shader_gather_info(shader->nir,
                             nir_shader_get_entrypoint(shader->nir));
   yttrium_shader_restore_tgsi_geometry_info(shader);

   if (!yttrium_shader_preflight_spirv_intrinsics(shader)) {
      YTTRIUM_WARN("yttrium: shader_compile_spirv skipped stage=%s id=%u reason=intrinsic_preflight_failed token_hash=0x%llx\n",
                   yttrium_shader_stage_name(shader->stage),
                   shader->id,
                   (unsigned long long)shader->token_hash);
      return false;
   }

   /* ntv_shader_prepare() uses GENERIC[n] as driver_location n but leaves
    * legacy COLOR/FOG/TEX slots at their raw Mesa slot values.  Reconcile the
    * two namespaces with a stable semantic map because VS and PS are compiled
    * separately.  Do not offset all generics to make room for legacy slots:
    * Nine deliberately uses sparse GENERIC indices (BLENDINDICES[0] is 20),
    * and adding 12 produced Location 32, beyond a 128-component vertex-output
    * limit.  Builtins are emitted with BuiltIn rather than Location. */
   {
      const mesa_shader_stage stage = shader->nir->info.stage;
      nir_foreach_variable_with_modes(io_var, shader->nir,
                                      nir_var_shader_in | nir_var_shader_out) {
         /* Vertex-shader inputs are vertex attributes (VERT_ATTRIB_*), and
          * fragment-shader outputs are render targets (FRAG_RESULT_*); neither
          * is an interstage varying, so leave ntv's assignment for them. */
         if (stage == MESA_SHADER_VERTEX && (io_var->data.mode & nir_var_shader_in))
            continue;
         if (stage == MESA_SHADER_FRAGMENT && (io_var->data.mode & nir_var_shader_out))
            continue;

         const unsigned slot = io_var->data.location;
         if (slot == VARYING_SLOT_POS)
            continue;

         unsigned location = 0;
         if (!yttrium_shader_interstage_location(slot, &location)) {
            YTTRIUM_WARN("yttrium: shader_compile_spirv skipped stage=%s id=%u reason=interstage_location_unsupported slot=%u token_hash=0x%llx\n",
                         yttrium_shader_stage_name(shader->stage),
                         shader->id, slot,
                         (unsigned long long)shader->token_hash);
            return false;
         }
         io_var->data.driver_location = location;
      }
   }

   if (!yttrium_shader_preflight_spirv_distance_io(shader))
      return false;

   if (!yttrium_shader_lower_formatless_images(ctx, shader))
      return false;

   yttrium_shader_scan_image_arrays(shader);

   shader->spirv = nir_to_spirv(shader->nir, &ntv_info);
   if (!shader->spirv || !shader->spirv->words ||
       !shader->spirv->num_words) {
      YTTRIUM_WARN("yttrium: shader_compile_spirv skipped stage=%s id=%u reason=nir_to_spirv_empty token_hash=0x%llx resource_free=%u ubo_only=%u texture_only=%u ubos=%u inputs=%u outputs=%u writes_memory=%u\n",
                   yttrium_shader_stage_name(shader->stage),
                   shader->id,
                   (unsigned long long)shader->token_hash,
                   shader->resource_free,
                   shader->uniform_buffer_only,
                   shader->sampled_texture_only,
                   shader->nir->info.num_ubos,
                   shader->info.num_inputs,
                   shader->info.num_outputs,
                   shader->nir->info.writes_memory);
      return false;
   }

   shader->spirv_word_count = (unsigned)shader->spirv->num_words;
   shader->spirv_hash = yttrium_shader_spirv_hash(shader->spirv);

   struct yttrium_spirv_probe probe;
   if (!yttrium_shader_probe_spirv(shader, &probe)) {
      yttrium_shader_log_spirv_probe(shader, &probe);
      YTTRIUM_WARN("yttrium: shader_compile_spirv skipped stage=%s id=%u reason=malformed_spirv failure=%s words=%u spirv_hash=0x%llx token_hash=0x%llx bad_offset=%u bad_words=%u bad_opcode=%u\n",
                   yttrium_shader_stage_name(shader->stage),
                   shader->id,
                   probe.failure ? probe.failure : "unknown",
                   shader->spirv_word_count,
                   (unsigned long long)shader->spirv_hash,
                   (unsigned long long)shader->token_hash,
                   probe.bad_word_offset,
                   probe.bad_word_count,
                   probe.bad_opcode);
      yttrium_shader_dump_spirv_failure(shader, "malformed_spirv", 0);
      return false;
   }
   yttrium_shader_log_spirv_probe(shader, &probe);

   YTTRIUM_LOG("yttrium: shader_compile_spirv stage=%s id=%u spirv_words=%u spirv_hash=0x%llx resource_free=%u ubo_only=%u texture_only=%u sampler_mask=0x%x ubo_mask=0x%x ubo_default=%u ubo_first=%u ubo_count=%u\n",
                yttrium_shader_stage_name(shader->stage),
                shader->id,
                shader->spirv_word_count,
                (unsigned long long)shader->spirv_hash,
                shader->resource_free,
                shader->uniform_buffer_only,
                shader->sampled_texture_only,
                shader->sampler_used_mask,
                shader->ubo_used_mask,
                shader->ubo_default,
                shader->ubo_first,
                shader->ubo_count);
   return true;
}

static bool
yttrium_shader_make_module(struct pipe_context *ctx,
                           struct yttrium_shader_state *shader)
{
   if (!ctx || !shader || !shader->spirv || !shader->spirv->words ||
       !shader->spirv->num_words)
      return false;

   struct yttrium_screen *screen = yttrium_screen(ctx->screen);
   const size_t code_size = shader->spirv->num_words * sizeof(uint32_t);
   struct yttrium_spirv_probe probe;
   if (!yttrium_shader_probe_spirv(shader, &probe)) {
      yttrium_shader_log_spirv_probe(shader, &probe);
      YTTRIUM_WARN("yttrium: shader_module_preflight failed stage=%s id=%u reason=malformed_spirv failure=%s spirv_words=%u spirv_hash=0x%llx token_hash=0x%llx bad_offset=%u bad_words=%u bad_opcode=%u\n",
                   yttrium_shader_stage_name(shader->stage),
                   shader->id,
                   probe.failure ? probe.failure : "unknown",
                   shader->spirv_word_count,
                   (unsigned long long)shader->spirv_hash,
                   (unsigned long long)shader->token_hash,
                   probe.bad_word_offset,
                   probe.bad_word_count,
                   probe.bad_opcode);
      yttrium_shader_dump_spirv_failure(shader, "module_preflight", 0);
      return false;
   }

   if (!probe.has_memory_model || !probe.entry_points ||
       !probe.has_stage_entry) {
      YTTRIUM_WARN("yttrium: shader_module_preflight suspicious stage=%s id=%u memory_model=%u entries=%u stage_entry=%u first_model=%u expected_model=%u spirv_words=%u spirv_hash=0x%llx token_hash=0x%llx\n",
                   yttrium_shader_stage_name(shader->stage),
                   shader->id,
                   probe.has_memory_model,
                   probe.entry_points,
                   probe.has_stage_entry,
                   probe.first_execution_model,
                   probe.expected_execution_model,
                   shader->spirv_word_count,
                   (unsigned long long)shader->spirv_hash,
                   (unsigned long long)shader->token_hash);
   }

   VkResult result = VK_SUCCESS;
   if (!yttrium_venus_create_shader_module(screen->venus,
                                           &shader->module_obj,
                                           &shader->module,
                                           shader->spirv->words,
                                           code_size,
                                           yttrium_shader_stage_name(shader->stage),
                                           &result)) {
      YTTRIUM_WARN("yttrium: shader_module_create failed stage=%s id=%u result=%d spirv_words=%u spirv_hash=0x%llx token_hash=0x%llx inputs=%u outputs=%u resource_free=%u memory_model=%u/%u entries=%u stage_entry=%u vars(input=%u output=%u uniform=%u uniform_constant=%u push=%u private=%u) locations=%u\n",
                   yttrium_shader_stage_name(shader->stage),
                   shader->id,
                   result,
                   shader->spirv_word_count,
                   (unsigned long long)shader->spirv_hash,
                   (unsigned long long)shader->token_hash,
                   shader->info.num_inputs,
                   shader->info.num_outputs,
                   shader->resource_free,
                   probe.memory_addressing_model,
                   probe.memory_model,
                   probe.entry_points,
                   probe.has_stage_entry,
                   probe.input_variables,
                   probe.output_variables,
                   probe.uniform_variables,
                   probe.uniform_constant_variables,
                   probe.push_constant_variables,
                   probe.private_variables,
                   probe.location_decorations);
      yttrium_shader_dump_spirv_failure(shader, "venus_create_module", result);
      return false;
   }

   YTTRIUM_LOG("yttrium: shader_module_create stage=%s id=%u module_id=%llu module=%p spirv_words=%u spirv_hash=0x%llx resource_free=%u\n",
                yttrium_shader_stage_name(shader->stage),
                shader->id,
                (unsigned long long)shader->module_obj.id,
                (void *)shader->module,
                shader->spirv_word_count,
                (unsigned long long)shader->spirv_hash,
                shader->resource_free);
   yttrium_shader_dump_sampled_success(shader);
   yttrium_shader_dump_stream_output_success(shader);
   return true;
}

static bool
yttrium_is_stream_output_placeholder_gs(
   const struct yttrium_shader_state *shader)
{
   return shader && shader->type == PIPE_SHADER_IR_TGSI &&
          shader->stage == MESA_SHADER_GEOMETRY && !shader->tokens &&
          shader->stream_output.num_outputs;
}

static struct yttrium_shader_state *
yttrium_shader_state_create_with_shared(struct pipe_context *ctx,
                                        const struct pipe_shader_state *state,
                                        mesa_shader_stage stage,
                                        unsigned static_shared_mem)
{
   struct yttrium_shader_state *shader =
      CALLOC_STRUCT(yttrium_shader_state);
   if (!shader) {
      /* pipe_shader_state transfers NIR ownership to the driver even when
       * state creation fails. */
      if (state && state->type == PIPE_SHADER_IR_NIR)
         ralloc_free(state->ir.nir);
      return NULL;
   }

   shader->id = (uint32_t)InterlockedIncrement(&yttrium_shader_sequence);
   shader->stage = stage;
   shader->type = state ? state->type : PIPE_SHADER_IR_TGSI;
   /* Not computed yet; CALLOC's zero would read as "no". */
   shader->uses_sample_shading = -1;
   shader->static_shared_mem = static_shared_mem;
   yttrium_shader_init_sampler_binding_map(shader);
   if (state)
      shader->stream_output = state->stream_output;

   if (state && state->type == PIPE_SHADER_IR_TGSI && state->tokens) {
      shader->tokens = tgsi_dup_tokens(state->tokens);
      if (!shader->tokens) {
         FREE(shader);
         return NULL;
      }

      shader->token_count = tgsi_num_tokens(shader->tokens);
      shader->token_hash = yttrium_shader_token_hash(shader);
      tgsi_scan_shader(shader->tokens, &shader->info);
      yttrium_shader_scan_sampler_view_map(shader);
      if (!yttrium_shader_apply_sampler_binding_map(
             shader, state->sampler_binding_map)) {
         yttrium_shader_state_destroy(
            ctx && ctx->screen ? yttrium_screen(ctx->screen) : NULL,
            shader);
         return NULL;
      }
      yttrium_shader_trace_sampler_info(shader, "create_tgsi_scan");
      yttrium_shader_trace_io_info(shader, "create_tgsi_scan");
      yttrium_trace_debug_stringf(
         "yttrium: shader_create stage=%s id=%u state=%p ir=tgsi token_count=%u token_hash=0x%llx inputs=%u outputs=%u constbufs=0x%x samplers=0x%x instructions=%u writes_z=%u uses_kill=%u",
         yttrium_shader_stage_name(stage),
         shader->id,
         shader,
         shader->token_count,
         (unsigned long long)shader->token_hash,
         shader->info.num_inputs,
         shader->info.num_outputs,
         shader->info.const_buffers_declared,
         yttrium_shader_info_sampled_texture_mask(&shader->info),
         shader->info.num_instructions,
         shader->info.writes_z,
         shader->info.uses_kill);
   } else if (state && state->type == PIPE_SHADER_IR_NIR && state->ir.nir) {
      shader->nir = state->ir.nir;
      nir_tgsi_scan_shader(shader->nir, &shader->info, true);
      yttrium_shader_trace_sampler_info(shader, "create_nir_scan");
      yttrium_shader_trace_io_info(shader, "create_nir_scan");
      yttrium_shader_trace_nir_samplers(shader, "create_nir_scan");
      yttrium_trace_debug_stringf(
         "yttrium: shader_create stage=%s id=%u state=%p ir=nir nir=%p inputs=%u outputs=%u",
         yttrium_shader_stage_name(stage),
         shader->id,
         shader,
         shader->nir,
         shader->info.num_inputs,
         shader->info.num_outputs);
   } else {
      yttrium_trace_debug_stringf(
         "yttrium: shader_create stage=%s id=%u state=%p ir=%u tokens=%p placeholder=1",
         yttrium_shader_stage_name(stage),
         shader->id,
         shader,
         state ? state->type : PIPE_SHADER_IR_TGSI,
         state ? state->tokens : NULL);
   }

   /* A tokenless GS carrying stream-output declarations is Gallium's
    * deliberate passthrough placeholder.  Its bind path compiles the current
    * vertex shader with these declarations, so this object itself neither
    * needs nor can produce NIR/SPIR-V. */
   const bool stream_output_placeholder =
      yttrium_is_stream_output_placeholder_gs(shader);
   const bool compile_enabled = yttrium_shader_compile_enabled();
   const bool module_enabled = yttrium_shader_module_enabled();
   const bool pipeline_draw_enabled = yttrium_shader_pipeline_draw_enabled();
   yttrium_trace_debug_stringf(
      "yttrium: shader_compile_gate stage=%s id=%u compile=%u module=%u pipeline_draw=%u state=%p ir=%u has_nir=%u",
      yttrium_shader_stage_name(stage),
      shader->id,
      compile_enabled,
      module_enabled,
      pipeline_draw_enabled,
      state,
      state ? state->type : PIPE_SHADER_IR_TGSI,
      shader->nir != NULL);

   if (compile_enabled && !shader->nir && state &&
       !stream_output_placeholder)
      yttrium_shader_make_nir(ctx, state, shader);

   /* Without NIR the SPIR-V and module block below is skipped wholesale.
    * Diagnose that root cause here; the fail-closed check below then rejects
    * the state instead of deferring it to pipeline creation and dropped draws.
    */
   if (compile_enabled && !shader->nir && !stream_output_placeholder) {
      YTTRIUM_WARN("yttrium: shader_no_nir stage=%s id=%u ir=%u token_hash=0x%llx state=%p; native shader unavailable\n",
                   yttrium_shader_stage_name(stage),
                   shader->id,
                   state ? state->type : 0,
                   (unsigned long long)shader->token_hash,
                   (const void *)state);
   }

   if (compile_enabled && shader->nir &&
       shader->static_shared_mem > shader->nir->info.shared_size)
      shader->nir->info.shared_size = shader->static_shared_mem;

   if (compile_enabled && shader->nir) {
      yttrium_shader_trace_io_info(shader, "pre_spirv");
      if (!yttrium_shader_make_spirv(ctx, shader)) {
         YTTRIUM_WARN("yttrium: shader_compile_spirv failed stage=%s id=%u ir=%u token_hash=0x%llx; native shader unavailable\n",
                      yttrium_shader_stage_name(stage),
                      shader->id,
                      shader->type,
                      (unsigned long long)shader->token_hash);
      } else if (module_enabled) {
         if (yttrium_shader_state_is_placeholder_module(shader)) {
            YTTRIUM_LOG("yttrium: shader_module_skip stage=%s id=%u reason=placeholder spirv_words=%u inputs=%u outputs=%u instructions=%u writes_z=%u uses_kill=%u\n",
                        yttrium_shader_stage_name(stage),
                        shader->id,
                        shader->spirv_word_count,
                        shader->info.num_inputs,
                        shader->info.num_outputs,
                        shader->info.num_instructions,
                        shader->info.writes_z,
                        shader->info.uses_kill);
         } else if (!yttrium_shader_make_module(ctx, shader)) {
            YTTRIUM_WARN("yttrium: shader_module_unavailable stage=%s id=%u spirv_words=%u spirv_hash=0x%llx; native shader unavailable\n",
                         yttrium_shader_stage_name(stage),
                         shader->id,
                         shader->spirv_word_count,
                         (unsigned long long)shader->spirv_hash);
         }
      }
   }

   /* Native shader creation is the last point where Gallium can report this
    * failure to its caller.  Returning a module-less, otherwise ordinary
    * shader defers the error until pipeline creation and turns every affected
    * draw into a silent no-op.  Keep the two intentional module-less cases:
    * tokenless stream-output GS placeholders are resolved when bound, and a
    * shader with no observable work is represented without a module.
    */
   if (module_enabled && !stream_output_placeholder &&
       !yttrium_shader_state_is_placeholder_module(shader) &&
       !yttrium_shader_state_has_module(shader)) {
      const char *reason = !shader->nir ? "nir_unavailable" :
                           !shader->spirv ? "spirv_unavailable" :
                                           "module_unavailable";
      YTTRIUM_WARN("yttrium: shader_state_create failed owner=yttrium stage=%s id=%u reason=%s token_hash=0x%llx\n",
                   yttrium_shader_stage_name(stage), shader->id, reason,
                   (unsigned long long)shader->token_hash);
      struct yttrium_screen *screen =
         ctx && ctx->screen ? yttrium_screen(ctx->screen) : NULL;
      yttrium_shader_state_destroy(screen, shader);
      return NULL;
   }

   return shader;
}

struct yttrium_shader_state *
yttrium_shader_state_create(struct pipe_context *ctx,
                            const struct pipe_shader_state *state,
                            mesa_shader_stage stage)
{
   return yttrium_shader_state_create_with_shared(ctx, state, stage, 0);
}

void
yttrium_shader_state_log_bind(const struct yttrium_shader_state *shader,
                              mesa_shader_stage stage)
{
   yttrium_trace_debug_stringf(
      "yttrium: shader_bind stage=%s state=%p id=%u token_hash=0x%llx spirv_hash=0x%llx module_id=%llu inputs=%u outputs=%u resource_free=%u ubo_only=%u texture_only=%u sampler_mask=0x%x ubo_mask=0x%x ubo_default=%u ubo_first=%u ubo_count=%u",
      yttrium_shader_stage_name(stage), shader,
      shader ? shader->id : 0,
      shader ? (unsigned long long)shader->token_hash : 0,
      shader ? (unsigned long long)shader->spirv_hash : 0,
      shader ? (unsigned long long)shader->module_obj.id : 0,
      shader ? shader->info.num_inputs : 0,
      shader ? shader->info.num_outputs : 0,
      shader ? shader->resource_free : 0,
      shader ? shader->uniform_buffer_only : 0,
      shader ? shader->sampled_texture_only : 0,
      shader ? shader->sampler_used_mask : 0,
      shader ? shader->ubo_used_mask : 0,
      shader ? shader->ubo_default : 0,
      shader ? shader->ubo_first : 0,
      shader ? shader->ubo_count : 0);
}

void
yttrium_shader_state_destroy(struct yttrium_screen *screen,
                             struct yttrium_shader_state *shader)
{
   if (!shader)
      return;

   yttrium_trace_debug_stringf(
      "yttrium: shader_delete stage=%s state=%p id=%u token_hash=0x%llx spirv_hash=0x%llx module_id=%llu",
      yttrium_shader_stage_name(shader->stage), shader,
      shader->id, (unsigned long long)shader->token_hash,
      (unsigned long long)shader->spirv_hash,
      (unsigned long long)shader->module_obj.id);
   if (screen && shader->module)
      yttrium_venus_destroy_shader_module(screen->venus,
                                          &shader->module_obj,
                                          shader->module);
   if (shader->spirv)
      spirv_shader_delete(shader->spirv);
   if (shader->nir)
      ralloc_free(shader->nir);
   if (shader->tokens)
      tgsi_free_tokens(shader->tokens);
   FREE(shader);
}

bool
yttrium_shader_state_has_module(const struct yttrium_shader_state *shader)
{
   return shader && shader->module && shader->module_obj.id &&
          shader->spirv && shader->spirv_word_count;
}

bool
yttrium_shader_state_is_resource_free(const struct yttrium_shader_state *shader)
{
   return shader && shader->resource_free;
}

bool
yttrium_shader_state_is_uniform_buffer_only(
   const struct yttrium_shader_state *shader)
{
   return shader && shader->uniform_buffer_only;
}

bool
yttrium_shader_state_is_sampled_texture_only(
   const struct yttrium_shader_state *shader)
{
   return shader && shader->sampled_texture_only;
}

bool
yttrium_shader_state_is_storage_image_only(
   const struct yttrium_shader_state *shader)
{
   return shader && shader->storage_image_only;
}

bool
yttrium_shader_state_is_sampled_storage_image_only(
   const struct yttrium_shader_state *shader)
{
   return shader && shader->sampled_storage_image_only;
}

uint32_t
yttrium_shader_state_sampler_used_mask(
   const struct yttrium_shader_state *shader)
{
   return shader ? shader->sampler_used_mask : 0;
}

uint64_t
yttrium_shader_state_image_used_mask(
   const struct yttrium_shader_state *shader)
{
   return shader ? shader->image_used_mask : 0;
}

unsigned
yttrium_shader_state_sampler_view_index(
   const struct yttrium_shader_state *shader,
   unsigned sampler_slot)
{
   if (!shader || sampler_slot >= PIPE_MAX_SAMPLERS)
      return sampler_slot;

   const unsigned view_slot = shader->sampler_view_index[sampler_slot];
   if (view_slot >= PIPE_MAX_SHADER_SAMPLER_VIEWS)
      return sampler_slot;

   return view_slot;
}

unsigned
yttrium_shader_state_sampler_state_index(
   const struct yttrium_shader_state *shader,
   unsigned sampler_slot)
{
   if (!shader || sampler_slot >= PIPE_MAX_SAMPLERS)
      return sampler_slot;

   const unsigned state_slot = shader->sampler_state_index[sampler_slot];
   if (state_slot > PIPE_MAX_SAMPLERS)
      return sampler_slot;

   return state_slot;
}

bool
yttrium_shader_state_has_explicit_sampler_binding(
   const struct yttrium_shader_state *shader,
   unsigned sampler_slot)
{
   return shader && sampler_slot < PIPE_MAX_SAMPLERS &&
          (shader->sampler_binding_valid_mask & (1u << sampler_slot));
}

void
yttrium_finalize_nir(struct pipe_screen *screen,
                     struct nir_shader *nir,
                     bool optimize)
{
   (void)screen;
   (void)optimize;

   if (!nir)
      return;

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
}

void
yttrium_screen_init_shader_compiler(struct yttrium_screen *screen)
{
   static const struct nir_shader_compiler_options default_options = {
      .io_options = nir_io_has_intrinsics | nir_io_mediump_is_32bit,
      .lower_ffma16 = true,
      .lower_ffma32 = true,
      .lower_ffma64 = true,
      .lower_scmp = true,
      .lower_fdph = true,
      .lower_flrp32 = true,
      .lower_fsat = true,
      .lower_hadd = true,
      .lower_iadd_sat = true,
      .lower_fisnormal = true,
      .lower_extract_byte = true,
      .lower_extract_word = true,
      .lower_insert_byte = true,
      .lower_insert_word = true,
      .has_ldexp = false,
      .lower_mul_high = true,
      .lower_to_scalar = true,
      .lower_uadd_carry = true,
      .compact_arrays = true,
      .lower_usub_borrow = true,
      .lower_uadd_sat = true,
      .lower_usub_sat = true,
      .lower_int64_options = ~0,
      .lower_doubles_options = ~0,
      .lower_uniforms_to_ubo = true,
      .has_fsub = true,
      .has_isub = true,
      .lower_mul_2x32_64 = true,
      .support_16bit_alu = true,
      .max_unroll_iterations = 0,
      .support_indirect_inputs = BITFIELD_BIT(MESA_SHADER_TESS_CTRL) |
                                  BITFIELD_BIT(MESA_SHADER_TESS_EVAL) |
                                  BITFIELD_BIT(MESA_SHADER_FRAGMENT),
      .support_indirect_outputs = (uint8_t)BITFIELD_MASK(MESA_SHADER_STAGES),
   };

   if (!screen)
      return;

   if (!screen->glsl_type_singleton_ref) {
      glsl_type_singleton_init_or_ref();
      screen->glsl_type_singleton_ref = true;
   }

   screen->nir_options = default_options;
   for (unsigned i = 0; i < MESA_SHADER_MESH_STAGES; i++)
      screen->base.nir_options[i] = &screen->nir_options;
   screen->base.finalize_nir = yttrium_finalize_nir;
}

static void *
yttrium_create_shader_state(struct pipe_context *ctx,
                            const struct pipe_shader_state *state,
                            mesa_shader_stage stage)
{
   return yttrium_shader_state_create(ctx, state, stage);
}

void *
yttrium_create_vs_state(struct pipe_context *ctx,
                        const struct pipe_shader_state *state)
{
   return yttrium_create_shader_state(ctx, state, MESA_SHADER_VERTEX);
}

void *
yttrium_create_fs_state(struct pipe_context *ctx,
                        const struct pipe_shader_state *state)
{
   return yttrium_create_shader_state(ctx, state, MESA_SHADER_FRAGMENT);
}

void *
yttrium_create_gs_state(struct pipe_context *ctx,
                        const struct pipe_shader_state *state)
{
   if (state && state->type == PIPE_SHADER_IR_TGSI && state->tokens &&
       state->stream_output.num_outputs &&
       tgsi_get_processor_type(state->tokens) == MESA_SHADER_VERTEX) {
      struct yttrium_shader_state *shader =
         yttrium_create_shader_state(ctx, state, MESA_SHADER_VERTEX);
      if (shader) {
         shader->vs_stream_output_gs = true;
         yttrium_trace_debug_stringf(
            "yttrium: shader_create_vs_stream_output_gs state=%p id=%u outputs=%u",
            shader, shader->id, shader->stream_output.num_outputs);
      }
      return shader;
   }

   return yttrium_create_shader_state(ctx, state, MESA_SHADER_GEOMETRY);
}

void *
yttrium_create_tcs_state(struct pipe_context *ctx,
                         const struct pipe_shader_state *state)
{
   return yttrium_create_shader_state(ctx, state, MESA_SHADER_TESS_CTRL);
}

void *
yttrium_create_tes_state(struct pipe_context *ctx,
                         const struct pipe_shader_state *state)
{
   return yttrium_create_shader_state(ctx, state, MESA_SHADER_TESS_EVAL);
}

void *
yttrium_create_compute_state(struct pipe_context *ctx,
                             const struct pipe_compute_state *state)
{
   if (!state)
      return yttrium_create_shader_state(ctx, NULL, MESA_SHADER_COMPUTE);

   struct pipe_shader_state shader_state;
   memset(&shader_state, 0, sizeof(shader_state));
   shader_state.type = state->ir_type;
   if (state->ir_type == PIPE_SHADER_IR_TGSI) {
      shader_state.tokens = (const struct tgsi_token *)state->prog;
      shader_state.sampler_binding_map = state->sampler_binding_map;
   } else if (state->ir_type == PIPE_SHADER_IR_NIR) {
      shader_state.ir.nir = (struct nir_shader *)state->prog;
   } else {
      return NULL;
   }

   return yttrium_shader_state_create_with_shared(ctx, &shader_state,
                                                  MESA_SHADER_COMPUTE,
                                                  state->static_shared_mem);
}

static void
yttrium_bind_shader_state(struct pipe_context *ctx, void *state,
                          mesa_shader_stage stage)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_shader_state *shader =
      (struct yttrium_shader_state *)state;

   if (stage < MESA_SHADER_STAGES)
      yctx->shaders[stage] = shader;

   yttrium_pipeline_state_changed(yctx);
   yttrium_shader_state_log_bind(shader, stage);
}

static struct yttrium_shader_state *
yttrium_get_vs_stream_output_variant(struct pipe_context *ctx,
                                     struct yttrium_shader_state *gs)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_shader_state *base_vs = yctx->base_vs;

   if (!yttrium_is_stream_output_placeholder_gs(gs) ||
       !base_vs || !base_vs->tokens)
      return NULL;

   if (gs->vs_stream_output_variant &&
       gs->vs_stream_output_base_id == base_vs->id)
      return gs->vs_stream_output_variant;

   if (gs->vs_stream_output_variant) {
      if (yctx->vs_stream_output_gs == gs->vs_stream_output_variant)
         yctx->vs_stream_output_gs = NULL;
      if (yctx->shaders[MESA_SHADER_VERTEX] == gs->vs_stream_output_variant)
         yctx->shaders[MESA_SHADER_VERTEX] = base_vs;
      yttrium_shader_state_destroy(yttrium_screen(ctx->screen),
                                   gs->vs_stream_output_variant);
      gs->vs_stream_output_variant = NULL;
      gs->vs_stream_output_base_id = 0;
   }

   struct pipe_shader_state state;
   memset(&state, 0, sizeof(state));
   state.type = PIPE_SHADER_IR_TGSI;
   state.tokens = base_vs->tokens;
   state.stream_output = gs->stream_output;

   struct yttrium_shader_state *variant =
      yttrium_create_shader_state(ctx, &state, MESA_SHADER_VERTEX);
   if (!variant)
      return NULL;

   yttrium_shader_copy_sampler_binding_map(variant, base_vs);
   variant->vs_stream_output_gs = true;
   gs->vs_stream_output_variant = variant;
   gs->vs_stream_output_base_id = base_vs->id;
   yttrium_trace_debug_stringf(
      "yttrium: shader_create_placeholder_vs_stream_output_gs gs=%p id=%u variant=%p variant_id=%u base_vs=%p base_vs_id=%u outputs=%u",
      gs, gs->id, variant, variant->id, base_vs, base_vs->id,
      gs->stream_output.num_outputs);
   return variant;
}

void
yttrium_bind_vs_state(struct pipe_context *ctx, void *state)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_shader_state *shader =
      (struct yttrium_shader_state *)state;

   yctx->base_vs = shader;
   if (yctx->vs_stream_output_source_gs) {
      yctx->vs_stream_output_gs =
         yttrium_get_vs_stream_output_variant(
            ctx, yctx->vs_stream_output_source_gs);
      if (!yctx->vs_stream_output_gs) {
         YTTRIUM_WARN("yttrium: shader bind failed owner=yttrium-shader "
                      "reason=tokenless_stream_output_variant_creation_failed "
                      "stage=vs action=disable_vertex_stage_and_fail_draw\n");
         InterlockedExchange(&yctx->draw_failure, 1);
      }
      yctx->shaders[MESA_SHADER_VERTEX] = yctx->vs_stream_output_gs;
   } else {
      yctx->shaders[MESA_SHADER_VERTEX] =
         yctx->vs_stream_output_gs ? yctx->vs_stream_output_gs : shader;
   }

   yttrium_pipeline_state_changed(yctx);
   yttrium_shader_state_log_bind(shader, MESA_SHADER_VERTEX);
}

void
yttrium_bind_fs_state(struct pipe_context *ctx, void *state)
{
   yttrium_bind_shader_state(ctx, state, MESA_SHADER_FRAGMENT);
}

void
yttrium_bind_gs_state(struct pipe_context *ctx, void *state)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_shader_state *shader =
      (struct yttrium_shader_state *)state;

   if (shader && shader->vs_stream_output_gs) {
      yctx->vs_stream_output_source_gs = NULL;
      yctx->vs_stream_output_gs = shader;
      yctx->shaders[MESA_SHADER_VERTEX] = shader;
      yctx->shaders[MESA_SHADER_GEOMETRY] = NULL;
      yttrium_pipeline_state_changed(yctx);
      yttrium_shader_state_log_bind(shader, MESA_SHADER_GEOMETRY);
      yttrium_trace_debug_stringf(
         "yttrium: shader_bind_vs_stream_output_gs state=%p id=%u base_vs=%p",
         shader, shader->id, yctx->base_vs);
      return;
   }

   if (yttrium_is_stream_output_placeholder_gs(shader)) {
      struct yttrium_shader_state *variant =
         yttrium_get_vs_stream_output_variant(ctx, shader);
      yctx->vs_stream_output_source_gs = shader;
      yctx->vs_stream_output_gs = variant;
      if (!variant) {
         YTTRIUM_WARN("yttrium: shader bind failed owner=yttrium-shader "
                      "reason=tokenless_stream_output_variant_creation_failed "
                      "stage=gs action=disable_vertex_stage_and_fail_draw\n");
         InterlockedExchange(&yctx->draw_failure, 1);
      }
      yctx->shaders[MESA_SHADER_VERTEX] = variant;
      yctx->shaders[MESA_SHADER_GEOMETRY] = NULL;
      yttrium_pipeline_state_changed(yctx);
      yttrium_shader_state_log_bind(shader, MESA_SHADER_GEOMETRY);
      yttrium_trace_debug_stringf(
         "yttrium: shader_bind_placeholder_vs_stream_output_gs gs=%p id=%u variant=%p base_vs=%p",
         shader, shader->id, variant, yctx->base_vs);
      return;
   }

   yctx->vs_stream_output_source_gs = NULL;
   yctx->vs_stream_output_gs = NULL;
   yctx->shaders[MESA_SHADER_VERTEX] = yctx->base_vs;
   yctx->shaders[MESA_SHADER_GEOMETRY] = shader;

   yttrium_pipeline_state_changed(yctx);
   yttrium_shader_state_log_bind(shader, MESA_SHADER_GEOMETRY);
}

void
yttrium_bind_tcs_state(struct pipe_context *ctx, void *state)
{
   yttrium_bind_shader_state(ctx, state, MESA_SHADER_TESS_CTRL);
}

void
yttrium_bind_tes_state(struct pipe_context *ctx, void *state)
{
   yttrium_bind_shader_state(ctx, state, MESA_SHADER_TESS_EVAL);
}

void
yttrium_bind_compute_state(struct pipe_context *ctx, void *state)
{
   yttrium_bind_shader_state(ctx, state, MESA_SHADER_COMPUTE);
}

static void
yttrium_delete_shader_state(struct pipe_context *ctx, void *state)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   struct yttrium_shader_state *shader =
      (struct yttrium_shader_state *)state;

   if (!shader)
      return;

   bool dirty = false;
   if (yctx->base_vs == shader) {
      yctx->base_vs = NULL;
      if (!yctx->vs_stream_output_gs &&
          yctx->shaders[MESA_SHADER_VERTEX] == shader)
         yctx->shaders[MESA_SHADER_VERTEX] = NULL;
      dirty = true;
   }

   if (yctx->vs_stream_output_source_gs == shader) {
      yctx->vs_stream_output_source_gs = NULL;
      dirty = true;
   }

   if (yctx->vs_stream_output_gs == shader) {
      yctx->vs_stream_output_gs = NULL;
      if (yctx->shaders[MESA_SHADER_VERTEX] == shader)
         yctx->shaders[MESA_SHADER_VERTEX] = yctx->base_vs;
      dirty = true;
   }

   if (shader->vs_stream_output_variant) {
      if (yctx->vs_stream_output_gs == shader->vs_stream_output_variant)
         yctx->vs_stream_output_gs = NULL;
      if (yctx->shaders[MESA_SHADER_VERTEX] ==
          shader->vs_stream_output_variant)
         yctx->shaders[MESA_SHADER_VERTEX] = yctx->base_vs;
      yttrium_shader_state_destroy(yttrium_screen(ctx->screen),
                                   shader->vs_stream_output_variant);
      shader->vs_stream_output_variant = NULL;
      shader->vs_stream_output_base_id = 0;
      dirty = true;
   }

   if (shader->stage < MESA_SHADER_STAGES &&
       yctx->shaders[shader->stage] == shader) {
      yctx->shaders[shader->stage] = NULL;
      dirty = true;
   }

   if (dirty)
      yttrium_pipeline_state_changed(yctx);

   yttrium_shader_state_destroy(yttrium_screen(ctx->screen), shader);
}

void
yttrium_delete_vs_state(struct pipe_context *ctx, void *state)
{
   yttrium_delete_shader_state(ctx, state);
}

void
yttrium_delete_fs_state(struct pipe_context *ctx, void *state)
{
   yttrium_delete_shader_state(ctx, state);
}

void
yttrium_delete_gs_state(struct pipe_context *ctx, void *state)
{
   yttrium_delete_shader_state(ctx, state);
}

void
yttrium_delete_tcs_state(struct pipe_context *ctx, void *state)
{
   yttrium_delete_shader_state(ctx, state);
}

void
yttrium_delete_tes_state(struct pipe_context *ctx, void *state)
{
   yttrium_delete_shader_state(ctx, state);
}

void
yttrium_delete_compute_state(struct pipe_context *ctx, void *state)
{
   if (state)
      yttrium_compute_pipeline_cache_invalidate(yttrium_context(ctx), state);
   yttrium_delete_shader_state(ctx, state);
}

void
yttrium_dump_capture_shader(const struct yttrium_shader_state *shader,
                            const char *path)
{
   FILE *file;

   if (!shader || !shader->tokens || !path || !path[0])
      return;

   file = fopen(path, "w");
   if (!file) {
      YTTRIUM_LOG("yttrium: textured draw capture shader fopen failed path=%s\n",
                   path);
      return;
   }

   tgsi_dump_to_file(shader->tokens, 0, file);
   fclose(file);
   YTTRIUM_LOG("yttrium: textured draw capture shader wrote path=%s\n",
                path);
}

void
yttrium_write_shader_capture_metadata(FILE *file,
                                      const struct yttrium_shader_state *shader)
{
   if (!file || !shader)
      return;

   fprintf(file,
           "shader.%s id=%u state=%p ir=%u tokens=%p token_count=%u token_hash=0x%llx spirv_words=%u spirv_hash=0x%llx inputs=%u outputs=%u constbufs=0x%x samplers=0x%x instructions=%u writes_z=%u uses_kill=%u\n",
           yttrium_shader_stage_name(shader->stage),
           shader->id,
           (const void *)shader,
           shader->type,
           (const void *)shader->tokens,
           shader->token_count,
           (unsigned long long)shader->token_hash,
           shader->spirv_word_count,
           (unsigned long long)shader->spirv_hash,
           shader->info.num_inputs,
           shader->info.num_outputs,
           shader->info.const_buffers_declared,
           yttrium_shader_info_sampled_texture_mask(&shader->info),
           shader->info.num_instructions,
           shader->info.writes_z,
           shader->info.uses_kill);

   for (unsigned i = 0; i < shader->info.num_inputs; i++) {
      fprintf(file,
              "shader.%s.input[%u] semantic=%u index=%u usage=0x%x interp=%u loc=%u\n",
              yttrium_shader_stage_name(shader->stage), i,
              shader->info.input_semantic_name[i],
              shader->info.input_semantic_index[i],
              shader->info.input_usage_mask[i],
              shader->info.input_interpolate[i],
              shader->info.input_interpolate_loc[i]);
   }

   for (unsigned i = 0; i < shader->info.num_outputs; i++) {
      fprintf(file,
              "shader.%s.output[%u] semantic=%u index=%u usage=0x%x streams=0x%x\n",
              yttrium_shader_stage_name(shader->stage), i,
              shader->info.output_semantic_name[i],
              shader->info.output_semantic_index[i],
              shader->info.output_usagemask[i],
              shader->info.output_streams[i]);
   }

   for (unsigned i = 0; i < MIN2(PIPE_MAX_SHADER_SAMPLER_VIEWS, 32); i++) {
      if (yttrium_shader_info_sampled_texture_mask(&shader->info) &
          (1u << i)) {
         fprintf(file,
                 "shader.%s.sampler[%u] target=%u type=%u\n",
                 yttrium_shader_stage_name(shader->stage), i,
                 shader->info.sampler_targets[i],
                 shader->info.sampler_type[i]);
      }
   }
}
