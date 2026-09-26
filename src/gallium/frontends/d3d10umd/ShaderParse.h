/**************************************************************************
 *
 * Copyright 2012-2021 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE COPYRIGHT HOLDERS, AUTHORS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 **************************************************************************/

/*
 * ShaderParse.h --
 *    Functions for parsing shader tokens.
 */

#ifndef SHADER_PARSE_H
#define SHADER_PARSE_H

#include "DriverIncludes.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Shader_header {
   D3D10_SB_TOKENIZED_PROGRAM_TYPE type;
   unsigned major_version;
   unsigned minor_version;
   unsigned size;
};

struct dx10_imm_const_buf {
   unsigned count;
   unsigned *data;
};

struct dx10_customdata {
   D3D10_SB_CUSTOMDATA_CLASS _class;
   union {
      struct dx10_imm_const_buf constbuf;
   } u;
};

struct dx10_indexable_temp {
   unsigned index;
   unsigned count;
   unsigned components;
};

struct dx10_global_flags {
   unsigned refactoring_allowed:1;
   unsigned force_early_depth_stencil:1;
};

struct dx10_thread_group {
   unsigned x;
   unsigned y;
   unsigned z;
};

struct dx10_tgsm {
   unsigned byte_count;
   unsigned structured_stride;
   unsigned structured_count;
};

struct Shader_relative_index {
   unsigned imm;
};

struct Shader_relative_operand {
   D3D10_SB_OPERAND_TYPE type;
   struct Shader_relative_index index[2];
   D3D10_SB_4_COMPONENT_NAME comp;
};

struct Shader_tessellation_io_signatures {
   const D3D10_SB_NAME *input_system_values;
   unsigned num_input_system_values;
   const D3D10_SB_NAME *output_system_values;
   unsigned num_output_system_values;
   const unsigned char *input_masks;
   const unsigned char *output_masks;
};

struct Shader_tessellation_properties {
   unsigned domain;
   unsigned partitioning;
   unsigned output_primitive;
};

struct Shader_index {
   unsigned imm;
   struct Shader_relative_operand rel;
   D3D10_SB_OPERAND_INDEX_REPRESENTATION index_rep;
};

struct Shader_operand {
   D3D10_SB_OPERAND_TYPE type;
   struct Shader_index index[2];
   unsigned index_dim;
};

struct Shader_dst_operand {
   struct Shader_operand base;
   unsigned mask;
};

union Shader_immediate {
   float f32;
   int i32;
   unsigned u32;
};

struct Shader_src_operand {
   struct Shader_operand base;
   union Shader_immediate imm[4];
   D3D10_SB_4_COMPONENT_NAME swizzle[4];
   D3D10_SB_OPERAND_MODIFIER modifier;
};

#define DX10_SM5_COMPUTE_SHADER ((D3D10_SB_TOKENIZED_PROGRAM_TYPE)5)
#define DX11_SM5_HULL_SHADER ((D3D10_SB_TOKENIZED_PROGRAM_TYPE)3)
#define DX11_SM5_DOMAIN_SHADER ((D3D10_SB_TOKENIZED_PROGRAM_TYPE)4)

#define DX10_SM5_OPERAND_TYPE_INPUT_THREAD_ID ((D3D10_SB_OPERAND_TYPE)32)
#define DX10_SM5_OPERAND_TYPE_THREAD_GROUP_SHARED_MEMORY ((D3D10_SB_OPERAND_TYPE)31)
#define DX10_SM5_OPERAND_TYPE_INPUT_THREAD_GROUP_ID ((D3D10_SB_OPERAND_TYPE)33)
#define DX10_SM5_OPERAND_TYPE_INPUT_THREAD_ID_IN_GROUP ((D3D10_SB_OPERAND_TYPE)34)
#define DX10_SM5_OPERAND_TYPE_INPUT_THREAD_ID_IN_GROUP_FLATTENED ((D3D10_SB_OPERAND_TYPE)36)

#define DX11_SM5_OPERAND_TYPE_OUTPUT_CONTROL_POINT_ID ((D3D10_SB_OPERAND_TYPE)22)
#define DX11_SM5_OPERAND_TYPE_INPUT_FORK_INSTANCE_ID ((D3D10_SB_OPERAND_TYPE)23)
#define DX11_SM5_OPERAND_TYPE_INPUT_JOIN_INSTANCE_ID ((D3D10_SB_OPERAND_TYPE)24)
#define DX11_SM5_OPERAND_TYPE_INPUT_CONTROL_POINT ((D3D10_SB_OPERAND_TYPE)25)
#define DX11_SM5_OPERAND_TYPE_OUTPUT_CONTROL_POINT ((D3D10_SB_OPERAND_TYPE)26)
#define DX11_SM5_OPERAND_TYPE_INPUT_PATCH_CONSTANT ((D3D10_SB_OPERAND_TYPE)27)
#define DX11_SM5_OPERAND_TYPE_INPUT_DOMAIN_POINT ((D3D10_SB_OPERAND_TYPE)28)

#define DX11_SM5_NAME_FINAL_QUAD_U_EQ_0_EDGE_TESSFACTOR ((D3D10_SB_NAME)11)
#define DX11_SM5_NAME_FINAL_QUAD_V_EQ_0_EDGE_TESSFACTOR ((D3D10_SB_NAME)12)
#define DX11_SM5_NAME_FINAL_QUAD_U_EQ_1_EDGE_TESSFACTOR ((D3D10_SB_NAME)13)
#define DX11_SM5_NAME_FINAL_QUAD_V_EQ_1_EDGE_TESSFACTOR ((D3D10_SB_NAME)14)
#define DX11_SM5_NAME_FINAL_QUAD_U_INSIDE_TESSFACTOR ((D3D10_SB_NAME)15)
#define DX11_SM5_NAME_FINAL_QUAD_V_INSIDE_TESSFACTOR ((D3D10_SB_NAME)16)
#define DX11_SM5_NAME_FINAL_TRI_U_EQ_0_EDGE_TESSFACTOR ((D3D10_SB_NAME)17)
#define DX11_SM5_NAME_FINAL_TRI_V_EQ_0_EDGE_TESSFACTOR ((D3D10_SB_NAME)18)
#define DX11_SM5_NAME_FINAL_TRI_W_EQ_0_EDGE_TESSFACTOR ((D3D10_SB_NAME)19)
#define DX11_SM5_NAME_FINAL_TRI_INSIDE_TESSFACTOR ((D3D10_SB_NAME)20)
#define DX11_SM5_NAME_FINAL_LINE_DETAIL_TESSFACTOR ((D3D10_SB_NAME)21)
#define DX11_SM5_NAME_FINAL_LINE_DENSITY_TESSFACTOR ((D3D10_SB_NAME)22)

#define DX10_SM5_OPCODE_BUFINFO ((D3D10_SB_OPCODE_TYPE)121)
#define DX10_SM5_OPCODE_DERIV_RTX_COARSE ((D3D10_SB_OPCODE_TYPE)122)
#define DX10_SM5_OPCODE_DERIV_RTX_FINE ((D3D10_SB_OPCODE_TYPE)123)
#define DX10_SM5_OPCODE_DERIV_RTY_COARSE ((D3D10_SB_OPCODE_TYPE)124)
#define DX10_SM5_OPCODE_DERIV_RTY_FINE ((D3D10_SB_OPCODE_TYPE)125)
#define DX10_SM5_OPCODE_RCP ((D3D10_SB_OPCODE_TYPE)129)
#define DX10_SM5_OPCODE_F32TOF16 ((D3D10_SB_OPCODE_TYPE)130)
#define DX10_SM5_OPCODE_F16TOF32 ((D3D10_SB_OPCODE_TYPE)131)
#define DX10_SM5_OPCODE_COUNTBITS ((D3D10_SB_OPCODE_TYPE)134)
#define DX10_SM5_OPCODE_FIRSTBIT_HI ((D3D10_SB_OPCODE_TYPE)135)
#define DX10_SM5_OPCODE_FIRSTBIT_LO ((D3D10_SB_OPCODE_TYPE)136)
#define DX10_SM5_OPCODE_FIRSTBIT_SHI ((D3D10_SB_OPCODE_TYPE)137)
#define DX10_SM5_OPCODE_UBFE ((D3D10_SB_OPCODE_TYPE)138)
#define DX10_SM5_OPCODE_IBFE ((D3D10_SB_OPCODE_TYPE)139)
#define DX10_SM5_OPCODE_BFI ((D3D10_SB_OPCODE_TYPE)140)
#define DX10_SM5_OPCODE_BFREV ((D3D10_SB_OPCODE_TYPE)141)
#define DX10_SM5_OPCODE_SWAPC ((D3D10_SB_OPCODE_TYPE)142)
#define DX11_SM5_OPCODE_HS_DECLS ((D3D10_SB_OPCODE_TYPE)113)
#define DX11_SM5_OPCODE_HS_CONTROL_POINT_PHASE ((D3D10_SB_OPCODE_TYPE)114)
#define DX11_SM5_OPCODE_HS_FORK_PHASE ((D3D10_SB_OPCODE_TYPE)115)
#define DX11_SM5_OPCODE_HS_JOIN_PHASE ((D3D10_SB_OPCODE_TYPE)116)
#define DX11_SM5_OPCODE_DCL_INPUT_CONTROL_POINT_COUNT ((D3D10_SB_OPCODE_TYPE)147)
#define DX11_SM5_OPCODE_DCL_OUTPUT_CONTROL_POINT_COUNT ((D3D10_SB_OPCODE_TYPE)148)
#define DX11_SM5_OPCODE_DCL_TESS_DOMAIN ((D3D10_SB_OPCODE_TYPE)149)
#define DX11_SM5_OPCODE_DCL_TESS_PARTITIONING ((D3D10_SB_OPCODE_TYPE)150)
#define DX11_SM5_OPCODE_DCL_TESS_OUTPUT_PRIMITIVE ((D3D10_SB_OPCODE_TYPE)151)
#define DX11_SM5_OPCODE_DCL_HS_MAX_TESSFACTOR ((D3D10_SB_OPCODE_TYPE)152)
#define DX11_SM5_OPCODE_DCL_HS_FORK_PHASE_INSTANCE_COUNT ((D3D10_SB_OPCODE_TYPE)153)
#define DX11_SM5_OPCODE_DCL_HS_JOIN_PHASE_INSTANCE_COUNT ((D3D10_SB_OPCODE_TYPE)154)
#define DX10_SM5_OPCODE_DCL_THREAD_GROUP ((D3D10_SB_OPCODE_TYPE)155)
#define DX10_SM5_OPCODE_DCL_UAV_RAW ((D3D10_SB_OPCODE_TYPE)157)
#define DX10_SM5_OPCODE_DCL_UAV_STRUCTURED ((D3D10_SB_OPCODE_TYPE)158)
#define DX10_SM5_OPCODE_DCL_TGSM_RAW ((D3D10_SB_OPCODE_TYPE)159)
#define DX10_SM5_OPCODE_DCL_TGSM_STRUCTURED ((D3D10_SB_OPCODE_TYPE)160)
#define DX10_SM5_OPCODE_DCL_RESOURCE_RAW ((D3D10_SB_OPCODE_TYPE)161)
#define DX10_SM5_OPCODE_DCL_RESOURCE_STRUCTURED ((D3D10_SB_OPCODE_TYPE)162)
#define DX10_SM5_OPCODE_LD_RAW ((D3D10_SB_OPCODE_TYPE)165)
#define DX10_SM5_OPCODE_STORE_RAW ((D3D10_SB_OPCODE_TYPE)166)
#define DX10_SM5_OPCODE_LD_STRUCTURED ((D3D10_SB_OPCODE_TYPE)167)
#define DX10_SM5_OPCODE_STORE_STRUCTURED ((D3D10_SB_OPCODE_TYPE)168)
#define DX10_SM5_OPCODE_ATOMIC_AND ((D3D10_SB_OPCODE_TYPE)169)
#define DX10_SM5_OPCODE_ATOMIC_OR ((D3D10_SB_OPCODE_TYPE)170)
#define DX10_SM5_OPCODE_ATOMIC_XOR ((D3D10_SB_OPCODE_TYPE)171)
#define DX10_SM5_OPCODE_ATOMIC_CMP_STORE ((D3D10_SB_OPCODE_TYPE)172)
#define DX10_SM5_OPCODE_ATOMIC_IADD ((D3D10_SB_OPCODE_TYPE)173)
#define DX10_SM5_OPCODE_ATOMIC_IMAX ((D3D10_SB_OPCODE_TYPE)174)
#define DX10_SM5_OPCODE_ATOMIC_IMIN ((D3D10_SB_OPCODE_TYPE)175)
#define DX10_SM5_OPCODE_ATOMIC_UMAX ((D3D10_SB_OPCODE_TYPE)176)
#define DX10_SM5_OPCODE_ATOMIC_UMIN ((D3D10_SB_OPCODE_TYPE)177)
#define DX10_SM5_OPCODE_IMM_ATOMIC_ALLOC ((D3D10_SB_OPCODE_TYPE)178)
#define DX10_SM5_OPCODE_IMM_ATOMIC_CONSUME ((D3D10_SB_OPCODE_TYPE)179)
#define DX10_SM5_OPCODE_IMM_ATOMIC_IADD ((D3D10_SB_OPCODE_TYPE)180)
#define DX10_SM5_OPCODE_IMM_ATOMIC_AND ((D3D10_SB_OPCODE_TYPE)181)
#define DX10_SM5_OPCODE_IMM_ATOMIC_OR ((D3D10_SB_OPCODE_TYPE)182)
#define DX10_SM5_OPCODE_IMM_ATOMIC_XOR ((D3D10_SB_OPCODE_TYPE)183)
#define DX10_SM5_OPCODE_IMM_ATOMIC_EXCH ((D3D10_SB_OPCODE_TYPE)184)
#define DX10_SM5_OPCODE_IMM_ATOMIC_CMP_EXCH ((D3D10_SB_OPCODE_TYPE)185)
#define DX10_SM5_OPCODE_IMM_ATOMIC_IMAX ((D3D10_SB_OPCODE_TYPE)186)
#define DX10_SM5_OPCODE_IMM_ATOMIC_IMIN ((D3D10_SB_OPCODE_TYPE)187)
#define DX10_SM5_OPCODE_IMM_ATOMIC_UMAX ((D3D10_SB_OPCODE_TYPE)188)
#define DX10_SM5_OPCODE_IMM_ATOMIC_UMIN ((D3D10_SB_OPCODE_TYPE)189)
#define DX10_SM5_OPCODE_SYNC ((D3D10_SB_OPCODE_TYPE)190)

#define SHADER_MAX_DST_OPERANDS 2
#define SHADER_MAX_SRC_OPERANDS 5

struct Shader_opcode {
   D3D10_SB_OPCODE_TYPE type;
   unsigned num_dst;
   unsigned num_src;
   struct Shader_dst_operand dst[SHADER_MAX_DST_OPERANDS];
   struct Shader_src_operand src[SHADER_MAX_SRC_OPERANDS];

   /* Opcode specific data.
    */
   union {
      D3D10_SB_RESOURCE_DIMENSION dcl_resource_dimension;
      D3D10_SB_SAMPLER_MODE dcl_sampler_mode;
      D3D10_SB_CONSTANT_BUFFER_ACCESS_PATTERN dcl_cb_access_pattern;
      D3D10_SB_INTERPOLATION_MODE dcl_in_ps_interp;
      D3D10_SB_PRIMITIVE_TOPOLOGY dcl_gs_output_primitive_topology;
      D3D10_SB_PRIMITIVE dcl_gs_input_primitive;
      D3D10_SB_INSTRUCTION_TEST_BOOLEAN test_boolean;
      D3D10_SB_RESINFO_INSTRUCTION_RETURN_TYPE resinfo_ret_type;
      unsigned dcl_max_output_vertex_count;
      unsigned dcl_gs_instance_count;
      unsigned dcl_num_temps;
      unsigned dcl_structured_stride;
      struct dx10_indexable_temp dcl_indexable_temp;
      unsigned index_range_count;
      unsigned stream;
      struct dx10_global_flags global_flags;
      struct dx10_thread_group dcl_thread_group;
      struct dx10_tgsm dcl_tgsm;
      unsigned dcl_input_control_point_count;
      unsigned dcl_output_control_point_count;
      unsigned dcl_tess_domain;
      unsigned dcl_tess_partitioning;
      unsigned dcl_tess_output_primitive;
      unsigned dcl_hs_max_tessfactor_bits;
      unsigned dcl_hs_phase_instance_count;
      unsigned sample_pos_compatibility;
   } specific;
   D3D10_SB_NAME dcl_siv_name;
   D3D10_SB_RESOURCE_RETURN_TYPE dcl_resource_ret_type[4];

   bool saturate;

   struct {
      int u:4;
      int v:4;
      int w:4;
   } imm_texel_offset;

   struct dx10_customdata customdata;
};

struct Shader_parser {
   const unsigned *code;
   const unsigned *curr;
   bool failed;

   struct Shader_header header;
};

void
Shader_parse_init(struct Shader_parser *parser,
                       const unsigned *code);

bool
Shader_parse_opcode(struct Shader_parser *parser,
                         struct Shader_opcode *opcode);

void
Shader_opcode_free(struct Shader_opcode *opcode);

bool
Shader_parse_tessellation_properties(
   const unsigned *code,
   struct Shader_tessellation_properties *properties);


const struct tgsi_token *
Shader_tgsi_translate(const unsigned *code,
                      unsigned *output_mapping,
                      unsigned *thread_group_size,
                      unsigned *shared_memory_size,
                      const struct Shader_tessellation_io_signatures *
                      tessellation_signatures,
                      const struct Shader_tessellation_properties *
                      tessellation_properties,
                      const D3D10DDIARG_STAGE_IO_SIGNATURES *stage_signatures,
                      struct pipe_shader_sampler_binding_map *
                      sampler_binding_map,
                      uint8_t *counter_image_slots);


#ifdef __cplusplus
}
#endif

#endif /* SHADER_PARSE_H */
