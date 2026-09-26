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
 * ShaderParse.c --
 *    Functions for parsing shader tokens.
 */

#include "Debug.h"
#include "ShaderParse.h"

#include "gallium/winsys/yttrium/gdi/yttrium_gdi_public.h"

#include "util/u_memory.h"


static bool
dword_count_fits_size_t(unsigned count)
{
   const size_t byte_count = (size_t)count * sizeof(unsigned);

   return !count || byte_count / sizeof(unsigned) == count;
}


void
Shader_parse_init(struct Shader_parser *parser,
                       const unsigned *code)
{
   if (!parser)
      return;

   memset(parser, 0, sizeof(*parser));
   parser->curr = parser->code = code;
   parser->failed = true;

   if (!code) {
      yttrium_gdi_trace_warnf(
         "yttrium: shader parse received NULL bytecode\n");
      return;
   }

   parser->header.type = DECODE_D3D10_SB_TOKENIZED_PROGRAM_TYPE(code[0]);
   parser->header.major_version =
      DECODE_D3D10_SB_TOKENIZED_PROGRAM_MAJOR_VERSION(code[0]);
   parser->header.minor_version =
      DECODE_D3D10_SB_TOKENIZED_PROGRAM_MINOR_VERSION(code[0]);
   parser->header.size = DECODE_D3D10_SB_TOKENIZED_PROGRAM_LENGTH(code[1]);

   if (parser->header.size < 2 ||
       !dword_count_fits_size_t(parser->header.size)) {
      yttrium_gdi_trace_warnf(
         "yttrium: shader parse invalid header size=%u\n",
         parser->header.size);
      return;
   }

   parser->curr = code + 2;
   parser->failed = false;
}

#define OP_NOT_DONE (1 << 0) /* not implemented yet */
#define OP_SATURATE (1 << 1) /* saturate in opcode specific control */
#define OP_TEST_BOOLEAN (1 << 2) /* test boolean in opcode specific control */
#define OP_DCL (1 << 3) /* custom opcode specific control */
#define OP_RESINFO_RET_TYPE (1 << 4) /* return type for resinfo */
#define OP_IGNORE_CONTROL (1 << 5) /* opcode-specific bits are handled elsewhere */

#ifndef D3D11_SB_GLOBAL_FLAG_FORCE_EARLY_DEPTH_STENCIL
#define D3D11_SB_GLOBAL_FLAG_FORCE_EARLY_DEPTH_STENCIL 0x2
#endif

struct dx10_opcode_info {
   D3D10_SB_OPCODE_TYPE type;
   const char *name;
   unsigned num_dst;
   unsigned num_src;
   unsigned flags;
};

#define _(_opcode) _opcode, #_opcode

static const struct dx10_opcode_info
opcode_info[D3D10_SB_NUM_OPCODES] = {
   {_(D3D10_SB_OPCODE_ADD),                              1, 2, OP_SATURATE},
   {_(D3D10_SB_OPCODE_AND),                              1, 2, 0},
   {_(D3D10_SB_OPCODE_BREAK),                            0, 0, 0},
   {_(D3D10_SB_OPCODE_BREAKC),                           0, 1, OP_TEST_BOOLEAN},
   {_(D3D10_SB_OPCODE_CALL),                             0, 1, 0},
   {_(D3D10_SB_OPCODE_CALLC),                            0, 2, OP_TEST_BOOLEAN},
   {_(D3D10_SB_OPCODE_CASE),                             0, 1, 0},
   {_(D3D10_SB_OPCODE_CONTINUE),                         0, 0, 0},
   {_(D3D10_SB_OPCODE_CONTINUEC),                        0, 1, OP_TEST_BOOLEAN},
   {_(D3D10_SB_OPCODE_CUT),                              0, 0, 0},
   {_(D3D10_SB_OPCODE_DEFAULT),                          0, 0, 0},
   {_(D3D10_SB_OPCODE_DERIV_RTX),                        1, 1, OP_SATURATE},
   {_(D3D10_SB_OPCODE_DERIV_RTY),                        1, 1, OP_SATURATE},
   {_(D3D10_SB_OPCODE_DISCARD),                          0, 1, OP_TEST_BOOLEAN},
   {_(D3D10_SB_OPCODE_DIV),                              1, 2, OP_SATURATE},
   {_(D3D10_SB_OPCODE_DP2),                              1, 2, OP_SATURATE},
   {_(D3D10_SB_OPCODE_DP3),                              1, 2, OP_SATURATE},
   {_(D3D10_SB_OPCODE_DP4),                              1, 2, OP_SATURATE},
   {_(D3D10_SB_OPCODE_ELSE),                             0, 0, 0},
   {_(D3D10_SB_OPCODE_EMIT),                             0, 0, 0},
   {_(D3D10_SB_OPCODE_EMITTHENCUT),                      0, 0, 0},
   {_(D3D10_SB_OPCODE_ENDIF),                            0, 0, 0},
   {_(D3D10_SB_OPCODE_ENDLOOP),                          0, 0, 0},
   {_(D3D10_SB_OPCODE_ENDSWITCH),                        0, 0, 0},
   {_(D3D10_SB_OPCODE_EQ),                               1, 2, 0},
   {_(D3D10_SB_OPCODE_EXP),                              1, 1, OP_SATURATE},
   {_(D3D10_SB_OPCODE_FRC),                              1, 1, OP_SATURATE},
   {_(D3D10_SB_OPCODE_FTOI),                             1, 1, 0},
   {_(D3D10_SB_OPCODE_FTOU),                             1, 1, 0},
   {_(D3D10_SB_OPCODE_GE),                               1, 2, 0},
   {_(D3D10_SB_OPCODE_IADD),                             1, 2, 0},
   {_(D3D10_SB_OPCODE_IF),                               0, 1, OP_TEST_BOOLEAN},
   {_(D3D10_SB_OPCODE_IEQ),                              1, 2, 0},
   {_(D3D10_SB_OPCODE_IGE),                              1, 2, 0},
   {_(D3D10_SB_OPCODE_ILT),                              1, 2, 0},
   {_(D3D10_SB_OPCODE_IMAD),                             1, 3, 0},
   {_(D3D10_SB_OPCODE_IMAX),                             1, 2, 0},
   {_(D3D10_SB_OPCODE_IMIN),                             1, 2, 0},
   {_(D3D10_SB_OPCODE_IMUL),                             2, 2, 0},
   {_(D3D10_SB_OPCODE_INE),                              1, 2, 0},
   {_(D3D10_SB_OPCODE_INEG),                             1, 1, 0},
   {_(D3D10_SB_OPCODE_ISHL),                             1, 2, 0},
   {_(D3D10_SB_OPCODE_ISHR),                             1, 2, 0},
   {_(D3D10_SB_OPCODE_ITOF),                             1, 1, 0},
   {_(D3D10_SB_OPCODE_LABEL),                            0, 1, 0},
   {_(D3D10_SB_OPCODE_LD),                               1, 2, 0},
   {_(D3D10_SB_OPCODE_LD_MS),                            1, 3, 0},
   {_(D3D10_SB_OPCODE_LOG),                              1, 1, OP_SATURATE},
   {_(D3D10_SB_OPCODE_LOOP),                             0, 0, 0},
   {_(D3D10_SB_OPCODE_LT),                               1, 2, 0},
   {_(D3D10_SB_OPCODE_MAD),                              1, 3, OP_SATURATE},
   {_(D3D10_SB_OPCODE_MIN),                              1, 2, OP_SATURATE},
   {_(D3D10_SB_OPCODE_MAX),                              1, 2, OP_SATURATE},
   {_(D3D10_SB_OPCODE_CUSTOMDATA),                       0, 0, 0},
   {_(D3D10_SB_OPCODE_MOV),                              1, 1, OP_SATURATE},
   {_(D3D10_SB_OPCODE_MOVC),                             1, 3, OP_SATURATE},
   {_(D3D10_SB_OPCODE_MUL),                              1, 2, OP_SATURATE},
   {_(D3D10_SB_OPCODE_NE),                               1, 2, 0},
   {_(D3D10_SB_OPCODE_NOP),                              0, 0, 0},
   {_(D3D10_SB_OPCODE_NOT),                              1, 1, 0},
   {_(D3D10_SB_OPCODE_OR),                               1, 2, 0},
   {_(D3D10_SB_OPCODE_RESINFO),                          1, 2, OP_RESINFO_RET_TYPE},
   {_(D3D10_SB_OPCODE_RET),                              0, 0, 0},
   {_(D3D10_SB_OPCODE_RETC),                             0, 1, OP_TEST_BOOLEAN},
   {_(D3D10_SB_OPCODE_ROUND_NE),                         1, 1, OP_SATURATE},
   {_(D3D10_SB_OPCODE_ROUND_NI),                         1, 1, OP_SATURATE},
   {_(D3D10_SB_OPCODE_ROUND_PI),                         1, 1, OP_SATURATE},
   {_(D3D10_SB_OPCODE_ROUND_Z),                          1, 1, OP_SATURATE},
   {_(D3D10_SB_OPCODE_RSQ),                              1, 1, OP_SATURATE},
   {_(D3D10_SB_OPCODE_SAMPLE),                           1, 3, 0},
   {_(D3D10_SB_OPCODE_SAMPLE_C),                         1, 4, 0},
   {_(D3D10_SB_OPCODE_SAMPLE_C_LZ),                      1, 4, 0},
   {_(D3D10_SB_OPCODE_SAMPLE_L),                         1, 4, 0},
   {_(D3D10_SB_OPCODE_SAMPLE_D),                         1, 5, 0},
   {_(D3D10_SB_OPCODE_SAMPLE_B),                         1, 4, 0},
   {_(D3D10_SB_OPCODE_SQRT),                             1, 1, OP_SATURATE},
   {_(D3D10_SB_OPCODE_SWITCH),                           0, 1, 0},
   {_(D3D10_SB_OPCODE_SINCOS),                           2, 1, OP_SATURATE},
   {_(D3D10_SB_OPCODE_UDIV),                             2, 2, 0},
   {_(D3D10_SB_OPCODE_ULT),                              1, 2, 0},
   {_(D3D10_SB_OPCODE_UGE),                              1, 2, 0},
   {_(D3D10_SB_OPCODE_UMUL),                             2, 2, 0},
   {_(D3D10_SB_OPCODE_UMAD),                             1, 3, 0},
   {_(D3D10_SB_OPCODE_UMAX),                             1, 2, 0},
   {_(D3D10_SB_OPCODE_UMIN),                             1, 2, 0},
   {_(D3D10_SB_OPCODE_USHR),                             1, 2, 0},
   {_(D3D10_SB_OPCODE_UTOF),                             1, 1, 0},
   {_(D3D10_SB_OPCODE_XOR),                              1, 2, 0},
   {_(D3D10_SB_OPCODE_DCL_RESOURCE),                     1, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_CONSTANT_BUFFER),              0, 1, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_SAMPLER),                      1, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_INDEX_RANGE),                  1, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_GS_OUTPUT_PRIMITIVE_TOPOLOGY), 0, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_GS_INPUT_PRIMITIVE),           0, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_MAX_OUTPUT_VERTEX_COUNT),      0, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_INPUT),                        1, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_INPUT_SGV),                    1, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_INPUT_SIV),                    1, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_INPUT_PS),                     1, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_INPUT_PS_SGV),                 1, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_INPUT_PS_SIV),                 1, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_OUTPUT),                       1, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_OUTPUT_SGV),                   1, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_OUTPUT_SIV),                   1, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_TEMPS),                        0, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_INDEXABLE_TEMP),               0, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_DCL_GLOBAL_FLAGS),                 0, 0, OP_DCL},
   {_(D3D10_SB_OPCODE_RESERVED0),                        0, 0, OP_NOT_DONE},
   {_(D3D10_1_SB_OPCODE_LOD),                            1, 3, 0},
   {_(D3D10_1_SB_OPCODE_GATHER4),                        1, 3, 0},
   {_(D3D10_1_SB_OPCODE_SAMPLE_POS),                     1, 2, 0},
   {_(D3D10_1_SB_OPCODE_SAMPLE_INFO),                    1, 1, OP_IGNORE_CONTROL},
   [D3D11_SB_OPCODE_EMIT_STREAM] = {
      _(D3D11_SB_OPCODE_EMIT_STREAM),                    0, 0, 0},
   [D3D11_SB_OPCODE_CUT_STREAM] = {
      _(D3D11_SB_OPCODE_CUT_STREAM),                     0, 0, 0},
   [D3D11_SB_OPCODE_EMITTHENCUT_STREAM] = {
      _(D3D11_SB_OPCODE_EMITTHENCUT_STREAM),             0, 0, 0},
   [D3D11_SB_OPCODE_DCL_STREAM] = {
      _(D3D11_SB_OPCODE_DCL_STREAM),                     0, 0, OP_DCL},
   [DX11_SM5_OPCODE_HS_DECLS] = {
      _(DX11_SM5_OPCODE_HS_DECLS),                        0, 0, OP_DCL},
   [DX11_SM5_OPCODE_HS_CONTROL_POINT_PHASE] = {
      _(DX11_SM5_OPCODE_HS_CONTROL_POINT_PHASE),          0, 0, OP_DCL},
   [DX11_SM5_OPCODE_HS_FORK_PHASE] = {
      _(DX11_SM5_OPCODE_HS_FORK_PHASE),                   0, 0, OP_DCL},
   [DX11_SM5_OPCODE_HS_JOIN_PHASE] = {
      _(DX11_SM5_OPCODE_HS_JOIN_PHASE),                   0, 0, OP_DCL},
   [DX11_SM5_OPCODE_DCL_INPUT_CONTROL_POINT_COUNT] = {
      _(DX11_SM5_OPCODE_DCL_INPUT_CONTROL_POINT_COUNT),   0, 0, OP_DCL},
   [DX11_SM5_OPCODE_DCL_OUTPUT_CONTROL_POINT_COUNT] = {
      _(DX11_SM5_OPCODE_DCL_OUTPUT_CONTROL_POINT_COUNT),  0, 0, OP_DCL},
   [DX11_SM5_OPCODE_DCL_TESS_DOMAIN] = {
      _(DX11_SM5_OPCODE_DCL_TESS_DOMAIN),                 0, 0, OP_DCL},
   [DX11_SM5_OPCODE_DCL_TESS_PARTITIONING] = {
      _(DX11_SM5_OPCODE_DCL_TESS_PARTITIONING),           0, 0, OP_DCL},
   [DX11_SM5_OPCODE_DCL_TESS_OUTPUT_PRIMITIVE] = {
      _(DX11_SM5_OPCODE_DCL_TESS_OUTPUT_PRIMITIVE),       0, 0, OP_DCL},
   [DX11_SM5_OPCODE_DCL_HS_MAX_TESSFACTOR] = {
      _(DX11_SM5_OPCODE_DCL_HS_MAX_TESSFACTOR),           0, 0, OP_DCL},
   [DX11_SM5_OPCODE_DCL_HS_FORK_PHASE_INSTANCE_COUNT] = {
      _(DX11_SM5_OPCODE_DCL_HS_FORK_PHASE_INSTANCE_COUNT), 0, 0, OP_DCL},
   [DX11_SM5_OPCODE_DCL_HS_JOIN_PHASE_INSTANCE_COUNT] = {
      _(DX11_SM5_OPCODE_DCL_HS_JOIN_PHASE_INSTANCE_COUNT), 0, 0, OP_DCL},
   [D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_TYPED] = {
      _(D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_TYPED), 1, 0, OP_DCL},
   [D3D11_SB_OPCODE_LD_UAV_TYPED] = {
      _(D3D11_SB_OPCODE_LD_UAV_TYPED),                  1, 2, 0},
   [D3D11_SB_OPCODE_STORE_UAV_TYPED] = {
      _(D3D11_SB_OPCODE_STORE_UAV_TYPED),               1, 2, 0},
   [D3D11_SB_OPCODE_GATHER4_C] = {
      _(D3D11_SB_OPCODE_GATHER4_C),                     1, 4, 0},
   [D3D11_SB_OPCODE_GATHER4_PO] = {
      _(D3D11_SB_OPCODE_GATHER4_PO),                    1, 4, 0},
   [D3D11_SB_OPCODE_GATHER4_PO_C] = {
      _(D3D11_SB_OPCODE_GATHER4_PO_C),                  1, 5, 0},
   [D3D11_SB_OPCODE_EVAL_SAMPLE_INDEX] = {
      _(D3D11_SB_OPCODE_EVAL_SAMPLE_INDEX),            1, 2, 0},
   [DX10_SM5_OPCODE_DCL_THREAD_GROUP] = {
      _(DX10_SM5_OPCODE_DCL_THREAD_GROUP),              0, 0, OP_DCL},
   [DX10_SM5_OPCODE_DCL_UAV_RAW] = {
      _(DX10_SM5_OPCODE_DCL_UAV_RAW),                   1, 0, OP_DCL},
   [DX10_SM5_OPCODE_DCL_UAV_STRUCTURED] = {
      _(DX10_SM5_OPCODE_DCL_UAV_STRUCTURED),            1, 0, OP_DCL},
   [DX10_SM5_OPCODE_DCL_TGSM_RAW] = {
      _(DX10_SM5_OPCODE_DCL_TGSM_RAW),                  1, 0, OP_DCL},
   [DX10_SM5_OPCODE_DCL_TGSM_STRUCTURED] = {
      _(DX10_SM5_OPCODE_DCL_TGSM_STRUCTURED),           1, 0, OP_DCL},
   [DX10_SM5_OPCODE_DCL_RESOURCE_RAW] = {
      _(DX10_SM5_OPCODE_DCL_RESOURCE_RAW),              1, 0, OP_DCL},
   [DX10_SM5_OPCODE_DCL_RESOURCE_STRUCTURED] = {
      _(DX10_SM5_OPCODE_DCL_RESOURCE_STRUCTURED),       1, 0, OP_DCL},
   [DX10_SM5_OPCODE_LD_RAW] = {
      _(DX10_SM5_OPCODE_LD_RAW),                        1, 2, 0},
   [DX10_SM5_OPCODE_STORE_RAW] = {
      _(DX10_SM5_OPCODE_STORE_RAW),                     1, 2, 0},
   [DX10_SM5_OPCODE_LD_STRUCTURED] = {
      _(DX10_SM5_OPCODE_LD_STRUCTURED),                 1, 3, 0},
   [DX10_SM5_OPCODE_STORE_STRUCTURED] = {
      _(DX10_SM5_OPCODE_STORE_STRUCTURED),              1, 3, 0},
   [DX10_SM5_OPCODE_ATOMIC_AND] = {
      _(DX10_SM5_OPCODE_ATOMIC_AND),                    1, 2, 0},
   [DX10_SM5_OPCODE_ATOMIC_OR] = {
      _(DX10_SM5_OPCODE_ATOMIC_OR),                     1, 2, 0},
   [DX10_SM5_OPCODE_ATOMIC_XOR] = {
      _(DX10_SM5_OPCODE_ATOMIC_XOR),                    1, 2, 0},
   [DX10_SM5_OPCODE_ATOMIC_CMP_STORE] = {
      _(DX10_SM5_OPCODE_ATOMIC_CMP_STORE),              1, 3, 0},
   [DX10_SM5_OPCODE_ATOMIC_IADD] = {
      _(DX10_SM5_OPCODE_ATOMIC_IADD),                   1, 2, 0},
   [DX10_SM5_OPCODE_ATOMIC_IMAX] = {
      _(DX10_SM5_OPCODE_ATOMIC_IMAX),                   1, 2, 0},
   [DX10_SM5_OPCODE_ATOMIC_IMIN] = {
      _(DX10_SM5_OPCODE_ATOMIC_IMIN),                   1, 2, 0},
   [DX10_SM5_OPCODE_ATOMIC_UMAX] = {
      _(DX10_SM5_OPCODE_ATOMIC_UMAX),                   1, 2, 0},
   [DX10_SM5_OPCODE_ATOMIC_UMIN] = {
      _(DX10_SM5_OPCODE_ATOMIC_UMIN),                   1, 2, 0},
   [DX10_SM5_OPCODE_IMM_ATOMIC_ALLOC] = {
      _(DX10_SM5_OPCODE_IMM_ATOMIC_ALLOC),              2, 0, 0},
   [DX10_SM5_OPCODE_IMM_ATOMIC_CONSUME] = {
      _(DX10_SM5_OPCODE_IMM_ATOMIC_CONSUME),            2, 0, 0},
   [DX10_SM5_OPCODE_IMM_ATOMIC_IADD] = {
      _(DX10_SM5_OPCODE_IMM_ATOMIC_IADD),               2, 2, 0},
   [DX10_SM5_OPCODE_IMM_ATOMIC_AND] = {
      _(DX10_SM5_OPCODE_IMM_ATOMIC_AND),                2, 2, 0},
   [DX10_SM5_OPCODE_IMM_ATOMIC_OR] = {
      _(DX10_SM5_OPCODE_IMM_ATOMIC_OR),                 2, 2, 0},
   [DX10_SM5_OPCODE_IMM_ATOMIC_XOR] = {
      _(DX10_SM5_OPCODE_IMM_ATOMIC_XOR),                2, 2, 0},
   [DX10_SM5_OPCODE_IMM_ATOMIC_EXCH] = {
      _(DX10_SM5_OPCODE_IMM_ATOMIC_EXCH),               2, 2, 0},
   [DX10_SM5_OPCODE_IMM_ATOMIC_CMP_EXCH] = {
      _(DX10_SM5_OPCODE_IMM_ATOMIC_CMP_EXCH),           2, 3, 0},
   [DX10_SM5_OPCODE_IMM_ATOMIC_IMAX] = {
      _(DX10_SM5_OPCODE_IMM_ATOMIC_IMAX),               2, 2, 0},
   [DX10_SM5_OPCODE_IMM_ATOMIC_IMIN] = {
      _(DX10_SM5_OPCODE_IMM_ATOMIC_IMIN),               2, 2, 0},
   [DX10_SM5_OPCODE_IMM_ATOMIC_UMAX] = {
      _(DX10_SM5_OPCODE_IMM_ATOMIC_UMAX),               2, 2, 0},
   [DX10_SM5_OPCODE_IMM_ATOMIC_UMIN] = {
      _(DX10_SM5_OPCODE_IMM_ATOMIC_UMIN),               2, 2, 0},
   [DX10_SM5_OPCODE_SYNC] = {
      _(DX10_SM5_OPCODE_SYNC),                          0, 0, OP_IGNORE_CONTROL},
   [DX10_SM5_OPCODE_BUFINFO] = {
      _(DX10_SM5_OPCODE_BUFINFO),                       1, 1, 0},
   [DX10_SM5_OPCODE_DERIV_RTX_COARSE] = {
      _(DX10_SM5_OPCODE_DERIV_RTX_COARSE),             1, 1, OP_SATURATE},
   [DX10_SM5_OPCODE_DERIV_RTX_FINE] = {
      _(DX10_SM5_OPCODE_DERIV_RTX_FINE),               1, 1, OP_SATURATE},
   [DX10_SM5_OPCODE_DERIV_RTY_COARSE] = {
      _(DX10_SM5_OPCODE_DERIV_RTY_COARSE),             1, 1, OP_SATURATE},
   [DX10_SM5_OPCODE_DERIV_RTY_FINE] = {
      _(DX10_SM5_OPCODE_DERIV_RTY_FINE),               1, 1, OP_SATURATE},
   [DX10_SM5_OPCODE_RCP] = {
      _(DX10_SM5_OPCODE_RCP),                          1, 1, OP_SATURATE},
   [DX10_SM5_OPCODE_F32TOF16] = {
      _(DX10_SM5_OPCODE_F32TOF16),                       1, 1, 0},
   [DX10_SM5_OPCODE_F16TOF32] = {
      _(DX10_SM5_OPCODE_F16TOF32),                       1, 1, 0},
   [DX10_SM5_OPCODE_COUNTBITS] = {
      _(DX10_SM5_OPCODE_COUNTBITS),                      1, 1, 0},
   [DX10_SM5_OPCODE_FIRSTBIT_HI] = {
      _(DX10_SM5_OPCODE_FIRSTBIT_HI),                    1, 1, 0},
   [DX10_SM5_OPCODE_FIRSTBIT_LO] = {
      _(DX10_SM5_OPCODE_FIRSTBIT_LO),                    1, 1, 0},
   [DX10_SM5_OPCODE_FIRSTBIT_SHI] = {
      _(DX10_SM5_OPCODE_FIRSTBIT_SHI),                   1, 1, 0},
   [DX10_SM5_OPCODE_UBFE] = {
      _(DX10_SM5_OPCODE_UBFE),                           1, 3, 0},
   [DX10_SM5_OPCODE_IBFE] = {
      _(DX10_SM5_OPCODE_IBFE),                           1, 3, 0},
   [DX10_SM5_OPCODE_BFI] = {
      _(DX10_SM5_OPCODE_BFI),                            1, 4, 0},
   [DX10_SM5_OPCODE_BFREV] = {
      _(DX10_SM5_OPCODE_BFREV),                          1, 1, 0},
   [DX10_SM5_OPCODE_SWAPC] = {
      _(DX10_SM5_OPCODE_SWAPC),                          2, 3, 0},
   [D3D11_SB_OPCODE_DCL_GS_INSTANCE_COUNT] = {
      _(D3D11_SB_OPCODE_DCL_GS_INSTANCE_COUNT),          0, 0, OP_DCL}
};

#undef _

static bool
read_token(const unsigned **curr, const unsigned *end, unsigned *value)
{
   if (!curr || !*curr || !value || *curr >= end)
      return false;

   *value = *(*curr)++;
   return true;
}

static bool
parse_operand(const unsigned **curr,
              const unsigned *end,
              struct Shader_operand *operand)
{
   unsigned token;

   if (!operand || !read_token(curr, end, &token))
      return false;

   operand->type = DECODE_D3D10_SB_OPERAND_TYPE(token);

   /* Index dimension. */
   switch (DECODE_D3D10_SB_OPERAND_INDEX_DIMENSION(token)) {
   case D3D10_SB_OPERAND_INDEX_0D:
      operand->index_dim = 0;
      break;
   case D3D10_SB_OPERAND_INDEX_1D:
      operand->index_dim = 1;
      break;
   case D3D10_SB_OPERAND_INDEX_2D:
      operand->index_dim = 2;
      break;
   default:
      return false;
   }

   if (operand->index_dim >= 1) {
      operand->index[0].index_rep =
         DECODE_D3D10_SB_OPERAND_INDEX_REPRESENTATION(0, token);
      if (operand->index_dim >= 2) {
         operand->index[1].index_rep =
            DECODE_D3D10_SB_OPERAND_INDEX_REPRESENTATION(1, token);
      }
   }

   return true;
}

static bool
parse_relative_operand(const unsigned **curr,
                       const unsigned *end,
                       struct Shader_relative_operand *operand)
{
   D3D10_SB_OPERAND_INDEX_DIMENSION index_dim;
   unsigned token;

   if (!operand || !read_token(curr, end, &token))
      return false;

   if (DECODE_IS_D3D10_SB_OPERAND_EXTENDED(token) ||
       DECODE_D3D10_SB_OPERAND_NUM_COMPONENTS(token) !=
          D3D10_SB_OPERAND_4_COMPONENT ||
       DECODE_D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE(token) !=
          D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_MODE)
      return false;

   operand->comp = DECODE_D3D10_SB_OPERAND_4_COMPONENT_SELECT_1(token);
   operand->type = DECODE_D3D10_SB_OPERAND_TYPE(token);
   if (operand->type == D3D10_SB_OPERAND_TYPE_IMMEDIATE32)
      return false;

   index_dim = DECODE_D3D10_SB_OPERAND_INDEX_DIMENSION(token);
   if (index_dim != D3D10_SB_OPERAND_INDEX_1D &&
       index_dim != D3D10_SB_OPERAND_INDEX_2D)
      return false;

   if (DECODE_D3D10_SB_OPERAND_INDEX_REPRESENTATION(0, token) !=
       D3D10_SB_OPERAND_INDEX_IMMEDIATE32)
      return false;
   if (index_dim == D3D10_SB_OPERAND_INDEX_2D &&
       DECODE_D3D10_SB_OPERAND_INDEX_REPRESENTATION(1, token) !=
          D3D10_SB_OPERAND_INDEX_IMMEDIATE32)
      return false;

   if (!read_token(curr, end, &operand->index[0].imm))
      return false;
   if (index_dim == D3D10_SB_OPERAND_INDEX_2D &&
       !read_token(curr, end, &operand->index[1].imm))
      return false;

   return true;
}

static bool
parse_index(const unsigned **curr,
            const unsigned *end,
            struct Shader_index *index)
{
   if (!index)
      return false;

   switch (index->index_rep) {
   case D3D10_SB_OPERAND_INDEX_IMMEDIATE32:
      return read_token(curr, end, &index->imm);
   case D3D10_SB_OPERAND_INDEX_RELATIVE:
      index->imm = 0;
      return parse_relative_operand(curr, end, &index->rel);
   case D3D10_SB_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE:
      return read_token(curr, end, &index->imm) &&
             parse_relative_operand(curr, end, &index->rel);
   default:
      /* Other index representations are not implemented. */
      return false;
   }
}

static bool
parse_operand_index(const unsigned **curr,
                    const unsigned *end,
                    struct Shader_operand *operand)
{
   if (!operand || operand->index_dim > 2)
      return false;

   if (operand->index_dim >= 1 &&
       !parse_index(curr, end, &operand->index[0]))
      return false;
   if (operand->index_dim >= 2 &&
       !parse_index(curr, end, &operand->index[1]))
      return false;

   return true;
}

static bool
parse_stream_operand(const unsigned **curr,
                     const unsigned *end,
                     struct Shader_opcode *opcode)
{
   struct Shader_operand stream;

   if (!opcode)
      return false;

   memset(&stream, 0, sizeof(stream));
   if (!parse_operand(curr, end, &stream) || stream.index_dim != 1 ||
       stream.index[0].index_rep != D3D10_SB_OPERAND_INDEX_IMMEDIATE32 ||
       !parse_operand_index(curr, end, &stream))
      return false;

   opcode->specific.stream = stream.index[0].imm;
   return true;
}

bool
Shader_parse_opcode(struct Shader_parser *parser,
                         struct Shader_opcode *opcode)
{
   const struct dx10_opcode_info *info;
   const unsigned *instruction_start;
   const unsigned *instruction_end;
   const unsigned *shader_end;
   const unsigned *curr;
   unsigned token0;
   unsigned length;
   unsigned i;

   if (!parser || !opcode) {
      if (parser)
         parser->failed = true;
      yttrium_gdi_trace_warnf(
         "yttrium: shader parse received invalid parser or opcode output\n");
      return false;
   }

   if (parser->failed)
      return false;

   if (!parser->code || !parser->curr || parser->header.size < 2 ||
       !dword_count_fits_size_t(parser->header.size)) {
      yttrium_gdi_trace_warnf(
         "yttrium: shader parse has invalid parser state shader_size=%u\n",
         parser->header.size);
      parser->failed = true;
      return false;
   }

   shader_end = parser->code + parser->header.size;
   instruction_start = curr = parser->curr;
   if (curr == shader_end)
      return false;
   if (curr < parser->code + 2 || curr > shader_end) {
      yttrium_gdi_trace_warnf(
         "yttrium: shader parse cursor out of bounds shader_size=%u\n",
         parser->header.size);
      parser->failed = true;
      return false;
   }

   memset(opcode, 0, sizeof(*opcode));
   token0 = *curr;
   opcode->type = DECODE_D3D10_SB_OPCODE_TYPE(token0);
   if (opcode->type >= D3D10_SB_NUM_OPCODES) {
      yttrium_gdi_trace_warnf(
         "yttrium: shader parse invalid opcode type=%u offset=%u shader_size=%u\n",
         opcode->type, (unsigned)(instruction_start - parser->code),
         parser->header.size);
      parser->failed = true;
      return false;
   }

   info = &opcode_info[opcode->type];
   if (!info->name || info->type != opcode->type ||
       info->num_dst > SHADER_MAX_DST_OPERANDS ||
       info->num_src > SHADER_MAX_SRC_OPERANDS) {
      yttrium_gdi_trace_warnf(
         "yttrium: shader parse missing or mismatched opcode metadata "
         "type=%u metadata_type=%u offset=%u\n",
         opcode->type, info->type,
         (unsigned)(instruction_start - parser->code));
      parser->failed = true;
      return false;
   }

   if (opcode->type == D3D10_SB_OPCODE_CUSTOMDATA) {
      unsigned custom_length;
      size_t allocation_size;

      opcode->customdata._class = DECODE_D3D10_SB_CUSTOMDATA_CLASS(token0);
      if (opcode->customdata._class !=
          D3D10_SB_CUSTOMDATA_DCL_IMMEDIATE_CONSTANT_BUFFER ||
          shader_end - instruction_start < 2) {
         yttrium_gdi_trace_warnf(
            "yttrium: shader parse unsupported or truncated custom data "
            "class=%u offset=%u shader_size=%u\n",
            opcode->customdata._class,
            (unsigned)(instruction_start - parser->code),
            parser->header.size);
         parser->failed = true;
         return false;
      }

      custom_length = instruction_start[1];
      if (custom_length < 2 ||
          custom_length > (unsigned)(shader_end - instruction_start)) {
         yttrium_gdi_trace_warnf(
            "yttrium: shader parse custom data length out of bounds "
            "offset=%u length=%u shader_size=%u\n",
            (unsigned)(instruction_start - parser->code), custom_length,
            parser->header.size);
         parser->failed = true;
         return false;
      }

      if ((custom_length - 2) % 4) {
         yttrium_gdi_trace_warnf(
            "yttrium: shader parse immediate constant buffer length "
            "is not vec4-aligned offset=%u length=%u payload=%u\n",
            (unsigned)(instruction_start - parser->code), custom_length,
            custom_length - 2);
         parser->failed = true;
         return false;
      }

      instruction_end = instruction_start + custom_length;
      opcode->customdata.u.constbuf.count = custom_length - 2;
      if (!dword_count_fits_size_t(
             opcode->customdata.u.constbuf.count)) {
         yttrium_gdi_trace_warnf(
            "yttrium: shader parse custom data allocation overflow "
            "offset=%u count=%u\n",
            (unsigned)(instruction_start - parser->code),
            opcode->customdata.u.constbuf.count);
         parser->failed = true;
         return false;
      }

      allocation_size = (size_t)opcode->customdata.u.constbuf.count *
                        sizeof(*opcode->customdata.u.constbuf.data);
      if (allocation_size) {
         opcode->customdata.u.constbuf.data = MALLOC(allocation_size);
         if (!opcode->customdata.u.constbuf.data) {
            yttrium_gdi_trace_warnf(
               "yttrium: shader parse custom data allocation failed "
               "offset=%u count=%u\n",
               (unsigned)(instruction_start - parser->code),
               opcode->customdata.u.constbuf.count);
            parser->failed = true;
            return false;
         }
         memcpy(opcode->customdata.u.constbuf.data,
                instruction_start + 2, allocation_size);
      }

      parser->curr = instruction_end;
      return true;
   }

   /* Decode and validate token0 before touching any instruction payload. */
   length = DECODE_D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH(token0);
   if (!length ||
       length > (unsigned)(shader_end - instruction_start)) {
      yttrium_gdi_trace_warnf(
         "yttrium: shader parse instruction length out of bounds "
         "opcode=%s type=%u offset=%u length=%u shader_size=%u\n",
         info->name, opcode->type,
         (unsigned)(instruction_start - parser->code), length,
         parser->header.size);
      parser->failed = true;
      return false;
   }
   instruction_end = instruction_start + length;

   if (info->flags & OP_NOT_DONE) {
      yttrium_gdi_trace_warnf(
         "yttrium: shader parse unsupported opcode opcode=%s type=%u "
         "offset=%u length=%u\n",
         info->name, opcode->type,
         (unsigned)(instruction_start - parser->code), length);
      parser->failed = true;
      return false;
   }

   opcode->dcl_siv_name = D3D10_SB_NAME_UNDEFINED;

   /* Opcode-specific fields carried by token0. */
   switch (opcode->type) {
   case D3D10_SB_OPCODE_DCL_RESOURCE:
   case D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_TYPED:
      opcode->specific.dcl_resource_dimension =
         DECODE_D3D10_SB_RESOURCE_DIMENSION(token0);
      break;
   case D3D10_SB_OPCODE_DCL_SAMPLER:
      opcode->specific.dcl_sampler_mode =
         DECODE_D3D10_SB_SAMPLER_MODE(token0);
      break;
   case D3D10_SB_OPCODE_DCL_GS_OUTPUT_PRIMITIVE_TOPOLOGY:
      opcode->specific.dcl_gs_output_primitive_topology =
         DECODE_D3D10_SB_GS_OUTPUT_PRIMITIVE_TOPOLOGY(token0);
      break;
   case D3D10_SB_OPCODE_DCL_GS_INPUT_PRIMITIVE:
      opcode->specific.dcl_gs_input_primitive =
         DECODE_D3D10_SB_GS_INPUT_PRIMITIVE(token0);
      break;
   case D3D10_SB_OPCODE_DCL_INPUT_PS:
   case D3D10_SB_OPCODE_DCL_INPUT_PS_SIV:
      opcode->specific.dcl_in_ps_interp =
         DECODE_D3D10_SB_INPUT_INTERPOLATION_MODE(token0);
      break;
   case D3D10_SB_OPCODE_DCL_CONSTANT_BUFFER:
      opcode->specific.dcl_cb_access_pattern =
         DECODE_D3D10_SB_CONSTANT_BUFFER_ACCESS_PATTERN(token0);
      break;
   case D3D10_SB_OPCODE_DCL_GLOBAL_FLAGS:
      opcode->specific.global_flags.refactoring_allowed =
         (DECODE_D3D10_SB_GLOBAL_FLAGS(token0) &
          D3D10_SB_GLOBAL_FLAG_REFACTORING_ALLOWED) ? 1 : 0;
      opcode->specific.global_flags.force_early_depth_stencil =
         (DECODE_D3D10_SB_GLOBAL_FLAGS(token0) &
          D3D11_SB_GLOBAL_FLAG_FORCE_EARLY_DEPTH_STENCIL) ? 1 : 0;
      break;
   case DX11_SM5_OPCODE_DCL_INPUT_CONTROL_POINT_COUNT:
      opcode->specific.dcl_input_control_point_count = (token0 >> 11) & 0x3f;
      break;
   case DX11_SM5_OPCODE_DCL_OUTPUT_CONTROL_POINT_COUNT:
      opcode->specific.dcl_output_control_point_count = (token0 >> 11) & 0x3f;
      break;
   case DX11_SM5_OPCODE_DCL_TESS_DOMAIN:
      opcode->specific.dcl_tess_domain = (token0 >> 11) & 0x3;
      break;
   case DX11_SM5_OPCODE_DCL_TESS_PARTITIONING:
      opcode->specific.dcl_tess_partitioning = (token0 >> 11) & 0x7;
      break;
   case DX11_SM5_OPCODE_DCL_TESS_OUTPUT_PRIMITIVE:
      opcode->specific.dcl_tess_output_primitive = (token0 >> 11) & 0x7;
      break;
   default:
      if (info->flags & OP_DCL) {
         /* No generic control field. */
      } else if (info->flags & OP_SATURATE) {
         opcode->saturate =
            !!DECODE_IS_D3D10_SB_INSTRUCTION_SATURATE_ENABLED(token0);
      } else if (info->flags & OP_TEST_BOOLEAN) {
         opcode->specific.test_boolean =
            DECODE_D3D10_SB_INSTRUCTION_TEST_BOOLEAN(token0);
      } else if (info->flags & OP_RESINFO_RET_TYPE) {
         opcode->specific.resinfo_ret_type =
            DECODE_D3D10_SB_RESINFO_INSTRUCTION_RETURN_TYPE(token0);
      } else if (!(info->flags & OP_IGNORE_CONTROL) &&
                 (token0 & ((1 << 24) - (1 << 11)))) {
         debug_printf(
            "warning: unexpected opcode-specific control in opcode %s\n",
            info->name);
      }
      break;
   }

   curr++;
   {
      bool opcode_is_extended =
         DECODE_IS_D3D10_SB_OPCODE_EXTENDED(token0);

      while (opcode_is_extended) {
         unsigned extended_token;

         if (!read_token(&curr, instruction_end, &extended_token))
            goto malformed;

         /* The SDK's opcode-double-extended decoder is broken; this is the
          * equivalent continuation bit used by the existing parser.
          */
         opcode_is_extended =
            !!((extended_token & D3D10_SB_OPERAND_DOUBLE_EXTENDED_MASK) >>
               D3D10_SB_OPERAND_DOUBLE_EXTENDED_SHIFT);

         switch (DECODE_D3D10_SB_EXTENDED_OPCODE_TYPE(extended_token)) {
         case D3D10_SB_EXTENDED_OPCODE_EMPTY:
            break;
         case D3D10_SB_EXTENDED_OPCODE_SAMPLE_CONTROLS:
            opcode->imm_texel_offset.u =
               DECODE_IMMEDIATE_D3D10_SB_ADDRESS_OFFSET(
                  D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_U, extended_token);
            opcode->imm_texel_offset.v =
               DECODE_IMMEDIATE_D3D10_SB_ADDRESS_OFFSET(
                  D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_V, extended_token);
            opcode->imm_texel_offset.w =
               DECODE_IMMEDIATE_D3D10_SB_ADDRESS_OFFSET(
                  D3D10_SB_IMMEDIATE_ADDRESS_OFFSET_W, extended_token);
            break;
         case D3D11_SB_EXTENDED_OPCODE_RESOURCE_DIM:
         case D3D11_SB_EXTENDED_OPCODE_RESOURCE_RETURN_TYPE:
            break;
         default:
            goto malformed;
         }
      }
   }

   opcode->num_dst = info->num_dst;
   opcode->num_src = info->num_src;

   /* Destination operands. */
   for (i = 0; i < info->num_dst; i++) {
      D3D10_SB_OPERAND_NUM_COMPONENTS num_components;
      unsigned operand_token;
      bool extended;

      if (curr >= instruction_end)
         goto malformed;
      operand_token = *curr;
      extended = DECODE_IS_D3D10_SB_OPERAND_EXTENDED(operand_token);
      num_components = DECODE_D3D10_SB_OPERAND_NUM_COMPONENTS(operand_token);

      if (num_components == D3D10_SB_OPERAND_4_COMPONENT) {
         if (DECODE_D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE(
                operand_token) != D3D10_SB_OPERAND_4_COMPONENT_MASK_MODE)
            goto malformed;
         opcode->dst[i].mask =
            DECODE_D3D10_SB_OPERAND_4_COMPONENT_MASK(operand_token);
      } else if (num_components == D3D10_SB_OPERAND_0_COMPONENT ||
                 num_components == D3D10_SB_OPERAND_1_COMPONENT) {
         opcode->dst[i].mask = D3D10_SB_OPERAND_4_COMPONENT_MASK_X;
      } else {
         goto malformed;
      }

      if (!parse_operand(&curr, instruction_end, &opcode->dst[i].base))
         goto malformed;

      if (extended) {
         unsigned extended_token;

         if (!read_token(&curr, instruction_end, &extended_token) ||
             ((extended_token & D3D10_SB_OPERAND_DOUBLE_EXTENDED_MASK) >>
              D3D10_SB_OPERAND_DOUBLE_EXTENDED_SHIFT))
            goto malformed;

         switch (DECODE_D3D10_SB_EXTENDED_OPERAND_TYPE(extended_token)) {
         case D3D10_SB_EXTENDED_OPERAND_EMPTY:
         case D3D10_SB_EXTENDED_OPERAND_MODIFIER:
            break;
         default:
            goto malformed;
         }
      }

      if (!parse_operand_index(&curr, instruction_end,
                               &opcode->dst[i].base))
         goto malformed;
   }

   /* Source operands. */
   for (i = 0; i < info->num_src; i++) {
      D3D10_SB_OPERAND_NUM_COMPONENTS num_components;
      unsigned operand_token;
      bool extended;

      if (curr >= instruction_end)
         goto malformed;
      operand_token = *curr;
      extended = DECODE_IS_D3D10_SB_OPERAND_EXTENDED(operand_token);
      num_components = DECODE_D3D10_SB_OPERAND_NUM_COMPONENTS(operand_token);

      if (num_components == D3D10_SB_OPERAND_4_COMPONENT) {
         D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE selection_mode =
            DECODE_D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE(operand_token);

         if (selection_mode == D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_MODE) {
            opcode->src[i].swizzle[0] =
               DECODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_SOURCE(
                  operand_token, 0);
            opcode->src[i].swizzle[1] =
               DECODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_SOURCE(
                  operand_token, 1);
            opcode->src[i].swizzle[2] =
               DECODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_SOURCE(
                  operand_token, 2);
            opcode->src[i].swizzle[3] =
               DECODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_SOURCE(
                  operand_token, 3);
         } else if (selection_mode ==
                    D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_MODE) {
            opcode->src[i].swizzle[0] =
               opcode->src[i].swizzle[1] =
               opcode->src[i].swizzle[2] =
               opcode->src[i].swizzle[3] =
                  DECODE_D3D10_SB_OPERAND_4_COMPONENT_SELECT_1(operand_token);
         } else if (selection_mode ==
                       D3D10_SB_OPERAND_4_COMPONENT_MASK_MODE &&
                    DECODE_D3D10_SB_OPERAND_4_COMPONENT_MASK(operand_token) ==
                       0 &&
                    DECODE_D3D10_SB_OPERAND_TYPE(operand_token) ==
                       D3D10_SB_OPERAND_TYPE_IMMEDIATE32) {
            opcode->src[i].swizzle[0] = D3D10_SB_4_COMPONENT_X;
            opcode->src[i].swizzle[1] = D3D10_SB_4_COMPONENT_Y;
            opcode->src[i].swizzle[2] = D3D10_SB_4_COMPONENT_Z;
            opcode->src[i].swizzle[3] = D3D10_SB_4_COMPONENT_W;
         } else {
            goto malformed;
         }
      } else if (num_components == D3D10_SB_OPERAND_1_COMPONENT) {
         opcode->src[i].swizzle[0] =
            opcode->src[i].swizzle[1] =
            opcode->src[i].swizzle[2] =
            opcode->src[i].swizzle[3] = D3D10_SB_4_COMPONENT_X;
      } else if (num_components == D3D10_SB_OPERAND_0_COMPONENT &&
                 (DECODE_D3D10_SB_OPERAND_TYPE(operand_token) ==
                     D3D10_SB_OPERAND_TYPE_SAMPLER ||
                  DECODE_D3D10_SB_OPERAND_TYPE(operand_token) ==
                     D3D10_SB_OPERAND_TYPE_LABEL)) {
         opcode->src[i].swizzle[0] = D3D10_SB_4_COMPONENT_X;
         opcode->src[i].swizzle[1] = D3D10_SB_4_COMPONENT_Y;
         opcode->src[i].swizzle[2] = D3D10_SB_4_COMPONENT_Z;
         opcode->src[i].swizzle[3] = D3D10_SB_4_COMPONENT_W;
      } else {
         goto malformed;
      }

      if (!parse_operand(&curr, instruction_end, &opcode->src[i].base))
         goto malformed;

      opcode->src[i].modifier = D3D10_SB_OPERAND_MODIFIER_NONE;
      if (extended) {
         unsigned extended_token;

         if (!read_token(&curr, instruction_end, &extended_token) ||
             ((extended_token & D3D10_SB_OPERAND_DOUBLE_EXTENDED_MASK) >>
              D3D10_SB_OPERAND_DOUBLE_EXTENDED_SHIFT))
            goto malformed;

         switch (DECODE_D3D10_SB_EXTENDED_OPERAND_TYPE(extended_token)) {
         case D3D10_SB_EXTENDED_OPERAND_EMPTY:
            break;
         case D3D10_SB_EXTENDED_OPERAND_MODIFIER:
            opcode->src[i].modifier =
               DECODE_D3D10_SB_OPERAND_MODIFIER(extended_token);
            break;
         default:
            goto malformed;
         }
      }

      if (!parse_operand_index(&curr, instruction_end,
                               &opcode->src[i].base))
         goto malformed;

      if (opcode->src[i].base.type == D3D10_SB_OPERAND_TYPE_IMMEDIATE32) {
         unsigned immediate;

         if (opcode->type == D3D10_1_SB_OPCODE_SAMPLE_POS && i == 1) {
            if (!read_token(&curr, instruction_end, &immediate))
               goto malformed;
            opcode->src[i].imm[0].u32 =
               opcode->src[i].imm[1].u32 =
               opcode->src[i].imm[2].u32 =
               opcode->src[i].imm[3].u32 = immediate;
            continue;
         }

         switch (num_components) {
         case D3D10_SB_OPERAND_1_COMPONENT:
            if (!read_token(&curr, instruction_end, &immediate))
               goto malformed;
            opcode->src[i].imm[0].u32 =
               opcode->src[i].imm[1].u32 =
               opcode->src[i].imm[2].u32 =
               opcode->src[i].imm[3].u32 = immediate;
            break;
         case D3D10_SB_OPERAND_4_COMPONENT:
            if (!read_token(&curr, instruction_end,
                            &opcode->src[i].imm[0].u32) ||
                !read_token(&curr, instruction_end,
                            &opcode->src[i].imm[1].u32) ||
                !read_token(&curr, instruction_end,
                            &opcode->src[i].imm[2].u32) ||
                !read_token(&curr, instruction_end,
                            &opcode->src[i].imm[3].u32))
               goto malformed;
            break;
         default:
            goto malformed;
         }
      }
   }

   /* Opcode-specific trailing payload. */
   switch (opcode->type) {
   case D3D10_SB_OPCODE_DCL_RESOURCE:
   case D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_TYPED: {
      unsigned return_types;
      if (!read_token(&curr, instruction_end, &return_types))
         goto malformed;
      opcode->dcl_resource_ret_type[0] =
         DECODE_D3D10_SB_RESOURCE_RETURN_TYPE(return_types, 0);
      opcode->dcl_resource_ret_type[1] =
         DECODE_D3D10_SB_RESOURCE_RETURN_TYPE(return_types, 1);
      opcode->dcl_resource_ret_type[2] =
         DECODE_D3D10_SB_RESOURCE_RETURN_TYPE(return_types, 2);
      opcode->dcl_resource_ret_type[3] =
         DECODE_D3D10_SB_RESOURCE_RETURN_TYPE(return_types, 3);
      break;
   }
   case D3D10_SB_OPCODE_DCL_MAX_OUTPUT_VERTEX_COUNT:
      if (!read_token(&curr, instruction_end,
                      &opcode->specific.dcl_max_output_vertex_count))
         goto malformed;
      break;
   case D3D11_SB_OPCODE_DCL_GS_INSTANCE_COUNT:
      if (!read_token(&curr, instruction_end,
                      &opcode->specific.dcl_gs_instance_count))
         goto malformed;
      break;
   case D3D10_SB_OPCODE_DCL_INPUT_SGV:
   case D3D10_SB_OPCODE_DCL_INPUT_SIV:
   case D3D10_SB_OPCODE_DCL_INPUT_PS_SGV:
   case D3D10_SB_OPCODE_DCL_INPUT_PS_SIV:
   case D3D10_SB_OPCODE_DCL_OUTPUT_SIV:
   case D3D10_SB_OPCODE_DCL_OUTPUT_SGV: {
      unsigned name;
      if (!read_token(&curr, instruction_end, &name))
         goto malformed;
      opcode->dcl_siv_name = DECODE_D3D10_SB_NAME(name);
      break;
   }
   case D3D10_SB_OPCODE_DCL_TEMPS:
      if (!read_token(&curr, instruction_end,
                      &opcode->specific.dcl_num_temps))
         goto malformed;
      break;
   case DX10_SM5_OPCODE_DCL_UAV_STRUCTURED:
   case DX10_SM5_OPCODE_DCL_RESOURCE_STRUCTURED:
      if (!read_token(&curr, instruction_end,
                      &opcode->specific.dcl_structured_stride))
         goto malformed;
      break;
   case DX10_SM5_OPCODE_DCL_TGSM_RAW:
      if (!read_token(&curr, instruction_end,
                      &opcode->specific.dcl_tgsm.byte_count))
         goto malformed;
      break;
   case DX10_SM5_OPCODE_DCL_TGSM_STRUCTURED:
      if (!read_token(&curr, instruction_end,
                      &opcode->specific.dcl_tgsm.structured_stride) ||
          !read_token(&curr, instruction_end,
                      &opcode->specific.dcl_tgsm.structured_count) ||
          (opcode->specific.dcl_tgsm.structured_stride &&
           opcode->specific.dcl_tgsm.structured_count >
              ~0u / opcode->specific.dcl_tgsm.structured_stride))
         goto malformed;
      opcode->specific.dcl_tgsm.byte_count =
         opcode->specific.dcl_tgsm.structured_stride *
         opcode->specific.dcl_tgsm.structured_count;
      break;
   case DX10_SM5_OPCODE_DCL_THREAD_GROUP:
      if (!read_token(&curr, instruction_end,
                      &opcode->specific.dcl_thread_group.x) ||
          !read_token(&curr, instruction_end,
                      &opcode->specific.dcl_thread_group.y) ||
          !read_token(&curr, instruction_end,
                      &opcode->specific.dcl_thread_group.z))
         goto malformed;
      break;
   case DX11_SM5_OPCODE_DCL_HS_MAX_TESSFACTOR:
      if (!read_token(&curr, instruction_end,
                      &opcode->specific.dcl_hs_max_tessfactor_bits))
         goto malformed;
      break;
   case DX11_SM5_OPCODE_DCL_HS_FORK_PHASE_INSTANCE_COUNT:
   case DX11_SM5_OPCODE_DCL_HS_JOIN_PHASE_INSTANCE_COUNT:
      if (!read_token(&curr, instruction_end,
                      &opcode->specific.dcl_hs_phase_instance_count))
         goto malformed;
      break;
   case D3D10_SB_OPCODE_DCL_INDEXABLE_TEMP:
      if (!read_token(&curr, instruction_end,
                      &opcode->specific.dcl_indexable_temp.index) ||
          !read_token(&curr, instruction_end,
                      &opcode->specific.dcl_indexable_temp.count) ||
          !read_token(&curr, instruction_end,
                      &opcode->specific.dcl_indexable_temp.components))
         goto malformed;
      break;
   case D3D10_SB_OPCODE_DCL_INDEX_RANGE:
      if (!read_token(&curr, instruction_end,
                      &opcode->specific.index_range_count))
         goto malformed;
      break;
   case D3D11_SB_OPCODE_DCL_STREAM:
   case D3D11_SB_OPCODE_EMIT_STREAM:
   case D3D11_SB_OPCODE_CUT_STREAM:
   case D3D11_SB_OPCODE_EMITTHENCUT_STREAM:
      if (!parse_stream_operand(&curr, instruction_end, opcode))
         goto malformed;
      break;
   default:
      break;
   }

   /* Some shader compilers append one compatibility DWORD to SAMPLE_POS.
    * Keep that accepted form explicit while rejecting all other slack.
    */
   if (opcode->type == D3D10_1_SB_OPCODE_SAMPLE_POS &&
       instruction_end - curr == 1) {
      if (!read_token(&curr, instruction_end,
                      &opcode->specific.sample_pos_compatibility))
         goto malformed;
   }

   if (curr != instruction_end)
      goto malformed;

   parser->curr = instruction_end;
   return true;

malformed:
   yttrium_gdi_trace_warnf(
      "yttrium: shader parse malformed instruction opcode=%s type=%u "
      "offset=%u consumed=%u length=%u\n",
      info->name, opcode->type,
      (unsigned)(instruction_start - parser->code),
      (unsigned)(curr - instruction_start), length);
   parser->failed = true;
   return false;
}

void
Shader_opcode_free(struct Shader_opcode *opcode)
{
   if (opcode->type == D3D10_SB_OPCODE_CUSTOMDATA) {
      if (opcode->customdata._class == D3D10_SB_CUSTOMDATA_DCL_IMMEDIATE_CONSTANT_BUFFER) {
         FREE(opcode->customdata.u.constbuf.data);
      }
   }
}

bool
Shader_parse_tessellation_properties(
   const unsigned *code,
   struct Shader_tessellation_properties *properties)
{
   struct Shader_parser parser;
   struct Shader_opcode opcode;

   if (!code || !properties)
      return false;

   memset(properties, 0, sizeof(*properties));
   Shader_parse_init(&parser, code);
   if (parser.header.type != DX11_SM5_HULL_SHADER)
      return false;

   while (Shader_parse_opcode(&parser, &opcode)) {
      switch (opcode.type) {
      case DX11_SM5_OPCODE_DCL_TESS_DOMAIN:
         properties->domain = opcode.specific.dcl_tess_domain;
         break;
      case DX11_SM5_OPCODE_DCL_TESS_PARTITIONING:
         properties->partitioning = opcode.specific.dcl_tess_partitioning;
         break;
      case DX11_SM5_OPCODE_DCL_TESS_OUTPUT_PRIMITIVE:
         properties->output_primitive =
            opcode.specific.dcl_tess_output_primitive;
         break;
      default:
         break;
      }
      Shader_opcode_free(&opcode);
   }

   if (parser.failed)
      return false;

   return properties->domain && properties->partitioning &&
          properties->output_primitive;
}
