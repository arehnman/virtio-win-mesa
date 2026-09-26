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
 * ShaderTGSI.c --
 *    Functions for translating shaders.
 */

#include "Debug.h"
#include "ShaderParse.h"

#include "pipe/p_shader_tokens.h"
#include "pipe/p_state.h"
#include "tgsi/tgsi_info.h"
#include "tgsi/tgsi_util.h"
#include "tgsi/tgsi_ureg.h"
#include "tgsi/tgsi_dump.h"
#include "util/macros.h"
#include "util/u_memory.h"

#include "ShaderDump.h"

/* For YTTRIUM_WARN: a shader the translator gives up on is an error the log
 * has to name, not a debug_printf into OutputDebugString that nobody reads. */
#include "gallium/winsys/yttrium/gdi/yttrium_trace.h"

extern bool use_old_tex_ops;

enum dx10_opcode_format {
   OF_FLOAT,
   OF_INT,
   OF_UINT
};

struct dx10_opcode_xlate {
   D3D10_SB_OPCODE_TYPE type;
   enum dx10_opcode_format format;
   uint tgsi_opcode;
};

/* Opcodes that we have not even attempted to implement:
 */
#define TGSI_LOG_UNSUPPORTED TGSI_OPCODE_LAST

/* Opcodes which do not translate directly to a TGSI opcode, but which
 * have at least a partial implemention coded below:
 */
#define TGSI_EXPAND          (TGSI_OPCODE_LAST+1)

static struct dx10_opcode_xlate opcode_xlate[D3D10_SB_NUM_OPCODES] = {
   {D3D10_SB_OPCODE_ADD,                              OF_FLOAT, TGSI_OPCODE_ADD},
   {D3D10_SB_OPCODE_AND,                              OF_UINT,  TGSI_OPCODE_AND},
   {D3D10_SB_OPCODE_BREAK,                            OF_FLOAT, TGSI_OPCODE_BRK},
   {D3D10_SB_OPCODE_BREAKC,                           OF_UINT,  TGSI_EXPAND},
   {D3D10_SB_OPCODE_CALL,                             OF_UINT,  TGSI_EXPAND},
   {D3D10_SB_OPCODE_CALLC,                            OF_UINT,  TGSI_EXPAND},
   {D3D10_SB_OPCODE_CASE,                             OF_UINT,  TGSI_OPCODE_CASE},
   {D3D10_SB_OPCODE_CONTINUE,                         OF_FLOAT, TGSI_OPCODE_CONT},
   {D3D10_SB_OPCODE_CONTINUEC,                        OF_UINT,  TGSI_EXPAND},
   {D3D10_SB_OPCODE_CUT,                              OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DEFAULT,                          OF_FLOAT, TGSI_OPCODE_DEFAULT},
   {D3D10_SB_OPCODE_DERIV_RTX,                        OF_FLOAT, TGSI_OPCODE_DDX},
   {D3D10_SB_OPCODE_DERIV_RTY,                        OF_FLOAT, TGSI_OPCODE_DDY},
   {D3D10_SB_OPCODE_DISCARD,                          OF_UINT,  TGSI_EXPAND},
   {D3D10_SB_OPCODE_DIV,                              OF_FLOAT, TGSI_OPCODE_DIV},
   {D3D10_SB_OPCODE_DP2,                              OF_FLOAT, TGSI_OPCODE_DP2},
   {D3D10_SB_OPCODE_DP3,                              OF_FLOAT, TGSI_OPCODE_DP3},
   {D3D10_SB_OPCODE_DP4,                              OF_FLOAT, TGSI_OPCODE_DP4},
   {D3D10_SB_OPCODE_ELSE,                             OF_FLOAT, TGSI_OPCODE_ELSE},
   {D3D10_SB_OPCODE_EMIT,                             OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_EMITTHENCUT,                      OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_ENDIF,                            OF_FLOAT, TGSI_OPCODE_ENDIF},
   {D3D10_SB_OPCODE_ENDLOOP,                          OF_FLOAT, TGSI_OPCODE_ENDLOOP},
   {D3D10_SB_OPCODE_ENDSWITCH,                        OF_FLOAT, TGSI_OPCODE_ENDSWITCH},
   {D3D10_SB_OPCODE_EQ,                               OF_FLOAT, TGSI_OPCODE_FSEQ},
   {D3D10_SB_OPCODE_EXP,                              OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_FRC,                              OF_FLOAT, TGSI_OPCODE_FRC},
   {D3D10_SB_OPCODE_FTOI,                             OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_FTOU,                             OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_GE,                               OF_FLOAT, TGSI_OPCODE_FSGE},
   {D3D10_SB_OPCODE_IADD,                             OF_INT,   TGSI_OPCODE_UADD},
   {D3D10_SB_OPCODE_IF,                               OF_UINT,  TGSI_EXPAND},
   {D3D10_SB_OPCODE_IEQ,                              OF_INT,   TGSI_OPCODE_USEQ},
   {D3D10_SB_OPCODE_IGE,                              OF_INT,   TGSI_OPCODE_ISGE},
   {D3D10_SB_OPCODE_ILT,                              OF_INT,   TGSI_OPCODE_ISLT},
   {D3D10_SB_OPCODE_IMAD,                             OF_INT,   TGSI_OPCODE_UMAD},
   {D3D10_SB_OPCODE_IMAX,                             OF_INT,   TGSI_OPCODE_IMAX},
   {D3D10_SB_OPCODE_IMIN,                             OF_INT,   TGSI_OPCODE_IMIN},
   {D3D10_SB_OPCODE_IMUL,                             OF_INT,   TGSI_EXPAND},
   {D3D10_SB_OPCODE_INE,                              OF_INT,   TGSI_OPCODE_USNE},
   {D3D10_SB_OPCODE_INEG,                             OF_INT,   TGSI_OPCODE_INEG},
   {D3D10_SB_OPCODE_ISHL,                             OF_INT,   TGSI_OPCODE_SHL},
   {D3D10_SB_OPCODE_ISHR,                             OF_INT,   TGSI_OPCODE_ISHR},
   {D3D10_SB_OPCODE_ITOF,                             OF_INT,   TGSI_OPCODE_I2F},
   {D3D10_SB_OPCODE_LABEL,                            OF_INT,   TGSI_EXPAND},
   {D3D10_SB_OPCODE_LD,                               OF_UINT,  TGSI_EXPAND},
   {D3D10_SB_OPCODE_LD_MS,                            OF_UINT,  TGSI_EXPAND},
   {D3D10_SB_OPCODE_LOG,                              OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_LOOP,                             OF_FLOAT, TGSI_OPCODE_BGNLOOP},
   {D3D10_SB_OPCODE_LT,                               OF_FLOAT, TGSI_OPCODE_FSLT},
   {D3D10_SB_OPCODE_MAD,                              OF_FLOAT, TGSI_OPCODE_MAD},
   {D3D10_SB_OPCODE_MIN,                              OF_FLOAT, TGSI_OPCODE_MIN},
   {D3D10_SB_OPCODE_MAX,                              OF_FLOAT, TGSI_OPCODE_MAX},
   {D3D10_SB_OPCODE_CUSTOMDATA,                       OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_MOV,                              OF_UINT,  TGSI_OPCODE_MOV},
   {D3D10_SB_OPCODE_MOVC,                             OF_UINT,  TGSI_OPCODE_UCMP},
   {D3D10_SB_OPCODE_MUL,                              OF_FLOAT, TGSI_OPCODE_MUL},
   {D3D10_SB_OPCODE_NE,                               OF_FLOAT, TGSI_OPCODE_FSNE},
   {D3D10_SB_OPCODE_NOP,                              OF_FLOAT, TGSI_OPCODE_NOP},
   {D3D10_SB_OPCODE_NOT,                              OF_UINT,  TGSI_OPCODE_NOT},
   {D3D10_SB_OPCODE_OR,                               OF_UINT,  TGSI_OPCODE_OR},
   {D3D10_SB_OPCODE_RESINFO,                          OF_UINT,  TGSI_EXPAND},
   {D3D10_SB_OPCODE_RET,                              OF_FLOAT, TGSI_OPCODE_RET},
   {D3D10_SB_OPCODE_RETC,                             OF_UINT,  TGSI_EXPAND},
   {D3D10_SB_OPCODE_ROUND_NE,                         OF_FLOAT, TGSI_OPCODE_ROUND},
   {D3D10_SB_OPCODE_ROUND_NI,                         OF_FLOAT, TGSI_OPCODE_FLR},
   {D3D10_SB_OPCODE_ROUND_PI,                         OF_FLOAT, TGSI_OPCODE_CEIL},
   {D3D10_SB_OPCODE_ROUND_Z,                          OF_FLOAT, TGSI_OPCODE_TRUNC},
   {D3D10_SB_OPCODE_RSQ,                              OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_SAMPLE,                           OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_SAMPLE_C,                         OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_SAMPLE_C_LZ,                      OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_SAMPLE_L,                         OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_SAMPLE_D,                         OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_SAMPLE_B,                         OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_SQRT,                             OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_SWITCH,                           OF_UINT,  TGSI_OPCODE_SWITCH},
   {D3D10_SB_OPCODE_SINCOS,                           OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_UDIV,                             OF_UINT,  TGSI_EXPAND},
   {D3D10_SB_OPCODE_ULT,                              OF_UINT,  TGSI_OPCODE_USLT},
   {D3D10_SB_OPCODE_UGE,                              OF_UINT,  TGSI_OPCODE_USGE},
   {D3D10_SB_OPCODE_UMUL,                             OF_UINT,  TGSI_EXPAND},
   {D3D10_SB_OPCODE_UMAD,                             OF_UINT,  TGSI_OPCODE_UMAD},
   {D3D10_SB_OPCODE_UMAX,                             OF_UINT,  TGSI_OPCODE_UMAX},
   {D3D10_SB_OPCODE_UMIN,                             OF_UINT,  TGSI_OPCODE_UMIN},
   {D3D10_SB_OPCODE_USHR,                             OF_UINT,  TGSI_OPCODE_USHR},
   {D3D10_SB_OPCODE_UTOF,                             OF_UINT,  TGSI_OPCODE_U2F},
   {D3D10_SB_OPCODE_XOR,                              OF_UINT,  TGSI_OPCODE_XOR},
   {D3D10_SB_OPCODE_DCL_RESOURCE,                     OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_CONSTANT_BUFFER,              OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_SAMPLER,                      OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_INDEX_RANGE,                  OF_FLOAT, TGSI_LOG_UNSUPPORTED},
   {D3D10_SB_OPCODE_DCL_GS_OUTPUT_PRIMITIVE_TOPOLOGY, OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_GS_INPUT_PRIMITIVE,           OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_MAX_OUTPUT_VERTEX_COUNT,      OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_INPUT,                        OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_INPUT_SGV,                    OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_INPUT_SIV,                    OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_INPUT_PS,                     OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_INPUT_PS_SGV,                 OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_INPUT_PS_SIV,                 OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_OUTPUT,                       OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_OUTPUT_SGV,                   OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_OUTPUT_SIV,                   OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_TEMPS,                        OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_INDEXABLE_TEMP,               OF_FLOAT, TGSI_EXPAND},
   {D3D10_SB_OPCODE_DCL_GLOBAL_FLAGS,                 OF_FLOAT, TGSI_LOG_UNSUPPORTED},
   {D3D10_SB_OPCODE_RESERVED0,                        OF_FLOAT, TGSI_LOG_UNSUPPORTED},
   {D3D10_1_SB_OPCODE_LOD,                            OF_FLOAT, TGSI_EXPAND},
   {D3D10_1_SB_OPCODE_GATHER4,                        OF_FLOAT, TGSI_EXPAND},
   {D3D10_1_SB_OPCODE_SAMPLE_POS,                     OF_UINT,  TGSI_EXPAND},
   {D3D10_1_SB_OPCODE_SAMPLE_INFO,                    OF_UINT,  TGSI_EXPAND},
   [D3D11_SB_OPCODE_EMIT_STREAM] = {
      D3D11_SB_OPCODE_EMIT_STREAM,                    OF_FLOAT, TGSI_EXPAND},
   [DX11_SM5_OPCODE_HS_DECLS] = {
      DX11_SM5_OPCODE_HS_DECLS,                       OF_FLOAT, TGSI_EXPAND},
   [DX11_SM5_OPCODE_HS_CONTROL_POINT_PHASE] = {
      DX11_SM5_OPCODE_HS_CONTROL_POINT_PHASE,         OF_FLOAT, TGSI_EXPAND},
   [DX11_SM5_OPCODE_HS_FORK_PHASE] = {
      DX11_SM5_OPCODE_HS_FORK_PHASE,                  OF_FLOAT, TGSI_EXPAND},
   [DX11_SM5_OPCODE_HS_JOIN_PHASE] = {
      DX11_SM5_OPCODE_HS_JOIN_PHASE,                  OF_FLOAT, TGSI_EXPAND},
   [DX11_SM5_OPCODE_DCL_INPUT_CONTROL_POINT_COUNT] = {
      DX11_SM5_OPCODE_DCL_INPUT_CONTROL_POINT_COUNT,  OF_FLOAT, TGSI_EXPAND},
   [DX11_SM5_OPCODE_DCL_OUTPUT_CONTROL_POINT_COUNT] = {
      DX11_SM5_OPCODE_DCL_OUTPUT_CONTROL_POINT_COUNT, OF_FLOAT, TGSI_EXPAND},
   [DX11_SM5_OPCODE_DCL_TESS_DOMAIN] = {
      DX11_SM5_OPCODE_DCL_TESS_DOMAIN,                OF_FLOAT, TGSI_EXPAND},
   [DX11_SM5_OPCODE_DCL_TESS_PARTITIONING] = {
      DX11_SM5_OPCODE_DCL_TESS_PARTITIONING,          OF_FLOAT, TGSI_EXPAND},
   [DX11_SM5_OPCODE_DCL_TESS_OUTPUT_PRIMITIVE] = {
      DX11_SM5_OPCODE_DCL_TESS_OUTPUT_PRIMITIVE,      OF_FLOAT, TGSI_EXPAND},
   [DX11_SM5_OPCODE_DCL_HS_MAX_TESSFACTOR] = {
      DX11_SM5_OPCODE_DCL_HS_MAX_TESSFACTOR,          OF_FLOAT, TGSI_EXPAND},
   [DX11_SM5_OPCODE_DCL_HS_FORK_PHASE_INSTANCE_COUNT] = {
      DX11_SM5_OPCODE_DCL_HS_FORK_PHASE_INSTANCE_COUNT,
                                                        OF_FLOAT, TGSI_EXPAND},
   [DX11_SM5_OPCODE_DCL_HS_JOIN_PHASE_INSTANCE_COUNT] = {
      DX11_SM5_OPCODE_DCL_HS_JOIN_PHASE_INSTANCE_COUNT,
                                                        OF_FLOAT, TGSI_EXPAND},
   [D3D11_SB_OPCODE_CUT_STREAM] = {
      D3D11_SB_OPCODE_CUT_STREAM,                     OF_FLOAT, TGSI_EXPAND},
   [D3D11_SB_OPCODE_EMITTHENCUT_STREAM] = {
      D3D11_SB_OPCODE_EMITTHENCUT_STREAM,             OF_FLOAT, TGSI_EXPAND},
   [D3D11_SB_OPCODE_DCL_STREAM] = {
      D3D11_SB_OPCODE_DCL_STREAM,                     OF_FLOAT, TGSI_EXPAND},
   [D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_TYPED] = {
      D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_TYPED, OF_FLOAT, TGSI_EXPAND},
   [D3D11_SB_OPCODE_LD_UAV_TYPED] = {
      D3D11_SB_OPCODE_LD_UAV_TYPED,                  OF_UINT,  TGSI_EXPAND},
   [D3D11_SB_OPCODE_STORE_UAV_TYPED] = {
      D3D11_SB_OPCODE_STORE_UAV_TYPED,               OF_UINT,  TGSI_EXPAND},
   [D3D11_SB_OPCODE_GATHER4_C] = {
      D3D11_SB_OPCODE_GATHER4_C,                     OF_FLOAT, TGSI_EXPAND},
   [D3D11_SB_OPCODE_GATHER4_PO] = {
      D3D11_SB_OPCODE_GATHER4_PO,                    OF_FLOAT, TGSI_EXPAND},
   [D3D11_SB_OPCODE_GATHER4_PO_C] = {
      D3D11_SB_OPCODE_GATHER4_PO_C,                  OF_FLOAT, TGSI_EXPAND},
   [D3D11_SB_OPCODE_EVAL_SAMPLE_INDEX] = {
      D3D11_SB_OPCODE_EVAL_SAMPLE_INDEX,            OF_FLOAT, TGSI_EXPAND},
   [DX10_SM5_OPCODE_DCL_UAV_RAW] = {
      DX10_SM5_OPCODE_DCL_UAV_RAW,                   OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_DCL_UAV_STRUCTURED] = {
      DX10_SM5_OPCODE_DCL_UAV_STRUCTURED,            OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_DCL_TGSM_RAW] = {
      DX10_SM5_OPCODE_DCL_TGSM_RAW,                  OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_DCL_TGSM_STRUCTURED] = {
      DX10_SM5_OPCODE_DCL_TGSM_STRUCTURED,           OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_DCL_RESOURCE_RAW] = {
      DX10_SM5_OPCODE_DCL_RESOURCE_RAW,              OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_DCL_RESOURCE_STRUCTURED] = {
      DX10_SM5_OPCODE_DCL_RESOURCE_STRUCTURED,       OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_LD_RAW] = {
      DX10_SM5_OPCODE_LD_RAW,                        OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_STORE_RAW] = {
      DX10_SM5_OPCODE_STORE_RAW,                     OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_LD_STRUCTURED] = {
      DX10_SM5_OPCODE_LD_STRUCTURED,                 OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_STORE_STRUCTURED] = {
      DX10_SM5_OPCODE_STORE_STRUCTURED,              OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_ATOMIC_AND] = {
      DX10_SM5_OPCODE_ATOMIC_AND,                    OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_ATOMIC_OR] = {
      DX10_SM5_OPCODE_ATOMIC_OR,                     OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_ATOMIC_XOR] = {
      DX10_SM5_OPCODE_ATOMIC_XOR,                    OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_ATOMIC_CMP_STORE] = {
      DX10_SM5_OPCODE_ATOMIC_CMP_STORE,              OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_ATOMIC_IADD] = {
      DX10_SM5_OPCODE_ATOMIC_IADD,                   OF_INT,   TGSI_EXPAND},
   [DX10_SM5_OPCODE_ATOMIC_IMAX] = {
      DX10_SM5_OPCODE_ATOMIC_IMAX,                   OF_INT,   TGSI_EXPAND},
   [DX10_SM5_OPCODE_ATOMIC_IMIN] = {
      DX10_SM5_OPCODE_ATOMIC_IMIN,                   OF_INT,   TGSI_EXPAND},
   [DX10_SM5_OPCODE_ATOMIC_UMAX] = {
      DX10_SM5_OPCODE_ATOMIC_UMAX,                   OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_ATOMIC_UMIN] = {
      DX10_SM5_OPCODE_ATOMIC_UMIN,                   OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_IMM_ATOMIC_ALLOC] = {
      DX10_SM5_OPCODE_IMM_ATOMIC_ALLOC,              OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_IMM_ATOMIC_CONSUME] = {
      DX10_SM5_OPCODE_IMM_ATOMIC_CONSUME,            OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_IMM_ATOMIC_IADD] = {
      DX10_SM5_OPCODE_IMM_ATOMIC_IADD,               OF_INT,   TGSI_EXPAND},
   [DX10_SM5_OPCODE_IMM_ATOMIC_AND] = {
      DX10_SM5_OPCODE_IMM_ATOMIC_AND,                OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_IMM_ATOMIC_OR] = {
      DX10_SM5_OPCODE_IMM_ATOMIC_OR,                 OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_IMM_ATOMIC_XOR] = {
      DX10_SM5_OPCODE_IMM_ATOMIC_XOR,                OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_IMM_ATOMIC_EXCH] = {
      DX10_SM5_OPCODE_IMM_ATOMIC_EXCH,               OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_IMM_ATOMIC_CMP_EXCH] = {
      DX10_SM5_OPCODE_IMM_ATOMIC_CMP_EXCH,           OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_IMM_ATOMIC_IMAX] = {
      DX10_SM5_OPCODE_IMM_ATOMIC_IMAX,               OF_INT,   TGSI_EXPAND},
   [DX10_SM5_OPCODE_IMM_ATOMIC_IMIN] = {
      DX10_SM5_OPCODE_IMM_ATOMIC_IMIN,               OF_INT,   TGSI_EXPAND},
   [DX10_SM5_OPCODE_IMM_ATOMIC_UMAX] = {
      DX10_SM5_OPCODE_IMM_ATOMIC_UMAX,               OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_IMM_ATOMIC_UMIN] = {
      DX10_SM5_OPCODE_IMM_ATOMIC_UMIN,               OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_SYNC] = {
      DX10_SM5_OPCODE_SYNC,                          OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_BUFINFO] = {
      DX10_SM5_OPCODE_BUFINFO,                       OF_UINT,  TGSI_EXPAND},
   [DX10_SM5_OPCODE_DERIV_RTX_COARSE] = {
      DX10_SM5_OPCODE_DERIV_RTX_COARSE,              OF_FLOAT, TGSI_OPCODE_DDX},
   [DX10_SM5_OPCODE_DERIV_RTX_FINE] = {
      DX10_SM5_OPCODE_DERIV_RTX_FINE,                OF_FLOAT, TGSI_OPCODE_DDX},
   [DX10_SM5_OPCODE_DERIV_RTY_COARSE] = {
      DX10_SM5_OPCODE_DERIV_RTY_COARSE,              OF_FLOAT, TGSI_OPCODE_DDY},
   [DX10_SM5_OPCODE_DERIV_RTY_FINE] = {
      DX10_SM5_OPCODE_DERIV_RTY_FINE,                OF_FLOAT, TGSI_OPCODE_DDY},
   [DX10_SM5_OPCODE_RCP] = {
      DX10_SM5_OPCODE_RCP,                           OF_FLOAT, TGSI_EXPAND},
   [DX10_SM5_OPCODE_F32TOF16] = {
      DX10_SM5_OPCODE_F32TOF16,                       OF_FLOAT, TGSI_EXPAND},
   [DX10_SM5_OPCODE_F16TOF32] = {
      DX10_SM5_OPCODE_F16TOF32,                       OF_UINT, TGSI_EXPAND},
   [DX10_SM5_OPCODE_COUNTBITS] = {
      DX10_SM5_OPCODE_COUNTBITS,                      OF_UINT, TGSI_OPCODE_POPC},
   [DX10_SM5_OPCODE_FIRSTBIT_HI] = {
      DX10_SM5_OPCODE_FIRSTBIT_HI,                    OF_UINT, TGSI_EXPAND},
   [DX10_SM5_OPCODE_FIRSTBIT_LO] = {
      DX10_SM5_OPCODE_FIRSTBIT_LO,                    OF_UINT, TGSI_OPCODE_LSB},
   [DX10_SM5_OPCODE_FIRSTBIT_SHI] = {
      DX10_SM5_OPCODE_FIRSTBIT_SHI,                   OF_INT, TGSI_EXPAND},
   [DX10_SM5_OPCODE_UBFE] = {
      DX10_SM5_OPCODE_UBFE,                           OF_UINT, TGSI_EXPAND},
   [DX10_SM5_OPCODE_IBFE] = {
      DX10_SM5_OPCODE_IBFE,                           OF_INT, TGSI_EXPAND},
   [DX10_SM5_OPCODE_BFI] = {
      DX10_SM5_OPCODE_BFI,                            OF_UINT, TGSI_EXPAND},
   [DX10_SM5_OPCODE_BFREV] = {
      DX10_SM5_OPCODE_BFREV,                          OF_UINT, TGSI_OPCODE_BREV},
   [DX10_SM5_OPCODE_SWAPC] = {
      DX10_SM5_OPCODE_SWAPC,                          OF_UINT, TGSI_EXPAND},
};

#define SHADER_MAX_TEMPS 4096
#define SHADER_MAX_ADDRS 3
#define SHADER_MAX_INPUTS 32
#define SHADER_MAX_OUTPUTS 32
#define SHADER_MAX_CONSTS 4096
#define SHADER_MAX_RESOURCES PIPE_MAX_SHADER_SAMPLER_VIEWS
#define SHADER_MAX_SAMPLERS PIPE_MAX_SAMPLERS
#define SHADER_MAX_IMAGES PIPE_MAX_SHADER_IMAGES
#define SHADER_MAX_TGSM 32
#define SHADER_MAX_INDEXABLE_TEMPS 4096

struct Shader_call {
   unsigned d3d_label;
   unsigned tgsi_label_token;
};

struct Shader_label {
   unsigned d3d_label;
   unsigned tgsi_insn_no;
};

struct Shader_resource {
   uint target;   /* TGSI_TEXTURE_x */
   uint structured_stride;
   bool raw;
   uint8_t mixed_return_component_mask;
   enum tgsi_return_type return_type[4];
};

struct Shader_sampler_binding_slot {
   unsigned resource;
   unsigned sampler;
   enum tgsi_texture_type target;
   enum tgsi_return_type return_type[4];
};

struct Shader_image {
   struct ureg_src reg;
   uint target;   /* TGSI_TEXTURE_x */
   uint structured_stride;
   bool raw;
   enum pipe_format format;
};

struct Shader_tgsm {
   unsigned byte_offset;
   unsigned byte_count;
   unsigned structured_stride;
};

struct Shader_xlate {
   struct ureg_program *ureg;

   uint vertices_in;
   uint declared_temps;

   struct ureg_dst temps[SHADER_MAX_TEMPS];
   struct ureg_dst output_depth;
   struct ureg_dst output_coverage_mask;
   struct ureg_dst addrs[SHADER_MAX_ADDRS];
   struct Shader_resource resources[SHADER_MAX_RESOURCES];
   struct Shader_image images[SHADER_MAX_IMAGES];
   uint8_t counter_image_slots[SHADER_MAX_IMAGES];
   struct ureg_src counter_images[SHADER_MAX_IMAGES];
   struct Shader_tgsm tgsm[SHADER_MAX_TGSM];
   struct ureg_src shared_memory;
   bool shared_memory_declared;
   unsigned shared_memory_size;
   struct ureg_src sv[SHADER_MAX_RESOURCES];
   struct ureg_src samplers[SHADER_MAX_SAMPLERS];
   struct pipe_shader_sampler_binding_map *sampler_binding_map;
   struct Shader_sampler_binding_slot
      sampler_binding_slots[SHADER_MAX_SAMPLERS];
   unsigned sampler_binding_count;
   struct ureg_src imms;
   struct ureg_src prim_id;
   struct ureg_src gs_instance_id;
   struct ureg_src sample_mask;
   struct ureg_src cs_block_id;
   struct ureg_src cs_thread_id;
   struct ureg_src cs_dispatch_thread_id;
   struct ureg_src cs_group_index;
   struct ureg_src tcs_invocation_id;
   struct ureg_dst tcs_phase_instance_id;
   struct ureg_src tes_coord;
   bool sample_mask_declared;
   bool output_coverage_mask_declared;
   bool cs_block_id_declared;
   bool cs_thread_id_declared;
   bool cs_dispatch_thread_id_declared;
   bool cs_group_index_declared;
   unsigned cs_thread_group_size[3];
   unsigned tcs_vertices_out;
   bool is_tcs;
   bool translation_failed;
   struct Shader_tessellation_properties tessellation_properties;

   uint addr_cur;
   uint temp_offset;
   uint indexable_temp_offsets[SHADER_MAX_INDEXABLE_TEMPS];

   struct {
      bool declared;
      uint writemask;
      uint siv_name;
      bool overloaded;
      struct ureg_src reg;
   } inputs[SHADER_MAX_INPUTS];
   struct {
      bool declared;
      struct ureg_src reg;
   } patch_inputs[SHADER_MAX_INPUTS];
   struct {
      uint first;
      uint count;
   } input_ranges[SHADER_MAX_INPUTS];
   uint num_input_ranges;

   struct {
      struct ureg_dst reg[4];
      struct ureg_dst gs_value;
      struct ureg_dst gs_packed_output;
      unsigned declared_mask;
      unsigned gs_scalar_mask;
   } outputs[SHADER_MAX_OUTPUTS];
   bool has_gs_scalar_outputs;
   unsigned tcs_tess_factor_writemask[SHADER_MAX_OUTPUTS];
   unsigned tcs_tess_factor_range_first[SHADER_MAX_OUTPUTS];
   unsigned tcs_tess_factor_range_end[SHADER_MAX_OUTPUTS];
   struct {
      const struct Shader_dst_operand *operand;
      struct ureg_dst value;
      struct ureg_dst index;
      unsigned first;
      unsigned end;
   } tcs_indexed_output_stores[SHADER_MAX_DST_OPERANDS];
   unsigned num_tcs_indexed_output_stores;
   const struct Shader_tessellation_io_signatures *tessellation_signatures;
   struct ureg_dst tcs_tess_outer;
   struct ureg_dst tcs_tess_inner;
   bool tcs_tess_outer_declared;
   bool tcs_tess_inner_declared;
   bool tcs_patch_phase_active;
   bool tcs_patch_phase_loop_active;
   unsigned tcs_patch_phase_instance_count;
   unsigned tcs_patch_phase_loop_label;
   bool tcs_control_point_phase_seen;
   bool tcs_control_point_barrier_emitted;
   bool tcs_implicit_control_point_passthrough;

   struct {
      uint d3d;
      uint tgsi;
   } clip_distance_mapping[SHADER_MAX_OUTPUTS];
   uint num_clip_distance_mappings;
   uint num_clip_distances_declared;
   uint num_cull_distances_declared;

   struct Shader_call *calls;
   uint num_calls;
   uint max_calls;
   struct Shader_label *labels;
   uint num_labels;
   uint max_labels;
   bool bufinfo_constants_declared;
   struct {
      bool valid;
      unsigned dst_index;
      unsigned declared_stride;
      unsigned record;
      unsigned remaining;
   } bufinfo_stride;
};

#define SHADER_TRANSLATION_UNSUPPORTED(sx, expr) \
   do { \
      if (expr) { \
         LogUnsupported(__FILE__, __LINE__, __func__, #expr); \
         (sx)->translation_failed = true; \
      } \
   } while (0)

static uint
translate_interpolation(D3D10_SB_INTERPOLATION_MODE interpolation)
{
   switch (interpolation) {
   case D3D10_SB_INTERPOLATION_UNDEFINED:
      return TGSI_INTERPOLATE_LINEAR;

   case D3D10_SB_INTERPOLATION_CONSTANT:
      return TGSI_INTERPOLATE_CONSTANT;
   case D3D10_SB_INTERPOLATION_LINEAR:
   case D3D10_SB_INTERPOLATION_LINEAR_CENTROID:
   case D3D10_SB_INTERPOLATION_LINEAR_SAMPLE:
      return TGSI_INTERPOLATE_PERSPECTIVE;
   case D3D10_SB_INTERPOLATION_LINEAR_NOPERSPECTIVE:
   case D3D10_SB_INTERPOLATION_LINEAR_NOPERSPECTIVE_CENTROID:
   case D3D10_SB_INTERPOLATION_LINEAR_NOPERSPECTIVE_SAMPLE:
      return TGSI_INTERPOLATE_LINEAR;
   }

   assert(0);
   return TGSI_INTERPOLATE_LINEAR;
}

static enum tgsi_interpolate_loc
translate_interpolation_location(D3D10_SB_INTERPOLATION_MODE interpolation)
{
   switch (interpolation) {
   case D3D10_SB_INTERPOLATION_LINEAR_CENTROID:
   case D3D10_SB_INTERPOLATION_LINEAR_NOPERSPECTIVE_CENTROID:
      return TGSI_INTERPOLATE_LOC_CENTROID;
   case D3D10_SB_INTERPOLATION_LINEAR_SAMPLE:
   case D3D10_SB_INTERPOLATION_LINEAR_NOPERSPECTIVE_SAMPLE:
      return TGSI_INTERPOLATE_LOC_SAMPLE;
   default:
      return TGSI_INTERPOLATE_LOC_CENTER;
   }
}

static uint
translate_system_name(struct Shader_xlate *sx, D3D10_SB_NAME name)
{
   switch (name) {
   case D3D10_SB_NAME_UNDEFINED:
      SHADER_TRANSLATION_UNSUPPORTED(sx, true);
      return TGSI_SEMANTIC_GENERIC;
   case D3D10_SB_NAME_POSITION:
      return TGSI_SEMANTIC_POSITION;
   case D3D10_SB_NAME_CLIP_DISTANCE:
   case D3D10_SB_NAME_CULL_DISTANCE:
      return TGSI_SEMANTIC_CLIPDIST;
   case D3D10_SB_NAME_PRIMITIVE_ID:
      return TGSI_SEMANTIC_PRIMID;
   case D3D10_SB_NAME_INSTANCE_ID:
      return TGSI_SEMANTIC_INSTANCEID;
   case D3D10_SB_NAME_VERTEX_ID:
      return TGSI_SEMANTIC_VERTEXID_NOBASE;
   case D3D10_SB_NAME_VIEWPORT_ARRAY_INDEX:
      return TGSI_SEMANTIC_VIEWPORT_INDEX;
   case D3D10_SB_NAME_RENDER_TARGET_ARRAY_INDEX:
      return TGSI_SEMANTIC_LAYER;
   case D3D10_SB_NAME_IS_FRONT_FACE:
      return TGSI_SEMANTIC_FACE;
   case D3D10_SB_NAME_SAMPLE_INDEX:
      return TGSI_SEMANTIC_SAMPLEID;
   case DX11_SM5_NAME_FINAL_QUAD_U_EQ_0_EDGE_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_QUAD_V_EQ_0_EDGE_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_QUAD_U_EQ_1_EDGE_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_QUAD_V_EQ_1_EDGE_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_TRI_U_EQ_0_EDGE_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_TRI_V_EQ_0_EDGE_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_TRI_W_EQ_0_EDGE_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_LINE_DETAIL_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_LINE_DENSITY_TESSFACTOR:
      return TGSI_SEMANTIC_TESSOUTER;
   case DX11_SM5_NAME_FINAL_QUAD_U_INSIDE_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_QUAD_V_INSIDE_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_TRI_INSIDE_TESSFACTOR:
      return TGSI_SEMANTIC_TESSINNER;
   default:
      SHADER_TRANSLATION_UNSUPPORTED(sx, true);
      return TGSI_SEMANTIC_GENERIC;
   }
}

static uint
translate_semantic_index(struct Shader_xlate *sx,
                         D3D10_SB_NAME name,
                         const struct Shader_dst_operand *operand)
{
   unsigned idx;
   switch (name) {
   case D3D10_SB_NAME_CLIP_DISTANCE:
   case D3D10_SB_NAME_CULL_DISTANCE:
      idx = 0;
      bool found = false;
      for (unsigned i = 0; i < sx->num_clip_distance_mappings; i++) {
         if (sx->clip_distance_mapping[i].d3d ==
             operand->base.index[0].imm) {
            idx = sx->clip_distance_mapping[i].tgsi;
            found = true;
            break;
         }
      }
      if (!found) {
         debug_printf("%s: unmapped clip/cull distance output d3d=%u clip=%u cull=%u, using semantic index 0\n",
                      __func__, operand->base.index[0].imm,
                      sx->num_clip_distances_declared,
                      sx->num_cull_distances_declared);
      }
      break;
   default:
      idx = 0;
   }
   return idx;
}

static enum tgsi_return_type
trans_dcl_ret_type(struct Shader_xlate *sx,
                   D3D10_SB_RESOURCE_RETURN_TYPE d3drettype) {
   switch (d3drettype) {
   case D3D10_SB_RETURN_TYPE_UNORM:
      return TGSI_RETURN_TYPE_UNORM;
   case D3D10_SB_RETURN_TYPE_SNORM:
      return TGSI_RETURN_TYPE_SNORM;
   case D3D10_SB_RETURN_TYPE_SINT:
      return TGSI_RETURN_TYPE_SINT;
   case D3D10_SB_RETURN_TYPE_UINT:
      return TGSI_RETURN_TYPE_UINT;
   case D3D10_SB_RETURN_TYPE_FLOAT:
      return TGSI_RETURN_TYPE_FLOAT;
   case D3D10_SB_RETURN_TYPE_MIXED:
      /* MIXED means that the declaration does not provide a concrete
       * component type.  TGSI has no typeless sampler-view return type;
       * preserve the long-standing D3D10 UMD and DXVK-compatible float
       * interpretation.  The declaration site emits a persistent warning
       * once per resource instead of silently hiding the approximation.
       */
      return TGSI_RETURN_TYPE_FLOAT;
   default:
      YTTRIUM_WARN("yttrium: shader TGSI translation failed "
                   "owner=d3d10umd-shader component=resource-return-type "
                   "value=%u action=abort-translation\n", d3drettype);
      SHADER_TRANSLATION_UNSUPPORTED(sx, true);
      return TGSI_RETURN_TYPE_FLOAT;
   }
}

static enum pipe_format
trans_image_format(struct Shader_xlate *sx,
                   D3D10_SB_RESOURCE_RETURN_TYPE d3drettype)
{
   switch (d3drettype) {
   /* The D3D declaration carries only the typed UAV return type; the bound
    * image view still provides the exact resource format.
    */
   case D3D10_SB_RETURN_TYPE_UNORM:
      return PIPE_FORMAT_R8G8B8A8_UNORM;
   case D3D10_SB_RETURN_TYPE_SNORM:
      return PIPE_FORMAT_R8G8B8A8_SNORM;
   case D3D10_SB_RETURN_TYPE_SINT:
      return PIPE_FORMAT_R32_SINT;
   case D3D10_SB_RETURN_TYPE_UINT:
      return PIPE_FORMAT_R32_UINT;
   case D3D10_SB_RETURN_TYPE_FLOAT:
      return PIPE_FORMAT_R32_FLOAT;
   default:
      SHADER_TRANSLATION_UNSUPPORTED(sx, true);
      return PIPE_FORMAT_R32_UINT;
   }
}

static void
declare_vertices_in(struct Shader_xlate *sx,
                    unsigned in)
{
   /* Make sure vertices_in is consistent with input primitive
    * and other input declarations.
    */
   if (sx->vertices_in) {
      assert(sx->vertices_in == in);
   } else {
      sx->vertices_in = in;
   }
}

struct swizzle_mapping {
   unsigned x;
   unsigned y;
   unsigned z;
   unsigned w;
};

/* mapping of writmask to swizzles */
static const struct swizzle_mapping writemask_to_swizzle[] = {
   { TGSI_SWIZZLE_X, TGSI_SWIZZLE_X, TGSI_SWIZZLE_X, TGSI_SWIZZLE_X }, //TGSI_WRITEMASK_NONE
   { TGSI_SWIZZLE_X, TGSI_SWIZZLE_X, TGSI_SWIZZLE_X, TGSI_SWIZZLE_X }, //TGSI_WRITEMASK_X
   { TGSI_SWIZZLE_Y, TGSI_SWIZZLE_Y, TGSI_SWIZZLE_Y, TGSI_SWIZZLE_Y }, //TGSI_WRITEMASK_Y
   { TGSI_SWIZZLE_X, TGSI_SWIZZLE_Y, TGSI_SWIZZLE_X, TGSI_SWIZZLE_Y }, //TGSI_WRITEMASK_XY
   { TGSI_SWIZZLE_Z, TGSI_SWIZZLE_Z, TGSI_SWIZZLE_Z, TGSI_SWIZZLE_Z }, //TGSI_WRITEMASK_Z
   { TGSI_SWIZZLE_X, TGSI_SWIZZLE_Z, TGSI_SWIZZLE_X, TGSI_SWIZZLE_Z }, //TGSI_WRITEMASK_XZ
   { TGSI_SWIZZLE_Y, TGSI_SWIZZLE_Z, TGSI_SWIZZLE_Y, TGSI_SWIZZLE_Z }, //TGSI_WRITEMASK_YZ
   { TGSI_SWIZZLE_X, TGSI_SWIZZLE_Y, TGSI_SWIZZLE_Z, TGSI_SWIZZLE_X }, //TGSI_WRITEMASK_XYZ
   { TGSI_SWIZZLE_W, TGSI_SWIZZLE_W, TGSI_SWIZZLE_W, TGSI_SWIZZLE_W }, //TGSI_WRITEMASK_W
   { TGSI_SWIZZLE_X, TGSI_SWIZZLE_W, TGSI_SWIZZLE_X, TGSI_SWIZZLE_W }, //TGSI_WRITEMASK_XW
   { TGSI_SWIZZLE_Y, TGSI_SWIZZLE_W, TGSI_SWIZZLE_Y, TGSI_SWIZZLE_W }, //TGSI_WRITEMASK_YW
   { TGSI_SWIZZLE_X, TGSI_SWIZZLE_Y, TGSI_SWIZZLE_W, TGSI_SWIZZLE_W }, //TGSI_WRITEMASK_XYW
   { TGSI_SWIZZLE_Z, TGSI_SWIZZLE_W, TGSI_SWIZZLE_Z, TGSI_SWIZZLE_W }, //TGSI_WRITEMASK_ZW
   { TGSI_SWIZZLE_X, TGSI_SWIZZLE_Y, TGSI_SWIZZLE_Z, TGSI_SWIZZLE_W }, //TGSI_WRITEMASK_XZW
   { TGSI_SWIZZLE_X, TGSI_SWIZZLE_Y, TGSI_SWIZZLE_Z, TGSI_SWIZZLE_W }, //TGSI_WRITEMASK_YZW
   { TGSI_SWIZZLE_X, TGSI_SWIZZLE_Y, TGSI_SWIZZLE_Z, TGSI_SWIZZLE_W }, //TGSI_WRITEMASK_XYZW
};

static struct ureg_src
swizzle_reg(struct ureg_src src, uint writemask,
            unsigned siv_name)
{
   switch (siv_name) {
   case D3D10_SB_NAME_PRIMITIVE_ID:
   case D3D10_SB_NAME_INSTANCE_ID:
   case D3D10_SB_NAME_VERTEX_ID:
   case D3D10_SB_NAME_VIEWPORT_ARRAY_INDEX:
   case D3D10_SB_NAME_RENDER_TARGET_ARRAY_INDEX:
   case D3D10_SB_NAME_IS_FRONT_FACE:
      return ureg_scalar(src, TGSI_SWIZZLE_X);
   default: {
      const struct swizzle_mapping *swizzle =
         &writemask_to_swizzle[writemask];
      return ureg_swizzle(src, swizzle->x, swizzle->y,
                          swizzle->z, swizzle->w);
   }
   }
}

static struct ureg_src
translate_load_src_for_txf(unsigned target, struct ureg_src src)
{
   switch (target) {
   case TGSI_TEXTURE_1D:
      return ureg_swizzle(src, TGSI_SWIZZLE_X, TGSI_SWIZZLE_Y,
                          TGSI_SWIZZLE_Z, TGSI_SWIZZLE_Y);
   case TGSI_TEXTURE_1D_ARRAY:
   case TGSI_TEXTURE_2D:
   case TGSI_TEXTURE_CUBE:
      return ureg_swizzle(src, TGSI_SWIZZLE_X, TGSI_SWIZZLE_Y,
                          TGSI_SWIZZLE_Z, TGSI_SWIZZLE_Z);
   default:
      return src;
   }
}

static void
dcl_base_output(struct Shader_xlate *sx,
                struct ureg_program *ureg,
                struct ureg_dst reg,
                const struct Shader_dst_operand *operand)
{
   unsigned writemask =
      operand->mask >> D3D10_SB_OPERAND_4_COMPONENT_MASK_SHIFT;
   unsigned idx =
      operand->base.index[operand->base.index_dim == 2 ? 1 : 0].imm;
   unsigned i;

   sx->outputs[idx].declared_mask |= writemask ?
      writemask : TGSI_WRITEMASK_XYZW;
   if (!writemask) {
      sx->outputs[idx].reg[0] = reg;
      sx->outputs[idx].reg[1] = reg;
      sx->outputs[idx].reg[2] = reg;
      sx->outputs[idx].reg[3] = reg;
      return;
   }

   for (i = 0; i < 4; ++i) {
      unsigned mask = 1 << i;
      if ((writemask & mask)) {
         sx->outputs[idx].reg[i] = reg;
      }
   }
}

static void
dcl_gs_scalar_output(struct Shader_xlate *sx,
                     const struct Shader_dst_operand *operand)
{
   const unsigned output = operand->base.index[0].imm;
   const unsigned mask =
      operand->mask >> D3D10_SB_OPERAND_4_COMPONENT_MASK_SHIFT;

   if (util_bitcount(mask) != 1) {
      YTTRIUM_WARN("yttrium: shader TGSI translation failed "
                   "owner=d3d10umd-shader component=gs-scalar-output "
                   "reason=invalid-declaration-mask output=%u mask=%x "
                   "action=abort-translation\n", output, mask);
      sx->translation_failed = true;
      return;
   }

   /* D3D can pack a scalar system value beside generic components in one
    * output register. Keep that complete register in a temporary so even a
    * single instruction writing both semantics retains its original mask.
    * TGSI requires LAYER, VIEWPORT_INDEX and PRIMID in separate outputs' X.
    */
   if (!sx->outputs[output].gs_scalar_mask) {
      sx->outputs[output].gs_value = ureg_DECL_temporary(sx->ureg);
      /* SO declarations address whole D3D registers and retain their D3D
       * component masks. Mirror the packed value in the existing GENERIC
       * semantic for this D3D register (or declare it for a scalar-only
       * register), separately from the canonical scalar system outputs.
       */
      sx->outputs[output].gs_packed_output =
         ureg_DECL_output(sx->ureg, TGSI_SEMANTIC_GENERIC, output);
   }
   sx->outputs[output].gs_scalar_mask |= mask;
   sx->has_gs_scalar_outputs = true;
}

static bool
is_gs_scalar_output_name(D3D10_SB_NAME name)
{
   return name == D3D10_SB_NAME_RENDER_TARGET_ARRAY_INDEX ||
          name == D3D10_SB_NAME_VIEWPORT_ARRAY_INDEX ||
          name == D3D10_SB_NAME_PRIMITIVE_ID;
}

static void
emit_gs_scalar_output_stores(struct Shader_xlate *sx)
{
   if (!sx->has_gs_scalar_outputs)
      return;

   for (unsigned output = 0; output < SHADER_MAX_OUTPUTS; ++output) {
      const unsigned scalar_mask = sx->outputs[output].gs_scalar_mask;
      if (!scalar_mask)
         continue;

      ureg_MOV(sx->ureg,
               ureg_writemask(sx->outputs[output].gs_packed_output,
                              sx->outputs[output].declared_mask),
               ureg_src(sx->outputs[output].gs_value));
      for (unsigned component = 0; component < 4; ++component) {
         const struct ureg_dst dst = sx->outputs[output].reg[component];
         if (ureg_dst_is_undef(dst) ||
             dst.Index == sx->outputs[output].gs_packed_output.Index)
            continue;

         const unsigned mask = scalar_mask & (1u << component) ?
            TGSI_WRITEMASK_X : 1u << component;
         ureg_MOV(sx->ureg, ureg_writemask(dst, mask),
                  ureg_scalar(ureg_src(sx->outputs[output].gs_value),
                               component));
      }
   }
}

static void
dcl_base_input(struct Shader_xlate *sx,
               struct ureg_program *ureg,
               const struct Shader_dst_operand *operand,
               struct ureg_src dcl_reg,
               uint index,
               uint siv_name)
{
   unsigned writemask =
      operand->mask >> D3D10_SB_OPERAND_4_COMPONENT_MASK_SHIFT;

   if (sx->inputs[index].declared && !sx->inputs[index].overloaded) {
      struct ureg_dst temp = ureg_DECL_temporary(sx->ureg);

      ureg_MOV(ureg,
               ureg_writemask(temp, sx->inputs[index].writemask),
               swizzle_reg(sx->inputs[index].reg, sx->inputs[index].writemask,
                           sx->inputs[index].siv_name));
      ureg_MOV(ureg, ureg_writemask(temp, writemask),
               swizzle_reg(dcl_reg, writemask, siv_name));
      sx->inputs[index].reg = ureg_src(temp);
      sx->inputs[index].overloaded = true;
      sx->inputs[index].writemask |= writemask;
   } else if (sx->inputs[index].overloaded) {
      struct ureg_dst temp = ureg_dst(sx->inputs[index].reg);
      ureg_MOV(ureg, ureg_writemask(temp, writemask),
               swizzle_reg(dcl_reg, writemask, siv_name));
      sx->inputs[index].writemask |= writemask;
   } else {
      assert(!sx->inputs[index].declared);

      sx->inputs[index].reg = dcl_reg;
      sx->inputs[index].declared = true;
      sx->inputs[index].writemask = writemask;
      sx->inputs[index].siv_name = siv_name;
   }
}

static void
dcl_vs_input(struct Shader_xlate *sx,
             struct ureg_program *ureg,
             const struct Shader_dst_operand *dst)
{
   struct ureg_src reg;
   assert(dst->base.index_dim == 1);
   assert(dst->base.index[0].imm < SHADER_MAX_INPUTS);

   reg = ureg_DECL_vs_input(ureg, dst->base.index[0].imm);

   dcl_base_input(sx, ureg, dst, reg, dst->base.index[0].imm,
                  D3D10_SB_NAME_UNDEFINED);
}

static void
dcl_input_index_range(struct Shader_xlate *sx,
                      const struct Shader_opcode *opcode)
{
   const struct Shader_dst_operand *dst = &opcode->dst[0];
   uint first, count;

   if (dst->base.type != D3D10_SB_OPERAND_TYPE_INPUT ||
       dst->base.index_dim != 1 ||
       dst->base.index[0].index_rep != D3D10_SB_OPERAND_INDEX_IMMEDIATE32) {
      SHADER_TRANSLATION_UNSUPPORTED(sx, true);
      return;
   }

   first = dst->base.index[0].imm;
   count = opcode->specific.index_range_count;
   if (!count || first >= SHADER_MAX_INPUTS ||
       count > SHADER_MAX_INPUTS - first) {
      SHADER_TRANSLATION_UNSUPPORTED(sx, true);
      return;
   }

   for (uint i = 0; i < sx->num_input_ranges; i++) {
      if (sx->input_ranges[i].first == first) {
         sx->input_ranges[i].count = MAX2(sx->input_ranges[i].count, count);
         return;
      }
   }

   if (sx->num_input_ranges >= SHADER_MAX_INPUTS) {
      SHADER_TRANSLATION_UNSUPPORTED(sx, true);
      return;
   }

   sx->input_ranges[sx->num_input_ranges].first = first;
   sx->input_ranges[sx->num_input_ranges].count = count;
   sx->num_input_ranges++;
}

static void
dcl_tcs_tess_factor_index_range(struct Shader_xlate *sx,
                                 const struct Shader_opcode *opcode)
{
   const struct Shader_dst_operand *dst = &opcode->dst[0];
   if (!sx->is_tcs || !sx->tcs_patch_phase_active ||
       dst->base.type != D3D10_SB_OPERAND_TYPE_OUTPUT ||
       dst->base.index_dim != 1 ||
       dst->base.index[0].index_rep != D3D10_SB_OPERAND_INDEX_IMMEDIATE32 ||
       dst->mask != D3D10_SB_OPERAND_4_COMPONENT_MASK_X) {
      SHADER_TRANSLATION_UNSUPPORTED(sx, true);
      return;
   }

   const unsigned first = dst->base.index[0].imm;
   const unsigned count = opcode->specific.index_range_count;
   if (!count || first >= SHADER_MAX_OUTPUTS ||
       count > SHADER_MAX_OUTPUTS - first) {
      SHADER_TRANSLATION_UNSUPPORTED(sx, true);
      return;
   }

   /* D3D scalar tessellation-factor registers are packed into components
    * of TGSI TESSOUTER/TESSINNER, not consecutive TGSI output registers.
    */
   for (unsigned i = first; i < first + count; ++i) {
      if (!sx->tcs_tess_factor_writemask[i]) {
         SHADER_TRANSLATION_UNSUPPORTED(sx, true);
         return;
      }
   }
   for (unsigned i = first; i < first + count; ++i) {
      sx->tcs_tess_factor_range_first[i] = first;
      sx->tcs_tess_factor_range_end[i] = first + count;
   }
}

static void
dcl_gs_input(struct Shader_xlate *sx,
             struct ureg_program *ureg,
             const struct Shader_dst_operand *dst)
{
   if (dst->base.index_dim == 2) {
      assert(dst->base.index[1].imm < SHADER_MAX_INPUTS);

      declare_vertices_in(sx, dst->base.index[0].imm);

      /* XXX: Implement declaration masks in gallium.
       */
      if (!sx->inputs[dst->base.index[1].imm].reg.File) {
         struct ureg_src reg =
            ureg_DECL_input(ureg,
                            TGSI_SEMANTIC_GENERIC,
                            dst->base.index[1].imm,
                            0, 1);
         dcl_base_input(sx, ureg, dst, reg, dst->base.index[1].imm,
                        D3D10_SB_NAME_UNDEFINED);
      }
   } else if (dst->base.type == D3D10_SB_OPERAND_TYPE_INPUT) {
      unsigned index;

      assert(dst->base.index_dim == 1);
      assert(dst->base.index[0].imm < SHADER_MAX_INPUTS);

      index = dst->base.index[0].imm;
      if (!sx->inputs[index].reg.File) {
         struct ureg_src reg =
            ureg_DECL_input(ureg,
                            TGSI_SEMANTIC_GENERIC,
                            index,
                            0, 1);
         dcl_base_input(sx, ureg, dst, reg, index,
                        D3D10_SB_NAME_UNDEFINED);
      }
   } else if (dst->base.type == D3D10_SB_OPERAND_TYPE_INPUT_PRIMITIVEID) {
      assert(dst->base.index_dim == 0);

      sx->prim_id = ureg_DECL_system_value(ureg, TGSI_SEMANTIC_PRIMID, 0);
   } else {
      assert(dst->base.type == D3D11_SB_OPERAND_TYPE_INPUT_GS_INSTANCE_ID);
      assert(dst->base.index_dim == 0);

      sx->gs_instance_id =
         ureg_DECL_system_value(ureg, TGSI_SEMANTIC_INVOCATIONID, 0);
   }
}

static unsigned
tessellation_clip_distance_semantic_index(
   const struct Shader_tessellation_io_signatures *signatures,
   bool output, unsigned reg)
{
   if (!signatures)
      return 0;

   const D3D10_SB_NAME *system_values = output ?
      signatures->output_system_values : signatures->input_system_values;
   const unsigned char *masks = output ?
      signatures->output_masks : signatures->input_masks;
   const unsigned count = output ? signatures->num_output_system_values :
      signatures->num_input_system_values;
   unsigned components = 0;

   if (!system_values || !masks)
      return 0;

   for (unsigned i = 0; i <= reg && i < count; ++i) {
      if (system_values[i] != D3D10_SB_NAME_CLIP_DISTANCE &&
          system_values[i] != D3D10_SB_NAME_CULL_DISTANCE)
         continue;
      if (i == reg)
         return components / 4;
      components += util_bitcount(masks[i]);
   }

   return 0;
}

static void
tessellation_translation_unsupported(struct Shader_xlate *sx,
                                     const char *reason)
{
   YTTRIUM_WARN("yttrium: shader TGSI tessellation translation unsupported owner=d3d10umd-shader component=tessellation reason=%s action=abort-translation\n",
                reason);
   sx->translation_failed = true;
}

static void
dcl_tess_input(struct Shader_xlate *sx,
               struct ureg_program *ureg,
               const struct Shader_dst_operand *dst)
{
   unsigned input;

   if (dst->base.type == DX11_SM5_OPERAND_TYPE_OUTPUT_CONTROL_POINT_ID ||
       dst->base.type == DX11_SM5_OPERAND_TYPE_INPUT_FORK_INSTANCE_ID ||
       dst->base.type == DX11_SM5_OPERAND_TYPE_INPUT_JOIN_INSTANCE_ID) {
      if (!sx->tcs_invocation_id.File) {
         sx->tcs_invocation_id =
            ureg_DECL_system_value(ureg, TGSI_SEMANTIC_INVOCATIONID, 0);
      }
      return;
   }

   if (dst->base.type == DX11_SM5_OPERAND_TYPE_INPUT_DOMAIN_POINT) {
      sx->tes_coord =
         ureg_DECL_system_value(ureg, TGSI_SEMANTIC_TESSCOORD, 0);
      return;
   }

   if (dst->base.type == DX11_SM5_OPERAND_TYPE_INPUT_PATCH_CONSTANT) {
      if (dst->base.index_dim != 1 ||
          dst->base.index[0].index_rep !=
             D3D10_SB_OPERAND_INDEX_IMMEDIATE32 ||
          dst->base.index[0].imm >= SHADER_MAX_INPUTS) {
         tessellation_translation_unsupported(
            sx, "invalid-patch-input-declaration");
         return;
      }
      input = dst->base.index[0].imm;
      if (!sx->patch_inputs[input].declared) {
         if (sx->is_tcs) {
            if (!sx->outputs[input].reg[0].File) {
               tessellation_translation_unsupported(
                  sx, "patch-input-has-no-prior-output");
               return;
            }
            /*
             * A hull join phase reads patch constants produced by an earlier
             * fork phase.  Keep that access in TGSI's output file so TTN can
             * lower it to a TCS output load instead of declaring a second
             * shader input interface.
             */
            sx->patch_inputs[input].reg =
               ureg_src(sx->outputs[input].reg[0]);
         } else {
            sx->patch_inputs[input].reg =
               ureg_DECL_input(ureg, TGSI_SEMANTIC_PATCH, input, 0, 1);
         }
         sx->patch_inputs[input].declared = true;
      }
      return;
   }

   if ((dst->base.type != D3D10_SB_OPERAND_TYPE_INPUT &&
        dst->base.type != DX11_SM5_OPERAND_TYPE_INPUT_CONTROL_POINT) ||
       dst->base.index_dim != 2 ||
       dst->base.index[0].index_rep !=
          D3D10_SB_OPERAND_INDEX_IMMEDIATE32 ||
       dst->base.index[1].index_rep !=
          D3D10_SB_OPERAND_INDEX_IMMEDIATE32) {
      tessellation_translation_unsupported(
         sx, "invalid-control-point-input-declaration");
      return;
   }

   declare_vertices_in(sx, dst->base.index[0].imm);
   input = dst->base.index[1].imm;
   if (input >= SHADER_MAX_INPUTS) {
      tessellation_translation_unsupported(
         sx, "control-point-input-register-out-of-range");
      return;
   }

   if (!sx->inputs[input].reg.File) {
      D3D10_SB_NAME system_value = D3D10_SB_NAME_UNDEFINED;
      unsigned semantic_index = input;
      if (sx->tessellation_signatures &&
          input < sx->tessellation_signatures->num_input_system_values) {
         system_value =
            sx->tessellation_signatures->input_system_values[input];
      }
      if (system_value == D3D10_SB_NAME_CLIP_DISTANCE ||
          system_value == D3D10_SB_NAME_CULL_DISTANCE) {
         semantic_index = tessellation_clip_distance_semantic_index(
            sx->tessellation_signatures, false, input);
      }
      struct ureg_src reg =
         ureg_DECL_input(ureg,
                         system_value == D3D10_SB_NAME_UNDEFINED ?
                            TGSI_SEMANTIC_GENERIC :
                            translate_system_name(sx, system_value),
                         semantic_index, 0, 1);
      dcl_base_input(sx, ureg, dst, reg, input, system_value);
   }
}

static void
dcl_tcs_output(struct Shader_xlate *sx,
               struct ureg_program *ureg,
               const struct Shader_dst_operand *dst)
{
   if ((dst->base.type != D3D10_SB_OPERAND_TYPE_OUTPUT &&
        dst->base.type != DX11_SM5_OPERAND_TYPE_OUTPUT_CONTROL_POINT) ||
       (dst->base.index_dim != 1 && dst->base.index_dim != 2)) {
      tessellation_translation_unsupported(
         sx, "invalid-control-point-output-declaration");
      return;
   }

   const unsigned output =
      dst->base.index[dst->base.index_dim - 1].imm;
   if (!sx->tcs_vertices_out || output >= SHADER_MAX_OUTPUTS) {
      tessellation_translation_unsupported(
         sx, "control-point-output-count-or-register-invalid");
      return;
   }

   D3D10_SB_NAME system_value = D3D10_SB_NAME_UNDEFINED;
   if (sx->tessellation_signatures &&
       output < sx->tessellation_signatures->num_output_system_values) {
      system_value =
         sx->tessellation_signatures->output_system_values[output];
   }
   enum tgsi_semantic semantic =
      sx->tcs_patch_phase_active ? TGSI_SEMANTIC_PATCH :
      system_value == D3D10_SB_NAME_UNDEFINED ?
         TGSI_SEMANTIC_GENERIC : translate_system_name(sx, system_value);
   unsigned semantic_index = output;
   if (system_value == D3D10_SB_NAME_CLIP_DISTANCE ||
       system_value == D3D10_SB_NAME_CULL_DISTANCE) {
      semantic_index = tessellation_clip_distance_semantic_index(
         sx->tessellation_signatures, true, output);
   }
   struct ureg_dst reg = ureg_DECL_output(ureg, semantic, semantic_index);
   dcl_base_output(sx, ureg, reg, dst);
}

static unsigned
tcs_tess_factor_writemask(D3D10_SB_NAME name)
{
   switch (name) {
   case DX11_SM5_NAME_FINAL_QUAD_U_EQ_0_EDGE_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_TRI_U_EQ_0_EDGE_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_LINE_DETAIL_TESSFACTOR:
      return TGSI_WRITEMASK_X;
   case DX11_SM5_NAME_FINAL_QUAD_V_EQ_0_EDGE_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_TRI_V_EQ_0_EDGE_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_LINE_DENSITY_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_QUAD_U_INSIDE_TESSFACTOR:
      return TGSI_WRITEMASK_Y;
   case DX11_SM5_NAME_FINAL_QUAD_U_EQ_1_EDGE_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_TRI_W_EQ_0_EDGE_TESSFACTOR:
      return TGSI_WRITEMASK_Z;
   case DX11_SM5_NAME_FINAL_QUAD_V_EQ_1_EDGE_TESSFACTOR:
      return TGSI_WRITEMASK_W;
   case DX11_SM5_NAME_FINAL_QUAD_V_INSIDE_TESSFACTOR:
   case DX11_SM5_NAME_FINAL_TRI_INSIDE_TESSFACTOR:
      return TGSI_WRITEMASK_X;
   default:
      return 0;
   }
}

static void
dcl_tes_tess_factor_input(struct Shader_xlate *sx,
                          struct ureg_program *ureg,
                          const struct Shader_dst_operand *dst,
                          D3D10_SB_NAME name)
{
   const unsigned component_mask = tcs_tess_factor_writemask(name);
   const unsigned input_mask =
      dst->mask >> D3D10_SB_OPERAND_4_COMPONENT_MASK_SHIFT;

   if (dst->base.type != DX11_SM5_OPERAND_TYPE_INPUT_PATCH_CONSTANT ||
       dst->base.index_dim != 1 ||
       dst->base.index[0].index_rep != D3D10_SB_OPERAND_INDEX_IMMEDIATE32 ||
       dst->base.index[0].imm >= SHADER_MAX_INPUTS ||
       !component_mask || util_bitcount(input_mask) != 1) {
      tessellation_translation_unsupported(
         sx, "invalid-tessellation-factor-input-declaration");
      return;
   }

   const unsigned input = dst->base.index[0].imm;
   if (sx->patch_inputs[input].declared) {
      tessellation_translation_unsupported(
         sx, "overlapping-tessellation-factor-input-declaration");
      return;
   }

   /* D3D declares each factor in a separate scalar patch register. TGSI packs
    * them into TESSOUTER/TESSINNER vectors, using the same component mapping
    * as hull outputs. Keep these separate from per-control-point inputs:
    * vpc0 and vicp[vertex][0] are different input namespaces.
    */
   struct ureg_src reg =
      ureg_DECL_input(ureg, translate_system_name(sx, name), 0, 0, 1);
   sx->patch_inputs[input].reg =
      ureg_scalar(reg, ffs(component_mask) - 1);
   sx->patch_inputs[input].declared = true;
}

static void
dcl_tcs_tess_factor_output(struct Shader_xlate *sx,
                           struct ureg_program *ureg,
                           const struct Shader_dst_operand *dst,
                           D3D10_SB_NAME name)
{
   const unsigned writemask = tcs_tess_factor_writemask(name);
   const bool inner =
      translate_system_name(sx, name) == TGSI_SEMANTIC_TESSINNER;
   struct ureg_dst reg;

   if (dst->base.index_dim != 1 || !writemask) {
      tessellation_translation_unsupported(
         sx, "invalid-tessellation-factor-output-declaration");
      return;
   }

   const unsigned output = dst->base.index[0].imm;
   if (output >= SHADER_MAX_OUTPUTS) {
      tessellation_translation_unsupported(
         sx, "tessellation-factor-output-register-out-of-range");
      return;
   }

   if (inner) {
      if (!sx->tcs_tess_inner_declared) {
         sx->tcs_tess_inner =
            ureg_DECL_output(ureg, TGSI_SEMANTIC_TESSINNER, 0);
         sx->tcs_tess_inner_declared = true;
      }
      reg = sx->tcs_tess_inner;
   } else {
      if (!sx->tcs_tess_outer_declared) {
         sx->tcs_tess_outer =
            ureg_DECL_output(ureg, TGSI_SEMANTIC_TESSOUTER, 0);
         sx->tcs_tess_outer_declared = true;
      }
      reg = sx->tcs_tess_outer;
   }

   for (unsigned i = 0; i < 4; ++i)
      sx->outputs[output].reg[i] = reg;
   sx->tcs_tess_factor_writemask[output] = writemask;
}

static void
tcs_end_patch_phase(struct Shader_xlate *sx, struct ureg_program *ureg)
{
   if (!sx->tcs_patch_phase_active)
      return;

   if (sx->tcs_patch_phase_loop_active) {
      struct ureg_dst predicate = ureg_DECL_temporary(ureg);
      unsigned label = 0;

      ureg_UADD(ureg,
                ureg_writemask(sx->tcs_phase_instance_id,
                               TGSI_WRITEMASK_X),
                ureg_scalar(ureg_src(sx->tcs_phase_instance_id),
                            TGSI_SWIZZLE_X),
                ureg_imm1u(ureg, sx->tcs_vertices_out));
      ureg_USGE(ureg, ureg_writemask(predicate, TGSI_WRITEMASK_X),
                ureg_scalar(ureg_src(sx->tcs_phase_instance_id),
                            TGSI_SWIZZLE_X),
                ureg_imm1u(ureg, sx->tcs_patch_phase_instance_count));
      ureg_UIF(ureg, ureg_src(predicate), &label);
      ureg_BRK(ureg);
      ureg_ENDIF(ureg);
      ureg_release_temporary(ureg, predicate);

      ureg_fixup_label(ureg, sx->tcs_patch_phase_loop_label,
                       ureg_get_instruction_number(ureg));
      ureg_ENDLOOP(ureg, &sx->tcs_patch_phase_loop_label);
      sx->tcs_patch_phase_loop_active = false;
   }

   ureg_ENDIF(ureg);
   ureg_BARRIER(ureg);
   sx->tcs_patch_phase_active = false;
}

static void
tcs_begin_patch_phase(struct Shader_xlate *sx, struct ureg_program *ureg,
                      unsigned instance_count)
{
   tcs_end_patch_phase(sx, ureg);
   if (!instance_count || instance_count > 32 ||
       !sx->tcs_vertices_out) {
      tessellation_translation_unsupported(
         sx, "invalid-patch-phase-instance-or-output-count");
      return;
   }
   if (!sx->tcs_invocation_id.File) {
      sx->tcs_invocation_id =
         ureg_DECL_system_value(ureg, TGSI_SEMANTIC_INVOCATIONID, 0);
   }
   if (!sx->tcs_phase_instance_id.File)
      sx->tcs_phase_instance_id = ureg_DECL_temporary(ureg);

   /*
    * D3D fork/join phases may have more instances than the hull shader has
    * output control points, while Vulkan launches exactly one TCS invocation
    * per output control point.  Distribute phase instances round-robin across
    * those invocations and loop only when an invocation owns multiple phase
    * instances.
    */
   ureg_MOV(ureg,
            ureg_writemask(sx->tcs_phase_instance_id, TGSI_WRITEMASK_X),
            ureg_scalar(sx->tcs_invocation_id, TGSI_SWIZZLE_X));

   struct ureg_dst predicate = ureg_DECL_temporary(ureg);
   struct ureg_src src[2] = {
      ureg_scalar(sx->tcs_invocation_id, TGSI_SWIZZLE_X),
      ureg_imm1u(ureg, instance_count),
   };
   unsigned label = 0;
   predicate = ureg_writemask(predicate, TGSI_WRITEMASK_X);
   ureg_insn(ureg, TGSI_OPCODE_USLT, &predicate, 1,
             src, ARRAY_SIZE(src), 0);
   ureg_UIF(ureg, ureg_src(predicate), &label);
   ureg_release_temporary(ureg, predicate);
   sx->tcs_patch_phase_active = true;
   sx->tcs_patch_phase_instance_count = instance_count;

   if (instance_count > sx->tcs_vertices_out) {
      ureg_BGNLOOP(ureg, &sx->tcs_patch_phase_loop_label);
      sx->tcs_patch_phase_loop_active = true;
   }
}

static void
tcs_set_clip_cull_properties(const struct Shader_xlate *sx,
                             struct ureg_program *ureg)
{
   const struct Shader_tessellation_io_signatures *signatures =
      sx->tessellation_signatures;
   unsigned output_clip_components = 0;
   unsigned output_cull_components = 0;

   if (!signatures || !signatures->output_system_values ||
       !signatures->output_masks)
      return;

   for (unsigned i = 0; i < signatures->num_output_system_values; ++i) {
      const unsigned writemask = signatures->output_masks[i];
      const D3D10_SB_NAME output_system_value =
         signatures->output_system_values[i];

      if (!writemask)
         continue;
      if (output_system_value == D3D10_SB_NAME_CLIP_DISTANCE)
         output_clip_components += util_bitcount(writemask);
      else if (output_system_value == D3D10_SB_NAME_CULL_DISTANCE)
         output_cull_components += util_bitcount(writemask);
   }

   if (output_clip_components)
      ureg_property(ureg, TGSI_PROPERTY_NUM_CLIPDIST_ENABLED,
                    output_clip_components);
   if (output_cull_components)
      ureg_property(ureg, TGSI_PROPERTY_NUM_CULLDIST_ENABLED,
                    output_cull_components);
}

static void
tcs_emit_implicit_control_point_passthrough(struct Shader_xlate *sx,
                                             struct ureg_program *ureg)
{
   const struct Shader_tessellation_io_signatures *signatures =
      sx->tessellation_signatures;
   if (!signatures || !signatures->input_masks ||
       !signatures->output_masks || !sx->vertices_in ||
       !sx->tcs_vertices_out)
      return;

   if (!sx->tcs_invocation_id.File) {
      sx->tcs_invocation_id =
         ureg_DECL_system_value(ureg, TGSI_SEMANTIC_INVOCATIONID, 0);
   }

   for (unsigned i = 0;
        i < signatures->num_input_system_values &&
        i < signatures->num_output_system_values; ++i) {
      const unsigned writemask =
         signatures->input_masks[i] & signatures->output_masks[i];
      if (!writemask)
         continue;

      const D3D10_SB_NAME input_name =
         signatures->input_system_values[i];
      const D3D10_SB_NAME output_name =
         signatures->output_system_values[i];
      enum tgsi_semantic input_semantic =
         input_name == D3D10_SB_NAME_UNDEFINED ?
            TGSI_SEMANTIC_GENERIC : translate_system_name(sx, input_name);
      enum tgsi_semantic output_semantic =
         output_name == D3D10_SB_NAME_UNDEFINED ?
            TGSI_SEMANTIC_GENERIC : translate_system_name(sx, output_name);
      unsigned input_index = i;
      unsigned output_index = i;
      if (input_name == D3D10_SB_NAME_CLIP_DISTANCE ||
          input_name == D3D10_SB_NAME_CULL_DISTANCE) {
         input_index = tessellation_clip_distance_semantic_index(
            signatures, false, i);
      }
      if (output_name == D3D10_SB_NAME_CLIP_DISTANCE ||
          output_name == D3D10_SB_NAME_CULL_DISTANCE) {
         output_index = tessellation_clip_distance_semantic_index(
            signatures, true, i);
      }
      /* Control points use the source dimension, not a register array.
       * TGSI-to-NIR adds the per-vertex array for tessellation inputs.
       */
      struct ureg_src input =
         ureg_DECL_input(ureg, input_semantic, input_index, 0, 1);
      struct ureg_dst output =
         ureg_DECL_output(ureg, output_semantic, output_index);
      ureg_MOV(ureg, ureg_writemask(output, writemask),
               ureg_src_dimension_indirect(input,
                                           sx->tcs_invocation_id, 0));
   }
}

static void
dcl_sgv_input(struct Shader_xlate *sx,
              struct ureg_program *ureg,
              const struct Shader_dst_operand *dst,
              uint dcl_siv_name)
{
   struct ureg_src reg;
   assert(dst->base.index_dim == 1);
   assert(dst->base.index[0].imm < SHADER_MAX_INPUTS);

   reg = ureg_DECL_system_value(
      ureg, translate_system_name(sx, dcl_siv_name), 0);

   dcl_base_input(sx, ureg, dst, reg, dst->base.index[0].imm,
                  dcl_siv_name);
}

static void
dcl_siv_input(struct Shader_xlate *sx,
              struct ureg_program *ureg,
              const struct Shader_dst_operand *dst,
              uint dcl_siv_name)
{
   struct ureg_src reg;
   assert(dst->base.index_dim == 2);
   assert(dst->base.index[1].imm < SHADER_MAX_INPUTS);

   declare_vertices_in(sx, dst->base.index[0].imm);

   reg = ureg_DECL_input(ureg,
                         translate_system_name(sx, dcl_siv_name), 0,
                         0, 1);

   dcl_base_input(sx, ureg, dst, reg, dst->base.index[1].imm,
                  dcl_siv_name);
}

static void
dcl_ps_input(struct Shader_xlate *sx,
             struct ureg_program *ureg,
             const struct Shader_dst_operand *dst,
             uint dcl_in_ps_interp)
{
   struct ureg_src reg;
   unsigned index;

   if (dst->base.type == D3D11_SB_OPERAND_TYPE_INPUT_COVERAGE_MASK) {
      if (!sx->sample_mask_declared) {
         sx->sample_mask =
            ureg_DECL_system_value(ureg, TGSI_SEMANTIC_SAMPLEMASK, 0);
         sx->sample_mask_declared = true;
      }
      return;
   }

   if (dst->base.index_dim == 2) {
      assert(dst->base.index[1].imm < SHADER_MAX_INPUTS);
      index = dst->base.index[1].imm;
   } else if (dst->base.index_dim == 0) {
      index = 0;
   } else {
      assert(dst->base.index_dim == 1);
      assert(dst->base.index[0].imm < SHADER_MAX_INPUTS);
      index = dst->base.index[0].imm;
   }

   reg = ureg_DECL_fs_input_centroid(ureg,
                                     TGSI_SEMANTIC_GENERIC,
                                     index,
                                     translate_interpolation(dcl_in_ps_interp),
                                     translate_interpolation_location(
                                        dcl_in_ps_interp),
                                     0, 1);

   dcl_base_input(sx, ureg, dst, reg, index,
                  D3D10_SB_NAME_UNDEFINED);
}

static void
dcl_ps_sgv_input(struct Shader_xlate *sx,
                 struct ureg_program *ureg,
                 const struct Shader_dst_operand *dst,
                 uint dcl_siv_name)
{
   struct ureg_src reg;
   assert(dst->base.index_dim == 1);
   assert(dst->base.index[0].imm < SHADER_MAX_INPUTS);

   if (dcl_siv_name == D3D10_SB_NAME_POSITION) {
      ureg_property(ureg,
                    TGSI_PROPERTY_FS_COORD_ORIGIN,
                    TGSI_FS_COORD_ORIGIN_UPPER_LEFT);
      ureg_property(ureg,
                    TGSI_PROPERTY_FS_COORD_PIXEL_CENTER,
                    TGSI_FS_COORD_PIXEL_CENTER_HALF_INTEGER);
   }

   if (dcl_siv_name == D3D10_SB_NAME_POSITION ||
       dcl_siv_name == D3D10_SB_NAME_SAMPLE_INDEX) {
      reg = ureg_DECL_system_value(ureg,
                                   translate_system_name(sx, dcl_siv_name),
                                   0);
   } else {
      reg = ureg_DECL_fs_input(ureg,
                               translate_system_name(sx, dcl_siv_name),
                               0,
                               TGSI_INTERPOLATE_CONSTANT);
   }

   if (dcl_siv_name == D3D10_SB_NAME_IS_FRONT_FACE) {
      /* We need to map gallium's front_face to the one expected
       * by D3D10 */
      struct ureg_dst tmp = ureg_DECL_temporary(ureg);

      tmp = ureg_writemask(tmp, TGSI_WRITEMASK_X);

      ureg_CMP(ureg, tmp, reg,
               ureg_imm1i(ureg, 0), ureg_imm1i(ureg, -1));

      reg = ureg_scalar(ureg_src(tmp), TGSI_SWIZZLE_X);
   }

   dcl_base_input(sx, ureg, dst, reg, dst->base.index[0].imm,
                  dcl_siv_name);
}

static void
dcl_ps_siv_input(struct Shader_xlate *sx,
                 struct ureg_program *ureg,
                 const struct Shader_dst_operand *dst,
                 uint dcl_siv_name, uint dcl_in_ps_interp)
{
   struct ureg_src reg;
   assert(dst->base.index_dim == 1);
   assert(dst->base.index[0].imm < SHADER_MAX_INPUTS);

   if (dcl_siv_name == D3D10_SB_NAME_POSITION) {
      ureg_property(ureg,
                    TGSI_PROPERTY_FS_COORD_ORIGIN,
                    TGSI_FS_COORD_ORIGIN_UPPER_LEFT);
      ureg_property(ureg,
                    TGSI_PROPERTY_FS_COORD_PIXEL_CENTER,
                    TGSI_FS_COORD_PIXEL_CENTER_HALF_INTEGER);
      if (translate_interpolation_location(dcl_in_ps_interp) ==
          TGSI_INTERPOLATE_LOC_SAMPLE)
         ureg_DECL_system_value(ureg, TGSI_SEMANTIC_SAMPLEID, 0);
   }

   if (dcl_siv_name == D3D10_SB_NAME_POSITION ||
       dcl_siv_name == D3D10_SB_NAME_SAMPLE_INDEX) {
      reg = ureg_DECL_system_value(ureg,
                                   translate_system_name(sx, dcl_siv_name),
                                   0);
   } else {
      reg = ureg_DECL_fs_input_centroid(ureg,
                                        translate_system_name(sx, dcl_siv_name),
                                        0,
                                        translate_interpolation(
                                           dcl_in_ps_interp),
                                        translate_interpolation_location(
                                           dcl_in_ps_interp),
                                        0, 1);
   }

   if (dcl_siv_name == D3D10_SB_NAME_POSITION) {
      /* D3D10 expects reciprocal of interpolated 1/w as 4th component,
       * gallium/GL just interpolated 1/w */
      struct ureg_dst tmp = ureg_DECL_temporary(ureg);

      ureg_MOV(ureg, tmp, reg);
      ureg_RCP(ureg, ureg_writemask(tmp, TGSI_WRITEMASK_W),
               ureg_scalar(ureg_src(tmp), TGSI_SWIZZLE_W));
      reg = ureg_src(tmp);
   }

   dcl_base_input(sx, ureg, dst, reg, dst->base.index[0].imm,
                  dcl_siv_name);
}

static struct ureg_src
get_sample_mask(struct Shader_xlate *sx)
{
   if (!sx->sample_mask_declared) {
      sx->sample_mask =
         ureg_DECL_system_value(sx->ureg, TGSI_SEMANTIC_SAMPLEMASK, 0);
      sx->sample_mask_declared = true;
   }

   return sx->sample_mask;
}

static struct ureg_dst
get_output_coverage_mask(struct Shader_xlate *sx)
{
   if (!sx->output_coverage_mask_declared) {
      sx->output_coverage_mask =
         ureg_DECL_output_masked(sx->ureg, TGSI_SEMANTIC_SAMPLEMASK, 0,
                                 TGSI_WRITEMASK_X, 0, 1);
      sx->output_coverage_mask =
         ureg_writemask(sx->output_coverage_mask, TGSI_WRITEMASK_X);
      sx->output_coverage_mask_declared = true;
   }

   return sx->output_coverage_mask;
}

static struct ureg_src
get_cs_block_id(struct Shader_xlate *sx)
{
   if (!sx->cs_block_id_declared) {
      sx->cs_block_id =
         ureg_DECL_system_value(sx->ureg, TGSI_SEMANTIC_BLOCK_ID, 0);
      sx->cs_block_id_declared = true;
   }

   return sx->cs_block_id;
}

static struct ureg_src
get_cs_thread_id(struct Shader_xlate *sx)
{
   if (!sx->cs_thread_id_declared) {
      sx->cs_thread_id =
         ureg_DECL_system_value(sx->ureg, TGSI_SEMANTIC_THREAD_ID, 0);
      sx->cs_thread_id_declared = true;
   }

   return sx->cs_thread_id;
}

static struct ureg_src
get_cs_dispatch_thread_id(struct Shader_xlate *sx)
{
   struct ureg_program *ureg = sx->ureg;
   struct ureg_src block_id, thread_id;
   struct ureg_dst dispatch_id;
   unsigned width, height, depth;

   if (sx->cs_dispatch_thread_id_declared)
      return sx->cs_dispatch_thread_id;

   block_id = get_cs_block_id(sx);
   thread_id = get_cs_thread_id(sx);
   dispatch_id = ureg_DECL_temporary(ureg);
   width = MAX2(sx->cs_thread_group_size[0], 1);
   height = MAX2(sx->cs_thread_group_size[1], 1);
   depth = MAX2(sx->cs_thread_group_size[2], 1);

   ureg_UMUL(ureg, ureg_writemask(dispatch_id, TGSI_WRITEMASK_XYZ),
             block_id, ureg_imm4u(ureg, width, height, depth, 0));
   ureg_UADD(ureg, ureg_writemask(dispatch_id, TGSI_WRITEMASK_XYZ),
             ureg_src(dispatch_id), thread_id);
   ureg_MOV(ureg, ureg_writemask(dispatch_id, TGSI_WRITEMASK_W),
            ureg_imm1u(ureg, 0));

   sx->cs_dispatch_thread_id = ureg_src(dispatch_id);
   sx->cs_dispatch_thread_id_declared = true;
   return sx->cs_dispatch_thread_id;
}

static struct ureg_src
get_cs_group_index(struct Shader_xlate *sx)
{
   struct ureg_program *ureg = sx->ureg;
   struct ureg_src thread_id;
   struct ureg_dst index, term;
   unsigned width, height;

   if (sx->cs_group_index_declared)
      return sx->cs_group_index;

   thread_id = get_cs_thread_id(sx);
   index = ureg_DECL_temporary(ureg);
   term = ureg_DECL_temporary(ureg);
   width = MAX2(sx->cs_thread_group_size[0], 1);
   height = MAX2(sx->cs_thread_group_size[1], 1);

   ureg_UMUL(ureg, ureg_writemask(index, TGSI_WRITEMASK_X),
             ureg_scalar(thread_id, TGSI_SWIZZLE_Y), ureg_imm1u(ureg, width));
   ureg_UADD(ureg, ureg_writemask(index, TGSI_WRITEMASK_X),
             ureg_scalar(ureg_src(index), TGSI_SWIZZLE_X),
             ureg_scalar(thread_id, TGSI_SWIZZLE_X));
   ureg_UMUL(ureg, ureg_writemask(term, TGSI_WRITEMASK_X),
             ureg_scalar(thread_id, TGSI_SWIZZLE_Z),
             ureg_imm1u(ureg, width * height));
   ureg_UADD(ureg, ureg_writemask(index, TGSI_WRITEMASK_X),
             ureg_scalar(ureg_src(index), TGSI_SWIZZLE_X),
             ureg_scalar(ureg_src(term), TGSI_SWIZZLE_X));
   ureg_MOV(ureg, ureg_writemask(index, TGSI_WRITEMASK_YZW),
            ureg_imm1u(ureg, 0));

   ureg_release_temporary(ureg, term);

   sx->cs_group_index = ureg_src(index);
   sx->cs_group_index_declared = true;
   return sx->cs_group_index;
}

static struct ureg_src
get_tgsm_memory(struct Shader_xlate *sx)
{
   if (!sx->shared_memory_declared) {
      sx->shared_memory =
         ureg_DECL_memory(sx->ureg, TGSI_MEMORY_TYPE_SHARED);
      sx->shared_memory_declared = true;
   }

   return sx->shared_memory;
}

static bool
tgsm_operand_slot(const struct Shader_operand *operand, unsigned *slot)
{
   if (!operand || !slot ||
       operand->type != DX10_SM5_OPERAND_TYPE_THREAD_GROUP_SHARED_MEMORY ||
       operand->index_dim != 1 ||
       operand->index[0].index_rep != D3D10_SB_OPERAND_INDEX_IMMEDIATE32 ||
       operand->index[0].imm >= SHADER_MAX_TGSM)
      return false;

   *slot = operand->index[0].imm;
   return true;
}

static struct ureg_src
tgsm_byte_offset(struct Shader_xlate *sx,
                 const struct Shader_operand *operand,
                 struct ureg_src offset,
                 struct ureg_dst *tmp)
{
   unsigned slot = 0;
   if (!tgsm_operand_slot(operand, &slot))
      return offset;

   const unsigned base = sx->tgsm[slot].byte_offset;
   if (!base)
      return offset;

   *tmp = ureg_DECL_temporary(sx->ureg);
   ureg_UADD(sx->ureg, *tmp, offset, ureg_imm1u(sx->ureg, base));
   return ureg_src(*tmp);
}

static struct ureg_src
translate_relative_operand_src(struct Shader_xlate *sx,
                               const struct Shader_relative_operand *operand)
{
   struct ureg_src reg;

   switch (operand->type) {
   case D3D10_SB_OPERAND_TYPE_TEMP:
      assert(operand->index[0].imm < SHADER_MAX_TEMPS);

      reg = ureg_src(sx->temps[sx->temp_offset + operand->index[0].imm]);
      break;

   case D3D10_SB_OPERAND_TYPE_INPUT_PRIMITIVEID:
      reg = sx->prim_id;
      break;

   case D3D11_SB_OPERAND_TYPE_INPUT_GS_INSTANCE_ID:
      reg = sx->gs_instance_id;
      break;

   case DX10_SM5_OPERAND_TYPE_INPUT_THREAD_GROUP_ID:
      reg = get_cs_block_id(sx);
      break;

   case DX10_SM5_OPERAND_TYPE_INPUT_THREAD_ID:
      reg = get_cs_dispatch_thread_id(sx);
      break;

   case DX10_SM5_OPERAND_TYPE_INPUT_THREAD_ID_IN_GROUP:
      reg = get_cs_thread_id(sx);
      break;

   case DX10_SM5_OPERAND_TYPE_INPUT_THREAD_ID_IN_GROUP_FLATTENED:
      reg = get_cs_group_index(sx);
      break;

   case D3D10_SB_OPERAND_TYPE_INDEXABLE_TEMP:
      assert(operand->index[1].imm < SHADER_MAX_TEMPS);

      reg = ureg_src(sx->temps[sx->indexable_temp_offsets[operand->index[0].imm] +
            operand->index[1].imm]);
      break;

   case D3D10_SB_OPERAND_TYPE_CONSTANT_BUFFER:
      assert(operand->index[1].imm < SHADER_MAX_CONSTS);
      assert(operand->index[0].imm + 1 < PIPE_MAX_CONSTANT_BUFFERS);

      reg = ureg_src_register(TGSI_FILE_CONSTANT, operand->index[1].imm);
      reg = ureg_src_dimension(reg, operand->index[0].imm + 1);
      break;

   case D3D10_SB_OPERAND_TYPE_IMMEDIATE_CONSTANT_BUFFER:
      reg = sx->imms;
      reg.Index += operand->index[0].imm;
      break;

   case D3D10_SB_OPERAND_TYPE_INPUT:
   case D3D10_SB_OPERAND_TYPE_OUTPUT:
   case D3D10_SB_OPERAND_TYPE_IMMEDIATE32:
   case D3D10_SB_OPERAND_TYPE_IMMEDIATE64:
   case D3D10_SB_OPERAND_TYPE_SAMPLER:
   case D3D10_SB_OPERAND_TYPE_RESOURCE:
   case D3D10_SB_OPERAND_TYPE_LABEL:
   case D3D10_SB_OPERAND_TYPE_OUTPUT_DEPTH:
   case D3D10_SB_OPERAND_TYPE_NULL:
   case D3D10_SB_OPERAND_TYPE_RASTERIZER:
   case D3D10_SB_OPERAND_TYPE_OUTPUT_COVERAGE_MASK:
      SHADER_TRANSLATION_UNSUPPORTED(sx, true);
      reg = ureg_src(ureg_DECL_temporary(sx->ureg));
      break;

   default:
      assert(0);                /* should never happen */
      reg = ureg_src(ureg_DECL_temporary(sx->ureg));
   }

   return ureg_scalar(reg, operand->comp);
}

static bool
is_output_depth_operand_type(D3D10_SB_OPERAND_TYPE type)
{
   return type == D3D10_SB_OPERAND_TYPE_OUTPUT_DEPTH ||
          type == D3D11_SB_OPERAND_TYPE_OUTPUT_DEPTH_GREATER_EQUAL ||
          type == D3D11_SB_OPERAND_TYPE_OUTPUT_DEPTH_LESS_EQUAL;
}

static bool
is_output_coverage_mask_operand_type(D3D10_SB_OPERAND_TYPE type)
{
   return type == D3D10_SB_OPERAND_TYPE_OUTPUT_COVERAGE_MASK;
}

static struct ureg_src
translate_relative_operand(struct Shader_xlate *sx,
                           const struct Shader_relative_operand *operand)
{
   struct ureg_dst addr_reg = sx->addrs[sx->addr_cur];
   ureg_UARL(sx->ureg, addr_reg,
             translate_relative_operand_src(sx, operand));
      
   sx->addr_cur++;
   if(sx->addr_cur >= SHADER_MAX_ADDRS) sx->addr_cur = 0;

   return ureg_src(addr_reg);
}

static bool
find_input_index_range(struct Shader_xlate *sx,
                       uint base,
                       uint *count)
{
   for (uint i = 0; i < sx->num_input_ranges; i++) {
      uint first = sx->input_ranges[i].first;
      uint range_count = sx->input_ranges[i].count;
      uint last = first + range_count;

      if (base >= first && base < last) {
         *count = last - base;
         return true;
      }
   }

   return false;
}

static void
prepare_tcs_indexed_output_stores(struct Shader_xlate *sx,
                                   const struct Shader_opcode *opcode)
{
   assert(!sx->num_tcs_indexed_output_stores);
   if (!sx->is_tcs || !sx->tcs_patch_phase_active)
      return;

   for (unsigned i = 0; i < opcode->num_dst; ++i) {
      const struct Shader_dst_operand *dst = &opcode->dst[i];
      if (dst->base.type != D3D10_SB_OPERAND_TYPE_OUTPUT ||
          dst->base.index_dim != 1)
         continue;

      const struct Shader_index *index = &dst->base.index[0];
      if (index->index_rep != D3D10_SB_OPERAND_INDEX_RELATIVE &&
          index->index_rep != D3D10_SB_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE)
         continue;

      const unsigned base =
         index->index_rep == D3D10_SB_OPERAND_INDEX_RELATIVE ? 0 : index->imm;
      if (base >= SHADER_MAX_OUTPUTS ||
          sx->tcs_tess_factor_range_end[base] <= base ||
          dst->mask != D3D10_SB_OPERAND_4_COMPONENT_MASK_X) {
         SHADER_TRANSLATION_UNSUPPORTED(sx, true);
         return;
      }

      const unsigned slot = sx->num_tcs_indexed_output_stores++;
      sx->tcs_indexed_output_stores[slot].operand = dst;
      sx->tcs_indexed_output_stores[slot].first =
         sx->tcs_tess_factor_range_first[base];
      sx->tcs_indexed_output_stores[slot].end =
         sx->tcs_tess_factor_range_end[base];
      sx->tcs_indexed_output_stores[slot].value =
         ureg_DECL_temporary(sx->ureg);
      sx->tcs_indexed_output_stores[slot].index =
         ureg_DECL_temporary(sx->ureg);

      /* Snapshot all indices before executing a multi-destination opcode:
       * another destination may overwrite the register holding an index.
       */
      ureg_UADD(sx->ureg,
                ureg_writemask(sx->tcs_indexed_output_stores[slot].index,
                               TGSI_WRITEMASK_X),
                translate_relative_operand_src(sx, &index->rel),
                ureg_imm1u(sx->ureg, base));
   }
}

static void
emit_tcs_indexed_output_stores(struct Shader_xlate *sx)
{
   if (!sx->num_tcs_indexed_output_stores)
      return;

   struct ureg_program *ureg = sx->ureg;
   struct ureg_dst predicate = ureg_DECL_temporary(ureg);
   for (unsigned slot = 0; slot < sx->num_tcs_indexed_output_stores; ++slot) {
      const unsigned first = sx->tcs_indexed_output_stores[slot].first;
      const unsigned end = sx->tcs_indexed_output_stores[slot].end;
      const struct ureg_src index = ureg_scalar(
         ureg_src(sx->tcs_indexed_output_stores[slot].index), TGSI_SWIZZLE_X);
      const struct ureg_src value = ureg_scalar(
         ureg_src(sx->tcs_indexed_output_stores[slot].value), TGSI_SWIZZLE_X);

      for (unsigned output = first; output < end; ++output) {
         unsigned label = 0;
         ureg_USEQ(ureg, ureg_writemask(predicate, TGSI_WRITEMASK_X),
                    index, ureg_imm1u(ureg, output));
         ureg_UIF(ureg, ureg_scalar(ureg_src(predicate), TGSI_SWIZZLE_X),
                   &label);
         ureg_MOV(ureg,
                  ureg_writemask(sx->outputs[output].reg[0],
                                 sx->tcs_tess_factor_writemask[output]),
                  value);
         ureg_ENDIF(ureg);
      }
      ureg_release_temporary(ureg, sx->tcs_indexed_output_stores[slot].index);
      ureg_release_temporary(ureg, sx->tcs_indexed_output_stores[slot].value);
   }
   ureg_release_temporary(ureg, predicate);
   sx->num_tcs_indexed_output_stores = 0;
}

static struct ureg_src
translate_indexed_input(struct Shader_xlate *sx,
                        uint base,
                        struct ureg_src rel)
{
   struct ureg_program *ureg = sx->ureg;
   struct ureg_dst selected, predicate;
   uint count;

   assert(base < SHADER_MAX_INPUTS);

   if (!find_input_index_range(sx, base, &count)) {
      SHADER_TRANSLATION_UNSUPPORTED(sx, true);
      return sx->inputs[base].reg;
   }

   if (count <= 1)
      return sx->inputs[base].reg;

   selected = ureg_DECL_temporary(ureg);
   predicate = ureg_DECL_temporary(ureg);

   ureg_MOV(ureg, selected, sx->inputs[base].reg);
   for (uint i = 1; i < count; i++) {
      if (!sx->inputs[base + i].declared) {
         SHADER_TRANSLATION_UNSUPPORTED(sx, true);
         continue;
      }

      ureg_USEQ(ureg, predicate, rel, ureg_imm1u(ureg, i));
      ureg_UCMP(ureg, selected, ureg_src(predicate),
                sx->inputs[base + i].reg, ureg_src(selected));
   }

   ureg_release_temporary(ureg, predicate);

   return ureg_src(selected);
}

static struct ureg_dst
translate_operand(struct Shader_xlate *sx,
                  const struct Shader_operand *operand,
                  unsigned writemask)
{
   struct ureg_dst reg;

   switch (operand->type) {
   case D3D10_SB_OPERAND_TYPE_TEMP:
      assert(operand->index_dim == 1);
      assert(operand->index[0].index_rep == D3D10_SB_OPERAND_INDEX_IMMEDIATE32);
      assert(operand->index[0].imm < SHADER_MAX_TEMPS);

      reg = sx->temps[sx->temp_offset + operand->index[0].imm];
      break;

   case D3D10_SB_OPERAND_TYPE_OUTPUT:
      assert(operand->index_dim == 1 || operand->index_dim == 2);
      const struct Shader_index *index =
         &operand->index[operand->index_dim == 2 ? 1 : 0];

      if (index->index_rep == D3D10_SB_OPERAND_INDEX_IMMEDIATE32) {
         unsigned output = index->imm;
         assert(output < SHADER_MAX_OUTPUTS);
         if (sx->outputs[output].gs_scalar_mask) {
            reg = sx->outputs[output].gs_value;
         } else if (!writemask) {
            reg = sx->outputs[output].reg[0];
         } else {
            unsigned i;
            /* Default to a valid component in case writemask has no low bits. */
            reg = sx->outputs[output].reg[0];
            for (i = 0; i < 4; ++i) {
               unsigned mask = 1 << i;
               if ((writemask & mask)) {
                  reg = sx->outputs[output].reg[i];
                  break;
               }
            }
         }
      } else if (index->index_rep == D3D10_SB_OPERAND_INDEX_RELATIVE ||
                 index->index_rep ==
                    D3D10_SB_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE) {
         if (sx->has_gs_scalar_outputs) {
            YTTRIUM_WARN("yttrium: shader TGSI translation failed "
                         "owner=d3d10umd-shader component=gs-scalar-output "
                         "reason=indexed-output-access "
                         "action=abort-translation\n");
            sx->translation_failed = true;
            reg = ureg_DECL_temporary(sx->ureg);
            break;
         }
         unsigned output =
            index->index_rep == D3D10_SB_OPERAND_INDEX_RELATIVE ?
               0 : index->imm;
         assert(output < SHADER_MAX_OUTPUTS);
         struct ureg_src addr =
            translate_relative_operand(sx, &index->rel);
         reg = ureg_dst_indirect(sx->outputs[output].reg[0], addr);
      } else {
         SHADER_TRANSLATION_UNSUPPORTED(sx, true);
         reg = ureg_DECL_temporary(sx->ureg);
      }

      if (operand->index_dim == 2) {
         switch (operand->index[0].index_rep) {
         case D3D10_SB_OPERAND_INDEX_IMMEDIATE32:
            reg = ureg_dst_dimension(reg, operand->index[0].imm);
            break;
         case D3D10_SB_OPERAND_INDEX_RELATIVE:
            reg = ureg_dst_dimension_indirect(
               reg, translate_relative_operand(sx, &operand->index[0].rel), 0);
            break;
         case D3D10_SB_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE:
            reg = ureg_dst_dimension_indirect(
               reg, translate_relative_operand(sx, &operand->index[0].rel),
               operand->index[0].imm);
            break;
         default:
            SHADER_TRANSLATION_UNSUPPORTED(sx, true);
            break;
         }
      }
      break;

   case D3D10_SB_OPERAND_TYPE_OUTPUT_DEPTH:
   case D3D11_SB_OPERAND_TYPE_OUTPUT_DEPTH_GREATER_EQUAL:
   case D3D11_SB_OPERAND_TYPE_OUTPUT_DEPTH_LESS_EQUAL:
      assert(operand->index_dim == 0);

      reg = sx->output_depth;
      break;

   case D3D10_SB_OPERAND_TYPE_OUTPUT_COVERAGE_MASK:
      assert(operand->index_dim == 0);

      reg = get_output_coverage_mask(sx);
      break;

   case D3D10_SB_OPERAND_TYPE_INPUT_PRIMITIVEID:
      assert(operand->index_dim == 0);

      reg = ureg_dst(sx->prim_id);
      break;

   case D3D10_SB_OPERAND_TYPE_INPUT:
   case DX11_SM5_OPERAND_TYPE_INPUT_CONTROL_POINT:
   case D3D10_SB_OPERAND_TYPE_INDEXABLE_TEMP:
   case D3D10_SB_OPERAND_TYPE_IMMEDIATE32:
   case D3D10_SB_OPERAND_TYPE_IMMEDIATE64:
   case D3D10_SB_OPERAND_TYPE_SAMPLER:
   case D3D10_SB_OPERAND_TYPE_RESOURCE:
   case D3D10_SB_OPERAND_TYPE_CONSTANT_BUFFER:
   case D3D10_SB_OPERAND_TYPE_IMMEDIATE_CONSTANT_BUFFER:
   case D3D10_SB_OPERAND_TYPE_LABEL:
   case D3D10_SB_OPERAND_TYPE_NULL:
   case D3D10_SB_OPERAND_TYPE_RASTERIZER:
   default:
      /* XXX: Translate more operands types.
       */
      SHADER_TRANSLATION_UNSUPPORTED(sx, true);
      reg = ureg_DECL_temporary(sx->ureg);
   }

   return reg;
}

static struct ureg_src
translate_indexable_temp(struct Shader_xlate *sx,
                         const struct Shader_operand *operand)
{
   struct ureg_src reg;
   switch (operand->index[1].index_rep) {
   case D3D10_SB_OPERAND_INDEX_IMMEDIATE32:
      reg = ureg_src(
         sx->temps[sx->indexable_temp_offsets[operand->index[0].imm] +
                   operand->index[1].imm]);
      break;
   case D3D10_SB_OPERAND_INDEX_RELATIVE:
      reg = ureg_src_indirect(
         ureg_src(sx->temps[
                     sx->indexable_temp_offsets[operand->index[0].imm]]),
         translate_relative_operand(sx,
                                    &operand->index[1].rel));
      break;
   case D3D10_SB_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE:
      reg = ureg_src_indirect(
         ureg_src(sx->temps[
                     operand->index[1].imm +
                     sx->indexable_temp_offsets[operand->index[0].imm]]),
         translate_relative_operand(sx,
                                    &operand->index[1].rel));
      break;
   default:
      /* XXX: Other index representations.
       */
      SHADER_TRANSLATION_UNSUPPORTED(sx, true);
      reg = ureg_src(ureg_DECL_temporary(sx->ureg));
   }
   return reg;
}

static struct ureg_dst
translate_dst_operand(struct Shader_xlate *sx,
                      const struct Shader_dst_operand *operand,
                      bool saturate)
{
   struct ureg_dst reg;
   unsigned writemask =
      operand->mask >> D3D10_SB_OPERAND_4_COMPONENT_MASK_SHIFT;

   assert((D3D10_SB_OPERAND_4_COMPONENT_MASK_SHIFT) == 4);
   assert((D3D10_SB_OPERAND_4_COMPONENT_MASK_X >> 4) == TGSI_WRITEMASK_X);
   assert((D3D10_SB_OPERAND_4_COMPONENT_MASK_Y >> 4) == TGSI_WRITEMASK_Y);
   assert((D3D10_SB_OPERAND_4_COMPONENT_MASK_Z >> 4) == TGSI_WRITEMASK_Z);
   assert((D3D10_SB_OPERAND_4_COMPONENT_MASK_W >> 4) == TGSI_WRITEMASK_W);

   for (unsigned i = 0; i < sx->num_tcs_indexed_output_stores; ++i) {
      if (sx->tcs_indexed_output_stores[i].operand == operand) {
         reg = ureg_writemask(sx->tcs_indexed_output_stores[i].value,
                              writemask);
         return saturate ? ureg_saturate(reg) : reg;
      }
   }

   if (operand->base.type == D3D10_SB_OPERAND_TYPE_OUTPUT &&
       operand->base.index_dim == 1 &&
       operand->base.index[0].index_rep ==
          D3D10_SB_OPERAND_INDEX_IMMEDIATE32) {
      const unsigned output = operand->base.index[0].imm;
      const unsigned mapped_writemask =
         output < SHADER_MAX_OUTPUTS ?
            sx->tcs_tess_factor_writemask[output] : 0;
      if (mapped_writemask && writemask == TGSI_WRITEMASK_X)
         writemask = mapped_writemask;
   }

   switch (operand->base.type) {
   case D3D10_SB_OPERAND_TYPE_INDEXABLE_TEMP:
      assert(operand->base.index_dim == 2);
      assert(operand->base.index[0].index_rep == D3D10_SB_OPERAND_INDEX_IMMEDIATE32);
      assert(operand->base.index[0].imm < SHADER_MAX_INDEXABLE_TEMPS);

      reg = ureg_dst(translate_indexable_temp(sx, &operand->base));
      break;

   default:
      reg = translate_operand(sx, &operand->base, writemask);
   }

   /* oDepth often has an empty writemask */
   if (is_output_coverage_mask_operand_type(operand->base.type)) {
      reg = ureg_writemask(reg, TGSI_WRITEMASK_X);
   } else if (!is_output_depth_operand_type(operand->base.type)) {
      reg = ureg_writemask(reg, writemask);
   }

   if (saturate) {
      reg = ureg_saturate(reg);
   }

   return reg;
}

static unsigned
shader_dst_writemask(const struct Shader_dst_operand *operand)
{
   return operand->mask >> D3D10_SB_OPERAND_4_COMPONENT_MASK_SHIFT;
}

static bool
shader_dst_is_simple_temp(const struct Shader_dst_operand *operand,
                          unsigned *index)
{
   if (operand->base.type != D3D10_SB_OPERAND_TYPE_TEMP ||
       operand->base.index_dim != 1 ||
       operand->base.index[0].index_rep != D3D10_SB_OPERAND_INDEX_IMMEDIATE32 ||
       operand->base.index[0].imm >= SHADER_MAX_TEMPS)
      return false;

   *index = operand->base.index[0].imm;
   return true;
}

static bool
sample_info_writes_float_color0(const struct Shader_dst_operand *operand)
{
   return operand->base.type == D3D10_SB_OPERAND_TYPE_OUTPUT &&
          operand->base.index_dim == 1 &&
          operand->base.index[0].index_rep == D3D10_SB_OPERAND_INDEX_IMMEDIATE32 &&
          operand->base.index[0].imm == 0;
}

static bool
try_emit_standard_4x_sample_position(struct ureg_program *ureg,
                                     struct ureg_dst dstreg,
                                     const struct Shader_src_operand *sample_index)
{
   static const float sample_positions[4][2] = {
      {-2.0f / 16.0f, -6.0f / 16.0f},
      { 6.0f / 16.0f, -2.0f / 16.0f},
      {-6.0f / 16.0f,  2.0f / 16.0f},
      { 2.0f / 16.0f,  6.0f / 16.0f},
   };
   unsigned index;

   if (sample_index->base.type != D3D10_SB_OPERAND_TYPE_IMMEDIATE32)
      return false;

   index = sample_index->imm[0].u32;
   if (index >= ARRAY_SIZE(sample_positions))
      return false;

   float value[4] = {0.0f, 0.0f, 0.0f, 0.0f};
   unsigned component = 0;
   for (unsigned i = 0; i < 4 && component < 2; i++) {
      if (!(dstreg.WriteMask & (1u << i)))
         continue;
      value[i] = sample_positions[index][component++];
   }
   ureg_MOV(ureg, dstreg,
            ureg_imm4f(ureg, value[0], value[1], value[2], value[3]));
   return true;
}

static struct ureg_src
declare_sampler_view_for_resource(struct Shader_xlate *sx,
                                  unsigned view_index,
                                  unsigned resource,
                                  enum tgsi_texture_type target)
{
   assert(view_index < SHADER_MAX_RESOURCES);
   assert(resource < SHADER_MAX_RESOURCES);

   if (ureg_src_is_undef(sx->sv[view_index])) {
      sx->sv[view_index] =
         ureg_DECL_sampler_view(sx->ureg, view_index, target,
                                sx->resources[resource].return_type[0],
                                sx->resources[resource].return_type[1],
                                sx->resources[resource].return_type[2],
                                sx->resources[resource].return_type[3]);
   }

   return sx->sv[view_index];
}

static struct ureg_src
translate_src_operand(struct Shader_xlate *sx,
                      const struct Shader_src_operand *operand,
                      const enum dx10_opcode_format format)
{
   struct ureg_src reg;

   switch (operand->base.type) {
   case D3D10_SB_OPERAND_TYPE_INPUT:
   case DX11_SM5_OPERAND_TYPE_INPUT_CONTROL_POINT:
      if (operand->base.index_dim == 0) {
         reg = sx->inputs[0].reg;
      } else if (operand->base.index_dim == 1) {
         switch (operand->base.index[0].index_rep) {
         case D3D10_SB_OPERAND_INDEX_IMMEDIATE32:
            assert(operand->base.index[0].imm < SHADER_MAX_INPUTS);
            reg = sx->inputs[operand->base.index[0].imm].reg;
            break;
         case D3D10_SB_OPERAND_INDEX_RELATIVE: {
            struct ureg_src rel =
               translate_relative_operand_src(sx, &operand->base.index[0].rel);
            reg = translate_indexed_input(sx, 0, rel);
         }
            break;
         case D3D10_SB_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE: {
            struct ureg_src rel =
               translate_relative_operand_src(sx, &operand->base.index[0].rel);
            reg = translate_indexed_input(sx, operand->base.index[0].imm, rel);
         }
            break;
        default:
           /* XXX: Other index representations.
            */
            SHADER_TRANSLATION_UNSUPPORTED(sx, true);
            /* Ensure reg is initialized even for unsupported index reps. */
            reg = ureg_src(ureg_DECL_temporary(sx->ureg));

         }
      } else {
         assert(operand->base.index_dim == 2);
         assert(operand->base.index[1].imm < SHADER_MAX_INPUTS);

         switch (operand->base.index[1].index_rep) {
         case D3D10_SB_OPERAND_INDEX_IMMEDIATE32:
            reg = sx->inputs[operand->base.index[1].imm].reg;
            break;
         case D3D10_SB_OPERAND_INDEX_RELATIVE: {
            struct ureg_src rel =
               translate_relative_operand_src(sx, &operand->base.index[1].rel);
            reg = translate_indexed_input(sx, 0, rel);
         }
            break;
         case D3D10_SB_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE: {
            struct ureg_src rel =
               translate_relative_operand_src(sx, &operand->base.index[1].rel);
            reg = translate_indexed_input(sx, operand->base.index[1].imm, rel);
         }
            break;
        default:
           /* XXX: Other index representations.
            */
            SHADER_TRANSLATION_UNSUPPORTED(sx, true);
            /* Ensure reg is initialized even for unsupported index reps. */
            reg = ureg_src(ureg_DECL_temporary(sx->ureg));
         }

         switch (operand->base.index[0].index_rep) {
         case D3D10_SB_OPERAND_INDEX_IMMEDIATE32:
            reg = ureg_src_dimension(reg, operand->base.index[0].imm);
            break;
         case D3D10_SB_OPERAND_INDEX_RELATIVE:{
            struct ureg_src tmp =
               translate_relative_operand(sx, &operand->base.index[0].rel);
            reg = ureg_src_dimension_indirect(reg, tmp, 0);
         }
            break;
         case D3D10_SB_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE: {
            struct ureg_src tmp =
               translate_relative_operand(sx, &operand->base.index[0].rel);
            reg = ureg_src_dimension_indirect(reg, tmp, operand->base.index[0].imm);
         }
            break;
        default:
           /* XXX: Other index representations.
            */
            SHADER_TRANSLATION_UNSUPPORTED(sx, true);
            /* Ensure reg is initialized even for unsupported index reps. */
            reg = ureg_src(ureg_DECL_temporary(sx->ureg));
         }
      }
      break;

   case DX11_SM5_OPERAND_TYPE_INPUT_PATCH_CONSTANT:
      assert(operand->base.index_dim == 1);
      assert(operand->base.index[0].index_rep ==
             D3D10_SB_OPERAND_INDEX_IMMEDIATE32);
      assert(operand->base.index[0].imm < SHADER_MAX_INPUTS);
      reg = sx->patch_inputs[operand->base.index[0].imm].reg;
      break;

   case D3D11_SB_OPERAND_TYPE_INPUT_COVERAGE_MASK:
      reg = get_sample_mask(sx);
      break;

   case DX11_SM5_OPERAND_TYPE_OUTPUT_CONTROL_POINT_ID:
      assert(operand->base.index_dim == 0);
      reg = sx->tcs_invocation_id;
      break;

   case DX11_SM5_OPERAND_TYPE_INPUT_FORK_INSTANCE_ID:
   case DX11_SM5_OPERAND_TYPE_INPUT_JOIN_INSTANCE_ID:
      assert(operand->base.index_dim == 0);
      reg = ureg_src(sx->tcs_phase_instance_id);
      break;

   case DX11_SM5_OPERAND_TYPE_INPUT_DOMAIN_POINT:
      assert(operand->base.index_dim == 0);
      reg = sx->tes_coord;
      break;

   case DX10_SM5_OPERAND_TYPE_INPUT_THREAD_GROUP_ID:
      assert(operand->base.index_dim == 0);
      reg = get_cs_block_id(sx);
      break;

   case DX10_SM5_OPERAND_TYPE_INPUT_THREAD_ID:
      assert(operand->base.index_dim == 0);
      reg = get_cs_dispatch_thread_id(sx);
      break;

   case DX10_SM5_OPERAND_TYPE_INPUT_THREAD_ID_IN_GROUP:
      assert(operand->base.index_dim == 0);
      reg = get_cs_thread_id(sx);
      break;

   case DX10_SM5_OPERAND_TYPE_INPUT_THREAD_ID_IN_GROUP_FLATTENED:
      assert(operand->base.index_dim == 0);
      reg = get_cs_group_index(sx);
      break;

   case D3D10_SB_OPERAND_TYPE_INDEXABLE_TEMP:
      assert(operand->base.index_dim == 2);
      assert(operand->base.index[0].index_rep == D3D10_SB_OPERAND_INDEX_IMMEDIATE32);
      assert(operand->base.index[0].imm < SHADER_MAX_INDEXABLE_TEMPS);

      reg = translate_indexable_temp(sx, &operand->base);
      break;

   case D3D10_SB_OPERAND_TYPE_IMMEDIATE32:
      switch (format) {
      case OF_FLOAT:
         reg = ureg_imm4f(sx->ureg,
                          operand->imm[0].f32,
                          operand->imm[1].f32,
                          operand->imm[2].f32,
                          operand->imm[3].f32);
         break;
      case OF_INT:
         reg = ureg_imm4i(sx->ureg,
                          operand->imm[0].i32,
                          operand->imm[1].i32,
                          operand->imm[2].i32,
                          operand->imm[3].i32);
         break;
      case OF_UINT:
         reg = ureg_imm4u(sx->ureg,
                          operand->imm[0].u32,
                          operand->imm[1].u32,
                          operand->imm[2].u32,
                          operand->imm[3].u32);
         break;
      default:
         assert(0);
         reg = ureg_src(ureg_DECL_temporary(sx->ureg));
      }
      break;

   case D3D10_SB_OPERAND_TYPE_SAMPLER:
      assert(operand->base.index_dim == 1);
      assert(operand->base.index[0].index_rep == D3D10_SB_OPERAND_INDEX_IMMEDIATE32);
      assert(operand->base.index[0].imm < SHADER_MAX_SAMPLERS);

      reg = sx->samplers[operand->base.index[0].imm];
      break;

   case D3D10_SB_OPERAND_TYPE_RESOURCE:
      assert(operand->base.index_dim == 1);
      assert(operand->base.index[0].index_rep == D3D10_SB_OPERAND_INDEX_IMMEDIATE32);
      assert(operand->base.index[0].imm < SHADER_MAX_RESOURCES);

      reg =
         declare_sampler_view_for_resource(
            sx, operand->base.index[0].imm, operand->base.index[0].imm,
            sx->resources[operand->base.index[0].imm].target);
      break;

   case D3D10_SB_OPERAND_TYPE_CONSTANT_BUFFER:
      assert(operand->base.index_dim == 2);

      assert(operand->base.index[0].index_rep == D3D10_SB_OPERAND_INDEX_IMMEDIATE32);
      assert(operand->base.index[0].imm + 1 < PIPE_MAX_CONSTANT_BUFFERS);

      switch (operand->base.index[1].index_rep) {
      case D3D10_SB_OPERAND_INDEX_IMMEDIATE32:
         assert(operand->base.index[1].imm < SHADER_MAX_CONSTS);

         reg = ureg_src_register(TGSI_FILE_CONSTANT, operand->base.index[1].imm);
         reg = ureg_src_dimension(reg, operand->base.index[0].imm + 1);
         break;
      case D3D10_SB_OPERAND_INDEX_RELATIVE:
      case D3D10_SB_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE:
         reg = ureg_src_register(TGSI_FILE_CONSTANT, operand->base.index[1].imm);
         reg = ureg_src_indirect(
            reg,
            translate_relative_operand(sx, &operand->base.index[1].rel));
         reg = ureg_src_dimension(reg, operand->base.index[0].imm + 1);
         break;
      default:
         /* XXX: Other index representations.
          */
         SHADER_TRANSLATION_UNSUPPORTED(sx, true);
         /* Ensure reg is initialized even for unsupported index reps. */
         reg = ureg_src(ureg_DECL_temporary(sx->ureg));
      }

      break;

   case D3D10_SB_OPERAND_TYPE_IMMEDIATE_CONSTANT_BUFFER:
      assert(operand->base.index_dim == 1);

      switch (operand->base.index[0].index_rep) {
      case D3D10_SB_OPERAND_INDEX_IMMEDIATE32:
         reg = sx->imms;
         reg.Index += operand->base.index[0].imm;
         break;
      case D3D10_SB_OPERAND_INDEX_RELATIVE:
      case D3D10_SB_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE:
         reg = sx->imms;
         reg.Index += operand->base.index[0].imm;
         reg = ureg_src_indirect(
            sx->imms,
            translate_relative_operand(sx, &operand->base.index[0].rel));
         break;
      default:
         /* XXX: Other index representations.
          */
         SHADER_TRANSLATION_UNSUPPORTED(sx, true);
         /* Ensure reg is initialized even for unsupported index reps. */
         reg = ureg_src(ureg_DECL_temporary(sx->ureg));
      }
      break;

   case D3D10_SB_OPERAND_TYPE_INPUT_PRIMITIVEID:
      reg = sx->prim_id;
      break;

   case D3D11_SB_OPERAND_TYPE_INPUT_GS_INSTANCE_ID:
      reg = sx->gs_instance_id;
      break;

   default:
      reg = ureg_src(translate_operand(sx, &operand->base, 0));
   }

   reg = ureg_swizzle(reg,
                      operand->swizzle[0],
                      operand->swizzle[1],
                      operand->swizzle[2],
                      operand->swizzle[3]);

   switch (operand->modifier) {
   case D3D10_SB_OPERAND_MODIFIER_NONE:
      break;
   case D3D10_SB_OPERAND_MODIFIER_NEG:
      reg = ureg_negate(reg);
      break;
   case D3D10_SB_OPERAND_MODIFIER_ABS:
      reg = ureg_abs(reg);
      break;
   case D3D10_SB_OPERAND_MODIFIER_ABSNEG:
      reg = ureg_negate(ureg_abs(reg));
      break;
   default:
      assert(0);
   }

   return reg;
}

static struct ureg_src
declare_compact_sampler_for_resource(struct Shader_xlate *sx,
                                     unsigned resource,
                                     unsigned sampler,
                                     enum tgsi_texture_type target,
                                     unsigned *binding_slot);

static bool
translate_binding_operand_index(struct Shader_xlate *sx,
                                const struct Shader_src_operand *operand,
                                unsigned expected_type,
                                unsigned limit,
                                const char *role,
                                unsigned *index)
{
   if (!operand || !index || operand->base.type != expected_type ||
       operand->base.index_dim != 1 ||
       operand->base.index[0].index_rep !=
          D3D10_SB_OPERAND_INDEX_IMMEDIATE32 ||
       operand->base.index[0].imm >= limit) {
      YTTRIUM_WARN("yttrium: shader TGSI translation failed "
                   "owner=d3d10umd-shader component=sampler-binding-map "
                   "reason=invalid-binding-operand role=%s type=%u "
                   "index_dim=%u action=abort-translation\n",
                   role ? role : "unknown",
                   operand ? operand->base.type : ~0u,
                   operand ? operand->base.index_dim : 0);
      sx->translation_failed = true;
      return false;
   }

   *index = operand->base.index[0].imm;
   return true;
}

static struct ureg_src
declare_compact_sampler_for_resource(struct Shader_xlate *sx,
                                     unsigned resource,
                                     unsigned sampler,
                                     enum tgsi_texture_type target,
                                     unsigned *binding_slot)
{
   const unsigned sampler_none = PIPE_MAX_SAMPLERS;
   unsigned slot = PIPE_MAX_SAMPLERS;

   assert(sx->sampler_binding_map);
   assert(resource < SHADER_MAX_RESOURCES);
   assert(sampler <= sampler_none);

   for (unsigned i = 0; i < sx->sampler_binding_count; i++) {
      const struct Shader_sampler_binding_slot *binding =
         &sx->sampler_binding_slots[i];
      bool same_view = binding->resource == resource &&
                       binding->target == target;
      for (unsigned component = 0; same_view && component < 4; component++)
         same_view = binding->return_type[component] ==
                     sx->resources[resource].return_type[component];
      if (!same_view)
         continue;

      if (sampler == sampler_none || binding->sampler == sampler) {
         slot = i;
         break;
      }

      if (binding->sampler == sampler_none) {
         sx->sampler_binding_slots[i].sampler = sampler;
         sx->sampler_binding_map->sampler_state_index[i] = (uint8_t)sampler;
         slot = i;
         break;
      }
   }

   if (slot == PIPE_MAX_SAMPLERS) {
      if (sx->sampler_binding_count >= PIPE_MAX_SAMPLERS) {
         YTTRIUM_WARN("yttrium: shader TGSI translation failed "
                      "owner=d3d10umd-shader component=sampler-binding-map "
                      "reason=combined-slot-limit resource=%u sampler=%u "
                      "target=%u slots=%u action=abort-translation\n",
                      resource, sampler, target,
                      sx->sampler_binding_count);
         sx->translation_failed = true;
         slot = 0;
      } else {
         slot = sx->sampler_binding_count++;
         struct Shader_sampler_binding_slot *binding =
            &sx->sampler_binding_slots[slot];
         binding->resource = resource;
         binding->sampler = sampler;
         binding->target = target;
         memcpy(binding->return_type, sx->resources[resource].return_type,
                sizeof(binding->return_type));

         sx->sampler_binding_map->valid_mask |= 1u << slot;
         sx->sampler_binding_map->sampler_view_index[slot] =
            (uint8_t)resource;
         sx->sampler_binding_map->sampler_state_index[slot] =
            (uint8_t)sampler;
      }
   }

   if (target == TGSI_TEXTURE_BUFFER &&
       sx->resources[resource].mixed_return_component_mask ==
          TGSI_WRITEMASK_XYZW)
      sx->sampler_binding_map->mixed_return_mask |= 1u << slot;

   if (ureg_src_is_undef(sx->samplers[slot]))
      sx->samplers[slot] = ureg_DECL_sampler(sx->ureg, slot);
   declare_sampler_view_for_resource(sx, slot, resource, target);
   if (binding_slot)
      *binding_slot = slot;
   return sx->samplers[slot];
}

static struct ureg_src
compact_sampler_view_src(struct Shader_xlate *sx,
                         unsigned slot,
                         const struct Shader_src_operand *operand)
{
   assert(slot < SHADER_MAX_SAMPLERS);
   assert(!ureg_src_is_undef(sx->sv[slot]));

   struct ureg_src view = sx->sv[slot];
   if (operand) {
      view = ureg_swizzle(view,
                          operand->swizzle[0], operand->swizzle[1],
                          operand->swizzle[2], operand->swizzle[3]);
      if (operand->modifier != D3D10_SB_OPERAND_MODIFIER_NONE) {
         YTTRIUM_WARN("yttrium: shader TGSI translation failed "
                      "owner=d3d10umd-shader component=sampler-binding-map "
                      "reason=modified-resource-operand slot=%u "
                      "action=abort-translation\n",
                      slot);
         sx->translation_failed = true;
      }
   }
   return view;
}

static struct ureg_src
translate_old_tex_sampler_for_resource(struct Shader_xlate *sx, unsigned resource,
                                       const struct Shader_src_operand *sampler)
{
   struct ureg_src translated_sampler;

   assert(resource < SHADER_MAX_RESOURCES);

   if (sx->sampler_binding_map) {
      unsigned sampler_index;
      if (!translate_binding_operand_index(
             sx, sampler, D3D10_SB_OPERAND_TYPE_SAMPLER,
             SHADER_MAX_SAMPLERS, "sampler", &sampler_index)) {
         return declare_compact_sampler_for_resource(
            sx, resource, PIPE_MAX_SAMPLERS,
            (enum tgsi_texture_type)sx->resources[resource].target, NULL);
      }

      return declare_compact_sampler_for_resource(
         sx, resource, sampler_index,
         (enum tgsi_texture_type)sx->resources[resource].target, NULL);
   }

   if (resource >= SHADER_MAX_SAMPLERS) {
      const unsigned sampler_index =
         sampler->base.index_dim == 1 &&
         sampler->base.index[0].index_rep ==
            D3D10_SB_OPERAND_INDEX_IMMEDIATE32 ?
               sampler->base.index[0].imm : ~0u;
      YTTRIUM_WARN("yttrium: shader TGSI compatibility path "
                   "owner=d3d10umd-shader operation=legacy-texture-sample "
                   "resource=%u sampler=%u reason=resource-exceeds-legacy-"
                   "combined-sampler-range action=continue\n",
                   resource, sampler_index);
      translated_sampler = translate_src_operand(sx, sampler, OF_FLOAT);
   } else {
      if (ureg_src_is_undef(sx->samplers[resource])) {
         sx->samplers[resource] =
            ureg_DECL_sampler(sx->ureg, resource);
      }

      translated_sampler = sx->samplers[resource];
   }

   /* Legacy TGSI texture instructions carry only a SAMPLER source, so
    * tgsi_to_nir uses that source index for both the sampler and image type.
    * D3D keeps resource and sampler registers independent.  Mirror the
    * resource declaration at the sampler index to preserve its return type
    * when those registers differ (for example t127 sampled through s15).
    */
   if (translated_sampler.File == TGSI_FILE_SAMPLER &&
       translated_sampler.Index < SHADER_MAX_RESOURCES)
      declare_sampler_view_for_resource(
         sx, translated_sampler.Index, resource,
         sx->resources[resource].target);

   return translated_sampler;
}

static uint
translate_resource_dimension(D3D10_SB_RESOURCE_DIMENSION dim)
{
   switch (dim) {
   case D3D10_SB_RESOURCE_DIMENSION_UNKNOWN:
      return TGSI_TEXTURE_UNKNOWN;
   case D3D10_SB_RESOURCE_DIMENSION_BUFFER:
      return TGSI_TEXTURE_BUFFER;
   case D3D10_SB_RESOURCE_DIMENSION_TEXTURE1D:
      return TGSI_TEXTURE_1D;
   case D3D10_SB_RESOURCE_DIMENSION_TEXTURE2D:
      return TGSI_TEXTURE_2D;
   case D3D10_SB_RESOURCE_DIMENSION_TEXTURE2DMS:
      return TGSI_TEXTURE_2D_MSAA;
   case D3D10_SB_RESOURCE_DIMENSION_TEXTURE3D:
      return TGSI_TEXTURE_3D;
   case D3D10_SB_RESOURCE_DIMENSION_TEXTURECUBE:
      return TGSI_TEXTURE_CUBE;
   case D3D10_SB_RESOURCE_DIMENSION_TEXTURE1DARRAY:
      return TGSI_TEXTURE_1D_ARRAY;
   case D3D10_SB_RESOURCE_DIMENSION_TEXTURE2DARRAY:
      return TGSI_TEXTURE_2D_ARRAY;
   case D3D10_SB_RESOURCE_DIMENSION_TEXTURE2DMSARRAY:
      return TGSI_TEXTURE_2D_ARRAY_MSAA;
   case D3D10_SB_RESOURCE_DIMENSION_TEXTURECUBEARRAY:
      return TGSI_TEXTURE_CUBE_ARRAY;
   default:
      assert(0);
      return TGSI_TEXTURE_UNKNOWN;
   }
}

static uint
translate_shadow_texture_target(uint target)
{
   switch (target) {
   case TGSI_TEXTURE_1D:
      return TGSI_TEXTURE_SHADOW1D;
   case TGSI_TEXTURE_2D:
      return TGSI_TEXTURE_SHADOW2D;
   case TGSI_TEXTURE_1D_ARRAY:
      return TGSI_TEXTURE_SHADOW1D_ARRAY;
   case TGSI_TEXTURE_2D_ARRAY:
      return TGSI_TEXTURE_SHADOW2D_ARRAY;
   case TGSI_TEXTURE_CUBE:
      return TGSI_TEXTURE_SHADOWCUBE;
   case TGSI_TEXTURE_CUBE_ARRAY:
      return TGSI_TEXTURE_SHADOWCUBE_ARRAY;
   default:
      return TGSI_TEXTURE_UNKNOWN;
   }
}

static uint
texture_dim_from_tgsi_target(unsigned tgsi_target)
{
   switch (tgsi_target) {
   case TGSI_TEXTURE_BUFFER:
   case TGSI_TEXTURE_1D:
   case TGSI_TEXTURE_1D_ARRAY:
   case TGSI_TEXTURE_SHADOW1D:
   case TGSI_TEXTURE_SHADOW1D_ARRAY:
      return 1;
   case TGSI_TEXTURE_2D:
   case TGSI_TEXTURE_2D_MSAA:
   case TGSI_TEXTURE_CUBE:
   case TGSI_TEXTURE_2D_ARRAY:
   case TGSI_TEXTURE_2D_ARRAY_MSAA:
   case TGSI_TEXTURE_SHADOW2D:
   case TGSI_TEXTURE_SHADOW2D_ARRAY:
   case TGSI_TEXTURE_SHADOWCUBE:
   case TGSI_TEXTURE_SHADOWCUBE_ARRAY:
      return 2;
   case TGSI_TEXTURE_3D:
      return 3;
   case TGSI_TEXTURE_UNKNOWN:
   default:
      assert(0);
      return 1;
   }
}

static unsigned
shadow_ref_writemask(unsigned shadow_target)
{
   const int ref_src = tgsi_util_get_shadow_ref_src_index(shadow_target);

   return ref_src >= 0 && ref_src < 4 ? TGSI_WRITEMASK_X << ref_src : 0;
}

static bool
operand_is_scalar(const struct Shader_src_operand *operand)
{
   return operand->swizzle[0] == operand->swizzle[1] &&
          operand->swizzle[1] == operand->swizzle[2] &&
          operand->swizzle[2] == operand->swizzle[3];
}

static bool
resource_swizzle_is_identity(const struct Shader_src_operand *operand)
{
   return operand->swizzle[0] == D3D10_SB_4_COMPONENT_X &&
          operand->swizzle[1] == D3D10_SB_4_COMPONENT_Y &&
          operand->swizzle[2] == D3D10_SB_4_COMPONENT_Z &&
          operand->swizzle[3] == D3D10_SB_4_COMPONENT_W;
}

static struct ureg_dst
old_tex_dst_for_resource_swizzle(struct ureg_program *ureg,
                                 struct ureg_dst dst,
                                 const struct Shader_src_operand *resource,
                                 struct ureg_dst *tmp)
{
   *tmp = ureg_dst_undef();
   if (resource_swizzle_is_identity(resource))
      return dst;

   *tmp = ureg_DECL_temporary(ureg);
   return *tmp;
}

static void
old_tex_apply_resource_swizzle(struct ureg_program *ureg,
                               struct ureg_dst dst,
                               const struct Shader_src_operand *resource,
                               struct ureg_dst tmp)
{
   struct ureg_src src;

   if (ureg_dst_is_undef(tmp))
      return;

   src = ureg_swizzle(ureg_src(tmp),
                      resource->swizzle[0],
                      resource->swizzle[1],
                      resource->swizzle[2],
                      resource->swizzle[3]);
   ureg_MOV(ureg, dst, src);
   ureg_release_temporary(ureg, tmp);
}

static void
Shader_add_call(struct Shader_xlate *sx,
                unsigned d3d_label,
                unsigned tgsi_label_token)
{
   ASSERT(sx->num_calls < sx->max_calls);

   sx->calls[sx->num_calls].d3d_label = d3d_label;
   sx->calls[sx->num_calls].tgsi_label_token = tgsi_label_token;
   sx->num_calls++;
}

static void
Shader_add_label(struct Shader_xlate *sx,
                 unsigned d3d_label,
                 unsigned tgsi_insn_no)
{
   ASSERT(sx->num_labels < sx->max_labels);

   sx->labels[sx->num_labels].d3d_label = d3d_label;
   sx->labels[sx->num_labels].tgsi_insn_no = tgsi_insn_no;
   sx->num_labels++;
}


static void
sample_ureg_emit_target(struct ureg_program *ureg,
                        unsigned tgsi_opcode,
                        unsigned num_src,
                        struct Shader_opcode *opcode,
                        const struct ureg_src *dynamic_texel_offset,
                        struct ureg_dst dst,
                        struct ureg_src *src,
                        enum tgsi_texture_type target)
{
   unsigned num_offsets = 0;
   struct tgsi_texture_offset texoffsets;

   memset(&texoffsets, 0, sizeof texoffsets);

   if (dynamic_texel_offset) {
      struct ureg_dst offset_tmp = ureg_DECL_temporary(ureg);
      struct ureg_src offsetreg;

      ureg_MOV(ureg, offset_tmp, *dynamic_texel_offset);
      offsetreg = ureg_src(offset_tmp);
      num_offsets = 1;
      texoffsets.File = offsetreg.File;
      texoffsets.Index = offsetreg.Index;
      texoffsets.SwizzleX = offsetreg.SwizzleX;
      texoffsets.SwizzleY = offsetreg.SwizzleY;
      texoffsets.SwizzleZ = offsetreg.SwizzleZ;
   } else if (opcode->imm_texel_offset.u ||
       opcode->imm_texel_offset.v ||
       opcode->imm_texel_offset.w) {
      struct ureg_src offsetreg;
      num_offsets = 1;
      /* don't actually always need all 3 values */
      offsetreg = ureg_imm3i(ureg,
                             opcode->imm_texel_offset.u,
                             opcode->imm_texel_offset.v,
                             opcode->imm_texel_offset.w);
      texoffsets.File = offsetreg.File;
      texoffsets.Index = offsetreg.Index;
      texoffsets.SwizzleX = offsetreg.SwizzleX;
      texoffsets.SwizzleY = offsetreg.SwizzleY;
      texoffsets.SwizzleZ = offsetreg.SwizzleZ;
   }

   ureg_tex_insn(ureg,
                 tgsi_opcode,
                 &dst, 1,
                 target,
                 &texoffsets, num_offsets,
                 src, num_src);
}

static void
sample_ureg_emit(struct ureg_program *ureg,
                 unsigned tgsi_opcode,
                 unsigned num_src,
                 struct Shader_opcode *opcode,
                 struct ureg_dst dst,
                 struct ureg_src *src)
{
   sample_ureg_emit_target(ureg, tgsi_opcode, num_src, opcode, NULL, dst, src,
                           TGSI_TEXTURE_UNKNOWN);
}

static void
sample_i_buffer_dwords_ureg_emit(struct ureg_program *ureg,
                                 struct Shader_opcode *opcode,
                                 struct ureg_dst dst,
                                 struct ureg_src coord,
                                 struct ureg_src resource,
                                 enum tgsi_texture_type target)
{
   struct ureg_dst base = ureg_DECL_temporary(ureg);
   struct ureg_dst fetch_coord = ureg_DECL_temporary(ureg);
   struct ureg_dst fetch = ureg_DECL_temporary(ureg);

   ureg_MOV(ureg, base, coord);

   for (unsigned chan = 0; chan < 4; chan++) {
      const unsigned writemask = TGSI_WRITEMASK_X << chan;
      struct ureg_dst scalar_dst = ureg_writemask(dst, writemask);
      struct ureg_src src[2];

      if (scalar_dst.WriteMask == TGSI_WRITEMASK_NONE)
         continue;

      if (chan) {
         ureg_UADD(ureg, fetch_coord,
                   ureg_scalar(ureg_src(base), TGSI_SWIZZLE_X),
                   ureg_imm1u(ureg, chan));
      } else {
         ureg_MOV(ureg, fetch_coord,
                  ureg_scalar(ureg_src(base), TGSI_SWIZZLE_X));
      }

      src[0] = ureg_src(fetch_coord);
      src[1] = resource;
      sample_ureg_emit_target(ureg, TGSI_OPCODE_SAMPLE_I, 2, opcode, NULL,
                              ureg_writemask(fetch, TGSI_WRITEMASK_X),
                              src, target);
      ureg_MOV(ureg, scalar_dst, ureg_scalar(ureg_src(fetch), TGSI_SWIZZLE_X));
   }

   ureg_release_temporary(ureg, fetch);
   ureg_release_temporary(ureg, fetch_coord);
   ureg_release_temporary(ureg, base);
}

static void
image_buffer_load_dwords_ureg_emit(struct ureg_program *ureg,
                                   struct ureg_dst dst,
                                   struct ureg_src image,
                                   struct ureg_src coord,
                                   enum tgsi_texture_type target,
                                   enum pipe_format format,
                                   unsigned component_stride,
                                   const D3D10_SB_4_COMPONENT_NAME *swizzle)
{
   struct ureg_dst base = ureg_DECL_temporary(ureg);
   struct ureg_dst fetch_coord = ureg_DECL_temporary(ureg);
   struct ureg_dst fetch = ureg_DECL_temporary(ureg);

   ureg_MOV(ureg, base, coord);

   for (unsigned chan = 0; chan < 4; chan++) {
      const unsigned writemask = TGSI_WRITEMASK_X << chan;
      struct ureg_dst scalar_dst = ureg_writemask(dst, writemask);
      struct ureg_src src[2];

      if (scalar_dst.WriteMask == TGSI_WRITEMASK_NONE)
         continue;

      /* The destination mask selects registers to write, not DWORDs to read:
       * ld_structured r0.y, ..., u0.xxxx still loads source component X. */
      const unsigned component = swizzle ? swizzle[chan] : chan;
      if (component) {
         ureg_UADD(ureg, fetch_coord,
                   ureg_scalar(ureg_src(base), TGSI_SWIZZLE_X),
                   ureg_imm1u(ureg, component * component_stride));
      } else {
         ureg_MOV(ureg, fetch_coord,
                  ureg_scalar(ureg_src(base), TGSI_SWIZZLE_X));
      }

      src[0] = image;
      src[1] = ureg_src(fetch_coord);
      struct ureg_dst fetch_dst = ureg_writemask(fetch, TGSI_WRITEMASK_XY);
      ureg_memory_insn(ureg, TGSI_OPCODE_LOAD,
                       &fetch_dst, 1, src, ARRAY_SIZE(src), 0, target, format);
      ureg_MOV(ureg, scalar_dst, ureg_scalar(ureg_src(fetch), TGSI_SWIZZLE_X));
   }

   ureg_release_temporary(ureg, fetch);
   ureg_release_temporary(ureg, fetch_coord);
   ureg_release_temporary(ureg, base);
}

typedef void (*unary_ureg_func)(struct ureg_program *ureg, struct ureg_dst dst,
                                struct ureg_src src);
static void
expand_unary_to_scalarf(struct ureg_program *ureg, unary_ureg_func func,
                        struct Shader_xlate *sx, struct Shader_opcode *opcode)
{
   struct ureg_dst tmp = ureg_DECL_temporary(ureg);
   struct ureg_dst dst = translate_dst_operand(sx, &opcode->dst[0],
                                               opcode->saturate);
   struct ureg_src src = translate_src_operand(sx, &opcode->src[0], OF_FLOAT);
   struct ureg_dst scalar_dst;
   ureg_MOV(ureg, tmp, src);
   src = ureg_src(tmp);

   scalar_dst = ureg_writemask(dst, TGSI_WRITEMASK_X);
   if (scalar_dst.WriteMask != TGSI_WRITEMASK_NONE) {
      func(ureg, scalar_dst,
           ureg_scalar(src, TGSI_SWIZZLE_X));
   }
   scalar_dst = ureg_writemask(dst, TGSI_WRITEMASK_Y);
   if (scalar_dst.WriteMask != TGSI_WRITEMASK_NONE) {
      func(ureg, scalar_dst,
           ureg_scalar(src, TGSI_SWIZZLE_Y));
   }
   scalar_dst = ureg_writemask(dst, TGSI_WRITEMASK_Z);
   if (scalar_dst.WriteMask != TGSI_WRITEMASK_NONE) {
      func(ureg, scalar_dst,
           ureg_scalar(src, TGSI_SWIZZLE_Z));
   }
   scalar_dst = ureg_writemask(dst, TGSI_WRITEMASK_W);
   if (scalar_dst.WriteMask != TGSI_WRITEMASK_NONE) {
      func(ureg, scalar_dst,
           ureg_scalar(src, TGSI_SWIZZLE_W));
   }
   ureg_release_temporary(ureg, tmp);
}

static void
expand_firstbit_hi(struct ureg_program *ureg, struct Shader_xlate *sx,
                   struct Shader_opcode *opcode, bool is_signed)
{
   struct ureg_dst raw = ureg_DECL_temporary(ureg);
   struct ureg_dst adjusted = ureg_DECL_temporary(ureg);
   struct ureg_dst no_bits = ureg_DECL_temporary(ureg);
   struct ureg_src src = translate_src_operand(sx, &opcode->src[0],
                                               is_signed ? OF_INT : OF_UINT);
   struct ureg_dst dst = translate_dst_operand(sx, &opcode->dst[0],
                                               opcode->saturate);

   if (is_signed)
      ureg_IMSB(ureg, raw, src);
   else
      ureg_UMSB(ureg, raw, src);

   ureg_INEG(ureg, adjusted, ureg_src(raw));
   ureg_UADD(ureg, adjusted, ureg_src(adjusted), ureg_imm1u(ureg, 31));
   ureg_USEQ(ureg, no_bits, ureg_src(raw), ureg_imm1u(ureg, 0xffffffff));
   ureg_UCMP(ureg, dst, ureg_src(no_bits),
             ureg_imm1u(ureg, 0xffffffff), ureg_src(adjusted));

   ureg_release_temporary(ureg, raw);
   ureg_release_temporary(ureg, adjusted);
   ureg_release_temporary(ureg, no_bits);
}

static void
expand_f16tof32(struct ureg_program *ureg, struct Shader_xlate *sx,
                struct Shader_opcode *opcode)
{
   struct ureg_dst half = ureg_DECL_temporary(ureg);
   struct ureg_dst sign = ureg_DECL_temporary(ureg);
   struct ureg_dst exp = ureg_DECL_temporary(ureg);
   struct ureg_dst mant = ureg_DECL_temporary(ureg);
   struct ureg_dst bits = ureg_DECL_temporary(ureg);
   struct ureg_dst abs_bits = ureg_DECL_temporary(ureg);
   struct ureg_dst is_zero = ureg_DECL_temporary(ureg);
   struct ureg_dst dst = translate_dst_operand(sx, &opcode->dst[0],
                                               opcode->saturate);
   struct ureg_src src = translate_src_operand(sx, &opcode->src[0], OF_UINT);

   ureg_AND(ureg, half, src, ureg_imm1u(ureg, 0xffff));

   ureg_AND(ureg, sign, ureg_src(half), ureg_imm1u(ureg, 0x8000));
   ureg_SHL(ureg, sign, ureg_src(sign), ureg_imm1u(ureg, 16));

   ureg_AND(ureg, exp, ureg_src(half), ureg_imm1u(ureg, 0x7c00));
   ureg_UADD(ureg, exp, ureg_src(exp), ureg_imm1u(ureg, 0x1c000));
   ureg_SHL(ureg, exp, ureg_src(exp), ureg_imm1u(ureg, 13));

   ureg_AND(ureg, mant, ureg_src(half), ureg_imm1u(ureg, 0x03ff));
   ureg_SHL(ureg, mant, ureg_src(mant), ureg_imm1u(ureg, 13));

   ureg_OR(ureg, bits, ureg_src(sign), ureg_src(exp));
   ureg_OR(ureg, bits, ureg_src(bits), ureg_src(mant));

   ureg_AND(ureg, abs_bits, ureg_src(half), ureg_imm1u(ureg, 0x7fff));
   ureg_USEQ(ureg, is_zero, ureg_src(abs_bits), ureg_imm1u(ureg, 0));
   ureg_UCMP(ureg, dst, ureg_src(is_zero), ureg_imm1u(ureg, 0),
             ureg_src(bits));

   ureg_release_temporary(ureg, half);
   ureg_release_temporary(ureg, sign);
   ureg_release_temporary(ureg, exp);
   ureg_release_temporary(ureg, mant);
   ureg_release_temporary(ureg, bits);
   ureg_release_temporary(ureg, abs_bits);
   ureg_release_temporary(ureg, is_zero);
}

static void
expand_f32tof16(struct ureg_program *ureg, struct Shader_xlate *sx,
                struct Shader_opcode *opcode)
{
   struct ureg_dst raw = ureg_DECL_temporary(ureg);
   struct ureg_dst sign = ureg_DECL_temporary(ureg);
   struct ureg_dst exp = ureg_DECL_temporary(ureg);
   struct ureg_dst mant = ureg_DECL_temporary(ureg);
   struct ureg_dst bits = ureg_DECL_temporary(ureg);
   struct ureg_dst abs_bits = ureg_DECL_temporary(ureg);
   struct ureg_dst is_zero = ureg_DECL_temporary(ureg);
   struct ureg_dst dst = translate_dst_operand(sx, &opcode->dst[0],
                                               opcode->saturate);
   struct ureg_src src = translate_src_operand(sx, &opcode->src[0], OF_FLOAT);

   ureg_MOV(ureg, raw, src);

   ureg_USHR(ureg, sign, ureg_src(raw), ureg_imm1u(ureg, 16));
   ureg_AND(ureg, sign, ureg_src(sign), ureg_imm1u(ureg, 0x8000));

   ureg_USHR(ureg, exp, ureg_src(raw), ureg_imm1u(ureg, 23));
   ureg_AND(ureg, exp, ureg_src(exp), ureg_imm1u(ureg, 0xff));
   ureg_UADD(ureg, exp, ureg_src(exp), ureg_imm1u(ureg, 0xffffff90));
   ureg_SHL(ureg, exp, ureg_src(exp), ureg_imm1u(ureg, 10));

   ureg_USHR(ureg, mant, ureg_src(raw), ureg_imm1u(ureg, 13));
   ureg_AND(ureg, mant, ureg_src(mant), ureg_imm1u(ureg, 0x03ff));

   ureg_OR(ureg, bits, ureg_src(sign), ureg_src(exp));
   ureg_OR(ureg, bits, ureg_src(bits), ureg_src(mant));

   ureg_AND(ureg, abs_bits, ureg_src(raw), ureg_imm1u(ureg, 0x7fffffff));
   ureg_USEQ(ureg, is_zero, ureg_src(abs_bits), ureg_imm1u(ureg, 0));
   ureg_UCMP(ureg, dst, ureg_src(is_zero), ureg_imm1u(ureg, 0),
             ureg_src(bits));

   ureg_release_temporary(ureg, raw);
   ureg_release_temporary(ureg, sign);
   ureg_release_temporary(ureg, exp);
   ureg_release_temporary(ureg, mant);
   ureg_release_temporary(ureg, bits);
   ureg_release_temporary(ureg, abs_bits);
   ureg_release_temporary(ureg, is_zero);
}

static void
emit_raw_buffer_bufinfo(struct ureg_program *ureg, struct ureg_dst dst,
                        struct ureg_src query)
{
   struct ureg_dst byte_count = ureg_DECL_temporary(ureg);

   ureg_UMUL(ureg, byte_count, ureg_scalar(query, TGSI_SWIZZLE_X),
             ureg_imm1u(ureg, sizeof(uint32_t)));
   ureg_MOV(ureg, dst, ureg_src(byte_count));
   ureg_release_temporary(ureg, byte_count);
}

static void
emit_structured_buffer_bufinfo(struct ureg_program *ureg, struct ureg_dst dst,
                               struct ureg_src query, unsigned stride,
                               unsigned count_divisor)
{
   struct ureg_dst value = ureg_DECL_temporary(ureg);

   ureg_UDIV(ureg, ureg_writemask(value, TGSI_WRITEMASK_X),
             ureg_scalar(query, TGSI_SWIZZLE_X),
             ureg_imm1u(ureg, MAX2(count_divisor, 1)));
   ureg_MOV(ureg, ureg_writemask(value, TGSI_WRITEMASK_Y),
            ureg_imm1u(ureg, stride));
   ureg_MOV(ureg, ureg_writemask(value, TGSI_WRITEMASK_Z),
            ureg_imm1u(ureg, 0));
   ureg_MOV(ureg, ureg_writemask(value, TGSI_WRITEMASK_W),
            ureg_imm1u(ureg, 1));
   ureg_MOV(ureg, dst, ureg_src(value));
   ureg_release_temporary(ureg, value);
}

static void
declare_driver_constants(struct Shader_xlate *sx)
{
   if (sx->bufinfo_constants_declared)
      return;

   ureg_DECL_constant2D(sx->ureg, 0,
                        D3D10UMD_DRIVER_CB_RECORD_COUNT - 1,
                        D3D10UMD_DRIVER_BUFINFO_CB_SLOT);
   sx->bufinfo_constants_declared = true;
}

static void
emit_dynamic_buffer_bufinfo(struct Shader_xlate *sx, struct ureg_dst dst,
                            unsigned record)
{
   declare_driver_constants(sx);

   ureg_MOV(sx->ureg, dst,
            ureg_src_dimension(ureg_src_register(TGSI_FILE_CONSTANT, record),
                               D3D10UMD_DRIVER_BUFINFO_CB_SLOT));
}

static void
emit_rasterizer_sample_info(struct Shader_xlate *sx, struct ureg_dst dst,
                            bool as_float)
{
   struct ureg_src sample_count;

   declare_driver_constants(sx);
   sample_count =
      ureg_src_dimension(
         ureg_src_register(TGSI_FILE_CONSTANT,
                           D3D10UMD_DRIVER_SAMPLE_INFO_RECORD),
         D3D10UMD_DRIVER_BUFINFO_CB_SLOT);
   sample_count = ureg_scalar(sample_count, TGSI_SWIZZLE_X);

   if (as_float)
      ureg_I2F(sx->ureg, dst, sample_count);
   else
      ureg_MOV(sx->ureg, dst, sample_count);
}

#define D3D10UMD_BUFINFO_STRIDE_REWRITE_WINDOW 5

static void
track_dynamic_structured_bufinfo_stride(struct Shader_xlate *sx,
                                        const struct Shader_dst_operand *dst,
                                        unsigned record,
                                        unsigned declared_stride)
{
   unsigned dst_index;

   sx->bufinfo_stride.valid = false;
   if (!declared_stride ||
       shader_dst_writemask(dst) != TGSI_WRITEMASK_X ||
       !shader_dst_is_simple_temp(dst, &dst_index))
      return;

   sx->bufinfo_stride.valid = true;
   sx->bufinfo_stride.dst_index = dst_index;
   sx->bufinfo_stride.declared_stride = declared_stride;
   sx->bufinfo_stride.record = record;
   sx->bufinfo_stride.remaining = D3D10UMD_BUFINFO_STRIDE_REWRITE_WINDOW;
}

static void
advance_dynamic_structured_bufinfo_stride(struct Shader_xlate *sx)
{
   if (!sx->bufinfo_stride.valid)
      return;

   if (sx->bufinfo_stride.remaining)
      sx->bufinfo_stride.remaining--;
   if (!sx->bufinfo_stride.remaining)
      sx->bufinfo_stride.valid = false;
}

static bool
try_emit_dynamic_structured_bufinfo_stride(struct Shader_xlate *sx,
                                           const struct Shader_opcode *opcode)
{
   unsigned dst_index, src_component;
   struct ureg_src info;

   if (!sx->bufinfo_stride.valid ||
       opcode->type != D3D10_SB_OPCODE_MOV ||
       opcode->num_dst != 1 ||
       opcode->num_src != 1 ||
       shader_dst_writemask(&opcode->dst[0]) != TGSI_WRITEMASK_Y ||
       !shader_dst_is_simple_temp(&opcode->dst[0], &dst_index) ||
       dst_index != sx->bufinfo_stride.dst_index ||
       opcode->src[0].base.type != D3D10_SB_OPERAND_TYPE_IMMEDIATE32 ||
       opcode->src[0].modifier != D3D10_SB_OPERAND_MODIFIER_NONE)
      return false;

   src_component = opcode->src[0].swizzle[TGSI_SWIZZLE_Y];
   if (src_component > TGSI_SWIZZLE_W ||
       opcode->src[0].imm[src_component].u32 !=
          sx->bufinfo_stride.declared_stride)
      return false;

   info = ureg_src_dimension(
      ureg_src_register(TGSI_FILE_CONSTANT, sx->bufinfo_stride.record),
      D3D10UMD_DRIVER_BUFINFO_CB_SLOT);
   ureg_MOV(sx->ureg,
            translate_dst_operand(sx, &opcode->dst[0], opcode->saturate),
            ureg_scalar(info, TGSI_SWIZZLE_Y));
   sx->bufinfo_stride.valid = false;
   return true;
}

static void
emit_tessellation_properties(
   struct ureg_program *ureg,
   const struct Shader_tessellation_properties *properties)
{
   switch (properties->domain) {
   case 1:
      ureg_property(ureg, TGSI_PROPERTY_TES_PRIM_MODE, MESA_PRIM_LINES);
      break;
   case 2:
      ureg_property(ureg, TGSI_PROPERTY_TES_PRIM_MODE, MESA_PRIM_TRIANGLES);
      break;
   case 3:
      ureg_property(ureg, TGSI_PROPERTY_TES_PRIM_MODE, MESA_PRIM_QUADS);
      break;
   default:
      break;
   }

   switch (properties->partitioning) {
   case 1:
      ureg_property(ureg, TGSI_PROPERTY_TES_SPACING,
                    PIPE_TESS_SPACING_EQUAL);
      break;
   case 2:
      /*
       * Vulkan has no power-of-two spacing mode.  The TCS rounds D3D pow2
       * factors explicitly, after which equal spacing is equivalent.
       */
      ureg_property(ureg, TGSI_PROPERTY_TES_SPACING,
                    PIPE_TESS_SPACING_EQUAL);
      break;
   case 3:
      ureg_property(ureg, TGSI_PROPERTY_TES_SPACING,
                    PIPE_TESS_SPACING_FRACTIONAL_ODD);
      break;
   case 4:
      ureg_property(ureg, TGSI_PROPERTY_TES_SPACING,
                    PIPE_TESS_SPACING_FRACTIONAL_EVEN);
      break;
   default:
      break;
   }

   switch (properties->output_primitive) {
   case 1:
      ureg_property(ureg, TGSI_PROPERTY_TES_POINT_MODE, 1);
      break;
   case 3:
      ureg_property(ureg, TGSI_PROPERTY_TES_VERTEX_ORDER_CW, 1);
      break;
   case 4:
      ureg_property(ureg, TGSI_PROPERTY_TES_VERTEX_ORDER_CW, 0);
      break;
   default:
      break;
   }
}

/*
 * The stage output signature is the union of the registers used by all
 * shaders which share that signature.  A particular shader may therefore
 * omit declarations which are nevertheless named by a tokenless
 * geometry-shader stream-output declaration.
 *
 * Yttrium translates D3D output registers to packed TGSI output registers,
 * so it cannot ignore that union the way hardware with an identity register
 * layout can.  Declare every generic signature output before translating the
 * bytecode.  This gives all shaders sharing the signature the same packed
 * layout, and leaves signature-only values undefined as required.  System
 * outputs are deliberately left to their real bytecode declarations because
 * several of them have stage-specific side effects or packing rules.
 */
static bool
dcl_stage_signature_generic_outputs(
   struct ureg_program *ureg,
   unsigned *output_mapping,
   const D3D10DDIARG_STAGE_IO_SIGNATURES *signatures)
{
   if (!signatures || !output_mapping)
      return true;

   if (signatures->NumOutputSignatureEntries &&
       !signatures->pOutputSignature) {
      YTTRIUM_WARN("yttrium: shader TGSI translation failed "
                   "owner=d3d10umd-shader component=stage-signature "
                   "reason=null-output-signature action=abort-translation\n");
      return false;
   }

   for (unsigned i = 0; i < signatures->NumOutputSignatureEntries; ++i) {
      const D3D10DDIARG_SIGNATURE_ENTRY *entry =
         &signatures->pOutputSignature[i];
      const unsigned mask = entry->Mask;

      if (entry->Register >= SHADER_MAX_OUTPUTS ||
          !mask || (mask & ~TGSI_WRITEMASK_XYZW)) {
         YTTRIUM_WARN("yttrium: shader TGSI translation failed "
                      "owner=d3d10umd-shader component=stage-signature "
                      "reason=invalid-generic-output entry=%u register=%u "
                      "mask=0x%x action=abort-translation\n",
                      i, entry->Register, mask);
         return false;
      }

      if (entry->SystemValue != D3D10_SB_NAME_UNDEFINED)
         continue;

      const struct ureg_dst output =
         ureg_DECL_output_masked(ureg, TGSI_SEMANTIC_GENERIC,
                                 entry->Register, mask, 0, 1);
      const unsigned old_mapping = output_mapping[entry->Register];
      if (old_mapping != ~0u && old_mapping != output.Index) {
         YTTRIUM_WARN("yttrium: shader TGSI translation failed "
                      "owner=d3d10umd-shader component=stage-signature "
                      "reason=ambiguous-generic-output entry=%u register=%u "
                      "old_tgsi=%u new_tgsi=%u action=abort-translation\n",
                      i, entry->Register, old_mapping, output.Index);
         return false;
      }
      output_mapping[entry->Register] = output.Index;
   }

   return true;
}

/* Hidden counters share Gallium's image namespace with application UAVs.
 * Allocate only unused declarations, including for sparse D3D11.1 UAV slots.
 * Never alias an application's image or silently drop an overflowing counter.
 */
static bool
find_counter_image_slots(const unsigned *code, uint8_t *slots)
{
   struct Shader_parser parser;
   struct Shader_opcode opcode;
   uint64_t declared = 0, counters = 0;
   bool valid = true;

   memset(slots, 0, SHADER_MAX_IMAGES * sizeof(*slots));
   Shader_parse_init(&parser, code);
   while (Shader_parse_opcode(&parser, &opcode)) {
      unsigned operand = 0;
      bool declaration = false, counter = false;
      switch (opcode.type) {
      case D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_TYPED:
      case DX10_SM5_OPCODE_DCL_UAV_RAW:
      case DX10_SM5_OPCODE_DCL_UAV_STRUCTURED:
         declaration = true;
         break;
      case DX10_SM5_OPCODE_IMM_ATOMIC_ALLOC:
      case DX10_SM5_OPCODE_IMM_ATOMIC_CONSUME:
         operand = 1;
         counter = true;
         break;
      default:
         break;
      }
      if (declaration || counter) {
         const struct Shader_operand *image = &opcode.dst[operand].base;
         if (image->index_dim != 1 ||
             image->index[0].index_rep != D3D10_SB_OPERAND_INDEX_IMMEDIATE32 ||
             image->index[0].imm >= SHADER_MAX_IMAGES) {
            valid = false;
         } else if (declaration) {
            declared |= UINT64_C(1) << image->index[0].imm;
         } else {
            counters |= UINT64_C(1) << image->index[0].imm;
         }
      }
      Shader_opcode_free(&opcode);
   }
   if (parser.failed || !valid || (counters & ~declared))
      return false;

   /* The native device enables fragmentStoresAndAtomics, but does not yet
    * enable vertexPipelineStoresAndAtomics. Never emit unsupported pre-raster
    * counter atomics merely because D3D11.1 allows those UAV declarations. */
   if (counters && parser.header.type != D3D10_SB_PIXEL_SHADER &&
       parser.header.type != DX10_SM5_COMPUTE_SHADER) {
      YTTRIUM_WARN("yttrium: shader translation rejected owner=d3d10umd "
                   "reason=hidden_counter_stage_unsupported stage=%u\n",
                   parser.header.type);
      return false;
   }

   for (unsigned slot = 0; slot < SHADER_MAX_IMAGES; ++slot) {
      if (!(counters & (UINT64_C(1) << slot)))
         continue;
      unsigned hidden;
      for (hidden = 0; hidden < SHADER_MAX_IMAGES; ++hidden)
         if (!(declared & (UINT64_C(1) << hidden)))
            break;
      if (hidden == SHADER_MAX_IMAGES) {
         YTTRIUM_WARN("yttrium: shader translation rejected owner=d3d10umd "
                      "reason=hidden_counter_image_slots_exhausted\n");
         return false;
      }
      slots[slot] = hidden + 1;
      declared |= UINT64_C(1) << hidden;
   }
   return true;
}

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
                      uint8_t *counter_image_slots)
{
   struct Shader_xlate *sx;
   struct Shader_parser parser;
   struct ureg_program *ureg = NULL;
   struct Shader_opcode opcode;
   const struct tgsi_token *tokens = NULL;
   uint nr_tokens;
   bool shader_dumped = false;
   bool inside_sub = false;
   uint i, j;

   /* DWM can translate shaders on threads with only a 128 KiB stack.
    * Keep the large scratch context off the caller's stack and private to
    * this translation, including concurrent or reentrant shader creation.
    */
   sx = CALLOC_STRUCT(Shader_xlate);
   if (!sx) {
      YTTRIUM_WARN("yttrium: shader translation failed owner=d3d10umd-shader "
                   "reason=context-allocation-failed action=abort-translation\n");
      return NULL;
   }
   sx->tessellation_signatures = tessellation_signatures;
   sx->sampler_binding_map = sampler_binding_map;
   if (sampler_binding_map)
      memset(sampler_binding_map, 0, sizeof(*sampler_binding_map));
   if (shared_memory_size)
      *shared_memory_size = 0;
   if (counter_image_slots)
      memset(counter_image_slots, 0, sizeof(sx->counter_image_slots));

   if (!find_counter_image_slots(code, sx->counter_image_slots)) {
      YTTRIUM_WARN("yttrium: shader translation rejected owner=d3d10umd "
                   "reason=invalid_hidden_counter_image_map\n");
      goto fail;
   }

   Shader_parse_init(&parser, code);
   if (parser.failed)
      goto fail;

   if (st_debug & ST_DEBUG_TGSI) {
      dx10_shader_dump_tokens(code);
      shader_dumped = true;
   }

   sx->max_calls = 64;
   sx->calls = (struct Shader_call *)MALLOC(sx->max_calls *
                                           sizeof(struct Shader_call));
   sx->num_calls = 0;

   sx->max_labels = 64;
   sx->labels = (struct Shader_label *)MALLOC(sx->max_labels *
                                             sizeof(struct Shader_call));
   sx->num_labels = 0;



   /* Header. */
   switch (parser.header.type) {
   case D3D10_SB_PIXEL_SHADER:
      ureg = ureg_create(MESA_SHADER_FRAGMENT);
      break;
   case D3D10_SB_VERTEX_SHADER:
      ureg = ureg_create(MESA_SHADER_VERTEX);
      break;
   case D3D10_SB_GEOMETRY_SHADER:
      ureg = ureg_create(MESA_SHADER_GEOMETRY);
      break;
   case DX11_SM5_HULL_SHADER:
      ureg = ureg_create(MESA_SHADER_TESS_CTRL);
      sx->is_tcs = true;
      break;
   case DX11_SM5_DOMAIN_SHADER:
      ureg = ureg_create(MESA_SHADER_TESS_EVAL);
      break;
   case DX10_SM5_COMPUTE_SHADER:
      ureg = ureg_create(MESA_SHADER_COMPUTE);
      break;
   default:
      UNREACHABLE("unsupported D3D10_SB_SHADER\n");
   }

   assert(ureg);
   if (parser.header.type == DX11_SM5_HULL_SHADER ||
       parser.header.type == DX11_SM5_DOMAIN_SHADER)
      ureg_set_any_inout_decl_range(ureg, true);
   sx->ureg = ureg;
   for (unsigned slot = 0; slot < SHADER_MAX_IMAGES; ++slot) {
      if (!sx->counter_image_slots[slot])
         continue;
      sx->counter_images[slot] = ureg_DECL_image(
         ureg, sx->counter_image_slots[slot] - 1, TGSI_TEXTURE_2D,
         PIPE_FORMAT_R32_UINT, true, false);
   }

   if ((parser.header.type == D3D10_SB_VERTEX_SHADER ||
        parser.header.type == D3D10_SB_GEOMETRY_SHADER) &&
       !dcl_stage_signature_generic_outputs(ureg, output_mapping,
                                            stage_signatures))
      goto fail;

   for(uint i = 0; i < SHADER_MAX_ADDRS; i++) {
      sx->addrs[i] = ureg_DECL_address(sx->ureg);
   }
   sx->addr_cur = 0;

   while (Shader_parse_opcode(&parser, &opcode)) {
      const struct dx10_opcode_xlate *ox;

      assert(opcode.type < D3D10_SB_NUM_OPCODES);
      ox = &opcode_xlate[opcode.type];

      prepare_tcs_indexed_output_stores(sx, &opcode);
      if (sx->translation_failed) {
         Shader_opcode_free(&opcode);
         goto fail;
      }

      switch (opcode.type) {
      case DX11_SM5_OPCODE_HS_DECLS:
         break;

      case DX11_SM5_OPCODE_HS_CONTROL_POINT_PHASE:
         tcs_end_patch_phase(sx, ureg);
         memset(sx->tcs_tess_factor_writemask, 0,
                sizeof(sx->tcs_tess_factor_writemask));
         memset(sx->tcs_tess_factor_range_end, 0,
                sizeof(sx->tcs_tess_factor_range_end));
         sx->tcs_control_point_phase_seen = true;
         break;

      case DX11_SM5_OPCODE_HS_FORK_PHASE:
      case DX11_SM5_OPCODE_HS_JOIN_PHASE:
         tcs_end_patch_phase(sx, ureg);
         memset(sx->tcs_tess_factor_writemask, 0,
                sizeof(sx->tcs_tess_factor_writemask));
         memset(sx->tcs_tess_factor_range_end, 0,
                sizeof(sx->tcs_tess_factor_range_end));
         if (sx->tcs_control_point_phase_seen &&
             !sx->tcs_control_point_barrier_emitted) {
            ureg_BARRIER(ureg);
            sx->tcs_control_point_barrier_emitted = true;
         }
         if (!sx->tcs_control_point_phase_seen &&
             !sx->tcs_implicit_control_point_passthrough) {
            tcs_emit_implicit_control_point_passthrough(sx, ureg);
            sx->tcs_implicit_control_point_passthrough = true;
         }
         /*
          * A fork/join phase without an instance-count declaration has one
          * instance.  Starting that default here also marks its declarations
          * as per-patch.  An explicit declaration immediately replaces the
          * empty default phase before any executable instruction.
          */
         tcs_begin_patch_phase(sx, ureg, 1);
         break;

      case DX11_SM5_OPCODE_DCL_HS_MAX_TESSFACTOR:
         ureg_property(ureg, TGSI_PROPERTY_TCS_TESS_FACTOR_MAX,
                       opcode.specific.dcl_hs_max_tessfactor_bits);
         break;

      case DX11_SM5_OPCODE_DCL_HS_FORK_PHASE_INSTANCE_COUNT:
      case DX11_SM5_OPCODE_DCL_HS_JOIN_PHASE_INSTANCE_COUNT:
         tcs_begin_patch_phase(sx, ureg,
                               opcode.specific.dcl_hs_phase_instance_count);
         break;

      case DX11_SM5_OPCODE_DCL_INPUT_CONTROL_POINT_COUNT:
         declare_vertices_in(
            sx, opcode.specific.dcl_input_control_point_count);
         break;

      case DX11_SM5_OPCODE_DCL_OUTPUT_CONTROL_POINT_COUNT:
         sx->tcs_vertices_out =
            opcode.specific.dcl_output_control_point_count;
         ureg_property(ureg, TGSI_PROPERTY_TCS_VERTICES_OUT,
                       sx->tcs_vertices_out);
         break;

      case DX11_SM5_OPCODE_DCL_TESS_DOMAIN:
         sx->tessellation_properties.domain =
            opcode.specific.dcl_tess_domain;
         break;

      case DX11_SM5_OPCODE_DCL_TESS_PARTITIONING:
         sx->tessellation_properties.partitioning =
            opcode.specific.dcl_tess_partitioning;
         if (opcode.specific.dcl_tess_partitioning == 2) {
            ureg_property(ureg,
                          TGSI_PROPERTY_TCS_TESS_FACTOR_ROUND_TO_POW2, 1);
         }
         break;

      case DX11_SM5_OPCODE_DCL_TESS_OUTPUT_PRIMITIVE:
         sx->tessellation_properties.output_primitive =
            opcode.specific.dcl_tess_output_primitive;
         break;

      case DX10_SM5_OPCODE_DCL_THREAD_GROUP:
         assert(parser.header.type == DX10_SM5_COMPUTE_SHADER);
         sx->cs_thread_group_size[0] = opcode.specific.dcl_thread_group.x;
         sx->cs_thread_group_size[1] = opcode.specific.dcl_thread_group.y;
         sx->cs_thread_group_size[2] = opcode.specific.dcl_thread_group.z;
         if (thread_group_size) {
            thread_group_size[0] = opcode.specific.dcl_thread_group.x;
            thread_group_size[1] = opcode.specific.dcl_thread_group.y;
            thread_group_size[2] = opcode.specific.dcl_thread_group.z;
         }
         ureg_property(ureg, TGSI_PROPERTY_CS_FIXED_BLOCK_WIDTH,
                       opcode.specific.dcl_thread_group.x);
         ureg_property(ureg, TGSI_PROPERTY_CS_FIXED_BLOCK_HEIGHT,
                       opcode.specific.dcl_thread_group.y);
         ureg_property(ureg, TGSI_PROPERTY_CS_FIXED_BLOCK_DEPTH,
                       opcode.specific.dcl_thread_group.z);
         break;
      case DX10_SM5_OPCODE_SYNC:
         ureg_BARRIER(ureg);
         break;
      case DX10_SM5_OPCODE_DCL_TGSM_RAW:
      case DX10_SM5_OPCODE_DCL_TGSM_STRUCTURED: {
         unsigned slot = 0;
         assert(parser.header.type == DX10_SM5_COMPUTE_SHADER);
         if (tgsm_operand_slot(&opcode.dst[0].base, &slot)) {
            sx->tgsm[slot].byte_offset = sx->shared_memory_size;
            sx->tgsm[slot].byte_count = opcode.specific.dcl_tgsm.byte_count;
            sx->tgsm[slot].structured_stride =
               opcode.specific.dcl_tgsm.structured_stride;
            sx->shared_memory_size += opcode.specific.dcl_tgsm.byte_count;
            get_tgsm_memory(sx);
         }
         break;
      }
      case D3D10_SB_OPCODE_EXP:
         expand_unary_to_scalarf(ureg, ureg_EX2, sx, &opcode);
         break;
      case D3D10_SB_OPCODE_SQRT:
         expand_unary_to_scalarf(ureg, ureg_SQRT, sx, &opcode);
         break;
      case D3D10_SB_OPCODE_RSQ:
         expand_unary_to_scalarf(ureg, ureg_RSQ, sx, &opcode);
         break;
      case DX10_SM5_OPCODE_RCP:
         expand_unary_to_scalarf(ureg, ureg_RCP, sx, &opcode);
         break;
      case D3D10_SB_OPCODE_LOG:
         expand_unary_to_scalarf(ureg, ureg_LG2, sx, &opcode);
         break;
      case DX10_SM5_OPCODE_F16TOF32:
         expand_f16tof32(ureg, sx, &opcode);
         break;
      case DX10_SM5_OPCODE_F32TOF16:
         expand_f32tof16(ureg, sx, &opcode);
         break;
      case DX10_SM5_OPCODE_FIRSTBIT_HI:
         expand_firstbit_hi(ureg, sx, &opcode, false);
         break;
      case DX10_SM5_OPCODE_FIRSTBIT_SHI:
         expand_firstbit_hi(ureg, sx, &opcode, true);
         break;
      case D3D10_SB_OPCODE_IMUL:
         if (opcode.dst[0].base.type != D3D10_SB_OPERAND_TYPE_NULL) {
            ureg_IMUL_HI(ureg,
                        translate_dst_operand(sx, &opcode.dst[0], opcode.saturate),
                        translate_src_operand(sx, &opcode.src[0], OF_INT),
                        translate_src_operand(sx, &opcode.src[1], OF_INT));
         }

         if (opcode.dst[1].base.type != D3D10_SB_OPERAND_TYPE_NULL) {
            ureg_UMUL(ureg,
                      translate_dst_operand(sx, &opcode.dst[1], opcode.saturate),
                      translate_src_operand(sx, &opcode.src[0], OF_INT),
                      translate_src_operand(sx, &opcode.src[1], OF_INT));
         }

         break;

      case D3D10_SB_OPCODE_FTOI: {
         /* XXX: tgsi (and just about everybody else, c, opencl, glsl) has
          * out-of-range (and NaN) values undefined for f2i/f2u, but d3d10
          * requires clamping to min and max representable value (as well as 0
          * for NaNs) (this applies to both ftoi and ftou). At least the online
          * docs state that - this is consistent with generic d3d10 conversion
          * rules.
          * For FTOI, we cheat a bit here - in particular depending on noone
          * caring about NaNs, and depending on the (undefined!) behavior of
          * F2I returning 0x80000000 for too negative values (which works with
          * x86 sse). Hence only need to clamp too positive values.
          * Note that it is impossible to clamp using a float, since 2^31 - 1
          * is not exactly representable with a float.
          */
         struct ureg_dst too_large = ureg_DECL_temporary(ureg);
         struct ureg_dst tmp = ureg_DECL_temporary(ureg);
         ureg_FSGE(ureg, too_large,
                   translate_src_operand(sx, &opcode.src[0], OF_FLOAT),
                   ureg_imm1f(ureg, 2147483648.0f));
         ureg_F2I(ureg, tmp,
                  translate_src_operand(sx, &opcode.src[0], OF_FLOAT));
         ureg_UCMP(ureg,
                   translate_dst_operand(sx, &opcode.dst[0], opcode.saturate),
                   ureg_src(too_large),
                   ureg_imm1i(ureg, 0x7fffffff),
                   ureg_src(tmp));
         ureg_release_temporary(ureg, too_large);
         ureg_release_temporary(ureg, tmp);
      }
         break;

      case D3D10_SB_OPCODE_FTOU: {
         /* For ftou, we need to do both clamps, which as a bonus also
          * gets us correct NaN behavior.
          * Note that it is impossible to clamp using a float against the upper
          * limit, since 2^32 - 1 is not exactly representable with a float,
          * but the clamp against 0.0 certainly works just fine.
          */
         struct ureg_dst too_large = ureg_DECL_temporary(ureg);
         struct ureg_dst tmp = ureg_DECL_temporary(ureg);
         ureg_FSGE(ureg, too_large,
                   translate_src_operand(sx, &opcode.src[0], OF_FLOAT),
                   ureg_imm1f(ureg, 4294967296.0f));
         /* clamp negative values + NaN to zero.
          * (Could be done slightly more efficient in llvmpipe due to
          * MAX NaN behavior handling.)
          */
         ureg_MAX(ureg, tmp,
                  ureg_imm1f(ureg, 0.0f),
                  translate_src_operand(sx, &opcode.src[0], OF_FLOAT));
         ureg_F2U(ureg, tmp,
                  ureg_src(tmp));
         ureg_UCMP(ureg,
                   translate_dst_operand(sx, &opcode.dst[0], opcode.saturate),
                   ureg_src(too_large),
                   ureg_imm1u(ureg, 0xffffffff),
                   ureg_src(tmp));
         ureg_release_temporary(ureg, too_large);
         ureg_release_temporary(ureg, tmp);
      }
         break;

      case D3D10_SB_OPCODE_LD_MS: {
         unsigned resource = opcode.src[1].base.index[0].imm;
         struct ureg_src sampler;
         struct ureg_dst coord = ureg_DECL_temporary(ureg);
         struct ureg_dst dst =
            translate_dst_operand(sx, &opcode.dst[0], opcode.saturate);
         struct ureg_dst tmp;
         struct ureg_dst tex_dst;

         assert(opcode.src[1].base.index_dim == 1);
         assert(resource < SHADER_MAX_RESOURCES);

         if (sx->sampler_binding_map) {
            if (!translate_binding_operand_index(
                   sx, &opcode.src[1], D3D10_SB_OPERAND_TYPE_RESOURCE,
                   SHADER_MAX_RESOURCES, "resource", &resource))
               break;
            sampler = declare_compact_sampler_for_resource(
               sx, resource, PIPE_MAX_SAMPLERS,
               (enum tgsi_texture_type)sx->resources[resource].target, NULL);
         } else {
            if (ureg_src_is_undef(sx->samplers[resource]))
               sx->samplers[resource] = ureg_DECL_sampler(ureg, resource);
            declare_sampler_view_for_resource(
               sx, resource, resource, sx->resources[resource].target);
            sampler = sx->samplers[resource];
         }

         ureg_MOV(ureg, coord,
                  translate_src_operand(sx, &opcode.src[0], OF_INT));
         ureg_MOV(ureg, ureg_writemask(coord, TGSI_WRITEMASK_W),
                  ureg_scalar(translate_src_operand(sx, &opcode.src[2],
                                                    OF_INT),
                              TGSI_SWIZZLE_X));
         tex_dst = old_tex_dst_for_resource_swizzle(ureg, dst,
                                                    &opcode.src[1], &tmp);
         ureg_TXF(ureg,
                  tex_dst,
                  sx->resources[resource].target,
                  ureg_src(coord),
                  sampler);
         old_tex_apply_resource_swizzle(ureg, dst, &opcode.src[1], tmp);
         ureg_release_temporary(ureg, coord);
      }
         break;
      case D3D10_SB_OPCODE_LD:
         if (sx->sampler_binding_map || use_old_tex_ops ||
             (st_debug & ST_DEBUG_OLD_TEX_OPS)) {
            unsigned resource = opcode.src[1].base.index[0].imm;
            struct ureg_src coord;
            struct ureg_src sampler;
            struct ureg_dst dst =
               translate_dst_operand(sx, &opcode.dst[0], opcode.saturate);
            struct ureg_dst tmp;
            struct ureg_dst tex_dst;
            assert(opcode.src[1].base.index_dim == 1);
            assert(opcode.src[1].base.index[0].imm < SHADER_MAX_RESOURCES);

            if (sx->sampler_binding_map) {
               if (!translate_binding_operand_index(
                      sx, &opcode.src[1], D3D10_SB_OPERAND_TYPE_RESOURCE,
                      SHADER_MAX_RESOURCES, "resource", &resource))
                  break;
               sampler = declare_compact_sampler_for_resource(
                  sx, resource, PIPE_MAX_SAMPLERS,
                  (enum tgsi_texture_type)sx->resources[resource].target,
                  NULL);
            } else {
               if (ureg_src_is_undef(sx->samplers[resource])) {
                  sx->samplers[resource] =
                     ureg_DECL_sampler(ureg, resource);
               }
               declare_sampler_view_for_resource(
                  sx, resource, resource, sx->resources[resource].target);
               sampler = sx->samplers[resource];
            }

            coord = translate_src_operand(sx, &opcode.src[0], OF_INT);
            if (opcode.type == D3D10_SB_OPCODE_LD)
               coord = translate_load_src_for_txf(sx->resources[resource].target, coord);

            tex_dst = old_tex_dst_for_resource_swizzle(ureg, dst,
                                                       &opcode.src[1], &tmp);
            ureg_TXF(ureg,
                     tex_dst,
                     sx->resources[resource].target,
                     coord,
                     sampler);
            old_tex_apply_resource_swizzle(ureg, dst, &opcode.src[1], tmp);
         }
         else {
            unsigned resource = opcode.src[1].base.index[0].imm;
            struct ureg_src srcreg[2];

            assert(opcode.src[1].base.index_dim == 1);
            assert(resource < SHADER_MAX_RESOURCES);

            srcreg[0] = translate_src_operand(sx, &opcode.src[0], OF_INT);
            srcreg[1] = translate_src_operand(sx, &opcode.src[1], OF_INT);

            sample_ureg_emit_target(ureg, TGSI_OPCODE_SAMPLE_I, 2, &opcode,
                                    NULL,
                                    translate_dst_operand(sx, &opcode.dst[0],
                                                          opcode.saturate),
                                    srcreg, sx->resources[resource].target);
         }
         break;

      case DX10_SM5_OPCODE_LD_RAW: {
         struct ureg_dst coord = ureg_DECL_temporary(ureg);
         struct ureg_src srcreg[2];

         if (opcode.src[1].base.type ==
             DX10_SM5_OPERAND_TYPE_THREAD_GROUP_SHARED_MEMORY) {
            struct ureg_src memory = get_tgsm_memory(sx);
            struct ureg_dst dst =
               translate_dst_operand(sx, &opcode.dst[0], opcode.saturate);
            struct ureg_dst fetch = ureg_DECL_temporary(ureg);
            struct ureg_dst byte_offset = ureg_dst_undef();
            struct ureg_dst fetch_offset = ureg_dst_undef();
            unsigned writemask =
               opcode.dst[0].mask >> D3D10_SB_OPERAND_4_COMPONENT_MASK_SHIFT;

            ureg_MOV(ureg, coord,
                     ureg_scalar(translate_src_operand(sx, &opcode.src[0], OF_UINT),
                                 TGSI_SWIZZLE_X));
            for (unsigned i = 0; i < 4; i++) {
               unsigned swizzle;
               if (!(writemask & (1u << i)))
                  continue;

               swizzle = opcode.src[1].swizzle[i];
               if (swizzle) {
                  if (ureg_dst_is_undef(fetch_offset))
                     fetch_offset = ureg_DECL_temporary(ureg);
                  ureg_UADD(ureg, fetch_offset, ureg_src(coord),
                            ureg_imm1u(ureg, swizzle * 4));
                  srcreg[1] = ureg_src(fetch_offset);
               } else {
                  srcreg[1] = ureg_src(coord);
               }

               srcreg[0] = memory;
               srcreg[1] = tgsm_byte_offset(sx, &opcode.src[1].base,
                                            srcreg[1], &byte_offset);
               struct ureg_dst fetch_dst =
                  ureg_writemask(fetch, TGSI_WRITEMASK_X);
               ureg_memory_insn(ureg, TGSI_OPCODE_LOAD,
                                &fetch_dst, 1, srcreg, ARRAY_SIZE(srcreg), 0,
                                TGSI_TEXTURE_BUFFER, PIPE_FORMAT_R32_UINT);
               ureg_MOV(ureg, ureg_writemask(dst, TGSI_WRITEMASK_X << i),
                        ureg_scalar(ureg_src(fetch), TGSI_SWIZZLE_X));
               if (!ureg_dst_is_undef(byte_offset)) {
                  ureg_release_temporary(ureg, byte_offset);
                  byte_offset = ureg_dst_undef();
               }
            }
            if (!ureg_dst_is_undef(fetch_offset))
               ureg_release_temporary(ureg, fetch_offset);
            ureg_release_temporary(ureg, fetch);
            ureg_release_temporary(ureg, coord);
            break;
         }

         assert(opcode.src[1].base.index_dim == 1);

         if (opcode.src[1].base.type ==
             D3D11_SB_OPERAND_TYPE_UNORDERED_ACCESS_VIEW) {
            unsigned image_index = opcode.src[1].base.index[0].imm;
            struct ureg_dst dst =
               translate_dst_operand(sx, &opcode.dst[0], opcode.saturate);

            assert(image_index < SHADER_MAX_IMAGES);

            ureg_USHR(ureg, coord,
                      ureg_scalar(translate_src_operand(sx, &opcode.src[0],
                                                        OF_UINT),
                                  TGSI_SWIZZLE_X),
                      ureg_imm1u(ureg, 2));
            image_buffer_load_dwords_ureg_emit(ureg, dst,
                                               sx->images[image_index].reg,
                                               ureg_src(coord),
                                               sx->images[image_index].target,
                                               sx->images[image_index].format,
                                               1, NULL);
            ureg_release_temporary(ureg, coord);
            break;
         }

         unsigned resource = opcode.src[1].base.index[0].imm;
         unsigned view_slot = resource;

         assert(resource < SHADER_MAX_RESOURCES);

         if (sx->sampler_binding_map) {
            if (!translate_binding_operand_index(
                   sx, &opcode.src[1], D3D10_SB_OPERAND_TYPE_RESOURCE,
                   SHADER_MAX_RESOURCES, "resource", &resource))
               break;
            declare_compact_sampler_for_resource(
               sx, resource, PIPE_MAX_SAMPLERS,
               (enum tgsi_texture_type)sx->resources[resource].target,
               &view_slot);
         } else if (ureg_src_is_undef(sx->sv[resource])) {
            sx->sv[resource] =
               ureg_DECL_sampler_view(ureg, resource, sx->resources[resource].target,
                                      TGSI_RETURN_TYPE_UINT,
                                      TGSI_RETURN_TYPE_UINT,
                                      TGSI_RETURN_TYPE_UINT,
                                      TGSI_RETURN_TYPE_UINT);
         }

         ureg_USHR(ureg, coord,
                   ureg_scalar(translate_src_operand(sx, &opcode.src[0], OF_UINT),
                               TGSI_SWIZZLE_X),
                   ureg_imm1u(ureg, 2));

         srcreg[0] = ureg_src(coord);
         srcreg[1] = sx->sampler_binding_map ?
            compact_sampler_view_src(sx, view_slot, &opcode.src[1]) :
            translate_src_operand(sx, &opcode.src[1], OF_UINT);
         sample_i_buffer_dwords_ureg_emit(
            ureg, &opcode,
            translate_dst_operand(sx, &opcode.dst[0], opcode.saturate),
            srcreg[0], srcreg[1], sx->resources[resource].target);
         ureg_release_temporary(ureg, coord);
         break;
      }

      case DX10_SM5_OPCODE_LD_STRUCTURED: {
         struct ureg_dst coord = ureg_DECL_temporary(ureg);
         struct ureg_dst byte_offset = ureg_DECL_temporary(ureg);
         struct ureg_src srcreg[2];

         if (opcode.src[2].base.type ==
             DX10_SM5_OPERAND_TYPE_THREAD_GROUP_SHARED_MEMORY) {
            unsigned slot = 0;
            struct ureg_src memory = get_tgsm_memory(sx);
            struct ureg_dst dst =
               translate_dst_operand(sx, &opcode.dst[0], opcode.saturate);
            struct ureg_dst fetch = ureg_DECL_temporary(ureg);
            struct ureg_dst load_offset = ureg_dst_undef();
            struct ureg_dst slot_offset = ureg_dst_undef();
            unsigned writemask =
               opcode.dst[0].mask >> D3D10_SB_OPERAND_4_COMPONENT_MASK_SHIFT;

            if (!tgsm_operand_slot(&opcode.src[2].base, &slot)) {
               ureg_release_temporary(ureg, fetch);
               ureg_release_temporary(ureg, byte_offset);
               ureg_release_temporary(ureg, coord);
               break;
            }

            ureg_UMUL(ureg, coord,
                      ureg_scalar(translate_src_operand(sx, &opcode.src[0], OF_UINT),
                                  TGSI_SWIZZLE_X),
                      ureg_imm1u(ureg, sx->tgsm[slot].structured_stride));
            ureg_UADD(ureg, coord, ureg_src(coord),
                      ureg_scalar(translate_src_operand(sx, &opcode.src[1], OF_UINT),
                                  TGSI_SWIZZLE_X));

            for (unsigned i = 0; i < 4; i++) {
               unsigned swizzle;
               if (!(writemask & (1u << i)))
                  continue;

               swizzle = opcode.src[2].swizzle[i];
               if (swizzle) {
                  if (ureg_dst_is_undef(load_offset))
                     load_offset = ureg_DECL_temporary(ureg);
                  ureg_UADD(ureg, load_offset, ureg_src(coord),
                            ureg_imm1u(ureg, swizzle * 4));
                  srcreg[1] = ureg_src(load_offset);
               } else {
                  srcreg[1] = ureg_src(coord);
               }

               srcreg[0] = memory;
               srcreg[1] = tgsm_byte_offset(sx, &opcode.src[2].base,
                                            srcreg[1], &slot_offset);
               struct ureg_dst fetch_dst =
                  ureg_writemask(fetch, TGSI_WRITEMASK_X);
               ureg_memory_insn(ureg, TGSI_OPCODE_LOAD,
                                &fetch_dst, 1, srcreg, ARRAY_SIZE(srcreg), 0,
                                TGSI_TEXTURE_BUFFER, PIPE_FORMAT_R32_UINT);
               ureg_MOV(ureg, ureg_writemask(dst, TGSI_WRITEMASK_X << i),
                        ureg_scalar(ureg_src(fetch), TGSI_SWIZZLE_X));
               if (!ureg_dst_is_undef(slot_offset)) {
                  ureg_release_temporary(ureg, slot_offset);
                  slot_offset = ureg_dst_undef();
               }
            }

            if (!ureg_dst_is_undef(load_offset))
               ureg_release_temporary(ureg, load_offset);
            ureg_release_temporary(ureg, fetch);
            ureg_release_temporary(ureg, byte_offset);
            ureg_release_temporary(ureg, coord);
            break;
         }

         assert(opcode.src[2].base.index_dim == 1);

         if (opcode.src[2].base.type ==
             D3D11_SB_OPERAND_TYPE_UNORDERED_ACCESS_VIEW) {
            unsigned image_index = opcode.src[2].base.index[0].imm;
            struct ureg_dst dst =
               translate_dst_operand(sx, &opcode.dst[0], opcode.saturate);

            assert(image_index < SHADER_MAX_IMAGES);

            ureg_UMUL(ureg, coord,
                      ureg_scalar(translate_src_operand(sx, &opcode.src[0],
                                                        OF_UINT),
                                  TGSI_SWIZZLE_X),
                      ureg_imm1u(ureg,
                                 sx->images[image_index].structured_stride));
            ureg_MOV(ureg, byte_offset,
                     ureg_scalar(translate_src_operand(sx, &opcode.src[1],
                                                       OF_UINT),
                                 TGSI_SWIZZLE_X));
            ureg_UADD(ureg, coord, ureg_src(coord), ureg_src(byte_offset));
            /* Image-buffer coordinates address R32_UINT texels, not bytes. */
            ureg_USHR(ureg, coord, ureg_src(coord), ureg_imm1u(ureg, 2));
            image_buffer_load_dwords_ureg_emit(ureg, dst,
                                               sx->images[image_index].reg,
                                               ureg_src(coord),
                                               sx->images[image_index].target,
                                               sx->images[image_index].format,
                                               1, opcode.src[2].swizzle);
            ureg_release_temporary(ureg, byte_offset);
            ureg_release_temporary(ureg, coord);
            break;
         }

         unsigned resource = opcode.src[2].base.index[0].imm;
         unsigned view_slot = resource;

         assert(resource < SHADER_MAX_RESOURCES);

         if (sx->sampler_binding_map) {
            if (!translate_binding_operand_index(
                   sx, &opcode.src[2], D3D10_SB_OPERAND_TYPE_RESOURCE,
                   SHADER_MAX_RESOURCES, "resource", &resource))
               break;
            declare_compact_sampler_for_resource(
               sx, resource, PIPE_MAX_SAMPLERS,
               (enum tgsi_texture_type)sx->resources[resource].target,
               &view_slot);
         } else if (ureg_src_is_undef(sx->sv[resource])) {
            sx->sv[resource] =
               ureg_DECL_sampler_view(ureg, resource, sx->resources[resource].target,
                                      TGSI_RETURN_TYPE_UINT,
                                      TGSI_RETURN_TYPE_UINT,
                                      TGSI_RETURN_TYPE_UINT,
                                      TGSI_RETURN_TYPE_UINT);
         }

         ureg_UMUL(ureg, coord,
                   ureg_scalar(translate_src_operand(sx, &opcode.src[0], OF_UINT),
                               TGSI_SWIZZLE_X),
                   ureg_imm1u(ureg, sx->resources[resource].structured_stride / 4));
         ureg_USHR(ureg, byte_offset,
                   ureg_scalar(translate_src_operand(sx, &opcode.src[1], OF_UINT),
                               TGSI_SWIZZLE_X),
                   ureg_imm1u(ureg, 2));
         ureg_UADD(ureg, coord, ureg_src(coord), ureg_src(byte_offset));

         srcreg[0] = ureg_src(coord);
         srcreg[1] = sx->sampler_binding_map ?
            compact_sampler_view_src(sx, view_slot, &opcode.src[2]) :
            translate_src_operand(sx, &opcode.src[2], OF_UINT);
         sample_i_buffer_dwords_ureg_emit(
            ureg, &opcode,
            translate_dst_operand(sx, &opcode.dst[0], opcode.saturate),
            srcreg[0], srcreg[1], sx->resources[resource].target);
         ureg_release_temporary(ureg, byte_offset);
         ureg_release_temporary(ureg, coord);
         break;
      }

      case D3D10_SB_OPCODE_CUSTOMDATA:
         if (opcode.customdata._class ==
             D3D10_SB_CUSTOMDATA_DCL_IMMEDIATE_CONSTANT_BUFFER) {
            sx->imms =
               ureg_DECL_immediate_block_uint(ureg,
                                              opcode.customdata.u.constbuf.data,
                                              opcode.customdata.u.constbuf.count);
         } else {
            assert(0);
         }
         break;

      case D3D10_SB_OPCODE_RESINFO:
         if (opcode.src[1].base.type == D3D11_SB_OPERAND_TYPE_UNORDERED_ACCESS_VIEW) {
            unsigned image_index = opcode.src[1].base.index[0].imm;
            struct ureg_dst dstreg =
               translate_dst_operand(sx, &opcode.dst[0], opcode.saturate);
            struct ureg_dst tmp = ureg_DECL_temporary(ureg);
            struct ureg_src image;
            struct ureg_src query;
            struct ureg_src swizzled_query;

            assert(opcode.src[1].base.index_dim == 1);
            assert(image_index < SHADER_MAX_IMAGES);

            image = sx->images[image_index].reg;
            ureg_memory_insn(ureg, TGSI_OPCODE_RESQ,
                             &tmp, 1, &image, 1, 0,
                             sx->images[image_index].target,
                             sx->images[image_index].format);

            query = ureg_src(tmp);
            swizzled_query = query;
            swizzled_query.SwizzleX = opcode.src[1].swizzle[0];
            swizzled_query.SwizzleY = opcode.src[1].swizzle[1];
            swizzled_query.SwizzleZ = opcode.src[1].swizzle[2];
            swizzled_query.SwizzleW = opcode.src[1].swizzle[3];

            if (opcode.specific.resinfo_ret_type ==
                D3D10_SB_RESINFO_INSTRUCTION_RETURN_UINT) {
               ureg_MOV(ureg, dstreg, swizzled_query);
            } else if (opcode.specific.resinfo_ret_type ==
                       D3D10_SB_RESINFO_INSTRUCTION_RETURN_FLOAT) {
               ureg_I2F(ureg, dstreg, swizzled_query);
            } else { /* D3D10_SB_RESINFO_INSTRUCTION_RETURN_RCPFLOAT */
               unsigned dims =
                  texture_dim_from_tgsi_target(sx->images[image_index].target);

               ureg_I2F(ureg, tmp, query);
               query = ureg_src(tmp);
               for (unsigned i = 0; i < 4; i++) {
                  unsigned dst_swizzle = opcode.src[1].swizzle[i];
                  struct ureg_dst masked = ureg_writemask(dstreg, 1 << i);

                  if (dst_swizzle < dims)
                     ureg_RCP(ureg, masked, ureg_scalar(query, dst_swizzle));
                  else
                     ureg_MOV(ureg, masked, ureg_scalar(query, dst_swizzle));
               }
            }
            ureg_release_temporary(ureg, tmp);
            break;
         }
         if (sx->sampler_binding_map || use_old_tex_ops ||
             (st_debug & ST_DEBUG_OLD_TEX_OPS)) {
            unsigned resource = opcode.src[1].base.index[0].imm;
            struct ureg_dst dstreg =
               translate_dst_operand(sx, &opcode.dst[0], opcode.saturate);
            struct ureg_src query;
            struct ureg_src swizzled_query;
            struct ureg_src sampler;
            struct ureg_dst tmp;
            assert(opcode.src[1].base.index_dim == 1);
            assert(opcode.src[1].base.index[0].imm < SHADER_MAX_RESOURCES);

            if (sx->sampler_binding_map) {
               if (!translate_binding_operand_index(
                      sx, &opcode.src[1], D3D10_SB_OPERAND_TYPE_RESOURCE,
                      SHADER_MAX_RESOURCES, "resource", &resource))
                  break;
               sampler = declare_compact_sampler_for_resource(
                  sx, resource, PIPE_MAX_SAMPLERS,
                  (enum tgsi_texture_type)sx->resources[resource].target,
                  NULL);
            } else {
               if (ureg_src_is_undef(sx->samplers[resource])) {
                  sx->samplers[resource] =
                     ureg_DECL_sampler(ureg, resource);
               }
               declare_sampler_view_for_resource(
                  sx, resource, resource, sx->resources[resource].target);
               sampler = sx->samplers[resource];
            }

            tmp = ureg_DECL_temporary(ureg);
            ureg_TXQ(ureg,
                     tmp,
                     sx->resources[resource].target,
                     translate_src_operand(sx, &opcode.src[0], OF_UINT),
                     sampler);

            query = ureg_src(tmp);
            swizzled_query = query;
            swizzled_query.SwizzleX = opcode.src[1].swizzle[0];
            swizzled_query.SwizzleY = opcode.src[1].swizzle[1];
            swizzled_query.SwizzleZ = opcode.src[1].swizzle[2];
            swizzled_query.SwizzleW = opcode.src[1].swizzle[3];
            if (opcode.specific.resinfo_ret_type ==
                D3D10_SB_RESINFO_INSTRUCTION_RETURN_UINT) {
               ureg_MOV(ureg, dstreg, swizzled_query);
            }
            else if (opcode.specific.resinfo_ret_type ==
                     D3D10_SB_RESINFO_INSTRUCTION_RETURN_FLOAT) {
               ureg_I2F(ureg, dstreg, swizzled_query);
            }
            else { /* D3D10_SB_RESINFO_INSTRUCTION_RETURN_RCPFLOAT */
               unsigned dims =
                  texture_dim_from_tgsi_target(sx->resources[resource].target);

               ureg_I2F(ureg, tmp, query);
               query = ureg_src(tmp);
               for (unsigned i = 0; i < 4; i++) {
                  unsigned dst_swizzle = opcode.src[1].swizzle[i];
                  struct ureg_dst masked = ureg_writemask(dstreg, 1 << i);

                  if (dst_swizzle < dims)
                     ureg_RCP(ureg, masked, ureg_scalar(query, dst_swizzle));
                  else
                     ureg_MOV(ureg, masked, ureg_scalar(query, dst_swizzle));
               }
            }
            ureg_release_temporary(ureg, tmp);
         }
         else {
            struct ureg_dst r0 = ureg_DECL_temporary(ureg);
            struct ureg_src tsrc = translate_src_operand(sx, &opcode.src[1], OF_UINT);
            struct ureg_dst dstreg = translate_dst_operand(sx, &opcode.dst[0],
                                                           opcode.saturate);

            /* while specs say swizzle is ignored better safe than sorry */
            tsrc.SwizzleX = TGSI_SWIZZLE_X;
            tsrc.SwizzleY = TGSI_SWIZZLE_Y;
            tsrc.SwizzleZ = TGSI_SWIZZLE_Z;
            tsrc.SwizzleW = TGSI_SWIZZLE_W;

            ureg_SVIEWINFO(ureg, r0,
                           translate_src_operand(sx, &opcode.src[0], OF_UINT),
                           tsrc);

            tsrc = ureg_src(r0);
            tsrc.SwizzleX = opcode.src[1].swizzle[0];
            tsrc.SwizzleY = opcode.src[1].swizzle[1];
            tsrc.SwizzleZ = opcode.src[1].swizzle[2];
            tsrc.SwizzleW = opcode.src[1].swizzle[3];

            if (opcode.specific.resinfo_ret_type ==
                D3D10_SB_RESINFO_INSTRUCTION_RETURN_UINT) {
               ureg_MOV(ureg, dstreg, tsrc);
            }
            else if (opcode.specific.resinfo_ret_type ==
                     D3D10_SB_RESINFO_INSTRUCTION_RETURN_FLOAT) {
                ureg_I2F(ureg, dstreg, tsrc);
            }
            else { /* D3D10_SB_RESINFO_INSTRUCTION_RETURN_RCPFLOAT */
               unsigned i;
               /*
                * Must apply rcp only to parts determined by dims,
                * (width/height/depth) but NOT to array size nor mip levels
                * hence need to figure that out here.
                * This is one sick modifier if you ask me!
                */
               unsigned res_index = opcode.src[1].base.index[0].imm;
               unsigned target = sx->resources[res_index].target;
               unsigned dims = texture_dim_from_tgsi_target(target);

               ureg_I2F(ureg, r0, ureg_src(r0));
               tsrc = ureg_src(r0);
               for (i = 0; i < 4; i++) {
                  unsigned dst_swizzle = opcode.src[1].swizzle[i];
                  struct ureg_dst dstregmasked = ureg_writemask(dstreg, 1 << i);
                  /*
                   * could do one mov with multiple write mask bits set
                   * but rcp is scalar anyway.
                   */
                  if (dst_swizzle < dims) {
                     ureg_RCP(ureg, dstregmasked, ureg_scalar(tsrc, dst_swizzle));
                  }
                  else {
                     ureg_MOV(ureg, dstregmasked, ureg_scalar(tsrc, dst_swizzle));
                  }
               }
            }
            ureg_release_temporary(ureg, r0);
         }
         break;

      case DX10_SM5_OPCODE_BUFINFO: {
         struct ureg_dst dstreg =
            translate_dst_operand(sx, &opcode.dst[0], opcode.saturate);
         struct ureg_dst tmp;

         assert(opcode.src[0].base.index_dim == 1);
         if (opcode.src[0].base.type == D3D11_SB_OPERAND_TYPE_UNORDERED_ACCESS_VIEW) {
            unsigned image_index = opcode.src[0].base.index[0].imm;
            struct ureg_src image;

            assert(image_index < SHADER_MAX_IMAGES);
            if (sx->images[image_index].target == TGSI_TEXTURE_BUFFER) {
               unsigned record = D3D10UMD_BUFINFO_UAV_RECORD_BASE + image_index;

               emit_dynamic_buffer_bufinfo(
                  sx, dstreg, record);
               track_dynamic_structured_bufinfo_stride(
                  sx, &opcode.dst[0], record,
                  sx->images[image_index].structured_stride);
               break;
            }

            tmp = ureg_DECL_temporary(ureg);
            image = sx->images[image_index].reg;
            ureg_memory_insn(ureg, TGSI_OPCODE_RESQ,
                             &tmp, 1, &image, 1, 0,
                             sx->images[image_index].target,
                             sx->images[image_index].format);
            if (sx->images[image_index].structured_stride) {
               emit_structured_buffer_bufinfo(ureg, dstreg, ureg_src(tmp),
                                              sx->images[image_index].structured_stride,
                                              sx->images[image_index].structured_stride);
            } else if (sx->images[image_index].raw) {
               emit_raw_buffer_bufinfo(ureg, dstreg, ureg_src(tmp));
            } else {
               ureg_MOV(ureg, dstreg, ureg_src(tmp));
            }
            ureg_release_temporary(ureg, tmp);
         } else {
            unsigned resource = opcode.src[0].base.index[0].imm;
            struct ureg_src sampler;

            assert(opcode.src[0].base.type == D3D10_SB_OPERAND_TYPE_RESOURCE);
            assert(resource < SHADER_MAX_RESOURCES);

            if (sx->resources[resource].target == TGSI_TEXTURE_BUFFER) {
               unsigned record = D3D10UMD_BUFINFO_SRV_RECORD_BASE + resource;

               emit_dynamic_buffer_bufinfo(
                  sx, dstreg, record);
               track_dynamic_structured_bufinfo_stride(
                  sx, &opcode.dst[0], record,
                  sx->resources[resource].structured_stride);
               break;
            }

            if (sx->sampler_binding_map) {
               if (!translate_binding_operand_index(
                      sx, &opcode.src[0], D3D10_SB_OPERAND_TYPE_RESOURCE,
                      SHADER_MAX_RESOURCES, "resource", &resource))
                  break;
               sampler = declare_compact_sampler_for_resource(
                  sx, resource, PIPE_MAX_SAMPLERS,
                  (enum tgsi_texture_type)sx->resources[resource].target,
                  NULL);
            } else {
               if (ureg_src_is_undef(sx->samplers[resource]))
                  sx->samplers[resource] = ureg_DECL_sampler(ureg, resource);
               declare_sampler_view_for_resource(
                  sx, resource, resource, sx->resources[resource].target);
               sampler = sx->samplers[resource];
            }

            tmp = ureg_DECL_temporary(ureg);
            ureg_TXQ(ureg, tmp, sx->resources[resource].target,
                     ureg_imm1u(ureg, 0), sampler);
            if (sx->resources[resource].structured_stride) {
               emit_structured_buffer_bufinfo(ureg, dstreg, ureg_src(tmp),
                                              sx->resources[resource].structured_stride,
                                              sx->resources[resource].structured_stride /
                                              sizeof(uint32_t));
            } else if (sx->resources[resource].raw) {
               emit_raw_buffer_bufinfo(ureg, dstreg, ureg_src(tmp));
            } else {
               ureg_MOV(ureg, dstreg, ureg_src(tmp));
            }
            ureg_release_temporary(ureg, tmp);
         }
         break;
      }

      case D3D11_SB_OPCODE_EVAL_SAMPLE_INDEX: {
         const struct Shader_operand *input = &opcode.src[0].base;
         if (parser.header.type != D3D10_SB_PIXEL_SHADER ||
             input->type != D3D10_SB_OPERAND_TYPE_INPUT ||
             input->index_dim != 1 ||
             input->index[0].index_rep !=
                D3D10_SB_OPERAND_INDEX_IMMEDIATE32 ||
             input->index[0].imm >= SHADER_MAX_INPUTS ||
             !sx->inputs[input->index[0].imm].declared ||
             sx->inputs[input->index[0].imm].reg.File != TGSI_FILE_INPUT) {
            YTTRIUM_WARN("yttrium: shader TGSI translation failed "
                         "owner=d3d10umd-shader component=eval-sample-index "
                         "reason=unsupported-input-form "
                         "action=abort-translation\n");
            sx->translation_failed = true;
            break;
         }

         /* Pull the attribute itself, not a temporary containing its ordinary
          * pixel-center value. TGSI retains the declaration's perspective
          * mode while INTERP_SAMPLE overrides its interpolation location.
          */
         ureg_INTERP_SAMPLE(
            ureg, translate_dst_operand(sx, &opcode.dst[0], false),
            translate_src_operand(sx, &opcode.src[0], OF_FLOAT),
            ureg_scalar(translate_src_operand(sx, &opcode.src[1], OF_UINT),
                        TGSI_SWIZZLE_X));
         break;
      }

      case D3D10_1_SB_OPCODE_SAMPLE_INFO: {
         struct ureg_dst dstreg =
            translate_dst_operand(sx, &opcode.dst[0], opcode.saturate);
         struct ureg_src src[2];
         struct ureg_dst query_dst = dstreg;
         struct ureg_dst tmp = ureg_dst_undef();
         bool dst_is_float =
            sample_info_writes_float_color0(&opcode.dst[0]);
         unsigned resource;
         unsigned view_slot;

         if (opcode.src[0].base.type == D3D10_SB_OPERAND_TYPE_RASTERIZER) {
            emit_rasterizer_sample_info(sx, dstreg, dst_is_float);
            break;
         }

         assert(opcode.src[0].base.type == D3D10_SB_OPERAND_TYPE_RESOURCE);
         assert(opcode.src[0].base.index_dim == 1);
         resource = opcode.src[0].base.index[0].imm;
         view_slot = resource;
         assert(resource < SHADER_MAX_RESOURCES);

         if (sx->sampler_binding_map) {
            if (!translate_binding_operand_index(
                   sx, &opcode.src[0], D3D10_SB_OPERAND_TYPE_RESOURCE,
                   SHADER_MAX_RESOURCES, "resource", &resource))
               break;
            declare_compact_sampler_for_resource(
               sx, resource, PIPE_MAX_SAMPLERS,
               (enum tgsi_texture_type)sx->resources[resource].target,
               &view_slot);
         } else if (ureg_src_is_undef(sx->sv[resource])) {
            sx->sv[resource] =
               ureg_DECL_sampler_view(ureg, resource, sx->resources[resource].target,
                                      TGSI_RETURN_TYPE_UINT,
                                      TGSI_RETURN_TYPE_UINT,
                                      TGSI_RETURN_TYPE_UINT,
                                      TGSI_RETURN_TYPE_UINT);
         }

         if (dst_is_float) {
            tmp = ureg_DECL_temporary(ureg);
            query_dst = tmp;
         }

         src[0] = ureg_imm1u(ureg, 0);
         src[1] = sx->sampler_binding_map ?
            compact_sampler_view_src(sx, view_slot, &opcode.src[0]) :
            translate_src_operand(sx, &opcode.src[0], OF_UINT);
         ureg_tex_insn(ureg, TGSI_OPCODE_SAMPLE_INFO, &query_dst, 1,
                       sx->resources[resource].target, NULL, 0, src, 2);
         if (dst_is_float) {
            ureg_I2F(ureg, dstreg,
                     ureg_scalar(ureg_src(tmp), TGSI_SWIZZLE_X));
            ureg_release_temporary(ureg, tmp);
         }
         break;
      }

      case D3D10_1_SB_OPCODE_SAMPLE_POS: {
         struct ureg_dst dstreg =
            translate_dst_operand(sx, &opcode.dst[0], opcode.saturate);
         struct ureg_src src[2];
         unsigned resource;
         unsigned view_slot;

         if (opcode.src[0].base.type == D3D10_SB_OPERAND_TYPE_RASTERIZER) {
            ureg_MOV(ureg, dstreg, ureg_imm4f(ureg, 0.5f, 0.5f, 0.0f, 0.0f));
            break;
         }

         assert(opcode.src[0].base.type == D3D10_SB_OPERAND_TYPE_RESOURCE);
         assert(opcode.src[0].base.index_dim == 1);
         resource = opcode.src[0].base.index[0].imm;
         view_slot = resource;
         assert(resource < SHADER_MAX_RESOURCES);

         if ((sx->resources[resource].target == TGSI_TEXTURE_2D_MSAA ||
              sx->resources[resource].target == TGSI_TEXTURE_2D_ARRAY_MSAA) &&
             try_emit_standard_4x_sample_position(ureg, dstreg, &opcode.src[1]))
            break;

         if (sx->sampler_binding_map) {
            if (!translate_binding_operand_index(
                   sx, &opcode.src[0], D3D10_SB_OPERAND_TYPE_RESOURCE,
                   SHADER_MAX_RESOURCES, "resource", &resource))
               break;
            declare_compact_sampler_for_resource(
               sx, resource, PIPE_MAX_SAMPLERS,
               (enum tgsi_texture_type)sx->resources[resource].target,
               &view_slot);
         } else if (ureg_src_is_undef(sx->sv[resource])) {
            sx->sv[resource] =
               ureg_DECL_sampler_view(ureg, resource, sx->resources[resource].target,
                                      TGSI_RETURN_TYPE_FLOAT,
                                      TGSI_RETURN_TYPE_FLOAT,
                                      TGSI_RETURN_TYPE_FLOAT,
                                      TGSI_RETURN_TYPE_FLOAT);
         }

         src[0] = sx->sampler_binding_map ?
            compact_sampler_view_src(sx, view_slot, &opcode.src[0]) :
            translate_src_operand(sx, &opcode.src[0], OF_UINT);
         src[1] = translate_src_operand(sx, &opcode.src[1], OF_UINT);
         ureg_tex_insn(ureg, TGSI_OPCODE_SAMPLE_POS, &dstreg, 1,
                       sx->resources[resource].target, NULL, 0, src, 2);
         break;
      }

      case D3D10_1_SB_OPCODE_GATHER4:
      case D3D11_SB_OPCODE_GATHER4_C:
      case D3D11_SB_OPCODE_GATHER4_PO:
      case D3D11_SB_OPCODE_GATHER4_PO_C: {
         struct ureg_src srcreg[3];
         struct ureg_src dynamic_offset;
         struct ureg_dst coord = ureg_dst_undef();
         const bool has_dynamic_offset =
            opcode.type == D3D11_SB_OPCODE_GATHER4_PO ||
            opcode.type == D3D11_SB_OPCODE_GATHER4_PO_C;
         const bool is_comparison =
            opcode.type == D3D11_SB_OPCODE_GATHER4_C ||
            opcode.type == D3D11_SB_OPCODE_GATHER4_PO_C;
         unsigned resource_src =
            has_dynamic_offset ? 2 : 1;
         unsigned sampler_src = resource_src + 1;
         unsigned compare_src =
            has_dynamic_offset ? 4 : 3;
         unsigned resource = opcode.src[resource_src].base.index[0].imm;
         unsigned sampler = opcode.src[sampler_src].base.index[0].imm;
         unsigned binding_slot = resource;

         if (sx->sampler_binding_map &&
             (!translate_binding_operand_index(
                 sx, &opcode.src[resource_src],
                 D3D10_SB_OPERAND_TYPE_RESOURCE, SHADER_MAX_RESOURCES,
                 "resource", &resource) ||
              !translate_binding_operand_index(
                 sx, &opcode.src[sampler_src],
                 D3D10_SB_OPERAND_TYPE_SAMPLER, SHADER_MAX_SAMPLERS,
                 "sampler", &sampler)))
            break;

         enum tgsi_texture_type target = sx->resources[resource].target;

         assert(opcode.src[resource_src].base.index_dim == 1);
         assert(opcode.src[sampler_src].base.index_dim == 1);
         assert(resource < SHADER_MAX_RESOURCES);
         assert(opcode.src[sampler_src].base.index[0].imm < SHADER_MAX_SAMPLERS);

         if (is_comparison) {
            const enum tgsi_texture_type shadow_target =
               translate_shadow_texture_target(target);

            SHADER_TRANSLATION_UNSUPPORTED(
               sx, shadow_target == TGSI_TEXTURE_UNKNOWN);
            if (shadow_target != TGSI_TEXTURE_UNKNOWN)
               target = shadow_target;

            coord = ureg_DECL_temporary(ureg);
            ureg_MOV(ureg, coord,
                     translate_src_operand(sx, &opcode.src[0], OF_FLOAT));
            ureg_MOV(ureg,
                     ureg_writemask(coord, shadow_ref_writemask(target)),
                     translate_src_operand(sx, &opcode.src[compare_src],
                                           OF_FLOAT));
            srcreg[0] = ureg_src(coord);
         } else {
            srcreg[0] = translate_src_operand(sx, &opcode.src[0], OF_FLOAT);
         }
         if (sx->sampler_binding_map) {
            declare_compact_sampler_for_resource(
               sx, resource, sampler, target, &binding_slot);
         } else if (ureg_src_is_undef(sx->sv[resource])) {
            sx->sv[resource] =
               ureg_DECL_sampler_view(ureg, resource, target,
                                      TGSI_RETURN_TYPE_FLOAT,
                                      TGSI_RETURN_TYPE_FLOAT,
                                      TGSI_RETURN_TYPE_FLOAT,
                                      TGSI_RETURN_TYPE_FLOAT);
         }
         if (sx->sampler_binding_map) {
            srcreg[1] = compact_sampler_view_src(
               sx, binding_slot, &opcode.src[resource_src]);
            srcreg[2] = compact_sampler_view_src(
               sx, binding_slot, &opcode.src[sampler_src]);
         } else {
            srcreg[1] = translate_src_operand(
               sx, &opcode.src[resource_src], OF_UINT);
            srcreg[2] = ureg_src_register(TGSI_FILE_SAMPLER_VIEW, resource);
            srcreg[2] = ureg_swizzle(srcreg[2],
                                     opcode.src[sampler_src].swizzle[0],
                                     opcode.src[sampler_src].swizzle[1],
                                     opcode.src[sampler_src].swizzle[2],
                                     opcode.src[sampler_src].swizzle[3]);
         }
         if (has_dynamic_offset)
            dynamic_offset = translate_src_operand(sx, &opcode.src[1], OF_INT);
         sample_ureg_emit_target(ureg, TGSI_OPCODE_TG4, 3, &opcode,
                                 has_dynamic_offset ? &dynamic_offset : NULL,
                                 translate_dst_operand(sx, &opcode.dst[0],
                                                       opcode.saturate),
                                 srcreg, target);
         if (!ureg_dst_is_undef(coord))
            ureg_release_temporary(ureg, coord);
         break;
      }

      case D3D10_1_SB_OPCODE_LOD: {
         unsigned resource = opcode.src[1].base.index[0].imm;
         unsigned sampler = opcode.src[2].base.index[0].imm;
         unsigned binding_slot = resource;
         struct ureg_src srcreg[2];

         assert(opcode.src[1].base.index_dim == 1);
         assert(resource < SHADER_MAX_RESOURCES);

         if (sx->sampler_binding_map &&
             (!translate_binding_operand_index(
                 sx, &opcode.src[1], D3D10_SB_OPERAND_TYPE_RESOURCE,
                 SHADER_MAX_RESOURCES, "resource", &resource) ||
              !translate_binding_operand_index(
                 sx, &opcode.src[2], D3D10_SB_OPERAND_TYPE_SAMPLER,
                 SHADER_MAX_SAMPLERS, "sampler", &sampler)))
            break;

         if (sx->sampler_binding_map) {
            declare_compact_sampler_for_resource(
               sx, resource, sampler,
               (enum tgsi_texture_type)sx->resources[resource].target,
               &binding_slot);
         } else if (ureg_src_is_undef(sx->sv[resource])) {
            sx->sv[resource] =
               ureg_DECL_sampler_view(ureg, resource, sx->resources[resource].target,
                                      TGSI_RETURN_TYPE_FLOAT,
                                      TGSI_RETURN_TYPE_FLOAT,
                                      TGSI_RETURN_TYPE_FLOAT,
                                      TGSI_RETURN_TYPE_FLOAT);
         }

         srcreg[0] = translate_src_operand(sx, &opcode.src[0], OF_FLOAT);
         srcreg[1] = sx->sampler_binding_map ?
            compact_sampler_view_src(sx, binding_slot, &opcode.src[1]) :
            translate_src_operand(sx, &opcode.src[1], OF_UINT);
         sample_ureg_emit_target(ureg, TGSI_OPCODE_LODQ, ARRAY_SIZE(srcreg),
                                 &opcode, NULL,
                                 translate_dst_operand(sx, &opcode.dst[0],
                                                       opcode.saturate),
                                 srcreg, sx->resources[resource].target);
         break;
      }

      case D3D10_SB_OPCODE_SAMPLE:
         if (sx->sampler_binding_map || use_old_tex_ops ||
             (st_debug & ST_DEBUG_OLD_TEX_OPS)) {
            unsigned resource = opcode.src[1].base.index[0].imm;
            struct ureg_dst dst;
            struct ureg_dst tmp;
            struct ureg_dst tex_dst;
            struct ureg_src coord;
            struct ureg_src sampler;
            assert(opcode.src[1].base.index_dim == 1);
            assert(resource < SHADER_MAX_RESOURCES);

            if (sx->sampler_binding_map &&
                !translate_binding_operand_index(
                   sx, &opcode.src[1], D3D10_SB_OPERAND_TYPE_RESOURCE,
                   SHADER_MAX_RESOURCES, "resource", &resource))
               break;

            dst = translate_dst_operand(sx, &opcode.dst[0],
                                        opcode.saturate);
            coord = translate_src_operand(sx, &opcode.src[0], OF_FLOAT);
            sampler = translate_old_tex_sampler_for_resource(sx, resource,
                                                            &opcode.src[2]);

            tex_dst = old_tex_dst_for_resource_swizzle(ureg, dst,
                                                       &opcode.src[1], &tmp);
            struct ureg_src tex_src[] = { coord, sampler };
            sample_ureg_emit_target(ureg, TGSI_OPCODE_TEX,
                                    ARRAY_SIZE(tex_src), &opcode, NULL,
                                    tex_dst, tex_src,
                                    sx->resources[resource].target);
            old_tex_apply_resource_swizzle(ureg, dst, &opcode.src[1], tmp);
         }
         else {
            struct ureg_src srcreg[3];
            srcreg[0] = translate_src_operand(sx, &opcode.src[0], OF_FLOAT);
            srcreg[1] = translate_src_operand(sx, &opcode.src[1], OF_UINT);
            srcreg[2] = translate_src_operand(sx, &opcode.src[2], OF_UINT);

            sample_ureg_emit(ureg, TGSI_OPCODE_SAMPLE, 3, &opcode,
                             translate_dst_operand(sx, &opcode.dst[0],
                                                   opcode.saturate),
                             srcreg);
         }
         break;

      case D3D10_SB_OPCODE_SAMPLE_C:
      {
         unsigned resource = opcode.src[1].base.index[0].imm;
         unsigned sampler = opcode.src[2].base.index[0].imm;
         unsigned binding_slot = resource;
         if (sx->sampler_binding_map &&
             (!translate_binding_operand_index(
                 sx, &opcode.src[1], D3D10_SB_OPERAND_TYPE_RESOURCE,
                 SHADER_MAX_RESOURCES, "resource", &resource) ||
              !translate_binding_operand_index(
                 sx, &opcode.src[2], D3D10_SB_OPERAND_TYPE_SAMPLER,
                 SHADER_MAX_SAMPLERS, "sampler", &sampler)))
            break;

         const uint target = sx->resources[resource].target;
         const uint shadow_target = translate_shadow_texture_target(target);
         const uint instruction_target =
            shadow_target != TGSI_TEXTURE_UNKNOWN ? shadow_target : target;
         struct ureg_src srcreg[4];

         assert(opcode.src[1].base.index_dim == 1);
         assert(opcode.src[2].base.index_dim == 1);
         assert(resource < SHADER_MAX_RESOURCES);

         SHADER_TRANSLATION_UNSUPPORTED(
            sx, shadow_target == TGSI_TEXTURE_UNKNOWN);

         if (sx->sampler_binding_map) {
            declare_compact_sampler_for_resource(
               sx, resource, sampler,
               (enum tgsi_texture_type)instruction_target, &binding_slot);
         } else if (ureg_src_is_undef(sx->sv[resource])) {
            sx->sv[resource] =
               ureg_DECL_sampler_view(ureg, resource, target,
                                      TGSI_RETURN_TYPE_FLOAT,
                                      TGSI_RETURN_TYPE_FLOAT,
                                      TGSI_RETURN_TYPE_FLOAT,
                                      TGSI_RETURN_TYPE_FLOAT);
         }

         srcreg[0] = translate_src_operand(sx, &opcode.src[0], OF_FLOAT);
         srcreg[1] = sx->sampler_binding_map ?
            compact_sampler_view_src(sx, binding_slot, &opcode.src[1]) :
            ureg_src_register(TGSI_FILE_SAMPLER_VIEW, resource);
         if (!sx->sampler_binding_map)
            srcreg[1] = ureg_swizzle(srcreg[1],
                                     opcode.src[1].swizzle[0],
                                     opcode.src[1].swizzle[1],
                                     opcode.src[1].swizzle[2],
                                     opcode.src[1].swizzle[3]);
         assert(opcode.src[1].modifier == D3D10_SB_OPERAND_MODIFIER_NONE);
         srcreg[2] = ureg_src_register(
            TGSI_FILE_SAMPLER_VIEW,
            sx->sampler_binding_map ? binding_slot : resource);
         srcreg[3] = translate_src_operand(sx, &opcode.src[3], OF_FLOAT);

         sample_ureg_emit_target(ureg, TGSI_OPCODE_SAMPLE_C, ARRAY_SIZE(srcreg),
                                 &opcode, NULL,
                                 translate_dst_operand(sx, &opcode.dst[0],
                                                       opcode.saturate),
                                 srcreg,
                                 instruction_target);
         break;
      }

      case D3D10_SB_OPCODE_SAMPLE_C_LZ:
      {
         unsigned resource = opcode.src[1].base.index[0].imm;
         unsigned sampler = opcode.src[2].base.index[0].imm;
         unsigned binding_slot = resource;
         if (sx->sampler_binding_map &&
             (!translate_binding_operand_index(
                 sx, &opcode.src[1], D3D10_SB_OPERAND_TYPE_RESOURCE,
                 SHADER_MAX_RESOURCES, "resource", &resource) ||
              !translate_binding_operand_index(
                 sx, &opcode.src[2], D3D10_SB_OPERAND_TYPE_SAMPLER,
                 SHADER_MAX_SAMPLERS, "sampler", &sampler)))
            break;

         const uint target = sx->resources[resource].target;
         const uint shadow_target = translate_shadow_texture_target(target);
         const uint instruction_target =
            shadow_target != TGSI_TEXTURE_UNKNOWN ? shadow_target : target;
         struct ureg_src srcreg[4];

         assert(opcode.src[1].base.index_dim == 1);
         assert(opcode.src[2].base.index_dim == 1);
         assert(resource < SHADER_MAX_RESOURCES);

         SHADER_TRANSLATION_UNSUPPORTED(
            sx, shadow_target == TGSI_TEXTURE_UNKNOWN);

         if (sx->sampler_binding_map) {
            declare_compact_sampler_for_resource(
               sx, resource, sampler,
               (enum tgsi_texture_type)instruction_target, &binding_slot);
         } else if (ureg_src_is_undef(sx->sv[resource])) {
            sx->sv[resource] =
               ureg_DECL_sampler_view(ureg, resource, target,
                                      TGSI_RETURN_TYPE_FLOAT,
                                      TGSI_RETURN_TYPE_FLOAT,
                                      TGSI_RETURN_TYPE_FLOAT,
                                      TGSI_RETURN_TYPE_FLOAT);
         }

         srcreg[0] = translate_src_operand(sx, &opcode.src[0], OF_FLOAT);
         srcreg[1] = sx->sampler_binding_map ?
            compact_sampler_view_src(sx, binding_slot, &opcode.src[1]) :
            ureg_src_register(TGSI_FILE_SAMPLER_VIEW, resource);
         if (!sx->sampler_binding_map)
            srcreg[1] = ureg_swizzle(srcreg[1],
                                     opcode.src[1].swizzle[0],
                                     opcode.src[1].swizzle[1],
                                     opcode.src[1].swizzle[2],
                                     opcode.src[1].swizzle[3]);
         assert(opcode.src[1].modifier == D3D10_SB_OPERAND_MODIFIER_NONE);
         srcreg[2] = ureg_src_register(
            TGSI_FILE_SAMPLER_VIEW,
            sx->sampler_binding_map ? binding_slot : resource);
         srcreg[3] = translate_src_operand(sx, &opcode.src[3], OF_FLOAT);

         sample_ureg_emit_target(ureg, TGSI_OPCODE_SAMPLE_C_LZ,
                                 ARRAY_SIZE(srcreg), &opcode, NULL,
                                 translate_dst_operand(sx, &opcode.dst[0],
                                                       opcode.saturate),
                                 srcreg,
                                 instruction_target);
         break;
      }

      case D3D10_SB_OPCODE_SAMPLE_L:
         if (sx->sampler_binding_map || use_old_tex_ops ||
             (st_debug & ST_DEBUG_OLD_TEX_OPS)) {
            unsigned resource = opcode.src[1].base.index[0].imm;
            struct ureg_dst dst =
               translate_dst_operand(sx, &opcode.dst[0], opcode.saturate);
            struct ureg_dst tmp;
            struct ureg_dst tex_dst;

            assert(opcode.src[1].base.index_dim == 1);
            assert(resource < SHADER_MAX_RESOURCES);

            if (sx->sampler_binding_map &&
                !translate_binding_operand_index(
                   sx, &opcode.src[1], D3D10_SB_OPERAND_TYPE_RESOURCE,
                   SHADER_MAX_RESOURCES, "resource", &resource))
               break;

            tex_dst = old_tex_dst_for_resource_swizzle(ureg, dst,
                                                       &opcode.src[1], &tmp);
            if (sx->resources[resource].target == TGSI_TEXTURE_CUBE_ARRAY) {
               /* Cube-array coordinates need .w for the array index. */
               struct ureg_src srcreg[3];
               srcreg[0] = translate_src_operand(sx, &opcode.src[0], OF_FLOAT);
               srcreg[1] = translate_src_operand(sx, &opcode.src[3], OF_FLOAT);
               srcreg[2] =
                  translate_old_tex_sampler_for_resource(sx, resource,
                                                         &opcode.src[2]);

               sample_ureg_emit_target(ureg, TGSI_OPCODE_TXL2,
                                       ARRAY_SIZE(srcreg), &opcode, NULL,
                                       tex_dst, srcreg,
                                       sx->resources[resource].target);
            } else {
               struct ureg_dst r0 = ureg_DECL_temporary(ureg);

               /* Insert LOD into .w component.
                */
               ureg_MOV(ureg,
                        ureg_writemask(r0, TGSI_WRITEMASK_XYZ),
                        translate_src_operand(sx, &opcode.src[0], OF_FLOAT));
               ureg_MOV(ureg,
                        ureg_writemask(r0, TGSI_WRITEMASK_W),
                        translate_src_operand(sx, &opcode.src[3], OF_FLOAT));

               struct ureg_src tex_src[] = {
                  ureg_src(r0),
                  translate_old_tex_sampler_for_resource(sx, resource,
                                                         &opcode.src[2]),
               };
               sample_ureg_emit_target(ureg, TGSI_OPCODE_TXL,
                                       ARRAY_SIZE(tex_src), &opcode, NULL,
                                       tex_dst, tex_src,
                                       sx->resources[resource].target);

               ureg_release_temporary(ureg, r0);
            }
            old_tex_apply_resource_swizzle(ureg, dst, &opcode.src[1], tmp);
         }
         else {
            struct ureg_src srcreg[4];
            srcreg[0] = translate_src_operand(sx, &opcode.src[0], OF_FLOAT);
            srcreg[1] = translate_src_operand(sx, &opcode.src[1], OF_UINT);
            srcreg[2] = translate_src_operand(sx, &opcode.src[2], OF_UINT);
            srcreg[3] = translate_src_operand(sx, &opcode.src[3], OF_FLOAT);

            sample_ureg_emit(ureg, TGSI_OPCODE_SAMPLE_L, 4, &opcode,
                             translate_dst_operand(sx, &opcode.dst[0],
                                                   opcode.saturate),
                             srcreg);
         }
         break;

      case D3D10_SB_OPCODE_SAMPLE_D:
         if (sx->sampler_binding_map || use_old_tex_ops ||
             (st_debug & ST_DEBUG_OLD_TEX_OPS)) {
            unsigned resource = opcode.src[1].base.index[0].imm;
            struct ureg_dst dst =
               translate_dst_operand(sx, &opcode.dst[0], opcode.saturate);
            struct ureg_dst tmp;
            struct ureg_dst tex_dst;
            assert(opcode.src[1].base.index_dim == 1);
            assert(resource < SHADER_MAX_RESOURCES);

            if (sx->sampler_binding_map &&
                !translate_binding_operand_index(
                   sx, &opcode.src[1], D3D10_SB_OPERAND_TYPE_RESOURCE,
                   SHADER_MAX_RESOURCES, "resource", &resource))
               break;

            tex_dst = old_tex_dst_for_resource_swizzle(ureg, dst,
                                                       &opcode.src[1], &tmp);
            struct ureg_src tex_src[] = {
               translate_src_operand(sx, &opcode.src[0], OF_FLOAT),
               translate_src_operand(sx, &opcode.src[3], OF_FLOAT),
               translate_src_operand(sx, &opcode.src[4], OF_FLOAT),
               translate_old_tex_sampler_for_resource(sx, resource,
                                                      &opcode.src[2]),
            };
            sample_ureg_emit_target(ureg, TGSI_OPCODE_TXD,
                                    ARRAY_SIZE(tex_src), &opcode, NULL,
                                    tex_dst, tex_src,
                                    sx->resources[resource].target);
            old_tex_apply_resource_swizzle(ureg, dst, &opcode.src[1], tmp);
         }
         else {
            struct ureg_src srcreg[5];
            srcreg[0] = translate_src_operand(sx, &opcode.src[0], OF_FLOAT);
            srcreg[1] = translate_src_operand(sx, &opcode.src[1], OF_UINT);
            srcreg[2] = translate_src_operand(sx, &opcode.src[2], OF_UINT);
            srcreg[3] = translate_src_operand(sx, &opcode.src[3], OF_FLOAT);
            srcreg[4] = translate_src_operand(sx, &opcode.src[4], OF_FLOAT);

            sample_ureg_emit(ureg, TGSI_OPCODE_SAMPLE_D, 5, &opcode,
                             translate_dst_operand(sx, &opcode.dst[0],
                                                   opcode.saturate),
                             srcreg);
         }
         break;

      case D3D10_SB_OPCODE_SAMPLE_B:
         if (sx->sampler_binding_map || use_old_tex_ops ||
             (st_debug & ST_DEBUG_OLD_TEX_OPS)) {
            struct ureg_dst r0 = ureg_DECL_temporary(ureg);
            unsigned resource = opcode.src[1].base.index[0].imm;
            struct ureg_dst dst =
               translate_dst_operand(sx, &opcode.dst[0], opcode.saturate);
            struct ureg_dst tmp;
            struct ureg_dst tex_dst;

            assert(opcode.src[1].base.index_dim == 1);
            assert(resource < SHADER_MAX_RESOURCES);

            if (sx->sampler_binding_map &&
                !translate_binding_operand_index(
                   sx, &opcode.src[1], D3D10_SB_OPERAND_TYPE_RESOURCE,
                   SHADER_MAX_RESOURCES, "resource", &resource))
               break;

            /* Insert LOD bias into .w component.
             */
            ureg_MOV(ureg,
                     ureg_writemask(r0, TGSI_WRITEMASK_XYZ),
                     translate_src_operand(sx, &opcode.src[0], OF_FLOAT));
            ureg_MOV(ureg,
                     ureg_writemask(r0, TGSI_WRITEMASK_W),
                     translate_src_operand(sx, &opcode.src[3], OF_FLOAT));

            tex_dst = old_tex_dst_for_resource_swizzle(ureg, dst,
                                                       &opcode.src[1], &tmp);
            struct ureg_src tex_src[] = {
               ureg_src(r0),
               translate_old_tex_sampler_for_resource(sx, resource,
                                                      &opcode.src[2]),
            };
            sample_ureg_emit_target(ureg, TGSI_OPCODE_TXB,
                                    ARRAY_SIZE(tex_src), &opcode, NULL,
                                    tex_dst, tex_src,
                                    sx->resources[resource].target);
            old_tex_apply_resource_swizzle(ureg, dst, &opcode.src[1], tmp);

            ureg_release_temporary(ureg, r0);
         }
         else {
            struct ureg_src srcreg[4];
            srcreg[0] = translate_src_operand(sx, &opcode.src[0], OF_FLOAT);
            srcreg[1] = translate_src_operand(sx, &opcode.src[1], OF_UINT);
            srcreg[2] = translate_src_operand(sx, &opcode.src[2], OF_UINT);
            srcreg[3] = translate_src_operand(sx, &opcode.src[3], OF_FLOAT);

            sample_ureg_emit(ureg, TGSI_OPCODE_SAMPLE_B, 4, &opcode,
                             translate_dst_operand(sx, &opcode.dst[0],
                                                   opcode.saturate),
                             srcreg);
         }
         break;

      case D3D10_SB_OPCODE_SINCOS: {
         struct ureg_dst src0 = ureg_DECL_temporary(ureg);
         ureg_MOV(ureg, src0, translate_src_operand(sx, &opcode.src[0], OF_FLOAT));
         if (opcode.dst[0].base.type != D3D10_SB_OPERAND_TYPE_NULL) {
            struct ureg_dst dst = translate_dst_operand(sx, &opcode.dst[0],
                                                        opcode.saturate);
            struct ureg_src src = ureg_src(src0);
            ureg_SIN(ureg, ureg_writemask(dst, TGSI_WRITEMASK_X),
                     ureg_scalar(src, TGSI_SWIZZLE_X));
            ureg_SIN(ureg, ureg_writemask(dst, TGSI_WRITEMASK_Y),
                     ureg_scalar(src, TGSI_SWIZZLE_Y));
            ureg_SIN(ureg, ureg_writemask(dst, TGSI_WRITEMASK_Z),
                     ureg_scalar(src, TGSI_SWIZZLE_Z));
            ureg_SIN(ureg, ureg_writemask(dst, TGSI_WRITEMASK_W),
                     ureg_scalar(src, TGSI_SWIZZLE_W));
         }
         if (opcode.dst[1].base.type != D3D10_SB_OPERAND_TYPE_NULL) {
            struct ureg_dst dst = translate_dst_operand(sx, &opcode.dst[1],
                                                        opcode.saturate);
            struct ureg_src src = ureg_src(src0);
            ureg_COS(ureg, ureg_writemask(dst, TGSI_WRITEMASK_X),
                     ureg_scalar(src, TGSI_SWIZZLE_X));
            ureg_COS(ureg, ureg_writemask(dst, TGSI_WRITEMASK_Y),
                     ureg_scalar(src, TGSI_SWIZZLE_Y));
            ureg_COS(ureg, ureg_writemask(dst, TGSI_WRITEMASK_Z),
                     ureg_scalar(src, TGSI_SWIZZLE_Z));
            ureg_COS(ureg, ureg_writemask(dst, TGSI_WRITEMASK_W),
                     ureg_scalar(src, TGSI_SWIZZLE_W));
         }
         ureg_release_temporary(ureg, src0);
      }
         break;

      case D3D10_SB_OPCODE_UDIV: {
         struct ureg_dst src0 = ureg_DECL_temporary(ureg);
         struct ureg_dst src1 = ureg_DECL_temporary(ureg);
         ureg_MOV(ureg, src0, translate_src_operand(sx, &opcode.src[0], OF_UINT));
         ureg_MOV(ureg, src1, translate_src_operand(sx, &opcode.src[1], OF_UINT));
         if (opcode.dst[0].base.type != D3D10_SB_OPERAND_TYPE_NULL) {
            ureg_UDIV(ureg,
                      translate_dst_operand(sx, &opcode.dst[0],
                                            opcode.saturate),
                      ureg_src(src0), ureg_src(src1));
         }
         if (opcode.dst[1].base.type != D3D10_SB_OPERAND_TYPE_NULL) {
            ureg_UMOD(ureg,
                      translate_dst_operand(sx, &opcode.dst[1],
                                            opcode.saturate),
                      ureg_src(src0), ureg_src(src1));
         }
         ureg_release_temporary(ureg, src0);
         ureg_release_temporary(ureg, src1);
      }
         break;
      case D3D10_SB_OPCODE_UMUL: {
         if (opcode.dst[0].base.type != D3D10_SB_OPERAND_TYPE_NULL) {
            ureg_UMUL_HI(ureg,
                         translate_dst_operand(sx, &opcode.dst[0],
                                               opcode.saturate),
                         translate_src_operand(sx, &opcode.src[0], OF_UINT),
                         translate_src_operand(sx, &opcode.src[1], OF_UINT));
         }
         if (opcode.dst[1].base.type != D3D10_SB_OPERAND_TYPE_NULL) {
            ureg_UMUL(ureg,
                      translate_dst_operand(sx, &opcode.dst[1],
                                            opcode.saturate),
                      translate_src_operand(sx, &opcode.src[0], OF_UINT),
                      translate_src_operand(sx, &opcode.src[1], OF_UINT));
         }
      }
         break;

      case D3D10_SB_OPCODE_DCL_RESOURCE:
      {
         unsigned target;
         unsigned res_index = opcode.dst[0].base.index[0].imm;
         assert(opcode.dst[0].base.index_dim == 1);
         assert(res_index < SHADER_MAX_RESOURCES);

         target = translate_resource_dimension(opcode.specific.dcl_resource_dimension);
         sx->resources[res_index].target = target;
         sx->resources[res_index].structured_stride = 0;
         sx->resources[res_index].raw = false;
         uint8_t mixed_return_component_mask = 0;
         for (unsigned component = 0; component < 4; component++) {
            if (opcode.dcl_resource_ret_type[component] ==
                D3D10_SB_RETURN_TYPE_MIXED)
               mixed_return_component_mask |= 1u << component;
            sx->resources[res_index].return_type[component] =
               trans_dcl_ret_type(sx,
                                  opcode.dcl_resource_ret_type[component]);
         }
         sx->resources[res_index].mixed_return_component_mask =
            mixed_return_component_mask;
         if (mixed_return_component_mask) {
            YTTRIUM_WARN("yttrium: WARNING: shader resource typeless return "
                         "mapped to float owner=d3d10umd stage=%u resource=%u "
                         "dimension=%u return_types=%u,%u,%u,%u\n",
                         parser.header.type, res_index,
                         opcode.specific.dcl_resource_dimension,
                         opcode.dcl_resource_ret_type[0],
                         opcode.dcl_resource_ret_type[1],
                         opcode.dcl_resource_ret_type[2],
                         opcode.dcl_resource_ret_type[3]);
         }
         if (sx->sampler_binding_map && target == TGSI_TEXTURE_BUFFER &&
             mixed_return_component_mask &&
             mixed_return_component_mask != TGSI_WRITEMASK_XYZW) {
            YTTRIUM_WARN("yttrium: shader TGSI translation failed "
                         "owner=d3d10umd-shader component=mixed-buffer-return "
                         "reason=partial-component-mask stage=%u resource=%u "
                         "mask=0x%x action=abort-translation\n",
                         parser.header.type, res_index,
                         mixed_return_component_mask);
            sx->translation_failed = true;
         }
         break;
      }

      case D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_TYPED:
      {
         unsigned target;
         unsigned image_index = opcode.dst[0].base.index[0].imm;
         assert(opcode.dst[0].base.index_dim == 1);
         assert(image_index < SHADER_MAX_IMAGES);

         target = translate_resource_dimension(opcode.specific.dcl_resource_dimension);
         sx->images[image_index].target = target;
         sx->images[image_index].structured_stride = 0;
         sx->images[image_index].raw = false;
         sx->images[image_index].format =
            trans_image_format(sx, opcode.dcl_resource_ret_type[0]);
         sx->images[image_index].reg =
            ureg_DECL_image(ureg, image_index, target,
                            sx->images[image_index].format, true, false);
         break;
      }

      case DX10_SM5_OPCODE_DCL_UAV_RAW:
      case DX10_SM5_OPCODE_DCL_UAV_STRUCTURED:
      {
         unsigned image_index = opcode.dst[0].base.index[0].imm;
         assert(opcode.dst[0].base.index_dim == 1);
         assert(image_index < SHADER_MAX_IMAGES);

         sx->images[image_index].target = TGSI_TEXTURE_BUFFER;
         sx->images[image_index].raw =
            opcode.type == DX10_SM5_OPCODE_DCL_UAV_RAW;
         sx->images[image_index].structured_stride = 0;
         if (opcode.type == DX10_SM5_OPCODE_DCL_UAV_STRUCTURED) {
            sx->images[image_index].structured_stride =
               opcode.specific.dcl_structured_stride;
            if (!sx->images[image_index].structured_stride ||
                (sx->images[image_index].structured_stride & 3)) {
               YTTRIUM_WARN("yttrium: shader translation rejected owner=d3d10umd "
                            "reason=invalid-structured-uav-dword-stride slot=%u stride=%u\n",
                            image_index, sx->images[image_index].structured_stride);
               sx->translation_failed = true;
               break;
            }
         }
         sx->images[image_index].format = PIPE_FORMAT_R32_UINT;
         sx->images[image_index].reg =
            ureg_DECL_image(ureg, image_index, TGSI_TEXTURE_BUFFER,
                            PIPE_FORMAT_R32_UINT, true, false);
         break;
      }

      case DX10_SM5_OPCODE_DCL_RESOURCE_RAW:
      case DX10_SM5_OPCODE_DCL_RESOURCE_STRUCTURED:
      {
         unsigned res_index = opcode.dst[0].base.index[0].imm;
         assert(opcode.dst[0].base.index_dim == 1);
         assert(res_index < SHADER_MAX_RESOURCES);

         sx->resources[res_index].target = TGSI_TEXTURE_BUFFER;
         sx->resources[res_index].raw =
            opcode.type == DX10_SM5_OPCODE_DCL_RESOURCE_RAW;
         sx->resources[res_index].structured_stride = 0;
         if (opcode.type == DX10_SM5_OPCODE_DCL_RESOURCE_STRUCTURED)
            sx->resources[res_index].structured_stride =
               opcode.specific.dcl_structured_stride;
         for (unsigned component = 0; component < 4; component++)
            sx->resources[res_index].return_type[component] =
               TGSI_RETURN_TYPE_UINT;
         if (!(sx->sampler_binding_map || use_old_tex_ops ||
               (st_debug & ST_DEBUG_OLD_TEX_OPS))) {
            sx->sv[res_index] =
               ureg_DECL_sampler_view(ureg, res_index, TGSI_TEXTURE_BUFFER,
                                      TGSI_RETURN_TYPE_UINT,
                                      TGSI_RETURN_TYPE_UINT,
                                      TGSI_RETURN_TYPE_UINT,
                                      TGSI_RETURN_TYPE_UINT);
         }
         break;
      }

      case D3D10_SB_OPCODE_DCL_CONSTANT_BUFFER: {
         unsigned num_constants = opcode.src[0].base.index[1].imm;

         assert(opcode.src[0].base.index[0].imm + 1 < PIPE_MAX_CONSTANT_BUFFERS);

         if (num_constants == 0) {
            num_constants = SHADER_MAX_CONSTS;
         } else {
            assert(num_constants <= SHADER_MAX_CONSTS);
         }

         ureg_DECL_constant2D(ureg,
                              0,
                              num_constants - 1,
                              opcode.src[0].base.index[0].imm + 1);
         break;
      }

      case D3D10_SB_OPCODE_DCL_SAMPLER:
         assert(opcode.dst[0].base.index_dim == 1);
         assert(opcode.dst[0].base.index[0].imm < SHADER_MAX_SAMPLERS);

         if (!sx->sampler_binding_map) {
            sx->samplers[opcode.dst[0].base.index[0].imm] =
               ureg_DECL_sampler(ureg,
                                 opcode.dst[0].base.index[0].imm);
         }
         break;

      case D3D10_SB_OPCODE_DCL_GS_OUTPUT_PRIMITIVE_TOPOLOGY:
         assert(parser.header.type == D3D10_SB_GEOMETRY_SHADER);

         switch (opcode.specific.dcl_gs_output_primitive_topology) {
         case D3D10_SB_PRIMITIVE_TOPOLOGY_POINTLIST:
            ureg_property(sx->ureg,
                          TGSI_PROPERTY_GS_OUTPUT_PRIM,
                          MESA_PRIM_POINTS);
            break;

         case D3D10_SB_PRIMITIVE_TOPOLOGY_LINESTRIP:
            ureg_property(sx->ureg,
                          TGSI_PROPERTY_GS_OUTPUT_PRIM,
                          MESA_PRIM_LINE_STRIP);
            break;

         case D3D10_SB_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP:
            ureg_property(sx->ureg,
                          TGSI_PROPERTY_GS_OUTPUT_PRIM,
                          MESA_PRIM_TRIANGLE_STRIP);
            break;

         default:
            assert(0);
         }
         break;

      case D3D10_SB_OPCODE_DCL_GS_INPUT_PRIMITIVE:
         assert(parser.header.type == D3D10_SB_GEOMETRY_SHADER);

         /* Figure out the second dimension of GS inputs.
          */
         switch (opcode.specific.dcl_gs_input_primitive) {
         case D3D10_SB_PRIMITIVE_POINT:
            declare_vertices_in(sx, 1);
            ureg_property(sx->ureg,
                          TGSI_PROPERTY_GS_INPUT_PRIM,
                          MESA_PRIM_POINTS);
            break;

         case D3D10_SB_PRIMITIVE_LINE:
            declare_vertices_in(sx, 2);
            ureg_property(sx->ureg,
                          TGSI_PROPERTY_GS_INPUT_PRIM,
                          MESA_PRIM_LINES);
            break;

         case D3D10_SB_PRIMITIVE_TRIANGLE:
            declare_vertices_in(sx, 3);
            ureg_property(sx->ureg,
                          TGSI_PROPERTY_GS_INPUT_PRIM,
                          MESA_PRIM_TRIANGLES);
            break;

         case D3D10_SB_PRIMITIVE_LINE_ADJ:
            declare_vertices_in(sx, 4);
            ureg_property(sx->ureg,
                          TGSI_PROPERTY_GS_INPUT_PRIM,
                          MESA_PRIM_LINES_ADJACENCY);
            break;

         case D3D10_SB_PRIMITIVE_TRIANGLE_ADJ:
            declare_vertices_in(sx, 6);
            ureg_property(sx->ureg,
                          TGSI_PROPERTY_GS_INPUT_PRIM,
                          MESA_PRIM_TRIANGLES_ADJACENCY);
            break;

         default:
            assert(0);
         }
         break;

      case D3D10_SB_OPCODE_DCL_MAX_OUTPUT_VERTEX_COUNT:
         assert(parser.header.type == D3D10_SB_GEOMETRY_SHADER);

         ureg_property(sx->ureg,
                       TGSI_PROPERTY_GS_MAX_OUTPUT_VERTICES,
                       opcode.specific.dcl_max_output_vertex_count);
         break;

      case D3D11_SB_OPCODE_DCL_GS_INSTANCE_COUNT:
         assert(parser.header.type == D3D10_SB_GEOMETRY_SHADER);

         ureg_property(sx->ureg,
                       TGSI_PROPERTY_GS_INVOCATIONS,
                       opcode.specific.dcl_gs_instance_count);
         break;

      case D3D10_SB_OPCODE_DCL_INPUT:
         if (parser.header.type == D3D10_SB_VERTEX_SHADER) {
            dcl_vs_input(sx, ureg, &opcode.dst[0]);
         } else if (parser.header.type == D3D10_SB_PIXEL_SHADER) {
            dcl_ps_input(sx, ureg, &opcode.dst[0],
                         D3D10_SB_INTERPOLATION_UNDEFINED);
         } else if (parser.header.type == DX10_SM5_COMPUTE_SHADER) {
            /* Compute system-value inputs are declared lazily as TGSI system values. */
         } else if (parser.header.type == DX11_SM5_HULL_SHADER ||
                    parser.header.type == DX11_SM5_DOMAIN_SHADER) {
            dcl_tess_input(sx, ureg, &opcode.dst[0]);
         } else {
            assert(parser.header.type == D3D10_SB_GEOMETRY_SHADER);
            dcl_gs_input(sx, ureg, &opcode.dst[0]);
         }
         break;

      case D3D10_SB_OPCODE_DCL_INPUT_SGV:
         assert(parser.header.type == D3D10_SB_VERTEX_SHADER);
         dcl_sgv_input(sx, ureg, &opcode.dst[0], opcode.dcl_siv_name);
         break;

      case D3D10_SB_OPCODE_DCL_INPUT_SIV:
         if (parser.header.type == DX11_SM5_DOMAIN_SHADER &&
             opcode.dst[0].base.type ==
                DX11_SM5_OPERAND_TYPE_INPUT_PATCH_CONSTANT) {
            dcl_tes_tess_factor_input(sx, ureg, &opcode.dst[0],
                                       opcode.dcl_siv_name);
         } else {
            assert(parser.header.type == D3D10_SB_GEOMETRY_SHADER);
            dcl_siv_input(sx, ureg, &opcode.dst[0], opcode.dcl_siv_name);
         }
         break;

      case D3D10_SB_OPCODE_DCL_INPUT_PS:
         assert(parser.header.type == D3D10_SB_PIXEL_SHADER);
         dcl_ps_input(sx, ureg, &opcode.dst[0],
                      opcode.specific.dcl_in_ps_interp);
         break;

      case D3D10_SB_OPCODE_DCL_INPUT_PS_SGV:
         assert(parser.header.type == D3D10_SB_PIXEL_SHADER);
         dcl_ps_sgv_input(sx, ureg, &opcode.dst[0],
                          opcode.dcl_siv_name);
         break;

      case D3D10_SB_OPCODE_DCL_INPUT_PS_SIV:
         assert(parser.header.type == D3D10_SB_PIXEL_SHADER);
         dcl_ps_siv_input(sx, ureg, &opcode.dst[0],
                          opcode.dcl_siv_name,
                          opcode.specific.dcl_in_ps_interp);
         break;

      case D3D10_SB_OPCODE_DCL_OUTPUT:
         if (parser.header.type == D3D10_SB_PIXEL_SHADER) {
            /* Pixel shader outputs. */
            if (is_output_depth_operand_type(opcode.dst[0].base.type)) {
               /* Depth output. */
               assert(opcode.dst[0].base.index_dim == 0);

               sx->output_depth = ureg_DECL_output_masked(ureg, TGSI_SEMANTIC_POSITION, 0, TGSI_WRITEMASK_Z, 0, 1);
               sx->output_depth = ureg_writemask(sx->output_depth, TGSI_WRITEMASK_Z);
            } else if (is_output_coverage_mask_operand_type(
                          opcode.dst[0].base.type)) {
               assert(opcode.dst[0].base.index_dim == 0);

               sx->output_coverage_mask = get_output_coverage_mask(sx);
            } else {
               /* Color outputs. */
               assert(opcode.dst[0].base.index_dim == 1);
               assert(opcode.dst[0].base.index[0].imm < SHADER_MAX_OUTPUTS);

               dcl_base_output(sx, ureg,
                               ureg_DECL_output(ureg,
                                                TGSI_SEMANTIC_COLOR,
                                                opcode.dst[0].base.index[0].imm),
                               &opcode.dst[0]);
            }
         } else if (parser.header.type == DX11_SM5_HULL_SHADER) {
            dcl_tcs_output(sx, ureg, &opcode.dst[0]);
         } else {
            assert(opcode.dst[0].base.index_dim == 1);
            assert(opcode.dst[0].base.index[0].imm < SHADER_MAX_OUTPUTS);

            const unsigned d3d_output = opcode.dst[0].base.index[0].imm;
            struct ureg_dst tgsi_output =
               ureg_DECL_output(ureg, TGSI_SEMANTIC_GENERIC, d3d_output);
            if (output_mapping)
               output_mapping[d3d_output] = tgsi_output.Index;
            dcl_base_output(sx, ureg,
                            tgsi_output,
                            &opcode.dst[0]);
         }
         break;

      case D3D10_SB_OPCODE_DCL_OUTPUT_SIV:
         assert(opcode.dst[0].base.index_dim == 1);
         assert(opcode.dst[0].base.index[0].imm < SHADER_MAX_OUTPUTS);

         if (parser.header.type == DX11_SM5_HULL_SHADER &&
             tcs_tess_factor_writemask(opcode.dcl_siv_name)) {
            dcl_tcs_tess_factor_output(sx, ureg, &opcode.dst[0],
                                       opcode.dcl_siv_name);
            break;
         }

         if (opcode.dcl_siv_name == D3D10_SB_NAME_CLIP_DISTANCE ||
             opcode.dcl_siv_name == D3D10_SB_NAME_CULL_DISTANCE) {
            const unsigned writemask =
               opcode.dst[0].mask >>
               D3D10_SB_OPERAND_4_COMPONENT_MASK_SHIFT;
            const unsigned component_count = util_bitcount(writemask);
            unsigned numcliporcull = sx->num_clip_distances_declared +
                                     sx->num_cull_distances_declared;
            if (sx->num_clip_distance_mappings <
                ARRAY_SIZE(sx->clip_distance_mapping)) {
               sx->clip_distance_mapping[sx->num_clip_distance_mappings].d3d =
                  opcode.dst[0].base.index[0].imm;
               sx->clip_distance_mapping[sx->num_clip_distance_mappings].tgsi =
                  numcliporcull / 4;
               sx->num_clip_distance_mappings++;
            } else {
               debug_printf("%s: too many combined clip/cull distance outputs d3d=%u count=%u max=%zu\n",
                            __func__, opcode.dst[0].base.index[0].imm,
                            numcliporcull,
                            ARRAY_SIZE(sx->clip_distance_mapping));
            }
            if (opcode.dcl_siv_name == D3D10_SB_NAME_CLIP_DISTANCE) {
               sx->num_clip_distances_declared += component_count;
               /* re-emit should be safe... */
               ureg_property(ureg, TGSI_PROPERTY_NUM_CLIPDIST_ENABLED,
                             sx->num_clip_distances_declared);
            } else {
               sx->num_cull_distances_declared += component_count;
               ureg_property(ureg, TGSI_PROPERTY_NUM_CULLDIST_ENABLED,
                             sx->num_cull_distances_declared);
            }
         }

         {
            const unsigned d3d_output = opcode.dst[0].base.index[0].imm;
            unsigned mask = opcode.dst[0].mask >>
               D3D10_SB_OPERAND_4_COMPONENT_MASK_SHIFT;
            if (parser.header.type == D3D10_SB_GEOMETRY_SHADER &&
                is_gs_scalar_output_name(opcode.dcl_siv_name)) {
               dcl_gs_scalar_output(sx, &opcode.dst[0]);
               mask = TGSI_WRITEMASK_X;
            }
            struct ureg_dst tgsi_output =
               ureg_DECL_output_masked(
                  ureg,
                  translate_system_name(sx, opcode.dcl_siv_name),
                  translate_semantic_index(sx, opcode.dcl_siv_name,
                                           &opcode.dst[0]),
                  mask,
                  0, 1);
            if (output_mapping)
               output_mapping[d3d_output] = tgsi_output.Index;
            dcl_base_output(sx, ureg, tgsi_output, &opcode.dst[0]);
         }
         break;

      case D3D10_SB_OPCODE_DCL_OUTPUT_SGV:
         assert(opcode.dst[0].base.index_dim == 1);
         assert(opcode.dst[0].base.index[0].imm < SHADER_MAX_OUTPUTS);

         {
            const unsigned d3d_output = opcode.dst[0].base.index[0].imm;
            const bool gs_scalar =
               parser.header.type == D3D10_SB_GEOMETRY_SHADER &&
               is_gs_scalar_output_name(opcode.dcl_siv_name);
            if (gs_scalar)
               dcl_gs_scalar_output(sx, &opcode.dst[0]);
            struct ureg_dst tgsi_output =
               ureg_DECL_output_masked(
                  ureg, translate_system_name(sx, opcode.dcl_siv_name),
                  0, gs_scalar ? TGSI_WRITEMASK_X : TGSI_WRITEMASK_XYZW,
                  0, 1);
            if (output_mapping)
               output_mapping[d3d_output] = tgsi_output.Index;
            dcl_base_output(sx, ureg, tgsi_output, &opcode.dst[0]);
         }
         break;

      case D3D10_SB_OPCODE_DCL_TEMPS:
         {
            uint i;

            assert(opcode.specific.dcl_num_temps + sx->declared_temps <=
                   SHADER_MAX_TEMPS);

            sx->temp_offset = sx->declared_temps;

            for (i = 0; i < opcode.specific.dcl_num_temps; i++) {
               sx->temps[sx->declared_temps + i] = ureg_DECL_temporary(ureg);
            }
            sx->declared_temps += opcode.specific.dcl_num_temps;
         }
         break;

      case D3D10_SB_OPCODE_DCL_INDEXABLE_TEMP:
         {
            uint i;

            /* XXX: Add true indexable temps to gallium.
             */

            assert(opcode.specific.dcl_indexable_temp.index <
                   SHADER_MAX_INDEXABLE_TEMPS);
            assert(opcode.specific.dcl_indexable_temp.count + sx->declared_temps <=
                   SHADER_MAX_TEMPS);

            sx->indexable_temp_offsets[opcode.specific.dcl_indexable_temp.index] =
               sx->declared_temps;

            for (i = 0; i < opcode.specific.dcl_indexable_temp.count; i++) {
               sx->temps[sx->declared_temps + i] = ureg_DECL_temporary(ureg);
            }
            sx->declared_temps += opcode.specific.dcl_indexable_temp.count;
         }
         break;
      case D3D10_SB_OPCODE_IF: {
         unsigned label = 0;
         if (opcode.specific.test_boolean == D3D10_SB_INSTRUCTION_TEST_ZERO) {
            struct ureg_src src =
               translate_src_operand(sx, &opcode.src[0], OF_INT);
            struct ureg_dst src_nz = ureg_DECL_temporary(ureg);
            ureg_USEQ(ureg, src_nz, src, ureg_imm1u(ureg, 0));
            ureg_UIF(ureg, ureg_src(src_nz), &label);
            ureg_release_temporary(ureg, src_nz);;
         } else {
            ureg_UIF(ureg, translate_src_operand(sx, &opcode.src[0], OF_INT), &label);
         }
      }
         break;
      case D3D10_SB_OPCODE_RETC:
      case D3D10_SB_OPCODE_CONTINUEC:
      case D3D10_SB_OPCODE_CALLC:
      case D3D10_SB_OPCODE_DISCARD:
      case D3D10_SB_OPCODE_BREAKC:
      {
         unsigned label = 0;
         assert(operand_is_scalar(&opcode.src[0]));
         if (opcode.specific.test_boolean == D3D10_SB_INSTRUCTION_TEST_ZERO) {
            struct ureg_src src =
               translate_src_operand(sx, &opcode.src[0], OF_INT);
            struct ureg_dst src_nz = ureg_DECL_temporary(ureg);
            ureg_USEQ(ureg, src_nz, src, ureg_imm1u(ureg, 0));
            ureg_UIF(ureg, ureg_src(src_nz), &label);
            ureg_release_temporary(ureg, src_nz);
         }
         else {
            ureg_UIF(ureg, translate_src_operand(sx, &opcode.src[0], OF_INT), &label);
         }
         switch (opcode.type) {
         case D3D10_SB_OPCODE_RETC:
            ureg_RET(ureg);
            break;
         case D3D10_SB_OPCODE_CONTINUEC:
            ureg_CONT(ureg);
            break;
         case D3D10_SB_OPCODE_CALLC: {
            unsigned label = opcode.src[1].base.index[0].imm;
            unsigned tgsi_token_label = 0;
            ureg_CAL(ureg, &tgsi_token_label);
            Shader_add_call(sx, label, tgsi_token_label);
         }
            break;
         case D3D10_SB_OPCODE_DISCARD:
            ureg_KILL(ureg);
            break;
         case D3D10_SB_OPCODE_BREAKC:
            ureg_BRK(ureg);
            break;
         default:
            assert(0);
            break;
         }
         ureg_ENDIF(ureg);
      }
         break;
      case D3D10_SB_OPCODE_RET:
         if (parser.header.type != DX11_SM5_HULL_SHADER)
            ureg_RET(ureg);
         break;
      case D3D10_SB_OPCODE_LABEL: {
         unsigned label = opcode.src[0].base.index[0].imm;
         unsigned tgsi_inst_no = 0;
         if (inside_sub) {
            ureg_ENDSUB(ureg);
         }
         tgsi_inst_no = ureg_get_instruction_number(ureg);
         ureg_BGNSUB(ureg);
         inside_sub = true;
         Shader_add_label(sx, label, tgsi_inst_no);
      }
         break;
      case D3D10_SB_OPCODE_CALL: {
         unsigned label = opcode.src[0].base.index[0].imm;
         unsigned tgsi_token_label = 0;
         ureg_CAL(ureg, &tgsi_token_label);
         Shader_add_call(sx, label, tgsi_token_label);
      }
         break;
      case D3D10_SB_OPCODE_EMIT:
         emit_gs_scalar_output_stores(sx);
         ureg_EMIT(ureg, ureg_imm1u(ureg, 0));
         break;
      case D3D10_SB_OPCODE_CUT:
         ureg_ENDPRIM(ureg, ureg_imm1u(ureg, 0));
         break;
      case D3D10_SB_OPCODE_EMITTHENCUT:
         emit_gs_scalar_output_stores(sx);
         ureg_EMIT(ureg, ureg_imm1u(ureg, 0));
         ureg_ENDPRIM(ureg, ureg_imm1u(ureg, 0));
         break;
      case D3D11_SB_OPCODE_EMIT_STREAM:
         emit_gs_scalar_output_stores(sx);
         ureg_EMIT(ureg, ureg_imm1u(ureg, opcode.specific.stream));
         break;
      case D3D11_SB_OPCODE_CUT_STREAM:
         ureg_ENDPRIM(ureg, ureg_imm1u(ureg, opcode.specific.stream));
         break;
      case D3D11_SB_OPCODE_EMITTHENCUT_STREAM:
         emit_gs_scalar_output_stores(sx);
         ureg_EMIT(ureg, ureg_imm1u(ureg, opcode.specific.stream));
         ureg_ENDPRIM(ureg, ureg_imm1u(ureg, opcode.specific.stream));
         break;
      case D3D11_SB_OPCODE_DCL_STREAM:
         break;
      case DX10_SM5_OPCODE_UBFE:
      case DX10_SM5_OPCODE_IBFE: {
         struct ureg_dst width = ureg_DECL_temporary(ureg);
         struct ureg_dst offset = ureg_DECL_temporary(ureg);
         struct ureg_dst dst = translate_dst_operand(sx, &opcode.dst[0],
                                                     opcode.saturate);
         struct ureg_src src[3];
         enum dx10_opcode_format format =
            opcode.type == DX10_SM5_OPCODE_IBFE ? OF_INT : OF_UINT;

         ureg_AND(ureg, width,
                  translate_src_operand(sx, &opcode.src[0], OF_UINT),
                  ureg_imm1u(ureg, 31));
         ureg_AND(ureg, offset,
                  translate_src_operand(sx, &opcode.src[1], OF_UINT),
                  ureg_imm1u(ureg, 31));

         src[0] = translate_src_operand(sx, &opcode.src[2], format);
         src[1] = ureg_src(offset);
         src[2] = ureg_src(width);
         ureg_insn(ureg,
                   opcode.type == DX10_SM5_OPCODE_IBFE ?
                   TGSI_OPCODE_IBFE : TGSI_OPCODE_UBFE,
                   &dst, 1, src, ARRAY_SIZE(src), 0);
         ureg_release_temporary(ureg, width);
         ureg_release_temporary(ureg, offset);
         break;
      }
      case DX10_SM5_OPCODE_BFI: {
         struct ureg_dst width = ureg_DECL_temporary(ureg);
         struct ureg_dst offset = ureg_DECL_temporary(ureg);

         ureg_AND(ureg, width,
                  translate_src_operand(sx, &opcode.src[0], OF_UINT),
                  ureg_imm1u(ureg, 31));
         ureg_AND(ureg, offset,
                  translate_src_operand(sx, &opcode.src[1], OF_UINT),
                  ureg_imm1u(ureg, 31));
         ureg_BFI(ureg,
                  translate_dst_operand(sx, &opcode.dst[0],
                                        opcode.saturate),
                  translate_src_operand(sx, &opcode.src[3], OF_UINT),
                  translate_src_operand(sx, &opcode.src[2], OF_UINT),
                  ureg_src(offset), ureg_src(width));
         ureg_release_temporary(ureg, width);
         ureg_release_temporary(ureg, offset);
         break;
      }
      case DX10_SM5_OPCODE_SWAPC: {
         struct ureg_dst src1_tmp = ureg_DECL_temporary(ureg);
         struct ureg_dst src2_tmp = ureg_DECL_temporary(ureg);
         struct ureg_src cond = translate_src_operand(sx, &opcode.src[0],
                                                      OF_UINT);
         struct ureg_src src1 = translate_src_operand(sx, &opcode.src[1],
                                                      OF_UINT);
         struct ureg_src src2 = translate_src_operand(sx, &opcode.src[2],
                                                      OF_UINT);

         ureg_MOV(ureg, src1_tmp, src1);
         ureg_MOV(ureg, src2_tmp, src2);
         src1 = ureg_src(src1_tmp);
         src2 = ureg_src(src2_tmp);

         ureg_UCMP(ureg,
                   translate_dst_operand(sx, &opcode.dst[0],
                                         opcode.saturate),
                   cond, src2, src1);
         ureg_UCMP(ureg,
                   translate_dst_operand(sx, &opcode.dst[1],
                                         opcode.saturate),
                   cond, src1, src2);
         ureg_release_temporary(ureg, src1_tmp);
         ureg_release_temporary(ureg, src2_tmp);
         break;
      }
      case D3D11_SB_OPCODE_LD_UAV_TYPED: {
         unsigned image_index = opcode.src[1].base.index[0].imm;
         struct ureg_dst dst = translate_dst_operand(sx, &opcode.dst[0],
                                                     opcode.saturate);
         struct ureg_dst load = ureg_DECL_temporary(ureg);
         struct ureg_src src[2];

         assert(opcode.src[1].base.index_dim == 1);
         assert(image_index < SHADER_MAX_IMAGES);

         src[0] = sx->images[image_index].reg;
         src[1] = translate_src_operand(sx, &opcode.src[0], OF_UINT);
         ureg_memory_insn(ureg, TGSI_OPCODE_LOAD,
                          &load, 1, src, ARRAY_SIZE(src), 0,
                          sx->images[image_index].target,
                          sx->images[image_index].format);
         ureg_MOV(ureg, dst, ureg_src(load));
         ureg_release_temporary(ureg, load);
         break;
      }
      case D3D11_SB_OPCODE_STORE_UAV_TYPED: {
         unsigned image_index = opcode.dst[0].base.index[0].imm;
         struct ureg_dst dst;
         struct ureg_src src[2];

         assert(opcode.dst[0].base.index_dim == 1);
         assert(image_index < SHADER_MAX_IMAGES);

         dst = ureg_dst(sx->images[image_index].reg);
         src[0] = translate_src_operand(sx, &opcode.src[0], OF_UINT);
         src[1] = translate_src_operand(sx, &opcode.src[1], OF_UINT);
         ureg_memory_insn(ureg, TGSI_OPCODE_STORE,
                          &dst, 1, src, ARRAY_SIZE(src), 0,
                          sx->images[image_index].target,
                          sx->images[image_index].format);
         break;
      }
      case DX10_SM5_OPCODE_STORE_RAW: {
         struct ureg_dst dst;
         struct ureg_dst index;
         struct ureg_dst coord;
         struct ureg_src data;
         struct ureg_src src[2];
         unsigned writemask =
            opcode.dst[0].mask >> D3D10_SB_OPERAND_4_COMPONENT_MASK_SHIFT;

         if (opcode.dst[0].base.type ==
             DX10_SM5_OPERAND_TYPE_THREAD_GROUP_SHARED_MEMORY) {
            struct ureg_dst byte_offset = ureg_dst_undef();

            dst = ureg_dst(get_tgsm_memory(sx));
            index = ureg_DECL_temporary(ureg);
            coord = ureg_DECL_temporary(ureg);
            ureg_MOV(ureg, index,
                     ureg_scalar(translate_src_operand(sx, &opcode.src[0], OF_UINT),
                                 TGSI_SWIZZLE_X));
            data = translate_src_operand(sx, &opcode.src[1], OF_UINT);
            for (unsigned i = 0; i < 4; i++) {
               if (!(writemask & (1u << i)))
                  continue;
               if (i) {
                  ureg_UADD(ureg, coord, ureg_src(index),
                            ureg_imm1u(ureg, i * 4));
                  src[0] = ureg_src(coord);
               } else {
                  src[0] = ureg_src(index);
               }
               src[0] = tgsm_byte_offset(sx, &opcode.dst[0].base,
                                         src[0], &byte_offset);
               src[1] = ureg_scalar(data, i);
               struct ureg_dst scalar_dst =
                  ureg_writemask(dst, TGSI_WRITEMASK_X);
               ureg_memory_insn(ureg, TGSI_OPCODE_STORE,
                                &scalar_dst, 1, src, ARRAY_SIZE(src), 0,
                                TGSI_TEXTURE_BUFFER, PIPE_FORMAT_R32_UINT);
               if (!ureg_dst_is_undef(byte_offset)) {
                  ureg_release_temporary(ureg, byte_offset);
                  byte_offset = ureg_dst_undef();
               }
            }
            ureg_release_temporary(ureg, coord);
            ureg_release_temporary(ureg, index);
            break;
         }

         unsigned image_index = opcode.dst[0].base.index[0].imm;

         assert(opcode.dst[0].base.index_dim == 1);
         assert(image_index < SHADER_MAX_IMAGES);

         dst = ureg_dst(sx->images[image_index].reg);
         index = ureg_DECL_temporary(ureg);
         coord = ureg_DECL_temporary(ureg);
         ureg_USHR(ureg, index,
                   ureg_scalar(translate_src_operand(sx, &opcode.src[0], OF_UINT),
                               TGSI_SWIZZLE_X),
                   ureg_imm1u(ureg, 2));
         data = translate_src_operand(sx, &opcode.src[1], OF_UINT);
         for (unsigned i = 0; i < 4; i++) {
            if (!(writemask & (1u << i)))
               continue;
            if (i) {
               ureg_UADD(ureg, coord, ureg_src(index), ureg_imm1u(ureg, i));
               src[0] = ureg_src(coord);
            } else {
               src[0] = ureg_src(index);
            }
            src[1] = ureg_scalar(data, i);
            ureg_memory_insn(ureg, TGSI_OPCODE_STORE,
                             &dst, 1, src, ARRAY_SIZE(src), 0,
                             sx->images[image_index].target,
                             sx->images[image_index].format);
         }
         ureg_release_temporary(ureg, coord);
         ureg_release_temporary(ureg, index);
         break;
      }
      case DX10_SM5_OPCODE_STORE_STRUCTURED: {
         struct ureg_dst dst;
         struct ureg_dst coord;
         struct ureg_dst byte_offset;
         struct ureg_src src[2];

         if (opcode.dst[0].base.type ==
             DX10_SM5_OPERAND_TYPE_THREAD_GROUP_SHARED_MEMORY) {
            unsigned slot = 0;
            struct ureg_dst store_offset = ureg_dst_undef();
            struct ureg_dst slot_offset = ureg_dst_undef();
            struct ureg_src data;
            unsigned writemask =
               opcode.dst[0].mask >> D3D10_SB_OPERAND_4_COMPONENT_MASK_SHIFT;

            if (!tgsm_operand_slot(&opcode.dst[0].base, &slot))
               break;

            dst = ureg_dst(get_tgsm_memory(sx));
            coord = ureg_DECL_temporary(ureg);
            byte_offset = ureg_DECL_temporary(ureg);
            ureg_UMUL(ureg, coord,
                      ureg_scalar(translate_src_operand(sx, &opcode.src[0], OF_UINT),
                                  TGSI_SWIZZLE_X),
                      ureg_imm1u(ureg, sx->tgsm[slot].structured_stride));
            ureg_UADD(ureg, coord, ureg_src(coord),
                      ureg_scalar(translate_src_operand(sx, &opcode.src[1], OF_UINT),
                                  TGSI_SWIZZLE_X));
            data = translate_src_operand(sx, &opcode.src[2], OF_UINT);

            for (unsigned i = 0; i < 4; i++) {
               if (!(writemask & (1u << i)))
                  continue;
               if (i) {
                  if (ureg_dst_is_undef(store_offset))
                     store_offset = ureg_DECL_temporary(ureg);
                  ureg_UADD(ureg, store_offset, ureg_src(coord),
                            ureg_imm1u(ureg, i * 4));
                  src[0] = ureg_src(store_offset);
               } else {
                  src[0] = ureg_src(coord);
               }

               src[0] = tgsm_byte_offset(sx, &opcode.dst[0].base,
                                         src[0], &slot_offset);
               src[1] = ureg_scalar(data, i);
               struct ureg_dst scalar_dst =
                  ureg_writemask(dst, TGSI_WRITEMASK_X);
               ureg_memory_insn(ureg, TGSI_OPCODE_STORE,
                                &scalar_dst, 1, src, ARRAY_SIZE(src), 0,
                                TGSI_TEXTURE_BUFFER, PIPE_FORMAT_R32_UINT);
               if (!ureg_dst_is_undef(slot_offset)) {
                  ureg_release_temporary(ureg, slot_offset);
                  slot_offset = ureg_dst_undef();
               }
            }

            if (!ureg_dst_is_undef(store_offset))
               ureg_release_temporary(ureg, store_offset);
            ureg_release_temporary(ureg, byte_offset);
            ureg_release_temporary(ureg, coord);
            break;
         }

         unsigned image_index = opcode.dst[0].base.index[0].imm;

         assert(opcode.dst[0].base.index_dim == 1);
         assert(image_index < SHADER_MAX_IMAGES);

         dst = ureg_dst(sx->images[image_index].reg);
         coord = ureg_DECL_temporary(ureg);
         byte_offset = ureg_DECL_temporary(ureg);
         struct ureg_dst store_offset = ureg_dst_undef();
         struct ureg_src data;
         unsigned writemask =
            opcode.dst[0].mask >> D3D10_SB_OPERAND_4_COMPONENT_MASK_SHIFT;

         ureg_UMUL(ureg, coord,
                   ureg_scalar(translate_src_operand(sx, &opcode.src[0], OF_UINT),
                               TGSI_SWIZZLE_X),
                   ureg_imm1u(ureg, sx->images[image_index].structured_stride));
         ureg_MOV(ureg, byte_offset,
                  ureg_scalar(translate_src_operand(sx, &opcode.src[1], OF_UINT),
                              TGSI_SWIZZLE_X));
         ureg_UADD(ureg, coord, ureg_src(coord), ureg_src(byte_offset));
         /* The D3D structure offset is in bytes; the R32_UINT view is not. */
         ureg_USHR(ureg, coord, ureg_src(coord), ureg_imm1u(ureg, 2));
         data = translate_src_operand(sx, &opcode.src[2], OF_UINT);
         for (unsigned i = 0; i < 4; i++) {
            if (!(writemask & (1u << i)))
               continue;
            if (i) {
               if (ureg_dst_is_undef(store_offset))
                  store_offset = ureg_DECL_temporary(ureg);
               ureg_UADD(ureg, store_offset, ureg_src(coord),
                         ureg_imm1u(ureg, i));
               src[0] = ureg_src(store_offset);
            } else {
               src[0] = ureg_src(coord);
            }
            src[1] = ureg_scalar(data, i);
            ureg_memory_insn(ureg, TGSI_OPCODE_STORE,
                             &dst, 1, src, ARRAY_SIZE(src), 0,
                             sx->images[image_index].target,
                             sx->images[image_index].format);
         }
         if (!ureg_dst_is_undef(store_offset))
            ureg_release_temporary(ureg, store_offset);
         ureg_release_temporary(ureg, byte_offset);
         ureg_release_temporary(ureg, coord);
         break;
      }
      case DX10_SM5_OPCODE_ATOMIC_AND:
      case DX10_SM5_OPCODE_ATOMIC_OR:
      case DX10_SM5_OPCODE_ATOMIC_XOR:
      case DX10_SM5_OPCODE_ATOMIC_CMP_STORE:
      case DX10_SM5_OPCODE_ATOMIC_IADD:
      case DX10_SM5_OPCODE_ATOMIC_IMAX:
      case DX10_SM5_OPCODE_ATOMIC_IMIN:
      case DX10_SM5_OPCODE_ATOMIC_UMAX:
      case DX10_SM5_OPCODE_ATOMIC_UMIN:
      case DX10_SM5_OPCODE_IMM_ATOMIC_ALLOC:
      case DX10_SM5_OPCODE_IMM_ATOMIC_CONSUME:
      case DX10_SM5_OPCODE_IMM_ATOMIC_IADD:
      case DX10_SM5_OPCODE_IMM_ATOMIC_AND:
      case DX10_SM5_OPCODE_IMM_ATOMIC_OR:
      case DX10_SM5_OPCODE_IMM_ATOMIC_XOR:
      case DX10_SM5_OPCODE_IMM_ATOMIC_EXCH:
      case DX10_SM5_OPCODE_IMM_ATOMIC_CMP_EXCH:
      case DX10_SM5_OPCODE_IMM_ATOMIC_IMAX:
      case DX10_SM5_OPCODE_IMM_ATOMIC_IMIN:
      case DX10_SM5_OPCODE_IMM_ATOMIC_UMAX:
      case DX10_SM5_OPCODE_IMM_ATOMIC_UMIN: {
         bool imm_atomic = opcode.num_dst == 2;
         unsigned image_dst = imm_atomic ? 1 : 0;
         unsigned image_index = opcode.dst[image_dst].base.index[0].imm;
         struct ureg_dst result = imm_atomic ?
            translate_dst_operand(sx, &opcode.dst[0], opcode.saturate) :
            ureg_DECL_temporary(ureg);
         struct ureg_dst coord_tmp;
         struct ureg_src src[4];
         unsigned tgsi_opcode;
         unsigned num_src = 3;
         enum dx10_opcode_format data_format = OF_UINT;
         bool release_coord_tmp = false;
         bool tgsm_atomic =
            opcode.dst[image_dst].base.type ==
            DX10_SM5_OPERAND_TYPE_THREAD_GROUP_SHARED_MEMORY;

         if (!tgsm_atomic) {
            assert(opcode.dst[image_dst].base.index_dim == 1);
            assert(image_index < SHADER_MAX_IMAGES);
         }

         switch (opcode.type) {
         case DX10_SM5_OPCODE_IMM_ATOMIC_ALLOC:
         case DX10_SM5_OPCODE_IMM_ATOMIC_CONSUME: {
            /* Use a typed 2D image, NOT the buffer-image atomic path that
             * previously hung the host.  The view owns its GPU counter and
             * the binding map keeps it separate from application UAVs.
             */
            assert(sx->counter_image_slots[image_index]);
            const bool consume =
               opcode.type == DX10_SM5_OPCODE_IMM_ATOMIC_CONSUME;
            const struct ureg_src counter_src[3] = {
               sx->counter_images[image_index],
               ureg_imm4u(ureg, 0, 0, 0, 0),
               ureg_imm1u(ureg, consume ? ~0u : 1),
            };
            struct ureg_dst old = ureg_DECL_temporary(ureg);
            ureg_memory_insn(ureg, TGSI_OPCODE_ATOMUADD,
                             &old, 1, counter_src, ARRAY_SIZE(counter_src),
                             0, TGSI_TEXTURE_2D, PIPE_FORMAT_R32_UINT);
            if (consume)
               ureg_UADD(ureg, result, ureg_scalar(ureg_src(old), 0),
                         ureg_imm1u(ureg, ~0u));
            else
               ureg_MOV(ureg, result, ureg_scalar(ureg_src(old), 0));
            ureg_release_temporary(ureg, old);
            break;
         }
         case DX10_SM5_OPCODE_ATOMIC_AND:
         case DX10_SM5_OPCODE_IMM_ATOMIC_AND:
            tgsi_opcode = TGSI_OPCODE_ATOMAND;
            break;
         case DX10_SM5_OPCODE_ATOMIC_OR:
         case DX10_SM5_OPCODE_IMM_ATOMIC_OR:
            tgsi_opcode = TGSI_OPCODE_ATOMOR;
            break;
         case DX10_SM5_OPCODE_ATOMIC_XOR:
         case DX10_SM5_OPCODE_IMM_ATOMIC_XOR:
            tgsi_opcode = TGSI_OPCODE_ATOMXOR;
            break;
         case DX10_SM5_OPCODE_IMM_ATOMIC_EXCH:
            tgsi_opcode = TGSI_OPCODE_ATOMXCHG;
            break;
         case DX10_SM5_OPCODE_ATOMIC_CMP_STORE:
         case DX10_SM5_OPCODE_IMM_ATOMIC_CMP_EXCH:
            tgsi_opcode = TGSI_OPCODE_ATOMCAS;
            num_src = 4;
            break;
         case DX10_SM5_OPCODE_ATOMIC_IADD:
         case DX10_SM5_OPCODE_IMM_ATOMIC_IADD:
            tgsi_opcode = TGSI_OPCODE_ATOMUADD;
            data_format = OF_INT;
            break;
         case DX10_SM5_OPCODE_ATOMIC_IMAX:
         case DX10_SM5_OPCODE_IMM_ATOMIC_IMAX:
            tgsi_opcode = TGSI_OPCODE_ATOMIMAX;
            data_format = OF_INT;
            break;
         case DX10_SM5_OPCODE_ATOMIC_IMIN:
         case DX10_SM5_OPCODE_IMM_ATOMIC_IMIN:
            tgsi_opcode = TGSI_OPCODE_ATOMIMIN;
            data_format = OF_INT;
            break;
         case DX10_SM5_OPCODE_ATOMIC_UMAX:
         case DX10_SM5_OPCODE_IMM_ATOMIC_UMAX:
            tgsi_opcode = TGSI_OPCODE_ATOMUMAX;
            break;
         case DX10_SM5_OPCODE_ATOMIC_UMIN:
         case DX10_SM5_OPCODE_IMM_ATOMIC_UMIN:
            tgsi_opcode = TGSI_OPCODE_ATOMUMIN;
            break;
         default:
            assert(0);
         }

         if (opcode.type == DX10_SM5_OPCODE_IMM_ATOMIC_ALLOC ||
             opcode.type == DX10_SM5_OPCODE_IMM_ATOMIC_CONSUME)
            break;

         if (tgsm_atomic) {
            unsigned slot = 0;
            struct ureg_dst byte_offset = ureg_dst_undef();

            src[0] = get_tgsm_memory(sx);
            coord_tmp = ureg_DECL_temporary(ureg);
            release_coord_tmp = true;
            if (tgsm_operand_slot(&opcode.dst[image_dst].base, &slot) &&
                sx->tgsm[slot].structured_stride) {
               struct ureg_src addr =
                  translate_src_operand(sx, &opcode.src[0], OF_UINT);
               ureg_UMUL(ureg, coord_tmp,
                         ureg_scalar(addr, TGSI_SWIZZLE_X),
                         ureg_imm1u(ureg, sx->tgsm[slot].structured_stride));
               ureg_UADD(ureg, coord_tmp, ureg_src(coord_tmp),
                         ureg_scalar(addr, TGSI_SWIZZLE_Y));
            } else {
               ureg_MOV(ureg, coord_tmp,
                        ureg_scalar(translate_src_operand(sx, &opcode.src[0], OF_UINT),
                                    TGSI_SWIZZLE_X));
            }
            src[1] = tgsm_byte_offset(sx, &opcode.dst[image_dst].base,
                                      ureg_src(coord_tmp), &byte_offset);
            src[2] = translate_src_operand(sx, &opcode.src[1], data_format);
            if (num_src == 4)
               src[3] = translate_src_operand(sx, &opcode.src[2], data_format);
            ureg_memory_insn(ureg, tgsi_opcode,
                             &result, 1, src, num_src, 0,
                             TGSI_TEXTURE_BUFFER, PIPE_FORMAT_R32_UINT);
            if (!ureg_dst_is_undef(byte_offset))
               ureg_release_temporary(ureg, byte_offset);
         } else {
            src[0] = sx->images[image_index].reg;
            struct ureg_src addr =
               translate_src_operand(sx, &opcode.src[0], OF_UINT);
            if (sx->images[image_index].target == TGSI_TEXTURE_BUFFER &&
                (sx->images[image_index].raw ||
                 sx->images[image_index].structured_stride)) {
               coord_tmp = ureg_DECL_temporary(ureg);
               release_coord_tmp = true;
               if (sx->images[image_index].structured_stride) {
                  /* Structured addresses are (element, byte offset). The
                   * R32_UINT backing view instead addresses dword texels. */
                  ureg_UMUL(ureg, coord_tmp,
                            ureg_scalar(addr, TGSI_SWIZZLE_X),
                            ureg_imm1u(ureg,
                               sx->images[image_index].structured_stride));
                  ureg_UADD(ureg, coord_tmp, ureg_src(coord_tmp),
                            ureg_scalar(addr, TGSI_SWIZZLE_Y));
               } else {
                  ureg_MOV(ureg, coord_tmp,
                           ureg_scalar(addr, TGSI_SWIZZLE_X));
               }
               ureg_USHR(ureg, coord_tmp, ureg_src(coord_tmp),
                          ureg_imm1u(ureg, 2));
               src[1] = ureg_src(coord_tmp);
            } else {
               /* Typed buffers and textures already use element coordinates. */
               src[1] = addr;
            }
            src[2] = translate_src_operand(sx, &opcode.src[1], data_format);
            if (num_src == 4)
               src[3] = translate_src_operand(sx, &opcode.src[2], data_format);
            ureg_memory_insn(ureg, tgsi_opcode,
                             &result, 1, src, num_src, 0,
                             sx->images[image_index].target,
                             sx->images[image_index].format);
         }
         if (release_coord_tmp)
            ureg_release_temporary(ureg, coord_tmp);
         if (!imm_atomic)
            ureg_release_temporary(ureg, result);
         break;
      }
      case D3D10_SB_OPCODE_DCL_INDEX_RANGE:
         if (opcode.dst[0].base.type == D3D10_SB_OPERAND_TYPE_OUTPUT)
            dcl_tcs_tess_factor_index_range(sx, &opcode);
         else
            dcl_input_index_range(sx, &opcode);
         break;
      case D3D10_SB_OPCODE_DCL_GLOBAL_FLAGS:
         if (opcode.specific.global_flags.force_early_depth_stencil) {
            ureg_property(ureg,
                          TGSI_PROPERTY_FS_EARLY_DEPTH_STENCIL,
                          1);
         }
         break;
      default:
         {
            uint i;
            struct ureg_dst dst[SHADER_MAX_DST_OPERANDS];
            struct ureg_src src[SHADER_MAX_SRC_OPERANDS];

            assert(ox->tgsi_opcode != TGSI_EXPAND);

            if (ox->tgsi_opcode == TGSI_LOG_UNSUPPORTED) {
               if (!shader_dumped) {
                  dx10_shader_dump_tokens(code);
                  shader_dumped = true;
               }
               YTTRIUM_WARN("yttrium: shader TGSI unsupported opcode d3d_type=%u; shader translation abandoned\n",
                            opcode.type);
               assert(ox->tgsi_opcode != TGSI_LOG_UNSUPPORTED);
               Shader_opcode_free(&opcode);
               goto fail;
            }

            if (try_emit_dynamic_structured_bufinfo_stride(sx, &opcode))
               break;

            /* Destination operands. */
            for (i = 0; i < opcode.num_dst; i++) {
               dst[i] = translate_dst_operand(sx, &opcode.dst[i],
                                              opcode.saturate);
            }

            /* Source operands. */
            for (i = 0; i < opcode.num_src; i++) {
               src[i] = translate_src_operand(sx, &opcode.src[i], ox->format);
            }

            /*
             * Re-route output depth to the Z channel.
             *
             * D3D's oDepth is scalar; TGSI carries depth in Z of the position
             * output.  So the destination is masked to Z, and for a
             * componentwise opcode the sources are scalarised to x, since Z of
             * the result then has to come from the value D3D put in x.
             *
             * Sources are deliberately left alone for anything that is not
             * componentwise.  A dot product already replicates its result
             * across all channels, so masking to Z is enough - scalarising it
             * would compute dot(a.xxx, b.xxx) instead, which is silently the
             * wrong number rather than a failure.
             *
             * This used to accept MOV only and abandon the whole shader for
             * anything else.  Superposition writes depth with a DIV, which
             * cost it every pixel shader that does so: no tokens, no module,
             * and every pipeline naming the shader failing to create.
             */
            if (is_output_depth_operand_type(opcode.dst[0].base.type)) {
               const struct tgsi_opcode_info *depth_info =
                  ox->tgsi_opcode < TGSI_OPCODE_LAST ?
                     tgsi_get_opcode_info(ox->tgsi_opcode) : NULL;

               dst[0] = ureg_writemask(dst[0], TGSI_WRITEMASK_Z);

               if (depth_info &&
                   depth_info->output_mode == TGSI_OUTPUT_COMPONENTWISE) {
                  for (i = 0; i < opcode.num_src; i++)
                     src[i] = ureg_scalar(src[i], TGSI_SWIZZLE_X);
               } else if (!depth_info ||
                          depth_info->output_mode != TGSI_OUTPUT_REPLICATE) {
                  /*
                   * Channel-dependent or otherwise unusual opcodes would need
                   * their own reasoning about which channel feeds Z, so keep
                   * refusing those rather than guessing.
                   */
                  if (!shader_dumped) {
                     dx10_shader_dump_tokens(code);
                     shader_dumped = true;
                  }
                  YTTRIUM_WARN("yttrium: shader TGSI output-depth write unsupported d3d_type=%u tgsi_opcode=%s(%u) output_mode=%d; shader translation abandoned\n",
                               opcode.type,
                               ox->tgsi_opcode < TGSI_OPCODE_LAST ?
                                  tgsi_get_opcode_name(ox->tgsi_opcode) :
                                  "(none)",
                               ox->tgsi_opcode,
                               depth_info ? (int)depth_info->output_mode : -1);
                  Shader_opcode_free(&opcode);
                  goto fail;
               }
            }

            ureg_insn(ureg,
                      ox->tgsi_opcode,
                      dst,
                      opcode.num_dst,
                      src,
                      opcode.num_src, 0);
         }
      }

      if (sx->translation_failed) {
         Shader_opcode_free(&opcode);
         goto fail;
      }

      emit_tcs_indexed_output_stores(sx);
      advance_dynamic_structured_bufinfo_stride(sx);
      Shader_opcode_free(&opcode);
   }

   if (parser.failed) {
      YTTRIUM_WARN("yttrium: shader TGSI translation failed "
                   "owner=d3d10umd-shader component=parser stage=%u "
                   "offset=%u shader_size=%u action=abort-translation\n",
                   parser.header.type,
                   (unsigned)(parser.curr - parser.code),
                   parser.header.size);
      goto fail;
   }

   /* Declarations may share a D3D register across generic and scalar system
    * semantics. Publish the whole-register SO mirror only after all of them
    * have been translated, so declaration order cannot overwrite it.
    */
   if (output_mapping && sx->has_gs_scalar_outputs) {
      for (unsigned output = 0; output < SHADER_MAX_OUTPUTS; ++output) {
         if (sx->outputs[output].gs_scalar_mask)
            output_mapping[output] = sx->outputs[output].gs_packed_output.Index;
      }
   }
   if (sx->has_gs_scalar_outputs &&
       ureg_get_nr_outputs(ureg) > PIPE_MAX_SHADER_OUTPUTS) {
      YTTRIUM_WARN("yttrium: shader TGSI translation failed "
                   "owner=d3d10umd-shader component=gs-scalar-output "
                   "reason=output-limit outputs=%u limit=%u "
                   "action=abort-translation\n",
                   ureg_get_nr_outputs(ureg), PIPE_MAX_SHADER_OUTPUTS);
      goto fail;
   }

   if (inside_sub) {
      ureg_ENDSUB(ureg);
   }

   tcs_end_patch_phase(sx, ureg);

   if (parser.header.type == DX11_SM5_HULL_SHADER)
      tcs_set_clip_cull_properties(sx, ureg);

   if (parser.header.type == DX11_SM5_DOMAIN_SHADER) {
      emit_tessellation_properties(
         ureg, tessellation_properties ?
                  tessellation_properties : &sx->tessellation_properties);
   }

   ureg_END(ureg);

   for (i = 0; i < sx->num_calls; ++i) {
      for (j = 0; j < sx->num_labels; ++j) {
         if (sx->calls[i].d3d_label == sx->labels[j].d3d_label) {
            ureg_fixup_label(sx->ureg,
                             sx->calls[i].tgsi_label_token,
                             sx->labels[j].tgsi_insn_no);
            break;
         }
      }
      ASSERT(j < sx->num_labels);
   }
   FREE(sx->labels);
   FREE(sx->calls);

   if (shared_memory_size)
      *shared_memory_size = sx->shared_memory_size;
   if (counter_image_slots)
      memcpy(counter_image_slots, sx->counter_image_slots,
             sizeof(sx->counter_image_slots));

   tokens = ureg_get_tokens(ureg, &nr_tokens);
   assert(tokens);
   ureg_destroy(ureg);

   if (st_debug & ST_DEBUG_TGSI) {
      tgsi_dump(tokens, 0);
   }

   FREE(sx);
   return tokens;

fail:
   FREE(sx->labels);
   FREE(sx->calls);
   if (ureg)
      ureg_destroy(ureg);
   FREE(sx);
   return NULL;
}
