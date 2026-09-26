/*
 * SPDX-FileCopyrightText: 2026 Ake Rehnman <ake.rehnman@gmail.com>
 * SPDX-License-Identifier: MPL-2.0
 */

#include "yttrium_state.h"

#include <string.h>

#include "util/blend.h"
#include "util/format/u_format.h"
#include "util/u_framebuffer.h"
#include "util/u_inlines.h"
#include "util/u_math.h"
#include "util/u_memory.h"

#include "yttrium_internal.h"
#include "yttrium_options.h"
#include "yttrium_pipeline.h"
#include "yttrium_shader.h"
#include "yttrium_trace.h"

static VkFormat
yttrium_vertex_format_to_vk(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8_UNORM:
      return VK_FORMAT_R8_UNORM;
   case PIPE_FORMAT_R8_SNORM:
      return VK_FORMAT_R8_SNORM;
   case PIPE_FORMAT_R8_USCALED:
      return VK_FORMAT_R8_USCALED;
   case PIPE_FORMAT_R8_SSCALED:
      return VK_FORMAT_R8_SSCALED;
   case PIPE_FORMAT_R8_UINT:
      return VK_FORMAT_R8_UINT;
   case PIPE_FORMAT_R8_SINT:
      return VK_FORMAT_R8_SINT;
   case PIPE_FORMAT_R8G8_UNORM:
      return VK_FORMAT_R8G8_UNORM;
   case PIPE_FORMAT_R8G8_SNORM:
      return VK_FORMAT_R8G8_SNORM;
   case PIPE_FORMAT_R8G8_USCALED:
      return VK_FORMAT_R8G8_USCALED;
   case PIPE_FORMAT_R8G8_SSCALED:
      return VK_FORMAT_R8G8_SSCALED;
   case PIPE_FORMAT_R8G8_UINT:
      return VK_FORMAT_R8G8_UINT;
   case PIPE_FORMAT_R8G8_SINT:
      return VK_FORMAT_R8G8_SINT;
   case PIPE_FORMAT_R8G8B8_UNORM:
      return VK_FORMAT_R8G8B8_UNORM;
   case PIPE_FORMAT_R8G8B8_SNORM:
      return VK_FORMAT_R8G8B8_SNORM;
   case PIPE_FORMAT_R8G8B8_USCALED:
      return VK_FORMAT_R8G8B8_USCALED;
   case PIPE_FORMAT_R8G8B8_SSCALED:
      return VK_FORMAT_R8G8B8_SSCALED;
   case PIPE_FORMAT_R8G8B8_UINT:
      return VK_FORMAT_R8G8B8_UINT;
   case PIPE_FORMAT_R8G8B8_SINT:
      return VK_FORMAT_R8G8B8_SINT;
   case PIPE_FORMAT_R8G8B8A8_UNORM:
      return VK_FORMAT_R8G8B8A8_UNORM;
   case PIPE_FORMAT_R8G8B8A8_SNORM:
      return VK_FORMAT_R8G8B8A8_SNORM;
   case PIPE_FORMAT_R8G8B8A8_USCALED:
      return VK_FORMAT_R8G8B8A8_USCALED;
   case PIPE_FORMAT_R8G8B8A8_SSCALED:
      return VK_FORMAT_R8G8B8A8_SSCALED;
   case PIPE_FORMAT_R8G8B8A8_UINT:
      return VK_FORMAT_R8G8B8A8_UINT;
   case PIPE_FORMAT_R8G8B8A8_SINT:
      return VK_FORMAT_R8G8B8A8_SINT;
   case PIPE_FORMAT_R10G10B10A2_UNORM:
      return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
   case PIPE_FORMAT_R10G10B10A2_UINT:
      return VK_FORMAT_A2B10G10R10_UINT_PACK32;
   case PIPE_FORMAT_B8G8R8A8_UNORM:
      return VK_FORMAT_B8G8R8A8_UNORM;
   case PIPE_FORMAT_B8G8R8A8_SNORM:
      return VK_FORMAT_B8G8R8A8_SNORM;
   case PIPE_FORMAT_B8G8R8A8_UINT:
      return VK_FORMAT_B8G8R8A8_UINT;
   case PIPE_FORMAT_B8G8R8A8_SINT:
      return VK_FORMAT_B8G8R8A8_SINT;
   case PIPE_FORMAT_R16_FLOAT:
      return VK_FORMAT_R16_SFLOAT;
   case PIPE_FORMAT_R16_UNORM:
      return VK_FORMAT_R16_UNORM;
   case PIPE_FORMAT_R16_SNORM:
      return VK_FORMAT_R16_SNORM;
   case PIPE_FORMAT_R16_USCALED:
      return VK_FORMAT_R16_USCALED;
   case PIPE_FORMAT_R16_SSCALED:
      return VK_FORMAT_R16_SSCALED;
   case PIPE_FORMAT_R16_UINT:
      return VK_FORMAT_R16_UINT;
   case PIPE_FORMAT_R16_SINT:
      return VK_FORMAT_R16_SINT;
   case PIPE_FORMAT_R16G16_FLOAT:
      return VK_FORMAT_R16G16_SFLOAT;
   case PIPE_FORMAT_R16G16_UNORM:
      return VK_FORMAT_R16G16_UNORM;
   case PIPE_FORMAT_R16G16_SNORM:
      return VK_FORMAT_R16G16_SNORM;
   case PIPE_FORMAT_R16G16_USCALED:
      return VK_FORMAT_R16G16_USCALED;
   case PIPE_FORMAT_R16G16_SSCALED:
      return VK_FORMAT_R16G16_SSCALED;
   case PIPE_FORMAT_R16G16_UINT:
      return VK_FORMAT_R16G16_UINT;
   case PIPE_FORMAT_R16G16_SINT:
      return VK_FORMAT_R16G16_SINT;
   case PIPE_FORMAT_R16G16B16_FLOAT:
      return VK_FORMAT_R16G16B16_SFLOAT;
   case PIPE_FORMAT_R16G16B16_UNORM:
      return VK_FORMAT_R16G16B16_UNORM;
   case PIPE_FORMAT_R16G16B16_SNORM:
      return VK_FORMAT_R16G16B16_SNORM;
   case PIPE_FORMAT_R16G16B16_USCALED:
      return VK_FORMAT_R16G16B16_USCALED;
   case PIPE_FORMAT_R16G16B16_SSCALED:
      return VK_FORMAT_R16G16B16_SSCALED;
   case PIPE_FORMAT_R16G16B16_UINT:
      return VK_FORMAT_R16G16B16_UINT;
   case PIPE_FORMAT_R16G16B16_SINT:
      return VK_FORMAT_R16G16B16_SINT;
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
   case PIPE_FORMAT_R16G16B16A16_UNORM:
      return VK_FORMAT_R16G16B16A16_UNORM;
   case PIPE_FORMAT_R16G16B16A16_SNORM:
      return VK_FORMAT_R16G16B16A16_SNORM;
   case PIPE_FORMAT_R16G16B16A16_USCALED:
      return VK_FORMAT_R16G16B16A16_USCALED;
   case PIPE_FORMAT_R16G16B16A16_SSCALED:
      return VK_FORMAT_R16G16B16A16_SSCALED;
   case PIPE_FORMAT_R16G16B16A16_UINT:
      return VK_FORMAT_R16G16B16A16_UINT;
   case PIPE_FORMAT_R16G16B16A16_SINT:
      return VK_FORMAT_R16G16B16A16_SINT;
   case PIPE_FORMAT_R32_FLOAT:
      return VK_FORMAT_R32_SFLOAT;
   case PIPE_FORMAT_R32_UINT:
      return VK_FORMAT_R32_UINT;
   case PIPE_FORMAT_R32_SINT:
      return VK_FORMAT_R32_SINT;
   case PIPE_FORMAT_R32G32_FLOAT:
      return VK_FORMAT_R32G32_SFLOAT;
   case PIPE_FORMAT_R32G32_UINT:
      return VK_FORMAT_R32G32_UINT;
   case PIPE_FORMAT_R32G32_SINT:
      return VK_FORMAT_R32G32_SINT;
   case PIPE_FORMAT_R32G32B32_FLOAT:
      return VK_FORMAT_R32G32B32_SFLOAT;
   case PIPE_FORMAT_R32G32B32_UINT:
      return VK_FORMAT_R32G32B32_UINT;
   case PIPE_FORMAT_R32G32B32_SINT:
      return VK_FORMAT_R32G32B32_SINT;
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
      return VK_FORMAT_R32G32B32A32_SFLOAT;
   case PIPE_FORMAT_R32G32B32A32_UINT:
      return VK_FORMAT_R32G32B32A32_UINT;
   case PIPE_FORMAT_R32G32B32A32_SINT:
      return VK_FORMAT_R32G32B32A32_SINT;
   default:
      return VK_FORMAT_UNDEFINED;
   }
}

static void
yttrium_build_vertex_input_state(struct yttrium_vertex_elements_state *state)
{
   int buffer_map[PIPE_MAX_ATTRIBS];

   if (!state)
      return;

   for (unsigned i = 0; i < PIPE_MAX_ATTRIBS; i++)
      buffer_map[i] = -1;

   state->vk_vertex_input_valid = true;

   for (unsigned i = 0; i < state->num_elements; i++) {
      const struct pipe_vertex_element *elem = &state->elements[i];
      const unsigned source_binding = elem->vertex_buffer_index;
      int mapped_binding;
      VkFormat vk_format;

      /* D3D input layouts are indexed by shader register and can have gaps.
       * A hole must not create a binding: its zero divisor could otherwise
       * conflict with real per-instance elements using the same buffer.
       */
      if (elem->src_format == PIPE_FORMAT_NONE)
         continue;

      if (source_binding >= PIPE_MAX_ATTRIBS) {
         state->vk_vertex_input_valid = false;
         continue;
      }

      mapped_binding = buffer_map[source_binding];
      if (mapped_binding < 0) {
         mapped_binding = state->num_bindings++;
         buffer_map[source_binding] = mapped_binding;
         state->binding_map[mapped_binding] = source_binding;
         state->binding_divisor[mapped_binding] = elem->instance_divisor;
         state->bindings[mapped_binding].binding = mapped_binding;
         state->bindings[mapped_binding].stride = elem->src_stride;
         state->bindings[mapped_binding].inputRate =
            elem->instance_divisor ? VK_VERTEX_INPUT_RATE_INSTANCE :
                                     VK_VERTEX_INPUT_RATE_VERTEX;
      } else {
         VkVertexInputBindingDescription *binding =
            &state->bindings[mapped_binding];
         const VkVertexInputRate input_rate =
            elem->instance_divisor ? VK_VERTEX_INPUT_RATE_INSTANCE :
                                     VK_VERTEX_INPUT_RATE_VERTEX;

         if (binding->stride != elem->src_stride ||
             binding->inputRate != input_rate ||
             state->binding_divisor[mapped_binding] !=
                elem->instance_divisor)
            state->vk_vertex_input_valid = false;
      }

      vk_format = yttrium_vertex_format_to_vk(elem->src_format);
      if (vk_format == VK_FORMAT_UNDEFINED)
         state->vk_vertex_input_valid = false;

      VkVertexInputAttributeDescription *attrib =
         &state->attribs[state->num_attribs++];
      attrib->location = i;
      attrib->binding = mapped_binding;
      attrib->format = vk_format;
      attrib->offset = elem->src_offset;
   }
}

void *
yttrium_create_vertex_elements_state(
   struct pipe_context *ctx,
   unsigned num_elements,
   const struct pipe_vertex_element *elements)
{
   struct yttrium_vertex_elements_state *state =
      CALLOC_STRUCT(yttrium_vertex_elements_state);
   if (!state)
      return NULL;

   state->num_elements = MIN2(num_elements, PIPE_MAX_ATTRIBS);
   if (state->num_elements && elements) {
      memcpy(state->elements, elements,
             state->num_elements * sizeof(state->elements[0]));
   }
   yttrium_build_vertex_input_state(state);

   YTTRIUM_LOG("yttrium: create vertex elements count=%u stored=%u vk_valid=%u vk_bindings=%u vk_attribs=%u\n",
                num_elements, state->num_elements,
                state->vk_vertex_input_valid, state->num_bindings,
                state->num_attribs);
   for (unsigned i = 0; i < state->num_bindings; i++) {
      const VkVertexInputBindingDescription *binding = &state->bindings[i];
      YTTRIUM_LOG("yttrium:   vi binding[%u] source_vb=%u stride=%u rate=%u divisor=%u\n",
                   i, state->binding_map[i], binding->stride,
                   binding->inputRate, state->binding_divisor[i]);
   }
   unsigned attrib_index = 0;
   for (unsigned i = 0; i < state->num_elements; i++) {
      const struct pipe_vertex_element *ve = &state->elements[i];
      const VkVertexInputAttributeDescription *attrib =
         attrib_index < state->num_attribs &&
         state->attribs[attrib_index].location == i ?
         &state->attribs[attrib_index++] : NULL;
      YTTRIUM_LOG("yttrium:   ve[%u] vb=%u offset=%u stride=%u format=%u divisor=%u vk_binding=%u vk_format=%u\n",
                   i, ve->vertex_buffer_index, ve->src_offset,
                   ve->src_stride, ve->src_format, ve->instance_divisor,
                   attrib ? attrib->binding : ~0u,
                   attrib ? attrib->format : VK_FORMAT_UNDEFINED);
   }

   return state;
}

void
yttrium_bind_vertex_elements_state(struct pipe_context *ctx, void *state)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   yctx->vertex_elements = (struct yttrium_vertex_elements_state *)state;
   yttrium_pipeline_state_changed(yctx);
   YTTRIUM_LOG("yttrium: bind vertex elements state=%p count=%u vk_valid=%u vk_bindings=%u\n",
                state,
                yctx->vertex_elements ? yctx->vertex_elements->num_elements : 0,
                yctx->vertex_elements ?
                   yctx->vertex_elements->vk_vertex_input_valid : 0,
                yctx->vertex_elements ? yctx->vertex_elements->num_bindings : 0);
}

void
yttrium_delete_vertex_elements_state(struct pipe_context *ctx, void *state)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   if (yctx->vertex_elements == state) {
      yctx->vertex_elements = NULL;
      yttrium_pipeline_state_changed(yctx);
   }
   YTTRIUM_LOG("yttrium: delete vertex elements state=%p\n", state);
   FREE(state);
}

void
yttrium_set_vertex_buffers(struct pipe_context *ctx,
                           unsigned count,
                           const struct pipe_vertex_buffer *buffers)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   const unsigned clamped = MIN2(count, PIPE_MAX_ATTRIBS);

   for (unsigned i = 0; i < PIPE_MAX_ATTRIBS; i++) {
      struct pipe_vertex_buffer *dst = &yctx->vertex_buffers[i];

      if (!dst->is_user_buffer)
         pipe_resource_reference(&dst->buffer.resource, NULL);

      memset(dst, 0, sizeof(*dst));
      dst->is_user_buffer = true;
   }

   yctx->num_vertex_buffers = clamped;

   for (unsigned i = 0; i < clamped; i++) {
      struct pipe_vertex_buffer *dst = &yctx->vertex_buffers[i];
      const struct pipe_vertex_buffer *src = buffers ? &buffers[i] : NULL;

      if (!src)
         continue;

      dst->buffer_offset = src->buffer_offset;
      dst->is_user_buffer = src->is_user_buffer;
      if (src->is_user_buffer) {
         dst->buffer.user = src->buffer.user;
      } else {
         dst->buffer.resource = NULL;
         pipe_resource_reference(&dst->buffer.resource, src->buffer.resource);
      }
   }

   YTTRIUM_LOG("yttrium: set vertex buffers count=%u stored=%u\n",
                count, yctx->num_vertex_buffers);
   for (unsigned i = 0; i < yctx->num_vertex_buffers; i++) {
      const struct pipe_vertex_buffer *vb = &yctx->vertex_buffers[i];
      const struct yttrium_resource *res =
         !vb->is_user_buffer && vb->buffer.resource ?
            yttrium_resource(vb->buffer.resource) : NULL;
      YTTRIUM_LOG("yttrium:   vb[%u] user=%u offset=%u resource=%p hAllocation=0x%lx size=0x%llx data=%p\n",
                   i, vb->is_user_buffer, vb->buffer_offset,
                   vb->is_user_buffer ? NULL : (void *)vb->buffer.resource,
                   res ? (unsigned long)res->hAllocation : 0,
                   res ? (unsigned long long)res->size : 0,
                   res ? res->data : NULL);
   }
}

void
yttrium_set_framebuffer_state(struct pipe_context *ctx,
                              const struct pipe_framebuffer_state *state)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   yttrium_pipeline_state_changed(yctx);
   util_copy_framebuffer_state(&yctx->fb, state);
   YTTRIUM_LOG("yttrium: set framebuffer nr_cbufs=%u size=%ux%u cbuf0=%p\n",
                yctx->fb.nr_cbufs, yctx->fb.width, yctx->fb.height,
                yctx->fb.nr_cbufs ? (void *)&yctx->fb.cbufs[0] : NULL);
}

void *
yttrium_create_rasterizer_state(struct pipe_context *ctx,
                                const struct pipe_rasterizer_state *state)
{
   struct yttrium_rasterizer_state *rast =
      CALLOC_STRUCT(yttrium_rasterizer_state);
   if (!rast)
      return NULL;

   rast->state = *state;
   YTTRIUM_LOG("yttrium: create rasterizer state=%p scissor=%u cull=0x%x front_ccw=%u fill=%u/%u\n",
                rast, rast->state.scissor, rast->state.cull_face,
                rast->state.front_ccw, rast->state.fill_front,
                rast->state.fill_back);
   return rast;
}

void
yttrium_bind_rasterizer_state(struct pipe_context *ctx, void *state)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   yctx->rasterizer = (struct yttrium_rasterizer_state *)state;
   yttrium_pipeline_state_changed(yctx);
   YTTRIUM_LOG("yttrium: bind rasterizer state=%p scissor=%u\n",
                state, yctx->rasterizer ? yctx->rasterizer->state.scissor : 0);
}

void
yttrium_delete_rasterizer_state(struct pipe_context *ctx, void *state)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   if (yctx->rasterizer == state) {
      yctx->rasterizer = NULL;
      yttrium_pipeline_state_changed(yctx);
   }
   YTTRIUM_LOG("yttrium: delete rasterizer state=%p\n", state);
   FREE(state);
}

static VkBlendFactor
yttrium_pipe_blend_factor(enum pipe_blendfactor factor)
{
   switch (factor) {
   case PIPE_BLENDFACTOR_ONE:
      return VK_BLEND_FACTOR_ONE;
   case PIPE_BLENDFACTOR_SRC_COLOR:
      return VK_BLEND_FACTOR_SRC_COLOR;
   case PIPE_BLENDFACTOR_SRC_ALPHA:
      return VK_BLEND_FACTOR_SRC_ALPHA;
   case PIPE_BLENDFACTOR_DST_ALPHA:
      return VK_BLEND_FACTOR_DST_ALPHA;
   case PIPE_BLENDFACTOR_DST_COLOR:
      return VK_BLEND_FACTOR_DST_COLOR;
   case PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE:
      return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
   case PIPE_BLENDFACTOR_CONST_COLOR:
      return VK_BLEND_FACTOR_CONSTANT_COLOR;
   case PIPE_BLENDFACTOR_CONST_ALPHA:
      return VK_BLEND_FACTOR_CONSTANT_ALPHA;
   case PIPE_BLENDFACTOR_SRC1_COLOR:
      return VK_BLEND_FACTOR_SRC1_COLOR;
   case PIPE_BLENDFACTOR_SRC1_ALPHA:
      return VK_BLEND_FACTOR_SRC1_ALPHA;
   case PIPE_BLENDFACTOR_ZERO:
      return VK_BLEND_FACTOR_ZERO;
   case PIPE_BLENDFACTOR_INV_SRC_COLOR:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
   case PIPE_BLENDFACTOR_INV_SRC_ALPHA:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
   case PIPE_BLENDFACTOR_INV_DST_ALPHA:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
   case PIPE_BLENDFACTOR_INV_DST_COLOR:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
   case PIPE_BLENDFACTOR_INV_CONST_COLOR:
      return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
   case PIPE_BLENDFACTOR_INV_CONST_ALPHA:
      return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
   case PIPE_BLENDFACTOR_INV_SRC1_COLOR:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
   case PIPE_BLENDFACTOR_INV_SRC1_ALPHA:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
   default:
      YTTRIUM_LOG("yttrium: unsupported blend factor %u, using ONE\n",
                   factor);
      return VK_BLEND_FACTOR_ONE;
   }
}

static VkBlendOp
yttrium_pipe_blend_op(enum pipe_blend_func func)
{
   switch (func) {
   case PIPE_BLEND_ADD:
      return VK_BLEND_OP_ADD;
   case PIPE_BLEND_SUBTRACT:
      return VK_BLEND_OP_SUBTRACT;
   case PIPE_BLEND_REVERSE_SUBTRACT:
      return VK_BLEND_OP_REVERSE_SUBTRACT;
   case PIPE_BLEND_MIN:
      return VK_BLEND_OP_MIN;
   case PIPE_BLEND_MAX:
      return VK_BLEND_OP_MAX;
   default:
      YTTRIUM_LOG("yttrium: unsupported blend op %u, using ADD\n", func);
      return VK_BLEND_OP_ADD;
   }
}

static VkLogicOp
yttrium_pipe_logic_op(unsigned logic_op)
{
   switch (logic_op) {
   case PIPE_LOGICOP_CLEAR:
      return VK_LOGIC_OP_CLEAR;
   case PIPE_LOGICOP_NOR:
      return VK_LOGIC_OP_NOR;
   case PIPE_LOGICOP_AND_INVERTED:
      return VK_LOGIC_OP_AND_INVERTED;
   case PIPE_LOGICOP_COPY_INVERTED:
      return VK_LOGIC_OP_COPY_INVERTED;
   case PIPE_LOGICOP_AND_REVERSE:
      return VK_LOGIC_OP_AND_REVERSE;
   case PIPE_LOGICOP_INVERT:
      return VK_LOGIC_OP_INVERT;
   case PIPE_LOGICOP_XOR:
      return VK_LOGIC_OP_XOR;
   case PIPE_LOGICOP_NAND:
      return VK_LOGIC_OP_NAND;
   case PIPE_LOGICOP_AND:
      return VK_LOGIC_OP_AND;
   case PIPE_LOGICOP_EQUIV:
      return VK_LOGIC_OP_EQUIVALENT;
   case PIPE_LOGICOP_NOOP:
      return VK_LOGIC_OP_NO_OP;
   case PIPE_LOGICOP_OR_INVERTED:
      return VK_LOGIC_OP_OR_INVERTED;
   case PIPE_LOGICOP_COPY:
      return VK_LOGIC_OP_COPY;
   case PIPE_LOGICOP_OR_REVERSE:
      return VK_LOGIC_OP_OR_REVERSE;
   case PIPE_LOGICOP_OR:
      return VK_LOGIC_OP_OR;
   case PIPE_LOGICOP_SET:
      return VK_LOGIC_OP_SET;
   default:
      YTTRIUM_LOG("yttrium: unsupported logic op %u, using COPY\n", logic_op);
      return VK_LOGIC_OP_COPY;
   }
}

static VkColorComponentFlags
yttrium_pipe_colormask(unsigned colormask)
{
   VkColorComponentFlags vk_mask = 0;

   if (colormask & PIPE_MASK_R)
      vk_mask |= VK_COLOR_COMPONENT_R_BIT;
   if (colormask & PIPE_MASK_G)
      vk_mask |= VK_COLOR_COMPONENT_G_BIT;
   if (colormask & PIPE_MASK_B)
      vk_mask |= VK_COLOR_COMPONENT_B_BIT;
   if (colormask & PIPE_MASK_A)
      vk_mask |= VK_COLOR_COMPONENT_A_BIT;

   return vk_mask;
}

static void
yttrium_init_blend_state(struct yttrium_blend_state *blend)
{
   blend->logic_op = yttrium_pipe_logic_op(blend->state.logicop_func);
   for (uint32_t i = 0; i < PIPE_MAX_COLOR_BUFS; i++) {
      const uint32_t rt_index = blend->state.independent_blend_enable ? i : 0;
      const struct pipe_rt_blend_state *rt = &blend->state.rt[rt_index];

      blend->rt_blend_enable[i] = rt->blend_enable ? VK_TRUE : VK_FALSE;
      blend->rt_color_write_mask[i] = yttrium_pipe_colormask(rt->colormask);
      blend->rt_src_color_blend_factor[i] =
         yttrium_pipe_blend_factor(rt->rgb_src_factor);
      blend->rt_dst_color_blend_factor[i] =
         yttrium_pipe_blend_factor(rt->rgb_dst_factor);
      blend->rt_color_blend_op[i] = yttrium_pipe_blend_op(rt->rgb_func);
      blend->rt_src_alpha_blend_factor[i] =
         yttrium_pipe_blend_factor(util_blendfactor_to_alpha(rt->alpha_src_factor));
      blend->rt_dst_alpha_blend_factor[i] =
         yttrium_pipe_blend_factor(util_blendfactor_to_alpha(rt->alpha_dst_factor));
      blend->rt_alpha_blend_op[i] = yttrium_pipe_blend_op(rt->alpha_func);
   }
}

void *
yttrium_create_blend_state(struct pipe_context *ctx,
                           const struct pipe_blend_state *state)
{
   struct yttrium_blend_state *blend =
      CALLOC_STRUCT(yttrium_blend_state);
   if (!blend)
      return NULL;

   blend->state = *state;
   yttrium_init_blend_state(blend);
   YTTRIUM_LOG("yttrium: create blend state=%p rt0 enable=%u rgb=(%u,%u,%u) alpha=(%u,%u,%u) mask=0x%x indep=%u logic=%u atoc=%u\n",
                blend, blend->state.rt[0].blend_enable,
                blend->state.rt[0].rgb_func,
                blend->state.rt[0].rgb_src_factor,
                blend->state.rt[0].rgb_dst_factor,
                blend->state.rt[0].alpha_func,
                blend->state.rt[0].alpha_src_factor,
                blend->state.rt[0].alpha_dst_factor,
                blend->state.rt[0].colormask,
                blend->state.independent_blend_enable,
                blend->state.logicop_enable,
                blend->state.alpha_to_coverage);
   return blend;
}

void
yttrium_bind_blend_state(struct pipe_context *ctx, void *state)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   yctx->blend = (struct yttrium_blend_state *)state;
   yttrium_pipeline_state_changed(yctx);
   YTTRIUM_LOG("yttrium: bind blend state=%p enable=%u mask=0x%x\n",
                state,
                yctx->blend ? yctx->blend->state.rt[0].blend_enable : 0,
                yctx->blend ? yctx->blend->state.rt[0].colormask : 0xf);
}

void
yttrium_delete_blend_state(struct pipe_context *ctx, void *state)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   if (yctx->blend == state) {
      yctx->blend = NULL;
      yttrium_pipeline_state_changed(yctx);
   }
   YTTRIUM_LOG("yttrium: delete blend state=%p\n", state);
   FREE(state);
}

void *
yttrium_create_dsa_state(
   struct pipe_context *ctx,
   const struct pipe_depth_stencil_alpha_state *state)
{
   struct yttrium_dsa_state *dsa = CALLOC_STRUCT(yttrium_dsa_state);
   if (!dsa)
      return NULL;

   if (state)
      dsa->state = *state;

   YTTRIUM_LOG("yttrium: create dsa state=%p depth=(enable=%u write=%u func=%u bounds=%u %f..%f) stencil0=(enable=%u func=%u fail=%u zpass=%u zfail=%u mask=0x%x write=0x%x) stencil1=(enable=%u func=%u fail=%u zpass=%u zfail=%u mask=0x%x write=0x%x) alpha=(enable=%u func=%u ref=%f)\n",
                dsa,
                dsa->state.depth_enabled,
                dsa->state.depth_writemask,
                dsa->state.depth_func,
                dsa->state.depth_bounds_test,
                dsa->state.depth_bounds_min,
                dsa->state.depth_bounds_max,
                dsa->state.stencil[0].enabled,
                dsa->state.stencil[0].func,
                dsa->state.stencil[0].fail_op,
                dsa->state.stencil[0].zpass_op,
                dsa->state.stencil[0].zfail_op,
                dsa->state.stencil[0].valuemask,
                dsa->state.stencil[0].writemask,
                dsa->state.stencil[1].enabled,
                dsa->state.stencil[1].func,
                dsa->state.stencil[1].fail_op,
                dsa->state.stencil[1].zpass_op,
                dsa->state.stencil[1].zfail_op,
                dsa->state.stencil[1].valuemask,
                dsa->state.stencil[1].writemask,
                dsa->state.alpha_enabled,
                dsa->state.alpha_func,
                dsa->state.alpha_ref_value);

   return dsa;
}

void
yttrium_bind_dsa_state(struct pipe_context *ctx, void *state)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   yctx->dsa = (struct yttrium_dsa_state *)state;
   yttrium_pipeline_state_changed(yctx);
   YTTRIUM_LOG("yttrium: bind dsa state=%p depth=%u stencil=%u/%u alpha=%u\n",
                state,
                yctx->dsa ? yctx->dsa->state.depth_enabled : 0,
                yctx->dsa ? yctx->dsa->state.stencil[0].enabled : 0,
                yctx->dsa ? yctx->dsa->state.stencil[1].enabled : 0,
                yctx->dsa ? yctx->dsa->state.alpha_enabled : 0);
}

void
yttrium_delete_dsa_state(struct pipe_context *ctx, void *state)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   if (!state)
      return;

   if (yctx->dsa == state)
      yctx->dsa = NULL;
   yttrium_pipeline_state_changed(yctx);

   YTTRIUM_LOG("yttrium: delete dsa state=%p\n", state);
   FREE(state);
}

void
yttrium_set_stencil_ref(struct pipe_context *ctx,
                        const struct pipe_stencil_ref state)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   yctx->stencil_ref = state;
   yttrium_pipeline_state_changed(yctx);
   YTTRIUM_LOG("yttrium: set stencil ref front=%u back=%u\n",
                yctx->stencil_ref.ref_value[0],
                yctx->stencil_ref.ref_value[1]);
}

void
yttrium_set_blend_color(struct pipe_context *ctx,
                        const struct pipe_blend_color *state)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   if (state)
      yctx->blend_color = *state;
   else
      memset(&yctx->blend_color, 0, sizeof(yctx->blend_color));

   YTTRIUM_LOG("yttrium: set blend color=(%f,%f,%f,%f)\n",
                yctx->blend_color.color[0], yctx->blend_color.color[1],
                yctx->blend_color.color[2], yctx->blend_color.color[3]);
}

void
yttrium_set_sample_mask(struct pipe_context *ctx, unsigned sample_mask)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   yctx->sample_mask = sample_mask;
   yttrium_pipeline_state_changed(yctx);
   YTTRIUM_LOG("yttrium: set sample mask=0x%x\n", yctx->sample_mask);
}

void
yttrium_set_viewport_states(struct pipe_context *ctx,
                            unsigned start_slot,
                            unsigned num_viewports,
                            const struct pipe_viewport_state *states)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   if (start_slot >= PIPE_MAX_VIEWPORTS) {
      YTTRIUM_LOG("yttrium: ignored viewport state start=%u count=%u\n",
                   start_slot, num_viewports);
      return;
   }

   const unsigned count =
      MIN2(num_viewports, PIPE_MAX_VIEWPORTS - start_slot);
   if (count && states)
      memcpy(&yctx->viewports[start_slot], states,
             sizeof(*states) * count);
   else if (count)
      memset(&yctx->viewports[start_slot], 0,
             sizeof(yctx->viewports[0]) * count);

   if (start_slot == 0)
      yctx->num_viewports = count;
   else
      yctx->num_viewports = MAX2(yctx->num_viewports, start_slot + count);

   if (count) {
      YTTRIUM_LOG("yttrium: set viewport start=%u count=%u first scale=(%f,%f,%f) translate=(%f,%f,%f)\n",
                   start_slot, count,
                   yctx->viewports[start_slot].scale[0],
                   yctx->viewports[start_slot].scale[1],
                   yctx->viewports[start_slot].scale[2],
                   yctx->viewports[start_slot].translate[0],
                   yctx->viewports[start_slot].translate[1],
                   yctx->viewports[start_slot].translate[2]);
   } else {
      memset(yctx->viewports, 0, sizeof(yctx->viewports));
      yctx->num_viewports = 0;
      YTTRIUM_LOG("yttrium: clear viewport state start=%u count=%u\n",
                   start_slot, num_viewports);
   }
}

void
yttrium_set_scissor_states(struct pipe_context *ctx,
                           unsigned start_slot,
                           unsigned num_scissors,
                           const struct pipe_scissor_state *states)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   if (start_slot >= PIPE_MAX_VIEWPORTS) {
      YTTRIUM_LOG("yttrium: ignored scissor state start=%u count=%u\n",
                   start_slot, num_scissors);
      return;
   }

   const unsigned count =
      MIN2(num_scissors, PIPE_MAX_VIEWPORTS - start_slot);
   if (count && states)
      memcpy(&yctx->scissors[start_slot], states,
             sizeof(*states) * count);
   else if (count)
      memset(&yctx->scissors[start_slot], 0,
             sizeof(yctx->scissors[0]) * count);

   if (start_slot == 0)
      yctx->num_scissors = count;
   else
      yctx->num_scissors = MAX2(yctx->num_scissors, start_slot + count);

   if (count) {
      YTTRIUM_LOG("yttrium: set scissor start=%u count=%u first %u,%u - %u,%u\n",
                   start_slot, count,
                   yctx->scissors[start_slot].minx,
                   yctx->scissors[start_slot].miny,
                   yctx->scissors[start_slot].maxx,
                   yctx->scissors[start_slot].maxy);
   } else {
      memset(yctx->scissors, 0, sizeof(yctx->scissors));
      yctx->num_scissors = 0;
      YTTRIUM_LOG("yttrium: clear scissor state start=%u count=%u\n",
                   start_slot, num_scissors);
   }
}

void
yttrium_set_constant_buffer(struct pipe_context *ctx,
                            mesa_shader_stage shader,
                            uint index,
                            const struct pipe_constant_buffer *cb)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   if (shader >= MESA_SHADER_STAGES || index >= PIPE_MAX_CONSTANT_BUFFERS) {
      YTTRIUM_LOG("yttrium: ignored constant buffer shader=%u index=%u\n",
                   shader, index);
      return;
   }

   struct yttrium_constant_buffer *slot =
      &yctx->constant_buffers[shader][index];
   pipe_resource_reference(&slot->buffer, NULL);
   slot->buffer_offset = 0;
   slot->buffer_size = 0;
   slot->user_buffer = NULL;

   if (cb) {
      pipe_resource_reference(&slot->buffer, cb->buffer);
      slot->buffer_offset = cb->buffer_offset;
      slot->buffer_size = cb->buffer_size;
      slot->user_buffer = cb->user_buffer;
   }

   const struct yttrium_resource *res =
      slot->buffer ? yttrium_resource(slot->buffer) : NULL;
   YTTRIUM_LOG("yttrium: set %s constant buffer[%u] resource=%p hAllocation=0x%lx offset=%u size=%u user=%p\n",
                yttrium_shader_stage_name(shader),
                index,
                (void *)slot->buffer,
                res ? (unsigned long)res->hAllocation : 0,
                slot->buffer_offset,
                slot->buffer_size,
                slot->user_buffer);
}

void *
yttrium_create_sampler_state(struct pipe_context *ctx,
                             const struct pipe_sampler_state *state)
{
   struct yttrium_sampler_state *sampler =
      CALLOC_STRUCT(yttrium_sampler_state);
   if (!sampler)
      return NULL;

   if (state)
      sampler->state = *state;

   YTTRIUM_LOG("yttrium: create sampler state=%p wrap=(%u,%u,%u) filter=(%u,%u,%u) lod=(%f,%f,%f) aniso=%u compare=(%u,%u)\n",
                sampler,
                sampler->state.wrap_s,
                sampler->state.wrap_t,
                sampler->state.wrap_r,
                sampler->state.min_img_filter,
                sampler->state.min_mip_filter,
                sampler->state.mag_img_filter,
                sampler->state.lod_bias,
                sampler->state.min_lod,
                sampler->state.max_lod,
                sampler->state.max_anisotropy,
                sampler->state.compare_mode,
                sampler->state.compare_func);

   return sampler;
}

void
yttrium_bind_sampler_states(struct pipe_context *ctx,
                            mesa_shader_stage shader,
                            unsigned start_slot,
                            unsigned num_samplers,
                            void **samplers)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   if (shader >= MESA_SHADER_STAGES || start_slot >= PIPE_MAX_SAMPLERS) {
      YTTRIUM_LOG("yttrium: ignored sampler state bind shader=%u start=%u count=%u\n",
                   shader, start_slot, num_samplers);
      return;
   }

   unsigned end = MIN2(start_slot + num_samplers, PIPE_MAX_SAMPLERS);
   unsigned bound = 0;
   bool changed = false;

   for (unsigned slot = start_slot; slot < end; slot++) {
      struct yttrium_sampler_state *sampler =
         samplers ? (struct yttrium_sampler_state *)samplers[slot - start_slot]
                  : NULL;
      changed |= yctx->sampler_states[shader][slot] != sampler;
      yctx->sampler_states[shader][slot] = sampler;
      if (sampler)
         bound++;
   }

   if (changed)
      yttrium_pipeline_fast_state_changed(yctx);

   YTTRIUM_LOG("yttrium: bind %s sampler states start=%u count=%u stored=%u bound=%u\n",
                yttrium_shader_stage_name(shader),
                start_slot,
                num_samplers,
                end - start_slot,
                bound);
}

void
yttrium_delete_sampler_state(struct pipe_context *ctx, void *state)
{
   struct yttrium_context *yctx = yttrium_context(ctx);
   bool changed = false;

   if (!state)
      return;

   for (unsigned stage = 0; stage < MESA_SHADER_STAGES; stage++) {
      for (unsigned slot = 0; slot < PIPE_MAX_SAMPLERS; slot++) {
         if (yctx->sampler_states[stage][slot] == state) {
            yctx->sampler_states[stage][slot] = NULL;
            changed = true;
         }
      }
   }

   if (changed)
      yttrium_pipeline_fast_state_changed(yctx);

   YTTRIUM_LOG("yttrium: delete sampler state=%p\n", state);
   FREE(state);
}

struct pipe_sampler_view *
yttrium_create_sampler_view(struct pipe_context *ctx,
                            struct pipe_resource *texture,
                            const struct pipe_sampler_view *state)
{
   struct pipe_sampler_view *view = CALLOC_STRUCT(pipe_sampler_view);
   if (!view)
      return NULL;

   if (state)
      *view = *state;

   view->texture = NULL;
   pipe_resource_reference(&view->texture, texture);
   pipe_reference_init(&view->reference, 1);
   view->context = ctx;

   const struct yttrium_resource *res =
      texture ? yttrium_resource(texture) : NULL;
   YTTRIUM_LOG("yttrium: create sampler view=%p texture=%p hAllocation=0x%lx format=%u target=%u levels=%u-%u layers=%u-%u swizzle=(%u,%u,%u,%u)\n",
                view,
                (void *)texture,
                res ? (unsigned long)res->hAllocation : 0,
                view->format,
                view->target,
                view->u.tex.first_level,
                view->u.tex.last_level,
                view->u.tex.first_layer,
                view->u.tex.last_layer,
                view->swizzle_r,
                view->swizzle_g,
                view->swizzle_b,
                view->swizzle_a);

   return view;
}

void
yttrium_set_sampler_views(struct pipe_context *ctx,
                          mesa_shader_stage shader,
                          unsigned start_slot,
                          unsigned num_views,
                          unsigned unbind_num_trailing_slots,
                          struct pipe_sampler_view **views)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   if (shader >= MESA_SHADER_STAGES ||
       start_slot >= PIPE_MAX_SHADER_SAMPLER_VIEWS) {
      YTTRIUM_LOG("yttrium: ignored sampler views bind shader=%u start=%u count=%u trailing=%u\n",
                   shader, start_slot, num_views, unbind_num_trailing_slots);
      return;
   }

   unsigned end = MIN2(start_slot + num_views,
                       PIPE_MAX_SHADER_SAMPLER_VIEWS);
   unsigned trailing_end = MIN2(end + unbind_num_trailing_slots,
                                PIPE_MAX_SHADER_SAMPLER_VIEWS);
   unsigned bound = 0;
   bool changed = false;

   for (unsigned slot = start_slot; slot < end; slot++) {
      struct pipe_sampler_view *view = views ? views[slot - start_slot] : NULL;
      changed |= yctx->sampler_views[shader][slot] != view;
      pipe_sampler_view_reference(&yctx->sampler_views[shader][slot], view);
      if (view)
         bound++;
   }

   for (unsigned slot = end; slot < trailing_end; slot++) {
      changed |= yctx->sampler_views[shader][slot] != NULL;
      pipe_sampler_view_reference(&yctx->sampler_views[shader][slot], NULL);
   }

   if (changed)
      yttrium_pipeline_fast_state_changed(yctx);

   YTTRIUM_LOG("yttrium: set %s sampler views start=%u count=%u stored=%u trailing=%u bound=%u\n",
                yttrium_shader_stage_name(shader),
                start_slot,
                num_views,
                end - start_slot,
                trailing_end - end,
                bound);
}

void
yttrium_set_shader_images(struct pipe_context *ctx,
                          mesa_shader_stage shader,
                          unsigned start_slot,
                          unsigned count,
                          unsigned unbind_num_trailing_slots,
                          const struct pipe_image_view *images)
{
   struct yttrium_context *yctx = yttrium_context(ctx);

   if (shader >= MESA_SHADER_STAGES || start_slot >= PIPE_MAX_SHADER_IMAGES)
      return;

   const unsigned end = MIN2(start_slot + count, PIPE_MAX_SHADER_IMAGES);
   const unsigned trailing_end =
      MIN2(end + unbind_num_trailing_slots, PIPE_MAX_SHADER_IMAGES);

   for (unsigned slot = start_slot; slot < end; slot++) {
      const struct pipe_image_view *image =
         images ? &images[slot - start_slot] : NULL;
      util_copy_image_view(&yctx->shader_images[shader][slot], image);
   }

   for (unsigned slot = end; slot < trailing_end; slot++)
      util_copy_image_view(&yctx->shader_images[shader][slot], NULL);

   yctx->num_shader_images[shader] =
      MAX2(yctx->num_shader_images[shader], trailing_end);
   while (yctx->num_shader_images[shader] &&
          !yctx->shader_images[shader]
                              [yctx->num_shader_images[shader] - 1].resource)
      yctx->num_shader_images[shader]--;

   yttrium_pipeline_state_changed(yctx);
}

void
yttrium_sampler_view_destroy(struct pipe_context *ctx,
                             struct pipe_sampler_view *view)
{
   if (!view)
      return;

   YTTRIUM_LOG("yttrium: destroy sampler view=%p texture=%p\n",
                view, (void *)view->texture);
   pipe_resource_reference(&view->texture, NULL);
   FREE(view);
}
