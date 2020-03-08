/*
        Copyright 2019 Intel Corporation.
        This software and the related documents are Intel copyrighted materials,
        and your use of them is governed by the express license under which they
        were provided to you (End User License Agreement for the Intel(R) Software
        Development Products (Version May 2017)). Unless the License provides
        otherwise, you may not use, modify, copy, publish, distribute, disclose or
        transmit this software or the related documents without Intel's prior
        written permission.

        This software and the related documents are provided as is, with no
        express or implied warranties, other than those that are expressly
        stated in the License.
*/
/* gradient.c */

#include "ipa-impl.h"

/* Include the cores */
#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 0
#include "gradient_c.h"
#endif

#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 1
#include "gradient_sse.h"
#endif

/* Render an axial gradient into a color buffer and optional mask buffer. */
void render_axial_gradient(ipa_context *ctx,
                           ipa_byte *ipa_restrict data_ptr,
                           int data_rowstride,
                           int base_x,
                           int base_y,
                           int offset_x,
                           int offset_y,
                           int size_x,
                           int size_y,
                           ipa_point axis_start,
                           ipa_point axis_end,
                           int color_count,
                           ipa_bool color_interpolation,
                           ipa_float colors[][IPA_GRADIENT_COLOR_COMPONENTS_MAX],
                           ipa_bool extend_start,
                           ipa_bool extend_end,
                           ipa_byte *ipa_restrict mask_ptr,
                           int mask_rowstride)
{
    if (ctx->use_sse_4_1) {
        render_axial_gradient_SSE(ctx,
                                  data_ptr,
                                  data_rowstride,
                                  base_x,
                                  base_y,
                                  offset_x,
                                  offset_y,
                                  size_x,
                                  size_y,
                                  axis_start,
                                  axis_end,
                                  color_count,
                                  color_interpolation,
                                  colors,
                                  extend_start,
                                  extend_end,
                                  mask_ptr,
                                  mask_rowstride);
    }
    else {
        render_axial_gradient_C(ctx,
                                data_ptr,
                                data_rowstride,
                                base_x,
                                base_y,
                                offset_x,
                                offset_y,
                                size_x,
                                size_y,
                                axis_start,
                                axis_end,
                                color_count,
                                color_interpolation,
                                colors,
                                extend_start,
                                extend_end,
                                mask_ptr,
                                mask_rowstride);
    }
}

/* Render a radial gradient into a color buffer and optional mask buffer. */
void render_radial_gradient(ipa_context *ctx,
                            ipa_byte *ipa_restrict data_ptr,
                            int data_rowstride,
                            int base_x,
                            int base_y,
                            int offset_x,
                            int offset_y,
                            int size_x,
                            int size_y,
                            ipa_point center,
                            ipa_float radius_start,
                            ipa_float radius_end,
                            int color_count,
                            ipa_bool color_interpolation,
                            ipa_float colors[][IPA_GRADIENT_COLOR_COMPONENTS_MAX],
                            ipa_bool extend_start,
                            ipa_bool extend_end,
                            ipa_byte *ipa_restrict mask_ptr,
                            int mask_rowstride)
{
    if (ctx->use_sse_4_1) {
        render_radial_gradient_SSE(ctx,
                                   data_ptr,
                                   data_rowstride,
                                   base_x,
                                   base_y,
                                   offset_x,
                                   offset_y,
                                   size_x,
                                   size_y,
                                   center,
                                   radius_start,
                                   radius_end,
                                   color_count,
                                   color_interpolation,
                                   colors,
                                   extend_start,
                                   extend_end,
                                   mask_ptr,
                                   mask_rowstride);
    }
    else {
        render_radial_gradient_C(ctx,
                                 data_ptr,
                                 data_rowstride,
                                 base_x,
                                 base_y,
                                 offset_x,
                                 offset_y,
                                 size_x,
                                 size_y,
                                 center,
                                 radius_start,
                                 radius_end,
                                 color_count,
                                 color_interpolation,
                                 colors,
                                 extend_start,
                                 extend_end,
                                 mask_ptr,
                                 mask_rowstride);
    }
}
