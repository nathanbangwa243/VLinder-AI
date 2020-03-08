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
/* PDF 1.4 transparency blending functions accelerated for SSE */

#include "ipa-impl.h"

typedef enum MASK_TYPE {
    MASK_TYPE_NONE=0,
    MASK_TYPE_PARTIAL,
    MASK_TYPE_FULL,
} MASK_TYPE;

/* Include the cores */
#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 0
#include "transparency_c.h"
#endif

#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 1
#include "transparency_sse.h"
#endif

/* Blend cases */

void
compose_blend_nonisolated_nomask_SSE(
    ipa_context *ctx,
    ipa_byte *ipa_restrict tos_ptr,
    int tos_rowstride,
    int tos_planestride,
    ipa_byte *ipa_restrict nos_ptr,
    int nos_rowstride,
    int nos_planestride,
    int n_chan,
    int x0, int y0, int x1, int y1,
    ipa_byte alpha)
{
    if (ctx->use_sse_4_1) {
        template_compose_group_SSE(ctx,
                                   tos_ptr,
                                   ipa_true, /* tos_isolated, */
                                   tos_rowstride,
                                   tos_planestride,
                                   nos_ptr,
                                   nos_rowstride,
                                   nos_planestride,
                                   MASK_TYPE_NONE,
                                   NULL, /* mask_ptr */
                                   0, /* mask_rowstride */
                                   0, /* mask_bg_alpha */
                                   NULL, /* mask_tr_fn */
                                   n_chan,
                                   x0, y0, x1, y1,
                                   0, 0, 0, 0,
                                   alpha);
    }
    else {
        template_compose_group_C(ctx,
                                 tos_ptr,
                                 ipa_true, /* tos_isolated, */
                                 tos_rowstride,
                                 tos_planestride,
                                 nos_ptr,
                                 nos_rowstride,
                                 nos_planestride,
                                 MASK_TYPE_NONE,
                                 NULL, /* mask_ptr */
                                 0, /* mask_rowstride */
                                 0, /* mask_bg_alpha */
                                 NULL, /* mask_tr_fn */
                                 n_chan,
                                 x0, y0, x1, y1,
                                 0, 0, 0, 0,
                                 alpha);
    }
}

void
compose_blend_nonisolated_partialmask_SSE(
    ipa_context *ctx,
    ipa_byte *ipa_restrict tos_ptr,
    int tos_rowstride,
    int tos_planestride,
    ipa_byte *ipa_restrict nos_ptr,
    int nos_rowstride,
    int nos_planestride,
    int n_chan,
    int x0, int y0, int x1, int y1,
    ipa_byte *ipa_restrict mask_ptr,
    int mask_rowstride,
    int mask_x0, int mask_y0, int mask_x1, int mask_y1,
    ipa_byte mask_bg_alpha,
    ipa_byte *ipa_restrict mask_tr_fn,
    ipa_byte alpha)
{
    if (ctx->use_sse_4_1) {
        template_compose_group_SSE(ctx,
                                   tos_ptr,
                                   ipa_false, /* tos_isolated, */
                                   tos_rowstride,
                                   tos_planestride,
                                   nos_ptr,
                                   nos_rowstride,
                                   nos_planestride,
                                   MASK_TYPE_PARTIAL,
                                   mask_ptr,
                                   mask_rowstride,
                                   mask_bg_alpha,
                                   mask_tr_fn,
                                   n_chan,
                                   x0, y0, x1, y1,
                                   mask_x0, mask_y0, mask_x1, mask_y1,
                                   alpha);
    }
    else {
          template_compose_group_C(ctx,
                                   tos_ptr,
                                   ipa_false, /* tos_isolated, */
                                   tos_rowstride,
                                   tos_planestride,
                                   nos_ptr,
                                   nos_rowstride,
                                   nos_planestride,
                                   MASK_TYPE_PARTIAL,
                                   mask_ptr,
                                   mask_rowstride,
                                   mask_bg_alpha,
                                   mask_tr_fn,
                                   n_chan,
                                   x0, y0, x1, y1,
                                   mask_x0, mask_y0, mask_x1, mask_y1,
                                   alpha);
    }
}

void
compose_blend_nonisolated_fullmask_SSE(
    ipa_context *ctx,
    ipa_byte *ipa_restrict tos_ptr,
    int tos_rowstride,
    int tos_planestride,
    ipa_byte *ipa_restrict nos_ptr,
    int nos_rowstride,
    int nos_planestride,
    int n_chan,
    int x0, int y0, int x1, int y1,
    ipa_byte *ipa_restrict mask_ptr,
    int mask_rowstride,
    ipa_byte mask_bg_alpha,
    ipa_byte *ipa_restrict mask_tr_fn,
    ipa_byte alpha)
{
    if (ctx->use_sse_4_1) {
        template_compose_group_SSE(ctx,
                                   tos_ptr,
                                   ipa_false, /* tos_isolated, */
                                   tos_rowstride,
                                   tos_planestride,
                                   nos_ptr,
                                   nos_rowstride,
                                   nos_planestride,
                                   MASK_TYPE_FULL,
                                   mask_ptr,
                                   mask_rowstride,
                                   mask_bg_alpha,
                                   mask_tr_fn,
                                   n_chan,
                                   x0, y0, x1, y1,
                                   x0, y0, x1, y1,
                                   alpha);
    }
    else {
          template_compose_group_C(ctx,
                                   tos_ptr,
                                   ipa_false, /* tos_isolated, */
                                   tos_rowstride,
                                   tos_planestride,
                                   nos_ptr,
                                   nos_rowstride,
                                   nos_planestride,
                                   MASK_TYPE_FULL,
                                   mask_ptr,
                                   mask_rowstride,
                                   mask_bg_alpha,
                                   mask_tr_fn,
                                   n_chan,
                                   x0, y0, x1, y1,
                                   x0, y0, x1, y1,
                                   alpha);
    }
}

void
compose_blend_isolated_nomask_SSE(
    ipa_context *ctx,
    ipa_byte *ipa_restrict tos_ptr,
    int tos_rowstride,
    int tos_planestride,
    ipa_byte *ipa_restrict nos_ptr,
    int nos_rowstride,
    int nos_planestride,
    int n_chan,
    int x0, int y0, int x1, int y1,
    ipa_byte alpha
    )
{
    if (ctx->use_sse_4_1) {
        template_compose_group_SSE(ctx,
                                   tos_ptr,
                                   ipa_true, /* tos_isolated, */
                                   tos_rowstride,
                                   tos_planestride,
                                   nos_ptr,
                                   nos_rowstride,
                                   nos_planestride,
                                   MASK_TYPE_NONE,
                                   NULL, /* mask_ptr */
                                   0, /* mask_rowstride */
                                   0, /* mask_bg_alpha */
                                   NULL, /* mask_tr_fn */
                                   n_chan,
                                   x0, y0, x1, y1,
                                   0, 0, 0, 0,
                                   alpha);
    }
    else {
        template_compose_group_C(ctx,
                                 tos_ptr,
                                 ipa_true, /* tos_isolated, */
                                 tos_rowstride,
                                 tos_planestride,
                                 nos_ptr,
                                 nos_rowstride,
                                 nos_planestride,
                                 MASK_TYPE_NONE,
                                 NULL, /* mask_ptr */
                                 0, /* mask_rowstride */
                                 0, /* mask_bg_alpha */
                                 NULL, /* mask_tr_fn */
                                 n_chan,
                                 x0, y0, x1, y1,
                                 0, 0, 0, 0,
                                 alpha);
    }
}

void
compose_blend_isolated_partialmask_SSE(
    ipa_context *ctx,
    ipa_byte *ipa_restrict tos_ptr,
    int tos_rowstride,
    int tos_planestride,
    ipa_byte *ipa_restrict nos_ptr,
    int nos_rowstride,
    int nos_planestride,
    int n_chan,
    int x0, int y0, int x1, int y1,
    ipa_byte *ipa_restrict mask_ptr,
    int mask_rowstride,
    int mask_x0, int mask_y0, int mask_x1, int mask_y1,
    ipa_byte mask_bg_alpha,
    ipa_byte *ipa_restrict mask_tr_fn,
    ipa_byte alpha)
{
    if (ctx->use_sse_4_1) {
        template_compose_group_SSE(ctx,
                                   tos_ptr,
                                   ipa_true, /* tos_isolated, */
                                   tos_rowstride,
                                   tos_planestride,
                                   nos_ptr,
                                   nos_rowstride,
                                   nos_planestride,
                                   MASK_TYPE_PARTIAL,
                                   mask_ptr,
                                   mask_rowstride,
                                   mask_bg_alpha,
                                   mask_tr_fn,
                                   n_chan,
                                   x0, y0, x1, y1,
                                   mask_x0, mask_y0, mask_x1, mask_y1,
                                   alpha);
    }
    else {
        template_compose_group_C(ctx,
                                 tos_ptr,
                                 ipa_true, /* tos_isolated, */
                                 tos_rowstride,
                                 tos_planestride,
                                 nos_ptr,
                                 nos_rowstride,
                                 nos_planestride,
                                 MASK_TYPE_PARTIAL,
                                 mask_ptr,
                                 mask_rowstride,
                                 mask_bg_alpha,
                                 mask_tr_fn,
                                 n_chan,
                                 x0, y0, x1, y1,
                                 mask_x0, mask_y0, mask_x1, mask_y1,
                                 alpha);
    }
}

void
compose_blend_isolated_fullmask_SSE(
    ipa_context *ctx,
    ipa_byte *ipa_restrict tos_ptr,
    int tos_rowstride,
    int tos_planestride,
    ipa_byte *ipa_restrict nos_ptr,
    int nos_rowstride,
    int nos_planestride,
    int n_chan,
    int x0, int y0, int x1, int y1,
    ipa_byte *ipa_restrict mask_ptr,
    int mask_rowstride,
    ipa_byte mask_bg_alpha,
    ipa_byte *ipa_restrict mask_tr_fn,
    ipa_byte alpha)
{
    if (ctx->use_sse_4_1) {
        template_compose_group_SSE(ctx,
                                   tos_ptr,
                                   ipa_true, /* tos_isolated, */
                                   tos_rowstride,
                                   tos_planestride,
                                   nos_ptr,
                                   nos_rowstride,
                                   nos_planestride,
                                   MASK_TYPE_FULL,
                                   mask_ptr,
                                   mask_rowstride,
                                   mask_bg_alpha,
                                   mask_tr_fn,
                                   n_chan,
                                   x0, y0, x1, y1,
                                   0, 0, 0, 0,
                                   alpha);
    }
    else {
        template_compose_group_C(ctx,
                                 tos_ptr,
                                 ipa_true, /* tos_isolated, */
                                 tos_rowstride,
                                 tos_planestride,
                                 nos_ptr,
                                 nos_rowstride,
                                 nos_planestride,
                                 MASK_TYPE_FULL,
                                 mask_ptr,
                                 mask_rowstride,
                                 mask_bg_alpha,
                                 mask_tr_fn,
                                 n_chan,
                                 x0, y0, x1, y1,
                                 0, 0, 0, 0,
                                 alpha);
    }
}
