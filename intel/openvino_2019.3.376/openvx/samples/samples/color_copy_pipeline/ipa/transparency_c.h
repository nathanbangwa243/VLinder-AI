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

#include <stdio.h>
#include <string.h>

#include "ipa-impl.h"

/* Definitions */

/* Routines */

static inline void
template_compose_group_C(ipa_context *ctx,
                         ipa_byte *ipa_restrict tos_ptr,
                         ipa_bool tos_isolated,
                         int tos_rowstride,
                         int tos_planestride,
                         ipa_byte *ipa_restrict nos_ptr,
                         int nos_rowstride,
                         int nos_planestride,
                         MASK_TYPE mask_type,
                         ipa_byte *ipa_restrict mask_ptr,
                         int mask_rowstride,
                         ipa_byte mask_bg_alpha,
                         ipa_byte *ipa_restrict mask_tr_fn,
                         int n_chan,
                         int x0, int y0, int x1, int y1,
                         int mask_x0, int mask_y0, int mask_x1, int mask_y1,
                         ipa_byte alpha)
{
    ipa_bool mask_tr_fn_is_linear;
    int width = x1 - x0;
    int x = 0;
    int y = 0;
    int i = 0;

    /* Optimize the case where we can bypass the mask transfer function. */
    mask_tr_fn_is_linear = ipa_true;
    if (mask_type != MASK_TYPE_NONE) {
        /* If we have a linear mask transfer function, we can skip the table lookup
         * when loading in the mask value, which should help performance with SSE.
         * If the mask transfer function is null we treat it as a linear transfer.
         */
        if (mask_tr_fn != NULL) {
            for (i = 0; i<256; i++) {
                if (i != mask_tr_fn[i]) {
                    mask_tr_fn_is_linear = ipa_false;
                    break;
                }
            }
        }
    }

    for (y=0; y<(y1 - y0); y++) {
        int true_y = y + y0;
        int true_x;
        ipa_byte *ipa_restrict mask_curr_ptr = mask_ptr;

        /* Process any remaining samples the old fashioned way. */
        for (x=0; x<width; x++) {
            ipa_bool mask_loaded;
            ipa_byte softmask;
            ipa_byte src_alpha;

            true_x = x + x0;

            mask_loaded = ipa_false;
            if (mask_type == MASK_TYPE_PARTIAL) {
                if ((true_y < mask_y0) || (true_y >= mask_y1) ||
                    (true_x < mask_x0) || (true_x >= mask_x1)) {
              
                    /* We are completely outside the mask so use the mask background color. */
                    softmask = mask_bg_alpha;
                    mask_curr_ptr++;
                    mask_loaded = ipa_true;
                }
            }
            if ((mask_type != MASK_TYPE_NONE) && !mask_loaded) {
                if (mask_tr_fn_is_linear) {
                    softmask = *mask_curr_ptr++;
                }
                else {
                    softmask = mask_tr_fn[*mask_curr_ptr++];
                }                  
            }
            src_alpha = tos_ptr[n_chan * tos_planestride];

            if (src_alpha == 0) {
            }
            else {
                ipa_byte dst_alpha = nos_ptr[n_chan * nos_planestride];
                ipa_byte pix_alpha;

                if (mask_type != MASK_TYPE_NONE) {
                    int tmp = alpha * softmask + 0x80;
                    pix_alpha = (tmp + (tmp >> 8)) >> 8;
                }
                else {
                    pix_alpha = alpha;
                }
                if (pix_alpha != 255) {
                    int tmp = src_alpha * pix_alpha + 0x80;
                    src_alpha = (tmp + (tmp >> 8)) >> 8;
                }

                if (dst_alpha == 0) {
                    /* Simple copy of colors plus alpha. */
                    for (i = 0; i < n_chan; i++) {
                        nos_ptr[i * nos_planestride] = tos_ptr[i * tos_planestride];
                    }
                    nos_ptr[i * nos_planestride] = src_alpha;
                }
                else {
                    /* Result alpha is union of backdrop and source alpha */
                    int tmp = (0xff - dst_alpha) * (0xff - src_alpha) + 0x80;
                    unsigned int a_r = 0xff - (((tmp >> 8) + tmp) >> 8);
                    int src_scale;

                    /* Store a_r back into the nos alpha value. */
                    nos_ptr[n_chan * nos_planestride] = a_r;

                    /* Compute src_alpha / a_r in 16.16 format */
                    src_scale = ((src_alpha << 16) + (a_r >> 1)) / a_r;

                    /* Do simple compositing of source over backdrop */
                    for (i=0; i<n_chan; i++) {
                        int c_s = tos_ptr[i * tos_planestride];
                        int c_b = nos_ptr[i * nos_planestride];
                        tmp = (c_b << 16) + src_scale * (c_s - c_b) + 0x8000;
                        nos_ptr[i * nos_planestride] = tmp >> 16;
                    }
                }
            }
            ++tos_ptr;
            ++nos_ptr;
        }
        tos_ptr += tos_rowstride - width;
        nos_ptr += nos_rowstride - width;
        if (mask_type != MASK_TYPE_NONE) {
            mask_ptr += mask_rowstride;
        }
    }
}
