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

#include <stdio.h>
#include <string.h>
#include <smmintrin.h>

#include "ipa-impl.h"

static inline void
template_compose_group_SSE(ipa_context *ctx,
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

    int chan = 0;

    /* While this code uses 128-bit SSE intrinsics, the algorithm is potentially
     * scalable to larger word sizes, so we'll keep things as general as possible.
     */
#define WORD_SIZE                     16
#define HALF_SHIFT      (WORD_SIZE >> 1)
#define QUARTER_SHIFT   (WORD_SIZE >> 2)

    /* Constants for SSE blend routines. */
    __m128i half_0x80;
    __m128i half_0xff;
    __m128i pack_16_to_8_hi;
    __m128i pack_32_to_4_hi;
    __m128i unpack_16_lo_to_32;
    __m128i unpack_8_lo_to_32;
    __m128i half_alpha;
    __m128i mask_bg_alpha_word;
    int nvalues;

    /* Storage area for quantities which are aligned to WORD_SIZE 
     * boundaries for best load/store performance.  We need enough
     * storage for all aligned words plus one extra word's worth
     * which is used to ensure proper alignment.
     */
    ipa_byte aligned_storage[WORD_SIZE*3];
    ipa_byte *aligned_storage_ptr = aligned_storage;
    ipa_byte *temp = NULL;

    /* Align output color storage to a WORD_SIZE boundary so we can do aligned loads/stores. */
    while ((long long)aligned_storage_ptr & (WORD_SIZE-1)) {
        aligned_storage_ptr++;
    }
    temp = aligned_storage_ptr;

    /* Initialize constants for SSE blend routines. */
    half_0x80 = _mm_set1_epi32(0x00800080);
    half_0xff = _mm_set1_epi32(0x00FF00FF);

    /* mm_shuffle_epi8 permutation vectors */

    /* Pack eight 16-bit values into the high 8 bytes of an m128i. */
    pack_16_to_8_hi = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, 10, 12, 14);

    /* Pack four 32-bit values into the high 4 bytes of an m128i. */
    pack_32_to_4_hi = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12);

    /* Unpack the lower 4 16-bit values into four 32-bit values. */
    unpack_16_lo_to_32 = _mm_setr_epi8(0, 1, -1, -1, 2, 3, -1, -1, 4, 5, -1, -1, 6, 7, -1, -1);

    /* Unpack the lower 4 8-bit values into four 32-bit values. */
    unpack_8_lo_to_32 = _mm_setr_epi8(0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1);

    /* Unlike the pure constants above this is a packing of the top-level alpha parameter. */
    half_alpha = _mm_set1_epi16(alpha);

    if (mask_type == MASK_TYPE_PARTIAL) {
        mask_bg_alpha_word = _mm_set1_epi8(mask_bg_alpha);
    }

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

#define N_CHAN_MAX   4   /* Windows does not allow non-constant array sizes in variable definitions so we must hardcode the array sizes. */
        for (x=0; x<width; x+=WORD_SIZE) {
            __m128i src_alpha_orig;
            __m128i dst_alpha_orig;
            __m128i src_color_orig[N_CHAN_MAX];
            __m128i dst_color_orig[N_CHAN_MAX];
            __m128i zero_src_alpha_mask;

            /* Working variables */
            __m128i src_alpha;
            __m128i src_color[N_CHAN_MAX];
            __m128i dst_alpha;
            __m128i dst_color[N_CHAN_MAX];

            __m128i dst_alpha_new = _mm_setzero_si128();
            __m128i dst_color_new[N_CHAN_MAX];
            __m128i softmask;
            __m128i zero_dst_alpha_new_mask;
            ipa_bool mask_loaded = ipa_false;
            true_x = x + x0;

            if (mask_type == MASK_TYPE_PARTIAL) {
                if ((true_y < mask_y0) || (true_y >= mask_y1) ||
                    (true_x + WORD_SIZE-1 < mask_x0) || (true_x >= mask_x1)) {
                    /* We are completely outside the mask so use the mask background color. */
                    softmask = mask_bg_alpha_word;
                    mask_curr_ptr += WORD_SIZE;
                    mask_loaded = ipa_true;
                }
                else {
                    ipa_byte softmask_bytes[WORD_SIZE*2];
                    ipa_byte *softmask_ptr = softmask_bytes;
                    while (((long long)softmask_ptr) & (WORD_SIZE-1)) {
                        /* Align to WORD_SIZE boundary. */
                        softmask_ptr++;
                    }
                    for (x1=0; x1<WORD_SIZE; x1++) {
                        if ((x + x1 < mask_x0) || (x + x1 >= mask_x1)) {
                            softmask_ptr[x1] = mask_bg_alpha;
                            mask_curr_ptr++;
                        }
                        else if (mask_tr_fn_is_linear) {
                            softmask_ptr[x1] = *(mask_curr_ptr++);
                        }
                    }
                    softmask = _mm_load_si128((const __m128i *)softmask_ptr);
                    mask_loaded = ipa_true;
                }
            }

            if ((mask_type != MASK_TYPE_NONE) && !mask_loaded) {
                /* Handle the partial-word case at the right edge of the line. */
                if (x + WORD_SIZE > width) {
                    if (mask_tr_fn_is_linear) { 
                        /* Linear - load sample directly. */
                        for(x1=0; x1<width-x; x1++) {
                            temp[x1] = mask_curr_ptr[x1];
                        }
                    }
                    else {
                        /* Nonlinear - map through transfer function. */
                        for(x1=0; x1<width-x; x1++) {
                            temp[x1] = mask_tr_fn[mask_curr_ptr[x1]];
                        }
                    }

                    /* Zero out any unused bytes in the mask word. */
                    for (; x1<WORD_SIZE; x1++) {
                        temp[x1] = 0;
                    }
                    softmask = _mm_load_si128((const __m128i *)temp);
                    mask_curr_ptr += width-x;
                }
                else {
                    /* Load WORD_SIZE bytes of soft mask data. */
                    if (mask_tr_fn_is_linear) { 
                        /* If the mask transfer function is linear, load the mask directly from memory. */
                        softmask = _mm_loadu_si128((const __m128i *)mask_curr_ptr);
                        mask_curr_ptr += WORD_SIZE;
                    }
                    else {
                        /* If the mask transfer function is nonlinear, we need to map each mask byte
                         * through the transfer function.
                         */
                        for(x1=0; x1<WORD_SIZE; x1++) {
                            temp[x1] = mask_tr_fn[mask_curr_ptr[x1]];
                        }
                        softmask = _mm_load_si128((const __m128i *)temp);
                        mask_curr_ptr += WORD_SIZE;
                    }
                }
            }

            if (x + WORD_SIZE > width) {

                /* Process the last partial word at the right edge of the line. */
                nvalues = width - x;

                /* Load in the source (TOS) and destination (NOS) data for each color channel. */
                for (chan=0; chan<n_chan; chan++) {
                    memcpy(temp, tos_ptr + chan * tos_planestride, nvalues);
                    src_color_orig[chan] = src_color[chan] = _mm_loadu_si128((const __m128i *)temp);

                    memcpy(temp, nos_ptr + chan * nos_planestride, nvalues);
                    dst_color_orig[chan] = dst_color[chan] = _mm_loadu_si128((const __m128i *)temp);

                    /* The new color value will be filled in one quarter at a time. */
                    dst_color_new[chan] = _mm_setzero_si128();
                }

                /* Load the source (TOS) alpha data. */
                memcpy(temp, tos_ptr + n_chan * tos_planestride, nvalues);
                src_alpha_orig = src_alpha = _mm_loadu_si128((const __m128i *)temp);

                /* Load the destination (NOS) alpha data. */
                memcpy(temp, nos_ptr + n_chan * nos_planestride, nvalues);
                dst_alpha_orig = dst_alpha = _mm_loadu_si128((const __m128i *)temp);
            }
            else {
                /* Load in the source (TOS) and destination (NOS) data for each color channel. */
                for (chan=0; chan<n_chan; chan++) {
                    src_color_orig[chan] = src_color[chan] = _mm_loadu_si128((const __m128i *)(tos_ptr + chan * tos_planestride));
                    dst_color_orig[chan] = dst_color[chan] = _mm_loadu_si128((const __m128i *)(nos_ptr + chan * nos_planestride));

                    /* The new color value will be filled in one quarter at a time. */
                    dst_color_new[chan] = _mm_setzero_si128();
                }

                /* Load the source (TOS) alpha data. */
                src_alpha_orig = src_alpha = _mm_loadu_si128((const __m128i *)(tos_ptr + n_chan * tos_planestride));

                /* Load the destination (NOS) alpha data. */
                dst_alpha_orig = dst_alpha = _mm_loadu_si128((const __m128i *)(nos_ptr + n_chan * nos_planestride));
            }

            /* Upscale from eight to 16 bits and process in two sets of eight bytes. */
			int half;
            for (half=0; half<2; half++) {
                __m128i half_src_alpha = _mm_cvtepu8_epi16(src_alpha);
                __m128i half_dst_alpha = _mm_cvtepu8_epi16(dst_alpha);
                __m128i half_softmask;
                __m128i tmp;
                __m128i half_pix_alpha;
                __m128i half_src_alpha_inverted;
                __m128i half_dst_alpha_inverted;
                __m128i half_a_r;

                /* Process the soft mask if present. */
                if (mask_type != MASK_TYPE_NONE) {
                    half_softmask = _mm_cvtepu8_epi16(softmask);

                    /* tmp = alpha * softmask + 0x80 */
                    tmp = _mm_mullo_epi16(half_alpha, half_softmask);                    /* tmp = alpha * softmask */
                    tmp = _mm_add_epi16(tmp, half_0x80);                                 /* tmp += 0x80 */

                    /* pix_alpha = (tmp + (tmp >> 8) >> 8 */
                    half_pix_alpha = tmp;                                               /* pix_alpha = tmp */
                    tmp = _mm_srli_epi16(tmp, 8);                                       /* tmp >>= 8 */
                    half_pix_alpha = _mm_add_epi16(half_pix_alpha, tmp);                /* pix_alpha += tmp */
                    half_pix_alpha = _mm_srli_epi16(half_pix_alpha, 8);                 /* pix_alpha >>= 8 */
                }
                else { /* no mask */
                    half_pix_alpha = half_alpha;
                }

                /* tmp = src_alpha * pix_alpha + 0x80 */
                tmp = _mm_mullo_epi16(half_src_alpha, half_pix_alpha);              /* tmp = src_alpha * pix_alpha */
                tmp = _mm_add_epi16(tmp, half_0x80);                                /* tmp += 0x80 */

                /* src_alpha = (tmp + (tmp >> 8)) >> 8 */
                half_src_alpha = tmp;                                               /* src_alpha = tmp */
                tmp = _mm_srli_epi16(tmp, 8);                                       /* tmp >>= 8 */
                half_src_alpha = _mm_add_epi16(half_src_alpha, tmp);                /* src_alpha += tmp */
                half_src_alpha = _mm_srli_epi16(half_src_alpha, 8);                 /* src_alpha >>= 8 */

                half_src_alpha_inverted = _mm_sub_epi16(half_0xff, half_src_alpha);  /* 0xff - src_alpha */
                half_dst_alpha_inverted = _mm_sub_epi16(half_0xff, half_dst_alpha);  /* 0xff - dst_alpha */

                /* tmp = (0xff - src_alpha) * (0xff - dst_alpha) + 0x80 */
                tmp = _mm_mullo_epi16(half_src_alpha_inverted, half_dst_alpha_inverted);
                tmp = _mm_add_epi16(tmp, half_0x80);                                              /* tmp += 0x80 */

                /* a_r = 0xff - (((tmp >> 8) + tmp) >> 8) */
                half_a_r = _mm_srli_epi16(tmp, 8);                /* a_r = (tmp >> 8) */
                half_a_r = _mm_adds_epu16(half_a_r, tmp);         /* a_r += tmp */
                half_a_r = _mm_srli_epi16(half_a_r, 8);           /* a_r >>= 8 */
                half_a_r = _mm_sub_epi16(half_0xff, half_a_r);    /* a_r = 0xff - a_r */

                /* Store a_r values into dst_alpha_new. */
                if (half) {
                    /* Shift the existing half-word to the right. */
                    dst_alpha_new = _mm_srli_si128(dst_alpha_new, HALF_SHIFT);
                }

                /* Or the new values into the high half of the word. */
                dst_alpha_new = _mm_or_si128(dst_alpha_new, _mm_shuffle_epi8(half_a_r, pack_16_to_8_hi));
                int quarter;
                for (quarter=0; quarter<2; quarter++) {
                    __m128i quarter_src_alpha_si32 = _mm_shuffle_epi8(half_src_alpha, unpack_16_lo_to_32);
                    __m128  quarter_src_alpha = _mm_cvtepi32_ps(quarter_src_alpha_si32);       /* Convert to floats */
                    __m128i quarter_a_r_si32 = _mm_shuffle_epi8(half_a_r, unpack_16_lo_to_32);
                    __m128  quarter_a_r = _mm_cvtepi32_ps(quarter_a_r_si32);                   /* Convert to floats */

                    /* src_scale = src_alpha / a_r */
                    /* Divisions by zero are ignored during this conversion and are masked out later in the process.
                     * Although it might be logical to handle this case here for each quarter-word, it is more
                     * efficient to process the zero-alpha masking on the full word of final 8-bit values, so this
                     * conversion is done after both halves have been processed.
                     */
                    __m128 quarter_src_scale = _mm_div_ps(quarter_src_alpha, quarter_a_r);

                    for (chan=0; chan<n_chan; chan++) {
                        __m128i quarter_src_color_si32 = _mm_shuffle_epi8(src_color[chan], unpack_8_lo_to_32);
                        __m128i quarter_dst_color_si32 = _mm_shuffle_epi8(dst_color[chan], unpack_8_lo_to_32);
                        __m128 quarter_src_color = _mm_cvtepi32_ps(quarter_src_color_si32);    /* Convert to floats */
                        __m128 quarter_dst_color = _mm_cvtepi32_ps(quarter_dst_color_si32);    /* Convert to floats */

                        __m128 quarter_dst_color_adjusted;
                        __m128i quarter_dst_color_adjusted_si32;

                        /* Do simple compositing of source over backdrop */
                        quarter_dst_color_adjusted = _mm_sub_ps(quarter_src_color, quarter_dst_color);          /* (c_s - c_b) */
                        quarter_dst_color_adjusted = _mm_mul_ps(quarter_src_scale, quarter_dst_color_adjusted); /* src_scale * (c_s - c_b) */
                        quarter_dst_color_adjusted = _mm_add_ps(quarter_dst_color_adjusted, quarter_dst_color); /* c_b + src_scale * (c_s - c_b) */
                        quarter_dst_color_adjusted = _mm_add_ps(quarter_dst_color_adjusted, _mm_set1_ps(0.5));  /* c_b + src_scale * (c_s - c_b) + 0.5 */

                        /* Convert floats back to 32-bit integers with truncation. */
                        quarter_dst_color_adjusted_si32 = _mm_cvttps_epi32(quarter_dst_color_adjusted);         /* Convert float to int32 with truncation */
                        quarter_dst_color_adjusted_si32 = _mm_shuffle_epi8(quarter_dst_color_adjusted_si32, pack_32_to_4_hi);

                        if (!half || !quarter) {
                            src_color[chan] = _mm_srli_si128(src_color[chan], QUARTER_SHIFT);            /* Shift one quarter word to the right. */
                            dst_color[chan] = _mm_srli_si128(dst_color[chan], QUARTER_SHIFT);            /* Shift one quarter word to the right. */
                        }

                        if (half || quarter) {
                            dst_color_new[chan] = _mm_srli_si128(dst_color_new[chan], QUARTER_SHIFT);            /* Shift one quarter word to the right. */
                        }
                        dst_color_new[chan] = _mm_or_si128(dst_color_new[chan], quarter_dst_color_adjusted_si32);      /* Or in four color values. */
                    }

                    /* Shift input variables for the next quarter. */
                    if (!quarter) {
                        if (mask_type != MASK_TYPE_NONE) {
                            half_softmask = _mm_srli_si128(half_softmask, HALF_SHIFT);
                        }
                        half_src_alpha = _mm_srli_si128(half_src_alpha, HALF_SHIFT);
                        half_dst_alpha = _mm_srli_si128(half_dst_alpha, HALF_SHIFT);
                        half_a_r = _mm_srli_si128(half_a_r, HALF_SHIFT);
                    }
                }

                /* Shift input variables for the next half. */
                if (half == 0) {
                    if (mask_type != MASK_TYPE_NONE) {
                        softmask = _mm_srli_si128(softmask, HALF_SHIFT);
                    }
                    src_alpha = _mm_srli_si128(src_alpha, HALF_SHIFT);
                    dst_alpha = _mm_srli_si128(dst_alpha, HALF_SHIFT);
                }
            }

            /* If any of the background alpha values are zero, mask and copy the
             * original source colors and alpha values into any zero spots.  The
             * previous values are invalid because of division by a zero alpha.
             */
            zero_dst_alpha_new_mask = _mm_cmpeq_epi8(dst_alpha_new, _mm_setzero_si128());

            /* Check the entire word to see if any of the destination alpha values are zero. */
            if (_mm_movemask_epi8(_mm_cmpeq_epi32(zero_dst_alpha_new_mask, _mm_setzero_si128())) != 0xFFFF) {

                /* One or more of the destination alpha values are zero, so we need to mask
                 * in the original source values in place of any zero values.  This is because
                 * the divide by zero produced an undefined result.
                 */
                dst_alpha_new = _mm_andnot_si128(zero_dst_alpha_new_mask, dst_alpha_new);
                dst_alpha_new = _mm_or_si128(dst_alpha_new, _mm_and_si128(zero_dst_alpha_new_mask, src_alpha_orig));

                for (chan=0; chan<n_chan; chan++) {
                    dst_color_new[chan] = _mm_andnot_si128(zero_dst_alpha_new_mask, dst_color_new[chan]);
                    dst_color_new[chan] = _mm_or_si128(dst_color_new[chan], _mm_and_si128(zero_dst_alpha_new_mask, src_color_orig[chan]));
                }
            }

            /* Create a mask containing 0xFF bytes for every zero source alpha value.
             * Source alpha values of 0 do not modify the destination plane.
             */
            zero_src_alpha_mask = _mm_cmpeq_epi8(src_alpha_orig, _mm_setzero_si128());

            /* Check the entire word to see if any of the source alpha values are zero. */
            if (_mm_movemask_epi8(_mm_cmpeq_epi32(zero_src_alpha_mask, _mm_setzero_si128())) != 0xFFFF) {
                /* One or more of the source alpha values are zero, so we need to mask in the original
                 * destination values in place of any zero values.  This is because a source alpha value
                 * of zero leaves the destination colors and alpha unchanged.
                 */
                dst_alpha_new = _mm_andnot_si128(zero_src_alpha_mask, dst_alpha_new);
                dst_alpha_new = _mm_or_si128(dst_alpha_new, _mm_and_si128(zero_src_alpha_mask, dst_alpha_orig));

                for (chan=0; chan<n_chan; chan++) {
                    dst_color_new[chan] = _mm_andnot_si128(zero_src_alpha_mask, dst_color_new[chan]);
                    dst_color_new[chan] = _mm_or_si128(dst_color_new[chan], _mm_and_si128(zero_src_alpha_mask, dst_color_orig[chan]));
                }
            }

            /* Store back the updated destination data for all channels. */
            if (x + WORD_SIZE > width) {
                /* Process the last partial word at the right edge of the line. */
                nvalues = width - x;

                for (chan=0; chan<n_chan; chan++) {
                    _mm_storeu_si128((__m128i *)temp, dst_color_new[chan]);
                    memcpy(nos_ptr + chan * nos_planestride, temp, nvalues);
                }

                /* Store back the updated destination alpha data. */
                _mm_storeu_si128((__m128i *)temp, dst_alpha_new);
                memcpy(nos_ptr + n_chan * nos_planestride, temp, nvalues);

                /* Update pointers to consume the processed data. */
                tos_ptr += nvalues;
                nos_ptr += nvalues;
            }
            else {
                for (chan=0; chan<n_chan; chan++) {
                    _mm_storeu_si128((__m128i *)(nos_ptr + chan * nos_planestride), dst_color_new[chan]);
                }

                /* Store back the updated destination alpha data. */
                _mm_storeu_si128((__m128i *)(nos_ptr + n_chan * nos_planestride), dst_alpha_new);

                /* Update pointers to consume the processed data. */
                tos_ptr += WORD_SIZE;
                nos_ptr += WORD_SIZE;
            }
        }
        tos_ptr += tos_rowstride - width;
        nos_ptr += nos_rowstride - width;
        if (mask_type != MASK_TYPE_NONE) {
            mask_ptr += mask_rowstride;
        }
    }
}
