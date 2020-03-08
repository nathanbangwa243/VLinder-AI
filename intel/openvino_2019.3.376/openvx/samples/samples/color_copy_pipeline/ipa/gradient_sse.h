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
/* Gradient blend routines - SSE 4.2 version */

#include <stdio.h>
#include <string.h>

#include <smmintrin.h>

#include "ipa-impl.h"

#define WORD_SIZE                       16
#define WORD_PS_COUNT      (WORD_SIZE >> 2)

/* Render an axial gradient into a color buffer and optional mask buffer. */
static void render_axial_gradient_SSE(ipa_context *ctx,
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
    int x = 0;
    int y = 0;
    int i = 0;
    int j = 0;
    ipa_byte *row_ptr = NULL;
    int ncomp = 4;
    uint32_t color_start_packed = 0;
    uint32_t color_end_packed = 0;

    __m128 colors_start[IPA_GRADIENT_COLOR_COMPONENTS_MAX];
    __m128 colors_end[IPA_GRADIENT_COLOR_COMPONENTS_MAX];

    __m128 x_bases = _mm_set_ps((ipa_float)base_x + (ipa_float)offset_x,
                                (ipa_float)base_x + (ipa_float)offset_x + 1.0f,
                                (ipa_float)base_x + (ipa_float)offset_x + 2.0f,
                                (ipa_float)base_x + (ipa_float)offset_x + 3.0f);

    /* Storage area for quantities which are aligned to WORD_SIZE 
     * boundaries for best load/store performance.  We need enough
     * storage for all aligned words plus one extra word's worth
     * which is used to ensure proper alignment.
     */
    ipa_byte aligned_storage[WORD_SIZE*4];
    ipa_byte *aligned_storage_ptr = aligned_storage;

    /* These pointers will be initialized to point to aligned storage. */
    ipa_byte *color_bytes_aligned = NULL;
    ipa_float *color_floats_aligned = NULL;
    ipa_float *next_color_floats_aligned = NULL;

    int32_t *color_ptr = NULL;
    int32_t *outdata_ptr = NULL;

    ipa_point axis_vector;
    ipa_float axis_length_squared;
    ipa_float axis_length;
    ipa_point axis_vector_normalized;
    __m128 sines;
    __m128 cosines;

    __m128i all_ones = _mm_cmpeq_epi8(_mm_setzero_si128(), _mm_setzero_si128());

    /* Set up the packed start color if the start extend flag is set. */
    if (extend_start) {
        for (j=0; j<WORD_PS_COUNT; j++) {
            color_start_packed <<= 8;
            color_start_packed |= (uint32_t) (colors[0][WORD_PS_COUNT-1-j] * 255.0 + 0.5);
        }
    }

    /* Set up the packed end color if the end extend flag is set. */
    if (extend_end) {
        for (j=0; j<WORD_PS_COUNT; j++) {
            color_end_packed <<= 8;
            color_end_packed |= (uint32_t) (colors[color_count-1][WORD_PS_COUNT-1-j] * 255.0);
        }
    }

    /* Pack four sets of color components into an m128i. */
    __m128i pack_colors[4] = {
        _mm_setr_epi8(12, -1, -1, -1,  8, -1, -1, -1,  4, -1, -1, -1,  0, -1, -1, -1),
        _mm_setr_epi8(-1, 12, -1, -1, -1,  8, -1, -1, -1,  4, -1, -1, -1,  0, -1, -1),
        _mm_setr_epi8(-1, -1, 12, -1, -1, -1,  8, -1, -1, -1,  4, -1, -1, -1,  0, -1),
        _mm_setr_epi8(-1, -1, -1, 12, -1, -1, -1,  8, -1, -1, -1,  4, -1, -1, -1,  0)
    };

    for (i=0; i<ncomp; i++) {
        colors_start[i] = _mm_set_ps1(colors[0][i]);
        colors_end[i] = _mm_set_ps1(colors[color_count-1][i]);
    }

    /* Align output color storage to a WORD_SIZE boundary so we can do aligned loads and stores. */
    while ((long long)aligned_storage_ptr & (WORD_SIZE-1)) {
        aligned_storage_ptr++;
    }
    color_bytes_aligned = (ipa_byte *)aligned_storage_ptr;
    color_floats_aligned = (ipa_float *)(aligned_storage_ptr + WORD_SIZE);
    next_color_floats_aligned = (ipa_float *)(aligned_storage_ptr + 2 * WORD_SIZE);

    /* Calculate the transformation matrix to rotate the gradient vector to zero degrees.
     * This allows position comparisons to be performed using only the X axis.
     */
    axis_vector.x = axis_end.x - axis_start.x;
    axis_vector.y = axis_end.y - axis_start.y;
    axis_length_squared = axis_vector.x * axis_vector.x + axis_vector.y * axis_vector.y;

    if (axis_length_squared == 0.0) {
        return;
    }
    axis_length = sqrtf(axis_length_squared);

    /* The X and Y components of the normalized axis vector are
     * the sine and cosine of the rotation angle theta.
     */
    axis_vector_normalized.x = axis_vector.x / axis_length;
    axis_vector_normalized.y = axis_vector.y / axis_length;

    /* Make full-word values for the cosines and sines. */
    cosines = _mm_set_ps1(axis_vector_normalized.x);
    sines = _mm_set_ps1(axis_vector_normalized.y);

    /* Loop through the grid filling in all pixels. */
    for (y=0; y<size_y; y++) {
        /* The Y distance is constant for the entire row. */
        ipa_float y_distance = (y + base_y + offset_y) - axis_vector.y;
        __m128 y_gradient_vectors = _mm_set_ps1(y_distance);

        /* Set up a pointer to the current mask row if enabled. */
        ipa_byte *mask_row_ptr = NULL;
        if (mask_ptr) {
            mask_row_ptr = mask_ptr + y * mask_rowstride;
        }

        /* Advance row pointer by the specified offset. */
        row_ptr = data_ptr + (y + offset_y) * data_rowstride + ncomp*offset_x;
        for (x=0; x<size_x; x+=WORD_PS_COUNT) {
            ipa_bool write_enable[WORD_PS_COUNT];
            int start_extend_mask;
            int end_extend_mask;
            __m128i final_colors;
            __m128 x_values;
            __m128 x_gradient_vectors;
            __m128 x_distances;
            __m128i start_extends;
            __m128i end_extends;

            final_colors = _mm_setzero_si128();
            x_values = _mm_set_ps1((float) x);
            x_gradient_vectors = _mm_sub_ps(_mm_add_ps(x_bases, x_values), _mm_set_ps1(axis_vector.x));

            /* Rotate the vector to zero degrees to allow position comparison
             * using only the X component.
             *  x' =  x*cos(theta) + y*sin(theta)
             *  y' = -x*sin(theta) + y*cos(theta)
             */
            x_distances = _mm_add_ps(_mm_mul_ps(x_gradient_vectors, cosines), _mm_mul_ps(y_gradient_vectors, sines));

            /* Determine which samples if any fall into one of the extend ranges. */
            start_extends = _mm_castps_si128(_mm_cmplt_ps(x_distances, _mm_set_ps1(0.0)));
            start_extend_mask = _mm_movemask_epi8(_mm_cmpeq_epi32(start_extends, all_ones));
            end_extends = _mm_castps_si128(_mm_cmpgt_ps(x_distances, _mm_set_ps1(axis_length)));
            end_extend_mask = _mm_movemask_epi8(_mm_cmpeq_epi32(end_extends, all_ones));

            /* Handle the case where all samples are in the start extend area. */
            if (start_extend_mask == 0xFFFF) {
                if (extend_start) {
                    /* Fill with start extend color */
                    outdata_ptr = (int32_t *) row_ptr;
                    for (j=0; j<WORD_PS_COUNT; j++) {
                        *(outdata_ptr++) = color_start_packed;
                    }

                    /* Set mask pointers to indicate marked pixels. */
                    if (mask_ptr) {
                        for (j=0; j<WORD_PS_COUNT; j++) {
                            *(mask_row_ptr++) = 0xFF;
                        }
                    }
                }
                else {
                    /* If not extending the start, the entire word is unfilled. */

                    /* Set mask pointers to indicate unmarked pixels. */
                    if (mask_ptr) {
                        for (j=0; j<WORD_PS_COUNT; j++) {
                            *(mask_row_ptr++) = 0x00;
                        }
                    }
                }
            }

            /* Handle the case where all samples are in the end extend area. */
            else if (end_extend_mask == 0xFFFF) {
                if (extend_end) {
                    /* Fill with end extend color */
                    outdata_ptr = (int32_t *) row_ptr;
                    for (j=0; j<WORD_PS_COUNT; j++) {
                        *(outdata_ptr++) = color_end_packed;
                    }

                    /* Set mask pointers to indicate marked pixels. */
                    if (mask_ptr) {
                        for (j=0; j<WORD_PS_COUNT; j++) {
                            *(mask_row_ptr++) = 0xFF;
                        }
                    }
                }
                else {
                    /* If not extending the end, the entire word is unfilled. */

                    /* Set mask pointers to indicate unmarked pixels. */
                    if (mask_ptr) {
                        for (j=0; j<WORD_PS_COUNT; j++) {
                            *(mask_row_ptr++) = 0x00;
                        }
                    }
                }
            }
            else {
                /* General case.  The word may contain a combination
                 * of samples in the gradient or in the start or end
                 * extend areas.
                 */
                int32_t *color_indices = NULL;
                __m128  ratios;
                __m128  scaled_ratios;
                __m128  color_values;
 
                int extend_byte_mask = 0xF000;
                for (j=0; j<WORD_PS_COUNT; j++) {
                    /* Normally the sample will be written out unless it's off the right side. */
                    write_enable[j] = (x + j < size_x) ? ipa_true : ipa_false;

                    /* If we are outside the minimum/maximum radius and not extending in that direction,
                     * clear the write_enable flag to allow the original color to show through.
                     */
                      if (!extend_start && (start_extend_mask & extend_byte_mask)) {
                          write_enable[j] = ipa_false;
                      }
                      if (!extend_end && (end_extend_mask & extend_byte_mask)) {
                          write_enable[j] = ipa_false;
                      }
                      extend_byte_mask >>= 4;
                }

                /* Determine the parametric variable ratios relative to the axis length. */
                ratios = _mm_div_ps(x_distances, _mm_set_ps1(axis_length));

                for (i=0; i<ncomp; i++) {
                    __m128i curr_table_indices;

                    if (color_count == 2) {
                        if (color_interpolation) {
                            color_values = _mm_add_ps(colors_start[i], _mm_mul_ps(ratios, _mm_sub_ps(colors_end[i], colors_start[i])));
                        }
                        else {
                            scaled_ratios = _mm_floor_ps(_mm_mul_ps(ratios, _mm_set1_ps(2.0)));
                            scaled_ratios = _mm_max_ps(scaled_ratios, _mm_set1_ps(0.0));
                            scaled_ratios = _mm_min_ps(scaled_ratios, _mm_set1_ps(1.0));
                            color_values = _mm_add_ps(colors_start[i], _mm_mul_ps(scaled_ratios, _mm_sub_ps(colors_end[i], colors_start[i])));
                        }
                    }
                    else if (color_interpolation) {
                        __m128  next_color_values;
                        __m128  scaled_indices;
                        __m128  remainders;
                        __m128  color_differences;

                        if (i == 0) {
                            /* Interpolated.  Look up the appropriate pair of table elements for each sample. */
                            scaled_ratios = _mm_mul_ps(ratios, _mm_set_ps1((float)color_count));

                            /* Determine the table index for each sample. */
                            curr_table_indices = _mm_cvtps_epi32(_mm_sub_ps(scaled_ratios, _mm_set1_ps(0.5)));
                            curr_table_indices = _mm_max_epi32(curr_table_indices, _mm_set1_epi32(0));
                            curr_table_indices = _mm_min_epi32(curr_table_indices, _mm_set1_epi32(color_count - 1));

                            /* Determine the remainders. */
                            scaled_indices = _mm_cvtepi32_ps(curr_table_indices);
                            remainders = _mm_sub_ps(scaled_ratios, scaled_indices);
                            remainders = _mm_max_ps(remainders, _mm_set1_ps(0.0f));
                            remainders = _mm_min_ps(remainders, _mm_set1_ps(1.0f));
                            /* Store table indices as 32-bit integers. */
                            _mm_store_si128((__m128i *)color_bytes_aligned, curr_table_indices);
                            color_indices = (int32_t *) color_bytes_aligned;
                        }

                        /* Look up each index in the color table. */
                        for (j=0; j<WORD_PS_COUNT; j++) {
                            int color_index = color_indices[j];
                            int next_color_index = color_index + 1;

                            if (next_color_index >= color_count) {
                                next_color_index = color_count - 1;
                            }

                            /* Look up this index in the color table. */
                            color_floats_aligned[j] = colors[color_index][i];
                            next_color_floats_aligned[j] = colors[next_color_index][i];
                        }
                        color_values = _mm_load_ps(color_floats_aligned);
                        next_color_values = _mm_load_ps(next_color_floats_aligned);

                        color_differences = _mm_sub_ps(next_color_values, color_values);
                        color_values = _mm_add_ps(color_values, _mm_mul_ps(remainders, color_differences));
                    }
                    else {
                        /* Non-interpolated.  Look up the appropriate table element for each sample. */
                        if (i == 0) {
                            scaled_ratios = _mm_mul_ps(ratios, _mm_set_ps1((float)color_count));
                            scaled_ratios = _mm_sub_ps(scaled_ratios, _mm_set1_ps(0.5));

                            /* Determine the table index for each sample. */
                            curr_table_indices = _mm_cvtps_epi32(scaled_ratios);
                            curr_table_indices = _mm_max_epi32(curr_table_indices, _mm_set1_epi32(0));
                            curr_table_indices = _mm_min_epi32(curr_table_indices, _mm_set1_epi32(color_count));

                            /* Store table indices as 32-bit integers. */
                            _mm_store_si128((__m128i *)color_bytes_aligned, curr_table_indices);
                            color_indices = (int32_t *) color_bytes_aligned;
                        }

                        /* Look up each index in the color table. */
                        for (j=0; j<WORD_PS_COUNT; j++) {
                            color_floats_aligned[j] = colors[color_indices[j]][i];
                        }
                        color_values = _mm_load_ps(color_floats_aligned);
                    }

                    /* Handle start extend case */
                    if (extend_start && start_extend_mask) {
                        color_values = _mm_castsi128_ps(_mm_andnot_si128(start_extends, _mm_castps_si128(color_values)));
                        color_values = _mm_castsi128_ps(_mm_or_si128(_mm_castps_si128(color_values), _mm_and_si128 (start_extends, _mm_castps_si128(colors_start[i]))));
                    }

                    /* Handle end extend case */
                    if (extend_end && end_extend_mask) {
                        color_values = _mm_castsi128_ps(_mm_andnot_si128(end_extends, _mm_castps_si128(color_values)));
                        color_values = _mm_castsi128_ps(_mm_or_si128(_mm_castps_si128(color_values), _mm_and_si128 (end_extends, _mm_castps_si128(colors_end[i]))));
                    }

                    /* Convert color values from floating-point to 8-bit unsigned format. */
                    color_values = _mm_mul_ps(color_values, _mm_set_ps1(255.0));
                    color_values = _mm_castsi128_ps(_mm_cvtps_epi32(color_values));
                    final_colors = _mm_or_si128(final_colors, _mm_shuffle_epi8(_mm_castps_si128(color_values), pack_colors[i]));
                }

                /* Store the final 8-byte color values. */
                _mm_store_si128((__m128i *)color_bytes_aligned, final_colors);

                /* Write out any unmasked data. */
                color_ptr = (int32_t *) color_bytes_aligned;
                outdata_ptr = (int32_t *) row_ptr;

                for (j=0; j<WORD_PS_COUNT; j++) {
                    if (write_enable[j]) {
                        *(outdata_ptr++) = *(color_ptr++);
                    }
                    else {
                        outdata_ptr++;
                        color_ptr++;
                    }

                    /* Write out the mask flag. */
                    if (mask_ptr) {
                        if (write_enable[j]) {
                            *(mask_row_ptr++) = 0xFF;
                        }
                        else if (x + j < size_x) {
                            *(mask_row_ptr++) = 0x00;
                        }
                    }
                }
            }
            row_ptr += ncomp * WORD_PS_COUNT;
        }
    }
}

/* Render a radial gradient into a color buffer and optional mask buffer. */
static void render_radial_gradient_SSE(ipa_context *ctx,
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
    int x = 0;
    int y = 0;
    int i = 0;
    int j = 0;
    ipa_byte *row_ptr = NULL;
    int ncomp = 4;
    uint32_t color_start_packed = 0;
    uint32_t color_end_packed = 0;

    __m128 colors_start[IPA_GRADIENT_COLOR_COMPONENTS_MAX];
    __m128 colors_end[IPA_GRADIENT_COLOR_COMPONENTS_MAX];

    __m128 radii_start = _mm_set_ps1(radius_start);
    __m128 radii_end = _mm_set_ps1(radius_end);

    __m128 x_bases = _mm_set_ps((ipa_float)base_x + (ipa_float)offset_x - center.x,
                                (ipa_float)base_x + (ipa_float)offset_x - center.x + 1.0f,
                                (ipa_float)base_x + (ipa_float)offset_x - center.x + 2.0f,
                                (ipa_float)base_x + (ipa_float)offset_x - center.x + 3.0f);

    /* Storage area for quantities which are aligned to WORD_SIZE 
     * boundaries for best load/store performance.  We need enough
     * storage for all aligned words plus one extra word's worth
     * which is used to ensure proper alignment.
     */
    ipa_byte aligned_storage[WORD_SIZE*4];
    ipa_byte *aligned_storage_ptr = aligned_storage;

    /* These pointers will be initialized to point to aligned storage. */
    ipa_byte *color_bytes_aligned = NULL;
    ipa_float *color_floats_aligned = NULL;
    ipa_float *next_color_floats_aligned = NULL;

    int32_t *color_ptr = NULL;
    int32_t *outdata_ptr = NULL;

    /* Set up the packed start color if the start extend flag is set. */
    if (extend_start) {
        for (j=0; j<WORD_PS_COUNT; j++) {
            color_start_packed <<= 8;
            color_start_packed |= (uint32_t) (colors[0][WORD_PS_COUNT-1-j] * 255.0 + 0.5);
        }
    }

    /* Set up the packed end color if the end extend flag is set. */
    if (extend_end) {
        for (j=0; j<WORD_PS_COUNT; j++) {
            color_end_packed <<= 8;
            color_end_packed |= (uint32_t) (colors[color_count-1][WORD_PS_COUNT-1-j] * 255.0);
        }
    }

    __m128i all_ones = _mm_cmpeq_epi8(_mm_setzero_si128(), _mm_setzero_si128());

    /* Pack four sets of color components into an m128i. */
    __m128i pack_colors[4] = {
        _mm_setr_epi8(12, -1, -1, -1,  8, -1, -1, -1,  4, -1, -1, -1,  0, -1, -1, -1),
        _mm_setr_epi8(-1, 12, -1, -1, -1,  8, -1, -1, -1,  4, -1, -1, -1,  0, -1, -1),
        _mm_setr_epi8(-1, -1, 12, -1, -1, -1,  8, -1, -1, -1,  4, -1, -1, -1,  0, -1),
        _mm_setr_epi8(-1, -1, -1, 12, -1, -1, -1,  8, -1, -1, -1,  4, -1, -1, -1,  0)
    };

    for (i=0; i<ncomp; i++) {
        colors_start[i] = _mm_set_ps1(colors[0][i]);
        colors_end[i] = _mm_set_ps1(colors[color_count-1][i]);
    }

    /* Align output color storage to a WORD_SIZE boundary so we can do aligned loads and stores. */
    while ((long long)aligned_storage_ptr & (WORD_SIZE-1)) {
        aligned_storage_ptr++;
    }
    color_bytes_aligned = (ipa_byte *)aligned_storage_ptr;
    color_floats_aligned = (ipa_float *)(aligned_storage_ptr + WORD_SIZE);
    next_color_floats_aligned = (ipa_float *)(aligned_storage_ptr + 2 * WORD_SIZE);

    /* Loop through the grid filling in all pixels. */
    for (y=0; y<size_y; y++) {
        /* The Y distance is constant for the entire row. */
        ipa_float y_distance = y + base_y + offset_y - center.y;
        ipa_float y_distance_squared = y_distance * y_distance;

        /* Set Y distance-squared values for all elements in the row. */
        __m128 y_distances_squared = _mm_set_ps1(y_distance_squared);

        /* Set up a pointer to the current mask row if enabled. */
        ipa_byte *mask_row_ptr = NULL;
        if (mask_ptr) {
            mask_row_ptr = mask_ptr + y * mask_rowstride;
        }

        /* Advance row pointer by the specified offset. */
        row_ptr = data_ptr + (y + offset_y) * data_rowstride + ncomp*offset_x;
        for (x=0; x<size_x; x+=WORD_PS_COUNT) {
            ipa_bool write_enable[WORD_PS_COUNT];
            __m128i final_colors = _mm_setzero_si128();
            __m128 x_values = _mm_set_ps1((float) x);
            __m128 x_distances = _mm_add_ps(x_bases, x_values);
            __m128 x_distances_squared = _mm_mul_ps(x_distances, x_distances);
            __m128 radii = _mm_sqrt_ps(_mm_add_ps(x_distances_squared, y_distances_squared));

            __m128i start_extends = _mm_castps_si128(_mm_cmplt_ps(radii, radii_start));
            int start_extend_mask = _mm_movemask_epi8(_mm_cmpeq_epi32(start_extends, all_ones));
            __m128i end_extends = _mm_castps_si128(_mm_cmpgt_ps(radii, radii_end));
            int end_extend_mask = _mm_movemask_epi8(_mm_cmpeq_epi32(end_extends, all_ones));

            /* Handle the case where all samples are in the start extend area. */
            if (start_extend_mask == 0xFFFF) {
                if (extend_start) {
                    /* Fill with start extend color */
                    outdata_ptr = (int32_t *) row_ptr;
                    for (j=0; j<WORD_PS_COUNT; j++) {
                        *(outdata_ptr++) = color_start_packed;
                    }

                    /* Set mask pointers to indicate marked pixels. */
                    if (mask_ptr) {
                        for (j=0; j<WORD_PS_COUNT; j++) {
                            *(mask_row_ptr++) = 0xFF;
                        }
                    }
                }
                else {
                    /* If not extending the start, the entire word is unfilled. */

                    /* Set mask pointers to indicate unmarked pixels. */
                    if (mask_ptr) {
                        for (j=0; j<WORD_PS_COUNT; j++) {
                            *(mask_row_ptr++) = 0x00;
                        }
                    }
                }
            }

            /* Handle the case where all samples are in the end extend area. */
            else if (end_extend_mask == 0xFFFF) {
                if (extend_end) {
                    /* Fill with end extend color */
                    outdata_ptr = (int32_t *) row_ptr;
                    for (j=0; j<WORD_PS_COUNT; j++) {
                        *(outdata_ptr++) = color_end_packed;
                    }

                    /* Set mask pointers to indicate marked pixels. */
                    if (mask_ptr) {
                        for (j=0; j<WORD_PS_COUNT; j++) {
                            *(mask_row_ptr++) = 0xFF;
                        }
                    }
                }
                else {
                    /* If not extending the end, the entire word is unfilled. */

                    /* Set mask pointers to indicate unmarked pixels. */
                    if (mask_ptr) {
                        for (j=0; j<WORD_PS_COUNT; j++) {
                            *(mask_row_ptr++) = 0x00;
                        }
                    }
                }
            }
            else {
                /* General case.  The word may contain a combination
                 * of samples in the gradient or in the start or end
                 * extend areas.
                 */
                int32_t *color_indices = NULL;
                __m128  ratios;
                __m128  scaled_ratios;
                __m128  color_values;

                int extend_byte_mask = 0xF000;
                for (j=0; j<WORD_PS_COUNT; j++) {
                    /* Normally the sample will be written out unless it's off the right side. */
                    write_enable[j] = (x + j < size_x) ? ipa_true : ipa_false;

                    /* If we are outside the minimum/maximum radius and not extending in that direction,
                     * clear the write_enable flag to allow the original color to show through.
                     */
                      if (!extend_start && (start_extend_mask & extend_byte_mask)) {
                          write_enable[j] = ipa_false;
                      }
                      if (!extend_end && (end_extend_mask & extend_byte_mask)) {
                          write_enable[j] = ipa_false;
                      }
                      extend_byte_mask >>= 4;
                }

                /* Determine the parametric variable ratios relative to the start/end radii. */
                ratios = _mm_div_ps(_mm_sub_ps(radii, radii_start), _mm_sub_ps(radii_end, radii_start));

                for (i=0; i<ncomp; i++) {
                    __m128i curr_table_indices;

                    if (color_count == 2) {
                        if (color_interpolation) {
                            color_values = _mm_add_ps(colors_start[i], _mm_mul_ps(ratios, _mm_sub_ps(colors_end[i], colors_start[i])));
                        }
                        else {
                            scaled_ratios = _mm_floor_ps(_mm_mul_ps(ratios, _mm_set1_ps(2.0)));
                            scaled_ratios = _mm_max_ps(scaled_ratios, _mm_set1_ps(0.0));
                            scaled_ratios = _mm_min_ps(scaled_ratios, _mm_set1_ps(1.0));
                            color_values = _mm_add_ps(colors_start[i], _mm_mul_ps(scaled_ratios, _mm_sub_ps(colors_end[i], colors_start[i])));
                        }
                    }
                    else if (color_interpolation) {
                        __m128  next_color_values;
                        __m128  scaled_indices;
                        __m128  remainders;
                        __m128  color_differences;

                        if (i == 0) {
                            /* Interpolated.  Look up the appropriate pair of table elements for each sample. */
                            scaled_ratios = _mm_mul_ps(ratios, _mm_set_ps1((float)color_count));

                            /* Determine the table index for each sample. */
                            curr_table_indices = _mm_cvtps_epi32(_mm_sub_ps(scaled_ratios, _mm_set1_ps(0.5)));
                            curr_table_indices = _mm_max_epi32(curr_table_indices, _mm_set1_epi32(0));
                            curr_table_indices = _mm_min_epi32(curr_table_indices, _mm_set1_epi32(color_count - 1));

                            /* Determine the remainders. */
                            scaled_indices = _mm_cvtepi32_ps(curr_table_indices);
                            remainders = _mm_sub_ps(scaled_ratios, scaled_indices);
                            remainders = _mm_max_ps(remainders, _mm_set1_ps(0.0f));
                            remainders = _mm_min_ps(remainders, _mm_set1_ps(1.0f));
                            /* Store table indices as 32-bit integers. */
                            _mm_store_si128((__m128i *)color_bytes_aligned, curr_table_indices);
                            color_indices = (int32_t *) color_bytes_aligned;
                        }

                        /* Look up each index in the color table. */
                        for (j=0; j<WORD_PS_COUNT; j++) {
                            int color_index = color_indices[j];
                            int next_color_index = color_index + 1;

                            if (next_color_index >= color_count) {
                                next_color_index = color_count - 1;
                            }

                            /* Look up this index in the color table. */
                            color_floats_aligned[j] = colors[color_index][i];
                            next_color_floats_aligned[j] = colors[next_color_index][i];
                        }
                        color_values = _mm_load_ps(color_floats_aligned);
                        next_color_values = _mm_load_ps(next_color_floats_aligned);

                        color_differences = _mm_sub_ps(next_color_values, color_values);
                        color_values = _mm_add_ps(color_values, _mm_mul_ps(remainders, color_differences));
                    }
                    else {
                        /* Non-interpolated.  Look up the appropriate table element for each sample. */
                        if (i == 0) {
                            scaled_ratios = _mm_mul_ps(ratios, _mm_set_ps1((float)color_count));
                            scaled_ratios = _mm_sub_ps(scaled_ratios, _mm_set1_ps(0.5));

                            /* Determine the table index for each sample. */
                            curr_table_indices = _mm_cvtps_epi32(scaled_ratios);
                            curr_table_indices = _mm_max_epi32(curr_table_indices, _mm_set1_epi32(0));
                            curr_table_indices = _mm_min_epi32(curr_table_indices, _mm_set1_epi32(color_count));

                            /* Store table indices as 32-bit integers. */
                            _mm_store_si128((__m128i *)color_bytes_aligned, curr_table_indices);
                            color_indices = (int32_t *) color_bytes_aligned;
                        }

                        /* Look up each index in the color table. */
                        for (j=0; j<WORD_PS_COUNT; j++) {
                            color_floats_aligned[j] = colors[color_indices[j]][i];
                        }
                        color_values = _mm_load_ps(color_floats_aligned);
                    }

                    /* Handle start extend case */
                    if (extend_start && start_extend_mask) {
                        color_values = _mm_castsi128_ps(_mm_andnot_si128(start_extends, _mm_castps_si128(color_values)));
                        color_values = _mm_castsi128_ps(_mm_or_si128(_mm_castps_si128(color_values), _mm_and_si128 (start_extends, _mm_castps_si128(colors_start[i]))));
                    }

                    /* Handle end extend case */
                    if (extend_end && end_extend_mask) {
                        color_values = _mm_castsi128_ps(_mm_andnot_si128(end_extends, _mm_castps_si128(color_values)));
                        color_values = _mm_castsi128_ps(_mm_or_si128(_mm_castps_si128(color_values), _mm_and_si128 (end_extends, _mm_castps_si128(colors_end[i]))));
                    }

                    /* Convert color values from floating-point to 8-bit unsigned format. */
                    color_values = _mm_mul_ps(color_values, _mm_set_ps1(255.0));
                    color_values = _mm_castsi128_ps(_mm_cvtps_epi32(color_values));
                    final_colors = _mm_or_si128(final_colors, _mm_shuffle_epi8(_mm_castps_si128(color_values), pack_colors[i]));
                }

                /* Store the final 8-byte color values. */
                _mm_store_si128((__m128i *)color_bytes_aligned, final_colors);

                /* Write out any unmasked data. */
                color_ptr = (int32_t *) color_bytes_aligned;
                outdata_ptr = (int32_t *) row_ptr;

                for (j=0; j<WORD_PS_COUNT; j++) {
                    if (write_enable[j]) {
                        *(outdata_ptr++) = *(color_ptr++);
                    }
                    else {
                        outdata_ptr++;
                        color_ptr++;
                    }

                    /* Write out the mask flag. */
                    if (mask_ptr) {
                        if (write_enable[j]) {
                            *(mask_row_ptr++) = 0xFF;
                        }
                        else if (x + j < size_x) {
                            *(mask_row_ptr++) = 0x00;
                        }
                    }
                }
            }
            row_ptr += ncomp * WORD_PS_COUNT;
        }
    }
}
