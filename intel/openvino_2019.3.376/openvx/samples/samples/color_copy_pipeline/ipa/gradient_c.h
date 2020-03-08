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
#include <math.h>

#include "ipa.h"

/* Render an axial gradient into a color buffer and optional mask buffer. */
static void render_axial_gradient_C(ipa_context *ctx,
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
    ipa_byte *row_ptr = NULL;
    int ncomp = 4;
    ipa_float ratio = 0.0;

    /* Calculate the transformation matrix to rotate the gradient vector to zero degrees.
     * This allows position comparisons to be performed using only the X axis.
     */
    ipa_point axis_vector;
    ipa_float axis_length_squared;
    ipa_float axis_length;
    ipa_point axis_vector_normalized;

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

    /* Loop through the grid filling in all pixels. */
    for (y=0; y<size_y; y++) {
        /* Vector from the start point to the current point. */
        ipa_point gradient_vector;
        ipa_point gradient_vector_rotated;
        ipa_byte *mask_row_ptr = NULL;
        if (mask_ptr) {
            mask_row_ptr = mask_ptr + y * mask_rowstride;
        }
        gradient_vector.y = (y + base_y + offset_y) - axis_vector.y;

        /* Advance row pointer by any specified offset. */
        row_ptr = data_ptr + (y + offset_y) * data_rowstride + ncomp*offset_x;
        for (x=0; x<size_x; x++) {
            gradient_vector.x = (x + base_x + offset_x) - axis_vector.x;

            /* Rotate the vector to zero degrees to allow position comparison
             * using only the X component.
             *  x' =  x*cos(theta) + y*sin(theta)
             *  y' = -x*sin(theta) + y*cos(theta)
             */
            gradient_vector_rotated.x = gradient_vector.x * axis_vector_normalized.x + gradient_vector.y * axis_vector_normalized.y;
            /* Handle start extend case */
            if (gradient_vector_rotated.x < 0.0) {
                if (extend_start) {
                    /* Output start extend color. */
                    for (i=0; i<ncomp; i++) {
                        *(row_ptr++) = (ipa_byte) (colors[0][i] * 255.0);
                    }

                    /* Set mask pointer to indicate marked pixel. */
                    if (mask_ptr) {
                        *(mask_row_ptr++) = 0xFF;
                    }
                }
                else {
                    /* Not extended -- leave original pixel contents unchanged */
                    row_ptr += ncomp;

                    /* Set mask pointer to indicate unmarked pixel. */
                    if (mask_ptr) {
                        *(mask_row_ptr++) = 0x00;
                    }
                }
            }

            /* Handle end extend case */
            else if (gradient_vector_rotated.x > axis_length) {
                if (extend_end) {
                    /* Output end extend color. */
                    for (i=0; i<ncomp; i++) {
                        *(row_ptr++) = (ipa_byte) (colors[color_count-1][i] * 255.0);
                    }

                    /* Set mask pointer to indicate marked pixel. */
                    if (mask_ptr) {
                        *(mask_row_ptr++) = 0xFF;
                    }
                }
                else {
                    /* Not extended -- leave original pixel contents unchanged */
                    row_ptr += ncomp;

                    /* Set mask pointer to indicate unmarked pixel. */
                    if (mask_ptr) {
                        *(mask_row_ptr++) = 0x00;
                    }
                }
            }

            /* Main case. */
            else {
                ratio = gradient_vector_rotated.x / axis_length;

                if (color_count == 2) {
                    if (color_interpolation) {
                        for (i=0; i<ncomp; i++) {
                            *(row_ptr++) = (ipa_byte)((colors[0][i] + ratio * (colors[1][i] - colors[0][i])) * 255.0);
                        }
                    }
                    else {
                        for (i=0; i<ncomp; i++) {
                            int color_index = (int)(ratio * 2.0);
                            *(row_ptr++) = (ipa_byte)(colors[color_index][i] * 255.0);
                        }
                    }
                }
                else {
                    ipa_float color_value = 0.0;
                    ipa_float remainder = 0.0;
                    int color_index = (int)(ratio * color_count);

                    if (color_index < 0) {
                        color_index = 0;
                    }
                    else if (color_index >= color_count) {
                        color_index = color_count - 1;
                    }
                    if (color_interpolation) {
                        int next_color_index = color_index + 1;

                        remainder = (ratio * color_count) - (ipa_float)color_index;

                        if (next_color_index >= color_count) {
                            next_color_index = color_count - 1;
                        }
                        for (i=0; i<ncomp; i++) {
                            color_value = colors[color_index][i] + remainder * (colors[next_color_index][i] - colors[color_index][i]);
                            *(row_ptr++) = (ipa_byte) (color_value * 255.0);
                        }
                    }
                    else {
                        /* No interpolation -- look up color table value and set directly. */
                        for (i=0; i<ncomp; i++) {
                            color_value = colors[color_index][i];
                            *(row_ptr++) = (ipa_byte) (color_value * 255.0);
                        }
                    }
                }
                
                /* Set mask pointer to indicate marked pixel. */
                if (mask_ptr) {
                    *(mask_row_ptr++) = 0xFF;
                }
            }
        }
    }
}

/* Render a radial gradient into a color buffer and optional mask buffer. */
static void render_radial_gradient_C(ipa_context *ctx,
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
    ipa_byte *row_ptr = NULL;
    int ncomp = 4;
    ipa_float ratio = 0.0;

    /* Loop through the grid filling in all pixels. */
    for (y=0; y<size_y; y++) {
        ipa_float y_distance = y + base_y - offset_y - center.y;
        ipa_float y_distance_squared = y_distance * y_distance;
        ipa_byte *mask_row_ptr = NULL;
        if (mask_ptr) {
            mask_row_ptr = mask_ptr + y * mask_rowstride;
        }

        /* Advance row pointer by any specified offset. */
        row_ptr = data_ptr + (y + offset_y) * data_rowstride + ncomp*offset_x;
        for (x=0; x<size_x; x++) {
            ipa_float x_distance = x + base_x + offset_x - center.x;
            ipa_float x_distance_squared = x_distance * x_distance;
            ipa_float radius = sqrtf(x_distance_squared + y_distance_squared);

            /* Handle start extend case */
            if (radius < radius_start) {
                if (extend_start) {
                    /* Output start extend color. */
                    for (i=0; i<ncomp; i++) {
                        *(row_ptr++) = (ipa_byte) (colors[0][i] * 255.0);
                    }

                    /* Set mask pointer to indicate marked pixel. */
                    if (mask_ptr) {
                        *(mask_row_ptr++) = 0xFF;
                    }
                }
                else {
                    /* Not extended -- leave original pixel contents unchanged */
                    row_ptr += ncomp;

                    /* Set mask pointer to indicate unmarked pixel. */
                    if (mask_ptr) {
                        *(mask_row_ptr++) = 0x00;
                    }
                }
            }

            /* Handle end extend case */
            else if (radius > radius_end) {
                if (extend_end) {
                    /* Output end extend color. */
                    for (i=0; i<ncomp; i++) {
                        *(row_ptr++) = (ipa_byte) (colors[color_count-1][i] * 255.0);
                    }

                    /* Set mask pointer to indicate marked pixel. */
                    if (mask_ptr) {
                        *(mask_row_ptr++) = 0xFF;
                    }
                }
                else {
                    /* Not extended -- leave original pixel contents unchanged */
                    row_ptr += ncomp;

                    /* Set mask pointer to indicate unmarked pixel. */
                    if (mask_ptr) {
                        *(mask_row_ptr++) = 0x00;
                    }
                }
            }

            /* Main case. */
            else {
                ratio = (radius - radius_start) / (radius_end - radius_start);

                if (color_count == 2) {
                    if (color_interpolation) {
                        for (i=0; i<ncomp; i++) {
                            *(row_ptr++) = (ipa_byte)((colors[0][i] + ratio * (colors[1][i] - colors[0][i])) * 255.0);
                        }
                    }
                    else {
                        for (i=0; i<ncomp; i++) {
                            int color_index = (int) (ratio * 2.0);
                            *(row_ptr++) = (ipa_byte)(colors[color_index][i] * 255.0);
                        }
                    }
                }
                else {
                    ipa_float color_value = 0.0;
                    ipa_float remainder = 0.0;
                    int color_index = (int)(ratio * color_count);

                    if (color_index < 0) {
                        color_index = 0;
                    }
                    else if (color_index >= color_count) {
                        color_index = color_count - 1;
                    }
                    if (color_interpolation) {
                        int next_color_index = color_index + 1;

                        remainder = (ratio * color_count) - (ipa_float)color_index;

                        if (next_color_index >= color_count) {
                            next_color_index = color_count - 1;
                        }
                        for (i=0; i<ncomp; i++) {
                            color_value = colors[color_index][i] + remainder * (colors[next_color_index][i] - colors[color_index][i]);
                            *(row_ptr++) = (ipa_byte) (color_value * 255.0);
                        }
                    }
                    else {
                        /* No interpolation -- look up color table value and set directly. */
                        for (i=0; i<ncomp; i++) {
                            color_value = colors[color_index][i];
                            *(row_ptr++) = (ipa_byte) (color_value * 255.0);
                        }
                    }
                }

                /* Set mask pointer to indicate marked pixel. */
                if (mask_ptr) {
                    *(mask_row_ptr++) = 0xFF;
                }
            }
        }
    }
}
