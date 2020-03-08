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

/* Apply filter to zoom horizontally from src to tmp. */
static void
zoom_x1to1(uint8_t       * ipa_restrict tmp,
           const void    * ipa_restrict src,
           int                          tmp_width,
           int                          num_colors,
           const index_t * ipa_restrict index,
           const int32_t * ipa_restrict weights)
{
    int c, i;

    for (c = 0; c < num_colors; ++c)
    {
        uint8_t *ipa_restrict tp = tmp + c;
        const index_t *ipa_restrict clp = index;
        const uint8_t *ipa_restrict raster = (const uint8_t *)src + c;

        for ( i = 0; i < tmp_width; tp += num_colors, ++clp, ++i )
        {
            int weight = WEIGHT_ROUND;
            int j = clp->n;
            const uint8_t *ipa_restrict pp = raster + clp->first_pixel;
            const int32_t *ipa_restrict wp = weights + clp->index;

            for ( ; j > 0; pp += num_colors, ++wp, --j )
                 weight += *pp * *wp;
            weight >>= WEIGHT_SHIFT;
            *tp = (uint8_t)CLAMP(weight, 0, 255);
        }
    }
}

static void
zoom_x1to1_1(uint8_t       * ipa_restrict tmp,
             const void    * ipa_restrict src,
             int                          tmp_width,
             int                          num_colors,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    for ( ; tmp_width != 0; --tmp_width )
    {
        int j = index->n;
        const uint8_t *ipa_restrict pp = ((const uint8_t *)src) + index->first_pixel;
        const int32_t *ipa_restrict wp = weights + (index++)->index;
        int weight0 = WEIGHT_ROUND;

        for ( ; j > 0; --j )
            weight0 += *pp++ * *wp++;
        weight0 >>= WEIGHT_SHIFT;
        *tmp++ = (uint8_t)CLAMP(weight0, 0, 255);
    }
}

static void
zoom_x1to1_3(uint8_t       * ipa_restrict tmp,
             const void    * ipa_restrict src,
             int                          tmp_width,
             int                          num_colors,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    for ( ; tmp_width != 0; --tmp_width )
    {
        int j = index->n;
        const uint8_t *ipa_restrict pp = ((const uint8_t *)src) + index->first_pixel;
        const int32_t *ipa_restrict wp = weights + (index++)->index;
        int weight0 = WEIGHT_ROUND;
        int weight1 = WEIGHT_ROUND;
        int weight2 = WEIGHT_ROUND;

        for ( ; j > 0; --j )
        {
            int weight = *wp++;
            weight0 += *pp++ * weight;
            weight1 += *pp++ * weight;
            weight2 += *pp++ * weight;
        }
        weight0 >>= WEIGHT_SHIFT;
        weight1 >>= WEIGHT_SHIFT;
        weight2 >>= WEIGHT_SHIFT;
        *tmp++ = (uint8_t)CLAMP(weight0, 0, 255);
        *tmp++ = (uint8_t)CLAMP(weight1, 0, 255);
        *tmp++ = (uint8_t)CLAMP(weight2, 0, 255);
    }
}

static void
zoom_x1to1_4(uint8_t       * ipa_restrict tmp,
             const void    * ipa_restrict src,
             int                          tmp_width,
             int                          num_colors,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    for ( ; tmp_width != 0; --tmp_width )
    {
        int j = index->n;
        const uint8_t *ipa_restrict pp = ((const uint8_t *)src) + index->first_pixel;
        const int32_t *ipa_restrict wp = weights + (index++)->index;
        int weight0 = WEIGHT_ROUND;
        int weight1 = WEIGHT_ROUND;
        int weight2 = WEIGHT_ROUND;
        int weight3 = WEIGHT_ROUND;

        for ( ; j > 0; --j )
        {
            int weight = *wp++;
            weight0 += *pp++ * weight;
            weight1 += *pp++ * weight;
            weight2 += *pp++ * weight;
            weight3 += *pp++ * weight;
        }
        weight0 >>= WEIGHT_SHIFT;
        weight1 >>= WEIGHT_SHIFT;
        weight2 >>= WEIGHT_SHIFT;
        weight3 >>= WEIGHT_SHIFT;
        *tmp++ = (uint8_t)CLAMP(weight0, 0, 255);
        *tmp++ = (uint8_t)CLAMP(weight1, 0, 255);
        *tmp++ = (uint8_t)CLAMP(weight2, 0, 255);
        *tmp++ = (uint8_t)CLAMP(weight3, 0, 255);
    }
}

static void
zoom_x2to2(uint8_t       * ipa_restrict tmp_,
           const void    * ipa_restrict src,
           int                          tmp_width,
           int                          num_colors,
           const index_t * ipa_restrict index,
           const int32_t * ipa_restrict weights)
{
    int c, i;
    uint16_t *ipa_restrict tmp = (uint16_t *)tmp_;

    for (c = 0; c < num_colors; ++c)
    {
        uint16_t *ipa_restrict tp = tmp + c;
        const index_t *ipa_restrict clp = index;
        const uint16_t *ipa_restrict raster = (const uint16_t *)src + c;

        for ( i = 0; i < tmp_width; tp += num_colors, ++clp, ++i )
        {
            int weight = WEIGHT_ROUND<<8;
            int j = clp->n;
            const uint16_t *ipa_restrict pp = raster + clp->first_pixel;
            const int32_t *ipa_restrict wp = weights + clp->index;

            for ( ; j > 0; pp += num_colors, ++wp, --j )
                weight += *pp * *wp;
            weight >>= WEIGHT_SHIFT;
            *tp = (uint16_t)CLAMP(weight, 0, 65535);
        }
    }
}

static void
zoom_x2to2_1(uint8_t       * ipa_restrict tmp_,
             const void    * ipa_restrict src,
             int                          tmp_width,
             int                          num_colors,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    uint16_t *ipa_restrict tmp = (uint16_t *)tmp_;

    for ( ; tmp_width != 0; --tmp_width )
    {
        int j = index->n;
        const uint16_t *ipa_restrict pp = ((const uint16_t *)src) + index->first_pixel;
        const int32_t *ipa_restrict wp = weights + (index++)->index;
        int weight0 = WEIGHT_ROUND;

        for ( ; j > 0; --j )
            weight0 += *pp++ * *wp++;
        weight0 >>= WEIGHT_SHIFT;
        *tmp++ = (uint16_t)CLAMP(weight0, 0, 65535);
    }
}

static void
zoom_x2to2_3(uint8_t       * ipa_restrict tmp_,
             const void    * ipa_restrict src,
             int                          tmp_width,
             int                          num_colors,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    uint16_t *ipa_restrict tmp = (uint16_t *)tmp_;

    for ( ; tmp_width != 0; --tmp_width )
    {
        int j = index->n;
        const uint16_t *ipa_restrict pp = ((const uint16_t *)src) + index->first_pixel;
        const int32_t *ipa_restrict wp = weights + (index++)->index;
        int weight0 = WEIGHT_ROUND;
        int weight1 = WEIGHT_ROUND;
        int weight2 = WEIGHT_ROUND;

        for ( ; j > 0; --j )
        {
            int weight = *wp++;
            weight0 += *pp++ * weight;
            weight1 += *pp++ * weight;
            weight2 += *pp++ * weight;
        }
        weight0 >>= WEIGHT_SHIFT;
        weight1 >>= WEIGHT_SHIFT;
        weight2 >>= WEIGHT_SHIFT;
        *tmp++ = (uint16_t)CLAMP(weight0, 0, 65535);
        *tmp++ = (uint16_t)CLAMP(weight1, 0, 65535);
        *tmp++ = (uint16_t)CLAMP(weight2, 0, 65535);
    }
}

static void
zoom_x2to2_4(uint8_t       * ipa_restrict tmp_,
             const void    * ipa_restrict src,
             int                          tmp_width,
             int                          num_colors,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    uint16_t *ipa_restrict tmp = (uint16_t *)tmp_;

    for ( ; tmp_width != 0; --tmp_width )
    {
        int j = index->n;
        const uint16_t *ipa_restrict pp = ((const uint16_t *)src) + index->first_pixel;
        const int32_t *ipa_restrict wp = weights + (index++)->index;
        int weight0 = WEIGHT_ROUND;
        int weight1 = WEIGHT_ROUND;
        int weight2 = WEIGHT_ROUND;
        int weight3 = WEIGHT_ROUND;

        for ( ; j > 0; --j )
        {
            int weight = *wp++;
            weight0 += *pp++ * weight;
            weight1 += *pp++ * weight;
            weight2 += *pp++ * weight;
            weight3 += *pp++ * weight;
        }
        weight0 >>= WEIGHT_SHIFT;
        weight1 >>= WEIGHT_SHIFT;
        weight2 >>= WEIGHT_SHIFT;
        weight3 >>= WEIGHT_SHIFT;
        *tmp++ = (uint16_t)CLAMP(weight0, 0, 65536);
        *tmp++ = (uint16_t)CLAMP(weight1, 0, 65535);
        *tmp++ = (uint16_t)CLAMP(weight2, 0, 65535);
        *tmp++ = (uint16_t)CLAMP(weight3, 0, 65535);
    }
}

/*
 * Apply filter to zoom vertically from tmp to dst.
 * This is simpler because we can treat all columns identically
 * without regard to the number of samples per pixel.
 */
static inline void
zoom_y1to1_4(void          * ipa_restrict dst,
             const uint8_t * ipa_restrict tmp,
             int                          width,
             int                          byte_stride,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    int first_pixel = index->first_pixel;
    int w0          = weights[index->index];
    int w1          = weights[index->index+1];
    int w2          = weights[index->index+2];
    int w3          = weights[index->index+3];
    uint8_t *ipa_restrict d;

    tmp += first_pixel;
    d = (uint8_t *)dst;
    for (; width > 0; width--)
    {
        int weight;

        weight  = tmp[            0] * w0;
        weight += tmp[  byte_stride] * w1;
        weight += tmp[2*byte_stride] * w2;
        weight += tmp[3*byte_stride] * w3;
        tmp++;

        weight = (weight + WEIGHT_ROUND)>>WEIGHT_SHIFT;
        *d++ = (uint8_t)CLAMP(weight, 0, 0xff);
    }
}

static inline void
zoom_y1to1_5(void          * ipa_restrict dst,
             const uint8_t * ipa_restrict tmp,
             int                          width,
             int                          byte_stride,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    int w0 = cbp[0];
    int w1 = cbp[1];
    int w2 = cbp[2];
    int w3 = cbp[3];
    int w4 = cbp[4];
    uint8_t *ipa_restrict d;

    tmp += first_pixel;
    d = (uint8_t *)dst;
    for (; width > 0; width--)
    {
        int weight;

        weight  = tmp[            0] * w0;
        weight += tmp[  byte_stride] * w1;
        weight += tmp[2*byte_stride] * w2;
        weight += tmp[3*byte_stride] * w3;
        weight += tmp[4*byte_stride] * w4;
        tmp++;

        weight = (weight + WEIGHT_ROUND)>>WEIGHT_SHIFT;
        *d++ = (uint8_t)CLAMP(weight, 0, 0xff);
    }
}

static inline void
template_zoom_y1to1(void          * ipa_restrict dst,
                    const uint8_t * ipa_restrict tmp,
                    int                          width,
                    int                          byte_stride,
                    const index_t * ipa_restrict index,
                    const int32_t * ipa_restrict weights,
                    int                          n)
{
    int cn = index->n;
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint8_t *ipa_restrict d;

    tmp += first_pixel;
    d = (uint8_t *)dst;
    for (; width > 0; width--)
    {
        int weight = WEIGHT_ROUND;
        const uint8_t *ipa_restrict pp = tmp++;
        int j;
        const int32_t *wp = cbp;

        for (j = cn; j > 0; pp += byte_stride, ++wp, --j)
            weight += *pp * *wp;
        weight >>= WEIGHT_SHIFT;
        *d++ = (uint8_t)CLAMP(weight, 0, 0xff);
    }
}

static void zoom_y1to1(void          * ipa_restrict dst,
                       const uint8_t * ipa_restrict tmp,
                       int                          width,
                       int                          byte_stride,
                       const index_t * ipa_restrict index,
                       const int32_t * ipa_restrict weights)
{
    switch(index->n)
    {
        case 4:
            zoom_y1to1_4(dst, tmp, width, byte_stride, index, weights);
            break;
        case 5:
            zoom_y1to1_5(dst, tmp, width, byte_stride, index, weights);
            break;
        default:
            template_zoom_y1to1(dst, tmp, width, byte_stride, index, weights, index->n);
            break;
    }
}

static inline void
zoom_y2to1_4(void          * ipa_restrict dst,
             const uint8_t * ipa_restrict tmp_,
             int                          width,
             int                          byte_stride,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    int first_pixel = index->first_pixel;
    int w0          = weights[index->index];
    int w1          = weights[index->index+1];
    int w2          = weights[index->index+2];
    int w3          = weights[index->index+3];
    uint8_t *ipa_restrict d;
    const uint16_t *ipa_restrict tmp = (const uint16_t *)tmp_;

    tmp += first_pixel;
    d = (uint8_t *)dst;
    for (; width > 0; width--)
    {
        int weight;

        weight  = tmp[            0] * w0;
        weight += tmp[  byte_stride] * w1;
        weight += tmp[2*byte_stride] * w2;
        weight += tmp[3*byte_stride] * w3;
        tmp++;

        weight = (weight + WEIGHT_ROUND)>>(WEIGHT_SHIFT+8);
        *d++ = (uint8_t)CLAMP(weight, 0, 0xff);
    }
}

static inline void
zoom_y2to1_5(void          * ipa_restrict dst,
             const uint8_t * ipa_restrict tmp_,
             int                          width,
             int                          byte_stride,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    int w0 = cbp[0];
    int w1 = cbp[1];
    int w2 = cbp[2];
    int w3 = cbp[3];
    int w4 = cbp[4];
    uint8_t *ipa_restrict d;
    const uint16_t *ipa_restrict tmp = (const uint16_t *)tmp_;

    tmp += first_pixel;
    d = (uint8_t *)dst;
    for (; width > 0; width--)
    {
        int weight;

        weight  = tmp[            0] * w0;
        weight += tmp[  byte_stride] * w1;
        weight += tmp[2*byte_stride] * w2;
        weight += tmp[3*byte_stride] * w3;
        weight += tmp[4*byte_stride] * w4;
        tmp++;

        weight = (weight + WEIGHT_ROUND)>>(WEIGHT_SHIFT+8);
        *d++ = (uint8_t)CLAMP(weight, 0, 0xff);
    }
}

static inline void
template_zoom_y2to1(void          * ipa_restrict dst,
                    const uint8_t * ipa_restrict tmp_,
                    int                          width,
                    int                          byte_stride,
                    const index_t * ipa_restrict index,
                    const int32_t * ipa_restrict weights,
                    int                          n)
{
    int cn = index->n;
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint8_t *ipa_restrict d;
    const uint16_t *ipa_restrict tmp = (const uint16_t *)tmp_;

    tmp += first_pixel;
    d = (uint8_t *)dst;
    for (; width > 0; width--)
    {
        int weight = WEIGHT_ROUND;
        const uint16_t *ipa_restrict pp = tmp++;
        int j;
        const int32_t *wp = cbp;

        for (j = cn; j > 0; pp += byte_stride, ++wp, --j)
            weight += *pp * *wp;
        weight >>= WEIGHT_SHIFT+8;
        *d++ = (uint8_t)CLAMP(weight, 0, 0xff);
    }
}

static void zoom_y2to1(void          * ipa_restrict dst,
                       const uint8_t * ipa_restrict tmp,
                       int                          width,
                       int                          byte_stride,
                       const index_t * ipa_restrict index,
                       const int32_t * ipa_restrict weights)
{
    byte_stride >>= 1;
    switch(index->n)
    {
        case 4:
            zoom_y2to1_4(dst, tmp, width, byte_stride, index, weights);
            break;
        case 5:
            zoom_y2to1_5(dst, tmp, width, byte_stride, index, weights);
            break;
        default:
            template_zoom_y2to1(dst, tmp, width, byte_stride, index, weights, index->n);
            break;
    }
}

static inline void
zoom_y1to2_2(void          * ipa_restrict dst,
             const uint8_t * ipa_restrict tmp,
             int                          width,
             int                          byte_stride,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d;
    int w0 = cbp[0];
    int w1 = cbp[1];

    tmp += first_pixel;
    d = (uint16_t *)dst;

    for (; width > 0; width--)
    {
        int weight;
        int pixel;

        weight  = tmp[          0] * w0;
        weight += tmp[byte_stride] * w1;
        tmp++;
        pixel = (weight + (WEIGHT_ROUND>>8)) >> (WEIGHT_SHIFT-8);
        *d++ = (uint16_t)CLAMP(pixel, 0, 0xffff);
    }
}

static inline void
zoom_y1to2_3(void          * ipa_restrict dst,
             const uint8_t * ipa_restrict tmp,
             int                          width,
             int                          byte_stride,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d;
    int w0 = cbp[0];
    int w1 = cbp[1];
    int w2 = cbp[2];

    tmp += first_pixel;
    d = (uint16_t *)dst;
    for (; width > 0; width--)
    {
        int weight;
        int pixel;

        weight  = tmp[              0] * w0;
        weight += tmp[    byte_stride] * w1;
        weight += tmp[2 * byte_stride] * w2;
        tmp++;
        pixel = (weight + (WEIGHT_ROUND>>8)) >> (WEIGHT_SHIFT-8);
        *d++ = (uint16_t)CLAMP(pixel, 0, 0xffff);
    }
}

static inline void
zoom_y1to2_4(void          * ipa_restrict dst,
             const uint8_t * ipa_restrict tmp,
             int             width,
             int             byte_stride,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d;
    int w0 = cbp[0];
    int w1 = cbp[1];
    int w2 = cbp[2];
    int w3 = cbp[3];

    tmp += first_pixel;
    d = (uint16_t *)dst;
    for (; width > 0; width--)
    {
        int weight;
        int pixel;

        weight  = tmp[            0] * w0;
        weight += tmp[  byte_stride] * w1;
        weight += tmp[2*byte_stride] * w2;
        weight += tmp[3*byte_stride] * w3;
        tmp++;
        pixel = (weight + (WEIGHT_ROUND>>8))>>(WEIGHT_SHIFT-8);
        *d++ = (uint16_t)CLAMP(pixel, 0, 0xffff);
    }
}

static inline void
zoom_y1to2_5(void          * ipa_restrict dst,
             const uint8_t * ipa_restrict tmp,
             int                          width,
             int                          byte_stride,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d;
    int w0 = cbp[0];
    int w1 = cbp[1];
    int w2 = cbp[2];
    int w3 = cbp[3];
    int w4 = cbp[4];

    tmp += first_pixel;
    d = (uint16_t *)dst;
    for (; width > 0; width--)
    {
        int weight;
        int pixel;

        weight  = tmp[            0] * w0;
        weight += tmp[  byte_stride] * w1;
        weight += tmp[2*byte_stride] * w2;
        weight += tmp[3*byte_stride] * w3;
        weight += tmp[4*byte_stride] * w4;
        tmp++;
        pixel = (weight + (WEIGHT_ROUND>>8))>>(WEIGHT_SHIFT-8);
        *d++ = (uint16_t)CLAMP(pixel, 0, 0xffff);
    }
}

static inline void
zoom_y1to2_n(void          * ipa_restrict dst,
             const uint8_t * ipa_restrict tmp,
             int                          width,
             int                          byte_stride,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    int cn = index->n;
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d;

    tmp += first_pixel;
    d = (uint16_t *)dst;
    for (; width > 0; width--)
    {
        int weight = 0;
        const uint8_t *ipa_restrict pp = tmp++;
        int pixel, j;
        const int32_t *ipa_restrict wp = cbp;

        for (j = cn; j > 0; pp += byte_stride, ++wp, --j)
            weight += *pp * *wp;
        pixel = (weight + (WEIGHT_ROUND>>8))>>(WEIGHT_SHIFT-8);
        *d++ = (uint16_t)CLAMP(pixel, 0, 0xffff);
    }
}

static void
zoom_y1to2(void          * ipa_restrict dst,
           const uint8_t * ipa_restrict tmp,
           int                          width,
           int                          byte_stride,
           const index_t * ipa_restrict index,
           const int32_t * ipa_restrict weights)
{
    switch (index->n)
    {
        case 2:
            zoom_y1to2_2(dst, tmp, width, byte_stride, index, weights);
            break;
        case 3:
            zoom_y1to2_3(dst, tmp, width, byte_stride, index, weights);
            break;
        case 4:
            zoom_y1to2_4(dst, tmp, width, byte_stride, index, weights);
            break;
        case 5:
            zoom_y1to2_5(dst, tmp, width, byte_stride, index, weights);
            break;
        default:
            zoom_y1to2_n(dst, tmp, width, byte_stride, index, weights);
            break;
    }
}

static inline void
zoom_y2to2_2(void          * ipa_restrict dst,
             const uint8_t * ipa_restrict tmp_,
             int                          width,
             int                          byte_stride,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d;
    int w0 = cbp[0];
    int w1 = cbp[1];
    const uint16_t * ipa_restrict tmp = (const uint16_t *)tmp_;

    tmp += first_pixel;
    d = (uint16_t *)dst;

    for (; width > 0; width--)
    {
        int weight;
        int pixel;

        weight  = tmp[          0] * w0;
        weight += tmp[byte_stride] * w1;
        tmp++;
        pixel = (weight + WEIGHT_ROUND) >> WEIGHT_SHIFT;
        *d++ = (uint16_t)CLAMP(pixel, 0, 0xffff);
    }
}

static inline void
zoom_y2to2_3(void          * ipa_restrict dst,
             const uint8_t * ipa_restrict tmp_,
             int                          width,
             int                          byte_stride,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d;
    int w0 = cbp[0];
    int w1 = cbp[1];
    int w2 = cbp[2];
    const uint16_t * ipa_restrict tmp = (const uint16_t *)tmp_;

    tmp += first_pixel;
    d = (uint16_t *)dst;
    for (; width > 0; width--)
    {
        int weight;
        int pixel;

        weight  = tmp[              0] * w0;
        weight += tmp[    byte_stride] * w1;
        weight += tmp[2 * byte_stride] * w2;
        tmp++;
        pixel = (weight + WEIGHT_ROUND) >> WEIGHT_SHIFT;
        *d++ = (uint16_t)CLAMP(pixel, 0, 0xffff);
    }
}

static inline void
zoom_y2to2_4(void          * ipa_restrict dst,
             const uint8_t * ipa_restrict tmp_,
             int             width,
             int             byte_stride,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d;
    int w0 = cbp[0];
    int w1 = cbp[1];
    int w2 = cbp[2];
    int w3 = cbp[3];
    const uint16_t * ipa_restrict tmp = (const uint16_t *)tmp_;

    tmp += first_pixel;
    d = (uint16_t *)dst;
    for (; width > 0; width--)
    {
        int weight;
        int pixel;

        weight  = tmp[            0] * w0;
        weight += tmp[  byte_stride] * w1;
        weight += tmp[2*byte_stride] * w2;
        weight += tmp[3*byte_stride] * w3;
        tmp++;
        pixel = (weight + WEIGHT_ROUND)>>WEIGHT_SHIFT;
        *d++ = (uint16_t)CLAMP(pixel, 0, 0xffff);
    }
}

static inline void
zoom_y2to2_5(void          * ipa_restrict dst,
             const uint8_t * ipa_restrict tmp_,
             int                          width,
             int                          byte_stride,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d;
    int w0 = cbp[0];
    int w1 = cbp[1];
    int w2 = cbp[2];
    int w3 = cbp[3];
    int w4 = cbp[4];
    const uint16_t * ipa_restrict tmp = (const uint16_t *)tmp_;

    tmp += first_pixel;
    d = (uint16_t *)dst;
    for (; width > 0; width--)
    {
        int weight;
        int pixel;

        weight  = tmp[            0] * w0;
        weight += tmp[  byte_stride] * w1;
        weight += tmp[2*byte_stride] * w2;
        weight += tmp[3*byte_stride] * w3;
        weight += tmp[4*byte_stride] * w4;
        tmp++;
        pixel = (weight + WEIGHT_ROUND)>>WEIGHT_SHIFT;
        *d++ = (uint16_t)CLAMP(pixel, 0, 0xffff);
    }
}

static inline void
zoom_y2to2_n(void          * ipa_restrict dst,
             const uint8_t * ipa_restrict tmp_,
             int                          width,
             int                          byte_stride,
             const index_t * ipa_restrict index,
             const int32_t * ipa_restrict weights)
{
    int cn = index->n;
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d;
    const uint16_t * ipa_restrict tmp = (const uint16_t *)tmp_;

    tmp += first_pixel;
    d = (uint16_t *)dst;
    for (; width > 0; width--)
    {
        int weight = 0;
        const uint16_t *ipa_restrict pp = tmp++;
        int pixel, j;
        const int32_t *ipa_restrict wp = cbp;

        for (j = cn; j > 0; pp += byte_stride, ++wp, --j)
            weight += *pp * *wp;
        pixel = (weight + WEIGHT_ROUND)>>WEIGHT_SHIFT;
        *d++ = (uint16_t)CLAMP(pixel, 0, 0xffff);
    }
}

static void
zoom_y2to2(void          * ipa_restrict dst,
           const uint8_t * ipa_restrict tmp,
           int                          width,
           int                          byte_stride,
           const index_t * ipa_restrict index,
           const int32_t * ipa_restrict weights)
{
    byte_stride >>= 1;
    switch (index->n)
    {
        case 2:
            zoom_y2to2_2(dst, tmp, width, byte_stride, index, weights);
            break;
        case 3:
            zoom_y2to2_3(dst, tmp, width, byte_stride, index, weights);
            break;
        case 4:
            zoom_y2to2_4(dst, tmp, width, byte_stride, index, weights);
            break;
        case 5:
            zoom_y2to2_5(dst, tmp, width, byte_stride, index, weights);
            break;
        default:
            zoom_y2to2_n(dst, tmp, width, byte_stride, index, weights);
            break;
    }
}

static inline void
zoom_y1to2_frac_4(void          * ipa_restrict dst,
                  const uint8_t * ipa_restrict tmp,
                  int                          width,
                  int                          byte_stride,
                  const index_t * ipa_restrict index,
                  const int32_t * ipa_restrict weights)
{
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d;
    int w0 = cbp[0];
    int w1 = cbp[1];
    int w2 = cbp[2];
    int w3 = cbp[3];

    tmp += first_pixel;
    d = (uint16_t *)dst;
    for (; width > 0; width--)
    {
        int weight;
        int pixel;

        weight  = tmp[            0]  * w0;
        weight += tmp[  byte_stride]  * w1;
        weight += tmp[2*byte_stride]  * w2;
        weight += tmp[3*byte_stride]  * w3;
        tmp++;
        pixel = (weight + WEIGHT_ROUND)>>WEIGHT_SHIFT;
        *d++ = (uint16_t)CLAMP(pixel, 0, frac_1);
    }
}

static inline void
zoom_y1to2_frac_5(void          * ipa_restrict dst,
                  const uint8_t * ipa_restrict tmp,
                  int                          width,
                  int                          byte_stride,
                  const index_t * ipa_restrict index,
                  const int32_t * ipa_restrict weights)
{
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d;
    int w0 = cbp[0];
    int w1 = cbp[1];
    int w2 = cbp[2];
    int w3 = cbp[3];
    int w4 = cbp[4];

    tmp += first_pixel;
    d = (uint16_t *)dst;
    for (; width > 0; width--)
    {
        int weight;
        int pixel;

        weight  = tmp[            0]  * w0;
        weight += tmp[  byte_stride]  * w1;
        weight += tmp[2*byte_stride]  * w2;
        weight += tmp[3*byte_stride]  * w3;
        weight += tmp[4*byte_stride]  * w4;
        tmp++;
        pixel = (weight + WEIGHT_ROUND)>>WEIGHT_SHIFT;
        *d++ = (uint16_t)CLAMP(pixel, 0, frac_1);
    }
}

static inline void
zoom_y1to2_frac_n(void          * ipa_restrict dst,
                  const uint8_t * ipa_restrict tmp,
                  int                          width,
                  int                          byte_stride,
                  const index_t * ipa_restrict index,
                  const int32_t * ipa_restrict weights)
{
    int cn = index->n;
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d;

    tmp += first_pixel;
    d = (uint16_t *)dst;
    for (; width > 0; width--)
    {
        int weight = 0;
        const uint8_t *ipa_restrict pp = tmp++;
        int pixel, j;
        const int32_t *ipa_restrict wp = cbp;

        for (j = cn; j > 0; pp += byte_stride, ++wp, --j)
            weight += *pp * *wp;
        pixel = (weight + WEIGHT_ROUND)>>WEIGHT_SHIFT;
        *d++ = (uint16_t)CLAMP(pixel, 0, frac_1);
    }
}

static void
zoom_y1to2_frac(void          * ipa_restrict dst,
                const uint8_t * ipa_restrict tmp,
                int                          width,
                int                          byte_stride,
                const index_t * ipa_restrict index,
                const int32_t * ipa_restrict weights)
{
    switch (index->n)
    {
        case 4:
            zoom_y1to2_frac_4(dst, tmp, width, byte_stride, index, weights);
            break;
        case 5:
            zoom_y1to2_frac_5(dst, tmp, width, byte_stride, index, weights);
            break;
        default:
            zoom_y1to2_frac_n(dst, tmp, width, byte_stride, index, weights);
            break;
    }
}

static inline void
zoom_y2to2_frac_4(void          * ipa_restrict dst,
                  const uint8_t * ipa_restrict tmp_,
                  int                          width,
                  int                          byte_stride,
                  const index_t * ipa_restrict index,
                  const int32_t * ipa_restrict weights)
{
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d;
    int w0 = cbp[0];
    int w1 = cbp[1];
    int w2 = cbp[2];
    int w3 = cbp[3];
    const uint16_t *ipa_restrict tmp = (const uint16_t *)tmp_;

    tmp += first_pixel;
    d = (uint16_t *)dst;
    for (; width > 0; width--)
    {
        int weight;
        int pixel;

        weight  = tmp[            0]  * w0;
        weight += tmp[  byte_stride]  * w1;
        weight += tmp[2*byte_stride]  * w2;
        weight += tmp[3*byte_stride]  * w3;
        tmp++;
        pixel = (weight + WEIGHT_ROUND)>>WEIGHT_SHIFT;
        *d++ = (uint16_t)CLAMP(pixel, 0, frac_1);
    }
}

static inline void
zoom_y2to2_frac_5(void          * ipa_restrict dst,
                  const uint8_t * ipa_restrict tmp_,
                  int                          width,
                  int                          byte_stride,
                  const index_t * ipa_restrict index,
                  const int32_t * ipa_restrict weights)
{
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d;
    int w0 = cbp[0];
    int w1 = cbp[1];
    int w2 = cbp[2];
    int w3 = cbp[3];
    int w4 = cbp[4];
    const uint16_t *ipa_restrict tmp = (const uint16_t *)tmp_;

    tmp += first_pixel;
    d = (uint16_t *)dst;
    for (; width > 0; width--)
    {
        int weight;
        int pixel;

        weight  = tmp[            0]  * w0;
        weight += tmp[  byte_stride]  * w1;
        weight += tmp[2*byte_stride]  * w2;
        weight += tmp[3*byte_stride]  * w3;
        weight += tmp[4*byte_stride]  * w4;
        tmp++;
        pixel = (weight + WEIGHT_ROUND)>>WEIGHT_SHIFT;
        *d++ = (uint16_t)CLAMP(pixel, 0, frac_1);
    }
}

static inline void
zoom_y2to2_frac_n(void          * ipa_restrict dst,
                  const uint8_t * ipa_restrict tmp_,
                  int                          width,
                  int                          byte_stride,
                  const index_t * ipa_restrict index,
                  const int32_t * ipa_restrict weights)
{
    int cn = index->n;
    int first_pixel = index->first_pixel;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d;
    const uint16_t *ipa_restrict tmp = (const uint16_t *)tmp_;

    tmp += first_pixel;
    d = (uint16_t *)dst;
    for (; width > 0; width--)
    {
        int weight = 0;
        const uint16_t *ipa_restrict pp = tmp++;
        int pixel, j;
        const int32_t *ipa_restrict wp = cbp;

        for (j = cn; j > 0; pp += byte_stride, ++wp, --j)
            weight += *pp * *wp;
        pixel = (weight + WEIGHT_ROUND)>>WEIGHT_SHIFT;
        *d++ = (uint16_t)CLAMP(pixel, 0, frac_1);
    }
}

static void
zoom_y2to2_frac(void          * ipa_restrict dst,
                const uint8_t * ipa_restrict tmp,
                int                          width,
                int                          byte_stride,
                const index_t * ipa_restrict index,
                const int32_t * ipa_restrict weights)
{
    byte_stride >>= 1;
    switch (index->n)
    {
        case 4:
            zoom_y2to2_frac_4(dst, tmp, width, byte_stride, index, weights);
            break;
        case 5:
            zoom_y2to2_frac_5(dst, tmp, width, byte_stride, index, weights);
            break;
        default:
            zoom_y2to2_frac_n(dst, tmp, width, byte_stride, index, weights);
            break;
    }
}
