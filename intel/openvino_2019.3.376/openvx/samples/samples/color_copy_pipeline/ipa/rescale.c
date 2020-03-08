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

#include "ipa-impl.h"

#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

//#define DEBUG_RESCALER

//int errprintf_nomem(const char *string, ...);
//#define printf errprintf_nomem

/*
 *    Image scaling code is based on public domain code from
 *      Graphics Gems III (pp. 414-424), Academic Press, 1992.
 */

 #define frac_1 ((int16_t)0x7ff8)

/* ---------------- ImageScaleEncode/Decode ---------------- */

#define WEIGHT_SHIFT 12
#define WEIGHT_SCALE (1<<WEIGHT_SHIFT)
#define WEIGHT_ROUND (1<<(WEIGHT_SHIFT-1))

/* Auxiliary structures. */
typedef struct
{
    uint32_t index;       /* index of first element in list of */
                          /* contributors */
    uint16_t n;           /* number of contributors */
                          /* (not multiplied by stride) */
    uint16_t slow;        /* Flag */
    uint32_t first_pixel; /* offset of first value in source data */
    uint32_t last_pixel;  /* last pixel number */
} index_t;

typedef void (zoom_y_fn)(void          *              dst,
                         const uint8_t * ipa_restrict tmp,
                         int                          sample_width,
                         int                          byte_stride,
                         const index_t * ipa_restrict index,
                         const int32_t * ipa_restrict weights);
typedef void (zoom_x_fn)(uint8_t       * ipa_restrict tmp,
                         const void    * ipa_restrict src,
                         int                          tmp_width,
                         int                          num_colors,
                         const index_t * ipa_restrict index,
                         const int32_t * ipa_restrict weights);

#define CLAMP(v, mn, mx)\
  (v < mn ? mn : v > mx ? mx : v)

/* Include the cores */
#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 0
#include "rescale_c.h"
#endif

#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 1
#include "rescale_sse.h"
#endif

struct ipa_rescaler_s
{
    /* The init procedure sets the following. */
    ipa_context  *ctx;
    void         *src;
    uint8_t      *tmp;
    uint32_t      src_h;
    uint32_t      dst_h;
    uint32_t      data_offset;
    uint32_t      dst_patch_x;
    uint32_t      dst_patch_y;
    uint32_t      dst_patch_w;
    uint32_t      dst_patch_h;
    uint32_t      src_patch_y;
    uint32_t      src_patch_h;
    uint32_t      channels;
    uint32_t      tmp_stride;
    uint32_t      max_support;
    index_t      *index_x;
    int32_t      *weights_x;
    index_t      *index_y;
    int32_t      *weights_y;
    /* The following are updated dynamically. */
    uint32_t      src_y;
    uint32_t      dst_y;
    zoom_y_fn    *zoom_y;
    zoom_x_fn    *zoom_x;
};

/* ------ Digital filter definition ------ */

#define LOG2_MAX_ISCALE_SUPPORT 3
#define MAX_ISCALE_SUPPORT (1 << LOG2_MAX_ISCALE_SUPPORT)

typedef struct
{
    double   (*filter)(double);
    uint32_t   filter_width;
    uint32_t (*contrib_pixels)(double scale);
    double     min_scale;
} filter_defn_t;

/* Mitchell filter definition */
#define Mitchell_support 2
#define Mitchell_min_scale ((Mitchell_support * 2) / (MAX_ISCALE_SUPPORT - 1.01))
#define B (1.0 / 3.0)
#define C (1.0 / 3.0)
static double
Mitchell_filter(double t)
{
    double t2 = t * t;

    if (t < 0)
        t = -t;

    if (t < 1)
        return
            ((12 - 9 * B - 6 * C) * (t * t2) +
             (-18 + 12 * B + 6 * C) * t2 +
             (6 - 2 * B)) / 6;
    else if (t < 2)
        return
            ((-1 * B - 6 * C) * (t * t2) +
             (6 * B + 30 * C) * t2 +
             (-12 * B - 48 * C) * t +
             (8 * B + 24 * C)) / 6;
    else
        return 0;
}

/* DogLeg Interpolated filter definition */
#define DogLeg_support 1
#define DogLeg_min_scale 0
static double
DogLeg_filter(double t)
{
    if (t < 0)
        t = -t;

    if (t >= 1)
        return 0;
    return 1 + (2*t -3)*t*t;
}

/* Linear Interpolated filter definition */
#define Linear_support 1
#define Linear_min_scale 0
static double
Linear_filter(double t)
{
    if (t < 0)
        t = -t;

    if (t >= 1)
        return 0;
    return t;
}

/* Nearest Neighbour filter definition */
#define Nearest_support 1
#define Nearest_min_scale 0
static double
Nearest_filter(double t)
{
    if (t < 0)
        t = -t;

    if (t > 0.5)
        return 0;
    return 1;
}

/*
 * The environment provides the following definitions:
 *      double fproc(double t)
 *      double fWidthIn
 *      PixelTmp {min,max,unit}PixelTmp
 */

/* ------ Auxiliary procedures ------ */

/* Calculate the support for a given scale. */
/* The value is always in the range 1..max_support (was MAX_ISCALE_SUPPORT). */
static uint32_t
Mitchell_contrib_pixels(double scale)
{
    if (scale == 0.0)
        return 1;
    if (scale > 1)
        scale = 1;
    else if (scale < Mitchell_min_scale)
        scale = Mitchell_min_scale;
    return (uint32_t)(((float)Mitchell_support) / scale * 2 + 1.5);
}

static uint32_t
DogLeg_contrib_pixels(double scale)
{
    if (scale == 0.0)
        return 1;
    return (uint32_t)(((float)DogLeg_support) / (scale >= 1.0 ? 1.0 : scale)
                 * 2 + 1.5);
}

static uint32_t
Linear_contrib_pixels(double scale)
{
    if (scale == 0.0)
        return 1;
    return (uint32_t)(((float)Linear_support) / (scale >= 1.0 ? 1.0 : scale)
                 * 2 + 1.5);
}

static uint32_t
Nearest_contrib_pixels(double scale)
{
    return 1;
}

/* Pre-calculate filter contributions for a row or a column. */
/* Return the highest input pixel index used. */
static void
calculate_contrib(index_t             *index,   /* Return weight list parameters in index[0 .. size-1]. */
                  int32_t             *weights, /* Store weights in weights[0 .. contrib_pixels(scale)*size-1]. */
                                                /* (Less space than this may actually be needed.) */
                  uint32_t             dst_w,
                  uint32_t             src_w,
                  uint32_t             patch_x,
                  uint32_t             patch_w,
                  uint32_t             data_x,
                  uint32_t             data_w,
                  uint32_t             stride,  /* Number of samples between pixels */
                  uint32_t             max_value,
                  double               rescale_factor,
                  const filter_defn_t *filter)
{
    double   scale          = ((double)dst_w)/src_w;
    int      fWidthIn       = filter->filter_width;
    double (*fproc)(double) = filter->filter;
    double   min_scale      = filter->min_scale;
    double   WidthIn, fscale;
    int      squeeze;
    int      npixels;
    uint32_t i;
    int      j;
    uint32_t sse_limit      = data_w * stride - 16;

    if (scale < 1.0)
    {
        double clamped_scale = scale;
        if (scale < min_scale)
            scale = min_scale;
        WidthIn = ((double)fWidthIn) / clamped_scale;
        fscale = 1.0 / clamped_scale;
        squeeze = 1;
    }
    else
    {
        WidthIn = (double)fWidthIn;
        fscale = 1.0;
        squeeze = 0;
    }
    npixels = (int)(WidthIn * 2 + 1);
    if (npixels < 4)
        npixels = 4;

    for (i = 0; i < patch_w; ++i)
    {
        /* Here we need :
           double scale = (double)dst_size / src_size;
           float dst_offset_fraction = floor(dst_offset) - dst_offset;
           double center = (i + dst_offset_fraction + 0.5) / scale - 0.5;
           int left = (int)ceil(center - WidthIn);
           int right = (int)floor(center + WidthIn);
           We can't compute 'right' in floats because float arithmetics is not associative.
           In older versions tt caused a 1 pixel bias of image bands due to
           rounding direction appears to depend on src_y_offset. So compute in rationals.
           Since pixel center fall to half integers, we subtract 0.5
           in the image space and add 0.5 in the device space.
         */
        int center_denom = dst_w * 2;
        int64_t center_num = /* center * center_denom * 2 = */
            (i+patch_x) * (int64_t)src_w * 2 + src_w - dst_w;
        int left = (int)ceil((center_num - WidthIn * center_denom) / center_denom);
        int right = (int)floor((center_num + WidthIn * center_denom) / center_denom);
        double center = (double)center_num / center_denom;
#define clamp_pixel(j) (j < 0 ? 0 : j >= (int)src_w ? (int)src_w - 1 : j)
        int first_pixel = clamp_pixel(left);
        int last_pixel = clamp_pixel(right);
        int32_t *p;

        index[i].first_pixel = (first_pixel - data_x) * stride;
        index[i].n = last_pixel - first_pixel + 1;
        index[i].index = i * npixels;
        index[i].last_pixel = last_pixel - data_x;
        /* slow set indicates that an SSE load of 16 bytes from the first pixel position would overrun the end of the line */
        index[i].slow = (index[i].first_pixel > sse_limit);
        p = weights + index[i].index;
        for (j = 0; j < npixels; ++j)
            p[j] = 0;
        if (squeeze)
        {
            double sum = 0;
            for (j = left; j <= right; ++j)
                sum += fproc((center - j) / fscale) / fscale;
            for (j = left; j <= right; ++j)
            {
                double weight = fproc((center - j) / fscale) / fscale / sum;
                int n = clamp_pixel(j);
                int k = n - first_pixel;

                p[k] +=
                    (int)((weight * rescale_factor) * WEIGHT_SCALE + 0.5);
            }
        }
        else
        {
            double sum = 0;
            for (j = left; j <= right; ++j)
                sum += fproc(center - j);
            for (j = left; j <= right; ++j)
            {
                double weight = fproc(center - j) / sum;
                int n = clamp_pixel(j);
                int k = n - first_pixel;

                p[k] +=
                    (int)((weight * rescale_factor) * WEIGHT_SCALE + 0.5);
            }
        }
    }
}

static void
permute_contribs(index_t *index,
                 int32_t *weights,
                 int      dst_h,
                 int      max_entries,
                 int      stride)
{
    int i, j;

    for (i = dst_h-1; i >= 0; i--)
    {
        int out = (i+1)*max_entries;
        int first = index[i].first_pixel / stride;
        int idx   = index[i].index;
        int n     = index[i].n;
        int out_first = first % max_entries;
        int out_last = (first+n-1) % max_entries;

        index[i].index = out;
        if (out_first <= out_last)
        {
            index[i].first_pixel = out_first * stride;
            while (n-- > 0)
                weights[out++] = weights[idx++];
        }
        else
        {
            index[i].first_pixel = 0;
            index[i].n = max_entries;
            for (j = 0; j <= out_last; j++)
                weights[out++] = weights[max_entries - out_first + idx + j];
            for (; j < out_first; j++)
                weights[out++] = 0;
            for (; j < max_entries; j++)
                weights[out++] = weights[idx++];
        }

    }
}

static const filter_defn_t Mitchell_defn =
{
    Mitchell_filter,
    Mitchell_support,
    Mitchell_contrib_pixels,
    Mitchell_min_scale
};

static const filter_defn_t DogLeg_defn =
{
    DogLeg_filter,
    DogLeg_support,
    DogLeg_contrib_pixels,
    DogLeg_min_scale
};

static const filter_defn_t Linear_defn =
{
    Linear_filter,
    Linear_support,
    Linear_contrib_pixels,
    Linear_min_scale
};

static const filter_defn_t Nearest_defn =
{
    Nearest_filter,
    Nearest_support,
    Nearest_contrib_pixels,
    Nearest_min_scale
};

/* Finalise a rescaler instance. */
void ipa_rescaler_fin(ipa_rescaler *rescaler, void *opaque)
{
    if (rescaler == NULL)
        return;

    ipa_free(rescaler->ctx, opaque, rescaler->weights_x);
    ipa_free(rescaler->ctx, opaque, rescaler->weights_y);
    ipa_free(rescaler->ctx, opaque, rescaler->index_x);
    ipa_free(rescaler->ctx, opaque, rescaler->index_y);
    ipa_free(rescaler->ctx, opaque, rescaler->tmp);
    ipa_free(rescaler->ctx, opaque, rescaler);
}

#ifdef DEBUG_RESCALER
static void
dump_contribs(const index_t *index,
              const int32_t *weights,
              int            dst_h,
              int            support,
              int            stride)
{
    int i, j, s;

    for (i = 0; i < dst_h; i++)
    {
        int n = index[i].n;
        int idx = index[i].index;
        printf("%d: fp = %d(%d) n=%d lp=%d\n  ", i, index[i].first_pixel/stride, index[i].first_pixel, n, index[i].last_pixel);
        s = 0;
        for (j = 0; j < n; j++)
        {
            s += weights[j+idx];
            printf(" %x", weights[j+idx]);
        }
        printf(" (%x)\n", s);
    }
}
#endif

/* Initialise a rescaler instance. */
ipa_rescaler *
ipa_rescaler_init(ipa_context *ctx,
                  void *opaque,
                  unsigned int src_w, unsigned int src_h,
                  unsigned int data_x, unsigned int data_y,
                  unsigned int data_w, unsigned int data_h,
                  unsigned int dst_w, unsigned int dst_h,
                  unsigned int patch_x, unsigned int patch_y,
                  unsigned int patch_w, unsigned int patch_h,
                  ipa_rescale_quality quality,
                  unsigned int src_bytes_per_channel,
                  unsigned int dst_bytes_per_channel,
                  unsigned int channels,
                  unsigned int src_max_value,
                  unsigned int dst_max_value)
{
    ipa_rescaler *rescaler;
    const filter_defn_t *horiz;
    const filter_defn_t *vert;
    unsigned int n_pixels;
    unsigned int tmp_stride_samples;
    int tmp_bytes_per_channel = src_bytes_per_channel;
    int tmp_max_value = tmp_bytes_per_channel == 1 ? 255 : 65535;
    int use_sse = ctx->use_sse_4_1;

#ifdef IPA_FORCE_SSE
#if IPA_FORCE_SSE
#define SSE_SWITCH(A,B) (A)
#else
#define SSE_SWITCH(A,B) (B)
#endif
#else
#define SSE_SWITCH(A,B) (use_sse ? (A) : (B))
#endif

#ifdef DEBUG_RESCALER
    printf("rescaler_init: %dx%d -> %dx%d\n", src_w, src_h, dst_w, dst_h);
    printf("  data=%d,%d+%d,%d\n", data_x, data_y, data_w, data_h);
    printf("  patch=%d,%d+%d,%d\n", patch_x, patch_y, patch_w, patch_h);
    printf("  src_bpp=%d dst_bpp=%d\n", src_bytes_per_channel, dst_bytes_per_channel);
    printf("  channels=%d values=%x->%x\n", channels, src_max_value, dst_max_value);
#endif

    /* Sanity check patch values */
    if (patch_x > dst_w || patch_y > dst_h)
        return NULL;
    if (patch_x + patch_w > dst_w || patch_y + patch_h > dst_h)
        return NULL;
    if (data_x > src_w || data_y > src_h)
        return NULL;
    if (data_x + data_w > src_w || data_y + data_h > src_h)
        return NULL;

    rescaler = ipa_malloc(ctx, opaque, sizeof(ipa_rescaler));
    if (rescaler == NULL)
        return NULL;

    rescaler->ctx         = ctx;
    rescaler->src_h       = src_h;
    rescaler->dst_h       = dst_h;
    rescaler->dst_patch_x = patch_x;
    rescaler->dst_patch_y = patch_y;
    rescaler->dst_patch_w = patch_w;
    rescaler->dst_patch_h = patch_h;
    rescaler->channels    = channels;
    rescaler->data_offset = data_x * channels * src_bytes_per_channel;

    switch (quality)
    {
        case IPA_MITCHELL:
            horiz = &Mitchell_defn;
            vert  = &Mitchell_defn;
            break;
        case IPA_DOGLEG:
            horiz = &DogLeg_defn;
            vert  = &DogLeg_defn;
            break;
        case IPA_LINEAR:
            horiz = &Linear_defn;
            vert  = &Linear_defn;
            break;
        case IPA_NEAREST:
        default:
            horiz = &Nearest_defn;
            vert  = &Nearest_defn;
            break;
    }

    /* By default we use the mitchell filter, but if we are scaling down
     * (either on the horizontal or the vertical axis) then use the simple
     * interpolation filter for that axis. */
    if (dst_w < src_w)
        horiz = &DogLeg_defn;
    if (dst_h < src_h)
        vert = &DogLeg_defn;

    rescaler->src_y = 0;
    rescaler->dst_y = 0;

    /* create intermediate image to hold horizontal zoom */
    rescaler->max_support = vert->contrib_pixels(((double)dst_h) / src_h);
    if (rescaler->max_support < 4)
        rescaler->max_support = 4;
    rescaler->tmp_stride  = patch_w * tmp_bytes_per_channel * channels;
    rescaler->tmp         = (uint8_t *)ipa_malloc(ctx, opaque, rescaler->max_support * rescaler->tmp_stride);
    rescaler->index_x     = (index_t *)ipa_malloc(ctx, opaque, patch_w * sizeof(index_t));
    rescaler->index_y     = (index_t *)ipa_malloc(ctx, opaque, patch_h * sizeof(index_t));
    n_pixels              = horiz->contrib_pixels((double)dst_w / src_w);
    if (n_pixels < 4)
        n_pixels = 4;
    rescaler->weights_x   = (int32_t *)ipa_malloc(ctx, opaque, n_pixels * patch_w * sizeof(int32_t) + 16);
    rescaler->weights_y   = (int32_t *)ipa_malloc(ctx, opaque, rescaler->max_support * (patch_h+1) * sizeof(int32_t));
    if (rescaler->tmp       == NULL ||
        rescaler->index_x   == NULL ||
        rescaler->weights_x == NULL ||
        rescaler->index_y   == NULL || 
        rescaler->weights_y == NULL)
    {
        ipa_rescaler_fin(rescaler, opaque);
        return NULL;
    }
#ifdef PACIFY_VALGRIND
    /* When we are scaling a subrectangle of an image, we calculate
     * the subrectangle, so that it's slightly larger than it needs
     * to be. Some of these 'extra' pixels are calculated using
     * bogus values (i.e. ones we don't bother copying/scaling into
     * the line buffer). These cause valgrind to be upset. To avoid
     * this, we preset the buffer to known values. */
    memset((uint8_t *)rescaler->tmp, 0,
           rescaler->max_support * dst_w * tmp_bytes_per_channel * channels);
#endif
    /* Pre-calculate filter contributions for a row. */
    calculate_contrib(rescaler->index_x,
                      rescaler->weights_x,
                      dst_w, src_w,
                      patch_x, patch_w,
                      data_x, data_w,
                      channels,
                      src_max_value,
                      (double)tmp_max_value / src_max_value,
                      horiz);

    /* Pre-calculate filter contributions for a column. */
    tmp_stride_samples = rescaler->tmp_stride / tmp_bytes_per_channel;
    calculate_contrib(rescaler->index_y,
                      rescaler->weights_y,
                      dst_h, src_h,
                      patch_y, patch_h,
                      data_y, data_h,
                      tmp_stride_samples,
                      dst_max_value,
                      (double)dst_max_value / (dst_bytes_per_channel == 2 ?  65535 : 255),
                      vert);

    /* Just occasionally, we can get called with dst_patch_w == 0.
     * Avoid this causing division by zero errors below. */
    if (tmp_stride_samples == 0)
        tmp_stride_samples = 1;

    /* Figure out what the source patch y range is. */
    {
        unsigned int i;
        int min, max;

        min = rescaler->index_y[0].first_pixel / tmp_stride_samples;
        max = min + rescaler->index_y[0].n;
        for (i = 1; i < patch_h; i++)
        {
            int f = rescaler->index_y[i].first_pixel / tmp_stride_samples;
            int n = rescaler->index_y[i].n;
            if (min > f)
                min = f;
            if (max < f + n)
                max = f + n;
        }
        rescaler->src_patch_y = min;
        rescaler->src_patch_h = max - min;
    }

#ifdef DEBUG_RESCALER
    printf("X weights:\n");
    dump_contribs(rescaler->index_x,
                  rescaler->weights_x,
                  patch_w,
                  rescaler->max_support,
                  channels);

    printf("Y weights:\n");
    dump_contribs(rescaler->index_y,
                  rescaler->weights_y,
                  patch_h,
                  rescaler->max_support,
                  tmp_stride_samples);
#endif

    /* Renumber/Reorder column contributions to allow for wraparound buffer */
    permute_contribs(rescaler->index_y,
                     rescaler->weights_y,
                     patch_h,
                     rescaler->max_support,
                     tmp_stride_samples);

#ifdef DEBUG_RESCALER
    dump_contribs(rescaler->index_y,
                  rescaler->weights_y,
                  patch_h,
                  rescaler->max_support,
                  tmp_stride_samples);
#endif

    /* Now we choose the subroutines we use to do the X and Y scales.
     * For the X scales, we get data as (e.g) RGBRGBRGBRGB and have to apply
     * weights to each of these in turn, w[0] to the first 3 bytes,
     * w[1] for the next 3 bytes etc. We therefore need different routines
     * for the different numbers of components we are handling.
     *
     * The x routines are therefore named: zoom_xAtoB_C{_sse}
     * where A = number of bytes for an input value
     *       B = number of bytes for an output value
     *       C = number of components.
     *
     * For the Y scales, all the values in the first line have w[0]
     * applied to them. All the values in the second line have w[1]
     * applied to them etc. As such, the number of components is irrelevant
     * to the operation (other than requiring us to multiply up the width
     * as required).
     *
     * For every application of the Y scales, the number of weights, and the
     * weights themselves are the same across the whole line. As such, we
     * call functions like zoom_yAtoB{_sse} from here, and those break down
     * into zoom_yAtoB_W, where A = number of bytes for an input value,
     * B = number of bytes for an output value, and W = number of weights for
     * this line.
     */

    if (tmp_bytes_per_channel == 1)
    {
        assert(src_bytes_per_channel == 1);
        switch (channels)
        {
            case 1:
                rescaler->zoom_x = SSE_SWITCH(zoom_x1to1_1_sse, zoom_x1to1_1);
                break;
            case 3:
                rescaler->zoom_x = SSE_SWITCH(zoom_x1to1_3_sse, zoom_x1to1_3);
                break;
            case 4:
                rescaler->zoom_x = SSE_SWITCH(zoom_x1to1_4_sse, zoom_x1to1_4);
                break;
            default:
                rescaler->zoom_x = SSE_SWITCH(NULL, zoom_x1to1);
                break;
        }

        if (dst_bytes_per_channel == 1)
            rescaler->zoom_y = SSE_SWITCH(zoom_y1to1_sse, zoom_y1to1);
        else if (dst_max_value == frac_1)
            rescaler->zoom_y = SSE_SWITCH(NULL, zoom_y1to2_frac);
        else
            rescaler->zoom_y = SSE_SWITCH(zoom_y1to2_sse, zoom_y1to2);
    }
    else
    {
        assert(src_bytes_per_channel == 2);
        switch (channels)
        {
            case 1:
                rescaler->zoom_x = SSE_SWITCH(zoom_x2to2_1_sse, zoom_x2to2_1);
                break;
            case 3:
                rescaler->zoom_x = SSE_SWITCH(zoom_x2to2_3_sse, zoom_x2to2_3);
                break;
            case 4:
                rescaler->zoom_x = SSE_SWITCH(zoom_x2to2_4_sse, zoom_x2to2_4);
                break;
            default:
                rescaler->zoom_x = SSE_SWITCH(NULL, zoom_x2to2);
                break;
        }

        if (dst_bytes_per_channel == 1)
            rescaler->zoom_y = SSE_SWITCH(zoom_y2to1_sse, zoom_y2to1);
#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 0
        else if (dst_max_value == frac_1)
            rescaler->zoom_y = zoom_y2to2_frac;
#endif
        else
            rescaler->zoom_y = SSE_SWITCH(zoom_y2to2_sse, zoom_y2to2);
    }

    if (rescaler->zoom_x == NULL || rescaler->zoom_y == NULL)
    {
        ipa_rescaler_fin(rescaler, opaque);
        return NULL;
    }

    return rescaler;
}

#ifdef DEBUG_RESCALER
static unsigned int
crc(const unsigned char *data, int w)
{
    unsigned int sum = 0;
    while (w--)
    {
        sum = (sum<<1) + *data++ + (sum>>31);
    }
    return sum;
}
#endif

/* Process data: */
int ipa_rescaler_process(ipa_rescaler *rescaler, void *opaque, const uint8_t *input, uint8_t *output)
{
    /* If we've given all the data we're going to give, no point in more
     * processing, but consume any extra data that we're being handed to
     * avoid silly callers getting stuck. */
    if (rescaler->dst_y == rescaler->dst_patch_h)
        return input == NULL ? 0 : 1;

    /* Check whether we need to deliver any output. */
    if (rescaler->src_y > rescaler->index_y[rescaler->dst_y].last_pixel)
    {
        /* We have enough horizontally scaled temporary rows */
        /* to generate a vertically scaled output row. */

        /* Apply filter to zoom vertically from tmp to dst. */
        if (rescaler->dst_y < rescaler->dst_patch_h)
        {
#ifdef DEBUG_RESCALER
            printf("Scaling tmp %d (%d->%d) to dst %d\n",
                   rescaler->index_y[rescaler->dst_y].first_pixel,
                   rescaler->index_y[rescaler->dst_y].last_pixel + 1 - rescaler->index_y[rescaler->dst_y].n,
                   rescaler->index_y[rescaler->dst_y].last_pixel,
                   rescaler->dst_y);
#endif
            rescaler->zoom_y(output,
                             rescaler->tmp,
                             rescaler->dst_patch_w * rescaler->channels,
                             rescaler->tmp_stride,
                             &rescaler->index_y[rescaler->dst_y],
                             rescaler->weights_y);
        }
        rescaler->dst_y++;
        return 2; /* Outputting new row */
    }

    /* Read input data and scale horizontally into tmp. */
    if (input == NULL)
        return 0; /* No input data! Disaster! */

    if (rescaler->src_y >= rescaler->src_patch_y &&
        rescaler->src_y < rescaler->src_patch_y + rescaler->src_patch_h)
    {
#ifdef DEBUG_RESCALER
        printf("Scaling src %d (%x) to tmp %d\n", rescaler->src_y, crc(input, rescaler->index_x[rescaler->dst_patch_w-1].last_pixel+1), rescaler->src_y % rescaler->max_support);
#endif
        rescaler->zoom_x(rescaler->tmp + (rescaler->src_y % rescaler->max_support) * rescaler->tmp_stride,
                         input - rescaler->data_offset,
                         rescaler->dst_patch_w,
                         rescaler->channels,
                         rescaler->index_x,
                         rescaler->weights_x);
    }
    rescaler->src_y++;

    return 1;
}

/* Reset a rescaler instance. */
void ipa_rescaler_reset(ipa_rescaler *rescaler, void *opaque)
{
    if (rescaler == NULL)
        return;

    rescaler->src_y = 0;
    rescaler->dst_y = 0;
}
