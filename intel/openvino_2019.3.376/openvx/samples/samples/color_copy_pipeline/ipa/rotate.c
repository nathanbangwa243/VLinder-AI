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
/* Bitmap rotation. */

#define _CRT_SECURE_NO_WARNINGS

#include "ipa-impl.h"

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//#define DEBUG_ROTATOR

//int errprintf_nomem(const char *string, ...);
//#define printf errprintf_nomem

/* We up the mallocs by a small amount to allow for SSE
 * reading overruns. */
#define SSE_SLOP 16

/* Some notes on the theory.

It is "well known" that a rotation (of between -90 and 90
degrees at least) can be performed as a process of 3 shears,
one on X, one on Y, one on X.

The standard rotation matrix is:

  R = ( cos(t)   -sin(t) )
      ( sin(t)    cos(t) )

For this to be equal to the above, we need:

 ShearX(a).ShearY(b).Shear(c) = R

 (1 a) (1 0) (1 c) = (1+ab a) (1 c) = (1+ab c+abc+a) = R
 (0 1) (b 1) (0 1)   (b    1) (0 1)   (b    1+bc   )

Solve this to get:

 b = sin(t)
 cos(t) = 1 + a.sin(t) => (cos(t)-1)/sin(t) = a => a = -tan(t/2)
 1+ab = 1+bc => ab = bc => a = c.

While a == c, we'll keep them separate in the following maths
for reasons that will hopefully become clear later.

The process we use to do sub-pixel capable shears allows us to
incorporate 1 dimensional scales for free. If the expected user of
this API is a scanner driver correcting for small errors in document
alignment, let's consider that they may also want to expand/reduce
at the same time.

I'm not sure whether pre or post scales will be most useful, so
let's do the maths allowing for both, as it'll end up as no
more work.

Our transformation pipeline can then be naively expressed as:

 (u') = (X 0) (1 a) (1 0) (1 c) (x 0) (u)
 (v')   (0 Y) (0 1) (b 1) (0 1) (0 y) (v)

(Where X,Y are 'post' scale factors, and x,y are 'pre' scale factors)

We need each step in the operation to be of the form:

 (C D)
 (0 1)

for some constants C and D to work with our 1-d scale/shear
mechanism. So, rearrange the pipeline:

 P = (X 0) (1 a) (1 0) (1 c) (x 0)
     (0 Y) (0 1) (b 1) (0 1) (0 y)

   = (X Xa) (1 0) (x cy)
     (0 Y ) (b 1) (0 y )

   = (X Xa/Y) (1 0) (1 0) (1 0) (x cy)
     (0 1   ) (0 Y) (b 1) (0 y) (0 1 )

   = (X Xa/Y) (1  0) (1 0) (x cy)
     (0 1   ) (bY Y) (0 y) (0 1 )

   = (X Xa/Y) (1   0) (x cy)
     (0 1   ) (bY Yy) (0 1 )

The first scale/shear involves us taking an input line of data,
and scaling it into temporary storage. Every line of data supplied
has the same scale done to it, but at a different X offset. We
will therefore have to generate sets of weights for several different
subpixel positions, and pick the appropriate one. 4 or 8 should
be enough?

For the second scale/shear, we don't run down individual scan-columns.
Instead, we use a block of lines (the output from the first phase) and
traverse through it diagonally, copying data out - effectively
producing 1 pixel output for each of the 'scan columns'.

This gives us a block of scanlines that we can apply the third
shear to.

Let us now consider the output position of each of the corners of
our src_w * src_h input rectangle under this transformation. For
simplity of notation, let's call these just 'w' and 'h'.

We know that by the properties of 2x2 matrices, (0, 0) maps to
(0, 0) and (w, h) maps the the sum of where (w,0) (0,h) map to,
so we only need calculate the image of 2 corners.

   (X Xa/Y) (1 0  ) (x cy) (w 0)
   (0 1   ) (bY Yy) (0 1 ) (0 h)

 = (X Xa/Y) (1  0 ) (wx  chy)
   (0 1   ) (bY Yy) (0   h  )

 = (X Xa/Y) (wx    chy        )
   (0 1   ) (bwxY  cbhYy + hYy)

 = (wxX + abwxX  chXy + acbhXy + ahXy)
   (bwxY         cbhYy + hYy         )

 = (C E)
   (D F)

where C = wxX + abwxX = wxX (1+ab)
      D = bwxY
      E = chXy + acbhXy + ahXy = hXy(c + acb + a)
      F = cbhYy + hYy = hYy (1+cb)

Some ASCII art to illustrate the images after the different
stages:

For angles 0..90:

  *--x => *----x   =>    __.x
  |  |     \    \      *'   \
  o--+      o----+      \  __.+
                         o'

For angles 0 to -90:

  *--x =>   *----x =>   *.__
  |  |     /    /      /    'x
  o--+    o----+      o.__  /
                          '+

How wide does temporary buffer 1 (where the results of the first
scale/shear are stored) need to be?

From the above diagrams we can see that for angles 0..90 it needs
to be wide enough to stretch horizontally from * to +. i.e. wx + chy

Similarly for angles 0..-90, it needs to be wide enough to stretch
horizontally from o to x. i.e. wx - chy

Given the properties of sin, we can see that it's wx + |c|hy in
all cases.


How tall is the image after the first scale/shear?

Well, the height doesn't change, so src_h.


How tall does temporary buffer 1 (where the results of the first
scale/shear are stored) need to be?

Consider the second stage. One of the scanlines produced will be
made by reading in a line of pixels from (0, 0) to (wx+|c|hy, i)
and outputting a horizontal line of pixels in the result. What is
the value i?

We know:

 (1  0 ) (wx+|c|hy)) = (wx+|c|hy))
 (bY Yy) (i        )   (0        )

 So: bY*(wx+|c|hy)) + iYy = 0

     i = -b*(wx+|c|hy)/y     (for non-zero Y and y)
       = -b*(tmp1_width)/y

(Sanity check: when t=0, b = 0, no lines needed. As the x
scale increases, we need more lines. Seems sane.)


How wide is the image after the second scale/shear?

Well, the shear on Y doesn't change the width, so it's still
wx + |c|hy.


How tall is the entire image after second scale/shear?

For angles 0 to 90, it's the difference in Y between o and x.
i.e. cbhYy + hYy - bwxY

For angles 0 to -90, it's the difference in Y between + and *.
i.e. cbhYy + hYy + bwxY

So cbhYy + hYy + |b|wxY in all cases.


What is the size of the final image?

For angles 0..90: Width is from * to +. Height is from o to x.

 W = C+E = wxX (1+ab) + hXy(c + acb + a)
 H = F-D = hYy (1+cb) - bwxY

For angles 0..-90: Width is from o to x: Height is from * to +.

 W = C-E = wxX (1+ab) - hXy(c + acb + a)
 H = F+D = hYy (1+cb) + bwxY

 Given the properties of sin and tan, we can say that:

 W = wxX (1+ab) + hXy|c + acb + a| = X * (wx(1+ab) + hy|c+acb+a|)
 H = hYy (1+cb) + |b|wxY = Y * (hy(1+cb) + |b|wx)


When we scale/shear from tmp2 to the output, what width does a
line of tmp2 scale to?

 (X Xa/Y) (tmp1_width 0) = (X*tmp1_width Xah/Y)
 (0 1   ) (0          h)   (0            h    )

So a line will end up being X*(tmp1_width + |a|h/Y) wide.

*/

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
    uint8_t  slow;        /* Flag */
    int32_t  first_pixel; /* offset of first value in source data */
    int32_t  last_pixel;  /* last pixel number */
} index_t;

typedef void (zoom_y_fn)(uint8_t       *              dst,
                         const uint8_t * ipa_restrict tmp,
                         const index_t * ipa_restrict index,
                         const int32_t * ipa_restrict weights,
                         uint32_t                     width,
                         uint32_t                     channels,
                         uint32_t                     mod,
                         int32_t                      y);
typedef void (zoom_x_fn)(uint8_t       * ipa_restrict tmp,
                         const uint8_t * ipa_restrict src,
                         const index_t * ipa_restrict index,
                         const int32_t * ipa_restrict weights,
                         uint32_t                     dst_w,
                         uint32_t                     src_w,
                         uint32_t                     channels,
                         const uint8_t * ipa_restrict bg);

#define CLAMP(v, mn, mx)\
  (v < mn ? mn : v > mx ? mx : v)

/* Include the cores */
#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 0
#include "rotate_c.h"
#endif

#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 1
#include "rotate_sse.h"
#endif

enum {
    SUB_PIX_X = 4,
    SUB_PIX_Y = 4,
};

struct ipa_rotator_s
{
    /* The init procedure sets the following. */
    ipa_context  *ctx;
    uint32_t      channels;
    uint32_t      src_w;
    uint32_t      src_h;
    uint32_t      dst_w;
    uint32_t      dst_h;
    index_t      *index_1[SUB_PIX_X];
    int32_t      *weights_1[SUB_PIX_X];
    index_t      *index_2[SUB_PIX_Y];
    int32_t      *weights_2[SUB_PIX_Y];
    index_t      *index_3[SUB_PIX_X];
    int32_t      *weights_3[SUB_PIX_X];
    uint8_t      *bg;

    double        pre_x_scale;
    double        pre_y_scale;
    double        post_x_scale;
    double        post_y_scale;
    double        alpha;
    double        beta;
    double        gamma;
    double        chy;
    double        extent1;
    double        w1;
    double        h2;
    double        diagonal_h;
    double        t_degrees;
    double        x_shift2;
    double        indent;
    uint32_t      tmp1_stride;
    uint32_t      tmp1_w;
    uint32_t      tmp1_h;
    uint32_t      tmp2_w;

    uint32_t      pre_fill_y;
    double        start_y;

    zoom_x_fn    *zoom_x1;
    zoom_y_fn    *zoom_y;
    zoom_x_fn    *zoom_x2;
};

static void
fill_bg(uint8_t *dst, const uint8_t *fill, uint32_t chan, uint32_t len)
{
    while (len--)
    {
        memcpy(dst, fill, chan);
        dst += chan;
    }
}

#define B (1.0f / 3.0f)
#define C (1.0f / 3.0f)
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

#define FILTER_WIDTH 2

static int
make_x_weights(ipa_context  *context,
               void         *opaque,
               index_t     **indexp,
               int32_t     **weightsp,
               uint32_t      src_w,
               uint32_t      dst_w,
               double        factor,
               uint32_t      offset_f,
               uint32_t      offset_n,
               int           sse_slow)
{
    double    squeeze;
    index_t  *index;
    int32_t  *weights;
    uint32_t  i;
    double    offset = ((double)offset_f)/offset_n;
    uint32_t  idx;
    uint32_t  max_weights;

    if (factor <= 1)
    {
        squeeze = 1;
        max_weights = 1 + FILTER_WIDTH * 2;
    }
    else
    {
        squeeze = factor;
        max_weights = (uint32_t)ceil(1 + squeeze * FILTER_WIDTH * 2);
        if (max_weights > 10)
        {
            max_weights = 10;
            squeeze = ((double)max_weights) / (FILTER_WIDTH * 2);
        }
    }

    if (max_weights <= 4)
        max_weights = 4;
    else if (max_weights <= 8)
        max_weights = 8;

    weights = ipa_malloc(context, opaque, max_weights * dst_w * sizeof(*weights) + SSE_SLOP);
    if (weights == NULL)
        return 1;
    memset(weights, 0, max_weights * dst_w * sizeof(*weights));
    index = ipa_malloc(context, opaque, sizeof(*index) * dst_w + SSE_SLOP);
    if (index == NULL)
    {
        ipa_free(context, opaque, weights);
        return 1;
    }
    *indexp   = index;
    *weightsp = weights;

    for (i = 0; i < dst_w; i++)
    {
        /* i is in 0..w (i.e. dst space).
         * centre, left, right are in 0..src_w (i.e. src_space)
         */
        double centre = (i+0.5f)*factor - offset;
        int32_t left = (int32_t)ceil(centre - squeeze*FILTER_WIDTH);
        int32_t right = (int32_t)floor(centre + squeeze*FILTER_WIDTH);
        int32_t j, k;
        idx = max_weights * i;

        if ((centre - left) >= squeeze * FILTER_WIDTH)
            left++;
        if ((right - centre) >= squeeze * FILTER_WIDTH)
            right--;

        /* When we're calculating the second set of X weights, the subpixel adjustment can cause us to
         * read too far to the right. Adjust for this hackily here. */
        if (left > (int32_t)src_w) {
            right -= left - src_w;
            centre -= left - src_w;
            left = src_w;
        }

        assert(right-left+1 <= (int)max_weights && right >= 0 && left <= (int32_t)src_w);
        index->index = idx;
        j = left;
        if (j < 0)
        {
            left = -1;
            weights[idx] = 0;
            for (; j < 0; j++)
            {
                double f = (centre - j) / squeeze;
                weights[idx] += (int32_t)(Mitchell_filter(f) * WEIGHT_SCALE / squeeze);
            }
            idx++;
        }
        k = right;
        if (k > (int32_t)src_w)
            k = (int32_t)src_w;
        for (; j <= k; j++)
        {
            double f = (centre - j) / squeeze;
            weights[idx++] = (int32_t)(Mitchell_filter(f) * WEIGHT_SCALE / squeeze);
        }
        for (; j < right; j++)
        {
            double f = (centre - j) / squeeze;
            weights[idx-1] += (int32_t)(Mitchell_filter(f) * WEIGHT_SCALE / squeeze);
        }
        index->first_pixel = left;
        index->last_pixel  = k;
        index->n           = k-left+1;
        index->slow        = left < 0 || k >= (int32_t)src_w;
        if (left + sse_slow > (int)src_w)
            index->slow = 1;
        index++;
    }

    return 0;
}

/* The calculations here are different.
 * We move from offset...offset+h1 in w steps.
 * At each point, we calculate the weights vertically
 * with a scale factor of dst_h/src_h.
 */
static int
make_y_weights(ipa_context  *context,
               void         *opaque,
               index_t     **indexp,
               int32_t     **weightsp,
               uint32_t      dst_w,
               double        factor,
               double        factor2,
               uint32_t      offset_f,
               uint32_t      offset_n,
               uint32_t      h)
{
    double     squeeze;
    index_t  *index;
    int32_t  *weights;
    uint32_t  i;
    double    offset = ((double)offset_f)/offset_n;
    uint32_t  idx;
    uint32_t  max_weights;

    if (factor >= 1)
    {
        squeeze = 1;
        max_weights = 1 + FILTER_WIDTH * 2;
    }
    else
    {
        squeeze = 1/factor;
        max_weights = (uint32_t)ceil(squeeze * FILTER_WIDTH * 2);
        if (max_weights > 10)
        {
            max_weights = 10;
            squeeze = ((double)max_weights) / (FILTER_WIDTH * 2);
        }
    }

    /* Ensure that we never try to access before 0 */
    offset += (double)FILTER_WIDTH/squeeze;

    weights = ipa_malloc(context, opaque, max_weights * dst_w * sizeof(*weights));
    if (weights == NULL)
        return 1;
    index = ipa_malloc(context, opaque, sizeof(*index) * dst_w);
    if (index == NULL)
    {
        ipa_free(context, opaque, weights);
        return 1;
    }
    *indexp   = index;
    *weightsp = weights;

    if (factor2 < 0)
        offset -= (dst_w-1) * factor2;

    idx = 0;
    for (i = 0; i < dst_w; i++)
    {
        /* i is in 0..dst_w (i.e. dst space).
         * centre, left, right are in 0..src_h (i.e. src_space)
         */
        double centre = (i+0.5f)*factor2 + offset;
        int32_t left = (int32_t)ceil(centre - squeeze*FILTER_WIDTH);
        int32_t right = (int32_t)floor(centre + squeeze*FILTER_WIDTH);
        int32_t j;

        if ((centre - left) >= squeeze * FILTER_WIDTH)
            left++;
        if ((right - centre) >= squeeze * FILTER_WIDTH)
            right--;

        assert(right-left+1 <= (int)max_weights);
        index->index       = idx;
        for (j = left; j <= right; j++)
        {
            double f = (centre - j) / squeeze;
            weights[idx++] = (int32_t)(Mitchell_filter(f) * WEIGHT_SCALE / squeeze);
        }
        index->last_pixel  = right;
        index->n           = right-left+1;
        if (left < 0)
            left += h;
        index->first_pixel = left;
        index->slow        = 0;
        index++;
    }

    return 0;
}

#ifdef DEBUG_ROTATOR
static void
dump_weights(const index_t *index,
             const uint32_t *weights,
             uint32_t w,
             const char *str)
{
    uint32_t i;

    printf("%s weights:\n", str);
    for (i = 0; i < w; i++)
    {
        uint32_t j;
        int32_t  sum = 0;
        uint32_t n = index[i].n;
        uint32_t idx = index[i].index;
        printf(" %d: %d->%d:", i, index[i].first_pixel, index[i].last_pixel);
        for (j = 0; j < n; j++)
        {
            sum += weights[idx];
            printf(" %x", weights[idx++]);
        }
        printf(" (%x)\n", sum);
    }
}
#endif

void
ipa_rotator_pre_init(unsigned int  src_w,
                     unsigned int  src_h,
                     unsigned int *dst_w,
                     unsigned int *dst_h,
                     double        t_degrees,
                     double        pre_x_scale,
                     double        pre_y_scale,
                     double        post_x_scale,
                     double        post_y_scale)
{
    double alpha = -tan(t_degrees * M_PI / 180 / 2);
    double beta  = sin(t_degrees * M_PI / 180);
    double gamma = alpha;

    /* After the second shear/scale, the image will be w1 x h2 in size. */

    /* Calculate the size of the destination buffer */
    double one_plus_ab = 1 + alpha * beta;
    double one_plus_cb = 1 + gamma * beta;
    /* W = X * (wx(1+ab) + hy|c+acb+a|) */
    double w2 = post_x_scale * (src_w * pre_x_scale * one_plus_ab + src_h * pre_y_scale * fabs(gamma * one_plus_ab + alpha));
    double h2_div_Y = src_h * pre_y_scale * one_plus_cb + fabs(src_w * pre_x_scale * beta);
    double h2 = post_y_scale * h2_div_Y;
    *dst_w = (uint32_t)ceil(w2);
    *dst_h = (uint32_t)ceil(h2);
}

ipa_rotator *
ipa_rotator_init(ipa_context   *context,
                 void          *opaque,
                 unsigned int   src_w,
                 unsigned int   src_h,
                 unsigned int  *dst_w,
                 unsigned int  *dst_h,
                 double         t_degrees,
                 double         pre_x_scale,
                 double         pre_y_scale,
                 double         post_x_scale,
                 double         post_y_scale,
                 unsigned char *bg,
                 unsigned int   channels)
{
    ipa_rotator *rotator;
    double w1, w2, h2, extent1, alpha, beta, gamma, chy, one_plus_ab, one_plus_cb;
    double diagonal_h, h2_div_Y, x_shift2, indent;
    int i;
    int use_sse = context->use_sse_4_1;

#ifdef IPA_FORCE_SSE
#if IPA_FORCE_SSE
#define SSE_SWITCH(A,B) (A)
#else
#define SSE_SWITCH(A,B) (B)
#endif
#else
#define SSE_SWITCH(A,B) (use_sse ? (A) : (B))
#endif

    rotator = ipa_malloc(context, opaque, sizeof(*rotator));
    if (!rotator)
        return NULL;

    memset(rotator, 0, sizeof(*rotator));
    rotator->ctx         = context;
    rotator->channels    = channels;

    /* Rotation coefficients */
    alpha = -tan(t_degrees * M_PI / 180 / 2);
    beta  = sin(t_degrees * M_PI / 180);
    gamma = alpha;

    /* After the first shear/scale, the image will be w1 x src_h in size. */
    chy = gamma * src_h * pre_y_scale;
    extent1 = pre_x_scale * src_w;
    w1 = extent1 + fabs(chy);
    diagonal_h = fabs(beta) * w1 / pre_y_scale;

    /* After the second shear/scale, the image will be w1 x h2 in size. */

    /* Calculate the size of the destination buffer */
    one_plus_ab = 1 + alpha * beta;
    one_plus_cb = 1 + gamma * beta;
    /* W = X * (wx(1+ab) + hy|c+acb+a|) */
    w2 = post_x_scale * (src_w * pre_x_scale * one_plus_ab + src_h * pre_y_scale * fabs(gamma * one_plus_ab + alpha));
    h2_div_Y = src_h * pre_y_scale * one_plus_cb + fabs(src_w * pre_x_scale * beta);
    h2 = post_y_scale * h2_div_Y;
    x_shift2 = post_x_scale * alpha * h2_div_Y;

    indent = pre_x_scale * beta * post_y_scale * src_w * x_shift2 / h2;

    rotator->src_w        = src_w;
    rotator->src_h        = src_h;
    rotator->dst_w        = *dst_w = (uint32_t)ceil(w2);
    rotator->dst_h        = *dst_h = (uint32_t)ceil(h2);
    rotator->pre_x_scale  = pre_x_scale;
    rotator->pre_y_scale  = pre_y_scale;
    rotator->post_x_scale = post_x_scale;
    rotator->post_y_scale = post_y_scale;
    rotator->alpha        = alpha;
    rotator->beta         = beta;
    rotator->gamma        = gamma;
    rotator->chy          = chy;
    rotator->extent1      = extent1;
    rotator->w1           = w1;
    rotator->h2           = h2;
    rotator->diagonal_h   = diagonal_h;
    rotator->t_degrees    = t_degrees;
    rotator->x_shift2     = x_shift2;
    rotator->indent       = indent;
    rotator->tmp1_stride  = (uint32_t)ceil(w1);
    rotator->tmp1_w       = (uint32_t)ceil(extent1)+1; /* +1 allows for subpixel positioning */
    rotator->tmp1_h       = (uint32_t)ceil(diagonal_h) + FILTER_WIDTH*2;
    if (rotator->tmp1_h == 0) /* Allow for the zero degree case */
        rotator->tmp1_h = 1;
    rotator->tmp1_h += FILTER_WIDTH * 2;
    rotator->tmp2_w      = (uint32_t)ceil(post_y_scale * w1);
    rotator->bg = ipa_malloc(context, opaque, channels);
    if (rotator->bg == NULL)
        goto fail;
    memcpy(rotator->bg, bg, channels);

    /*
     * Once we have scaled/sheared lines into tmp1, we have
     * data such as:
     *
     *    -ve angles     +ve angles
     *      +------+  or +-----+
     *     /      /       \     \
     *    /      /         \     \
     *   /      /           \     \
     *  /      /             \     \
     * +-----+                +-----+
     *
     * We then need to copy data out into tmp2. This is done by
     * reading a series of parallel diagonal lines out. The first
     * one is as shown here:
     *
     *            _.   or    ._            ----
     *        _.-'             '-._              } pre fill region
     *     _.+-----+         +-----+._     ----
     *  .-' /     /           \     \ '-.  ____  } lines of data required before we can start
     *     /     /             \     \
     *    /     /               \     \
     *   /     /                 \     \
     *  +-----+                   +-----+
     *
     * We can see that we need to fill the temporary buffer with some
     * empty lines to start with.
     *
     * The total Y extent of the diagonal line has been calculated as
     * diagonal_h already. The length of the horizontal edge of the data
     * region is extent1, and the remainder of the width is |c|hy.
     *
     * Thus we need diagonal_h * extent1 / (extent1 + |c|hy) in the pre
     * fill region. We can generate our first line out once we have
     * h1 lines in.
     */
    rotator->pre_fill_y = ((uint32_t)ceil(rotator->diagonal_h*rotator->extent1/rotator->w1) + FILTER_WIDTH*2);

    for (i = 0; i < SUB_PIX_X; i++)
    {
        if (make_x_weights(context, opaque,
                           &rotator->index_1[i],
                           &rotator->weights_1[i],
                           src_w,
                           rotator->tmp1_w,
                           (double)src_w / extent1,
                           i, SUB_PIX_X,
                           use_sse ? (16+channels-1)/channels : 0))
            goto fail;
#ifdef DEBUG_ROTATOR
        {
            char text[16];
            sprintf(text, "X1[%d]", i);
            dump_weights(rotator->index_1[i],
                         rotator->weights_1[i],
                         rotator->tmp1_w,
                         text);
        }
#endif
    }

    for (i = 0; i < SUB_PIX_Y; i++)
    {
        if (make_y_weights(context,
                           opaque,
                           &rotator->index_2[i],
                           &rotator->weights_2[i],
                           rotator->tmp1_stride,
                           post_y_scale * pre_y_scale,
                           (t_degrees >= 0 ? diagonal_h : -diagonal_h) / w1,
                           i,
                           SUB_PIX_Y,
                           rotator->tmp1_h))
            goto fail;
#ifdef DEBUG_ROTATOR
        {
            char text[16];
            sprintf(text, "Y[%d]", i);
            dump_weights(rotator->index_2[i],
                         rotator->weights_2[i],
                         rotator->tmp1_stride,
                         text);
        }
#endif
    }

    for (i = 0; i < SUB_PIX_X; i++)
    {
        if (make_x_weights(context, opaque,
                           &rotator->index_3[i],
                           &rotator->weights_3[i],
                           rotator->tmp1_stride,
                           rotator->dst_w,
                           post_x_scale,
                           i, SUB_PIX_X,
                           0))
            goto fail;
#ifdef DEBUG_ROTATOR
        {
            char text[16];
            sprintf(text, "X2[%d]", i);
            dump_weights(rotator->index_3[i],
                         rotator->weights_3[i],
                         rotator->dst_w,
                         text);
        }
#endif
    }

    switch (channels)
    {
        case 1:
            rotator->zoom_x1 = SSE_SWITCH(zoom_x1_sse, zoom_x1);
            rotator->zoom_y  = SSE_SWITCH(zoom_y1_sse, zoom_y1);
            rotator->zoom_x2 = SSE_SWITCH(zoom_x1_sse, zoom_x1);
            break;
        case 3:
            rotator->zoom_x1 = SSE_SWITCH(zoom_x3_sse, zoom_x3);
            rotator->zoom_y  = SSE_SWITCH(zoom_y3_sse, zoom_y3);
            rotator->zoom_x2 = SSE_SWITCH(zoom_x3_sse, zoom_x3);
            break;
        case 4:
            rotator->zoom_x1 = SSE_SWITCH(zoom_x4_sse, zoom_x4);
            rotator->zoom_y  = SSE_SWITCH(zoom_y4_sse, zoom_y4);
            rotator->zoom_x2 = SSE_SWITCH(zoom_x4_sse, zoom_x4);
            break;
        default:
#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 0
            rotator->zoom_x1 = zoom_x;
            rotator->zoom_y  = zoom_y;
            rotator->zoom_x2 = zoom_x;
#endif
            break;
    }

    return rotator;
fail:
    ipa_rotator_fin(rotator, opaque);
    return NULL;
}

void
ipa_rotator_map_band(ipa_rotator  *rotator,
                     unsigned int  dst_y0,
                     unsigned int  dst_y1,
                     unsigned int *src_y0,
                     unsigned int *src_y1,
                     unsigned int *src_w)
{
    double diag_y = FILTER_WIDTH - (double)rotator->pre_fill_y;
    double diag_dy = (rotator->src_h + rotator->diagonal_h * (2 * rotator->extent1 / rotator->w1 - 1)) / rotator->h2;
    double diag_y0, diag_y1;
    int iy, which_y, need_y;
    unsigned int dst_left = 0;
    unsigned int dst_right = rotator->tmp1_stride - 1;
    *src_w = rotator->src_w;

    /* Do a half step on y, and round for sub pix. */
    diag_y += diag_dy / 2 + 1.0 / (2 * SUB_PIX_Y);
    diag_y0 = diag_y + dst_y0 * diag_dy;
    diag_y1 = diag_y + dst_y1 * diag_dy;

    iy = (int)floor(diag_y0);
    which_y = (int)floor((diag_y0 - iy) * SUB_PIX_Y);

    need_y = rotator->index_2[which_y][rotator->t_degrees < 0 ? dst_right-1 : dst_left].first_pixel + iy;
    if (need_y < 0)
        need_y = 0;
    *src_y0 = need_y;

    iy = (int)floor(diag_y1);
    which_y = (int)floor((diag_y1 - iy) * SUB_PIX_Y);

    need_y = rotator->index_2[which_y][rotator->t_degrees >= 0 ? dst_right-1 : dst_left].last_pixel + iy;
    if (need_y > (int)rotator->src_h)
        need_y = (int)rotator->src_h;
    *src_y1 = need_y;
}

int
ipa_rotator_band(ipa_rotator         *rotator,
                 void                *opaque,
                 const unsigned char *src,
                 int                  src_stride,
                 unsigned int         src_y0,
                 unsigned int         src_y1,
                 unsigned char       *dst,
                 int                  dst_stride,
                 unsigned int         dst_y0,
                 unsigned int         dst_y1)
{
    double diag_y, diag_dy;
    uint32_t in_y, out_y;
    uint32_t tmp_size = rotator->tmp1_stride * rotator->channels * rotator->tmp1_h;
    uint8_t *tmp1, *tmp2;
    double x1, dx1, x2, dx2;

    tmp1 = (uint8_t *)ipa_malloc(rotator->ctx, opaque, tmp_size + SSE_SLOP);
    if (tmp1 == NULL)
        return 1;
    tmp2 = (uint8_t *)ipa_malloc(rotator->ctx, opaque, rotator->tmp1_stride * rotator->channels);
    if (tmp2 == NULL)
    {
        ipa_free(rotator->ctx, opaque, tmp1);
        return 1;
    }

    /* Each line we scale into tmp1 starts at a different x position.
     * We hold that as x1. Over src_h lines, we want to move from
     * 0 to chy (or chy to 0). */
    x1 = (rotator->chy >= 0 ? rotator->chy : 0);
    dx1 = -rotator->chy/rotator->src_h;
    /* Do a half step */
    x1 += dx1/2;
    x1 += src_y0 * dx1;

    x2 = rotator->indent;
    if (rotator->x_shift2 <= 0)
    {
        dx2 = -rotator->x_shift2 / rotator->h2;
    }
    else
    {
        x2 += rotator->x_shift2;
        dx2 = -rotator->x_shift2 / rotator->h2;
    }
    /* Do a half step */
    x2 += dx2/2;
    x2 += dst_y0 * dx2;

    diag_y = FILTER_WIDTH-(double)rotator->pre_fill_y;
    diag_dy = (rotator->src_h + rotator->diagonal_h * (2 * rotator->extent1 / rotator->w1 - 1)) / rotator->h2;
    /* Do a half step on y */
    diag_y += diag_dy/2;
    diag_y += dst_y0 * diag_dy;

    /* diag_y = Where we start to read the diagonal line from */
    /* in_y = All lines < this have been written into tmp. */
    /* out_y = All lines smaller than this have been written out */

    /* The first source line we have is src_y0. After the X skew this will still be src_y0.
     * The Y skew will mean that this line extends across a range of output scanlines.
     * Find the range of scanlines that we touch. */
    in_y = src_y0;
    /* We need to fill the lines up to and including tmp_y with the background color. */
    {
        int stride = rotator->tmp1_stride;
        int first_y = (in_y - rotator->pre_fill_y + rotator->tmp1_h) % rotator->tmp1_h;
        int count = rotator->pre_fill_y;
        if (first_y + count > (int)rotator->tmp1_h)
        {
            int s = rotator->tmp1_h - first_y;
            fill_bg(tmp1 + first_y * stride * rotator->channels, rotator->bg, rotator->channels, stride * s);
            first_y = 0;
            count -= s;
        }
        fill_bg(tmp1 + first_y * stride * rotator->channels, rotator->bg, rotator->channels, stride * count);
    }

    out_y = dst_y0;
    while (out_y < dst_y1)
    {
        /* If we have enough data to make a new y2 line, do so. */
        double y = diag_y + 1.0/(2 * SUB_PIX_Y);
        int iy = (int)floor(y);
        int which_y = (int)floor((y - iy) * SUB_PIX_Y);

        /* Which source line do we need to have to generate the next destination line? */
        int need_y = rotator->index_2[which_y][rotator->t_degrees >= 0 ? rotator->tmp1_stride - 1 : 0].last_pixel + iy;

        if ((int)in_y >= need_y)
        {
            double x;
            int xi, xs, xe;
            int which;
            uint8_t *output = dst + dst_stride * (out_y - dst_y0);
            rotator->zoom_y(tmp2,
                            tmp1,
                            &rotator->index_2[which_y][0],
                            rotator->weights_2[which_y],
                            rotator->tmp1_stride,
                            rotator->channels,
                            tmp_size,
                            (iy + rotator->tmp1_h) % rotator->tmp1_h);
            out_y++;
            diag_y += diag_dy;

            /* Now scale tmp2 into output */
            x = x2 + (1.0 / (2 * SUB_PIX_X));
            xi = (int)floor(x);
            which = (int)floor((x - xi) * SUB_PIX_X);
            xs = xi;
            if (xs > 0)
                fill_bg(output, rotator->bg, rotator->channels, xs);
            else
                xs = 0;

            xe = xi + rotator->tmp2_w;
            if ((int)rotator->dst_w > xe)
                fill_bg(&output[xe * rotator->channels],
                        rotator->bg, rotator->channels,
                        rotator->dst_w - xe);
            else
                xe = (int)rotator->dst_w;

            rotator->zoom_x2(&output[xs * rotator->channels],
                             tmp2,
                             &rotator->index_3[which][xs-xi],
                             rotator->weights_3[which],
                             xe - xs,
                             rotator->tmp2_w,
                             rotator->channels,
                             rotator->bg);
            x2 += dx2;
        }
        else if (in_y < rotator->src_h)
        {
            uint8_t *line = &tmp1[rotator->tmp1_stride * rotator->channels * (in_y % rotator->tmp1_h)];
            double x = x1 + 1.0 / (SUB_PIX_X * 2);
            int xi = (int)floor(x);
            int which = (int)floor((x - xi) * SUB_PIX_X);

            {
                int r = (int)rotator->tmp1_stride;
                if (r > xi)
                    r = xi;
                if (0 < r)
                    fill_bg(line,
                            rotator->bg,
                            rotator->channels,
                            r);
            }
            {
                int l = xi + rotator->tmp1_w - 1; /* Undo the +1 earlier */
                if (l < 0)
                    l = 0;
                if (l < (int)rotator->tmp1_stride)
                    fill_bg(&line[l * rotator->channels],
                            rotator->bg,
                            rotator->channels,
                            (int)rotator->tmp1_stride - l);
            }
            {
                int l = xi;
                int r = xi + rotator->tmp1_w;
                if (l < 0)
                    l = 0;
                if (r > (int)rotator->tmp1_stride)
                    r = rotator->tmp1_stride;
                if (l < r)
                {
                    rotator->zoom_x1(&line[l * rotator->channels],
                                     src + src_stride * (in_y - src_y0),
                                     &rotator->index_1[which][l - xi],
                                     rotator->weights_1[which],
                                     r-l,
                                     rotator->src_w,
                                     rotator->channels,
                                     rotator->bg);
                 }
            }
            in_y++;
            x1 += dx1;
        }
        else
        {
            uint8_t *line = &tmp1[rotator->tmp1_stride * rotator->channels * (in_y % rotator->tmp1_h)];
            fill_bg(line, rotator->bg, rotator->channels, rotator->tmp1_stride);
            in_y++;
        }
    }

    ipa_free(rotator->ctx, opaque, tmp2);
    ipa_free(rotator->ctx, opaque, tmp1);
    return 0;
}

void ipa_rotator_fin(ipa_rotator *rotator, void *opaque)
{
    int i;

    if (!rotator)
        return;

    for (i = 0; i < SUB_PIX_X; i++)
    {
        ipa_free(rotator->ctx, opaque, rotator->index_1[i]);
        ipa_free(rotator->ctx, opaque, rotator->weights_1[i]);
    }
    for (i = 0; i < SUB_PIX_Y; i++)
    {
        ipa_free(rotator->ctx, opaque, rotator->index_2[i]);
        ipa_free(rotator->ctx, opaque, rotator->weights_2[i]);
    }
    for (i = 0; i < SUB_PIX_X; i++)
    {
        ipa_free(rotator->ctx, opaque, rotator->index_3[i]);
        ipa_free(rotator->ctx, opaque, rotator->weights_3[i]);
    }

    ipa_free(rotator->ctx, opaque, rotator);
}
