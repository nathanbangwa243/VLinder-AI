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
/* Bitmap halftoning. */

#define _CRT_SECURE_NO_WARNINGS

#include "ipa-impl.h"

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

/* HT_ALIGN must be a power of 2, and at least 32. */
#define HT_ALIGN 32
/* HT_LANDSCAPE must be a power of 2, and at least HT_ALIGN. */
#define HT_LANDSCAPE 32

enum {
    MAX_PLANES = 4
};

typedef struct
{
    float x0;
    float y0;
    float x1;
    float y1;
} ipa_rect;

typedef struct
{
    int x0;
    int y0;
    int x1;
    int y1;
} ipa_irect;

typedef struct
{
    int      w;
    int      h;
    int      x_phase;
    int      y_phase;
    uint8_t *data;
} ipa_screen;

/* Structure to represent a bresenham.
 *
 * At any given instance the value is i + (d-f)/d.
 */
typedef struct
{
    int i;
    int f; /* 0 < f <= d */
    int d;
    int di;
    int df;
} bresenham_t;

static inline int divmod_fixed(int n, int d, int *r)
{
    int i;
    assert(d >= 0);
    if (d == 0) /* Avoid division by zero. */
        d = 1;
    if (n >= 0)
        i = n/d;
    else
        i = -((d-1-n)/d);
    *r = n - i*d;
    return i;
}

static void
bresenham_init_fixed(bresenham_t *b, int i, int n, int e, int s)
{
    int step, step_i, step_f;
    int d = e - s;
    if (d == 0) /* Avoid division by zero. */
        d = 1;
    b->i = i;
    b->di = divmod_fixed(n<<8, d, &b->df);
    b->d = d;
    b->f = b->d;
    /* Step to midpoint */
    step = (((s + 127) & ~255) + 128) - s;
    assert(0 <= step && step < 256);
    step = (n * step);
    step_i = divmod_fixed(step, d, &step_f);
    b->i += step_i;
    b->f -= step_f;
    if (b->f <= 0)
        b->i++, b->f = b->d;
    assert(d > 0 && b->df >= 0 && b->df < b->d && b->f > 0 && b->f <= b->d);
}

static inline void
bresenham_step(bresenham_t *b)
{
    b->i += b->di;
    b->f -= b->df;
    if (b->f <= 0)
        b->f += b->d, b->i++;
}

static inline void
bresenham_advance(bresenham_t *b, int w)
{
    while (w--)
    {
        bresenham_step(b);
    }
}

typedef void (copy_contone_fn)(ipa_halftone *, const uint8_t **buffer);
typedef void (halftone_fn)(uint8_t *halftone, const uint8_t *contone, const uint8_t *screen, int w);

struct ipa_halftone_s
{
    ipa_context       *ctx;
    int                landscape;
    int                num_planes;
    int                w;
    int                h;
    ipa_matrix         mat;
    ipa_irect          drect; /* Destination rectangle */
    ipa_irect          fdrect; /* Destination rectangle (24.8 fixed point) */
    ipa_irect          cdrect; /* Clipped Destination rectangle */
    int                contone_stride;
    int                halftone_stride;
    uint8_t           *contone;
    uint8_t           *screen;
    uint8_t           *halftone;
    int                left;  /* The leftmost pixel represented in contone/screen/halftone. */
    int                right; /* The first pixel beyond contone/screen/halftone. */
    int                screen_fill;
    ipa_screen        *screens[MAX_PLANES];
    bresenham_t        brx;
    bresenham_t        bry;
    uint8_t           *cache;
    copy_contone_fn   *copy_contone;
    halftone_fn       *core;
    int                out_idx;
    int                in_idx;
};

/* The following transform operations are only guaranteed to work on
 * orthogonal transforms! */
static void
rect_transform(ipa_rect *dst, const ipa_rect *src, const ipa_matrix *mat)
{
    float x = mat->xx * src->x0 + mat->xy * src->y0 + mat->tx;
    float y = mat->yx * src->x0 + mat->yy * src->y0 + mat->ty;
    dst->x0 = dst->x1 = x;
    dst->y0 = dst->y1 = y;
    x = mat->xx * src->x1 + mat->xy * src->y1 + mat->tx;
    y = mat->yx * src->x1 + mat->yy * src->y1 + mat->ty;
    if (dst->x0 > x)
        dst->x0 = x;
    else
        dst->x1 = x;
    if (dst->y0 > y)
        dst->y0 = y;
    else
        dst->y1 = y;
}

static void frect_from_rect(ipa_irect *fr, ipa_rect *r)
{
    fr->x0 = (int)floor(r->x0 * 256);
    fr->y0 = (int)floor(r->y0 * 256);
    fr->x1 = (int)floor(r->x1 * 256);
    fr->y1 = (int)floor(r->y1 * 256);
}

static void irect_from_frect(ipa_irect *ir, ipa_irect *fr)
{
    ir->x0 = (fr->x0 + 127)>>8;
    ir->y0 = (fr->y0 + 127)>>8;
    ir->x1 = (fr->x1 + 127)>>8;
    ir->y1 = (fr->y1 + 127)>>8;
}

static void
copy_contone(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int w;
    int dst_off = (int)ht->cdrect.x0 - ht->left;
    int dw = ht->cdrect.x1 - ht->cdrect.x0;

    for (plane = 0; plane < ht->num_planes; plane++)
    {
        bresenham_t brx = ht->brx;
        const uint8_t *b = buffer[plane];
        uint8_t *d = &ht->contone[dst_off + ht->contone_stride * plane];
        for (w = dw; w > 0; w--)
        {
            *d++ = b[brx.i];
            bresenham_step(&brx);
        }
    }
}

static void
copy_contone_1to1(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int dst_off = (int)ht->cdrect.x0 - ht->left;
    int dw = ht->cdrect.x1 - ht->cdrect.x0;

    for (plane = 0; plane < ht->num_planes; plane++)
    {
        const uint8_t *b = buffer[plane] + ht->brx.i;
        uint8_t *d = &ht->contone[dst_off + ht->contone_stride * plane];
        memcpy(d, b, dw);
    }
}

static void
copy_contone_rev_1to1(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int w;
    int dst_off = (int)ht->cdrect.x0 - ht->left;
    int dw = ht->cdrect.x1 - ht->cdrect.x0;

    for (plane = 0; plane < ht->num_planes; plane++)
    {
        const uint8_t *b = buffer[plane] + ht->brx.i;
        uint8_t *d = &ht->contone[dst_off + ht->contone_stride * plane];
        for (w = dw; w > 0; w--)
        {
            *d++ = *b--;
        }
    }
}

static void
copy_contone_1to2(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int dst_off = (int)ht->cdrect.x0 - ht->left;
    int dw = ht->cdrect.x1 - ht->cdrect.x0 - 1;

    for (plane = 0; plane < ht->num_planes; plane++)
    {
        const uint8_t *b = buffer[plane] + ht->brx.i;
        uint8_t *d = &ht->contone[dst_off + ht->contone_stride * plane];
        int w = dw;
        if (ht->brx.f <= ht->brx.df)
            *d++ = *b++, w--;
        for (; w > 0; w -= 2)
        {
            uint8_t c = *b++;
            *d++ = c;
            *d++ = c;
        }
        if (w == 0)
            *d = *b;
    }
}

static void
copy_contone_rev_1to2(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int dst_off = (int)ht->cdrect.x0 - ht->left;
    int dw = ht->cdrect.x1 - ht->cdrect.x0 - 1;

    for (plane = 0; plane < ht->num_planes; plane++)
    {
        const uint8_t *b = buffer[plane] + ht->brx.i;
        uint8_t *d = &ht->contone[dst_off + ht->contone_stride * plane];
        int w = dw;
        if (ht->brx.f > ht->brx.df)
            *d++ = *b--, w--;
        for (; w > 0; w -= 2)
        {
            uint8_t c = *b--;
            *d++ = c;
            *d++ = c;
        }
        if (w == 0)
            *d = *b;
    }
}

static void
copy_contone_landscape(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int h;
    int dst_off = ht->out_idx & (HT_LANDSCAPE-1);
    int dh = ht->cdrect.y1 - ht->cdrect.y0;
    uint8_t *d = &ht->contone[dst_off];

    for (plane = 0; plane < ht->num_planes; plane++)
    {
        bresenham_t brx = ht->brx;
        const uint8_t *b = buffer[plane];
        for (h = dh; h > 0; h--)
        {
            *d = b[brx.i];
            d += HT_LANDSCAPE;
            bresenham_step(&brx);
        }
    }
}

static void
copy_contone_landscape_1to1(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int h;
    int dst_off = ht->out_idx & (HT_LANDSCAPE-1);
    int dh = ht->cdrect.y1 - ht->cdrect.y0;
    uint8_t *d = &ht->contone[dst_off];

    for (plane = 0; plane < ht->num_planes; plane++)
    {
        const uint8_t *b = buffer[plane] + ht->brx.i;
        for (h = dh; h > 0; h--)
        {
            *d = *b++;
            d += HT_LANDSCAPE;
        }
    }
}

static void
copy_contone_landscape_rev_1to1(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int h;
    int dst_off = ht->out_idx & (HT_LANDSCAPE-1);
    int dh = ht->cdrect.y1 - ht->cdrect.y0;
    uint8_t *d = &ht->contone[dst_off];

    for (plane = 0; plane < ht->num_planes; plane++)
    {
        const uint8_t *b = buffer[plane] + ht->brx.i;
        for (h = dh; h > 0; h--)
        {
            *d = *b--;
            d += HT_LANDSCAPE;
        }
    }
}

static void
copy_contone_landscape_1to2(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int dst_off = ht->out_idx & (HT_LANDSCAPE-1);
    int dh = ht->cdrect.y1 - ht->cdrect.y0;
    uint8_t *d = &ht->contone[dst_off];

    for (plane = 0; plane < ht->num_planes; plane++)
    {
        const uint8_t *b = buffer[plane] + ht->brx.i;
        int h = dh-1;
        if (ht->brx.f <= ht->brx.df)
            *d = *b++, d += HT_LANDSCAPE, h--;
        for (; h > 0; h -= 2)
        {
            uint8_t c = *b++;
            *d = c;
            d += HT_LANDSCAPE;
            *d = c;
            d += HT_LANDSCAPE;
        }
        if (h == 0)
            *d = *b, d += HT_LANDSCAPE;
    }
}

static void
copy_contone_landscape_rev_1to2(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int dst_off = ht->out_idx & (HT_LANDSCAPE-1);
    int dh = ht->cdrect.y1 - ht->cdrect.y0;
    uint8_t *d = &ht->contone[dst_off];

    for (plane = 0; plane < ht->num_planes; plane++)
    {
        const uint8_t *b = buffer[plane] + ht->brx.i;
        int h = dh-1;
        if (ht->brx.f > ht->brx.df)
            *d = *b--, d += HT_LANDSCAPE, h--;
        for (; h > 0; h -= 2)
        {
            uint8_t c = *b--;
            *d = c;
            d += HT_LANDSCAPE;
            *d = c;
            d += HT_LANDSCAPE;
        }
        if (h == 0)
            *d = *b, d += HT_LANDSCAPE;
    }
}

static void
copy_contone_cache(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int w;
    int dst_off = (int)ht->cdrect.x0 - ht->left;
    int dw = ht->cdrect.x1 - ht->cdrect.x0;
    uint8_t *cache = ht->cache;
    int n = ht->num_planes;
    int stride = ht->contone_stride;
    bresenham_t brx = ht->brx;
    uint8_t *dst = &ht->contone[dst_off];
    const uint8_t *b = buffer[0];

    for (w = dw; w > 0; w--)
    {
        uint8_t v = b[brx.i];
        uint8_t *d = dst++;
        uint8_t *c = &cache[v*n];
        for (plane = n; plane > 0; plane--)
            *d = *c++, d += stride;
        bresenham_step(&brx);
    }
}

static void
copy_contone_cache_1to1(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int w;
    int dst_off = (int)ht->cdrect.x0 - ht->left;
    int dw = ht->cdrect.x1 - ht->cdrect.x0;
    uint8_t *cache = ht->cache;
    int n = ht->num_planes;
    int stride = ht->contone_stride;
    bresenham_t brx = ht->brx;
    uint8_t *dst = &ht->contone[dst_off];
    const uint8_t *b = &buffer[0][brx.i];

    for (w = dw; w > 0; w--)
    {
        uint8_t v = *b++;
        uint8_t *d = dst++;
        uint8_t *c = &cache[v*n];
        for (plane = n; plane > 0; plane--)
            *d = *c++, d += stride;
    }
}

static void
copy_contone_cache_rev_1to1(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int w;
    int dst_off = (int)ht->cdrect.x0 - ht->left;
    int dw = ht->cdrect.x1 - ht->cdrect.x0;
    uint8_t *cache = ht->cache;
    int n = ht->num_planes;
    int stride = ht->contone_stride;
    bresenham_t brx = ht->brx;
    uint8_t *dst = &ht->contone[dst_off];
    const uint8_t *b = &buffer[0][brx.i];

    for (w = dw; w > 0; w--)
    {
        uint8_t v = *b--;
        uint8_t *d = dst++;
        uint8_t *c = &cache[v*n];
        for (plane = n; plane > 0; plane--)
            *d = *c++, d += stride;
    }
}

static void
copy_contone_cache_1to2(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int dst_off = (int)ht->cdrect.x0 - ht->left;
    int w = ht->cdrect.x1 - ht->cdrect.x0 - 1;
    uint8_t *cache = ht->cache;
    int n = ht->num_planes;
    int stride = ht->contone_stride;
    bresenham_t brx = ht->brx;
    uint8_t *dst = &ht->contone[dst_off];
    const uint8_t *b = &buffer[0][brx.i];

    if (ht->brx.f <= ht->brx.df)
    {
        uint8_t v = *b++;
        uint8_t *d = dst++;
        uint8_t *c = &cache[v*n];
        for (plane = n; plane > 0; plane--)
            *d = *c++, d += stride;
        w--;
    }
    for (; w > 0; w -= 2)
    {
        uint8_t v = *b++;
        uint8_t *d = dst;
        uint8_t *c = &cache[v*n];
        for (plane = n; plane > 0; plane--)
        {
            uint8_t e = *c++;
            d[0] = d[1] = e;
            d += stride;
        }
        dst += 2;
    }
    if (w == 0)
    {
        uint8_t v = *b;
        uint8_t *c = &cache[v*n];
        for (plane = n; plane > 0; plane--)
            *dst = *c++, dst += stride;
    }
}

static void
copy_contone_cache_rev_1to2(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int dst_off = (int)ht->cdrect.x0 - ht->left;
    int w = ht->cdrect.x1 - ht->cdrect.x0 - 1;
    uint8_t *cache = ht->cache;
    int n = ht->num_planes;
    int stride = ht->contone_stride;
    bresenham_t brx = ht->brx;
    uint8_t *dst = &ht->contone[dst_off];
    const uint8_t *b = &buffer[0][brx.i];

    if (ht->brx.f <= ht->brx.df)
    {
        uint8_t v = *b--;
        uint8_t *d = dst++;
        uint8_t *c = &cache[v*n];
        for (plane = n; plane > 0; plane--)
            *d = *c++, d += stride;
        w--;
    }
    for (; w > 0; w -= 2)
    {
        uint8_t v = *b--;
        uint8_t *d = dst;
        uint8_t *c = &cache[v*n];
        for (plane = n; plane > 0; plane--)
        {
            d[0] = d[1] = *c++;
            d += stride;
        }
        dst += 2;
    }
    if (w == 0)
    {
        uint8_t v = *b;
        uint8_t *c = &cache[v*n];
        for (plane = n; plane > 0; plane--)
            *dst = *c++, dst += stride;
    }
}

static void
copy_contone_cache_landscape(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int h;
    int dst_off = ht->out_idx & (HT_LANDSCAPE-1);
    int dh = ht->cdrect.y1 - ht->cdrect.y0;
    uint8_t *dst = &ht->contone[dst_off];
    bresenham_t brx = ht->brx;
    const uint8_t *b = buffer[0];
    int stride = HT_LANDSCAPE * dh;
    uint8_t *cache = ht->cache;
    int n = ht->num_planes;

    for (h = dh; h > 0; h--)
    {
        uint8_t v = b[brx.i];
        uint8_t *c = &cache[v * n];
        uint8_t *d = dst;
        dst += HT_LANDSCAPE;
        for (plane = n; plane > 0; plane--)
        {
            *d = *c++;
            d += stride;
        }
        bresenham_step(&brx);
    }
}

static void
copy_contone_cache_landscape_1to1(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int h;
    int dst_off = ht->out_idx & (HT_LANDSCAPE-1);
    int dh = ht->cdrect.y1 - ht->cdrect.y0;
    uint8_t *dst = &ht->contone[dst_off];
    bresenham_t brx = ht->brx;
    const uint8_t *b = &buffer[0][brx.i];
    int stride = HT_LANDSCAPE * dh;
    uint8_t *cache = ht->cache;
    int n = ht->num_planes;

    for (h = dh; h > 0; h--)
    {
        uint8_t v = *b++;
        uint8_t *c = &cache[v * n];
        uint8_t *d = dst;
        dst += HT_LANDSCAPE;
        for (plane = n; plane > 0; plane--)
        {
            *d = *c++;
            d += stride;
        }
    }
}

static void
copy_contone_cache_landscape_rev_1to1(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int h;
    int dst_off = ht->out_idx & (HT_LANDSCAPE-1);
    int dh = ht->cdrect.y1 - ht->cdrect.y0;
    uint8_t *dst = &ht->contone[dst_off];
    bresenham_t brx = ht->brx;
    const uint8_t *b = &buffer[0][brx.i];
    int stride = HT_LANDSCAPE * dh;
    uint8_t *cache = ht->cache;
    int n = ht->num_planes;

    for (h = dh; h > 0; h--)
    {
        uint8_t v = *b--;
        uint8_t *c = &cache[v * n];
        uint8_t *d = dst;
        dst += HT_LANDSCAPE;
        for (plane = n; plane > 0; plane--)
        {
            *d = *c++;
            d += stride;
        }
    }
}

static void
copy_contone_cache_landscape_1to2(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int dst_off = ht->out_idx & (HT_LANDSCAPE-1);
    int h = ht->cdrect.y1 - ht->cdrect.y0;
    uint8_t *dst = &ht->contone[dst_off];
    bresenham_t brx = ht->brx;
    const uint8_t *b = &buffer[0][brx.i];
    int stride = HT_LANDSCAPE * h;
    uint8_t *cache = ht->cache;
    int n = ht->num_planes;

    h--;
    if (ht->brx.f <= ht->brx.df)
    {
        uint8_t v = *b++;
        uint8_t *c = &cache[v * n];
        uint8_t *d = dst;
        dst += HT_LANDSCAPE;
        for (plane = n; plane > 0; plane--)
        {
            *d = *c++;
            d += stride;
        }
        h--;
    }
    for (; h > 0; h -= 2)
    {
        uint8_t v = *b++;
        uint8_t *c = &cache[v * n];
        uint8_t *d = dst;
        dst += HT_LANDSCAPE;
        for (plane = n; plane > 0; plane--)
        {
            d[0] = d[1] = *c++;
            d += stride;
        }
    }
    if (h == 0)
    {
        uint8_t v = *b;
        uint8_t *c = &cache[v * n];
        uint8_t *d = dst;
        dst += HT_LANDSCAPE;
        for (plane = n; plane > 0; plane--)
        {
            *d = *c++;
            d += stride;
        }
    }
}

static void
copy_contone_cache_landscape_rev_1to2(ipa_halftone *ht, const unsigned char **buffer)
{
    int plane;
    int dst_off = ht->out_idx & (HT_LANDSCAPE-1);
    int h = ht->cdrect.y1 - ht->cdrect.y0;
    uint8_t *dst = &ht->contone[dst_off];
    bresenham_t brx = ht->brx;
    const uint8_t *b = &buffer[0][brx.i];
    int stride = HT_LANDSCAPE * h;
    uint8_t *cache = ht->cache;
    int n = ht->num_planes;

    h--;
    if (ht->brx.f <= ht->brx.df)
    {
        uint8_t v = *b--;
        uint8_t *c = &cache[v * n];
        uint8_t *d = dst;
        dst += HT_LANDSCAPE;
        for (plane = n; plane > 0; plane--)
        {
            *d = *c++;
            d += stride;
        }
        h--;
    }
    for (; h > 0; h -= 2)
    {
        uint8_t v = *b--;
        uint8_t *c = &cache[v * n];
        uint8_t *d = dst;
        dst += HT_LANDSCAPE;
        for (plane = n; plane > 0; plane--)
        {
            d[0] = d[1] = *c++;
            d += stride;
        }
    }
    if (h == 0)
    {
        uint8_t v = *b;
        uint8_t *c = &cache[v * n];
        uint8_t *d = dst;
        dst += HT_LANDSCAPE;
        for (plane = n; plane > 0; plane--)
        {
            *d = *c++;
            d += stride;
        }
    }
}

/* Include the cores */
#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 0
#include "halftone_c.h"
#endif

#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 1
#include "halftone_sse.h"
#endif

void ipa_halftone_reset(ipa_context *ctx, ipa_halftone *ht)
{
    ipa_matrix *mat = &ht->mat;
    int w = ht->w;
    int h = ht->h;

    if (!ht->landscape) {
        /* Set up the bresenham. We do "destination width" number of steps, and the integer
         * value of the bresenham says which source pixel goes there. */
        /* We always want to write left to right, so reverse the bresenham if required. */
        if (mat->xx > 0)
        {
            bresenham_init_fixed(&ht->brx, 0, w, ht->fdrect.x1, ht->fdrect.x0);
        }
        else
        {
            bresenham_init_fixed(&ht->brx, w, -w, ht->fdrect.x1,  ht->fdrect.x0);
        }
        bresenham_advance(&ht->brx, ht->cdrect.x0 - ht->drect.x0);
        if (mat->yy > 0)
        {
            bresenham_init_fixed(&ht->bry, 0, h, ht->fdrect.y1, ht->fdrect.y0);
            bresenham_advance(&ht->bry, ht->cdrect.y0 - ht->drect.y0);
            ht->out_idx = ht->cdrect.y0;
            ht->in_idx = 0;
        }
        else
        {
            bresenham_init_fixed(&ht->bry, h, -h, ht->fdrect.y1, ht->fdrect.y0);
            bresenham_advance(&ht->bry, ht->drect.y1 - ht->cdrect.y1);
            ht->out_idx = ht->cdrect.y1 - 1;
            ht->in_idx = h-1;
        }
    } else {
        /* Set up the bresenham. We do "destination width" number of steps, and the integer
         * value of the bresenham says which source pixel goes there. */
        /* We always want to copy_contone "downwards" in the buffer. So for -ve xy,
         * we run the bresenham backwards. */
        if (mat->xy > 0)
        {
            bresenham_init_fixed(&ht->brx, 0, w, ht->fdrect.y1, ht->fdrect.y0);
        }
        else
        {
            bresenham_init_fixed(&ht->brx, w, -w, ht->fdrect.y1, ht->fdrect.y0);
        }
        bresenham_advance(&ht->brx, ht->cdrect.y0 - ht->drect.y0);
        if (mat->yx > 0)
        {
            bresenham_init_fixed(&ht->bry, 0, h, ht->fdrect.x1, ht->fdrect.x0);
            bresenham_advance(&ht->bry, ht->cdrect.x0 - ht->drect.x0);
            ht->out_idx = ht->cdrect.x0;
            ht->in_idx = 0;
        }
        else
        {
            bresenham_init_fixed(&ht->bry, h, -h, ht->fdrect.x1, ht->fdrect.x0);
            bresenham_advance(&ht->bry, ht->drect.x1 - ht->cdrect.x1);
            ht->out_idx = ht->cdrect.x1 - 1;
            ht->in_idx = h-1;
        }
    }
}

ipa_halftone *
ipa_halftone_init(ipa_context       *ctx,
                  void              *opaque,
                  int                w,
                  int                h,
                  const ipa_matrix  *mat,
                  unsigned int       num_planes,
                  unsigned char     *cache,
                  int                clip_x,
                  int                clip_y,
                  int                clip_w,
                  int                clip_h,
                  int app)
{
    ipa_halftone *ht;
    int           landscape;
    ipa_rect      srect, drect;
    int           use_sse = ctx->use_sse_4_1;
    int           left, right;

#ifdef IPA_FORCE_SSE
#if IPA_FORCE_SSE
#define SSE_SWITCH(A,B) (A)
#else
#define SSE_SWITCH(A,B) (B)
#endif
#else
#define SSE_SWITCH(A,B) (use_sse ? (A) : (B))
#endif

    if (mat == NULL || w < 0 || h < 0 || clip_w < 0 || clip_h < 0)
        return NULL;

    if (mat->xx != 0 && mat->xy == 0 && mat->yx == 0 && mat->yy != 0)
        landscape = 0;
    else if (mat->xx == 0 && mat->xy != 0 && mat->yx != 0 && mat->yy == 0)
        landscape = 1;
    else
        return NULL;

    ht = ipa_malloc(ctx, opaque, sizeof(*ht));
    if (!ht)
        return NULL;
    memset(ht, 0, sizeof(*ht));

    ht->ctx = ctx;
    ht->landscape = landscape;
    ht->mat = *mat;
    ht->num_planes = num_planes;

    if (cache)
    {
        ht->cache = ipa_malloc(ctx, opaque, 256 * ht->num_planes);
        if (ht->cache == NULL)
            goto fail;
        memcpy(ht->cache, cache, 256 * ht->num_planes);
    }

    /* Figure out the destination rectangle */
    srect.x0 = 0;
    srect.y0 = 0;
    srect.x1 = (float)w;
    srect.y1 = (float)h;
    rect_transform(&drect, &srect, mat);
    frect_from_rect(&ht->fdrect, &drect);

    /* Bend the destination for any-part-of-pixel */
    if (app)
    {
        if ((ht->fdrect.x0 & 255) >= 128)
            ht->fdrect.x0 = (ht->fdrect.x0 & ~255) | 128;
        if ((ht->fdrect.y0 & 255) >= 128)
            ht->fdrect.y0 = (ht->fdrect.y0 & ~255) | 128;
        if ((ht->fdrect.x1 & 255) < 128)
            ht->fdrect.x1 = (ht->fdrect.x1 & ~255) | 128;
        if ((ht->fdrect.y1 & 255) < 128)
            ht->fdrect.y1 = (ht->fdrect.y1 & ~255) | 128;
    }

    irect_from_frect(&ht->drect, &ht->fdrect);

    /* Now clip that */
    ht->cdrect.x0 = (ht->drect.x0 > clip_x) ? ht->drect.x0 : clip_x;
    ht->cdrect.y0 = (ht->drect.y0 > clip_y) ? ht->drect.y0 : clip_y;
    ht->cdrect.x1 = (ht->drect.x1 < clip_x + clip_w) ? ht->drect.x1 : clip_x + clip_w;
    ht->cdrect.y1 = (ht->drect.y1 < clip_y + clip_h) ? ht->drect.y1 : clip_y + clip_h;

    if (ht->cdrect.x0 > ht->cdrect.x1)
        ht->cdrect.x1 = ht->cdrect.x0;
    if (ht->cdrect.y0 > ht->cdrect.y1)
        ht->cdrect.y1 = ht->cdrect.y0;

    /* A goal of this code is so that the halftone data ends
     * up aligned as it would be in the final device space to
     * facilitate fast copying of the data. (i.e. we'd like the
     * caller to be able to do 32 (HT_ALIGN) bit int sized copies
     * without having to rotate bits). */
    left  = (ht->cdrect.x0     ) & ~31;
    right = (ht->cdrect.x1 + 31) & ~31;
    ht->w = w;
    ht->h = h;
    ipa_halftone_reset(ctx, ht);
    if (!landscape)
    {
        /* So, in the portrait case, each time we are called, we
         * will copy the input data into a contone buffer, using
         * a bresenham to transform it as required. Note, this
         * may include reversing it's direction. We will make
         * the corresponding screen data, and then apply the
         * halftone operation to it. */
        int dw = right - left;

        ht->contone_stride = dw;
        ht->contone = ipa_malloc_aligned(ctx, opaque, ht->num_planes * ht->contone_stride, HT_ALIGN);
        if (ht->contone == NULL)
            goto fail;
        ht->screen = ipa_malloc_aligned(ctx, opaque, ht->num_planes * ht->contone_stride, HT_ALIGN);
        if (ht->screen == NULL)
            goto fail;
        ht->halftone_stride = dw>>3;
        ht->halftone = ipa_malloc_aligned(ctx, opaque, ht->num_planes * ht->halftone_stride, HT_ALIGN>>3);
        if (ht->halftone == NULL)
            goto fail;
        ht->left = left;
        ht->right = right;

        ht->copy_contone = cache ? copy_contone_cache : copy_contone;
        if (ht->brx.df == 0)
        {
            if (ht->brx.di == 1)
                ht->copy_contone = cache ? copy_contone_cache_1to1 : copy_contone_1to1;
            else if (ht->brx.di == -1)
                ht->copy_contone = cache ? copy_contone_cache_rev_1to1 : copy_contone_rev_1to1;
        }
        else if (ht->brx.df*2 == ht->brx.d)
        {
            if (ht->brx.di == 0)
                ht->copy_contone = cache ? copy_contone_cache_1to2 : copy_contone_1to2;
            else if (ht->brx.di == -1)
                ht->copy_contone = cache ? copy_contone_cache_rev_1to2 : copy_contone_rev_1to2;
        }
    }
    else
    {
        /* In the landscape case, we are still fed scanlines of data. These
         * correspond to scan columns of data in the output. We gather the
         * incoming data into a (HT_LANDSCAPE x height) buffer, and when it's full
         * we scan convert the whole block of data at once.
         *
         * For alignment reasons, the first and last scan converted buffers
         * may not be full.
         */
        int dh = ht->cdrect.y1 - ht->cdrect.y0;

        ht->contone_stride = HT_LANDSCAPE;
        ht->contone = ipa_malloc_aligned(ctx, opaque, ht->num_planes * ht->contone_stride * dh, HT_LANDSCAPE);
        if (ht->contone == NULL)
            goto fail;
        ht->screen = ipa_malloc_aligned(ctx, opaque, ht->num_planes * ht->contone_stride * dh, HT_LANDSCAPE);
        if (ht->screen == NULL)
            goto fail;
        ht->halftone_stride = HT_LANDSCAPE>>3;
        ht->halftone = ipa_malloc_aligned(ctx, opaque, ht->num_planes * ht->halftone_stride * dh, HT_LANDSCAPE>>3);
        if (ht->halftone == NULL)
            goto fail;
        ht->left = left;
        ht->right = right;

        ht->copy_contone = cache ? copy_contone_cache_landscape : copy_contone_landscape;
        if (ht->brx.df == 0)
        {
            if (ht->brx.di == 1)
                ht->copy_contone = cache ? copy_contone_cache_landscape_1to1 : copy_contone_landscape_1to1;
            else if (ht->brx.di == -1)
                ht->copy_contone = cache ? copy_contone_cache_landscape_rev_1to1 : copy_contone_landscape_rev_1to1;
        }
        else if (ht->brx.df*2 == ht->brx.d)
        {
            if (ht->brx.di == 0)
                ht->copy_contone = cache ? copy_contone_cache_landscape_1to2 : copy_contone_landscape_1to2;
            else if (ht->brx.di == -1)
                ht->copy_contone = cache ? copy_contone_cache_landscape_rev_1to2 : copy_contone_landscape_rev_1to2;
        }
    }

    ht->core = SSE_SWITCH(core_halftone_sse, core_halftone);

    return ht;

fail:
    ipa_halftone_fin(ht, opaque);

    return NULL;
}

static int modulo(int a, int b)
{
    int c = a % b;
    if (a >= 0)
        return c;

    if (c != 0)
        c += b;
    return c;
}

/* Set a screen for a colorant. */
int ipa_halftone_add_screen(ipa_context   *ctx,
                            void          *opaque,
                            ipa_halftone  *ht,
                            int            invert,
                            unsigned int   width,
                            unsigned int   height,
                            unsigned int   x_phase,
                            unsigned int   y_phase,
                            unsigned char *values)
{
    ipa_screen *screen;
    unsigned int n = width * height;
    unsigned int i;

    if (ht == NULL || ht->screen_fill >= ht->num_planes)
        return 1;

    screen = ipa_malloc(ctx, opaque, sizeof(*screen) + n + HT_ALIGN);
    if (screen == NULL)
        return 1;
    memset(screen, 0, sizeof(*screen));

    screen->w       = width;
    screen->h       = height;
    screen->x_phase = modulo(x_phase, width);
    screen->y_phase = modulo(y_phase, height);
    screen->data    = (uint8_t *)(void *)&screen[1];
    screen->data    = (uint8_t *)((((intptr_t)screen->data) + HT_ALIGN-1) & ~(HT_ALIGN-1));

    if (ctx->use_sse_4_1)
    {
        int v = invert ? 0x7f : 0x80;
        for (i = 0; i < n; i++)
            screen->data[i] = values[i] ^ v;
    }
    else
    {
        if (invert)
            for (i = 0; i < n; i++)
                screen->data[i] = ~values[i];
        else
            memcpy(screen->data, values, n);
    }

    ht->screens[ht->screen_fill++] = screen;

    return 0;
}

/* Return non-zero if the next scanlines data will be used. */
int ipa_halftone_next_line_required(ipa_halftone *ht)
{
    if (ht->bry.di >= 0)
    {
        if (ht->bry.i > ht->in_idx)
        {
            ht->in_idx++;
            return 0;
        }
    }
    else
    {
        if (ht->bry.i < ht->in_idx)
        {
            ht->in_idx--;
            return 0;
        }
    }
    return 1;
}

static int
planes_have_offset(const uint8_t **buffer, int num_planes, int stride)
{
    int i;
    for (i = 1; i < num_planes; i++)
        if (buffer[0] + stride*i != buffer[i])
            return 0;
    return 1;
}

/* Halftone some data. */
int ipa_halftone_process_planar(ipa_halftone         *ht,
                                void                 *opaque,
                                const unsigned char **buffer,
                                ipa_ht_callback_t    *callback,
                                void                 *callback_arg)
{
    ipa_halftone_data_t data;
    int plane, y;

    if (!ht || !buffer || ht->screen_fill != ht->num_planes)
        return 1;

    /* If we are scaling down, we occasionally need to skip lines */
    if (ht->bry.di >= 0)
    {
        if (ht->bry.i > ht->in_idx++)
            return 0;
    }
    else
    {
        if (ht->bry.i < ht->in_idx--)
            return 0;
    }

    data.data         = ht->halftone;
    data.raster       = ht->halftone_stride;
    data.x            = ht->cdrect.x0;
    data.h            = ht->landscape ? ht->cdrect.y1 - ht->cdrect.y0 : 1;
    data.plane_raster = ht->halftone_stride * data.h;
    if (!ht->landscape)
    {
        int buffer_w = ht->right     - ht->left;
        int dw       = ht->cdrect.x1 - ht->cdrect.x0;
        int dst_off  = ht->cdrect.x0 - ht->left;
        const uint8_t *contone = ht->contone;

        /* Copy the contone data. */
        if (ht->copy_contone == copy_contone_1to1 &&
            dst_off == 0 &&
            planes_have_offset(buffer, ht->num_planes, ht->contone_stride))
            contone = buffer[0];
        else
            ht->copy_contone(ht, buffer);

        for (y = ht->bry.i; y == ht->bry.i; bresenham_step(&ht->bry))
        {
            /* Make the screen. */
            for (plane = 0; plane < ht->num_planes; plane++)
            {
                uint8_t *d = &ht->screen[dst_off + ht->contone_stride * plane];
                int w = dw;
                int row_w = ht->screens[plane]->w;
                int y_off = ((ht->out_idx + ht->screens[plane]->y_phase) % ht->screens[plane]->h) * row_w;
                int x_off = modulo(ht->screens[plane]->x_phase + ht->cdrect.x0, row_w);
                const uint8_t *ht_row = &ht->screens[plane]->data[y_off];
                while (w > 0)
                {
                    int w2 = w;
                    if (w2 > row_w - x_off)
                        w2 = row_w - x_off;
                    memcpy(d, ht_row + x_off, w2);
                    x_off = 0;
                    d += w2;
                    w -= w2;
                }
            }

            /* Halftone! */
            ht->core(ht->halftone,
                     contone,
                     ht->screen, buffer_w * ht->num_planes);

            /* Callback */
            data.offset_x = (-ht->left)>>3;
            data.y        = ht->out_idx;
            data.w        = dw;
            callback(&data, callback_arg);
            if (ht->mat.yy >= 0)
                ht->out_idx++;
            else
                ht->out_idx--;
        }
    }
    else
    {
        uint8_t *d;
        int dh = ht->cdrect.y1 - ht->cdrect.y0;
        int x;

        for (x = ht->bry.i; x == ht->bry.i; bresenham_step(&ht->bry))
        {
            /* Copy the contone data. */
            ht->copy_contone(ht, buffer);

            /* Increment out_idx, and skip the rest of the loop if we haven't got a block to halftone yet. */
            if (ht->mat.yx >= 0)
            {
                int left = ht->out_idx++;
                if ((((ht->out_idx) & (HT_LANDSCAPE-1)) != 0) && ht->out_idx != ht->cdrect.x1)
                    continue;
                left &= ~(HT_LANDSCAPE-1);
                if (left < ht->cdrect.x0)
                    left = ht->cdrect.x0;
                data.x = left;
                data.w = ht->out_idx - left;
            }
            else
            {
                data.x = ht->out_idx--;
                if ((((ht->out_idx) & (HT_LANDSCAPE-1)) != (HT_LANDSCAPE-1)) && ht->out_idx != ht->cdrect.x0-1)
                    continue;
                data.w = ht->contone_stride - (data.x & (HT_LANDSCAPE-1));
                if (data.w > ht->cdrect.x1 - data.x)
                    data.w = ht->cdrect.x1 - data.x;
            }

            data.offset_x = -((data.x & ~(HT_LANDSCAPE-1))>>3);
            data.y        = ht->cdrect.y0;

            /* Make the screen. */
            d = ht->screen + (data.x & (HT_LANDSCAPE-1));
            for (plane = 0; plane < ht->num_planes; plane++)
            {
                int row_w = ht->screens[plane]->w;
                int y_off = ((ht->cdrect.y0 + ht->screens[plane]->y_phase) % ht->screens[plane]->h) * row_w;
                int x_off = (ht->screens[plane]->x_phase + data.x) % row_w;
                int y_off_max = ht->screens[plane]->h * row_w;
                const uint8_t *ht_base = ht->screens[plane]->data;
                for (y = dh; y > 0; y--)
                {
                    uint8_t *d2 = d;
                    int w = data.w;
                    const uint8_t *ht_row = ht_base + y_off;
                    int x_off2 = x_off;

                    y_off += row_w;
                    if (y_off >= y_off_max)
                        y_off -= y_off_max;
                    while (w > 0)
                    {
                        int w2 = w;
                        if (w2 > row_w - x_off2)
                            w2 = row_w - x_off2;
                        memcpy(d2, ht_row + x_off2, w2);
                        x_off2 = 0;
                        d2 += w2;
                        w -= w2;
                    }
                    d += HT_LANDSCAPE;
                }
            }

            ht->core(ht->halftone,
                        ht->contone,
                        ht->screen, HT_LANDSCAPE * data.h * ht->num_planes);

            /* Callback */
            callback(&data, callback_arg);
        }
    }

    return 0;
}

/* Maybe:
int ipa_halftone_process_chunky(ipa_halftone        *ht,
                                void                *opaque,
                                const unsigned char *buffer);
*/

/* Finalize a halftone instance. */
void ipa_halftone_fin(ipa_halftone *ht, void *opaque)
{
    ipa_context *ctx;
    int i;

    if (ht == NULL)
        return;

    ctx = ht->ctx;
    ipa_free_aligned(ctx, opaque, ht->contone);
    ipa_free_aligned(ctx, opaque, ht->screen);
    ipa_free_aligned(ctx, opaque, ht->halftone);

    for (i = 0; i < ht->screen_fill; i++)
        ipa_free(ctx, opaque, ht->screens[i]);

    ipa_free(ctx, opaque, ht->cache);

    ipa_free(ctx, opaque, ht);
}
