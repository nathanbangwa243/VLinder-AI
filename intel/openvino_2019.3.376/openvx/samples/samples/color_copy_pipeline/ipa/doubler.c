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
/* Bitmap "doubling". */

#define _CRT_SECURE_NO_WARNINGS

#include "ipa-impl.h"

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

//#define DEBUG_DOUBLER

//int errprintf_nomem(const char *string, ...);
//#define printf errprintf_nomem

typedef int (double_fn)(uint8_t       ** ipa_restrict dst,
                        const uint8_t ** ipa_restrict in_lines,
                        ipa_doubler    * ipa_restrict doubler);

#define CLAMP(v, mn, mx)\
  (v < mn ? mn : v > mx ? mx : v)


struct ipa_doubler_s
{
    /* The init procedure sets the following. */
    ipa_context        *ctx;
    uint32_t            channels;
    uint32_t            src_w;
    uint32_t            src_h;
    uint32_t            factor;
    uint32_t            in_y; /* How many lines we have read in. */
    uint32_t            out_y; /* How many lines we have written out. */
    uint32_t            support;
    ipa_double_quality  quality;
    double_fn          *action;
    double_fn          *top;
    uint32_t            tmp_stride;
    uint32_t           *tmp;
    int32_t             tmp_y;
};

/* Include the cores */
#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 0
#include "double_c.h"
#endif

#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 1
#include "double_sse.h"
#endif

enum {
    SUB_PIX_X = 4,
    SUB_PIX_Y = 4,
    MITCHELL_SUPPORT = 4
};

ipa_doubler *ipa_doubler_init(ipa_context        *context,
                              void               *opaque,
                              unsigned int        src_w,
                              unsigned int        src_h,
                              unsigned int        factor,
                              ipa_double_quality  quality,
                              unsigned int        channels,
                              unsigned int       *in_lines)
{
    ipa_doubler *doubler;
    int use_sse = context->use_sse_4_1;
    int needs_tmp = 0;

#ifdef IPA_FORCE_SSE
#if IPA_FORCE_SSE
#define SSE_SWITCH(A,B) (A)
#else
#define SSE_SWITCH(A,B) (B)
#endif
#else
#define SSE_SWITCH(A,B) (use_sse ? (A) : (B))
#endif

    if (in_lines == NULL)
        return NULL;

    if (factor != 2 && factor != 4 && factor != 8)
        return NULL;

    if (quality == IPA_DOUBLE_MITCHELL && src_w < 4)
        quality = IPA_DOUBLE_INTERP;
    if (quality == IPA_DOUBLE_INTERP && src_w < 2)
        quality = IPA_DOUBLE_NEAREST;

    switch (quality)
    {
        case IPA_DOUBLE_NEAREST:
            *in_lines = 1;
            break;
        case IPA_DOUBLE_INTERP:
            *in_lines = 2;
            break;
        case IPA_DOUBLE_MITCHELL:
            *in_lines = 4;
            break;
        default:
            return NULL;
    }

    doubler = ipa_malloc(context, opaque, sizeof(*doubler));
    if (!doubler)
        return NULL;

    memset(doubler, 0, sizeof(*doubler));
    doubler->ctx         = context;
    doubler->channels    = channels;
    doubler->src_w       = src_w;
    doubler->src_h       = src_h;
    doubler->factor      = factor;
    doubler->in_y        = 0;
    doubler->out_y       = 0;
    doubler->support     = 0;

    if (factor == 2)
    {
        switch (channels)
        {
            case 1:
                switch(quality)
                {
                    case IPA_DOUBLE_MITCHELL:
                        doubler->action = SSE_SWITCH((needs_tmp=1, double_mitchell1_sse) , double_mitchell1);
                        doubler->top = SSE_SWITCH(double_mitchell1_top_sse, double_mitchell1_top);
                        break;
                    case IPA_DOUBLE_INTERP:
                        doubler->action = SSE_SWITCH(double_interp1_sse, double_interp1);
                        doubler->top = SSE_SWITCH(double_interp1_top_sse, double_interp1_top);
                        break;
                    default:
                        quality = IPA_DOUBLE_NEAREST;
                        doubler->action = SSE_SWITCH(double_near1_sse, double_near1);
                        break;
                }
                break;
            case 3:
                switch(quality)
                {
                    case IPA_DOUBLE_MITCHELL:
                        doubler->action = SSE_SWITCH((needs_tmp=1, double_mitchell3_sse) , double_mitchell3);
                        doubler->top = SSE_SWITCH(double_mitchell3_top_sse, double_mitchell3_top);
                        break;
                    case IPA_DOUBLE_INTERP:
                        doubler->action = SSE_SWITCH(double_interp3_sse, double_interp3);
                        doubler->top = SSE_SWITCH(double_interp3_top_sse, double_interp3_top);
                        break;
                    default:
                        quality = IPA_DOUBLE_NEAREST;
                        doubler->action = SSE_SWITCH(double_near3_sse, double_near3);
                        break;
                }
                break;
            case 4:
                switch(quality)
                {
                    case IPA_DOUBLE_MITCHELL:
                        doubler->action = SSE_SWITCH((needs_tmp=1, double_mitchell4_sse) , double_mitchell4);
                        doubler->top = SSE_SWITCH(double_mitchell4_top_sse, double_mitchell4_top);
                        break;
                    case IPA_DOUBLE_INTERP:
                        doubler->action = SSE_SWITCH(double_interp4_sse, double_interp4);
                        doubler->top = SSE_SWITCH(double_interp4_top_sse, double_interp4_top);
                        break;
                    default:
                        quality = IPA_DOUBLE_NEAREST;
                        doubler->action = SSE_SWITCH(double_near4_sse, double_near4);
                        break;
                }
                break;
            default:
#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 0
                switch(quality)
                {
                    case IPA_DOUBLE_MITCHELL:
                    case IPA_DOUBLE_INTERP:
                        break;
                    default:
                        quality = IPA_DOUBLE_NEAREST;
                        doubler->action = double_near;
                        break;
                }
#endif
                break;
        }
    }
    else if (factor == 4)
    {
        switch (channels)
        {
            case 1:
                switch (quality)
                {
                    case IPA_DOUBLE_MITCHELL:
                        doubler->action = SSE_SWITCH((needs_tmp=1, quad_mitchell1_sse), quad_mitchell1);
                        doubler->top = SSE_SWITCH(quad_mitchell1_top_sse, quad_mitchell1_top);
                        break;
                    case IPA_DOUBLE_INTERP:
                        doubler->action = SSE_SWITCH(quad_interp1_sse, quad_interp1);
                        doubler->top = SSE_SWITCH(quad_interp1_top_sse, quad_interp1_top);
                        break;
                    default:
                        quality = IPA_DOUBLE_NEAREST;
                        doubler->action = SSE_SWITCH(quad_near1_sse, quad_near1);
                        break;
                }
                break;
            case 3:
                switch (quality)
                {
                    case IPA_DOUBLE_MITCHELL:
                        doubler->action = SSE_SWITCH((needs_tmp=1, quad_mitchell3_sse), quad_mitchell3);
                        doubler->top = SSE_SWITCH(quad_mitchell3_top_sse, quad_mitchell3_top);
                        break;
                    case IPA_DOUBLE_INTERP:
                        doubler->action = SSE_SWITCH(quad_interp3_sse, quad_interp3);
                        doubler->top = SSE_SWITCH(quad_interp3_top_sse, quad_interp3_top);
                        break;
                    default:
                        quality = IPA_DOUBLE_NEAREST;
                        doubler->action = SSE_SWITCH(quad_near3_sse, quad_near3);
                        break;
                }
                break;
            case 4:
                switch (quality)
                {
                    case IPA_DOUBLE_MITCHELL:
                        doubler->action = SSE_SWITCH((needs_tmp=1, quad_mitchell4_sse), quad_mitchell4);
                        doubler->top = SSE_SWITCH(quad_mitchell4_top_sse, quad_mitchell4_top);
                        break;
                    case IPA_DOUBLE_INTERP:
                        doubler->action = SSE_SWITCH(quad_interp4_sse, quad_interp4);
                        doubler->top = SSE_SWITCH(quad_interp4_top_sse, quad_interp4_top);
                        break;
                    default:
                        quality = IPA_DOUBLE_NEAREST;
                        doubler->action = SSE_SWITCH(quad_near4_sse, quad_near4);
                        break;
                }
                break;
            default:
#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 0
                switch (quality)
                {
                    case IPA_DOUBLE_MITCHELL:
                    case IPA_DOUBLE_INTERP:
                        break;
                    default:
                        quality = IPA_DOUBLE_NEAREST;
                        doubler->action = quad_near;
                        break;
                }
#endif
                break;
        }
    }
    else if (factor == 8)
    {
        quality = IPA_DOUBLE_NEAREST;
        switch (channels)
        {
            case 1:
                doubler->action = SSE_SWITCH(octo_near1_sse, octo_near1);
                break;
            case 3:
                doubler->action = SSE_SWITCH(octo_near3_sse, octo_near3);
                break;
            case 4:
                doubler->action = SSE_SWITCH(octo_near4_sse, octo_near4);
                break;
            default:
#if !defined(IPA_FORCE_SSE) || IPA_FORCE_SSE == 0
                doubler->action = octo_near;
#endif
                break;
        }
    }
    doubler->quality     = quality;

    if (needs_tmp)
    {
        doubler->tmp_stride = doubler->src_w * doubler->factor * channels;
        doubler->tmp = ipa_malloc(context, opaque, MITCHELL_SUPPORT * doubler->tmp_stride * sizeof(*doubler->tmp));
        if (doubler->tmp == NULL)
            goto fail;
    }

    if (doubler->action == NULL)
    {
fail:
        ipa_free(context, opaque, doubler->tmp);
        ipa_free(context, opaque, doubler);
        return NULL;
    }

    return doubler;
}

int ipa_doubler_process(ipa_doubler *doubler, void *opaque,
                        const unsigned char **input,
                        unsigned char **output)
{
    int ret;
    const unsigned char *ins[4];
    unsigned char *outs[4];

    switch (doubler->quality)
    {
        case IPA_DOUBLE_INTERP:
            if (input[0] == NULL)
            {
                ins[0] = ins[1] = input[1];
                goto top_or_bottom;
            }
            else if (input[1] == NULL)
            {
                ins[0] = ins[1] = input[0];
                goto top_or_bottom;
            }
            break;
        case IPA_DOUBLE_MITCHELL:
            if (doubler->tmp != NULL)
            {
                if (input[0] == NULL || input[3] == NULL)
                    goto tmp_top_or_bottom;
                break;
            }
            /* For factor = 2:
             *  Output: | A | C | E | G | I | K | M | O |
             *  Input:  |   a   |   b   |   c   |   d   |
             * Normally, we generate:
             *  E from a*MW0 + b*MW1 + c*MW2 + d*MW3
             *  G from a*MW4 + b*MW5 + c*MW6 + d*MW7
             *  I from a*MW7 + b*MW6 + c*MW5 + d*MW4
             *  K from a*MW3 + b*MW2 + c*MW1 + d*MW0
             *
             * So special case leading/trailing code is required for A, and
             * normal code for C/E.
             *
             * For factor = 4:
             *  Output: |A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|
             *  Input:  |   a   |   b   |   c   |   d   |
             * Normally, we generate:
             *  G from a*MW0 + b*MW1 + c*MW2 + d*MW3
             *  H from a*MW4 + b*MW5 + c*MW6 + d*MW7
             *  I from a*MW7 + b*MW6 + c*MW5 + d*MW4
             *  J from a*MW3 + b*MW2 + c*MW1 + d*MW0
             *
             * So special case leading code is required for A/B, and normal
             * code for C/D/E/F.
             */
            if (input[0] == NULL)
            {
                if (input[2] == NULL)
                    return 0;
                if (input[1] == NULL)
                {
                    /* 2 leading lines of data.
                     * Factor 2 - generate A from a,a,a,b (reversed)
                     * Factor 4 - generate A/B from a,a,a,b (reversed)
                     */
                    ins[1] = ins[2] = ins[3] = input[2];
                    ins[0] = input[3];
                    if (doubler->factor == 4)
                    {
                        outs[0] = output[1];
                        outs[1] = output[0];
                        output = outs;
                    }
                    goto top_or_bottom;
                }
                /* 3 leading lines of data.
                 * Factor 2 - generate C/E from a,a,b,c
                 * Factor 4 - generate C/D/E/F from a,a,b,c
                 */
                ins[0] = ins[1] = input[1];
                ins[2] = input[2];
                ins[3] = input[3];
                input = ins;
            }
            else if (input[3] == NULL)
            {
                if (input[1] == NULL)
                    return 0;
                if (input[2] == NULL)
                {
                    /* 2 trailing lines of data.
                     * Factor 2 - generate O from c,d,d,d
                     * Factor 4 - generate O/P from c,d,d,d
                     */
                    ins[0] = input[0];
                    ins[1] = ins[2] = ins[3] = input[1];
                    goto top_or_bottom;
                }
                /* 3 trailing lines of data.
                 * Factor 2 - generate K/M from b,c,d,d
                 * Factor 2 - generate K/L/M/N from b,c,d,d
                 */
                ins[0] = input[0];
                ins[1] = input[1];
                ins[2] = ins[3] = input[2];
                input = ins;
            }
            break;
        default:
            break;
    }

    if (0)
    {
top_or_bottom:
        input = ins;
tmp_top_or_bottom:
        ret = doubler->top(output, input, doubler);
    }
    else
        ret = doubler->action(output, input, doubler);

    doubler->in_y++;
    doubler->out_y += ret;

    return ret;
}

void ipa_doubler_fin(ipa_doubler *doubler, void *opaque)
{
    if (!doubler)
        return;

    ipa_free(doubler->ctx, opaque, doubler->tmp);
    ipa_free(doubler->ctx, opaque, doubler);
}
