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

static int
double_near1(uint8_t       ** ipa_restrict dsts,
             const uint8_t ** ipa_restrict srcs,
             ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];

    for (w = doubler->src_w; w > 0; w--)
    {
        d0[0] = d0[1] = d1[0] = d1[1] = *s0++;
        d0 += 2;
        d1 += 2;
    }

    return 2;
}

static int
double_near3(uint8_t       ** ipa_restrict dsts,
             const uint8_t ** ipa_restrict srcs,
             ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];

    for (w = doubler->src_w; w > 0; w--)
    {
        d0[0] = d0[3] = d1[0] = d1[3] = *s0++;
        d0[1] = d0[4] = d1[1] = d1[4] = *s0++;
        d0[2] = d0[5] = d1[2] = d1[5] = *s0++;
        d0 += 6;
        d1 += 6;
    }

    return 2;
}

static int
double_near4(uint8_t       ** ipa_restrict dsts,
             const uint8_t ** ipa_restrict srcs,
             ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];

    for (w = doubler->src_w; w > 0; w--)
    {
        d0[0] = d0[4] = d1[0] = d1[4] = *s0++;
        d0[1] = d0[5] = d1[1] = d1[5] = *s0++;
        d0[2] = d0[6] = d1[2] = d1[6] = *s0++;
        d0[3] = d0[7] = d1[3] = d1[7] = *s0++;
        d0 += 8;
        d1 += 8;
    }

    return 2;
}

static int
double_near(uint8_t       ** ipa_restrict dsts,
            const uint8_t ** ipa_restrict srcs,
            ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    uint32_t channels = doubler->channels;

    for (w = doubler->src_w; w > 0; w--)
    {
        uint32_t j = channels;
        do {
            d0[0] = d0[channels] = d1[0] = d1[channels] = *s0++;
            d0++;
            d1++;
            j--;
        } while (j != 0);
        d0 += channels;
        d1 += channels;
    }

    return 2;
}

/* Simple 3/4 + 1/4 interpolation */

#define COMBINE(L,R) { int t = L*3+R; R = L+R*3; L = t; }
static int
double_interp1(uint8_t       ** ipa_restrict dsts,
               const uint8_t ** ipa_restrict srcs,
               ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    int tl = *s0++;
    int bl = *s1++;
    int tr;
    int br;

    /* Leading single pixel */
    COMBINE(tl, bl);
    *d0++ = (tl+2)>>2;
    *d1++ = (bl+2)>>2;

    for (w = doubler->src_w-1; w > 0; w--)
    {
        tr = *s0++;
        br = *s1++;
        COMBINE(tr, br);
        d0[0] = (tl*3+tr  + 8)>>4;
        d0[1] = (tl  +tr*3+ 8)>>4;
        d1[0] = (bl*3+br  + 8)>>4;
        d1[1] = (bl  +br*3+ 8)>>4;
        d0 += 2;
        d1 += 2;
        tl = tr;
        bl = br;
    }

    /* Trailing single pixel */
    d0[0] = (tl+2)>>2;
    d1[0] = (bl+2)>>2;

    return 2;
}

static int
double_interp1_top(uint8_t       ** ipa_restrict dsts,
                   const uint8_t ** ipa_restrict srcs,
                   ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d = dsts[0];
    const uint8_t * ipa_restrict s = srcs[0];
    int l = *s++;
    int r;

    /* Leading single pixel */
    d[0] = l;
    d++;

    for (w = doubler->src_w-1; w > 0; w--)
    {
        r = *s++;
        d[0] = (l*3+r  + 2)>>2;
        d[1] = (l  +r*3+ 2)>>2;
        d += 2;
        l = r;
    }

    /* Trailing single pixel */
    d[0] = l;

    return 1;
}

static int
double_interp3(uint8_t       ** ipa_restrict dsts,
               const uint8_t ** ipa_restrict srcs,
               ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    int tl_r = *s0++;
    int tl_g = *s0++;
    int tl_b = *s0++;
    int bl_r = *s1++;
    int bl_g = *s1++;
    int bl_b = *s1++;
    int tr_r, tr_g, tr_b;
    int br_r, br_g, br_b;

    /* Leading single pixel */
    COMBINE(tl_r, bl_r);
    COMBINE(tl_g, bl_g);
    COMBINE(tl_b, bl_b);
    d0[0] = (tl_r+2)>>2;
    d0[1] = (tl_g+2)>>2;
    d0[2] = (tl_b+2)>>2;
    d1[0] = (bl_r+2)>>2;
    d1[1] = (bl_g+2)>>2;
    d1[2] = (bl_b+2)>>2;
    d0 += 3;
    d1 += 3;

    for (w = doubler->src_w-1; w > 0; w--)
    {
        tr_r = *s0++;
        tr_g = *s0++;
        tr_b = *s0++;
        br_r = *s1++;
        br_g = *s1++;
        br_b = *s1++;
        COMBINE(tr_r, br_r);
        COMBINE(tr_g, br_g);
        COMBINE(tr_b, br_b);
        d0[0] = (tl_r*3+tr_r  + 8)>>4;
        d0[1] = (tl_g*3+tr_g  + 8)>>4;
        d0[2] = (tl_b*3+tr_b  + 8)>>4;
        d0[3] = (tl_r  +tr_r*3+ 8)>>4;
        d0[4] = (tl_g  +tr_g*3+ 8)>>4;
        d0[5] = (tl_b  +tr_b*3+ 8)>>4;
        d1[0] = (bl_r*3+br_r  + 8)>>4;
        d1[1] = (bl_g*3+br_g  + 8)>>4;
        d1[2] = (bl_b*3+br_b  + 8)>>4;
        d1[3] = (bl_r  +br_r*3+ 8)>>4;
        d1[4] = (bl_g  +br_g*3+ 8)>>4;
        d1[5] = (bl_b  +br_b*3+ 8)>>4;
        d0 += 6;
        d1 += 6;
        tl_r = tr_r;
        tl_g = tr_g;
        tl_b = tr_b;
        bl_r = br_r;
        bl_g = br_g;
        bl_b = br_b;
    }

    /* Trailing single pixel */
    d0[0] = (tl_r+2)>>2;
    d0[1] = (tl_g+2)>>2;
    d0[2] = (tl_b+2)>>2;
    d1[0] = (bl_r+2)>>2;
    d1[1] = (bl_g+2)>>2;
    d1[2] = (bl_b+2)>>2;

    return 2;
}

static int
double_interp3_top(uint8_t       ** ipa_restrict dsts,
                   const uint8_t ** ipa_restrict srcs,
                   ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d = dsts[0];
    const uint8_t * ipa_restrict s = srcs[0];
    int l_r = *s++;
    int l_g = *s++;
    int l_b = *s++;
    int r_r, r_g, r_b;

    /* Leading single pixel */
    d[0] = l_r;
    d[1] = l_g;
    d[2] = l_b;
    d += 3;

    for (w = doubler->src_w-1; w > 0; w--)
    {
        r_r = *s++;
        r_g = *s++;
        r_b = *s++;
        d[0] = (l_r*3+r_r  + 2)>>2;
        d[1] = (l_g*3+r_g  + 2)>>2;
        d[2] = (l_b*3+r_b  + 2)>>2;
        d[3] = (l_r  +r_r*3+ 2)>>2;
        d[4] = (l_g  +r_g*3+ 2)>>2;
        d[5] = (l_b  +r_b*3+ 2)>>2;
        d += 6;
        l_r = r_r;
        l_g = r_g;
        l_b = r_b;
    }

    /* Trailing single pixel */
    d[0] = l_r;
    d[1] = l_g;
    d[2] = l_b;

    return 1;
}

static int
double_interp4(uint8_t       ** ipa_restrict dsts,
               const uint8_t ** ipa_restrict srcs,
               ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    int tl_r = *s0++;
    int tl_g = *s0++;
    int tl_b = *s0++;
    int tl_k = *s0++;
    int bl_r = *s1++;
    int bl_g = *s1++;
    int bl_b = *s1++;
    int bl_k = *s1++;
    int tr_r, tr_g, tr_b, tr_k;
    int br_r, br_g, br_b, br_k;

    /* Leading single pixel */
    COMBINE(tl_r, bl_r);
    COMBINE(tl_g, bl_g);
    COMBINE(tl_b, bl_b);
    COMBINE(tl_k, bl_k);
    d0[0] = (tl_r+2)>>2;
    d0[1] = (tl_g+2)>>2;
    d0[2] = (tl_b+2)>>2;
    d0[3] = (tl_k+2)>>2;
    d1[0] = (bl_r+2)>>2;
    d1[1] = (bl_g+2)>>2;
    d1[2] = (bl_b+2)>>2;
    d1[3] = (bl_k+2)>>2;
    d0 += 4;
    d1 += 4;

    for (w = doubler->src_w-1; w > 0; w--)
    {
        tr_r = *s0++;
        tr_g = *s0++;
        tr_b = *s0++;
        tr_k = *s0++;
        br_r = *s1++;
        br_g = *s1++;
        br_b = *s1++;
        br_k = *s1++;
        COMBINE(tr_r, br_r);
        COMBINE(tr_g, br_g);
        COMBINE(tr_b, br_b);
        COMBINE(tr_k, br_k);
        d0[0] = (tl_r*3+tr_r  + 8)>>4;
        d0[1] = (tl_g*3+tr_g  + 8)>>4;
        d0[2] = (tl_b*3+tr_b  + 8)>>4;
        d0[3] = (tl_k*3+tr_k  + 8)>>4;
        d0[4] = (tl_r  +tr_r*3+ 8)>>4;
        d0[5] = (tl_g  +tr_g*3+ 8)>>4;
        d0[6] = (tl_b  +tr_b*3+ 8)>>4;
        d0[7] = (tl_k  +tr_k*3+ 8)>>4;
        d1[0] = (bl_r*3+br_r  + 8)>>4;
        d1[1] = (bl_g*3+br_g  + 8)>>4;
        d1[2] = (bl_b*3+br_b  + 8)>>4;
        d1[3] = (bl_k*3+br_k  + 8)>>4;
        d1[4] = (bl_r  +br_r*3+ 8)>>4;
        d1[5] = (bl_g  +br_g*3+ 8)>>4;
        d1[6] = (bl_b  +br_b*3+ 8)>>4;
        d1[7] = (bl_k  +br_k*3+ 8)>>4;
        d0 += 8;
        d1 += 8;
        tl_r = tr_r;
        tl_g = tr_g;
        tl_b = tr_b;
        tl_k = tr_k;
        bl_r = br_r;
        bl_g = br_g;
        bl_b = br_b;
        bl_k = br_k;
    }

    /* Trailing single pixel */
    d0[0] = (tl_r+2)>>2;
    d0[1] = (tl_g+2)>>2;
    d0[2] = (tl_b+2)>>2;
    d0[3] = (tl_k+2)>>2;
    d1[0] = (bl_r+2)>>2;
    d1[1] = (bl_g+2)>>2;
    d1[2] = (bl_b+2)>>2;
    d1[3] = (bl_k+2)>>2;

    return 2;
}

static int
double_interp4_top(uint8_t       ** ipa_restrict dsts,
                   const uint8_t ** ipa_restrict srcs,
                   ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d = dsts[0];
    const uint8_t * ipa_restrict s = srcs[0];
    int l_r = *s++;
    int l_g = *s++;
    int l_b = *s++;
    int l_k = *s++;
    int r_r, r_g, r_b, r_k;

    /* Leading single pixel */
    d[0] = l_r;
    d[1] = l_g;
    d[2] = l_b;
    d[3] = l_k;
    d += 4;

    for (w = doubler->src_w-1; w > 0; w--)
    {
        r_r = *s++;
        r_g = *s++;
        r_b = *s++;
        r_k = *s++;
        d[0] = (l_r*3+r_r  + 2)>>2;
        d[1] = (l_g*3+r_g  + 2)>>2;
        d[2] = (l_b*3+r_b  + 2)>>2;
        d[3] = (l_k*3+r_k  + 2)>>2;
        d[4] = (l_r  +r_r*3+ 2)>>2;
        d[5] = (l_g  +r_g*3+ 2)>>2;
        d[6] = (l_b  +r_b*3+ 2)>>2;
        d[7] = (l_k  +r_k*3+ 2)>>2;
        d += 8;
        l_r = r_r;
        l_g = r_g;
        l_b = r_b;
        l_k = r_k;
    }

    /* Trailing single pixel */
    d[0] = l_r;
    d[1] = l_g;
    d[2] = l_b;
    d[3] = l_k;

    return 1;
}
#undef COMBINE

/* Mitchell Filtered interpolation */

#define WEIGHT_SHIFT 10
#define MW0 (-24)
#define MW1 (801)
#define MW2 (262)
#define MW3 (-15)

#define WEIGHT_SCALE (1<<WEIGHT_SHIFT)
#define WEIGHT_ROUND (1<<(WEIGHT_SHIFT-1))
#define COMBINE(A,B,C,D) {\
    int A_ = A * MW0 + B * MW1 + C * MW2 + D * MW3;\
    B      = A * MW3 + B * MW2 + C * MW1 + D * MW0;\
    A = A_; }
#define COMBINE2(A,B,C,D) (((A)*MW0 + (B) * MW1 + (C) * MW2 + (D) * MW3 + (WEIGHT_SCALE<<WEIGHT_SHIFT))>>(WEIGHT_SHIFT*2))
#define COMBINE3(A,B,C,D) (((A)*MW3 + (B) * MW2 + (C) * MW1 + (D) * MW0 + (WEIGHT_SCALE<<WEIGHT_SHIFT))>>(WEIGHT_SHIFT*2))

static uint8_t clamp(int in)
{
    if (in < 0)
        return 0;
    if (in > 255)
        return 255;
    return (uint8_t)in;
}

static int
double_mitchell1(uint8_t       ** ipa_restrict dsts,
                 const uint8_t ** ipa_restrict srcs,
                 ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    const uint8_t * ipa_restrict s2 = srcs[2];
    const uint8_t * ipa_restrict s3 = srcs[3];
    int s00;
    int s10;
    int s01 = *s0++;
    int s11 = *s1++;
    int s21 = *s2++;
    int s31 = *s3++;
    int s02 = *s0++;
    int s12 = *s1++;
    int s22 = *s2++;
    int s32 = *s3++;
    int s03 = *s0++;
    int s13 = *s1++;
    int s23 = *s2++;
    int s33 = *s3++;

    /*
     * dst  |A|B|C|D|E|F|G|H|
     * src  | a | b | c | d |
     *
     * D = a*M(-1.25) + b*M(-.25) + c*M(.75) + d*M(1.75)
     * E = a*M(-1.75) + b*M(-.75) + c*M(.25) + d*M(1.25)
     *
     * MW0 = M(-1.25) = M( 1.25)
     * MW1 = M(- .25) = M(  .25)
     * MW2 = M(  .75) = M(- .75)
     * MW3 = M( 1.75) = M(-1.75)
     */

    /* Combine them vertically */
    COMBINE(s01, s11, s21, s31); /* s21, s31 now unused */
    COMBINE(s02, s12, s22, s32); /* s22, s32 now unused */
    COMBINE(s03, s13, s23, s33); /* s23, s33 now unused */

    /* Now do the leading pixels. */
    d0[0] = clamp(COMBINE3(s01,s01,s01,s02)); /* A */
    d0[1] = clamp(COMBINE2(s01,s01,s02,s03)); /* B */
    d0[2] = clamp(COMBINE3(s01,s01,s02,s03)); /* C */
    d1[0] = clamp(COMBINE3(s11,s11,s11,s12)); /* A */
    d1[1] = clamp(COMBINE2(s11,s11,s12,s13)); /* B */
    d1[2] = clamp(COMBINE3(s11,s11,s12,s13)); /* C */
    d0 += 3; d1 += 3;

    for (w = doubler->src_w-3; w > 0; w--)
    {
        s00 = s01; s10 = s11;
        s01 = s02; s11 = s12;
        s02 = s03; s12 = s13;
        s03 = *s0++;
        s13 = *s1++;
        s23 = *s2++;
        s33 = *s3++;
        COMBINE(s03, s13, s23, s33); /* s23, s33 now unused */
        d0[0] = clamp(COMBINE2(s00,s01,s02,s03)); /* D */
        d0[1] = clamp(COMBINE3(s00,s01,s02,s03)); /* E */
        d1[0] = clamp(COMBINE2(s10,s11,s12,s13));
        d1[1] = clamp(COMBINE3(s10,s11,s12,s13));
        d0 += 2;
        d1 += 2;
    }

    /* Trailing pixels */
    d0[0] = clamp(COMBINE2(s01,s02,s03,s03)); /* F */
    d0[1] = clamp(COMBINE3(s01,s02,s03,s03)); /* G */
    d0[2] = clamp(COMBINE2(s02,s03,s03,s03)); /* H */
    d1[0] = clamp(COMBINE2(s11,s12,s13,s13)); /* F */
    d1[1] = clamp(COMBINE3(s11,s12,s13,s13)); /* G */
    d1[2] = clamp(COMBINE2(s12,s13,s13,s13)); /* H */

    return 2;
}

static int
double_mitchell1_top(uint8_t       ** ipa_restrict dsts,
                     const uint8_t ** ipa_restrict srcs,
                     ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    const uint8_t * ipa_restrict s2 = srcs[2];
    const uint8_t * ipa_restrict s3 = srcs[3];
    int s00;
    int s01 = *s0++;
    int s11 = *s1++;
    int s21 = *s2++;
    int s31 = *s3++;
    int s02 = *s0++;
    int s12 = *s1++;
    int s22 = *s2++;
    int s32 = *s3++;
    int s03 = *s0++;
    int s13 = *s1++;
    int s23 = *s2++;
    int s33 = *s3++;

    /*
     * dst  |A|B|C|D|E|F|G|H|
     * src  | a | b | c | d |
     *
     * D = a*M(-1.25) + b*M(-.25) + c*M(.75) + d*M(1.75)
     * E = a*M(-1.75) + b*M(-.75) + c*M(.25) + d*M(1.25)
     *
     * MW0 = M(-1.25) = M( 1.25)
     * MW1 = M(- .25) = M(  .25)
     * MW2 = M(  .75) = M(- .75)
     * MW3 = M( 1.75) = M(-1.75)
     */

    /* Combine them vertically */
    COMBINE(s01, s11, s21, s31); /* s21, s31 now unused */
    COMBINE(s02, s12, s22, s32); /* s22, s32 now unused */
    COMBINE(s03, s13, s23, s33); /* s23, s33 now unused */

    /* Now do the leading pixels. */
    d0[0] = clamp(COMBINE3(s01,s01,s01,s02)); /* A */
    d0[1] = clamp(COMBINE2(s01,s01,s02,s03)); /* B */
    d0[2] = clamp(COMBINE3(s01,s01,s02,s03)); /* C */
    d0 += 3;

    for (w = doubler->src_w-3; w > 0; w--)
    {
        s00 = s01;
        s01 = s02;
        s02 = s03;
        s03 = *s0++;
        s13 = *s1++;
        s23 = *s2++;
        s33 = *s3++;
        COMBINE(s03, s13, s23, s33); /* s23, s33 now unused */
        d0[0] = clamp(COMBINE2(s00,s01,s02,s03)); /* D */
        d0[1] = clamp(COMBINE3(s00,s01,s02,s03)); /* E */
        d0 += 2;
    }

    /* Trailing pixels */
    d0[0] = clamp(COMBINE2(s01,s02,s03,s03)); /* F */
    d0[1] = clamp(COMBINE3(s01,s02,s03,s03)); /* G */
    d0[2] = clamp(COMBINE2(s02,s03,s03,s03)); /* H */

    return 1;
}

#define RGB_DECLARE(A) int A##_r,A##_g,A##_b
#define RGB_COMBINE(A,B,C,D) {\
    COMBINE(A##_r,B##_r,C##_r,D##_r);\
    COMBINE(A##_g,B##_g,C##_g,D##_g);\
    COMBINE(A##_b,B##_b,C##_b,D##_b);\
}
#define RGB_LOAD(A,B) {\
    A##_r = B[0];\
    A##_g = B[1];\
    A##_b = B[2];\
    B += 3;\
}
#define RGB_ASSIGN(A,B) { A##_r = B##_r; A##_g = B##_g; A##_b = B##_b; }
#define RGB_STORE2(DST,A,B,C,D) {\
    DST[0] = clamp(COMBINE2(A##_r,B##_r,C##_r,D##_r));\
    DST[1] = clamp(COMBINE2(A##_g,B##_g,C##_g,D##_g));\
    DST[2] = clamp(COMBINE2(A##_b,B##_b,C##_b,D##_b));\
    DST += 3;\
}
#define RGB_STORE3(DST,A,B,C,D) {\
    DST[0] = clamp(COMBINE3(A##_r,B##_r,C##_r,D##_r));\
    DST[1] = clamp(COMBINE3(A##_g,B##_g,C##_g,D##_g));\
    DST[2] = clamp(COMBINE3(A##_b,B##_b,C##_b,D##_b));\
    DST += 3;\
}

static int
double_mitchell3(uint8_t       ** ipa_restrict dsts,
                 const uint8_t ** ipa_restrict srcs,
                 ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    const uint8_t * ipa_restrict s2 = srcs[2];
    const uint8_t * ipa_restrict s3 = srcs[3];
    RGB_DECLARE(s00); RGB_DECLARE(s10);
    RGB_DECLARE(s01); RGB_DECLARE(s11); RGB_DECLARE(s21); RGB_DECLARE(s31);
    RGB_DECLARE(s02); RGB_DECLARE(s12); RGB_DECLARE(s22); RGB_DECLARE(s32);
    RGB_DECLARE(s03); RGB_DECLARE(s13); RGB_DECLARE(s23); RGB_DECLARE(s33);
    RGB_LOAD(s01, s0); RGB_LOAD(s02, s0); RGB_LOAD(s03, s0);
    RGB_LOAD(s11, s1); RGB_LOAD(s12, s1); RGB_LOAD(s13, s1);
    RGB_LOAD(s21, s2); RGB_LOAD(s22, s2); RGB_LOAD(s23, s2);
    RGB_LOAD(s31, s3); RGB_LOAD(s32, s3); RGB_LOAD(s33, s3);

    /*
     * dst  |A|B|C|D|E|F|G|H|
     * src  | a | b | c | d |
     *
     * D = a*M(-1.25) + b*M(-.25) + c*M(.75) + d*M(1.75)
     * E = a*M(-1.75) + b*M(-.75) + c*M(.25) + d*M(1.25)
     *
     * MW0 = M(-1.25) = M( 1.25)
     * MW1 = M(- .25) = M(  .25)
     * MW2 = M(  .75) = M(- .75)
     * MW3 = M( 1.75) = M(-1.75)
     */

    /* Combine them vertically */
    RGB_COMBINE(s01, s11, s21, s31); /* s21, s31 now unused */
    RGB_COMBINE(s02, s12, s22, s32); /* s22, s32 now unused */
    RGB_COMBINE(s03, s13, s23, s33); /* s23, s33 now unused */

    /* Now do the leading pixels. */
    RGB_STORE3(d0,s01,s01,s01,s02); /* A */
    RGB_STORE2(d0,s01,s01,s02,s03); /* B */
    RGB_STORE3(d0,s01,s01,s02,s03); /* C */
    RGB_STORE3(d1,s11,s11,s11,s12); /* A */
    RGB_STORE2(d1,s11,s11,s12,s13); /* B */
    RGB_STORE3(d1,s11,s11,s12,s13); /* C */

    for (w = doubler->src_w-3; w > 0; w--)
    {
        RGB_ASSIGN(s00, s01); RGB_ASSIGN(s10, s11);
        RGB_ASSIGN(s01, s02); RGB_ASSIGN(s11, s12);
        RGB_ASSIGN(s02, s03); RGB_ASSIGN(s12, s13);
        RGB_LOAD(s03, s0); RGB_LOAD(s13, s1); RGB_LOAD(s23, s2); RGB_LOAD(s33, s3);
        RGB_COMBINE(s03, s13, s23, s33); /* s23, s33 now unused */
        RGB_STORE2(d0,s00,s01,s02,s03); /* D */
        RGB_STORE3(d0,s00,s01,s02,s03); /* E */
        RGB_STORE2(d1,s10,s11,s12,s13); /* D */
        RGB_STORE3(d1,s10,s11,s12,s13); /* E */
    }

    /* Trailing pixels */
    RGB_STORE2(d0,s01,s02,s03,s03); /* F */
    RGB_STORE3(d0,s01,s02,s03,s03); /* G */
    RGB_STORE2(d0,s02,s03,s03,s03); /* H */
    RGB_STORE2(d1,s11,s12,s13,s13); /* F */
    RGB_STORE3(d1,s11,s12,s13,s13); /* G */
    RGB_STORE2(d1,s12,s13,s13,s13); /* H */

    return 2;
}

static int
double_mitchell3_top(uint8_t       ** ipa_restrict dsts,
                     const uint8_t ** ipa_restrict srcs,
                     ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    const uint8_t * ipa_restrict s2 = srcs[2];
    const uint8_t * ipa_restrict s3 = srcs[3];
    RGB_DECLARE(s00);
    RGB_DECLARE(s01); RGB_DECLARE(s11); RGB_DECLARE(s21); RGB_DECLARE(s31);
    RGB_DECLARE(s02); RGB_DECLARE(s12); RGB_DECLARE(s22); RGB_DECLARE(s32);
    RGB_DECLARE(s03); RGB_DECLARE(s13); RGB_DECLARE(s23); RGB_DECLARE(s33);
    RGB_LOAD(s01, s0); RGB_LOAD(s02, s0); RGB_LOAD(s03, s0);
    RGB_LOAD(s11, s1); RGB_LOAD(s12, s1); RGB_LOAD(s13, s1);
    RGB_LOAD(s21, s2); RGB_LOAD(s22, s2); RGB_LOAD(s23, s2);
    RGB_LOAD(s31, s3); RGB_LOAD(s32, s3); RGB_LOAD(s33, s3);

    /*
     * dst  |A|B|C|D|E|F|G|H|
     * src  | a | b | c | d |
     *
     * D = a*M(-1.25) + b*M(-.25) + c*M(.75) + d*M(1.75)
     * E = a*M(-1.75) + b*M(-.75) + c*M(.25) + d*M(1.25)
     *
     * MW0 = M(-1.25) = M( 1.25)
     * MW1 = M(- .25) = M(  .25)
     * MW2 = M(  .75) = M(- .75)
     * MW3 = M( 1.75) = M(-1.75)
     */

    /* Combine them vertically */
    RGB_COMBINE(s01, s11, s21, s31); /* s21, s31 now unused */
    RGB_COMBINE(s02, s12, s22, s32); /* s22, s32 now unused */
    RGB_COMBINE(s03, s13, s23, s33); /* s23, s33 now unused */

    /* Now do the leading pixels. */
    RGB_STORE3(d0,s01,s01,s01,s02); /* A */
    RGB_STORE2(d0,s01,s01,s02,s03); /* B */
    RGB_STORE3(d0,s01,s01,s02,s03); /* C */

    for (w = doubler->src_w-3; w > 0; w--)
    {
        RGB_ASSIGN(s00, s01);
        RGB_ASSIGN(s01, s02);
        RGB_ASSIGN(s02, s03);
        RGB_LOAD(s03, s0); RGB_LOAD(s13, s1); RGB_LOAD(s23, s2); RGB_LOAD(s33, s3);
        RGB_COMBINE(s03, s13, s23, s33); /* s23, s33 now unused */
        RGB_STORE2(d0,s00,s01,s02,s03); /* D */
        RGB_STORE3(d0,s00,s01,s02,s03); /* E */
    }

    /* Trailing pixels */
    RGB_STORE2(d0,s01,s02,s03,s03); /* F */
    RGB_STORE3(d0,s01,s02,s03,s03); /* G */
    RGB_STORE2(d0,s02,s03,s03,s03); /* H */

    return 1;
}
#undef RGB_DECLARE
#undef RGB_LOAD
#undef RGB_COMBINE
#undef RGB_ASSIGN
#undef RGB_STORE2
#undef RGB_STORE3

#define CMYK_DECLARE(A) int A##_r,A##_g,A##_b,A##_k
#define CMYK_COMBINE(A,B,C,D) {\
    COMBINE(A##_r,B##_r,C##_r,D##_r);\
    COMBINE(A##_g,B##_g,C##_g,D##_g);\
    COMBINE(A##_b,B##_b,C##_b,D##_b);\
    COMBINE(A##_k,B##_k,C##_k,D##_k);\
}
#define CMYK_LOAD(A,B) {\
    A##_r = B[0];\
    A##_g = B[1];\
    A##_b = B[2];\
    A##_k = B[3];\
    B += 4;\
}
#define CMYK_ASSIGN(A,B) { A##_r = B##_r; A##_g = B##_g; A##_b = B##_b; A##_k = B##_k; }
#define CMYK_STORE2(DST,A,B,C,D) {\
    DST[0] = clamp(COMBINE2(A##_r,B##_r,C##_r,D##_r));\
    DST[1] = clamp(COMBINE2(A##_g,B##_g,C##_g,D##_g));\
    DST[2] = clamp(COMBINE2(A##_b,B##_b,C##_b,D##_b));\
    DST[3] = clamp(COMBINE2(A##_k,B##_k,C##_k,D##_k));\
    DST += 4;\
}
#define CMYK_STORE3(DST,A,B,C,D) {\
    DST[0] = clamp(COMBINE3(A##_r,B##_r,C##_r,D##_r));\
    DST[1] = clamp(COMBINE3(A##_g,B##_g,C##_g,D##_g));\
    DST[2] = clamp(COMBINE3(A##_b,B##_b,C##_b,D##_b));\
    DST[3] = clamp(COMBINE3(A##_k,B##_k,C##_k,D##_k));\
    DST += 4;\
}
static int
double_mitchell4(uint8_t       ** ipa_restrict dsts,
                 const uint8_t ** ipa_restrict srcs,
                 ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    const uint8_t * ipa_restrict s2 = srcs[2];
    const uint8_t * ipa_restrict s3 = srcs[3];
    CMYK_DECLARE(s00); CMYK_DECLARE(s10);
    CMYK_DECLARE(s01); CMYK_DECLARE(s11); CMYK_DECLARE(s21); CMYK_DECLARE(s31);
    CMYK_DECLARE(s02); CMYK_DECLARE(s12); CMYK_DECLARE(s22); CMYK_DECLARE(s32);
    CMYK_DECLARE(s03); CMYK_DECLARE(s13); CMYK_DECLARE(s23); CMYK_DECLARE(s33);
    CMYK_LOAD(s01, s0); CMYK_LOAD(s02, s0); CMYK_LOAD(s03, s0);
    CMYK_LOAD(s11, s1); CMYK_LOAD(s12, s1); CMYK_LOAD(s13, s1);
    CMYK_LOAD(s21, s2); CMYK_LOAD(s22, s2); CMYK_LOAD(s23, s2);
    CMYK_LOAD(s31, s3); CMYK_LOAD(s32, s3); CMYK_LOAD(s33, s3);

    /*
     * dst  |A|B|C|D|E|F|G|H|
     * src  | a | b | c | d |
     *
     * D = a*M(-1.25) + b*M(-.25) + c*M(.75) + d*M(1.75)
     * E = a*M(-1.75) + b*M(-.75) + c*M(.25) + d*M(1.25)
     *
     * MW0 = M(-1.25) = M( 1.25)
     * MW1 = M(- .25) = M(  .25)
     * MW2 = M(  .75) = M(- .75)
     * MW3 = M( 1.75) = M(-1.75)
     */

    /* Combine them vertically */
    CMYK_COMBINE(s01, s11, s21, s31); /* s21, s31 now unused */
    CMYK_COMBINE(s02, s12, s22, s32); /* s22, s32 now unused */
    CMYK_COMBINE(s03, s13, s23, s33); /* s23, s33 now unused */

    /* Now do the leading pixels. */
    CMYK_STORE3(d0,s01,s01,s01,s02); /* A */
    CMYK_STORE2(d0,s01,s01,s02,s03); /* B */
    CMYK_STORE3(d0,s01,s01,s02,s03); /* C */
    CMYK_STORE3(d1,s11,s11,s11,s12); /* A */
    CMYK_STORE2(d1,s11,s11,s12,s13); /* B */
    CMYK_STORE3(d1,s11,s11,s12,s13); /* C */

    for (w = doubler->src_w-3; w > 0; w--)
    {
        CMYK_ASSIGN(s00, s01); CMYK_ASSIGN(s10, s11);
        CMYK_ASSIGN(s01, s02); CMYK_ASSIGN(s11, s12);
        CMYK_ASSIGN(s02, s03); CMYK_ASSIGN(s12, s13);
        CMYK_LOAD(s03, s0); CMYK_LOAD(s13, s1); CMYK_LOAD(s23, s2); CMYK_LOAD(s33, s3);
        CMYK_COMBINE(s03, s13, s23, s33); /* s23, s33 now unused */
        CMYK_STORE2(d0,s00,s01,s02,s03); /* D */
        CMYK_STORE3(d0,s00,s01,s02,s03); /* E */
        CMYK_STORE2(d1,s10,s11,s12,s13); /* D */
        CMYK_STORE3(d1,s10,s11,s12,s13); /* E */
    }

    /* Trailing pixels */
    CMYK_STORE2(d0,s01,s02,s03,s03); /* F */
    CMYK_STORE3(d0,s01,s02,s03,s03); /* G */
    CMYK_STORE2(d0,s02,s03,s03,s03); /* H */
    CMYK_STORE2(d1,s11,s12,s13,s13); /* F */
    CMYK_STORE3(d1,s11,s12,s13,s13); /* G */
    CMYK_STORE2(d1,s12,s13,s13,s13); /* H */

    return 2;
}

static int
double_mitchell4_top(uint8_t       ** ipa_restrict dsts,
                     const uint8_t ** ipa_restrict srcs,
                     ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    const uint8_t * ipa_restrict s2 = srcs[2];
    const uint8_t * ipa_restrict s3 = srcs[3];
    CMYK_DECLARE(s00);
    CMYK_DECLARE(s01); CMYK_DECLARE(s11); CMYK_DECLARE(s21); CMYK_DECLARE(s31);
    CMYK_DECLARE(s02); CMYK_DECLARE(s12); CMYK_DECLARE(s22); CMYK_DECLARE(s32);
    CMYK_DECLARE(s03); CMYK_DECLARE(s13); CMYK_DECLARE(s23); CMYK_DECLARE(s33);
    CMYK_LOAD(s01, s0); CMYK_LOAD(s02, s0); CMYK_LOAD(s03, s0);
    CMYK_LOAD(s11, s1); CMYK_LOAD(s12, s1); CMYK_LOAD(s13, s1);
    CMYK_LOAD(s21, s2); CMYK_LOAD(s22, s2); CMYK_LOAD(s23, s2);
    CMYK_LOAD(s31, s3); CMYK_LOAD(s32, s3); CMYK_LOAD(s33, s3);

    /*
     * dst  |A|B|C|D|E|F|G|H|
     * src  | a | b | c | d |
     *
     * D = a*M(-1.25) + b*M(-.25) + c*M(.75) + d*M(1.75)
     * E = a*M(-1.75) + b*M(-.75) + c*M(.25) + d*M(1.25)
     *
     * MW0 = M(-1.25) = M( 1.25)
     * MW1 = M(- .25) = M(  .25)
     * MW2 = M(  .75) = M(- .75)
     * MW3 = M( 1.75) = M(-1.75)
     */

    /* Combine them vertically */
    CMYK_COMBINE(s01, s11, s21, s31); /* s21, s31 now unused */
    CMYK_COMBINE(s02, s12, s22, s32); /* s22, s32 now unused */
    CMYK_COMBINE(s03, s13, s23, s33); /* s23, s33 now unused */

    /* Now do the leading pixels. */
    CMYK_STORE3(d0,s01,s01,s01,s02); /* A */
    CMYK_STORE2(d0,s01,s01,s02,s03); /* B */
    CMYK_STORE3(d0,s01,s01,s02,s03); /* C */

    for (w = doubler->src_w-3; w > 0; w--)
    {
        CMYK_ASSIGN(s00, s01);
        CMYK_ASSIGN(s01, s02);
        CMYK_ASSIGN(s02, s03);
        CMYK_LOAD(s03, s0); CMYK_LOAD(s13, s1); CMYK_LOAD(s23, s2); CMYK_LOAD(s33, s3);
        CMYK_COMBINE(s03, s13, s23, s33); /* s23, s33 now unused */
        CMYK_STORE2(d0,s00,s01,s02,s03); /* D */
        CMYK_STORE3(d0,s00,s01,s02,s03); /* E */
    }

    /* Trailing pixels */
    CMYK_STORE2(d0,s01,s02,s03,s03); /* F */
    CMYK_STORE3(d0,s01,s02,s03,s03); /* G */
    CMYK_STORE2(d0,s02,s03,s03,s03); /* H */

    return 1;
}
#undef CMYK_DECLARE
#undef CMYK_LOAD
#undef CMYK_COMBINE
#undef CMYK_ASSIGN
#undef CMYK_STORE2
#undef CMYK_STORE3

#undef MW0
#undef MW1
#undef MW2
#undef MW3
#undef COMBINE
#undef COMBINE2
#undef COMBINE3

#undef WEIGHT_SCALE
#undef WEIGHT_SHIFT
#undef WEIGHT_ROUND

static int
quad_near1(uint8_t       ** ipa_restrict dsts,
           const uint8_t ** ipa_restrict srcs,
           ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    const uint8_t * ipa_restrict s0 = srcs[0];

    for (w = doubler->src_w; w > 0; w--)
    {
        d0[0] = d0[1] = d0[2] = d0[3] =
            d1[0] = d1[1] = d1[2] = d1[3] =
            d2[0] = d2[1] = d2[2] = d2[3] =
            d3[0] = d3[1] = d3[2] = d3[3] = *s0++;
        d0 += 4;
        d1 += 4;
        d2 += 4;
        d3 += 4;
    }

    return 4;
}

static int
quad_near3(uint8_t       ** ipa_restrict dsts,
           const uint8_t ** ipa_restrict srcs,
           ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    const uint8_t * ipa_restrict s0 = srcs[0];

    for (w = doubler->src_w; w > 0; w--)
    {
        d0[0] = d0[3] = d0[6] = d0[9] =
            d1[0] = d1[3] = d1[6] = d1[9] =
            d2[0] = d2[3] = d2[6] = d2[9] =
            d3[0] = d3[3] = d3[6] = d3[9] = *s0++;
        d0[1] = d0[4] = d0[7] = d0[10] =
            d1[1] = d1[4] = d1[7] = d1[10] =
            d2[1] = d2[4] = d2[7] = d2[10] =
            d3[1] = d3[4] = d3[7] = d3[10] = *s0++;
        d0[2] = d0[5] = d0[8] = d0[11] =
            d1[2] = d1[5] = d1[8] = d1[11] =
            d2[2] = d2[5] = d2[8] = d2[11] =
            d3[2] = d3[5] = d3[8] = d3[11] = *s0++;
        d0 += 12;
        d1 += 12;
        d2 += 12;
        d3 += 12;
    }

    return 4;
}

static int
quad_near4(uint8_t       ** ipa_restrict dsts,
           const uint8_t ** ipa_restrict srcs,
           ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    const uint8_t * ipa_restrict s0 = srcs[0];

    for (w = doubler->src_w; w > 0; w--)
    {
        d0[0] = d0[4] = d0[8] = d0[12] =
            d1[0] = d1[4] = d1[8] = d1[12] =
            d2[0] = d2[4] = d2[8] = d2[12] =
            d3[0] = d3[4] = d3[8] = d3[12] = *s0++;
        d0[1] = d0[5] = d0[9] = d0[13] =
            d1[1] = d1[5] = d1[9] = d1[13] =
            d2[1] = d2[5] = d2[9] = d2[13] =
            d3[1] = d3[5] = d3[9] = d3[13] = *s0++;
        d0[2] = d0[6] = d0[10] = d0[14] =
            d1[2] = d1[6] = d1[10] = d1[14] =
            d2[2] = d2[6] = d2[10] = d2[14] =
            d3[2] = d3[6] = d3[10] = d3[14] = *s0++;
        d0[3] = d0[7] = d0[11] = d0[15] =
            d1[3] = d1[7] = d1[11] = d1[15] =
            d2[3] = d2[7] = d2[11] = d2[15] =
            d3[3] = d3[7] = d3[11] = d3[15] = *s0++;
        d0 += 16;
        d1 += 16;
        d2 += 16;
        d3 += 16;
    }

    return 4;
}

static int
quad_near(uint8_t       ** ipa_restrict dsts,
          const uint8_t ** ipa_restrict srcs,
          ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    const uint8_t * ipa_restrict s0 = srcs[0];
    uint32_t channels = doubler->channels;

    for (w = doubler->src_w; w > 0; w--)
    {
        uint32_t j = channels;
        do {
            d0[0] = d0[channels] = d0[channels*2] = d0[channels*3] =
                d1[0] = d1[channels] = d1[channels*2] = d1[channels*3] =
                d2[0] = d2[channels] = d2[channels*2] = d2[channels*3] =
                d3[0] = d3[channels] = d3[channels*2] = d3[channels*3] = *s0++;
            d0++;
            d1++;
            d2++;
            d3++;
            j--;
        } while (j != 0);
        d0 += channels*3;
        d1 += channels*3;
        d2 += channels*3;
        d3 += channels*3;
    }

    return 4;
}

/* Simple 15/16 + 1/16 then 5/8, 3/8 interpolation */

#define COMBINE(A,B,C,D,L,R) { A = 15*L+R; B = 10*L+6*R; C = 6*L+10*R; D = L+15*R; }
static int
quad_interp1(uint8_t       ** ipa_restrict dsts,
             const uint8_t ** ipa_restrict srcs,
             ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    int t = *s0++;
    int b = *s1++;
    int v00, v01, v02, v03;

    /* Leading pixels */
    COMBINE(v00, v01, v02, v03, t, b);
    d0[0] = d0[1] = (v00+8)>>4;
    d1[0] = d1[1] = (v01+8)>>4;
    d2[0] = d2[1] = (v02+8)>>4;
    d3[0] = d3[1] = (v03+8)>>4;
    d0 += 2;
    d1 += 2;
    d2 += 2;
    d3 += 2;

    for (w = doubler->src_w-1; w > 0; w--)
    {
        int v10, v11, v12, v13;
        int a, b, c, d;
        t = *s0++;
        b = *s1++;
        COMBINE(v10, v11, v12, v13, t, b);
        COMBINE(a, b, c, d, v00, v10);
        d0[0] = (a+128)>>8; d0[1] = (b+128)>>8; d0[2] = (c+128)>>8; d0[3] = (d+128)>>8;
        COMBINE(a, b, c, d, v01, v11);
        d1[0] = (a+128)>>8; d1[1] = (b+128)>>8; d1[2] = (c+128)>>8; d1[3] = (d+128)>>8;
        COMBINE(a, b, c, d, v02, v12);
        d2[0] = (a+128)>>8; d2[1] = (b+128)>>8; d2[2] = (c+128)>>8; d2[3] = (d+128)>>8;
        COMBINE(a, b, c, d, v03, v13);
        d3[0] = (a+128)>>8; d3[1] = (b+128)>>8; d3[2] = (c+128)>>8; d3[3] = (d+128)>>8;
        d0 += 4;
        d1 += 4;
        d2 += 4;
        d3 += 4;
        v00 = v10;
        v01 = v11;
        v02 = v12;
        v03 = v13;
    }

    /* Trailing pixels */
    d0[0] = d0[1] = (v00+8)>>4;
    d1[0] = d1[1] = (v01+8)>>4;
    d2[0] = d2[1] = (v02+8)>>4;
    d3[0] = d3[1] = (v03+8)>>4;

    return 4;
}

static int
quad_interp1_top(uint8_t       ** ipa_restrict dsts,
                 const uint8_t ** ipa_restrict srcs,
                 ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s = srcs[0];
    int l = *s++;
    int v0, v1, v2, v3;

    /* Leading pixels */
    d0[0] = l;
    d1[1] = l;
    d0 += 2;
    d1 += 2;

    for (w = doubler->src_w-1; w > 0; w--)
    {
        int r = *s++;
        COMBINE(v0, v1, v2, v3, l, r);
        d0[0] = d1[0] = (v0+8)>>4;
        d0[1] = d1[1] = (v1+8)>>4;
        d0[2] = d1[2] = (v2+8)>>4;
        d0[3] = d1[3] = (v3+8)>>4;
        d0 += 4;
        d1 += 4;
        l = r;
    }

    /* Trailing pixels */
    d0[0] = d0[1] = l;
    d1[0] = d1[1] = l;

    return 2;
}

static int
quad_interp3(uint8_t       ** ipa_restrict dsts,
             const uint8_t ** ipa_restrict srcs,
             ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    int t_r = *s0++;
    int t_g = *s0++;
    int t_b = *s0++;
    int b_r = *s1++;
    int b_g = *s1++;
    int b_b = *s1++;
    int v00_r, v00_g, v00_b, v01_r, v01_g, v01_b, v02_r, v02_g, v02_b, v03_r, v03_g, v03_b;

    /* Leading pixels */
    COMBINE(v00_r, v01_r, v02_r, v03_r, t_r, b_r);
    d0[0] = d0[3] = (v00_r+8)>>4;
    d1[0] = d1[3] = (v01_r+8)>>4;
    d2[0] = d2[3] = (v02_r+8)>>4;
    d3[0] = d3[3] = (v03_r+8)>>4;
    COMBINE(v00_g, v01_g, v02_g, v03_g, t_g, b_g);
    d0[1] = d0[4] = (v00_g+8)>>4;
    d1[1] = d1[4] = (v01_g+8)>>4;
    d2[1] = d2[4] = (v02_g+8)>>4;
    d3[1] = d3[4] = (v03_g+8)>>4;
    COMBINE(v00_b, v01_b, v02_b, v03_b, t_b, b_b);
    d0[2] = d0[5] = (v00_b+8)>>4;
    d1[2] = d1[5] = (v01_b+8)>>4;
    d2[2] = d2[5] = (v02_b+8)>>4;
    d3[2] = d3[5] = (v03_b+8)>>4;
    d0 += 6;
    d1 += 6;
    d2 += 6;
    d3 += 6;

    for (w = doubler->src_w-1; w > 0; w--)
    {
        int v10_r, v10_g, v10_b, v11_r, v11_g, v11_b, v12_r, v12_g, v12_b, v13_r, v13_g, v13_b;
        int a, b, c, d;
        t_r = *s0++;
        t_g = *s0++;
        t_b = *s0++;
        b_r = *s1++;
        b_g = *s1++;
        b_b = *s1++;
        COMBINE(v10_r, v11_r, v12_r, v13_r, t_r, b_r);
        COMBINE(v10_g, v11_g, v12_g, v13_g, t_g, b_g);
        COMBINE(v10_b, v11_b, v12_b, v13_b, t_b, b_b);
        COMBINE(a, b, c, d, v00_r, v10_r);
        d0[0] = (a+128)>>8; d0[3] = (b+128)>>8; d0[6] = (c+128)>>8; d0[9] = (d+128)>>8;
        COMBINE(a, b, c, d, v00_g, v10_g);
        d0[1] = (a+128)>>8; d0[4] = (b+128)>>8; d0[7] = (c+128)>>8; d0[10] = (d+128)>>8;
        COMBINE(a, b, c, d, v00_b, v10_b);
        d0[2] = (a+128)>>8; d0[5] = (b+128)>>8; d0[8] = (c+128)>>8; d0[11] = (d+128)>>8;
        COMBINE(a, b, c, d, v01_r, v11_r);
        d1[0] = (a+128)>>8; d1[3] = (b+128)>>8; d1[6] = (c+128)>>8; d1[9] = (d+128)>>8;
        COMBINE(a, b, c, d, v01_g, v11_g);
        d1[1] = (a+128)>>8; d1[4] = (b+128)>>8; d1[7] = (c+128)>>8; d1[10] = (d+128)>>8;
        COMBINE(a, b, c, d, v01_b, v11_b);
        d1[2] = (a+128)>>8; d1[5] = (b+128)>>8; d1[8] = (c+128)>>8; d1[11] = (d+128)>>8;
        COMBINE(a, b, c, d, v02_r, v12_r);
        d2[0] = (a+128)>>8; d2[3] = (b+128)>>8; d2[6] = (c+128)>>8; d2[9] = (d+128)>>8;
        COMBINE(a, b, c, d, v02_g, v12_g);
        d2[1] = (a+128)>>8; d2[4] = (b+128)>>8; d2[7] = (c+128)>>8; d2[10] = (d+128)>>8;
        COMBINE(a, b, c, d, v02_b, v12_b);
        d2[2] = (a+128)>>8; d2[5] = (b+128)>>8; d2[8] = (c+128)>>8; d2[11] = (d+128)>>8;
        COMBINE(a, b, c, d, v03_r, v13_r);
        d3[0] = (a+128)>>8; d3[3] = (b+128)>>8; d3[6] = (c+128)>>8; d3[9] = (d+128)>>8;
        COMBINE(a, b, c, d, v03_g, v13_g);
        d3[1] = (a+128)>>8; d3[4] = (b+128)>>8; d3[7] = (c+128)>>8; d3[10] = (d+128)>>8;
        COMBINE(a, b, c, d, v03_b, v13_b);
        d3[2] = (a+128)>>8; d3[5] = (b+128)>>8; d3[8] = (c+128)>>8; d3[11] = (d+128)>>8;
        d0 += 12;
        d1 += 12;
        d2 += 12;
        d3 += 12;
        v00_r = v10_r; v00_g = v10_g; v00_b = v10_b;
        v01_r = v11_r; v01_g = v11_g; v01_b = v11_b;
        v02_r = v12_r; v02_g = v12_g; v02_b = v12_b;
        v03_r = v13_r; v03_g = v13_g; v03_b = v13_b;
    }

    /* Trailing pixels */
    d0[0] = d0[3] = (v00_r+8)>>4;
    d0[1] = d0[4] = (v00_g+8)>>4;
    d0[2] = d0[5] = (v00_b+8)>>4;
    d1[0] = d1[3] = (v01_r+8)>>4;
    d1[1] = d1[4] = (v01_g+8)>>4;
    d1[2] = d1[5] = (v01_b+8)>>4;
    d2[0] = d2[3] = (v02_r+8)>>4;
    d2[1] = d2[4] = (v02_g+8)>>4;
    d2[2] = d2[5] = (v02_b+8)>>4;
    d3[0] = d3[3] = (v03_r+8)>>4;
    d3[1] = d3[4] = (v03_g+8)>>4;
    d3[2] = d3[5] = (v03_b+8)>>4;

    return 4;
}

static int
quad_interp3_top(uint8_t       ** ipa_restrict dsts,
                 const uint8_t ** ipa_restrict srcs,
                 ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    int l_r = *s0++;
    int l_g = *s0++;
    int l_b = *s0++;

    /* Leading pixels */
    d0[0] = d0[3] = d1[0] = d1[3] = l_r;
    d0[1] = d0[4] = d1[1] = d1[4] = l_g;
    d0[2] = d0[5] = d1[2] = d1[5] = l_b;
    d0 += 6;
    d1 += 6;

    for (w = doubler->src_w-1; w > 0; w--)
    {
        int r_r, r_g, r_b;
        int a, b, c, d;
        r_r = *s0++;
        r_g = *s0++;
        r_b = *s0++;
        COMBINE(a, b, c, d, l_r, r_r);
        d0[0] = d1[0] = (a+8)>>4; d0[3] = d1[3] = (b+8)>>4; d0[6] = d1[6] = (c+8)>>4; d0[9] = d1[9] = (d+8)>>4;
        COMBINE(a, b, c, d, l_g, r_g);
        d0[1] = d1[1] = (a+8)>>4; d0[4] = d1[4] = (b+8)>>4; d0[7] = d1[7] = (c+8)>>4; d0[10] = d1[10] = (d+8)>>4;
        COMBINE(a, b, c, d, l_b, r_b);
        d0[2] = d1[2] = (a+8)>>4; d0[5] = d1[5] = (b+8)>>4; d0[8] = d1[8] = (c+8)>>4; d0[11] = d1[11] = (d+8)>>4;
        d0 += 12;
        d1 += 12;
        l_r = r_r; l_g = r_g; l_b = r_b;
    }

    /* Trailing pixels */
    d0[0] = d0[3] = d1[0] = d1[3] = l_r;
    d0[1] = d0[4] = d1[1] = d1[4] = l_g;
    d0[2] = d0[5] = d1[2] = d1[5] = l_b;

    return 2;
}

static int
quad_interp4(uint8_t       ** ipa_restrict dsts,
             const uint8_t ** ipa_restrict srcs,
             ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    int t_r = *s0++;
    int t_g = *s0++;
    int t_b = *s0++;
    int t_k = *s0++;
    int b_r = *s1++;
    int b_g = *s1++;
    int b_b = *s1++;
    int b_k = *s1++;
    int v00_r, v00_g, v00_b, v00_k, v01_r, v01_g, v01_b, v01_k, v02_r, v02_g, v02_b, v02_k, v03_r, v03_g, v03_b, v03_k;

    /* Leading pixels */
    COMBINE(v00_r, v01_r, v02_r, v03_r, t_r, b_r);
    d0[0] = d0[4] = (v00_r+8)>>4;
    d1[0] = d1[4] = (v01_r+8)>>4;
    d2[0] = d2[4] = (v02_r+8)>>4;
    d3[0] = d3[4] = (v03_r+8)>>4;
    COMBINE(v00_g, v01_g, v02_g, v03_g, t_g, b_g);
    d0[1] = d0[5] = (v00_g+8)>>4;
    d1[1] = d1[5] = (v01_g+8)>>4;
    d2[1] = d2[5] = (v02_g+8)>>4;
    d3[1] = d3[5] = (v03_g+8)>>4;
    COMBINE(v00_b, v01_b, v02_b, v03_b, t_b, b_b);
    d0[2] = d0[6] = (v00_b+8)>>4;
    d1[2] = d1[6] = (v01_b+8)>>4;
    d2[2] = d2[6] = (v02_b+8)>>4;
    d3[2] = d3[6] = (v03_b+8)>>4;
    COMBINE(v00_k, v01_k, v02_k, v03_k, t_k, b_k);
    d0[3] = d0[7] = (v00_k+8)>>4;
    d1[3] = d1[7] = (v01_k+8)>>4;
    d2[3] = d2[7] = (v02_k+8)>>4;
    d3[3] = d3[7] = (v03_k+8)>>4;
    d0 += 8;
    d1 += 8;
    d2 += 8;
    d3 += 8;

    for (w = doubler->src_w-1; w > 0; w--)
    {
        int v10_r, v10_g, v10_b, v10_k, v11_r, v11_g, v11_b, v11_k, v12_r, v12_g, v12_b, v12_k, v13_r, v13_g, v13_b, v13_k;
        int a, b, c, d;
        t_r = *s0++;
        t_g = *s0++;
        t_b = *s0++;
        t_k = *s0++;
        b_r = *s1++;
        b_g = *s1++;
        b_b = *s1++;
        b_k = *s1++;
        COMBINE(v10_r, v11_r, v12_r, v13_r, t_r, b_r);
        COMBINE(v10_g, v11_g, v12_g, v13_g, t_g, b_g);
        COMBINE(v10_b, v11_b, v12_b, v13_b, t_b, b_b);
        COMBINE(v10_k, v11_k, v12_k, v13_k, t_k, b_k);
        COMBINE(a, b, c, d, v00_r, v10_r);
        d0[0] = (a+128)>>8; d0[4] = (b+128)>>8; d0[8] = (c+128)>>8; d0[12] = (d+128)>>8;
        COMBINE(a, b, c, d, v00_g, v10_g);
        d0[1] = (a+128)>>8; d0[5] = (b+128)>>8; d0[9] = (c+128)>>8; d0[13] = (d+128)>>8;
        COMBINE(a, b, c, d, v00_b, v10_b);
        d0[2] = (a+128)>>8; d0[6] = (b+128)>>8; d0[10] = (c+128)>>8; d0[14] = (d+128)>>8;
        COMBINE(a, b, c, d, v00_k, v10_k);
        d0[3] = (a+128)>>8; d0[7] = (b+128)>>8; d0[11] = (c+128)>>8; d0[15] = (d+128)>>8;
        COMBINE(a, b, c, d, v01_r, v11_r);
        d1[0] = (a+128)>>8; d1[4] = (b+128)>>8; d1[8] = (c+128)>>8; d1[12] = (d+128)>>8;
        COMBINE(a, b, c, d, v01_g, v11_g);
        d1[1] = (a+128)>>8; d1[5] = (b+128)>>8; d1[9] = (c+128)>>8; d1[13] = (d+128)>>8;
        COMBINE(a, b, c, d, v01_b, v11_b);
        d1[2] = (a+128)>>8; d1[6] = (b+128)>>8; d1[10] = (c+128)>>8; d1[14] = (d+128)>>8;
        COMBINE(a, b, c, d, v01_k, v11_k);
        d1[3] = (a+128)>>8; d1[7] = (b+128)>>8; d1[11] = (c+128)>>8; d1[15] = (d+128)>>8;
        COMBINE(a, b, c, d, v02_r, v12_r);
        d2[0] = (a+128)>>8; d2[4] = (b+128)>>8; d2[8] = (c+128)>>8; d2[12] = (d+128)>>8;
        COMBINE(a, b, c, d, v02_g, v12_g);
        d2[1] = (a+128)>>8; d2[5] = (b+128)>>8; d2[9] = (c+128)>>8; d2[13] = (d+128)>>8;
        COMBINE(a, b, c, d, v02_b, v12_b);
        d2[2] = (a+128)>>8; d2[6] = (b+128)>>8; d2[10] = (c+128)>>8; d2[14] = (d+128)>>8;
        COMBINE(a, b, c, d, v02_k, v12_k);
        d2[3] = (a+128)>>8; d2[7] = (b+128)>>8; d2[11] = (c+128)>>8; d2[15] = (d+128)>>8;
        COMBINE(a, b, c, d, v03_r, v13_r);
        d3[0] = (a+128)>>8; d3[4] = (b+128)>>8; d3[8] = (c+128)>>8; d3[12] = (d+128)>>8;
        COMBINE(a, b, c, d, v03_g, v13_g);
        d3[1] = (a+128)>>8; d3[5] = (b+128)>>8; d3[9] = (c+128)>>8; d3[13] = (d+128)>>8;
        COMBINE(a, b, c, d, v03_b, v13_b);
        d3[2] = (a+128)>>8; d3[6] = (b+128)>>8; d3[10] = (c+128)>>8; d3[14] = (d+128)>>8;
        COMBINE(a, b, c, d, v03_k, v13_k);
        d3[3] = (a+128)>>8; d3[7] = (b+128)>>8; d3[11] = (c+128)>>8; d3[15] = (d+128)>>8;
        d0 += 16;
        d1 += 16;
        d2 += 16;
        d3 += 16;
        v00_r = v10_r; v00_g = v10_g; v00_b = v10_b; v00_k = v10_k;
        v01_r = v11_r; v01_g = v11_g; v01_b = v11_b; v01_k = v11_k;
        v02_r = v12_r; v02_g = v12_g; v02_b = v12_b; v02_k = v12_k;
        v03_r = v13_r; v03_g = v13_g; v03_b = v13_b; v03_k = v13_k;
    }

    /* Trailing pixels */
    d0[0] = d0[4] = (v00_r+8)>>4;
    d0[1] = d0[5] = (v00_g+8)>>4;
    d0[2] = d0[6] = (v00_b+8)>>4;
    d0[3] = d0[7] = (v00_k+8)>>4;
    d1[0] = d1[4] = (v01_r+8)>>4;
    d1[1] = d1[5] = (v01_g+8)>>4;
    d1[2] = d1[6] = (v01_b+8)>>4;
    d1[3] = d1[7] = (v01_k+8)>>4;
    d2[0] = d2[4] = (v02_r+8)>>4;
    d2[1] = d2[5] = (v02_g+8)>>4;
    d2[2] = d2[6] = (v02_b+8)>>4;
    d2[3] = d2[7] = (v02_k+8)>>4;
    d3[0] = d3[4] = (v03_r+8)>>4;
    d3[1] = d3[5] = (v03_g+8)>>4;
    d3[2] = d3[6] = (v03_b+8)>>4;
    d3[3] = d3[7] = (v03_k+8)>>4;

    return 4;
}

static int
quad_interp4_top(uint8_t       ** ipa_restrict dsts,
                 const uint8_t ** ipa_restrict srcs,
                 ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    int l_r = *s0++;
    int l_g = *s0++;
    int l_b = *s0++;
    int l_k = *s0++;

    /* Leading pixels */
    d0[0] = d0[4] = d1[0] = d1[4] = l_r;
    d0[1] = d0[5] = d1[1] = d1[5] = l_g;
    d0[2] = d0[6] = d1[2] = d1[6] = l_b;
    d0[3] = d0[7] = d1[3] = d1[7] = l_k;
    d0 += 8;
    d1 += 8;

    for (w = doubler->src_w-1; w > 0; w--)
    {
        int r_r, r_g, r_b, r_k;
        int a, b, c, d;
        r_r = *s0++;
        r_g = *s0++;
        r_b = *s0++;
        r_k = *s0++;
        COMBINE(a, b, c, d, l_r, r_r);
        d0[0] = d1[0] = (a+8)>>4; d0[4] = d1[4] = (b+8)>>4; d0[8] = d1[8] = (c+8)>>4; d0[12] = d1[12] = (d+8)>>4;
        COMBINE(a, b, c, d, l_g, r_g);
        d0[1] = d1[1] = (a+8)>>4; d0[5] = d1[5] = (b+8)>>4; d0[9] = d1[9] = (c+8)>>4; d0[13] = d1[13] = (d+8)>>4;
        COMBINE(a, b, c, d, l_b, r_b);
        d0[2] = d1[2] = (a+8)>>4; d0[6] = d1[6] = (b+8)>>4; d0[10] = d1[10] = (c+8)>>4; d0[14] = d1[14] = (d+8)>>4;
        COMBINE(a, b, c, d, l_k, r_k);
        d0[3] = d1[3] = (a+8)>>4; d0[7] = d1[7] = (b+8)>>4; d0[11] = d1[11] = (c+8)>>4; d0[15] = d1[15] = (d+8)>>4;
        d0 += 16;
        d1 += 16;
        l_r = r_r; l_g = r_g; l_b = r_b; l_k = r_k;
    }

    /* Trailing pixels */
    d0[0] = d0[4] = d1[0] = d1[4] = l_r;
    d0[1] = d0[5] = d1[1] = d1[5] = l_g;
    d0[2] = d0[6] = d1[2] = d1[6] = l_b;
    d0[3] = d0[7] = d1[3] = d1[7] = l_k;

    return 2;
}

#undef COMBINE

#define WEIGHT_SHIFT 10
#define MW0 (5)    /* 0.00531684 */
#define MW1 (881)  /* 0.859918 */
#define MW2 (143)  /* 0.139214 */
#define MW3 (-5)   /* -0.00444878 */
#define MW4 (-36)  /* -0.0352648 */
#define MW5 (685)  /* 0.669162 */
#define MW6 (402)  /* 0.39247 */
#define MW7 (-27)  /* -0.0263672 */

#define WEIGHT_SCALE (1<<WEIGHT_SHIFT)
#define WEIGHT_ROUND (1<<(WEIGHT_SHIFT-1))
#define COMBINE0(A,B,C,D) (A * MW0 + B * MW1 + C * MW2 + D * MW3)
#define COMBINE1(A,B,C,D) (A * MW4 + B * MW5 + C * MW6 + D * MW7)
#define COMBINE2(A,B,C,D) (A * MW7 + B * MW6 + C * MW5 + D * MW4)
#define COMBINE3(A,B,C,D) (A * MW3 + B * MW2 + C * MW1 + D * MW0)
#define COMBINE(a,b,c,d,s0,s1,s2,s3) \
do { a = COMBINE0(s0,s1,s2,s3); b = COMBINE1(s0,s1,s2,s3); c = COMBINE2(s0,s1,s2,s3); d = COMBINE3(s0,s1,s2,s3); } while (0)
#define COMBINE_HALF(a,b,s0,s1,s2,s3) \
do { a = COMBINE0(s0,s1,s2,s3); b = COMBINE1(s0,s1,s2,s3); } while (0)

#define RCLAMP(A) clamp((A + (WEIGHT_ROUND<<WEIGHT_SHIFT))>>(2*WEIGHT_SHIFT))

static int
quad_mitchell1(uint8_t       ** ipa_restrict dsts,
               const uint8_t ** ipa_restrict srcs,
               ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    const uint8_t * ipa_restrict s2 = srcs[2];
    const uint8_t * ipa_restrict s3 = srcs[3];
    int s00 = *s0++;
    int s01 = *s0++;
    int s02 = *s0++;
    int s10 = *s1++;
    int s11 = *s1++;
    int s12 = *s1++;
    int s20 = *s2++;
    int s21 = *s2++;
    int s22 = *s2++;
    int s30 = *s3++;
    int s31 = *s3++;
    int s32 = *s3++;
    int v00, v01, v02, v10, v11, v12, v20, v21, v22, v30, v31, v32;

    /* Combine them vertically */
    COMBINE(v00, v10, v20, v30, s00, s10, s20, s30);
    COMBINE(v01, v11, v21, v31, s01, s11, s21, s31);
    COMBINE(v02, v12, v22, v32, s02, s12, s22, s32);

    /* Now do the leading pixels. */
    d0[0] = RCLAMP(COMBINE2(v00,v00,v00,v01)); /* A */
    d0[1] = RCLAMP(COMBINE3(v00,v00,v00,v01)); /* B */
    d0[2] = RCLAMP(COMBINE0(v00,v00,v01,v02)); /* C */
    d0[3] = RCLAMP(COMBINE1(v00,v00,v01,v02)); /* D */
    d0[4] = RCLAMP(COMBINE2(v00,v00,v01,v02)); /* E */
    d0[5] = RCLAMP(COMBINE3(v00,v00,v01,v02)); /* F */
    d1[0] = RCLAMP(COMBINE2(v10,v10,v10,v11)); /* A */
    d1[1] = RCLAMP(COMBINE3(v10,v10,v10,v11)); /* B */
    d1[2] = RCLAMP(COMBINE0(v10,v10,v11,v12)); /* C */
    d1[3] = RCLAMP(COMBINE1(v10,v10,v11,v12)); /* D */
    d1[4] = RCLAMP(COMBINE2(v10,v10,v11,v12)); /* E */
    d1[5] = RCLAMP(COMBINE3(v10,v10,v11,v12)); /* F */
    d2[0] = RCLAMP(COMBINE2(v20,v20,v20,v21)); /* A */
    d2[1] = RCLAMP(COMBINE3(v20,v20,v20,v21)); /* B */
    d2[2] = RCLAMP(COMBINE0(v20,v20,v21,v22)); /* C */
    d2[3] = RCLAMP(COMBINE1(v20,v20,v21,v22)); /* D */
    d2[4] = RCLAMP(COMBINE2(v20,v20,v21,v22)); /* E */
    d2[5] = RCLAMP(COMBINE3(v20,v20,v21,v22)); /* F */
    d3[0] = RCLAMP(COMBINE2(v30,v30,v30,v31)); /* A */
    d3[1] = RCLAMP(COMBINE3(v30,v30,v30,v31)); /* B */
    d3[2] = RCLAMP(COMBINE0(v30,v30,v31,v32)); /* C */
    d3[3] = RCLAMP(COMBINE1(v30,v30,v31,v32)); /* D */
    d3[4] = RCLAMP(COMBINE2(v30,v30,v31,v32)); /* E */
    d3[5] = RCLAMP(COMBINE3(v30,v30,v31,v32)); /* F */
    d0 += 6; d1 += 6; d2 += 6; d3 += 6;

    for (w = doubler->src_w-3; w > 0; w--)
    {
        int s03 = *s0++;
        int s13 = *s1++;
        int s23 = *s2++;
        int s33 = *s3++;
        int v03, v13, v23, v33;
        COMBINE(v03, v13, v23, v33, s03, s13, s23, s33);
        d0[0] = RCLAMP(COMBINE0(v00,v01,v02,v03)); /* G */
        d0[1] = RCLAMP(COMBINE1(v00,v01,v02,v03)); /* H */
        d0[2] = RCLAMP(COMBINE2(v00,v01,v02,v03)); /* I */
        d0[3] = RCLAMP(COMBINE3(v00,v01,v02,v03)); /* J */
        d1[0] = RCLAMP(COMBINE0(v10,v11,v12,v13)); /* G */
        d1[1] = RCLAMP(COMBINE1(v10,v11,v12,v13)); /* H */
        d1[2] = RCLAMP(COMBINE2(v10,v11,v12,v13)); /* I */
        d1[3] = RCLAMP(COMBINE3(v10,v11,v12,v13)); /* J */
        d2[0] = RCLAMP(COMBINE0(v20,v21,v22,v23)); /* G */
        d2[1] = RCLAMP(COMBINE1(v20,v21,v22,v23)); /* H */
        d2[2] = RCLAMP(COMBINE2(v20,v21,v22,v23)); /* I */
        d2[3] = RCLAMP(COMBINE3(v20,v21,v22,v23)); /* J */
        d3[0] = RCLAMP(COMBINE0(v30,v31,v32,v33)); /* G */
        d3[1] = RCLAMP(COMBINE1(v30,v31,v32,v33)); /* H */
        d3[2] = RCLAMP(COMBINE2(v30,v31,v32,v33)); /* I */
        d3[3] = RCLAMP(COMBINE3(v30,v31,v32,v33)); /* J */
        d0 += 4;
        d1 += 4;
        d2 += 4;
        d3 += 4;
        v00 = v01; v01 = v02; v02 = v03;
        v10 = v11; v11 = v12; v12 = v13;
        v20 = v21; v21 = v22; v22 = v23;
        v30 = v31; v31 = v32; v32 = v33;
    }

    /* Trailing pixels */
    d0[0] = RCLAMP(COMBINE0(v00,v01,v02,v02)); /* A */
    d0[1] = RCLAMP(COMBINE1(v00,v01,v02,v02)); /* B */
    d0[2] = RCLAMP(COMBINE2(v00,v01,v02,v02)); /* C */
    d0[3] = RCLAMP(COMBINE3(v00,v01,v02,v02)); /* D */
    d0[4] = RCLAMP(COMBINE0(v01,v02,v02,v02)); /* E */
    d0[5] = RCLAMP(COMBINE1(v01,v02,v02,v02)); /* F */
    d1[0] = RCLAMP(COMBINE0(v10,v11,v12,v12)); /* A */
    d1[1] = RCLAMP(COMBINE1(v10,v11,v12,v12)); /* B */
    d1[2] = RCLAMP(COMBINE2(v10,v11,v12,v12)); /* C */
    d1[3] = RCLAMP(COMBINE3(v10,v11,v12,v12)); /* D */
    d1[4] = RCLAMP(COMBINE0(v11,v12,v12,v12)); /* E */
    d1[5] = RCLAMP(COMBINE1(v11,v12,v12,v12)); /* F */
    d2[0] = RCLAMP(COMBINE0(v20,v21,v22,v22)); /* A */
    d2[1] = RCLAMP(COMBINE1(v20,v21,v22,v22)); /* B */
    d2[2] = RCLAMP(COMBINE2(v20,v21,v22,v22)); /* C */
    d2[3] = RCLAMP(COMBINE3(v20,v21,v22,v22)); /* D */
    d2[4] = RCLAMP(COMBINE0(v21,v22,v22,v22)); /* E */
    d2[5] = RCLAMP(COMBINE1(v21,v22,v22,v22)); /* F */
    d3[0] = RCLAMP(COMBINE0(v30,v31,v32,v32)); /* A */
    d3[1] = RCLAMP(COMBINE1(v30,v31,v32,v32)); /* B */
    d3[2] = RCLAMP(COMBINE2(v30,v31,v32,v32)); /* C */
    d3[3] = RCLAMP(COMBINE3(v30,v31,v32,v32)); /* D */
    d3[4] = RCLAMP(COMBINE0(v31,v32,v32,v32)); /* E */
    d3[5] = RCLAMP(COMBINE1(v31,v32,v32,v32)); /* F */

    return 4;
}

static int
quad_mitchell1_top(uint8_t       ** ipa_restrict dsts,
                   const uint8_t ** ipa_restrict srcs,
                   ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    const uint8_t * ipa_restrict s2 = srcs[2];
    const uint8_t * ipa_restrict s3 = srcs[3];
    int s00 = *s0++;
    int s01 = *s0++;
    int s02 = *s0++;
    int s10 = *s1++;
    int s11 = *s1++;
    int s12 = *s1++;
    int s20 = *s2++;
    int s21 = *s2++;
    int s22 = *s2++;
    int s30 = *s3++;
    int s31 = *s3++;
    int s32 = *s3++;
    int v00, v01, v02, v10, v11, v12;

    /* Combine them vertically */
    COMBINE_HALF(v00, v10, s00, s10, s20, s30);
    COMBINE_HALF(v01, v11, s01, s11, s21, s31);
    COMBINE_HALF(v02, v12, s02, s12, s22, s32);

    /* Now do the leading pixels. */
    d0[0] = RCLAMP(COMBINE2(v00,v00,v00,v01)); /* A */
    d0[1] = RCLAMP(COMBINE3(v00,v00,v00,v01)); /* B */
    d0[2] = RCLAMP(COMBINE0(v00,v00,v01,v02)); /* C */
    d0[3] = RCLAMP(COMBINE1(v00,v00,v01,v02)); /* D */
    d0[4] = RCLAMP(COMBINE2(v00,v00,v01,v02)); /* E */
    d0[5] = RCLAMP(COMBINE3(v00,v00,v01,v02)); /* F */
    d1[0] = RCLAMP(COMBINE2(v10,v10,v10,v11)); /* A */
    d1[1] = RCLAMP(COMBINE3(v10,v10,v10,v11)); /* B */
    d1[2] = RCLAMP(COMBINE0(v10,v10,v11,v12)); /* C */
    d1[3] = RCLAMP(COMBINE1(v10,v10,v11,v12)); /* D */
    d1[4] = RCLAMP(COMBINE2(v10,v10,v11,v12)); /* E */
    d1[5] = RCLAMP(COMBINE3(v10,v10,v11,v12)); /* F */
    d0 += 6; d1 += 6;

    for (w = doubler->src_w-3; w > 0; w--)
    {
        int s03 = *s0++;
        int s13 = *s1++;
        int s23 = *s2++;
        int s33 = *s3++;
        int v03, v13;
        COMBINE_HALF(v03, v13, s03, s13, s23, s33);
        d0[0] = RCLAMP(COMBINE0(v00,v01,v02,v03)); /* G */
        d0[1] = RCLAMP(COMBINE1(v00,v01,v02,v03)); /* H */
        d0[2] = RCLAMP(COMBINE2(v00,v01,v02,v03)); /* I */
        d0[3] = RCLAMP(COMBINE3(v00,v01,v02,v03)); /* J */
        d1[0] = RCLAMP(COMBINE0(v10,v11,v12,v13)); /* G */
        d1[1] = RCLAMP(COMBINE1(v10,v11,v12,v13)); /* H */
        d1[2] = RCLAMP(COMBINE2(v10,v11,v12,v13)); /* I */
        d1[3] = RCLAMP(COMBINE3(v10,v11,v12,v13)); /* J */
        d0 += 4;
        d1 += 4;
        v00 = v01; v01 = v02; v02 = v03;
        v10 = v11; v11 = v12; v12 = v13;
    }

    /* Trailing pixels */
    d0[0] = RCLAMP(COMBINE0(v00,v01,v02,v02)); /* A */
    d0[1] = RCLAMP(COMBINE1(v00,v01,v02,v02)); /* B */
    d0[2] = RCLAMP(COMBINE2(v00,v01,v02,v02)); /* C */
    d0[3] = RCLAMP(COMBINE3(v00,v01,v02,v02)); /* D */
    d0[4] = RCLAMP(COMBINE0(v01,v02,v02,v02)); /* E */
    d0[5] = RCLAMP(COMBINE1(v01,v02,v02,v02)); /* F */
    d1[0] = RCLAMP(COMBINE0(v10,v11,v12,v12)); /* A */
    d1[1] = RCLAMP(COMBINE1(v10,v11,v12,v12)); /* B */
    d1[2] = RCLAMP(COMBINE2(v10,v11,v12,v12)); /* C */
    d1[3] = RCLAMP(COMBINE3(v10,v11,v12,v12)); /* D */
    d1[4] = RCLAMP(COMBINE0(v11,v12,v12,v12)); /* E */
    d1[5] = RCLAMP(COMBINE1(v11,v12,v12,v12)); /* F */

    return 2;
}

#define DECLARE(T) int T##_r, T##_g, T##_b
#define LOAD(T,S) do { T##_r = *(S++); T##_g = *(S++); T##_b = *(S++); } while (0)
#define COMBINE_RGB(a,b,c,d,s0,s1,s2,s3) \
do { COMBINE(a##_r,b##_r,c##_r,d##_r,s0##_r,s1##_r,s2##_r,s3##_r); \
     COMBINE(a##_g,b##_g,c##_g,d##_g,s0##_g,s1##_g,s2##_g,s3##_g); \
     COMBINE(a##_b,b##_b,c##_b,d##_b,s0##_b,s1##_b,s2##_b,s3##_b); \
} while (0)
#define COMBINE0_RGB(D,V0,V1,V2,V3) \
do { D[0] = RCLAMP(COMBINE0(V0##_r,V1##_r,V2##_r,V3##_r));\
     D[1] = RCLAMP(COMBINE0(V0##_g,V1##_g,V2##_g,V3##_g));\
     D[2] = RCLAMP(COMBINE0(V0##_b,V1##_b,V2##_b,V3##_b));\
} while (0)
#define COMBINE1_RGB(D,V0,V1,V2,V3) \
do { D[0] = RCLAMP(COMBINE1(V0##_r,V1##_r,V2##_r,V3##_r));\
     D[1] = RCLAMP(COMBINE1(V0##_g,V1##_g,V2##_g,V3##_g));\
     D[2] = RCLAMP(COMBINE1(V0##_b,V1##_b,V2##_b,V3##_b));\
} while (0)
#define COMBINE2_RGB(D,V0,V1,V2,V3) \
do { D[0] = RCLAMP(COMBINE2(V0##_r,V1##_r,V2##_r,V3##_r));\
     D[1] = RCLAMP(COMBINE2(V0##_g,V1##_g,V2##_g,V3##_g));\
     D[2] = RCLAMP(COMBINE2(V0##_b,V1##_b,V2##_b,V3##_b));\
} while (0)
#define COMBINE3_RGB(D,V0,V1,V2,V3) \
do { D[0] = RCLAMP(COMBINE3(V0##_r,V1##_r,V2##_r,V3##_r));\
     D[1] = RCLAMP(COMBINE3(V0##_g,V1##_g,V2##_g,V3##_g));\
     D[2] = RCLAMP(COMBINE3(V0##_b,V1##_b,V2##_b,V3##_b));\
} while (0)
#define ASSIGN(D,S) do { D##_r = S##_r; D##_g = S##_g; D##_b = S##_b; } while (0)
static int
quad_mitchell3(uint8_t       ** ipa_restrict dsts,
               const uint8_t ** ipa_restrict srcs,
               ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    const uint8_t * ipa_restrict s2 = srcs[2];
    const uint8_t * ipa_restrict s3 = srcs[3];
    DECLARE(s00); DECLARE(s01); DECLARE(s02);
    DECLARE(s10); DECLARE(s11); DECLARE(s12);
    DECLARE(s20); DECLARE(s21); DECLARE(s22);
    DECLARE(s30); DECLARE(s31); DECLARE(s32);
    DECLARE(v00); DECLARE(v01); DECLARE(v02);
    DECLARE(v10); DECLARE(v11); DECLARE(v12);
    DECLARE(v20); DECLARE(v21); DECLARE(v22);
    DECLARE(v30); DECLARE(v31); DECLARE(v32);

    /* Combine them vertically */
    LOAD(s00, s0); LOAD(s01, s0); LOAD(s02, s0);
    LOAD(s10, s1); LOAD(s11, s1); LOAD(s12, s1);
    LOAD(s20, s2); LOAD(s21, s2); LOAD(s22, s2);
    LOAD(s30, s3); LOAD(s31, s3); LOAD(s32, s3);
    COMBINE_RGB(v00, v10, v20, v30, s00, s10, s20, s30);
    COMBINE_RGB(v01, v11, v21, v31, s01, s11, s21, s31);
    COMBINE_RGB(v02, v12, v22, v32, s02, s12, s22, s32);

    /* Now do the leading pixels. */
    COMBINE2_RGB(d0,v00,v00,v00,v01); d0 += 3;
    COMBINE3_RGB(d0,v00,v00,v00,v01); d0 += 3;
    COMBINE0_RGB(d0,v00,v00,v01,v02); d0 += 3;
    COMBINE1_RGB(d0,v00,v00,v01,v02); d0 += 3;
    COMBINE2_RGB(d0,v00,v00,v01,v02); d0 += 3;
    COMBINE3_RGB(d0,v00,v00,v01,v02); d0 += 3;
    COMBINE2_RGB(d1,v10,v10,v10,v11); d1 += 3;
    COMBINE3_RGB(d1,v10,v10,v10,v11); d1 += 3;
    COMBINE0_RGB(d1,v10,v10,v11,v12); d1 += 3;
    COMBINE1_RGB(d1,v10,v10,v11,v12); d1 += 3;
    COMBINE2_RGB(d1,v10,v10,v11,v12); d1 += 3;
    COMBINE3_RGB(d1,v10,v10,v11,v12); d1 += 3;
    COMBINE2_RGB(d2,v20,v20,v20,v21); d2 += 3;
    COMBINE3_RGB(d2,v20,v20,v20,v21); d2 += 3;
    COMBINE0_RGB(d2,v20,v20,v21,v22); d2 += 3;
    COMBINE1_RGB(d2,v20,v20,v21,v22); d2 += 3;
    COMBINE2_RGB(d2,v20,v20,v21,v22); d2 += 3;
    COMBINE3_RGB(d2,v20,v20,v21,v22); d2 += 3;
    COMBINE2_RGB(d3,v30,v30,v30,v31); d3 += 3;
    COMBINE3_RGB(d3,v30,v30,v30,v31); d3 += 3;
    COMBINE0_RGB(d3,v30,v30,v31,v32); d3 += 3;
    COMBINE1_RGB(d3,v30,v30,v31,v32); d3 += 3;
    COMBINE2_RGB(d3,v30,v30,v31,v32); d3 += 3;
    COMBINE3_RGB(d3,v30,v30,v31,v32); d3 += 3;

    for (w = doubler->src_w-3; w > 0; w--)
    {
        DECLARE(s03); DECLARE(s13); DECLARE(s23); DECLARE(s33);
        DECLARE(v03); DECLARE(v13); DECLARE(v23); DECLARE(v33);
        LOAD(s03, s0); LOAD(s13, s1); LOAD(s23, s2); LOAD(s33, s3);
        COMBINE_RGB(v03, v13, v23, v33, s03, s13, s23, s33);
        COMBINE0_RGB(d0,v00,v01,v02,v03); d0 += 3;
        COMBINE1_RGB(d0,v00,v01,v02,v03); d0 += 3;
        COMBINE2_RGB(d0,v00,v01,v02,v03); d0 += 3;
        COMBINE3_RGB(d0,v00,v01,v02,v03); d0 += 3;
        COMBINE0_RGB(d1,v10,v11,v12,v13); d1 += 3;
        COMBINE1_RGB(d1,v10,v11,v12,v13); d1 += 3;
        COMBINE2_RGB(d1,v10,v11,v12,v13); d1 += 3;
        COMBINE3_RGB(d1,v10,v11,v12,v13); d1 += 3;
        COMBINE0_RGB(d2,v20,v21,v22,v23); d2 += 3;
        COMBINE1_RGB(d2,v20,v21,v22,v23); d2 += 3;
        COMBINE2_RGB(d2,v20,v21,v22,v23); d2 += 3;
        COMBINE3_RGB(d2,v20,v21,v22,v23); d2 += 3;
        COMBINE0_RGB(d3,v30,v31,v32,v33); d3 += 3;
        COMBINE1_RGB(d3,v30,v31,v32,v33); d3 += 3;
        COMBINE2_RGB(d3,v30,v31,v32,v33); d3 += 3;
        COMBINE3_RGB(d3,v30,v31,v32,v33); d3 += 3;
        ASSIGN(v00, v01); ASSIGN(v01, v02); ASSIGN(v02, v03);
        ASSIGN(v10, v11); ASSIGN(v11, v12); ASSIGN(v12, v13);
        ASSIGN(v20, v21); ASSIGN(v21, v22); ASSIGN(v22, v23);
        ASSIGN(v30, v31); ASSIGN(v31, v32); ASSIGN(v32, v33);
    }

    /* Trailing pixels */
    COMBINE0_RGB(d0,v00,v01,v02,v02); d0 += 3;
    COMBINE1_RGB(d0,v00,v01,v02,v02); d0 += 3;
    COMBINE2_RGB(d0,v00,v01,v02,v02); d0 += 3;
    COMBINE3_RGB(d0,v00,v01,v02,v02); d0 += 3;
    COMBINE0_RGB(d0,v01,v02,v02,v02); d0 += 3;
    COMBINE1_RGB(d0,v01,v02,v02,v02);
    COMBINE0_RGB(d1,v10,v11,v12,v12); d1 += 3;
    COMBINE1_RGB(d1,v10,v11,v12,v12); d1 += 3;
    COMBINE2_RGB(d1,v10,v11,v12,v12); d1 += 3;
    COMBINE3_RGB(d1,v10,v11,v12,v12); d1 += 3;
    COMBINE0_RGB(d1,v11,v12,v12,v12); d1 += 3;
    COMBINE1_RGB(d1,v11,v12,v12,v12);
    COMBINE0_RGB(d2,v20,v21,v22,v22); d2 += 3;
    COMBINE1_RGB(d2,v20,v21,v22,v22); d2 += 3;
    COMBINE2_RGB(d2,v20,v21,v22,v22); d2 += 3;
    COMBINE3_RGB(d2,v20,v21,v22,v22); d2 += 3;
    COMBINE0_RGB(d2,v21,v22,v22,v22); d2 += 3;
    COMBINE1_RGB(d2,v21,v22,v22,v22);
    COMBINE0_RGB(d3,v30,v31,v32,v32); d3 += 3;
    COMBINE1_RGB(d3,v30,v31,v32,v32); d3 += 3;
    COMBINE2_RGB(d3,v30,v31,v32,v32); d3 += 3;
    COMBINE3_RGB(d3,v30,v31,v32,v32); d3 += 3;
    COMBINE0_RGB(d3,v31,v32,v32,v32); d3 += 3;
    COMBINE1_RGB(d3,v31,v32,v32,v32);

    return 4;
}

#define COMBINE_HALF_RGB(a,b,s0,s1,s2,s3) \
do { COMBINE_HALF(a##_r,b##_r,s0##_r,s1##_r,s2##_r,s3##_r); \
     COMBINE_HALF(a##_g,b##_g,s0##_g,s1##_g,s2##_g,s3##_g); \
     COMBINE_HALF(a##_b,b##_b,s0##_b,s1##_b,s2##_b,s3##_b); \
} while (0)
static int
quad_mitchell3_top(uint8_t       ** ipa_restrict dsts,
                   const uint8_t ** ipa_restrict srcs,
                   ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    const uint8_t * ipa_restrict s2 = srcs[2];
    const uint8_t * ipa_restrict s3 = srcs[3];
    DECLARE(s00); DECLARE(s01); DECLARE(s02);
    DECLARE(s10); DECLARE(s11); DECLARE(s12);
    DECLARE(s20); DECLARE(s21); DECLARE(s22);
    DECLARE(s30); DECLARE(s31); DECLARE(s32);
    DECLARE(v00); DECLARE(v01); DECLARE(v02);
    DECLARE(v10); DECLARE(v11); DECLARE(v12);

    /* Combine them vertically */
    LOAD(s00, s0); LOAD(s01, s0); LOAD(s02, s0);
    LOAD(s10, s1); LOAD(s11, s1); LOAD(s12, s1);
    LOAD(s20, s2); LOAD(s21, s2); LOAD(s22, s2);
    LOAD(s30, s3); LOAD(s31, s3); LOAD(s32, s3);
    COMBINE_HALF_RGB(v00, v10, s00, s10, s20, s30);
    COMBINE_HALF_RGB(v01, v11, s01, s11, s21, s31);
    COMBINE_HALF_RGB(v02, v12, s02, s12, s22, s32);

    /* Now do the leading pixels. */
    COMBINE2_RGB(d0,v00,v00,v00,v01); d0 += 3;
    COMBINE3_RGB(d0,v00,v00,v00,v01); d0 += 3;
    COMBINE0_RGB(d0,v00,v00,v01,v02); d0 += 3;
    COMBINE1_RGB(d0,v00,v00,v01,v02); d0 += 3;
    COMBINE2_RGB(d0,v00,v00,v01,v02); d0 += 3;
    COMBINE3_RGB(d0,v00,v00,v01,v02); d0 += 3;
    COMBINE2_RGB(d1,v10,v10,v10,v11); d1 += 3;
    COMBINE3_RGB(d1,v10,v10,v10,v11); d1 += 3;
    COMBINE0_RGB(d1,v10,v10,v11,v12); d1 += 3;
    COMBINE1_RGB(d1,v10,v10,v11,v12); d1 += 3;
    COMBINE2_RGB(d1,v10,v10,v11,v12); d1 += 3;
    COMBINE3_RGB(d1,v10,v10,v11,v12); d1 += 3;

    for (w = doubler->src_w-3; w > 0; w--)
    {
        DECLARE(s03); DECLARE(s13); DECLARE(s23); DECLARE(s33);
        DECLARE(v03); DECLARE(v13);
        LOAD(s03, s0); LOAD(s13, s1); LOAD(s23, s2); LOAD(s33, s3);
        COMBINE_HALF_RGB(v03, v13, s03, s13, s23, s33);
        COMBINE0_RGB(d0,v00,v01,v02,v03); d0 += 3;
        COMBINE1_RGB(d0,v00,v01,v02,v03); d0 += 3;
        COMBINE2_RGB(d0,v00,v01,v02,v03); d0 += 3;
        COMBINE3_RGB(d0,v00,v01,v02,v03); d0 += 3;
        COMBINE0_RGB(d1,v10,v11,v12,v13); d1 += 3;
        COMBINE1_RGB(d1,v10,v11,v12,v13); d1 += 3;
        COMBINE2_RGB(d1,v10,v11,v12,v13); d1 += 3;
        COMBINE3_RGB(d1,v10,v11,v12,v13); d1 += 3;
        ASSIGN(v00, v01); ASSIGN(v01, v02); ASSIGN(v02, v03);
        ASSIGN(v10, v11); ASSIGN(v11, v12); ASSIGN(v12, v13);
    }

    /* Trailing pixels */
    COMBINE0_RGB(d0,v00,v01,v02,v02); d0 += 3;
    COMBINE1_RGB(d0,v00,v01,v02,v02); d0 += 3;
    COMBINE2_RGB(d0,v00,v01,v02,v02); d0 += 3;
    COMBINE3_RGB(d0,v00,v01,v02,v02); d0 += 3;
    COMBINE0_RGB(d0,v01,v02,v02,v02); d0 += 3;
    COMBINE1_RGB(d0,v01,v02,v02,v02);
    COMBINE0_RGB(d1,v10,v11,v12,v12); d1 += 3;
    COMBINE1_RGB(d1,v10,v11,v12,v12); d1 += 3;
    COMBINE2_RGB(d1,v10,v11,v12,v12); d1 += 3;
    COMBINE3_RGB(d1,v10,v11,v12,v12); d1 += 3;
    COMBINE0_RGB(d1,v11,v12,v12,v12); d1 += 3;
    COMBINE1_RGB(d1,v11,v12,v12,v12);

    return 2;
}
#undef LOAD
#undef DECLARE
#undef ASSIGN
#undef COMBINE_RGB
#undef COMBINE_HALF_RGB
#undef COMBINE0_RGB
#undef COMBINE1_RGB
#undef COMBINE2_RGB
#undef COMBINE3_RGB

#define DECLARE(T) int T##_r, T##_g, T##_b, T##_k
#define LOAD(T,S) do { T##_r = *(S++); T##_g = *(S++); T##_b = *(S++); T##_k = *(S++); } while (0)
#define COMBINE_CMYK(a,b,c,d,s0,s1,s2,s3) \
do { COMBINE(a##_r,b##_r,c##_r,d##_r,s0##_r,s1##_r,s2##_r,s3##_r); \
     COMBINE(a##_g,b##_g,c##_g,d##_g,s0##_g,s1##_g,s2##_g,s3##_g); \
     COMBINE(a##_b,b##_b,c##_b,d##_b,s0##_b,s1##_b,s2##_b,s3##_b); \
     COMBINE(a##_k,b##_k,c##_k,d##_k,s0##_k,s1##_k,s2##_k,s3##_k); \
} while (0)
#define COMBINE0_CMYK(D,V0,V1,V2,V3) \
do { D[0] = RCLAMP(COMBINE0(V0##_r,V1##_r,V2##_r,V3##_r));\
     D[1] = RCLAMP(COMBINE0(V0##_g,V1##_g,V2##_g,V3##_g));\
     D[2] = RCLAMP(COMBINE0(V0##_b,V1##_b,V2##_b,V3##_b));\
     D[3] = RCLAMP(COMBINE0(V0##_k,V1##_k,V2##_k,V3##_k));\
} while (0)
#define COMBINE1_CMYK(D,V0,V1,V2,V3) \
do { D[0] = RCLAMP(COMBINE1(V0##_r,V1##_r,V2##_r,V3##_r));\
     D[1] = RCLAMP(COMBINE1(V0##_g,V1##_g,V2##_g,V3##_g));\
     D[2] = RCLAMP(COMBINE1(V0##_b,V1##_b,V2##_b,V3##_b));\
     D[3] = RCLAMP(COMBINE1(V0##_k,V1##_k,V2##_k,V3##_k));\
} while (0)
#define COMBINE2_CMYK(D,V0,V1,V2,V3) \
do { D[0] = RCLAMP(COMBINE2(V0##_r,V1##_r,V2##_r,V3##_r));\
     D[1] = RCLAMP(COMBINE2(V0##_g,V1##_g,V2##_g,V3##_g));\
     D[2] = RCLAMP(COMBINE2(V0##_b,V1##_b,V2##_b,V3##_b));\
     D[3] = RCLAMP(COMBINE2(V0##_k,V1##_k,V2##_k,V3##_k));\
} while (0)
#define COMBINE3_CMYK(D,V0,V1,V2,V3) \
do { D[0] = RCLAMP(COMBINE3(V0##_r,V1##_r,V2##_r,V3##_r));\
     D[1] = RCLAMP(COMBINE3(V0##_g,V1##_g,V2##_g,V3##_g));\
     D[2] = RCLAMP(COMBINE3(V0##_b,V1##_b,V2##_b,V3##_b));\
     D[3] = RCLAMP(COMBINE3(V0##_k,V1##_k,V2##_k,V3##_k));\
} while (0)
#define ASSIGN(D,S) do { D##_r = S##_r; D##_g = S##_g; D##_b = S##_b; D##_k = S##_k; } while (0)
static int
quad_mitchell4(uint8_t       ** ipa_restrict dsts,
               const uint8_t ** ipa_restrict srcs,
               ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    const uint8_t * ipa_restrict s2 = srcs[2];
    const uint8_t * ipa_restrict s3 = srcs[3];
    DECLARE(s00); DECLARE(s01); DECLARE(s02);
    DECLARE(s10); DECLARE(s11); DECLARE(s12);
    DECLARE(s20); DECLARE(s21); DECLARE(s22);
    DECLARE(s30); DECLARE(s31); DECLARE(s32);
    DECLARE(v00); DECLARE(v01); DECLARE(v02);
    DECLARE(v10); DECLARE(v11); DECLARE(v12);
    DECLARE(v20); DECLARE(v21); DECLARE(v22);
    DECLARE(v30); DECLARE(v31); DECLARE(v32);

    /* Combine them vertically */
    LOAD(s00, s0); LOAD(s01, s0); LOAD(s02, s0);
    LOAD(s10, s1); LOAD(s11, s1); LOAD(s12, s1);
    LOAD(s20, s2); LOAD(s21, s2); LOAD(s22, s2);
    LOAD(s30, s3); LOAD(s31, s3); LOAD(s32, s3);
    COMBINE_CMYK(v00, v10, v20, v30, s00, s10, s20, s30);
    COMBINE_CMYK(v01, v11, v21, v31, s01, s11, s21, s31);
    COMBINE_CMYK(v02, v12, v22, v32, s02, s12, s22, s32);

    /* Now do the leading pixels. */
    COMBINE2_CMYK(d0,v00,v00,v00,v01); d0 += 4;
    COMBINE3_CMYK(d0,v00,v00,v00,v01); d0 += 4;
    COMBINE0_CMYK(d0,v00,v00,v01,v02); d0 += 4;
    COMBINE1_CMYK(d0,v00,v00,v01,v02); d0 += 4;
    COMBINE2_CMYK(d0,v00,v00,v01,v02); d0 += 4;
    COMBINE3_CMYK(d0,v00,v00,v01,v02); d0 += 4;
    COMBINE2_CMYK(d1,v10,v10,v10,v11); d1 += 4;
    COMBINE3_CMYK(d1,v10,v10,v10,v11); d1 += 4;
    COMBINE0_CMYK(d1,v10,v10,v11,v12); d1 += 4;
    COMBINE1_CMYK(d1,v10,v10,v11,v12); d1 += 4;
    COMBINE2_CMYK(d1,v10,v10,v11,v12); d1 += 4;
    COMBINE3_CMYK(d1,v10,v10,v11,v12); d1 += 4;
    COMBINE2_CMYK(d2,v20,v20,v20,v21); d2 += 4;
    COMBINE3_CMYK(d2,v20,v20,v20,v21); d2 += 4;
    COMBINE0_CMYK(d2,v20,v20,v21,v22); d2 += 4;
    COMBINE1_CMYK(d2,v20,v20,v21,v22); d2 += 4;
    COMBINE2_CMYK(d2,v20,v20,v21,v22); d2 += 4;
    COMBINE3_CMYK(d2,v20,v20,v21,v22); d2 += 4;
    COMBINE2_CMYK(d3,v30,v30,v30,v31); d3 += 4;
    COMBINE3_CMYK(d3,v30,v30,v30,v31); d3 += 4;
    COMBINE0_CMYK(d3,v30,v30,v31,v32); d3 += 4;
    COMBINE1_CMYK(d3,v30,v30,v31,v32); d3 += 4;
    COMBINE2_CMYK(d3,v30,v30,v31,v32); d3 += 4;
    COMBINE3_CMYK(d3,v30,v30,v31,v32); d3 += 4;

    for (w = doubler->src_w-3; w > 0; w--)
    {
        DECLARE(s03); DECLARE(s13); DECLARE(s23); DECLARE(s33);
        DECLARE(v03); DECLARE(v13); DECLARE(v23); DECLARE(v33);
        LOAD(s03, s0); LOAD(s13, s1); LOAD(s23, s2); LOAD(s33, s3);
        COMBINE_CMYK(v03, v13, v23, v33, s03, s13, s23, s33);
        COMBINE0_CMYK(d0,v00,v01,v02,v03); d0 += 4;
        COMBINE1_CMYK(d0,v00,v01,v02,v03); d0 += 4;
        COMBINE2_CMYK(d0,v00,v01,v02,v03); d0 += 4;
        COMBINE3_CMYK(d0,v00,v01,v02,v03); d0 += 4;
        COMBINE0_CMYK(d1,v10,v11,v12,v13); d1 += 4;
        COMBINE1_CMYK(d1,v10,v11,v12,v13); d1 += 4;
        COMBINE2_CMYK(d1,v10,v11,v12,v13); d1 += 4;
        COMBINE3_CMYK(d1,v10,v11,v12,v13); d1 += 4;
        COMBINE0_CMYK(d2,v20,v21,v22,v23); d2 += 4;
        COMBINE1_CMYK(d2,v20,v21,v22,v23); d2 += 4;
        COMBINE2_CMYK(d2,v20,v21,v22,v23); d2 += 4;
        COMBINE3_CMYK(d2,v20,v21,v22,v23); d2 += 4;
        COMBINE0_CMYK(d3,v30,v31,v32,v33); d3 += 4;
        COMBINE1_CMYK(d3,v30,v31,v32,v33); d3 += 4;
        COMBINE2_CMYK(d3,v30,v31,v32,v33); d3 += 4;
        COMBINE3_CMYK(d3,v30,v31,v32,v33); d3 += 4;
        ASSIGN(v00, v01); ASSIGN(v01, v02); ASSIGN(v02, v03);
        ASSIGN(v10, v11); ASSIGN(v11, v12); ASSIGN(v12, v13);
        ASSIGN(v20, v21); ASSIGN(v21, v22); ASSIGN(v22, v23);
        ASSIGN(v30, v31); ASSIGN(v31, v32); ASSIGN(v32, v33);
    }

    /* Trailing pixels */
    COMBINE0_CMYK(d0,v00,v01,v02,v02); d0 += 4;
    COMBINE1_CMYK(d0,v00,v01,v02,v02); d0 += 4;
    COMBINE2_CMYK(d0,v00,v01,v02,v02); d0 += 4;
    COMBINE3_CMYK(d0,v00,v01,v02,v02); d0 += 4;
    COMBINE0_CMYK(d0,v01,v02,v02,v02); d0 += 4;
    COMBINE1_CMYK(d0,v01,v02,v02,v02);
    COMBINE0_CMYK(d1,v10,v11,v12,v12); d1 += 4;
    COMBINE1_CMYK(d1,v10,v11,v12,v12); d1 += 4;
    COMBINE2_CMYK(d1,v10,v11,v12,v12); d1 += 4;
    COMBINE3_CMYK(d1,v10,v11,v12,v12); d1 += 4;
    COMBINE0_CMYK(d1,v11,v12,v12,v12); d1 += 4;
    COMBINE1_CMYK(d1,v11,v12,v12,v12);
    COMBINE0_CMYK(d2,v20,v21,v22,v22); d2 += 4;
    COMBINE1_CMYK(d2,v20,v21,v22,v22); d2 += 4;
    COMBINE2_CMYK(d2,v20,v21,v22,v22); d2 += 4;
    COMBINE3_CMYK(d2,v20,v21,v22,v22); d2 += 4;
    COMBINE0_CMYK(d2,v21,v22,v22,v22); d2 += 4;
    COMBINE1_CMYK(d2,v21,v22,v22,v22);
    COMBINE0_CMYK(d3,v30,v31,v32,v32); d3 += 4;
    COMBINE1_CMYK(d3,v30,v31,v32,v32); d3 += 4;
    COMBINE2_CMYK(d3,v30,v31,v32,v32); d3 += 4;
    COMBINE3_CMYK(d3,v30,v31,v32,v32); d3 += 4;
    COMBINE0_CMYK(d3,v31,v32,v32,v32); d3 += 4;
    COMBINE1_CMYK(d3,v31,v32,v32,v32);

    return 4;
}

#define COMBINE_HALF_CMYK(a,b,s0,s1,s2,s3) \
do { COMBINE_HALF(a##_r,b##_r,s0##_r,s1##_r,s2##_r,s3##_r); \
     COMBINE_HALF(a##_g,b##_g,s0##_g,s1##_g,s2##_g,s3##_g); \
     COMBINE_HALF(a##_b,b##_b,s0##_b,s1##_b,s2##_b,s3##_b); \
     COMBINE_HALF(a##_k,b##_k,s0##_k,s1##_k,s2##_k,s3##_k); \
} while (0)
static int
quad_mitchell4_top(uint8_t       ** ipa_restrict dsts,
                   const uint8_t ** ipa_restrict srcs,
                   ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    const uint8_t * ipa_restrict s2 = srcs[2];
    const uint8_t * ipa_restrict s3 = srcs[3];
    DECLARE(s00); DECLARE(s01); DECLARE(s02);
    DECLARE(s10); DECLARE(s11); DECLARE(s12);
    DECLARE(s20); DECLARE(s21); DECLARE(s22);
    DECLARE(s30); DECLARE(s31); DECLARE(s32);
    DECLARE(v00); DECLARE(v01); DECLARE(v02);
    DECLARE(v10); DECLARE(v11); DECLARE(v12);

    /* Combine them vertically */
    LOAD(s00, s0); LOAD(s01, s0); LOAD(s02, s0);
    LOAD(s10, s1); LOAD(s11, s1); LOAD(s12, s1);
    LOAD(s20, s2); LOAD(s21, s2); LOAD(s22, s2);
    LOAD(s30, s3); LOAD(s31, s3); LOAD(s32, s3);
    COMBINE_HALF_CMYK(v00, v10, s00, s10, s20, s30);
    COMBINE_HALF_CMYK(v01, v11, s01, s11, s21, s31);
    COMBINE_HALF_CMYK(v02, v12, s02, s12, s22, s32);

    /* Now do the leading pixels. */
    COMBINE2_CMYK(d0,v00,v00,v00,v01); d0 += 4;
    COMBINE3_CMYK(d0,v00,v00,v00,v01); d0 += 4;
    COMBINE0_CMYK(d0,v00,v00,v01,v02); d0 += 4;
    COMBINE1_CMYK(d0,v00,v00,v01,v02); d0 += 4;
    COMBINE2_CMYK(d0,v00,v00,v01,v02); d0 += 4;
    COMBINE3_CMYK(d0,v00,v00,v01,v02); d0 += 4;
    COMBINE2_CMYK(d1,v10,v10,v10,v11); d1 += 4;
    COMBINE3_CMYK(d1,v10,v10,v10,v11); d1 += 4;
    COMBINE0_CMYK(d1,v10,v10,v11,v12); d1 += 4;
    COMBINE1_CMYK(d1,v10,v10,v11,v12); d1 += 4;
    COMBINE2_CMYK(d1,v10,v10,v11,v12); d1 += 4;
    COMBINE3_CMYK(d1,v10,v10,v11,v12); d1 += 4;

    for (w = doubler->src_w-3; w > 0; w--)
    {
        DECLARE(s03); DECLARE(s13); DECLARE(s23); DECLARE(s33);
        DECLARE(v03); DECLARE(v13);
        LOAD(s03, s0); LOAD(s13, s1); LOAD(s23, s2); LOAD(s33, s3);
        COMBINE_HALF_CMYK(v03, v13, s03, s13, s23, s33);
        COMBINE0_CMYK(d0,v00,v01,v02,v03); d0 += 4;
        COMBINE1_CMYK(d0,v00,v01,v02,v03); d0 += 4;
        COMBINE2_CMYK(d0,v00,v01,v02,v03); d0 += 4;
        COMBINE3_CMYK(d0,v00,v01,v02,v03); d0 += 4;
        COMBINE0_CMYK(d1,v10,v11,v12,v13); d1 += 4;
        COMBINE1_CMYK(d1,v10,v11,v12,v13); d1 += 4;
        COMBINE2_CMYK(d1,v10,v11,v12,v13); d1 += 4;
        COMBINE3_CMYK(d1,v10,v11,v12,v13); d1 += 4;
        ASSIGN(v00, v01); ASSIGN(v01, v02); ASSIGN(v02, v03);
        ASSIGN(v10, v11); ASSIGN(v11, v12); ASSIGN(v12, v13);
    }

    /* Trailing pixels */
    COMBINE0_CMYK(d0,v00,v01,v02,v02); d0 += 4;
    COMBINE1_CMYK(d0,v00,v01,v02,v02); d0 += 4;
    COMBINE2_CMYK(d0,v00,v01,v02,v02); d0 += 4;
    COMBINE3_CMYK(d0,v00,v01,v02,v02); d0 += 4;
    COMBINE0_CMYK(d0,v01,v02,v02,v02); d0 += 4;
    COMBINE1_CMYK(d0,v01,v02,v02,v02);
    COMBINE0_CMYK(d1,v10,v11,v12,v12); d1 += 4;
    COMBINE1_CMYK(d1,v10,v11,v12,v12); d1 += 4;
    COMBINE2_CMYK(d1,v10,v11,v12,v12); d1 += 4;
    COMBINE3_CMYK(d1,v10,v11,v12,v12); d1 += 4;
    COMBINE0_CMYK(d1,v11,v12,v12,v12); d1 += 4;
    COMBINE1_CMYK(d1,v11,v12,v12,v12);

    return 2;
}
#undef LOAD
#undef DECLARE
#undef ASSIGN
#undef COMBINE_CMYK
#undef COMBINE_HALF_CMYK
#undef COMBINE0_CMYK
#undef COMBINE1_CMYK
#undef COMBINE2_CMYK
#undef COMBINE3_CMYK

#undef WEIGHT_SHIFT
#undef MW0
#undef MW1
#undef MW2
#undef MW3
#undef MW4
#undef MW5
#undef MW6
#undef MW7

#undef WEIGHT_SCALE
#undef WEIGHT_ROUND
#undef COMBINE0
#undef COMBINE1
#undef COMBINE2
#undef COMBINE3
#undef COMBINE

#undef RCLAMP

static int
octo_near(uint8_t       ** ipa_restrict dsts,
          const uint8_t ** ipa_restrict srcs,
          ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    uint8_t * ipa_restrict d4 = dsts[4];
    uint8_t * ipa_restrict d5 = dsts[5];
    uint8_t * ipa_restrict d6 = dsts[6];
    uint8_t * ipa_restrict d7 = dsts[7];
    const uint8_t * ipa_restrict s0 = srcs[0];
    uint32_t channels = doubler->channels;

    for (w = doubler->src_w; w > 0; w--)
    {
        uint32_t j = channels;
        do {
            d0[0] = d0[channels] = d0[channels*2] = d0[channels*3] = d0[channels*4] = d0[channels*5] = d0[channels*6] = d0[channels*7] =
                d1[0] = d1[channels] = d1[channels*2] = d1[channels*3] = d1[channels*4] = d1[channels*5] = d1[channels*6] = d1[channels*7] =
                d2[0] = d2[channels] = d2[channels*2] = d2[channels*3] = d2[channels*4] = d2[channels*5] = d2[channels*6] = d2[channels*7] =
                d3[0] = d3[channels] = d3[channels*2] = d3[channels*3] = d3[channels*4] = d3[channels*5] = d3[channels*6] = d3[channels*7] =
                d4[0] = d4[channels] = d4[channels*2] = d4[channels*3] = d4[channels*4] = d4[channels*5] = d4[channels*6] = d4[channels*7] =
                d5[0] = d5[channels] = d5[channels*2] = d5[channels*3] = d5[channels*4] = d5[channels*5] = d5[channels*6] = d5[channels*7] =
                d6[0] = d6[channels] = d6[channels*2] = d6[channels*3] = d6[channels*4] = d6[channels*5] = d6[channels*6] = d6[channels*7] =
                d7[0] = d7[channels] = d7[channels*2] = d7[channels*3] = d7[channels*4] = d7[channels*5] = d7[channels*6] = d7[channels*7] = *s0++;
            d0++;
            d1++;
            d2++;
            d3++;
            d4++;
            d5++;
            d6++;
            d7++;
            j--;
        } while (j != 0);
        d0 += channels*7;
        d1 += channels*7;
        d2 += channels*7;
        d3 += channels*7;
        d4 += channels*7;
        d5 += channels*7;
        d6 += channels*7;
        d7 += channels*7;
    }

    return 8;
}

static int
octo_near1(uint8_t       ** ipa_restrict dsts,
           const uint8_t ** ipa_restrict srcs,
           ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    uint8_t * ipa_restrict d4 = dsts[4];
    uint8_t * ipa_restrict d5 = dsts[5];
    uint8_t * ipa_restrict d6 = dsts[6];
    uint8_t * ipa_restrict d7 = dsts[7];
    const uint8_t * ipa_restrict s0 = srcs[0];

    for (w = doubler->src_w; w > 0; w--)
    {
        d0[0] = d0[1] = d0[2] = d0[3] = d0[4] = d0[5] = d0[6] = d0[7] =
            d1[0] = d1[1] = d1[2] = d1[3] = d1[4] = d1[5] = d1[6] = d1[7] =
            d2[0] = d2[1] = d2[2] = d2[3] = d2[4] = d2[5] = d2[6] = d2[7] =
            d3[0] = d3[1] = d3[2] = d3[3] = d3[4] = d3[5] = d3[6] = d3[7] =
            d4[0] = d4[1] = d4[2] = d4[3] = d4[4] = d4[5] = d4[6] = d4[7] =
            d5[0] = d5[1] = d5[2] = d5[3] = d5[4] = d5[5] = d5[6] = d5[7] =
            d6[0] = d6[1] = d6[2] = d6[3] = d6[4] = d6[5] = d6[6] = d6[7] =
            d7[0] = d7[1] = d7[2] = d7[3] = d7[4] = d7[5] = d7[6] = d7[7] = *s0++;
        d0 += 8;
        d1 += 8;
        d2 += 8;
        d3 += 8;
        d4 += 8;
        d5 += 8;
        d6 += 8;
        d7 += 8;
    }

    return 8;
}

static int
octo_near3(uint8_t       ** ipa_restrict dsts,
           const uint8_t ** ipa_restrict srcs,
           ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    uint8_t * ipa_restrict d4 = dsts[4];
    uint8_t * ipa_restrict d5 = dsts[5];
    uint8_t * ipa_restrict d6 = dsts[6];
    uint8_t * ipa_restrict d7 = dsts[7];
    const uint8_t * ipa_restrict s0 = srcs[0];

    for (w = doubler->src_w; w > 0; w--)
    {
        d0[0] = d0[3] = d0[6] = d0[9] = d0[12] = d0[15] = d0[18] = d0[21] =
            d1[0] = d1[3] = d1[6] = d1[9] = d1[12] = d1[15] = d1[18] = d1[21] =
            d2[0] = d2[3] = d2[6] = d2[9] = d2[12] = d2[15] = d2[18] = d2[21] =
            d3[0] = d3[3] = d3[6] = d3[9] = d3[12] = d3[15] = d3[18] = d3[21] =
            d4[0] = d4[3] = d4[6] = d4[9] = d4[12] = d4[15] = d4[18] = d4[21] =
            d5[0] = d5[3] = d5[6] = d5[9] = d5[12] = d5[15] = d5[18] = d5[21] =
            d6[0] = d6[3] = d6[6] = d6[9] = d6[12] = d6[15] = d6[18] = d6[21] =
            d7[0] = d7[3] = d7[6] = d7[9] = d7[12] = d7[15] = d7[18] = d7[21] = *s0++;
        d0[1] = d0[4] = d0[7] = d0[10] = d0[13] = d0[16] = d0[19] = d0[22] =
            d1[1] = d1[4] = d1[7] = d1[10] = d1[13] = d1[16] = d1[19] = d1[22] =
            d2[1] = d2[4] = d2[7] = d2[10] = d2[13] = d2[16] = d2[19] = d2[22] =
            d3[1] = d3[4] = d3[7] = d3[10] = d3[13] = d3[16] = d3[19] = d3[22] =
            d4[1] = d4[4] = d4[7] = d4[10] = d4[13] = d4[16] = d4[19] = d4[22] =
            d5[1] = d5[4] = d5[7] = d5[10] = d5[13] = d5[16] = d5[19] = d5[22] =
            d6[1] = d6[4] = d6[7] = d6[10] = d6[13] = d6[16] = d6[19] = d6[22] =
            d7[1] = d7[4] = d7[7] = d7[10] = d7[13] = d7[16] = d7[19] = d7[22] = *s0++;
        d0[2] = d0[5] = d0[8] = d0[11] = d0[14] = d0[17] = d0[20] = d0[23] =
            d1[2] = d1[5] = d1[8] = d1[11] = d1[14] = d1[17] = d1[20] = d1[23] =
            d2[2] = d2[5] = d2[8] = d2[11] = d2[14] = d2[17] = d2[20] = d2[23] =
            d3[2] = d3[5] = d3[8] = d3[11] = d3[14] = d3[17] = d3[20] = d3[23] =
            d4[2] = d4[5] = d4[8] = d4[11] = d4[14] = d4[17] = d4[20] = d4[23] =
            d5[2] = d5[5] = d5[8] = d5[11] = d5[14] = d5[17] = d5[20] = d5[23] =
            d6[2] = d6[5] = d6[8] = d6[11] = d6[14] = d6[17] = d6[20] = d6[23] =
            d7[2] = d7[5] = d7[8] = d7[11] = d7[14] = d7[17] = d7[20] = d7[23] = *s0++;
        d0 += 24;
        d1 += 24;
        d2 += 24;
        d3 += 24;
        d4 += 24;
        d5 += 24;
        d6 += 24;
        d7 += 24;
    }

    return 8;
}

static int
octo_near4(uint8_t       ** ipa_restrict dsts,
           const uint8_t ** ipa_restrict srcs,
           ipa_doubler    * ipa_restrict doubler)
{
    uint32_t w;
    uint32_t * ipa_restrict d0 = (uint32_t *)(dsts[0]);
    uint32_t * ipa_restrict d1 = (uint32_t *)(dsts[1]);
    uint32_t * ipa_restrict d2 = (uint32_t *)(dsts[2]);
    uint32_t * ipa_restrict d3 = (uint32_t *)(dsts[3]);
    uint32_t * ipa_restrict d4 = (uint32_t *)(dsts[4]);
    uint32_t * ipa_restrict d5 = (uint32_t *)(dsts[5]);
    uint32_t * ipa_restrict d6 = (uint32_t *)(dsts[6]);
    uint32_t * ipa_restrict d7 = (uint32_t *)(dsts[7]);
    const uint32_t * ipa_restrict s0 = (uint32_t *)(srcs[0]);

    for (w = doubler->src_w; w > 0; w--)
    {
        d0[0] = d0[1] = d0[2] = d0[3] = d0[4] = d0[5] = d0[6] = d0[7] =
            d1[0] = d1[1] = d1[2] = d1[3] = d1[4] = d1[5] = d1[6] = d1[7] =
            d2[0] = d2[1] = d2[2] = d2[3] = d2[4] = d2[5] = d2[6] = d2[7] =
            d3[0] = d3[1] = d3[2] = d3[3] = d3[4] = d3[5] = d3[6] = d3[7] =
            d4[0] = d4[1] = d4[2] = d4[3] = d4[4] = d4[5] = d4[6] = d4[7] =
            d5[0] = d5[1] = d5[2] = d5[3] = d5[4] = d5[5] = d5[6] = d5[7] =
            d6[0] = d6[1] = d6[2] = d6[3] = d6[4] = d6[5] = d6[6] = d6[7] =
            d7[0] = d7[1] = d7[2] = d7[3] = d7[4] = d7[5] = d7[6] = d7[7] = *s0++;
        d0 += 8;
        d1 += 8;
        d2 += 8;
        d3 += 8;
        d4 += 8;
        d5 += 8;
        d6 += 8;
        d7 += 8;
    }

    return 8;
}
