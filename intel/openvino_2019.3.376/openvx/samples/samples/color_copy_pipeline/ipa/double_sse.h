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

#include <emmintrin.h>
#include <smmintrin.h>

/* Define a constant for us to use in _mm_set_epi8() calls */
#define ZZ -128

static int
double_near1_sse(uint8_t       ** ipa_restrict dsts,
                 const uint8_t ** ipa_restrict srcs,
                 ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];

    for (w = doubler->src_w-16; w >= 0; w -= 16)
    {
        __m128i mm0, mm1;

        mm0 = _mm_loadu_si128((const __m128i *)s0);
        mm1 = _mm_unpacklo_epi8(mm0,mm0);
        mm0 = _mm_unpackhi_epi8(mm0,mm0);
        _mm_storeu_si128((__m128i *)d0,mm1);
        _mm_storeu_si128((__m128i *)(d0+16),mm0);
        _mm_storeu_si128((__m128i *)d1,mm1);
        _mm_storeu_si128((__m128i *)(d1+16),mm0);
        s0 += 16;
        d0 += 32;
        d1 += 32;
    }

    w += 16;
    if (w)
    {
        __m128i mm0, mm1;
        uint8_t local[16];

        memcpy(local, s0, w);
        mm0 = _mm_loadu_si128((const __m128i *)local);
        mm1 = _mm_unpacklo_epi8(mm0,mm0);
        mm0 = _mm_unpackhi_epi8(mm0,mm0);
        switch (w)
        {
        case 15: ((uint16_t *)d0)[14] = ((uint16_t *)d1)[14] = _mm_extract_epi16(mm0, 6);
        case 14: ((uint16_t *)d0)[13] = ((uint16_t *)d1)[13] = _mm_extract_epi16(mm0, 5);
        case 13: ((uint16_t *)d0)[12] = ((uint16_t *)d1)[12] = _mm_extract_epi16(mm0, 4);
        case 12: ((uint16_t *)d0)[11] = ((uint16_t *)d1)[11] = _mm_extract_epi16(mm0, 3);
        case 11: ((uint16_t *)d0)[10] = ((uint16_t *)d1)[10] = _mm_extract_epi16(mm0, 2);
        case 10: ((uint16_t *)d0)[ 9] = ((uint16_t *)d1)[ 9] = _mm_extract_epi16(mm0, 1);
        case  9: ((uint16_t *)d0)[ 8] = ((uint16_t *)d1)[ 8] = _mm_extract_epi16(mm0, 0);
        case  8: ((uint16_t *)d0)[ 7] = ((uint16_t *)d1)[ 7] = _mm_extract_epi16(mm1, 7);
        case  7: ((uint16_t *)d0)[ 6] = ((uint16_t *)d1)[ 6] = _mm_extract_epi16(mm1, 6);
        case  6: ((uint16_t *)d0)[ 5] = ((uint16_t *)d1)[ 5] = _mm_extract_epi16(mm1, 5);
        case  5: ((uint16_t *)d0)[ 4] = ((uint16_t *)d1)[ 4] = _mm_extract_epi16(mm1, 4);
        case  4: ((uint16_t *)d0)[ 3] = ((uint16_t *)d1)[ 3] = _mm_extract_epi16(mm1, 3);
        case  3: ((uint16_t *)d0)[ 2] = ((uint16_t *)d1)[ 2] = _mm_extract_epi16(mm1, 2);
        case  2: ((uint16_t *)d0)[ 1] = ((uint16_t *)d1)[ 1] = _mm_extract_epi16(mm1, 1);
        case  1: ((uint16_t *)d0)[ 0] = ((uint16_t *)d1)[ 0] = _mm_extract_epi16(mm1, 0);
        }
    }

    return 2;
}

static int
double_near3_sse(uint8_t       ** ipa_restrict dsts,
                 const uint8_t ** ipa_restrict srcs,
                 ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    __m128i mm_shuf0, mm_shuf1, mm_shuf2, mm_shuf3, mm_shuf4;

    mm_shuf0 = _mm_set_epi8( 6, 8, 7, 6, 5, 4, 3, 5, 4, 3, 2, 1, 0, 2, 1, 0);
    mm_shuf1 = _mm_set_epi8(ZZ,15,14,13,12,14,13,12,11,10, 9,11,10, 9, 8, 7);
    mm_shuf2 = _mm_set_epi8(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,15,14,ZZ);
    mm_shuf3 = _mm_set_epi8( 7, 6, 5, 7, 6, 5, 4, 3, 2, 4, 3, 2, 1,ZZ,ZZ, 1);
    mm_shuf4 = _mm_set_epi8(ZZ,ZZ,ZZ,ZZ,13,12,11,13,12,11,10, 9, 8,10, 9, 8);
    for (w = doubler->src_w-10; w >= 0; w -= 10)
    {
        __m128i mm0, mm1, mm2;

        mm0 = _mm_loadu_si128((const __m128i *)s0);      // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);           // mm1 = ggiihhggffeeddffeeddccbbaaccbbaa
        _mm_storeu_si128((__m128i *)d0,mm1);
        _mm_storeu_si128((__m128i *)d1,mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf1);           // mm1 = 00ppoonnmmoonnmmllkkjjllkkjjiihh
        mm0 = _mm_loadu_si128((const __m128i *)(s0+16)); // mm0 = FFEEDDCCBBAAzzyyxxwwvvuuttssrrqq
        mm2 = _mm_slli_si128(mm0,15);                    // mm2 = qq000000000000000000000000000000
        mm1 = _mm_or_si128(mm1,mm2);                     // mm1 = qqppoonnmmoonnmmllkkjjllkkjjiihh
        _mm_storeu_si128((__m128i *)(d0+16),mm1);
        _mm_storeu_si128((__m128i *)(d1+16),mm1);
        mm1 = _mm_shuffle_epi8(mm1,mm_shuf2);            // mm1 = 00000000000000000000000000qqpp00
        mm2 = _mm_shuffle_epi8(mm0, mm_shuf3);           // mm2 = xxwwvvxxwwvvuuttssuuttssrr0000rr
        mm1 = _mm_or_si128(mm1,mm2);                     // mm1 = xxwwvvxxwwvvuuttssuuttssrrqqpprr
        _mm_storeu_si128((__m128i *)(d0+32),mm1);
        _mm_storeu_si128((__m128i *)(d1+32),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf4);           // mm1 = 00000000DDCCBBDDCCBBAAzzyyAAzzyy
        _mm_storeu_si128((__m128i *)(d0+48),mm1);
        _mm_storeu_si128((__m128i *)(d1+48),mm1);
        d0 += 60;
        d1 += 60;
        s0 += 30;
    }

    w += 10;
    if (w)
    {
        __m128i mm0, mm1, mm2;
        uint8_t local[16];

        memcpy(local, s0, w*3);
        mm0 = _mm_loadu_si128((const __m128i *)local);   // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);           // mm1 = ggiihhggffeeddffeeddccbbaaccbbaa
        switch (w)
        {
            default:
                ((uint32_t *)d0)[ 3] = ((uint32_t *)d1)[ 3] = _mm_extract_epi32(mm1, 3);
            case 2:
                ((uint32_t *)d0)[ 2] = ((uint32_t *)d1)[ 2] = _mm_extract_epi32(mm1, 2);
                ((uint16_t *)d0)[ 3] = ((uint16_t *)d1)[ 3] = _mm_extract_epi16(mm1, 3);
            case 1:
                ((uint16_t *)d0)[ 2] = ((uint16_t *)d1)[ 2] = _mm_extract_epi16(mm1, 2);
                ((uint32_t *)d0)[ 0] = ((uint32_t *)d1)[ 0] = _mm_extract_epi32(mm1, 0);
        }
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf1);           // mm1 = 00ppoonnmmoonnmmllkkjjllkkjjiihh
        mm0 = _mm_loadu_si128((const __m128i *)(s0+16)); // mm0 = FFEEDDCCBBAAzzyyxxwwvvuuttssrrqq
        mm2 = _mm_slli_si128(mm0,15);                    // mm2 = qq000000000000000000000000000000
        mm1 = _mm_or_si128(mm1,mm2);                     // mm1 = qqppoonnmmoonnmmllkkjjllkkjjiihh
        switch (w)
        {
            default:
                ((uint16_t *)d0)[15] = ((uint16_t *)d1)[15] = _mm_extract_epi16(mm1, 7);
            case 5:
                ((uint16_t *)d0)[14] = ((uint16_t *)d1)[14] = _mm_extract_epi16(mm1, 6);
                ((uint32_t *)d0)[ 6] = ((uint32_t *)d1)[ 6] = _mm_extract_epi32(mm1, 2);
            case 4:
                ((uint32_t *)d0)[ 5] = ((uint32_t *)d1)[ 5] = _mm_extract_epi32(mm1, 1);
                ((uint16_t *)d0)[ 9] = ((uint16_t *)d1)[ 9] = _mm_extract_epi16(mm1, 1);
            case 3:
                ((uint16_t *)d0)[ 8] = ((uint16_t *)d1)[ 8] = _mm_extract_epi16(mm1, 0);
            case 2: case 1: {/* Done */}
        }
        mm1 = _mm_shuffle_epi8(mm1,mm_shuf2);            // mm1 = 00000000000000000000000000qqpp00
        mm2 = _mm_shuffle_epi8(mm0, mm_shuf3);           // mm2 = xxwwvvxxwwvvuuttssuuttssrr0000rr
        mm1 = _mm_or_si128(mm1,mm2);                     // mm1 = xxwwvvxxwwvvuuttssuuttssrrqqpprr
        switch (w)
        {
            default:
            case 8:
                ((uint32_t *)d0)[11] = ((uint32_t *)d1)[11] = _mm_extract_epi32(mm1, 3);
                ((uint16_t *)d0)[21] = ((uint16_t *)d1)[21] = _mm_extract_epi16(mm1, 5);
            case 7:
                ((uint16_t *)d0)[20] = ((uint16_t *)d1)[20] = _mm_extract_epi16(mm1, 4);
                ((uint32_t *)d0)[ 9] = ((uint32_t *)d1)[ 9] = _mm_extract_epi32(mm1, 1);
            case 6:
                ((uint32_t *)d0)[ 8] = ((uint32_t *)d1)[ 8] = _mm_extract_epi32(mm1, 0);
            case 5: case 4: case 3: case 2: case 1: {/* Done */}
        }
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf4);           // mm1 = 00000000DDCCBBDDCCBBAAzzyyAAzzyy
        switch (w)
        {
            default:
            case 10:
                ((uint32_t *)d0)[14] = ((uint32_t *)d1)[14] = _mm_extract_epi32(mm1, 2);
                ((uint16_t *)d0)[27] = ((uint16_t *)d1)[27] = _mm_extract_epi16(mm1, 3);
            case 9:
                ((uint16_t *)d0)[26] = ((uint16_t *)d1)[26] = _mm_extract_epi16(mm1, 2);
                ((uint32_t *)d0)[12] = ((uint32_t *)d1)[12] = _mm_extract_epi32(mm1, 0);
            case 8: case 7: case 6: case 5: case 4: case 3: case 2: case 1: {/* Done */}
        }
    }

    return 2;
}

static int
double_near4_sse(uint8_t       ** ipa_restrict dsts,
                 const uint8_t ** ipa_restrict srcs,
                 ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];

    for (w = doubler->src_w-4; w >= 0; w -= 4)
    {
        __m128i mm0, mm1;

        mm0 = _mm_loadu_si128((const __m128i *)s0);
        mm1 = _mm_unpacklo_epi32(mm0,mm0);
        mm0 = _mm_unpackhi_epi32(mm0,mm0);
        _mm_storeu_si128((__m128i *)d0,mm1);
        _mm_storeu_si128((__m128i *)(d0+16),mm0);
        _mm_storeu_si128((__m128i *)d1,mm1);
        _mm_storeu_si128((__m128i *)(d1+16),mm0);
        s0 += 16;
        d0 += 32;
        d1 += 32;
    }

    w += 4;
    if (w)
    {
        __m128i mm0, mm1;
        uint8_t local[16];

        memcpy(local, s0, w*4);
        mm0 = _mm_loadu_si128((const __m128i *)local);
        mm1 = _mm_unpacklo_epi32(mm0,mm0);
        mm0 = _mm_unpackhi_epi32(mm0,mm0);
        switch (w)
        {
#if defined (_M_X64)
        case  3: ((uint64_t *)d0)[ 2] = ((uint64_t *)d1)[ 2] = _mm_extract_epi64(mm0, 0);
        case  2: ((uint64_t *)d0)[ 1] = ((uint64_t *)d1)[ 1] = _mm_extract_epi64(mm1, 1);
        case  1: ((uint64_t *)d0)[ 0] = ((uint64_t *)d1)[ 0] = _mm_extract_epi64(mm1, 0);
#else
        case  3:
            ((uint32_t *)d0)[ 5] = ((uint32_t *)d1)[ 5] = _mm_extract_epi32(mm0, 1);
            ((uint32_t *)d0)[ 4] = ((uint32_t *)d1)[ 4] = _mm_extract_epi32(mm0, 0);
        case  2:
            ((uint32_t *)d0)[ 3] = ((uint32_t *)d1)[ 3] = _mm_extract_epi32(mm1, 3);
            ((uint32_t *)d0)[ 2] = ((uint32_t *)d1)[ 2] = _mm_extract_epi32(mm1, 2);
        case  1:
            ((uint32_t *)d0)[ 1] = ((uint32_t *)d1)[ 1] = _mm_extract_epi32(mm1, 1);
            ((uint32_t *)d0)[ 0] = ((uint32_t *)d1)[ 0] = _mm_extract_epi32(mm1, 0);
#endif
        default: {/* Never happens */}
        }
    }

    return 2;
}

/* Rather than use the exact:
      d[0] = s[0]*3/4 + s[1]*1/4
      d[1] = s[0]*1/4 + s[1]*3/4
   formulation here, we use a trick that keeps it all into single bytes.
   I had considered:
      t = (s[0]>>2)-(s[1]>>2)
      d[0] = s[0] - t
      d[1] = s[1] + t
   which produces a maximum error of 1 (with a sum of absolute differences
   of 16384 over all the possible inputs), but instead, we use a formulation
   spotted by Michael Vrhel:
      t = (s[0]+s[1] + 1)>>1
      d[0] = (s[0] + t + 1)>>1
      d[1] = (s[1] + t + 1)>>1
   This has the same worst case error of 1, but a larger sad of 32768. It
   codes up more nicely into SSE though.
 */

static inline __m128i
shift_down(__m128i reg, int w)
{
    if (w & 8)
        reg = _mm_srli_si128(reg,8);
    if (w & 4)
        reg = _mm_srli_si128(reg,4);
    if (w & 2)
        reg = _mm_srli_si128(reg,2);
    if (w & 1)
        reg = _mm_srli_si128(reg,1);

    return reg;
}

#define COMBINE(L,R) { int t = (L+R+1)>>1; L = (L+t+1)>>1; R = (R+t+1)>>1; }
static int
double_interp1_sse(uint8_t       ** ipa_restrict dsts,
                   const uint8_t ** ipa_restrict srcs,
                   ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    int tl = *s0++;
    int bl = *s1++;
    __m128i mm0, mm1, mm2, mm3, mm4, mm5, mm6;

    /* Leading single pixel */
    COMBINE(tl, bl);
    *d0++ = tl;
    *d1++ = bl;

    mm4 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,tl);
    mm5 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,bl);

    for (w = doubler->src_w-17; w >= 0; w -= 16)
    {
        // mm4, mm5 = single (combined) pixel carried forward.
        // Load raw pixels into mm0 and mm1 (source pixels n to n+15)
        mm0 = _mm_loadu_si128((const __m128i *)s0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        mm1 = _mm_loadu_si128((const __m128i *)s1); // mm1 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        // Combine vertically
        mm3 = _mm_avg_epu8(mm0,mm1);                // mm3 = (mm0 + mm1)>>1
        mm0 = _mm_avg_epu8(mm0,mm3);                // mm0 = (mm0 + mm3)>>1
        mm1 = _mm_avg_epu8(mm1,mm3);                // mm1 = (mm1 + mm3)>>1
        // mm0 and mm1 are vertically combined pixels (n to n+15).
        mm2 = _mm_slli_si128(mm0,1);
        mm3 = _mm_slli_si128(mm1,1);
        mm2 = _mm_or_si128(mm2,mm4);
        mm3 = _mm_or_si128(mm3,mm5);
        // mm2 and mm3 are vertically combined pixels (n-1 to n+14)
        // Make mm4 and mm5 ready for next iteration.
        mm4 = _mm_srli_si128(mm0,15);
        mm5 = _mm_srli_si128(mm1,15);
        // Now, we need to horizontally combine the 2 sets of combined pixels
        mm6 = _mm_avg_epu8(mm0,mm2);
        mm0 = _mm_avg_epu8(mm0,mm6);
        mm2 = _mm_avg_epu8(mm2,mm6);
        mm6 = _mm_avg_epu8(mm1,mm3);
        mm1 = _mm_avg_epu8(mm1,mm6);
        mm3 = _mm_avg_epu8(mm3,mm6);
        // So our output pixel values are in mm0/mm2 and mm1/mm3
        // But they are in the wrong order.
        mm6 = _mm_unpacklo_epi8(mm2,mm0);
        mm0 = _mm_unpackhi_epi8(mm2,mm0);
        mm2 = _mm_unpacklo_epi8(mm3,mm1);
        mm3 = _mm_unpackhi_epi8(mm3,mm1);
        // Results are in mm6/mm0, mm2/mm3
        _mm_storeu_si128((__m128i *)d0, mm6);
        _mm_storeu_si128((__m128i *)(d0+16), mm0);
        _mm_storeu_si128((__m128i *)d1, mm2);
        _mm_storeu_si128((__m128i *)(d1+16), mm3);
        d0 += 32;
        d1 += 32;
        s0 += 16;
        s1 += 16;
    }

    w += 16;
    if (w)
    {
        uint8_t local0[16], local1[16];
        memcpy(local0, s0, w+1);
        memcpy(local1, s1, w+1);
        // mm4, mm5 = single (combined) pixel carried forward.
        // Load raw pixels into mm0 and mm1 (source pixels n to n+15)
        mm0 = _mm_loadu_si128((const __m128i *)local0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        mm1 = _mm_loadu_si128((const __m128i *)local1); // mm1 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        // Combine vertically
        mm3 = _mm_avg_epu8(mm0,mm1);                // mm3 = (mm0 + mm1)>>1
        mm0 = _mm_avg_epu8(mm0,mm3);                // mm0 = (mm0 + mm3)>>1
        mm1 = _mm_avg_epu8(mm1,mm3);                // mm1 = (mm1 + mm3)>>1
        // mm0 and mm1 are vertically combined pixels (n to n+15).
        mm2 = _mm_slli_si128(mm0,1);
        mm3 = _mm_slli_si128(mm1,1);
        mm2 = _mm_or_si128(mm2,mm4);
        mm3 = _mm_or_si128(mm3,mm5);
        // mm2 and mm3 are vertically combined pixels (n-1 to n+14)
        // Make mm4 and mm5 ready for next iteration.
        mm4 = shift_down(mm0, w-1);
        mm5 = shift_down(mm1, w-1);
        // Now, we need to horizontally combine the 2 sets of combined pixels
        mm6 = _mm_avg_epu8(mm0,mm2);
        mm0 = _mm_avg_epu8(mm0,mm6);
        mm2 = _mm_avg_epu8(mm2,mm6);
        mm6 = _mm_avg_epu8(mm1,mm3);
        mm1 = _mm_avg_epu8(mm1,mm6);
        mm3 = _mm_avg_epu8(mm3,mm6);
        // So our output pixel values are in mm0/mm2 and mm1/mm3
        // But they are in the wrong order.
        mm6 = _mm_unpacklo_epi8(mm2,mm0);
        mm0 = _mm_unpackhi_epi8(mm2,mm0);
        mm2 = _mm_unpacklo_epi8(mm3,mm1);
        mm3 = _mm_unpackhi_epi8(mm3,mm1);
        // Results are in mm6/mm0, mm2/mm3
        switch (w)
        {
        case 15: ((uint16_t *)d0)[14] = _mm_extract_epi16(mm0, 6);
                 ((uint16_t *)d1)[14] = _mm_extract_epi16(mm3, 6);
        case 14: ((uint16_t *)d0)[13] = _mm_extract_epi16(mm0, 5);
                 ((uint16_t *)d1)[13] = _mm_extract_epi16(mm3, 5);
        case 13: ((uint16_t *)d0)[12] = _mm_extract_epi16(mm0, 4);
                 ((uint16_t *)d1)[12] = _mm_extract_epi16(mm3, 4);
        case 12: ((uint16_t *)d0)[11] = _mm_extract_epi16(mm0, 3);
                 ((uint16_t *)d1)[11] = _mm_extract_epi16(mm3, 3);
        case 11: ((uint16_t *)d0)[10] = _mm_extract_epi16(mm0, 2);
                 ((uint16_t *)d1)[10] = _mm_extract_epi16(mm3, 2);
        case 10: ((uint16_t *)d0)[ 9] = _mm_extract_epi16(mm0, 1);
                 ((uint16_t *)d1)[ 9] = _mm_extract_epi16(mm3, 1);
        case  9: ((uint16_t *)d0)[ 8] = _mm_extract_epi16(mm0, 0);
                 ((uint16_t *)d1)[ 8] = _mm_extract_epi16(mm3, 0);
        case  8: ((uint16_t *)d0)[ 7] = _mm_extract_epi16(mm6, 7);
                 ((uint16_t *)d1)[ 7] = _mm_extract_epi16(mm2, 7);
        case  7: ((uint16_t *)d0)[ 6] = _mm_extract_epi16(mm6, 6);
                 ((uint16_t *)d1)[ 6] = _mm_extract_epi16(mm2, 6);
        case  6: ((uint16_t *)d0)[ 5] = _mm_extract_epi16(mm6, 5);
                 ((uint16_t *)d1)[ 5] = _mm_extract_epi16(mm2, 5);
        case  5: ((uint16_t *)d0)[ 4] = _mm_extract_epi16(mm6, 4);
                 ((uint16_t *)d1)[ 4] = _mm_extract_epi16(mm2, 4);
        case  4: ((uint16_t *)d0)[ 3] = _mm_extract_epi16(mm6, 3);
                 ((uint16_t *)d1)[ 3] = _mm_extract_epi16(mm2, 3);
        case  3: ((uint16_t *)d0)[ 2] = _mm_extract_epi16(mm6, 2);
                 ((uint16_t *)d1)[ 2] = _mm_extract_epi16(mm2, 2);
        case  2: ((uint16_t *)d0)[ 1] = _mm_extract_epi16(mm6, 1);
                 ((uint16_t *)d1)[ 1] = _mm_extract_epi16(mm2, 1);
        case  1: ((uint16_t *)d0)[ 0] = _mm_extract_epi16(mm6, 0);
                 ((uint16_t *)d1)[ 0] = _mm_extract_epi16(mm2, 0);
        }
    }
    /* Trailing pixel */
    d0[w*2] = _mm_extract_epi8(mm4,0);
    d1[w*2] = _mm_extract_epi8(mm5,0);

    return 2;
}

static int
double_interp1_top_sse(uint8_t       ** ipa_restrict dsts,
                       const uint8_t ** ipa_restrict srcs,
                       ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    const uint8_t * ipa_restrict s0 = srcs[0];
    int tl = *s0++;
    __m128i mm0, mm2, mm4, mm6;

    /* Leading single pixel */
    *d0++ = tl;

    mm4 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,tl);

    for (w = doubler->src_w-17; w >= 0; w -= 16)
    {
        // mm4 = single pixel carried forward.
        // Load raw pixels into mm0 and mm1 (source pixels n to n+15)
        mm0 = _mm_loadu_si128((const __m128i *)s0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        // mm0 is vertically combined pixels (n to n+15).
        mm2 = _mm_slli_si128(mm0,1);
        mm2 = _mm_or_si128(mm2,mm4);
        // mm2 is vertically combined pixels (n-1 to n+14)
        // Make mm4 ready for next iteration.
        mm4 = _mm_srli_si128(mm0,15);
        // Now, we need to horizontally combine the pixels
        mm6 = _mm_avg_epu8(mm0,mm2);
        mm0 = _mm_avg_epu8(mm0,mm6);
        mm2 = _mm_avg_epu8(mm2,mm6);
        // So our output pixel values are in mm0/mm2
        // But they are in the wrong order.
        mm6 = _mm_unpacklo_epi8(mm2,mm0);
        mm0 = _mm_unpackhi_epi8(mm2,mm0);
        // Results are in mm6/mm0
        _mm_storeu_si128((__m128i *)d0, mm6);
        _mm_storeu_si128((__m128i *)(d0+16), mm0);
        d0 += 32;
        s0 += 16;
    }

    w += 16;
    if (w)
    {
        uint8_t local[16];
        memcpy(local, s0, w+1);
        // mm4 = single pixel carried forward.
        // Load raw pixels into mm0 and mm1 (source pixels n to n+15)
        mm0 = _mm_loadu_si128((const __m128i *)local); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        // mm0 is vertically combined pixels (n to n+15).
        mm2 = _mm_slli_si128(mm0,1);
        mm2 = _mm_or_si128(mm2,mm4);
        // mm2 is vertically combined pixels (n-1 to n+14)
        // Make mm4 ready for next iteration.
        mm4 = shift_down(mm0,w-1);
        // Now, we need to horizontally combine the pixels
        mm6 = _mm_avg_epu8(mm0,mm2);
        mm0 = _mm_avg_epu8(mm0,mm6);
        mm2 = _mm_avg_epu8(mm2,mm6);
        // So our output pixel values are in mm0/mm2
        // But they are in the wrong order.
        mm6 = _mm_unpacklo_epi8(mm2,mm0);
        mm0 = _mm_unpackhi_epi8(mm2,mm0);
        // Results are in mm6/mm0
        _mm_storeu_si128((__m128i *)d0, mm6);
        _mm_storeu_si128((__m128i *)(d0+16), mm0);
        switch (w)
        {
        case 15: ((uint16_t *)d0)[14] = _mm_extract_epi16(mm0, 6);
        case 14: ((uint16_t *)d0)[13] = _mm_extract_epi16(mm0, 5);
        case 13: ((uint16_t *)d0)[12] = _mm_extract_epi16(mm0, 4);
        case 12: ((uint16_t *)d0)[11] = _mm_extract_epi16(mm0, 3);
        case 11: ((uint16_t *)d0)[10] = _mm_extract_epi16(mm0, 2);
        case 10: ((uint16_t *)d0)[ 9] = _mm_extract_epi16(mm0, 1);
        case  9: ((uint16_t *)d0)[ 8] = _mm_extract_epi16(mm0, 0);
        case  8: ((uint16_t *)d0)[ 7] = _mm_extract_epi16(mm6, 7);
        case  7: ((uint16_t *)d0)[ 6] = _mm_extract_epi16(mm6, 6);
        case  6: ((uint16_t *)d0)[ 5] = _mm_extract_epi16(mm6, 5);
        case  5: ((uint16_t *)d0)[ 4] = _mm_extract_epi16(mm6, 4);
        case  4: ((uint16_t *)d0)[ 3] = _mm_extract_epi16(mm6, 3);
        case  3: ((uint16_t *)d0)[ 2] = _mm_extract_epi16(mm6, 2);
        case  2: ((uint16_t *)d0)[ 1] = _mm_extract_epi16(mm6, 1);
        case  1: ((uint16_t *)d0)[ 0] = _mm_extract_epi16(mm6, 0);
        }
    }

    /* Trailing single pixel */
    d0[w*2] = _mm_extract_epi8(mm4,0);

    return 1;
}
#undef COMBINE

#define DECLARE(A) int A##_r,A##_g,A##_b
#define LOAD(A,S) do { A##_r=S[0]; A##_g=S[1]; A##_b=S[2]; S+=3; } while (0)
#define STORE(A,D) do { D[0]=A##_r; D[1]=A##_g; D[2]=A##_b; D+=3; } while (0)
#define COMBINE(L,R) do {\
    int t = (L##_r + R##_r + 1)>>1;\
    L##_r = (L##_r + t + 1)>>1;\
    R##_r = (R##_r + t + 1)>>1;\
    t     = (L##_g + R##_g + 1)>>1;\
    L##_g = (L##_g + t + 1)>>1;\
    R##_g = (R##_g + t + 1)>>1;\
    t     = (L##_b + R##_b + 1)>>1;\
    L##_b = (L##_b + t + 1)>>1;\
    R##_b = (R##_b + t + 1)>>1;\
} while (0)
#define ASSIGN(D,S) do { D##_r=S##_r; D##_g=S##_g; D##_b=S##_b; } while (0)
static int
double_interp3_sse(uint8_t       ** ipa_restrict dsts,
                   const uint8_t ** ipa_restrict srcs,
                   ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    DECLARE(tl);
    DECLARE(bl);
    __m128i mm_shuf1, mm_shuf2, mm_shuf3, mm_shuf4;
    __m128i mm_shuf5, mm_shuf6, mm_shuf7, mm_shuf8;
    __m128i mm_shuf9, mm_shuf10,mm_shuf11,mm_shuf12;
    __m128i mm_fwd0, mm_fwd1;

    /* Leading single pixel */
    LOAD(tl,s0);
    LOAD(bl,s1);
    COMBINE(tl, bl);
    STORE(tl,d0);
    STORE(bl,d1);

    mm_fwd0 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,tl_b,tl_g,tl_r);
    mm_fwd1 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,bl_b,bl_g,bl_r);

    // d00 and d10 are combined using the following:
    mm_shuf1 = _mm_set_epi8(ZZ, 8, 7, 6,ZZ,ZZ,ZZ, 5, 4, 3,ZZ,ZZ,ZZ, 2, 1, 0);
    mm_shuf2 = _mm_set_epi8( 6,ZZ,ZZ,ZZ, 5, 4, 3,ZZ,ZZ,ZZ, 2, 1, 0,ZZ,ZZ,ZZ);
    // d01 and d11 are combined using the following:
    mm_shuf3 = _mm_set_epi8(ZZ,15,ZZ,ZZ,ZZ,14,13,12,ZZ,ZZ,ZZ,11,10, 9,ZZ,ZZ);
    //                      ^^ 16 - won't be available yet.
    mm_shuf4 = _mm_set_epi8(ZZ,ZZ,14,13,12,ZZ,ZZ,ZZ,11,10, 9,ZZ,ZZ,ZZ, 8, 7);
    // d02 and d12 are combined using the following:
    mm_shuf5 = _mm_set_epi8(ZZ,ZZ,ZZ,23,22,21,ZZ,ZZ,ZZ,20,19,18,ZZ,ZZ,ZZ,17);
    mm_shuf6 = _mm_set_epi8(23,22,21,ZZ,ZZ,ZZ,20,19,18,ZZ,ZZ,ZZ,17,16,ZZ,ZZ);
    //                                    15 carried over from before ^^
    // d03 and d13 are combined using the following:
    mm_shuf7 = _mm_set_epi8(ZZ,ZZ,31,30,ZZ,ZZ,ZZ,29,28,27,ZZ,ZZ,ZZ,26,25,24);
    //                         ^^ 32 - won't be available yet
    mm_shuf8 = _mm_set_epi8(30,ZZ,ZZ,ZZ,29,28,27,ZZ,ZZ,ZZ,26,25,24,ZZ,ZZ,ZZ);
    // d04 and d14 are combined using the following:
    mm_shuf9 = _mm_set_epi8(40,39,ZZ,ZZ,ZZ,38,37,36,ZZ,ZZ,ZZ,35,34,33,ZZ,ZZ);
    mm_shuf10= _mm_set_epi8(ZZ,ZZ,38,37,36,ZZ,ZZ,ZZ,35,34,33,ZZ,ZZ,ZZ,32,ZZ);
    //                                       31 carried over from before ^^
    // d05 and d15 are combined using the following:
    mm_shuf11= _mm_set_epi8(ZZ,ZZ,ZZ,47,46,45,ZZ,ZZ,ZZ,44,43,42,ZZ,ZZ,ZZ,41);
    mm_shuf12= _mm_set_epi8(47,46,45,ZZ,ZZ,ZZ,44,43,42,ZZ,ZZ,ZZ,41,40,39,ZZ);
    for (w = doubler->src_w-17; w >= 0; w -= 16)
    {
        __m128i mm_s00, mm_s01, mm_s02, mm_s10, mm_s11, mm_s12;
        __m128i mm_d00, mm_d01, mm_d02, mm_d03, mm_d04, mm_d05;
        __m128i mm_d10, mm_d11, mm_d12, mm_d13, mm_d14, mm_d15;
        __m128i mm_v00, mm_v01, mm_v02, mm_v10, mm_v11, mm_v12;
        __m128i mm_w00, mm_w01, mm_w02, mm_w10, mm_w11, mm_w12;
        __m128i mm_vh00, mm_vh01, mm_vh02, mm_vh10, mm_vh11, mm_vh12;
        __m128i mm_wh00, mm_wh01, mm_wh02, mm_wh10, mm_wh11, mm_wh12;
        __m128i mm_tmp;

        // Load raw pixels into mm0 and mm1 (source pixel bytes n to n+15)
        mm_s00 = _mm_loadu_si128((const __m128i *)s0);
        mm_s10 = _mm_loadu_si128((const __m128i *)s1);
        // Combine vertically
        mm_tmp = _mm_avg_epu8(mm_s00,mm_s10);
        mm_v00 = _mm_avg_epu8(mm_s00,mm_tmp);
        mm_v10 = _mm_avg_epu8(mm_s10,mm_tmp);
        // mm_v00 and mm_v10 are vertically combined pixel bytes (n to n+15).
        mm_w00 = _mm_slli_si128(mm_v00,3);
        mm_w10 = _mm_slli_si128(mm_v10,3);
        mm_w00 = _mm_or_si128(mm_w00,mm_fwd0);
        mm_w10 = _mm_or_si128(mm_w10,mm_fwd1);
        // mm_w00 and mm_w01 are vertically combined pixel bytes, delayed by a pixel (n-3 to n+12)
        mm_fwd0 = _mm_srli_si128(mm_v00,13);
        mm_fwd1 = _mm_srli_si128(mm_v10,13);
        // mm_fwd0 and mm_fwd1 are the 2 vertically combined pixels that will be carried forward.
        // Now, we need to horizontally combine the 2 sets of combined pixels
        mm_tmp  = _mm_avg_epu8(mm_v00,mm_w00);
        mm_vh00 = _mm_avg_epu8(mm_v00,mm_tmp);
        mm_wh00 = _mm_avg_epu8(mm_w00,mm_tmp);
        mm_tmp  = _mm_avg_epu8(mm_v10,mm_w10);
        mm_vh10 = _mm_avg_epu8(mm_v10,mm_tmp);
        mm_wh10 = _mm_avg_epu8(mm_w10,mm_tmp);
        // So our output pixel values are in mm_vh00/mv_wh00 and mm_vh10/mm_wh10, but in the wrong order
        mm_tmp = _mm_shuffle_epi8(mm_wh00, mm_shuf1);
        mm_d00 = _mm_shuffle_epi8(mm_vh00, mm_shuf2);
        mm_d00 = _mm_or_si128(mm_d00, mm_tmp);
        _mm_storeu_si128((__m128i *)d0, mm_d00);
        mm_tmp = _mm_shuffle_epi8(mm_wh10, mm_shuf1);
        mm_d10 = _mm_shuffle_epi8(mm_vh10, mm_shuf2);
        mm_d10 = _mm_or_si128(mm_d10, mm_tmp);
        _mm_storeu_si128((__m128i *)d1, mm_d10);
        mm_tmp = _mm_shuffle_epi8(mm_wh00, mm_shuf3);
        mm_d01 = _mm_shuffle_epi8(mm_vh00, mm_shuf4);
        mm_d01 = _mm_or_si128(mm_d01, mm_tmp);
        mm_tmp = _mm_shuffle_epi8(mm_wh10, mm_shuf3);
        mm_d11 = _mm_shuffle_epi8(mm_vh10, mm_shuf4);
        mm_d11 = _mm_or_si128(mm_d11, mm_tmp);
        // Unstored results are in mm_d01 and mm_d11 - both need the top byte filling in.
        mm_vh00 = _mm_srli_si128(mm_vh00, 15);
        mm_vh10 = _mm_srli_si128(mm_vh10, 15);
        // We need to carry the top byte from mm_vh00 and mm_vh10 forward into the next results word.
        // Load raw pixels into mm_s01 and mm_s11 (source pixel bytes n+16 to n+31)
        mm_s01 = _mm_loadu_si128((const __m128i *)(s0+16));
        mm_s11 = _mm_loadu_si128((const __m128i *)(s1+16));
        // Combine vertically
        mm_tmp = _mm_avg_epu8(mm_s01,mm_s11);
        mm_v01 = _mm_avg_epu8(mm_s01,mm_tmp);
        mm_v11 = _mm_avg_epu8(mm_s11,mm_tmp);
        // mm_v00 and mm_v01 are vertically combined pixel bytes (n+16 to n+31).
        mm_w01 = _mm_slli_si128(mm_v01,3);
        mm_w11 = _mm_slli_si128(mm_v11,3);
        mm_w01 = _mm_or_si128(mm_w01,mm_fwd0);
        mm_w11 = _mm_or_si128(mm_w11,mm_fwd1);
        // mm_w01 and mm_w11 are vertically combined pixel bytes, delayed by a pixel (n+13 to n+28)
        mm_fwd0 = _mm_srli_si128(mm_v01,13);
        mm_fwd1 = _mm_srli_si128(mm_v11,13);
        // Now, we need to horizontally combine the 2 sets of combined pixels
        mm_tmp  = _mm_avg_epu8(mm_v01,mm_w01);
        mm_vh01 = _mm_avg_epu8(mm_v01,mm_tmp);
        mm_wh01 = _mm_avg_epu8(mm_w01,mm_tmp);
        mm_tmp  = _mm_avg_epu8(mm_v11,mm_w11);
        mm_vh11 = _mm_avg_epu8(mm_v11,mm_tmp);
        mm_wh11 = _mm_avg_epu8(mm_w11,mm_tmp);
        // So our output pixel values are in mm_vh01/mm_wh01 and mm_vh11/mm_wh11, but in the wrong order
        // Finish off the previous results words
        mm_tmp = _mm_slli_si128(mm_wh01,15);
        mm_d01 = _mm_or_si128(mm_d01,mm_tmp);
        _mm_storeu_si128((__m128i *)(d0+16), mm_d01);
        mm_tmp = _mm_slli_si128(mm_wh11,15);
        mm_d11 = _mm_or_si128(mm_d11,mm_tmp);
        _mm_storeu_si128((__m128i *)(d1+16), mm_d11);
        // Now reorder results.
        mm_tmp = _mm_shuffle_epi8(mm_wh01, mm_shuf5);
        mm_d02 = _mm_shuffle_epi8(mm_vh01, mm_shuf6);
        mm_d02 = _mm_or_si128(mm_d02, mm_tmp);
        mm_vh00 = _mm_slli_si128(mm_vh00, 1);
        mm_d02 = _mm_or_si128(mm_d02, mm_vh00);
        _mm_storeu_si128((__m128i *)(d0+32), mm_d02);
        mm_tmp = _mm_shuffle_epi8(mm_wh11, mm_shuf5);
        mm_d12 = _mm_shuffle_epi8(mm_vh11, mm_shuf6);
        mm_d12 = _mm_or_si128(mm_d12, mm_tmp);
        mm_vh10 = _mm_slli_si128(mm_vh10, 1);
        mm_d12 = _mm_or_si128(mm_d12, mm_vh10);
        _mm_storeu_si128((__m128i *)(d1+32), mm_d12);
        mm_tmp = _mm_shuffle_epi8(mm_wh01, mm_shuf7);
        mm_d03 = _mm_shuffle_epi8(mm_vh01, mm_shuf8);
        mm_d03 = _mm_or_si128(mm_d03, mm_tmp);
        mm_tmp = _mm_shuffle_epi8(mm_wh11, mm_shuf7);
        mm_d13 = _mm_shuffle_epi8(mm_vh11, mm_shuf8);
        mm_d13 = _mm_or_si128(mm_d13, mm_tmp);
        // Unstored results are in mm_d03 and mm_d13 - both need the next to top byte filling in.
        mm_vh01 = _mm_srli_si128(mm_vh01, 15);
        mm_vh11 = _mm_srli_si128(mm_vh11, 15);
        // We need to carry the top byte from mm_vh00 and mm_vh10 forward into the next results word.
        // Load raw pixels into mm_s02 and mm_s12 (source pixel bytes n+32 to n+47)
        mm_s02 = _mm_loadu_si128((const __m128i *)(s0+32));
        mm_s12 = _mm_loadu_si128((const __m128i *)(s1+32));
        // Combine vertically
        mm_tmp = _mm_avg_epu8(mm_s02,mm_s12);
        mm_v02 = _mm_avg_epu8(mm_s02,mm_tmp);
        mm_v12 = _mm_avg_epu8(mm_s12,mm_tmp);
        // mm_v02 and mm_v12 are vertically combined pixels (n+32 to n+47).
        mm_w02 = _mm_slli_si128(mm_v02,3);
        mm_w12 = _mm_slli_si128(mm_v12,3);
        mm_w02 = _mm_or_si128(mm_w02,mm_fwd0);
        mm_w12 = _mm_or_si128(mm_w12,mm_fwd1);
        // mm_w02 and mm_w12 are vertically combined pixels delayed by a pixel (n+29 to n+44)
        mm_fwd0 = _mm_srli_si128(mm_v02,13);
        mm_fwd1 = _mm_srli_si128(mm_v12,13);
        // Now, we need to horizontally combine the 2 sets of combined pixels
        mm_tmp  = _mm_avg_epu8(mm_v02,mm_w02);
        mm_vh02 = _mm_avg_epu8(mm_v02,mm_tmp);
        mm_wh02 = _mm_avg_epu8(mm_w02,mm_tmp);
        mm_tmp  = _mm_avg_epu8(mm_v12,mm_w12);
        mm_vh12 = _mm_avg_epu8(mm_v12,mm_tmp);
        mm_wh12 = _mm_avg_epu8(mm_w12,mm_tmp);
        // So our output pixel values are in mm_v02/mm_w02 and mm_v12/mm_w12, but in the wrong order
        // Finish off the previous results words
        mm_tmp = _mm_slli_si128(mm_wh02,15);
        mm_tmp = _mm_srli_si128(mm_tmp,1);
        mm_d03 = _mm_or_si128(mm_d03,mm_tmp);
        _mm_storeu_si128((__m128i *)(d0+48), mm_d03);
        mm_tmp = _mm_slli_si128(mm_wh12,15);
        mm_tmp = _mm_srli_si128(mm_tmp,1);
        mm_d13 = _mm_or_si128(mm_d13,mm_tmp);
        _mm_storeu_si128((__m128i *)(d1+48), mm_d13);
        // Now reorder results.
        mm_tmp = _mm_shuffle_epi8(mm_wh02, mm_shuf9);
        mm_d04 = _mm_shuffle_epi8(mm_vh02, mm_shuf10);
        mm_d04 = _mm_or_si128(mm_d04, mm_tmp);
        mm_d04 = _mm_or_si128(mm_d04, mm_vh01);
        _mm_storeu_si128((__m128i *)(d0+64), mm_d04);
        mm_tmp = _mm_shuffle_epi8(mm_wh12, mm_shuf9);
        mm_d14 = _mm_shuffle_epi8(mm_vh12, mm_shuf10);
        mm_d14 = _mm_or_si128(mm_d14, mm_tmp);
        mm_d14 = _mm_or_si128(mm_d14, mm_vh11);
        _mm_storeu_si128((__m128i *)(d1+64), mm_d14);
        mm_tmp = _mm_shuffle_epi8(mm_wh02, mm_shuf11);
        mm_d05 = _mm_shuffle_epi8(mm_vh02, mm_shuf12);
        mm_d05 = _mm_or_si128(mm_d05, mm_tmp);
        _mm_storeu_si128((__m128i *)(d0+80), mm_d05);
        mm_tmp = _mm_shuffle_epi8(mm_wh12, mm_shuf11);
        mm_d15 = _mm_shuffle_epi8(mm_vh12, mm_shuf12);
        mm_d15 = _mm_or_si128(mm_d15, mm_tmp);
        _mm_storeu_si128((__m128i *)(d1+80), mm_d15);
        d0 += 96;
        d1 += 96;
        s0 += 48;
        s1 += 48;
    }

    w += 16;
    if (w)
    {
        __m128i mm_s00, mm_s01, mm_s02, mm_s10, mm_s11, mm_s12;
        __m128i mm_d00, mm_d01, mm_d02, mm_d03, mm_d04, mm_d05;
        __m128i mm_d10, mm_d11, mm_d12, mm_d13, mm_d14, mm_d15;
        __m128i mm_v00, mm_v01, mm_v02, mm_v10, mm_v11, mm_v12;
        __m128i mm_w00, mm_w01, mm_w02, mm_w10, mm_w11, mm_w12;
        __m128i mm_vh00, mm_vh01, mm_vh02, mm_vh10, mm_vh11, mm_vh12;
        __m128i mm_wh00, mm_wh01, mm_wh02, mm_wh10, mm_wh11, mm_wh12;
        __m128i mm_tmp;
        uint8_t local0[16*3], local1[16*3];

        memcpy(local0, s0, (w+1)*3);
        memcpy(local1, s1, (w+1)*3);
        // Load raw pixels into mm0 and mm1 (source pixel bytes n to n+15)
        mm_s00 = _mm_loadu_si128((const __m128i *)local0);
        mm_s10 = _mm_loadu_si128((const __m128i *)local1);
        // Combine vertically
        mm_tmp = _mm_avg_epu8(mm_s00,mm_s10);
        mm_v00 = _mm_avg_epu8(mm_s00,mm_tmp);
        mm_v10 = _mm_avg_epu8(mm_s10,mm_tmp);
        // mm_v00 and mm_v10 are vertically combined pixel bytes (n to n+15).
        mm_w00 = _mm_slli_si128(mm_v00,3);
        mm_w10 = _mm_slli_si128(mm_v10,3);
        mm_w00 = _mm_or_si128(mm_w00,mm_fwd0);
        mm_w10 = _mm_or_si128(mm_w10,mm_fwd1);
        // mm_w00 and mm_w01 are vertically combined pixel bytes, delayed by a pixel (n-3 to n+12)
        mm_fwd0 = _mm_srli_si128(mm_v00,13);
        mm_fwd1 = _mm_srli_si128(mm_v10,13);
        // mm_fwd0 and mm_fwd1 are the 2 vertically combined pixels that will be carried forward.
        // Now, we need to horizontally combine the 2 sets of combined pixels
        mm_tmp  = _mm_avg_epu8(mm_v00,mm_w00);
        mm_vh00 = _mm_avg_epu8(mm_v00,mm_tmp);
        mm_wh00 = _mm_avg_epu8(mm_w00,mm_tmp);
        mm_tmp  = _mm_avg_epu8(mm_v10,mm_w10);
        mm_vh10 = _mm_avg_epu8(mm_v10,mm_tmp);
        mm_wh10 = _mm_avg_epu8(mm_w10,mm_tmp);
        // So our output pixel values are in mm_vh00/mv_wh00 and mm_vh10/mm_wh10, but in the wrong order
        mm_tmp = _mm_shuffle_epi8(mm_wh00, mm_shuf1);
        mm_d00 = _mm_shuffle_epi8(mm_vh00, mm_shuf2);
        mm_d00 = _mm_or_si128(mm_d00, mm_tmp);
        mm_tmp = _mm_shuffle_epi8(mm_wh10, mm_shuf1);
        mm_d10 = _mm_shuffle_epi8(mm_vh10, mm_shuf2);
        mm_d10 = _mm_or_si128(mm_d10, mm_tmp);
        switch (w)
        {
        default:
            ((uint32_t *)d0)[ 3] = _mm_extract_epi32(mm_d00, 3);
            ((uint32_t *)d1)[ 3] = _mm_extract_epi32(mm_d10, 3);
        case  2:
            ((uint32_t *)d0)[ 2] = _mm_extract_epi32(mm_d00, 2);
            ((uint16_t *)d0)[ 3] = _mm_extract_epi16(mm_d00, 3);
            ((uint32_t *)d1)[ 2] = _mm_extract_epi32(mm_d10, 2);
            ((uint16_t *)d1)[ 3] = _mm_extract_epi16(mm_d10, 3);
        case  1:
            ((uint16_t *)d0)[ 2] = _mm_extract_epi16(mm_d00, 2);
            ((uint32_t *)d0)[ 0] = _mm_extract_epi32(mm_d00, 0);
            ((uint16_t *)d1)[ 2] = _mm_extract_epi16(mm_d10, 2);
            ((uint32_t *)d1)[ 0] = _mm_extract_epi32(mm_d10, 0);
        }
        mm_tmp = _mm_shuffle_epi8(mm_wh00, mm_shuf3);
        mm_d01 = _mm_shuffle_epi8(mm_vh00, mm_shuf4);
        mm_d01 = _mm_or_si128(mm_d01, mm_tmp);
        mm_tmp = _mm_shuffle_epi8(mm_wh10, mm_shuf3);
        mm_d11 = _mm_shuffle_epi8(mm_vh10, mm_shuf4);
        mm_d11 = _mm_or_si128(mm_d11, mm_tmp);
        // Unstored results are in mm_d01 and mm_d11 - both need the top byte filling in.
        mm_vh00 = _mm_srli_si128(mm_vh00, 15);
        mm_vh10 = _mm_srli_si128(mm_vh10, 15);
        // We need to carry the top byte from mm_vh00 and mm_vh10 forward into the next results word.
        // Load raw pixels into mm_s01 and mm_s11 (source pixel bytes n+16 to n+31)
        mm_s01 = _mm_loadu_si128((const __m128i *)(local0+16));
        mm_s11 = _mm_loadu_si128((const __m128i *)(local1+16));
        // Combine vertically
        mm_tmp = _mm_avg_epu8(mm_s01,mm_s11);
        mm_v01 = _mm_avg_epu8(mm_s01,mm_tmp);
        mm_v11 = _mm_avg_epu8(mm_s11,mm_tmp);
        // mm_v00 and mm_v01 are vertically combined pixel bytes (n+16 to n+31).
        mm_w01 = _mm_slli_si128(mm_v01,3);
        mm_w11 = _mm_slli_si128(mm_v11,3);
        mm_w01 = _mm_or_si128(mm_w01,mm_fwd0);
        mm_w11 = _mm_or_si128(mm_w11,mm_fwd1);
        // mm_w01 and mm_w11 are vertically combined pixel bytes, delayed by a pixel (n+13 to n+28)
        mm_fwd0 = _mm_srli_si128(mm_v01,13);
        mm_fwd1 = _mm_srli_si128(mm_v11,13);
        // Now, we need to horizontally combine the 2 sets of combined pixels
        mm_tmp  = _mm_avg_epu8(mm_v01,mm_w01);
        mm_vh01 = _mm_avg_epu8(mm_v01,mm_tmp);
        mm_wh01 = _mm_avg_epu8(mm_w01,mm_tmp);
        mm_tmp  = _mm_avg_epu8(mm_v11,mm_w11);
        mm_vh11 = _mm_avg_epu8(mm_v11,mm_tmp);
        mm_wh11 = _mm_avg_epu8(mm_w11,mm_tmp);
        // So our output pixel values are in mm_vh01/mm_wh01 and mm_vh11/mm_wh11, but in the wrong order
        // Finish off the previous results words
        mm_tmp = _mm_slli_si128(mm_wh01,15);
        mm_d01 = _mm_or_si128(mm_d01,mm_tmp);
        mm_tmp = _mm_slli_si128(mm_wh11,15);
        mm_d11 = _mm_or_si128(mm_d11,mm_tmp);
        switch (w)
        {
        default:
            ((uint16_t *)d0)[15] = _mm_extract_epi16(mm_d01, 7);
            ((uint16_t *)d1)[15] = _mm_extract_epi16(mm_d11, 7);
        case  5:
            ((uint16_t *)d0)[14] = _mm_extract_epi16(mm_d01, 6);
            ((uint32_t *)d0)[ 6] = _mm_extract_epi32(mm_d01, 2);
            ((uint16_t *)d1)[14] = _mm_extract_epi16(mm_d11, 6);
            ((uint32_t *)d1)[ 6] = _mm_extract_epi32(mm_d11, 2);
        case  4:
            ((uint32_t *)d0)[ 5] = _mm_extract_epi32(mm_d01, 1);
            ((uint16_t *)d0)[ 9] = _mm_extract_epi16(mm_d01, 1);
            ((uint32_t *)d1)[ 5] = _mm_extract_epi32(mm_d11, 1);
            ((uint16_t *)d1)[ 9] = _mm_extract_epi16(mm_d11, 1);
        case  3:
            ((uint16_t *)d0)[ 8] = _mm_extract_epi16(mm_d01, 0);
            ((uint16_t *)d1)[ 8] = _mm_extract_epi16(mm_d11, 0);
        case 2: case 1: {/* Nothing to do */}
        }
        // Now reorder results.
        mm_tmp = _mm_shuffle_epi8(mm_wh01, mm_shuf5);
        mm_d02 = _mm_shuffle_epi8(mm_vh01, mm_shuf6);
        mm_d02 = _mm_or_si128(mm_d02, mm_tmp);
        mm_vh00 = _mm_slli_si128(mm_vh00, 1);
        mm_d02 = _mm_or_si128(mm_d02, mm_vh00);
        mm_tmp = _mm_shuffle_epi8(mm_wh11, mm_shuf5);
        mm_d12 = _mm_shuffle_epi8(mm_vh11, mm_shuf6);
        mm_d12 = _mm_or_si128(mm_d12, mm_tmp);
        mm_vh10 = _mm_slli_si128(mm_vh10, 1);
        mm_d12 = _mm_or_si128(mm_d12, mm_vh10);
        switch (w)
        {
        default:
        case 8:
            ((uint32_t *)d0)[11] = _mm_extract_epi32(mm_d02, 3);
            ((uint16_t *)d0)[21] = _mm_extract_epi16(mm_d02, 5);
            ((uint32_t *)d1)[11] = _mm_extract_epi32(mm_d12, 3);
            ((uint16_t *)d1)[21] = _mm_extract_epi16(mm_d12, 5);
        case 7:
            ((uint16_t *)d0)[20] = _mm_extract_epi16(mm_d02, 4);
            ((uint32_t *)d0)[ 9] = _mm_extract_epi32(mm_d02, 1);
            ((uint16_t *)d1)[20] = _mm_extract_epi16(mm_d12, 4);
            ((uint32_t *)d1)[ 9] = _mm_extract_epi32(mm_d12, 1);
        case 6:
            ((uint32_t *)d0)[ 8] = _mm_extract_epi32(mm_d02, 0);
            ((uint32_t *)d1)[ 8] = _mm_extract_epi32(mm_d12, 0);
        case 5: case 4: case 3: case 2: case 1: {/* Nothing to do */}
        }
        mm_tmp = _mm_shuffle_epi8(mm_wh01, mm_shuf7);
        mm_d03 = _mm_shuffle_epi8(mm_vh01, mm_shuf8);
        mm_d03 = _mm_or_si128(mm_d03, mm_tmp);
        mm_tmp = _mm_shuffle_epi8(mm_wh11, mm_shuf7);
        mm_d13 = _mm_shuffle_epi8(mm_vh11, mm_shuf8);
        mm_d13 = _mm_or_si128(mm_d13, mm_tmp);
        // Unstored results are in mm_d03 and mm_d13 - both need the next to top byte filling in.
        mm_vh01 = _mm_srli_si128(mm_vh01, 15);
        mm_vh11 = _mm_srli_si128(mm_vh11, 15);
        // We need to carry the top byte from mm_vh00 and mm_vh10 forward into the next results word.
        // Load raw pixels into mm_s02 and mm_s12 (source pixel bytes n+32 to n+47)
        mm_s02 = _mm_loadu_si128((const __m128i *)(local0+32));
        mm_s12 = _mm_loadu_si128((const __m128i *)(local1+32));
        // Combine vertically
        mm_tmp = _mm_avg_epu8(mm_s02,mm_s12);
        mm_v02 = _mm_avg_epu8(mm_s02,mm_tmp);
        mm_v12 = _mm_avg_epu8(mm_s12,mm_tmp);
        // mm_v02 and mm_v12 are vertically combined pixels (n+32 to n+47).
        mm_w02 = _mm_slli_si128(mm_v02,3);
        mm_w12 = _mm_slli_si128(mm_v12,3);
        mm_w02 = _mm_or_si128(mm_w02,mm_fwd0);
        mm_w12 = _mm_or_si128(mm_w12,mm_fwd1);
        // mm_w02 and mm_w12 are vertically combined pixels delayed by a pixel (n+29 to n+44)
        switch (w)
        {
        case 15:
            mm_fwd0 = _mm_srli_si128(mm_v02,10);
            mm_fwd1 = _mm_srli_si128(mm_v12,10);
            break;
        case 14:
            mm_fwd0 = _mm_srli_si128(mm_v02,7);
            mm_fwd1 = _mm_srli_si128(mm_v12,7);
            break;
        case 13:
            mm_fwd0 = _mm_srli_si128(mm_v02,4);
            mm_fwd1 = _mm_srli_si128(mm_v12,4);
            break;
        case 12:
            mm_fwd0 = _mm_srli_si128(mm_v02,1);
            mm_fwd1 = _mm_srli_si128(mm_v12,1);
            break;
        case 11:
            mm_fwd0 = _mm_or_si128(_mm_srli_si128(mm_v01,14), _mm_slli_si128(mm_v02,2));
            mm_fwd1 = _mm_or_si128(_mm_srli_si128(mm_v11,14), _mm_slli_si128(mm_v12,2));
            break;
        case 10:
            mm_fwd0 = _mm_srli_si128(mm_v01,11);
            mm_fwd1 = _mm_srli_si128(mm_v11,11);
            break;
        case  9:
            mm_fwd0 = _mm_srli_si128(mm_v01,8);
            mm_fwd1 = _mm_srli_si128(mm_v11,8);
            break;
        case  8:
            mm_fwd0 = _mm_srli_si128(mm_v01,5);
            mm_fwd1 = _mm_srli_si128(mm_v11,5);
            break;
        case  7:
            mm_fwd0 = _mm_srli_si128(mm_v01,2);
            mm_fwd1 = _mm_srli_si128(mm_v11,2);
            break;
        case  6:
            mm_fwd0 = _mm_or_si128(_mm_srli_si128(mm_v00,15), _mm_slli_si128(mm_v01,1));
            mm_fwd1 = _mm_or_si128(_mm_srli_si128(mm_v10,15), _mm_slli_si128(mm_v11,1));
            break;
        case  5:
            mm_fwd0 = _mm_srli_si128(mm_v00,12);
            mm_fwd1 = _mm_srli_si128(mm_v10,12);
            break;
        case  4:
            mm_fwd0 = _mm_srli_si128(mm_v00,9);
            mm_fwd1 = _mm_srli_si128(mm_v10,9);
            break;
        case  3:
            mm_fwd0 = _mm_srli_si128(mm_v00,6);
            mm_fwd1 = _mm_srli_si128(mm_v10,6);
            break;
        case  2:
            mm_fwd0 = _mm_srli_si128(mm_v00,3);
            mm_fwd1 = _mm_srli_si128(mm_v10,3);
            break;
        case  1:
            mm_fwd0 = mm_v00;
            mm_fwd1 = mm_v10;
            break;
        }
        // Now, we need to horizontally combine the 2 sets of combined pixels
        mm_tmp  = _mm_avg_epu8(mm_v02,mm_w02);
        mm_vh02 = _mm_avg_epu8(mm_v02,mm_tmp);
        mm_wh02 = _mm_avg_epu8(mm_w02,mm_tmp);
        mm_tmp  = _mm_avg_epu8(mm_v12,mm_w12);
        mm_vh12 = _mm_avg_epu8(mm_v12,mm_tmp);
        mm_wh12 = _mm_avg_epu8(mm_w12,mm_tmp);
        // So our output pixel values are in mm_v02/mm_w02 and mm_v12/mm_w12, but in the wrong order
        // Finish off the previous results words
        mm_tmp = _mm_slli_si128(mm_wh02,15);
        mm_tmp = _mm_srli_si128(mm_tmp,1);
        mm_d03 = _mm_or_si128(mm_d03,mm_tmp);
        mm_tmp = _mm_slli_si128(mm_wh12,15);
        mm_tmp = _mm_srli_si128(mm_tmp,1);
        mm_d13 = _mm_or_si128(mm_d13,mm_tmp);
        switch (w)
        {
        default:
            ((uint32_t *)d0)[15] = _mm_extract_epi32(mm_d03, 3);
            ((uint32_t *)d1)[15] = _mm_extract_epi32(mm_d13, 3);
        case 10:
            ((uint32_t *)d0)[14] = _mm_extract_epi32(mm_d03, 2);
            ((uint16_t *)d0)[27] = _mm_extract_epi16(mm_d03, 3);
            ((uint32_t *)d1)[14] = _mm_extract_epi32(mm_d13, 2);
            ((uint16_t *)d1)[27] = _mm_extract_epi16(mm_d13, 3);
        case 9:
            ((uint16_t *)d0)[26] = _mm_extract_epi16(mm_d03, 2);
            ((uint32_t *)d0)[12] = _mm_extract_epi32(mm_d03, 0);
            ((uint16_t *)d1)[26] = _mm_extract_epi16(mm_d13, 2);
            ((uint32_t *)d1)[12] = _mm_extract_epi32(mm_d13, 0);
        case 8: case 7: case 6: case 5: case 4: case 3: case 2: case 1: {/* Nothing to do */}
        }
        // Now reorder results.
        mm_tmp = _mm_shuffle_epi8(mm_wh02, mm_shuf9);
        mm_d04 = _mm_shuffle_epi8(mm_vh02, mm_shuf10);
        mm_d04 = _mm_or_si128(mm_d04, mm_tmp);
        mm_d04 = _mm_or_si128(mm_d04, mm_vh01);
        mm_tmp = _mm_shuffle_epi8(mm_wh12, mm_shuf9);
        mm_d14 = _mm_shuffle_epi8(mm_vh12, mm_shuf10);
        mm_d14 = _mm_or_si128(mm_d14, mm_tmp);
        mm_d14 = _mm_or_si128(mm_d14, mm_vh11);
        switch (w)
        {
        default:
            ((uint16_t *)d0)[39] = _mm_extract_epi16(mm_d04, 7);
            ((uint16_t *)d1)[39] = _mm_extract_epi16(mm_d14, 7);
        case 13:
            ((uint16_t *)d0)[38] = _mm_extract_epi16(mm_d04, 6);
            ((uint32_t *)d0)[18] = _mm_extract_epi32(mm_d04, 2);
            ((uint16_t *)d1)[38] = _mm_extract_epi16(mm_d14, 6);
            ((uint32_t *)d1)[18] = _mm_extract_epi32(mm_d14, 2);
        case 12:
            ((uint32_t *)d0)[17] = _mm_extract_epi32(mm_d04, 1);
            ((uint16_t *)d0)[33] = _mm_extract_epi16(mm_d04, 1);
            ((uint32_t *)d1)[17] = _mm_extract_epi32(mm_d14, 1);
            ((uint16_t *)d1)[33] = _mm_extract_epi16(mm_d14, 1);
        case 11:
            ((uint16_t *)d0)[32] = _mm_extract_epi16(mm_d04, 0);
            ((uint16_t *)d1)[32] = _mm_extract_epi16(mm_d14, 0);
        case 10: case 9: case 8: case 7: case 6: case 5: case 4: case 3: case 2: case 1: {/* Nothing to do */}
        }
        mm_tmp = _mm_shuffle_epi8(mm_wh02, mm_shuf11);
        mm_d05 = _mm_shuffle_epi8(mm_vh02, mm_shuf12);
        mm_d05 = _mm_or_si128(mm_d05, mm_tmp);
        mm_tmp = _mm_shuffle_epi8(mm_wh12, mm_shuf11);
        mm_d15 = _mm_shuffle_epi8(mm_vh12, mm_shuf12);
        mm_d15 = _mm_or_si128(mm_d15, mm_tmp);
        switch (w)
        {
        case 15:
            ((uint16_t *)d0)[44] = _mm_extract_epi16(mm_d05, 4);
            ((uint32_t *)d0)[21] = _mm_extract_epi32(mm_d05, 1);
            ((uint16_t *)d1)[44] = _mm_extract_epi16(mm_d15, 4);
            ((uint32_t *)d1)[21] = _mm_extract_epi32(mm_d15, 1);
        case 14:
            ((uint32_t *)d0)[20] = _mm_extract_epi32(mm_d05, 0);
            ((uint32_t *)d1)[20] = _mm_extract_epi32(mm_d15, 0);
        default: {/* Nothing to do */}
        }
        d0 += w*6;
        d1 += w*6;
    }
    /* Trailing single pixel */
    d0[0] = _mm_extract_epi8(mm_fwd0,0);
    d0[1] = _mm_extract_epi8(mm_fwd0,1);
    d0[2] = _mm_extract_epi8(mm_fwd0,2);
    d1[0] = _mm_extract_epi8(mm_fwd1,0);
    d1[1] = _mm_extract_epi8(mm_fwd1,1);
    d1[2] = _mm_extract_epi8(mm_fwd1,2);

    return 2;
}

static int
double_interp3_top_sse(uint8_t       ** ipa_restrict dsts,
                       const uint8_t ** ipa_restrict srcs,
                       ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    const uint8_t * ipa_restrict s0 = srcs[0];
    DECLARE(tl);
    __m128i mm_shuf1, mm_shuf2, mm_shuf3, mm_shuf4;
    __m128i mm_shuf5, mm_shuf6, mm_shuf7, mm_shuf8;
    __m128i mm_shuf9, mm_shuf10,mm_shuf11,mm_shuf12;
    __m128i mm_fwd0;

    /* Leading single pixel */
    LOAD(tl,s0);
    STORE(tl,d0);

    mm_fwd0 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,tl_b,tl_g,tl_r);

    // d00 and d10 are combined using the following:
    mm_shuf1 = _mm_set_epi8(ZZ, 8, 7, 6,ZZ,ZZ,ZZ, 5, 4, 3,ZZ,ZZ,ZZ, 2, 1, 0);
    mm_shuf2 = _mm_set_epi8( 6,ZZ,ZZ,ZZ, 5, 4, 3,ZZ,ZZ,ZZ, 2, 1, 0,ZZ,ZZ,ZZ);
    // d01 and d11 are combined using the following:
    mm_shuf3 = _mm_set_epi8(ZZ,15,ZZ,ZZ,ZZ,14,13,12,ZZ,ZZ,ZZ,11,10, 9,ZZ,ZZ);
    //                      ^^ 16 - won't be available yet.
    mm_shuf4 = _mm_set_epi8(ZZ,ZZ,14,13,12,ZZ,ZZ,ZZ,11,10, 9,ZZ,ZZ,ZZ, 8, 7);
    // d02 and d12 are combined using the following:
    mm_shuf5 = _mm_set_epi8(ZZ,ZZ,ZZ,23,22,21,ZZ,ZZ,ZZ,20,19,18,ZZ,ZZ,ZZ,17);
    mm_shuf6 = _mm_set_epi8(23,22,21,ZZ,ZZ,ZZ,20,19,18,ZZ,ZZ,ZZ,17,16,ZZ,ZZ);
    //                                    15 carried over from before ^^
    // d03 and d13 are combined using the following:
    mm_shuf7 = _mm_set_epi8(ZZ,ZZ,31,30,ZZ,ZZ,ZZ,29,28,27,ZZ,ZZ,ZZ,26,25,24);
    //                         ^^ 32 - won't be available yet
    mm_shuf8 = _mm_set_epi8(30,ZZ,ZZ,ZZ,29,28,27,ZZ,ZZ,ZZ,26,25,24,ZZ,ZZ,ZZ);
    // d04 and d14 are combined using the following:
    mm_shuf9 = _mm_set_epi8(40,39,ZZ,ZZ,ZZ,38,37,36,ZZ,ZZ,ZZ,35,34,33,ZZ,ZZ);
    mm_shuf10= _mm_set_epi8(ZZ,ZZ,38,37,36,ZZ,ZZ,ZZ,35,34,33,ZZ,ZZ,ZZ,32,ZZ);
    //                                       31 carried over from before ^^
    // d05 and d15 are combined using the following:
    mm_shuf11= _mm_set_epi8(ZZ,ZZ,ZZ,47,46,45,ZZ,ZZ,ZZ,44,43,42,ZZ,ZZ,ZZ,41);
    mm_shuf12= _mm_set_epi8(47,46,45,ZZ,ZZ,ZZ,44,43,42,ZZ,ZZ,ZZ,41,40,39,ZZ);
    for (w = doubler->src_w-17; w >= 0; w -= 16)
    {
        __m128i mm_s00, mm_s01, mm_s02;
        __m128i mm_d00, mm_d01, mm_d02, mm_d03, mm_d04, mm_d05;
        __m128i mm_w00, mm_w01, mm_w02;
        __m128i mm_vh00, mm_vh01, mm_vh02;
        __m128i mm_wh00, mm_wh01, mm_wh02;
        __m128i mm_tmp;

        // Load raw pixels into mm0 and mm1 (source pixel bytes n to n+15)
        mm_s00 = _mm_loadu_si128((const __m128i *)s0);
        mm_w00 = _mm_slli_si128(mm_s00,3);
        mm_w00 = _mm_or_si128(mm_w00,mm_fwd0);
        // mm_w00 are vertically combined pixel bytes, delayed by a pixel (n-3 to n+12)
        mm_fwd0 = _mm_srli_si128(mm_s00,13);
        // mm_fwd0 are the pixels that will be carried forward.
        // Now, we need to horizontally combine
        mm_tmp  = _mm_avg_epu8(mm_s00,mm_w00);
        mm_vh00 = _mm_avg_epu8(mm_s00,mm_tmp);
        mm_wh00 = _mm_avg_epu8(mm_w00,mm_tmp);
        // So our output pixel values are in mm_vh00/mv_wh00, but in the wrong order
        mm_tmp = _mm_shuffle_epi8(mm_wh00, mm_shuf1);
        mm_d00 = _mm_shuffle_epi8(mm_vh00, mm_shuf2);
        mm_d00 = _mm_or_si128(mm_d00, mm_tmp);
        _mm_storeu_si128((__m128i *)d0, mm_d00);
        mm_tmp = _mm_shuffle_epi8(mm_wh00, mm_shuf3);
        mm_d01 = _mm_shuffle_epi8(mm_vh00, mm_shuf4);
        mm_d01 = _mm_or_si128(mm_d01, mm_tmp);
        // Unstored results are in mm_d01 - both need the top byte filling in.
        mm_vh00 = _mm_srli_si128(mm_vh00, 15);
        // We need to carry the top byte from mm_vh00 forward into the next results word.
        // Load raw pixels into mm_s01 and mm_s11 (source pixel bytes n+16 to n+31)
        mm_s01 = _mm_loadu_si128((const __m128i *)(s0+16));
        // mm_v00 and mm_v01 are vertically combined pixel bytes (n+16 to n+31).
        mm_w01 = _mm_slli_si128(mm_s01,3);
        mm_w01 = _mm_or_si128(mm_w01,mm_fwd0);
        // mm_w01 and mm_w11 are vertically combined pixel bytes, delayed by a pixel (n+13 to n+28)
        mm_fwd0 = _mm_srli_si128(mm_s01,13);
        // Now, we need to horizontally combine the 2 sets of combined pixels
        mm_tmp  = _mm_avg_epu8(mm_s01,mm_w01);
        mm_vh01 = _mm_avg_epu8(mm_s01,mm_tmp);
        mm_wh01 = _mm_avg_epu8(mm_w01,mm_tmp);
        // So our output pixel values are in mm_vh01/mm_wh01 and mm_vh11/mm_wh11, but in the wrong order
        // Finish off the previous results words
        mm_tmp = _mm_slli_si128(mm_wh01,15);
        mm_d01 = _mm_or_si128(mm_d01,mm_tmp);
        _mm_storeu_si128((__m128i *)(d0+16), mm_d01);
        // Now reorder results.
        mm_tmp = _mm_shuffle_epi8(mm_wh01, mm_shuf5);
        mm_d02 = _mm_shuffle_epi8(mm_vh01, mm_shuf6);
        mm_d02 = _mm_or_si128(mm_d02, mm_tmp);
        mm_vh00 = _mm_slli_si128(mm_vh00, 1);
        mm_d02 = _mm_or_si128(mm_d02, mm_vh00);
        _mm_storeu_si128((__m128i *)(d0+32), mm_d02);
        mm_tmp = _mm_shuffle_epi8(mm_wh01, mm_shuf7);
        mm_d03 = _mm_shuffle_epi8(mm_vh01, mm_shuf8);
        mm_d03 = _mm_or_si128(mm_d03, mm_tmp);
        // Unstored results are in mm_d03 - needs the next to top byte filling in.
        mm_vh01 = _mm_srli_si128(mm_vh01, 15);
        // We need to carry the top byte from mm_vh00 forward into the next results word.
        // Load raw pixels into mm_s02 and mm_s12 (source pixel bytes n+32 to n+47)
        mm_s02 = _mm_loadu_si128((const __m128i *)(s0+32));
        // mm_v02 are vertically combined pixels (n+32 to n+47).
        mm_w02 = _mm_slli_si128(mm_s02,3);
        mm_w02 = _mm_or_si128(mm_w02,mm_fwd0);
        // mm_w02 are vertically combined pixels delayed by a pixel (n+29 to n+44)
        mm_fwd0 = _mm_srli_si128(mm_s02,13);
        // Now, we need to horizontally combine
        mm_tmp  = _mm_avg_epu8(mm_s02,mm_w02);
        mm_vh02 = _mm_avg_epu8(mm_s02,mm_tmp);
        mm_wh02 = _mm_avg_epu8(mm_w02,mm_tmp);
        // So our output pixel values are in mm_v02/mm_w02, but in the wrong order
        // Finish off the previous results words
        mm_tmp = _mm_slli_si128(mm_wh02,15);
        mm_tmp = _mm_srli_si128(mm_tmp,1);
        mm_d03 = _mm_or_si128(mm_d03,mm_tmp);
        _mm_storeu_si128((__m128i *)(d0+48), mm_d03);
        // Now reorder results.
        mm_tmp = _mm_shuffle_epi8(mm_wh02, mm_shuf9);
        mm_d04 = _mm_shuffle_epi8(mm_vh02, mm_shuf10);
        mm_d04 = _mm_or_si128(mm_d04, mm_tmp);
        mm_d04 = _mm_or_si128(mm_d04, mm_vh01);
        _mm_storeu_si128((__m128i *)(d0+64), mm_d04);
        mm_tmp = _mm_shuffle_epi8(mm_wh02, mm_shuf11);
        mm_d05 = _mm_shuffle_epi8(mm_vh02, mm_shuf12);
        mm_d05 = _mm_or_si128(mm_d05, mm_tmp);
        _mm_storeu_si128((__m128i *)(d0+80), mm_d05);
        d0 += 96;
        s0 += 48;
    }

    w += 16;
    if (w)
    {
        __m128i mm_s00, mm_s01, mm_s02;
        __m128i mm_d00, mm_d01, mm_d02, mm_d03, mm_d04, mm_d05;
        __m128i mm_w00, mm_w01, mm_w02;
        __m128i mm_vh00, mm_vh01, mm_vh02;
        __m128i mm_wh00, mm_wh01, mm_wh02;
        __m128i mm_tmp;
        uint8_t local0[16*3];

        memcpy(local0, s0, (w+1)*3);
        // Load raw pixels into mm0 and mm1 (source pixel bytes n to n+15)
        mm_s00 = _mm_loadu_si128((const __m128i *)local0);
        mm_w00 = _mm_slli_si128(mm_s00,3);
        mm_w00 = _mm_or_si128(mm_w00,mm_fwd0);
        // mm_w00 are vertically combined pixel bytes, delayed by a pixel (n-3 to n+12)
        mm_fwd0 = _mm_srli_si128(mm_s00,13);
        // mm_fwd0 are the pixels that will be carried forward.
        // Now, we need to horizontally combine
        mm_tmp  = _mm_avg_epu8(mm_s00,mm_w00);
        mm_vh00 = _mm_avg_epu8(mm_s00,mm_tmp);
        mm_wh00 = _mm_avg_epu8(mm_w00,mm_tmp);
        // So our output pixel values are in mm_vh00/mv_wh00, but in the wrong order
        mm_tmp = _mm_shuffle_epi8(mm_wh00, mm_shuf1);
        mm_d00 = _mm_shuffle_epi8(mm_vh00, mm_shuf2);
        mm_d00 = _mm_or_si128(mm_d00, mm_tmp);
        switch (w)
        {
        default:
            ((uint32_t *)d0)[ 3] = _mm_extract_epi32(mm_d00, 3);
        case  2:
            ((uint32_t *)d0)[ 2] = _mm_extract_epi32(mm_d00, 2);
            ((uint16_t *)d0)[ 3] = _mm_extract_epi16(mm_d00, 3);
        case  1:
            ((uint16_t *)d0)[ 2] = _mm_extract_epi16(mm_d00, 2);
            ((uint32_t *)d0)[ 0] = _mm_extract_epi32(mm_d00, 0);
        }
        mm_tmp = _mm_shuffle_epi8(mm_wh00, mm_shuf3);
        mm_d01 = _mm_shuffle_epi8(mm_vh00, mm_shuf4);
        mm_d01 = _mm_or_si128(mm_d01, mm_tmp);
        // Unstored results are in mm_d01 - both need the top byte filling in.
        mm_vh00 = _mm_srli_si128(mm_vh00, 15);
        // We need to carry the top byte from mm_vh00 forward into the next results word.
        // Load raw pixels into mm_s01 and mm_s11 (source pixel bytes n+16 to n+31)
        mm_s01 = _mm_loadu_si128((const __m128i *)(local0+16));
        // mm_v00 and mm_v01 are vertically combined pixel bytes (n+16 to n+31).
        mm_w01 = _mm_slli_si128(mm_s01,3);
        mm_w01 = _mm_or_si128(mm_w01,mm_fwd0);
        // mm_w01 and mm_w11 are vertically combined pixel bytes, delayed by a pixel (n+13 to n+28)
        mm_fwd0 = _mm_srli_si128(mm_s01,13);
        // Now, we need to horizontally combine the 2 sets of combined pixels
        mm_tmp  = _mm_avg_epu8(mm_s01,mm_w01);
        mm_vh01 = _mm_avg_epu8(mm_s01,mm_tmp);
        mm_wh01 = _mm_avg_epu8(mm_w01,mm_tmp);
        // So our output pixel values are in mm_vh01/mm_wh01 and mm_vh11/mm_wh11, but in the wrong order
        // Finish off the previous results words
        mm_tmp = _mm_slli_si128(mm_wh01,15);
        mm_d01 = _mm_or_si128(mm_d01,mm_tmp);
        switch (w)
        {
        default:
            ((uint16_t *)d0)[15] = _mm_extract_epi16(mm_d01, 7);
        case 5:
            ((uint16_t *)d0)[14] = _mm_extract_epi16(mm_d01, 6);
            ((uint32_t *)d0)[ 6] = _mm_extract_epi32(mm_d01, 2);
        case  4:
            ((uint32_t *)d0)[ 5] = _mm_extract_epi32(mm_d01, 1);
            ((uint16_t *)d0)[ 9] = _mm_extract_epi16(mm_d01, 1);
        case  3:
            ((uint16_t *)d0)[ 8] = _mm_extract_epi16(mm_d01, 0);
        case 2: case 1: {/* Nothing to do */}
        }
        // Now reorder results.
        mm_tmp = _mm_shuffle_epi8(mm_wh01, mm_shuf5);
        mm_d02 = _mm_shuffle_epi8(mm_vh01, mm_shuf6);
        mm_d02 = _mm_or_si128(mm_d02, mm_tmp);
        mm_vh00 = _mm_slli_si128(mm_vh00, 1);
        mm_d02 = _mm_or_si128(mm_d02, mm_vh00);
        switch (w)
        {
        default:
        case 8:
            ((uint32_t *)d0)[11] = _mm_extract_epi32(mm_d02, 3);
            ((uint16_t *)d0)[21] = _mm_extract_epi16(mm_d02, 5);
        case 7:
            ((uint16_t *)d0)[20] = _mm_extract_epi16(mm_d02, 4);
            ((uint32_t *)d0)[ 9] = _mm_extract_epi32(mm_d02, 1);
        case 6:
            ((uint32_t *)d0)[ 8] = _mm_extract_epi32(mm_d02, 0);
        case 5: case 4: case 3: case 2: case 1: {/* Nothing to do */}
        }
        mm_tmp = _mm_shuffle_epi8(mm_wh01, mm_shuf7);
        mm_d03 = _mm_shuffle_epi8(mm_vh01, mm_shuf8);
        mm_d03 = _mm_or_si128(mm_d03, mm_tmp);
        // Unstored results are in mm_d03 - needs the next to top byte filling in.
        mm_vh01 = _mm_srli_si128(mm_vh01, 15);
        // We need to carry the top byte from mm_vh00 forward into the next results word.
        // Load raw pixels into mm_s02 and mm_s12 (source pixel bytes n+32 to n+47)
        mm_s02 = _mm_loadu_si128((const __m128i *)(local0+32));
        // mm_v02 are vertically combined pixels (n+32 to n+47).
        mm_w02 = _mm_slli_si128(mm_s02,3);
        mm_w02 = _mm_or_si128(mm_w02,mm_fwd0);
        // mm_w02 are vertically combined pixels delayed by a pixel (n+29 to n+44)
        switch (w)
        {
        case 14:
            mm_fwd0 = _mm_srli_si128(mm_w02,10);
            break;
        case 13:
            mm_fwd0 = _mm_srli_si128(mm_w02,7);
            break;
        case 12:
            mm_fwd0 = _mm_srli_si128(mm_w02,4);
            break;
        case 11:
            mm_fwd0 = _mm_srli_si128(mm_w02,1);
            break;
        case 10:
            mm_fwd0 = _mm_or_si128(_mm_srli_si128(mm_w01,14), _mm_slli_si128(mm_w02,2));
            break;
        case 9:
            mm_fwd0 = _mm_srli_si128(mm_w01,11);
            break;
        case 8:
            mm_fwd0 = _mm_srli_si128(mm_w01,8);
            break;
        case 7:
            mm_fwd0 = _mm_srli_si128(mm_w01,5);
            break;
        case 6:
            mm_fwd0 = _mm_srli_si128(mm_w01,2);
            break;
        case 5:
            mm_fwd0 = _mm_or_si128(_mm_srli_si128(mm_w00,15), _mm_slli_si128(mm_w01,1));
            break;
        case 4:
            mm_fwd0 = _mm_srli_si128(mm_w00,12);
            break;
        case 3:
            mm_fwd0 = _mm_srli_si128(mm_w00,9);
            break;
        case 2:
            mm_fwd0 = _mm_srli_si128(mm_w00,6);
            break;
        case 1:
            mm_fwd0 = _mm_srli_si128(mm_w00,3);
            break;
        }
        // Now, we need to horizontally combine
        mm_tmp  = _mm_avg_epu8(mm_s02,mm_w02);
        mm_vh02 = _mm_avg_epu8(mm_s02,mm_tmp);
        mm_wh02 = _mm_avg_epu8(mm_w02,mm_tmp);
        // So our output pixel values are in mm_v02/mm_w02, but in the wrong order
        // Finish off the previous results words
        mm_tmp = _mm_slli_si128(mm_wh02,15);
        mm_tmp = _mm_srli_si128(mm_tmp,1);
        mm_d03 = _mm_or_si128(mm_d03,mm_tmp);
        switch (w)
        {
        default:
            ((uint32_t *)d0)[15] = _mm_extract_epi32(mm_d03, 3);
        case 10:
            ((uint32_t *)d0)[14] = _mm_extract_epi32(mm_d03, 2);
            ((uint16_t *)d0)[27] = _mm_extract_epi16(mm_d03, 3);
        case 9:
            ((uint16_t *)d0)[26] = _mm_extract_epi16(mm_d03, 2);
            ((uint32_t *)d0)[12] = _mm_extract_epi32(mm_d03, 0);
        case 8: case 7: case 6: case 5: case 4: case 3: case 2: case 1: {/* Nothing to do */}
        }
        // Now reorder results.
        mm_tmp = _mm_shuffle_epi8(mm_wh02, mm_shuf9);
        mm_d04 = _mm_shuffle_epi8(mm_vh02, mm_shuf10);
        mm_d04 = _mm_or_si128(mm_d04, mm_tmp);
        mm_d04 = _mm_or_si128(mm_d04, mm_vh01);
        switch (w)
        {
        default:
            ((uint16_t *)d0)[39] = _mm_extract_epi16(mm_d04, 7);
        case 13:
            ((uint16_t *)d0)[38] = _mm_extract_epi16(mm_d04, 6);
            ((uint32_t *)d0)[18] = _mm_extract_epi32(mm_d04, 2);
        case 12:
            ((uint32_t *)d0)[17] = _mm_extract_epi32(mm_d04, 1);
            ((uint16_t *)d0)[33] = _mm_extract_epi16(mm_d04, 1);
        case 11:
            ((uint16_t *)d0)[32] = _mm_extract_epi16(mm_d04, 0);
        case 10: case 9: case 8: case 7: case 6: case 5: case 4: case 3: case 2: case 1: {/* Nothing to do */}
        }
        mm_tmp = _mm_shuffle_epi8(mm_wh02, mm_shuf11);
        mm_d05 = _mm_shuffle_epi8(mm_vh02, mm_shuf12);
        mm_d05 = _mm_or_si128(mm_d05, mm_tmp);
        switch (w)
        {
        case 15:
            ((uint16_t *)d0)[44] = _mm_extract_epi16(mm_d05, 4);
            ((uint32_t *)d0)[21] = _mm_extract_epi32(mm_d05, 1);
        case 14:
            ((uint32_t *)d0)[20] = _mm_extract_epi32(mm_d05, 0);
        default: {/* Nothing to do */}
        }
        d0 += w*6;
    }

    /* Trailing single pixel */
    d0[0] = _mm_extract_epi8(mm_fwd0,0);
    d0[1] = _mm_extract_epi8(mm_fwd0,1);
    d0[2] = _mm_extract_epi8(mm_fwd0,2);

    return 2;
}
#undef DECLARE
#undef LOAD
#undef STORE
#undef COMBINE

#define COMBINE(L,R) {\
    uint32_t l0 = L & 0xFF00FF, r0 = R & 0xFF00FF, l1 = (L>>8) & 0xFF00FF, r1 = (R>>8) & 0xFF00FF;\
    uint32_t t0 = (l0+r0+1)>>1, t1 = (l1+r1+1)>>1;\
    L  = ((l0+t0+1)>>1) & 0xFF00FF;\
    L += ((l1+t1+1)<<7) & ~0xFF00FF;\
    R  = ((r0+t0+1)>>1) & 0xFF00FF;\
    R += ((r1+t1+1)<<7) & ~0xFF00FF;\
}
static int
double_interp4_sse(uint8_t       ** ipa_restrict dsts,
                   const uint8_t ** ipa_restrict srcs,
                   ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint32_t * ipa_restrict d0 = ((uint32_t **)dsts)[0];
    uint32_t * ipa_restrict d1 = ((uint32_t **)dsts)[1];
    const uint32_t * ipa_restrict s0 = ((const uint32_t **)srcs)[0];
    const uint32_t * ipa_restrict s1 = ((const uint32_t **)srcs)[1];
    uint32_t tl = *s0++;
    uint32_t bl = *s1++;
    __m128i mm0, mm1, mm2, mm3, mm4, mm5, mm6;

    /* Leading single pixel */
    COMBINE(tl, bl);
    *d0++ = tl;
    *d1++ = bl;

    mm4 = _mm_set_epi32(0,0,0,tl);
    mm5 = _mm_set_epi32(0,0,0,bl);

    for (w = doubler->src_w-5; w >= 0; w -= 4)
    {
        // mm4, mm5 = single (combined) pixel carried forward.
        // Load raw pixels into mm0 and mm1 (source pixels n to n+3)
        mm0 = _mm_loadu_si128((const __m128i *)s0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        mm1 = _mm_loadu_si128((const __m128i *)s1); // mm1 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        // Combine vertically
        mm3 = _mm_avg_epu8(mm0,mm1);                // mm3 = (mm0 + mm1)>>1
        mm0 = _mm_avg_epu8(mm0,mm3);                // mm0 = (mm0 + mm3)>>1
        mm1 = _mm_avg_epu8(mm1,mm3);                // mm1 = (mm1 + mm3)>>1
        // mm0 and mm1 are vertically combined pixels (n to n+3).
        mm2 = _mm_slli_si128(mm0,4);
        mm3 = _mm_slli_si128(mm1,4);
        mm2 = _mm_or_si128(mm2,mm4);
        mm3 = _mm_or_si128(mm3,mm5);
        // mm2 and mm3 are vertically combined pixels (n-1 to n+2)
        // Make mm4 and mm5 ready for next iteration.
        mm4 = _mm_srli_si128(mm0,12);
        mm5 = _mm_srli_si128(mm1,12);
        // Now, we need to horizontally combine the 2 sets of combined pixels
        mm6 = _mm_avg_epu8(mm0,mm2);
        mm0 = _mm_avg_epu8(mm0,mm6);
        mm2 = _mm_avg_epu8(mm2,mm6);
        mm6 = _mm_avg_epu8(mm1,mm3);
        mm1 = _mm_avg_epu8(mm1,mm6);
        mm3 = _mm_avg_epu8(mm3,mm6);
        // So our output pixel values are in mm0/mm2 and mm1/mm3
        // But they are in the wrong order.
        mm6 = _mm_unpacklo_epi32(mm2,mm0);
        mm0 = _mm_unpackhi_epi32(mm2,mm0);
        mm2 = _mm_unpacklo_epi32(mm3,mm1);
        mm3 = _mm_unpackhi_epi32(mm3,mm1);
        // Results are in mm6/mm0, mm2/mm3
        _mm_storeu_si128((__m128i *)d0, mm6);
        _mm_storeu_si128((__m128i *)(d0+4), mm0);
        _mm_storeu_si128((__m128i *)d1, mm2);
        _mm_storeu_si128((__m128i *)(d1+4), mm3);
        d0 += 8;
        d1 += 8;
        s0 += 4;
        s1 += 4;
    }

    w += 4;
    if (w)
    {
        uint8_t local0[16], local1[16];

        memcpy(local0, s0, (w+1)*4);
        memcpy(local1, s1, (w+1)*4);
        // mm4, mm5 = single (combined) pixel carried forward.
        // Load raw pixels into mm0 and mm1 (source pixels n to n+3)
        mm0 = _mm_loadu_si128((const __m128i *)local0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        mm1 = _mm_loadu_si128((const __m128i *)local1); // mm1 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        // Combine vertically
        mm3 = _mm_avg_epu8(mm0,mm1);                // mm3 = (mm0 + mm1)>>1
        mm0 = _mm_avg_epu8(mm0,mm3);                // mm0 = (mm0 + mm3)>>1
        mm1 = _mm_avg_epu8(mm1,mm3);                // mm1 = (mm1 + mm3)>>1
        // mm0 and mm1 are vertically combined pixels (n to n+3).
        mm2 = _mm_slli_si128(mm0,4);
        mm3 = _mm_slli_si128(mm1,4);
        mm2 = _mm_or_si128(mm2,mm4);
        mm3 = _mm_or_si128(mm3,mm5);
        // mm2 and mm3 are vertically combined pixels (n-1 to n+2)
        // Make mm4 and mm5 ready for next iteration.
        switch (w)
        {
        case 3:
            mm4 = _mm_srli_si128(mm0,8);
            mm5 = _mm_srli_si128(mm1,8);
            break;
        case 2:
            mm4 = _mm_srli_si128(mm0,4);
            mm5 = _mm_srli_si128(mm1,4);
            break;
        case 1:
            mm4 = mm0;
            mm5 = mm1;
            break;
        }
        // Now, we need to horizontally combine the 2 sets of combined pixels
        mm6 = _mm_avg_epu8(mm0,mm2);
        mm0 = _mm_avg_epu8(mm0,mm6);
        mm2 = _mm_avg_epu8(mm2,mm6);
        mm6 = _mm_avg_epu8(mm1,mm3);
        mm1 = _mm_avg_epu8(mm1,mm6);
        mm3 = _mm_avg_epu8(mm3,mm6);
        // So our output pixel values are in mm0/mm2 and mm1/mm3
        // But they are in the wrong order.
        mm6 = _mm_unpacklo_epi32(mm2,mm0);
        mm0 = _mm_unpackhi_epi32(mm2,mm0);
        mm2 = _mm_unpacklo_epi32(mm3,mm1);
        mm3 = _mm_unpackhi_epi32(mm3,mm1);
        // Results are in mm6/mm0, mm2/mm3
        switch (w)
        {
#if defined (_M_X64)
        default:
            ((uint64_t *)d0)[2] = _mm_extract_epi64(mm0, 0);
            ((uint64_t *)d1)[2] = _mm_extract_epi64(mm3, 0);
        case 2:
            ((uint64_t *)d0)[1] = _mm_extract_epi64(mm6, 1);
            ((uint64_t *)d1)[1] = _mm_extract_epi64(mm2, 1);
        case 1:
            ((uint64_t *)d0)[0] = _mm_extract_epi64(mm6, 0);
            ((uint64_t *)d1)[0] = _mm_extract_epi64(mm2, 0);
#else
        default:
            d0[5] = _mm_extract_epi32(mm0, 1);
            d1[5] = _mm_extract_epi32(mm3, 1);
            d0[4] = _mm_extract_epi32(mm0, 0);
            d1[4] = _mm_extract_epi32(mm3, 0);
        case 2:
            d0[3] = _mm_extract_epi32(mm6, 3);
            d1[3] = _mm_extract_epi32(mm2, 3);
            d0[2] = _mm_extract_epi32(mm6, 2);
            d1[2] = _mm_extract_epi32(mm2, 2);
        case 1:
            d0[1] = _mm_extract_epi32(mm6, 1);
            d1[1] = _mm_extract_epi32(mm2, 1);
            d0[0] = _mm_extract_epi32(mm6, 0);
            d1[0] = _mm_extract_epi32(mm2, 0);
#endif
        }
        d0 += w*2;
        d1 += w*2;
    }
    /* Trailing single pixel */
    d0[0] = _mm_extract_epi32(mm4,0);
    d1[0] = _mm_extract_epi32(mm5,0);

    return 2;
}

static int
double_interp4_top_sse(uint8_t       ** ipa_restrict dsts,
                       const uint8_t ** ipa_restrict srcs,
                       ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint32_t * ipa_restrict d0 = ((uint32_t **)dsts)[0];
    const uint32_t * ipa_restrict s0 = ((const uint32_t **)srcs)[0];
    uint32_t tl = *s0++;
    __m128i mm0, mm2, mm4, mm6;

    /* Leading single pixel */
    *d0++ = tl;

    mm4 = _mm_set_epi32(0,0,0,tl);

    for (w = doubler->src_w-5; w >= 0; w -= 4)
    {
        // mm4 = single (combined) pixel carried forward.
        // Load raw pixels into mm0 (source pixels n to n+3)
        mm0 = _mm_loadu_si128((const __m128i *)s0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        // mm0 is pixels (n to n+3).
        mm2 = _mm_slli_si128(mm0,4);
        mm2 = _mm_or_si128(mm2,mm4);
        // mm2 is pixels (n-1 to n+2)
        // Make mm4 ready for next iteration.
        mm4 = _mm_srli_si128(mm0,12);
        // Now, we need to horizontally combine the pixels
        mm6 = _mm_avg_epu8(mm0,mm2);
        mm0 = _mm_avg_epu8(mm0,mm6);
        mm2 = _mm_avg_epu8(mm2,mm6);
        // So our output pixel values are in mm0/mm2
        // But they are in the wrong order.
        mm6 = _mm_unpacklo_epi32(mm2,mm0);
        mm0 = _mm_unpackhi_epi32(mm2,mm0);
        // Results are in mm6/mm0, mm2/mm3
        _mm_storeu_si128((__m128i *)d0, mm6);
        _mm_storeu_si128((__m128i *)(d0+4), mm0);
        d0 += 8;
        s0 += 4;
    }

    w += 4;
    if (w)
    {
        uint8_t local0[16];

        memcpy(local0, s0, (w+1)*4);
        // mm4 = single (combined) pixel carried forward.
        // Load raw pixels into mm0 (source pixels n to n+3)
        mm0 = _mm_loadu_si128((const __m128i *)local0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        // mm0 is pixels (n to n+3).
        mm2 = _mm_slli_si128(mm0,4);
        mm2 = _mm_or_si128(mm2,mm4);
        // mm2 is pixels (n-1 to n+2)
        // Now, we need to horizontally combine the pixels
        mm6 = _mm_avg_epu8(mm0,mm2);
        mm0 = _mm_avg_epu8(mm0,mm6);
        mm2 = _mm_avg_epu8(mm2,mm6);
        // So our output pixel values are in mm0/mm2
        // But they are in the wrong order.
        mm6 = _mm_unpacklo_epi32(mm2,mm0);
        mm0 = _mm_unpackhi_epi32(mm2,mm0);
        // Results are in mm6/mm0, mm2/mm3
        switch (w)
        {
#if defined (_M_X64)
            default:((uint64_t *)d0)[2] = _mm_extract_epi64(mm0, 0);
            case 2: ((uint64_t *)d0)[1] = _mm_extract_epi64(mm6, 1);
            case 1: ((uint64_t *)d0)[0] = _mm_extract_epi64(mm6, 0);
#else
            default:
                d0[5] = _mm_extract_epi32(mm0, 1);
                d0[4] = _mm_extract_epi32(mm0, 0);
            case 2:
                d0[3] = _mm_extract_epi32(mm6, 3);
                d0[2] = _mm_extract_epi32(mm6, 2);
            case 1:
                d0[1] = _mm_extract_epi32(mm6, 1);
                d0[0] = _mm_extract_epi32(mm6, 0);
#endif
        }
        d0 += w * 2;
    }

    /* Trailing single pixel */
    d0[0] = _mm_extract_epi32(mm4,0);

    return 1;
}
#undef COMBINE

#define WEIGHT_SHIFT 10
#define MW0 (-24)
#define MW1 (801)
#define MW2 (262)
#define MW3 (-15)

#define WEIGHT_SCALE (1<<WEIGHT_SHIFT)
#define WEIGHT_ROUND (1<<(WEIGHT_SHIFT-1))
#define COMBINE0(A,B,C,D) (A * MW0 + B * MW1 + C * MW2 + D * MW3)
#define COMBINE1(A,B,C,D) (A * MW3 + B * MW2 + C * MW1 + D * MW0)

static void
double_mitchell1_to_tmp(uint32_t *t0, const uint8_t *s0, uint32_t src_w)
{
    int32_t w;
    int a, b, c;
    __m128i mm_w0123, mm_w3210;
    uint8_t local0[16];

    /* Leading pixels */
    a = s0[0];
    b = s0[1];
    c = s0[2];
    *t0++ = COMBINE1(a,a,a,b);
    *t0++ = COMBINE0(a,a,b,c);
    *t0++ = COMBINE1(a,a,b,c);

    mm_w0123 = _mm_set_epi32(MW3,MW2,MW1,MW0);
    mm_w3210 = _mm_set_epi32(MW0,MW1,MW2,MW3);

    for (w = src_w-3-13; w >= 0; w -= 13)
    {
        __m128i mm_s0, mm_s00, mm_s01;

        // Load raw pixels into mm0 and mm1 (source pixels n to n+3)
        mm_s0 = _mm_loadu_si128((const __m128i *)s0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        // 0: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 1: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 2: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 3: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 4: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 5: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 6: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 7: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 8: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 9: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 10: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 11: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 12: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        s0 += 13;
    }

    w += 13;
    while (w) /* Only a while so we can break out */
    {
        __m128i mm_s0, mm_s00, mm_s01;

        memcpy(local0, s0, w+3);
        s0 += w;
        // Load raw pixels into mm0 and mm1 (source pixels n to n+3)
        mm_s0 = _mm_loadu_si128((const __m128i *)local0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        // 0: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        if (w == 1) break;
        // 1: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        if (w == 2) break;
        // 2: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        if (w == 3) break;
        // 3: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        if (w == 4) break;
        // 4: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        if (w == 5) break;
        // 5: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        if (w == 6) break;
        // 6: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        if (w == 7) break;
        // 7: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        if (w == 8) break;
        // 8: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        if (w == 9) break;
        // 9: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        if (w == 10) break;
        // 10: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        if (w == 11) break;
        // 11: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        if (w == 12) break;
        // 12: Explode to 32 bits
        mm_s01 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        *t0++ = _mm_extract_epi32(mm_s00, 0);
        *t0++ = _mm_extract_epi32(mm_s00, 1);
        break;
    }

    a = *s0++;
    b = *s0++;
    c = *s0++;
    /* Trailing pixels */
    *t0++ = COMBINE0(a,b,c,c);
    *t0++ = COMBINE1(a,b,c,c);
    *t0++ = COMBINE0(b,c,c,c);
}

static int
double_mitchell(uint8_t        * ipa_restrict d0,
                uint8_t        * ipa_restrict d1,
                const uint32_t * ipa_restrict t0,
                const uint32_t * ipa_restrict t1,
                const uint32_t * ipa_restrict t2,
                const uint32_t * ipa_restrict t3,
                int32_t          src_w)
{
    int32_t w;
    __m128i mm_w0, mm_w1, mm_w2, mm_w3, mm_round;

    mm_w0 = _mm_set1_epi32(MW0);
    mm_w1 = _mm_set1_epi32(MW1);
    mm_w2 = _mm_set1_epi32(MW2);
    mm_w3 = _mm_set1_epi32(MW3);
    mm_round = _mm_set1_epi32(WEIGHT_ROUND<<WEIGHT_SHIFT);
    for (w = 2*src_w-4; w >= 0; w -= 4)
    {
        __m128i mm_s0, mm_s1, mm_s2, mm_s3;
        __m128i mm_s00, mm_s01, mm_s02, mm_s03;
        __m128i mm_s10, mm_s11, mm_s12, mm_s13;

        mm_s0 = _mm_loadu_si128((const __m128i *)t0);
        // Combine vertically
        mm_s00 = _mm_mullo_epi32(mm_s0, mm_w0);
        mm_s10 = _mm_mullo_epi32(mm_s0, mm_w3);
        mm_s1 = _mm_loadu_si128((const __m128i *)t1);
        mm_s01 = _mm_mullo_epi32(mm_s1, mm_w1);
        mm_s11 = _mm_mullo_epi32(mm_s1, mm_w2);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s01);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s11);
        mm_s2 = _mm_loadu_si128((const __m128i *)t2);
        mm_s02 = _mm_mullo_epi32(mm_s2, mm_w2);
        mm_s12 = _mm_mullo_epi32(mm_s2, mm_w1);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s02);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s12);
        mm_s3 = _mm_loadu_si128((const __m128i *)t3);
        mm_s03 = _mm_mullo_epi32(mm_s3, mm_w3);
        mm_s13 = _mm_mullo_epi32(mm_s3, mm_w0);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s03);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s13);
        // Round, shift, extract
        mm_s00 = _mm_add_epi32(mm_s00, mm_round);
        mm_s10 = _mm_add_epi32(mm_s10, mm_round);
        mm_s00 = _mm_srai_epi32(mm_s00, WEIGHT_SHIFT*2);
        mm_s10 = _mm_srai_epi32(mm_s10, WEIGHT_SHIFT*2);
        mm_s00 = _mm_packus_epi32(mm_s00, mm_s00); // Clamp to 0 to 65535 range.
        mm_s10 = _mm_packus_epi32(mm_s10, mm_s10); // Clamp to 0 to 65535 range.
        mm_s00 = _mm_packus_epi16(mm_s00, mm_s00); // Clamp to 0 to 255 range.
        mm_s10 = _mm_packus_epi16(mm_s10, mm_s10); // Clamp to 0 to 255 range.
        *(uint32_t *)d0 = _mm_extract_epi32(mm_s00, 0);
        *(uint32_t *)d1 = _mm_extract_epi32(mm_s10, 0);
        d0 += 4;
        d1 += 4;
        t0 += 4;
        t1 += 4;
        t2 += 4;
        t3 += 4;
    }

    w += 4;
    if (w)
    {
        __m128i mm_s0, mm_s1, mm_s2, mm_s3;
        __m128i mm_s00, mm_s01, mm_s02, mm_s03;
        __m128i mm_s10, mm_s11, mm_s12, mm_s13;

        /* tmp is always large enough to allow for the overrun */
        mm_s0 = _mm_loadu_si128((const __m128i *)t0);
        // Combine vertically
        mm_s00 = _mm_mullo_epi32(mm_s0, mm_w0);
        mm_s10 = _mm_mullo_epi32(mm_s0, mm_w3);
        mm_s1 = _mm_loadu_si128((const __m128i *)t1);
        mm_s01 = _mm_mullo_epi32(mm_s1, mm_w1);
        mm_s11 = _mm_mullo_epi32(mm_s1, mm_w2);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s01);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s11);
        mm_s2 = _mm_loadu_si128((const __m128i *)t2);
        mm_s02 = _mm_mullo_epi32(mm_s2, mm_w2);
        mm_s12 = _mm_mullo_epi32(mm_s2, mm_w1);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s02);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s12);
        mm_s3 = _mm_loadu_si128((const __m128i *)t3);
        mm_s03 = _mm_mullo_epi32(mm_s3, mm_w3);
        mm_s13 = _mm_mullo_epi32(mm_s3, mm_w0);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s03);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s13);
        // Round, shift, extract
        mm_s00 = _mm_add_epi32(mm_s00, mm_round);
        mm_s10 = _mm_add_epi32(mm_s10, mm_round);
        mm_s00 = _mm_srai_epi32(mm_s00, WEIGHT_SHIFT*2);
        mm_s10 = _mm_srai_epi32(mm_s10, WEIGHT_SHIFT*2);
        mm_s00 = _mm_packus_epi32(mm_s00, mm_s00); // Clamp to 0 to 65535 range.
        mm_s10 = _mm_packus_epi32(mm_s10, mm_s10); // Clamp to 0 to 65535 range.
        mm_s00 = _mm_packus_epi16(mm_s00, mm_s00); // Clamp to 0 to 255 range.
        mm_s10 = _mm_packus_epi16(mm_s10, mm_s10); // Clamp to 0 to 255 range.
        switch (w)
        {
            case 3:
                d0[2] = _mm_extract_epi8(mm_s00, 2);
                d1[2] = _mm_extract_epi8(mm_s10, 2);
            case 2:
                d0[1] = _mm_extract_epi8(mm_s00, 1);
                d1[1] = _mm_extract_epi8(mm_s10, 1);
            case 1:
                d0[0] = _mm_extract_epi8(mm_s00, 0);
                d1[0] = _mm_extract_epi8(mm_s10, 0);
        }
    }

    return 2;
}

static int
double_mitchell1_sse(uint8_t       ** ipa_restrict dsts,
                     const uint8_t ** ipa_restrict srcs,
                     ipa_doubler    * ipa_restrict doubler)
{
    uint8_t * ipa_restrict d0 = ((uint8_t **)dsts)[0];
    uint8_t * ipa_restrict d1 = ((uint8_t **)dsts)[1];
    const uint8_t * ipa_restrict s3 = ((const uint8_t **)srcs)[3];
    uint32_t *t3 = doubler->tmp + (((doubler->tmp_y    )&3) * doubler->tmp_stride);
    uint32_t *t2 = doubler->tmp + (((doubler->tmp_y + 3)&3) * doubler->tmp_stride);
    uint32_t *t1 = doubler->tmp + (((doubler->tmp_y + 2)&3) * doubler->tmp_stride);
    uint32_t *t0 = doubler->tmp + (((doubler->tmp_y + 1)&3) * doubler->tmp_stride);

    double_mitchell1_to_tmp(t3, s3, doubler->src_w);
    doubler->tmp_y++;

    return double_mitchell(d0, d1, t0, t1, t2, t3, doubler->src_w);
}

static int
double_mitchell_final(uint8_t       ** ipa_restrict dsts,
                      ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = ((uint8_t **)dsts)[0];
    uint32_t *t3 = doubler->tmp + (((doubler->tmp_y    )&3) * doubler->tmp_stride);
    uint32_t *t2 = doubler->tmp + (((doubler->tmp_y + 3)&3) * doubler->tmp_stride);
    uint32_t *t1 = doubler->tmp + (((doubler->tmp_y + 2)&3) * doubler->tmp_stride);
    uint32_t *t0 = doubler->tmp + (((doubler->tmp_y + 1)&3) * doubler->tmp_stride);
    __m128i mm_w0, mm_w1, mm_w2, mm_w3, mm_round;
    int tmp_y = doubler->tmp_y++;

    if (tmp_y == 0)
        return 0;
    if (tmp_y == 1)
    {
        /* A,A,A,B reversed */
        t0 = t3;
        t1 = t3 = t2;
    }
    else if (tmp_y == 2)
    {
        /* A,A,B,C */
        uint8_t * ipa_restrict d1 = ((uint8_t **)dsts)[1];
        return double_mitchell(d0, d1, t1, t1, t2, t3, doubler->src_w * doubler->channels);
    }
    else if (tmp_y == doubler->src_h)
    {
        /* A,B,C,C */
        uint8_t * ipa_restrict d1 = ((uint8_t **)dsts)[1];
        return double_mitchell(d0, d1, t0, t1, t2, t2, doubler->src_w * doubler->channels);
    }
    else
    {
        assert(tmp_y == doubler->src_h+1);
        /* A,B,B,B */
        t2 = t3 = t1;
    }

    mm_w0 = _mm_set1_epi32(MW0);
    mm_w1 = _mm_set1_epi32(MW1);
    mm_w2 = _mm_set1_epi32(MW2);
    mm_w3 = _mm_set1_epi32(MW3);
    mm_round = _mm_set1_epi32(WEIGHT_ROUND<<WEIGHT_SHIFT);
    for (w = doubler->src_w * 2 * doubler->channels - 4; w >= 0; w -= 4)
    {
        __m128i mm_s0, mm_s1, mm_s2, mm_s3;
        __m128i mm_s00, mm_s01, mm_s02, mm_s03;

        mm_s0 = _mm_loadu_si128((const __m128i *)t0);
        // Combine vertically
        mm_s00 = _mm_mullo_epi32(mm_s0, mm_w0);
        mm_s1 = _mm_loadu_si128((const __m128i *)t1);
        mm_s01 = _mm_mullo_epi32(mm_s1, mm_w1);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s01);
        mm_s2 = _mm_loadu_si128((const __m128i *)t2);
        mm_s02 = _mm_mullo_epi32(mm_s2, mm_w2);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s02);
        mm_s3 = _mm_loadu_si128((const __m128i *)t3);
        mm_s03 = _mm_mullo_epi32(mm_s3, mm_w3);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s03);
        // Round, shift, extract
        mm_s00 = _mm_add_epi32(mm_s00, mm_round);
        mm_s00 = _mm_srai_epi32(mm_s00, WEIGHT_SHIFT*2);
        mm_s00 = _mm_packus_epi32(mm_s00, mm_s00); // Clamp to 0 to 65535 range.
        mm_s00 = _mm_packus_epi16(mm_s00, mm_s00); // Clamp to 0 to 255 range.
        *(uint32_t *)d0 = _mm_extract_epi32(mm_s00, 0);
        d0 += 4;
        t0 += 4;
        t1 += 4;
        t2 += 4;
        t3 += 4;
    }

    w += 4;
    if (w)
    {
        __m128i mm_s0, mm_s1, mm_s2, mm_s3;
        __m128i mm_s00, mm_s01, mm_s02, mm_s03;

        /* tmp is always large enough to allow for the overrun */
        mm_s0 = _mm_loadu_si128((const __m128i *)t0);
        // Combine vertically
        mm_s00 = _mm_mullo_epi32(mm_s0, mm_w0);
        mm_s1 = _mm_loadu_si128((const __m128i *)t1);
        mm_s01 = _mm_mullo_epi32(mm_s1, mm_w1);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s01);
        mm_s2 = _mm_loadu_si128((const __m128i *)t2);
        mm_s02 = _mm_mullo_epi32(mm_s2, mm_w2);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s02);
        mm_s3 = _mm_loadu_si128((const __m128i *)t3);
        mm_s03 = _mm_mullo_epi32(mm_s3, mm_w3);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s03);
        // Round, shift, extract
        mm_s00 = _mm_add_epi32(mm_s00, mm_round);
        mm_s00 = _mm_srai_epi32(mm_s00, WEIGHT_SHIFT*2);
        mm_s00 = _mm_packus_epi32(mm_s00, mm_s00); // Clamp to 0 to 65535 range.
        mm_s00 = _mm_packus_epi16(mm_s00, mm_s00); // Clamp to 0 to 255 range.
        switch (w)
        {
            case 3: d0[2] = _mm_extract_epi8(mm_s00, 2);
            case 2: d0[1] = _mm_extract_epi8(mm_s00, 1);
            case 1: d0[0] = _mm_extract_epi8(mm_s00, 0);
        }
    }

    return 1;
}

static int
double_mitchell1_top_sse(uint8_t       ** ipa_restrict dsts,
                         const uint8_t ** ipa_restrict srcs,
                         ipa_doubler    * ipa_restrict doubler)
{
    const uint8_t * ipa_restrict s3 = ((const uint8_t **)srcs)[3];
    uint32_t *t3 = doubler->tmp + (((doubler->tmp_y    )&3) * doubler->tmp_stride);

    if (doubler->tmp_y < (int)doubler->src_h)
        double_mitchell1_to_tmp(t3, s3, doubler->src_w);

    return double_mitchell_final(dsts, doubler);
}

static void
double_mitchell3_to_tmp(uint32_t *t0, const uint8_t *s0, uint32_t src_w)
{
    int32_t w;
    int a_r, a_g, a_b, b_r, b_g, b_b, c_r, c_g, c_b;
    __m128i mm_w0123, mm_w3210, mm_shuffle;

    /* Leading pixels */
    a_r = s0[0];
    a_g = s0[1];
    a_b = s0[2];
    b_r = s0[3];
    b_g = s0[4];
    b_b = s0[5];
    c_r = s0[6];
    c_g = s0[7];
    c_b = s0[8];
    *t0++ = COMBINE1(a_r,a_r,a_r,b_r);
    *t0++ = COMBINE1(a_g,a_g,a_g,b_g);
    *t0++ = COMBINE1(a_b,a_b,a_b,b_b);
    *t0++ = COMBINE0(a_r,a_r,b_r,c_r);
    *t0++ = COMBINE0(a_g,a_g,b_g,c_g);
    *t0++ = COMBINE0(a_b,a_b,b_b,c_b);
    *t0++ = COMBINE1(a_r,a_r,b_r,c_r);
    *t0++ = COMBINE1(a_g,a_g,b_g,c_g);
    *t0++ = COMBINE1(a_b,a_b,b_b,c_b);

    mm_w0123 = _mm_set_epi32(MW3,MW2,MW1,MW0);
    mm_w3210 = _mm_set_epi32(MW0,MW1,MW2,MW3);
    mm_shuffle = _mm_set_epi8(ZZ,ZZ,ZZ, 9,ZZ,ZZ,ZZ, 6,ZZ,ZZ,ZZ, 3,ZZ,ZZ,ZZ, 0);

    for (w = src_w-3-2; w >= 0; w -= 2)
    {
        __m128i mm_s0, mm_s00, mm_s01;

        // Load raw pixels into mm0 and mm1 (source pixels n to n+3)
        mm_s0 = _mm_loadu_si128((const __m128i *)s0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        // 0: Explode to 32 bits
        mm_s01 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        t0[0] = _mm_extract_epi32(mm_s00, 0);
        t0[3] = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 1: Explode to 32 bits
        mm_s01 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        t0[1] = _mm_extract_epi32(mm_s00, 0);
        t0[4] = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 2: Explode to 32 bits
        mm_s01 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        t0[2] = _mm_extract_epi32(mm_s00, 0);
        t0[5] = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 3: Explode to 32 bits
        mm_s01 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        t0[6] = _mm_extract_epi32(mm_s00, 0);
        t0[9] = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 4: Explode to 32 bits
        mm_s01 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        t0[7] = _mm_extract_epi32(mm_s00, 0);
        t0[10] = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 5: Explode to 32 bits
        mm_s01 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        t0[8] = _mm_extract_epi32(mm_s00, 0);
        t0[11] = _mm_extract_epi32(mm_s00, 1);
        s0 += 6;
        t0 += 12;
    }

    w += 2;
    if (w)
    {
        __m128i mm_s0, mm_s00, mm_s01;

        // Load raw pixels into mm0 and mm1 (source pixels n to n+3)
        mm_s0 = _mm_loadu_si128((const __m128i *)s0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        // 0: Explode to 32 bits
        mm_s01 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        t0[0] = _mm_extract_epi32(mm_s00, 0);
        t0[3] = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 1: Explode to 32 bits
        mm_s01 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        t0[1] = _mm_extract_epi32(mm_s00, 0);
        t0[4] = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 2: Explode to 32 bits
        mm_s01 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        t0[2] = _mm_extract_epi32(mm_s00, 0);
        t0[5] = _mm_extract_epi32(mm_s00, 1);
        s0 += 3;
        t0 += 6;
    }

    a_r = s0[0]; a_g = s0[1]; a_b = s0[2];
    b_r = s0[3]; b_g = s0[4]; b_b = s0[5];
    c_r = s0[6]; c_g = s0[7]; c_b = s0[8];
    s0 += 9;

    /* Trailing pixels */
    *t0++ = COMBINE0(a_r,b_r,c_r,c_r);
    *t0++ = COMBINE0(a_g,b_g,c_g,c_g);
    *t0++ = COMBINE0(a_b,b_b,c_b,c_b);
    *t0++ = COMBINE1(a_r,b_r,c_r,c_r);
    *t0++ = COMBINE1(a_g,b_g,c_g,c_g);
    *t0++ = COMBINE1(a_b,b_b,c_b,c_b);
    *t0++ = COMBINE0(b_r,c_r,c_r,c_r);
    *t0++ = COMBINE0(b_g,c_g,c_g,c_g);
    *t0++ = COMBINE0(b_b,c_b,c_b,c_b);
}

static int
double_mitchell3_sse(uint8_t       ** ipa_restrict dsts,
                     const uint8_t ** ipa_restrict srcs,
                     ipa_doubler    * ipa_restrict doubler)
{
    uint8_t * ipa_restrict d0 = ((uint8_t **)dsts)[0];
    uint8_t * ipa_restrict d1 = ((uint8_t **)dsts)[1];
    const uint8_t * ipa_restrict s3 = ((const uint8_t **)srcs)[3];
    uint32_t *t3 = doubler->tmp + (((doubler->tmp_y    )&3) * doubler->tmp_stride);
    uint32_t *t2 = doubler->tmp + (((doubler->tmp_y + 3)&3) * doubler->tmp_stride);
    uint32_t *t1 = doubler->tmp + (((doubler->tmp_y + 2)&3) * doubler->tmp_stride);
    uint32_t *t0 = doubler->tmp + (((doubler->tmp_y + 1)&3) * doubler->tmp_stride);

    double_mitchell3_to_tmp(t3, s3, doubler->src_w);
    doubler->tmp_y++;

    return double_mitchell(d0, d1, t0, t1, t2, t3, doubler->src_w*3);
}

static int
double_mitchell3_top_sse(uint8_t       ** ipa_restrict dsts,
                         const uint8_t ** ipa_restrict srcs,
                         ipa_doubler    * ipa_restrict doubler)
{
    const uint8_t * ipa_restrict s3 = ((const uint8_t **)srcs)[3];
    uint32_t *t3 = doubler->tmp + (((doubler->tmp_y    )&3) * doubler->tmp_stride);

    if (doubler->tmp_y < (int)doubler->src_h)
        double_mitchell3_to_tmp(t3, s3, doubler->src_w);

    return double_mitchell_final(dsts, doubler);
}

static void
double_mitchell4_to_tmp(uint32_t *t0, const uint8_t *s0, uint32_t src_w)
{
    int32_t w;
    int a_r, a_g, a_b, a_k, b_r, b_g, b_b, b_k, c_r, c_g, c_b, c_k;
    __m128i mm_w0123, mm_w3210, mm_shuffle;

    /* Leading pixels */
    a_r = s0[0];
    a_g = s0[1];
    a_b = s0[2];
    a_k = s0[3];
    b_r = s0[4];
    b_g = s0[5];
    b_b = s0[6];
    b_k = s0[7];
    c_r = s0[8];
    c_g = s0[9];
    c_b = s0[10];
    c_k = s0[11];
    *t0++ = COMBINE1(a_r,a_r,a_r,b_r);
    *t0++ = COMBINE1(a_g,a_g,a_g,b_g);
    *t0++ = COMBINE1(a_b,a_b,a_b,b_b);
    *t0++ = COMBINE1(a_k,a_k,a_k,b_k);
    *t0++ = COMBINE0(a_r,a_r,b_r,c_r);
    *t0++ = COMBINE0(a_g,a_g,b_g,c_g);
    *t0++ = COMBINE0(a_b,a_b,b_b,c_b);
    *t0++ = COMBINE0(a_k,a_k,b_k,c_k);
    *t0++ = COMBINE1(a_r,a_r,b_r,c_r);
    *t0++ = COMBINE1(a_g,a_g,b_g,c_g);
    *t0++ = COMBINE1(a_b,a_b,b_b,c_b);
    *t0++ = COMBINE1(a_k,a_k,b_k,c_k);

    mm_w0123 = _mm_set_epi32(MW3,MW2,MW1,MW0);
    mm_w3210 = _mm_set_epi32(MW0,MW1,MW2,MW3);
    mm_shuffle = _mm_set_epi8(ZZ,ZZ,ZZ,12,ZZ,ZZ,ZZ, 8,ZZ,ZZ,ZZ, 4,ZZ,ZZ,ZZ, 0);

    for (w = src_w-3-1; w >= 0; w--)
    {
        __m128i mm_s0, mm_s00, mm_s01;

        // Load raw pixels into mm0 and mm1 (source pixels n to n+3)
        mm_s0 = _mm_loadu_si128((const __m128i *)s0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        // 0: Explode to 32 bits
        mm_s01 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        t0[0] = _mm_extract_epi32(mm_s00, 0);
        t0[4] = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 1: Explode to 32 bits
        mm_s01 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        t0[1] = _mm_extract_epi32(mm_s00, 0);
        t0[5] = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 2: Explode to 32 bits
        mm_s01 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        t0[2] = _mm_extract_epi32(mm_s00, 0);
        t0[6] = _mm_extract_epi32(mm_s00, 1);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 3: Explode to 32 bits
        mm_s01 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s01, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s01, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s00);
        // Store out horizontally combined values.
        t0[3] = _mm_extract_epi32(mm_s00, 0);
        t0[7] = _mm_extract_epi32(mm_s00, 1);
        s0 += 4;
        t0 += 8;
    }

    a_r = s0[0]; a_g = s0[1]; a_b = s0[2]; a_k = s0[3];
    b_r = s0[4]; b_g = s0[5]; b_b = s0[6]; b_k = s0[7];
    c_r = s0[8]; c_g = s0[9]; c_b = s0[10];c_k = s0[11];
    /* Trailing pixels */
    *t0++ = COMBINE0(a_r,b_r,c_r,c_r);
    *t0++ = COMBINE0(a_g,b_g,c_g,c_g);
    *t0++ = COMBINE0(a_b,b_b,c_b,c_b);
    *t0++ = COMBINE0(a_k,b_k,c_k,c_k);
    *t0++ = COMBINE1(a_r,b_r,c_r,c_r);
    *t0++ = COMBINE1(a_g,b_g,c_g,c_g);
    *t0++ = COMBINE1(a_b,b_b,c_b,c_b);
    *t0++ = COMBINE1(b_k,b_k,c_k,c_k);
    *t0++ = COMBINE0(b_r,c_r,c_r,c_r);
    *t0++ = COMBINE0(b_g,c_g,c_g,c_g);
    *t0++ = COMBINE0(b_b,c_b,c_b,c_b);
    *t0++ = COMBINE0(b_k,c_k,c_k,c_k);
}

static int
double_mitchell4_sse(uint8_t       ** ipa_restrict dsts,
                     const uint8_t ** ipa_restrict srcs,
                     ipa_doubler    * ipa_restrict doubler)
{
    uint8_t * ipa_restrict d0 = ((uint8_t **)dsts)[0];
    uint8_t * ipa_restrict d1 = ((uint8_t **)dsts)[1];
    const uint8_t * ipa_restrict s3 = ((const uint8_t **)srcs)[3];
    uint32_t *t3 = doubler->tmp + (((doubler->tmp_y    )&3) * doubler->tmp_stride);
    uint32_t *t2 = doubler->tmp + (((doubler->tmp_y + 3)&3) * doubler->tmp_stride);
    uint32_t *t1 = doubler->tmp + (((doubler->tmp_y + 2)&3) * doubler->tmp_stride);
    uint32_t *t0 = doubler->tmp + (((doubler->tmp_y + 1)&3) * doubler->tmp_stride);

    double_mitchell4_to_tmp(t3, s3, doubler->src_w);
    doubler->tmp_y++;

    return double_mitchell(d0, d1, t0, t1, t2, t3, doubler->src_w*4);
}

static int
double_mitchell4_top_sse(uint8_t       ** ipa_restrict dsts,
                         const uint8_t ** ipa_restrict srcs,
                         ipa_doubler    * ipa_restrict doubler)
{
    const uint8_t * ipa_restrict s3 = ((const uint8_t **)srcs)[3];
    uint32_t *t3 = doubler->tmp + (((doubler->tmp_y    )&3) * doubler->tmp_stride);

    if (doubler->tmp_y < (int)doubler->src_h)
        double_mitchell4_to_tmp(t3, s3, doubler->src_w);

    return double_mitchell_final(dsts, doubler);
}

#undef COMBINE0
#undef COMBINE1
#undef MW0
#undef MW1
#undef MW2
#undef MW3
#undef WEIGHT_SHIFT
#undef WEIGHT_SCALE
#undef WEIGHT_ROUND

static int
quad_near1_sse(uint8_t       ** ipa_restrict dsts,
               const uint8_t ** ipa_restrict srcs,
               ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    const uint8_t * ipa_restrict s0 = srcs[0];
    __m128i mm_shuf0, mm_shuf1, mm_shuf2, mm_shuf3;

    mm_shuf0 = _mm_set_epi8( 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    mm_shuf1 = _mm_set_epi8( 7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4);
    mm_shuf2 = _mm_set_epi8(11,11,11,11,10,10,10,10, 9, 9, 9, 9, 8, 8, 8, 8);
    mm_shuf3 = _mm_set_epi8(15,15,15,15,14,14,14,14,13,13,13,13,12,12,12,12);

    for (w = doubler->src_w-16; w >= 0; w -= 16)
    {
        __m128i mm0, mm1;

        mm0 = _mm_loadu_si128((const __m128i *)s0);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)d0,mm1);
        _mm_storeu_si128((__m128i *)d1,mm1);
        _mm_storeu_si128((__m128i *)d2,mm1);
        _mm_storeu_si128((__m128i *)d3,mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf1);
        _mm_storeu_si128((__m128i *)(d0+16),mm1);
        _mm_storeu_si128((__m128i *)(d1+16),mm1);
        _mm_storeu_si128((__m128i *)(d2+16),mm1);
        _mm_storeu_si128((__m128i *)(d3+16),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf2);
        _mm_storeu_si128((__m128i *)(d0+32),mm1);
        _mm_storeu_si128((__m128i *)(d1+32),mm1);
        _mm_storeu_si128((__m128i *)(d2+32),mm1);
        _mm_storeu_si128((__m128i *)(d3+32),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf3);
        _mm_storeu_si128((__m128i *)(d0+48),mm1);
        _mm_storeu_si128((__m128i *)(d1+48),mm1);
        _mm_storeu_si128((__m128i *)(d2+48),mm1);
        _mm_storeu_si128((__m128i *)(d3+48),mm1);
        s0 += 16;
        d0 += 64;
        d1 += 64;
        d2 += 64;
        d3 += 64;
    }

    w += 16;
    while (w) /* So we can break out */
    {
        uint8_t local0[16];
        __m128i mm0, mm1;

        memcpy(local0, s0, w);
        mm0 = _mm_loadu_si128((const __m128i *)local0);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        switch (w)
        {
            default: ((uint32_t *)d0)[3] = ((uint32_t *)d1)[3] = ((uint32_t *)d2)[3] = ((uint32_t *)d3)[3] = _mm_extract_epi32(mm1, 3);
            case  3: ((uint32_t *)d0)[2] = ((uint32_t *)d1)[2] = ((uint32_t *)d2)[2] = ((uint32_t *)d3)[2] = _mm_extract_epi32(mm1, 2);
            case  2: ((uint32_t *)d0)[1] = ((uint32_t *)d1)[1] = ((uint32_t *)d2)[1] = ((uint32_t *)d3)[1] = _mm_extract_epi32(mm1, 1);
            case  1: ((uint32_t *)d0)[0] = ((uint32_t *)d1)[0] = ((uint32_t *)d2)[0] = ((uint32_t *)d3)[0] = _mm_extract_epi32(mm1, 0);
        }
        if (w <= 4) break;
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf1);
        switch (w)
        {
            default: ((uint32_t *)d0)[7] = ((uint32_t *)d1)[7] = ((uint32_t *)d2)[7] = ((uint32_t *)d3)[7] = _mm_extract_epi32(mm1, 3);
            case  7: ((uint32_t *)d0)[6] = ((uint32_t *)d1)[6] = ((uint32_t *)d2)[6] = ((uint32_t *)d3)[6] = _mm_extract_epi32(mm1, 2);
            case  6: ((uint32_t *)d0)[5] = ((uint32_t *)d1)[5] = ((uint32_t *)d2)[5] = ((uint32_t *)d3)[5] = _mm_extract_epi32(mm1, 1);
            case  5: ((uint32_t *)d0)[4] = ((uint32_t *)d1)[4] = ((uint32_t *)d2)[4] = ((uint32_t *)d3)[4] = _mm_extract_epi32(mm1, 0);
        }
        if (w <= 8) break;
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf2);
        switch (w)
        {
            default: ((uint32_t *)d0)[11] = ((uint32_t *)d1)[11] = ((uint32_t *)d2)[11] = ((uint32_t *)d3)[11] = _mm_extract_epi32(mm1, 3);
            case 11: ((uint32_t *)d0)[10] = ((uint32_t *)d1)[10] = ((uint32_t *)d2)[10] = ((uint32_t *)d3)[10] = _mm_extract_epi32(mm1, 2);
            case 10: ((uint32_t *)d0)[ 9] = ((uint32_t *)d1)[ 9] = ((uint32_t *)d2)[ 9] = ((uint32_t *)d3)[ 9] = _mm_extract_epi32(mm1, 1);
            case  9: ((uint32_t *)d0)[ 8] = ((uint32_t *)d1)[ 8] = ((uint32_t *)d2)[ 8] = ((uint32_t *)d3)[ 8] = _mm_extract_epi32(mm1, 0);
        }
        if (w <= 12) break;
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf3);
        switch (w)
        {
            case 15: ((uint32_t *)d0)[14] = ((uint32_t *)d1)[14] = ((uint32_t *)d2)[14] = ((uint32_t *)d3)[14] = _mm_extract_epi32(mm1, 2);
            case 14: ((uint32_t *)d0)[13] = ((uint32_t *)d1)[13] = ((uint32_t *)d2)[13] = ((uint32_t *)d3)[13] = _mm_extract_epi32(mm1, 1);
            case 13: ((uint32_t *)d0)[12] = ((uint32_t *)d1)[12] = ((uint32_t *)d2)[12] = ((uint32_t *)d3)[12] = _mm_extract_epi32(mm1, 0);
        }
        break;
    }

    return 4;
}

static int
quad_near3_sse(uint8_t       ** ipa_restrict dsts,
               const uint8_t ** ipa_restrict srcs,
               ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    const uint8_t * ipa_restrict s0 = srcs[0];
    __m128i mm_shuf0, mm_shuf1, mm_shuf2, mm_shuf3;

    mm_shuf0 = _mm_set_epi8( 3, 5, 4, 3, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0);
    mm_shuf1 = _mm_set_epi8( 7, 6, 8, 7, 6, 8, 7, 6, 5, 4, 3, 5, 4, 3, 5, 4);
    mm_shuf2 = _mm_set_epi8(11,10, 9,11,10, 9,11,10, 9,11,10, 9, 8, 7, 6, 8);
    mm_shuf3 = _mm_set_epi8(ZZ,ZZ,ZZ,ZZ,14,13,12,14,13,12,14,13,12,14,13,12);

    for (w = doubler->src_w-5; w >= 0; w -= 5)
    {
        __m128i mm0, mm1;

        mm0 = _mm_loadu_si128((const __m128i *)s0);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)d0,mm1);
        _mm_storeu_si128((__m128i *)d1,mm1);
        _mm_storeu_si128((__m128i *)d2,mm1);
        _mm_storeu_si128((__m128i *)d3,mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf1);
        _mm_storeu_si128((__m128i *)(d0+16),mm1);
        _mm_storeu_si128((__m128i *)(d1+16),mm1);
        _mm_storeu_si128((__m128i *)(d2+16),mm1);
        _mm_storeu_si128((__m128i *)(d3+16),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf2);
        _mm_storeu_si128((__m128i *)(d0+32),mm1);
        _mm_storeu_si128((__m128i *)(d1+32),mm1);
        _mm_storeu_si128((__m128i *)(d2+32),mm1);
        _mm_storeu_si128((__m128i *)(d3+32),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf3);
        _mm_storeu_si128((__m128i *)(d0+48),mm1);
        _mm_storeu_si128((__m128i *)(d1+48),mm1);
        _mm_storeu_si128((__m128i *)(d2+48),mm1);
        _mm_storeu_si128((__m128i *)(d3+48),mm1);
        s0 += 15;
        d0 += 60;
        d1 += 60;
        d2 += 60;
        d3 += 60;
    }

    w += 5;
    while (w) /* Just so we can break out */
    {
        uint8_t local[16];
        __m128i mm0, mm1;

        memcpy(local, s0, w*3);
        mm0 = _mm_loadu_si128((const __m128i *)local);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        switch (w)
        {
        default:
            ((uint32_t *)d0)[3] = ((uint32_t *)d1)[3] = ((uint32_t *)d2)[3] = ((uint32_t *)d3)[3] = _mm_extract_epi32(mm1, 3);
        case  1:
            ((uint32_t *)d0)[2] = ((uint32_t *)d1)[2] = ((uint32_t *)d2)[2] = ((uint32_t *)d3)[2] = _mm_extract_epi32(mm1, 2);
            ((uint32_t *)d0)[1] = ((uint32_t *)d1)[1] = ((uint32_t *)d2)[1] = ((uint32_t *)d3)[1] = _mm_extract_epi32(mm1, 1);
            ((uint32_t *)d0)[0] = ((uint32_t *)d1)[0] = ((uint32_t *)d2)[0] = ((uint32_t *)d3)[0] = _mm_extract_epi32(mm1, 0);
        }
        if (w <= 1) break;
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf1);
        switch (w)
        {
        default:
            ((uint32_t *)d0)[7] = ((uint32_t *)d1)[7] = ((uint32_t *)d2)[7] = ((uint32_t *)d3)[7] = _mm_extract_epi32(mm1, 3);
            ((uint32_t *)d0)[6] = ((uint32_t *)d1)[6] = ((uint32_t *)d2)[6] = ((uint32_t *)d3)[6] = _mm_extract_epi32(mm1, 2);
        case  2:
            ((uint32_t *)d0)[5] = ((uint32_t *)d1)[5] = ((uint32_t *)d2)[5] = ((uint32_t *)d3)[5] = _mm_extract_epi32(mm1, 1);
            ((uint32_t *)d0)[4] = ((uint32_t *)d1)[4] = ((uint32_t *)d2)[4] = ((uint32_t *)d3)[4] = _mm_extract_epi32(mm1, 0);
        }
        if (w <= 2) break;
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf2);
        switch (w)
        {
        default:
            ((uint32_t *)d0)[11] = ((uint32_t *)d1)[11] = ((uint32_t *)d2)[11] = ((uint32_t *)d3)[11] = _mm_extract_epi32(mm1, 3);
            ((uint32_t *)d0)[10] = ((uint32_t *)d1)[10] = ((uint32_t *)d2)[10] = ((uint32_t *)d3)[10] = _mm_extract_epi32(mm1, 2);
            ((uint32_t *)d0)[ 9] = ((uint32_t *)d1)[ 9] = ((uint32_t *)d2)[ 9] = ((uint32_t *)d3)[ 9] = _mm_extract_epi32(mm1, 1);
        case  3:
            ((uint32_t *)d0)[ 8] = ((uint32_t *)d1)[ 8] = ((uint32_t *)d2)[ 8] = ((uint32_t *)d3)[ 8] = _mm_extract_epi32(mm1, 0);
        }
        break;
    }

    return 4;
}

static int
quad_near4_sse(uint8_t       ** ipa_restrict dsts,
               const uint8_t ** ipa_restrict srcs,
               ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    const uint8_t * ipa_restrict s0 = srcs[0];

    for (w = doubler->src_w-4; w >= 0; w -= 4)
    {
        __m128i mm0, mm1;

        mm0 = _mm_loadu_si128((const __m128i *)s0);
        mm1 = _mm_shuffle_epi32(mm0, 0);
        _mm_storeu_si128((__m128i *)d0,mm1);
        _mm_storeu_si128((__m128i *)d1,mm1);
        _mm_storeu_si128((__m128i *)d2,mm1);
        _mm_storeu_si128((__m128i *)d3,mm1);
        mm1 = _mm_shuffle_epi32(mm0, 1 * 0x55);
        _mm_storeu_si128((__m128i *)(d0+16),mm1);
        _mm_storeu_si128((__m128i *)(d1+16),mm1);
        _mm_storeu_si128((__m128i *)(d2+16),mm1);
        _mm_storeu_si128((__m128i *)(d3+16),mm1);
        mm1 = _mm_shuffle_epi32(mm0, 2 * 0x55);
        _mm_storeu_si128((__m128i *)(d0+32),mm1);
        _mm_storeu_si128((__m128i *)(d1+32),mm1);
        _mm_storeu_si128((__m128i *)(d2+32),mm1);
        _mm_storeu_si128((__m128i *)(d3+32),mm1);
        mm1 = _mm_shuffle_epi32(mm0, 3 * 0x55);
        _mm_storeu_si128((__m128i *)(d0+48),mm1);
        _mm_storeu_si128((__m128i *)(d1+48),mm1);
        _mm_storeu_si128((__m128i *)(d2+48),mm1);
        _mm_storeu_si128((__m128i *)(d3+48),mm1);
        s0 += 16;
        d0 += 64;
        d1 += 64;
        d2 += 64;
        d3 += 64;
    }

    w += 4;
    while (w) /* Just so we can break out */
    {
        __m128i mm0, mm1;
        uint8_t local[16];

        memcpy(local, s0, w*4);
        mm0 = _mm_loadu_si128((const __m128i *)s0);
        mm1 = _mm_shuffle_epi32(mm0, 0);
        ((uint32_t *)d0)[3] = ((uint32_t *)d1)[3] = ((uint32_t *)d2)[3] = ((uint32_t *)d3)[3] = _mm_extract_epi32(mm1, 3);
        ((uint32_t *)d0)[2] = ((uint32_t *)d1)[2] = ((uint32_t *)d2)[2] = ((uint32_t *)d3)[2] = _mm_extract_epi32(mm1, 2);
        ((uint32_t *)d0)[1] = ((uint32_t *)d1)[1] = ((uint32_t *)d2)[1] = ((uint32_t *)d3)[1] = _mm_extract_epi32(mm1, 1);
        ((uint32_t *)d0)[0] = ((uint32_t *)d1)[0] = ((uint32_t *)d2)[0] = ((uint32_t *)d3)[0] = _mm_extract_epi32(mm1, 0);
        if (w <= 1) break;
        mm1 = _mm_shuffle_epi32(mm0, 1 * 0x55);
        ((uint32_t *)d0)[7] = ((uint32_t *)d1)[7] = ((uint32_t *)d2)[7] = ((uint32_t *)d3)[7] = _mm_extract_epi32(mm1, 3);
        ((uint32_t *)d0)[6] = ((uint32_t *)d1)[6] = ((uint32_t *)d2)[6] = ((uint32_t *)d3)[6] = _mm_extract_epi32(mm1, 2);
        ((uint32_t *)d0)[5] = ((uint32_t *)d1)[5] = ((uint32_t *)d2)[5] = ((uint32_t *)d3)[5] = _mm_extract_epi32(mm1, 1);
        ((uint32_t *)d0)[4] = ((uint32_t *)d1)[4] = ((uint32_t *)d2)[4] = ((uint32_t *)d3)[4] = _mm_extract_epi32(mm1, 0);
        if (w <= 2) break;
        mm1 = _mm_shuffle_epi32(mm0, 2 * 0x55);
        ((uint32_t *)d0)[11] = ((uint32_t *)d1)[11] = ((uint32_t *)d2)[11] = ((uint32_t *)d3)[11] = _mm_extract_epi32(mm1, 3);
        ((uint32_t *)d0)[10] = ((uint32_t *)d1)[10] = ((uint32_t *)d2)[10] = ((uint32_t *)d3)[10] = _mm_extract_epi32(mm1, 2);
        ((uint32_t *)d0)[ 9] = ((uint32_t *)d1)[ 9] = ((uint32_t *)d2)[ 9] = ((uint32_t *)d3)[ 9] = _mm_extract_epi32(mm1, 1);
        ((uint32_t *)d0)[ 8] = ((uint32_t *)d1)[ 8] = ((uint32_t *)d2)[ 8] = ((uint32_t *)d3)[ 8] = _mm_extract_epi32(mm1, 0);
        break;
    }

    return 4;
}

/*
 * For the filter here, we are using A*15/16 + B/16 and
 * A*10/16 + B*6/16.
 *
 * avg(A,B) = (A+B)/2
 * avg(A,avg(A,B)) = (A*3+B)/4
 * avg(A,avg(A,avg(A,B))) = (A*7+B)/8
 * avg(A,avg(A,avg(A,avg(A,B)))) = (A*15+B)/16
 * avg(avg(A,B),avg(A,avg(A,B))) = (A*5 + B*3)/8 = (A*10 + B*6)/16
 */

#define COMBINE(a,b,c,d,L,R) \
{ int avg = (L+R+1)>>1; int l3 = (L+avg+1)>>1; int l7 = (L+l3+1)>>1; int r3 = (R+avg+1)>>1; int r7 = (R+r3+1)>>1;\
  a = (L+l7+1)>>1; b = (avg+l3+1)>>1; c = (avg+r3+1)>>1; d = (R+r7+1)>>1; }
static int
quad_interp1_sse(uint8_t       ** ipa_restrict dsts,
                 const uint8_t ** ipa_restrict srcs,
                 ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    int tl = *s0++;
    int bl = *s1++;
    int v00, v01, v02, v03;
    __m128i mm_fwd0, mm_fwd1, mm_fwd2, mm_fwd3;
    __m128i mm_v00, mm_v01, mm_v02, mm_v03;

    /* Leading single pixel */
    COMBINE(v00, v01, v02, v03, tl, bl);
    d0[0] = d0[1] = v00; d0 += 2;
    d1[0] = d1[1] = v01; d1 += 2;
    d2[0] = d2[1] = v02; d2 += 2;
    d3[0] = d3[1] = v03; d3 += 2;

    mm_fwd0 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,v00);
    mm_fwd1 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,v01);
    mm_fwd2 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,v02);
    mm_fwd3 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,v03);

    for (w = doubler->src_w-17; w >= 0; w -= 16)
    {
        __m128i mm_s0, mm_s1, mm_avg, mm_l3, mm_l7, mm_r3, mm_r7;
        __m128i mm_v10, mm_v11, mm_v12, mm_v13;
        __m128i mm_a, mm_b, mm_c, mm_d;
        __m128i mm_ab0, mm_ab1, mm_cd0, mm_cd1;
        __m128i mm_abcd;
        // mm_fwd0,1,2,3 = single (combined) pixels carried forward.
        // Load raw pixels into mm_s0 and mm_s1 (source pixels n to n+15)
        mm_s0 = _mm_loadu_si128((const __m128i *)s0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        mm_s1 = _mm_loadu_si128((const __m128i *)s1); // mm1 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        // Combine vertically
        mm_avg = _mm_avg_epu8(mm_s0,mm_s1);           // mm3 = (mm_s0 + mm_s1)>>1
        mm_l3  = _mm_avg_epu8(mm_s0,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_s0,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_s1,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_s1,mm_r3);
        mm_v10 = _mm_avg_epu8(mm_s0,mm_l7);
        mm_v11 = _mm_avg_epu8(mm_avg,mm_l3);
        mm_v12 = _mm_avg_epu8(mm_avg,mm_r3);
        mm_v13 = _mm_avg_epu8(mm_s1,mm_r7);
        // mm_v10,11,12,13 are vertically interpolations of pixels (n to n+15).
        mm_v00 = _mm_slli_si128(mm_v10,1);
        mm_v01 = _mm_slli_si128(mm_v11,1);
        mm_v02 = _mm_slli_si128(mm_v12,1);
        mm_v03 = _mm_slli_si128(mm_v13,1);
        mm_v00 = _mm_or_si128(mm_v00,mm_fwd0);
        mm_v01 = _mm_or_si128(mm_v01,mm_fwd1);
        mm_v02 = _mm_or_si128(mm_v02,mm_fwd2);
        mm_v03 = _mm_or_si128(mm_v03,mm_fwd3);
        // mm_v0,01,02,03 are vertical interpolations of pixels (n-1 to n+14)
        // Make mm_fwd0,1,2,3 ready for next iteration.
        mm_fwd0 = _mm_srli_si128(mm_v10,15);
        mm_fwd1 = _mm_srli_si128(mm_v11,15);
        mm_fwd2 = _mm_srli_si128(mm_v12,15);
        mm_fwd3 = _mm_srli_si128(mm_v13,15);
        // Now, we need to horizontally combine the 2 sets of combined pixels
        // First row
        mm_avg = _mm_avg_epu8(mm_v00,mm_v10);
        mm_l3  = _mm_avg_epu8(mm_v00,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v00,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v10,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v10,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v00,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v10,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi8(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi8(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi8(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi8(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi16(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)d0, mm_abcd);
        mm_abcd = _mm_unpackhi_epi16(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)(d0+16), mm_abcd);
        mm_abcd = _mm_unpacklo_epi16(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d0+32), mm_abcd);
        mm_abcd = _mm_unpackhi_epi16(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d0+48), mm_abcd);
        // Second row
        mm_avg = _mm_avg_epu8(mm_v01,mm_v11);
        mm_l3  = _mm_avg_epu8(mm_v01,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v01,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v11,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v11,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v01,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v11,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi8(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi8(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi8(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi8(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi16(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)d1, mm_abcd);
        mm_abcd = _mm_unpackhi_epi16(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)(d1+16), mm_abcd);
        mm_abcd = _mm_unpacklo_epi16(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d1+32), mm_abcd);
        mm_abcd = _mm_unpackhi_epi16(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d1+48), mm_abcd);
        // Third row
        mm_avg = _mm_avg_epu8(mm_v02,mm_v12);
        mm_l3  = _mm_avg_epu8(mm_v02,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v02,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v12,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v12,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v02,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v12,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi8(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi8(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi8(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi8(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi16(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)d2, mm_abcd);
        mm_abcd = _mm_unpackhi_epi16(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)(d2+16), mm_abcd);
        mm_abcd = _mm_unpacklo_epi16(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d2+32), mm_abcd);
        mm_abcd = _mm_unpackhi_epi16(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d2+48), mm_abcd);
        // Fourth row
        mm_avg = _mm_avg_epu8(mm_v03,mm_v13);
        mm_l3  = _mm_avg_epu8(mm_v03,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v03,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v13,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v13,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v03,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v13,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi8(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi8(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi8(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi8(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi16(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)d3, mm_abcd);
        mm_abcd = _mm_unpackhi_epi16(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)(d3+16), mm_abcd);
        mm_abcd = _mm_unpacklo_epi16(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d3+32), mm_abcd);
        mm_abcd = _mm_unpackhi_epi16(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d3+48), mm_abcd);
        d0 += 64;
        d1 += 64;
        d2 += 64;
        d3 += 64;
        s0 += 16;
        s1 += 16;
    }

    w += 16;
    if (w)
    {
        __m128i mm_s0, mm_s1, mm_avg, mm_l3, mm_l7, mm_r3, mm_r7;
        __m128i mm_v10, mm_v11, mm_v12, mm_v13;
        __m128i mm_a, mm_b, mm_c, mm_d;
        __m128i mm_ab0, mm_ab1, mm_cd0, mm_cd1;
        __m128i mm_abcd;
        // mm_fwd0,1,2,3 = single (combined) pixels carried forward.
        // Load raw pixels into mm_s0 and mm_s1 (source pixels n to n+15)
        mm_s0 = _mm_loadu_si128((const __m128i *)s0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        mm_s1 = _mm_loadu_si128((const __m128i *)s1); // mm1 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        // Combine vertically
        mm_avg = _mm_avg_epu8(mm_s0,mm_s1);           // mm3 = (mm_s0 + mm_s1)>>1
        mm_l3  = _mm_avg_epu8(mm_s0,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_s0,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_s1,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_s1,mm_r3);
        mm_v10 = _mm_avg_epu8(mm_s0,mm_l7);
        mm_v11 = _mm_avg_epu8(mm_avg,mm_l3);
        mm_v12 = _mm_avg_epu8(mm_avg,mm_r3);
        mm_v13 = _mm_avg_epu8(mm_s1,mm_r7);
        // mm_v10,11,12,13 are vertically interpolations of pixels (n to n+15).
        mm_v00 = _mm_slli_si128(mm_v10,1);
        mm_v01 = _mm_slli_si128(mm_v11,1);
        mm_v02 = _mm_slli_si128(mm_v12,1);
        mm_v03 = _mm_slli_si128(mm_v13,1);
        mm_v00 = _mm_or_si128(mm_v00,mm_fwd0);
        mm_v01 = _mm_or_si128(mm_v01,mm_fwd1);
        mm_v02 = _mm_or_si128(mm_v02,mm_fwd2);
        mm_v03 = _mm_or_si128(mm_v03,mm_fwd3);
        // mm_v0,01,02,03 are vertical interpolations of pixels (n-1 to n+14)
        // Make mm_fwd0,1,2,3 ready for next iteration.
        mm_fwd0 = shift_down(mm_v10,w-1);
        mm_fwd1 = shift_down(mm_v11,w-1);
        mm_fwd2 = shift_down(mm_v12,w-1);
        mm_fwd3 = shift_down(mm_v13,w-1);
        // Now, we need to horizontally combine the 2 sets of combined pixels
        // First row
        mm_avg = _mm_avg_epu8(mm_v00,mm_v10);
        mm_l3  = _mm_avg_epu8(mm_v00,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v00,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v10,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v10,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v00,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v10,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi8(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi8(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi8(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi8(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi16(mm_ab0, mm_cd0);
        switch (w)
        {
            default: ((uint32_t *)d0)[3] = _mm_extract_epi32(mm_abcd, 3);
            case  3: ((uint32_t *)d0)[2] = _mm_extract_epi32(mm_abcd, 2);
            case  2: ((uint32_t *)d0)[1] = _mm_extract_epi32(mm_abcd, 1);
            case  1: ((uint32_t *)d0)[0] = _mm_extract_epi32(mm_abcd, 0);
        }
        mm_abcd = _mm_unpackhi_epi16(mm_ab0, mm_cd0);
        switch (w)
        {
            default: ((uint32_t *)d0)[7] = _mm_extract_epi32(mm_abcd, 3);
            case  7: ((uint32_t *)d0)[6] = _mm_extract_epi32(mm_abcd, 2);
            case  6: ((uint32_t *)d0)[5] = _mm_extract_epi32(mm_abcd, 1);
            case  5: ((uint32_t *)d0)[4] = _mm_extract_epi32(mm_abcd, 0);
            case  4: case 3: case 2: case 1: {/* Nothing to do */}
        }
        mm_abcd = _mm_unpacklo_epi16(mm_ab1, mm_cd1);
        switch (w)
        {
            default: ((uint32_t *)d0)[11] = _mm_extract_epi32(mm_abcd, 3);
            case 11: ((uint32_t *)d0)[10] = _mm_extract_epi32(mm_abcd, 2);
            case 10: ((uint32_t *)d0)[ 9] = _mm_extract_epi32(mm_abcd, 1);
            case  9: ((uint32_t *)d0)[ 8] = _mm_extract_epi32(mm_abcd, 0);
            case  8: case 7: case 6: case 5: case 4: case 3: case 2: case 1: {/* Nothing to do */}
        }
        mm_abcd = _mm_unpackhi_epi16(mm_ab1, mm_cd1);
        switch (w)
        {
            case 15: ((uint32_t *)d0)[14] = _mm_extract_epi32(mm_abcd, 2);
            case 14: ((uint32_t *)d0)[13] = _mm_extract_epi32(mm_abcd, 1);
            case 13: ((uint32_t *)d0)[12] = _mm_extract_epi32(mm_abcd, 0);
            default: {/* Nothing to do */}
        }
        // Second row
        mm_avg = _mm_avg_epu8(mm_v01,mm_v11);
        mm_l3  = _mm_avg_epu8(mm_v01,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v01,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v11,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v11,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v01,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v11,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi8(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi8(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi8(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi8(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi16(mm_ab0, mm_cd0);
        switch (w)
        {
            default: ((uint32_t *)d1)[3] = _mm_extract_epi32(mm_abcd, 3);
            case  3: ((uint32_t *)d1)[2] = _mm_extract_epi32(mm_abcd, 2);
            case  2: ((uint32_t *)d1)[1] = _mm_extract_epi32(mm_abcd, 1);
            case  1: ((uint32_t *)d1)[0] = _mm_extract_epi32(mm_abcd, 0);
        }
        mm_abcd = _mm_unpackhi_epi16(mm_ab0, mm_cd0);
        switch (w)
        {
            default: ((uint32_t *)d1)[7] = _mm_extract_epi32(mm_abcd, 3);
            case  7: ((uint32_t *)d1)[6] = _mm_extract_epi32(mm_abcd, 2);
            case  6: ((uint32_t *)d1)[5] = _mm_extract_epi32(mm_abcd, 1);
            case  5: ((uint32_t *)d1)[4] = _mm_extract_epi32(mm_abcd, 0);
            case  4: case 3: case 2: case 1: {/* Nothing to do */}
        }
        mm_abcd = _mm_unpacklo_epi16(mm_ab1, mm_cd1);
        switch (w)
        {
            default: ((uint32_t *)d1)[11] = _mm_extract_epi32(mm_abcd, 3);
            case 11: ((uint32_t *)d1)[10] = _mm_extract_epi32(mm_abcd, 2);
            case 10: ((uint32_t *)d1)[ 9] = _mm_extract_epi32(mm_abcd, 1);
            case  9: ((uint32_t *)d1)[ 8] = _mm_extract_epi32(mm_abcd, 0);
            case  8: case 7: case 6: case 5: case 4: case 3: case 2: case 1: {/* Nothing to do */}
        }
        mm_abcd = _mm_unpackhi_epi16(mm_ab1, mm_cd1);
        switch (w)
        {
            case 15: ((uint32_t *)d1)[14] = _mm_extract_epi32(mm_abcd, 2);
            case 14: ((uint32_t *)d1)[13] = _mm_extract_epi32(mm_abcd, 1);
            case 13: ((uint32_t *)d1)[12] = _mm_extract_epi32(mm_abcd, 0);
            default: {/* Nothing to do */}
        }
        // Third row
        mm_avg = _mm_avg_epu8(mm_v02,mm_v12);
        mm_l3  = _mm_avg_epu8(mm_v02,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v02,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v12,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v12,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v02,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v12,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi8(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi8(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi8(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi8(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi16(mm_ab0, mm_cd0);
        switch (w)
        {
            default: ((uint32_t *)d2)[3] = _mm_extract_epi32(mm_abcd, 3);
            case  3: ((uint32_t *)d2)[2] = _mm_extract_epi32(mm_abcd, 2);
            case  2: ((uint32_t *)d2)[1] = _mm_extract_epi32(mm_abcd, 1);
            case  1: ((uint32_t *)d2)[0] = _mm_extract_epi32(mm_abcd, 0);
        }
        mm_abcd = _mm_unpackhi_epi16(mm_ab0, mm_cd0);
        switch (w)
        {
            default: ((uint32_t *)d2)[7] = _mm_extract_epi32(mm_abcd, 3);
            case  7: ((uint32_t *)d2)[6] = _mm_extract_epi32(mm_abcd, 2);
            case  6: ((uint32_t *)d2)[5] = _mm_extract_epi32(mm_abcd, 1);
            case  5: ((uint32_t *)d2)[4] = _mm_extract_epi32(mm_abcd, 0);
            case  4: case 3: case 2: case 1: {/* Nothing to do */}
        }
        mm_abcd = _mm_unpacklo_epi16(mm_ab1, mm_cd1);
        switch (w)
        {
            default: ((uint32_t *)d2)[11] = _mm_extract_epi32(mm_abcd, 3);
            case 11: ((uint32_t *)d2)[10] = _mm_extract_epi32(mm_abcd, 2);
            case 10: ((uint32_t *)d2)[ 9] = _mm_extract_epi32(mm_abcd, 1);
            case  9: ((uint32_t *)d2)[ 8] = _mm_extract_epi32(mm_abcd, 0);
            case  8: case 7: case 6: case 5: case 4: case 3: case 2: case 1: {/* Nothing to do */}
        }
        mm_abcd = _mm_unpackhi_epi16(mm_ab1, mm_cd1);
        switch (w)
        {
            case 15: ((uint32_t *)d2)[14] = _mm_extract_epi32(mm_abcd, 2);
            case 14: ((uint32_t *)d2)[13] = _mm_extract_epi32(mm_abcd, 1);
            case 13: ((uint32_t *)d2)[12] = _mm_extract_epi32(mm_abcd, 0);
            default: {/* Nothing to do */}
        }
        // Fourth row
        mm_avg = _mm_avg_epu8(mm_v03,mm_v13);
        mm_l3  = _mm_avg_epu8(mm_v03,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v03,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v13,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v13,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v03,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v13,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi8(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi8(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi8(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi8(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi16(mm_ab0, mm_cd0);
        switch (w)
        {
            default: ((uint32_t *)d3)[3] = _mm_extract_epi32(mm_abcd, 3);
            case  3: ((uint32_t *)d3)[2] = _mm_extract_epi32(mm_abcd, 2);
            case  2: ((uint32_t *)d3)[1] = _mm_extract_epi32(mm_abcd, 1);
            case  1: ((uint32_t *)d3)[0] = _mm_extract_epi32(mm_abcd, 0);
        }
        mm_abcd = _mm_unpackhi_epi16(mm_ab0, mm_cd0);
        switch (w)
        {
            default: ((uint32_t *)d3)[7] = _mm_extract_epi32(mm_abcd, 3);
            case  7: ((uint32_t *)d3)[6] = _mm_extract_epi32(mm_abcd, 2);
            case  6: ((uint32_t *)d3)[5] = _mm_extract_epi32(mm_abcd, 1);
            case  5: ((uint32_t *)d3)[4] = _mm_extract_epi32(mm_abcd, 0);
            case  4: case 3: case 2: case 1: {/* Nothing to do */}
        }
        mm_abcd = _mm_unpacklo_epi16(mm_ab1, mm_cd1);
        switch (w)
        {
            default: ((uint32_t *)d3)[11] = _mm_extract_epi32(mm_abcd, 3);
            case 11: ((uint32_t *)d3)[10] = _mm_extract_epi32(mm_abcd, 2);
            case 10: ((uint32_t *)d3)[ 9] = _mm_extract_epi32(mm_abcd, 1);
            case  9: ((uint32_t *)d3)[ 8] = _mm_extract_epi32(mm_abcd, 0);
            case  8: case 7: case 6: case 5: case 4: case 3: case 2: case 1: {/* Nothing to do */}
        }
        mm_abcd = _mm_unpackhi_epi16(mm_ab1, mm_cd1);
        switch (w)
        {
            case 15: ((uint32_t *)d3)[14] = _mm_extract_epi32(mm_abcd, 2);
            case 14: ((uint32_t *)d3)[13] = _mm_extract_epi32(mm_abcd, 1);
            case 13: ((uint32_t *)d3)[12] = _mm_extract_epi32(mm_abcd, 0);
            default: {/* Nothing to do */}
        }
        d0 += w * 4;
        d1 += w * 4;
        d2 += w * 4;
        d3 += w * 4;
    }

    /* Trailing single pixel */
    d0[0] = d0[1] = _mm_extract_epi8(mm_fwd0,0);
    d1[0] = d1[1] = _mm_extract_epi8(mm_fwd1,0);
    d2[0] = d2[1] = _mm_extract_epi8(mm_fwd2,0);
    d3[0] = d3[1] = _mm_extract_epi8(mm_fwd3,0);

    return 4;
}

static int
quad_interp1_top_sse(uint8_t       ** ipa_restrict dsts,
                     const uint8_t ** ipa_restrict srcs,
                     ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    int l = *s0++;
    __m128i mm_fwd0;

    /* Leading single pixel */
    d0[0] = d0[1] = l; d0 += 2;
    d1[0] = d1[1] = l; d1 += 2;

    mm_fwd0 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,l);

    for (w = doubler->src_w-17; w >= 0; w -= 16)
    {
        __m128i mm_s0, mm_s1, mm_avg, mm_l3, mm_l7, mm_r3, mm_r7;
        __m128i mm_a, mm_b, mm_c, mm_d;
        __m128i mm_ab0, mm_ab1, mm_cd0, mm_cd1;
        __m128i mm_abcd;
        // mm_fwd0 = single (combined) pixel carried forward.
        // Load raw pixels into mm_s1 (source pixels n to n+15)
        mm_s1 = _mm_loadu_si128((const __m128i *)s0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        mm_s0 = _mm_slli_si128(mm_s1,1);
        mm_s0 = _mm_or_si128(mm_s0,mm_fwd0);
        // mm_s1,01,02,03 is pixels (n-1 to n+14)
        // Make mm_fwd0 ready for next iteration.
        mm_fwd0 = _mm_srli_si128(mm_s1,15);
        // Now, we need to horizontally combine the 2 sets of pixels
        mm_avg = _mm_avg_epu8(mm_s0,mm_s1);
        mm_l3  = _mm_avg_epu8(mm_s0,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_s0,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_s1,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_s1,mm_r3);
        mm_a   = _mm_avg_epu8(mm_s0,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_s1,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi8(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi8(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi8(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi8(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi16(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)d0, mm_abcd);
        _mm_storeu_si128((__m128i *)d1, mm_abcd);
        mm_abcd = _mm_unpackhi_epi16(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)(d0+16), mm_abcd);
        _mm_storeu_si128((__m128i *)(d1+16), mm_abcd);
        mm_abcd = _mm_unpacklo_epi16(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d0+32), mm_abcd);
        _mm_storeu_si128((__m128i *)(d1+32), mm_abcd);
        mm_abcd = _mm_unpackhi_epi16(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d0+48), mm_abcd);
        _mm_storeu_si128((__m128i *)(d1+48), mm_abcd);
        d0 += 64;
        d1 += 64;
        s0 += 16;
    }

    w += 16;
    if (w)
    {
        __m128i mm_s0, mm_s1, mm_avg, mm_l3, mm_l7, mm_r3, mm_r7;
        __m128i mm_a, mm_b, mm_c, mm_d;
        __m128i mm_ab0, mm_ab1, mm_cd0, mm_cd1;
        __m128i mm_abcd;
        uint8_t local[16];

        memcpy(local, s0, w+1);
        // mm_fwd0 = single (combined) pixel carried forward.
        // Load raw pixels into mm_s1 (source pixels n to n+15)
        mm_s1 = _mm_loadu_si128((const __m128i *)local); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        mm_s0 = _mm_slli_si128(mm_s1,1);
        mm_s0 = _mm_or_si128(mm_s0,mm_fwd0);
        // mm_s1,01,02,03 is pixels (n-1 to n+14)
        // Make mm_fwd0 ready for next iteration.
        mm_fwd0 = shift_down(mm_s1,w-1);
        // Now, we need to horizontally combine the 2 sets of pixels
        mm_avg = _mm_avg_epu8(mm_s0,mm_s1);
        mm_l3  = _mm_avg_epu8(mm_s0,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_s0,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_s1,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_s1,mm_r3);
        mm_a   = _mm_avg_epu8(mm_s0,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_s1,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi8(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi8(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi8(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi8(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi16(mm_ab0, mm_cd0);
        switch (w)
        {
            default: ((uint32_t *)d0)[3] = ((uint32_t *)d1)[3] = _mm_extract_epi32(mm_abcd, 3);
            case  3: ((uint32_t *)d0)[2] = ((uint32_t *)d1)[2] = _mm_extract_epi32(mm_abcd, 2);
            case  2: ((uint32_t *)d0)[1] = ((uint32_t *)d1)[1] = _mm_extract_epi32(mm_abcd, 1);
            case  1: ((uint32_t *)d0)[0] = ((uint32_t *)d1)[0] = _mm_extract_epi32(mm_abcd, 0);
        }
        mm_abcd = _mm_unpackhi_epi16(mm_ab0, mm_cd0);
        switch (w)
        {
            default: ((uint32_t *)d0)[7] = ((uint32_t *)d1)[7] = _mm_extract_epi32(mm_abcd, 3);
            case  7: ((uint32_t *)d0)[6] = ((uint32_t *)d1)[6] = _mm_extract_epi32(mm_abcd, 2);
            case  6: ((uint32_t *)d0)[5] = ((uint32_t *)d1)[5] = _mm_extract_epi32(mm_abcd, 1);
            case  5: ((uint32_t *)d0)[4] = ((uint32_t *)d1)[4] = _mm_extract_epi32(mm_abcd, 0);
            case  4: case 3: case 2: case 1: {/* Nothing to do */}
        }
        mm_abcd = _mm_unpacklo_epi16(mm_ab1, mm_cd1);
        switch (w)
        {
            default: ((uint32_t *)d0)[11] = ((uint32_t *)d1)[11] = _mm_extract_epi32(mm_abcd, 3);
            case 11: ((uint32_t *)d0)[10] = ((uint32_t *)d1)[10] = _mm_extract_epi32(mm_abcd, 2);
            case 10: ((uint32_t *)d0)[ 9] = ((uint32_t *)d1)[ 9] = _mm_extract_epi32(mm_abcd, 1);
            case  9: ((uint32_t *)d0)[ 8] = ((uint32_t *)d1)[ 8] = _mm_extract_epi32(mm_abcd, 0);
            case  8: case 7: case 6: case 5: case 4: case 3: case 2: case 1: {/* Nothing to do */}
        }
        mm_abcd = _mm_unpackhi_epi16(mm_ab1, mm_cd1);
        switch (w)
        {
            case 15: ((uint32_t *)d0)[14] = ((uint32_t *)d1)[14] = _mm_extract_epi32(mm_abcd, 2);
            case 14: ((uint32_t *)d0)[13] = ((uint32_t *)d1)[13] = _mm_extract_epi32(mm_abcd, 1);
            case 13: ((uint32_t *)d0)[12] = ((uint32_t *)d1)[12] = _mm_extract_epi32(mm_abcd, 0);
            default: {/* Nothing to do */}
        }
        d0 += w * 4;
        d1 += w * 4;
    }

    /* Trailing single pixel */
    d0[0] = d0[1] = d1[0] = d1[1] = _mm_extract_epi8(mm_fwd0,0);

    return 2;
}

static int
quad_interp3_sse(uint8_t       ** ipa_restrict dsts,
                 const uint8_t ** ipa_restrict srcs,
                 ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    const uint8_t * ipa_restrict s0 = srcs[0];
    const uint8_t * ipa_restrict s1 = srcs[1];
    int t_r = s0[0];
    int t_g = s0[1];
    int t_b = s0[2];
    int b_r = s1[0];
    int b_g = s1[1];
    int b_b = s1[2];
    int v00_r, v00_g, v00_b, v01_r, v01_g, v01_b, v02_r, v02_g, v02_b, v03_r, v03_g, v03_b;
    __m128i mm_shuf0, mm_shuf1, mm_shuf2, mm_shuf3, mm_shuf4, mm_shuf5;
    __m128i mm_shuf6, mm_shuf7, mm_shuf8, mm_shuf9, mm_shuf10, mm_shuf11;

    /* Leading single pixel */
    COMBINE(v00_r, v01_r, v02_r, v03_r, t_r, b_r);
    d0[0] = d0[3] = v00_r; d1[0] = d1[3] = v01_r; d2[0] = d2[3] = v02_r; d3[0] = d3[3] = v03_r;
    COMBINE(v00_g, v01_g, v02_g, v03_g, t_g, b_g);
    d0[1] = d0[4] = v00_g; d1[1] = d1[4] = v01_g; d2[1] = d2[4] = v02_g; d3[1] = d3[4] = v03_g;
    COMBINE(v00_b, v01_b, v02_b, v03_b, t_b, b_b);
    d0[2] = d0[5] = v00_b; d1[2] = d1[5] = v01_b; d2[2] = d2[5] = v02_b; d3[2] = d3[5] = v03_b;
    d0 += 6;
    d1 += 6;
    d2 += 6;
    d3 += 6;

    mm_shuf0 = _mm_set_epi8(ZZ, 5, 4, 3,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ, 2, 1, 0);
    mm_shuf1 = _mm_set_epi8( 3,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ, 2, 1, 0,ZZ,ZZ,ZZ);
    mm_shuf2 = _mm_set_epi8(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ, 2, 1, 0,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ);
    mm_shuf3 = _mm_set_epi8(ZZ,ZZ,ZZ,ZZ, 2, 1, 0,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ);
    mm_shuf4 = _mm_set_epi8(ZZ,ZZ,ZZ,ZZ,ZZ, 8, 7, 6,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ);
    mm_shuf5 = _mm_set_epi8(ZZ,ZZ, 8, 7, 6,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ, 5, 4);
    mm_shuf6 = _mm_set_epi8( 7, 6,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ, 5, 4, 3,ZZ,ZZ);
    mm_shuf7 = _mm_set_epi8(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ, 5, 4, 3,ZZ,ZZ,ZZ,ZZ,ZZ);
    mm_shuf8 = _mm_set_epi8(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,11,10, 9,ZZ,ZZ,ZZ,ZZ);
    mm_shuf9 = _mm_set_epi8(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,11,10, 9,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ);
    mm_shuf10= _mm_set_epi8(ZZ,ZZ,ZZ,11,10, 9,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ, 8);
    mm_shuf11= _mm_set_epi8(11,10, 9,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ, 8, 7, 6,ZZ);
    for (w = doubler->src_w-5; w >= 0; w -= 4)
    {
        __m128i mm_s0, mm_s1, mm_avg, mm_l3, mm_l7, mm_r3, mm_r7;
        __m128i mm_v00, mm_v01, mm_v02, mm_v03;
        __m128i mm_v10, mm_v11, mm_v12, mm_v13;
        __m128i mm_a, mm_b, mm_c, mm_d;
        __m128i mm_ab0, mm_abcd;
        // Load raw pixels into mm_s0 and mm_s1 (source bytes n to n+15)
        mm_s0 = _mm_loadu_si128((const __m128i *)s0); // mm_s0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        mm_s1 = _mm_loadu_si128((const __m128i *)s1); // mm_s1 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        // Combine vertically
        mm_avg = _mm_avg_epu8(mm_s0,mm_s1);           // mm3 = (mm_s0 + mm_s1)>>1
        mm_l3  = _mm_avg_epu8(mm_s0,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_s0,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_s1,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_s1,mm_r3);
        mm_v00 = _mm_avg_epu8(mm_s0,mm_l7);
        mm_v01 = _mm_avg_epu8(mm_avg,mm_l3);
        mm_v02 = _mm_avg_epu8(mm_avg,mm_r3);
        mm_v03 = _mm_avg_epu8(mm_s1,mm_r7);
        // mm_v0,01,02,03 are vertical interpolations of pixels bytes (n to n+15)
        mm_v10 = _mm_srli_si128(mm_v00,3);
        mm_v11 = _mm_srli_si128(mm_v01,3);
        mm_v12 = _mm_srli_si128(mm_v02,3);
        mm_v13 = _mm_srli_si128(mm_v03,3);
        // mm_v10,11,12,13 are vertically interpolations of pixel bytes (n+3 to n+15).
        // Now, we need to horizontally combine the 2 sets of combined pixels
        // First row
        mm_avg = _mm_avg_epu8(mm_v00,mm_v10);
        mm_l3  = _mm_avg_epu8(mm_v00,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v00,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v10,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v10,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v00,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v10,mm_r7);
        // Rejig bytes into output order
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf0); // mm_abcd = 00ffeedd000000000000000000ccbbaa
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf1); // mm_abcd = dd000000000000000000ccbbaa000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf2); // mm_abcd = 00000000000000ccbbaa000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf3); // mm_abcd = 00000000ccbbaa000000000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        _mm_storeu_si128((__m128i *)d0, mm_abcd);
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf4); // mm_abcd = 0000000000iihhgg0000000000000000
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf5); // mm_abcd = 0000iihhgg000000000000000000ffee
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf6); // mm_abcd = hhgg000000000000000000ffeedd0000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf7); // mm_abcd = 0000000000000000ffeedd0000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        _mm_storeu_si128((__m128i *)(d0+16), mm_abcd);
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf8); // mm_abcd = 000000000000000000llkkjj00000000
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf9); // mm_abcd = 000000000000llkkjj00000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf10);// mm_abcd = 000000llkkjj000000000000000000ii
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf11);// mm_abcd = llkkjj000000000000000000iihhgg00
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        _mm_storeu_si128((__m128i *)(d0+32), mm_abcd);
        // Second row
        mm_avg = _mm_avg_epu8(mm_v01,mm_v11);
        mm_l3  = _mm_avg_epu8(mm_v01,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v01,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v11,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v11,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v01,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v11,mm_r7);
        // Rejig bytes into output order
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf0); // mm_abcd = 00ffeedd000000000000000000ccbbaa
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf1); // mm_abcd = dd000000000000000000ccbbaa000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf2); // mm_abcd = 00000000000000ccbbaa000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf3); // mm_abcd = 00000000ccbbaa000000000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        _mm_storeu_si128((__m128i *)d1, mm_abcd);
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf4); // mm_abcd = 0000000000iihhgg0000000000000000
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf5); // mm_abcd = 0000iihhgg000000000000000000ffee
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf6); // mm_abcd = hhgg000000000000000000ffeedd0000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf7); // mm_abcd = 0000000000000000ffeedd0000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        _mm_storeu_si128((__m128i *)(d1+16), mm_abcd);
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf8); // mm_abcd = 000000000000000000llkkjj00000000
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf9); // mm_abcd = 000000000000llkkjj00000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf10);// mm_abcd = 000000llkkjj000000000000000000ii
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf11);// mm_abcd = llkkjj000000000000000000iihhgg00
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        _mm_storeu_si128((__m128i *)(d1+32), mm_abcd);
        // Third row
        mm_avg = _mm_avg_epu8(mm_v02,mm_v12);
        mm_l3  = _mm_avg_epu8(mm_v02,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v02,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v12,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v12,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v02,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v12,mm_r7);
        // Rejig bytes into output order
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf0); // mm_abcd = 00ffeedd000000000000000000ccbbaa
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf1); // mm_abcd = dd000000000000000000ccbbaa000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf2); // mm_abcd = 00000000000000ccbbaa000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf3); // mm_abcd = 00000000ccbbaa000000000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        _mm_storeu_si128((__m128i *)d2, mm_abcd);
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf4); // mm_abcd = 0000000000iihhgg0000000000000000
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf5); // mm_abcd = 0000iihhgg000000000000000000ffee
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf6); // mm_abcd = hhgg000000000000000000ffeedd0000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf7); // mm_abcd = 0000000000000000ffeedd0000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        _mm_storeu_si128((__m128i *)(d2+16), mm_abcd);
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf8); // mm_abcd = 000000000000000000llkkjj00000000
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf9); // mm_abcd = 000000000000llkkjj00000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf10);// mm_abcd = 000000llkkjj000000000000000000ii
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf11);// mm_abcd = llkkjj000000000000000000iihhgg00
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        _mm_storeu_si128((__m128i *)(d2+32), mm_abcd);
        // Fourth row
        mm_avg = _mm_avg_epu8(mm_v03,mm_v13);
        mm_l3  = _mm_avg_epu8(mm_v03,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v03,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v13,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v13,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v03,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v13,mm_r7);
        // Rejig bytes into output order
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf0); // mm_abcd = 00ffeedd000000000000000000ccbbaa
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf1); // mm_abcd = dd000000000000000000ccbbaa000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf2); // mm_abcd = 00000000000000ccbbaa000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf3); // mm_abcd = 00000000ccbbaa000000000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        _mm_storeu_si128((__m128i *)d3, mm_abcd);
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf4); // mm_abcd = 0000000000iihhgg0000000000000000
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf5); // mm_abcd = 0000iihhgg000000000000000000ffee
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf6); // mm_abcd = hhgg000000000000000000ffeedd0000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf7); // mm_abcd = 0000000000000000ffeedd0000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        _mm_storeu_si128((__m128i *)(d3+16), mm_abcd);
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf8); // mm_abcd = 000000000000000000llkkjj00000000
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf9); // mm_abcd = 000000000000llkkjj00000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf10);// mm_abcd = 000000llkkjj000000000000000000ii
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf11);// mm_abcd = llkkjj000000000000000000iihhgg00
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        _mm_storeu_si128((__m128i *)(d3+32), mm_abcd);
        d0 += 48;
        d1 += 48;
        d2 += 48;
        d3 += 48;
        s0 += 12;
        s1 += 12;
    }

    w += 4;
    if (w)
    {
        __m128i mm_s0, mm_s1, mm_avg, mm_l3, mm_l7, mm_r3, mm_r7;
        __m128i mm_v00, mm_v01, mm_v02, mm_v03;
        __m128i mm_v10, mm_v11, mm_v12, mm_v13;
        __m128i mm_a, mm_b, mm_c, mm_d;
        __m128i mm_ab0, mm_abcd;
        uint8_t local0[16], local1[16];

        memcpy(local0, s0, (w+1)*4);
        memcpy(local1, s1, (w+1)*4);
        // Load raw pixels into mm_s0 and mm_s1 (source bytes n to n+15)
        mm_s0 = _mm_loadu_si128((const __m128i *)local0); // mm_s0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        mm_s1 = _mm_loadu_si128((const __m128i *)local1); // mm_s1 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        // Combine vertically
        mm_avg = _mm_avg_epu8(mm_s0,mm_s1);           // mm3 = (mm_s0 + mm_s1)>>1
        mm_l3  = _mm_avg_epu8(mm_s0,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_s0,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_s1,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_s1,mm_r3);
        mm_v00 = _mm_avg_epu8(mm_s0,mm_l7);
        mm_v01 = _mm_avg_epu8(mm_avg,mm_l3);
        mm_v02 = _mm_avg_epu8(mm_avg,mm_r3);
        mm_v03 = _mm_avg_epu8(mm_s1,mm_r7);
        // mm_v0,01,02,03 are vertical interpolations of pixels bytes (n to n+15)
        mm_v10 = _mm_srli_si128(mm_v00,3);
        mm_v11 = _mm_srli_si128(mm_v01,3);
        mm_v12 = _mm_srli_si128(mm_v02,3);
        mm_v13 = _mm_srli_si128(mm_v03,3);
        // mm_v10,11,12,13 are vertically interpolations of pixel bytes (n+3 to n+15).
        // Now, we need to horizontally combine the 2 sets of combined pixels
        // First row
        mm_avg = _mm_avg_epu8(mm_v00,mm_v10);
        mm_l3  = _mm_avg_epu8(mm_v00,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v00,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v10,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v10,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v00,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v10,mm_r7);
        // Rejig bytes into output order
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf0); // mm_abcd = 00ffeedd000000000000000000ccbbaa
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf1); // mm_abcd = dd000000000000000000ccbbaa000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf2); // mm_abcd = 00000000000000ccbbaa000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf3); // mm_abcd = 00000000ccbbaa000000000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        switch (w)
        {
        default:
            ((uint32_t *)d0)[3] = _mm_extract_epi32(mm_abcd, 3);
        case 1:
            ((uint32_t *)d0)[2] = _mm_extract_epi32(mm_abcd, 2);
            ((uint32_t *)d0)[1] = _mm_extract_epi32(mm_abcd, 1);
            ((uint32_t *)d0)[0] = _mm_extract_epi32(mm_abcd, 0);
        }
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf4); // mm_abcd = 0000000000iihhgg0000000000000000
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf5); // mm_abcd = 0000iihhgg000000000000000000ffee
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf6); // mm_abcd = hhgg000000000000000000ffeedd0000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf7); // mm_abcd = 0000000000000000ffeedd0000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        switch (w)
        {
        default:
            ((uint32_t *)d0)[7] = _mm_extract_epi32(mm_abcd, 3);
            ((uint32_t *)d0)[6] = _mm_extract_epi32(mm_abcd, 2);
        case 2:
            ((uint32_t *)d0)[5] = _mm_extract_epi32(mm_abcd, 1);
            ((uint32_t *)d0)[4] = _mm_extract_epi32(mm_abcd, 0);
        case 1: {/* Do nothing */}
        }
        mm_abcd = _mm_shuffle_epi8(mm_c,mm_shuf10);// mm_abcd = 000000llkkjj000000000000000000ii
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf11);// mm_abcd = llkkjj000000000000000000iihhgg00
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        if (w == 3)
            ((uint32_t *)d0)[8] = _mm_extract_epi32(mm_abcd, 0);
        // Second row
        mm_avg = _mm_avg_epu8(mm_v01,mm_v11);
        mm_l3  = _mm_avg_epu8(mm_v01,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v01,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v11,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v11,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v01,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v11,mm_r7);
        // Rejig bytes into output order
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf0); // mm_abcd = 00ffeedd000000000000000000ccbbaa
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf1); // mm_abcd = dd000000000000000000ccbbaa000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf2); // mm_abcd = 00000000000000ccbbaa000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf3); // mm_abcd = 00000000ccbbaa000000000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        switch (w)
        {
        default:
            ((uint32_t *)d1)[3] = _mm_extract_epi32(mm_abcd, 3);
        case 1:
            ((uint32_t *)d1)[2] = _mm_extract_epi32(mm_abcd, 2);
            ((uint32_t *)d1)[1] = _mm_extract_epi32(mm_abcd, 1);
            ((uint32_t *)d1)[0] = _mm_extract_epi32(mm_abcd, 0);
        }
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf4); // mm_abcd = 0000000000iihhgg0000000000000000
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf5); // mm_abcd = 0000iihhgg000000000000000000ffee
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf6); // mm_abcd = hhgg000000000000000000ffeedd0000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf7); // mm_abcd = 0000000000000000ffeedd0000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        switch (w)
        {
        default:
            ((uint32_t *)d1)[7] = _mm_extract_epi32(mm_abcd, 3);
            ((uint32_t *)d1)[6] = _mm_extract_epi32(mm_abcd, 2);
        case 2:
            ((uint32_t *)d1)[5] = _mm_extract_epi32(mm_abcd, 1);
            ((uint32_t *)d1)[4] = _mm_extract_epi32(mm_abcd, 0);
        case 1: {/* Do nothing */}
        }
        mm_abcd = _mm_shuffle_epi8(mm_c,mm_shuf10);// mm_abcd = 000000llkkjj000000000000000000ii
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf11);// mm_abcd = llkkjj000000000000000000iihhgg00
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        if (w == 3)
            ((uint32_t *)d1)[8] = _mm_extract_epi32(mm_abcd, 0);
        // Third row
        mm_avg = _mm_avg_epu8(mm_v02,mm_v12);
        mm_l3  = _mm_avg_epu8(mm_v02,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v02,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v12,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v12,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v02,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v12,mm_r7);
        // Rejig bytes into output order
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf0); // mm_abcd = 00ffeedd000000000000000000ccbbaa
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf1); // mm_abcd = dd000000000000000000ccbbaa000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf2); // mm_abcd = 00000000000000ccbbaa000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf3); // mm_abcd = 00000000ccbbaa000000000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        switch (w)
        {
        default:
            ((uint32_t *)d2)[3] = _mm_extract_epi32(mm_abcd, 3);
        case 1:
            ((uint32_t *)d2)[2] = _mm_extract_epi32(mm_abcd, 2);
            ((uint32_t *)d2)[1] = _mm_extract_epi32(mm_abcd, 1);
            ((uint32_t *)d2)[0] = _mm_extract_epi32(mm_abcd, 0);
        }
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf4); // mm_abcd = 0000000000iihhgg0000000000000000
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf5); // mm_abcd = 0000iihhgg000000000000000000ffee
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf6); // mm_abcd = hhgg000000000000000000ffeedd0000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf7); // mm_abcd = 0000000000000000ffeedd0000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        switch (w)
        {
        default:
            ((uint32_t *)d2)[7] = _mm_extract_epi32(mm_abcd, 3);
            ((uint32_t *)d2)[6] = _mm_extract_epi32(mm_abcd, 2);
        case 2:
            ((uint32_t *)d2)[5] = _mm_extract_epi32(mm_abcd, 1);
            ((uint32_t *)d2)[4] = _mm_extract_epi32(mm_abcd, 0);
        case 1: {/* Do nothing */}
        }
        mm_abcd = _mm_shuffle_epi8(mm_c,mm_shuf10);// mm_abcd = 000000llkkjj000000000000000000ii
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf11);// mm_abcd = llkkjj000000000000000000iihhgg00
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        if (w == 3)
            ((uint32_t *)d2)[8] = _mm_extract_epi32(mm_abcd, 0);
        // Fourth row
        mm_avg = _mm_avg_epu8(mm_v03,mm_v13);
        mm_l3  = _mm_avg_epu8(mm_v03,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v03,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v13,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v13,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v03,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v13,mm_r7);
        // Rejig bytes into output order
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf0); // mm_abcd = 00ffeedd000000000000000000ccbbaa
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf1); // mm_abcd = dd000000000000000000ccbbaa000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf2); // mm_abcd = 00000000000000ccbbaa000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf3); // mm_abcd = 00000000ccbbaa000000000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        switch (w)
        {
        default:
            ((uint32_t *)d3)[3] = _mm_extract_epi32(mm_abcd, 3);
        case 1:
            ((uint32_t *)d3)[2] = _mm_extract_epi32(mm_abcd, 2);
            ((uint32_t *)d3)[1] = _mm_extract_epi32(mm_abcd, 1);
            ((uint32_t *)d3)[0] = _mm_extract_epi32(mm_abcd, 0);
        }
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf4); // mm_abcd = 0000000000iihhgg0000000000000000
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf5); // mm_abcd = 0000iihhgg000000000000000000ffee
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf6); // mm_abcd = hhgg000000000000000000ffeedd0000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf7); // mm_abcd = 0000000000000000ffeedd0000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        switch (w)
        {
        default:
            ((uint32_t *)d3)[7] = _mm_extract_epi32(mm_abcd, 3);
            ((uint32_t *)d3)[6] = _mm_extract_epi32(mm_abcd, 2);
        case 2:
            ((uint32_t *)d3)[5] = _mm_extract_epi32(mm_abcd, 1);
            ((uint32_t *)d3)[4] = _mm_extract_epi32(mm_abcd, 0);
        case 1: {/* Do nothing */}
        }
        mm_abcd = _mm_shuffle_epi8(mm_c,mm_shuf10);// mm_abcd = 000000llkkjj000000000000000000ii
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf11);// mm_abcd = llkkjj000000000000000000iihhgg00
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        if (w == 3)
            ((uint32_t *)d3)[8] = _mm_extract_epi32(mm_abcd, 0);
        d0 += w * 12;
        d1 += w * 12;
        d2 += w * 12;
        d3 += w * 12;
        s0 += w * 3;
        s1 += w * 3;
    }
    t_r = s0[0];
    t_g = s0[1];
    t_b = s0[2];
    b_r = s1[0];
    b_g = s1[1];
    b_b = s1[2];
    COMBINE(v00_r, v01_r, v02_r, v03_r, t_r, b_r);
    COMBINE(v00_g, v01_g, v02_g, v03_g, t_g, b_g);
    COMBINE(v00_b, v01_b, v02_b, v03_b, t_b, b_b);

    /* Trailing single pixel */
    d0[0] = d0[3] = v00_r; d0[1] = d0[4] = v00_g; d0[2] = d0[5] = v00_b;
    d1[0] = d1[3] = v01_r; d1[1] = d1[4] = v01_g; d1[2] = d1[5] = v01_b;
    d2[0] = d2[3] = v02_r; d2[1] = d2[4] = v02_g; d2[2] = d2[5] = v02_b;
    d3[0] = d3[3] = v03_r; d3[1] = d3[4] = v03_g; d3[2] = d3[5] = v03_b;

    return 4;
}

static int
quad_interp3_top_sse(uint8_t       ** ipa_restrict dsts,
                     const uint8_t ** ipa_restrict srcs,
                     ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    int l_r = s0[0];
    int l_g = s0[1];
    int l_b = s0[2];
    __m128i mm_shuf0, mm_shuf1, mm_shuf2, mm_shuf3, mm_shuf4, mm_shuf5;
    __m128i mm_shuf6, mm_shuf7, mm_shuf8, mm_shuf9, mm_shuf10, mm_shuf11;

    /* Leading single pixel */
    d0[0] = d0[3] = d1[0] = d1[3] = l_r;
    d0[1] = d0[4] = d1[1] = d1[4] = l_g;
    d0[2] = d0[5] = d1[2] = d1[5] = l_b;
    d0 += 6;
    d1 += 6;

    mm_shuf0 = _mm_set_epi8(ZZ, 5, 4, 3,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ, 2, 1, 0);
    mm_shuf1 = _mm_set_epi8( 3,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ, 2, 1, 0,ZZ,ZZ,ZZ);
    mm_shuf2 = _mm_set_epi8(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ, 2, 1, 0,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ);
    mm_shuf3 = _mm_set_epi8(ZZ,ZZ,ZZ,ZZ, 2, 1, 0,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ);
    mm_shuf4 = _mm_set_epi8(ZZ,ZZ,ZZ,ZZ,ZZ, 8, 7, 6,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ);
    mm_shuf5 = _mm_set_epi8(ZZ,ZZ, 8, 7, 6,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ, 5, 4);
    mm_shuf6 = _mm_set_epi8( 7, 6,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ, 5, 4, 3,ZZ,ZZ);
    mm_shuf7 = _mm_set_epi8(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ, 5, 4, 3,ZZ,ZZ,ZZ,ZZ,ZZ);
    mm_shuf8 = _mm_set_epi8(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,11,10, 9,ZZ,ZZ,ZZ,ZZ);
    mm_shuf9 = _mm_set_epi8(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,11,10, 9,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ);
    mm_shuf10= _mm_set_epi8(ZZ,ZZ,ZZ,11,10, 9,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ, 8);
    mm_shuf11= _mm_set_epi8(11,10, 9,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ, 8, 7, 6,ZZ);
    for (w = doubler->src_w-5; w >= 0; w -= 4)
    {
        __m128i mm_s0, mm_s1, mm_avg, mm_l3, mm_l7, mm_r3, mm_r7;
        __m128i mm_a, mm_b, mm_c, mm_d;
        __m128i mm_ab0, mm_abcd;
        // Load raw pixels into mm_s0 and mm_s1 (source bytes n to n+15)
        mm_s0 = _mm_loadu_si128((const __m128i *)s0); // mm_s1 = ppoonnmmllkkjjiihhggffeeddccbbaa
        // mm_s0 is pixel bytes (n-4 to n+12)
        mm_s1 = _mm_srli_si128(mm_s0,3);
        // mm_s1 is pixel bytes (n to n+15).
        // Now, we need to horizontally combine the 2 sets of pixels
        mm_avg = _mm_avg_epu8(mm_s0,mm_s1);
        mm_l3  = _mm_avg_epu8(mm_s0,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_s0,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_s1,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_s1,mm_r3);
        mm_a   = _mm_avg_epu8(mm_s0,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_s1,mm_r7);
        // Rejig bytes into output order
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf0); // mm_abcd = 00ffeedd000000000000000000ccbbaa
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf1); // mm_abcd = dd000000000000000000ccbbaa000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf2); // mm_abcd = 00000000000000ccbbaa000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf3); // mm_abcd = 00000000ccbbaa000000000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        _mm_storeu_si128((__m128i *)d0, mm_abcd);
        _mm_storeu_si128((__m128i *)d1, mm_abcd);
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf4); // mm_abcd = 0000000000iihhgg0000000000000000
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf5); // mm_abcd = 0000iihhgg000000000000000000ffee
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf6); // mm_abcd = hhgg000000000000000000ffeedd0000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf7); // mm_abcd = 0000000000000000ffeedd0000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        _mm_storeu_si128((__m128i *)(d0+16), mm_abcd);
        _mm_storeu_si128((__m128i *)(d1+16), mm_abcd);
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf8); // mm_abcd = 000000000000000000llkkjj00000000
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf9); // mm_abcd = 000000000000llkkjj00000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf10);// mm_abcd = 000000llkkjj000000000000000000ii
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf11);// mm_abcd = llkkjj000000000000000000iihhgg00
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        _mm_storeu_si128((__m128i *)(d0+32), mm_abcd);
        _mm_storeu_si128((__m128i *)(d1+32), mm_abcd);
        d0 += 48;
        d1 += 48;
        s0 += 12;
    }

    w += 4;
    if (w)
    {
        __m128i mm_s0, mm_s1, mm_avg, mm_l3, mm_l7, mm_r3, mm_r7;
        __m128i mm_a, mm_b, mm_c, mm_d;
        __m128i mm_ab0, mm_abcd;
        uint8_t local[16];

        memcpy(local, s0, (w+1)*4);
        // Load raw pixels into mm_s0 and mm_s1 (source bytes n to n+15)
        mm_s0 = _mm_loadu_si128((const __m128i *)s0); // mm_s1 = ppoonnmmllkkjjiihhggffeeddccbbaa
        // mm_s0 is pixel bytes (n-4 to n+12)
        mm_s1 = _mm_srli_si128(mm_s0,3);
        // mm_s1 is pixel bytes (n to n+15).
        // Now, we need to horizontally combine the 2 sets of pixels
        mm_avg = _mm_avg_epu8(mm_s0,mm_s1);
        mm_l3  = _mm_avg_epu8(mm_s0,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_s0,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_s1,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_s1,mm_r3);
        mm_a   = _mm_avg_epu8(mm_s0,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_s1,mm_r7);
        // Rejig bytes into output order
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf0); // mm_abcd = 00ffeedd000000000000000000ccbbaa
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf1); // mm_abcd = dd000000000000000000ccbbaa000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf2); // mm_abcd = 00000000000000ccbbaa000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf3); // mm_abcd = 00000000ccbbaa000000000000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        switch (w)
        {
        default:
            ((uint32_t *)d0)[3] = ((uint32_t *)d1)[3] = _mm_extract_epi32(mm_abcd, 3);
        case 1:
            ((uint32_t *)d0)[2] = ((uint32_t *)d1)[2] = _mm_extract_epi32(mm_abcd, 2);
            ((uint32_t *)d0)[1] = ((uint32_t *)d1)[1] = _mm_extract_epi32(mm_abcd, 1);
            ((uint32_t *)d0)[0] = ((uint32_t *)d1)[0] = _mm_extract_epi32(mm_abcd, 0);
        }
        mm_abcd = _mm_shuffle_epi8(mm_a,mm_shuf4); // mm_abcd = 0000000000iihhgg0000000000000000
        mm_ab0  = _mm_shuffle_epi8(mm_b,mm_shuf5); // mm_abcd = 0000iihhgg000000000000000000ffee
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_c,mm_shuf6); // mm_abcd = hhgg000000000000000000ffeedd0000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf7); // mm_abcd = 0000000000000000ffeedd0000000000
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        switch (w)
        {
        default:
            ((uint32_t *)d0)[7] = ((uint32_t *)d1)[7] = _mm_extract_epi32(mm_abcd, 3);
            ((uint32_t *)d0)[6] = ((uint32_t *)d1)[6] = _mm_extract_epi32(mm_abcd, 2);
        case 2:
            ((uint32_t *)d0)[5] = ((uint32_t *)d1)[5] = _mm_extract_epi32(mm_abcd, 1);
            ((uint32_t *)d0)[4] = ((uint32_t *)d1)[4] = _mm_extract_epi32(mm_abcd, 0);
        case 1: {/* Do nothing */}
        }
        mm_abcd = _mm_shuffle_epi8(mm_c,mm_shuf10);// mm_abcd = 000000llkkjj000000000000000000ii
        mm_ab0  = _mm_shuffle_epi8(mm_d,mm_shuf11);// mm_abcd = llkkjj000000000000000000iihhgg00
        mm_abcd = _mm_or_si128(mm_abcd, mm_ab0);
        if (w == 3)
            ((uint32_t *)d0)[8] = ((uint32_t *)d1)[8] = _mm_extract_epi32(mm_abcd, 0);
        d0 += w * 12;
        d1 += w * 12;
        s0 += w;
    }

    /* Trailing single pixel */
    d0[0] = d0[3] = d1[0] = d1[3] = s0[0];
    d0[1] = d0[4] = d1[1] = d1[4] = s0[1];
    d0[2] = d0[5] = d1[2] = d1[5] = s0[2];

    return 2;
}

static int
quad_interp4_sse(uint8_t       ** ipa_restrict dsts,
                 const uint8_t ** ipa_restrict srcs,
                 ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
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
    int v00_r, v00_g, v00_b, v00_k, v01_r, v01_g, v01_b, v01_k, v02_r, v02_g, v02_b, v02_k, v03_r, v03_g, v03_b, v03_k;
    __m128i mm_fwd0, mm_fwd1, mm_fwd2, mm_fwd3;
    __m128i mm_v00, mm_v01, mm_v02, mm_v03;

    /* Leading single pixel */
    COMBINE(v00_r, v01_r, v02_r, v03_r, tl_r, bl_r);
    COMBINE(v00_g, v01_g, v02_g, v03_g, tl_g, bl_g);
    COMBINE(v00_b, v01_b, v02_b, v03_b, tl_b, bl_b);
    COMBINE(v00_k, v01_k, v02_k, v03_k, tl_k, bl_k);
    d0[0] = d0[4] = v00_r; d0[1] = d0[5] = v00_g; d0[2] = d0[6] = v00_b; d0[3] = d0[7] = v00_k;
    d1[0] = d1[4] = v01_r; d1[1] = d1[5] = v01_g; d1[2] = d1[6] = v01_b; d1[3] = d1[7] = v01_k;
    d2[0] = d2[4] = v02_r; d2[1] = d2[5] = v02_g; d2[2] = d2[6] = v02_b; d2[3] = d2[7] = v02_k;
    d3[0] = d3[4] = v03_r; d3[1] = d3[5] = v03_g; d3[2] = d3[6] = v03_b; d3[3] = d3[7] = v03_k;
    d0 += 8;
    d1 += 8;
    d2 += 8;
    d3 += 8;

    mm_fwd0 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,v00_k,v00_b,v00_g,v00_r);
    mm_fwd1 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,v01_k,v01_b,v01_g,v01_r);
    mm_fwd2 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,v02_k,v02_b,v02_g,v02_r);
    mm_fwd3 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,v03_k,v03_b,v03_g,v03_r);

    for (w = doubler->src_w-5; w >= 0; w -= 4)
    {
        __m128i mm_s0, mm_s1, mm_avg, mm_l3, mm_l7, mm_r3, mm_r7;
        __m128i mm_v10, mm_v11, mm_v12, mm_v13;
        __m128i mm_a, mm_b, mm_c, mm_d;
        __m128i mm_ab0, mm_ab1, mm_cd0, mm_cd1;
        __m128i mm_abcd;
        // mm_fwd0,1,2,3 = single (combined) pixels carried forward.
        // Load raw pixels into mm_s0 and mm_s1 (source pixels n to n+15)
        mm_s0 = _mm_loadu_si128((const __m128i *)s0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        mm_s1 = _mm_loadu_si128((const __m128i *)s1); // mm1 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        // Combine vertically
        mm_avg = _mm_avg_epu8(mm_s0,mm_s1);           // mm3 = (mm_s0 + mm_s1)>>1
        mm_l3  = _mm_avg_epu8(mm_s0,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_s0,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_s1,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_s1,mm_r3);
        mm_v10 = _mm_avg_epu8(mm_s0,mm_l7);
        mm_v11 = _mm_avg_epu8(mm_avg,mm_l3);
        mm_v12 = _mm_avg_epu8(mm_avg,mm_r3);
        mm_v13 = _mm_avg_epu8(mm_s1,mm_r7);
        // mm_v10,11,12,13 are vertically interpolations of pixels (n to n+3).
        mm_v00 = _mm_slli_si128(mm_v10,4);
        mm_v01 = _mm_slli_si128(mm_v11,4);
        mm_v02 = _mm_slli_si128(mm_v12,4);
        mm_v03 = _mm_slli_si128(mm_v13,4);
        mm_v00 = _mm_or_si128(mm_v00,mm_fwd0);
        mm_v01 = _mm_or_si128(mm_v01,mm_fwd1);
        mm_v02 = _mm_or_si128(mm_v02,mm_fwd2);
        mm_v03 = _mm_or_si128(mm_v03,mm_fwd3);
        // mm_v0,01,02,03 are vertical interpolations of pixels (n-1 to n+2)
        // Make mm_fwd0,1,2,3 ready for next iteration.
        mm_fwd0 = _mm_srli_si128(mm_v10,12);
        mm_fwd1 = _mm_srli_si128(mm_v11,12);
        mm_fwd2 = _mm_srli_si128(mm_v12,12);
        mm_fwd3 = _mm_srli_si128(mm_v13,12);
        // Now, we need to horizontally combine the 2 sets of combined pixels
        // First row
        mm_avg = _mm_avg_epu8(mm_v00,mm_v10);
        mm_l3  = _mm_avg_epu8(mm_v00,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v00,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v10,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v10,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v00,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v10,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi32(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi32(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi32(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi32(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi64(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)d0, mm_abcd);
        mm_abcd = _mm_unpackhi_epi64(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)(d0+16), mm_abcd);
        mm_abcd = _mm_unpacklo_epi64(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d0+32), mm_abcd);
        mm_abcd = _mm_unpackhi_epi64(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d0+48), mm_abcd);
        // Second row
        mm_avg = _mm_avg_epu8(mm_v01,mm_v11);
        mm_l3  = _mm_avg_epu8(mm_v01,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v01,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v11,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v11,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v01,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v11,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi32(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi32(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi32(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi32(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi64(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)d1, mm_abcd);
        mm_abcd = _mm_unpackhi_epi64(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)(d1+16), mm_abcd);
        mm_abcd = _mm_unpacklo_epi64(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d1+32), mm_abcd);
        mm_abcd = _mm_unpackhi_epi64(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d1+48), mm_abcd);
        // Third row
        mm_avg = _mm_avg_epu8(mm_v02,mm_v12);
        mm_l3  = _mm_avg_epu8(mm_v02,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v02,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v12,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v12,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v02,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v12,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi32(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi32(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi32(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi32(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi64(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)d2, mm_abcd);
        mm_abcd = _mm_unpackhi_epi64(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)(d2+16), mm_abcd);
        mm_abcd = _mm_unpacklo_epi64(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d2+32), mm_abcd);
        mm_abcd = _mm_unpackhi_epi64(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d2+48), mm_abcd);
        // Fourth row
        mm_avg = _mm_avg_epu8(mm_v03,mm_v13);
        mm_l3  = _mm_avg_epu8(mm_v03,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v03,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v13,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v13,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v03,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v13,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi32(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi32(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi32(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi32(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi64(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)d3, mm_abcd);
        mm_abcd = _mm_unpackhi_epi64(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)(d3+16), mm_abcd);
        mm_abcd = _mm_unpacklo_epi64(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d3+32), mm_abcd);
        mm_abcd = _mm_unpackhi_epi64(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d3+48), mm_abcd);
        d0 += 64;
        d1 += 64;
        d2 += 64;
        d3 += 64;
        s0 += 16;
        s1 += 16;
    }

    w += 4;
    if (w)
    {
        __m128i mm_s0, mm_s1, mm_avg, mm_l3, mm_l7, mm_r3, mm_r7;
        __m128i mm_v10, mm_v11, mm_v12, mm_v13;
        __m128i mm_a, mm_b, mm_c, mm_d;
        __m128i mm_ab0, mm_ab1, mm_cd0, mm_cd1;
        __m128i mm_abcd;
        uint8_t local0[16], local1[16];

        memcpy(local0, s0, (w+1)*4);
        memcpy(local1, s1, (w+1)*4);
        // mm_fwd0,1,2,3 = single (combined) pixels carried forward.
        // Load raw pixels into mm_s0 and mm_s1 (source pixels n to n+15)
        mm_s0 = _mm_loadu_si128((const __m128i *)local0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        mm_s1 = _mm_loadu_si128((const __m128i *)local1); // mm1 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        // Combine vertically
        mm_avg = _mm_avg_epu8(mm_s0,mm_s1);           // mm3 = (mm_s0 + mm_s1)>>1
        mm_l3  = _mm_avg_epu8(mm_s0,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_s0,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_s1,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_s1,mm_r3);
        mm_v10 = _mm_avg_epu8(mm_s0,mm_l7);
        mm_v11 = _mm_avg_epu8(mm_avg,mm_l3);
        mm_v12 = _mm_avg_epu8(mm_avg,mm_r3);
        mm_v13 = _mm_avg_epu8(mm_s1,mm_r7);
        // mm_v10,11,12,13 are vertically interpolations of pixels (n to n+3).
        mm_v00 = _mm_slli_si128(mm_v10,4);
        mm_v01 = _mm_slli_si128(mm_v11,4);
        mm_v02 = _mm_slli_si128(mm_v12,4);
        mm_v03 = _mm_slli_si128(mm_v13,4);
        mm_v00 = _mm_or_si128(mm_v00,mm_fwd0);
        mm_v01 = _mm_or_si128(mm_v01,mm_fwd1);
        mm_v02 = _mm_or_si128(mm_v02,mm_fwd2);
        mm_v03 = _mm_or_si128(mm_v03,mm_fwd3);
        // mm_v0,01,02,03 are vertical interpolations of pixels (n-1 to n+2)
        // Make mm_fwd0,1,2,3 ready for next iteration.
        mm_fwd0 = shift_down(mm_v10,(w-1)*4);
        mm_fwd1 = shift_down(mm_v11,(w-1)*4);
        mm_fwd2 = shift_down(mm_v12,(w-1)*4);
        mm_fwd3 = shift_down(mm_v13,(w-1)*4);
        // Now, we need to horizontally combine the 2 sets of combined pixels
        // First row
        mm_avg = _mm_avg_epu8(mm_v00,mm_v10);
        mm_l3  = _mm_avg_epu8(mm_v00,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v00,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v10,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v10,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v00,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v10,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi32(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi32(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi32(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi32(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi64(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)d0, mm_abcd);
        mm_abcd = _mm_unpackhi_epi64(mm_ab0, mm_cd0);
        if (w >= 2)
            _mm_storeu_si128((__m128i *)(d0+16), mm_abcd);
        mm_abcd = _mm_unpacklo_epi64(mm_ab1, mm_cd1);
        if (w >= 3)
            _mm_storeu_si128((__m128i *)(d0+32), mm_abcd);
        // Second row
        mm_avg = _mm_avg_epu8(mm_v01,mm_v11);
        mm_l3  = _mm_avg_epu8(mm_v01,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v01,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v11,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v11,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v01,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v11,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi32(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi32(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi32(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi32(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi64(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)d1, mm_abcd);
        mm_abcd = _mm_unpackhi_epi64(mm_ab0, mm_cd0);
        if (w >= 2)
            _mm_storeu_si128((__m128i *)(d1+16), mm_abcd);
        mm_abcd = _mm_unpacklo_epi64(mm_ab1, mm_cd1);
        if (w >= 3)
            _mm_storeu_si128((__m128i *)(d1+32), mm_abcd);
        // Third row
        mm_avg = _mm_avg_epu8(mm_v02,mm_v12);
        mm_l3  = _mm_avg_epu8(mm_v02,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v02,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v12,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v12,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v02,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v12,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi32(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi32(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi32(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi32(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi64(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)d2, mm_abcd);
        mm_abcd = _mm_unpackhi_epi64(mm_ab0, mm_cd0);
        if (w >= 2)
            _mm_storeu_si128((__m128i *)(d2+16), mm_abcd);
        mm_abcd = _mm_unpacklo_epi64(mm_ab1, mm_cd1);
        if (w >= 3)
            _mm_storeu_si128((__m128i *)(d2+32), mm_abcd);
        // Fourth row
        mm_avg = _mm_avg_epu8(mm_v03,mm_v13);
        mm_l3  = _mm_avg_epu8(mm_v03,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_v03,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_v13,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_v13,mm_r3);
        mm_a   = _mm_avg_epu8(mm_v03,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_v13,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi32(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi32(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi32(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi32(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi64(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)d3, mm_abcd);
        mm_abcd = _mm_unpackhi_epi64(mm_ab0, mm_cd0);
        if (w >= 2)
            _mm_storeu_si128((__m128i *)(d3+16), mm_abcd);
        mm_abcd = _mm_unpacklo_epi64(mm_ab1, mm_cd1);
        if (w >= 3)
            _mm_storeu_si128((__m128i *)(d3+32), mm_abcd);
        d0 += w * 16;
        d1 += w * 16;
        d2 += w * 16;
        d3 += w * 16;
    }

    /* Trailing single pixel */
    d0[0] = d0[4] = _mm_extract_epi8(mm_fwd0,0);
    d0[1] = d0[5] = _mm_extract_epi8(mm_fwd0,1);
    d0[2] = d0[6] = _mm_extract_epi8(mm_fwd0,2);
    d0[3] = d0[7] = _mm_extract_epi8(mm_fwd0,3);
    d1[0] = d1[4] = _mm_extract_epi8(mm_fwd1,0);
    d1[1] = d1[5] = _mm_extract_epi8(mm_fwd1,1);
    d1[2] = d1[6] = _mm_extract_epi8(mm_fwd1,2);
    d1[3] = d1[7] = _mm_extract_epi8(mm_fwd1,3);
    d2[0] = d2[4] = _mm_extract_epi8(mm_fwd2,0);
    d2[1] = d2[5] = _mm_extract_epi8(mm_fwd2,1);
    d2[2] = d2[6] = _mm_extract_epi8(mm_fwd2,2);
    d2[3] = d2[7] = _mm_extract_epi8(mm_fwd2,3);
    d3[0] = d3[4] = _mm_extract_epi8(mm_fwd3,0);
    d3[1] = d3[5] = _mm_extract_epi8(mm_fwd3,1);
    d3[2] = d3[6] = _mm_extract_epi8(mm_fwd3,2);
    d3[3] = d3[7] = _mm_extract_epi8(mm_fwd3,3);

    return 4;
}

static int
quad_interp4_top_sse(uint8_t       ** ipa_restrict dsts,
                     const uint8_t ** ipa_restrict srcs,
                     ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    const uint8_t * ipa_restrict s0 = srcs[0];
    int l_r = *s0++;
    int l_g = *s0++;
    int l_b = *s0++;
    int l_k = *s0++;
    __m128i mm_fwd0;

    /* Leading single pixel */
    d0[0] = d0[4] = d1[0] = d1[4] = l_r;
    d0[1] = d0[5] = d1[1] = d1[5] = l_g;
    d0[2] = d0[6] = d1[2] = d1[6] = l_b;
    d0[3] = d0[7] = d1[3] = d1[7] = l_k;
    d0 += 8;
    d1 += 8;

    mm_fwd0 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,l_k,l_b,l_g,l_r);

    for (w = doubler->src_w-5; w >= 0; w -= 4)
    {
        __m128i mm_s0, mm_s1, mm_avg, mm_l3, mm_l7, mm_r3, mm_r7;
        __m128i mm_a, mm_b, mm_c, mm_d;
        __m128i mm_ab0, mm_ab1, mm_cd0, mm_cd1;
        __m128i mm_abcd;
        // mm_fwd0 = single (combined) pixel carried forward.
        // Load raw pixels into mm_s1 (source pixels n to n+15)
        mm_s1 = _mm_loadu_si128((const __m128i *)s0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        mm_s0 = _mm_slli_si128(mm_s1,4);
        mm_s0 = _mm_or_si128(mm_s0,mm_fwd0);
        // mm_s1,01,02,03 is pixels (n-1 to n+14)
        // Make mm_fwd0 ready for next iteration.
        mm_fwd0 = _mm_srli_si128(mm_s1,12);
        // Now, we need to horizontally combine the 2 sets of pixels
        mm_avg = _mm_avg_epu8(mm_s0,mm_s1);
        mm_l3  = _mm_avg_epu8(mm_s0,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_s0,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_s1,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_s1,mm_r3);
        mm_a   = _mm_avg_epu8(mm_s0,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_s1,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi32(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi32(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi32(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi32(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi64(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)d0, mm_abcd);
        _mm_storeu_si128((__m128i *)d1, mm_abcd);
        mm_abcd = _mm_unpackhi_epi64(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)(d0+16), mm_abcd);
        _mm_storeu_si128((__m128i *)(d1+16), mm_abcd);
        mm_abcd = _mm_unpacklo_epi64(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d0+32), mm_abcd);
        _mm_storeu_si128((__m128i *)(d1+32), mm_abcd);
        mm_abcd = _mm_unpackhi_epi64(mm_ab1, mm_cd1);
        _mm_storeu_si128((__m128i *)(d0+48), mm_abcd);
        _mm_storeu_si128((__m128i *)(d1+48), mm_abcd);
        d0 += 64;
        d1 += 64;
        s0 += 16;
    }

    w += 4;
    if (w)
    {
        __m128i mm_s0, mm_s1, mm_avg, mm_l3, mm_l7, mm_r3, mm_r7;
        __m128i mm_a, mm_b, mm_c, mm_d;
        __m128i mm_ab0, mm_ab1, mm_cd0, mm_cd1;
        __m128i mm_abcd;
        uint8_t local[16];

        memcpy(local, s0, (w+1)*4);
        // mm_fwd0 = single (combined) pixel carried forward.
        // Load raw pixels into mm_s1 (source pixels n to n+15)
        mm_s1 = _mm_loadu_si128((const __m128i *)local); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        mm_s0 = _mm_slli_si128(mm_s1,4);
        mm_s0 = _mm_or_si128(mm_s0,mm_fwd0);
        // mm_s1,01,02,03 is pixels (n-1 to n+14)
        // Make mm_fwd0 ready for next iteration.
        mm_fwd0 = shift_down(mm_s1,(w-1)*4);
        // Now, we need to horizontally combine the 2 sets of pixels
        mm_avg = _mm_avg_epu8(mm_s0,mm_s1);
        mm_l3  = _mm_avg_epu8(mm_s0,mm_avg);
        mm_l7  = _mm_avg_epu8(mm_s0,mm_l3);
        mm_r3  = _mm_avg_epu8(mm_s1,mm_avg);
        mm_r7  = _mm_avg_epu8(mm_s1,mm_r3);
        mm_a   = _mm_avg_epu8(mm_s0,mm_l7);
        mm_b   = _mm_avg_epu8(mm_avg,mm_l3);
        mm_c   = _mm_avg_epu8(mm_avg,mm_r3);
        mm_d   = _mm_avg_epu8(mm_s1,mm_r7);
        // Rejig bytes into output order
        mm_ab0 = _mm_unpacklo_epi32(mm_a, mm_b);
        mm_ab1 = _mm_unpackhi_epi32(mm_a, mm_b);
        mm_cd0 = _mm_unpacklo_epi32(mm_c, mm_d);
        mm_cd1 = _mm_unpackhi_epi32(mm_c, mm_d);
        mm_abcd = _mm_unpacklo_epi64(mm_ab0, mm_cd0);
        _mm_storeu_si128((__m128i *)d0, mm_abcd);
        _mm_storeu_si128((__m128i *)d1, mm_abcd);
        mm_abcd = _mm_unpackhi_epi64(mm_ab0, mm_cd0);
        if (w >= 2)
        {
            _mm_storeu_si128((__m128i *)(d0+16), mm_abcd);
            _mm_storeu_si128((__m128i *)(d1+16), mm_abcd);
        }
        mm_abcd = _mm_unpacklo_epi64(mm_ab1, mm_cd1);
        if (w >= 3)
        {
            _mm_storeu_si128((__m128i *)(d0+32), mm_abcd);
            _mm_storeu_si128((__m128i *)(d1+32), mm_abcd);
        }
        d0 += w * 16;
        d1 += w * 16;
        s0 += w;
    }

    /* Trailing single pixel */
    d0[0] = d0[4] = d1[0] = d1[4] = _mm_extract_epi8(mm_fwd0,0);
    d0[1] = d0[5] = d1[1] = d1[5] = _mm_extract_epi8(mm_fwd0,1);
    d0[2] = d0[6] = d1[2] = d1[6] = _mm_extract_epi8(mm_fwd0,2);
    d0[3] = d0[7] = d1[3] = d1[7] = _mm_extract_epi8(mm_fwd0,3);

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

static void
quad_mitchell1_to_tmp(uint32_t *t0, const uint8_t *s0, uint32_t src_w)
{
    int32_t w;
    int a, b, c;
    __m128i mm_w0123, mm_w4567, mm_w7654, mm_w3210;

    /* Leading pixels */
    a = s0[0];
    b = s0[1];
    c = s0[2];
    *t0++ = COMBINE2(a,a,a,b);
    *t0++ = COMBINE3(a,a,a,b);
    *t0++ = COMBINE0(a,a,b,c);
    *t0++ = COMBINE1(a,a,b,c);
    *t0++ = COMBINE2(a,a,b,c);
    *t0++ = COMBINE3(a,a,b,c);

    mm_w0123 = _mm_set_epi32(MW3,MW2,MW1,MW0);
    mm_w4567 = _mm_set_epi32(MW7,MW6,MW5,MW4);
    mm_w7654 = _mm_set_epi32(MW4,MW5,MW6,MW7);
    mm_w3210 = _mm_set_epi32(MW0,MW1,MW2,MW3);

    for (w = src_w-3-13; w >= 0; w -= 13)
    {
        __m128i mm_s0, mm_s00, mm_s01, mm_s02, mm_s03;

        mm_s0 = _mm_loadu_si128((const __m128i *)s0);     // mm_s0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        // 0: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 1: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 2: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 3: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 4: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 5: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 6: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 7: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 8: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 9: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 10: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 11: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 12: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        s0 += 13;
    }

    w += 13;
    while (w) /* So we can break out */
    {
        __m128i mm_s0, mm_s00, mm_s01, mm_s02, mm_s03;
        uint8_t local[16];

        memcpy(local, s0, w+3);
        s0 += w;
        mm_s0 = _mm_loadu_si128((const __m128i *)local);     // mm_s0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        // 0: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        if (w == 1) break;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 1: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        if (w == 2) break;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 2: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        if (w == 3) break;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 3: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        if (w == 4) break;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 4: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        if (w == 5) break;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 5: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        if (w == 6) break;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 6: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        if (w == 7) break;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 7: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        if (w == 8) break;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 8: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        if (w == 9) break;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 9: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        if (w == 10) break;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 10: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        if (w == 11) break;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 11: Explode to 32 bits
        mm_s03 = _mm_cvtepu8_epi32(mm_s0); // mm_s01 = 000000dd000000cc000000bb000000aa
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        _mm_storeu_si128((__m128i *)t0, mm_s00);
        t0 += 4;
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        break;
    }
    a = s0[0];
    b = s0[1];
    c = s0[2];

    /* Trailing pixels */
    *t0++ = COMBINE0(a,b,c,c);
    *t0++ = COMBINE1(a,b,c,c);
    *t0++ = COMBINE2(a,b,c,c);
    *t0++ = COMBINE3(a,b,c,c);
    *t0++ = COMBINE0(b,c,c,c);
    *t0++ = COMBINE1(b,c,c,c);
}

static int
quad_mitchell(uint8_t        * ipa_restrict d0,
              uint8_t        * ipa_restrict d1,
              uint8_t        * ipa_restrict d2,
              uint8_t        * ipa_restrict d3,
              const uint32_t * ipa_restrict t0,
              const uint32_t * ipa_restrict t1,
              const uint32_t * ipa_restrict t2,
              const uint32_t * ipa_restrict t3,
              int32_t          src_w)
{
    int32_t w;
    __m128i mm_w0, mm_w1, mm_w2, mm_w3, mm_w4, mm_w5, mm_w6, mm_w7, mm_round;

    mm_w0 = _mm_set1_epi32(MW0);
    mm_w1 = _mm_set1_epi32(MW1);
    mm_w2 = _mm_set1_epi32(MW2);
    mm_w3 = _mm_set1_epi32(MW3);
    mm_w4 = _mm_set1_epi32(MW4);
    mm_w5 = _mm_set1_epi32(MW5);
    mm_w6 = _mm_set1_epi32(MW6);
    mm_w7 = _mm_set1_epi32(MW7);
    mm_round = _mm_set1_epi32(WEIGHT_ROUND<<WEIGHT_SHIFT);
    for (w = 4*src_w-4; w >= 0; w -= 4)
    {
        __m128i mm_s0, mm_s1, mm_s2, mm_s3;
        __m128i mm_s00, mm_s01, mm_s02, mm_s03;
        __m128i mm_s10, mm_s11, mm_s12, mm_s13;
        __m128i mm_s20, mm_s21, mm_s22, mm_s23;
        __m128i mm_s30, mm_s31, mm_s32, mm_s33;

        mm_s0 = _mm_loadu_si128((const __m128i *)t0);
        // Combine vertically
        mm_s00 = _mm_mullo_epi32(mm_s0, mm_w0);
        mm_s10 = _mm_mullo_epi32(mm_s0, mm_w4);
        mm_s20 = _mm_mullo_epi32(mm_s0, mm_w7);
        mm_s30 = _mm_mullo_epi32(mm_s0, mm_w3);
        mm_s1 = _mm_loadu_si128((const __m128i *)t1);
        mm_s01 = _mm_mullo_epi32(mm_s1, mm_w1);
        mm_s11 = _mm_mullo_epi32(mm_s1, mm_w5);
        mm_s21 = _mm_mullo_epi32(mm_s1, mm_w6);
        mm_s31 = _mm_mullo_epi32(mm_s1, mm_w2);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s01);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s11);
        mm_s20 = _mm_add_epi32(mm_s20, mm_s21);
        mm_s30 = _mm_add_epi32(mm_s30, mm_s31);
        mm_s2 = _mm_loadu_si128((const __m128i *)t2);
        mm_s02 = _mm_mullo_epi32(mm_s2, mm_w2);
        mm_s12 = _mm_mullo_epi32(mm_s2, mm_w6);
        mm_s22 = _mm_mullo_epi32(mm_s2, mm_w5);
        mm_s32 = _mm_mullo_epi32(mm_s2, mm_w1);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s02);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s12);
        mm_s20 = _mm_add_epi32(mm_s20, mm_s22);
        mm_s30 = _mm_add_epi32(mm_s30, mm_s32);
        mm_s3 = _mm_loadu_si128((const __m128i *)t3);
        mm_s03 = _mm_mullo_epi32(mm_s3, mm_w3);
        mm_s13 = _mm_mullo_epi32(mm_s3, mm_w7);
        mm_s23 = _mm_mullo_epi32(mm_s3, mm_w4);
        mm_s33 = _mm_mullo_epi32(mm_s3, mm_w0);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s03);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s13);
        mm_s20 = _mm_add_epi32(mm_s20, mm_s23);
        mm_s30 = _mm_add_epi32(mm_s30, mm_s33);
        // Round, shift, extract
        mm_s00 = _mm_add_epi32(mm_s00, mm_round);
        mm_s10 = _mm_add_epi32(mm_s10, mm_round);
        mm_s20 = _mm_add_epi32(mm_s20, mm_round);
        mm_s30 = _mm_add_epi32(mm_s30, mm_round);
        mm_s00 = _mm_srai_epi32(mm_s00, WEIGHT_SHIFT*2);
        mm_s10 = _mm_srai_epi32(mm_s10, WEIGHT_SHIFT*2);
        mm_s20 = _mm_srai_epi32(mm_s20, WEIGHT_SHIFT*2);
        mm_s30 = _mm_srai_epi32(mm_s30, WEIGHT_SHIFT*2);
        mm_s00 = _mm_packus_epi32(mm_s00, mm_s00); // Clamp to 0 to 65535 range.
        mm_s10 = _mm_packus_epi32(mm_s10, mm_s10); // Clamp to 0 to 65535 range.
        mm_s20 = _mm_packus_epi32(mm_s20, mm_s20); // Clamp to 0 to 65535 range.
        mm_s30 = _mm_packus_epi32(mm_s30, mm_s30); // Clamp to 0 to 65535 range.
        mm_s00 = _mm_packus_epi16(mm_s00, mm_s00); // Clamp to 0 to 255 range.
        mm_s10 = _mm_packus_epi16(mm_s10, mm_s10); // Clamp to 0 to 255 range.
        mm_s20 = _mm_packus_epi16(mm_s20, mm_s20); // Clamp to 0 to 255 range.
        mm_s30 = _mm_packus_epi16(mm_s30, mm_s30); // Clamp to 0 to 255 range.
        *(uint32_t *)d0 = _mm_extract_epi32(mm_s00, 0);
        *(uint32_t *)d1 = _mm_extract_epi32(mm_s10, 0);
        *(uint32_t *)d2 = _mm_extract_epi32(mm_s20, 0);
        *(uint32_t *)d3 = _mm_extract_epi32(mm_s30, 0);
        d0 += 4;
        d1 += 4;
        d2 += 4;
        d3 += 4;
        t0 += 4;
        t1 += 4;
        t2 += 4;
        t3 += 4;
    }

    w += 4;
    if (w)
    {
        __m128i mm_s0, mm_s1, mm_s2, mm_s3;
        __m128i mm_s00, mm_s01, mm_s02, mm_s03;
        __m128i mm_s10, mm_s11, mm_s12, mm_s13;
        __m128i mm_s20, mm_s21, mm_s22, mm_s23;
        __m128i mm_s30, mm_s31, mm_s32, mm_s33;

        // tmp is safe to overread.
        mm_s0 = _mm_loadu_si128((const __m128i *)t0);
        // Combine vertically
        mm_s00 = _mm_mullo_epi32(mm_s0, mm_w0);
        mm_s10 = _mm_mullo_epi32(mm_s0, mm_w4);
        mm_s20 = _mm_mullo_epi32(mm_s0, mm_w7);
        mm_s30 = _mm_mullo_epi32(mm_s0, mm_w3);
        mm_s1 = _mm_loadu_si128((const __m128i *)t1);
        mm_s01 = _mm_mullo_epi32(mm_s1, mm_w1);
        mm_s11 = _mm_mullo_epi32(mm_s1, mm_w5);
        mm_s21 = _mm_mullo_epi32(mm_s1, mm_w6);
        mm_s31 = _mm_mullo_epi32(mm_s1, mm_w2);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s01);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s11);
        mm_s20 = _mm_add_epi32(mm_s20, mm_s21);
        mm_s30 = _mm_add_epi32(mm_s30, mm_s31);
        mm_s2 = _mm_loadu_si128((const __m128i *)t2);
        mm_s02 = _mm_mullo_epi32(mm_s2, mm_w2);
        mm_s12 = _mm_mullo_epi32(mm_s2, mm_w6);
        mm_s22 = _mm_mullo_epi32(mm_s2, mm_w5);
        mm_s32 = _mm_mullo_epi32(mm_s2, mm_w1);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s02);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s12);
        mm_s20 = _mm_add_epi32(mm_s20, mm_s22);
        mm_s30 = _mm_add_epi32(mm_s30, mm_s32);
        mm_s3 = _mm_loadu_si128((const __m128i *)t3);
        mm_s03 = _mm_mullo_epi32(mm_s3, mm_w3);
        mm_s13 = _mm_mullo_epi32(mm_s3, mm_w7);
        mm_s23 = _mm_mullo_epi32(mm_s3, mm_w4);
        mm_s33 = _mm_mullo_epi32(mm_s3, mm_w0);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s03);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s13);
        mm_s20 = _mm_add_epi32(mm_s20, mm_s23);
        mm_s30 = _mm_add_epi32(mm_s30, mm_s33);
        // Round, shift, extract
        mm_s00 = _mm_add_epi32(mm_s00, mm_round);
        mm_s10 = _mm_add_epi32(mm_s10, mm_round);
        mm_s20 = _mm_add_epi32(mm_s20, mm_round);
        mm_s30 = _mm_add_epi32(mm_s30, mm_round);
        mm_s00 = _mm_srai_epi32(mm_s00, WEIGHT_SHIFT*2);
        mm_s10 = _mm_srai_epi32(mm_s10, WEIGHT_SHIFT*2);
        mm_s20 = _mm_srai_epi32(mm_s20, WEIGHT_SHIFT*2);
        mm_s30 = _mm_srai_epi32(mm_s30, WEIGHT_SHIFT*2);
        mm_s00 = _mm_packus_epi32(mm_s00, mm_s00); // Clamp to 0 to 65535 range.
        mm_s10 = _mm_packus_epi32(mm_s10, mm_s10); // Clamp to 0 to 65535 range.
        mm_s20 = _mm_packus_epi32(mm_s20, mm_s20); // Clamp to 0 to 65535 range.
        mm_s30 = _mm_packus_epi32(mm_s30, mm_s30); // Clamp to 0 to 65535 range.
        mm_s00 = _mm_packus_epi16(mm_s00, mm_s00); // Clamp to 0 to 255 range.
        mm_s10 = _mm_packus_epi16(mm_s10, mm_s10); // Clamp to 0 to 255 range.
        mm_s20 = _mm_packus_epi16(mm_s20, mm_s20); // Clamp to 0 to 255 range.
        mm_s30 = _mm_packus_epi16(mm_s30, mm_s30); // Clamp to 0 to 255 range.
        switch (w)
        {
            case 2:
                d0[2] = _mm_extract_epi8(mm_s00, 2);
                d1[2] = _mm_extract_epi8(mm_s10, 2);
                d2[2] = _mm_extract_epi8(mm_s20, 2);
                d3[2] = _mm_extract_epi8(mm_s30, 2);
            case 3:
                d0[1] = _mm_extract_epi8(mm_s00, 1);
                d1[1] = _mm_extract_epi8(mm_s10, 1);
                d2[1] = _mm_extract_epi8(mm_s20, 1);
                d3[1] = _mm_extract_epi8(mm_s30, 1);
            case 1:
                d0[0] = _mm_extract_epi8(mm_s00, 0);
                d1[0] = _mm_extract_epi8(mm_s10, 0);
                d2[0] = _mm_extract_epi8(mm_s20, 0);
                d3[0] = _mm_extract_epi8(mm_s30, 0);
        }
    }

    return 4;
}

static int
quad_mitchell1_sse(uint8_t       ** ipa_restrict dsts,
                   const uint8_t ** ipa_restrict srcs,
                   ipa_doubler    * ipa_restrict doubler)
{
    uint8_t * ipa_restrict d0 = ((uint8_t **)dsts)[0];
    uint8_t * ipa_restrict d1 = ((uint8_t **)dsts)[1];
    uint8_t * ipa_restrict d2 = ((uint8_t **)dsts)[2];
    uint8_t * ipa_restrict d3 = ((uint8_t **)dsts)[3];
    const uint8_t * ipa_restrict s3 = ((const uint8_t **)srcs)[3];
    uint32_t *t3 = doubler->tmp + (((doubler->tmp_y    )&3) * doubler->tmp_stride);
    uint32_t *t2 = doubler->tmp + (((doubler->tmp_y + 3)&3) * doubler->tmp_stride);
    uint32_t *t1 = doubler->tmp + (((doubler->tmp_y + 2)&3) * doubler->tmp_stride);
    uint32_t *t0 = doubler->tmp + (((doubler->tmp_y + 1)&3) * doubler->tmp_stride);

    quad_mitchell1_to_tmp(t3, s3, doubler->src_w);
    doubler->tmp_y++;

    return quad_mitchell(d0, d1, d2, d3, t0, t1, t2, t3, doubler->src_w);
}

static int
quad_mitchell_final(uint8_t       ** ipa_restrict dsts,
                    ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = ((uint8_t **)dsts)[0];
    uint8_t * ipa_restrict d1 = ((uint8_t **)dsts)[1];
    uint32_t *t3 = doubler->tmp + (((doubler->tmp_y    )&3) * doubler->tmp_stride);
    uint32_t *t2 = doubler->tmp + (((doubler->tmp_y + 3)&3) * doubler->tmp_stride);
    uint32_t *t1 = doubler->tmp + (((doubler->tmp_y + 2)&3) * doubler->tmp_stride);
    uint32_t *t0 = doubler->tmp + (((doubler->tmp_y + 1)&3) * doubler->tmp_stride);
    __m128i mm_w0, mm_w1, mm_w2, mm_w3, mm_w4, mm_w5, mm_w6, mm_w7, mm_round;
    int tmp_y = doubler->tmp_y++;

    if (tmp_y == 0)
        return 0;
    if (tmp_y == 1)
    {
        /* A,A,A,B reversed */
        t0 = t3;
        t1 = t3 = t2;
    }
    else if (tmp_y == 2)
    {
        /* A,A,B,C */
        uint8_t * ipa_restrict d2 = ((uint8_t **)dsts)[2];
        uint8_t * ipa_restrict d3 = ((uint8_t **)dsts)[3];
        return quad_mitchell(d0, d1, d2, d3, t1, t1, t2, t3, doubler->src_w * doubler->channels);
    }
    else if (tmp_y == doubler->src_h)
    {
        /* A,B,C,C */
        uint8_t * ipa_restrict d2 = ((uint8_t **)dsts)[2];
        uint8_t * ipa_restrict d3 = ((uint8_t **)dsts)[3];
        return quad_mitchell(d0, d1, d2, d3, t0, t1, t2, t2, doubler->src_w * doubler->channels);
    }
    else
    {
        assert(tmp_y == doubler->src_h+1);
        /* A,B,B,B */
        t2 = t3 = t1;
    }

    mm_w0 = _mm_set1_epi32(MW0);
    mm_w1 = _mm_set1_epi32(MW1);
    mm_w2 = _mm_set1_epi32(MW2);
    mm_w3 = _mm_set1_epi32(MW3);
    mm_w4 = _mm_set1_epi32(MW4);
    mm_w5 = _mm_set1_epi32(MW5);
    mm_w6 = _mm_set1_epi32(MW6);
    mm_w7 = _mm_set1_epi32(MW7);
    mm_round = _mm_set1_epi32(WEIGHT_ROUND<<WEIGHT_SHIFT);
    for (w = doubler->src_w * 4 * doubler->channels - 4; w >= 0; w -= 4)
    {
        __m128i mm_s0, mm_s1, mm_s2, mm_s3;
        __m128i mm_s00, mm_s01, mm_s02, mm_s03;
        __m128i mm_s10, mm_s11, mm_s12, mm_s13;

        // tmp allows for overreading.
        mm_s0 = _mm_loadu_si128((const __m128i *)t0);
        // Combine vertically
        mm_s00 = _mm_mullo_epi32(mm_s0, mm_w0);
        mm_s10 = _mm_mullo_epi32(mm_s0, mm_w4);
        mm_s1 = _mm_loadu_si128((const __m128i *)t1);
        mm_s01 = _mm_mullo_epi32(mm_s1, mm_w1);
        mm_s11 = _mm_mullo_epi32(mm_s1, mm_w5);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s01);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s11);
        mm_s2 = _mm_loadu_si128((const __m128i *)t2);
        mm_s02 = _mm_mullo_epi32(mm_s2, mm_w2);
        mm_s12 = _mm_mullo_epi32(mm_s2, mm_w6);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s02);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s12);
        mm_s3 = _mm_loadu_si128((const __m128i *)t3);
        mm_s03 = _mm_mullo_epi32(mm_s3, mm_w3);
        mm_s13 = _mm_mullo_epi32(mm_s3, mm_w7);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s03);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s13);
        // Round, shift, extract
        mm_s00 = _mm_add_epi32(mm_s00, mm_round);
        mm_s10 = _mm_add_epi32(mm_s10, mm_round);
        mm_s00 = _mm_srai_epi32(mm_s00, WEIGHT_SHIFT*2);
        mm_s10 = _mm_srai_epi32(mm_s10, WEIGHT_SHIFT*2);
        mm_s00 = _mm_packus_epi32(mm_s00, mm_s00); // Clamp to 0 to 65535 range.
        mm_s10 = _mm_packus_epi32(mm_s10, mm_s10); // Clamp to 0 to 65535 range.
        mm_s00 = _mm_packus_epi16(mm_s00, mm_s00); // Clamp to 0 to 255 range.
        mm_s10 = _mm_packus_epi16(mm_s10, mm_s10); // Clamp to 0 to 255 range.
        *(uint32_t *)d0 = _mm_extract_epi32(mm_s00, 0);
        *(uint32_t *)d1 = _mm_extract_epi32(mm_s10, 0);
        d0 += 4;
        d1 += 4;
        t0 += 4;
        t1 += 4;
        t2 += 4;
        t3 += 4;
    }

    w += 4;
    if (w)
    {
        __m128i mm_s0, mm_s1, mm_s2, mm_s3;
        __m128i mm_s00, mm_s01, mm_s02, mm_s03;
        __m128i mm_s10, mm_s11, mm_s12, mm_s13;

        mm_s0 = _mm_loadu_si128((const __m128i *)t0);
        // Combine vertically
        mm_s00 = _mm_mullo_epi32(mm_s0, mm_w0);
        mm_s10 = _mm_mullo_epi32(mm_s0, mm_w4);
        mm_s1 = _mm_loadu_si128((const __m128i *)t1);
        mm_s01 = _mm_mullo_epi32(mm_s1, mm_w1);
        mm_s11 = _mm_mullo_epi32(mm_s1, mm_w5);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s01);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s11);
        mm_s2 = _mm_loadu_si128((const __m128i *)t2);
        mm_s02 = _mm_mullo_epi32(mm_s2, mm_w2);
        mm_s12 = _mm_mullo_epi32(mm_s2, mm_w6);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s02);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s12);
        mm_s3 = _mm_loadu_si128((const __m128i *)t3);
        mm_s03 = _mm_mullo_epi32(mm_s3, mm_w3);
        mm_s13 = _mm_mullo_epi32(mm_s3, mm_w7);
        mm_s00 = _mm_add_epi32(mm_s00, mm_s03);
        mm_s10 = _mm_add_epi32(mm_s10, mm_s13);
        // Round, shift, extract
        mm_s00 = _mm_add_epi32(mm_s00, mm_round);
        mm_s10 = _mm_add_epi32(mm_s10, mm_round);
        mm_s00 = _mm_srai_epi32(mm_s00, WEIGHT_SHIFT*2);
        mm_s10 = _mm_srai_epi32(mm_s10, WEIGHT_SHIFT*2);
        mm_s00 = _mm_packus_epi32(mm_s00, mm_s00); // Clamp to 0 to 65535 range.
        mm_s10 = _mm_packus_epi32(mm_s10, mm_s10); // Clamp to 0 to 65535 range.
        mm_s00 = _mm_packus_epi16(mm_s00, mm_s00); // Clamp to 0 to 255 range.
        mm_s10 = _mm_packus_epi16(mm_s10, mm_s10); // Clamp to 0 to 255 range.
        switch (w)
        {
        case 3:
            d0[2] = _mm_extract_epi8(mm_s00, 2);
            d1[2] = _mm_extract_epi8(mm_s10, 2);
        case 2:
            d0[1] = _mm_extract_epi8(mm_s00, 1);
            d1[1] = _mm_extract_epi8(mm_s10, 1);
        case 1:
            d0[0] = _mm_extract_epi8(mm_s00, 0);
            d1[0] = _mm_extract_epi8(mm_s10, 0);
        }
    }

    return 2;
}

static int
quad_mitchell1_top_sse(uint8_t       ** ipa_restrict dsts,
                       const uint8_t ** ipa_restrict srcs,
                       ipa_doubler    * ipa_restrict doubler)
{
    const uint8_t * ipa_restrict s3 = ((const uint8_t **)srcs)[3];
    uint32_t *t3 = doubler->tmp + (((doubler->tmp_y    )&3) * doubler->tmp_stride);

    if (doubler->tmp_y < (int)doubler->src_h)
        quad_mitchell1_to_tmp(t3, s3, doubler->src_w);

    return quad_mitchell_final(dsts, doubler);
}

static void
quad_mitchell3_to_tmp(uint32_t *t0, const uint8_t *s0, uint32_t src_w)
{
    int32_t w;
    int a_r, a_g, a_b, b_r, b_g, b_b, c_r, c_g, c_b;
    __m128i mm_w0123, mm_w4567, mm_w7654, mm_w3210, mm_shuffle;

    /* Leading pixels */
    a_r = s0[0];
    a_g = s0[1];
    a_b = s0[2];
    b_r = s0[3];
    b_g = s0[4];
    b_b = s0[5];
    c_r = s0[6];
    c_g = s0[7];
    c_b = s0[8];
    *t0++ = COMBINE2(a_r,a_r,a_r,b_r);
    *t0++ = COMBINE2(a_g,a_g,a_g,b_g);
    *t0++ = COMBINE2(a_b,a_b,a_b,b_b);
    *t0++ = COMBINE3(a_r,a_r,a_r,b_r);
    *t0++ = COMBINE3(a_g,a_g,a_g,b_g);
    *t0++ = COMBINE3(a_b,a_b,a_b,b_b);
    *t0++ = COMBINE0(a_r,a_r,b_r,c_r);
    *t0++ = COMBINE0(a_g,a_g,b_g,c_g);
    *t0++ = COMBINE0(a_b,a_b,b_b,c_b);
    *t0++ = COMBINE1(a_r,a_r,b_r,c_r);
    *t0++ = COMBINE1(a_g,a_g,b_g,c_g);
    *t0++ = COMBINE1(a_b,a_b,b_b,c_b);
    *t0++ = COMBINE2(a_r,a_r,b_r,c_r);
    *t0++ = COMBINE2(a_g,a_g,b_g,c_g);
    *t0++ = COMBINE2(a_b,a_b,b_b,c_b);
    *t0++ = COMBINE3(a_r,a_r,b_r,c_r);
    *t0++ = COMBINE3(a_g,a_g,b_g,c_g);
    *t0++ = COMBINE3(a_b,a_b,b_b,c_b);

    mm_w0123 = _mm_set_epi32(MW3,MW2,MW1,MW0);
    mm_w4567 = _mm_set_epi32(MW7,MW6,MW5,MW4);
    mm_w7654 = _mm_set_epi32(MW4,MW5,MW6,MW7);
    mm_w3210 = _mm_set_epi32(MW0,MW1,MW2,MW3);
    mm_shuffle = _mm_set_epi8(ZZ,ZZ,ZZ, 9,ZZ,ZZ,ZZ, 6,ZZ,ZZ,ZZ, 3,ZZ,ZZ,ZZ, 0);

    for (w = src_w-3-2; w >= 0; w -= 2)
    {
        __m128i mm_s0, mm_s00, mm_s01, mm_s02, mm_s03;

        // Load raw pixels into mm0 and mm1 (source pixels n to n+3)
        mm_s0 = _mm_loadu_si128((const __m128i *)s0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        // 0: Explode to 32 bits
        mm_s03 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        t0[0] = _mm_extract_epi32(mm_s00, 0);
        t0[3] = _mm_extract_epi32(mm_s00, 1);
        t0[6] = _mm_extract_epi32(mm_s00, 2);
        t0[9] = _mm_extract_epi32(mm_s00, 3);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 1: Explode to 32 bits
        mm_s03 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        t0[1] = _mm_extract_epi32(mm_s00, 0);
        t0[4] = _mm_extract_epi32(mm_s00, 1);
        t0[7] = _mm_extract_epi32(mm_s00, 2);
        t0[10] = _mm_extract_epi32(mm_s00, 3);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 2: Explode to 32 bits
        mm_s03 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        t0[2] = _mm_extract_epi32(mm_s00, 0);
        t0[5] = _mm_extract_epi32(mm_s00, 1);
        t0[8] = _mm_extract_epi32(mm_s00, 2);
        t0[11] = _mm_extract_epi32(mm_s00, 3);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 3: Explode to 32 bits
        mm_s03 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        t0[12] = _mm_extract_epi32(mm_s00, 0);
        t0[15] = _mm_extract_epi32(mm_s00, 1);
        t0[18] = _mm_extract_epi32(mm_s00, 2);
        t0[21] = _mm_extract_epi32(mm_s00, 3);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 4: Explode to 32 bits
        mm_s03 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        t0[13] = _mm_extract_epi32(mm_s00, 0);
        t0[16] = _mm_extract_epi32(mm_s00, 1);
        t0[19] = _mm_extract_epi32(mm_s00, 2);
        t0[22] = _mm_extract_epi32(mm_s00, 3);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 5: Explode to 32 bits
        mm_s03 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        t0[14] = _mm_extract_epi32(mm_s00, 0);
        t0[17] = _mm_extract_epi32(mm_s00, 1);
        t0[20] = _mm_extract_epi32(mm_s00, 2);
        t0[23] = _mm_extract_epi32(mm_s00, 3);
        s0 += 6;
        t0 += 24;
    }

    w += 2;
    if (w)
    {
        __m128i mm_s0, mm_s00, mm_s01, mm_s02, mm_s03;
        uint8_t local[16];

        memcpy(local, s0, 12);
        // Load raw pixels into mm0 and mm1 (source pixels n to n+3)
        mm_s0 = _mm_loadu_si128((const __m128i *)local); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        // 0: Explode to 32 bits
        mm_s03 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        t0[0] = _mm_extract_epi32(mm_s00, 0);
        t0[3] = _mm_extract_epi32(mm_s00, 1);
        t0[6] = _mm_extract_epi32(mm_s00, 2);
        t0[9] = _mm_extract_epi32(mm_s00, 3);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 1: Explode to 32 bits
        mm_s03 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        t0[1] = _mm_extract_epi32(mm_s00, 0);
        t0[4] = _mm_extract_epi32(mm_s00, 1);
        t0[7] = _mm_extract_epi32(mm_s00, 2);
        t0[10] = _mm_extract_epi32(mm_s00, 3);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 2: Explode to 32 bits
        mm_s03 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        t0[2] = _mm_extract_epi32(mm_s00, 0);
        t0[5] = _mm_extract_epi32(mm_s00, 1);
        t0[8] = _mm_extract_epi32(mm_s00, 2);
        t0[11] = _mm_extract_epi32(mm_s00, 3);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        s0 += 3;
        t0 += 12;
    }

    a_r = s0[0]; a_g = s0[1]; a_b = s0[2];
    b_r = s0[3]; b_g = s0[4]; b_b = s0[5];
    c_r = s0[6]; c_g = s0[7]; c_b = s0[8];

    /* Trailing pixels */
    *t0++ = COMBINE0(a_r,b_r,c_r,c_r);
    *t0++ = COMBINE0(a_g,b_g,c_g,c_g);
    *t0++ = COMBINE0(a_b,b_b,c_b,c_b);
    *t0++ = COMBINE1(a_r,b_r,c_r,c_r);
    *t0++ = COMBINE1(a_g,b_g,c_g,c_g);
    *t0++ = COMBINE1(a_b,b_b,c_b,c_b);
    *t0++ = COMBINE2(a_r,b_r,c_r,c_r);
    *t0++ = COMBINE2(a_g,b_g,c_g,c_g);
    *t0++ = COMBINE2(a_b,b_b,c_b,c_b);
    *t0++ = COMBINE3(a_r,b_r,c_r,c_r);
    *t0++ = COMBINE3(a_g,b_g,c_g,c_g);
    *t0++ = COMBINE3(a_b,b_b,c_b,c_b);
    *t0++ = COMBINE0(b_r,c_r,c_r,c_r);
    *t0++ = COMBINE0(b_g,c_g,c_g,c_g);
    *t0++ = COMBINE0(b_b,c_b,c_b,c_b);
    *t0++ = COMBINE1(b_r,c_r,c_r,c_r);
    *t0++ = COMBINE1(b_g,c_g,c_g,c_g);
    *t0++ = COMBINE1(b_b,c_b,c_b,c_b);
}

static int
quad_mitchell3_sse(uint8_t       ** ipa_restrict dsts,
                   const uint8_t ** ipa_restrict srcs,
                   ipa_doubler    * ipa_restrict doubler)
{
    uint8_t * ipa_restrict d0 = ((uint8_t **)dsts)[0];
    uint8_t * ipa_restrict d1 = ((uint8_t **)dsts)[1];
    uint8_t * ipa_restrict d2 = ((uint8_t **)dsts)[2];
    uint8_t * ipa_restrict d3 = ((uint8_t **)dsts)[3];
    const uint8_t * ipa_restrict s3 = ((const uint8_t **)srcs)[3];
    uint32_t *t3 = doubler->tmp + (((doubler->tmp_y    )&3) * doubler->tmp_stride);
    uint32_t *t2 = doubler->tmp + (((doubler->tmp_y + 3)&3) * doubler->tmp_stride);
    uint32_t *t1 = doubler->tmp + (((doubler->tmp_y + 2)&3) * doubler->tmp_stride);
    uint32_t *t0 = doubler->tmp + (((doubler->tmp_y + 1)&3) * doubler->tmp_stride);

    quad_mitchell3_to_tmp(t3, s3, doubler->src_w);
    doubler->tmp_y++;

    return quad_mitchell(d0, d1, d2, d3, t0, t1, t2, t3, doubler->src_w*3);
}

static int
quad_mitchell3_top_sse(uint8_t       ** ipa_restrict dsts,
                       const uint8_t ** ipa_restrict srcs,
                       ipa_doubler    * ipa_restrict doubler)
{
    const uint8_t * ipa_restrict s3 = ((const uint8_t **)srcs)[3];
    uint32_t *t3 = doubler->tmp + (((doubler->tmp_y    )&3) * doubler->tmp_stride);

    if (doubler->tmp_y < (int)doubler->src_h)
        quad_mitchell3_to_tmp(t3, s3, doubler->src_w);

    return quad_mitchell_final(dsts, doubler);
}

static void
quad_mitchell4_to_tmp(uint32_t *t0, const uint8_t *s0, uint32_t src_w)
{
    int32_t w;
    int a_r, a_g, a_b, a_k, b_r, b_g, b_b, b_k, c_r, c_g, c_b, c_k;
    __m128i mm_w0123, mm_w3210, mm_w4567, mm_w7654, mm_shuffle;

    /* Leading pixels */
    a_r = s0[0];
    a_g = s0[1];
    a_b = s0[2];
    a_k = s0[3];
    b_r = s0[4];
    b_g = s0[5];
    b_b = s0[6];
    b_k = s0[7];
    c_r = s0[8];
    c_g = s0[9];
    c_b = s0[10];
    c_k = s0[11];
    *t0++ = COMBINE2(a_r,a_r,a_r,b_r);
    *t0++ = COMBINE2(a_g,a_g,a_g,b_g);
    *t0++ = COMBINE2(a_b,a_b,a_b,b_b);
    *t0++ = COMBINE2(a_k,a_k,a_k,b_k);
    *t0++ = COMBINE3(a_r,a_r,a_r,b_r);
    *t0++ = COMBINE3(a_g,a_g,a_g,b_g);
    *t0++ = COMBINE3(a_b,a_b,a_b,b_b);
    *t0++ = COMBINE3(a_k,a_k,a_k,b_k);
    *t0++ = COMBINE0(a_r,a_r,b_r,c_r);
    *t0++ = COMBINE0(a_g,a_g,b_g,c_g);
    *t0++ = COMBINE0(a_b,a_b,b_b,c_b);
    *t0++ = COMBINE0(a_k,a_k,b_k,c_k);
    *t0++ = COMBINE1(a_r,a_r,b_r,c_r);
    *t0++ = COMBINE1(a_g,a_g,b_g,c_g);
    *t0++ = COMBINE1(a_b,a_b,b_b,c_b);
    *t0++ = COMBINE1(a_k,a_k,b_k,c_k);
    *t0++ = COMBINE2(a_r,a_r,b_r,c_r);
    *t0++ = COMBINE2(a_g,a_g,b_g,c_g);
    *t0++ = COMBINE2(a_b,a_b,b_b,c_b);
    *t0++ = COMBINE2(a_k,a_k,b_k,c_k);
    *t0++ = COMBINE3(a_r,a_r,b_r,c_r);
    *t0++ = COMBINE3(a_g,a_g,b_g,c_g);
    *t0++ = COMBINE3(a_b,a_b,b_b,c_b);
    *t0++ = COMBINE3(a_k,a_k,b_k,c_k);

    mm_w0123 = _mm_set_epi32(MW3,MW2,MW1,MW0);
    mm_w4567 = _mm_set_epi32(MW7,MW6,MW5,MW4);
    mm_w7654 = _mm_set_epi32(MW4,MW5,MW6,MW7);
    mm_w3210 = _mm_set_epi32(MW0,MW1,MW2,MW3);
    mm_shuffle = _mm_set_epi8(ZZ,ZZ,ZZ,12,ZZ,ZZ,ZZ, 8,ZZ,ZZ,ZZ, 4,ZZ,ZZ,ZZ, 0);

    for (w = src_w-3-1; w >= 0; w--)
    {
        __m128i mm_s0, mm_s00, mm_s01, mm_s02, mm_s03;

        // Load raw pixels into mm0 and mm1 (source pixels n to n+3)
        mm_s0 = _mm_loadu_si128((const __m128i *)s0); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa
        // 0: Explode to 32 bits
        mm_s03 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        t0[0] = _mm_extract_epi32(mm_s00, 0);
        t0[4] = _mm_extract_epi32(mm_s00, 1);
        t0[8] = _mm_extract_epi32(mm_s00, 2);
        t0[12]= _mm_extract_epi32(mm_s00, 3);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 1: Explode to 32 bits
        mm_s03 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        t0[1] = _mm_extract_epi32(mm_s00, 0);
        t0[5] = _mm_extract_epi32(mm_s00, 1);
        t0[9] = _mm_extract_epi32(mm_s00, 2);
        t0[13]= _mm_extract_epi32(mm_s00, 3);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 2: Explode to 32 bits
        mm_s03 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        t0[2] = _mm_extract_epi32(mm_s00, 0);
        t0[6] = _mm_extract_epi32(mm_s00, 1);
        t0[10]= _mm_extract_epi32(mm_s00, 2);
        t0[14]= _mm_extract_epi32(mm_s00, 3);
        mm_s0 = _mm_srli_si128(mm_s0, 1);
        // 3: Explode to 32 bits
        mm_s03 = _mm_shuffle_epi8(mm_s0, mm_shuffle);
        // Combine horizontally
        mm_s00 = _mm_mullo_epi32(mm_s03, mm_w0123);
        mm_s01 = _mm_mullo_epi32(mm_s03, mm_w4567);
        mm_s02 = _mm_mullo_epi32(mm_s03, mm_w7654);
        mm_s03 = _mm_mullo_epi32(mm_s03, mm_w3210);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s01);
        mm_s02 = _mm_hadd_epi32(mm_s02, mm_s03);
        mm_s00 = _mm_hadd_epi32(mm_s00, mm_s02);
        // Store out horizontally combined values.
        t0[3] = _mm_extract_epi32(mm_s00, 0);
        t0[7] = _mm_extract_epi32(mm_s00, 1);
        t0[11]= _mm_extract_epi32(mm_s00, 2);
        t0[15]= _mm_extract_epi32(mm_s00, 3);
        s0 += 4;
        t0 += 16;
    }

    a_r = s0[0]; a_g = s0[1]; a_b = s0[2]; a_k = s0[3];
    b_r = s0[4]; b_g = s0[5]; b_b = s0[6]; b_k = s0[7];
    c_r = s0[8]; c_g = s0[9]; c_b = s0[10];c_k = s0[11];

    /* Trailing pixels */
    *t0++ = COMBINE0(a_r,b_r,c_r,c_r);
    *t0++ = COMBINE0(a_g,b_g,c_g,c_g);
    *t0++ = COMBINE0(a_b,b_b,c_b,c_b);
    *t0++ = COMBINE0(a_k,b_k,c_k,c_k);
    *t0++ = COMBINE1(a_r,b_r,c_r,c_r);
    *t0++ = COMBINE1(a_g,b_g,c_g,c_g);
    *t0++ = COMBINE1(a_b,b_b,c_b,c_b);
    *t0++ = COMBINE1(b_k,b_k,c_k,c_k);
    *t0++ = COMBINE2(a_r,b_r,c_r,c_r);
    *t0++ = COMBINE2(a_g,b_g,c_g,c_g);
    *t0++ = COMBINE2(a_b,b_b,c_b,c_b);
    *t0++ = COMBINE2(a_k,b_k,c_k,c_k);
    *t0++ = COMBINE3(a_r,b_r,c_r,c_r);
    *t0++ = COMBINE3(a_g,b_g,c_g,c_g);
    *t0++ = COMBINE3(a_b,b_b,c_b,c_b);
    *t0++ = COMBINE3(b_k,b_k,c_k,c_k);
    *t0++ = COMBINE0(b_r,c_r,c_r,c_r);
    *t0++ = COMBINE0(b_g,c_g,c_g,c_g);
    *t0++ = COMBINE0(b_b,c_b,c_b,c_b);
    *t0++ = COMBINE0(b_k,c_k,c_k,c_k);
    *t0++ = COMBINE1(b_r,c_r,c_r,c_r);
    *t0++ = COMBINE1(b_g,c_g,c_g,c_g);
    *t0++ = COMBINE1(b_b,c_b,c_b,c_b);
    *t0++ = COMBINE1(b_k,c_k,c_k,c_k);
}

static int
quad_mitchell4_sse(uint8_t       ** ipa_restrict dsts,
                   const uint8_t ** ipa_restrict srcs,
                   ipa_doubler    * ipa_restrict doubler)
{
    uint8_t * ipa_restrict d0 = ((uint8_t **)dsts)[0];
    uint8_t * ipa_restrict d1 = ((uint8_t **)dsts)[1];
    uint8_t * ipa_restrict d2 = ((uint8_t **)dsts)[2];
    uint8_t * ipa_restrict d3 = ((uint8_t **)dsts)[3];
    const uint8_t * ipa_restrict s3 = ((const uint8_t **)srcs)[3];
    uint32_t *t3 = doubler->tmp + (((doubler->tmp_y    )&3) * doubler->tmp_stride);
    uint32_t *t2 = doubler->tmp + (((doubler->tmp_y + 3)&3) * doubler->tmp_stride);
    uint32_t *t1 = doubler->tmp + (((doubler->tmp_y + 2)&3) * doubler->tmp_stride);
    uint32_t *t0 = doubler->tmp + (((doubler->tmp_y + 1)&3) * doubler->tmp_stride);

    quad_mitchell4_to_tmp(t3, s3, doubler->src_w);
    doubler->tmp_y++;

    return quad_mitchell(d0, d1, d2, d3, t0, t1, t2, t3, doubler->src_w*4);
}

static int
quad_mitchell4_top_sse(uint8_t       ** ipa_restrict dsts,
                       const uint8_t ** ipa_restrict srcs,
                       ipa_doubler    * ipa_restrict doubler)
{
    const uint8_t * ipa_restrict s3 = ((const uint8_t **)srcs)[3];
    uint32_t *t3 = doubler->tmp + (((doubler->tmp_y    )&3) * doubler->tmp_stride);

    if (doubler->tmp_y < (int)doubler->src_h)
        quad_mitchell4_to_tmp(t3, s3, doubler->src_w);

    return quad_mitchell_final(dsts, doubler);
}

static int
octo_near1_sse(uint8_t       ** ipa_restrict dsts,
               const uint8_t ** ipa_restrict srcs,
               ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    uint8_t * ipa_restrict d4 = dsts[4];
    uint8_t * ipa_restrict d5 = dsts[5];
    uint8_t * ipa_restrict d6 = dsts[6];
    uint8_t * ipa_restrict d7 = dsts[7];
    const uint8_t * ipa_restrict s0 = srcs[0];
    __m128i mm_shuf0;

    mm_shuf0 = _mm_set_epi8(1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0);

    for (w = doubler->src_w-16; w >= 0; w -= 16)
    {
        __m128i mm0, mm1;

        mm0 = _mm_loadu_si128((const __m128i *)s0);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)d0,mm1);
        _mm_storeu_si128((__m128i *)d1,mm1);
        _mm_storeu_si128((__m128i *)d2,mm1);
        _mm_storeu_si128((__m128i *)d3,mm1);
        mm0 = _mm_srli_si128(mm0,2);
        _mm_storeu_si128((__m128i *)d4,mm1);
        _mm_storeu_si128((__m128i *)d5,mm1);
        _mm_storeu_si128((__m128i *)d6,mm1);
        _mm_storeu_si128((__m128i *)d7,mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)(d0+16),mm1);
        _mm_storeu_si128((__m128i *)(d1+16),mm1);
        _mm_storeu_si128((__m128i *)(d2+16),mm1);
        _mm_storeu_si128((__m128i *)(d3+16),mm1);
        mm0 = _mm_srli_si128(mm0,2);
        _mm_storeu_si128((__m128i *)(d4+16),mm1);
        _mm_storeu_si128((__m128i *)(d5+16),mm1);
        _mm_storeu_si128((__m128i *)(d6+16),mm1);
        _mm_storeu_si128((__m128i *)(d7+16),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)(d0+32),mm1);
        _mm_storeu_si128((__m128i *)(d1+32),mm1);
        _mm_storeu_si128((__m128i *)(d2+32),mm1);
        _mm_storeu_si128((__m128i *)(d3+32),mm1);
        mm0 = _mm_srli_si128(mm0,2);
        _mm_storeu_si128((__m128i *)(d4+32),mm1);
        _mm_storeu_si128((__m128i *)(d5+32),mm1);
        _mm_storeu_si128((__m128i *)(d6+32),mm1);
        _mm_storeu_si128((__m128i *)(d7+32),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)(d0+48),mm1);
        _mm_storeu_si128((__m128i *)(d1+48),mm1);
        _mm_storeu_si128((__m128i *)(d2+48),mm1);
        _mm_storeu_si128((__m128i *)(d3+48),mm1);
        mm0 = _mm_srli_si128(mm0,2);
        _mm_storeu_si128((__m128i *)(d4+48),mm1);
        _mm_storeu_si128((__m128i *)(d5+48),mm1);
        _mm_storeu_si128((__m128i *)(d6+48),mm1);
        _mm_storeu_si128((__m128i *)(d7+48),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)(d0+64),mm1);
        _mm_storeu_si128((__m128i *)(d1+64),mm1);
        _mm_storeu_si128((__m128i *)(d2+64),mm1);
        _mm_storeu_si128((__m128i *)(d3+64),mm1);
        mm0 = _mm_srli_si128(mm0,2);
        _mm_storeu_si128((__m128i *)(d4+64),mm1);
        _mm_storeu_si128((__m128i *)(d5+64),mm1);
        _mm_storeu_si128((__m128i *)(d6+64),mm1);
        _mm_storeu_si128((__m128i *)(d7+64),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)(d0+80),mm1);
        _mm_storeu_si128((__m128i *)(d1+80),mm1);
        _mm_storeu_si128((__m128i *)(d2+80),mm1);
        _mm_storeu_si128((__m128i *)(d3+80),mm1);
        mm0 = _mm_srli_si128(mm0,2);
        _mm_storeu_si128((__m128i *)(d4+80),mm1);
        _mm_storeu_si128((__m128i *)(d5+80),mm1);
        _mm_storeu_si128((__m128i *)(d6+80),mm1);
        _mm_storeu_si128((__m128i *)(d7+80),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)(d0+96),mm1);
        _mm_storeu_si128((__m128i *)(d1+96),mm1);
        _mm_storeu_si128((__m128i *)(d2+96),mm1);
        _mm_storeu_si128((__m128i *)(d3+96),mm1);
        mm0 = _mm_srli_si128(mm0,2);
        _mm_storeu_si128((__m128i *)(d4+96),mm1);
        _mm_storeu_si128((__m128i *)(d5+96),mm1);
        _mm_storeu_si128((__m128i *)(d6+96),mm1);
        _mm_storeu_si128((__m128i *)(d7+96),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)(d0+112),mm1);
        _mm_storeu_si128((__m128i *)(d1+112),mm1);
        _mm_storeu_si128((__m128i *)(d2+112),mm1);
        _mm_storeu_si128((__m128i *)(d3+112),mm1);
        _mm_storeu_si128((__m128i *)(d4+112),mm1);
        _mm_storeu_si128((__m128i *)(d5+112),mm1);
        _mm_storeu_si128((__m128i *)(d6+112),mm1);
        _mm_storeu_si128((__m128i *)(d7+112),mm1);
        s0 += 16;
        d0 += 128;
        d1 += 128;
        d2 += 128;
        d3 += 128;
        d4 += 128;
        d5 += 128;
        d6 += 128;
        d7 += 128;
    }

    w += 16;
    while (w) /* So we can break out */
    {
        __m128i mm0, mm1;
        uint8_t local[16];

        memcpy(local, s0, w);
        mm0 = _mm_loadu_si128((const __m128i *)local);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)d0,mm1);
        _mm_storeu_si128((__m128i *)d1,mm1);
        _mm_storeu_si128((__m128i *)d2,mm1);
        _mm_storeu_si128((__m128i *)d3,mm1);
        mm0 = _mm_srli_si128(mm0,2);
        _mm_storeu_si128((__m128i *)d4,mm1);
        _mm_storeu_si128((__m128i *)d5,mm1);
        _mm_storeu_si128((__m128i *)d6,mm1);
        _mm_storeu_si128((__m128i *)d7,mm1);
        if (w == 1) break;
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)(d0+16),mm1);
        _mm_storeu_si128((__m128i *)(d1+16),mm1);
        _mm_storeu_si128((__m128i *)(d2+16),mm1);
        _mm_storeu_si128((__m128i *)(d3+16),mm1);
        mm0 = _mm_srli_si128(mm0,2);
        _mm_storeu_si128((__m128i *)(d4+16),mm1);
        _mm_storeu_si128((__m128i *)(d5+16),mm1);
        _mm_storeu_si128((__m128i *)(d6+16),mm1);
        _mm_storeu_si128((__m128i *)(d7+16),mm1);
        if (w == 2) break;
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)(d0+32),mm1);
        _mm_storeu_si128((__m128i *)(d1+32),mm1);
        _mm_storeu_si128((__m128i *)(d2+32),mm1);
        _mm_storeu_si128((__m128i *)(d3+32),mm1);
        mm0 = _mm_srli_si128(mm0,2);
        _mm_storeu_si128((__m128i *)(d4+32),mm1);
        _mm_storeu_si128((__m128i *)(d5+32),mm1);
        _mm_storeu_si128((__m128i *)(d6+32),mm1);
        _mm_storeu_si128((__m128i *)(d7+32),mm1);
        if (w == 3) break;
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)(d0+48),mm1);
        _mm_storeu_si128((__m128i *)(d1+48),mm1);
        _mm_storeu_si128((__m128i *)(d2+48),mm1);
        _mm_storeu_si128((__m128i *)(d3+48),mm1);
        mm0 = _mm_srli_si128(mm0,2);
        _mm_storeu_si128((__m128i *)(d4+48),mm1);
        _mm_storeu_si128((__m128i *)(d5+48),mm1);
        _mm_storeu_si128((__m128i *)(d6+48),mm1);
        _mm_storeu_si128((__m128i *)(d7+48),mm1);
        if (w == 4) break;
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)(d0+64),mm1);
        _mm_storeu_si128((__m128i *)(d1+64),mm1);
        _mm_storeu_si128((__m128i *)(d2+64),mm1);
        _mm_storeu_si128((__m128i *)(d3+64),mm1);
        mm0 = _mm_srli_si128(mm0,2);
        _mm_storeu_si128((__m128i *)(d4+64),mm1);
        _mm_storeu_si128((__m128i *)(d5+64),mm1);
        _mm_storeu_si128((__m128i *)(d6+64),mm1);
        _mm_storeu_si128((__m128i *)(d7+64),mm1);
        if (w == 5) break;
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)(d0+80),mm1);
        _mm_storeu_si128((__m128i *)(d1+80),mm1);
        _mm_storeu_si128((__m128i *)(d2+80),mm1);
        _mm_storeu_si128((__m128i *)(d3+80),mm1);
        mm0 = _mm_srli_si128(mm0,2);
        _mm_storeu_si128((__m128i *)(d4+80),mm1);
        _mm_storeu_si128((__m128i *)(d5+80),mm1);
        _mm_storeu_si128((__m128i *)(d6+80),mm1);
        _mm_storeu_si128((__m128i *)(d7+80),mm1);
        if (w == 6) break;
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)(d0+96),mm1);
        _mm_storeu_si128((__m128i *)(d1+96),mm1);
        _mm_storeu_si128((__m128i *)(d2+96),mm1);
        _mm_storeu_si128((__m128i *)(d3+96),mm1);
        mm0 = _mm_srli_si128(mm0,2);
        _mm_storeu_si128((__m128i *)(d4+96),mm1);
        _mm_storeu_si128((__m128i *)(d5+96),mm1);
        _mm_storeu_si128((__m128i *)(d6+96),mm1);
        _mm_storeu_si128((__m128i *)(d7+96),mm1);
        break;
    }

    return 8;
}

static int
octo_near3_sse(uint8_t       ** ipa_restrict dsts,
               const uint8_t ** ipa_restrict srcs,
               ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint8_t * ipa_restrict d0 = dsts[0];
    uint8_t * ipa_restrict d1 = dsts[1];
    uint8_t * ipa_restrict d2 = dsts[2];
    uint8_t * ipa_restrict d3 = dsts[3];
    uint8_t * ipa_restrict d4 = dsts[4];
    uint8_t * ipa_restrict d5 = dsts[5];
    uint8_t * ipa_restrict d6 = dsts[6];
    uint8_t * ipa_restrict d7 = dsts[7];
    const uint8_t * ipa_restrict s0 = srcs[0];
    __m128i mm_shuf0, mm_shuf1, mm_shuf2;

    mm_shuf0 = _mm_set_epi8(0,2,1,0,2,1,0,2,1,0,2,1,0,2,1,0);
    mm_shuf1 = _mm_set_epi8(4,3,5,4,3,5,4,3,2,1,0,2,1,0,2,1);
    mm_shuf2 = _mm_set_epi8(5,4,3,5,4,3,5,4,3,5,4,3,5,4,3,5);

    for (w = doubler->src_w-5; w >= 0; w -= 5)
    {
        __m128i mm0, mm1;

        mm0 = _mm_loadu_si128((const __m128i *)s0);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)d0,mm1);
        _mm_storeu_si128((__m128i *)d1,mm1);
        _mm_storeu_si128((__m128i *)d2,mm1);
        _mm_storeu_si128((__m128i *)d3,mm1);
        _mm_storeu_si128((__m128i *)d4,mm1);
        _mm_storeu_si128((__m128i *)d5,mm1);
        _mm_storeu_si128((__m128i *)d6,mm1);
        _mm_storeu_si128((__m128i *)d7,mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf1);
        _mm_storeu_si128((__m128i *)(d0+16),mm1);
        _mm_storeu_si128((__m128i *)(d1+16),mm1);
        _mm_storeu_si128((__m128i *)(d2+16),mm1);
        _mm_storeu_si128((__m128i *)(d3+16),mm1);
        _mm_storeu_si128((__m128i *)(d4+16),mm1);
        _mm_storeu_si128((__m128i *)(d5+16),mm1);
        _mm_storeu_si128((__m128i *)(d6+16),mm1);
        _mm_storeu_si128((__m128i *)(d7+16),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf2);
        _mm_storeu_si128((__m128i *)(d0+32),mm1);
        _mm_storeu_si128((__m128i *)(d1+32),mm1);
        _mm_storeu_si128((__m128i *)(d2+32),mm1);
        _mm_storeu_si128((__m128i *)(d3+32),mm1);
        mm0 = _mm_srli_si128(mm0,6);
        _mm_storeu_si128((__m128i *)(d4+32),mm1);
        _mm_storeu_si128((__m128i *)(d5+32),mm1);
        _mm_storeu_si128((__m128i *)(d6+32),mm1);
        _mm_storeu_si128((__m128i *)(d7+32),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)(d0+48),mm1);
        _mm_storeu_si128((__m128i *)(d1+48),mm1);
        _mm_storeu_si128((__m128i *)(d2+48),mm1);
        _mm_storeu_si128((__m128i *)(d3+48),mm1);
        _mm_storeu_si128((__m128i *)(d4+48),mm1);
        _mm_storeu_si128((__m128i *)(d5+48),mm1);
        _mm_storeu_si128((__m128i *)(d6+48),mm1);
        _mm_storeu_si128((__m128i *)(d7+48),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf1);
        _mm_storeu_si128((__m128i *)(d0+64),mm1);
        _mm_storeu_si128((__m128i *)(d1+64),mm1);
        _mm_storeu_si128((__m128i *)(d2+64),mm1);
        _mm_storeu_si128((__m128i *)(d3+64),mm1);
        _mm_storeu_si128((__m128i *)(d4+64),mm1);
        _mm_storeu_si128((__m128i *)(d5+64),mm1);
        _mm_storeu_si128((__m128i *)(d6+64),mm1);
        _mm_storeu_si128((__m128i *)(d7+64),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf2);
        _mm_storeu_si128((__m128i *)(d0+80),mm1);
        _mm_storeu_si128((__m128i *)(d1+80),mm1);
        _mm_storeu_si128((__m128i *)(d2+80),mm1);
        _mm_storeu_si128((__m128i *)(d3+80),mm1);
        mm0 = _mm_srli_si128(mm0,6);
        _mm_storeu_si128((__m128i *)(d4+80),mm1);
        _mm_storeu_si128((__m128i *)(d5+80),mm1);
        _mm_storeu_si128((__m128i *)(d6+80),mm1);
        _mm_storeu_si128((__m128i *)(d7+80),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)(d0+96),mm1);
        _mm_storeu_si128((__m128i *)(d1+96),mm1);
        _mm_storeu_si128((__m128i *)(d2+96),mm1);
        _mm_storeu_si128((__m128i *)(d3+96),mm1);
        _mm_storeu_si128((__m128i *)(d4+96),mm1);
        _mm_storeu_si128((__m128i *)(d5+96),mm1);
        _mm_storeu_si128((__m128i *)(d6+96),mm1);
        _mm_storeu_si128((__m128i *)(d7+96),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf1);
        _mm_storeu_si128((__m128i *)(d0+112),mm1);
        _mm_storeu_si128((__m128i *)(d1+112),mm1);
        _mm_storeu_si128((__m128i *)(d2+112),mm1);
        _mm_storeu_si128((__m128i *)(d3+112),mm1);
        _mm_storeu_si128((__m128i *)(d4+112),mm1);
        _mm_storeu_si128((__m128i *)(d5+112),mm1);
        _mm_storeu_si128((__m128i *)(d6+112),mm1);
        _mm_storeu_si128((__m128i *)(d7+112),mm1);
        s0 += 15;
        d0 += 120;
        d1 += 120;
        d2 += 120;
        d3 += 120;
        d4 += 120;
        d5 += 120;
        d6 += 120;
        d7 += 120;
    }

    w += 5;
    while (w) /* So we can break out */
    {
        __m128i mm0, mm1;
        uint8_t local[16];

        memcpy(local, s0, w);
        mm0 = _mm_loadu_si128((const __m128i *)local);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)d0,mm1);
        _mm_storeu_si128((__m128i *)d1,mm1);
        _mm_storeu_si128((__m128i *)d2,mm1);
        _mm_storeu_si128((__m128i *)d3,mm1);
        _mm_storeu_si128((__m128i *)d4,mm1);
        _mm_storeu_si128((__m128i *)d5,mm1);
        _mm_storeu_si128((__m128i *)d6,mm1);
        _mm_storeu_si128((__m128i *)d7,mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf1);
        if (w == 1)
        {
            ((uint32_t *)d0)[4] = ((uint32_t *)d1)[4] = ((uint32_t *)d2)[4] = ((uint32_t *)d3)[4] = _mm_extract_epi32(mm1, 0);
            ((uint32_t *)d0)[5] = ((uint32_t *)d1)[5] = ((uint32_t *)d2)[5] = ((uint32_t *)d3)[5] = _mm_extract_epi32(mm1, 1);
            break;
        }
        _mm_storeu_si128((__m128i *)(d0+16),mm1);
        _mm_storeu_si128((__m128i *)(d1+16),mm1);
        _mm_storeu_si128((__m128i *)(d2+16),mm1);
        _mm_storeu_si128((__m128i *)(d3+16),mm1);
        _mm_storeu_si128((__m128i *)(d4+16),mm1);
        _mm_storeu_si128((__m128i *)(d5+16),mm1);
        _mm_storeu_si128((__m128i *)(d6+16),mm1);
        _mm_storeu_si128((__m128i *)(d7+16),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf2);
        _mm_storeu_si128((__m128i *)(d0+32),mm1);
        _mm_storeu_si128((__m128i *)(d1+32),mm1);
        _mm_storeu_si128((__m128i *)(d2+32),mm1);
        _mm_storeu_si128((__m128i *)(d3+32),mm1);
        mm0 = _mm_srli_si128(mm0,6);
        _mm_storeu_si128((__m128i *)(d4+32),mm1);
        _mm_storeu_si128((__m128i *)(d5+32),mm1);
        _mm_storeu_si128((__m128i *)(d6+32),mm1);
        _mm_storeu_si128((__m128i *)(d7+32),mm1);
        if (w == 2) break;
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf0);
        _mm_storeu_si128((__m128i *)(d0+48),mm1);
        _mm_storeu_si128((__m128i *)(d1+48),mm1);
        _mm_storeu_si128((__m128i *)(d2+48),mm1);
        _mm_storeu_si128((__m128i *)(d3+48),mm1);
        _mm_storeu_si128((__m128i *)(d4+48),mm1);
        _mm_storeu_si128((__m128i *)(d5+48),mm1);
        _mm_storeu_si128((__m128i *)(d6+48),mm1);
        _mm_storeu_si128((__m128i *)(d7+48),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf1);
        if (w == 3)
        {
            ((uint32_t *)d0)[16] = ((uint32_t *)d1)[16] = ((uint32_t *)d2)[16] = ((uint32_t *)d3)[16] = _mm_extract_epi32(mm1, 0);
            ((uint32_t *)d0)[17] = ((uint32_t *)d1)[17] = ((uint32_t *)d2)[17] = ((uint32_t *)d3)[17] = _mm_extract_epi32(mm1, 1);
            break;
        }
        _mm_storeu_si128((__m128i *)(d0+64),mm1);
        _mm_storeu_si128((__m128i *)(d1+64),mm1);
        _mm_storeu_si128((__m128i *)(d2+64),mm1);
        _mm_storeu_si128((__m128i *)(d3+64),mm1);
        _mm_storeu_si128((__m128i *)(d4+64),mm1);
        _mm_storeu_si128((__m128i *)(d5+64),mm1);
        _mm_storeu_si128((__m128i *)(d6+64),mm1);
        _mm_storeu_si128((__m128i *)(d7+64),mm1);
        mm1 = _mm_shuffle_epi8(mm0, mm_shuf2);
        _mm_storeu_si128((__m128i *)(d0+80),mm1);
        _mm_storeu_si128((__m128i *)(d1+80),mm1);
        _mm_storeu_si128((__m128i *)(d2+80),mm1);
        _mm_storeu_si128((__m128i *)(d3+80),mm1);
        mm0 = _mm_srli_si128(mm0,6);
        _mm_storeu_si128((__m128i *)(d4+80),mm1);
        _mm_storeu_si128((__m128i *)(d5+80),mm1);
        _mm_storeu_si128((__m128i *)(d6+80),mm1);
        _mm_storeu_si128((__m128i *)(d7+80),mm1);
        break;
    }

    return 8;
}

static int
octo_near4_sse(uint8_t       ** ipa_restrict dsts,
               const uint8_t ** ipa_restrict srcs,
               ipa_doubler    * ipa_restrict doubler)
{
    int32_t w;
    uint32_t * ipa_restrict d0 = (uint32_t *)(dsts[0]);
    uint32_t * ipa_restrict d1 = (uint32_t *)(dsts[1]);
    uint32_t * ipa_restrict d2 = (uint32_t *)(dsts[2]);
    uint32_t * ipa_restrict d3 = (uint32_t *)(dsts[3]);
    uint32_t * ipa_restrict d4 = (uint32_t *)(dsts[4]);
    uint32_t * ipa_restrict d5 = (uint32_t *)(dsts[5]);
    uint32_t * ipa_restrict d6 = (uint32_t *)(dsts[6]);
    uint32_t * ipa_restrict d7 = (uint32_t *)(dsts[7]);
    const uint8_t * ipa_restrict s0 = srcs[0];

    for (w = doubler->src_w-4; w >= 0; w -= 4)
    {
        __m128i mm0, mm1;

        mm0 = _mm_loadu_si128((const __m128i *)s0);
        mm1 = _mm_shuffle_epi32(mm0, 0);
        _mm_storeu_si128((__m128i *)d0,mm1);
        _mm_storeu_si128((__m128i *)(d0+4),mm1);
        _mm_storeu_si128((__m128i *)d1,mm1);
        _mm_storeu_si128((__m128i *)(d1+4),mm1);
        _mm_storeu_si128((__m128i *)d2,mm1);
        _mm_storeu_si128((__m128i *)(d2+4),mm1);
        _mm_storeu_si128((__m128i *)d3,mm1);
        _mm_storeu_si128((__m128i *)(d3+4),mm1);
        _mm_storeu_si128((__m128i *)d4,mm1);
        _mm_storeu_si128((__m128i *)(d4+4),mm1);
        _mm_storeu_si128((__m128i *)d5,mm1);
        _mm_storeu_si128((__m128i *)(d5+4),mm1);
        _mm_storeu_si128((__m128i *)d6,mm1);
        _mm_storeu_si128((__m128i *)(d6+4),mm1);
        _mm_storeu_si128((__m128i *)d7,mm1);
        _mm_storeu_si128((__m128i *)(d7+4),mm1);
        mm1 = _mm_shuffle_epi32(mm0, 1 * 0x55);
        _mm_storeu_si128((__m128i *)(d0+8),mm1);
        _mm_storeu_si128((__m128i *)(d0+12),mm1);
        _mm_storeu_si128((__m128i *)(d1+8),mm1);
        _mm_storeu_si128((__m128i *)(d1+12),mm1);
        _mm_storeu_si128((__m128i *)(d2+8),mm1);
        _mm_storeu_si128((__m128i *)(d2+12),mm1);
        _mm_storeu_si128((__m128i *)(d3+8),mm1);
        _mm_storeu_si128((__m128i *)(d3+12),mm1);
        _mm_storeu_si128((__m128i *)(d4+8),mm1);
        _mm_storeu_si128((__m128i *)(d4+12),mm1);
        _mm_storeu_si128((__m128i *)(d5+8),mm1);
        _mm_storeu_si128((__m128i *)(d5+12),mm1);
        _mm_storeu_si128((__m128i *)(d6+8),mm1);
        _mm_storeu_si128((__m128i *)(d6+12),mm1);
        _mm_storeu_si128((__m128i *)(d7+8),mm1);
        _mm_storeu_si128((__m128i *)(d7+12),mm1);
        mm1 = _mm_shuffle_epi32(mm0, 2 * 0x55);
        _mm_storeu_si128((__m128i *)(d0+16),mm1);
        _mm_storeu_si128((__m128i *)(d0+20),mm1);
        _mm_storeu_si128((__m128i *)(d1+16),mm1);
        _mm_storeu_si128((__m128i *)(d1+20),mm1);
        _mm_storeu_si128((__m128i *)(d2+16),mm1);
        _mm_storeu_si128((__m128i *)(d2+20),mm1);
        _mm_storeu_si128((__m128i *)(d3+16),mm1);
        _mm_storeu_si128((__m128i *)(d3+20),mm1);
        _mm_storeu_si128((__m128i *)(d4+16),mm1);
        _mm_storeu_si128((__m128i *)(d4+20),mm1);
        _mm_storeu_si128((__m128i *)(d5+16),mm1);
        _mm_storeu_si128((__m128i *)(d5+20),mm1);
        _mm_storeu_si128((__m128i *)(d6+16),mm1);
        _mm_storeu_si128((__m128i *)(d6+20),mm1);
        _mm_storeu_si128((__m128i *)(d7+16),mm1);
        _mm_storeu_si128((__m128i *)(d7+20),mm1);
        mm1 = _mm_shuffle_epi32(mm0, 3 * 0x55);
        _mm_storeu_si128((__m128i *)(d0+24),mm1);
        _mm_storeu_si128((__m128i *)(d0+28),mm1);
        _mm_storeu_si128((__m128i *)(d1+24),mm1);
        _mm_storeu_si128((__m128i *)(d1+28),mm1);
        _mm_storeu_si128((__m128i *)(d2+24),mm1);
        _mm_storeu_si128((__m128i *)(d2+28),mm1);
        _mm_storeu_si128((__m128i *)(d3+24),mm1);
        _mm_storeu_si128((__m128i *)(d3+28),mm1);
        _mm_storeu_si128((__m128i *)(d4+24),mm1);
        _mm_storeu_si128((__m128i *)(d4+28),mm1);
        _mm_storeu_si128((__m128i *)(d5+24),mm1);
        _mm_storeu_si128((__m128i *)(d5+28),mm1);
        _mm_storeu_si128((__m128i *)(d6+24),mm1);
        _mm_storeu_si128((__m128i *)(d6+28),mm1);
        _mm_storeu_si128((__m128i *)(d7+24),mm1);
        _mm_storeu_si128((__m128i *)(d7+28),mm1);
        s0 += 16;
        d0 += 32;
        d1 += 32;
        d2 += 32;
        d3 += 32;
        d4 += 32;
        d5 += 32;
        d6 += 32;
        d7 += 32;
    }

    w += 4;
    while (w) /* So we can break out */
    {
        __m128i mm0, mm1;
        uint8_t local[16];

        memcpy(local, s0, w*4);
        mm0 = _mm_loadu_si128((const __m128i *)local);
        mm1 = _mm_shuffle_epi32(mm0, 0);
        _mm_storeu_si128((__m128i *)d0,mm1);
        _mm_storeu_si128((__m128i *)(d0+4),mm1);
        _mm_storeu_si128((__m128i *)d1,mm1);
        _mm_storeu_si128((__m128i *)(d1+4),mm1);
        _mm_storeu_si128((__m128i *)d2,mm1);
        _mm_storeu_si128((__m128i *)(d2+4),mm1);
        _mm_storeu_si128((__m128i *)d3,mm1);
        _mm_storeu_si128((__m128i *)(d3+4),mm1);
        _mm_storeu_si128((__m128i *)d4,mm1);
        _mm_storeu_si128((__m128i *)(d4+4),mm1);
        _mm_storeu_si128((__m128i *)d5,mm1);
        _mm_storeu_si128((__m128i *)(d5+4),mm1);
        _mm_storeu_si128((__m128i *)d6,mm1);
        _mm_storeu_si128((__m128i *)(d6+4),mm1);
        _mm_storeu_si128((__m128i *)d7,mm1);
        _mm_storeu_si128((__m128i *)(d7+4),mm1);
        if (w == 1) break;
        mm1 = _mm_shuffle_epi32(mm0, 1 * 0x55);
        _mm_storeu_si128((__m128i *)(d0+8),mm1);
        _mm_storeu_si128((__m128i *)(d0+12),mm1);
        _mm_storeu_si128((__m128i *)(d1+8),mm1);
        _mm_storeu_si128((__m128i *)(d1+12),mm1);
        _mm_storeu_si128((__m128i *)(d2+8),mm1);
        _mm_storeu_si128((__m128i *)(d2+12),mm1);
        _mm_storeu_si128((__m128i *)(d3+8),mm1);
        _mm_storeu_si128((__m128i *)(d3+12),mm1);
        _mm_storeu_si128((__m128i *)(d4+8),mm1);
        _mm_storeu_si128((__m128i *)(d4+12),mm1);
        _mm_storeu_si128((__m128i *)(d5+8),mm1);
        _mm_storeu_si128((__m128i *)(d5+12),mm1);
        _mm_storeu_si128((__m128i *)(d6+8),mm1);
        _mm_storeu_si128((__m128i *)(d6+12),mm1);
        _mm_storeu_si128((__m128i *)(d7+8),mm1);
        _mm_storeu_si128((__m128i *)(d7+12),mm1);
        if (w == 2) break;
        mm1 = _mm_shuffle_epi32(mm0, 2 * 0x55);
        _mm_storeu_si128((__m128i *)(d0+16),mm1);
        _mm_storeu_si128((__m128i *)(d0+20),mm1);
        _mm_storeu_si128((__m128i *)(d1+16),mm1);
        _mm_storeu_si128((__m128i *)(d1+20),mm1);
        _mm_storeu_si128((__m128i *)(d2+16),mm1);
        _mm_storeu_si128((__m128i *)(d2+20),mm1);
        _mm_storeu_si128((__m128i *)(d3+16),mm1);
        _mm_storeu_si128((__m128i *)(d3+20),mm1);
        _mm_storeu_si128((__m128i *)(d4+16),mm1);
        _mm_storeu_si128((__m128i *)(d4+20),mm1);
        _mm_storeu_si128((__m128i *)(d5+16),mm1);
        _mm_storeu_si128((__m128i *)(d5+20),mm1);
        _mm_storeu_si128((__m128i *)(d6+16),mm1);
        _mm_storeu_si128((__m128i *)(d6+20),mm1);
        _mm_storeu_si128((__m128i *)(d7+16),mm1);
        _mm_storeu_si128((__m128i *)(d7+20),mm1);
        break;
    }

    return 8;
}
