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

/* This file is included from halftone.c if C cores are allowed. */

static void
core_halftone(uint8_t *dst, const uint8_t *contone, const uint8_t *screen, int w)
{
    int m = 128;
    int v = 0;
    while (w--)
    {
        uint8_t c = *contone++;
        uint8_t s = *screen++;
        if (c < s)
            v |= m;
        m >>= 1;
        if (m == 0)
            *dst++ = v, m = 128, v = 0;
    }
}
