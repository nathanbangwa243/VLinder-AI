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
#ifndef ipa_impl_h_INCLUDED
#define ipa_impl_h_INCLUDED

#include "ipa.h"

#include <stdint.h>

struct ipa_context_s {
    ipa_allocators alloc;

    /* Indicates that the CPU supports sse 4.1 */
    int has_sse_4_1;

    /* Indicates that SSE 4.1 should be used.  The ipa_force_sse routine
     * can be used to disable SSE 4.1 but it will be forced to enabled
     * only if the hardware supports it.
     */
    int use_sse_4_1;
};

void *ipa_malloc(ipa_context *ctx, void *opaque, size_t size);

void ipa_free(ipa_context *ctx, void *opaque, void *ptr);

void *ipa_realloc(ipa_context *ctx, void *opaque, void *ptr, size_t new_size);

/*
    align must be a power of 2.
    Any block returned by ipa_malloc_aligned must be freed by ipa_free_aligned.
 */
void *ipa_malloc_aligned(ipa_context *ctx, void *opaque, size_t size, int align);

void ipa_free_aligned(ipa_context *ctx, void *opaque, void *ptr);

#endif
