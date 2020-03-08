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
/* Intel Printing Acceleration library */

#include "ipa-impl.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
/* Runtime detection of SSE */

#ifdef _MSC_VER
#include <intrin.h>
#endif

static int is_sse41_supported()
{
#ifdef _MSC_VER
   uint32_t info[4];
   __cpuid(info, 1);
   return (info[2] & (1U << 19)) != 0;
#else
#ifdef __GNUC__
   uint32_t eax=1;
   uint32_t ebx=0;
   uint32_t ecx=0;
   uint32_t edx=0;
   __asm__ ( "cpuid" : "+b"(ebx),"+a" (eax), "+c"(ecx), "=d"(edx) );
   return (ecx & (1U << 19)) != 0;
#else
   return 0;
#endif
#endif
}

/* External API to create/destroy instances */

/* Initialise an instance. */
ipa_context *ipa_init(const ipa_allocators *alloc, void *opaque)
{
    ipa_context *ctx;
    int has_sse_4_1;

    if (alloc == NULL)
	return NULL;

    has_sse_4_1 = is_sse41_supported();

#if defined(IPA_FORCE_SSE) && IPA_FORCE_SSE == 1
    /* SSE is forced on (we only have SSE implementations), but this host doesn't have SSE. Fail to init. */
    if (has_sse_4_1 == 0)
        return NULL;
#endif

    ctx = alloc->ipa_malloc(opaque, sizeof(*ctx));
    if (!ctx)
	return NULL;

    memcpy(&ctx->alloc, alloc, sizeof(*alloc));

    ctx->has_sse_4_1 = has_sse_4_1;
    ctx->use_sse_4_1 = ctx->has_sse_4_1;

    return ctx;
}

/* Finalise an instance. */
void ipa_fin(ipa_context *ctx, void *opaque)
{
    if (ctx == NULL)
	return;

    ctx->alloc.ipa_free(opaque, ctx);
}

/* Internal malloc APIs */

void *ipa_malloc(ipa_context *ctx, void *opaque, size_t size)
{
    if (ctx == NULL)
	return NULL;

    return ctx->alloc.ipa_malloc(opaque, size);
}

void ipa_free(ipa_context *ctx, void *opaque, void *ptr)
{
    if (ptr == NULL || ctx == NULL)
	return;

    ctx->alloc.ipa_free(opaque, ptr);
}

void *ipa_realloc(ipa_context *ctx, void *opaque, void *ptr, size_t new_size)
{
    if (ctx == NULL)
	return NULL;

    if (ptr == NULL)
    {
	if (new_size == 0)
	    return NULL;
	return ipa_malloc(ctx, opaque, new_size);
    }

    if (new_size == 0)
    {
	ipa_free(ctx, opaque, ptr);
	return NULL;
    }

    return ctx->alloc.ipa_realloc(opaque, ptr, new_size);
}

/*
    To manage alignment, we will store a struct before the pointer.
    We could probably get away without storing the real_size and
    alignment if we didn't want to allow for reallocing later.
*/


typedef struct {
    size_t   real_size;
    uint16_t align;
    uint16_t back;
} ipa_blkalign;

#define HEADER_SIZE     (2*sizeof(size_t))
#define PTR_TO_HDR(ptr) ((ipa_blkalign)&(((size_t *)(void *)(ptr))[-2]))
#define HDR_TO_BLK(hdr) ((void *)(&((char *)(hdr))[-(hdr)->back]))

/* align is assumed to be a power of 2. */
void *ipa_malloc_aligned(ipa_context *ctx, void *opaque, size_t size, int align)
{
    unsigned char *block;
    intptr_t ptr;
    ipa_blkalign *hdr;

    if (size == 0)
	return NULL;

    if (align < sizeof(size_t))
	align = sizeof(size_t);
    block = ipa_malloc(ctx, opaque, size + HEADER_SIZE + align);
    if (block == NULL)
	return NULL;

    ptr = (intptr_t)block;
    ptr = (ptr + HEADER_SIZE + align-1) & ~(intptr_t)(align-1);
    hdr = (ipa_blkalign *)(ptr - HEADER_SIZE);
    hdr->back = (unsigned short)((unsigned char *)hdr - block);
    hdr->align = (unsigned short)align;
    hdr->real_size = size;

    return (void *)ptr;
}

void ipa_free_aligned(ipa_context *ctx, void *opaque, void *ptr)
{
    ipa_blkalign *hdr;
    unsigned char *block;

    if (ptr == NULL)
	return;

    hdr = (ipa_blkalign *)(&((size_t *)ptr)[-2]);
    block = &((unsigned char *)(hdr))[-hdr->back];
    ipa_free(ctx, opaque, block);
}

int ipa_force_sse(ipa_context *ctx, int sse)
{
    if (!ctx)
        return 0;

#if defined(IPA_FORCE_SSE)
#if IPA_FORCE_SSE == 0
    /* We only have non-SSE implementations. */
    if (sse)
        return 1;
#endif
#if IPA_FORCE_SSE == 1
    /* We only have SSE implementations. */
    if (!sse)
        return 1;
#endif
#endif

    /* Always use the software version if the CPU doesn't support SSE 4.1 */
    ctx->use_sse_4_1 = (ctx->has_sse_4_1 ? !!sse : 0);

    return 0;
}

int ipa_cpu_supports_sse_4_1(ipa_context *ctx)
{
    return ctx ? ctx->has_sse_4_1 : 0;
}
