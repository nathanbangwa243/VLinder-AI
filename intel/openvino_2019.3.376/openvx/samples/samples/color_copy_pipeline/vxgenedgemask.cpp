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

#include <VX/vx_intel_volatile.h>
#include <stdio.h>
#include <mmintrin.h>
#include <immintrin.h>
#include "vx_user_pipeline_nodes.h"

static inline void ProduceDilatedOutput(vx_uint8 *pScratchDilate0,
                                        vx_uint8 *pScratchDilate1,
                                        vx_uint8 *pScratchDilate2,
                                        vx_uint8 *pOutput,
                                        vx_uint32 tile_width)
{
  vx_uint32 unrolled_iterations = tile_width / 128;
  vx_uint32 remainder = (tile_width % 128);

  __m128i *pScratch0_128 = (__m128i *)pScratchDilate0;
  __m128i *pScratch1_128 = (__m128i *)pScratchDilate1;
  __m128i *pScratch2_128 = (__m128i *)pScratchDilate2;
  vx_uint16 *pDst = (vx_uint16 *)pOutput;

  const __m128i vunpacklower = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
  const __m128i vmask = _mm_setr_epi8(0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80);
  const __m128i vReversemask = _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);

  for( int i = 0; i < unrolled_iterations; i++ )
  {
    __m128i scratch0 = _mm_load_si128(pScratch0_128++);
    __m128i scratch1 = _mm_load_si128(pScratch1_128++);
    __m128i scratch2 = _mm_load_si128(pScratch2_128++);

    __m128i dilated_packed = _mm_or_si128(scratch0, _mm_or_si128(scratch1, scratch2));

    //This loop reverses the bits in every 16-bit chunk
    #pragma unroll(8)
    for( int i = 0; i < 8; i++)
    {
      __m128i unpacked = _mm_shuffle_epi8(dilated_packed, vunpacklower);
      unpacked = _mm_and_si128(unpacked, vmask);
      unpacked = _mm_cmpeq_epi8(unpacked, vmask);
      unpacked = _mm_shuffle_epi8(unpacked, vReversemask);
      *pDst++ = _mm_movemask_epi8(unpacked);
      dilated_packed = _mm_srli_si128(dilated_packed, 2);
    }
  }

  if( remainder )
  {
     //we know we have some extra space in the scratch buffer to load from.
     __m128i scratch0 = _mm_load_si128(pScratch0_128++);
     __m128i scratch1 = _mm_load_si128(pScratch1_128++);
     __m128i scratch2 = _mm_load_si128(pScratch2_128++);
     __m128i dilated_packed = _mm_or_si128(scratch0, _mm_or_si128(scratch1, scratch2));
     for( int i = 0; i < (remainder/16); i++)
     {
       __m128i unpacked = _mm_shuffle_epi8(dilated_packed, vunpacklower);
       unpacked = _mm_and_si128(unpacked, vmask);
       unpacked = _mm_cmpeq_epi8(unpacked, vmask);
       unpacked = _mm_shuffle_epi8(unpacked, vReversemask);
       *pDst++ = _mm_movemask_epi8(unpacked);
       dilated_packed = _mm_srli_si128(dilated_packed, 2);
     }
  }
}

//Shifting whole 128-bit SSE vectors by a specified bit is
// a bit tricky, as there's only SSE instructions available
// to shift by whole bytes. The following helper functions
// define small instruction sequences for the bit shifts that
// we need for dilation.
static inline __m128i ShiftLeft1Bit(__m128i v)
{
   __m128i lostbit = _mm_srli_epi64(v, 63);
           lostbit = _mm_slli_si128(lostbit, 8);
   __m128i result = _mm_or_si128(lostbit, _mm_slli_epi64(v, 1));
   return result;
}

static inline __m128i ShiftRight1Bit(__m128i v)
{
   __m128i lostbit = _mm_slli_epi64(v, 63);
           lostbit = _mm_srli_si128(lostbit, 8);
   __m128i result = _mm_or_si128(lostbit, _mm_srli_epi64(v, 1));
   return result;
}

static inline __m128i ShiftLeft127Bit(__m128i v)
{
   v = _mm_slli_epi64(v, 63);
   v = _mm_slli_si128(v, 8);
   return v;
}

static inline __m128i ShiftRight127Bit(__m128i v)
{
   v = _mm_srli_epi64(v, 63);
   v = _mm_srli_si128(v, 8);
   return v;
}


static inline void DilatePackedLine(vx_uint8 *pScratch, vx_uint32 tile_width)
{
   //we will process 128 pixels per loop iteration
   vx_uint32 loop_unrolled = tile_width / 128;

   //perform one extra loop iteration if the tile width is
   // not divisible by 64. Remember, we are reading out-of /
   // writing into scratch buffers which we intentionally added
   // some extra padding bytes for this purpose.
   if( tile_width % 128 )
   {
      loop_unrolled++;

      //we don't want any garbage pixels to affect the dilated output, so
      // we want to zero out the first "non-valid" byte
      pScratch[tile_width/8] = 0;
   }

   __m128i prevpacked = _mm_setzero_si128();

   __m128i *pScratch_128 = (__m128i *)pScratch;
   __m128i *pDilatedResult_128 = (__m128i *)pScratch;

   __m128i currpacked = _mm_load_si128(pScratch_128++);
   __m128i nextpacked = _mm_load_si128(pScratch_128++);

   for( int i = 0; i < loop_unrolled; i++ )
   {
      __m128i result = ShiftRight127Bit(prevpacked);
      result = _mm_or_si128(result, currpacked);
      result = _mm_or_si128(result, ShiftLeft1Bit(currpacked));
      result = _mm_or_si128(result, ShiftRight1Bit(currpacked));
      result = _mm_or_si128(result, ShiftLeft127Bit(nextpacked));

      _mm_store_si128(pDilatedResult_128++, result);

      prevpacked = currpacked;
      currpacked = nextpacked;

      nextpacked = _mm_load_si128(pScratch_128++);
  }
}

static inline void DilateHighYAndLowDilateX(vx_uint8 *pScratchThreshHigh0,
                                            vx_uint8 *pScratchThreshHigh1,
                                            vx_uint8 *pScratchThreshHigh2,
                                            vx_uint8 *pScratchThreshLow,
                                            vx_uint8 *pScratchDilate,
                                            vx_uint32 tile_width)
{

   //we will process 128 pixels per loop iteration
   vx_uint32 loop_unrolled = (tile_width) / 128;

   //perform one extra loop iteration if the tile width is
   // not divisible by 128. Remember, we are reading out-of /
   // writing into scratch buffers which we intentionally added
   // some extra padding bytes for this purpose.
   if( tile_width % 128 )
   {
      loop_unrolled++;
   }

   __m128i *pThreshHigh0_128 = (__m128i *)pScratchThreshHigh0;
   __m128i *pThreshHigh1_128 = (__m128i *)pScratchThreshHigh1;
   __m128i *pThreshHigh2_128 = (__m128i *)pScratchThreshHigh2;
   __m128i *pThreshLow128 = (__m128i *)pScratchThreshLow;
   __m128i *pDilate128 = (__m128i *)pScratchDilate;

   for( int i = 0; i < loop_unrolled; i++)
   {
      __m128i thresh_high0 = _mm_load_si128(pThreshHigh0_128++);
      __m128i thresh_high1 = _mm_load_si128(pThreshHigh1_128++);
      __m128i thresh_high2 = _mm_load_si128(pThreshHigh2_128++);

      //dilate the high threshold in the Y direction
      __m128i thresh_high = _mm_or_si128(thresh_high0, _mm_or_si128(thresh_high1, thresh_high2));

      //and this with the thresh low
      __m128i thresh_low = _mm_load_si128(pThreshLow128++);
      __m128i result = _mm_and_si128(thresh_low, thresh_high);

      _mm_store_si128(pDilate128++, result);
  }

   //dilate this output line
   DilatePackedLine(pScratchDilate, tile_width);
}

static inline void ThresholdHighAndLowDilateX(vx_uint8 *pInputLine,
                                              vx_uint8 *pScratchLineThreshHigh,
                                              vx_uint8 *pScratchLineThreshLow,
                                              vx_int16 thresh_high,
                                              vx_int16 thresh_low,
                                              vx_uint32 tile_width)
{

   vx_uint16 *pDstHigh = (vx_uint16 *)pScratchLineThreshHigh;
   vx_uint16 *pDstLow = (vx_uint16 *)pScratchLineThreshLow;

   __m128i vthreshhigh = _mm_set1_epi16(thresh_high);
   __m128i vthreshlow = _mm_set1_epi16(thresh_low);
   __m128i vmask = _mm_set1_epi16(0xff);

   //we will process 16 pixels per loop iteration
   vx_uint32 loop_unrolled = tile_width / 16;

   __m128i *pInput128 = (__m128i *)(pInputLine);

   //step 1, iterate through the input line, thresholding / packing into the output buffer
   for( int i = 0; i < loop_unrolled; i++)
   {
      //grab 8 pixels worth of input
      __m128i input128 = _mm_load_si128(pInput128++);

      //perform the threshold
      __m128i result_high0 = _mm_and_si128(_mm_cmpgt_epi16(input128, vthreshhigh), vmask);
      __m128i result_low0 = _mm_and_si128(_mm_cmpgt_epi16(input128, vthreshlow), vmask);

      //grab the next 8 pixels
      input128 = _mm_load_si128(pInput128++);

      //threshold
      __m128i result_high1 = _mm_and_si128(_mm_cmpgt_epi16(input128, vthreshhigh), vmask);
      __m128i result_low1 = _mm_and_si128(_mm_cmpgt_epi16(input128, vthreshlow), vmask);

      __m128i result_high = _mm_packus_epi16(result_high0, result_high1);
      __m128i result_low = _mm_packus_epi16(result_low0, result_low1);

      //grab the upper bit from all 16 bytes in the vector and store in scratch buffer
      *pDstHigh++ = _mm_movemask_epi8(result_high);
      *pDstLow++ = _mm_movemask_epi8(result_low);
   }

   //Step 2, dilate the packed high threshold line in the X-direction
   DilatePackedLine(pScratchLineThreshHigh, tile_width);

}


static vx_status vxGenEdgeMaskKernel(vx_node node,
                                     void *   parameters[],
                                     vx_uint32 num,
                                     void *   tile_memory,
                                     vx_size tile_memory_size)
{
   vx_tile_intel_t *pInTile = (vx_tile_intel_t *)parameters[0];
   vx_uint8 *pSrc = pInTile->base[0];
   vx_int32 srcStep = pInTile->addr[0].stride_y;

   vx_tile_intel_t *pOutTile = (vx_tile_intel_t *)parameters[1];
   vx_uint8 *pDst = pOutTile->base[0];
   vx_int32 dstStep = pOutTile->addr[0].stride_y;

   //tile_width is the width in 1bpp pixels. Since the input/output image are U8,
   // we multiply this by 8, as each byte implies 8 pixels for 1bpp.
   vx_uint32 tile_width = pOutTile->addr[0].dim_x*8;
   vx_uint32 tile_height = pOutTile->addr[0].dim_y;

   vx_size uStart = (vx_size)tile_memory;
   if( (uStart % 64) != 0 )
   {
      uStart = ((uStart/64)+1)*64;
   }
   vx_uint8 *pTileMemory = (vx_uint8 *)uStart;

   vx_size valid_scratch_bytes = tile_width/8;
   vx_size scratchLineStep = valid_scratch_bytes;
   if( (scratchLineStep % 128) != 0 )
   {
      scratchLineStep = ((scratchLineStep/128)+1)*128;
   }

   vx_uint8 *scratch_lines[11];
   for( int i = 0; i < 11; i++ )
   {
      scratch_lines[i] = pTileMemory + i*scratchLineStep;
   }

   const vx_int16 thresh_low = 400;
   const vx_int16 thresh_high = 400;

   vx_int32 prev2line = pOutTile->tile_y - 2;
   vx_int32 prev1line = pOutTile->tile_y - 1;
   vx_int32 currline = pOutTile->tile_y;
   vx_int32 next1line = pOutTile->tile_y + 1;
   vx_int32 next2line = pOutTile->tile_y + 2;

   vx_uint8 *pCurrent = pSrc + (currline - pInTile->tile_y)*srcStep;
   vx_uint8 *pPrev1Line = pCurrent - srcStep;
   vx_uint8 *pPrev2Line = pPrev1Line - srcStep;
   vx_uint8 *pNext1Line = pCurrent + srcStep;
   vx_uint8 *pNext2Line = pNext1Line + srcStep;

   vx_uint8 *pThreshHighPrev2Scratch = scratch_lines[0];
   vx_uint8 *pThreshHighPrev1Scratch = scratch_lines[1];
   vx_uint8 *pThreshCurrentScratch = scratch_lines[2];
   vx_uint8 *pThreshHighNext1Scratch = scratch_lines[3];
   vx_uint8 *pThreshHighNext2Scratch = scratch_lines[4];
   vx_uint8 *pDilatedScratch0 = scratch_lines[5];
   vx_uint8 *pDilatedScratch1 = scratch_lines[6];
   vx_uint8 *pDilatedScratch2 = scratch_lines[7];
   vx_uint8 *pThreshLow = scratch_lines[8];
   vx_uint8 *pThreshLowNext = scratch_lines[9];


   //prep the first 4 lines of our thresh-high scratch buffer, and first 3 lines of our thresh-low scratch buffers
   if( prev2line < 0 )
   {
      memset(pThreshHighPrev2Scratch, 0, valid_scratch_bytes);
   }
   else
   {
     //technically we don't need to generate the low threshold output for this line, but it gets generated anyway. It will
     // be overridden.
     ThresholdHighAndLowDilateX(pPrev2Line, pThreshHighPrev2Scratch, pThreshLow, thresh_high, thresh_low, tile_width);
   }

   if( prev1line < 0 )
   {
      memset(pThreshHighPrev1Scratch, 0, valid_scratch_bytes);
      memset(pThreshLow, 0, valid_scratch_bytes);
   }
   else
   {
     ThresholdHighAndLowDilateX(pPrev1Line, pThreshHighPrev1Scratch, pThreshLow, thresh_high, thresh_low, tile_width);
   }

   ThresholdHighAndLowDilateX(pCurrent, pThreshCurrentScratch, pThreshLowNext, thresh_high, thresh_low, tile_width);

   //generate the first output dilate line
   DilateHighYAndLowDilateX(pThreshHighPrev2Scratch, pThreshHighPrev1Scratch, pThreshCurrentScratch, pThreshLow, pDilatedScratch0, tile_width);

   //swap
   vx_uint8 *pTmpSwap = pThreshLow;
   pThreshLow = pThreshLowNext;
   pThreshLowNext = pTmpSwap;

   if( next1line >= pOutTile->image.height )
   {
      memset(pThreshHighNext1Scratch, 0, valid_scratch_bytes);
      memset(pThreshLowNext, 0, valid_scratch_bytes);
   }
   else
   {
      ThresholdHighAndLowDilateX(pNext1Line, pThreshHighNext1Scratch, pThreshLowNext, thresh_high, thresh_low, tile_width);
   }

   //generate the second output dilate line
   DilateHighYAndLowDilateX(pThreshHighPrev1Scratch, pThreshCurrentScratch, pThreshHighNext1Scratch, pThreshLow, pDilatedScratch1, tile_width);

   pTmpSwap = pThreshLow;
   pThreshLow = pThreshLowNext;
   pThreshLowNext = pTmpSwap;

   for( vx_int32 y = 0; y < tile_height; y++ )
   {
      if( next2line++ >= pOutTile->image.height )
      {
         memset(pThreshHighNext2Scratch, 0, valid_scratch_bytes);
         memset(pThreshLowNext, 0, valid_scratch_bytes);
      }
      else
      {
        ThresholdHighAndLowDilateX(pNext2Line, pThreshHighNext2Scratch, pThreshLowNext, thresh_high, thresh_low, tile_width);
      }

      DilateHighYAndLowDilateX(pThreshCurrentScratch, pThreshHighNext1Scratch, pThreshHighNext2Scratch, pThreshLow, pDilatedScratch2, tile_width);
      ProduceDilatedOutput(pDilatedScratch0, pDilatedScratch1, pDilatedScratch2, pDst, tile_width);

      //swap our circular buffer ptr's..
      //low thresh ptr's
      pTmpSwap = pThreshLow;
      pThreshLow = pThreshLowNext;
      pThreshLowNext = pTmpSwap;

      //high thresh ptr's
      pTmpSwap = pThreshHighPrev2Scratch;
      pThreshHighPrev2Scratch = pThreshHighPrev1Scratch;
      pThreshHighPrev1Scratch = pThreshCurrentScratch;
      pThreshCurrentScratch = pThreshHighNext1Scratch;
      pThreshHighNext1Scratch = pThreshHighNext2Scratch;
      pThreshHighNext2Scratch = pTmpSwap;

      //dilate ptr's
      pTmpSwap = pDilatedScratch0;
      pDilatedScratch0 = pDilatedScratch1;
      pDilatedScratch1 = pDilatedScratch2;
      pDilatedScratch2 = pTmpSwap;

      pNext2Line += srcStep;
      pDst += dstStep;
   }

   return VX_SUCCESS;
}

static vx_status VX_CALLBACK vxGenEdgeMaskValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
   vx_status status = VX_ERROR_INVALID_PARAMETERS;
   if(num!=2)
   {
      return status;
   }

   vx_df_image format = 0;
   vx_uint32 width = 0;
   vx_uint32 height = 0;

   if(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format)) == VX_SUCCESS)
   {
      if (format == VX_DF_IMAGE_S16)
      {
         status = VX_SUCCESS;
      }
      else
      {
         status = VX_ERROR_INVALID_VALUE;
         vxAddLogEntry((vx_reference)node, status, "vxGenEdgeMaskValidator Input Validation failed: invalid input image 0 format, it must be S16\n");
         return status;
      }
   }
   status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &width, sizeof(width));
   if (width%16!=0)
   {
      status = VX_ERROR_INVALID_VALUE;
      vxAddLogEntry((vx_reference)node, status, "vxGenEdgeMaskValidator Input Validation failed: input image 0 width must be evenly divisible by 16\n");
      return status;
   }

   width/=8;

   status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height));
   format = VX_DF_IMAGE_U8;

   PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[1], VX_IMAGE_WIDTH, &width, sizeof(width)));
   PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[1], VX_IMAGE_HEIGHT, &height, sizeof(height)));
   PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[1], VX_IMAGE_FORMAT, &format, sizeof(format)));

   return status;
}

static vx_status vxGenEdgeMaskTileMapping (vx_node node,
                                         vx_reference parameters[],
                                         const vx_tile_t_attributes_intel_t* dstRectIn,
                                         vx_tile_t_attributes_intel_t* srcRectOut,
                                         vx_uint32 param_num)
{
   vx_uint32 height = 0;
   vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height));

   srcRectOut->x = dstRectIn->x*8;
   srcRectOut->tile_block.width = dstRectIn->tile_block.width*8;
   srcRectOut->y = dstRectIn->y - 2;
   srcRectOut->tile_block.height = dstRectIn->tile_block.height + 4;

   if( srcRectOut->y < 0 )
   {
      srcRectOut->tile_block.height += srcRectOut->y;
      srcRectOut->y = 0;
   }

   if( (srcRectOut->y + srcRectOut->tile_block.height) > height )
   {
      srcRectOut->tile_block.height = height - srcRectOut->y;
   }

   return VX_SUCCESS;
}

static vx_status vxGenEdgeMaskTileDimensionsInitialize(vx_node node,
                                                const vx_reference *parameters,
                                                vx_uint32 param_num,
                                                const vx_tile_block_size_intel_t *tile_dimensions)
{

   //we want to have dedicated 1bpp scratch memory allocated for each thread
   //Set the 'tile memory size'
   //+256 for alignment purposes, and in case we need to read/write slightly past the
   // "valid" line boundary (as opposed to implementing scalar remainder code)
   vx_size tileMemorySize = ((tile_dimensions->width) + 256)*11;

  return vxSetNodeAttribute(node, VX_NODE_TILE_MEMORY_SIZE_INTEL, &tileMemorySize, sizeof(tileMemorySize));;
}


vx_status VX_API_CALL PublishGenEdgeMaskKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;
    //create a kernel via the vxAddAdvancedTilingKernelIntel interface
    vx_kernel kernel = vxAddAdvancedTilingKernelIntel(context,
        (char*)VX_KERNEL_NAME_USER_GENEDGEMASK,
        VX_KERNEL_NAME_GENEDGEMASK,
        vxGenEdgeMaskKernel,
        vxGenEdgeMaskTileMapping,
        2,
        vxGenEdgeMaskValidator,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        vxGenEdgeMaskTileDimensionsInitialize);

    PROCESS_VX_STATUS(context, vxGetStatus((vx_reference)kernel));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));

    SAFE_VX_CALL(status, context, vxFinalizeKernel(kernel));

    if( VX_SUCCESS != status )
        vxRemoveKernel( kernel );

    return status;
}
