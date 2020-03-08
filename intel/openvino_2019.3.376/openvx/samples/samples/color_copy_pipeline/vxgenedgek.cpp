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
#include <immintrin.h>
#include "vx_user_pipeline_nodes.h"

static inline void ProduceEdgeK(vx_uint8 *pScratchLineThreshHigh0,
                                vx_uint8 *pScratchLineThreshHigh1,
                                vx_uint8 *pScratchLineThreshHigh2,
                                vx_uint8 *pScratchLineThreshLow,
                                vx_uint8 *pInputRenderedK,
                                vx_uint8 *pInputEdgeMask,
                                vx_uint8 *pDst,
                                vx_uint32 tile_width)
{
   //as all inputs are packed, we can process 128 pixels per loop iteration
   vx_uint32 loop_unrolled = tile_width / 128;
   vx_uint32 remainder = tile_width % 128;

   __m128i *pScratchLineThreshHigh0_128 = (__m128i *)pScratchLineThreshHigh0;
   __m128i *pScratchLineThreshHigh1_128 = (__m128i *)pScratchLineThreshHigh1;
   __m128i *pScratchLineThreshHigh2_128 = (__m128i *)pScratchLineThreshHigh2;

   __m128i *pThreshLow_128 = (__m128i *)pScratchLineThreshLow;
   __m128i *pRenderedK_128 = (__m128i *)pInputRenderedK;
   __m128i *pEdgeMask_128 = (__m128i *)pInputEdgeMask;
   __m128i *pDst_128 = (__m128i *)pDst;

   for( int i = 0; i < loop_unrolled; i++ )
   {
      __m128i th0 = _mm_load_si128(pScratchLineThreshHigh0_128++);
      __m128i th1 = _mm_load_si128(pScratchLineThreshHigh1_128++);
      __m128i th2 = _mm_load_si128(pScratchLineThreshHigh2_128++);
      __m128i inpuT1_128 = _mm_or_si128(th0, _mm_or_si128(th1, th2));

      __m128i inpuT2_128 = _mm_load_si128(pThreshLow_128++);
      __m128i inputN128 = _mm_load_si128(pEdgeMask_128++);
      __m128i inputK128 = _mm_load_si128(pRenderedK_128++);

      __m128i n_and_t1 = _mm_and_si128(inputN128, inpuT1_128);
      __m128i n_and_t1_and_t2 = _mm_and_si128(n_and_t1, inpuT2_128);
      __m128i notn_and_t1_and_k = _mm_andnot_si128(n_and_t1, inputK128);
      __m128i output128 = _mm_or_si128(notn_and_t1_and_k, n_and_t1_and_t2);

      _mm_store_si128(pDst_128++, output128);
   }

   if( remainder )
   {
      vx_uint8 *pScratchHigh0_8 = (vx_uint8 *)pScratchLineThreshHigh0_128;
      vx_uint8 *pScratchHigh1_8 = (vx_uint8 *)pScratchLineThreshHigh1_128;
      vx_uint8 *pScratchHigh2_8 = (vx_uint8 *)pScratchLineThreshHigh2_128;

      vx_uint8 *pThreshLow_8 = (vx_uint8 *)pThreshLow_128;
      vx_uint8 *pEdgeMask_8 = (vx_uint8 *)pEdgeMask_128;
      vx_uint8 *pRenderedK_8 = (vx_uint8 *)pRenderedK_128;
      vx_uint8 *pDst8 = (vx_uint8 *)pDst_128;

      for( int i = 0; i < remainder/8; i++ )
      {
         vx_uint8 t1 = pScratchHigh0_8[i] | pScratchHigh1_8[i] | pScratchHigh2_8[i];
         vx_uint8 t2 = pThreshLow_8[i];
         vx_uint8 n = pEdgeMask_8[i];
         vx_uint8 k = pRenderedK_8[i];

         vx_uint8 n_and_t1 = n & t1;
         vx_uint8 n_and_t1_and_t2 = n_and_t1 & t2;
         vx_uint8 notn_and_t1_and_k = (~n_and_t1) & k;
         vx_uint8 output = notn_and_t1_and_k | n_and_t1_and_t2;
         pDst8[i] = output;
      }
   }
}


//Takes the 8bpp input line, thresholds it using thresh_high and thresh_low
// The high thresholded result is dilated in the X direction.
static inline void ThresholdHighAndLow(vx_uint8 *pInputLine,
                                       vx_uint8 *pScratchLineThreshHigh,
                                       vx_uint8 *pScratchLineThreshLow,
                                       vx_uint8 thresh_high,
                                       vx_uint8 thresh_low,
                                       vx_uint32 tile_width)
{

  vx_uint16 *pDstHigh = (vx_uint16 *)pScratchLineThreshHigh;
  vx_uint16 *pDstLow = (vx_uint16 *)pScratchLineThreshLow;

  const __m128i vthreshhigh = _mm_set1_epi8(thresh_high);
  const __m128i vthreshlow = _mm_set1_epi8(thresh_low);
  const __m128i vReversemask = _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);

  //we will process 16 pixels per loop iteration
  vx_uint32 loop_unrolled = tile_width / 16;

  __m128i *pInput128 = (__m128i *)(pInputLine);

  __m128i threshhighprev = _mm_setzero_si128();
  __m128i threshhighcurrent;
  {
     __m128i input128 = _mm_load_si128(pInput128++);

     __m128i max = _mm_max_epu8(input128, vthreshlow);
     __m128i threshlowresult = _mm_cmpeq_epi8(max, input128);
     threshlowresult = _mm_shuffle_epi8(threshlowresult, vReversemask);
     *pDstLow++ = _mm_movemask_epi8(threshlowresult);

     max = _mm_max_epu8(input128, vthreshhigh);
     threshhighcurrent = _mm_cmpeq_epi8(max, input128);

     input128 = _mm_load_si128(pInput128++);

     max = _mm_max_epu8(input128, vthreshlow);
     threshlowresult = _mm_cmpeq_epi8(max, input128);
     threshlowresult = _mm_shuffle_epi8(threshlowresult, vReversemask);
     *pDstLow++ = _mm_movemask_epi8(threshlowresult);

     max = _mm_max_epu8(input128, vthreshhigh);
     __m128i threshhighnext = _mm_cmpeq_epi8(max, input128);

     __m128i shr = _mm_alignr_epi8(threshhighnext, threshhighcurrent, 1);
     __m128i shl = _mm_slli_si128(threshhighcurrent, 1);

     __m128i threshhighresult = _mm_or_si128(threshhighprev,
                                             _mm_or_si128(shl,
                                             _mm_or_si128(threshhighcurrent, shr)));

     threshhighresult = _mm_shuffle_epi8(threshhighresult, vReversemask);
     *pDstHigh++ = _mm_movemask_epi8(threshhighresult);

     threshhighprev = _mm_srli_si128(threshhighcurrent, 15);
     threshhighcurrent = threshhighnext;

  }

  loop_unrolled -= 2;

  for( int i = 0; i < loop_unrolled; i++)
  {
    __m128i input128 = _mm_load_si128(pInput128++);
    __m128i max = _mm_max_epu8(input128, vthreshlow);
     __m128i threshlowresult = _mm_cmpeq_epi8(max, input128);
     threshlowresult = _mm_shuffle_epi8(threshlowresult, vReversemask);
     *pDstLow++ = _mm_movemask_epi8(threshlowresult);

     max = _mm_max_epu8(input128, vthreshhigh);
     __m128i threshhighnext = _mm_cmpeq_epi8(max, input128);
     __m128i shr = _mm_alignr_epi8(threshhighnext, threshhighcurrent, 1);
     __m128i shl = _mm_slli_si128(threshhighcurrent, 1);
     __m128i threshhighresult = threshhighnext;
     threshhighresult = _mm_or_si128(threshhighprev,
                                     _mm_or_si128(shl,
                                     _mm_or_si128(threshhighcurrent, shr)));
     threshhighresult = _mm_shuffle_epi8(threshhighresult, vReversemask);
     *pDstHigh++ = _mm_movemask_epi8(threshhighresult);

     threshhighprev = _mm_srli_si128(threshhighcurrent, 15);
     threshhighcurrent = threshhighnext;
  }

  {
     __m128i shr = _mm_srli_si128(threshhighcurrent, 1);
     __m128i shl = _mm_slli_si128(threshhighcurrent, 1);
     __m128i threshhighresult = _mm_or_si128(threshhighprev,
                                             _mm_or_si128(shl,
                                             _mm_or_si128(threshhighcurrent, shr)));
     threshhighresult = _mm_shuffle_epi8(threshhighresult, vReversemask);
     *pDstHigh++ = _mm_movemask_epi8(threshhighresult);
  }

}


static vx_status vxGenEdgeKKernel(vx_node node,
                                  void *   parameters[],
                                  vx_uint32 num,
                                  void *   tile_memory,
                                  vx_size tile_memory_size)
{
    vx_tile_intel_t *pInNEdgeMaskTile = (vx_tile_intel_t *)parameters[0];
    vx_uint8 *pSrcNEdgeMask = pInNEdgeMaskTile->base[0];
    vx_int32 srcStepNEdgeMask = pInNEdgeMaskTile->addr[0].stride_y;

    vx_tile_intel_t *pInContoneKTile = (vx_tile_intel_t *)parameters[1];
    vx_uint8 *pSrcContoneK= pInContoneKTile->base[0];
    vx_int32 srcStepContoneK = pInContoneKTile->addr[0].stride_y;

    vx_tile_intel_t *pInRenderedKTile = (vx_tile_intel_t *)parameters[2];
    vx_uint8 *pSrcRenderedK = pInRenderedKTile->base[0];
    vx_int32 srcStepRenderedK = pInRenderedKTile->addr[0].stride_y;

    vx_tile_intel_t *pOutTile = (vx_tile_intel_t *)parameters[5];
    vx_uint8 *pDst = pOutTile->base[0];
    vx_int32 dstStep = pOutTile->addr[0].stride_y;

    //tile_width is the width in 1bpp pixels. Since the input/output image are U8,
    // we multiply this by 8, as each byte implies 8 pixels for 1bpp.
    vx_uint32 tile_width = pOutTile->addr[0].dim_x*8;
    vx_uint32 tile_height = pOutTile->addr[0].dim_y;

    vx_scalar sThreshLow = (vx_scalar)parameters[3];
    vx_scalar sThreshHigh = (vx_scalar)parameters[4];

    vx_uint8 thresholdLow;
    vxCopyScalar(sThreshLow, &thresholdLow, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    vx_uint8 thresholdHigh;
    vxCopyScalar(sThreshHigh, &thresholdHigh, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    vx_size uStart = (vx_size)tile_memory;
    if( (uStart % 64) != 0 )
    {
       uStart = ((uStart/64)+1)*64;
    }
    vx_uint8 *pTileMemory = (vx_uint8 *)uStart;

    vx_size valid_scratch_bytes = pOutTile->addr[0].dim_x;
    vx_size scratchLineStep = valid_scratch_bytes;
    if( (scratchLineStep % 128) != 0 )
    {
       scratchLineStep = ((scratchLineStep/128)+1)*128;
    }

    vx_uint8 *scratch_lines[5];
    for( int i = 0; i < 5; i++ )
    {
      scratch_lines[i] = pTileMemory + i*scratchLineStep;
    }

    vx_int32 prevline = pOutTile->tile_y - 1;
    vx_int32 currline = pOutTile->tile_y;
    vx_int32 nextline = pOutTile->tile_y + 1;

    vx_uint8 *pCurrent = pSrcContoneK + (currline - pInContoneKTile->tile_y)*srcStepContoneK;
    vx_uint8 *pPrevLine = pCurrent - srcStepContoneK;
    vx_uint8 *pNextLine = pCurrent + srcStepContoneK;

    vx_uint8 *pThreshHighPrevScratch = scratch_lines[0];
    vx_uint8 *pThreshHighCurrentScratch = scratch_lines[1];
    vx_uint8 *pThreshHighNextScratch = scratch_lines[2];

    vx_uint8 *pThreshLowCurrent = scratch_lines[3];
    vx_uint8 *pThreshLowNext = scratch_lines[4];

    //prep the first 4 lines of our thresh-high scratch buffer, and first 3 lines of our thresh-low scratch buffers
    if( prevline < 0 )
    {
       memset(pThreshHighPrevScratch, 0, valid_scratch_bytes);
    }
    else
    {
      //technically we don't need to generate the low threshold output for this line, but it gets generated anyway. It will
      // be overridden.
      ThresholdHighAndLow(pPrevLine, pThreshHighPrevScratch, pThreshLowCurrent, thresholdHigh, thresholdLow, tile_width);
    }

    ThresholdHighAndLow(pCurrent, pThreshHighCurrentScratch, pThreshLowCurrent, thresholdHigh, thresholdLow, tile_width);


    for( vx_int32 y = 0; y < tile_height; y++ )
    {
       if( nextline++ >= pOutTile->image.height )
       {
          memset(pThreshHighNextScratch, 0, valid_scratch_bytes);
       }
       else
       {
         ThresholdHighAndLow(pNextLine, pThreshHighNextScratch, pThreshLowNext, thresholdHigh, thresholdLow, tile_width);
       }

       ProduceEdgeK(pThreshHighPrevScratch,
                    pThreshHighCurrentScratch,
                    pThreshHighNextScratch,
                    pThreshLowCurrent,
                    pSrcRenderedK,
                    pSrcNEdgeMask,
                    pDst,
                    tile_width);

       //swap our circular buffer ptr's
       //low thresh ptr's
       vx_uint8 *pTmpSwap = pThreshLowCurrent;
       pThreshLowCurrent = pThreshLowNext;
       pThreshLowNext = pTmpSwap;

       //high thresh ptr's
       pTmpSwap = pThreshHighPrevScratch;
       pThreshHighPrevScratch = pThreshHighCurrentScratch;
       pThreshHighCurrentScratch = pThreshHighNextScratch;
       pThreshHighNextScratch = pTmpSwap;

       pNextLine += srcStepContoneK;
       pSrcRenderedK += srcStepRenderedK;
       pSrcNEdgeMask += srcStepNEdgeMask;
       pDst += dstStep;
    }

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK vxGenEdgeKValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    if(num!=6)
    {
        return status;
    }

    vx_df_image format = 0;
    vx_uint32 width = 0;
    vx_uint32 height = 0;

    if(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format)) == VX_SUCCESS)
    {
        if (format == VX_DF_IMAGE_U8)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "vxGenEdgeKValidator Input Validation failed: invalid input image 0 format, it must be U8\n");
            return status;
        }
    }
    status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &width, sizeof(width));
    status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height));
    format = VX_DF_IMAGE_U8;

    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[5], VX_IMAGE_WIDTH, &width, sizeof(width)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[5], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[5], VX_IMAGE_FORMAT, &format, sizeof(format)));

    return status;
}

static vx_status vxGenEdgeKTileMapping (vx_node node,
                                         vx_reference parameters[],
                                         const vx_tile_t_attributes_intel_t* dstRectIn,
                                         vx_tile_t_attributes_intel_t* srcRectOut,
                                         vx_uint32 param_num)
{
   if( param_num == 1)
   {
      vx_uint32 height = 0;
      vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height));

      srcRectOut->x = dstRectIn->x*8;
      srcRectOut->tile_block.width = dstRectIn->tile_block.width*8;

      srcRectOut->y = dstRectIn->y - 1;
      srcRectOut->tile_block.height = dstRectIn->tile_block.height + 2;

      if( srcRectOut->y < 0 )
      {
        srcRectOut->tile_block.height += srcRectOut->y;
        srcRectOut->y = 0;
      }

      if( (srcRectOut->y + srcRectOut->tile_block.height) > height )
      {
        srcRectOut->tile_block.height = height - srcRectOut->y;
      }
   }
   else
   {
      srcRectOut->x = dstRectIn->x;
      srcRectOut->y = dstRectIn->y;
      srcRectOut->tile_block.width = dstRectIn->tile_block.width;
      srcRectOut->tile_block.height = dstRectIn->tile_block.height;
   }

   return VX_SUCCESS;
}

static vx_status vxGenEdgeKTileDimensionsInitialize(vx_node node,
                                             const vx_reference *parameters,
                                             vx_uint32 param_num,
                                             const vx_tile_block_size_intel_t *tile_dimensions)
{
   //we want to have dedicated scratch memory allocated for each thread
   //Set the 'tile memory size'
   // +256 for alignment purposes, and in case we need to read past "valid" pixel
   // boundaries for remainder conditions
   vx_size tileMemorySize = ((tile_dimensions->width) + 256)*5;

   return vxSetNodeAttribute(node, VX_NODE_TILE_MEMORY_SIZE_INTEL, &tileMemorySize, sizeof(tileMemorySize));
}

vx_status VX_API_CALL PublishGenEdgeKKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;
    //create a kernel via the vxAddAdvancedTilingKernelIntel interface
    vx_kernel kernel = vxAddAdvancedTilingKernelIntel(context,
        (char*)VX_KERNEL_NAME_USER_GENEDGEK,
        VX_KERNEL_USER_GENEDGEK,
        vxGenEdgeKKernel,
        vxGenEdgeKTileMapping,
        6,
        vxGenEdgeKValidator,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        vxGenEdgeKTileDimensionsInitialize);

    PROCESS_VX_STATUS(context, vxGetStatus((vx_reference)kernel));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 5, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxFinalizeKernel(kernel));

    if( VX_SUCCESS != status )
        vxRemoveKernel( kernel );

    return status;
}
