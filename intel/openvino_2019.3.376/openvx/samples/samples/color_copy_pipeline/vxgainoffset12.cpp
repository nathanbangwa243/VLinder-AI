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
#include <immintrin.h>
#include "vx_user_pipeline_nodes.h"

//This function takes in a vector of 8 U16 pixels, which it will apply gain / offset to, and return
// the result vector, which is also a vector of 8 U16 pixels.
static inline __m128i ProcessGainOffset16BPP(__m128i input, float *&pGain, float *&pOffset)
{
   __m128i result32_0;
   {
      //convert the lower 4 pixels to 32f
      __m128 tmp32f = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(input));

      //load 4 gain values
      __m128 gain128 = _mm_load_ps(pGain);
      pGain += 4;

      //load 4 offset values
      __m128 offset128 = _mm_load_ps(pOffset);
      pOffset += 4;

      //result = (input * gain + offset) * agcline
      __m128 result = _mm_mul_ps(tmp32f, gain128);
             result = _mm_add_ps(result, offset128);

      result32_0 = _mm_cvtps_epi32(result);
   }

   //shift the input over by 8 (4 pixels) to prepare for the next conversion to 32f
   input = _mm_srli_si128(input, 8);

   __m128i result32_1;
   {
      //convert the lower 4 pixels to 32f
      __m128 tmp32f = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(input));

      //load 4 gain values
      __m128 gain128 = _mm_load_ps(pGain);
      pGain += 4;

      //load 4 offset values
      __m128 offset128 = _mm_load_ps(pOffset);
      pOffset += 4;

      //result = (input * gain + offset) * agcline
      __m128 result = _mm_mul_ps(tmp32f, gain128);
             result = _mm_add_ps(result, offset128);

      result32_1 = _mm_cvtps_epi32(result);
   }

   return _mm_packus_epi32(result32_0, result32_1);
}

static void GetGainOffsetStartPtr(void *pScratch, vx_float32 *&pGain, vx_float32 *&pOffset, vx_uint32 width)
{
  vx_size uStart = (vx_size)pScratch;
  if( (uStart % 64) != 0 )
  {
     uStart = ((uStart/64)+1)*64;
  }
  vx_uint8 *pStart = (vx_uint8 *)uStart;

  vx_size gainoffsetbytes = width * sizeof(vx_float32);
  if( (gainoffsetbytes % 64) != 0 )
  {
     gainoffsetbytes = ((gainoffsetbytes/64)+1)*64;
  }


  pGain = (vx_float32 *)pStart;
  pOffset = (vx_float32 *)(pStart + gainoffsetbytes);
}

static vx_status vxGainOffset12PreProcess(vx_node node,
                                          const vx_reference *parameters,
                                          vx_uint32 num_parameters,
                                          void * tile_memory[],
                                          vx_uint32 num_tile_memory_elements,
                                          vx_size tile_memory_size)
{
   void *pScratch = 0;
   PROCESS_VX_STATUS(node, vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &pScratch, sizeof(pScratch)));

   vx_image output = (vx_image)parameters[1];
   vx_int32 width;
   vxQueryImage(output, VX_IMAGE_WIDTH, &width, sizeof(width));

   vx_float32 *pScratchGain, *pScratchOffset;
   GetGainOffsetStartPtr(pScratch, pScratchGain, pScratchOffset, width);

   vx_array gain = (vx_array)parameters[2];
   vx_array offset = (vx_array)parameters[3];

   vx_size num_items = 0;
   vx_size stride = 0;

   vx_float32 *pGain = 0;
   vx_map_id gain_map_id;

   PROCESS_VX_STATUS(node, vxQueryArray(gain, VX_ARRAY_CAPACITY, &num_items, sizeof(num_items)));
   //we need to have, at least, a gain entry per pixel
   if( num_items < width )
   {
      vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_VALUE, "vxGainOffset12PreProcess: The number of gain entries must greater than or equal to the image width\n");
      return VX_FAILURE;
   }
   PROCESS_VX_STATUS(node, vxMapArrayRange(gain, 0, num_items, &gain_map_id, &stride, reinterpret_cast<void**>(&pGain), VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));

   vx_float32 *pOffset = 0;
   vx_map_id offset_map_id;
   vxQueryArray(offset, VX_ARRAY_CAPACITY, &num_items, sizeof(num_items));
   //we need to have, at least, an offset entry per pixel
   if( num_items < width )
   {
      vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_VALUE, "vxGainOffset12PreProcess: The number of offset entries must greater than or equal to the image width\n");
      return VX_FAILURE;
   }
   PROCESS_VX_STATUS(node, vxMapArrayRange(offset, 0, num_items, &offset_map_id, &stride, reinterpret_cast<void**>(&pOffset), VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));

   vx_scalar sAgc = (vx_scalar)parameters[4];
   vx_float32 agc = 0;
   PROCESS_VX_STATUS(node, vxCopyScalar(sAgc, &agc, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

   //calculate our "new" gain / offset
   for( vx_int32 i = 0; i < width; i++ )
   {
      pScratchGain[i] = pGain[i] * agc;
      pScratchOffset[i] = pOffset[i] * agc;
   }

   PROCESS_VX_STATUS(node, vxUnmapArrayRange(gain, gain_map_id));
   PROCESS_VX_STATUS(node, vxUnmapArrayRange(offset, offset_map_id));

   return VX_SUCCESS;
}

static vx_status vxGainOffset12Kernel(vx_node node,
                                      void *   parameters[],
                                      vx_uint32 num,
                                      void *   tile_memory,
                                      vx_size tile_memory_size)
{
    vx_tile_intel_t *pInTileIn = (vx_tile_intel_t *)parameters[0];
    vx_tile_intel_t *pInTileOut = (vx_tile_intel_t *)parameters[1];

    vx_uint8 *pSrc = pInTileIn->base[0];
    vx_uint8 *pDst = pInTileOut->base[0];

    vx_int32 srcStep = pInTileIn->addr[0].stride_y;
    vx_int32 dstStep = pInTileOut->addr[0].stride_y;

    void *pScratch = 0;
    PROCESS_VX_STATUS(node, vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &pScratch, sizeof(pScratch)));

    vx_float32 *pScratchGain, *pScratchOffset;
    GetGainOffsetStartPtr(pScratch, pScratchGain, pScratchOffset, pInTileOut->image.width);

    vx_uint32 output_tile_width = pInTileOut->addr[0].dim_x;
    vx_uint32 output_tile_height = pInTileOut->addr[0].dim_y;

    //each iteration we are going to produce 32 pixels
    vx_uint32 unrolled_iterations = output_tile_width / 32;

    vx_uint8 *pPackedTmp = pSrc;
    vx_uint8 *pUnpackedTmp = pDst;

    __m128i maskeven = _mm_setr_epi8(0xff, 0x0f, 0xff, 0xff, 0x0f, 0xff, 0xff, 0x0f, 0xff, 0xff, 0x0f, 0xff, 0xff, 0x0f, 0xff, 0xff);
    __m128i extracteven = _mm_setr_epi8(0, 1, 3, 4, 6, 7, 9, 10, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff);

    __m128i maskodd =  _mm_setr_epi8(0xff, 0xf0, 0xff, 0xff, 0xf0, 0xff, 0xff, 0xf0, 0xff, 0xff, 0xf0, 0xff, 0xff, 0xf0, 0xff, 0xff);
    __m128i extractodd =  _mm_setr_epi8(1, 2, 4, 5, 7, 8, 10, 11, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff);

    //get the starting offsets to the offset, gain, and agc
    vx_float32 *pTileGain = &pScratchGain[pInTileOut->tile_x];
    vx_float32 *pTileOffset = &pScratchOffset[pInTileOut->tile_x];

    //convert the unpacked input to the packed input
    for( vx_uint32 y = 0; y < output_tile_height; y++ )
    {
       __m128i *pPackedLine = (__m128i *)(pPackedTmp);
       __m128i *pUnpackedLine = (__m128i *)(pUnpackedTmp);

       vx_float32 *tmpGain = pTileGain;
       vx_float32 *tmpOffset = pTileOffset;

       for( vx_uint32 x = 0; x < unrolled_iterations; x++ )
       {
          __m128i inputpacked0 = _mm_load_si128(pPackedLine++);

          __m128i result16_0;
          //first 8 pixels
          {
             __m128i inputpacked = inputpacked0;

             __m128i tmpeven = _mm_and_si128(inputpacked, maskeven);
                     tmpeven = _mm_shuffle_epi8(tmpeven, extracteven);

             __m128i tmpodd = _mm_and_si128(inputpacked, maskodd);
                     tmpodd = _mm_shuffle_epi8(tmpodd, extractodd);
                     tmpodd = _mm_srli_epi16(tmpodd, 4);


             __m128i unpacked = _mm_unpacklo_epi16(tmpeven, tmpodd);

             //Apply Gain/Offset to these 8 pixels
             result16_0 = ProcessGainOffset16BPP(unpacked, tmpGain, tmpOffset);
           }

          __m128i inputpacked1 = _mm_load_si128(pPackedLine++);

          __m128i result16_1;
          //second 8 pixels
          {
             __m128i inputpacked = _mm_alignr_epi8(inputpacked1, inputpacked0, 12);

             __m128i tmpeven = _mm_and_si128(inputpacked, maskeven);
                     tmpeven = _mm_shuffle_epi8(tmpeven, extracteven);


             __m128i tmpodd = _mm_and_si128(inputpacked, maskodd);
                     tmpodd = _mm_shuffle_epi8(tmpodd, extractodd);
                     tmpodd = _mm_srli_epi16(tmpodd, 4);


             __m128i unpacked = _mm_unpacklo_epi16(tmpeven, tmpodd);

             result16_1 = ProcessGainOffset16BPP(unpacked, tmpGain, tmpOffset);
          }

          //pack 16 U16 pixels into 16 U8 pixels
          __m128i result8 = _mm_packus_epi16(result16_0, result16_1);

          //write it to memory
          _mm_store_si128(pUnpackedLine++, result8);

          __m128i inputpacked2 = _mm_load_si128(pPackedLine++);
          //third 8 pixels
          {
             __m128i inputpacked = _mm_alignr_epi8(inputpacked2, inputpacked1, 8);

             __m128i tmpeven = _mm_and_si128(inputpacked, maskeven);
                     tmpeven = _mm_shuffle_epi8(tmpeven, extracteven);


             __m128i tmpodd = _mm_and_si128(inputpacked, maskodd);
                     tmpodd = _mm_shuffle_epi8(tmpodd, extractodd);
                     tmpodd = _mm_srli_epi16(tmpodd, 4);


             __m128i unpacked = _mm_unpacklo_epi16(tmpeven, tmpodd);

             //write these 8 pixels
             result16_0 = ProcessGainOffset16BPP(unpacked, tmpGain, tmpOffset);
          }

          //fourth 8 pixels
          {
             __m128i inputpacked = _mm_srli_si128(inputpacked2, 4);

             __m128i tmpeven = _mm_and_si128(inputpacked, maskeven);
                     tmpeven = _mm_shuffle_epi8(tmpeven, extracteven);


             __m128i tmpodd = _mm_and_si128(inputpacked, maskodd);
                     tmpodd = _mm_shuffle_epi8(tmpodd, extractodd);
                     tmpodd = _mm_srli_epi16(tmpodd, 4);

              __m128i unpacked = _mm_unpacklo_epi16(tmpeven, tmpodd);

              result16_1 = ProcessGainOffset16BPP(unpacked, tmpGain, tmpOffset);
          }

          result8 = _mm_packus_epi16(result16_0, result16_1);

          //write it to memory
          _mm_store_si128(pUnpackedLine++, result8);
       }

       pPackedTmp += srcStep;
       pUnpackedTmp += dstStep;

    }

    return VX_SUCCESS;

}

static vx_status vxGainOffset12TileMapping (vx_node node,
                                           vx_reference parameters[],
                                           const vx_tile_t_attributes_intel_t* dstRectIn,
                                           vx_tile_t_attributes_intel_t* srcRectOut,
                                           vx_uint32 param_num)
{
    vx_int32 inputbitstart = (dstRectIn->x * 12);
    vx_int32 inputbitwidth = (dstRectIn->tile_block.width * 12);

    srcRectOut->x = inputbitstart / 8;
    srcRectOut->tile_block.width = inputbitwidth / 8;
    srcRectOut->y = dstRectIn->y;
    srcRectOut->tile_block.height = dstRectIn->tile_block.height;

    return VX_SUCCESS;

}

static vx_status vxGainOffset12SetTileDimensions(vx_node node,
                                                 const vx_reference *parameters,
                                                 vx_uint32 param_num,
                                                 const vx_tile_block_size_intel_t *current_tile_dimensions,
                                                 vx_tile_block_size_intel_t *updated_tile_dimensions)
{
   updated_tile_dimensions->width = current_tile_dimensions->width;
   updated_tile_dimensions->height = current_tile_dimensions->height;

   //this node outputs 64-pixels per loop iteration, so we want the output tile width to be divisible by 32
   if( (updated_tile_dimensions->width % 32) != 0 )
   {
     updated_tile_dimensions->width = ((updated_tile_dimensions->width/32)+1)*32;
   }

   return VX_SUCCESS;
}

vx_status VX_CALLBACK vxGainOffset12Validator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    if(num!=5)
    {
        return status;
    }

    vx_df_image format = 0;
    if(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format)) == VX_SUCCESS)
    {
        if ((format == VX_DF_IMAGE_U8))
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "vxGainOffset12 Validation failed: invalid input image format, it must be U8\n");
        }
    }

    vx_enum item_type;
    vxQueryArray((vx_array)parameters[2], VX_ARRAY_ITEMTYPE, &item_type, sizeof(item_type));
    if( item_type != VX_TYPE_FLOAT32 )
    {
       status = VX_ERROR_INVALID_VALUE;
       vxAddLogEntry((vx_reference)node, status, "vxGainOffset12 Validation failed: Gain array must be of type VX_TYPE_FLOAT32\n");
    }

    vxQueryArray((vx_array)parameters[3], VX_ARRAY_ITEMTYPE, &item_type, sizeof(item_type));
    if( item_type != VX_TYPE_FLOAT32 )
    {
       status = VX_ERROR_INVALID_VALUE;
       vxAddLogEntry((vx_reference)node, status, "vxGainOffset12 Validation failed: Offset array must be of type VX_TYPE_FLOAT32\n");
    }

    vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &item_type, sizeof(item_type));
    if( item_type != VX_TYPE_FLOAT32 )
    {
       status = VX_ERROR_INVALID_VALUE;
       vxAddLogEntry((vx_reference)node, status, "vxGainOffset12 Validation failed: AGC scalar must be of type VX_TYPE_FLOAT32\n");
    }

    vx_uint32 width = 0;
    status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &width, sizeof(width));
    if (width%48!=0)
    {
        status = VX_ERROR_INVALID_VALUE;
        vxAddLogEntry((vx_reference)node, status, "vxGainOffset12 Validation failed: input image width must be evenly divisible by 48 bytes\n");
    }

    vx_uint32 height = 0;
    status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height));

    //calculate the output width
    vx_uint32 inputbitsperline = width * 8;
    vx_uint32 outputwidth = inputbitsperline / 12;

    //output format will be 8bpp
    format = VX_DF_IMAGE_U8;

    status |= vxSetMetaFormatAttribute(metas[1], VX_IMAGE_WIDTH, &outputwidth, sizeof(width));
    status |= vxSetMetaFormatAttribute(metas[1], VX_IMAGE_HEIGHT, &height, sizeof(height));
    status |= vxSetMetaFormatAttribute(metas[1], VX_IMAGE_FORMAT, &format, sizeof(format));

    return status;
}

static vx_status vxGainOffset12Initialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{

   vx_image output = (vx_image)parameters[1];
   vx_int32 width;
   vxQueryImage(output, VX_IMAGE_WIDTH, &width, sizeof(width));

   //The operation we want to achieve looks like this:
   // output8[pixel]=  (input12[pixel] * gain[pixel] + offset[pixel]) * agc
   // Since AGC is a constant (per page), we can save on a multiply by
   //  precomputing dedicated gain & offset:
   // new_gain[pixel] = gain[pixel] * agc
   // new_offset[pixel] = offset[pixel] * agc
   //So, we want to have the engine reserve dedicated scratch memory which
   // we will compute the "new" gain & offset into within preprocess for each
   // page. The size is a function of the width, and we add some extra bytes
   // to guarantee 64-byte alignment.
   vx_size gain_bytes = width * sizeof(vx_float32);
   if( (gain_bytes % 64) != 0 )
   {
      gain_bytes = ((gain_bytes/64)+1)*64;
   }

   vx_size offset_bytes = width * sizeof(vx_float32);
   if( (offset_bytes % 64) != 0 )
   {
      offset_bytes = ((offset_bytes/64)+1)*64;
   }

   //(+64 here so that we can align the start ptr of scratch buffer if needed)
   vx_size scratch_size = gain_bytes + offset_bytes + 64;

   return vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_SIZE, &scratch_size, sizeof(scratch_size));
}


vx_status VX_API_CALL PublishGainOffset12Kernel(vx_context context)
{
    vx_status status = VX_SUCCESS;
    //create a kernel via the vxAddAdvancedTilingKernelIntel interface
    vx_kernel kernel = vxAddAdvancedTilingKernelIntel(context,
        (char*)VX_KERNEL_NAME_USER_GAINOFFSET12,
        VX_KERNEL_USER_GAINOFFSET12,
        vxGainOffset12Kernel,
        vxGainOffset12TileMapping,
        5,
        vxGainOffset12Validator,
        vxGainOffset12Initialize,
        NULL,
        vxGainOffset12PreProcess,
        NULL,
        vxGainOffset12SetTileDimensions,
        NULL);

    PROCESS_VX_STATUS(context, vxGetStatus((vx_reference)kernel));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxFinalizeKernel(kernel));

    if( VX_SUCCESS != status )
        vxRemoveKernel( kernel );

    return status;
}
