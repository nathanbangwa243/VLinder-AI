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
#include <malloc.h>
#include <immintrin.h>
#include "ipa.h"

#include "vx_user_pipeline_nodes.h"


static void *my_ipa_malloc(void *opaque, size_t size)
{
   (void)opaque;
   return malloc(size);
}

static void *my_ipa_realloc(void *opaque, void *ptr, size_t newsize)
{
   (void)opaque;
   return realloc(ptr, newsize);
}

static void my_ipa_free(void *opaque, void *ptr)
{
   (void)opaque;
   free(ptr);
}

static ipa_allocators my_ipa_allocators =
{
   &my_ipa_malloc,
   &my_ipa_realloc,
   &my_ipa_free
};

struct IPAHalftoneLocal
{
   ipa_halftone *ht = 0;
   ipa_context *ipacontext = 0;

};

struct HTCallbackArgs
{
   vx_uint8 *pDstLine;
   vx_uint32 width;
};

static void ht_callback(ipa_halftone_data_t *data, void *args)
{
   HTCallbackArgs *pArg = (HTCallbackArgs *)args;

   //unpack, and invert the output which IPAContone produced.
   vx_int32 unrolled_iterations = pArg->width / 128;
   vx_int32 remaining = (pArg->width % 128)/8;

   const __m128i vunpacklower = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
   const __m128i vmask = _mm_setr_epi8(0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1, 0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1);

   __m128i *inputpacked128 = (__m128i *)(data->data);
   __m128i *outputunpacked128 = (__m128i *)(pArg->pDstLine);

   for( vx_int32 x = 0; x < unrolled_iterations; x++)
   {
      //read 128 bits from the input
      __m128i inputpacked = _mm_load_si128( inputpacked128++ );

#pragma unroll(8)
      for( vx_int32 i = 0; i < 8; i++)
      {
         __m128i unpacked = _mm_shuffle_epi8(inputpacked, vunpacklower);
         unpacked = _mm_and_si128(unpacked, vmask);
         unpacked = _mm_cmpeq_epi8(unpacked, _mm_setzero_si128());
         _mm_store_si128(outputunpacked128++, unpacked);
         inputpacked = _mm_srli_si128(inputpacked, 2);
      }

      vx_uint8 *pInputRemaining = (vx_uint8 *)inputpacked128;
      vx_uint8 *pOutputRemaining = (vx_uint8 *)outputunpacked128;
      //manually unpack each byte for the remaining (up to 15) input bytes
      for (int fs = 0; fs < remaining; fs++)
      {
         vx_uint8 input = pInputRemaining[fs];

         //write 8 output bytes per each input byte
         *pOutputRemaining++ = (input & 0x80) ? 255 : 0;
         *pOutputRemaining++ = (input & 0x40) ? 255 : 0;
         *pOutputRemaining++ = (input & 0x20) ? 255 : 0;
         *pOutputRemaining++ = (input & 0x10) ? 255 : 0;
         *pOutputRemaining++ = (input & 0x8) ? 255 : 0;
         *pOutputRemaining++ = (input & 0x4) ? 255 : 0;
         *pOutputRemaining++ = (input & 0x2) ? 255 : 0;
         *pOutputRemaining++ = (input & 0x1) ? 255 : 0;
      }

   }

}

static vx_status vxIPAHalftonePreProcess(vx_node node,
                                         const vx_reference *parameters,
                                         vx_uint32 num_parameters,
                                         void * tile_memory[],
                                         vx_uint32 num_tile_memory_elements,
                                         vx_size tile_memory_size)
{
   IPAHalftoneLocal *pLocal = 0;
   if( vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &pLocal, sizeof(pLocal)) != VX_SUCCESS )
      return VX_FAILURE;
   if( !pLocal )
      return VX_FAILURE;

   //Reset the halftone instance to a initial state.
   ipa_halftone_reset(pLocal->ipacontext, pLocal->ht);

   return VX_SUCCESS;
}

static vx_status vxIPAHalftoneKernel(vx_node node,
                                           void *   parameters[],
                                           vx_uint32 num,
                                           void *   tile_memory,
                                           vx_size tile_memory_size)
{
   IPAHalftoneLocal *pLocal = 0;
   if( vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &pLocal, sizeof(pLocal)) != VX_SUCCESS )
      return VX_FAILURE;
   if( !pLocal )
      return VX_FAILURE;

   vx_tile_intel_t *pInTile = (vx_tile_intel_t *)parameters[0];
   vx_tile_intel_t *pOutTile = (vx_tile_intel_t *)parameters[5];

   vx_uint8 *pSrc = pInTile->base[0];
   vx_uint8 *pDst = pOutTile->base[0];

   vx_int32 srcStep = pInTile->addr[0].stride_y;
   vx_int32 dstStep = pOutTile->addr[0].stride_y;

   vx_uint32 tile_width = pOutTile->addr[0].dim_x;
   vx_uint32 tile_height = pOutTile->addr[0].dim_y;

   HTCallbackArgs callbackArgs = {pDst, tile_width};

   for( vx_int32 y = 0; y < tile_height; y++)
   {
      if( ipa_halftone_process_planar(pLocal->ht,
                                      NULL,
                                      (const unsigned char **)&pSrc,
                                      ht_callback,
                                      &callbackArgs) )
      {
         vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_VALUE, "ipa_halftone_process_planar failed\n");
         return VX_FAILURE;
      }

      pSrc += srcStep;
      callbackArgs.pDstLine += dstStep;
   }

   return VX_SUCCESS;
}

vx_status VX_CALLBACK vxIPAHalftoneValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    if(num!=6)
    {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    vx_df_image format = 0;
    vx_uint32 width = 0;
    vx_uint32 height = 0;
    if(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format)) == VX_SUCCESS)
    {
        if (format != VX_DF_IMAGE_U8)
        {
            vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_VALUE, "IPAHalftone Input Validation failed: invalid input image format, it must be VX_DF_IMAGE_U8\n");
            return VX_ERROR_INVALID_VALUE;
        }
    }
    else
    {
       return VX_ERROR_INVALID_PARAMETERS;
    }

    vx_enum array_type;
    if( vxQueryArray((vx_array)parameters[1], VX_ARRAY_ITEMTYPE, &array_type, sizeof(array_type)) == VX_SUCCESS )
    {
        if( array_type != VX_TYPE_UINT8 )
        {
           vxAddLogEntry((vx_reference)node, VX_ERROR_INVALID_VALUE, "IPAHalftone Input Validation failed: invalid vx_array format for screendata. It must be VX_TYPE_UINT8\n");
           return VX_ERROR_INVALID_VALUE;
        }
    }
    else
    {
       return VX_ERROR_INVALID_PARAMETERS;
    }

    PROCESS_VX_STATUS(node, vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    PROCESS_VX_STATUS(node, vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &width, sizeof(width)));

    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[5], VX_IMAGE_WIDTH, &width, sizeof(width)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[5], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[5], VX_IMAGE_FORMAT, &format, sizeof(format)));

    return VX_SUCCESS;
}

static void CleanupLocal(IPAHalftoneLocal *pLocal)
{
   if( pLocal )
   {
      if( pLocal->ht )
        ipa_halftone_fin(pLocal->ht, NULL);

      if( pLocal->ipacontext )
        ipa_fin(pLocal->ipacontext, NULL);

     delete pLocal;
   }
}

vx_status vxIPAHalftoneInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
  vx_image input = (vx_image)parameters[0];
  vx_int32 width, height;
  vxQueryImage(input, VX_IMAGE_WIDTH, &width, sizeof(width));
  vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(height));

  IPAHalftoneLocal *pLocal = new IPAHalftoneLocal();
  pLocal->ipacontext = ipa_init(&my_ipa_allocators, NULL);
  if( !pLocal->ipacontext )
  {
     vxAddLogEntry((vx_reference)node, VX_FAILURE, "ERROR in ipa_init\n");
     CleanupLocal(pLocal);
     return VX_FAILURE;

  }

  //Check to make sure that the IPA context initialized itself with support for SSE4.1
  if (!ipa_cpu_supports_sse_4_1(pLocal->ipacontext))
  {
    vxAddLogEntry((vx_reference)node, VX_FAILURE, "THIS TARGET DOES NOT SUPPORT SSE 4.1\n");
    CleanupLocal(pLocal);
    return VX_FAILURE;
  }

  //1:1 transform. We aren't scaling, rotating, mirroring, etc.
  ipa_matrix mat;
  mat.xx = 1.0f;
  mat.xy = 0;
  mat.yx = 0;
  mat.yy = 1.0f;
  mat.tx = 0;
  mat.ty = 0;

  //Create an instance of IPAHalftone
  pLocal->ht = ipa_halftone_init(pLocal->ipacontext,
                                 NULL, //opaque pointer to be passed to ipa_malloc callbacks
                                 width, height,  //image width / height
                                 &mat, //transformation matrix
                                 1, //number of planes
                                 NULL, //cache
                                 0,  //clip_x
                                 0,  //clip_y
                                 width, //clip_width
                                 height, //clip_height
                                 0);

  if( !pLocal->ht )
  {
     vxAddLogEntry((vx_reference)node, VX_FAILURE, "ipa_halftone_init failed\n");
     CleanupLocal(pLocal);
     return VX_FAILURE;
  }

  //Obtain the screen data from the user parameters
  vx_array screendata_array = (vx_array)parameters[1];
  vx_uint8 *pScreenData = 0;
  vx_size stride = 0;
  vx_size num_items = 0;

  if( vxQueryArray(screendata_array, VX_ARRAY_CAPACITY, &num_items, sizeof(num_items)) != VX_SUCCESS )
  {
     CleanupLocal(pLocal);
     return VX_FAILURE;
  }

  vx_map_id map_id;
  if( vxMapArrayRange(screendata_array, 0, num_items, &map_id, &stride, reinterpret_cast<void**>(&pScreenData), VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0) != VX_SUCCESS)
  {
    CleanupLocal(pLocal);
    return VX_FAILURE;
  }

  vx_scalar sScreenWidth = (vx_scalar)parameters[2];
  vx_int32 screenWidth;
  if( vxCopyScalar(sScreenWidth, &screenWidth, VX_READ_ONLY, VX_MEMORY_TYPE_HOST) != VX_SUCCESS )
  {
     CleanupLocal(pLocal);
     return VX_FAILURE;
  }

  vx_scalar sScreenHeight = (vx_scalar)parameters[3];
  vx_int32 screenHeight;
  if( vxCopyScalar(sScreenHeight, &screenHeight, VX_READ_ONLY, VX_MEMORY_TYPE_HOST) != VX_SUCCESS )
  {
     CleanupLocal(pLocal);
     return VX_FAILURE;
  }

  vx_scalar sScreenShift = (vx_scalar)parameters[4];
  vx_int32 screenShift;
  if( vxCopyScalar(sScreenShift, &screenShift, VX_READ_ONLY, VX_MEMORY_TYPE_HOST) != VX_SUCCESS )
  {
     CleanupLocal(pLocal);
     return VX_FAILURE;
  }

  if( num_items != screenWidth*screenHeight )
  {
     vxAddLogEntry((vx_reference)node, VX_FAILURE, "The number of screendata array items must be equal to screenwidth * screenheight\n");
     CleanupLocal(pLocal);
     return VX_FAILURE;
  }

  //Add the screen to our IPAHalftone instance
  if( ipa_halftone_add_screen(pLocal->ipacontext, NULL,
                                    pLocal->ht,
                                    0, /* Not inverted */
                                    screenWidth,
                                    screenHeight,
                                    screenShift,
                                    0,
                                    pScreenData) )
  {
     vxAddLogEntry((vx_reference)node, VX_FAILURE, "ipa_halftone_add_screen failed\n");
     CleanupLocal(pLocal);
     return VX_FAILURE;
  }

  vxUnmapArrayRange(screendata_array, map_id);

  //if a previous vxVerifyGraph has set a local IPAHalftoneLocal for this node, free
  // it before allocating and setting a new one.
  IPAHalftoneLocal *pCurrent = 0;
  vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &pCurrent, sizeof(pCurrent));
  if( pCurrent ) CleanupLocal(pCurrent);

  //set the 'local data ptr' for this node to our IPAHalftoneLocal ptr
  vx_status status = vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &pLocal, sizeof(pLocal));
  if( status != VX_SUCCESS )
  {
     CleanupLocal(pCurrent);
     return status;
  }

  return VX_SUCCESS;

}

vx_status vxIPAHalftoneDeInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
   IPAHalftoneLocal *pCurrent = 0;
   vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &pCurrent, sizeof(pCurrent));
   if( pCurrent )
      CleanupLocal(pCurrent);

   pCurrent = 0;
   return vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &pCurrent, sizeof(pCurrent));
}

vx_status vxIPAHalftoneSetTileDimensions(vx_node node,
                                         const vx_reference *parameters,
                                         vx_uint32 param_num,
                                         const vx_tile_block_size_intel_t *current_tile_dimensions,
                                         vx_tile_block_size_intel_t *updated_tile_dimensions)
{
   //Since the underlying IPAHalftone function works with entire lines, we need to force the output tile
   // size to the width of the image.
   vx_uint32 image_width = 0;
   PROCESS_VX_STATUS(node, vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &image_width, sizeof(image_width)));
   updated_tile_dimensions->width =image_width;

   //no need to change the tile height, so keep it at the "current" tile dimension height
   updated_tile_dimensions->height = current_tile_dimensions->height;

   return VX_SUCCESS;
}


vx_status VX_API_CALL PublishIPAHalftoneKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;
    //create a kernel via the vxAddAdvancedTilingKernelIntel interface
    vx_kernel kernel = vxAddAdvancedTilingKernelIntel(context,
        (char*)VX_KERNEL_NAME_USER_IPAHALFTONE,
        VX_KERNEL_USER_IPAHALFTONE,
        vxIPAHalftoneKernel,
        NULL,
        6,
        vxIPAHalftoneValidator,
        vxIPAHalftoneInitialize,
        vxIPAHalftoneDeInitialize,
        vxIPAHalftonePreProcess,
        NULL,
        vxIPAHalftoneSetTileDimensions,
        NULL);

    PROCESS_VX_STATUS(context, vxGetStatus((vx_reference)kernel));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 5, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));

    //An underlying IPAHalftone instance has it's own state, and needs to be fed input image lines from top to bottom.
    // We need to give a hint to the OpenVX scheduler that tiles for this kernel need to be scheduled in-order from top to
    // bottom, and to not allow any tiles to be scheduled concurrently. A side effect of this is that only a single thread
    // will only ever be processing a tile (within vxIPAHalftoneKernel()) at any given time.
    vx_serial_type_intel_e serial_type = VX_SERIAL_LEFTTOP_TO_RIGHTBOTTOM_INTEL;
    SAFE_VX_CALL(status, context, vxSetKernelAttribute(kernel, VX_KERNEL_SERIAL_TYPE_INTEL, &serial_type, sizeof(serial_type)));

    SAFE_VX_CALL(status, context, vxFinalizeKernel(kernel));

    if( VX_SUCCESS != status )
        vxRemoveKernel( kernel );

    return status;
}
