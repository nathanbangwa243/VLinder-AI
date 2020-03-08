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

#include "vx_user_census_nodes.h"
#include <stdio.h>

#include <immintrin.h>

#include <VX/vx_intel_volatile.h>


// Reuse validator from a regular user kernel implementation (see vx_censustransform_module.c)
extern vx_status VX_CALLBACK CensusTransformValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);


//! An internal definition of the order of the parameters to the function
//! This list must match the parameter list in the function and in the
//! publish kernel list.
typedef enum _census_transformtiled_params_e {
    CENSUSTRANSFORMTILED_PARAM_INPUT = 0,
    CENSUSTRANSFORMTILED_PARAM_OUTPUT
} census_transformtiled_params_e;


//!*************************************************************************************************************
//! Function Name    :  CensusTransformTileMapping
//! Argument 1       :  Handle to the node                                                              [IN]
//! Argument 2       :  Input parameters                                                                [IN]
//! Argument 3       :  Destination tile attributes                                                     [IN]
//! Argument 4       :  Source tile attributes required to produce destination tile                     [OUT]
//! Argument 5       :  Index of input image parameter the engine is requesting to fill the srcRect for [IN]
//! Returns          :  Status
//! Description      :  For each given output tile, it is required that the
//!                  :  kernel creator describe the input tile dependencies through the tile mapping function.
//!*************************************************************************************************************
static vx_status CensusTransformTileMapping (vx_node node,
                                             vx_reference parameters[],
                                             const vx_tile_t_attributes_intel_t* dstRectIn,
                                             vx_tile_t_attributes_intel_t* srcRectOut,
                                             vx_uint32 param_num)
{
    //To produce a dst tile, a source tile which is 2 columns / 2 lines
    // larger is required.
    // dst image size is srcWidth-2, srcHeight�2, so no need to adjust
    // x, y because of smaller dst tile size
    srcRectOut->x = dstRectIn->x;
    srcRectOut->y = dstRectIn->y;
    srcRectOut->tile_block.width = dstRectIn->tile_block.width + 2;
    srcRectOut->tile_block.height = dstRectIn->tile_block.height + 2;
    return VX_SUCCESS;
}


//!****************************************************************************************************
//! Function Name        :  CensusTransformTilingKernel
//! Argument 1           :  Handle to the node			        [IN]
//! Argument 2           :  Input parameters		            [IN]
//! Argument 3           :  Number of parameters				[IN]
//! Argument 4           :  Tile local memory 	                [IN]
//! Argument 5           :  Tile local memory size				[IN]
//! Returns              :  Status
//! Description          :  The private kernel function for CensusTransform custom tiling Kernel
//!                      :  This function will be called for each tile contained within
//!                      :  the output image, during vxProcessGraph.
//!                      :  Pixels cooresponding to those of the output tile are required
//!                      :  to be produced within this function.
//!****************************************************************************************************
static vx_status CensusTransformTilingKernel(vx_node node, //The handle to the node that contains this kernel.
                                             void *   parameters[], //The array abstract pointers to parameters.
                                             vx_uint32 num, //The number of parameters.
                                             void *   tile_memory, //The local tile memory pointer if requested, otherwise NULL.
                                             vx_size tile_memory_size) //The size of the local tile memory, if not requested, 0.

{
    vx_tile_intel_t *pInTile = (vx_tile_intel_t *)parameters[CENSUSTRANSFORMTILED_PARAM_INPUT];
    vx_tile_intel_t *pOutTile = (vx_tile_intel_t *)parameters[CENSUSTRANSFORMTILED_PARAM_OUTPUT];

    vx_int16 *pSrc = (vx_int16 *)pInTile->base[0];
    vx_uint8 *pDst = pOutTile->base[0];

    vx_int32 srcStride = pInTile->addr[0].stride_y;
    vx_int32 dstStride = pOutTile->addr[0].stride_y;

    vx_uint32 dst_tile_width = pOutTile->addr[0].dim_x;
    vx_uint32 dst_tile_height = pOutTile->addr[0].dim_y;

    return censustransform(pSrc, srcStride, pDst, dstStride, dst_tile_width, dst_tile_height);

}


//!***********************************************************************
//! Function Name        :  CensusTransformTiledInitialize
//! Argument 1           :  Handle to the node			        [IN]
//! Argument 2           :  Input parameters		            [IN]
//! Argument 3           :  Number of parameters				[IN]
//! Returns              :  Status
//! Description          :  An initializer function for CensusTransform
//!					     :  node handle
//!***********************************************************************
vx_status VX_CALLBACK CensusTransformTiledInitialize(vx_node node, const vx_reference *parameters,
                                                     vx_uint32 num)
{
    /* CensusTransformTiledInitialize requires no initialization of memory or resources */
    return VX_SUCCESS;
}


//!***********************************************************************
//! Function Name        :  CensusTransformTiledDeinitialize
//! Argument 1           :  Handle to the node		            [IN]
//! Argument 2           :  Input parameters		            [IN]
//! Argument 3           :  Number of parameters				[IN]
//! Returns              :  Status
//! Description          :  A deinitializer function
//!***********************************************************************
vx_status VX_CALLBACK CensusTransformTiledDeinitialize(vx_node node, const vx_reference *parameters,
                                                       vx_uint32 num)
{
    /* CensusTransformTiledDeinitialize requires no de-initialization of memory or resources */
    return VX_SUCCESS;
}


//!***********************************************************************
//! Function Name        :  CensusTransformTiledPreProcess
//! Argument 1           :  Handle to the node		            [IN]
//! Argument 2           :  Input parameters		            [IN]
//! Argument 3           :  Number of parameters				[IN]
//! Returns              :  Status
//! Description          :  A preprocess function
//!***********************************************************************
vx_status VX_CALLBACK CensusTransformTiledPreProcess(vx_node node,
                                                     const vx_reference *parameters,
                                                     vx_uint32 num_parameters,
                                                     void *   tile_memory[],
                                                     vx_uint32 num_tile_memory_elements,
                                                     vx_size tile_memory_size)
{
    /* CensusTransformTiledPreProcess requires no pre-processing. Stub implementation. */
    return VX_SUCCESS;
}


//!***********************************************************************
//! Function Name        :  CensusTransformTiledPostProcess
//! Argument 1           :  Handle to the node		            [IN]
//! Argument 2           :  Input parameters		            [IN]
//! Argument 3           :  Number of parameters				[IN]
//! Returns              :  Status
//! Description          :  A preprocess function
//!***********************************************************************
vx_status VX_CALLBACK CensusTransformTiledPostProcess(vx_node node,
                                                      const vx_reference *parameters,
                                                      vx_uint32 num_parameters,
                                                      void *   tile_memory[],
                                                      vx_uint32 num_tile_memory_elements,
                                                      vx_size tile_memory_size)
{
    /* CensusTransformTiledPostProcess requires no post-processing. Stub implementation. */
    return VX_SUCCESS;
}


//!***********************************************************************
//! Function Name        :  CensusTransformTiledSetTileDimensions
//! Argument 1           :  Handle to the node		            [IN]
//! Argument 2           :  Input parameters		            [IN]
//! Argument 3           :  Number of parameters				[IN]
//! Argument 3           :  Current tile dimensions				[IN]
//! Argument 3           :  New tile diemnsions 				[OUT]
//! Returns              :  Status
//! Description          :  A preprocess function
//!***********************************************************************
vx_status VX_CALLBACK CensusTransformTiledSetTileDimensions(vx_node node,
                                                            const vx_reference *parameters,
                                                            vx_uint32 num,
                                                            const vx_tile_block_size_intel_t *current_tile_dimensions,
                                                            vx_tile_block_size_intel_t *updated_tile_dimensions)
{

    /* No need specific tile dimensions for CT */
    /* Stub implementation retrieves current tile dimension and set is target */
    updated_tile_dimensions->width = current_tile_dimensions->width;
    updated_tile_dimensions->height = current_tile_dimensions->height;

    return VX_SUCCESS;
}


//!**************************************************************************
//! Function Name        :  PublishCensusTransformTiledKernel
//! Argument 1           :  Context		                                [IN]
//! Returns              :  Status
//! Description          :  This function publishes the user defined kernels
//!**************************************************************************
vx_status VX_API_CALL PublishCensusTransformTiledKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;

    vx_kernel kernel = vxAddAdvancedTilingKernelIntel(context,
        VX_KERNEL_NAME_USER_CENSUSTRANSFORMTILED, //The string to use to match the kernel.
        VX_KERNEL_USER_CENSUSTRANSFORMTILED, //The enumerated value of the kernel to be used by clients.
        CensusTransformTilingKernel, //The process-local function pointer to be invoked.
        CensusTransformTileMapping,  //The tile mapping function pointer.
        2, //The number of parameters for this kernel.
        CensusTransformValidator, //The pointer to callback function, which validates the input and output parameters to this kernel.
        CensusTransformTiledInitialize, //The kernel initialization function.
        CensusTransformTiledDeinitialize, //The kernel de-initialization function.
        CensusTransformTiledPreProcess, //The optional pre-process function. No need pre-process for CT. Stub implementation.
        CensusTransformTiledPostProcess, //The optional post-process function. No need post-process for CT. Stub implementation.
        CensusTransformTiledSetTileDimensions, //The optional custom tile dimensions setup function. No need custom tile dimensions for CT. Stub implementation.
        NULL); //The optional 'tile dimensions initialize' function. No need for CT, pass NULL

    if (kernel)
    {
        status |= vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
        if (status != VX_SUCCESS) goto exit;

        status |= vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
        if (status != VX_SUCCESS) goto exit;

        status |= vxFinalizeKernel(kernel);
        if (status != VX_SUCCESS) goto exit;
    }
exit:
    if (status != VX_SUCCESS) {
        vxRemoveKernel(kernel);
        vxAddLogEntry((vx_reference)context, status, "CT Tiled kernel publish failed\n");
    }
    return status;
}
