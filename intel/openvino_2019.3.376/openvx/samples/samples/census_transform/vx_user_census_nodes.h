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

#ifndef _VX_USER_CENSUS_H_
#define _VX_USER_CENSUS_H_

#include "VX/vx.h"
#include "stdio.h"


//! User kernel names.
#define VX_KERNEL_NAME_USER_CENSUSTRANSFORM "com.intel.sample.censustransform"
#define VX_KERNEL_NAME_USER_CENSUSTRANSFORMTILED "com.intel.sample.censustransformtiled"
#define VX_KERNEL_NAME_USER_CENSUSTRANSFORM_OPENCL "com.intel.sample.censustransform.opencl"


#define VX_LIBRARY_SAMPLE_CENSUS_TRANSFORM (0x2)

//! The list of kernels enum.
enum vx_kernel_intel_sample_census_transform_e {
    VX_KERNEL_USER_CENSUSTRANSFORM = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_CENSUS_TRANSFORM) + 0x0,
    VX_KERNEL_USER_CENSUSTRANSFORMTILED = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_CENSUS_TRANSFORM) + 0x1,
    VX_KERNEL_USER_CENSUSTRANSFORM_OPENCL = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_CENSUS_TRANSFORM) + 0x2,
};


#ifdef __cplusplus
extern "C" {
#endif

    //! Functions for user kernels publishing.
#if _WIN32
	__declspec(dllexport)
#endif
	vx_status VX_API_CALL vxPublishKernels(vx_context context);
    vx_status VX_API_CALL PublishCensusTransformKernel(vx_context context);
    vx_status VX_API_CALL PublishCensusTransformTiledKernel(vx_context context);
    vx_status VX_API_CALL PublishCensusTransformOpenCLKernel(vx_context context);

    //!**********************************************************************************************
    //! Function Name      :  vxCensusTransformNode
    //! Argument 1         :  The handle to the graph in which to instantiate the node   [IN]
    //! Argument 2         :  Input VX_DF_IMAGE_S16 image 		                         [IN]
    //! Argument 3         :  Output CT VX_DF_IMAGE_U8 image	                         [OUT]
    //! Returns            :  Status
    //! Description        :  Implementation of the census transform node which executes
    //!					   :  in the Graph to invoke the censusTransform kernel.
    //!                    :  This function performs the creation of the node and
    //!			           :  adds the created node to the graph. It retrieves the
    //!			           :  kernel that performs the censusTransform operation
    //!***********************************************************************************************
    vx_node vxCensusTransformNode(vx_graph graph, vx_image input, vx_image output);

    //!***********************************************************************************************
    //! Function Name      :  vxuCensusTransform
    //! Argument 1         :  The overall context of the implementation	                 [IN]
    //! Argument 2         :  Input VX_DF_IMAGE_S16 image  	                             [IN]
    //! Argument 3         :  Output CT VX_DF_IMAGE_U8 image	                         [OUT]
    //! Returns            :  Status
    //! Description        :  This function provides a immediate node mode
    //!					   :  implementation of the vxuCensusTransform function
    //!***********************************************************************************************
    vx_status vxuCensusTransform(vx_context context, vx_image input, vx_image output);

    //!***********************************************************************************************
    //! Function Name      :  vxCensusTransformTiledNode
    //! Argument 1         :  The handle to the graph in which to instantiate the node   [IN]
    //! Argument 2         :  Input VX_DF_IMAGE_S16 image 		                         [IN]
    //! Argument 3         :  Output CT VX_DF_IMAGE_U8 image	                         [OUT]
    //! Returns            :  Status
    //! Description        :  Implementation of the census transform tiled node which executes
    //!					   :  in the Graph to invoke the censusTransform kernel.
    //!                    :  This function performs the creation of the node and
    //!			           :  adds the created node to the graph. It retrieves the
    //!			           :  kernel that performs the censusTransform operation
    //!************************************************************************************************
    vx_node vxCensusTransformTiledNode(vx_graph graph, vx_image input, vx_image output);

    //!************************************************************************************************
    //! Function Name      :  vxuCensusTransformTiled
    //! Argument 1         :  The overall context of the implementation	                 [IN]
    //! Argument 2         :  Input VX_DF_IMAGE_S16 image  	                             [IN]
    //! Argument 3         :  Output CT VX_DF_IMAGE_U8 image	                         [OUT]
    //! Returns            :  Status
    //! Description        :  This function provides a immediate node mode
    //!					   :  implementation of the vxuCensusTransformTiled function
    //!************************************************************************************************
    vx_status vxuCensusTransformTiled(vx_context context, vx_image input, vx_image output);

#ifdef __cplusplus
}
#endif

    //!**********************************************************************************************
    //! Function Name      :  vxCensusTransformOpenCLNode
    //! Argument 1         :  The handle to the graph in which to instantiate the node   [IN]
    //! Argument 2         :  Input VX_DF_IMAGE_S16 image 		                         [IN]
    //! Argument 3         :  Output CT VX_DF_IMAGE_U8 image	                         [OUT]
    //! Returns            :  Status
    //! Description        :  Implementation of the census transform node which executes
    //!					   :  in the Graph to invoke the OpenCL censusTransform kernel.
    //!***********************************************************************************************
    vx_node vxCensusTransformOpenCLNode(vx_graph graph, vx_image input, vx_image output);

#ifdef __cplusplus
extern "C" {
#endif
    //!************************************************************************************************
    //! Function Name      :  censustransform
    //! Argument 1         :  Pointer to source data tile	                            [IN]
    //! Argument 2         :  Source data tile stride  	                                [IN]
    //! Argument 3         :  Pointer to destination data tile                          [IN/OUT]
    //! Argument 4         :  Destination data tile stride  	                        [IN]
    //! Argument 5         :  Deestination data tile width                              [IN]
    //! Argument 6         :  Destination data tile height  	                        [IN]
    //! Returns            :  Status
    //! Description        :  SSE optimized implementation of Census Transform
    //!************************************************************************************************
    vx_status censustransform(vx_int16 *pSrc,
        vx_int32 srcStride,
        vx_uint8 *pDst,
        vx_int32 dstStride,
        vx_uint32 dstWidth,
        vx_uint32 dstHeight);

#ifdef __cplusplus
}
#endif

#endif /* _VX_USER_CENSUS_H_ */

