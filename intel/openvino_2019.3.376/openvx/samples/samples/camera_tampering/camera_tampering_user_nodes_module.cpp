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


#include <cassert>

#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/perfprof.hpp>
#include <stdarg.h>
#include <iostream>
#include "camera_tampering_core.h"
#include "camera_tampering_user_nodes_lib.h"

// ------------------------------------------------------------------------------------------------
// Note, here RETURN_VX_STATUS macro is used from common sample infrastructure for easier debugging.
// This macro will terminate the program in case when returned status is not VX_SUCCESS.
// It is not a valid behaviour for an input/output validators, kernel functions or vxPublishKernels.
// Instead the status should be reported as a return value.
// Use program termination (like here) in educational and debugging purposes only.
// ------------------------------------------------------------------------------------------------

typedef enum _connected_component_labeling_params_e {
    CONNECTED_COMPONENT_LABELING_PARAM_INPUT = 0,
    CONNECTED_COMPONENT_LABELING_PARAM_THRESHOLD,
    CONNECTED_COMPONENT_LABELING_PARAM_OUTPUT,
    CONNECTED_COMPONENT_LABELING_PARAM_RECTANGLES,
} connected_component_labeling_params_e;

namespace
{

// --------------------------------------------------------------------------
// Next several functions help to verify node parameters
// --------------------------------------------------------------------------

/*****************************************************************************
    Next 5 functions define Connected Component Labeling node behaviour
*****************************************************************************/

vx_status VX_CALLBACK ConnectedComponentLabelingValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;
    vx_df_image format = 0;

    if(num != 4)
    {
        IntelVXSample::logger(1) << "ConnectedComponentLabeling need 4 parameters" << std::endl;
        return VX_ERROR_INVALID_PARAMETERS;
    }

    status |= vxQueryImage((vx_image)parameters[CONNECTED_COMPONENT_LABELING_PARAM_INPUT], VX_IMAGE_FORMAT, &format, sizeof(format));
    if ((status != VX_SUCCESS) || (format != VX_DF_IMAGE_U8))
    {
        IntelVXSample::logger(1) << "The parameter input of ConnectedComponentLabeling validate failed" << std::endl;
        return VX_ERROR_INVALID_PARAMETERS;
    }

    vx_enum type = 0;
    status |= vxQueryScalar((vx_scalar)parameters[CONNECTED_COMPONENT_LABELING_PARAM_THRESHOLD], VX_SCALAR_TYPE, &type, sizeof(type));
    if ((status != VX_SUCCESS) || (type != VX_TYPE_UINT32))
    {
        IntelVXSample::logger(1) << "The parameter threshold of ConnectedComponentLabeling validate failed" << std::endl;
        return VX_ERROR_INVALID_PARAMETERS;
    }

    vx_uint32 width = 0;
    vx_uint32 height = 0;

    vxQueryImage((vx_image)parameters[CONNECTED_COMPONENT_LABELING_PARAM_INPUT], VX_IMAGE_WIDTH, &width,  sizeof(width));
    vxQueryImage((vx_image)parameters[CONNECTED_COMPONENT_LABELING_PARAM_INPUT], VX_IMAGE_HEIGHT, &height, sizeof(height));
    format = VX_DF_IMAGE_U32;

    status |= vxSetMetaFormatAttribute(metas[CONNECTED_COMPONENT_LABELING_PARAM_OUTPUT], VX_IMAGE_WIDTH, &width, sizeof(width));
    status |= vxSetMetaFormatAttribute(metas[CONNECTED_COMPONENT_LABELING_PARAM_OUTPUT], VX_IMAGE_HEIGHT, &height, sizeof(height));
    status |= vxSetMetaFormatAttribute(metas[CONNECTED_COMPONENT_LABELING_PARAM_OUTPUT], VX_IMAGE_FORMAT, &format, sizeof(format));
    if(status != VX_SUCCESS)
    {
        IntelVXSample::logger(1) << "The parameter output of ConnectedComponentLabeling validate failed" << std::endl;
        return VX_ERROR_INVALID_PARAMETERS;
    }

    vx_enum itemType = VX_TYPE_RECTANGLE;
    vx_size capacity = 4096;

    status |= vxSetMetaFormatAttribute(metas[CONNECTED_COMPONENT_LABELING_PARAM_RECTANGLES], VX_ARRAY_ITEMTYPE, &itemType, sizeof(itemType));
    status |= vxSetMetaFormatAttribute(metas[CONNECTED_COMPONENT_LABELING_PARAM_RECTANGLES], VX_ARRAY_CAPACITY, &capacity, sizeof(capacity));  // TODO: need to check if this parameter is properly set
    if(status != VX_SUCCESS)
    {
        IntelVXSample::logger(1) << "The parameter rectangles of ConnectedComponentLabeling validate failed" << std::endl;
        return VX_ERROR_INVALID_PARAMETERS;
    }

    return VX_SUCCESS;
}

/// This is a kernel function that is called when the node is executed in the graph
vx_status VX_CALLBACK ConnectedComponentLabelingKernel(vx_node node, const vx_reference* parameters, vx_uint32 num)
{
    IntelVXSample::logger(2) << "Start ConnectedComponentLabelingKernel" << std::endl;
    vx_status status = VX_ERROR_INVALID_PARAMETERS;

    if (num == 4)
    {
        vx_image                            input       = (vx_image)parameters[CONNECTED_COMPONENT_LABELING_PARAM_INPUT];
        vx_scalar                           threshold   = (vx_scalar)parameters[CONNECTED_COMPONENT_LABELING_PARAM_THRESHOLD];
        vx_image                            output      = (vx_image)parameters[CONNECTED_COMPONENT_LABELING_PARAM_OUTPUT];
        vx_array                            rectangles  = (vx_array)parameters[CONNECTED_COMPONENT_LABELING_PARAM_RECTANGLES];
        vx_uint32                           width = 0, height = 0, srcImgStep = 0, dstImgStep = 0;
        vx_uint8                            *pSrc = NULL;
        vx_uint32                           *pDst = NULL;
        ConnectedComponentLabelingConfig    config;
        ConnectedComponentLabelingClass     cConnectedComponentLabeling;    // Connected component labeling object
        vx_rectangle_t                      objectItems[MAXIMUM_RECTANGLE_NUMBER];
        vx_int32                            nObjects    = 0;
        vx_uint32                           nThreshold  = 0;

        vxQueryImage(input, VX_IMAGE_WIDTH, &width,  sizeof(width));
        vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(height));

        void *base;
        vx_imagepatch_addressing_t src_addr, dst_addr;
        vx_rectangle_t rect     = {0u, 0u, (unsigned int)width, (unsigned int)height};

        // Get input buffer parameters: step & address
        base = nullptr;
        vx_map_id map_id_input;
        status = vxMapImagePatch(input, &rect, 0, &map_id_input, &src_addr, &base, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);
        if (status != VX_SUCCESS)
        {
            std::cerr << "[ ERROR ] Access image patch failed with status " << status << "\n";
            return status;
        }

        // Set input buffer parameters
        srcImgStep  = src_addr.stride_y;
        pSrc        = (vx_uint8 *)base;

        status = vxUnmapImagePatch(input, map_id_input);
        if (status != VX_SUCCESS)
        {
            std::cerr << "[ ERROR ] Commit image patch failed with status " << status << "\n";
            return status;
        }

        // Get output buffer parameters: step & address
        base = nullptr;
        vx_map_id map_id_output;
        status = vxMapImagePatch(output, &rect, 0, &map_id_output, &dst_addr, &base, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);
        if (status != VX_SUCCESS)
        {
            std::cerr << "[ ERROR ] Access image patch failed with status " << status << "\n";
            return status;
        }

        // Set output buffer parameters
        dstImgStep  = dst_addr.stride_y;
        pDst        = (vx_uint32 *)base;

        status = vxUnmapImagePatch(output, map_id_output);
        if (status != VX_SUCCESS)
        {
            std::cerr << "[ ERROR ] Commit image patch failed with status " << status << "\n";
            return status;
        }
        vxCopyScalar(threshold, &nThreshold, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        config.threshold = nThreshold;

        cConnectedComponentLabeling.setConfig(&config);
        cConnectedComponentLabeling.init(width, height, srcImgStep, dstImgStep);
        nObjects = cConnectedComponentLabeling.Do(pSrc, pDst, objectItems, MAXIMUM_RECTANGLE_NUMBER);

        // Set vx_array size as 0
        vxTruncateArray(rectangles, 0);

        if ((nObjects > 0) && (nObjects <= MAXIMUM_RECTANGLE_NUMBER))
        {
            vxAddArrayItems(rectangles, nObjects, objectItems, 0);
        }

        status = VX_SUCCESS;
    }

    IntelVXSample::logger(2) << "End ConnectedComponentLabelingKernel" << std::endl;

    return status;
}

/// This function is called once when node instance is initialized in a graph and may contain appropriate one-time initialization
vx_status VX_CALLBACK ConnectedComponentLabelingInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    return VX_SUCCESS;
}

/// This function is called when node instance is destroyed from a graph
vx_status VX_CALLBACK ConnectedComponentLabelingDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    return VX_SUCCESS;
}


/**
 * Input validator is called for each input parameter for a node and should check parameter attributes
 */

vx_status VX_CALLBACK count_non_zero_input_validator( vx_node   node,
                                                   vx_uint32 index )
{
    vx_reference ref       = NULL;
    vx_parameter parameter = vxGetParameterByIndex( node, index );
    CHECK_VX_STATUS( vxQueryParameter( parameter, VX_PARAMETER_ATTRIBUTE_REF, &ref,
									sizeof( ref ) ) );
    CHECK_VX_STATUS( vxReleaseParameter( &parameter ) );
    CHECK_VX_OBJ( ref );

    if( index == 0 )
    {
        // parameter #0 -- input image of format VX_DF_IMAGE_U8
        vx_df_image format = VX_DF_IMAGE_VIRT;
        CHECK_VX_STATUS( vxQueryImage( ( vx_image )ref, VX_IMAGE_ATTRIBUTE_FORMAT, &format,
		            sizeof( format ) ) );
        CHECK_VX_STATUS( vxReleaseImage( ( vx_image * )&ref ) );
		//Only allow U8 image in this sample
        if( format != VX_DF_IMAGE_U8 )
        {
            return VX_ERROR_INVALID_FORMAT;
        }
    }
    else
    {
        // invalid input parameter
        return VX_ERROR_INVALID_PARAMETERS;
    }

    return VX_SUCCESS;
}

/**
 * Output validator is called for each output parameter and should set requirements for parameter attributes
 */

vx_status VX_CALLBACK count_non_zero_output_validator( vx_node        node,
                                                    vx_uint32      index,
                                                    vx_meta_format meta )
{
    if( index == 1 )
    {
        //meta.dim.scalar.type = VX_TYPE_FLOAT64;
        vx_enum type = VX_TYPE_FLOAT64;
        CHECK_VX_STATUS( vxSetMetaFormatAttribute( meta, VX_SCALAR_ATTRIBUTE_TYPE, &type,
		    						sizeof( type ) ) );
    }
    else
    {
        // invalid input parameter
        return VX_ERROR_INVALID_PARAMETERS;
    }

    return VX_SUCCESS;
}

/**
 * This is a kernel function that is called when the node is executed in the graph
 */

vx_status VX_CALLBACK count_non_zero_host_side_function( vx_node node,
							const vx_reference * refs,
							vx_uint32 num )
{
    vx_image  input   = ( vx_image ) refs[0];
    vx_scalar s_total = ( vx_scalar) refs[1];
    vx_float64 total;

    vx_uint32 width = 0, height = 0;
    CHECK_VX_STATUS( vxQueryImage( input, VX_IMAGE_ATTRIBUTE_WIDTH,  &width,  sizeof( width ) ) );
    CHECK_VX_STATUS( vxQueryImage( input, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof( height ) ) );

    vx_rectangle_t rect = { 0, 0, width, height };
    vx_imagepatch_addressing_t addr_input = { 0 };
    void * ptr_input = NULL;
    CHECK_VX_STATUS(vxAccessImagePatch( input, &rect, 0, &addr_input, &ptr_input, VX_READ_ONLY));

    cv::Mat mat_input(  height, width, CV_8U, ptr_input,  addr_input .stride_y );
    total = cv::countNonZero(mat_input);

    CHECK_VX_STATUS( vxCommitImagePatch( input,  &rect, 0, &addr_input,  ptr_input ) );

    CHECK_VX_STATUS( vxWriteScalarValue(s_total, &total) );

    return VX_SUCCESS;
}

extern "C"
#if _WIN32
__declspec(dllexport)
#endif
vx_status VX_API_CALL vxPublishKernels(vx_context context)
{
    IntelVXSample::logger(1) << "[ INFO ] Start vxPublishKernels(" << context << ")" << std::endl;
    IntelVXSample::logger(1) << "[ INFO ] vxAddKernel(ConnectedComponentLabelingKernel)"
								<< std::endl;
    vx_kernel kernel = vxAddUserKernel(
        context,
        VX_KERNEL_NAME_INTEL_SAMPLE_CONNECTED_COMPONENT_LABELING,
        VX_KERNEL_SAMPLE_CAMERATAMPERING_CONNECTED_COMPONENT_LABELING,
        ConnectedComponentLabelingKernel,
        4,
        ConnectedComponentLabelingValidator,
        ConnectedComponentLabelingInitialize,
        ConnectedComponentLabelingDeinitialize
    );

    if (kernel)
    {
        // Note, here RETURN_VX_STATUS macro is used from common sample infrastructure for easier debugging.
        // This macro will terminate the program in case when returned status is not VX_SUCCESS.
        // It is not a valid behaviour for an input/output validator. Instead the status should be
        // reported as a return value. Use program termination (like here) in educational and debugging purposes.
        RETURN_VX_STATUS(vxAddParameterToKernel(kernel, CONNECTED_COMPONENT_LABELING_PARAM_INPUT,      VX_INPUT,  VX_TYPE_IMAGE,  VX_PARAMETER_STATE_REQUIRED));
        RETURN_VX_STATUS(vxAddParameterToKernel(kernel, CONNECTED_COMPONENT_LABELING_PARAM_THRESHOLD,  VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        RETURN_VX_STATUS(vxAddParameterToKernel(kernel, CONNECTED_COMPONENT_LABELING_PARAM_OUTPUT,     VX_OUTPUT, VX_TYPE_IMAGE,  VX_PARAMETER_STATE_REQUIRED));
        RETURN_VX_STATUS(vxAddParameterToKernel(kernel, CONNECTED_COMPONENT_LABELING_PARAM_RECTANGLES, VX_OUTPUT, VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED));

        RETURN_VX_STATUS(vxFinalizeKernel(kernel));
    }

    IntelVXSample::logger(1) << "[ INFO ] Start vxPublishKernels(" << context << ")" << std::endl;
    IntelVXSample::logger(1) << "[ INFO ] vxAddKernel()" << std::endl;

    vx_kernel kernel_count_non_zero = vxAddKernel( context,
                                    VX_KERNEL_NAME_INTEL_SAMPLE_COUNT_NON_ZERO,
                                    VX_USER_KERNEL_COUNT_NON_ZERO,
                                    count_non_zero_host_side_function,
                                    2,   // numParams
                                    count_non_zero_input_validator,
                                    count_non_zero_output_validator,
                                    NULL,
                                    NULL );
    CHECK_VX_OBJ( kernel_count_non_zero );

    if (kernel)
    {
	RETURN_VX_STATUS( vxAddParameterToKernel( kernel_count_non_zero, 0, VX_INPUT,  VX_TYPE_IMAGE,  VX_PARAMETER_STATE_REQUIRED ) ); // input
	RETURN_VX_STATUS( vxAddParameterToKernel( kernel_count_non_zero, 1, VX_OUTPUT, VX_TYPE_SCALAR,  VX_PARAMETER_STATE_REQUIRED ) ); // output
	RETURN_VX_STATUS( vxFinalizeKernel( kernel_count_non_zero ) );
	RETURN_VX_STATUS( vxReleaseKernel( &kernel_count_non_zero ) );
    }


    return VX_SUCCESS;
}

}