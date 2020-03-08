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
#include "video_stabilization_user_nodes_lib.h"
#include "debug_visualization_lib.hpp"
#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/perfprof.hpp>
#include <stdarg.h>

#include <iostream>


// ------------------------------------------------------------------------------------------------
// Note, here CHECK_VX_STATUS macro is used from common sample infrastructure for easier debugging.
// This macro will terminate the program in case when returned status is not VX_SUCCESS.
// It is not a valid behaviour for an input/output validators, kernel functions or vxPublishKernels.
// Instead the status should be reported as a return value.
// Use program termination (like here) in educational and debugging purposes only.
// ------------------------------------------------------------------------------------------------


typedef enum _estimate_transform_params_e {
    ESTIMATE_TRANSFORM_PARAM_OLD_POINTS = 0,
    ESTIMATE_TRANSFORM_PARAM_NEW_POINTS,
    ESTIMATE_TRANSFORM_PARAM_RECT,
    ESTIMATE_TRANSFORM_PARAM_TRANSFORM,
} estimate_transform_params_e;


typedef enum _debug_visualization_params_e {
    DEBUG_VISUALIZATION_PARAM_INPUT = 0,
    DEBUG_VISUALIZATION_PARAM_OLD_POINTS,
    DEBUG_VISUALIZATION_PARAM_NEW_POINTS,
    DEBUG_VISUALIZATION_PARAM_OUTPUT,
} debug_visualization_params_e;


namespace
{

// --------------------------------------------------------------------------
// Next several functions help to verify node parameters
// --------------------------------------------------------------------------

// Verifies that param is vx_array and item type of the array is itemTypeRequired
vx_status verifyArrayType (vx_parameter param, vx_enum itemTypeRequired)
{
    vx_enum itemType;
    vx_status status = VX_ERROR_INVALID_PARAMETERS;

    if (vxQueryArray((vx_array)param, VX_ARRAY_ITEMTYPE, &itemType, sizeof(itemType)) == VX_SUCCESS)
    {
        if (itemType == itemTypeRequired)
            status = VX_SUCCESS;
        else
            status = VX_ERROR_INVALID_VALUE;
    }
    
    return status;
}

vx_status verifyPointArray (vx_parameter param)
{
    return verifyArrayType(param, VX_TYPE_KEYPOINT);
}


// Checks if a given argument can be treated as video stabilization state
vx_status verifyStateArray (vx_parameter param)
{
    vx_type_e itemType;

    if(
        vxQueryParameter(param, VX_PARAMETER_TYPE, &itemType, sizeof(itemType)) == VX_SUCCESS &&
        itemType == VX_TYPE_ARRAY
    )
    {
        return VX_SUCCESS;
    }
    else
    {
        return VX_ERROR_INVALID_VALUE;
    }
}



/*****************************************************************************
    Next 5 functions define Estimate Transform node behaviour
*****************************************************************************/

vx_status VX_CALLBACK EstimateTransformValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{

    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    if(num!=
        4
        )
    {
        return status;
    }

    RETURN_VX_STATUS(verifyPointArray((vx_parameter)parameters[ESTIMATE_TRANSFORM_PARAM_OLD_POINTS]));
    RETURN_VX_STATUS(verifyPointArray((vx_parameter)parameters[ESTIMATE_TRANSFORM_PARAM_NEW_POINTS]));

    // Should check that this is a vx_array with a single element of type VX_TYPE_RECTANGLE.

    vx_enum itemType;
    vx_size numItems;

    if (vxQueryArray((vx_array)parameters[ESTIMATE_TRANSFORM_PARAM_RECT], VX_ARRAY_ITEMTYPE, &itemType, sizeof(itemType)) == VX_SUCCESS &&
        vxQueryArray((vx_array)parameters[ESTIMATE_TRANSFORM_PARAM_RECT], VX_ARRAY_NUMITEMS, &numItems, sizeof(numItems)) == VX_SUCCESS &&
        itemType == VX_TYPE_RECTANGLE &&
        numItems == 1
        )
    {
        status = VX_SUCCESS;
    }
    else
    {
        status = VX_ERROR_INVALID_VALUE;
    }
    RETURN_VX_STATUS(status);

    vx_enum type = VX_TYPE_FLOAT32;
    RETURN_VX_STATUS(vxSetMetaFormatAttribute(metas[ESTIMATE_TRANSFORM_PARAM_TRANSFORM], VX_MATRIX_TYPE, &type, sizeof(type)));

    vx_size rows = 3;
    RETURN_VX_STATUS(vxSetMetaFormatAttribute(metas[ESTIMATE_TRANSFORM_PARAM_TRANSFORM], VX_MATRIX_ROWS, &rows, sizeof(rows)));

    vx_size cols = 2;
    RETURN_VX_STATUS(vxSetMetaFormatAttribute(metas[ESTIMATE_TRANSFORM_PARAM_TRANSFORM], VX_MATRIX_COLUMNS, &cols, sizeof(cols)));

    status = VX_SUCCESS;

    return status;
}


/// Input validator is called for each input parameter for a node and should check parameter attributes
vx_status VX_CALLBACK EstimateTransformInputValidator(vx_node node, vx_uint32 index)
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    vx_parameter param = vxGetParameterByIndex(node, index);

    if (
        index == ESTIMATE_TRANSFORM_PARAM_OLD_POINTS ||
        index == ESTIMATE_TRANSFORM_PARAM_NEW_POINTS
    )
    {
        status = verifyPointArray(param);
    }
    else if (index == ESTIMATE_TRANSFORM_PARAM_RECT)
    {
        // Should check that this is a vx_array with a single element of type VX_TYPE_RECTANGLE.

        vx_array array;
        vx_enum itemType;
        vx_size numItems;

        if (vxQueryParameter(param, VX_PARAMETER_REF, &array, sizeof(vx_array)) == VX_SUCCESS)
        {
            if (vxQueryArray(array, VX_ARRAY_ITEMTYPE, &itemType, sizeof(itemType)) == VX_SUCCESS &&
                vxQueryArray(array, VX_ARRAY_NUMITEMS, &numItems, sizeof(numItems)) == VX_SUCCESS &&
                itemType == VX_TYPE_RECTANGLE &&
                numItems == 1
            )
            {
                status = VX_SUCCESS;
            }
            else
            {
                status = VX_ERROR_INVALID_VALUE;
            }
            RETURN_VX_STATUS(vxReleaseArray(&array));
        }
    }

	RETURN_VX_STATUS(vxReleaseParameter(&param));
    return status;
}


/// Output validator is called for each output parameter and should set requirements for parameter attributes
vx_status VX_CALLBACK EstimateTransformOutputValidator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    vx_parameter param = vxGetParameterByIndex(node, index);

    if (index == ESTIMATE_TRANSFORM_PARAM_TRANSFORM)
    {
        vx_enum type = VX_TYPE_FLOAT32;
        RETURN_VX_STATUS(vxSetMetaFormatAttribute(meta, VX_MATRIX_TYPE, &type, sizeof(type)));

        vx_size rows = 3;
        RETURN_VX_STATUS(vxSetMetaFormatAttribute(meta, VX_MATRIX_ROWS, &rows, sizeof(rows)));

        vx_size cols = 2;
        RETURN_VX_STATUS(vxSetMetaFormatAttribute(meta, VX_MATRIX_COLUMNS, &cols, sizeof(cols)));

        status = VX_SUCCESS;

    }

    RETURN_VX_STATUS(vxReleaseParameter(&param));
    return status;
}


PERFPROF_REGION_DEFINE(EstimateTransformKernel);


/// This is a kernel function that is called when the node is executed in the graph
vx_status VX_CALLBACK EstimateTransformKernel(vx_node node, const vx_reference* parameters, vx_uint32 num)
{
    PERFPROF_REGION_AUTO(EstimateTransformKernel);

    IntelVXSample::logger(2) << "Start EstimateTransformKernel" << std::endl;
    vx_status status = VX_ERROR_INVALID_PARAMETERS;

    if (num == 4)
    {
        // Extract transform state from node attributes, state is a pointer by definition
        // so we can use it directly as VX_NODE_LOCAL_DATA_PTR
        TrackingStateOpenVX state;
        RETURN_VX_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &state, sizeof(state)));

        vx_array old_points = (vx_array)parameters[ESTIMATE_TRANSFORM_PARAM_OLD_POINTS];
        vx_array new_points = (vx_array)parameters[ESTIMATE_TRANSFORM_PARAM_NEW_POINTS];
        vx_array rect = (vx_array)parameters[ESTIMATE_TRANSFORM_PARAM_RECT];
        vx_matrix transform =  (vx_matrix)parameters[ESTIMATE_TRANSFORM_PARAM_TRANSFORM];

        status = estimateTransform(state, old_points, new_points, rect, transform);
    }

    IntelVXSample::logger(2) << "End EstimateTransformKernel" << std::endl;

    return status;
}


/// This function is called once when node instance is initialized in a graph and may contain appropriate one-time initialization
vx_status VX_CALLBACK EstimateTransformInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    // Create a tracking state and assign it to node as an attribute to be accessible when kernel is executed
    // This state is changed every node kernel execution and tracks current movements in the stabilization process.
    TrackingStateOpenVX state = createEstimateTransformState();
    // State is a pointer by definition so we can use it directly as VX_NODE_ATTRIBUTE_LOCAL_DATA_PTR
    RETURN_VX_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &state, sizeof(state)));
    return VX_SUCCESS;
}


/// This function is called when node instance is destroyed from a graph
vx_status VX_CALLBACK EstimateTransformDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    TrackingStateOpenVX state;
    RETURN_VX_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &state, sizeof(state)));
    releaseEstimateTransformState(&state);
    state = 0;
    // set local data pointer to null to avoid double deletion of it in OpenVX run-time
    RETURN_VX_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &state, sizeof(state)));
    return VX_SUCCESS;
}



/*****************************************************************************
    Next 5 functions define Debug Visualization node behaviour
*****************************************************************************/
vx_status VX_CALLBACK DebugVisualizationValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{

    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    if(num!=4)
    {
        return status;
    }

    vx_df_image imageFormat;
    if(vxQueryImage((vx_image)parameters[DEBUG_VISUALIZATION_PARAM_INPUT], VX_IMAGE_FORMAT, &imageFormat, sizeof(imageFormat)) == VX_SUCCESS &&
        imageFormat == VX_DF_IMAGE_RGB
        )
    {
        status = VX_SUCCESS;
    }
    RETURN_VX_STATUS(status);

    RETURN_VX_STATUS(verifyPointArray((vx_parameter)parameters[DEBUG_VISUALIZATION_PARAM_OLD_POINTS]));
    RETURN_VX_STATUS(verifyPointArray((vx_parameter)parameters[DEBUG_VISUALIZATION_PARAM_OLD_POINTS]));

    vx_uint32 inputWidth, inputHeight;
    if(
        vxQueryImage((vx_image)parameters[DEBUG_VISUALIZATION_PARAM_INPUT], VX_IMAGE_WIDTH,   &inputWidth,  sizeof( inputWidth)) == VX_SUCCESS &&
        vxQueryImage((vx_image)parameters[DEBUG_VISUALIZATION_PARAM_INPUT], VX_IMAGE_HEIGHT,  &inputHeight, sizeof(inputHeight)) == VX_SUCCESS
        )
    {
        vx_df_image imageFormat = VX_DF_IMAGE_RGB;

        // Note, here CHECK_VX_STATUS macro is used from common sample infrastructure for easier debugging.
        // This macro will terminate the program in case when returned status is not VX_SUCCESS.
        // It is not a valid behaviour for an input/output validator. Instead the status should be
        // reported as a return value. Use program termination (like here) in educational and debugging purposes.
        RETURN_VX_STATUS(vxSetMetaFormatAttribute(metas[DEBUG_VISUALIZATION_PARAM_OUTPUT], VX_IMAGE_FORMAT, &imageFormat, sizeof(imageFormat)));
        RETURN_VX_STATUS(vxSetMetaFormatAttribute(metas[DEBUG_VISUALIZATION_PARAM_OUTPUT], VX_IMAGE_WIDTH, &inputWidth, sizeof(inputWidth)));
        RETURN_VX_STATUS(vxSetMetaFormatAttribute(metas[DEBUG_VISUALIZATION_PARAM_OUTPUT], VX_IMAGE_HEIGHT, &inputHeight, sizeof(inputHeight)));

        status = VX_SUCCESS;
    }

    return status;
}


/// Input validator is called for each input parameter for a node and should check parameter attributes
vx_status VX_CALLBACK DebugVisualizationInputValidator(vx_node node, vx_uint32 index)
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    vx_parameter param = vxGetParameterByIndex(node, index);

    if(index == DEBUG_VISUALIZATION_PARAM_INPUT)
    {
        // Verify that input image is a 3-channel RGB image

        vx_image image;
        vx_df_image imageFormat;

        if(vxQueryParameter(param, VX_PARAMETER_REF, &image, sizeof(image)) == VX_SUCCESS)
        {
            if(vxQueryImage(image, VX_IMAGE_FORMAT, &imageFormat, sizeof(imageFormat)) == VX_SUCCESS &&
                imageFormat == VX_DF_IMAGE_RGB
            )
            {
                status = VX_SUCCESS;
            }

            RETURN_VX_STATUS(vxReleaseImage(&image));
        }
    }
    else if (
        index == DEBUG_VISUALIZATION_PARAM_OLD_POINTS ||
        index == DEBUG_VISUALIZATION_PARAM_NEW_POINTS
    )
    {
        status = verifyPointArray(param);
    }

    RETURN_VX_STATUS(vxReleaseParameter(&param));
    return status;
}


/// Output validator is called for each output parameter and should set requirements for parameter attributes
vx_status VX_CALLBACK DebugVisualizationOutputValidator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    vx_parameter param = vxGetParameterByIndex(node, index);

    if(index == DEBUG_VISUALIZATION_PARAM_OUTPUT)
    {
        // Query _input_ image dimensions and set output requirements based on that

        vx_parameter input = vxGetParameterByIndex(node, DEBUG_VISUALIZATION_PARAM_INPUT);
        vx_image inputImage;
        vx_uint32 inputWidth, inputHeight;

        if(vxQueryParameter(input, VX_PARAMETER_REF, &inputImage,  sizeof(inputImage)) == VX_SUCCESS)
        {
            if(
                vxQueryImage(inputImage, VX_IMAGE_WIDTH,   &inputWidth,  sizeof( inputWidth)) == VX_SUCCESS &&
                vxQueryImage(inputImage, VX_IMAGE_HEIGHT,  &inputHeight, sizeof(inputHeight)) == VX_SUCCESS
            )
            {
                vx_df_image imageFormat = VX_DF_IMAGE_RGB;

                // Note, here CHECK_VX_STATUS macro is used from common sample infrastructure for easier debugging.
                // This macro will terminate the program in case when returned status is not VX_SUCCESS.
                // It is not a valid behaviour for an input/output validator. Instead the status should be
                // reported as a return value. Use program termination (like here) in educational and debugging purposes.
                RETURN_VX_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_FORMAT, &imageFormat, sizeof(imageFormat)));
                RETURN_VX_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_WIDTH, &inputWidth, sizeof(inputWidth)));
                RETURN_VX_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_HEIGHT, &inputHeight, sizeof(inputHeight)));

                status = VX_SUCCESS;
            }

            RETURN_VX_STATUS(vxReleaseImage(&inputImage));
        }
        RETURN_VX_STATUS(vxReleaseParameter(&input));
    }
 
    RETURN_VX_STATUS(vxReleaseParameter(&param));
    return status;
}


PERFPROF_REGION_DEFINE(DebugVisualizationKernel);

/// This is a kernel function that is called when the node is executed in the graph
vx_status VX_CALLBACK DebugVisualizationKernel(vx_node node, const vx_reference* parameters, vx_uint32 num)
{
    PERFPROF_REGION_AUTO(DebugVisualizationKernel);
    IntelVXSample::logger(2) << "Start DebugVisualizationKernel" << std::endl;
    vx_status status = VX_ERROR_INVALID_PARAMETERS;

    if (num == 4)
    {
        vx_image input = (vx_image)parameters[DEBUG_VISUALIZATION_PARAM_INPUT];
        vx_array oldPoints = (vx_array)parameters[DEBUG_VISUALIZATION_PARAM_OLD_POINTS];
        vx_array newPoints = (vx_array)parameters[DEBUG_VISUALIZATION_PARAM_NEW_POINTS];
        vx_image output = (vx_image)parameters[DEBUG_VISUALIZATION_PARAM_OUTPUT];

        drawDebugVisualization(input, oldPoints, newPoints, output);
        status = VX_SUCCESS; // if this point is executed, that all is really OK
    }

    IntelVXSample::logger(2) << "End DebugVisualizationKernel" << std::endl;

    return status;
}


/// This function is called once when node instance is initialized in a graph and may contain appropriate one-time initialization
vx_status VX_CALLBACK DebugVisualizationInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    return VX_SUCCESS;
}


/// This function is called when node instance is destroyed from a graph
vx_status VX_CALLBACK DebugVisualizationDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    return VX_SUCCESS;
}

}


extern "C"
#if _WIN32
__declspec(dllexport)
#endif
vx_status VX_API_CALL vxPublishKernels(vx_context context)
{
    IntelVXSample::logger(1) << "[ INFO ] Start vxPublishKernels(" << context << ")" << std::endl;

    IntelVXSample::logger(1) << "[ INFO ] vxAddKernel(EstimateTransformKernel)" << std::endl;

    vx_kernel kernel = vxAddUserKernel(
        context,
        VX_KERNEL_NAME_INTEL_SAMPLE_ESTIMATE_TRANSFORM,
        VX_KERNEL_SAMPLE_STABILIZATION_ESTIMATE_TRANSFORM,
        EstimateTransformKernel,
        4,
        EstimateTransformValidator,
        EstimateTransformInitialize,
        EstimateTransformDeinitialize
    );
    if (kernel)
    {
        // Note, here CHECK_VX_STATUS macro is used from common sample infrastructure for easier debugging.
        // This macro will terminate the program in case when returned status is not VX_SUCCESS.
        // It is not a valid behaviour for an input/output validator. Instead the status should be
        // reported as a return value. Use program termination (like here) in educational and debugging purposes.
        RETURN_VX_STATUS(vxAddParameterToKernel(kernel, ESTIMATE_TRANSFORM_PARAM_OLD_POINTS, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        RETURN_VX_STATUS(vxAddParameterToKernel(kernel, ESTIMATE_TRANSFORM_PARAM_NEW_POINTS, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        RETURN_VX_STATUS(vxAddParameterToKernel(kernel, ESTIMATE_TRANSFORM_PARAM_RECT, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        RETURN_VX_STATUS(vxAddParameterToKernel(kernel, ESTIMATE_TRANSFORM_PARAM_TRANSFORM, VX_OUTPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED));

        RETURN_VX_STATUS(vxFinalizeKernel(kernel));
    }

    IntelVXSample::logger(1) << "[ INFO ] vxAddKernel(DebugVisualizationKernel)" << std::endl;

    kernel = vxAddUserKernel(
        context,
        VX_KERNEL_NAME_INTEL_SAMPLE_DEBUG_VISUALIZATION,
        VX_KERNEL_SAMPLE_STABILIZATION_DEBUG_VISUALIZATION,
        DebugVisualizationKernel,
        4,
        DebugVisualizationValidator,
        DebugVisualizationInitialize,
        DebugVisualizationDeinitialize
    );
    if (kernel)
    {
        RETURN_VX_STATUS(vxAddParameterToKernel(kernel, DEBUG_VISUALIZATION_PARAM_INPUT,  VX_INPUT,  VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        RETURN_VX_STATUS(vxAddParameterToKernel(kernel, DEBUG_VISUALIZATION_PARAM_OLD_POINTS, VX_INPUT,  VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        RETURN_VX_STATUS(vxAddParameterToKernel(kernel, DEBUG_VISUALIZATION_PARAM_NEW_POINTS, VX_INPUT,  VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        RETURN_VX_STATUS(vxAddParameterToKernel(kernel, DEBUG_VISUALIZATION_PARAM_OUTPUT, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));

        RETURN_VX_STATUS(vxFinalizeKernel(kernel));
    }

    IntelVXSample::logger(1) << "[ INFO ] End vxPublishKernels(" << context << ")" << std::endl;

    return VX_SUCCESS;
}
