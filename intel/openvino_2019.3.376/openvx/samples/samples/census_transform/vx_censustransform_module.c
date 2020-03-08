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


//! An internal definition of the order of the parameters to the function
//! This list must match the parameter list in the function and in the
//! publish kernel list.
typedef enum _census_transform_params_e {
    CENSUSTRANSFORM_PARAM_INPUT = 0,
    CENSUSTRANSFORM_PARAM_OUTPUT
} census_transform_params_e;

//!*********************************************************************************************************
//! Function Name        :  CensusTransformValidator
//! Argument 1           :  Handle to the node                       [IN]
//! Argument 2           :  The array of parameters to be validated  [IN]
//! Argument 3           :  Number of parameters to be validated     [IN]
//! Argument 4           :  The metadata used to check the parameter [IN]
//! Returns              :  Status
//! Description          :  Input parameter validator for the Census Transform node
//!                      :  The function, which validates the input parameters to this user custom kernel
//!*********************************************************************************************************
vx_status VX_CALLBACK CensusTransformValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{

    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    if(num!=2)
    {
        return status;
    }

    vx_df_image df_image = 0;
    if(vxQueryImage((vx_image)parameters[CENSUSTRANSFORM_PARAM_INPUT], VX_IMAGE_FORMAT, &df_image, sizeof(df_image)) == VX_SUCCESS)
    {
        if (df_image == VX_DF_IMAGE_S16)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "CT Validation failed: invalid input image format\n");
            return status;
        }
    }

    vx_uint32 output_width = 0;
    vx_uint32 output_height = 0;
    vx_uint32 input_width = 0;
    vx_uint32 input_height = 0;

    //Query the input image
    status = vxQueryImage((vx_image)parameters[CENSUSTRANSFORM_PARAM_INPUT], VX_IMAGE_WIDTH, &input_width, sizeof(input_width));
    status |= vxQueryImage((vx_image)parameters[CENSUSTRANSFORM_PARAM_INPUT], VX_IMAGE_HEIGHT, &input_height, sizeof(input_height));

    //this node will actually output w-2xh-2
    // since it only processes 'valid' pixels
    output_width = input_width-2;
    output_height = input_height-2;

    vx_df_image  format = VX_DF_IMAGE_U8;
    //Input is of S16 type and output image is of type U8
    //Set width and height for validation as well
    status |= vxSetMetaFormatAttribute(metas[CENSUSTRANSFORM_PARAM_OUTPUT], VX_IMAGE_WIDTH, &output_width, sizeof(output_width));
    status |= vxSetMetaFormatAttribute(metas[CENSUSTRANSFORM_PARAM_OUTPUT], VX_IMAGE_HEIGHT, &output_height, sizeof(output_height));
    status |= vxSetMetaFormatAttribute(metas[CENSUSTRANSFORM_PARAM_OUTPUT], VX_IMAGE_FORMAT, &format, sizeof(format));

    return status;
}


//!*****************************************************************************************************
//! Function Name        :  CensusTransformKernel
//! Argument 1           :  Handle to the node                  [IN]
//! Argument 2           :  Input parameters                    [IN]
//! Argument 3           :  Number of parameters                [IN]
//! Returns              :  Status
//! Description          :  The private kernel function for CensusTransformKernel custom kernel
//!*****************************************************************************************************
vx_status VX_CALLBACK CensusTransformKernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    vx_image input  = (vx_image) parameters[CENSUSTRANSFORM_PARAM_INPUT];
    vx_image output = (vx_image) parameters[CENSUSTRANSFORM_PARAM_OUTPUT];

    if (num == 2)
    {
        vx_uint32 p = 0;

        void *src, *dst;
        vx_imagepatch_addressing_t src_addr, dst_addr;
        vx_rectangle_t inputrect;
        vx_rectangle_t outputrect;
        vx_map_id input_id, output_id;
        src = dst = NULL;
        status = vxGetValidRegionImage(input, &inputrect);
        status |= vxMapImagePatch(input, &inputrect, p, &input_id, &src_addr, &src, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
        status |= vxGetValidRegionImage(output, &outputrect);
        status |= vxMapImagePatch(output, &outputrect, p, &output_id, &dst_addr, &dst, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);

        if(status == VX_SUCCESS)
        {

            censustransform((vx_int16 *)src,
                src_addr.stride_y,
                (vx_uint8 *)dst,
                dst_addr.stride_y,
                dst_addr.dim_x,
                dst_addr.dim_y);

            status |= vxUnmapImagePatch(input, input_id);
            status |= vxUnmapImagePatch(output, output_id);
        }


    }

    return status;
}


//!***********************************************************************
//! Function Name        :  CensusTransformInitialize
//! Argument 1           :  Handle to the node                  [IN]
//! Argument 2           :  Input parameters                    [IN]
//! Argument 3           :  Number of parameters                [IN]
//! Returns              :  Status
//! Description          :  An initializer function for CensusTransform
//!                         :  node handle
//!***********************************************************************
vx_status VX_CALLBACK CensusTransformInitialize(vx_node node, const vx_reference *parameters,
                                                vx_uint32 num)
{
    /* CensusTransformInitialize requires no initialization of memory or resources */
    return VX_SUCCESS;
}


//!***********************************************************************
//! Function Name        :  CensusTransformDeinitialize
//! Argument 1           :  Handle to the node                  [IN]
//! Argument 2           :  Input parameters                    [IN]
//! Argument 3           :  Number of parameters                [IN]
//! Returns              :  Status
//! Description          :  A deinitializer function
//!***********************************************************************
vx_status VX_CALLBACK CensusTransformDeinitialize(vx_node node, const vx_reference *parameters,
                                                  vx_uint32 num)
{
    /* CensusTransformDeinitialize requires no de-initialization of memory or resources */
    return VX_SUCCESS;
}


//!**************************************************************************
//! Function Name        :  PublishCensusTransformKernel
//! Argument 1           :  Context                             [IN]
//! Returns              :  Status
//! Description          :  This function publishes the user defined kernels
//!**************************************************************************
vx_status VX_API_CALL PublishCensusTransformKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;

    vx_kernel kernel = vxAddUserKernel(context,
        VX_KERNEL_NAME_USER_CENSUSTRANSFORM, //The string to use to match the kernel.
        VX_KERNEL_USER_CENSUSTRANSFORM, //The enumerated value of the kernel to be used by clients.
        CensusTransformKernel, //The process-local function pointer to be invoked.
        2, //The number of parameters for this kernel.
        CensusTransformValidator, //The pointer to callback function, which validates the input and output parameters to this kernel.
        CensusTransformInitialize, //The kernel initialization function.
        CensusTransformDeinitialize); //The kernel de-initialization function.
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
        vxAddLogEntry((vx_reference)context, status, "CT kernel publish failed\n");
    }
    return status;
}

#if _WIN32
__declspec(dllexport)
#endif
vx_status VX_API_CALL vxPublishKernels(vx_context context)
{
    vx_status status = VX_SUCCESS;
    if((status = PublishCensusTransformKernel(context)) != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "CensusTransform kernel publishing failed\n");
    }
    if((status = PublishCensusTransformOpenCLKernel(context)) != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "CensusTransformOpenCL kernel publishing failed\n");
    }
    if((status = PublishCensusTransformTiledKernel(context)) != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "CensusTransformTiled kernel publishing failed\n");
    }

    return VX_SUCCESS;
}

