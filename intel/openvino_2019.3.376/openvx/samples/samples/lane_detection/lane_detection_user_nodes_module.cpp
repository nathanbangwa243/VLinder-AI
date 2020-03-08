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


#include <VX/vx.h>
#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/perfprof.hpp>
#include "collect_lane_marks.hpp"


#define PROCESS_VX_STATUS(NODE, COMMAND)                                \
    {                                                           \
        vx_status __local_status = COMMAND;                     \
        if(__local_status != VX_SUCCESS)                        \
        {                                                       \
            std::string msg = std::string("Code:") + IntelVXSample::vxStatusToStr(__local_status) + std::string(" COMMAND: ") + std::string(#COMMAND);\
            vxAddLogEntry((vx_reference)NODE, __local_status, msg.c_str());\
            return __local_status;                                       \
        }                                                       \
    }

#define SAFE_VX_CALL(STATUS, NODE, COMMAND)                                \
    if(STATUS == VX_SUCCESS){                                                           \
        vx_status __local_status = COMMAND;                     \
        if(__local_status != VX_SUCCESS)                        \
        {                                                       \
            std::string msg = std::string("Code:") + IntelVXSample::vxStatusToStr(__local_status) + std::string(" COMMAND: ") + std::string(#COMMAND);\
            vxAddLogEntry((vx_reference)NODE, __local_status, msg.c_str());\
            (STATUS) = __local_status;                                       \
        }                                                       \
    }


#define VX_LIBRARY_SAMPLE_LANEDETECTION (0x3)

#define VX_KERNEL_NAME_SAMPLE_LANEDETECTION_COLLECT_LANE_MARKS "com.intel.sample.collectlanemarks"

enum vx_kernel_intel_sample_lanedetection_e {
    VX_KERNEL_SAMPLE_LANEDETECTION_COLLECT_LANE_MARKS = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_LANEDETECTION) + 0x0,
};

typedef enum _collect_lane_marks_params_e {
    COLLECT_LANE_MARKS_PARAM_EDGES = 0,
    COLLECT_LANE_MARKS_PARAM_EDGE_THRESHOLD,
    COLLECT_LANE_MARKS_PARAM_LINES,
    COLLECT_LANE_MARKS_PARAM_LINE_COUNT,
    COLLECT_LANE_MARKS_PARAM_PERSPECTIVE_MATRIX,
    COLLECT_LANE_MARKS_PARAM_INPUT_IMAGE,
    COLLECT_LANE_MARKS_PARAM_OUTPUT_IMAGE,
    COLLECT_LANE_MARKS_PARAM_NUM
} _collect_lane_marks_params_e;


/*****************************************************************************
    Next 4 functions define Collect Lane Marks node behaviour
    CollectLaneMarksValidator  - to check input/output OpenVX parameters
    CollectLaneMarksInitialize      - to allocate and initialize internal node data
    CollectLaneMarksDeinitialize    - to release internal node data
    CollectLaneMarksKernel          - to make main job
*****************************************************************************/
// Validator is called for each input/outpu parameter for a node and should check parameter attributes
vx_status VX_CALLBACK CollectLaneMarksValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;

    {
        vx_df_image imageType = 0;
        if( vxQueryImage((vx_image)parameters[COLLECT_LANE_MARKS_PARAM_EDGES], VX_IMAGE_FORMAT, &imageType, sizeof(imageType)) == VX_SUCCESS)
        {
            if(imageType == VX_DF_IMAGE_U8)
                status = VX_SUCCESS;
            else
                vxAddLogEntry((vx_reference)node, status, "CollectLaneMarks: 'edges' is not VX_DF_IMAGE_U8 image\n");
        }
    }
    {
        vx_enum   scalarType;
        if (vxQueryScalar((vx_scalar)parameters[COLLECT_LANE_MARKS_PARAM_EDGE_THRESHOLD], VX_SCALAR_TYPE, &scalarType, sizeof(scalarType)) == VX_SUCCESS)
        {
            if (scalarType == VX_TYPE_INT32)
                status = VX_SUCCESS;
            else
                vxAddLogEntry((vx_reference)node, status, "CollectLaneMarks: 'edgeThreshold' is not VX_TYPE_INT32 scalar\n");
        }
    }
    {
        vx_enum  arrayType;
        if (vxQueryArray((vx_array)parameters[COLLECT_LANE_MARKS_PARAM_LINES], VX_ARRAY_ITEMTYPE, &arrayType, sizeof(arrayType)) == VX_SUCCESS)
        {
            if (arrayType == VX_TYPE_RECTANGLE)
                status = VX_SUCCESS;
            else
                vxAddLogEntry((vx_reference)node, status, "CollectLaneMarks: lines is not VX_TYPE_RECTANGLE array\n");
        }
    }
    {
        vx_enum   scalarType;
        if (vxQueryScalar((vx_scalar)parameters[COLLECT_LANE_MARKS_PARAM_LINE_COUNT], VX_SCALAR_TYPE, &scalarType, sizeof(scalarType)) == VX_SUCCESS)
        {
            if (scalarType == VX_TYPE_INT32)
                status = VX_SUCCESS;
            else
                vxAddLogEntry((vx_reference)node, status, "CollectLaneMarks: lineCount is not VX_TYPE_INT32\n");
        }
    }
    {
        vx_enum     matrixType;
        vx_size     rows;
        vx_size     cols;
        if( vxQueryMatrix((vx_matrix)parameters[COLLECT_LANE_MARKS_PARAM_PERSPECTIVE_MATRIX], VX_MATRIX_TYPE,    &matrixType, sizeof(matrixType)) == VX_SUCCESS &&
            vxQueryMatrix((vx_matrix)parameters[COLLECT_LANE_MARKS_PARAM_PERSPECTIVE_MATRIX], VX_MATRIX_ROWS,    &rows, sizeof(rows)) == VX_SUCCESS &&
            vxQueryMatrix((vx_matrix)parameters[COLLECT_LANE_MARKS_PARAM_PERSPECTIVE_MATRIX], VX_MATRIX_COLUMNS, &cols, sizeof(cols)) == VX_SUCCESS)
        {
            if(matrixType == VX_TYPE_FLOAT32 && rows == 3 && cols == 3)
                status = VX_SUCCESS;
            else
                vxAddLogEntry((vx_reference)node, status, "CollectLaneMarks: matrix is not 3x3 VX_TYPE_FLOAT32\n");
        }
    }
    {
        vx_df_image imageType = 0;
        if( vxQueryImage((vx_image)parameters[COLLECT_LANE_MARKS_PARAM_INPUT_IMAGE], VX_IMAGE_FORMAT, &imageType, sizeof(imageType)) == VX_SUCCESS)
        {
            if(imageType == VX_DF_IMAGE_RGB)
                status = VX_SUCCESS;
            else
                vxAddLogEntry((vx_reference)node, status, "CollectLaneMarks: input is not VX_DF_IMAGE_RGB image\n");
        }
    }


    {
        vx_df_image type = VX_DF_IMAGE_RGB;
        vx_uint32   width = 0;
        vx_uint32   height = 0;

        //Query the input image width and height
        PROCESS_VX_STATUS(node, vxQueryImage((vx_image)parameters[COLLECT_LANE_MARKS_PARAM_INPUT_IMAGE], VX_IMAGE_WIDTH, &width, sizeof(width)) );
        PROCESS_VX_STATUS(node, vxQueryImage((vx_image)parameters[COLLECT_LANE_MARKS_PARAM_INPUT_IMAGE], VX_IMAGE_HEIGHT, &height, sizeof(height)) );

        //Set width, height and type for validation
        PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[COLLECT_LANE_MARKS_PARAM_OUTPUT_IMAGE], VX_IMAGE_WIDTH, &width, sizeof(width)) );
        PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[COLLECT_LANE_MARKS_PARAM_OUTPUT_IMAGE], VX_IMAGE_HEIGHT,&height, sizeof(height)) );
        PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[COLLECT_LANE_MARKS_PARAM_OUTPUT_IMAGE], VX_IMAGE_FORMAT,&type, sizeof(type)) );

        status = VX_SUCCESS;
    }

    return status;
}


// This function is called once when node instance is initialized in a graph and may contain appropriate one-time initialization
vx_status VX_CALLBACK CollectLaneMarksInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    CollectLaneMarks* obj = new CollectLaneMarks;
    // State is a pointer by definition so we can use it directly as VX_NODE_LOCAL_DATA_PTR
    PROCESS_VX_STATUS(node,vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &obj, sizeof(obj)));
    return VX_SUCCESS;
}


// This function is called when node instance is destroyed from a graph
vx_status VX_CALLBACK CollectLaneMarksDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    CollectLaneMarks* obj = NULL;
    PROCESS_VX_STATUS(node,vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &obj, sizeof(obj)));
    delete obj;
    obj = NULL;
    // set local data pointer to null to avoid double deletion of it in OpenVX run-time
    PROCESS_VX_STATUS(node,vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &obj, sizeof(obj)));
    return VX_SUCCESS;
}


PERFPROF_REGION_DEFINE(CollectLaneMarksKernel);
vx_status VX_CALLBACK CollectLaneMarksKernel(vx_node node, const vx_reference* parameters, vx_uint32 num)
{
    PERFPROF_REGION_AUTO(CollectLaneMarksKernel);

    if (num == COLLECT_LANE_MARKS_PARAM_NUM)
    {
        // Extract CollectLaneMarks class pointer from node attributes, it is a pointer by definition
        // so we can use it directly as VX_NODE_LOCAL_DATA_PTR
        CollectLaneMarks* obj = NULL;
        PROCESS_VX_STATUS(node, vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &obj, sizeof(obj)));

        // get reference to all OpenVX parameters
        vx_image  ovxImgEdges     = (vx_image) parameters[COLLECT_LANE_MARKS_PARAM_EDGES];
        vx_scalar ovxEdgeThreshold= (vx_scalar)parameters[COLLECT_LANE_MARKS_PARAM_EDGE_THRESHOLD];
        vx_array  ovxLineArray    = (vx_array) parameters[COLLECT_LANE_MARKS_PARAM_LINES];
        vx_scalar ovxLineCount    = (vx_scalar)parameters[COLLECT_LANE_MARKS_PARAM_LINE_COUNT];
        vx_matrix ovxH            = (vx_matrix)parameters[COLLECT_LANE_MARKS_PARAM_PERSPECTIVE_MATRIX];
        vx_image  ovxInpImg       = (vx_image) parameters[COLLECT_LANE_MARKS_PARAM_INPUT_IMAGE];
        vx_image  ovxOutImg       = (vx_image) parameters[COLLECT_LANE_MARKS_PARAM_OUTPUT_IMAGE];

        vx_int32                edgeThreshold;
        std::vector<cv::Vec4i>  lines; // will be filled later

        PROCESS_VX_STATUS(node, vxCopyScalar(ovxEdgeThreshold, &edgeThreshold, VX_READ_ONLY, VX_MEMORY_TYPE_HOST) );

        {//get line segments from OpenVX array and run final step
            vx_int32    LineCountOVX = -1;
            PROCESS_VX_STATUS(node, vxCopyScalar(ovxLineCount, &LineCountOVX, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
            vx_size stride=0;
            char*   ptr=NULL;
            if(LineCountOVX>0)
            {
                lines.resize(LineCountOVX);
                vx_map_id map_id;
                PROCESS_VX_STATUS(node, vxMapArrayRange(ovxLineArray, 0, LineCountOVX, &map_id, &stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0) );
                for(int i=0; i<LineCountOVX; ++i)
                {//iterate over all lines and copy data from OpenVX array into vector
                    vx_rectangle_t* pL = (vx_rectangle_t*)(ptr + stride*i);
                    lines[i][0] = pL->start_x;
                    lines[i][1] = pL->start_y;
                    lines[i][2] = pL->end_x;
                    lines[i][3] = pL->end_y;
                }
                PROCESS_VX_STATUS(node,vxUnmapArrayRange(ovxLineArray, map_id));
            }
        }

        {// map filter responce and run final stage
            vx_map_id map_id;
            cv::Mat imgEdges = IntelVXSample::mapAsMat(ovxImgEdges, VX_READ_ONLY, &map_id);
            obj->Process(imgEdges, edgeThreshold, lines);
            IntelVXSample::unmapAsMat(ovxImgEdges, imgEdges, map_id);
        }

        {// draw OpenVX result
            vx_map_id map_id_in;
            vx_map_id map_id_out;
            cv::Mat     inpImg = IntelVXSample::mapAsMat(ovxInpImg, VX_READ_ONLY, &map_id_in);
            cv::Mat     outImg = IntelVXSample::mapAsMat(ovxOutImg, VX_READ_AND_WRITE, &map_id_out);
            vx_float32  ovxData[9];
            PROCESS_VX_STATUS(node, vxCopyMatrix(ovxH, (void*)ovxData, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
            double ocvData[9] = {
                ovxData[0],ovxData[3],ovxData[6],
                ovxData[1],ovxData[4],ovxData[7],
                ovxData[2],ovxData[5],ovxData[8]
            };
            inpImg.copyTo(outImg);
            obj->DrawResult(outImg, cv::Mat(3,3,CV_64F,ocvData), 2, false);
            IntelVXSample::unmapAsMat(ovxInpImg, inpImg, map_id_in);
            IntelVXSample::unmapAsMat(ovxOutImg, outImg, map_id_out);
        }
    }
    return VX_SUCCESS;
}

// define and implement function that will publich node inside OpenVX runtime
extern "C"
#if _WIN32
__declspec(dllexport)
#endif
vx_status VX_API_CALL vxPublishKernels(vx_context context)
{
    vx_status status = VX_SUCCESS;
    vx_kernel kernel = vxAddUserKernel(
        context,
        VX_KERNEL_NAME_SAMPLE_LANEDETECTION_COLLECT_LANE_MARKS,
        VX_KERNEL_SAMPLE_LANEDETECTION_COLLECT_LANE_MARKS,
        CollectLaneMarksKernel,
        COLLECT_LANE_MARKS_PARAM_NUM,
        CollectLaneMarksValidator,
        CollectLaneMarksInitialize,
        CollectLaneMarksDeinitialize
    );

    PROCESS_VX_STATUS(context,vxGetStatus((vx_reference)kernel));

    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, COLLECT_LANE_MARKS_PARAM_EDGES,             VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, COLLECT_LANE_MARKS_PARAM_EDGE_THRESHOLD,    VX_INPUT, VX_TYPE_SCALAR,VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, COLLECT_LANE_MARKS_PARAM_LINES,             VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, COLLECT_LANE_MARKS_PARAM_LINE_COUNT,        VX_INPUT, VX_TYPE_SCALAR,VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, COLLECT_LANE_MARKS_PARAM_PERSPECTIVE_MATRIX,VX_INPUT, VX_TYPE_MATRIX,VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, COLLECT_LANE_MARKS_PARAM_INPUT_IMAGE,       VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, COLLECT_LANE_MARKS_PARAM_OUTPUT_IMAGE,      VX_OUTPUT,VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));

    SAFE_VX_CALL(status,context,vxFinalizeKernel(kernel));

    if( VX_SUCCESS != status )
        vxRemoveKernel( kernel );

    return status;
}
