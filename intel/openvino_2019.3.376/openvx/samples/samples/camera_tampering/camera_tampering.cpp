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

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <VX/vx.h>
#include <VX/vx_api.h>
#include <VX/vx_compatibility.h>
#include <VX/vx_intel_volatile.h>
#include <opencv2/opencv.hpp>
#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/perfprof.hpp>
#include <intel/vx_samples/basic.hpp>
#include <intel/vx_samples/hetero.hpp>
#include "cmdoptions.hpp"
#include "camera_tampering_user_nodes_lib.h"
#include "camera_tampering_common.h"

using namespace std;

#define CANNY_THRESHOLD1    100
#define CANNY_THRESHOLD2    200

enum TamperingType
{
    NO_TAMPERING        = 0,
    DEFOCUS             = 1,
    OCCLUSION           = 2,
};

const std::map<TamperingType, const char*> kTamperingTypeStrings
{
        { TamperingType::NO_TAMPERING       , "NoTampering"        },
        { TamperingType::DEFOCUS            , "Defocus"            },
        { TamperingType::OCCLUSION          , "Occlusion"          },
};

std::string to_string(const TamperingType& type)
{
    auto it = kTamperingTypeStrings.find(type);
    if (it != kTamperingTypeStrings.end())
	{
        return it->second;
    }
	else
	{
        return "UnknownTampering";
    }
}

std::vector<std::string> to_strings(const TamperingType& type)
{
  int val_type = static_cast<int>(type);
  int mask = 1;
  std::vector<std::string> out;

  if (val_type == 0)
  {
    return { kTamperingTypeStrings.find(static_cast<TamperingType>(mask & val_type))->second };
  }

  for (size_t i = 0; i < sizeof(int) * 8; ++i)
  {
    if ((mask & val_type) != 0)
	{
      auto tampering = static_cast<TamperingType>(mask & val_type);
      auto tampering_string = to_string(tampering);
      out.emplace_back(tampering_string);
    }
    mask <<= 1;
  }

  return out;
}

std::ostream& operator<<(std::ostream& stream, const TamperingType& type)
{
  auto strings = to_strings(type);
  for (size_t i = 0; i < strings.size() - 1; ++i)
    stream << strings[i] << "|";
  stream << strings.back();
  return stream;
}

const size_t      kPaddingSize        = 12;
const int         kFontFace           = cv::FONT_HERSHEY_COMPLEX;
const double      kFontScale          = 0.6;
const int         kFontThickness      = 1;

void DrawFrameInfo(cv::Mat& frame, const int tampering_type)
{
  const auto background_color = cv::Scalar(190, 235, 255);
  const auto foreground_color = cv::Scalar(0, 0, 255);

  if (tampering_type != TamperingType::NO_TAMPERING)
  {
    size_t frame_width = frame.cols, frame_height = frame.rows;
    std::stringstream tampering_text;

    tampering_text << "Tampering: " << static_cast<TamperingType>(tampering_type);

    int baseline = -1;
    cv::Size text_size = cv::getTextSize(tampering_text.str(),
                                         kFontFace, kFontScale, kFontThickness, &baseline);

    const cv::Point tl((frame_width - text_size.width) / 2 - kPaddingSize,
                       frame_height - kPaddingSize - kPaddingSize - text_size.height - baseline);
    const cv::Point br((frame_width + text_size.width) / 2 + kPaddingSize,
                       frame_height - kPaddingSize);
    const cv::Point text_offset(kPaddingSize, text_size.height + kPaddingSize / 2);

    cv::rectangle(frame, tl, br, background_color, CV_FILLED);

    cv::putText(frame, tampering_text.str(), tl + text_offset,
                kFontFace, kFontScale, foreground_color, kFontThickness);
  }
}

void DrawBoundingBoxes(
        cv::Mat                 &inImg,         // Input image, can be input color images, or output binary image
        vector<vx_rectangle_t>  &componentBoxes,
        int                     scaleFactor)
{
    // Draw bounding box for detected moving objects
    for (int i = 0; i < componentBoxes.size(); i++)
    {
        cv::rectangle(inImg, cv::Point(componentBoxes[i].start_x * scaleFactor,
		    componentBoxes[i].start_y * scaleFactor),
		    cv::Point(componentBoxes[i].end_x * scaleFactor,
		    componentBoxes[i].end_y * scaleFactor), cv::Scalar(255, 255, 255));
    }
}

/* There's an optional mode to scale large input images down before processing
 * to save computations. When this mode is enabled, this function creates the
 * nodes for scaling. */
void CreateScaleNodes(
    vx_context                          context,
    vx_graph                            graph,
    vx_image                            input,          // input image
    vx_image                            scaleImages[6], // the images used by the scaling nodes
    vx_image                            scaleOutput,    // output image after scaling
    IntelVXSample::HeteroScheduleConfig &heteroConfig,  // heterogeneity configuration
    std::vector<vx_node>                &vxNodes)
{
    vx_node channelExtractNodes[3];
    vx_node scaleImageNodes[3];
    vx_node channelCombineNode;

    // Step 1: channel extract
    channelExtractNodes[0] = vxChannelExtractNode(graph, input, VX_CHANNEL_R, scaleImages[0]);
    CHECK_VX_OBJ(channelExtractNodes[0]);
    if(vx_enum target = heteroConfig.getTargetByNodeName("vxChannelExtractNode(0)"))
    {
        CHECK_VX_STATUS(vxSetNodeTarget(channelExtractNodes[0], target, 0));
    }
    vxNodes.push_back(channelExtractNodes[0]);

    channelExtractNodes[1] = vxChannelExtractNode(graph, input, VX_CHANNEL_G, scaleImages[1]);
    CHECK_VX_OBJ(channelExtractNodes[1]);
    if(vx_enum target = heteroConfig.getTargetByNodeName("vxChannelExtractNode(1)"))
    {
        CHECK_VX_STATUS(vxSetNodeTarget(channelExtractNodes[1], target, 0));
    }
    vxNodes.push_back(channelExtractNodes[1]);

    channelExtractNodes[2] = vxChannelExtractNode(graph, input, VX_CHANNEL_B, scaleImages[2]);
    CHECK_VX_OBJ(channelExtractNodes[2]);
    if(vx_enum target = heteroConfig.getTargetByNodeName("vxChannelExtractNode(2)"))
    {
        CHECK_VX_STATUS(vxSetNodeTarget(channelExtractNodes[2], target, 0));
    }
    vxNodes.push_back(channelExtractNodes[2]);

    // Step 2: scale image
    scaleImageNodes[0] = vxScaleImageNode(graph, scaleImages[0], scaleImages[3], VX_INTERPOLATION_TYPE_BILINEAR);
    CHECK_VX_OBJ(scaleImageNodes[0]);
    if(vx_enum target = heteroConfig.getTargetByNodeName("vxScaleImageNode(0)"))
    {
        CHECK_VX_STATUS(vxSetNodeTarget(scaleImageNodes[0], target, 0));
    }
    vxNodes.push_back(scaleImageNodes[0]);

    scaleImageNodes[1] = vxScaleImageNode(graph, scaleImages[1], scaleImages[4], VX_INTERPOLATION_TYPE_BILINEAR);
    CHECK_VX_OBJ(scaleImageNodes[1]);
    if(vx_enum target = heteroConfig.getTargetByNodeName("vxScaleImageNode(1)"))
    {
        CHECK_VX_STATUS(vxSetNodeTarget(scaleImageNodes[1], target, 0));
    }
    vxNodes.push_back(scaleImageNodes[1]);

    scaleImageNodes[2] = vxScaleImageNode(graph, scaleImages[2], scaleImages[5], VX_INTERPOLATION_TYPE_BILINEAR);
    CHECK_VX_OBJ(scaleImageNodes[2]);
    if(vx_enum target = heteroConfig.getTargetByNodeName("vxScaleImageNode(2)"))
    {
        CHECK_VX_STATUS(vxSetNodeTarget(scaleImageNodes[2], target, 0));
    }
    vxNodes.push_back(scaleImageNodes[2]);

    // Step 3: channel combine
    channelCombineNode = vxChannelCombineNode(graph, scaleImages[3], scaleImages[4], scaleImages[5], NULL, scaleOutput);
    CHECK_VX_OBJ(channelCombineNode);
    if(vx_enum target = heteroConfig.getTargetByNodeName("vxChannelCombineNode"))
    {
        CHECK_VX_STATUS(vxSetNodeTarget(channelCombineNode, target, 0));
    }
    vxNodes.push_back(channelCombineNode);

    // Step 4: release the virtual scale images
    for (int i=0; i<6; i++)
    {
        CHECK_VX_STATUS(vxReleaseImage(&scaleImages[i]));
    }
}

/* This function create the graph for motion detection */
vx_graph CreateCameraTamperingGraph(
    vx_context                      context,
    vx_uint32                       imgWidth,       // input image width
    vx_uint32                       imgHeight,      // input image height
    vx_df_image                     color,          // input image format: RGB
    vx_uint32                       threshold,      // threshold for size filter
    bool                            scaleImage,     // if input image should be checked to make sure if scaling down is needed
    bool                            &scaleFlag,     // if input image should be scaled down before processing
    int                             &scaleFactor,   // scale factor in X & Y axis
    cv::Mat                         &input,         // the matrix which holds input image data
    cv::Mat                         &output,        // the matrix which holds input image data
    cv::Mat                         &labelImg,      // the matrix which holds label image data
    vx_scalar                       &countNonZero_scalar,
    const CmdParserMotionDetection  &cmdparser,     // command parser
    std::vector<vx_node>            &vxNodes,       // vector of nodes
    std::vector<vx_image>           &vxImages,      // vector of images
    std::vector<vx_array>           &vxArrays)      // vector of arrays
{
    IntelVXSample::HeteroScheduleConfig heteroConfig(cmdparser.hetero_config.getValue());
    IntelVXSample::logger(0) << "[ INFO ] "<<cmdparser.hetero_config.getValue() << "\n";

    heteroConfig.pupulateSupportedTargets();

    // GPU can support grayscale input format for background subtraction MOG2 node. It has better performance compared with RGB input.
    bool bgsubMOG2_gray_input_flag = false;
    char *pcEnv = std::getenv("BGSUBMOG2_GRAYSCALE_GPU");
    if (pcEnv != NULL)
    {
        string sEnv(pcEnv);
	vx_enum target_gpu = VX_TARGET_GPU_INTEL;

        if ((sEnv.compare("1") == 0) && (heteroConfig.getTargetByNodeName("vxBackgroundSubMOG2Node") == target_gpu))
        {
            bgsubMOG2_gray_input_flag = true;
        }
    }

    // Create graph
    vx_graph graph = vxCreateGraph(context);
    CHECK_VX_OBJ(graph);

    vx_imagepatch_addressing_t inputFormat, outputFormat, labelFormat;

    inputFormat.dim_x       = imgWidth;
    inputFormat.dim_y       = imgHeight;
    inputFormat.stride_x    = input.elemSize();           // for three channels: R, G, and B
    inputFormat.stride_y    = input.step;                 // number of bytes each matrix row occupies
    inputFormat.scale_x     = VX_SCALE_UNITY;
    inputFormat.scale_y     = VX_SCALE_UNITY;
    inputFormat.step_x      = 1;
    inputFormat.step_y      = 1;
    vx_image vxInput        = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &inputFormat, (void**)&input.data, VX_MEMORY_TYPE_HOST);
    CHECK_VX_OBJ(vxInput);
    vxImages.push_back(vxInput);

    vx_array rectangles     = vxCreateArray(context, VX_TYPE_RECTANGLE, 4096);  /*output rectangles*/
    CHECK_VX_OBJ(rectangles);
    vxArrays.push_back(rectangles);

    // Query image to get image resolution
    vx_uint32 width = 0, height = 0;
    CHECK_VX_STATUS(vxQueryImage(vxInput, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
    CHECK_VX_STATUS(vxQueryImage(vxInput, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));

    // Set internal image dimensions
    int internalWidth = width, internalHeight = height;
    SetScaleParameters(scaleImage, width, height, scaleFlag, internalWidth, internalHeight, scaleFactor);

    vx_image scaleImages[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
    vx_image scaleOutput = NULL;

    if (scaleFlag == true)
    {
        // Create scale images
        CreateScaleImages(graph, color, width, height, internalWidth, internalHeight, scaleImages, scaleOutput);

        // Create scale nodes
        CreateScaleNodes(context, graph, vxInput, scaleImages, scaleOutput, heteroConfig, vxNodes);

        // Update threshold
        threshold = (vx_uint32)(threshold / (scaleFactor * scaleFactor));
    }

    // Create output and label images, they may have different resolution from input image
    output.create(cv::Size(internalWidth, internalHeight), CV_8UC1);
    outputFormat.dim_x      = internalWidth;
    outputFormat.dim_y      = internalHeight;
    outputFormat.stride_x   = output.elemSize();                            // for 1 channel U8
    outputFormat.stride_y   = output.step;                  // number of bytes each matrix row occupies
    outputFormat.scale_x    = VX_SCALE_UNITY;
    outputFormat.scale_y    = VX_SCALE_UNITY;
    outputFormat.step_x     = 1;
    outputFormat.step_y     = 1;
    vx_image vxOutput       = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &outputFormat, (void**)&output.data, VX_IMPORT_TYPE_HOST);
    CHECK_VX_OBJ(vxOutput);
    vxImages.push_back(vxOutput);

    labelImg.create(cv::Size(internalWidth, internalHeight), CV_32SC1);
    labelFormat.dim_x       = internalWidth;
    labelFormat.dim_y       = internalHeight;
    labelFormat.stride_x    = labelImg.elemSize();        // for 1 channel U32
    labelFormat.stride_y    = labelImg.step;              // number of bytes each matrix row occupies
    labelFormat.scale_x     = VX_SCALE_UNITY;
    labelFormat.scale_y     = VX_SCALE_UNITY;
    labelFormat.step_x      = 1;                          // Set it as 1, from specification: step_x is step of X Dimension in pixels
    labelFormat.step_y      = 1;
    vx_image vxLabel        = vxCreateImageFromHandle(context, VX_DF_IMAGE_U32, &labelFormat, (void**)&labelImg.data, VX_MEMORY_TYPE_HOST);
    CHECK_VX_OBJ(vxLabel);
    vxImages.push_back(vxLabel);

    // Virtual images which are connected to internal nodes
    vx_image virtImages[3];

    for (int i=0; i<3; i++)
    {
        virtImages[i] = vxCreateVirtualImage(graph, internalWidth, internalHeight, VX_DF_IMAGE_U8);
        CHECK_VX_OBJ(virtImages[i]);
    }

    vx_image input_image = NULL;
    vx_image bgsubMOG2_input_image = NULL;
    vx_image tempImages[2] = {NULL, NULL};

    if (scaleFlag == true)
    {
        input_image = scaleOutput;
    }
    else
    {
        input_image = vxInput;
    }

    if (bgsubMOG2_gray_input_flag == true)
    {
        vx_image colorConvert_input_image = input_image;
        tempImages[0] = vxCreateVirtualImage(graph, internalWidth, internalHeight, VX_DF_IMAGE_NV12);
        CHECK_VX_OBJ(tempImages[0]);
        tempImages[1] = vxCreateVirtualImage(graph, internalWidth, internalHeight, VX_DF_IMAGE_U8);
        CHECK_VX_OBJ(tempImages[1]);

        vx_node colorConvertNode = vxColorConvertNode(graph, colorConvert_input_image, tempImages[0]);
        CHECK_VX_OBJ(colorConvertNode);
        if(vx_enum target = heteroConfig.getTargetByNodeName("vxColorConvertNode"))
        {
            CHECK_VX_STATUS(vxSetNodeTarget(colorConvertNode, target, 0));
        }
        vxNodes.push_back(colorConvertNode);

        vx_node channelExtractNode = vxChannelExtractNode(graph, tempImages[0], VX_CHANNEL_Y, tempImages[1]);
        CHECK_VX_OBJ(channelExtractNode);
        if(vx_enum target = heteroConfig.getTargetByNodeName("vxChannelExtractNode(3)"))
        {
            CHECK_VX_STATUS(vxSetNodeTarget(channelExtractNode, target, 0));
        }
        vxNodes.push_back(channelExtractNode);

        bgsubMOG2_input_image = tempImages[1];
    }
    else
    {
        bgsubMOG2_input_image = input_image;
    }

    // Camera Tampering nodes
    unsigned int ct_enable = cmdparser.ct_enable.getValue();

    IntelVXSample::logger(0) << "[ INFO ] ct_enable :" << ct_enable << "\n";

    if (ct_enable)
    {

        float ct_scale = cmdparser.ct_scale.getValue();
        vx_uint32 w2 = width * ct_scale;
        vx_uint32 h2 = ((vx_uint32)(height * ct_scale)) & 0xfffffffe;

        IntelVXSample::logger(0) << "[ INFO ] ct_scale :" << ct_scale << "\n";
        IntelVXSample::logger(0) << "[ INFO ] w2 :" << w2 << ",Original :" << width << "\n";
        IntelVXSample::logger(0) << "[ INFO ] h2 :" << h2 << ",Original :" << height <<"\n";

        vx_image vxImgYUV = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_IYUV);
        vx_image vxImgGray = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
        vx_image vxImgScaleGray = vxCreateVirtualImage(graph, w2, h2, VX_DF_IMAGE_U8);
        vx_image vxImgOutput = vxCreateVirtualImage(graph, w2, h2, VX_DF_IMAGE_U8);

        vx_node ct_node1 = vxColorConvertNode(graph, vxInput, vxImgYUV );

        CHECK_VX_OBJ(ct_node1);
        if(vx_enum target = heteroConfig.getTargetByNodeName("CT_vxColorConvertNode"))
        {
            CHECK_VX_STATUS(vxSetNodeTarget(ct_node1, target, 0));
        }
        vxNodes.push_back(ct_node1);

        vx_node ct_node2 = vxChannelExtractNode(graph, vxImgYUV, VX_CHANNEL_Y, vxImgGray);
        CHECK_VX_OBJ(ct_node2);
        if(vx_enum target = heteroConfig.getTargetByNodeName("CT_vxChannelExtractNode"))
        {
            CHECK_VX_STATUS(vxSetNodeTarget(ct_node2, target, 0));
        }
        vxNodes.push_back(ct_node2);

        vx_node ct_node3 = vxScaleImageNode(graph, vxImgGray, vxImgScaleGray, VX_INTERPOLATION_TYPE_AREA);//VX_INTERPOLATION_TYPE_AREA);
        CHECK_VX_OBJ(ct_node3);
        if(vx_enum target = heteroConfig.getTargetByNodeName("CT_vxScaleImageNode"))
        {
            CHECK_VX_STATUS(vxSetNodeTarget(ct_node3, target, 0));
        }
        vxNodes.push_back(ct_node3);

        vx_threshold hyst = vxCreateThreshold(context, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_UINT8);
        vx_int32 lower = CANNY_THRESHOLD1, upper = CANNY_THRESHOLD2;
        vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, &lower, sizeof(lower));
        vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, &upper, sizeof(upper));

        vx_node ct_node4 = vxCannyEdgeDetectorNode(graph, vxImgScaleGray, hyst, 3, VX_NORM_L1, vxImgOutput);
        CHECK_VX_OBJ(ct_node4);
        if(vx_enum target = heteroConfig.getTargetByNodeName("CT_vxCannyEdgeDetectorNode"))
        {
            CHECK_VX_STATUS(vxSetNodeTarget(ct_node4, target, 0));
        }
        vxNodes.push_back(ct_node4);

        vx_node ct_node5 = vxUserCountNonZeroNode(graph, vxImgOutput, countNonZero_scalar);
	CHECK_VX_OBJ(ct_node5);
        vxNodes.push_back(ct_node5);

        CHECK_VX_STATUS(vxReleaseImage(&vxImgYUV));
        CHECK_VX_STATUS(vxReleaseImage(&vxImgGray));
        CHECK_VX_STATUS(vxReleaseImage(&vxImgScaleGray));
        CHECK_VX_STATUS(vxReleaseImage(&vxImgOutput));
    }

    /* Background subtraction MOG2 node
     * The min, max var is of huge gap to vars in clips from video clips from customer.
     * Set the max var to adapt the vars from customer video clips
     * Set the shadow as background in SubMog2 output. If applciation want to process shadow
     * specially, we can set the shadow as a special value that could be identified.
     */
    vx_bg_state_intel bgState = vxCreateBGStateIntel(context, internalWidth, internalHeight, 25, 5, 15.0*15.0f,
					    4.0*4.0f, 23.0*23.0f, 25.0f, 0.9f, 12.0f, 0.95f, 1, 0, 0.5);
    CHECK_VX_OBJ(bgState);
    vx_node bgsubMOG2Node = vxBackgroundSubMOG2NodeIntel(graph, bgsubMOG2_input_image, bgState, virtImages[0]);
    CHECK_VX_OBJ(bgsubMOG2Node);

    if(vx_enum target = heteroConfig.getTargetByNodeName("vxBackgroundSubMOG2Node"))
    {
	CHECK_VX_STATUS(vxSetNodeTarget(bgsubMOG2Node, target, 0));
    }
    vxNodes.push_back(bgsubMOG2Node);

    if (scaleFlag == true)
    {
        CHECK_VX_STATUS(vxReleaseImage(&scaleOutput));
    }

    if (bgsubMOG2_gray_input_flag == true)
    {
        CHECK_VX_STATUS(vxReleaseImage(&tempImages[0]));
        CHECK_VX_STATUS(vxReleaseImage(&tempImages[1]));
    }

    // Morphology nodes
    vx_node dilateNode1 = vxDilate3x3Node(graph, virtImages[0], virtImages[1]);

    CHECK_VX_OBJ(dilateNode1);
    if(vx_enum target = heteroConfig.getTargetByNodeName("vxDilate3x3Node(1)"))
    {
	CHECK_VX_STATUS(vxSetNodeTarget(dilateNode1, target, 0));
    }
    vxNodes.push_back(dilateNode1);

    vx_node erodeNode = vxErode3x3Node(graph, virtImages[1], virtImages[2]);
    CHECK_VX_OBJ(erodeNode);
    if(vx_enum target = heteroConfig.getTargetByNodeName("vxErode3x3Node"))
    {
	CHECK_VX_STATUS(vxSetNodeTarget(erodeNode, target, 0));
    }
    vxNodes.push_back(erodeNode);

    vx_node dilateNode2 = vxDilate3x3Node(graph, virtImages[2], vxOutput);
    CHECK_VX_OBJ(dilateNode2);
    if(vx_enum target = heteroConfig.getTargetByNodeName("vxDilate3x3Node(2)"))
    {
	CHECK_VX_STATUS(vxSetNodeTarget(dilateNode2, target, 0));
    }
    vxNodes.push_back(dilateNode2);

    vx_node connectedComponentLabelingNode = vxConnectedComponentLabelingNode(graph, vxOutput, threshold, vxLabel, rectangles);
    CHECK_VX_OBJ(connectedComponentLabelingNode);
    vxNodes.push_back(connectedComponentLabelingNode);

    // Release virtual images
    for (int i=0; i<3; i++)
    {
        CHECK_VX_STATUS(vxReleaseImage(&virtImages[i]));
    }

    return graph;
}

int camera_tampering_openvx(
        const CmdParserMotionDetection &cmdparser)
{
    vx_uint32 imgWidth, imgHeight;
    vx_uint32 imgNumber;
    vx_uint32 imgCount;
    vx_uint32 threshold = 0;        // threshold used by size filter to filter out small components(noise), it's the count of pixels
    bool      mergeBoxes = false;   // a flag to decide if adjacent boxes should be merged if they're close enough to each other
    bool      scaleImage = false;   // a flag to indicate if input image can be scaled down to save computations
    bool      scaleFlag = false;    // a flag to indicate if input image is scaled down to save computations, when scaleImage is TRUE and input image width is larger than 640, this flag will be set as TRUE
    cv::Mat   input, output, dstImg, labelImg;
    cv::VideoCapture cap;           // used to read from input video file

    cv::VideoWriter videoWriter;    // used to write result to output video file

    int       scaleFactor = 1;

    PERFPROF_REGION_DEFINE(vxVerifyGraph);
    PERFPROF_REGION_DEFINE(vxProcessGraph);
    PERFPROF_REGION_DEFINE(Frame);
    PERFPROF_REGION_DEFINE(ReadFrame);
    PERFPROF_REGION_DEFINE(ProcessFrame);

    // Process input arguments
    imgNumber   = cmdparser.max_frames.getValue();
    threshold   = cmdparser.threshold.getValue();
    mergeBoxes  = cmdparser.merge.getValue();
    scaleImage  = cmdparser.scale.getValue();

    unsigned int ct_enable = cmdparser.ct_enable.getValue();
    float ct_ratio_threshold = cmdparser.ct_ratio_threshold.getValue();
    float ct_scale = cmdparser.ct_scale.getValue();

    std::cout <<"[input setting] imgNumber = " << imgNumber << " threshold = " << threshold <<
		" merge = " << mergeBoxes << " scale = " << scaleImage <<
		" ct_ratio_threshold = " << ct_ratio_threshold <<
		" ct_scale = " << ct_scale << "\n";

    // Read from input video file, convert each frame to RGB
    string inputName = cmdparser.input.getValue();
    if(inputName.size() == 1 && isdigit(inputName[0]))
    {
        unsigned int cameraId = 0;
        stringstream ss(inputName);
        ss >> cameraId;
        cap.open(cameraId);

        if(!cap.isOpened())
        {
            std::cerr << "[ ERROR ] Cannot open Camera ID " << cameraId << "\n";
            return 1;
        }
    }
    else
    {
        cap.open(cmdparser.input.getValue());

        if(!cap.isOpened())
        {
            std::cerr << "[ ERROR ] Cannot open input video file " << cmdparser.input.getValue() << "\n";
            return 1;
        }
    }
    cap.set(cv::CAP_PROP_CONVERT_RGB, 1);
    vx_df_image color = VX_DF_IMAGE_RGB;

    // Create context and images
    // Note: It is workaround to fix opencl driver init failure issue on some HWs
    //       vxCreateContext must be ahead of 'cap >> input'
    //       OCL driver error in cl::Platform::get() in clhal.cpp:Init();
    vx_context context = vxCreateContext();
    CHECK_VX_OBJ(context);

    // Get image resolution by reading the first video frame
    PERFPROF_REGION_BEGIN(ReadFrame);
    cap >> input;
    PERFPROF_REGION_END(ReadFrame);

    if(input.empty())
    {
        std::cerr << "[ ERROR ] Cannot read the first frame from video\n";
        return 1;
    }

    imgWidth    = input.cols;
    imgHeight   = input.rows;

    std::cout << "Video frame size: " << imgWidth << "x" << imgHeight << "\n";

    // Write result to a video file
    std::string outputName;
    if (cmdparser.output.isSet())
    {
        outputName = cmdparser.output.getValue();
    }
    else
    {
        outputName = cmdparser.input.getValue();
        std::size_t pos = outputName.rfind("/");

        if (pos == std::string::npos)
        {
            outputName = "output." + outputName;
        }
        else
        {
            outputName.insert(pos + 1, "output.");
        }
    }

    videoWriter.open(outputName, cap.get(cv::CAP_PROP_FOURCC), 30, input.size());

    if(!videoWriter.isOpened())
    {
        std::cerr << "[ ERROR ] Cannot open output video file " << outputName << "\n";
        return -1;
    }

    // Verify image & object size
    if ((imgWidth <= 0) || (imgHeight <= 0))
    {
        std::cerr << "Image resolution should be larger than 0\n";
        return -1;
    }

    // Images
    dstImg.create(cv::Size(imgWidth, imgHeight), CV_8UC3);

    // Register callback to receive diagnostics messages from OpenVX run-time
    // Predefined function IntelVXSample::errorReceiver is used from common sample infrastructure,
    // and just put all the messages to std::cerr to be read by the user.
    vxRegisterLogCallback(context, IntelVXSample::errorReceiver, vx_true_e);

    // To use two user-nodes that is compiled separately, vxLoadKernels should be called
    // with the name of the library without prefix (e.g. lib) or suffix (e.g. so).
    CHECK_VX_STATUS(vxLoadKernels(context, "camera_tampering_user_nodes_module"));

    std::vector<vx_node>    vxNodes;
    std::vector<vx_image>   vxImages;
    std::vector<vx_array>   vxArrays;

    // Set threshold which is used by size filter, 0 is the default value: 1% of total pixels in the whole image
    if (threshold == 0)
    {
        threshold = (imgWidth * imgHeight) / SIZE_FILTER_DEFAULT_THRESHOLD_DIVISOR;
    }

    vx_float64 countNonZero = 0;
    vx_scalar countNonZero_scalar = vxCreateScalar(context, VX_TYPE_FLOAT64, &countNonZero);

    vx_graph graph = CreateCameraTamperingGraph(context, imgWidth, imgHeight, color, threshold, scaleImage, scaleFlag, scaleFactor, input, output, labelImg, countNonZero_scalar, cmdparser, vxNodes, vxImages, vxArrays);

    // Verify graph
    PERFPROF_REGION_BEGIN(vxVerifyGraph);
    CHECK_VX_STATUS(vxVerifyGraph(graph));
    PERFPROF_REGION_END(vxVerifyGraph);

    IntelVXSample::logger(1) << "[ INFO ] Verified graph " << graph << "\n";

    // A loop to process graph frame by frame
    for (imgCount=0; /* loop termination criteria is checked inside the loop body before reading next frame */; imgCount++)
    {
        PERFPROF_REGION_AUTO(Frame);

        IntelVXSample::logger(1) << "Frame " << imgCount << "\n";

        PERFPROF_REGION_BEGIN(ProcessFrame);

        // Process graph
        PERFPROF_REGION_BEGIN(vxProcessGraph);
        CHECK_VX_STATUS(vxProcessGraph(graph));
        PERFPROF_REGION_END(vxProcessGraph);

        // Get a copy of input image, draw result rectangles on it, then write to output file
        input.copyTo(dstImg);

        if (ct_enable)
	{
            //Camera tampering process
	    vxReadScalarValue(countNonZero_scalar, &countNonZero);

            int tampering_type = TamperingType::NO_TAMPERING;//OCCLUSION
            vx_float64 edge_count_ratio = countNonZero /
					((vx_float64) input.size().area()*ct_scale*ct_scale);
            if (edge_count_ratio < ct_ratio_threshold)
	    {
                //IntelVXSample::logger(0) << "Tampering \n";
                tampering_type = TamperingType::OCCLUSION | TamperingType::DEFOCUS;
            }
            //IntelVXSample::logger(0) << "count :" << countNonZero << ",ratio :" <<
            //    edge_count_ratio << ",ct_ratio_th: "<< ct_ratio_threshold << "\n";
            DrawFrameInfo(dstImg, tampering_type);
        }
        // Get rectangles from vx_array
        vector<vx_rectangle_t>   objectList;
        vx_rectangle_t           *pObjectListPtr = NULL;
        vx_size                  stride          = 0;
        vx_size                  numItems        = 0;

        CHECK_VX_STATUS(vxQueryArray(vxArrays.back(), VX_ARRAY_ATTRIBUTE_NUMITEMS, &numItems, sizeof(numItems)));

        if ((numItems > 0) && (numItems <= MAXIMUM_RECTANGLE_NUMBER))
        {
            vx_map_id map_id;
            CHECK_VX_STATUS(vxMapArrayRange(vxArrays.back(), 0, numItems, &map_id, &stride, reinterpret_cast<void**>(&pObjectListPtr), VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));

            objectList.reserve(numItems);
            std::copy(pObjectListPtr, pObjectListPtr + numItems, std::back_inserter(objectList));

            CHECK_VX_STATUS(vxUnmapArrayRange(vxArrays.back(), map_id));

            // Merge overlapping rectangles
            if (mergeBoxes == true)
            {
                vx_uint32 thresholdX = imgWidth / HORIZONTAL_MERGE_NEAR_THRESHOLD_DIVISOR;  // The threshold to decide if 2 horizontally neighboring boxes can be merged or not
                vx_uint32 thresholdY = imgHeight / VERTICAL_MERGE_NEAR_THRESHOLD_DIVISOR;   // The threshold to decide if 2 vertically neighboring boxes can be merged or not

                // Merge overlapped bounding boxes
                MergeBoundingBoxes(objectList, thresholdX, thresholdY);
            }
        }
        else
        {
            IntelVXSample::logger(1) << "No object detected!\n";
        }

        PERFPROF_REGION_END(ProcessFrame);

        // Draw rectangles
        DrawBoundingBoxes(dstImg, objectList, scaleFactor);

        // Show the video on screen
        if (!cmdparser.no_show.isSet())
        {
            cv::imshow("Camera Tampering Result", dstImg);
            int key = cv::waitKey(cmdparser.frame_wait.getValue()) & 0xff;
            if(key == 27)   // 27 is ESC, press ESC to exit
            {
                break;
            }
        }

        // Save binary image with bounding rectangles to output file
        videoWriter.write(dstImg);

        // Read next frame
        if (imgNumber != 0 && imgCount == (imgNumber - 1))
        {
            std::cout << "Reached the maximum number of frames requested (" << imgNumber << ")\n";
            break;
        }

        PERFPROF_REGION_BEGIN(ReadFrame);

        void *base;
        vx_imagepatch_addressing_t src_addr;
        vx_rectangle_t rect = {0u, 0u, (unsigned int)imgWidth, (unsigned int)imgHeight};

        // Refresh input buffer
        base = nullptr;
        vx_map_id map_id;
        CHECK_VX_STATUS(vxMapImagePatch(vxImages.front(), &rect, 0, &map_id, &src_addr,
				&base, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));
        assert(base == input.data);
        cap >> input;

        CHECK_VX_STATUS(vxUnmapImagePatch(vxImages.front(), map_id));

        if(input.empty())
        {
            std::cout << "Reached end of video file\n";
	    PERFPROF_REGION_END(ReadFrame);
            break;
        }
        PERFPROF_REGION_END(ReadFrame);
        cv::waitKey(16);
    }

    for (int i=0; i<vxNodes.size(); i++)
    {
        CHECK_VX_STATUS(vxReleaseNode(&vxNodes[i]));
    }

    for (int i=0; i<vxImages.size(); i++)
    {
        CHECK_VX_STATUS(vxReleaseImage(&vxImages[i]));
    }

    for (int i=0; i<vxArrays.size(); i++)
    {
        CHECK_VX_STATUS(vxReleaseArray(&vxArrays[i]));
    }

    CHECK_VX_STATUS(vxReleaseGraph(&graph));
    CHECK_VX_STATUS(vxReleaseContext(&context));

    videoWriter.release();

    return 0;
}

int main(
        int argc,
        const char** argv)
{
    try
    {
        //Parse command line arguments.
        CmdParserMotionDetection cmdparser(argc, argv);
        cmdparser.parse();

        if(cmdparser.help.isSet())
        {
            // Immediately exit if user wanted to see the usage information only.
            return 0;
        }

        return camera_tampering_openvx(cmdparser);

    }

    catch(const CmdParser::Error& error)
    {
        cerr
            << "[ ERROR ] In command line: " << error.what() << "\n"
            << "Run " << argv[0] << " -h for usage info.\n";
        return 1;
    }
    catch(const std::exception& error)
    {
        cerr << "[ ERROR ] " << error.what() << "\n";
        return 1;
    }
    catch(...)
    {
        cerr << "[ ERROR ] Unknown/internal exception happened.\n";
        return 1;
    }
}
