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

#include <stdio.h>
#include <vector>
#include <string>
#include <cmath>
#include <VX/vx.h>
#include <VX/vx_api.h>
#include <VX/vx_intel_volatile.h>
#include <VX/vx_compatibility.h>
#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/hetero.hpp>
#include "camera_tampering_user_nodes_lib.h"
#include "camera_tampering_common.h"
#include "libwl_ctd.h"

using namespace std;
#define CANNY_THRESHOLD1    100
#define CANNY_THRESHOLD2    200
#define CT_SCALE 1
#define CT_RATIO_THRESHOLD 0.01
typedef struct _VX_MD_Session
{
    vx_context    context;
    vx_graph      graph;
    vx_array      rectangles;   // Output bounding rectangles on the moving objects

    vx_float64 countNonZero;
    vx_scalar countNonZero_scalar;

    bool          inited;
    int           width;
    int           height;
    int           color;
    vx_uint32     threshold;    // The threshold to filter out little components(noise), it's the number of pixels, here 0 is the default number
    bool          mergeBoxes;   // A flag to indicate if it's necessary to merge overlapped boxes, or those boxes very close to each other
    bool          scaleImage;   // A flag to indicate if it's necessary to scale larger input images into smaller images to improve performance
    bool          scaleFlag;    // A flag to indicate if scale image node is used in the graph for performance optimization, , when scaleImage is TRUE and input image width is larger than 640, this flag will be set as TRUE
    int           scaleFactor;  // Scale factor, it's the same for both horizontal and vertical directions
    vx_enum       heterogeneity;// A flag to set OpenVX Nodes to run on heterogenous hardware platforms: CPU, GPU, IPU, CVE, etc.
    std::string heterogeneity_config_file;
    std::vector<vx_node>    vxNodes;
    std::vector<vx_image>   vxImages;
    std::vector<vx_array>   vxArrays;
}VX_MD_Session;

vx_status assignNodeTarget(vx_context context, vx_node node, const char* nodename, vx_enum heterogeneity)
{
    vx_status status = VX_FAILURE;

    vx_target_intel target_cpu    = vxGetTargetByNameIntel(context, "intel.cpu");
    vx_target_intel target_cpuext = vxGetTargetByNameIntel(context, "intel.ext");
    vx_target_intel target_gpu    = vxGetTargetByNameIntel(context, "intel.gpu");
    vx_target_intel target_ipu    = vxGetTargetByNameIntel(context, "intel.ipu4m");

    switch(heterogeneity)
    {
        case WL_HETER_CPU_ONLY:
            if(target_cpu || target_cpuext)
            {
                if (target_cpu)
                {
                    status = vxAssignNodeAffinityIntel(node, target_cpu);
                    if (status == VX_SUCCESS)
                    {
                        IntelVXSample::logger(0) << "[ INFO ] " << nodename << " is assigned to CPU" << '\n';
                    }
                }

                if((status != VX_SUCCESS) && (target_cpuext))
                {
                    status = vxAssignNodeAffinityIntel(node, target_cpuext);
                    CHECK_VX_STATUS(status);
                    IntelVXSample::logger(0) << "[ INFO ] " << nodename << " is assigned to CPUExt" << '\n';
                }
            }
            else
            {
                std::cerr << "[ ERROR ] CPU/CPUEXT not supported: " << "\n";
                status = VX_ERROR_NOT_SUPPORTED;
                CHECK_VX_STATUS(status);
            }
            break;

        case WL_HETER_GPU_ONLY:
            if(target_gpu)
            {
                status = vxAssignNodeAffinityIntel(node, target_gpu);
                CHECK_VX_STATUS(status);
                IntelVXSample::logger(0) << "[ INFO ] " << nodename << " is assigned to GPU" << '\n';
            }
            else
            {
                std::cerr << "[ ERROR ] GPU not supported: " << "\n";
                status = VX_ERROR_NOT_SUPPORTED;
                CHECK_VX_STATUS(status);
            }
            break;

        case WL_HETER_IPU_ONLY:
            if(target_ipu)
            {
                status = vxAssignNodeAffinityIntel(node, target_ipu);
                CHECK_VX_STATUS(status);
                IntelVXSample::logger(0) << "[ INFO ] " << nodename << " is assigned to IPU" << '\n';
            }
            else
            {
                std::cerr << "[ ERROR ] IPU not supported: " << "\n";
                status = VX_ERROR_NOT_SUPPORTED;
                CHECK_VX_STATUS(status);
            }
            break;

        case WL_HETER_CPU_PREF:
            if(target_cpu)
            {
                status = vxAssignNodeAffinityIntel(node, target_cpu);
                if (status == VX_SUCCESS)
                {
                    IntelVXSample::logger(0) << "[ INFO ] " << nodename << " is assigned to CPU" << '\n';
                }
            }

            if ((status != VX_SUCCESS) && target_cpuext)
            {
                status = vxAssignNodeAffinityIntel(node, target_cpuext);
                if (status == VX_SUCCESS)
                {
                    IntelVXSample::logger(0) << "[ INFO ] " << nodename << " is assigned to CPUExt" << '\n';
                }
            }

            if ((status != VX_SUCCESS) && target_gpu)
            {
                status = vxAssignNodeAffinityIntel(node, target_gpu);
                if (status == VX_SUCCESS)
                {
                    IntelVXSample::logger(0) << "[ INFO ] " << nodename << " is assigned to GPU" << '\n';
                }
            }

            if ((status != VX_SUCCESS) && target_ipu)
            {
                status = vxAssignNodeAffinityIntel(node, target_ipu);
                if (status == VX_SUCCESS)
                {
                    IntelVXSample::logger(0) << "[ INFO ] " << nodename << " is assigned to IPU" << '\n';
                }
            }
            break;

        case WL_HETER_GPU_PREF:
            if(target_gpu)
            {
                status = vxAssignNodeAffinityIntel(node, target_gpu);
                if (status == VX_SUCCESS)
                {
                    IntelVXSample::logger(0) << "[ INFO ] " << nodename << " is assigned to GPU" << '\n';
                }
            }

            if((status != VX_SUCCESS) && target_cpu)
            {
                status = vxAssignNodeAffinityIntel(node, target_cpu);
                if (status == VX_SUCCESS)
                {
                    IntelVXSample::logger(0) << "[ INFO ] " << nodename << " is assigned to CPU" << '\n';
                }
            }

            if ((status != VX_SUCCESS) && target_cpuext)
            {
                status = vxAssignNodeAffinityIntel(node, target_cpuext);
                if (status == VX_SUCCESS)
                {
                    IntelVXSample::logger(0) << "[ INFO ] " << nodename << " is assigned to CPUExt" << '\n';
                }
            }

            if ((status != VX_SUCCESS) && target_ipu)
            {
                status = vxAssignNodeAffinityIntel(node, target_ipu);
                if (status == VX_SUCCESS)
                {
                    IntelVXSample::logger(0) << "[ INFO ] " << nodename << " is assigned to IPU" << '\n';
                }
            }
            break;

        case WL_HETER_IPU_PREF:
            if(target_ipu)
            {
                status = vxAssignNodeAffinityIntel(node, target_ipu);
                if (status == VX_SUCCESS)
                {
                    IntelVXSample::logger(0) << "[ INFO ] " << nodename << " is assigned to IPU" << '\n';
                }
            }

            if((status != VX_SUCCESS) && target_cpu)
            {
                status = vxAssignNodeAffinityIntel(node, target_cpu);
                if (status == VX_SUCCESS)
                {
                    IntelVXSample::logger(0) << "[ INFO ] " << nodename << " is assigned to CPU" << '\n';
                }
            }

            if ((status != VX_SUCCESS) && target_cpuext)
            {
                status = vxAssignNodeAffinityIntel(node, target_cpuext);
                if (status == VX_SUCCESS)
                {
                    IntelVXSample::logger(0) << "[ INFO ] " << nodename << " is assigned to CPUExt" << '\n';
                }
            }

            if ((status != VX_SUCCESS) && target_gpu)
            {
                status = vxAssignNodeAffinityIntel(node, target_gpu);
                if (status == VX_SUCCESS)
                {
                    IntelVXSample::logger(0) << "[ INFO ] " << nodename << " is assigned to GPU" << '\n';
                }
            }
            break;

        default:
            std::cerr << "[ ERROR ] CVE not supported: " << "\n";
            status = VX_ERROR_NOT_SUPPORTED;
            break;
    }

    return status;
}

/* There's an optional mode to scale large input images down before processing
 * to save computations. When this mode is enabled, this function creates the
 * nodes for scaling. */
void CreateScaleNodes(
    vx_context              context,
    vx_graph                graph,
    vx_df_image             color,          // input image format: RGB, NV12 or IYUV(I420)
    vx_enum                 heterogeneity,  // heterogeneity configuration
    vx_image                input,          // input image
    vx_image                scaleImages[6], // the images used by the scaling nodes
    vx_image                scaleOutput,    // output image after scaling
    std::vector<vx_node>    &vxNodes)
{
    vx_node channelExtractNodes[3];
    vx_node scaleImageNodes[3];
    vx_node channelCombineNode;

    // Step 1: channel extract
    if (color == VX_DF_IMAGE_RGB)
    {
        channelExtractNodes[0] = vxChannelExtractNode(graph, input, VX_CHANNEL_R, scaleImages[0]);
        CHECK_VX_OBJ(channelExtractNodes[0]);
        assignNodeTarget(context, channelExtractNodes[0], "vxChannelExtractNode(R)", heterogeneity);
        vxNodes.push_back(channelExtractNodes[0]);

        channelExtractNodes[1] = vxChannelExtractNode(graph, input, VX_CHANNEL_G, scaleImages[1]);
        CHECK_VX_OBJ(channelExtractNodes[1]);
        assignNodeTarget(context, channelExtractNodes[1], "vxChannelExtractNode(G)", heterogeneity);
        vxNodes.push_back(channelExtractNodes[1]);

        channelExtractNodes[2] = vxChannelExtractNode(graph, input, VX_CHANNEL_B, scaleImages[2]);
        CHECK_VX_OBJ(channelExtractNodes[2]);
        assignNodeTarget(context, channelExtractNodes[2], "vxChannelExtractNode(B)", heterogeneity);
        vxNodes.push_back(channelExtractNodes[2]);
    }
    else if ((color == VX_DF_IMAGE_NV12) || (color == VX_DF_IMAGE_IYUV))
    {
        channelExtractNodes[0] = vxChannelExtractNode(graph, input, VX_CHANNEL_Y, scaleImages[0]);
        CHECK_VX_OBJ(channelExtractNodes[0]);
        assignNodeTarget(context, channelExtractNodes[0], "vxChannelExtractNode(Y)", heterogeneity);
        vxNodes.push_back(channelExtractNodes[0]);

        channelExtractNodes[1] = vxChannelExtractNode(graph, input, VX_CHANNEL_U, scaleImages[1]);
        CHECK_VX_OBJ(channelExtractNodes[1]);
        assignNodeTarget(context, channelExtractNodes[1], "vxChannelExtractNode(U)", heterogeneity);
        vxNodes.push_back(channelExtractNodes[1]);

        channelExtractNodes[2] = vxChannelExtractNode(graph, input, VX_CHANNEL_V, scaleImages[2]);
        CHECK_VX_OBJ(channelExtractNodes[2]);
        assignNodeTarget(context, channelExtractNodes[2], "vxChannelExtractNode(V)", heterogeneity);
        vxNodes.push_back(channelExtractNodes[2]);
    }

    // Step 2: scale image
    scaleImageNodes[0] = vxScaleImageNode(graph, scaleImages[0], scaleImages[3], VX_INTERPOLATION_TYPE_BILINEAR);
    CHECK_VX_OBJ(scaleImageNodes[0]);
    assignNodeTarget(context, scaleImageNodes[0], "vxScaleImageNode(0)", heterogeneity);
    vxNodes.push_back(scaleImageNodes[0]);

    scaleImageNodes[1] = vxScaleImageNode(graph, scaleImages[1], scaleImages[4], VX_INTERPOLATION_TYPE_BILINEAR);
    CHECK_VX_OBJ(scaleImageNodes[1]);
    assignNodeTarget(context, scaleImageNodes[1], "vxScaleImageNode(1)", heterogeneity);
    vxNodes.push_back(scaleImageNodes[1]);

    scaleImageNodes[2] = vxScaleImageNode(graph, scaleImages[2], scaleImages[5], VX_INTERPOLATION_TYPE_BILINEAR);
    CHECK_VX_OBJ(scaleImageNodes[2]);
    assignNodeTarget(context, scaleImageNodes[2], "vxScaleImageNode(2)", heterogeneity);
    vxNodes.push_back(scaleImageNodes[2]);

    // Step 3: channel combine
    channelCombineNode = vxChannelCombineNode(graph, scaleImages[3], scaleImages[4], scaleImages[5], NULL, scaleOutput);
    CHECK_VX_OBJ(channelCombineNode);
    assignNodeTarget(context, channelCombineNode, "vxChannelCombineNode", heterogeneity);
    vxNodes.push_back(channelCombineNode);

    // Step 4: release the virtual scale images
    for (int i=0; i<6; i++)
    {
        CHECK_VX_STATUS(vxReleaseImage(&scaleImages[i]));
    }
}

/* This function create the graph for motion detection */
vx_graph CreateCameraTamperingGraph(
    vx_context              context,
    int                     width,          // input image width
    int                     height,         // input image height
    vx_scalar               &countNonZero_scalar,
    vx_df_image             color,          // input image format: RGB, NV12 or IYUV(I420)
    vx_uint32               threshold,      // threshold for size filter
    bool                    scaleImage,     // if input image should be checked to make sure if scaling down is needed
    bool                    &scaleFlag,     // if input image should be scaled down before processing
    vx_enum                 heterogeneity,  // heterogeneity configuration
    std::string heterogeneity_config_file,  // heterogeneity configuration file
    int                     &scaleFactor,   // scale factor in X & Y axis
    std::vector<vx_node>    &vxNodes,       // vector of nodes
    std::vector<vx_image>   &vxImages,      // vector of images
    std::vector<vx_array>   &vxArrays)      // vector of arrays
{
#if (VX_VERSION == VX_VERSION_1_0)
    vx_uint32 numOfTargets = 0;
    CHECK_VX_STATUS(vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_TARGETS, &numOfTargets, sizeof(numOfTargets)));
    IntelVXSample::logger(0) << "[ INFO ] Number of targets (VX_CONTEXT_ATTRIBUTE_TARGETS): " << numOfTargets << '\n';
    for(int i = 0; i < numOfTargets; ++i)
    {
        IntelVXSample::logger(0) << "[ INFO ]     Target[" << i << "] name: ";
        vx_target_intel target = vxGetTargetByIndexIntel(context, i);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)target));
        vx_char targetName[VX_MAX_TARGET_NAME];
        CHECK_VX_STATUS(vxQueryTargetIntel(target, VX_TARGET_ATTRIBUTE_NAME_INTEL, targetName, sizeof(targetName)));
        IntelVXSample::logger(0) << targetName << '\n';
    }
#else
    IntelVXSample::HeteroScheduleConfig heteroConfig(heterogeneity_config_file);
    heteroConfig.pupulateSupportedTargets();
#endif

    // GPU can support grayscale input format for background subtraction MOG2 node. It has better performance compared with RGB input.
    bool bgsubMOG2_gray_input_flag = false;
    char *pcEnv = std::getenv("BGSUBMOG2_GRAYSCALE_GPU");

    if (pcEnv != NULL)
    {
        string sEnv(pcEnv);

        if ((sEnv.compare("1") == 0) && ((heterogeneity == WL_HETER_GPU_ONLY) || (heterogeneity == WL_HETER_GPU_PREF)))
        {
            bgsubMOG2_gray_input_flag = true;
        }
    }

    vx_graph graph = vxCreateGraph(context);
    CHECK_VX_OBJ(graph);

    // Initialize
    vx_image input = vxCreateImage(context, width, height, color);

    CHECK_VX_OBJ(input);
    vxImages.push_back(input);

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
        CreateScaleNodes(context, graph, color, heterogeneity, input, scaleImages, scaleOutput, vxNodes);

        // Update threshold
        threshold = (vx_uint32)(threshold / (scaleFactor * scaleFactor));
    }

    vx_image output = vxCreateImage(context, internalWidth, internalHeight, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(output);
    vxImages.push_back(output);
    vx_image labelImg = vxCreateImage(context, internalWidth, internalHeight, VX_DF_IMAGE_U32);
    CHECK_VX_OBJ(labelImg);
    vxImages.push_back(labelImg);
    vx_array rectangles = vxCreateArray(context, VX_TYPE_RECTANGLE, 4096);
    CHECK_VX_OBJ(rectangles);
    vxArrays.push_back(rectangles);

    // Virtual images which are connected to internal nodes
    vx_image virtImages[3];

    for (int i=0; i<3; i++)
    {
        virtImages[i] = vxCreateVirtualImage(graph, internalWidth, internalHeight, VX_DF_IMAGE_U8);
        CHECK_VX_OBJ(virtImages[i]);
    }

    vx_image input_image = NULL;
    vx_image output_image = NULL;
    vx_image bgsubMOG2_input_image = NULL;
    vx_image tempImages[2] = {NULL, NULL};

    if (scaleFlag == true)
    {
        input_image = scaleOutput;
    }
    else
    {
        input_image = input;
    }

    if (color == VX_DF_IMAGE_RGB)
    {
        if (bgsubMOG2_gray_input_flag == true)
        {
            vx_image colorConvert_input_image = input_image;
            tempImages[0] = vxCreateVirtualImage(graph, internalWidth, internalHeight, VX_DF_IMAGE_NV12);
            CHECK_VX_OBJ(tempImages[0]);
            tempImages[1] = vxCreateVirtualImage(graph, internalWidth, internalHeight, VX_DF_IMAGE_U8);
            CHECK_VX_OBJ(tempImages[1]);

            vx_node colorConvertNode = vxColorConvertNode(graph, colorConvert_input_image, tempImages[0]);
            CHECK_VX_OBJ(colorConvertNode);
            assignNodeTarget(context, colorConvertNode, "vxColorConvertNode", heterogeneity);
            vxNodes.push_back(colorConvertNode);

            vx_node channelExtractNode = vxChannelExtractNode(graph, tempImages[0], VX_CHANNEL_Y, tempImages[1]);
            CHECK_VX_OBJ(channelExtractNode);
            assignNodeTarget(context, channelExtractNode, "vxChannelExtractNode", heterogeneity);
            vxNodes.push_back(channelExtractNode);

            bgsubMOG2_input_image = tempImages[1];
        }
        else
        {
            bgsubMOG2_input_image = input_image;
        }
    }
    else
    {
        if (bgsubMOG2_gray_input_flag == true)
        {
            output_image = vxCreateVirtualImage(graph, internalWidth, internalHeight, VX_DF_IMAGE_U8);
            CHECK_VX_OBJ(output_image);
            vx_node channelExtactNode = vxChannelExtractNode(graph, input_image, VX_CHANNEL_Y, output_image);
            CHECK_VX_OBJ(channelExtactNode);
            assignNodeTarget(context, channelExtactNode, "vxChannelExtractNode", heterogeneity);
            vxNodes.push_back(channelExtactNode);
        }
        else
        {
            output_image = vxCreateVirtualImage(graph, internalWidth, internalHeight, VX_DF_IMAGE_RGB);
            CHECK_VX_OBJ(output_image);

            vx_node colorConvertNode = vxColorConvertNode(graph, input_image, output_image);
            CHECK_VX_OBJ(colorConvertNode);
            assignNodeTarget(context, colorConvertNode, "vxColorConvertNode", heterogeneity);
            vxNodes.push_back(colorConvertNode);
        }

        bgsubMOG2_input_image = output_image;
    }

// camera tampering
    float ct_scale = 1;
    vx_uint32 w2 = width * ct_scale;
    vx_uint32 h2 = ((vx_uint32)(height * ct_scale)) & 0xfffffffe;

    vx_image ct_input_image = NULL;
    vx_image vxImgYUV = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_IYUV);
    vx_image vxImgGray = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_image vxImgScaleGray = vxCreateVirtualImage(graph, w2, h2, VX_DF_IMAGE_U8);
    vx_image vxImgOutput = vxCreateVirtualImage(graph, w2, h2, VX_DF_IMAGE_U8);
    if (color == VX_DF_IMAGE_RGB) {
        vx_node ct_node1 = vxColorConvertNode(graph, input, vxImgYUV );

        CHECK_VX_OBJ(ct_node1);
        assignNodeTarget(context,ct_node1,"CT_vxColorConvertNode",heterogeneity);
        vxNodes.push_back(ct_node1);
        ct_input_image = vxImgYUV;
    } else {
        ct_input_image = input;
    }
    vx_node ct_node2 = vxChannelExtractNode(graph, ct_input_image, VX_CHANNEL_Y, vxImgGray);
    CHECK_VX_OBJ(ct_node2);
    assignNodeTarget(context,ct_node2,"CT_vxChannelExtractNode",heterogeneity);
    vxNodes.push_back(ct_node2);


    vx_node ct_node3 = vxScaleImageNode(graph, vxImgGray, vxImgScaleGray, VX_INTERPOLATION_TYPE_AREA);//VX_INTERPOLATION_TYPE_AREA);
    CHECK_VX_OBJ(ct_node3);
    assignNodeTarget(context,ct_node3,"CT_vxScaleImageNode",heterogeneity);
    vxNodes.push_back(ct_node3);


    vx_threshold hyst = vxCreateThreshold(context, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_UINT8);
    vx_int32 lower = CANNY_THRESHOLD1, upper = CANNY_THRESHOLD2;
    vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, &lower, sizeof(lower));
    vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, &upper, sizeof(upper));

    vx_node ct_node4 = vxCannyEdgeDetectorNode(graph, vxImgScaleGray, hyst, 3, VX_NORM_L1, vxImgOutput);
    CHECK_VX_OBJ(ct_node4);
    assignNodeTarget(context,ct_node4,"CT_vxCannyEdgeDetectorNode",heterogeneity);
    vxNodes.push_back(ct_node4);


    vx_node ct_node5 = vxUserCountNonZeroNode(graph, vxImgOutput, countNonZero_scalar);
    CHECK_VX_OBJ(ct_node5);
    vxNodes.push_back(ct_node5);

    CHECK_VX_STATUS(vxReleaseImage(&vxImgYUV));
    CHECK_VX_STATUS(vxReleaseImage(&vxImgGray));
    CHECK_VX_STATUS(vxReleaseImage(&vxImgScaleGray));
    CHECK_VX_STATUS(vxReleaseImage(&vxImgOutput));

    // Background subtraction MOG2 node
    vx_bg_state_intel bgState = vxCreateBGStateIntel(context, internalWidth, internalHeight, 50, 5, 15.0, 4.0, 75.0, 4.0f * 4.0f, 0.9f, 3.0f * 3.0f, 0.05f, 0, 0, 0.5);
    CHECK_VX_OBJ(bgState);

    vx_node bgsubMOG2Node = vxBackgroundSubMOG2NodeIntel(graph, bgsubMOG2_input_image, bgState, virtImages[0]);
    CHECK_VX_OBJ(bgsubMOG2Node);
    assignNodeTarget(context, bgsubMOG2Node, "vxBackgroundSubMOG2Node", heterogeneity);
    vxNodes.push_back(bgsubMOG2Node);

    if ((color == VX_DF_IMAGE_RGB) && (scaleFlag == true))
    {
        CHECK_VX_STATUS(vxReleaseImage(&scaleOutput));
    }

    if (output_image)
    {
        CHECK_VX_STATUS(vxReleaseImage(&output_image));
    }

    if ((color == VX_DF_IMAGE_RGB) && (bgsubMOG2_gray_input_flag == true))
    {
        CHECK_VX_STATUS(vxReleaseImage(&tempImages[0]));
        CHECK_VX_STATUS(vxReleaseImage(&tempImages[1]));
    }

    // Morphology nodes
    vx_node dilateNode1 = vxDilate3x3Node(graph, virtImages[0], virtImages[1]);
    CHECK_VX_OBJ(dilateNode1);
    assignNodeTarget(context, dilateNode1, "vxDilate3x3Node(1)", heterogeneity);
    vxNodes.push_back(dilateNode1);

    vx_node erodeNode = vxErode3x3Node(graph, virtImages[1], virtImages[2]);
    CHECK_VX_OBJ(erodeNode);
    assignNodeTarget(context, erodeNode, "vxErode3x3Node", heterogeneity);
    vxNodes.push_back(erodeNode);

    vx_node dilateNode2 = vxDilate3x3Node(graph, virtImages[2], output);
    CHECK_VX_OBJ(dilateNode2);
    assignNodeTarget(context, dilateNode2, "vxDilate3x3Node(2)", heterogeneity);
    vxNodes.push_back(dilateNode2);

    // Connected component labeling node
    vx_node connectedComponentLabelingNode = vxConnectedComponentLabelingNode(graph, output, threshold, labelImg, rectangles);
    CHECK_VX_OBJ(connectedComponentLabelingNode);
    vxNodes.push_back(connectedComponentLabelingNode);

    // Release virtual images
    for (int i=0; i<3; i++)
    {
        CHECK_VX_STATUS(vxReleaseImage(&virtImages[i]));
    }

    return graph;
}

static WL_Status vx_md_init (void *workload)
{
    vx_df_image color = 0;

    //validate parameters
    if (workload == NULL)
        return WL_BAD_VALUE;

    VX_MD_Session *md = (VX_MD_Session*) workload;

    if (md->inited == true)
        return WL_OK;

    if (md->color == WL_COLOR_UNKNOWN)
    {
        std::cerr << "[ ERROR ] Color Format not supported: " << "\n";
        return WL_NOT_SUPPORTED;
    }

    if (md->color == WL_COLOR_RGB)
    {
        color = VX_DF_IMAGE_RGB;
    }
    else if (md->color == WL_COLOR_NV12)
    {
        color = VX_DF_IMAGE_NV12;
    }
    else if (md->color == WL_COLOR_I420)
    {
        color = VX_DF_IMAGE_IYUV;
    }

    // Set threshold which is used by size filter, 0 is the default value: 1% of total pixels in the whole image
    if (md->threshold == 0)
    {
        md->threshold = (md->width * md->height) / SIZE_FILTER_DEFAULT_THRESHOLD_DIVISOR;
    }

    // Register callback to receive diagnostics messages from OpenVX run-time
    // Predefined function IntelVXSample::errorReceiver is used from common sample infrastructure,
    // and just put all the input images to std::cerr to be read by the user.
    vxRegisterLogCallback(md->context, IntelVXSample::errorReceiver, vx_true_e);

    // To use two user-nodes that is compiled separately, vxLoadKernels should be called
    // with the name of the library without prefix (e.g. lib) or suffix (e.g. so).
    CHECK_VX_STATUS(vxLoadKernels(md->context, "camera_tampering_user_nodes_module"));

    md->countNonZero = 0;
    md->countNonZero_scalar = vxCreateScalar(md->context, VX_TYPE_FLOAT64, &(md->countNonZero));


    md->graph = CreateCameraTamperingGraph(md->context, md->width, md->height, md->countNonZero_scalar, color, md->threshold, md->scaleImage, md->scaleFlag, md->heterogeneity, md->heterogeneity_config_file, md->scaleFactor, md->vxNodes, md->vxImages, md->vxArrays);
    if( md->graph == NULL)
    {
        std::cerr << "[ ERROR ] Failed to CreateCameraTamperingGraph: " << "\n";
        return WL_OPERATION_FAIL;
    }

    // Verify graph
    CHECK_VX_STATUS(vxVerifyGraph(md->graph));

    md->inited = true;
    return WL_OK;
}

static WL_Status vx_md_config (void *workload, WL_Cfg_Index index, void* config)
{
    if (workload == NULL)
        return WL_BAD_VALUE;

    VX_MD_Session *md = (VX_MD_Session*) workload;
    switch (index) {
      case WL_CFG_INPUT:
      {
          WL_Image_Info* input = (WL_Image_Info*) config;

          if((input->width < MINIMUM_IMAGE_WIDTH) || (input->width > MAXIMUM_IMAGE_WIDTH) || (input->height < MINIMUM_IMAGE_HEIGHT) || (input->height > MAXIMUM_IMAGE_HEIGHT))
          {
              std::cerr << "[ ERROR ] Image size not supported" << "\n";
              return WL_NOT_SUPPORTED;
          }

          if((input->color <= WL_COLOR_UNKNOWN) || (input->color >= WL_COLOR_NUM))
          {
              std::cerr << "[ ERROR ] Color format not supported" << "\n";
              return WL_NOT_SUPPORTED;
          }

          md->width = input->width;
          md->height = input->height;
          md->color = input->color;
          break;
      }

      case WL_CFG_CAMERATAMPERING:
      {
          WL_MDConfig *mdConfig = (WL_MDConfig *)config;

          if (mdConfig->threshold >= (md->width * md->height))
          {
              std::cerr << "[ ERROR ] The threshold must locate in [0, width x height) pixel" << "\n";
              return WL_NOT_SUPPORTED;
          }

          if ((mdConfig->mergeBoxes != 0) && (mdConfig->mergeBoxes != 1))
          {
            std::cerr << "[ ERROR ] The merge boxes flag must be 0 or 1" << "\n";
            return WL_NOT_SUPPORTED;
          }

          if ((mdConfig->scaleImage != 0) && (mdConfig->scaleImage != 1))
          {
            std::cerr << "[ ERROR ] The merge image flag must be 0 or 1" << "\n";
            return WL_NOT_SUPPORTED;
          }

          md->threshold     = mdConfig->threshold;
          md->mergeBoxes    = mdConfig->mergeBoxes;
          md->scaleImage    = mdConfig->scaleImage;
          break;
      }

      case WL_CFG_HETEROGENEITY:
      {
#if (VX_VERSION == VX_VERSION_1_0)
          vx_enum heterogeneity = *((vx_enum*)config);
          if((heterogeneity <= WL_HETER_UNKNOWN) || (heterogeneity >= WL_HETER_MAX_NUM))
          {
              std::cerr << "[ ERROR ] The heterogeneity type not be supported" << "\n";
              return WL_NOT_SUPPORTED;
          }

          md->heterogeneity = heterogeneity;
#else
          WL_Heterogeneity_Info *hetero_info = (WL_Heterogeneity_Info *)config;
          if((hetero_info->heterogeneity_pref <= WL_HETER_UNKNOWN) ||
		  (hetero_info->heterogeneity_pref >= WL_HETER_MAX_NUM))
          {
              std::cerr << "[ ERROR ] The heterogeneity type not be supported" << "\n";
              return WL_NOT_SUPPORTED;
          }

          md->heterogeneity = hetero_info->heterogeneity_pref;

          if(md->heterogeneity == WL_HETER_CUSTOM)
          {
              if(hetero_info->heterogeneity_config_file == NULL)
              {
                  std::cerr << "[ ERROR ] When heterogeneity property is 9, we must assign the heterogeneity configure file by heterogeneity-config-file property" << "\n";
                  return WL_NOT_SUPPORTED;
              }
              else
              {
                  md->heterogeneity_config_file = hetero_info->heterogeneity_config_file;
                  std::cout << "hetero_config_file: " << md->heterogeneity_config_file << "\n";
              }
          }
#endif
          break;
      }

      case WL_CFG_ROI:
      {
          std::cerr << "[ ERROR ] WL_CFG_ROI is not supported by motion detection workload library" << "\n";
          return WL_NOT_SUPPORTED;
      }

      default:
          break;
    }

    return WL_OK;

}

static WL_Status vx_md_process (void *workload, WL_Images *inbufs, WL_Images *outbufs, void* outmetadata)
{
    vx_uint32 thresholdX;
    vx_uint32 thresholdY;

    if ((workload == NULL) || (inbufs == NULL) || (outmetadata == NULL))
    {
        std::cerr << "[ ERROR ] The parameter cannot be NULL" << "\n";
        return WL_BAD_VALUE;
    }

    VX_MD_Session* md = (VX_MD_Session*) workload;

    vx_rectangle_t rect;
    vx_imagepatch_addressing_t addr;
    vx_uint32 plane = 0;
    void *base_ptr = NULL;

    // Plane 0
    plane           = 0;
    base_ptr        = inbufs->imgs[0].buf + inbufs->imgs[0].offset[0];
    rect.start_x    = 0;
    rect.start_y    = 0;
    rect.end_x      = md->width;
    rect.end_y      = md->height;

    addr.dim_x      = md->width;
    addr.dim_y      = md->height;
    addr.scale_x    = VX_SCALE_UNITY;
    addr.scale_y    = VX_SCALE_UNITY;
    addr.step_x     = 1;
    addr.step_y     = 1;

    if (md->color == WL_COLOR_RGB)
    {
        addr.stride_x   = 3;
    }
    else
    {
        addr.stride_x   = 1;
    }

    addr.stride_y = inbufs->imgs[0].stride[0];
    CHECK_VX_STATUS(vxAccessImagePatch(md->vxImages.front(), &rect, plane, &addr, &base_ptr, VX_WRITE_ONLY));
    CHECK_VX_STATUS(vxCommitImagePatch(md->vxImages.front(), &rect, plane, &addr, base_ptr));

    // Plane 1
    if ((md->color == WL_COLOR_NV12) || (md->color == WL_COLOR_I420))
    {
        plane       = 1;
        base_ptr    = inbufs->imgs[0].buf + inbufs->imgs[0].offset[1];
        rect.end_x  = md->width / 2;
        rect.end_y  = md->height / 2;
        addr.dim_x  = md->width / 2;
        addr.dim_y  = md->height / 2;

        if (md->color == WL_COLOR_NV12)
        {
            addr.stride_x   = 2;
        }
        else
        {
            addr.stride_x   = 1;
        }

        addr.stride_y = inbufs->imgs[0].stride[1];
        CHECK_VX_STATUS(vxAccessImagePatch(md->vxImages.front(), &rect, plane, &addr, &base_ptr, VX_WRITE_ONLY));
        CHECK_VX_STATUS(vxCommitImagePatch(md->vxImages.front(), &rect, plane, &addr, base_ptr));
    }

    // Plane 2
    if (md->color == WL_COLOR_I420)
    {
        plane           = 2;
        base_ptr        = inbufs->imgs[0].buf + inbufs->imgs[0].offset[2];
        rect.end_x      = md->width / 2;
        rect.end_y      = md->height / 2;
        addr.dim_x      = md->width / 2;
        addr.dim_y      = md->height / 2;
        addr.stride_x   = 1;
        addr.stride_y   = inbufs->imgs[0].stride[2];

        CHECK_VX_STATUS(vxAccessImagePatch(md->vxImages.front(), &rect, plane, &addr, &base_ptr, VX_WRITE_ONLY));
        CHECK_VX_STATUS(vxCommitImagePatch(md->vxImages.front(), &rect, plane, &addr, base_ptr));
    }

    CHECK_VX_STATUS(vxProcessGraph(md->graph));

    //Camera tampering process
//    vxAccessScalarValue(md->countNonZero_scalar, &md->countNonZero);
    vxWriteScalarValue(md->countNonZero_scalar, &md->countNonZero);

    vx_float64 ct_scale = CT_SCALE;
    vx_float64 ct_ratio_threshold = CT_RATIO_THRESHOLD;
    vx_float64 edge_count_ratio = md->countNonZero / ((vx_float64) md->width * md->height*ct_scale*ct_scale);
    if (edge_count_ratio < ct_ratio_threshold)
	{
        IntelVXSample::logger(0) << "Tampering \n";
    }
	else
	{
        IntelVXSample::logger(0) << "No Tampering \n";
    }
    IntelVXSample::logger(0) << "count :" << md->countNonZero << ",ratio :" << edge_count_ratio << "\n";
    // Get rectangles from vx_array
    vector<vx_rectangle_t>   objectList;
    vx_rectangle_t           *pObjectListPtr = NULL;
    vx_size                  stride          = 0;
    vx_size                  numItems        = 0;

    CHECK_VX_STATUS(vxQueryArray(md->vxArrays.back(), VX_ARRAY_ATTRIBUTE_NUMITEMS, &numItems, sizeof(numItems)));

    if (numItems != 0)
    {
        CHECK_VX_STATUS(vxAccessArrayRange(md->vxArrays.back(), 0, numItems, &stride, reinterpret_cast<void**>(&pObjectListPtr), VX_READ_ONLY));

        objectList.reserve(numItems);
        std::copy(pObjectListPtr, pObjectListPtr + numItems, std::back_inserter(objectList));

        CHECK_VX_STATUS(vxCommitArrayRange(md->vxArrays.back(), 0, numItems, pObjectListPtr));

        // Merge overlapped bounding boxes
        if (md->mergeBoxes == true)
        {
            thresholdX      = md->width / HORIZONTAL_MERGE_NEAR_THRESHOLD_DIVISOR;
            thresholdY      = md->height / VERTICAL_MERGE_NEAR_THRESHOLD_DIVISOR;
            MergeBoundingBoxes(objectList, thresholdX, thresholdY);
        }

        // Draw bounding box for detected moving objects
        WL_Rois *output     = (WL_Rois*)outmetadata;
        output->count       = min(objectList.size(), sizeof(output->rois) / sizeof(WL_Roi));

        for (int i = 0; i < objectList.size(); i++)
        {
            if (md->scaleFlag == true)
            {
                output->rois[i].x       = objectList[i].start_x * md->scaleFactor;
                output->rois[i].y       = objectList[i].start_y * md->scaleFactor;
                output->rois[i].width   = (objectList[i].end_x - objectList[i].start_x) * md->scaleFactor;
                output->rois[i].height  = (objectList[i].end_y - objectList[i].start_y) * md->scaleFactor;
            }
            else
            {
                output->rois[i].x       = objectList[i].start_x;
                output->rois[i].y       = objectList[i].start_y;
                output->rois[i].width   = objectList[i].end_x - objectList[i].start_x;
                output->rois[i].height  = objectList[i].end_y - objectList[i].start_y;
            }
        }
    }
    else
    {
        WL_Rois *output = (WL_Rois*)outmetadata;
        output->count   = 0;
    }

    return WL_OK;
}

static WL_Status vx_md_deinit (void *workload)
{
    if (workload == NULL)
        return WL_BAD_VALUE;

    VX_MD_Session* md = (VX_MD_Session*) workload;

    if (md->inited == false)
        return WL_OK;

    for (int i=0; i<md->vxNodes.size(); i++)
    {
        CHECK_VX_STATUS(vxReleaseNode(&md->vxNodes[i]));
    }

    for (int i=0; i<md->vxImages.size(); i++)
    {
        CHECK_VX_STATUS(vxReleaseImage(&md->vxImages[i]));
    }

    for (int i=0; i<md->vxArrays.size(); i++)
    {
        CHECK_VX_STATUS(vxReleaseArray(&md->vxArrays[i]));
    }
    vxReleaseGraph(&md->graph);

    md->vxNodes.clear();
    md->vxImages.clear();
    md->vxArrays.clear();

    md->inited = false;
    return WL_OK;
}

extern "C" __attribute__ ((visibility ("default"))) WLContext* WLCREATE()
{

    VX_MD_Session* md = new VX_MD_Session;
    if (md == NULL)
        return NULL;

    md->context = vxCreateContext();
    if (md->context == NULL)
    {
        delete md;
        return NULL;
    }
    md->inited          = false;
    md->width           = 0;
    md->height          = 0;
    md->color           = WL_COLOR_UNKNOWN;
    md->threshold       = 0;
    md->mergeBoxes      = true;
    md->scaleImage      = false;
    md->scaleFlag       = false;
    md->scaleFactor     = 1;
    md->heterogeneity   = WL_HETER_CPU_ONLY;

    WLContext* wlctx = new WLContext();
    if (wlctx == NULL)
    {
        delete md;
        return NULL;
    }
    wlctx->wl = (void*) md;
    wlctx->Init = vx_md_init;
    wlctx->Config = vx_md_config;
    wlctx->Process = vx_md_process;
    wlctx->Deinit = vx_md_deinit;

    return wlctx;
}

extern "C" __attribute__ ((visibility ("default"))) void WLDESTROY(WLContext* wlctx)
{
    if (wlctx == NULL)
      return;

    if (wlctx->wl == NULL)
        return;

    VX_MD_Session* md = (VX_MD_Session*) wlctx->wl;
    wlctx->Deinit(md);

    vxReleaseContext(&md->context);

    delete md;
    delete wlctx;
    return;
}

extern "C" __attribute__ ((visibility ("default"))) WLType WLTYPES()
{
    return WL_TYPE_ROIDETECTOR;
}

extern "C" __attribute__ ((visibility ("default"))) int WLVERSION()
{
    return WL_VER(IOTG_MD_VER_MAJOR, IOTG_MD_VER_MINOR);
}
