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
#include <iostream>
#include <algorithm>
#include <vector>
#include <stdio.h>


#include <exception>

#include <VX/vx.h>
#include <VX/vxu.h>


// Some common for all samples infrastructure stuff:
#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/perfprof.hpp>
#include <intel/vx_samples/basic.hpp>
#include <intel/vx_samples/hetero.hpp>
#include "video_stabilization.hpp"
#include "video_stabilization_user_nodes_lib.h"
#include "debug_visualization_lib.hpp"


#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


/// Builds one instance of the video stabilization graph in a given context
/** The function receives all intermediate data to glue nodes. The caller should correctly initialize
    all the data object, this function just use them when it calls vx-function.
    Constant parameters to tune specific graph nodes (like Harris Corners or Optical flow) are
    read directly from command line parser passed as one of the arguments. */
void buildVideoStabilizationGraph (
    vx_context context,
    vx_graph graph,
    unsigned int inputWidth,
    unsigned int inputHeight,
    vx_image imageOrig,   // defined externally because it is populated out of the graph
    vx_array prevPoints,   // defined externally because it is a part of vx_delay
    vx_array nextPoints,   // defined externally because it is a part of vx_delay
    vx_pyramid prevPyramid,   // defined externally because it is a part of vx_delay
    vx_pyramid nextPyramid,   // defined externally because it is a part of vx_delay
    vx_image imageWarp,
    std::vector<vx_node>& outNodes,
    std::vector<const char*>& nodeNames,
    const CmdParserVideoStabilization& cmdparser
)
{
    // To do the RGB to gray-scale conversion with the standard nodes, the two-stage
    // process is required. It involves conversion of RGB image to some other format,
    // that holds gray/luminance component as an explicit channel. Here NV12 is chosen
    // as such a format.
    vx_image nv12Image;
    if(cmdparser.no_virtual.isSet())
    {
        nv12Image = vxCreateImage(context, inputWidth, inputHeight, VX_DF_IMAGE_NV12);
    }
    else
    {
        nv12Image = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_NV12);
    }
    CHECK_VX_STATUS(vxGetStatus((vx_reference)nv12Image));

    vx_image grayImage;
    if(cmdparser.no_virtual.isSet())
    {
        grayImage = vxCreateImage(context, inputWidth, inputHeight, VX_DF_IMAGE_U8);
    }
    else
    {
        grayImage = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_VIRT);
    }
    CHECK_VX_STATUS(vxGetStatus((vx_reference)grayImage));


    // Input image for warping. It can be either imageOrig or intermediate image with debug visualization
    // depending on visualization level required by the user from command line.
    vx_image imageForWarp;
    if(cmdparser.debug_output.isSet())
    {
        if(cmdparser.no_virtual.isSet())
        {
            imageForWarp = vxCreateImage(context, inputWidth, inputHeight, VX_DF_IMAGE_RGB);
        }
        else
        {
            imageForWarp = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_VIRT);
        }
        CHECK_VX_STATUS(vxGetStatus((vx_reference)imageForWarp));
    }
    else
    {
        imageForWarp = imageOrig;
    }

    // To do image warping with the nodes from OpenVX standard, we need to handle
    // each channel of RGB image separatelly.
    vx_image imageForWarpR, imageForWarpG, imageForWarpB;
    if(cmdparser.no_virtual.isSet())
    {
        imageForWarpR = vxCreateImage(context, inputWidth, inputHeight, VX_DF_IMAGE_U8);
        imageForWarpG = vxCreateImage(context, inputWidth, inputHeight, VX_DF_IMAGE_U8);
        imageForWarpB = vxCreateImage(context, inputWidth, inputHeight, VX_DF_IMAGE_U8);
    }
    else
    {
        imageForWarpR = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_VIRT);
        imageForWarpG = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_VIRT);
        imageForWarpB = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_VIRT);
    }
    CHECK_VX_STATUS(vxGetStatus((vx_reference)imageForWarpR));
    CHECK_VX_STATUS(vxGetStatus((vx_reference)imageForWarpG));
    CHECK_VX_STATUS(vxGetStatus((vx_reference)imageForWarpB));

    // The next three images hold the result of warping of three separated channels of input image
    vx_image imageWarpR, imageWarpG, imageWarpB;
    if(cmdparser.no_virtual.isSet())
    {
        imageWarpR = vxCreateImage(context, inputWidth, inputHeight, VX_DF_IMAGE_U8);
        imageWarpG = vxCreateImage(context, inputWidth, inputHeight, VX_DF_IMAGE_U8);
        imageWarpB = vxCreateImage(context, inputWidth, inputHeight, VX_DF_IMAGE_U8);
    }
    else
    {
        imageWarpR = vxCreateVirtualImage(graph, inputWidth, inputHeight, VX_DF_IMAGE_U8);
        imageWarpG = vxCreateVirtualImage(graph, inputWidth, inputHeight, VX_DF_IMAGE_U8);
        imageWarpB = vxCreateVirtualImage(graph, inputWidth, inputHeight, VX_DF_IMAGE_U8);
    }
    CHECK_VX_STATUS(vxGetStatus((vx_reference)imageWarpR));
    CHECK_VX_STATUS(vxGetStatus((vx_reference)imageWarpG));
    CHECK_VX_STATUS(vxGetStatus((vx_reference)imageWarpB));

    // Array holds new estimations for feature points obtained by optical flow node
    // This array will be used as an input to movement estimator and (optionally) debug
    // visualization node.
    vx_array movedPoints;
    if(cmdparser.no_virtual.isSet())
    {
        movedPoints = vxCreateArray(context, VX_TYPE_KEYPOINT, cmdparser.max_corners.getValue());
    }
    else
    {
        movedPoints = vxCreateVirtualArray(graph, 0, cmdparser.max_corners.getValue());
    }
    CHECK_VX_STATUS(vxGetStatus((vx_reference)movedPoints));

    // One of the user nodes should know the valid size of the frame to filter
    // feature points for more robust transform estimation results.
    // Current version of OpenVX doesn't allow to pass a structure like vx_rectangle_t
    // as a node parameter. So we need to do a trick: create vx_array with a single
    // element of type vx_rectangle_t. And then this vx_array will be passed to a node as
    // a paramter.
    vx_rectangle_t frameRectVal;
    frameRectVal.start_x = 0;
    frameRectVal.start_y = 0;
    frameRectVal.end_x = inputWidth;
    frameRectVal.end_y = inputHeight;
    vx_array frameRect = vxCreateArray(context, VX_TYPE_RECTANGLE, 1);
    CHECK_VX_STATUS(vxGetStatus((vx_reference)frameRect));
    CHECK_VX_STATUS(vxAddArrayItems(frameRect, 1, &frameRectVal, sizeof(frameRectVal)));

    // This matrix is to pass affine transform matrix produced by vxEstimateTransformNode user-node
    // to vxWarpAffineNode to do real image transformation.
    vx_matrix transform = vxCreateMatrix(context, VX_TYPE_FLOAT32, 2, 3);
    CHECK_VX_STATUS(vxGetStatus((vx_reference)transform));

    // Define necessary scalar parameters to be used in nodes Harris Corners and Optical Flow.
    // Part of the parameters should be passed as vx_scalars, and they should be prepared here.
    // Another part of parameters can be accepted as values of native types (like int or float),
    // and they are passed directly to vx-functions.
    // Please refer to those nodes' documentation for reference.

    float strengthThreshVal = cmdparser.strength_thresh.getValue();
    vx_scalar strengthThresh = vxCreateScalar(context, VX_TYPE_FLOAT32, &strengthThreshVal);
    CHECK_VX_STATUS(vxGetStatus((vx_reference)strengthThresh));
    
    float minDistanceVal = cmdparser.min_distance.getValue();
    vx_scalar minDistance = vxCreateScalar(context, VX_TYPE_FLOAT32, &minDistanceVal);
    CHECK_VX_STATUS(vxGetStatus((vx_reference)minDistance));
    
    float sensitivityVal = cmdparser.sensitivity.getValue();
    vx_scalar sensitivity = vxCreateScalar(context, VX_TYPE_FLOAT32, &sensitivityVal);
    CHECK_VX_STATUS(vxGetStatus((vx_reference)sensitivity));

    vx_float32 terminateEpsilonVal = cmdparser.optical_flow_epsilon.getValue();
    vx_scalar terminateEpsilon = vxCreateScalar(context, VX_TYPE_FLOAT32, &terminateEpsilonVal);
    CHECK_VX_STATUS(vxGetStatus((vx_reference)terminateEpsilon));

    vx_uint32 terminateIterationsVal = cmdparser.optical_flow_iterations.getValue();
    vx_scalar terminateIterations = vxCreateScalar(context, VX_TYPE_UINT32, &terminateIterationsVal);
    CHECK_VX_STATUS(vxGetStatus((vx_reference)terminateIterations));

    vx_bool useInitialEstimateVal = vx_true_e;
    vx_scalar useInitialEstimations = vxCreateScalar(context, VX_TYPE_BOOL, &useInitialEstimateVal);
    CHECK_VX_STATUS(vxGetStatus((vx_reference)useInitialEstimations));
    IntelVXSample::HeteroScheduleConfig heteroConfig(cmdparser.hetero_config.getValue());
    heteroConfig.pupulateSupportedTargets();

    // Populate graph with nodes

    // Each node in the graph is created with vx-function, which returns vx_node object.
    // Besides node creation in a graph, we also need to do some things with this node
    // on the application side. To keep the code shorter, we use the following macro.
    //
    // This is a helping macro that accepts vx_node object and user-defined name for a node and
    //    - checks the vx_node object for correct creation with CHECK_VX_STATUS macro
    //    - adds vx_node object to outNodes vector to be used outside this function
    //    - adds the node name to nodeNames vector to be also be used outside this function
    //    - assigns a specific target to the node object if it is listed in the hetero configuration
    //
    // We apply this macro for each node in the graph.
    //
    #define SAMPLE_ADD_NEW_NODE(NODE_NAME, NODE)                                \
    {                                                                           \
        vx_node node = (NODE);    /* NODE is a statement with node creation */  \
                                                                                \
        CHECK_VX_STATUS(vxGetStatus((vx_reference)node));                       \
        outNodes.push_back(node);                                               \
        nodeNames.push_back(NODE_NAME);                                         \
                                                                                \
        if(vx_enum target = heteroConfig.getTargetByNodeName(NODE_NAME))        \
        {                                                                       \
            CHECK_VX_STATUS(vxSetNodeTarget(node, target, 0));                     \
        }                                                                       \
    }
    // RGB to GRAY conversion: two nodes - convert and extract
    {
        SAMPLE_ADD_NEW_NODE(
            "vxColorConvertNode",
            vxColorConvertNode(graph, imageOrig, nv12Image)
        );

        SAMPLE_ADD_NEW_NODE(
            "vxChannelExtractNode",
            vxChannelExtractNode(graph, nv12Image, VX_CHANNEL_Y, grayImage)
        );
    }
        
    SAMPLE_ADD_NEW_NODE(
        "vxHarrisCornersNode",
        vxHarrisCornersNode(
            graph,
            grayImage,
            strengthThresh,
            minDistance,
            sensitivity,
            3,
            cmdparser.block_size.getValue(),
            nextPoints,
            0
        )
    );

    SAMPLE_ADD_NEW_NODE(
        "vxGaussianPyramidNode",
        vxGaussianPyramidNode(graph, grayImage, nextPyramid)
    );

    SAMPLE_ADD_NEW_NODE(
        "vxOpticalFlowPyrLKNode",
        vxOpticalFlowPyrLKNode(
            graph,
            prevPyramid,
            nextPyramid,
            prevPoints,
            prevPoints,
            movedPoints,
            VX_TERM_CRITERIA_BOTH,
            terminateEpsilon,
            terminateIterations,
            useInitialEstimations,
            cmdparser.optical_flow_window.getValue()
        )
    );

    SAMPLE_ADD_NEW_NODE(
        "vxEstimateTransformNode",
        vxEstimateTransformNode(
            graph,
            prevPoints,
            movedPoints,
            frameRect,
            transform
        )
    );
    if (cmdparser.debug_output.isSet())
    {
        // Optionally add debug visualization node
        // that will modify input image to put feature points and their
        // new estimated positions on that image. We incorporate this
        // code as a user node to avoid splitting of the graph to two pieces, and
        // to demonstrate how it can be done naturally as a part of the graph.
        SAMPLE_ADD_NEW_NODE(
            "vxDebugVisualizationNode",
            vxDebugVisualizationNode(
                graph,
                imageOrig,
                prevPoints,
                movedPoints,
                imageForWarp
            )
        );
    }
    else
    {
        // this is guarantied by the code above that creates imageForWarp
        assert(imageForWarp == imageOrig);
    }

    // Before doing image warping, it should be decomposed into separate channels (R, G and B in our case).
    // It should be done, because vxWarpAffineNode accepts only single channel images.

    // So, extract...

    SAMPLE_ADD_NEW_NODE(
        "vxChannelExtractNode(R)",
        vxChannelExtractNode(graph, imageForWarp, VX_CHANNEL_R, imageForWarpR)
    );

    SAMPLE_ADD_NEW_NODE(
        "vxChannelExtractNode(G)",
        vxChannelExtractNode(graph, imageForWarp, VX_CHANNEL_G, imageForWarpG)
    );

    SAMPLE_ADD_NEW_NODE(
        "vxChannelExtractNode(B)",
        vxChannelExtractNode(graph, imageForWarp, VX_CHANNEL_B, imageForWarpB)
    );

    // ... then warp ...

    SAMPLE_ADD_NEW_NODE(
        "vxWarpAffineNode(R)",
        vxWarpAffineNode(graph, imageForWarpR, transform, VX_INTERPOLATION_BILINEAR, imageWarpR)
    );

    SAMPLE_ADD_NEW_NODE(
        "vxWarpAffineNode(G)",
        vxWarpAffineNode(graph, imageForWarpG, transform, VX_INTERPOLATION_BILINEAR, imageWarpG)
    );

    SAMPLE_ADD_NEW_NODE(
        "vxWarpAffineNode(B)",
        vxWarpAffineNode(graph, imageForWarpB, transform, VX_INTERPOLATION_BILINEAR, imageWarpB)
    );

    // ... and finaly combine back into a single image.

    SAMPLE_ADD_NEW_NODE(
        "vxChannelCombineNode(warp)",
        vxChannelCombineNode(
            graph,
            imageWarpR,
            imageWarpG,
            imageWarpB,
            0,
            imageWarp
        )
    );

    // Before we leave this function, all OpenVX objects (arrays, images, scalars etc.) that were
    // created in this function should be released to decrement reference counter for each of them.
    // If it is not done here there will no any chance to do it lately and it will lead to resource leaks
    // if this function is called repetedly for some reason.
    // It is not so critical for this sample code, because this function is called only once and all
    // resources will be destroyed in the end of the program anyway.

    CHECK_VX_STATUS(vxReleaseImage(&nv12Image));
    CHECK_VX_STATUS(vxReleaseImage(&grayImage));

    if (cmdparser.debug_output.isSet())
    {
        CHECK_VX_STATUS(vxReleaseImage(&imageForWarp));
    }

    CHECK_VX_STATUS(vxReleaseImage(&imageForWarpR));
    CHECK_VX_STATUS(vxReleaseImage(&imageForWarpG));
    CHECK_VX_STATUS(vxReleaseImage(&imageForWarpB));
    CHECK_VX_STATUS(vxReleaseImage(&imageWarpR));
    CHECK_VX_STATUS(vxReleaseImage(&imageWarpG));
    CHECK_VX_STATUS(vxReleaseImage(&imageWarpB));

    CHECK_VX_STATUS(vxReleaseArray(&movedPoints));
    CHECK_VX_STATUS(vxReleaseArray(&frameRect));
    CHECK_VX_STATUS(vxReleaseMatrix(&transform));
    
    CHECK_VX_STATUS(vxReleaseScalar(&strengthThresh));
    CHECK_VX_STATUS(vxReleaseScalar(&minDistance));
    CHECK_VX_STATUS(vxReleaseScalar(&sensitivity));
    CHECK_VX_STATUS(vxReleaseScalar(&terminateEpsilon));
    CHECK_VX_STATUS(vxReleaseScalar(&terminateIterations));
    CHECK_VX_STATUS(vxReleaseScalar(&useInitialEstimations));
}


int video_stabilization_openvx (const CmdParserVideoStabilization& cmdparser)
{
    // Define regions for further profiling in the code.
    // PERFPROF_REGION_DEFINE is a macro from samles infrastructure
    // that helps to measure performance of code regions.

    PERFPROF_REGION_DEFINE(vxProcessGraph);
    PERFPROF_REGION_DEFINE(vxVerifyGraph);
    PERFPROF_REGION_DEFINE(Frame);
    PERFPROF_REGION_DEFINE(ReadFrame);
    PERFPROF_REGION_DEFINE(ProcessFrame);

    vx_context context = vxCreateContext();
    CHECK_VX_STATUS(vxGetStatus((vx_reference)context));
    CHECK_VX_STATUS(vxDirective((vx_reference)context, VX_DIRECTIVE_ENABLE_PERFORMANCE));

    // Register callback to receive diagnostics messages from OpenVX run-time
    // Predefined function IntelVXSample::errorReceiver is used from common sample infrastructure,
    // and just put all the input messages to std::cerr to be read by the user.
    vxRegisterLogCallback(context, IntelVXSample::errorReceiver, vx_true_e);

    // To use two user-nodes that is compiled separately, vxLoadKernels should be called
    // with the name of the library without prefix (e.g. lib) or suffix (e.g. so).
    CHECK_VX_STATUS(vxLoadKernels(context, "video_stabilization_user_nodes_module"));

    // OpenCV is used here for two main purposes: to read a video file and show input and output images.
    cv::VideoCapture cap;
    cv::VideoWriter ocvWriter;  // will be used in case if --output is provided
    cv::Mat frame;    // is used for sharing with OpenVX, but really used if input file is present

    unsigned int inputWidth;
    unsigned int inputHeight;

    cap.open(cmdparser.input.getValue());
    if(!cap.isOpened())
    {
        std::cerr << "[ ERROR ] Cannot open input video file " << cmdparser.input.getValue() << "\n";
        return 1;
    }

    if(cmdparser.output.isSet())
    {
        IntelVXSample::openVideoWriterByCapture(cap, ocvWriter, cmdparser.output.getValue());
    }

    cap.set(cv::CAP_PROP_CONVERT_RGB, 1);
    {
        PERFPROF_REGION_AUTO(ReadFrame);
        cap >> frame;
    }

    if(frame.empty())
    {
        std::cerr << "[ ERROR ] Cannot read the first frame from video\n";
        return 1;
    }

    inputWidth = frame.cols;
    inputHeight = frame.rows;

    std::cout << "Video frame size: " << inputWidth << "x" << inputHeight << "\n";

    unsigned int maxIterations = cmdparser.max_frames.getValue();
    unsigned int iterations = 0;

    // The following struct holds the valid size of the frame.
    // Then it will be used to access input vx_image to fill it with a new data.
    vx_rectangle_t frameRect;
    frameRect.start_x = 0;
    frameRect.start_y = 0;
    frameRect.end_x = inputWidth;
    frameRect.end_y = inputHeight;


    // Prepare structures for correct sharing with OpenCV
    // Here vx_array instance is created over cv::Mat data structure
    // To do that, we need to describe data alignement in vx_imagepatch_addressing_t
    // structure instance and then pass it to vxCreateImageFromHandle function.

    vx_imagepatch_addressing_t frameFormat;
    frameFormat.dim_x = inputWidth;
    frameFormat.dim_y = inputHeight;
    frameFormat.stride_x = 3;   // for three channels: R, G, and B
    frameFormat.stride_y = frame.step;  // number of bytes each matrix row occupies
    frameFormat.scale_x = VX_SCALE_UNITY;
    frameFormat.scale_y = VX_SCALE_UNITY;
    frameFormat.step_x = 1;
    frameFormat.step_y = 1;
    
    // Input image to feed the graph.
    // Will be updated every iteration to process each frame
    vx_image imageOrig = vxCreateImageFromHandle(
        context,
        VX_DF_IMAGE_RGB,
        &frameFormat,
        (void**)&frame.data,
        VX_MEMORY_TYPE_HOST
    );

    CHECK_VX_STATUS(vxGetStatus((vx_reference)imageOrig));

    // Output image, this is the resulting image produced by the graph.
    // Each iteration it will hold the resulting stabilized video frame based on
    // input frame in imageOrig.
    vx_image imageWarp  = vxCreateImage(context, inputWidth, inputHeight, VX_DF_IMAGE_RGB);
    CHECK_VX_STATUS(vxGetStatus((vx_reference)imageWarp));


    // Create graph object
    vx_graph graph = vxCreateGraph(context);
    CHECK_VX_STATUS(vxGetStatus((vx_reference)graph));

    
    // There are several objects that should be produced by one graph processing iteration and
    // should be consumed by next graph processing iteration. We cannot use the same data object to
    // be consumed and then written in the same graph. So vx_delay is used to hold such objects.
    // An instance of vx_delay is a ring buffer with specified number of element.
    // In our case the size of each vx_delay instance is 2, because only two iterations of the graph
    // processing execution should be connected.

    // First create an exemplar for vx_delay for the pyramids
    vx_pyramid pyramidExemplar = vxCreatePyramid(context, cmdparser.pyramid_levels.getValue(), VX_SCALE_PYRAMID_HALF, inputWidth, inputHeight, VX_DF_IMAGE_U8);
    CHECK_VX_STATUS(vxGetStatus((vx_reference)pyramidExemplar));
    // And then use this exemplar to create a vx_delay instance itself
    vx_delay pyramidRing = vxCreateDelay(context, (vx_reference)pyramidExemplar, 2);
    CHECK_VX_STATUS(vxGetStatus((vx_reference)pyramidRing));
    // After vx_delay creation we don't need to keep the exemplar that was used for that delay
    CHECK_VX_STATUS(vxReleasePyramid(&pyramidExemplar));

    // Do similar steps for feature points vx_dalay
    vx_array pointsExemplar = vxCreateArray(context, VX_TYPE_KEYPOINT, cmdparser.max_corners.getValue());
    CHECK_VX_STATUS(vxGetStatus((vx_reference)pointsExemplar));
    vx_delay pointsRing = vxCreateDelay(context, (vx_reference)pointsExemplar, 2);
    CHECK_VX_STATUS(vxGetStatus((vx_reference)pointsRing));
    CHECK_VX_STATUS(vxReleaseArray(&pointsExemplar));

    // While building the graph in buildVideoStabilizationGraph function, after each node creation,
    // a node id (an instance of vx_node) and descriptive node name (just a string) are stored in
    // the following two vectors: nodes and nodeNames. They will be used later while debug
    // visualization is formed to show node execution times at the timeline drawn by means of OpenCV.
    // It is activated only if --debug-output is set in the command line.
    // It is not directly related to OpenVX core stuff, but serves a handy tool to debug and profile
    // OpenVX programs.
    std::vector<vx_node> nodes;
    std::vector<const char*> nodeNames;

    // OK, now finally, build the graph.
    // It is written in a separate function to avoid code bloating.
    buildVideoStabilizationGraph(
        context,
        graph,
        inputWidth,
        inputHeight,
        imageOrig,
        (vx_array)vxGetReferenceFromDelay(pointsRing, -1),  // indices of vx_delay is in range [-n+1,... 0]
        (vx_array)vxGetReferenceFromDelay(pointsRing, 0),  // indices of vx_delay is in range [-n+1,... 0]
        (vx_pyramid)vxGetReferenceFromDelay(pyramidRing, -1),  // indices of vx_delay is in range [-n+1,... 0]
        (vx_pyramid)vxGetReferenceFromDelay(pyramidRing, 0),  // indices of vx_delay is in range [-n+1,... 0]
        imageWarp,
        nodes,
        nodeNames,
        cmdparser
    );
    
    {
        PERFPROF_REGION_AUTO(vxVerifyGraph);
        CHECK_VX_STATUS(vxVerifyGraph(graph));
    }

    IntelVXSample::logger(1) << "[ INFO ] Verified graph " << graph << "\n";

    // Loop for frames
    for(;;)
    {
        PERFPROF_REGION_AUTO(Frame);
        iterations++;
        IntelVXSample::logger(1) << "Frame " << iterations << "\n";

        {
            PERFPROF_REGION_AUTO(ProcessFrame)

            {
                PERFPROF_REGION_AUTO(vxProcessGraph);
                CHECK_VX_STATUS(vxProcessGraph(graph));
            }

            // Between the previous and the next processing of the graph,
            // all vx_delay objects should be aged by vxAgeDelay function.
            // It automatically updates references in the graph so that the object,
            // that were written in the previous graph execution, are linked to
            // input data parameters in the next graph processing.
            CHECK_VX_STATUS(vxAgeDelay(pointsRing));
            CHECK_VX_STATUS(vxAgeDelay(pyramidRing));
        }

        // Draw images with OpenCV by mapping vx_image as cv::Mat using helper function mapAsMat
        // from common samples infrastructure.
        // mapAsMat/unmapAsMat can be interesting for the education, so please refer to
        // samples/common/src/helper.cpp where they are defined and and examine the code.

        vx_map_id map_id;
        if(cmdparser.debug_output.isSet() || !cmdparser.no_show.isSet())
        {
            cv::Mat imageResizedMat = IntelVXSample::mapAsMat(imageOrig, VX_READ_ONLY, &map_id);
            if (!cmdparser.no_show.isSet())
            {
                cv::imshow("Input Frame", imageResizedMat);
            }
            IntelVXSample::unmapAsMat(imageOrig, imageResizedMat, map_id);
        }

        if(cmdparser.debug_output.isSet() || !cmdparser.no_show.isSet() || ocvWriter.isOpened())
        {
            cv::Mat imageWarpMat = IntelVXSample::mapAsMat(imageWarp, VX_READ_ONLY, &map_id);
            if(!cmdparser.no_show.isSet())
            {
                cv::imshow("Stabilized Frame", imageWarpMat);
            }
            if(ocvWriter.isOpened())
            {
                ocvWriter.write(imageWarpMat);
            }
            IntelVXSample::unmapAsMat(imageWarp, imageWarpMat, map_id);
        }

        if(maxIterations > 0 && iterations >= maxIterations)
        {   
            std::cout << "Reached specified maximum number of iterations\n";
            break;
        }

        {
            PERFPROF_REGION_AUTO(ReadFrame);
            vx_imagepatch_addressing_t outPatchAddr;
            void* mappedArea = 0;
            // To read the next frame, map imageOrig to be
            // able to populate it with new data by OpenCV
            vx_map_id map_id;
            CHECK_VX_STATUS(vxMapImagePatch(imageOrig, &frameRect, 0, &map_id, &outPatchAddr, &mappedArea, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));
            assert(mappedArea == frame.data);

            cap >> frame;
            
            CHECK_VX_STATUS(vxUnmapImagePatch(imageOrig, map_id));

            if(frame.empty())
            {
                std::cout << "Reached end of video file\n";
                break;
            }

        }

        if(cmdparser.debug_output.isSet() && !cmdparser.no_show.isSet())
        {
            // Here is where collected nodes and nodeNames are used.
            // drawNodesAtTimeline makes additional OpenCV window with graphical
            // representation of the latest execution times of the nodes drawn on the time-line.
            // Examine this function if you want to learn how to obtain performance information
            // from the nodes to do your own performance measurement code.
            IntelVXSample::drawNodesAtTimeline(&nodes[0], nodes.size(), &nodeNames[0]);
        }

        if(!cmdparser.no_show.isSet())
        {
            int key = cv::waitKey(cmdparser.frame_wait.getValue()) & 0xff;
            if(key == 27)   // 27 is ESC, press ESC to exit
            {
                break;
            }
        }
    }

    std::cout << "Processed " << iterations << " iterations\n";

    CHECK_VX_STATUS(vxReleaseImage(&imageOrig));
    CHECK_VX_STATUS(vxReleaseImage(&imageWarp));
    CHECK_VX_STATUS(vxReleaseDelay(&pyramidRing));
    CHECK_VX_STATUS(vxReleaseDelay(&pointsRing));

    for (int n = 0; n < nodes.size(); n++)
    {
        CHECK_VX_STATUS(vxReleaseNode(&nodes[n]));
    }
    
    CHECK_VX_STATUS(vxReleaseGraph(&graph));

    CHECK_VX_STATUS(vxReleaseContext(&context));
    std::cout << "Sample was finished successfully\n";

    return 0;
}

