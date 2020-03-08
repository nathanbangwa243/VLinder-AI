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


#include <opencv2/opencv.hpp>
#include <VX/vx.h>
#include <VX/vx_intel_volatile.h>
#include <iostream>

#include <intel/vx_samples/perfprof.hpp>
#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/cmdparser.hpp>


int main(int argc, const char** argv)
{
    try
    {
        // Define command line parameters and parse them

        CmdParserWithHelp cmd(argc, argv);

        CmdOption<int> max_frames(
            cmd,
            0,
            "max-frames","<integer>",
            "Number of frames from input video file to be read. "
                "May be useful for benchmarking purposes and/or when "
                "input video file is to large to be processed completely. "
                "-1 means that entire files is processed.",
            -1
        );

        CmdOption<string> input(
            cmd,
            'i',
            "input",
            "<file name>",
            "Input video file.",
            exe_dir() + "toy_flower_512x512.mp4"
        );

        CmdOptionOutputVideo output(cmd);

        CmdOptionFrameWait frame_wait(cmd);

        CmdOptionNoShow noshow(cmd);

        cmd.parse();
        if (cmd.help.isSet())
        {
            // Immediatly exit if user wanted to see the usage information only.
            return 0;
        }

        // ---------------------------------------------------------------------------------------

        cv::VideoCapture    ocvCapture;     // video caputure to read input frames from video file
        cv::VideoWriter     ocvWriter;      // video writer to write output frames (if enabled)
        cv::Mat             ocvInpBGR;      // input image captured by OpenCV in in BGR format
        int                 width, height;  // width and height of input image
        vx_context          ovxContext;     // OpenVX context
        vx_graph            ovxGraph;       // OpenVX graph

        // Define performance counter to measure execution time
        PERFPROF_REGION_DEFINE(vxProcessGraph);
        PERFPROF_REGION_DEFINE(ReadFrame);


        // Open input video file
        if(!ocvCapture.open(input.getValue()))
        {
            std::cerr << "[ ERROR ] " << input.getValue() << " is not opened" << std::endl;
            return EXIT_FAILURE;
        }
        std::cout <<  input.getValue() << " is opened" << std::endl;

        // Get width and height of an input frame
        width = (int)ocvCapture.get(cv::CAP_PROP_FRAME_WIDTH);
        height = (int)ocvCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cout <<  "Input frame size: " << width << "x" << height << std::endl;

        if(output.isSet())
        {
            IntelVXSample::openVideoWriterByCapture(ocvCapture, ocvWriter, output.getValue());
        }

        // Create OpenVX context, graph and register callback to print errors to
        ovxContext = vxCreateContext();
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxContext));
        vxRegisterLogCallback(ovxContext, IntelVXSample::errorReceiver, vx_true_e);
        ovxGraph = vxCreateGraph(ovxContext);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxGraph));

        // ---------------------------------------------------------------------------------------

        // Create image data objects to hold input, intermediate and output data in the pipeline
        // All intermediate images are virtual because there is no need to access them outside
        // the pipeline.

        // input RGB image
        vx_image    ovxImgRGB      = vxCreateImage(ovxContext, width, height, VX_DF_IMAGE_RGB);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxImgRGB));

        // R channel input and output images for Canny
        vx_image    ovxImgRIn      = vxCreateVirtualImage(ovxGraph, 0, 0, VX_DF_IMAGE_U8);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxImgRIn));
        vx_image    ovxImgROut     = vxCreateVirtualImage(ovxGraph, 0, 0, VX_DF_IMAGE_U8);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxImgROut));

        // G channel input and output images for Canny
        vx_image    ovxImgGIn      = vxCreateVirtualImage(ovxGraph, 0, 0, VX_DF_IMAGE_U8);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxImgGIn));
        vx_image    ovxImgGOut     = vxCreateVirtualImage(ovxGraph, 0, 0, VX_DF_IMAGE_U8);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxImgGOut));

        // B channel input and output images for Canny
        vx_image    ovxImgBIn      = vxCreateVirtualImage(ovxGraph, 0, 0, VX_DF_IMAGE_U8);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxImgBIn));
        vx_image    ovxImgBOut     = vxCreateVirtualImage(ovxGraph, 0, 0, VX_DF_IMAGE_U8);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxImgBOut));

        // output RGB image where all three channels after Canny nodes are combined
        vx_image    ovxImgOut      = vxCreateImage(ovxContext, width, height, VX_DF_IMAGE_RGB);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxImgOut));

        // ---------------------------------------------------------------------------------------

        // Later a graph contatinig Canny nodes will be constructed. To instantiate a Canny
        // node, some parameters should be defined: hysteresis, gradient size and norm type.
        // Define them here, and then pass the same parameters for each channel.

        vx_threshold hyst = vxCreateThreshold(ovxContext, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_UINT8);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)hyst));
        vx_int32 hystLo = 80, hystUp = 150;
        CHECK_VX_STATUS(vxSetThresholdAttribute(hyst, VX_THRESHOLD_THRESHOLD_LOWER, &hystLo, sizeof(hystLo)));
        CHECK_VX_STATUS(vxSetThresholdAttribute(hyst, VX_THRESHOLD_THRESHOLD_UPPER, &hystUp, sizeof(hystUp)));

        vx_int32 gradient_size = 3;
        vx_enum norm_type = VX_NORM_L1;

        // ---------------------------------------------------------------------------------------

        // Create nodes connected by images created above
        // For each channel, Canny is executed on a separate target
        //
        //                -->  extractRNode  -->  cannyRNode(CPU)  --
        //               /                                           \
        //   ovxImgRGB  ---->  extractGNode  -->  cannyGNode(GPU)  ----  combineNode  --> ovxImgOut
        //               \                                           /
        //                -->  extractBNode  -->  cannyBNode(IPU)  --
        //

        vx_node extractRNode   = vxChannelExtractNode(ovxGraph, ovxImgRGB, VX_CHANNEL_R, ovxImgRIn);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)extractRNode));
        vx_node cannyRNode = vxCannyEdgeDetectorNode(ovxGraph, ovxImgRIn, hyst, gradient_size, norm_type, ovxImgROut);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)cannyRNode));

        vx_node extractGNode   = vxChannelExtractNode(ovxGraph, ovxImgRGB, VX_CHANNEL_G, ovxImgGIn);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)extractGNode));
        vx_node cannyGNode = vxCannyEdgeDetectorNode(ovxGraph, ovxImgGIn, hyst, gradient_size, norm_type, ovxImgGOut);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)cannyGNode));

        vx_node extractBNode   = vxChannelExtractNode(ovxGraph, ovxImgRGB, VX_CHANNEL_B, ovxImgBIn);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)extractBNode));
        vx_node cannyBNode = vxCannyEdgeDetectorNode(ovxGraph, ovxImgBIn, hyst, gradient_size, norm_type, ovxImgBOut);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)cannyBNode));

        // and finaly channel combine node is instantiated to produce the final RGB image
        vx_node combinedNode = vxChannelCombineNode(ovxGraph, ovxImgBOut, ovxImgGOut, ovxImgROut, 0, ovxImgOut);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)combinedNode));

        // ---------------------------------------------------------------------------------------

        // Targets can be identified by enum or by name.
        // The following targets are supported:
        //
        //           Enum         |     Name
        // -----------------------+---------------
        //   VX_TARGET_CPU_INTEL  |  "intel.cpu"
        //   VX_TARGET_GPU_INTEL  |  "intel.gpu"
        //   VX_TARGET_IPU_INTEL  |  "intel.ipu"
        //
        // Not all of the targets are available on any platform and for any kernel.
        // In the next code we are trying to set desired target for each of the Canny nodes.
        // If a specific target is not available for a node, the node will remain assigned to
        // default target VX_TARGET_ANY that let run-time decide which target to choose.

        if(vxSetNodeTarget(cannyRNode, VX_TARGET_CPU_INTEL, 0) == VX_SUCCESS)
        {
            std::cout << "[ INFO ] Target intel.cpu is set for channel R.\n";
        }
        else
        {
            std::cout << "[ WARNING ] Target intel.cpu is NOT available. ";
            std::cout << "Nodes related to channel R won't be assigned to any specific target.\n";
        }

        if(vxSetNodeTarget(cannyGNode, VX_TARGET_GPU_INTEL, 0) == VX_SUCCESS)
        {
            std::cout << "[ INFO ] Target intel.gpu is set for channel G.\n";
        }
        else
        {
            std::cout << "[ WARNING ] Target intel.gpu is NOT available. ";
            std::cout << "Nodes related to channel G won't be assigned to any specific target.\n";
        }

        if(vxSetNodeTarget(cannyBNode, VX_TARGET_IPU_INTEL, 0) == VX_SUCCESS)
        {
            std::cout << "[ INFO ] Target intel.ipu is set for channel B.\n";
        }
        else
        {
            std::cout << "[ WARNING ] Target intel.ipu is NOT available. ";
            std::cout << "Nodes related to channel B won't be assigned to any specific target.\n";
        }

        // ---------------------------------------------------------------------------------------

        // Before running the main loop over frames we need to verify the graph as usual
        CHECK_VX_STATUS(vxVerifyGraph(ovxGraph));

        // The main loop iterating over frames
        for(unsigned int frame=0; frame < (unsigned int)max_frames.getValue(); frame++)
        {
            {
                // This is read frame section that is measured separatly
                PERFPROF_REGION_AUTO(ReadFrame);

                // Image capture section
                if(!ocvCapture.read(ocvInpBGR))
                {
                    std::cout << std::endl << "Break on ocvCapture.read";
                    break; // break if there is no any image
                }


                // Copy input OpenCV BGR image into input OpenVX RGB image.
                // It is necessary to make BGR->RGB conversion because OpenVX support only RGB image format
                // mapAsMat is used to map OpenVX image into HOST mempry and wrap it by OpenCV Mat.
                vx_map_id map_id;
                cv::Mat  imgRGB = IntelVXSample::mapAsMat(ovxImgRGB, VX_READ_ONLY, &map_id);
                cv::cvtColor(ocvInpBGR,imgRGB,cv::COLOR_BGR2RGB);
                IntelVXSample::unmapAsMat(ovxImgRGB,imgRGB, map_id);
            }

            {
                // Run OpenVX graph on new capture frame
                PERFPROF_REGION_AUTO(vxProcessGraph);
                CHECK_VX_STATUS(vxProcessGraph(ovxGraph));
            }

            if(!noshow.isSet() || ocvWriter.isOpened())
            {
                vx_map_id map_id;
                cv::Mat visualizationMat = IntelVXSample::mapAsMat(ovxImgOut, VX_READ_ONLY, &map_id);
                if(!noshow.isSet())
                {
                    cv::imshow("OVXResult", visualizationMat);
                }
                if(ocvWriter.isOpened())
                {
                    ocvWriter.write(visualizationMat);
                }
                IntelVXSample::unmapAsMat(ovxImgOut, visualizationMat, map_id);

                if(!noshow.isSet())
                {
                    // read key value and process it
                    int key = cv::waitKey(frame_wait.getValue()) & 0xff;
                    if(key == 27) // ESC is pressed. exit
                        break;
                }
            }// visualisation end

        }//next frame

        // ---------------------------------------------------------------------------------------

        // Release all stuff

        std::cout << std::endl << "Release data..." << std::endl;

        // Release nodes
        CHECK_VX_STATUS(vxReleaseNode(&extractRNode));
        CHECK_VX_STATUS(vxReleaseNode(&cannyRNode));
        CHECK_VX_STATUS(vxReleaseNode(&extractGNode));
        CHECK_VX_STATUS(vxReleaseNode(&cannyGNode));
        CHECK_VX_STATUS(vxReleaseNode(&extractBNode));
        CHECK_VX_STATUS(vxReleaseNode(&cannyBNode));
        CHECK_VX_STATUS(vxReleaseNode(&combinedNode));

        // Release images
        CHECK_VX_STATUS(vxReleaseImage(&ovxImgRGB));
        CHECK_VX_STATUS(vxReleaseImage(&ovxImgRIn));
        CHECK_VX_STATUS(vxReleaseImage(&ovxImgROut));
        CHECK_VX_STATUS(vxReleaseImage(&ovxImgGIn));
        CHECK_VX_STATUS(vxReleaseImage(&ovxImgGOut));
        CHECK_VX_STATUS(vxReleaseImage(&ovxImgBIn));
        CHECK_VX_STATUS(vxReleaseImage(&ovxImgBOut));
        CHECK_VX_STATUS(vxReleaseImage(&ovxImgOut));

        // Release threshold
        CHECK_VX_STATUS(vxReleaseThreshold(&hyst));

        // Release graph and context
        CHECK_VX_STATUS(vxReleaseGraph(&ovxGraph));
        CHECK_VX_STATUS(vxReleaseContext(&ovxContext));

        // ---------------------------------------------------------------------------------------

        return EXIT_SUCCESS;
    }
    catch(const CmdParser::Error& error)
    {
        cerr << "[ ERROR ] In command line: " << error.what() << std::endl
             << "Run " << argv[0] << " -h for usage info." << std::endl;
        return EXIT_FAILURE;
    }
}
