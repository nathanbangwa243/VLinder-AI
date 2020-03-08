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

#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/cmdparser.hpp>
#include <intel/vx_samples/perfprof.hpp>

#if INTEL_SAMPLE_USE_OPENCV
#include <opencv2/opencv.hpp>
#endif


#include "vx_user_census_nodes.h"


using namespace std;

class CmdParserCT: public CmdParserWithHelp
{
public:
    CmdParserCT(int argc, const char** argv) : CmdParser(argc, argv), CmdParserWithHelp(argc, argv),
        input(
            *this,
            'i',
            "input",
            "<file name>",
            "Input video file.",
            exe_dir() + "toy_flower.mp4"
        ),
        no_show(*this),
        debug_output(*this),
        frame_wait(*this),
        no_tiled(
            *this,
            0,
            "no-tiled",
            "",
            "CT non-tiled implementation. Use non-tiled implementation of Census Transform user node.",
            false
        ),
        opencl(
            *this,
            0,
            "opencl",
            "",
            "Switch to OpenCL custom kernel implementation for user node. Requires --no-tiled, because there is no tiled implementation yet.",
            false
        ),
        frames(
            *this,
            'f',
            "max-frames",
            "<integer>",
            "Maximal number of video frames to be processed or maximal number of times the same graph is executed in a loop. The whole video file will be processed by default.",
            -1
        ),
        output(*this)
    {
    }
    CmdOption<std::string> input;
    CmdOptionNoShow no_show;
    CmdOptionDebugOutput debug_output;
    CmdOptionOutputVideo output;
    CmdOptionFrameWait frame_wait;
    CmdOption<bool> no_tiled;
    CmdOption<unsigned int> frames;
    CmdOption<bool> opencl;


    virtual void parse()
    {
        CmdParserWithHelp::parse();
        if (input.getValue().empty())
        {
            throw CmdParser::Error("Input file name is required. Use --input FILE to provide input video file name.");
        }

        if(opencl.isSet() && !no_tiled.isSet())
        {
            throw CmdParser::Error("No --no-tiled option is provided when --opencl option is set. This configuration is not supported.");
        }
    }
};


PERFPROF_REGION_DEFINE(vxProcessGraph);

int main(int argc, const char *argv[])
{
    try
    {
        // Parse command line arguments.
        // See CmdParserCT for command line knobs description.
        CmdParserCT cmdparser(argc, argv);
        cmdparser.parse();

        if (cmdparser.help.isSet())
        {
            // Immediatly exit if user wanted to see the usage information only.
            return 0;
        }


        int i32frameCount = 0;
        cv::Mat inframe;

        int frameHeight, frameWidth;

        //CENTRIST histogram drawing specific
        int histSize = 256;
        int histWinWidth = 1024;
        int histWinHeight = 256;
        int binWidth = cvRound((float)histWinWidth/histSize);
        cv::Mat histImage(histWinHeight, histWinWidth, CV_8UC3, cv::Scalar(0,0,0));

        //create context
        vx_context context = vxCreateContext();
        CHECK_VX_STATUS(vxGetStatus((vx_reference)context));
        CHECK_VX_STATUS(vxDirective((vx_reference)context, VX_DIRECTIVE_ENABLE_PERFORMANCE));
        //Register log callback
        vxRegisterLogCallback(context, IntelVXSample::errorReceiver, vx_true_e);

        //Initialize CV capture
        cv::VideoCapture cap(cmdparser.input.getValue());
        if (!cap.isOpened())
        {
            std::cerr << "[ ERROR ] Cannot open file"<<endl;
            std::exit(1);
        }

        cv::VideoWriter     ocvWriter;      // video writer to write output frames (if enabled)


        //Retrieve video frame width and height from CV capture
        frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        //Create video writer for output frames which will be 2 pixels smaller in each dimension
        if(cmdparser.output.isSet())
        {
            IntelVXSample::openVideoWriterByCapture(cap, ocvWriter, cmdparser.output.getValue(), frameWidth - 2, frameHeight - 2);
        }

        //Set CENTRIST histgram normalizer
        float histNorm = (float)(frameWidth*frameHeight)/16.0f;


        //create graph
        vx_graph graph_handle = vxCreateGraph(context);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)graph_handle));


        // Create an input frame (vx_image) for the graph
        vx_image input_image = vxCreateImage(context, frameWidth, frameHeight, VX_DF_IMAGE_RGB);
        // Output vx_image frame (output of CT node)
        vx_image output_image = vxCreateImage(context, frameWidth - 2, frameHeight - 2, VX_DF_IMAGE_U8);


        // Create 'virtual' images which represent connections internal to the graph.
        // By creating them as virtual, we are acknowledging that we don't need
        // access to these images outside the scope of graph execution.

        vx_image yuv_image = vxCreateVirtualImage(graph_handle, frameWidth, frameHeight, VX_DF_IMAGE_YUV4);
        // Output vx_image frame (output of RGB to Gray node)
        vx_image y_image = vxCreateVirtualImage(graph_handle, frameWidth, frameHeight, VX_DF_IMAGE_U8);
        // Output vx_image frame (output of sobel node)
        vx_image grad_x_image = vxCreateVirtualImage(graph_handle, frameWidth, frameHeight, VX_DF_IMAGE_S16);
        vx_image grad_y_image = vxCreateVirtualImage(graph_handle, frameWidth, frameHeight, VX_DF_IMAGE_S16);
        // sobel image output
        vx_image sobel_image = vxCreateVirtualImage(graph_handle, frameWidth, frameHeight, VX_DF_IMAGE_S16);
        // CT distribution histogram data object creation
        // The second argument is the number of bins in the distribution which is histSize = 256.
        // The third argument is the start offset into the range value that marks
        // the beginning of the 1D Distribution. We set it to 0 (no offset).
        // The fourth argument is histogram range or total number of the values in histogram.
        // We set it to histSize = 256 as well. As result we will sort U8 values [0, 255]
        // in 256 histogram bins.
        vx_distribution CT_distribution = vxCreateDistribution (context, histSize, 0, histSize);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)CT_distribution));


		//Load kernels
		CHECK_VX_STATUS(vxLoadKernels(context, "census_transform_lib"));

        // Correct color space (optional).
        // The sample implements CENTRIST video descriptor calculation which is used as
        // front end for Object Recogntion applications. If training data is already collected
        // and ITU 601 color space was used during training process, it's necessary to
        // adjsut OpenVX color space atribute of the input image to achieve correspondence
        // with training data. By default OpenVX use ITU 709 color space. This step is optional
        // and can be skiped if the same OpenVX code will be used during training stage of the
        // whole objest recogntion application.
        // The code below request and prints current color space of the input image,
        // tries to switch to ITU 601 color space and finally checks and prints result color space.
        vx_enum space;
        CHECK_VX_STATUS(vxQueryImage(input_image, VX_IMAGE_SPACE, &space, sizeof(vx_enum)));
        switch (space) {
        case VX_COLOR_SPACE_NONE:
            cout<<"Current color space is VX_COLOR_SPACE_NONE " <<endl;
            break;
        case VX_COLOR_SPACE_BT601_525:
            cout<<"Current color space is VX_COLOR_SPACE_BT601_525 " <<endl;
            break;
        case VX_COLOR_SPACE_BT601_625:
            cout<<"Current color space is VX_COLOR_SPACE_BT601_625 " <<endl;
            break;
        case VX_COLOR_SPACE_BT709:
            cout<<"Current color space is VX_COLOR_SPACE_BT709 == VX_COLOR_SPACE_DEFAULT" <<endl;
            break;
        default:
            cout<<"Unknown color space " <<endl;
        }

        vx_enum desired_space = VX_COLOR_SPACE_BT601_625;
        CHECK_VX_STATUS(vxSetImageAttribute(input_image, VX_IMAGE_SPACE, &desired_space, sizeof(vx_enum)));
        CHECK_VX_STATUS(vxSetImageAttribute(yuv_image, VX_IMAGE_SPACE, &desired_space, sizeof(vx_enum)));
        CHECK_VX_STATUS(vxQueryImage(input_image, VX_IMAGE_SPACE, &space, sizeof(vx_enum)));

        switch (space) {
        case VX_COLOR_SPACE_NONE:
            cout<<"Switched to VX_COLOR_SPACE_NONE " <<endl;
            break;
        case VX_COLOR_SPACE_BT601_525:
            cout<<"Switched to VX_COLOR_SPACE_BT601_525 " <<endl;
            break;
        case VX_COLOR_SPACE_BT601_625:
            cout<<"Switched to VX_COLOR_SPACE_BT601_625 " <<endl;
            break;
        case VX_COLOR_SPACE_BT709:
            cout<<"Switched to VX_COLOR_SPACE_BT709 == VX_COLOR_SPACE_DEFAULT" <<endl;
            break;
        default:
            cout<<"Unknown color space " <<endl;
        }


        //Assemble the nodes in to the graph
        std::vector<vx_node> nodes;
        std::vector<const char*> nodeNames;

        vx_node nColorConvert = vxColorConvertNode(graph_handle, input_image, yuv_image);
        CHECK_VX_OBJ(nColorConvert);
        nodes.push_back(nColorConvert);
        nodeNames.push_back("vxColorConvertNode");

        vx_node nChannelExtract = vxChannelExtractNode(graph_handle, yuv_image, VX_CHANNEL_Y, y_image);
        CHECK_VX_OBJ(nChannelExtract);
        nodes.push_back(nChannelExtract);
        nodeNames.push_back("vxChannelExtractNode");

        vx_node nSobel3x3 = vxSobel3x3Node(graph_handle, y_image, grad_x_image, grad_y_image);
        CHECK_VX_OBJ(nSobel3x3);
        nodes.push_back(nSobel3x3);
        nodeNames.push_back("vxSobel3x3Node");

        vx_node nMagnitude = vxMagnitudeNode(graph_handle, grad_x_image, grad_y_image, sobel_image);
        CHECK_VX_OBJ(nMagnitude);
        nodes.push_back(nMagnitude);
        nodeNames.push_back("vxMagnitudeNode");


        vx_node nCensusTransform;
        if(cmdparser.no_tiled.isSet())
        {
            if(cmdparser.opencl.isSet())
            {
                nCensusTransform = vxCensusTransformOpenCLNode(graph_handle, sobel_image, output_image);
                nodeNames.push_back("vxCensusTransformOpenCLNode");
            }
            else
            {
                nCensusTransform = vxCensusTransformNode(graph_handle, sobel_image, output_image);
                nodeNames.push_back("vxCensusTransformNode");
            }
        }
        else
        {
            nCensusTransform = vxCensusTransformTiledNode(graph_handle,sobel_image,output_image);
            nodeNames.push_back("vxCensusTransformTiledNode");
        }
        CHECK_VX_OBJ(nCensusTransform);
        nodes.push_back(nCensusTransform);

        vx_node nHistogram = vxHistogramNode (graph_handle, output_image, CT_distribution);
        CHECK_VX_OBJ(nHistogram);
        nodes.push_back(nHistogram);
        nodeNames.push_back("vxHistogramNode");

        // Validating the Graph
        CHECK_VX_STATUS(vxVerifyGraph(graph_handle));

        const unsigned int frames = cmdparser.frames.getValue();
        while(1)
        {
            bool bSuccess;

            // Read the next frame
            bSuccess = cap.read(inframe);

            if (!bSuccess||(i32frameCount==frames))
            {
                cout<<"Frames are over"<<endl;
                break;
            }

            {
                // Populate the vx_image using CV::Mat contents
                vx_map_id map_id;
                cv::Mat rgb_frame = IntelVXSample::mapAsMat (input_image, VX_READ_ONLY, &map_id);
                // OpenCV capture is in BGR format and pipeline expects data in RGB
                cv::cvtColor(inframe, rgb_frame, CV_BGR2RGB);
                IntelVXSample::unmapAsMat (input_image, rgb_frame, map_id);
            }

            PERFPROF_REGION_BEGIN(vxProcessGraph);
            CHECK_VX_STATUS(vxProcessGraph(graph_handle));
            PERFPROF_REGION_END(vxProcessGraph);

            i32frameCount++;

            // Display using imshow
            if(!cmdparser.no_show.isSet())
            {
                //Nodes time break-down
                if(cmdparser.debug_output.isSet())
                {
                    IntelVXSample::drawNodesAtTimeline(&nodes[0], nodes.size(), &nodeNames[0]);
                }

                {
                    vx_map_id map_id;
                    cv::Mat gray_frame =  IntelVXSample::mapAsMat (output_image, VX_READ_ONLY, &map_id);
                    cv::imshow("Sobel3x3 + Census Transform Output",gray_frame);
                    IntelVXSample::unmapAsMat (output_image, gray_frame, map_id);
                }

                //Access histogram data and draw histogram
                void* histData = NULL;
                vx_map_id map_id;
                CHECK_VX_STATUS(vxMapDistribution (CT_distribution, &map_id, &histData, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
                histImage.setTo(cv::Scalar(0,0,0));
                //Draw histogram
                for(int i=0; i<histSize; i++)
                {
                    cv::line(histImage, cv::Point(binWidth*(i), histWinHeight - cvRound((float)histWinHeight*((float)((vx_int32*)histData)[i]/histNorm))),
                        cv::Point(binWidth*(i), histWinHeight),
                        cv::Scalar(0, 255, 0), 2, 4, 0);
                }
                //Unmap distribution resource
                CHECK_VX_STATUS(vxUnmapDistribution(CT_distribution, map_id));

                cv::imshow("CENTRIST - Census Transform Histogram values", histImage );
                int key = cv::waitKey(cmdparser.frame_wait.getValue()) & 0xff;
                if(key == 27) // ESC is pressed. exit
                    break;
            }

            if(cmdparser.output.isSet())
            {
                vx_map_id map_id;
                cv::Mat gray_frame =  IntelVXSample::mapAsMat (output_image, VX_READ_ONLY, &map_id);
                cv::Mat outframe;
                cvtColor(gray_frame,outframe,CV_GRAY2BGR);
                ocvWriter.write(outframe);
                IntelVXSample::unmapAsMat (output_image, gray_frame, map_id);
            }
        }
        cout<<"Number of processed frames is : "<<  i32frameCount <<endl;

        //Resource clean up
        //Release nodes
        CHECK_VX_STATUS(vxReleaseNode(&nColorConvert));
        CHECK_VX_STATUS(vxReleaseNode(&nChannelExtract));
        CHECK_VX_STATUS(vxReleaseNode(&nSobel3x3));
        CHECK_VX_STATUS(vxReleaseNode(&nMagnitude));
        CHECK_VX_STATUS(vxReleaseNode(&nCensusTransform));
        CHECK_VX_STATUS(vxReleaseNode(&nHistogram));

        CHECK_VX_STATUS(vxReleaseImage(&input_image));
        CHECK_VX_STATUS(vxReleaseImage(&yuv_image));
        CHECK_VX_STATUS(vxReleaseImage(&y_image));
        CHECK_VX_STATUS(vxReleaseImage(&grad_x_image));
        CHECK_VX_STATUS(vxReleaseImage(&grad_y_image));
        CHECK_VX_STATUS(vxReleaseImage(&sobel_image));
        CHECK_VX_STATUS(vxReleaseImage(&output_image));
        CHECK_VX_STATUS(vxReleaseDistribution (&CT_distribution));
        CHECK_VX_STATUS(vxReleaseGraph (&graph_handle));
        CHECK_VX_STATUS(vxReleaseContext(&context));
    }
    catch (const CmdParser::Error& error)
    {
        cerr
            << "[ ERROR ] In command line: " << error.what() << "\n"
            << "Run " << argv[0] << " -h for usage info.\n";
        return 1;
    }
    catch (const std::exception& error)
    {
        cerr << "[ ERROR ] " << error.what() << "\n";
        return 1;
    }
    catch (...)
    {
        cerr << "[ ERROR ] Unknown/internal exception happened.\n";
        return 1;
    }

}
