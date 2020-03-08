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

#include <string>
#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/cmdparser.hpp>
#include <intel/vx_samples/perfprof.hpp>


using namespace std;

class CmdParserAutoContrast: public CmdParserWithHelp
{
public:
    CmdParserAutoContrast(int argc, const char** argv) : CmdParser(argc, argv), CmdParserWithHelp(argc, argv),
        input(
            *this,
            'i',
            "input",
            "<file name>",
            "Input image file.",
            "./low_contrast_vga.jpg"
            ),
        output(*this),
        as_gray(
            *this,
            'g',
            "gray",
            "read image as gray-scale",
            "Read image file in gray-scale (single channel).",
            false
            ),
        noshow(*this),
        loops(
            *this,
            'l',
            "loops",
            "number of times the graph is looped",
            "Number of times the same graph is executed in a loop.",
            1
            )
    {
    }
    CmdOption<std::string> input;
    CmdOptionOutputImage output;
    CmdOption<bool> as_gray;
    CmdOptionNoShow noshow;
    CmdOption<unsigned int> loops;

    virtual void parse()
    {
        CmdParserWithHelp::parse();
        if (input.getValue().empty())
        {
            throw CmdParser::Error("Input file name is required. Use --input FILE to provide input video file name.");
        }

    }
};

#if INTEL_SAMPLE_USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

PERFPROF_REGION_DEFINE(vxProcessGraph);

int main(int argc, const char* argv[])
{
    try
    {
        // Parse command line arguments.
        // See CmdParserVideoStabilization for command line knobs description.
        CmdParserAutoContrast cmdparser(argc, argv);
        cmdparser.parse();

        if (cmdparser.help.isSet())
        {
            // Immediatly exit if user wanted to see the usage information only.
            return 0;
        }

        /***************OpenVX init************************/
        vx_status status = VX_SUCCESS;
        vx_context context = vxCreateContext();
        status = vxGetStatus((vx_reference)context);
        CHECK_VX_STATUS(status);
        vxRegisterLogCallback(context, IntelVXSample::errorReceiver, vx_true_e);
        vx_graph graph = vxCreateGraph(context);
        status = vxGetStatus((vx_reference)graph);
        CHECK_VX_STATUS(status);
        // a vector for OpenVX nodes (merely for resource clean-up purposes)
        std::vector<vx_node> vxnodes;
        // a vector for OpenVX images (merely for resource clean-up purposes)
        std::vector<vx_image> vximages;
        /***************\OpenVX init************************/

        /***************inputs and outputs ************************/
        int inputWidth, inputHeight;
        int inputChannels = cmdparser.as_gray.getValue() ? 1 : 3;
#if INTEL_SAMPLE_USE_OPENCV
        std::cerr << "[ INFO ] Reading input file using OpenCV file I/O" <<
                     cmdparser.input.getValue() << endl;
        cv::Mat src = cv::imread(cmdparser.input.getValue(), cmdparser.as_gray.getValue() ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED);
        if(!src.data)
        {
            std::cerr << "[ ERROR ] cannot read input file\n";
            std::exit(1);
        }
        inputWidth = src.size().width;
        inputHeight = src.size().height;
        inputChannels = src.channels();
        std::cerr << "[ INFO ] Input file ok: " << inputWidth <<"x"<<inputHeight<<
                    " and "<< inputChannels<<" channel(s)"<<endl;
        //structure used by the host to address image pixels
        vx_imagepatch_addressing_t frameFormat;
        //dimension in x, in pixels
        frameFormat.dim_x = src.cols;
        //dimension in y, in pixels
        frameFormat.dim_y = src.rows;
        //distance (in bytes) from a pixel to the next adjacent pixel in the positive x direction
        frameFormat.stride_x = (vx_uint32)src.elemSize();
        //distance (in bytes) from a pixel to the next adjacent pixel in the positive y direction
        frameFormat.stride_y = (vx_uint32)src.step;
        //scaling from the primary plane (zero indexed plane) to this plane (relevant only to the multi-plane images)
        frameFormat.scale_x = VX_SCALE_UNITY;
        frameFormat.scale_y = VX_SCALE_UNITY;
        //number of pixels to skip to arrive at the next pixel (relevant only to the multi-plane images)
        //e.g. on a plane that is half-scaled, the step of 2 would indicate that every other pixel is an alias
        frameFormat.step_x = 1;
        frameFormat.step_y = 1;
        void* frameData = src.data;
#else
        inputWidth = 512;
        inputHeight = 512;
        std::cerr << "[ INFO ] Input image generated: " << inputWidth << "x" << inputHeight <<
                    " and " << inputChannels << " channel(s)" << endl;
        vx_imagepatch_addressing_t frameFormat;
        frameFormat.dim_x = inputWidth;
        frameFormat.dim_y = inputHeight;
        frameFormat.stride_x = inputChannels*sizeof(unsigned char);
        frameFormat.stride_y = frameFormat.stride_x*inputWidth;
        frameFormat.scale_x = VX_SCALE_UNITY;
        frameFormat.scale_y = VX_SCALE_UNITY;
        frameFormat.step_x = 1;
        frameFormat.step_y = 1;
        const int mem = inputWidth*inputHeight*inputChannels*sizeof(unsigned char);
        void* frameData = malloc(mem);
        memset(frameData, 128, mem);//alternatively you can populate image data with any other pattern like color gradient
#endif
        //creating input image, by wrapping the host side pointer (e.g. when the frame comes from OpenCV)
        vx_image imageOrig = vxCreateImageFromHandle(
            context,
            (1==inputChannels) ? VX_DF_IMAGE_U8 : VX_DF_IMAGE_RGB,
            &frameFormat,
            &frameData,
            VX_MEMORY_TYPE_HOST
            );
        vximages.push_back(imageOrig);
        //creating output image, which matches the input
        vx_image imageRes = vxCreateImage(context, inputWidth, inputHeight, (inputChannels== 1) ? VX_DF_IMAGE_U8 : VX_DF_IMAGE_RGB);
        if (imageOrig == NULL || imageRes == NULL)
        {
            std::cerr << "[ ERROR ] cannot create input/output images\n";
            std::exit(1);
        }
         vximages.push_back(imageRes);
        /***************\inputs and outputs ************************/

        /***************actual graph machinery************************/
        if (3 == inputChannels)
        {
            //a virtual image that will store the results of input image conversion into the NV12 format
            vx_image imageNV12 = vxCreateVirtualImage(graph, inputWidth, inputHeight, VX_DF_IMAGE_NV12);
            vximages.push_back(imageNV12);
            //a virtual image that will store the results of image processing (in the NV12 format)
            vx_image imageNV12Eq     = vxCreateVirtualImage(graph, inputWidth, inputHeight, VX_DF_IMAGE_NV12);
            vximages.push_back(imageNV12Eq);

            //a separate virtual image to store extracted Y plane, for processing            
            vx_image imageChannelY  = vxCreateVirtualImage(graph, inputWidth, inputHeight, VX_DF_IMAGE_U8);
            vximages.push_back(imageChannelY);
            //a separate virtual image to store results of processing the Y plane            
            vx_image imageChannelYEq = vxCreateVirtualImage(graph, inputWidth, inputHeight, VX_DF_IMAGE_U8);
            vximages.push_back(imageChannelYEq);
            //a separate virtual image to store extracted U plane, left intact            
            vx_image imageU = vxCreateVirtualImage(graph, inputWidth / 2, inputHeight / 2, VX_DF_IMAGE_U8);
            vximages.push_back(imageU);
           //a separate virtual image to store extracted V plane, left intact            
            vx_image imageV = vxCreateVirtualImage(graph, inputWidth / 2, inputHeight / 2, VX_DF_IMAGE_U8);
            vximages.push_back(imageV);
            /*********************************************************************************************
                                         >|channel extract|->Y->|equalize_hist|--
                                        /                                        \
            RGB (input)->|convert|->NV12->|channel extract|->U------------------ ->|convert|->RGB (output)
                                        \                                        /
                                         >|channel extract|->V-------------------  
            *********************************************************************************************/
            vxnodes.push_back(
                vxColorConvertNode(graph, imageOrig, imageNV12));
            vxnodes.push_back(
                vxChannelExtractNode(graph, imageNV12, VX_CHANNEL_Y, imageChannelY));
            vxnodes.push_back(
                vxEqualizeHistNode(graph, imageChannelY, imageChannelYEq));
            vxnodes.push_back(
                vxChannelExtractNode(graph, imageNV12, VX_CHANNEL_U, imageU));
            vxnodes.push_back(
                vxChannelExtractNode(graph, imageNV12, VX_CHANNEL_V, imageV));
            vxnodes.push_back(
                vxChannelCombineNode(graph, imageChannelYEq, imageU, imageV, 0, imageNV12Eq));
            vxnodes.push_back(
                vxColorConvertNode(graph, imageNV12Eq, imageRes));
        }
        else
        {
            vxnodes.push_back(
                vxEqualizeHistNode(graph, imageOrig, imageRes));
        }

        status = vxVerifyGraph(graph);
        CHECK_VX_STATUS(status);
        std::cerr << "[ INFO ] Verifying the graph: OK " << graph << "\n";
        const unsigned int loops = cmdparser.loops.getValue();
        //looping the same graph to measure the performance more accurately ( e.g. by averaging the time)
        for (unsigned int i = 0; i < loops; i++)
        {
            PERFPROF_REGION_BEGIN(vxProcessGraph);
            status = vxProcessGraph(graph);
            PERFPROF_REGION_END(vxProcessGraph);
            CHECK_VX_STATUS(status);
        }
         std::cerr << "[ INFO ] Graph execution is OK " << graph << "\n";
        /***************\actual graph machinery ************************/

        /*************** saving and displaying the results************************/
#if INTEL_SAMPLE_USE_OPENCV
        vx_map_id map_id;
        cv::Mat res = IntelVXSample::mapAsMat(imageRes, VX_READ_ONLY, &map_id);
        if (cmdparser.output.isSet())
        {
            cv::imwrite(cmdparser.output.getValue(), res);
        }
        if(!cmdparser.noshow.isSet())
        {
            cv::imshow("Input image", src);
            cv::imshow("Processed image:", res);
            cv::waitKey(0);
        }
        IntelVXSample::unmapAsMat(imageRes,res,map_id);
#else
        free(frameData);
#endif
        /***************\saving and displaying the results************************/

        /*************** clean-up ************************/
        for (int n = 0; n < vxnodes.size(); n++)
        {
            //release each node
            status = vxReleaseNode(&vxnodes[n]);
            CHECK_VX_STATUS(status)
        }
        for (int n = 0; n < vximages.size(); n++)
        {
            //release each image
            status = vxReleaseImage(&vximages[n]);
            CHECK_VX_STATUS(status)
        }
        //release graph
        status = vxReleaseGraph(&graph);
        CHECK_VX_STATUS(status)
        //release context
        status = vxReleaseContext(&context);
        CHECK_VX_STATUS(status)
        /*************** \clean-up ************************/

        printf("End of execution\n");
        return 0;
    }
    /***************\ saving and displaying the results************************/
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
