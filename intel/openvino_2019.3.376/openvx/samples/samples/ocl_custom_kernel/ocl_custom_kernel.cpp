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


#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <VX/vx.h>
#include <VX/vx_intel_volatile.h>
#include <iostream>

#include <intel/vx_samples/perfprof.hpp>
#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/cmdparser.hpp>


// Define necessary IDs required for OpenCL custom kernel registration later in the main function
// They are similar to regular OpenVX user kernel, no OpenCL specific here

// Library ID is used to construct a unique Kernel ID later; Library ID is chosen in a way to avoid
// conflicts with other group of kernels.
// Value is chosen for illustrative purpuses to avoid conflicts with other samples
#define VX_LIBRARY_SAMPLE_OCL_CUSTOM_KERNEL (0x7)

// OpenCL custom kernel ID, formed as for a regulr user OpenVX kernels
enum vx_kernel_intel_sample_ocl_custom_kernel_e {
    VX_KERNEL_SAMPLE_OCL_CUSTOM = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_OCL_CUSTOM_KERNEL) + 0x0
};


// The following 3 functions are callbacks for OpenCL custom kernel
// The signatures and guidelines for the code in the callbacks are the same as for
// regular OpenVX user nodes. See how they are used in the main function later in this file.

// Input and output validator is called for each input parameter of OpenVX custom node
vx_status VX_CALLBACK oclCustomKernelValidate(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    if(num!=2)
    {
        return status;
    }

    vx_df_image imageFormat = 0;
    if(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &imageFormat, sizeof(imageFormat)) == VX_SUCCESS)
    {
        if (imageFormat == VX_DF_IMAGE_RGB)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "oclCustomKernel Validation failed: invalid input image format\n");
            return status;
        }
    }

    vx_uint32 inputWidth = 0;
    vx_uint32 inputHeight = 0;

    //Query the input image
    status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &inputWidth, sizeof(inputWidth));
    status |= vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &inputHeight, sizeof(inputHeight));


    //Input/output is of RGB type 
    //Set width and height for validation as well
    status |= vxSetMetaFormatAttribute(metas[1], VX_IMAGE_WIDTH, &inputWidth, sizeof(inputWidth));
    status |= vxSetMetaFormatAttribute(metas[1], VX_IMAGE_HEIGHT, &inputHeight, sizeof(inputHeight));
    status |= vxSetMetaFormatAttribute(metas[1], VX_IMAGE_FORMAT, &imageFormat, sizeof(imageFormat));

    return status;
}



vx_status VX_CALLBACK oclCustomKernelInitialize (vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    // This function is empty for this sample.
    // In this case, alternatively you can provide a null pointer to oclCustomKernelInitialize and don't
    // implement the function.
    return VX_SUCCESS;
}


vx_status VX_CALLBACK oclCustomKernelDeinitialize (vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    // This function is empty for this sample.
    // In this case, alternatively you can provide a null pointer to oclCustomKernelInitialize and don't
    // implement the function.
    return VX_SUCCESS;
}


/////////////////////////////////////////////////////////////////////////////////////////////////


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
            exe_dir() + "toy_flower.mp4"
        );

        CmdOptionOutputVideo output(cmd);

        CmdOptionNoShow no_show(cmd);

        CmdOptionFrameWait frame_wait(cmd);

        cmd.parse();

        if (cmd.help.isSet())
        {
            // Immediately exit if user wanted to see the usage information only.
            return 0;
        }


        // ---------------------------------------------------------------------------------------
        // Define necessary OpenVX and OpenCV objects.
        // OpenCV is used as a convinient way to load video frames from a file and store them to
        // a file or display on the screen. There is nothing specific related to OpenCL custom
        // kernel here yet.

        cv::VideoCapture    ocvCapture;     // video caputure to read input frames from video file
        cv::VideoWriter     ocvWriter;      // video writer to write output frames (if enabled)
        int                 width, height;  // width and height of input image
        vx_context          ovxContext;     // OpenVX context
        vx_graph            ovxGraph;       // OpenVX graph

        // Define performance counters to measure execution time
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
            // open output file for writing with the same properties as input file

            std::cout << "Open file " << output.getValue() << " for writing" << std::endl;

            int fourcc =(int)ocvCapture.get(cv::CAP_PROP_FOURCC);
            int fps =(int)ocvCapture.get(cv::CAP_PROP_FPS);

            ocvWriter.open(output.getValue(), fourcc, fps, cv::Size(width, height));
            if(!ocvWriter.isOpened())
            {
                std::cerr << "[ ERROR ] Can not open output file: " << output.getValue() << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        // Create OpenVX context
        ovxContext = vxCreateContext();
        CHECK_VX_OBJ(ovxContext);

        // Register user-defined callback function to receive message from inside OpenVX run-time
        // The function IntelVXSample::errorReceiver is used here from common sample infrastructure
        // for convenience. This function will just redirect all the messages to std::cout.
        vxRegisterLogCallback(ovxContext, IntelVXSample::errorReceiver, vx_true_e);

        // Create an OpenVX graph that will be populated with node(s) later
        ovxGraph = vxCreateGraph(ovxContext);
        CHECK_VX_OBJ(ovxGraph);

        // ---------------------------------------------------------------------------------------

        // Create image data objects to hold input and output data in the pipeline

        // Input RGB image; this will be read from a file and passed as an input to OpenCL node
        vx_image    ovxImgIn       = vxCreateImage(ovxContext, width, height, VX_DF_IMAGE_RGB);
        CHECK_VX_OBJ(ovxImgIn);

        // output RGB image; will be passed as an output parameter for OpenCL node and will be
        // stored to file or displayed on the screen`
        vx_image    ovxImgOut      = vxCreateImage(ovxContext, width, height, VX_DF_IMAGE_RGB);
        CHECK_VX_OBJ(ovxImgOut);

        // ---------------------------------------------------------------------------------------
        // Let's prepare OpenCL program and register OpenCL kernel as an OpenVX kernel in the run-time

        // When an OpenVX kernels are organized in a separate module (dynamicly linked library),
        // then all these steps should be placed in vxPublishKernels function of the module
        // similarly to regular OpenVX user node kernels.
        // Here the separate module is not used, and kernels are registered directly from
        // the main application. It is a convenient way for kernels that aren't reused
        // in multiple applications.

        // First prepare source of an OpenCL program. It is just read from a file.
        // The API that we use below also accepts an OpenCL program in a binary form,
        // here the source form of a program is demonstrated only. Program in a binary form
        // is prepared similarly.
        std::string oclSource = IntelVXSample::readTextFile(exe_dir() + "source.cl");

        // Register OpenCL program as an OpenVX Device Kernel Library. Device kernel library is
        // a container for device kernels. So to be able to call at least one OpenCL device kernel, a device
        // kernel library should be created first with an OpenCL program that includes the kernel(s)
        // implementation.
        //
        // Two types of libraries are supported:
        //   - VX_OPENCL_LIBRARY_SOURCE: created from OpenCL program in source form
        //   - VX_OPENCL_LIBRARY_BINARY: created from OpenCL program in binary form
        //
        vx_device_kernel_library_intel kernelLibrary = vxAddDeviceLibraryIntel(
            ovxContext,
            oclSource.length(),         // Size in bytes of the OpenCL source
            oclSource.c_str(),          // Pointer to the program source or binary

            "",                               // Pointer to the compilation flag string
                                              // Convenient way to pass host-side defined constants as macros
                                              // to OpenCL program.
                                              // Ignored if library_type == VX_OPENCL_LIBRARY_BINARY

            VX_OPENCL_LIBRARY_SOURCE_INTEL,   // VX_OPENCL_LIBRARY_SOURCE or VX_OPENCL_LIBRARY_BINARY
            "intel.gpu"                       // Default vx_target name that should run kernels; see OpenVX Target API for reference
        );

        CHECK_VX_OBJ(kernelLibrary);

        // For each OpenVX node parameter, parameter type and direction are defined in
        // the following two arrays. Index in the arrays corresponds to an index of a parameter.
        // So we have a node with two parameters here: both images, the first one is an input parameter
        // and the second one is an output parameter
        vx_enum param_types[] = { VX_TYPE_IMAGE, VX_TYPE_IMAGE };
        vx_enum param_directions[] = { VX_INPUT, VX_OUTPUT };

        // Call vxIntelAddDeviceKernel serves the similar purpose as a regular vxAddKernel function call
        // but accepts OpenCL kernel specific arguments
        vx_kernel oclKernel = vxAddDeviceKernelIntel(
            ovxContext,
            "com.intel.sample.ocl_custom_kernel.oclKernel",  // The name of the kernel in OpenVX nomenclature
            VX_KERNEL_SAMPLE_OCL_CUSTOM,    // enum value for OpenVX kernel, formed in the same way as for regular user nodes
            kernelLibrary,                  // kernel library, just created earlier here in the code
            "oclKernel",                    // OpenCL kernel name as it appears in OpenCL program
            2,                              // number of OpenVX Kernel parameters
            param_types,                    // Types of parameters: array
            param_directions,               // Directions for each parameter: array
            VX_BORDER_UNDEFINED,       // border mode: this sample doesn't care; other values are not supported yet

            // the following 3 parameters are similar to regular user nodes
            oclCustomKernelValidate,        // input and output parameter validator
            oclCustomKernelInitialize,      // node initialization
            oclCustomKernelDeinitialize     // node deinitialization
        );

        CHECK_VX_OBJ(oclKernel);

        // Each work-item in the OpenCL kernel process a single pixel by default.
        // All input images should be the same size and this size is used as a global size for
        // the kernel. X image dimension is mapped to 0-th dimension of the NDRange and Y image
        // dimension is mapped to 1-th dimension of the NDRange.

        // Number of pixels processed by a single work item can be modified for each dimension
        // by calling vxSetKernelAttribute with VX_OPENCL_WORK_ITEM_XSIZE and VX_OPENCL_WORK_ITEM_YSIZE. 
        // In case of this example it is set to default values 1x1, because the OpenCL kernel is written
        // to process a single pixel. If values greater than 1 are provided, the global size of NDRange
        // is reduced accordingly: width/xAreaSize x height/yAreaSize, and the kernel should be modified
        // to process multiple pixels in one work item.
        vx_size  xAreaSize = 1;
        vx_size  yAreaSize = 1;
        CHECK_VX_STATUS(vxSetKernelAttribute(oclKernel, VX_OPENCL_WORK_ITEM_XSIZE_INTEL, &xAreaSize, sizeof(xAreaSize)));
        CHECK_VX_STATUS(vxSetKernelAttribute(oclKernel, VX_OPENCL_WORK_ITEM_YSIZE_INTEL, &yAreaSize, sizeof(yAreaSize)));

		//Finalize kernel. The kernel is ready to be used.
		CHECK_VX_STATUS(vxFinalizeKernel(oclKernel));

        // ---------------------------------------------------------------------------------------
        // After kernel is created successfully, now it's time to create a node in the graph with this kernel
        // and set input and output parameters. It is absolutelly identical to regular user node creation,
        // no OpenCL specifics here.

        vx_node node = vxCreateGenericNode(ovxGraph, oclKernel);
        CHECK_VX_OBJ(node);
        CHECK_VX_STATUS(vxSetParameterByIndex(node, 0, (vx_reference)ovxImgIn));
        CHECK_VX_STATUS(vxSetParameterByIndex(node, 1, (vx_reference)ovxImgOut));

        // ---------------------------------------------------------------------------------------

        // Before running the main loop over frames we need to verify the graph as usual
        CHECK_VX_STATUS(vxVerifyGraph(ovxGraph));

        vx_map_id map_id_in;
        vx_map_id map_id_out;
        // The main loop iterating over frames
        for(unsigned int frame=0; frame < (unsigned int)max_frames.getValue(); frame++)
        {
            {
                // This is read frame section that is measured separatly
                PERFPROF_REGION_AUTO(ReadFrame);

                // Read input image from video file directly to vx_image by mapping it as cv::Mat
                cv::Mat ocvImgIn = IntelVXSample::mapAsMat(ovxImgIn, VX_WRITE_ONLY, &map_id_in);

                // Read frame
                if(!ocvCapture.read(ocvImgIn))
                {
                    std::cout << std::endl << "Break on ocvCapture.read";
                    IntelVXSample::unmapAsMat(ovxImgIn, ocvImgIn, map_id_in);
                    break; // break if there is no any image
                }

                // Unmap vx_image allowing using it inside OpenVX graph
                IntelVXSample::unmapAsMat(ovxImgIn, ocvImgIn, map_id_in);
            }

            {
                // Run OpenVX graph on new capture frame
                PERFPROF_REGION_AUTO(vxProcessGraph);
                CHECK_VX_STATUS(vxProcessGraph(ovxGraph));
            }

            if(!no_show.isSet() || output.isSet())
            {
                cv::Mat ocvImgOut = IntelVXSample::mapAsMat(ovxImgOut, VX_READ_ONLY, &map_id_out);
                if(!no_show.isSet())
                {
                    cv::Mat ocvImgIn = IntelVXSample::mapAsMat(ovxImgIn, VX_READ_ONLY, &map_id_in);
                    cv::imshow("Input", ocvImgIn);
                    IntelVXSample::unmapAsMat(ovxImgIn, ocvImgIn, map_id_in);
                    cv::imshow("Output", ocvImgOut);
                    if(frame == 0)
                    {
                        // Move windows at the first frame to avoid default
                        // positioning that sometime is not convenient
                        cv::moveWindow("Input", 0, 0);
                        cv::moveWindow("Output", width/2, height/2);
                    }
                }
                if(ocvWriter.isOpened())
                {
                    ocvWriter.write(ocvImgOut);
                }
                IntelVXSample::unmapAsMat(ovxImgOut, ocvImgOut, map_id_out);

                if(!no_show.isSet())
                {
                    // read key value and process it
                    int key=cv::waitKey(frame_wait.getValue()) & 0xff;
                    if(key == 27) // ESC is pressed. exit
                        break;
                }
            }// visualisation end

        }//next frame

        // ---------------------------------------------------------------------------------------

        // Release all stuff

        std::cout << std::endl << "Release data..." << std::endl;

        // Release node(s)
        CHECK_VX_STATUS(vxReleaseNode(&node));

        // Release images
        CHECK_VX_STATUS(vxReleaseImage(&ovxImgIn));
        CHECK_VX_STATUS(vxReleaseImage(&ovxImgOut));

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
    catch(const SampleError& error)
    {
		cerr << "[ ERROR ] Sample error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
