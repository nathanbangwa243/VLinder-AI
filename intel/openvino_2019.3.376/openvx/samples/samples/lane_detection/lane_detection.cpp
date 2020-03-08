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
// Include main OpenVX file
#include <VX/vx.h>
// The sample uses Hough Transform node that is a OpenVX extention.
// So the additional include has to be used
#include <VX/vx_intel_volatile.h>

// Include OpenCV headers for reference pipline and visualization implementation
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//Include common sample headers
#include <intel/vx_samples/perfprof.hpp>
#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/cmdparser.hpp>

#include "collect_lane_marks.hpp"


//--- define parameters for pipeline ---

// width and height for intermidiate images
#define INTERNAL_WIDTH  220
#define INTERNAL_HEIGHT 240

// define filter size and filter kernel values
// the size and shape of the filter depends on real lane markers
#define FILTER2D_W  3
#define FILTER2D_H  11
static short gFilter2D_Data[FILTER2D_W*FILTER2D_H] =
{
    -5, -5, -5,
    -5, -5, -5,
    -5, -5, -5,
     6,  6,  6,
     6,  6,  6,
     6,  6,  6,
     6,  6,  6,
     6,  6,  6,
    -5, -5, -5,
    -5, -5, -5,
    -5, -5, -5
};
// Scale factor for filter. It has to be power of 2 for OpenVX 1.0
#define FILTER2D_SCALE  16

// Threshold for filtered image to detect lane marker pixels
#define THRESHOLD_VALUE 170

//parameters for HoughTransformP function
#define HOUGH_RHO_RESOLUTION   2
#define HOUGH_THETA_RESOLUTION (CV_PI / 180)
#define HOUGH_THRESHOLD        30
#define HOUGH_MIN_LINE_LENGTH  16
#define HOUGH_MAX_LINE_GAP     6
#define HOUGH_MAX_LINES        100

// Below are several parameters to define road area for mapping and processing.

// Relative height of vanishing point or distance from bottom camera view border to horizon line.
// It is supposed that horizon is parallel to camera view borders
#define REMAP_HORIZON      0.435f

// Relative height of point where left and right lines can be distinguished.
// It is distance from bottom camera view border
// to the farest point on the road that will be mapped.
// It has to be slightly less than REMAP_HORIZON
#define REMAP_FAR_CLIP      ((REMAP_HORIZON)-0.05f)

// Relative height of hood’s front not to detect line segments on it.
// It is distance from bottom camera border
// to the nearest point on the road that will be mapped.
#define REMAP_NEAR_CLIP     0.17f

// Relative width of the mapped road area for the REMAP_NEAR_CLIP
// This value correlates with road width that will be processed
#define REMAP_NEAR_WIDTH    0.95f
//   ^
//   |               Camera View
//   +1.0 +------------------------------------+------- top border (1.0)
//   |    |                                    |
//   |    |                                    |
//   |    |                                    |
//   |    |                                    |
//   |    |-----------------.------------------|-'-REMAP_HORIZON (0.435)
//   +0.5 |                . .                 | |
//   |    |               1---3 - - - - - - - -|-+-'-REMAP_FAR_CLIP (0.385)
//   |    |              /M A P\               | | |
//   |    |             /A R E A\              | | |
//   |    |            0---------2 - - - - - - |-+-+-'-REMAP_NEAR_CLIP (0.17)
//   |    |                                    | | | |
//   +0.0 +------------------------------------+-'-'-'-- bottom border (0.0)
//                     |         |
//                     '---------'
//                     NEAR_WIDTH

static void calcLaneArea(int width, int height, cv::Point2f laneArea[4])
{
    // Calc lane area from REMAP_HORIZON, REMAP_FAR_CLIP, REMAP_NEAR_CLIP and REMAP_NEAR_WIDTH parameters.
    // In general any 4 points can be defined below
    float dxFar = 0.5f * REMAP_NEAR_WIDTH * (REMAP_FAR_CLIP - REMAP_HORIZON) / (REMAP_NEAR_CLIP - REMAP_HORIZON);
    float dxNear = 0.5f * REMAP_NEAR_WIDTH;
    laneArea[0].x = (0.5f-dxNear)*width; laneArea[0].y = (1.0f-REMAP_NEAR_CLIP)*height;
    laneArea[1].x = (0.5f-dxFar) *width; laneArea[1].y = (1.0f-REMAP_FAR_CLIP) *height;
    laneArea[2].x = (0.5f+dxNear)*width; laneArea[2].y = (1.0f-REMAP_NEAR_CLIP)*height;
    laneArea[3].x = (0.5f+dxFar) *width; laneArea[3].y = (1.0f-REMAP_FAR_CLIP) *height;
}
// function that calculate perspective transformation matrix from camera (width,height) map area to
// whole internal top view (0,0)-(INTERNAL_WIDTH, INTERNAL_HEIGHT)
static cv::Mat calcPerspectiveTransform(int width, int height)
{// calc perspective transform from 4 points
    // define persepctive transform by 4 points placed on road
    cv::Point2f srcP[4];
    calcLaneArea(width,height,srcP);
    cv::Point2f dstP[4] =
    {
        {0                      , 0},
        {(float)(INTERNAL_WIDTH), 0},
        {0                      , (float)(INTERNAL_HEIGHT)},
        {(float)(INTERNAL_WIDTH), (float)(INTERNAL_HEIGHT)}
    };
    return cv::getPerspectiveTransform(dstP,srcP);
}

// this class implements the reference pipeline using OpenCV functionality
class OCVPipeline
{
public:
    cv::Mat                 m_ImgGray;          // gray image with original no resized frame
    cv::Mat                 m_ImgWarped;        // warped gray image
    cv::Mat                 m_Edges16S;         // 16S image to store convolution result
    cv::Mat                 m_Edges8U;          // 8U image to store scaled convolution result
    cv::Mat                 m_EdgesThresholded; // binary image with lane mark pixel candidates
    cv::Mat                 m_H;                // matrix for perspective transformation
    std::vector<cv::Vec4i>  m_Lines;            // array of line segments returned by HoughP transform
    cv::Size                m_InternalSize;     // size of internal image (INTERNAL_WIDTH, INTERNAL_HEIGHT)

    // this function makes initialization for OpenCV pipline including perspective transform parameters calculation
    void Init(int width, int height)
    {
        //declare destination size for warping
        m_InternalSize = cv::Size(INTERNAL_WIDTH, INTERNAL_HEIGHT);
        m_ImgGray.create(width, height, CV_8UC1);
        m_ImgWarped.create(m_InternalSize, CV_8UC1);
        m_Edges16S.create(m_InternalSize, CV_16SC1);
        m_Edges8U.create(m_InternalSize, CV_8UC1);
        m_EdgesThresholded.create(m_InternalSize, CV_8UC1);
        m_H = calcPerspectiveTransform(width,height);
    };
    // this function makes all processing using OpenCV functions
    void    Process(cv::Mat& inp)
    {
        // convert color image into gray scale image
        cv::cvtColor(inp,m_ImgGray,cv::COLOR_BGR2GRAY);

        //warp grayimage into smaller image to extract and align road plane area
        cv::warpPerspective(m_ImgGray, m_ImgWarped, m_H, m_InternalSize, cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);

        //detect lane mark pixels on warped image using linear filter
        cv::Mat filter(FILTER2D_H,FILTER2D_W, CV_16SC1, (void*)gFilter2D_Data);
        filter2D(m_ImgWarped, m_Edges16S, CV_16S, filter);
        m_Edges16S.convertTo(m_Edges8U,CV_8U,1.0f/FILTER2D_SCALE);

        //threshold filter responce to get binary images for Hough line transform
        cv::threshold(m_Edges8U,m_EdgesThresholded,THRESHOLD_VALUE,255,cv::THRESH_BINARY);

        //detect lines on binary image result and put it into m_Lines array
        cv::HoughLinesP(m_EdgesThresholded,m_Lines,HOUGH_RHO_RESOLUTION, HOUGH_THETA_RESOLUTION, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);
    };
};

int main(int argc, const char** argv)
{
    try
    {
        // define CMD line parameters and parse it
        CmdParserWithHelp   cmd( argc, argv);
        CmdOption<int>   max_frames(
            cmd,
            0,
            "max-frames","<integer>",
            "Number of frames from input video file to be read. May be useful for benchmarking purposes and/or when "
                "input video file is to large to be processed completely. "
                "-1 means that entire file is processed.",
            -1
        );
        CmdOption<string>   input(
            cmd,
            'i',
            "input",
            "<file name>",
            "Input video file.",
            exe_dir() + "road_lane.mp4"
        );
        CmdOptionNoShow no_show(cmd);
        CmdOptionDebugOutput debug_output(cmd);
        CmdOption<bool> disable_ref(
            cmd,
            0,
            "disable-ref",
            "",
            "Disable reference OpenCV implementation.",
            false);
        CmdOptionFrameWait frame_wait(cmd);
        CmdOptionOutputVideo output(cmd);
        cmd.parse();
        if(cmd.help.isSet())
        {
            // Immediately exit if user wanted to see the usage information only.
            return EXIT_SUCCESS;
        }

        // other local variables
        int                 vis = debug_output.isSet() ? 2 : 1;
        OCVPipeline         ocvPipeline;    // reference OpenCV pipline
        cv::VideoCapture    ocvCapture;     // video caputure to read input frames
        cv::Mat             ocvInpBGR;      // input image captured by OpenCV in in BGR format
        int                 width, height;  // width and height of input image
        vx_context          ovxContext;     // OpenVX context
        vx_graph            ovxGraph;       // OpenVX graph

        //some variable and modules for final processing for OpenCV and OpenVX
        std::vector<cv::Vec4i>  linesOVX; // array of line segments returned by HoughP transform
        CollectLaneMarks        finalOVX;
        CollectLaneMarks        finalOCV;

        //opencv images for debug purposes
        cv::Mat outDebugOVX;
        cv::Mat outDebugOCV;

        // define performance counter to measure execution time
        PERFPROF_REGION_DEFINE(vxProcessGraph)
        PERFPROF_REGION_DEFINE(ProcessOpenCVReference)
        PERFPROF_REGION_DEFINE(ReadFrame)

        // open input video file
        if(!ocvCapture.open(input.getValue()))
        {
            std::cerr << "[ ERROR ] " << input.getValue() << " is not opened" << std::endl;
            return EXIT_FAILURE;
        }
        std::cout <<  input.getValue() << " is opened" << std::endl;

        // get width and height of input video
        width = (int)ocvCapture.get(cv::CAP_PROP_FRAME_WIDTH);
        height = (int)ocvCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cout <<  "Input frame size: " << width << "x" << height << std::endl;

        // initiailze OpenCV pipeline
        ocvPipeline.Init(width,height);

        // init openvx context, graph and register callback to print errors
        ovxContext = vxCreateContext();
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxContext));
        CHECK_VX_STATUS(vxDirective((vx_reference)ovxContext, VX_DIRECTIVE_ENABLE_PERFORMANCE));

        vxRegisterLogCallback(ovxContext, IntelVXSample::errorReceiver, vx_true_e);

        ovxGraph = vxCreateGraph(ovxContext);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxGraph));

        // create and init perspective transform OpenVX matrix vxH
        cv::Mat ocvH;
        calcPerspectiveTransform(width,height).convertTo(ocvH,CV_32F);
        vx_matrix ovxH = vxCreateMatrix(ovxContext, VX_TYPE_FLOAT32, 3, 3);
        vx_float32  data[9] =
        {
            ocvH.at<float>(0,0),ocvH.at<float>(1,0),ocvH.at<float>(2,0),
            ocvH.at<float>(0,1),ocvH.at<float>(1,1),ocvH.at<float>(2,1),
            ocvH.at<float>(0,2),ocvH.at<float>(1,2),ocvH.at<float>(2,2)
        };
        printf("Warp Perspective Matrix = 9::%f,%f,%f,%f,%f,%f,%f,%f,%f\n",data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8]);
        CHECK_VX_STATUS( vxCopyMatrix(ovxH, &data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST) );

        // create filter to get high responce on lane mark pixels
        vx_uint32       filterScale = FILTER2D_SCALE;
        vx_convolution  ovxFilter = vxCreateConvolution(ovxContext, FILTER2D_W, FILTER2D_H);
        CHECK_VX_STATUS(vxCopyConvolutionCoefficients(ovxFilter, &gFilter2D_Data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        CHECK_VX_STATUS(vxSetConvolutionAttribute(ovxFilter, VX_CONVOLUTION_SCALE, &filterScale, sizeof(filterScale)));

        // create and init threshold for thershold node
        vx_threshold    ovxThreshold = vxCreateThreshold(ovxContext, VX_THRESHOLD_TYPE_BINARY, VX_TYPE_UINT8);
        vx_int32        threshValue = THRESHOLD_VALUE;
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxThreshold));
        CHECK_VX_STATUS(vxSetThresholdAttribute(ovxThreshold, VX_THRESHOLD_THRESHOLD_VALUE, &threshValue, sizeof(threshValue)));

        // create array for line segments returned by Hough transform node
        vx_array  ovxLineArray = vxCreateArray(ovxContext, VX_TYPE_RECTANGLE, HOUGH_MAX_LINES);
        vx_scalar ovxLineCount = vxCreateScalar(ovxContext, VX_TYPE_INT32, NULL);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxLineArray));
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxLineCount));

        // create other OpenVX images that are used in the graph
        const int   IW = INTERNAL_WIDTH;
        const int   IH = INTERNAL_HEIGHT;
        vx_image    ovxImgRGB      = vxCreateImage(ovxContext,width,height,VX_DF_IMAGE_RGB);  // 0 input RGB image (it is not virtual becasue we need access to it)
        vx_image    ovxImgYUV      = vxCreateVirtualImage(ovxGraph,0,  0,VX_DF_IMAGE_YUV4);   // 1 YUV image
        vx_image    ovxImgGray     = vxCreateVirtualImage(ovxGraph,0,  0,VX_DF_IMAGE_U8);     // 2 Y image
        vx_image    ovxImgMapped   = vxCreateVirtualImage(ovxGraph,IW,IH,VX_DF_IMAGE_U8);     // 3 remaped image
        vx_image    ovxImgEdges    = vxCreateImage(ovxContext,     IW,IH,VX_DF_IMAGE_U8);     // 4 filtered image
        vx_image    ovxImgBin      = vxCreateVirtualImage(ovxGraph,IW,IH,VX_DF_IMAGE_U8);     // 5 thresholded edge response

        // check status of created images
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxImgRGB));
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxImgYUV));
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxImgGray));
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxImgMapped));
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxImgEdges));
        CHECK_VX_STATUS(vxGetStatus((vx_reference)ovxImgBin));

        // Create nodes connected by images.
        // this is the place where graph is created.
        // array of names is needed to draw execution time line by IntelVXSample::drawNodesAtTimeline
        const char* node_names[]= {"cvtConvert","ChannelExtract","WarpPercpective","Convolve","Threshold","HoughLinesP"};
        vx_node nodes[] =
        {
            vxColorConvertNode(    ovxGraph, ovxImgRGB, ovxImgYUV ),
            vxChannelExtractNode(  ovxGraph, ovxImgYUV, VX_CHANNEL_Y, ovxImgGray),
            vxWarpPerspectiveNode( ovxGraph, ovxImgGray, ovxH, VX_INTERPOLATION_BILINEAR, ovxImgMapped),
            vxConvolveNode(        ovxGraph, ovxImgMapped, ovxFilter, ovxImgEdges),
            vxThresholdNode(       ovxGraph, ovxImgEdges, ovxThreshold, ovxImgBin),
            vxHoughLinesPNodeIntel(     ovxGraph, ovxImgBin, HOUGH_RHO_RESOLUTION, HOUGH_THETA_RESOLUTION, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP, HOUGH_MAX_LINES, ovxLineArray, ovxLineCount)
        };

        // check statuses of created nodes
        for(int i=0; i<sizeof(nodes)/sizeof(nodes[0]); ++i)
            CHECK_VX_STATUS(vxGetStatus((vx_reference)nodes[i]));

        // Verify created graph
        CHECK_VX_STATUS(vxVerifyGraph(ovxGraph));

        // Making iteration while ESC is not pressed
        int delay = 1;  // 1 means that there is no wait between frames to press anykey
                        // 0 means that sample will wait when anykey is pressed to process next frame
        unsigned int frame;
        for(frame=0; frame<(unsigned int)max_frames.getValue(); frame++)
        {
            {// this is read frame section that is measured separatly
                PERFPROF_REGION_AUTO(ReadFrame)

                if(!ocvCapture.read(ocvInpBGR))
                {
                    break; // break if there is no any image
                }

                // copy input OpenCV BGR inage into input OpenVX RGB image
                // it is necessary to make BGR->RGB conversion because OpenVX support only RGB image format
                // mapAsMat is used to map OpenVX image into HOST memory and wrap it by OpenCV Mat
                vx_map_id map_id;
                cv::Mat imgRGB = IntelVXSample::mapAsMat(ovxImgRGB, VX_READ_ONLY, &map_id);
                cv::cvtColor(ocvInpBGR, imgRGB, cv::COLOR_BGR2RGB);
                IntelVXSample::unmapAsMat(ovxImgRGB, imgRGB, map_id);
            }

            if(!disable_ref.getValue())
            {//run opencv pipeline

                PERFPROF_REGION_BEGIN(ProcessOpenCVReference)
                ocvPipeline.Process(ocvInpBGR);
                PERFPROF_REGION_END(ProcessOpenCVReference)

                //run final step
                finalOCV.Process(ocvPipeline.m_Edges8U, THRESHOLD_VALUE, ocvPipeline.m_Lines);
            }

            {//run OpenVX graph on input vxImgRGB frame
                PERFPROF_REGION_BEGIN(vxProcessGraph)
                CHECK_VX_STATUS(vxProcessGraph(ovxGraph));
                PERFPROF_REGION_END(vxProcessGraph)

                {//get line segments from OpenVX array and run final step
                    vx_int32    LineCountOVX = -1;
                    CHECK_VX_STATUS(vxCopyScalar(ovxLineCount, &LineCountOVX, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));					
                    // get acess to lines array
                    vx_size stride=0;
                    char*   ptr=NULL;
                    linesOVX.clear();
                    if(LineCountOVX)
                    {
                        linesOVX.resize(LineCountOVX);
                        vx_map_id map_id;
                        CHECK_VX_STATUS(vxMapArrayRange(ovxLineArray, 0, LineCountOVX, &map_id, &stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
                        for(int i=0; i<LineCountOVX; ++i)
                        {//iterate over all lines and copy data from OpenVX array into vector
                            vx_rectangle_t* pL = (vx_rectangle_t*)(ptr + stride*i);
                            linesOVX[i][0] = pL->start_x;
                            linesOVX[i][1] = pL->start_y;
                            linesOVX[i][2] = pL->end_x;
                            linesOVX[i][3] = pL->end_y;
                        }
                        CHECK_VX_STATUS(vxUnmapArrayRange(ovxLineArray, map_id));
                    }
                }
                // map filter responce and run final stage
                vx_map_id map_id;
                cv::Mat  imgEdges = IntelVXSample::mapAsMat(ovxImgEdges, VX_READ_ONLY, &map_id);
                finalOVX.Process(imgEdges , THRESHOLD_VALUE, linesOVX);
                IntelVXSample::unmapAsMat(ovxImgEdges, imgEdges, map_id);
            }

            // the rest of the code is for debug purposes only
            if(!no_show.isSet() && debug_output.isSet())
            {// draw OVX timeline
                IntelVXSample::drawNodesAtTimeline(nodes, sizeof(nodes)/sizeof(vx_node), node_names);
            }
            if(!disable_ref.getValue())
            {// draw OpenCV result
                ocvInpBGR.copyTo(outDebugOCV);
                finalOCV.DrawResult(outDebugOCV,ocvH,vis);
                if(!no_show.isSet())
                    cv::imshow("Lane Detection OpenCV Reference Result", outDebugOCV);
            }
            // draw OpenVX result
            ocvInpBGR.copyTo(outDebugOVX);
            finalOVX.DrawResult(outDebugOVX,ocvH,vis);
            if(!no_show.isSet())
                cv::imshow("Lane Detection OpenVX Result", outDebugOVX);
                if(output.isSet())
                {
                    static cv::VideoWriter ocvWriter;
                    if(!ocvWriter.isOpened())
                    {   // open output file for writing
                        std::cout << "open file "<< output.getValue() <<" for writing" << std::endl;
                        int fourcc =(int)ocvCapture.get(cv::CAP_PROP_FOURCC);
                        int fps =(int)ocvCapture.get(cv::CAP_PROP_FPS);
                        ocvWriter.open(output.getValue(), fourcc, fps, outDebugOVX.size());
                        if(!ocvWriter.isOpened())
                        {
                            std::cout << "Can not open output file: " << output.getValue() << std::endl;
                            break;
                        }
                    }

                    ocvWriter.write(outDebugOVX);
                }
            // read key value and process it
            int key = cv::waitKey(delay * frame_wait.getValue()) & 0xff;
            if(key == 32)
                delay = delay?0:1;
            if(key == 27) // ESC is pressed. exit
                break;
            if(key == 'v') // change visualization mode between 1 and 2
                vis = 3-vis;

            if((frame%10) == 0)
            {// simulate a progress bar by printing one dot for each 10 processed frames
                std::cout << "Frame: " << frame << "\r";
                std::cout.flush();

            }
        }//next frame
        std::cout << "Frame: " << frame << std::endl;
        std::cout << "Release data..." << std::endl;
        // release all stuff
        CHECK_VX_STATUS(vxReleaseThreshold(&ovxThreshold));
        CHECK_VX_STATUS(vxReleaseArray(&ovxLineArray));
        CHECK_VX_STATUS(vxReleaseScalar(&ovxLineCount));
        CHECK_VX_STATUS(vxReleaseConvolution(&ovxFilter));

        //release nodes
        for(int i=0; i<sizeof(nodes)/sizeof(vx_node); i++)
            CHECK_VX_STATUS(vxReleaseNode(nodes+i));

        //release images
        CHECK_VX_STATUS(vxReleaseImage(&ovxImgRGB));
        CHECK_VX_STATUS(vxReleaseImage(&ovxImgYUV));
        CHECK_VX_STATUS(vxReleaseImage(&ovxImgGray));
        CHECK_VX_STATUS(vxReleaseImage(&ovxImgMapped));
        CHECK_VX_STATUS(vxReleaseImage(&ovxImgEdges));
        CHECK_VX_STATUS(vxReleaseImage(&ovxImgBin));

        CHECK_VX_STATUS(vxReleaseGraph(&ovxGraph));
        CHECK_VX_STATUS(vxReleaseContext(&ovxContext));
        return EXIT_SUCCESS;
    }
    catch(const CmdParser::Error& error)
    {
        cerr << "[ ERROR ] In command line: " << error.what() << std::endl
             << "Run " << argv[0] << " -h for usage info." << std::endl;
        return EXIT_FAILURE;
    }
}
