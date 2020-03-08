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

#ifndef _PIPELINECONTROL_H_
#define _PIPELINECONTROL_H_

#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <VX/vx_intel_volatile.h>
#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/cmdparser.hpp>
#include <intel/vx_samples/perfprof.hpp>

class CmdParserPipeline: public CmdParserWithHelp
{
public:
    CmdParserPipeline(int argc, const char** argv) : CmdParser(argc, argv), CmdParserWithHelp(argc, argv),
        input(
            *this,
            'i',
            "input",
            "<file name>",
            "(5120x6592 RAW RGB (8bpp)) input raw file",
            exe_dir() + "low_contrast_5120x6592_I444.raw"
        ),
        width(*this,
            0,
            "width",
            "<integer>",
            "Specified width of raw RGB input image",
            5120
        ),
        height(*this,
            0,
            "height",
            "<integer>",
            "Specified height of raw RGB input image",
            6592
        ),
        staticmempoolsize(*this,
            0,
            "staticmempoolsize",
            "<integer>",
            "Size (in bytes) for static memory pool",
            1024*1024*450
        ),
        output(
            *this,
            'o',
            "output",
            "<file name>",
            "(5120x6592 RAW CMYK (1bpp)) output raw file",
            "output.raw"
        ),
        nthreads(
            *this,
            0,
            "nthreads",
            "<integer>",
            "How many CPU Worker threads OpenVX runtime is allowed to use. Default is the number of CPU cores detected on the system running.",
            0
        ),
        tileheight(
            *this,
            0,
            "tileheight",
            "<integer>",
            "Tile height in # of scanlines for tiling extension.",
            206
        ),
        lightness(
            *this,
            0,
            "lightness",
            "<integer>",
            "Lightness [ --lightness x ] (where x ranges -3 to 3).",
            0
        ),
        contrast(
            *this,
            0,
            "contrast",
            "<integer>",
            "Contrast [ --contrast x ] (where x ranges -2 to 2).",
            0
        ),
        gpuboxfilter(
            *this,
            0,
            "gpuboxfilter",
            "",
            "Offload Box Filters to GPU.",
            false
        ),
        gpusobelfilter(
            *this,
            0,
            "gpusobelfilter",
            "",
            "Offload Sobel Filters to GPU.",
            false
        ),
        ipurgb2lab(
            *this,
            0,
            "ipurgb2lab",
            "",
            "Offload RGB-to-CIELab to IPU.",
            false
        ),
        gpurgb2lab(
            *this,
            0,
            "gpurgb2lab",
            "",
            "Offload RGB-to-CIELab to GPU.",
            false
        ),
        gpulut(
            *this,
            0,
            "gpulut",
            "",
            "Offload Lightness / Darkness LUT to GPU.",
            false
        ),
        gpulab2cmyk(
            *this,
            0,
            "gpulab2cmyk",
            "",
            "Offload Lab-to-CMYK to GPU.",
            false
        ),
        ipusymm7x7(
            *this,
            0,
            "ipusymm7x7",
            "",
            "Offload Symmetrical 7x7 Filter to IPU.",
            false
        ),
        gpusymm7x7(
            *this,
            0,
            "gpusymm7x7",
            "",
            "Offload Symmetrical 7x7 Filter to GPU.",
            false
        ),
        gpusymm7x7_custom(
            *this,
            0,
            "gpusymm7x7_custom",
            "",
            "Offload Custom Symmetrical 7x7 Filter to GPU.",
            false
        ),
        gpuremovefringe(
            *this,
            0,
            "gpuremovefringe",
            "",
            "Offload RemoveFringe to GPU.",
            false
        ),
        frames(
            *this,
            'f',
            "max-frames",
            "<integer>",
            "Number of iterations input frame to be processed. Performance metrics are averaged over this number.",
            10
        ),
        clnontiled(
            *this,
            0,
            "clnontiled",
            "",
            "Disable GPU tiling.",
            false
        ),
        tetrainterp(
           *this,
           0,
           "tetrainterp",
           "",
           "Use VX_INTERPOLATION_TETRAHEDRAL_INTEL instead of default VX_INTERPOLATION_TRILINEAR_INTEL for LUT3D CIELab->CMYK conversion",
           false
       ),
       nlatticepoints33(
           *this,
           0,
           "nlatticepoints33",
           "",
           "Use 33 lattice points for LUT3D CIELab->CMYK conversion, instead of the default 17",
           false
       ),
       ipahalftone(
           *this,
           0,
           "ipahalftone",
           "",
           "Use IPAHalftone node(s) instead of Error Diffusion",
           false
        ),
       sppseparate(
           *this,
           0,
           "sppseparate",
           "",
           "Run SPP(Scan Pre-Process), but separate from main copy graph",
           false
        ),
        sppconnected(
           *this,
           0,
           "sppconnected",
           "",
           "Run SPP(Scan Pre-Process) as part of the main copy graph",
           false
        ),
        sppskew(
           *this,
           0,
           "sppskew",
           "",
           "Skew angle. The input image will be skewed (rotated) using this, SPP will correct it.",
           0.0f
        ),
        gpuskew(
           *this,
           0,
           "gpuskew",
           "",
           "Offload skew correction (vxWarpAffine) to GPU",
           false
        ),
        sppbits(*this,
            0,
            "sppbits",
            "<integer>",
            "Number of bits to create uncalibrated R, G. B with. 10 or 12 are supported.",
            0
        ),
        halftonepath(
           *this,
           0,
           "halftonepath",
           "",
           "Use the halftone path for the RGB-to-CMYK graph",
           false
        ),
        edpath(
           *this,
           0,
           "edpath",
           "",
           "Use the error diffusion path for the RGB-to-CMYK graph",
           false
        ),
        skewinterptype(
           *this,
           0,
           "skewinterptype",
           "",
           "Interpolation Type for Skew Correction. 0 - NN, 1 - Bilinear, 2 - BiCubic",
           1
        )
    {
    }
    //Sample command line options. Detailed description for each is above.
    CmdOption<std::string> input;
    CmdOption<std::string> output;
    CmdOption<unsigned int> nthreads;
    CmdOption<unsigned int> tileheight;
    CmdOption<unsigned int> staticmempoolsize;
    CmdOption<int> lightness;
    CmdOption<int> contrast;
    CmdOption<bool> gpuboxfilter;
    CmdOption<bool> gpusobelfilter;
    CmdOption<bool> gpurgb2lab;
    CmdOption<bool> gpulut;
    CmdOption<bool> gpulab2cmyk;
    CmdOption<bool> ipurgb2lab;
    CmdOption<bool> ipusymm7x7;
    CmdOption<bool> gpusymm7x7;
    CmdOption<bool> gpusymm7x7_custom;
    CmdOption<bool> gpuremovefringe;
    CmdOption<unsigned int> frames;
    CmdOption<bool> clnontiled;
    CmdOption<bool> tetrainterp;
    CmdOption<bool> nlatticepoints33;
    CmdOption<bool> ipahalftone;
    CmdOption<bool> sppseparate;
    CmdOption<bool> sppconnected;
    CmdOption<bool> halftonepath;
    CmdOption<bool> edpath;
    CmdOption<float> sppskew;
    CmdOption<bool> gpuskew;
    CmdOption<int> skewinterptype;
    CmdOption<unsigned int> sppbits;
    CmdOption<unsigned int> width;
    CmdOption<unsigned int> height;

    virtual void parse()
    {
        CmdParserWithHelp::parse();
        if (help.isSet())
        {
            // Immediatly exit if user wanted to see the usage information only.
            return;
        }
        if (input.getValue().empty())
        {
            throw CmdParser::Error("Input file name is required. Use --input FILE to provide input raw image file name.");
        }

        int flavor = (int)edpath.isSet() + (int)halftonepath.isSet();
        if ( flavor != 1)
        {
            throw CmdParser::Error("Must specify either --edpath or--halftonepath (but not both).");
        }
    }
};


class PipelineControl
{
public:

    PipelineControl(CmdParserPipeline *cmdparser);
    ~PipelineControl();

    //"warms" up the GPU (OpenCL) kernels by running small "dummy" executions
    void WarmGPUKernels();

    //Step 1
    int GetInputImage();

    //Step 2
    int ScanPreProcess();

    //Step 2.5
    void AttachCommonSubGraph();

    //Step 3
    void AssembleHalftoneGraph();
    //or
    void AssembleErrorDiffusionGraph();

    //Step 4
    void ExecuteGraph();

    //Step 5
    void SaveOutputImage();

    static void *Allocate(void* opaque, vx_size size);
    static void Free(void* opaque, void* data);

private:
    //OpenVX graph and context.
    vx_graph m_graph;
    vx_context m_context;

    //Lookup table for RGB2LAB conversion using vxLUT3DNode.
    vx_array m_rgb2lab_nodevals17x17x17;

    //Lookup table for LAB2CMYK conversion using vxLUT3DNode.
    vx_array m_lab2cmyk_nodevals;

    //Input images containing separate R, G, B channels.
    vx_image m_srcImageR;
    vx_image m_srcImageG;
    vx_image m_srcImageB;

    //The packed 10-bit or 12-bit "raw" images that
    // get generated if --sppbits is specified. These
    // are used as the input to vxGainOffset10 / vxGainOffset12
    vx_image m_rawGainOffsetInputR;
    vx_image m_rawGainOffsetInputG;
    vx_image m_rawGainOffsetInputB;

    vx_enum  m_warpInterpType;

    vx_image m_copyInputR;
    vx_image m_copyInputG;
    vx_image m_copyInputB;

    //Output neutral edge mask for high3 and high6 graphs.
    vx_image m_dstNeutralEdgeImage;

    //Output images containing separate C, M, Y, K channels.
    vx_image m_dstImageC;
    vx_image m_dstImageM;
    vx_image m_dstImageY;
    vx_image m_dstImageK;

    //Virtual images for the output of Background Suppression. Output of common subgraph.
    vx_image m_iBackgroundSuppress_0;
    vx_image m_iBackgroundSuppress_1;
    vx_image m_iBackgroundSuppress_2;

    //Virtual image for the output of Lab2CMYK. Output of common subgraph.
    vx_image m_iLab2CMYK;


    //Virtual image for the output of lightness / darkness / contrast (table lookup)
    vx_image m_iLightnessDarknessContrast;

    //Helper function for lightness and contarst lookup table initialization.
    void InitializeLightnessDarknessContrastLUT( int lightness, int contrast, vx_uint8 lut[256]);

    //RGB2LAB color conversion subgraph.
    void AttachRGB2LabSubGraph(vx_image r_in,
        vx_image g_in,
        vx_image b_in,
        vx_image l_out,
        vx_image a_out,
        vx_image b_out);

    //LAB2CMYK color conversion subgraph.
    void AttachLab2CMYKSubGraph(vx_image l_in,
        vx_image a_in,
        vx_image b_in,
        vx_image cmyk_out);

    //Attaches a neutral edge detection sub-graph to m_graph... pass in the input links (CIELab channels),
    // will pass back the neutral edge graph
    void AttachNeutralEdgeDetectSubGraph(vx_image l_in, vx_image a_in, vx_image b_in, vx_image neutral_edge_out);

    //Attaches the rendering portion (contone CMYK to bitone CMYK) to the graph
    void AttachErrorDiffusionSubGraph(vx_image cmyk_contone,
                                     vx_image c_bitone,
                                     vx_image m_bitone,
                                     vx_image y_bitone,
                                     vx_image k_bitone);


    void AttachHalftoneSubGraph(vx_image c_contone,
                                vx_image m_contone,
                                vx_image y_contone,
                                vx_image k_contone,
                                vx_image c_bitone,
                                vx_image m_bitone,
                                vx_image y_bitone,
                                vx_image k_bitone);

    //For memory allocation profiling.
    unsigned char *m_pMemoryPool;
    size_t m_currentOffset;
    vx_allocator_intel_t m_customAllocator;

    //Command line parser
    CmdParserPipeline *m_cmdparser;

    vx_enum m_3dlut_interp_type_lab2cmyk;
    vx_int32 m_3dlut_nlatticepoints_lab2cmyk;

    int m_tileWidth = 0;

    //scan-pre-process
    vx_matrix m_correctionMatrix;

    unsigned int m_inputImageWidth;
    unsigned int m_inputImageHeight;
};


#endif
