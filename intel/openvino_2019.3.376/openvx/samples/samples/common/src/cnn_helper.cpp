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
#include <intel/vx_samples/cnn_helper.hpp>

namespace IntelVXSample
{

#if INTEL_SAMPLE_USE_OPENCV
// This function takes OpenCV image, pre-processes and stores in F32 format
// The following steps are done sequentially:
// 1. Convert input image into F32 format.
// 2. Resize result image to CNN input size.
// 3. subtract mean image from resized image, if it's specified. The CNN models are often supposed
//    that mean image is subtracted from image before training. So the same
//    subtract operation has to be done before classification step.
// 4. Scale result image's values (to fit Q78 data range by all activations)
cv::Mat preProcessImage(cv::Mat image, CNNInputDimensions dims, cv::Mat mean, float scale)
{
    cv::Mat inp32F;
    // 1. Convert input image into F32 format
    image.convertTo(inp32F, CV_32FC3);

    // 2. Resize image into CNN input size
    cv::resize(inp32F, inp32F, cv::Size(dims.width, dims.height));

    // 3. subtract mean image if it's specified
    if (!mean.empty())
    {
        cv::Mat meanResized;
        // resize mean image to CNN input size
        cv::resize(mean, meanResized, inp32F.size());
        // subtract mean image from image
        inp32F = (inp32F - meanResized);
    }

    // 4. Scale result image's values.
    inp32F = inp32F * scale;

    return inp32F;
}
#endif // INTEL_SAMPLE_USE_OPENCV

#if INTEL_SAMPLE_USE_OVX_1_0_1
vx_df_intel_mddata_e getMDDataType(vx_intel_md_data mddata)
{
    vx_df_intel_mddata_e dt;
    CHECK_VX_STATUS( vxQueryIntelMDData(mddata, VX_INTEL_MDDATA_DATA_FORMAT, &dt, sizeof(dt)) );
    return dt;
}
#else
// retrieve data type for vx_tensor
vx_type_e getMDDataType(vx_tensor mddata)
{
    vx_type_e dt;
    CHECK_VX_STATUS( vxQueryTensor(mddata, VX_TENSOR_DATA_TYPE, &dt, sizeof(dt)) );
    return dt;
}
#endif

#ifndef INTEL_SAMPLE_USE_OVX_1_0_1
int getMDDataElemSize(vx_type_e dt)
{
    int elemSize = 0;
    switch(dt)
    {
    case VX_TYPE_INT16:     //Q78
    case VX_TYPE_FLOAT16_INTEL:   //FP16
        elemSize = sizeof(vx_int16);
    break;
    case VX_TYPE_FLOAT32:
        elemSize = sizeof(vx_float32);
    break;
    }
    return elemSize;
}
#endif

#if INTEL_SAMPLE_USE_OVX_1_0_1
std::vector<CNNResponse> mdData2Responses(vx_intel_md_data output)
{
    // Get reference to the graph output
    std::vector<CNNResponse> response;
    {// get responses from OpenVX response

        // Process graph output

        // retrieve data type for vx_tensor
        vx_df_intel_mddata_e            dt;
        CHECK_VX_STATUS( vxQueryIntelMDData(output, VX_INTEL_MDDATA_DATA_FORMAT, &dt, sizeof(dt)) );

        // get access to data
        void*                           ptr = NULL;
        vx_intel_mdview_addressing_t    addr;
        vx_intel_mdview_t               view = getMDDataView(output);
        CHECK_VX_STATUS( vxAccessIntelMDDataPatch(output, &view, &addr, &ptr, VX_READ_ONLY) );

        response.resize( addr.dim[0] );

        // copy data from mddataOutput into response array
        for(int i=0; i < addr.dim[0]; ++i)
        {
            float val = mddata2float((char*)ptr + i*addr.stride[0], dt);
            response[i].m_ID = i;
            response[i].m_Prob = val;
        }

        CHECK_VX_STATUS( vxCommitIntelMDDataPatch(output, &view, &addr, ptr) );
    }
    // sort responses in descending order
    std::sort(response.begin(), response.end(), compCNNResponse );
    return response;
}
#else
// This function takes vx_tensor and extracts vector of responses for classification CNN model
ResponseVector tensor2Responses(vx_tensor tensorOut, unsigned int classDimId)
{
    vx_type_e dt = getMDDataType(tensorOut);

    vx_size numDims = 0;
    CHECK_VX_STATUS(vxQueryTensor(tensorOut, VX_TENSOR_NUMBER_OF_DIMS, &numDims, sizeof(numDims)));

    std::vector<vx_size> dims(numDims);
    CHECK_VX_STATUS(vxQueryTensor(tensorOut, VX_TENSOR_DIMS, dims.data(), dims.size()*sizeof(vx_size)));
    
    if(classDimId >= numDims - 1)
    {
        std::cout << "Output tensor dimensions:\n";
        for (std::vector<vx_size>::reverse_iterator rit = dims.rbegin(); rit!= dims.rend(); ++rit)
            std::cout << *rit << " ";
        std::cout << "]";
        std::cout << classDimId << " dim is set by user\n";
        throw SampleError("Invalid class dimension index set by user!\n");
    }

    int elemSize = getMDDataElemSize(dt);

    std::vector<vx_size> szStart(dims.size(), 0);
    std::vector<vx_size> szStride(dims.size(), 0);

    vx_size total_size = 1;
    for(vx_size k=0; k<dims.size(); ++k)
    {
        if( dims[k] )
            total_size *= dims[k];
    }

    szStride[0] = elemSize;
    int i=1;
    for (; i < dims.size(); i++)
        szStride[i] = szStride[i - 1] * dims[i - 1];

    std::vector<char> tensorBuf(elemSize * total_size);
    char* bufPtr = tensorBuf.data();

    CHECK_VX_STATUS(
        vxCopyTensorPatch(
            tensorOut,
            dims.size(),
            szStart.data(),
            dims.data(),
            szStride.data(),
            (void*)bufPtr,
            VX_READ_ONLY,
            VX_MEMORY_TYPE_HOST
        )
    );
    
    ResponseVector results;
    
    // copy data from mddataOutput into response array
    unsigned int batch_size = dims[classDimId + 1];

    for(unsigned int b=0; b < batch_size; ++b)
    {
        std::vector<CNNResponse> response( dims[classDimId] );
        
        for(int i=0; i < dims[classDimId]; ++i)
        {
            float val = IntelVXSample::mddata2float(
                    (char*)bufPtr +                  //tensor base
                    b * szStride[classDimId+1] +     //batch dimension stride
                    i * szStride[classDimId],        //class dimension stride
                    dt);
            response[i].m_ID = i;
            response[i].m_Prob = val;
        }
        // sort responses in descending order
        std::sort(response.begin(), response.end(), compCNNResponse );
        
        results.push_back(response);
    }
    return results;
}
#endif

#if INTEL_SAMPLE_USE_OVX_1_0_1
CNNInputDimensions getInputDimensions(vx_intel_md_data mddataInp)
{
    void* ptr = NULL;
    vx_intel_mdview_addressing_t addr;
    vx_intel_mdview_t view = getMDDataView(mddataInp);
    CHECK_VX_STATUS(vxAccessIntelMDDataPatch(mddataInp, &view, &addr, &ptr, VX_READ_ONLY));
    CNNInputDimensions result(addr.dim[0], addr.dim[1]);
    CHECK_VX_STATUS(vxCommitIntelMDDataPatch(mddataInp, &view, &addr, ptr));
    return result;
}
#else
// This function checks and returns dimensions of vx_tensor
CNNInputDimensions getInputDimensions(vx_tensor tensorInp)
{
    vx_mdview_addressing_intel_t addr;
    vx_size dims[VX_MAX_TENSOR_DIMS_INTEL] = { 0 };
    CHECK_VX_STATUS(vxQueryTensor(tensorInp, VX_TENSOR_DIMS, dims, sizeof(dims)));
    CNNInputDimensions result(dims[0], dims[1]);
    return result;
}
#endif

#if INTEL_SAMPLE_USE_OPENCV
#if INTEL_SAMPLE_USE_OVX_1_0_1
// This function takes an array of OpenCV images in F32 format as input and initializes vx_tensor from them.
void image2IntelMDData(vx_intel_md_data mddataInp, cv::Mat inp32F)
{
    // retrieve data type for vx_tensor
    vx_df_intel_mddata_e dt;
    CHECK_VX_STATUS( vxQueryIntelMDData(mddataInp, VX_INTEL_MDDATA_DATA_FORMAT, &dt, sizeof(dt)) );
    std::cout << "DataType: " << vxMDDataTypeToStr(dt) << std::endl;

    vx_intel_mdview_addressing_t    addr; 
    vx_intel_mdview_t               view = getMDDataView( mddataInp );
    void*                           ptr = NULL;
    CHECK_VX_STATUS( vxAccessIntelMDDataPatch(mddataInp, &view, &addr, &ptr, VX_WRITE_ONLY) );

    for(int y=0;y<inp32F.rows;++y)for(int x=0;x<inp32F.cols;++x)
    {
        float* pInp = (float*)inp32F.ptr(y,x);
        for(int c=0;c<inp32F.channels();++c)
        {
            //    Convert and store result F32 image values into vx_intel_md_data
            //    To calculate the position of current value in mddata the
            //    addr.stride[0,1,2] are used. The stride values were got by
            //    vxAccessIntelMDDataPatch called earlier. Each stride value is
            //    distance in bytes between values along correspondence dimension
            //    addr.stride[0] is distance between mddata(c,y,x) and mddata(c,y,x+1)
            //    addr.stride[1] is distance between mddata(c,y,x) and mddata(c,y+1,x)
            //    addr.stride[2] is distance between mddata(c,y,x) and mddata(c+1,y,x)
            float2mddata(
                // Prepared float data value
                pInp[c],
                // Destination address
                (char*)ptr + c*addr.stride[2] + y*addr.stride[1] + x*addr.stride[0],
                // Destination type
                dt);
        }
    }
    CHECK_VX_STATUS( vxCommitIntelMDDataPatch(mddataInp, &view, &addr, ptr) );
#ifdef DUMP_DATA
    dumpMDData(mddataInp,"input_data.txt");
#endif
}
#else
// This function takes OpenCV image in F32 format as input and initializes vx_tensor from it.
void image2Tensor(vx_tensor tensorInp, std::vector<cv::Mat> inp32F)
{
    // retrieve data type for vx_tensor
    vx_type_e dt = getMDDataType(tensorInp);
    std::cout << "DataType: " << vxMDDataTypeToStr(dt) << std::endl;

    vx_mdview_addressing_intel_t    addr;
    void*                           ptr = NULL;

    vx_size numDims = 0;
    CHECK_VX_STATUS(vxQueryTensor(tensorInp, VX_TENSOR_NUMBER_OF_DIMS, &numDims, sizeof(numDims)));

    std::vector<vx_size> viewEnd(numDims, 0);
    std::vector<vx_size> viewStart(numDims, 0);
    CHECK_VX_STATUS(vxQueryTensor(tensorInp, VX_TENSOR_DIMS, viewEnd.data(), viewEnd.size()*sizeof(vx_size)));

    vx_map_id map_id;
    CHECK_VX_STATUS( vxMapTensorPatchIntel(tensorInp,
                                    numDims,
                                    viewStart.data(), viewEnd.data(),
                                    &map_id,
                                    &addr, &ptr, VX_WRITE_ONLY,
                                    VX_MEMORY_TYPE_HOST, VX_NOGAP_X) );

    if(inp32F.size() != viewEnd.back())
        throw SampleError("[ ERROR ] Input tensor size doesn't match number of images!\n");

    for(unsigned int b=0; b < inp32F.size(); ++b)
    {
        for(int y=0;y<inp32F[b].rows;++y)
            for(int x=0;x<inp32F[b].cols;++x)
        {
            float* pInp = (float*)inp32F[b].ptr(y,x);
            for(int c=0;c<inp32F[b].channels();++c)
            {
                //    Convert and store result F32 image values into vx_tensor
                //    To calculate the position of current value in mddata the
                //    addr.stride[0,1,2] are used. The stride values were got by
                //    vxAccessIntelMDDataPatch called earlier. Each stride value is
                //    distance in bytes between values along correspondence dimension
                //    addr.stride[0] is distance between mddata(c,y,x) and mddata(c,y,x+1)
                //    addr.stride[1] is distance between mddata(c,y,x) and mddata(c,y+1,x)
                //    addr.stride[2] is distance between mddata(c,y,x) and mddata(c+1,y,x)
                //    addr.stride[3] is distance between images in a batch
                float2mddata(
                    // Prepared float data value
                    pInp[c],
                    // Destination address
                    (char*)ptr +            //tensor address base
                    b*addr.stride[3] +      //image number in a batch
                    c*addr.stride[2] +      //channel
                    y*addr.stride[1] +      //y
                    x*addr.stride[0],       //x
                    // Destination type
                    dt);
            }
        }
    }
    CHECK_VX_STATUS( vxUnmapTensorPatchIntel(tensorInp, map_id) );
#ifdef DUMP_DATA
    dumpTensor(tensorInp,"input_data.txt");
#endif
}
#endif
#endif // INTEL_SAMPLE_USE_OPENCV

#if INTEL_SAMPLE_USE_OVX_1_0_1
const char* vxMDDataTypeToStr (vx_df_intel_mddata_e dt)
{
#define VX_DT_TO_STR_ENTRY(DT) case DT: return #DT;
    switch(dt)
    {
        VX_DT_TO_STR_ENTRY(VX_DF_INTEL_MDDATA_S16)
        VX_DT_TO_STR_ENTRY(VX_DF_INTEL_MDDATA_Q78)
        VX_DT_TO_STR_ENTRY(VX_DF_INTEL_MDDATA_FLOAT16)
        VX_DT_TO_STR_ENTRY(VX_DF_INTEL_MDDATA_FLOAT32)
        default: return "UNKNOWN vx_df_intel_mddata_e";
    }
#undef VX_DT_TO_STR_ENTRY
}
#else
const char* vxMDDataTypeToStr (vx_type_e dt)
{
#define VX_DT_TO_STR_ENTRY(DT) case DT: return #DT;
    switch(dt)
    {
        VX_DT_TO_STR_ENTRY(VX_TYPE_INT16)
        VX_DT_TO_STR_ENTRY(VX_TYPE_FLOAT16_INTEL)
        VX_DT_TO_STR_ENTRY(VX_TYPE_FLOAT32)
        default: return "UNSUPPORTED vx_type_e";
    }
#undef VX_DT_TO_STR_ENTRY
}
#endif



// Function to convert F32 into Q78
int16_t float2Q78(float x)
{
    float x_scaled = x * 256;
    if (x_scaled > SHRT_MAX)
        return SHRT_MAX;
    if (x_scaled < SHRT_MIN)
        return SHRT_MIN;
    if(x_scaled<0)
        return (int16_t)(x_scaled - 0.5);
    else
        return (int16_t)(x_scaled + 0.5);
}
// Function to convert value from Q78 into F32
float Q782float(int16_t x)
{
    return (float)x*(1.0f/256.0f);
}

#ifndef USE_F16_INRINSICS
// Function to convert F32 into F16
// F32: exp_bias:127 SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM.
// F16: exp_bias:15  SEEEEEMM MMMMMMMM
#define EXP_MASK_F32 0x7F800000U
#define EXP_MASK_F16     0x7C00U

//small helper function to represent uint32_t value as float32
inline float asfloat(uint32_t v)
{
    return *(float*)&v;
}

// This function convert f32 to f16 with rounding to nearest value to minimize error
// the denormal values are converted to 0.
uint16_t f32tof16(float x)
{
    //create minimal positive normal f16 value in f32 format
    //exp:-14,mantissa:0 -> 2^-14 * 1.0
    static float min16 = asfloat((127 - 14) << 23);

    //create maximal positive normal f16 value in f32 and f16 formats
    //exp:15,mantissa:11111 -> 2^15 * 1.(11111)
    static float    max16 = asfloat(((127 + 15) << 23) | 0x007FE000 );
    static uint32_t max16f16 =      (( 15 + 15) << 10) | 0x3FF;

    // define and declare variable for intermidiate and output result
    // the union is used to simplify representation changing
    union
    {
        float f;
        uint32_t u;
    } v;
    v.f = x;

    // get sign in 16bit format
    uint32_t    s = (v.u >> 16) & 0x8000; // sign 16:  00000000 00000000 10000000 00000000

    // make it abs
    v.u &= 0x7FFFFFFF;                    // abs mask: 01111111 11111111 11111111 11111111

    // check NAN and INF
    if( (v.u & EXP_MASK_F32) == EXP_MASK_F32)
    {
        if(v.u & 0x007FFFFF)
            return s | (v.u >> (23 - 10)) | 0x0200; // return NAN f16
        else
            return s | (v.u >> (23 - 10)); // return INF f16
    }

    // to make f32 round to nearest f16
    // create halfULP for f16 and add it to origin value
    float halfULP = asfloat( v.u & EXP_MASK_F32 ) * asfloat((127 - 11) << 23);
    v.f += halfULP;

    // if input value is not fit normalized f16 then return 0
    // denormals are not covered by this code and just converted to 0
    if(v.f < min16*0.5F)
        return s;

    // if input value between min16/2 and min16 then return min16
    if(v.f < min16)
        return s | (1<<10);

    // if input value more than maximal allowed value for f16
    // then return this maximal value
    if(v.f >= max16 )
        return max16f16 | s;

    // change exp bias from 127 to 15
    v.u -= ((127 - 15) << 23);

    // round to f16
    v.u >>= (23-10);

    return v.u | s;
}

// Function to convert F32 into F16
float f16tof32(uint16_t x)
{
    // this is storage for output result
    uint32_t u = x;

    // get sign in 32bit format
    uint32_t s = ((u & 0x8000) << 16);

    // check for NAN and INF
    if( (u & EXP_MASK_F16) == EXP_MASK_F16 )
    {
        //keep mantissa only
        u &= 0x03FF;

        // check if it is NAN and raise 10 bit to be align with intrin
        if(u)
            u |= 0x0200;

        u <<= (23 - 10);
        u |= EXP_MASK_F32;
        u |= s;
    }
    // check for zero and denormals. both are converted to zero
    else if( (x & EXP_MASK_F16) == 0 )
    {
        u = s;
    }
    else
    {
        //abs
        u = (u & 0x7FFF);

        // shift mantissa and exp from f16 to f32 position
        u <<= (23-10);

        //new bias for exp (f16 bias is 15 and f32 bias is 127)
        u += ((127-15) << 23);

        //add sign
        u |= s;
    }

    //finaly represent result as float and return
    return asfloat( u );
}

#endif // USE_F16_INRINSICS

#if INTEL_SAMPLE_USE_OVX_1_0_1
float mddata2float(char* ptr, vx_df_intel_mddata_e dt)
#else
float mddata2float(char* ptr, vx_type_e dt)
#endif
{
    float v;
#if INTEL_SAMPLE_USE_OVX_1_0_1
    if(dt == VX_DF_INTEL_MDDATA_Q78)
#else
    if(dt == VX_TYPE_INT16)
#endif
    {
        v = Q782float(*(int16_t*)ptr);
    }
    else
    if( dt == VX_TYPE_FLOAT32 )
    {
        v = *(float*)ptr;
    }
    else
// TODO: restriction from vx_types.h
#if INTEL_SAMPLE_USE_OVX_1_0_1
    if( dt == VX_DF_INTEL_MDDATA_FLOAT16 )
#else
    if( (unsigned)dt == (unsigned)VX_TYPE_FLOAT16_INTEL )
#endif
    {
#ifdef USE_F16_INRINSICS
        v = _cvtsh_ss(*(short*)ptr);
#else
        v = f16tof32( *(uint32_t*)ptr ) ;
#endif
    }
    else
    {
        std::cerr
            << "[ ERROR ] Sample does not support vx_df_intel_mddata_e = " << dt
            << std::endl
            << "[ ERROR ] only "
            << "VX_TYPE_FLOAT32, "
#if defined(EXPERIMENTAL_PLATFORM_SUPPORTS_16_FLOAT)
            << "VX_TYPE_FLOAT16, "
#endif
            << "VX_TYPE_INT16 are supported\n";
        std::exit( EXIT_FAILURE );                                       \
    }
    return v;
}

// Function to convert value from float to vx_tensor data types
#if INTEL_SAMPLE_USE_OVX_1_0_1
void float2mddata(float v, char* ptr, vx_df_intel_mddata_e dt)
#else
void float2mddata(float v, char* ptr, vx_type_e dt)
#endif
{
#if INTEL_SAMPLE_USE_OVX_1_0_1
    if( dt == VX_DF_INTEL_MDDATA_Q78 )
#else
    if( dt == VX_TYPE_INT16 )
#endif
    {
        *(int16_t*)ptr = float2Q78(v);
    }
    else
    if( dt == VX_TYPE_FLOAT32 )
    {
        *(float*)ptr = v;
    }
    else
// TODO: restriction from vx_types.h
#if INTEL_SAMPLE_USE_OVX_1_0_1
    if( dt == VX_DF_INTEL_MDDATA_FLOAT16 )
#else
    if( (unsigned)dt == (unsigned)VX_TYPE_FLOAT16_INTEL )
#endif
    {
#ifdef USE_F16_INRINSICS
        *(uint16_t*)ptr = _cvtss_sh( v, 0 );
#else
        *(uint16_t*)ptr = f32tof16( v );
#endif
    }
    else
    {
        std::cerr
            << "[ ERROR ] Sample does not support vx_df_intel_mddata_e = " << dt
            << std::endl
            << "[ ERROR ] only "
            << "VX_TYPE_FLOAT32"
#if defined(EXPERIMENTAL_PLATFORM_SUPPORTS_16_FLOAT)
            << "VX_TYPE_FLOAT16, "
#endif
            << "VX_TYPE_INT16 are supported\n";
        std::exit( EXIT_FAILURE );                                       \
    }
}

#if INTEL_SAMPLE_USE_OVX_1_0_1
// this function calculate view for whole mddata
vx_intel_mdview_t getMDDataView(vx_intel_md_data mddata)
{
    vx_intel_mdview_t               view;
    vx_uint32                       dims[VX_INTEL_MD_DIM_MAX];
    CHECK_VX_STATUS( vxQueryIntelMDData(mddata, VX_INTEL_MDDATA_DIMS, dims, VX_INTEL_MD_DIM_MAX*sizeof(dims[0])) );
    for(int k=0;k<VX_INTEL_MD_DIM_MAX ;++k)
    {// Iterate over all dimensions to calculate total data size
     //  and set dimension to 1 for absent dimensions
        view.start[k] = view.end[k] = 0;
        if( dims[k] )
            view.end[k] = dims[k];
    }
    return view;
}
#endif

#ifdef DUMP_DATA
#if INTEL_SAMPLE_USE_OVX_1_0_1
// This function damp data into text file.
// It can be used to control internals of vx_intel_md_data
void dumpMDData(vx_intel_md_data mddata, const char* pName)
{
    FILE* out = fopen(pName,"wt");
    if(!out)
    {
        printf("Can not open %s for writing\n", pName);
        exit(EXIT_FAILURE);
    }

    // retrive data type for vx_intel_md_data
    vx_df_intel_mddata_e dt;
    CHECK_VX_STATUS( vxQueryIntelMDData(mddata, VX_INTEL_MDDATA_DATA_FORMAT, &dt, sizeof(dt)) );

    vx_intel_mdview_addressing_t    addr;
    vx_intel_mdview_t               view = getMDDataView( mddata );
    void*                           ptr = NULL;
    size_t                          totalSize = 1;
    vx_uint32                       dims[VX_INTEL_MD_DIM_MAX];

    CHECK_VX_STATUS( vxAccessIntelMDDataPatch(mddata, &view, &addr, &ptr, VX_READ_ONLY) );
    for(int k=0;k<VX_INTEL_MD_DIM_MAX ;++k)
    {// Iterate over all dimensions to calculate total data size
     //  and set dimension to 1 for absent dimensions
        dims[k] = addr.dim[k]>0 ? addr.dim[k] : 1;
        totalSize *= dims[k];
    }

    // Iterate over all values of the mddata
    for(int i5=0; i5<dims[5]; ++i5)
    for(int i4=0; i4<dims[4]; ++i4)
    for(int i3=0; i3<dims[3]; ++i3)
    for(int i2=0; i2<dims[2]; ++i2)
    for(int i1=0; i1<dims[1]; ++i1)
    for(int i0=0; i0<dims[0]; ++i0)
    {
        // Calculate address of current mddata value and load it
        size_t index = 0;
        index += i5 * addr.stride[5];
        index += i4 * addr.stride[4];
        index += i3 * addr.stride[3];
        index += i2 * addr.stride[2];
        index += i1 * addr.stride[1];
        index += i0 * addr.stride[0];
        if(dt == VX_DF_INTEL_MDDATA_Q78 || dt == VX_DF_INTEL_MDDATA_S16 )
        {
            int16_t v = *(int16_t*)((char*)ptr + index);
            fprintf(out, "%d,%d,%d,%d,%d,%d: %d\n",i5,i4,i3,i2,i1,i0,v);
        }
        else
        {
            float v = mddata2float((char*)ptr + index, dt);
            uint32_t u = *(uint32_t*)&v;
            fprintf(out, "%d,%d,%d,%d,%d,%d: (%08x:%1x-%02X-%06X): %f\n",i5,i4,i3,i2,i1,i0, u, (u>>31)&1, (u>>23)&0xFF, u&((1<<23)-1), v );
        }
    }// Next value from the mddata

    CHECK_VX_STATUS( vxCommitIntelMDDataPatch(mddata, &view, &addr, ptr) );
    fclose(out);
}
#else
void dumpMDData(vx_tensor mddata, const char* pName)
{
    FILE* out = fopen(pName,"wt");
    if(!out)
    {
        printf("Can not open %s for writing\n", pName);
        exit(EXIT_FAILURE);
    }

    // retrive data type for vx_intel_md_data
    vx_type_e dt;
    vxQueryTensor(mddata, VX_TENSOR_DATA_TYPE, &dt, sizeof(dt));


    size_t                          totalSize = 1;
    vx_uint32                       dims[VX_MAX_TENSOR_DIMS_INTEL];

    vx_mdview_addressing_intel_t    addr;
    void*                           ptr = NULL;
    const vx_size viewStart[VX_MAX_TENSOR_DIMS_INTEL] = { 0 };
    vx_size viewEnd[VX_MAX_TENSOR_DIMS_INTEL] = { 0 };
    vx_size numDims = 0;
    CHECK_VX_STATUS(vxQueryTensor(mddata, VX_TENSOR_NUMBER_OF_DIMS, &numDims, sizeof(numDims)));

    CHECK_VX_STATUS(vxQueryTensor(mddata, VX_TENSOR_DIMS, viewEnd, sizeof(viewEnd)));

    vx_map_id map_id;
    CHECK_VX_STATUS( vxMapTensorPatchIntel(mddata,
                                    numDims,
                                    viewStart, viewEnd,
                                    &map_id,
                                    &addr, &ptr, VX_READ_ONLY,
                                    VX_MEMORY_TYPE_HOST, VX_NOGAP_X) );

    for(int k=0;k<VX_MAX_TENSOR_DIMS_INTEL ;++k)
    {// Iterate over all dimensions to calculate total data size
     //  and set dimension to 1 for absent dimensions
        dims[k] = addr.dim[k]>0 ? addr.dim[k] : 1;
        totalSize *= dims[k];
    }

    // Iterate over all values of the mddata
    for(int i5=0; i5<dims[5]; ++i5)
    for(int i4=0; i4<dims[4]; ++i4)
    for(int i3=0; i3<dims[3]; ++i3)
    for(int i2=0; i2<dims[2]; ++i2)
    for(int i1=0; i1<dims[1]; ++i1)
    for(int i0=0; i0<dims[0]; ++i0)
    {
        // Calculate address of current mddata value and load it
        size_t index = 0;
        index += i5 * addr.stride[5];
        index += i4 * addr.stride[4];
        index += i3 * addr.stride[3];
        index += i2 * addr.stride[2];
        index += i1 * addr.stride[1];
        index += i0 * addr.stride[0];
        {
            float v = mddata2float((char*)ptr + index, dt);
            uint32_t u = *(uint32_t*)&v;
            fprintf(out, "%d,%d,%d,%d,%d,%d: (%08x:%1x-%02X-%06X): %f\n",i5,i4,i3,i2,i1,i0, u, (u>>31)&1, (u>>23)&0xFF, u&((1<<23)-1), v );
        }
    }// Next value from the mddata

    CHECK_VX_STATUS( vxUnmapTensorPatchIntel(mddata, map_id) );
    fclose(out);
}
#if INTEL_SAMPLE_USE_OPENCV
void dumpTensor(vx_tensor tensor, std::string fname)
{
    std::cout << std::endl << "Dumping " << fname;
    // retrieve data type for vx_tensor
    vx_type_e dt = getMDDataType(tensor);

    vx_size numDims = 0;
    CHECK_VX_STATUS(vxQueryTensor(tensor, VX_TENSOR_NUMBER_OF_DIMS, &numDims, sizeof(numDims)));
    std::vector<vx_size> dims(numDims);
    CHECK_VX_STATUS(vxQueryTensor(tensor, VX_TENSOR_DIMS, dims.data(), dims.size()*sizeof(vx_size)));

    if(numDims < 2)
    {
        std::cout << ". Only " << numDims << " dimension(s)" << std::endl;
        return;
    }

    //calculate total tensor size (in elements)
    size_t total_size = 1;
    size_t total_images = 1;

    std::cout << ", " << vxMDDataTypeToStr(dt) << ", [";
    for (std::vector<vx_size>::reverse_iterator rit = dims.rbegin(); rit!= dims.rend(); ++rit)
        std::cout << *rit << " ";
    std::cout << "]";

    for(vx_size k=0; k<dims.size(); ++k)
    {
        if( dims[k] )
        {
            total_size *= dims[k];

            if(k > 1)
                total_images *= dims[k];
        }
    }

    if(total_size < 100)
    {
        std::cout << ". Image size is too small" << std::endl;
        return;
    }

    if(total_images < 1)
    {
        std::cout << ". Has zero 2D images" << std::endl;
        return;
    }

    if(dims[0] < 10 || dims[1] < 10)
    {
        std::cout << ". Dimensions are too small to dump - " << dims[1] << "x" << dims[0] << std::endl;
        return;
    }
    if(dims[0] > 2048 || dims[1] > 2048)
    {
        std::cout << ". Too large to dump - " << dims[1] << "x" << dims[0] << std::endl;
        return;
    }
    else
    {
        float ratio = (float)dims[0] / (float)dims[1];

        if(ratio < 1.0) ratio = 1.0/ratio;

        if(ratio > 8)
        {
            std::cout << ". Suspiciously high aspect ratio - " << ratio << std::endl;
            return;
        }
    }

    std::cout << ", " << total_images << " image files to write ..." << std::endl;

    int elemSize = getMDDataElemSize(dt);

    std::vector<vx_size> szStart(dims.size(), 0);
    std::vector<vx_size> szStride(dims.size(), 0);

    szStride[0] = elemSize;
    int i=1;
    for (; i < dims.size(); i++)
        szStride[i] = szStride[i - 1] * dims[i - 1];

    std::vector<char> tensorBuf(elemSize*total_size);
    char* bufPtr = tensorBuf.data();

    CHECK_VX_STATUS(
        vxCopyTensorPatch(
            tensor,
            dims.size(),
            szStart.data(),
            dims.data(),
            szStride.data(),
            (void*)bufPtr,
            VX_READ_ONLY,
            VX_MEMORY_TYPE_HOST
        )
    );

    //quickly test that all elements are zero
    bool zeros = std::all_of(tensorBuf.begin(), tensorBuf.end(), [](unsigned char i) { return i==0; });

    if(zeros)
    {
        std::cout << "Tensor " << fname << " is all zero" << std::endl;
        return;
    }

    //output all data treating tensor as a set of many 2D images
    char* imgPtr = bufPtr;
    if(3 == dims[2])
    {
        total_images /= 3;

        for(vx_size m=0; m<total_images; m++)
        {
            cv::Mat outmat(dims[0], dims[1], CV_32FC3);

            for(int y=0;y<outmat.rows;++y)
            for(int x=0;x<outmat.cols;++x)
            {
                float* pInp = (float*)outmat.ptr(y,x);

                for(int c=0;c<3;++c)
                    pInp[c] = mddata2float(imgPtr + c * szStride[2] + y * szStride[1] + x * szStride[0], dt);
            }

            //normalize and write image to disk
            cv::Mat norm;
            cv::normalize(outmat, norm, 255.0, 0.0, cv::NORM_MINMAX);

            cv::Mat cnv;
            norm.convertTo(cnv, CV_8U);

            for(int y=0;y<outmat.rows;++y)
            for(int x=0;x<outmat.cols;++x)
            {
                float* pInp = (float*)outmat.ptr(y,x);
                unsigned char* pConv = (unsigned char*)cnv.ptr(y,x);

                for(int c=0;c<3;++c)
                {
                    unsigned u0 = *(unsigned int*)(pInp+c);

                    if( 0 == u0 )
                        pConv[2] = 255;
                }
            }

            cv::imwrite(std::string(fname) + "_" + std::to_string(m) + ".png", cnv);

            imgPtr += szStride[3];
        }
    }
    else
    {
        for(vx_size m=0; m<total_images; m++)
        {
            cv::Mat outmat(dims[0], dims[1], CV_32FC1);

            for(int y=0;y<outmat.rows;++y)
            for(int x=0;x<outmat.cols;++x)
            {
                float* pInp = (float*)outmat.ptr(y,x);

                *pInp = mddata2float(imgPtr + y * szStride[1] + x * szStride[0], dt);
            }

            //normalize and write image to disk
            cv::Mat norm;
            cv::normalize(outmat, norm, 255.0, 0.0, cv::NORM_MINMAX);

            cv::Mat cnv, cnvColor;
            norm.convertTo(cnv, CV_8U);

            cvtColor(cnv, cnvColor, cv::COLOR_GRAY2RGB);

            for(int y=0;y<outmat.rows;++y)
            for(int x=0;x<outmat.cols;++x)
            {
                float* pInp = (float*)outmat.ptr(y,x);
                unsigned char* pConv = (unsigned char*)cnvColor.ptr(y,x);

                unsigned u0 = *(unsigned int*)pInp;

                if( 0 == u0 )
                    pConv[2] = 255;
            }

            cv::imwrite(std::string(fname) + "_" + std::to_string(m) + ".png", cnvColor);

            imgPtr += szStride[2];
        }
    }

    std::cout << "Finished OK." << std::endl;
}
#else
void dumpTensor(vx_tensor tensor, const char* fname)
{
    return;
}
#endif
#endif
#endif


}
