/* ////////////////////////////////////////////////////////////////////////////////////
//                    INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2016-2019 Intel Corporation. All Rights Reserved.
//M*/

/*! \file vx_intel.h
\brief The vx_intel.h file provides declarations of Intel extensions for OpenVX* 1.1.
*/
#ifndef VX_INTEL_H
#define VX_INTEL_H

#include <VX/vx.h>

#ifdef  __cplusplus
extern "C"
{
#endif

#define VX_LIBRARY_INTEL (0x00)

enum vx_kernel_ext_intel_e
{
    VX_KERNEL_DIVIDE_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_INTEL) + 0x0,
    VX_KERNEL_COMPARE_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_INTEL) + 0x1,
    VX_KERNEL_NORM_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_INTEL) + 0x2,
    VX_KERNEL_LOG_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_INTEL) + 0x3,
    VX_KERNEL_ADAPTIVE_THRESHOLD_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_INTEL) + 0x4,
    VX_KERNEL_CHANNEL_SEPARATE_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_INTEL) + 0x5,
    VX_KERNEL_HOUGH_CIRCLES_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_INTEL) + 0x6,
    VX_KERNEL_SQRT_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_INTEL) + 0x7
};

enum vx_enum_ext_intel_e
{
    VX_ENUM_VENDOR_INTEL = 0x80,
    VX_ENUM_COMP_METRIC_TYPE_INTEL = VX_ENUM_VENDOR_INTEL + 0x1,
    VX_ENUM_DIVIDE_TYPE_INTEL = VX_ENUM_VENDOR_INTEL + 0x2,
    VX_ENUM_MATRIX_NAME_TYPE_INTEL = VX_ENUM_VENDOR_INTEL + 0x3,
    VX_ENUM_RANSAC_LINE_TYPE_INTEL = VX_ENUM_VENDOR_INTEL + 0x4,
    VX_ENUM_WARP_LENS_TYPE_INTEL = VX_ENUM_VENDOR_INTEL + 0x5,
    VX_ENUM_WARP_OUTPUT_TYPE_INTEL = VX_ENUM_VENDOR_INTEL + 0x6,
    VX_ENUM_ADAPTIVE_THRESHOLD_TYPE_INTEL = VX_ENUM_VENDOR_INTEL + 0x7,
    VX_ENUM_CONVOLUTION_PATTERN_INTEL = VX_ENUM_VENDOR_INTEL + 0x8,
};

typedef enum _vx_adaptive_threshold_type_intel_e
{
    VX_ADAPTIVE_THRESHOLD_FILTER_BOX_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_ADAPTIVE_THRESHOLD_TYPE_INTEL) + 0x0,
    VX_ADAPTIVE_THRESHOLD_FILTER_GAUSS_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_ADAPTIVE_THRESHOLD_TYPE_INTEL) + 0x1,
} vx_adaptive_threshold_type_intel_e;


enum vx_comp_metric_intel_e
{
    VX_COMPARE_HAMMING_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_COMP_METRIC_TYPE_INTEL) + 0x0,
    VX_COMPARE_L1_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_COMP_METRIC_TYPE_INTEL) + 0x1,
    VX_COMPARE_L2_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_COMP_METRIC_TYPE_INTEL) + 0x2,
    VX_COMPARE_CCORR_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_COMP_METRIC_TYPE_INTEL) + 0x3,
    VX_COMPARE_L2_NORM_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_COMP_METRIC_TYPE_INTEL) + 0x4,
    VX_COMPARE_CCORR_NORM_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_COMP_METRIC_TYPE_INTEL) + 0x5
};

enum vx_divide_options_intel_e
{
    VX_DIVIDE_OUTPUT_QUOTIENT_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_DIVIDE_TYPE_INTEL) + 0x0,
    VX_DIVIDE_OUTPUT_REMAINDER_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_DIVIDE_TYPE_INTEL) + 0x1,
    VX_DIVIDE_OUTPUT_FULL_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_DIVIDE_TYPE_INTEL) + 0x2
};


typedef enum _vx_matrix_name_intel_e
{
    VX_MATRIX_MIRROR_HORIZONTAL_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MATRIX_NAME_TYPE_INTEL) + 0x1,
    VX_MATRIX_MIRROR_VERTICAL_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MATRIX_NAME_TYPE_INTEL) + 0x2,
    VX_MATRIX_ROTATE90_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MATRIX_NAME_TYPE_INTEL) + 0x3,
    VX_MATRIX_ROTATE180_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MATRIX_NAME_TYPE_INTEL) + 0x4,
    VX_MATRIX_ROTATE270_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MATRIX_NAME_TYPE_INTEL) + 0x5,
    VX_MATRIX_TRANSPOSE_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MATRIX_NAME_TYPE_INTEL) + 0x6,
}vx_matrix_name_intel_e;

typedef struct _vx_circle_intel_t
{
    vx_float32 x;
    vx_float32 y;
    vx_float32 radius;
} vx_circle_intel_t;


/*! \brief Pre-defined convolution and gradient filters that can be created by <tt>\ref vxCreateConvolutionFromPatternIntel</tt>.
 * Here are the definitions for x gradients. The y gradients are the transpose. \n
 * 3x3 VX_CONVOLUTION_PATTERN_SOBEL_X_INTEL =    \f$\left(\begin{array}{ccc}
 * -1 & 0 & 1\\
 * -2 & 0 & 2\\
 * -1 & 0 & 1
 * \end{array}\right) \f$
 * Convolution scale is set to 1
 *
 * 5x5 VX_CONVOLUTION_PATTERN_SOBEL_X_INTEL =      \f$\left(\begin{array}{ccccc}
 * -1 & -2 & 0 & 2 & 1\\
 * -4 & -8 & 0 & 8 & 4\\
 * -6 & -12 & 0 & 12 & 6\\
 * -4 & -8 & 0 & 8 & 4\\
 * -1 & -2 & 0 & 2 & 1
 *  \end{array}\right) \f$
 * Convolution scale is set to 1
 *
 * 7x7 VX_CONVOLUTION_PATTERN_SOBEL_X_INTEL =      \f$\left(\begin{array}{ccccccc}
 *-1 & -4 & -5 & 0 & 5 & 4 & 1\\
 * -6 & -24 & -30 & 0 & 30 & 24 & 6\\
 * -15 & -60 & -75 & 0 & 75 & 60 & 15\\
 * -20 & -80 & -100 & 0 & 100 & 80 & 20\\
 * -15 & -60 & -75 & 0 & 75 & 60 & 15\\
 * -6 & -24 & -30 & 0 & 30 & 24 & 6\\
 * -1 & -4 & -5 & 0 & 5 & 4 & 1
 *  \end{array}\right) \f$
 * Convolution scale is set to 1
 *
 * 3x1 VX_CONVOLUTION_PATTERN_SOBEL_X_INTEL =    \f$\left(\begin{array}{ccc}
 * -1 & 0 & 1
 * \end{array}\right) \f$
 * Convolution scale is set to 1
 *
 * 5x1 VX_CONVOLUTION_PATTERN_SOBEL_X_INTEL =      \f$\left(\begin{array}{ccccc}
 * -1 & -2 & 0 & 2 & 1
 *  \end{array}\right) \f$
 * Convolution scale is set to 1
 *
 * 7x1 VX_CONVOLUTION_PATTERN_SOBEL_X_INTEL =      \f$\left(\begin{array}{ccccccc}
 * -1 & -4 & -5 & 0 & 5 & 4 & 1
 *  \end{array}\right) \f$
 * Convolution scale is set to 1
 *
 * 3x3 VX_CONVOLUTION_PATTERN_SCHARR_X_INTEL =    \f$\left(\begin{array}{ccc}
 * -3 & 0 & 3\\
 * -10 & 0 & 10\\
 * -3 & 0 & 3
 * \end{array}\right) \f$
 * Convolution scale is set to 1
 *
 * 3x3 VX_CONVOLUTION_PATTERN_GAUSSIAN_INTEL =    \f$\left(\begin{array}{ccc}
 * 1 & 2 & 1\\
 * 2 & 4 & 2\\
 * 1 & 2 & 1
 * \end{array}\right) \f$
 * Convolution scale is set to 1/16
 *
 * 5x5 VX_CONVOLUTION_PATTERN_GAUSSIAN_INTEL =    \f$\left(\begin{array}{ccccc}
 * 1 & 4 & 6 & 4 & 1\\
 * 4 & 16 & 24 & 16 & 4\\
 * 6 & 24 & 36 & 24 & 6\\
 * 4 & 16 & 24 & 16 & 4\\
 * 1 & 4 & 6 & 4 & 1
 * \end{array}\right) \f$
 * Convolution scale is set to 1/256
 *
 * 7x7 VX_CONVOLUTION_PATTERN_GAUSSIAN_INTEL =    \f$\left(\begin{array}{ccccccc}
 * 1 & 6 & 15 & 20 & 15 & 6 & 1\\
 * 6 & 36 & 90 & 120 & 90 & 36 & 6\\
 * 15 & 90 & 225 & 300 & 225 & 90 & 15\\
 * 20 & 120 & 300 & 400 & 300 & 120 & 20\\
 * 15 & 90 & 225 & 300 & 225 & 90 & 15\\
 * 6 & 36 & 90 & 120 & 90 & 36 & 6\\
 * 1 & 6 & 15 & 20 & 15 & 6 & 1
 * \end{array}\right) \f$
 * Convolution scale is set to 1/4096
 *
 * VX_CONVOLUTION_PATTERN_BOX_INTEL - all coefficients are set to 1
 * Convolution scale is set to 1/(filter_width * filter_height)
 */
enum vx_convolution_pattern_intel_e {
    /*! \brief The Scharr gradient in horizontal direction. Only 3x3 size is supported. */
    VX_CONVOLUTION_PATTERN_SCHARR_X_INTEL  = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_CONVOLUTION_PATTERN_INTEL) + 0x1,
    /*! \brief The Scharr gradients in vertical direction. Only 3x3 size is supported. */
    VX_CONVOLUTION_PATTERN_SCHARR_Y_INTEL  = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_CONVOLUTION_PATTERN_INTEL) + 0x2,
    /*! \brief The Sobel gradients in horizontal direction. Supported 3x3, 5x5, 7x7 and 3x1, 5x1, 7x1 sizes. */
    VX_CONVOLUTION_PATTERN_SOBEL_X_INTEL   = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_CONVOLUTION_PATTERN_INTEL) + 0x3,
    /*! \brief The Sobel gradients in vertical direction. Supported 3x3, 5x5, 7x7 and 1x3, 1x5, 1x7 sizes. */
    VX_CONVOLUTION_PATTERN_SOBEL_Y_INTEL   = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_CONVOLUTION_PATTERN_INTEL) + 0x4,
    /*! \brief The Gaussian smoothing filter. Supported 3x3, 5x5, 7x7 sizes. */
    VX_CONVOLUTION_PATTERN_GAUSSIAN_INTEL  = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_CONVOLUTION_PATTERN_INTEL) + 0x5,
    /*! \brief The box smoothing filter. Supported all odd sizes of square filter from 3x3 to 31x31. */
    VX_CONVOLUTION_PATTERN_BOX_INTEL       = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_CONVOLUTION_PATTERN_INTEL) + 0x6,
};


/*! \brief [Graph] Applies an adaptive threshold to an image.
 * In case of adaptive threshold box type the following equation apply
 * \f$ intermediate(y,x)=\frac{1}{N^{2}}\sum_{i=\frac{N-1}{2}}^{\frac{{N-1}}{2}}\sum_{j=\frac{{N-1}}{2}}^{\frac{{N-1}}{2}}input(y+j,x+i)\\
 *  \f$
 * In case of adaptive threshold Gaussian type the following equation apply
 * \f$ intermediate(y,x)=\frac{1}{N^{2}}\sum_{i=\frac{N-1}{2}}^{\frac{{N-1}}{2}}\sum_{j=\frac{{N-1}}{2}}^{\frac{{N-1}}{2}}G(i,j)input(y+j,x+i)\\
 * \f$
 * Where \f$G(i,j) \f$ is the Gaussian filter. Currently only Gaussian of 3x3 and 5x5 are supported.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [in] filter A filtering type from <tt>\ref vx_adaptive_threshold_filter_intel_e</tt> enumeration. Possible values: <tt>\ref VX_ADAPTIVE_THRESHOLD_FILTER_BOX_INTEL</tt>, <tt>\ref VX_ADAPTIVE_THRESHOLD_FILTER_GAUSS_INTEL</tt>.
 * \param [in] ksize Linear size of the filtering kernel. Supported values are 3 and 5 for <tt>\ref VX_ADAPTIVE_THRESHOLD_FILTER_GAUSS_INTEL</tt> and any odd nuber from 3 to 31 for <tt>\ref VX_ADAPTIVE_THRESHOLD_FILTER_BOX_INTEL</tt>.
 * \param [in] threshold <tt>\ref vx_threshold</tt> of type <tt>\ref VX_THRESHOLD_TYPE_BINARY</tt>.
 * \param [out] output The output image <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxAdaptiveThresholdNodeIntel(vx_graph graph,
        vx_image input, vx_enum filter, vx_uint32 ksize, vx_threshold threshold,
        vx_image output);

/*! \brief Splits multi-channel image into separate channels.
 * \param [in] graph The reference to the graph.
 * \param [in] input input image in <tt>\ref VX_DF_IMAGE_RGB</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt> formats.
 * \param [out] plane0 Output channel 0.
 * \param [out] plane1 Output channel 1.
 * \param [out] plane2 Output channel 2.[optional]
 * \param [out] plane3 Output channel 3 [optional] Outputs VX_DF_IMAGE_U8 images.
 * \return vx node A node reference.
 * \Any possible errors preventing a successful creation should be checked using vxGetStatus
 */
VX_API_ENTRY vx_node VX_API_CALL vxChannelSeparateNodeIntel(vx_graph graph,
        vx_image input, vx_image plane0, vx_image plane1, vx_image plane2,
        vx_image plane3);

/*! \brief Performs comparison between elements (pixels) of two source matrices
 * \param [in] graph The reference to the graph.
 * \param [in] input1 first input.
 * \param [in] input2 Second input Supported input data types:
 * <tt>\ref VX_DF_IMAGE_RGBX</tt>
 * <tt>\ref VX_DF_IMAGE_RGB</tt>
 * <tt>\ref VX_DF_IMAGE_U8</tt>
 * <tt>\ref VX_DF_IMAGE_U16</tt>
 * <tt>\ref VX_DF_IMAGE_S16</tt>
 * Both input images must be of the same type and size.
 * \param [out] output The output image Output data type is either <tt>\ref VX_DF_IMAGE_RGBX</tt>,
 * <tt>\ref VX_DF_IMAGE_RGB</tt> for corresponding input image types, or <tt>\ref VX_DF_IMAGE_U8</tt> for all other data types.
 * Output pixels values equal to 255 if input pixels meet the compare condition, and 0 other-wise.
 * \return vx node A node reference.
 * \Any possible errors preventing a successful creation should be checked using vxGetStatus
 */

VX_API_ENTRY vx_node VX_API_CALL vxCompareNodeIntel(vx_graph graph,
        vx_image input1, vx_image input2, vx_enum type, vx_image output);

/*! \brief The kernel performs element-wise integer division between two images in <tt>\ref VX_DF_IMAGE_U8</tt> or
 * <tt>\ref VX_DF_IMAGE_S16</tt> formats, and computes the quotient and the reminder per pixel in two separate output
 * images. If both input images are <tt>\ref VX_DF_IMAGE_U8</tt> then the output images will be <tt>\ref VX_DF_IMAGE_U8</tt>.
 *  If any of the input images is <tt>\ref VX_DF_IMAGE_S16</tt> then the output images will be <tt>\ref VX_DF_IMAGE_S16</tt>.
 * \param [in] graph The reference to the graph.
 * \param [in] divident Dividend image of <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [in] divisor Divisor image of <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [out] quotient Quotient image of <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [out] reminder Reminder image of <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \return vx node A node reference.
 * \Any possible errors preventing a successful creation should be checked using vxGetStatus
 */

VX_API_ENTRY vx_node VX_API_CALL vxDivideNodeIntel(vx_graph graph,
        vx_image divident, vx_image divisor, vx_image quotient,
        vx_image reminder);

/*! \brief [Graph] Computes an absolute difference norm of a given image
 *(images) or relative difference norm of two given images.
 * \param [in] graph The reference to the graph.
 * \param [in] input1 The first input image of <tt>\ref VX_DF_IMAGE_U8</tt> or
 * <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [in] input2 [Optional] The second input image of the same data format as input1.
 * \param [in] metric  A norm type from the <tt>\ref vx_norm_intel_e</tt>
 * enumeration. L1, L2, INF norm types are supported.
 * \param [out] output The output norm value of type <tt>\ref VX_TYPE_UINT32</tt>.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful
 * creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxNormNodeIntel(vx_graph graph,
        vx_image input1, vx_image input2, vx_enum metric, vx_scalar output);

/*! \brief Calculate a log of the image.
 * \param [in] graph The reference to the graph.
 * \param [in] src The input image of type <tt>\ref VX_DF_IMAGE_S16</tt>.
 * \param [in] scale A Scale const parameter. The result will be multiplied by the scale
 * factor before converted to S16.
 * \param [in] overflow_policy policy to use when the result overflowed of type vx_convert_polict_e.
 * \param [in] rounding_policy policy to use in truncation. of type vx_round_policy_e.
 * \param [out] dst log of the input image of type <tt>\ref VX_DF_IMAGE_S16</tt>.
 * \return vx node A node reference.
 * \Any possible errors preventing a successful creation should be checked using vxGetStatus
 */

VX_API_ENTRY vx_node VX_API_CALL vxLogNodeIntel(vx_graph graph, vx_image src,
        vx_int32 scale, vx_enum overflow_policy, vx_enum rounding_policy,
        vx_image dst);

/*! \brief [Graph] Detects circles using Hough circle transform (Hough gradient base) and stores the result in an array.
 * Circles are defined using three parameters: center (x0, y0) and radius r.
 * The result is stored in a vx_array of type <tt>\ref vx_coordinates3d_t</tt>, the first two values in it
 * corresponds to center coordinates, and the third one is the radius.
 * \param [in] graph The reference to the graph.
 * \param [in] input An Input binary image of edges in <tt>\ref VX_DF_IMAGE_U8</tt> format
 * \param [in] dx An Input image of dx gradients in <tt>\ref VX_DF_IMAGE_S16</tt> format
 * \param [in] dy An Input image of dy gradients in <tt>\ref VX_DF_IMAGE_S16</tt> format
 * \param [in] minDistance Non-negative <tt>\ref VX_TYPE_INT32</tt> minimum distance between centers of detected circles. Used for non-maxima suppresison.
 * \param [in] minRadius Non-negative <tt>\ref VX_TYPE_INT32</tt> minimum radius of detected circle.
 * \param [in] maxRadius Non-negative <tt>\ref VX_TYPE_INT32</tt> maximum radius of detected circle.
 * \param [in] minEvidencePointsThreshold <tt>\ref VX_TYPE_INT32</tt> minimum points count in the suitable neighborhood to propose center.
 * \param [in] minCircleFilledThreshold <tt>\ref VX_TYPE_FLOAT32</tt> in range [0..1] minimum percent of circle line filling to detect circle.
 * \param [out] circles The array of <tt>\ref VX_TYPE_COORDINATES3D</tt> objects. The order of the keypoints in this array is implementation dependent.
 * \param [out] count [Optional] The total number of detected circles. Use a <tt>\ref VX_TYPE_SIZE</tt> scalar.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxHoughCirclesNodeIntel(vx_graph graph,
        vx_image input, vx_image dx, vx_image dy, vx_scalar minDistance,
        vx_scalar minRadius, vx_scalar maxRadius,
        vx_scalar minEvidencePointsThreshold, vx_scalar minCircleFilledThreshold,
        vx_array circles, vx_scalar count);

/*! \brief Computes square roots of pixel values of a source image and writes them
 * into the destination image. values must be less or equal 2 ^ 15-1 for <tt>\ref VX_DF_IMAGE_S16</tt>.
 * and less than 2 ^ 31-1 for <tt>\ref VX_DF_IMAGE_S32</tt>. And only positive values.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image of type <tt>\ref VX_DF_IMAGE_S16</tt> or <tt>\ref VX_DF_IMAGE_S32</tt>.
 * \param [out] output The output image of type <tt>\ref VX_DF_IMAGE_S16</tt>.
 * \return vx node A node reference.
 * \Any possible errors preventing a successful creation should be checked using vxGetStatus
 */

VX_API_ENTRY vx_node VX_API_CALL vxSqrtNodeIntel(vx_graph graph, vx_image input,
        vx_image output);

/*! \brief Create a <tt>vx_matrix</tt> object by a specified named matrix transform.
 * \param [in] context The reference to the overall context.
 * \param [in] transform The name of transform from <tt>\ref vx_matrix_name_intel_e</tt> enumeration.
 * \param [in] image_width The first dimensionality of transformed image.
 * \param [in] image_height The second dimensionality of transformed image.
 * \returns An 2x3 affine transform matrix reference <tt>\ref vx_matrix</tt> of type <tt>\ref vx_float32</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_matrix VX_API_CALL vxCreateNamedMatrixAffineTransformIntel(
        vx_context context, vx_enum transform, vx_uint32 image_width,
        vx_uint32 image_height);

/*! \brief Create a <tt>\ref vx_convolution</tt> object by a specified named convolution.
* \param [in] context The reference to the overall context.
* \param [in] pattern The pattern of the convloution from <tt>\ref vx_convolution_pattern_intel_e</tt> enumeration.
* \param [in] columns The first dimensionality.
* \param [in] rows The second dimensionality.
* \returns An convolution reference <tt>\ref vx_convolution</tt> of type <tt>\ref vx_int16</tt>. Any possible errors preventing a
* successful creation should be checked using <tt>\ref vxGetStatus</tt>.
*/
VX_API_ENTRY vx_convolution VX_API_CALL vxCreateConvolutionFromPatternIntel(vx_context context, vx_enum pattern, vx_size columns, vx_size rows);

/*! \brief [Graph] Creates a Shi Tomasi  Corners Node.(TODO: copy explanation from wikipedia)
* \param [in] graph The reference to the graph.
* \param [in] input The input <tt>\ref VX_DF_IMAGE_U8</tt> image.
* \param [in] strength_thresh The <tt>\ref VX_TYPE_FLOAT32</tt> threshold for minimal eigen value with which to eliminate.
* \param [in] min_distance The <tt>\ref VX_TYPE_FLOAT32</tt> radial Euclidean distance for non-maximum suppression.
* \param [in] gradient_size The gradient window size to use on the input. The
* implementation must support at least 3, 5, and 7.
* \param [in] block_size The block window size used to compute the eigen value.
* The implementation must support at least 3, 5, and 7.
* \param [out] corners The array of <tt>\ref VX_TYPE_KEYPOINT</tt> objects. The order of the keypoints in this array is implementation dependent.
* \param [out] num_corners [Optional] The total number of detected corners in image. Use a <tt>\ref VX_TYPE_SIZE</tt> scalar.
* \return <tt>\ref vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
*/
VX_API_ENTRY vx_node VX_API_CALL vxShiTomasiCornersNodeIntel(vx_graph graph,
    vx_image input,
    vx_scalar strength_thresh,
    vx_scalar min_distance,
    vx_int32 gradient_size,
    vx_int32 block_size,
    vx_array corners,
    vx_scalar num_corners);

#ifdef  __cplusplus
} // extern "C"
#endif
#endif /*VX_INTEL_H */
