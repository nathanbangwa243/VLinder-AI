/* ////////////////////////////////////////////////////////////////////////////////////
//                    INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2016-2019 Intel Corporation. All Rights Reserved.
//M*/

/*! \file vx_intel_volatile.h
\brief The vx_intel_volatile.h file provides declarations of experimental Intel extensions for OpenVX* 1.1
*/
#ifndef VX_INTEL_VOLATILE_H
#define VX_INTEL_VOLATILE_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include <VX/vx.h>
#include <VX/vx_intel.h>
#include <VX/vx_khr_nn.h>

#ifdef  __cplusplus
extern "C"
{
#endif

/*! \brief The Object Type Enumeration for Targets.
 * \ingroup group_target
 */
enum {
    VX_TYPE_PARAM_STRUCT_INTEL = VX_TYPE_VENDOR_OBJECT_START + 0x2, /*!< \brief A <tt>\ref vx_param_struct_intel</tt>. */
    VX_TYPE_BG_STATE_INTEL = VX_TYPE_VENDOR_OBJECT_START + 0x3, /*!< \brief A <tt>\ref vx_bg_state_intel</tt>. */
    VX_TYPE_SVM_PARAMS_INTEL = VX_TYPE_VENDOR_OBJECT_START + 0x4, /*!< \brief A <tt>\ref vx_svm_params_intel</tt>. */
    VX_TYPE_SEPFILTER2D_INTEL = VX_TYPE_VENDOR_OBJECT_START + 0x5, /*!< \brief A <tt>\ref vx_sepfilter2d</tt>. */
    VX_TYPE_DEVICE_LIBRARY_INTEL = VX_TYPE_VENDOR_OBJECT_START + 0x6, /*!< \brief A <tt>\ref vx_device_kernel_library</tt>. */
    VX_TYPE_TARGET_INTEL = VX_TYPE_VENDOR_OBJECT_START + 0x7, /*!< \brief A <tt>\ref vx_target_intel</tt> */
};


/*! \brief The Object Type Enumeration for extensions of basic features.
 * \ingroup group_basic_features_ext
 */
enum {
    VX_TYPE_KEYPOINT_F32_INTEL = VX_TYPE_VENDOR_OBJECT_START + 0x8,/*!< \brief A <tt>\ref vx_keypoint_f32_t</tt>. */
};


/*! \brief The keypoint data structure with floating-point x, y coordinates.
 * \ingroup group_basic_features_ext
 */
typedef struct _vx_keypoint_f32_intel_t {
    vx_float32 x;               /*!< \brief The x coordinate. */
    vx_float32 y;               /*!< \brief The y coordinate. */
    vx_float32 strength;        /*!< \brief The strength of the keypoint. Its definition is specific to the corner detector. */
    vx_float32 scale;           /*!< \brief Initialized to 0 by corner detectors. */
    vx_float32 orientation;     /*!< \brief Initialized to 0 by corner detectors. */
    vx_int32 tracking_status;   /*!< \brief A zero indicates a lost point. Initialized to 1 by corner detectors. */
    vx_float32 error;           /*!< \brief A tracking method specific error. Initialized to 0 by corner detectors. */
} vx_keypoint_f32_intel_t;


/*! \brief The set of additional supported enumerations in OpenVX.
 * \details These can be extracted from enumerated values using <tt>\ref VX_ENUM_TYPE</tt>.
 * \ingroup group_intel_basic_features
 */
enum vx_enum_intel_e {
    VX_ENUM_MDDATA_DF_INTEL           = 0xA0, /*!< \brief MD Data data format. */
};

/*!
 * \brief Returns the name of a status code constant.
 * \param [status] The status code.
 * \return If \a status corresponds to one of the constants from vx_status_e, a pointer to a string
 * with the name of that constant; otherwise, NULL.
 * The string will remain valid for the entire lifetime of the library and must not be freed.
 * \ingroup group_intel_basic_features
 */
VX_API_ENTRY const char* VX_API_CALL vxGetStatusNameIntel(vx_status status);

#define VX_MAX_TENSOR_DIMS_INTEL                 (6)

/*! \brief Input parameters for a convolution operation.
 * \ingroup group_intel_cnn
 */
typedef struct _vx_conv_params_intel_t {
    vx_uint32 stride_x;     /*!< \brief Distance in elements between successive filter applications in the x dimension. */
    vx_uint32 stride_y;     /*!< \brief Distance in elements between successive filter applications in the y dimension. */
    vx_uint32 pad_x;        /*!< \brief Number of elements added at each side in the x dimension of the input. */
    vx_uint32 pad_y;        /*!< \brief Number of elements added at each side in the y dimension of the input. */
} vx_conv_params_intel_t;

/*! \brief Input parameters for a pooling operation.
 * \ingroup group_intel_cnn
 */
typedef struct _vx_pool_params_intel_t {
    vx_uint32 pool_size_x;  /*!< \brief Size of the pooling region in the x dimension. */
    vx_uint32 pool_size_y;  /*!< \brief Size of the pooling region in the y dimension. */
    vx_uint32 pool_pad_x;   /*!< \brief Padding size in the x dimension. */
    vx_uint32 pool_pad_y;   /*!< \brief Padding size in the y dimension. */
    vx_uint32 pool_stride_x;/*!< \brief Distance in elements between successive pooling applications in the x dimension. */
    vx_uint32 pool_stride_y;/*!< \brief Distance in elements between successive pooling applications in the y dimension. */
    vx_enum rounding;       /*!< \brief Rounding method of output dimensions. See vx_intel_cnn_rounding_type_e */
} vx_pool_params_intel_t;

/*! \brief Input parameters for a normalization operation.
 * \ingroup group_intel_cnn
 */
typedef struct _vx_norm_params_intel_t {
    vx_uint32 norm_size;    /*!< \brief Number of elements to normalize across. */
    vx_float32 alpha;       /*!< \brief Normalization alpha. */
    vx_float32 beta;        /*!< \brief Normalization beta. */
} vx_norm_params_intel_t;

enum vx_df_mddata_intel_e {
   VX_DF_MDDATA_S16_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MDDATA_DF_INTEL) + 0x0, /*! \brief A 16-bit signed integer element */
   VX_DF_MDDATA_Q78_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MDDATA_DF_INTEL) + 0x1, /*! \brief A 16-bit Q1.7.8 fixed point element */
   VX_DF_MDDATA_FLOAT16_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MDDATA_DF_INTEL) + 0x2, /*! \brief A 16-bit floating point element */
   VX_DF_MDDATA_FLOAT32_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MDDATA_DF_INTEL) + 0x3 /*! \brief A 32-bit floating point element */
};

/*! \brief Parameter struct attributes list.
* \ingroup group_intel_param_struct
*/
enum vx_param_struct_attribute_intel_e {
    /*! \brief Queries the user-given sub-type of parameter struct. Read only. Use a <tt>\ref vx_size</tt>. */
    VX_PARAM_STRUCT_SUBTYPE_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_PARAM_STRUCT_INTEL) + 0x0,
    /*! \brief Queries the size of parameter struct in bytes. Read only. Use a <tt>\ref vx_size</tt>. */
    VX_PARAM_STRUCT_SIZE_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_PARAM_STRUCT_INTEL) + 0x1,
};


typedef struct _vx_mdview_addressing_intel_t {
   vx_uint32 dim[VX_MAX_TENSOR_DIMS_INTEL];        /*!< \brief Length of patch in all dimensions in elements. */
   vx_uint32 stride[VX_MAX_TENSOR_DIMS_INTEL];     /*!< \brief Stride in all dimensions in bytes. */
} vx_mdview_addressing_intel_t;


/*! \brief Input parameters for a non linear operation.
* \ingroup group_intel_cnn
*/
typedef struct vx_non_linear_params_intel_t {
    vx_int32 a; /*!< \brief Non Linear first parameter. */
    vx_int32 b; /*!< \brief Non Linear second parameter. */
} vx_non_linear_params_intel;

/*! \brief This structure is used to pass a large list of parameters to a node.
 * \see vxCreateIntelParameterStruct
 * \ingroup group_param_struct
 * \extends vx_reference
 */
typedef struct _vx_parameter_struct_intel_t *vx_parameter_struct_intel;

#define VX_LIBRARY_EXPERIMENTAL_INTEL (0x01)

/*! \brief Input parameters for a roi pooling operation.
 * \ingroup group_intel_cnn
 */
typedef struct _vx_roi_pool_params_intel_t {
    vx_enum pool_type;  /*!< \brief Only Max pooling is supported (see vx_intel_cnn_pooling_type_e). */
} vx_roi_pool_params_intel_t;

/*! \brief [Graph] Creates a CNN RegionOfInterest Pooling Layer Node.
 *  This node pools each of the ROIs given in input_rois within the input_data into a fixed size output.
 *
 * \param [in] graph The handle to the graph.
 * \param [in] input_data The input md data. 3 lower dims represent a single input with dimensions [width, height, IFM], and an optional 4th dimension for batch of inputs.
 * \param [in] input_rois The rois md data. The dimensions of which are [4, roi_count] and an optional batch count which must match input_datas. The 4-tuples are coordinates of ROIs in the input_data space, [x0, y0, x1, y1] where x0, y0 are the top left corner and x1, y1 are the bot right corner.
 * \param [in] pool_params Parameter struct containing the roi pooling parameters (see vx_roi_pool_params_intel_t). ATM, it must be set to max pooling (see vx_intel_cnn_pooling_type_e).
 * \param [out] output_arr The output md data. Output dimensions are [width, height, IFM, roi_count] and an optional batch count. width and height here set the roi pooling target size. IFM, roi_count and the optional batch count must match those provided in the inputs.
 * \ingroup group_intel_cnn
 */
VX_API_ENTRY vx_node VX_API_CALL vxCNNROIPoolingNodeIntel(vx_graph graph,
        vx_tensor input_data, vx_tensor input_rois,
        vx_parameter_struct_intel roi_pool_params, vx_tensor output_arr);

/*! \brief [Graph] Creates a CNN tensor Convert Node.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor.
 * \param [out] outputs The output tensor. Output will have the same number of dimensions as input.
 * \ingroup group_intel_cnn
 */
VX_API_ENTRY vx_node VX_API_CALL vxCNNConvertNodeIntel(vx_graph graph, vx_tensor inputs, vx_tensor outputs);

/*! \brief [Graph] Creates a tensor copy Node.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor.
 * \param [out] outputs The output tensor. Output will be a copy of the input.
 * \ingroup group_intel_cnn
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorCopyNodeIntel(vx_graph graph, vx_tensor inputs, vx_tensor outputs);

/*! \brief Creates a tensor from user-allocated memory
 * \param [in] context The reference to the implementation context.
 * \param [in] number_of_dims The number of dimensions.
 * \param [in] dims Dimensions sizes in elements.
 * \param [in] data_type The <tt>vx_type_t</tt> that represents the data type of the tensor data elements.
 * \param [in] fixed_point_position Specifies the fixed point position when the input element type is int16, if 0 calculations are performed in integer math
 * \param [in] user_ptr The pointer to user-allocated tensor data
 * \param [in] user_strides The pointer to array of strides of user data
 * \param [in] <tt>\ref vx_memory_type_e</tt>. When passing <tt>\ref VX_MEMORY_TYPE_HOST</tt>
 * the \a ptr is assumed to be HOST accessible pointer to memory.
 * \return A tensor data reference. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_tensor VX_API_CALL vxCreateTensorFromHandleIntel(vx_context context,
                                                            vx_size number_of_dims,
                                                            const vx_size *dims,
                                                            vx_enum data_type,
                                                            vx_uint8 fixed_point_position,
                                                            void * user_ptr,
                                                            const vx_size * user_strides,
                                                            vx_enum memory_type);

/*==============================================================================
 CONVOLUTION
 =============================================================================*/

/*! \brief [Graph] Creates a CNN dilated convolution Node.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data. 3 lower dimensions represent a single input, all following dimensions represent number of batches, possibly nested.
 * The dimensions order is [width, height, #IFM, #batches]\n. Implementations must support input tensor data type <tt>VX_TYPE_INT16</tt>. with fixed_point_position 8.
 * \param [in] weights Weights are 4d tensor with dimensions [kernel_x, kernel_y, #IFM, #OFM].\n Implementations must support input tensor data type <tt>VX_TYPE_INT16</tt>. with fixed_point_position 8.
 * \param [in] biases Optional, ignored if NULL. The biases, which may be shared (one per ofm) or unshared (one per ofm * output location). The possible layouts are
 * either [#OFM] or [width, height, #OFM]. Implementations must support input tensor data type <tt>VX_TYPE_INT16</tt>. with fixed_point_position 8.
 * \param [in] padding_x Number of elements added at each side in the x dimension of the input.
 * \param [in] padding_y Number of elements added at each side in the y dimension of the input. In fully connected layers this input is ignored.
 * \param [in] overflow_policy A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_convert_policy_e</tt> enumeration.
 * \param [in] rounding_policy A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_round_policy_e</tt> enumeration.
 * \param [in] down_scale_size_rounding Rounding method for calculating output dimensions. See <tt>\ref vx_nn_rounding_type_e</tt>
 * \param [in] dilation_x A <tt> VX_TYPE_SCALAR</tt>. Use a <tt>\ref vx_uint32</tt> parameter.
 * \param [in] dilation_y A <tt> VX_TYPE_SCALAR</tt>. Use a <tt>\ref vx_uint32</tt> parameter.
 * \param [out] outputs The output tensor. Output will have the same number of dimensions as input.
 * \ingroup group_intel_cnn
 */
VX_API_ENTRY vx_node VX_API_CALL vxCNNConvolutionLayerDilationNodeIntel(vx_graph graph,
                                                           vx_tensor inputs,
                                                           vx_tensor weights,
                                                           vx_tensor biases,
                                                           vx_size pad_x,
                                                           vx_size pad_y,
                                                           vx_enum overflow_policy,
                                                           vx_enum rounding_policy,
                                                           vx_enum down_scale_size_rounding,
                                                           vx_uint32 dilation_x,
                                                           vx_uint32 dilation_y,
                                                           vx_tensor outputs);

/*! \brief Creates an opaque reference to an object wrapping a constant struct of input parameters for a node.
 * \param [in] context The reference to the implementation context.
 * \param [in] subtype The sub-type of the wrapped struct. This may have per-kernel meaning.
 * \param [in] size The size of the wrapped struct in bytes.
 * \param [in] data A pointer to the initial data for struct of parameters.
 * If NULL was provided, then object's data will not be initialized.
 * \return A parameter struct reference.
 * \retval 0 No Param struct object was created.
 * \retval * A Param struct was created or an error occurred. Use <tt>\ref vxGetStatus</tt> to determine.
 * \ingroup group_intel_param_struct
 */
VX_API_ENTRY vx_parameter_struct_intel VX_API_CALL vxCreateParamStructIntel(
        vx_context context, vx_size subtype, vx_size size, const void *data);

/*! \brief Releases a reference to a parameter struct object.
 * The object may not be garbage collected until its total reference count is zero.
 * \param [in] param_struct The pointer to the parameter struct to release.
 * \post After returning from this function the reference is zeroed.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \ingroup group_intel_param_struct
 */
VX_API_ENTRY vx_status VX_API_CALL vxReleaseParamStructIntel(
        vx_parameter_struct_intel *param_struct);

/*! \brief Retrieves various attributes of a parameter struct.
 * \param [in] param_struct The parameter struct to query.
 * \param [in] attribute The attribute to query. Use a <tt>\ref vx_intel_param_struct_attribute_e</tt>.
 * \param [out] ptr The location at which to store the resulting value.
 * \param [in] size The size of the container to which \a ptr points.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_PARAMETERS If any of the other parameters are incorrect.
 * \ingroup group_intel_param_struct
 */
VX_API_ENTRY vx_status VX_API_CALL vxQueryParamStructIntel(
        vx_parameter_struct_intel param_struct, vx_enum attribute, void *ptr,
        vx_size size);

/*! \brief Copy the data of the parameter struct to user memory or copy user memory to parameter struct.
 * \param [in] param_struct The accessed parameter struct.
 * \param [in] user_ptr The pointer to user memory from which to read from or write to.
 * \param [in] usage Whether copy operation is a read or a write.
 * If usage is VX_READ_ONLY then parameter struct data will be copied to user memory.
 * Otherwise, if usage is VX_WRITE_ONLY then user memory will be copied to
 * the parameter struct. No other value is accepted for usage.
 * \param [in] user_mem_type In the current spec version this parameter is ignored.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_ERROR_INVALID_PARAMETERS Invalid <tt>\ref vx_parameter_struct_intel</tt> or null ptr.
 * \ingroup group_intel_param_struct
 */
VX_API_ENTRY vx_status VX_API_CALL vxCopyParamStructIntel(
        vx_parameter_struct_intel param_struct, void* user_ptr, vx_enum usage,
        vx_enum user_mem_type);


enum vx_kernel_experimental_ext_intel
{
    VX_KERNEL_POLARTOCART_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x0,
    VX_KERNEL_HOUGH_LINES_P_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x1,
    VX_KERNEL_MATCH_TEMPLATE_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x3,
    VX_KERNEL_BILATERAL_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x4,
    VX_KERNEL_CASCADECLASSIFIER_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x5,
    VX_KERNEL_RANSACLINE_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x6,
    VX_KERNEL_MINMAXPIXEL_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x7,
    VX_KERNEL_WARP_PROJECTION_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x8,
    VX_KERNEL_EUCLIDEAN_NMS_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x9,
    VX_KERNEL_SVMPREDICTION_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0xA,
    VX_KERNEL_HOG_INTEL = VX_KERNEL_BASE( VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL ) + 0xB,
    VX_KERNEL_RANSACHOMOGRAPHY_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0xC,
    VX_KERNEL_SHITOMASICORNERS_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0xD,
    VX_KERNEL_FEATUREMATCHING_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0xE,
    VX_KERNEL_BLOCKMATCHING_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0xF,
    VX_KERNEL_CREATE_NAMED_MATRIXAFFINE_TRANSFORM_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x10,
    VX_KERNEL_MIRROR_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x11,
    VX_KERNEL_NON_MAX_SUPPRESSION_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x12,
    VX_KERNEL_OTSU_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x13,
    VX_KERNEL_FFT_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x14,
    VX_KERNEL_GRADIENT_VECTOR_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x15,
    VX_KERNEL_LBP_IMAGE_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x16,
    VX_KERNEL_HAAR_CLASSIFIER_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x17,
    VX_KERNEL_CONVERT_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x18,
    VX_KERNEL_ROTATE_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x19,
    VX_KERNEL_ROTATE90_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x1A,
    VX_KERNEL_MASKED_COPY_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x1B,
    VX_KERNEL_PSR_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x1C,
    VX_KERNEL_PEAK_LISTER_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x1D,
    VX_KERNEL_SQR_INTEGRAL_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x1E,
    VX_KERNEL_CROP_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x1F,
    VX_KERNEL_DCT_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x20,
    VX_KERNEL_COLORTOGRAY_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x21,
    VX_KERNEL_MULSPECTRUMS_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL)+ 0x22,
    VX_KERNEL_HOUGH_LINES_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL)+ 0x23,
    VX_KERNEL_MORPHOLOGYEX_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL)+ 0x24,
    VX_KERNEL_TRANSPOSE_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL)+ 0x25,
    VX_KERNEL_NORMALIZE_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL)+ 0x26,
    VX_KERNEL_TRUEDISTANCETRANSFORM_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x27,
    VX_KERNEL_CORNERMINEIGENVAL_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x28,
    VX_KERNEL_RANSACFITTING_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x29,
    VX_KERNEL_SPHERICAL_WARPER_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x2A,
    VX_KERNEL_CYLINDRICAL_WARPER_INTEL = VX_KERNEL_BASE(VX_ID_INTEL,VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x2B,
    VX_KERNEL_STEREOBM_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL)+ 0x2C,
    VX_KERNEL_BGSUBMOG2_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL)+ 0x2D,
    VX_KERNEL_SYMM7x7_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL)+ 0x2E,
    VX_KERNEL_SEP_FILTER_2D_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x30,
    VX_KERNEL_BOX_NXN_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL)+ 0x31,
    VX_KERNEL_IN_RANGE_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL)+ 0x32,
    VX_KERNEL_SVMCLASSIFIER_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x33,
    VX_KERNEL_RGBTOYCBCR_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL)+ 0x34,
    VX_KERNEL_RGBTOLAB_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL)+ 0x35,
    VX_KERNEL_HALFTONE_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL)+ 0x36,
    VX_KERNEL_TETRAHEDRAL_INTERPOLATION_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL)+ 0x37,
    VX_KERNEL_LUT_3D_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x38,
    VX_KERNEL_CNNROIPOOLING_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x39,
    VX_KERNEL_TENSOR_CONVERT_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x3A,
    VX_KERNEL_CONVOLUTION_LAYER_DILATION_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x3B,
    VX_KERNEL_TENSOR_COPY_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x3C,
    VX_KERNEL_SYMMNxN_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x3D,
    VX_KERNEL_PAD_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x3E,
    VX_KERNEL_PACK8TO1_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x40,
    VX_KERNEL_UNPACK1TO8_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x41,
    VX_KERNEL_PACK8TO2_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x42,
    VX_KERNEL_UNPACK2TO8_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x43,
    VX_KERNEL_PACK8TO4_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x44,
    VX_KERNEL_UNPACK4TO8_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_EXPERIMENTAL_INTEL) + 0x45,
};

enum {
    VX_ENUM_ALPHA_TYPE_INTEL = 0xC0,
    VX_ENUM_MIRROR_TYPE_INTEL = 0xC1,
    VX_ENUM_FFT_NORM_TYPE_INTEL = 0xC2,
    VX_ENUM_ROUND_INTEL = 0xC3,
    VX_ENUM_OPERATOR_TYPE_INTEL = 0xC4,
    VX_ENUM_HAAR_INTEL = 0xC5,
    VX_ENUM_COMPARE_TYPE_INTEL = 0xC6,
    VX_ENUM_HOUGH_LINES_P_INTEL = 0xC7,
    VX_ENUM_NORM_ALL_TYPE_INTEL = 0xC8,
    VX_ENUM_MORPH_TYPE_INTEL = 0xC9,
    VX_ENUM_MASK_SIZE_INTEL = 0xCA,
    VX_ENUM_FILTER_TYPE_INTEL = 0xCB,
    VX_ENUM_MATCH_TEMPLATE_ALG_INTEL = 0xCD,
    VX_ENUM_TIMESCALE_TYPE_INTEL = 0xCE,
    VX_ENUM_MATH_MODE_INTEL = 0xCF,
    VX_ENUM_SERIAL_TYPE_INTEL = 0xD0,
    VX_ENUM_SVM_MODEL_TYPE_INTEL = 0xD1,
    VX_ENUM_CLASSIFIER_FEATURE_TYPE_INTEL = 0xD2,
    VX_ENUM_CASCADE_MODEL_TYPE_INTEL = 0xD3,
    VX_ENUM_HOMOGRAPHY_TYPE_INTEL = 0xD4,
    VX_ENUM_LBP_FORMAT_TYPE_INTEL = 0xD5,
    VX_ENUM_SYMMNXN_SIZE_INTEL = 0xD6,
};

typedef struct _vx_deconv_params_intel_t
{
    vx_uint32 stride_x;
    vx_uint32 stride_y;
    vx_uint32 pad_x;
    vx_uint32 pad_y;
} vx_deconv_params_intel_t;

enum vx_cascade_model_format_intel_e
{
    VX_CASCADE_MODEL_OPENCV_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_CASCADE_MODEL_TYPE_INTEL) + 0x0
};

enum vx_homography_type_intel_e
{
    VX_HOMOGRAPHY_AFFINE_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_HOMOGRAPHY_TYPE_INTEL) + 0x1
};

enum vx_lbp_format_intel_t
{
    VX_LBP_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_LBP_FORMAT_TYPE_INTEL) + 0x0,
    VX_MLBP_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_LBP_FORMAT_TYPE_INTEL) + 0x1,
    VX_ULBP_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_LBP_FORMAT_TYPE_INTEL) + 0x2
};

enum vx_memory_type_intel_e
{
    VX_MEMORY_TYPE_DEVICE_MAPPED_HANDLE_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MEMORY_TYPE) + 0x0,
    VX_MEMORY_TYPE_EXTERNALLY_MAPPED_HANDLE_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MEMORY_TYPE) + 0x1
};

enum vx_ransac_line_intel_e
{
    VX_RANSAC_FIT_LINE_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_RANSAC_LINE_TYPE_INTEL) + 0x0,
    VX_RANSAC_FIT_PARABOLA_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_RANSAC_LINE_TYPE_INTEL) + 0x1
};

enum vx_warp_lens_projection_intel_e
{
    VX_WARP_LENS_PROJECTION_RECTILINEAR_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_WARP_LENS_TYPE_INTEL) + 0x0,
    VX_WARP_LENS_PROJECTION_FISHEYE_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_WARP_LENS_TYPE_INTEL) + 0x1
};

enum vx_warp_output_projection_intel_e
{
    VX_WARP_OUTPUT_PROJECTION_PLANAR_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_WARP_OUTPUT_TYPE_INTEL) + 0x0,
    VX_WARP_OUTPUT_PROJECTION_CYLINDRICAL_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_WARP_OUTPUT_TYPE_INTEL) + 0x1,
    VX_WARP_OUTPUT_PROJECTION_SPHERICAL_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_WARP_OUTPUT_TYPE_INTEL) + 0x2
};

enum vx_symmnxn_size_intel_e
{
    VX_SYMM3X3_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_SYMMNXN_SIZE_INTEL) + 0x0,
    VX_SYMM5X5_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_SYMMNXN_SIZE_INTEL) + 0x1,
    VX_SYMM7X7_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_SYMMNXN_SIZE_INTEL) + 0x2,
    VX_SYMM9X9_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_SYMMNXN_SIZE_INTEL) + 0x3
};

typedef struct _vx_gemm_params_intel_t
{
    vx_uint8 fixed_point_pos;
    vx_uint8 accumulator_bits;
    vx_bool transpose_input1;
    vx_bool transpose_input2;
    vx_bool transpose_input3;
} vx_gemm_params_intel_t;

typedef struct _vx_haar_weak_classifier_intel_t
{
    vx_rectangle_t rect[3];
    vx_int32 weight[3];
    vx_float32 threshold;
    vx_float32 left;
    vx_float32 right;
} vx_haar_weak_classifier_intel_t;

typedef struct _vx_hog_intel_t
{
    vx_int32 cell_size;
    vx_int32 block_size;
    vx_int32 block_stide;
    vx_int32 num_bins;
    vx_int32 window_size;
    vx_int32 window_stride;
    vx_float32 threshold;
} vx_hog_intel_t;

typedef struct _vx_hough_lines_p_intel_t
{
    vx_float32 rho;
    vx_float32 theta;
    vx_int32 threshold;
    vx_int32 line_gap;
    vx_int32 line_length;
    vx_float32 theta_max;
    vx_float32 theta_min;
} vx_hough_lines_p_intel_t;

typedef struct _vx_lbp_weak_classifier_intel_t
{
    vx_rectangle_t rect;
    vx_int32 lut[8];
    vx_float32 left;
    vx_float32 right;
} vx_lbp_weak_classifier_intel_t;

typedef struct _vx_line2d_intel_t
{
    vx_float32 start_x;
    vx_float32 start_y;
    vx_float32 end_x;
    vx_float32 end_y;
} vx_line2d_intel_t;

typedef struct _vx_ransac_homography_parameters_intel_t
{
    /*! \brief homography type <tt>\ref vx_homography_type_intel_e</tt>. */
    vx_enum homographyType;
    /*! \brief Number of iterations for RANSAC method. */
    vx_int32 numIterations;
    /*! \brief The threshold of the distances between points and the fitting transform. */
    vx_int32 threshDist;
    /*! \brief The minimum number of samples required to fit the model. */
    vx_int32 threshNumInliers;
    /*! \brief Seed for the random numbers generator. */
    vx_int64 rngSeed;
} vx_ransac_homography_parameters_intel_t;

typedef struct _vx_ransacline_parameters_line_t
{
    vx_enum fitFunc;
    vx_int32 minPoints;
    vx_int32 numIterations;
    vx_int32 threshDist;
    vx_int32 threshNumSamples;
} vx_ransacline_parameters_line_t;

typedef struct _vx_warp_intel_t
{
    vx_float32 K[3][3];
    vx_float32 R[4][4];
    vx_float32 Distortion[8];
    vx_enum lens_projection;
    vx_enum output_projection;
    vx_enum interpolation;
} vx_warp_intel_t;


typedef struct _vx_parameter_struct_intel_t *vx_parameter_struct_intel;



enum vx_classifier_feature_type_intel_e
{
    VX_CLASSIFIER_HAAR_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_CLASSIFIER_FEATURE_TYPE_INTEL) + 0x1,
    VX_CLASSIFIER_LBP_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_CLASSIFIER_FEATURE_TYPE_INTEL) + 0x2,
    VX_CLASSIFIER_MLBP_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_CLASSIFIER_FEATURE_TYPE_INTEL) + 0x3,
    VX_CLASSIFIER_MCT_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_CLASSIFIER_FEATURE_TYPE_INTEL) + 0x4,
};

enum vx_svm_model_format_intel_e
{
    VX_SVM_MODEL_LIBSVM_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_SVM_MODEL_TYPE_INTEL) + 0X0
};

/*! \brief The serial mode extensions
 * \note Indicate that the kernel must be executed in a serial fashion, individual tiles must be executed in order
 * \ingroup group_tiling
 */
enum vx_serial_type_intel_e {
    /*! \brief Serial left-right, top-bottom
     *  \ref VX_NODE_ATTRIBUTE_SERIAL_TYPE.
     */
    VX_SERIAL_LEFT_TO_RIGHT_TOP_TO_BOTTOM_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_SERIAL_TYPE_INTEL) + 0x0,
   /*! \brief Serial upper-left to lower-right
    *  \ref VX_NODE_ATTRIBUTE_SERIAL_TYPE.
    */
   VX_SERIAL_LEFTTOP_TO_RIGHTBOTTOM_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_SERIAL_TYPE_INTEL) + 0x1,
   /*! \brief No restriction
    *  \ref VX_NODE_ATTRIBUTE_SERIAL_TYPE.
    */
   VX_SERIAL_UNDEFINED_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_SERIAL_TYPE_INTEL) + 0x2,
};

/*! \brief The User Kernel Advanced Tiling Attributes.
 * \ingroup group_tiling
 */
enum vx_kernel_attribute_tiling_intel_e {
    /*! \brief This allows a tiling mode kernel to set its serial type. */
    VX_KERNEL_SERIAL_TYPE_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_KERNEL) + 0x1,
    /*! \brief This allows to execute kernel in in-place mode. */
    VX_KERNEL_INPLACE_KERNEL_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_KERNEL) + 0x2,
    /*! \brief Specify which input parameter is associated to output parameter for in-place mode. */
    VX_KERNEL_INPLACE_PARAMS_FUNC_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_KERNEL) + 0x3,
    /*! \brief This allows to execute kernel in blocking mode. */
    VX_KERNEL_BLOCKING_MODE_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_KERNEL) + 0x4,
    /*! \brief This allows a tiling mode kernel to set its input neighborhood. */
    VX_KERNEL_INPUT_NEIGHBORHOOD_INTEL      = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_KERNEL) + 0x5,
    /*! \brief This allows a tiling mode kernel to set its output tile block size. */
    VX_KERNEL_OUTPUT_TILE_BLOCK_SIZE_INTEL  = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_KERNEL) + 0x6,
    /*! \brief This allows the author to set the border mode on the tiling kernel. */
    VX_KERNEL_BORDER_INTEL                  = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_KERNEL) + 0x7,
    /*! \brief This determines the per tile memory allocation. */
    VX_KERNEL_TILE_MEMORY_SIZE_INTEL        = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_KERNEL) + 0x8,
};

/*! \brief The User Node Advanced Tiling Attributes.
 * \ingroup group_tiling
 */
enum vx_node_attribute_tiling_intel_e {
    /*! \brief This allows a tiling mode node to set its serial type. */
    VX_NODE_SERIAL_TYPE_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_NODE) + 0x1,
    /*! \brief This allows to execute node in in-place mode. */
    VX_NODE_INPLACE_KERNEL_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_NODE) + 0x2,
    /*! \brief Specify wich input paramenter is associated to output parameter for in-place mode. */
    VX_NODE_INPLACE_PARAMS_FUNC_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_NODE) + 0x3,
    /*! \brief This allows to execute node in blocking mode. */
    VX_NODE_BLOCKING_MODE_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_NODE) + 0x4,
    /*! \brief Can be used by some nodes to implement tiling. */
    VX_EXT_NODE_ADDITIONAL_TILE_NEIGHBORHOOD_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_NODE) + 0x5,
    /*! \brief This allows a tiling mode node to get its input neighborhood. */
    VX_NODE_INPUT_NEIGHBORHOOD_INTEL      = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_NODE) + 0x6,
    /*! \brief This allows a tiling mode node to get its output tile block size. */
    VX_NODE_OUTPUT_TILE_BLOCK_SIZE_INTEL  = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_NODE) + 0x7,
    /*! \brief This is the size of the tile local memory area. */
    VX_NODE_TILE_MEMORY_SIZE_INTEL        = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_NODE) + 0x8,
};

/*! \file
 * \brief The OpenVX Target API Definition
 */

/*! \brief The extension name.
 * \ingroup group_target
 */
#define OPENVX_EXT_TARGET_INTEL "vx_ext_target"

/*! \brief Defines the maximum number of characters in a target string.
 * \ingroup group_target
 */
#define VX_MAX_TARGET_NAME_INTEL (64)

enum vx_ext_target_context_attribute_intel_e {
    /*! \brief Used to query the context for the number of active targets. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_CONTEXT_TARGETS_INTEL= VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_CONTEXT) + 0xE,
};

/*! \brief An abstract handle to a target.
 * \ingroup group_target
 */
typedef struct _vx_target_intel *vx_target_intel;

/*! \brief The target attributes list
 * \ingroup group_target
 */
enum vx_target_attribute_intel_e {
    /*! \brief Returns the index of the given target. Use a <tt>\ref vx_uint32</tt> parameter.*/
    VX_TARGET_ATTRIBUTE_INDEX_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_TARGET_INTEL) + 0x0,
    /*! \brief Returns the name of the given target in the format "vendor.vendor_string".
     * Use a <tt>\ref vx_char</tt>[<tt>\ref VX_MAX_TARGET_NAME_INTEL</tt>] array
     */
    VX_TARGET_ATTRIBUTE_NAME_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_TARGET_INTEL) + 0x1,
    /*! \brief Returns the number of kernels that the target is capable of processing.
     * This is then used to allocate a table which is then filled when <tt>\ref vxQueryTargetIntel</tt>
     * is called with <tt>\ref VX_TARGET_ATTRIBUTE_KERNELTABLE</tt>.
     * Use a <tt>\ref vx_uint32</tt> parameter.
     */
    VX_TARGET_ATTRIBUTE_NUMKERNELS_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_TARGET_INTEL) + 0x2,
    /*! \brief Returns the table of all the kernels that a given target can execute.
     *  Use a <tt>vx_kernel_info_t</tt> array.
     * \pre You must call <tt>\ref vxQueryTargetIntel</tt> with <tt>\ref VX_TARGET_ATTRIBUTE_NUMKERNELS</tt>
     * to compute the necessary size of the array.
     */
    VX_TARGET_ATTRIBUTE_KERNELTABLE_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_TARGET_INTEL) + 0x3,
};


/*!
 * \brief The Intel Target Enumeration.
 */
enum vx_target_intel_e {
    /*! \brief intel.cpu target */
    VX_TARGET_CPU_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_TARGET) + 0x0,
    /*! \brief intel.gpu target */
    VX_TARGET_GPU_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_TARGET) + 0x1,
    /*! \brief intel.ipu target */
    VX_TARGET_IPU_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_TARGET) + 0x2,
};

/*! \brief The Background Subtractor State Object. This contains
 * pointer to the structure needed for working the
 * background subtractor.
 * \extends vx_reference
 * \ingroup group_bg_state
 */
typedef struct _vx_bg_state_intel *vx_bg_state_intel;

/*! \brief The set of working parameters for SVM classifier.
 * \ingroup group_svm_params
 */
typedef struct _vx_svm_params_intel *vx_svm_params_intel;
/*! \brief The SepFilter2D Object. A user-defined filter with kernels and divisors for rows and columns.
 * \extends vx_reference
 * \ingroup group_sepfilter2d
 */
typedef struct _vx_sepfilter2d_intel_t *vx_sepfilter2d_intel;

/*! \brief The set of working parameters for device kernel library.
 */
typedef struct _vx_device_kernel_library_intel* vx_device_kernel_library_intel;

typedef int16_t vx_float16_intel;

enum
{
   VX_TYPE_COORDINATES4D_INTEL        = VX_TYPE_VENDOR_STRUCT_START + 0x1,/*!< \brief A <tt>\ref vx_coordinates4d_t</tt>. */
   VX_TYPE_COORDINATES_POLAR_INTEL    = VX_TYPE_VENDOR_STRUCT_START + 0x2,/*!< \brief A <tt>\ref vx_coordinates_polar_t</tt>. */
   VX_TYPE_HAAR_WEAK_CLASSIFIER_INTEL = VX_TYPE_VENDOR_STRUCT_START + 0x3,/*!< \brief A <tt>\ref vx_haar_weak_classifier_t</tt>. */
   VX_TYPE_LBP_WEAK_CLASSIFIER_INTEL  = VX_TYPE_VENDOR_STRUCT_START + 0x4,/*!< \brief A <tt>\ref vx_lbp_weak_classifier_t</tt>. */
   VX_TYPE_FLOAT16_INTEL              = VX_TYPE_VENDOR_STRUCT_START + 0x5,/*!< \brief A <tt>\ref vx_float16_intel</tt>. */
};

enum vx_sepfilter2d_attribute_intel_e
{
   /*! \brief The size of kernel for filtering rows. Use a <tt>\ref vx_size</tt> parameter. */
   VX_SEPFILTER2D_ROW_KERNEL_SIZE_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_SEPFILTER2D_INTEL) + 0x0,
   /*! \brief The size of kernel for filtering columns. Use a <tt>\ref vx_size</tt> parameter. */
   VX_SEPFILTER2D_COLUMN_KERNEL_SIZE_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_SEPFILTER2D_INTEL) + 0x1,
   /*! \brief Divisor for the row kernel. Use a <tt>\ref vx_int32</tt> parameter. */
   VX_SEPFILTER2D_ROW_DIVISOR_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_SEPFILTER2D_INTEL) + 0x2,
   /*! \brief Divisor for the column kernel. Use a <tt>\ref vx_int32</tt> parameter. */
   VX_SEPFILTER2D_COLUMN_DIVISOR_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_SEPFILTER2D_INTEL) + 0x3
};

/*!
 * \brief Additional CNN activation functions supported by Intel as vendor extension.
 */
enum vx_df_nn_activation_function_intel_e
{
   VX_NN_ACTIVATION_RELU_NEGATIVE_SLOPE_INTEL =  VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_NN_ACTIVATION_FUNCTION) + 0x0,
};

enum  {
   /*! \brief Output values are defined by triliniar interpolation between the pixels whose centers are closest
     * to the sample position, weighted linearly by the distance of the sample from the pixel centers. */
   VX_INTERPOLATION_TRILINEAR_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_INTERPOLATION) + 0x0,
   /*! \brief Output values are defined by tetrahedral interpolation */
   VX_INTERPOLATION_TETRAHEDRAL_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_INTERPOLATION) + 0x1,
   /*! \brief Output values are defined by bicubic interpolation */
   VX_INTERPOLATION_BICUBIC_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_INTERPOLATION) + 0x2,
};

enum {
    VX_DF_IMAGE_F32_INTEL  = VX_DF_IMAGE('F', '0', '3', '2'),
      /*! \brief A single plane of float 64-bit data.  */
    VX_DF_IMAGE_F64_INTEL  = VX_DF_IMAGE('F', '0', '6', '4'),
};

typedef struct _vx_md_data_intel_t *vx_md_data_intel;

enum vx_mddata_attribute_intel_e {
    /*! \brief Queries an MD Data for its number of dimensions. Read only. Use a <tt>\ref vx_uint32</tt> parameter. */
   VX_MDDATA_NUM_OF_DIMS_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_TENSOR) + 0x0,
    /*! \brief Queries an MD Data for its dimensions. Read only. Use a <tt>\ref vx_size</tt>[<tt>\ref VX_MAX_TENSOR_DIMS_INTEL</tt>] array. */
   VX_MDDATA_DIMS_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_TENSOR) + 0x1,
    /*! \brief Queries an MD Data for its format. Read only. Use a <tt>\ref vx_enum</tt> parameter. */
   VX_MDDATA_DATA_FORMAT_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_TENSOR) + 0x2,
};

/*! \brief Device kernel attributes list.
*/
enum vx_device_kernel_attribute_intel_e {
    VX_OPENCL_WORK_ITEM_XSIZE_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_DEVICE_LIBRARY_INTEL) + 0x0,
    VX_OPENCL_WORK_ITEM_YSIZE_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_DEVICE_LIBRARY_INTEL) + 0x1,
};

enum vx_device_library_type_intel_e
{
    VX_OPENCL_LIBRARY_SOURCE_INTEL = 0,
    VX_OPENCL_LIBRARY_BINARY_INTEL,
    VX_OPENCL_LIBRARY_SPIRV_INTEL,
};

enum vx_context_attribute_ext_intel_e
{
    VX_CONTEXT_MEM_OPTIMIZATION_LEVEL_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_CONTEXT) + 0x0,
    VX_CONTEXT_ALLOCATOR_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_CONTEXT) + 0x1,
    VX_CONTEXT_POOL_THREAD_COUNT_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL,VX_TYPE_CONTEXT) + 0x2,
    VX_CONTEXT_MAX_POOL_THREAD_COUNT_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_CONTEXT) + 0x3,
    VX_CONTEXT_POOL_THREAD_PARAMS_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL,VX_TYPE_CONTEXT) + 0x4,
};

typedef struct
{
    void* opaque;
    void* (*alloc_ptr)(void* opaque, vx_size size);
    void (*free_ptr)(void* opaque, void* data);
} vx_allocator_intel_t;

typedef struct
{
    vx_int32 threadId;
    vx_int32 schedPolicy;
    vx_int32 schedPriority;
    vx_uint32 affinity;
    const char* name;
} vx_thread_params_intel_t;

enum vx_internal_node_attribute_intel_e
{
    /*! \brief Queries node unique id in the graph. Read-only. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_NODE_ID_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_NODE) + 0x9,

    /*! \brief Queries node math mode: fast, strict, or balanced.
     Use a <tt>\ref vx_math_mode_e</tt> parameter. */
    VX_NODE_MATH_MODE_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_NODE) + 0xA,

    /*! \brief Provides a hint to the runtime that input parameters set for this
     *   node won't be changed post-vxVerifyGraph(), which may allow further optimization
     *   opportunities.
     *  Use a <tt>\ref vx_bool</tt> parameter. Default is vx_false_e.
     */
    VX_NODE_ENABLE_STATIC_PARAMETER_OPTIMIZATION = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_NODE) + 0xB,
};

enum vx_graph_attribute_ext_intel_e
{
    VX_GRAPH_MEM_OPTIMIZATION_LEVEL_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_GRAPH) + 0x0,
    VX_GRAPH_ALLOCATOR_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_GRAPH) + 0x1,
    VX_GRAPH_TILE_WIDTH_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_GRAPH) + 0x3,
    VX_GRAPH_TILE_HEIGHT_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_GRAPH) + 0x4,
    VX_GRAPH_PRIORITY_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_GRAPH) + 0x5,
    VX_GRAPH_PROFILE_MEMORY_SIZE_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_GRAPH) + 0x6,
    VX_GRAPH_PROFILE_BASIC_INFO_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_GRAPH) + 0x7,
    VX_GRAPH_PROFILE_NODES_INFO_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_GRAPH) + 0x8,
    VX_GRAPH_PROFILE_TILES_INFO_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_GRAPH) + 0x9,
    VX_GRAPH_DISABLE_PROFILING_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_GRAPH) + 0xA,
    VX_GRAPH_TILE_POOL_USE_CUSTOM_ALLOCATOR_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_GRAPH) + 0xD,
    VX_GRAPH_TILE_POOL_PROFILE_MEMORY_INTEL = VX_ATTRIBUTE_BASE(VX_ID_INTEL, VX_TYPE_GRAPH) + 0xE,
};

/*!
 * \brief These enumerations are given to the \c vxDirective API to enable/disable
 * platform optimizations and/or features.
 */
enum vx_directive_ext_intel_e {
    /*! \brief Disables nodes fusion in provided graph. */
    VX_DIRECTIVE_DISABLE_FUSION_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_DIRECTIVE) + 0x0,
    /*! \brief Enables nodes fusion in provided graph. */
    VX_DIRECTIVE_ENABLE_FUSION_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_DIRECTIVE) + 0x1,
};

enum vx_ext_timescale_intel_e
{
    /*! \brief. */
    NANOSEC_INTEL,
    MILISEC_INTEL
};

typedef enum vx_device_intel_e
{
    CPU_INTEL,
    GPU_INTEL,
    IPU_INTEL
} vx_device_intel_e;

/*! \brief Parameters for RANSAC line kernel.
 */
typedef struct _vx_ransacline_parameters_intel_t
{
    /*! \brief Curve type <tt>\ref vx_ransac_fit_intel_e</tt>. */
    vx_enum fitFunc;
    /*! \brief Number of iterations for RANSAC method. */
    vx_int32 numIterations;
    /*! \brief The threshold of the distances between points and the fitted curve. */
    vx_int32 threshDist;
    /*! \brief The minimum number of samples required to fit the model. */
    vx_int32 threshNumInliers;
    /*! \brief Seed for the random numbers generator. */
    vx_int64 rngSeed;
} vx_ransacline_parameters_intel_t;


typedef struct
{
    vx_int32 cols;
    vx_int32 rows;
    vx_char name[VX_MAX_KERNEL_NAME];
    vx_int32 id;
    vx_bool externalOutput;
    vx_device_intel_e deviceType;
} vx_nodeInfo_intel_t;

typedef struct
{
    vx_int32 index;
    vx_int32 tileY;
    vx_int32 tileX;
} vx_nodeId_intel_t;

typedef struct
{
    vx_int32 threadID;
    vx_int32 lineIndex;
    vx_int32 execBlockIndex;
    vx_uint64 lock;
    vx_uint64 begin;
    vx_uint64 end;
    vx_nodeId_intel_t nodeId;
    vx_int32 rectX;
    vx_int32 rectY;
    vx_int32 rectWidth;
    vx_int32 rectHeight;
    vx_int64 lockTime;
    vx_int64 startTime;
    vx_int64 endTime;
    vx_int64 duration;
} vx_hostExecLineInfo_intel_t;

typedef struct
{
    int nodeIndex;
    int partIndex;
    int execLineIndex;
    int rectX;
    int rectY;
    int rectWidth;
    int rectHeight;
    vx_int64 lockTime;
    vx_int64 startTime;
    vx_int64 endTime;
    vx_int64 duration;
} vx_hwPartInfo_intel_t;

typedef struct
{
    vx_device_intel_e deviceType;
    union device
    {
        vx_hostExecLineInfo_intel_t host;
        vx_hwPartInfo_intel_t hw;
    } device;
} vx_execLineInfo_intel_t;

typedef struct
{
    vx_int32 numThreads;
    vx_int32 numNodes;
    vx_int32 numExecLines;
    vx_enum timescale;
    vx_int64 baseTime;
} vx_profileInfo_intel_t;


enum vx_enum_morph_intel_e
{
    VX_MORPH_OPEN_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MORPH_TYPE_INTEL) + 0x1,
    VX_MORPH_CLOSE_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MORPH_TYPE_INTEL) + 0x2,
    VX_MORPH_GRADIENT_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MORPH_TYPE_INTEL) + 0x3,
    VX_MORPH_TOPHAT_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MORPH_TYPE_INTEL) + 0x4,
    VX_MORPH_BLACKHAT_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MORPH_TYPE_INTEL) + 0x5,
};

enum vx_math_mode_intel_e
{
    VX_MATH_MODE_FAST_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MATH_MODE_INTEL) + 0x1,
    VX_MATH_MODE_STRICT_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MATH_MODE_INTEL) + 0x2,
    VX_MATH_MODE_BALANCED_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MATH_MODE_INTEL) + 0x3,
};

enum vx_alpha_composition_intel_e
{
    VX_ALPHA_OVER_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_ALPHA_TYPE_INTEL) + 0x1,
    VX_ALPHA_IN_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_ALPHA_TYPE_INTEL) + 0x2,
    VX_ALPHA_OUT_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_ALPHA_TYPE_INTEL) + 0x3,
    VX_ALPHA_ATOP_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_ALPHA_TYPE_INTEL) + 0x4,
    VX_ALPHA_XOR_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_ALPHA_TYPE_INTEL) + 0x5,
    VX_ALPHA_PLUS_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_ALPHA_TYPE_INTEL) + 0x6
};

enum vx_intel_round_policy_intel_e
{
    VX_ROUND_POLICY_FRACTIONAL_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_ROUND_INTEL) + 0x1
};

typedef struct
{
    vx_float32 rho;
    vx_float32 theta;
} vx_coordinates_polar_intel_t;

enum vx_fft_norm_intel_e
{
    VX_FFT_DIV_FWD_BY_N_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_FFT_NORM_TYPE_INTEL) + 0x1,
    VX_FFT_DIV_INV_BY_N_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_FFT_NORM_TYPE_INTEL) + 0x2,
    VX_FFT_DIV_BY_SQRTN_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_FFT_NORM_TYPE_INTEL) + 0x3,
    VX_FFT_NODIV_BY_ANY_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_FFT_NORM_TYPE_INTEL) + 0x4
};

typedef enum vx_fft_norm_intel_e vx_fft_normalization_intel;

enum vx_mirror_intel_e
{
    VX_MIRROR_HORIZONTAL_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MIRROR_TYPE_INTEL) + 0x1,
    VX_MIRROR_VERTICAL_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MIRROR_TYPE_INTEL) + 0x2,
    VX_MIRROR_BOTH_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MIRROR_TYPE_INTEL) + 0x3,
    VX_MIRROR_45_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MIRROR_TYPE_INTEL) + 0x4,
    VX_MIRROR_135_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MIRROR_TYPE_INTEL) + 0x5
};

enum vx_haar_intel_e
{
    VX_HAAR_FROM_FILE_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_HAAR_INTEL) + 0x1,
    VX_HAAR_FROM_STRING_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_HAAR_INTEL) + 0x2,
};

typedef enum _vx_operator_intel_e_
{
    VX_OPERATOR_PREWITT_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_OPERATOR_TYPE_INTEL) + 0x1,
    VX_OPERATOR_SCHARR_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_OPERATOR_TYPE_INTEL) + 0x2,
    VX_OPERATOR_SOBEL_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_OPERATOR_TYPE_INTEL) + 0x3
} vx_operator_intel_e;

enum vx_compare_intel_e
{
    VX_COMPARE_LESS_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_COMPARE_TYPE_INTEL) + 0x1,
    VX_COMPARE_LESS_EQ_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_COMPARE_TYPE_INTEL) + 0x2,
    VX_COMPARE_EQ_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_COMPARE_TYPE_INTEL) + 0x3,
    VX_COMPARE_GREATER_EQ_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_COMPARE_TYPE_INTEL) + 0x4,
    VX_COMPARE_GREATER_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_COMPARE_TYPE_INTEL) + 0x5,
    VX_COMPARE_NOT_EQ_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_COMPARE_TYPE_INTEL) + 0x6
};

enum vx_norm_intel_e
{
    VX_NORM_INF_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_NORM_ALL_TYPE_INTEL) + 0x1,
    VX_NORM_RELATIVE_L1_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_NORM_ALL_TYPE_INTEL) + 0x2,
    VX_NORM_RELATIVE_L2_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_NORM_ALL_TYPE_INTEL) + 0x3,
    VX_NORM_RELATIVE_INF_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_NORM_ALL_TYPE_INTEL) + 0x4,
    VX_NORM_HAMMING_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_NORM_TYPE) + 0x5
};

enum vx_adaptive_threshold_mask_intel_e
{
    VX_MASK_SIZE_3X3_INTEL, VX_MASK_SIZE_5X5_INTEL
};

/*! \brief Creates an opaque reference to a vx_tensor buffer with data from a given image.
* Dimensions will be determined by image data format. Interleaved images will be converted to planar.
* \param [in] image The image from which to create the md data.
* \return A md data reference or zero when an error is encountered.
* \see vxMapTensorPatchIntel to obtain direct memory access to the tensor
* \ingroup group_intel_mddata
*/
VX_API_ENTRY vx_tensor VX_API_CALL vxCreateMDDataFromImageIntel(vx_image image);


/*! \brief Retrieves the valid region of the vx_tensor as a view.
* \param [in] tensor The md data from which to retrieve the valid region.
* \param [out] view The destination view.
* \return vx_status
* \retval VX_ERROR_INVALID_REFERENCE Invalid md data.
* \retval VX_ERROR_INVALID_PARAMETERS Invalid view.
* \retval VX_STATUS Valid md data.
* \note This view can be passed directly to <tt>\ref vxMapTensorPatchIntel</tt> to get
* the full valid region of the md data.
* \ingroup group_intel_mddata
*/
VX_API_ENTRY vx_status VX_API_CALL vxGetValidRegionTensorIntel(vx_tensor tensor, vx_size *view_start, vx_size *view_end);

/*! \brief Allows the application to get direct access to a patch of an vx_tensor object
* \param [in] tensor The reference to the md data from which to extract the patch.
* \param [in] ndims
* \param [in] view_start The coordinates to which to set the patch.
* \param [in] view_end The coordinates of the end of the patch
* \param [out] map_id The address of a vx_map_id variable where the function returns a map identifier
* \param [out] addr The addressing information for the tensor
* \param [out] ptr The pointer of a location from which to read the data.
* \param [in] usage <tt>\ref vx_accessor_e</tt> enumeration
* \param [in] memory_type The only supported value is VX_MEMORY_TYPE_HOST
* \param [in] flags flags from <tt>\ref vx_map_flag_e</tt> enumeration
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \ingroup group_intel_mddata
*/
VX_API_ENTRY vx_status VX_API_CALL vxMapTensorPatchIntel(vx_tensor tensor,
                                vx_size ndims,
                                const vx_size *view_start, const vx_size *view_end,
                                vx_map_id *map_id,
                                vx_mdview_addressing_intel_t *addr,
                                void **ptr,
                                vx_enum usage,
                                vx_enum memory_type,
                                vx_uint32 flags);

/*! \brief Unmap and commit potential changes to a tensor object patch that were previously mapped.
* \param [in] tensor The reference to the tensor from which patch were extracted.
* \param [in] map_id The unique map identifier that was returned by vxMapTensorPatchIntel
*/
VX_API_ENTRY vx_status VX_API_CALL vxUnmapTensorPatchIntel(vx_tensor tensor, vx_map_id map_id);


enum vx_match_template_alg_intel_e
{
    VX_TM_SQDIFF_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MATCH_TEMPLATE_ALG_INTEL) + 0x1,
    VX_TM_CRCORR_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MATCH_TEMPLATE_ALG_INTEL) + 0x2,
    VX_TM_CRCOEFF_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MATCH_TEMPLATE_ALG_INTEL) + 0x3,
};

enum vx_match_template_impl_alg_intel_e
{
    VX_MATCH_TEMPLATE_DIRECT_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MATCH_TEMPLATE_ALG_INTEL) + 0xA,
    VX_MATCH_TEMPLATE_FFT_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_MATCH_TEMPLATE_ALG_INTEL) + 0xB,
};

/*enum vx_sumsqdiff_alg_intel_e
{
    VX_SUMSQDIFF_DIRECT_INTEL = VX_MATCH_TEMPLATE_DIRECT_INTEL,
    VX_SUMSQDIFF_FFT_INTEL = VX_MATCH_TEMPLATE_FFT_INTEL,
};*/

enum vx_debug_zone_intel_e
{
    VX_ZONE_ERROR_INTEL = 0, /*!< Used for most errors */
    VX_ZONE_WARNING_INTEL = 1, /*!< Used to warning developers of possible issues */
    VX_ZONE_API_INTEL = 2, /*!< Used to trace API calls and return values */
    VX_ZONE_INFO_INTEL = 3, /*!< Used to show run-time processing debug */

    VX_ZONE_PERF_INTEL = 4, /*!< Used to show performance information */
    VX_ZONE_CONTEXT_INTEL = 5,
    VX_ZONE_OSAL_INTEL = 6,
    VX_ZONE_REFERENCE_INTEL = 7,

    VX_ZONE_ARRAY_INTEL = 8,
    VX_ZONE_IMAGE_INTEL = 9,
    VX_ZONE_SCALAR_INTEL = 10,
    VX_ZONE_KERNEL_INTEL = 11,

    VX_ZONE_GRAPH_INTEL = 12,
    VX_ZONE_NODE_INTEL = 13,
    VX_ZONE_PARAMETER_INTEL = 14,
    VX_ZONE_DELAY_INTEL = 15,

    VX_ZONE_TARGET_INTEL = 16,
    VX_ZONE_LOG_INTEL = 17,
    VX_ZONE_SHIM_INTEL = 18,

    VX_ZONE_MAX_INTEL = 32
};

enum vx_svm_type_intel_e
{
    VX_SVM_C_SVC_INTEL, /*! \brief Classification mode */
    VX_SVM_NU_SVC_INTEL, /*! \brief Classification mode */
    VX_SVM_ONE_CLASS_INTEL, /*! \brief Single classification mode */
    VX_SVM_EPS_SVR_INTEL, /*! \brief Regression mode */
    VX_SVM_NU_SVR_INTEL /*! \brief Regression mode */
};

enum vx_svm_kernel_type_intel_e
{
    VX_SVM_LINEAR_KERNEL_INTEL,
    VX_SVM_POLY_KERNEL_INTEL,
    VX_SVM_RBF_KERNEL_INTEL,
    VX_SVM_SIGMOID_KERNEL_INTEL
};

#define VX_LIBRARY_PI_INTEL 0x02

/*! \brief The list of available Print Imaging extension kernels.
 */
enum vx_kernel_pi_intel_e
{
    VX_KERNEL_PI_HALFTONE_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_PI_INTEL) + 0x0,
    VX_KERNEL_PI_ERROR_DIFFUSION_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_PI_INTEL) + 0x1,
    VX_KERNEL_PI_EDCMYK_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_PI_INTEL) + 0x2,
    VX_KERNEL_PI_LUT3D_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_PI_INTEL) + 0x3,
    VX_KERNEL_PI_TETRAHEDRAL_INTERPOLATION_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_PI_INTEL) + 0x4,
    VX_KERNEL_PI_EDCMYKPLANAR_INTEL = VX_KERNEL_BASE(VX_ID_INTEL, VX_LIBRARY_PI_INTEL) + 0x5,
};

/*! \brief The set of supported enumerations in Print Imaging extension.
 */
enum vx_enum_pi_intel_e
{
    VX_ENUM_DITHER_INTEL = 0xFE,
};

/*! \brief The 4D Coordinates structure. */
typedef struct _vx_coordinates4d_intel_t
{
    vx_uint32 a; /*!< \brief The X coordinate. */
    vx_uint32 b; /*!< \brief The Y coordinate. */
    vx_uint32 c; /*!< \brief The Z coordinate. */
    vx_uint32 d; /*!< \brief The W coordinate. */
} vx_coordinates4d_intel_t;

/*! \brief The dithering type enumeration for error diffusion.
 */
typedef enum _vx_dither_pi_intel_e
{
    /*! \brief */
    VX_ED_FS_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_DITHER_INTEL) + 0x0,
    /*! \brief */
    VX_ED_JNN_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_DITHER_INTEL) + 0x1,
    /*! \brief */
    VX_ED_STUCKI_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_DITHER_INTEL) + 0x2,
    /*! \brief */
    VX_ED_BAYER_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_DITHER_INTEL) + 0x3
} vx_dither_pi_intel_e;


/*! \brief Add a Device Kernel Library into an OpenVX context.
 * \param [in] context The OpenVX Context.
 * \param [in] source The source of the library, can pre-compiled or source.
 * \param [in] target_enum The vx_target_e the default target for nodes instantiated from this kernel.
 * \param [in] target_string The target name ASCII string. This contains a valid value when target_enum is set to VX_TARGET_STRING, otherwise it is ignored.
 * Returns a kernel library reference for the created library
 */
VX_API_ENTRY vx_device_kernel_library_intel VX_API_CALL vxAddDeviceLibraryIntel(
        vx_context context, vx_size length, const char* source, const char *flags,
        vx_enum library_type, const char* target_string);

/*! \brief Releases a reference to Device Kernel Library.
 * \param [in] library Pointer to the reference to Device Kernel Library.
 */
VX_API_ENTRY vx_status VX_API_CALL vxReleaseDeviceLibraryIntel(
        vx_device_kernel_library_intel* library);

/*! \brief Add a non-tiled Device Kernel into an OpenVX context.
 * \param [in] context The OpenVX Context.
 * \param [in] vx_name The name of the kernel in OpenVX nomenclature.
 * \param [in] enum The OpenVX kernel enumeration used to identify this kernel.
 * \param [in] library The Kernel library that contains the kernel.
 * \param [in] device_name The name of the kernel to call in the library.
 * \param [in] numParams The number of parameters to the OpenVX kernel.
 * \param [in] param_types An array of types of parameters to the OpenVX kernel.
 * \param [in] param_directions An array of directions of parameters to the OpenVX kernel.
 * \param [in] border_mode. Select the border mode supported by the kernel
 * \param [in] validate The input/output validator.
 * \param [in] initialize The initialization call-back.
 * \param [in] deinitialize The De-initialization call-back.
 */
VX_API_ENTRY vx_kernel VX_API_CALL vxAddDeviceKernelIntel(vx_context context,
        const vx_char vx_name[], vx_enum enummerator,
        vx_device_kernel_library_intel library,
        const vx_char device_name[VX_MAX_KERNEL_NAME], vx_uint32 numParams,
        vx_enum *param_types, vx_enum *param_direction, vx_enum border_mode,
        vx_kernel_validate_f validate,
        vx_kernel_initialize_f initialize,
        vx_kernel_deinitialize_f deinitialize);

/*! \brief Add a tiled Device Kernel into an OpenVX context.
 * \param [in] context The OpenVX Context.
 * \param [in] vx_name The name of the kernel in OpenVX nomenclature.
 * \param [in] enum The OpenVX kernel enumeration used to identify this kernel.
 * \param [in] library The Kernel library that contains the kernel.
 * \param [in] device_name The name of the kernel to call in the library.
 * \param [in] numParams The number of parameters to the OpenVX kernel.
 * \param [in] param_types An array of types of parameters to the OpenVX kernel.
 * \param [in] param_directions An array of directions of parameters to the OpenVX kernel.
 * \param [in] border_mode. Select the border mode supported by the kernel
 * \param [in] validate The input/output validator.
 * \param [in] initialize The initialization call-back.
 * \param [in] deinitialize The De-initialization call-back.
 */
VX_API_ENTRY vx_kernel VX_API_CALL vxAddDeviceTilingKernelIntel(
        vx_context context, vx_char vx_name[], vx_enum enummerator,
        vx_device_kernel_library_intel library,
        const vx_char device_name[VX_MAX_KERNEL_NAME], vx_uint32 numParams,
        vx_enum *param_types, vx_enum *param_direction, vx_enum border_mode,
        vx_kernel_validate_f validate,
        vx_kernel_initialize_f initialize,
        vx_kernel_deinitialize_f deinitialize);

// CNN defines

/*! \brief [Immediate] Fast color space conversion using tetrahedral interpolation.
 * \param [in] context The reference to the overall context.
 * \param [in] input1 The input image for 1st channel.
 * \param [in] input2 The input image for 2nd channel.
 * \param [in] input3 The input image for 3rd channel.
 * \param [in] map Maps source channels from (0..255) to (0..15) with 4.7 precision. Array element type <tt>\ref vx_coordinates3d_t</tt>.
 * \param [in] values Set of 4096 samples of output channels. Array element type <tt>\ref vx_coordinates4d_intel_t</tt>.
 * \param [out] output1 The output image for 1st channel.
 * \param [out] output2 The output image for 2nd channel.
 * \param [out] output3 The output image for 3rd channel.
 * \param [out] output4 The output image fir 4th channel.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */

VX_API_ENTRY vx_status VX_API_CALL vxuTetrahedralInterpolationIntel(vx_context context, vx_image input1, vx_image input2, vx_image input3,
                                                  vx_array map, vx_array values,
                                                  vx_image output1, vx_image output2, vx_image output3, vx_image output4);

/*! \brief [Graph] Performs forward or inverse Fast Fourier Transform to a 2^n x 2^m image.
 * \param [in] graph The reference to the graph.
 * \param [in] image The input image VX_DF_IMAGE_F32_INTEL.
 * \param [in] norm Normalization factor, <tt>\ref VX_TYPE_ENUM</tt> of the<tt>\ref vx_fft_norm_e</tt>
 * \param [in] is_inverse Forward FFT or inverse FFT.
 * \param [out] output The output image VX_DF_IMAGE_F32_INTEL.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxFFTImageNodeIntel(vx_graph graph,
        vx_image input, vx_enum norm, vx_bool is_inverse, vx_image output);
/*! \brief [Immediate] Performs forward or inverse Fast Fourier Transform to a 2^n x 2^m image.
 * \param [in] context The reference to the overall context.
 * \param [in] image The input image VX_DF_IMAGE_F32_INTEL.
 * \param [in] norm Normalization factor, <tt>\ref VX_TYPE_ENUM</tt> of the<tt>\ref vx_fft_norm_e</tt>
 * \param [in] is_inverse Forward FFT or inverse FFT.
 * \param [out] output The output image VX_DF_IMAGE_F32_INTEL.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuFFTImageIntel(vx_context context,
        vx_image input, vx_enum norm, vx_bool is_inverse, vx_image output);

/*! \brief [Graph] Creates a Rotate node.
 * \param [in] graph The reference to the graph.
 * \param [in] input An input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>, or <tt>\ref VX_DF_IMAGE_RGB</tt> format
 * \param [in] angle Angle in degrees to rotate image counterclockwise
 * \param [in] xShift Shift rotated image along horizontal axis
 * \param [in] yShift Shift rotated image along vertical axis
 * \param [in] interpolation Interpolation type. Supported values: <tt>\ref VX_INTERPOLATION_NEAREST_NEIGHBOR</tt> or <tt>\ref VX_INTERPOLATION_BILINEAR</tt>
 * \param [out] out The output image in the same format as the input image
 * \ingroup group_vision_function_rotate
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxRotateNodeIntel(vx_graph graph, vx_image input,
        vx_float64 angle, vx_float64 xShift, vx_float64 yShift,
        vx_enum interpolation, vx_image output);

/*! \brief [Immedate] Rotates the input image by the given angle.
 * \param [in] context The reference to the overall context.
 * \param [in] input An input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>, or <tt>\ref VX_DF_IMAGE_RGB</tt> format
 * \param [in] angle Angle in degrees to rotate image counterclockwise
 * \param [in] xShift Shift rotated image along horizontal axis
 * \param [in] yShift Shift rotated image along vertical axis
 * \param [in] interpolation Interpolation type. Supported values: <tt>\ref VX_INTERPOLATION_NEAREST_NEIGHBOR</tt> or <tt>\ref VX_INTERPOLATION_BILINEAR</tt>
 * \param [out] out The output image in the same format as the input image
 * \ingroup group_vision_function_rotate
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuRotateIntel(vx_context context,
        vx_image input, vx_float64 angle, vx_float64 xShift, vx_float64 yShift,
        vx_enum interpolation, vx_image output);

/*! \brief [Graph] Rotates input image counterclockwise to 90, 180, 270 degrees.
 * \param [in] graph The reference to the graph.
 * \param [in] image The input image.
 * \param [in] angle Angle (0, 90, 180, 270) in degrees to rotate image counterclockwise.
 * \param [out] output The output image.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxRotate90NodeIntel(vx_graph graph,
        vx_image input, vx_int32 angle, vx_image output);
/*! \brief [Immediate] Rotates input image counterclockwise to 90, 180, 270 degrees.
 * \param [in] context The reference to the overall context.
 * \param [in] image The input image.
 * \param [in] angle Angle (0, 90, 180, 270) in degrees to rotate image counterclockwise.
 * \param [out] output The output image.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuRotate90Intel(vx_context context,
        vx_image input, vx_int32 angle, vx_image output);



/*! \brief [Graph] Mirrors an image about the specified axis (axes).
 * \param [in] graph The reference to the graph.
 * \param [in] image The input image.
 * \param [in] axis Axis to mirror the image about <tt>\ref vx_mirror_e</tt>.
 * \param [out] output The output image.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxMirrorNodeIntel(vx_graph graph, vx_image input,
        vx_enum axis, vx_image output);
/*! \brief [Immediate] Mirrors an image about the specified axis (axes).
 * \param [in] context The reference to the overall context.
 * \param [in] image The input image.
 * \param [in] axis Axis to mirror the image about <tt>\ref vx_mirror_e</tt>.
 * \param [out] output The output image.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuMirrorIntel(vx_context context,
        vx_image input, vx_enum axis, vx_image output);

/*! \brief [Graph] Creates a node for calculating Otsu threshold
 * \param [in] graph The reference to the graph
 * \param [in] input The input image
 * \param [out] output The output image
 * \param [out] threshold Found threshold value (optional)
 * \return <tt>\ref vx_node</tt>
 * \retval 0 Node could not be created
 * \retval * Node handle
 */
VX_API_ENTRY vx_node VX_API_CALL vxOtsuNodeIntel(vx_graph graph, vx_image input,
        vx_image output, vx_threshold threshold);
/*! \brief [Immediate] Calculates Otsu threshold
 * \param [in] context The reference to the context
 * \param [in] input The input image
 * \param [out] output The output image
 * \param [out] threshold Found threshold value (optional)
 * \return A <tt>\ref vx_status_e</tt> enumeration
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>
 */
VX_API_ENTRY vx_status VX_API_CALL vxuOtsuIntel(vx_context context,
        vx_image input, vx_image output, vx_threshold threshold);

/*! \brief [Graph] Creates node for converting image pixels from one datatype to another.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>, <tt>\ref VX_DF_IMAGE_U32</tt>, or <tt>\ref VX_DF_IMAGE_S32</tt> format.
 * \param [in] round A <tt>\ref VX_TYPE_ENUM</tt> of the <tt>\ref vx_round_policy_e</tt> enumeration.
* \param [in] scale A multiplying coefficient for the output pixel values by 2**(-scale) to preserve precision or value range. Scale is supported for reducing bit depth (e.g. U32 to U8 or U16 to S16).
 * \param [out] output The output image in the desired format. The types of input and output images determines the conversion type.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxConvertNodeIntel(vx_graph graph,
        vx_image input, vx_image output, vx_enum round, vx_int32 scale);

/*! \brief [Immediate] Converts image pixel from one datatype to another.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>, <tt>\ref VX_DF_IMAGE_U32</tt>, or <tt>\ref VX_DF_IMAGE_S32</tt> format.
 * \param [in] round A <tt>\ref VX_TYPE_ENUM</tt> of the <tt>\ref vx_round_policy_e</tt> enumeration.
 * \param [in] scale A multiplying coefficient for the output pixel values by 2**(-scale) to preserve precision or value range. Scale is supported for reducing bit depth (e.g. U32 to U8 or U16 to S16).
 * \param [out] output The output image in the desired format. The types of input and output images determines the conversion type.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success.
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuConvertIntel(vx_context context,
        vx_image input, vx_image output, vx_enum round, vx_int32 scale);

/*! \brief [Immediate] Performs error diffusion on image.
 * \param [in] context The reference to the overall context.
 * \param [in] inputCMYK. The input CMYK image (VX_DF_IMAGE_RGBX)
 * \param [out] outputCMYK. The output CMYK image (VX_DF_IMAGE_RGBX)
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuErrorDiffusionCMYKIntel(vx_context context,
        vx_image inputCMYK, vx_image outputCMYK);

/*! \brief [Immediate] Performs error diffusion on image.
 * \param [in] context The reference to the overall context.
 * \param [in] inputCMYK. The input CMYK image (VX_DF_IMAGE_RGBX)
 * \param [out] outputC. The output CMYK image C channel (VX_DF_IMAGE_U8)
 * \param [out] outputM. The output CMYK image M channel (VX_DF_IMAGE_U8)
 * \param [out] outputY. The output CMYK image Y channel (VX_DF_IMAGE_U8)
 * \param [out] outputK. The output CMYK image K channel (VX_DF_IMAGE_U8)
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuErrorDiffusionCMYKPlanarIntel(vx_context context,
        vx_image inputCMYK, vx_image outputC, vx_image outputM, vx_image outputY, vx_image outputK);

/*! \brief [Graph] Fast symmetrical 7x7 filter
 * \param [in] graph The reference to the graph.
 * \param [in] input image. Only type supported is VX_DF_IMAGE_U8
 * \param [in] coefficients Array of 10 coefficients of vx_int32 type: \f$\left(\begin{array}{ccc}
 * A & B & C & D & E & F & G & H & I & J
 * \end{array}\right) \f$ that forms a symmetrical 7x7 filter \f$\left(\begin{array}{ccc}
 * J & I & H & G & H & I & J\\
 * I & F & E & D & E & F & I\\
 * H & E & C & B & C & E & H\\
 * G & D & B & A & B & D & G\\
 * H & E & C & B & C & E & H\\
 * I & F & E & D & E & F & I\\
 * J & I & H & G & H & I & J\\
 * \end{array}\right) \f$
 * \param [in] shift right value. Shifts the result of the filter adds / multiplies by this value.
 * \param [out] output image. Only type supported is VX_DF_IMAGE_U8
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxSymmetrical7x7FilterNodeIntel(vx_graph graph,
        vx_image input, vx_array coefficients, vx_int32 shift_right,
        vx_image output);

/*! \brief [Immediate] Performs Fast symmetrical 7x7 filter.
 * \param [in] context The reference to the overall context.
 * \param [in] input image. Only type supported is VX_DF_IMAGE_U8
 * \param [in] coefficients Array of 10 coefficients of vx_int32 type: \f$\left(\begin{array}{ccc}
 * A & B & C & D & E & F & G & H & I & J
 * \end{array}\right) \f$ that forms a symmetrical 7x7 filter \f$\left(\begin{array}{ccc}
 * J & I & H & G & H & I & J\\
 * I & F & E & D & E & F & I\\
 * H & E & C & B & C & E & H\\
 * G & D & B & A & B & D & G\\
 * H & E & C & B & C & E & H\\
 * I & F & E & D & E & F & I\\
 * J & I & H & G & H & I & J\\
 * \end{array}\right) \f$
 * \param [in] shift right value. Shifts the result of the filter adds / multiplies by this value.
 * \param [out] output image. Only type supported is VX_DF_IMAGE_U8
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuSymmetrical7x7FilterIntel(
        vx_context context, vx_image input, vx_array coefficients,
        vx_int32 shift_right, vx_image output);

/*! \brief [Graph] Fast symmetrical NxN filter
* \param [in] graph The reference to the graph.
* \param [in] input image. Only type supported is VX_DF_IMAGE_U8
* \param [in] filter size <tt>\ref vx_enum</tt> type, <tt>\ref VX_SYMM3X3_INTEL</tt> or <tt>\ref VX_SYMM5X5_INTEL</tt> or <tt>\ref VX_SYMM7X7_INTEL</tt> or <tt>\ref VX_SYMM9X9_INTEL</tt>.
* \param [in] coefficients Array of 3, 6, 10 or 15 coefficients of vx_int32 type: \f$\left(\begin{array}{ccc}
* A & B & C
* \end{array}\right) \f$ that forms a symmetrical 3x3 filter \f$\left(\begin{array}{ccc}
* C & B & C\\
* B & A & B\\
* C & B & C\\
* \end{array}\right) \f$
* A & B & C & D & E & F
* \end{array}\right) \f$ that forms a symmetrical 5x5 filter \f$\left(\begin{array}{ccc}
* F & E & D & E & F\\
* E & C & B & C & E\\
* D & B & A & B & D\\
* E & C & B & C & E\\
* F & E & D & E & F\\
* \end{array}\right) \f$
* A & B & C & D & E & F & G & H & I & J
* \end{array}\right) \f$ that forms a symmetrical 7x7 filter \f$\left(\begin{array}{ccc}
* J & I & H & G & H & I & J\\
* I & F & E & D & E & F & I\\
* H & E & C & B & C & E & H\\
* G & D & B & A & B & D & G\\
* H & E & C & B & C & E & H\\
* I & F & E & D & E & F & I\\
* J & I & H & G & H & I & J\\
* \end{array}\right) \f$
* A & B & C & D & E & F & G & H & I & J & K & L & M & N & O
* \end{array}\right) \f$ that forms a symmetrical 9x9 filter \f$\left(\begin{array}{ccc}
* O & N & M & L & K & L & M & N & O\\
* N & J & I & H & G & H & I & J & N\\
* M & I & F & E & D & E & F & I & M\\
* L & H & E & C & B & C & E & H & L\\
* K & G & D & B & A & B & D & G & K\\
* L & H & E & C & B & C & E & H & L\\
* M & I & F & E & D & E & F & I & M\\
* N & J & I & H & G & H & I & J & N\\
* O & N & M & L & K & L & M & N & O\\
* \end{array}\right) \f$
* \param [in] shift right value. Shifts the result of the filter adds / multiplies by this value.
* \param [out] output image. Only type supported is VX_DF_IMAGE_U8
* \return <tt>\ref vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
*/
VX_API_ENTRY vx_node VX_API_CALL vxSymmetricalNxNFilterNodeIntel(vx_graph graph,
    vx_image input, vx_enum size, vx_array coefficients, vx_int32 shift_right,
    vx_image output);

/*! \brief [Immediate] Performs Fast symmetrical NxN filter.
* \param [in] context The reference to the overall context.
* \param [in] input image. Only type supported is VX_DF_IMAGE_U8
* \param [in] filter size <tt>\ref vx_enum</tt> type, <tt>\ref VX_SYMM3X3_INTEL</tt> or <tt>\ref VX_SYMM5X5_INTEL</tt> or <tt>\ref VX_SYMM7X7_INTEL</tt> or <tt>\ref VX_SYMM9X9_INTEL</tt>.
* \param [in] coefficients Array of 3, 6, 10 or 15 coefficients of vx_int32 type: \f$\left(\begin{array}{ccc}
* A & B & C
* \end{array}\right) \f$ that forms a symmetrical 3x3 filter \f$\left(\begin{array}{ccc}
* C & B & C\\
* B & A & B\\
* C & B & C\\
* \end{array}\right) \f$
* A & B & C & D & E & F
* \end{array}\right) \f$ that forms a symmetrical 5x5 filter \f$\left(\begin{array}{ccc}
* F & E & D & E & F\\
* E & C & B & C & E\\
* D & B & A & B & D\\
* E & C & B & C & E\\
* F & E & D & E & F\\
* \end{array}\right) \f$
* A & B & C & D & E & F & G & H & I & J
* \end{array}\right) \f$ that forms a symmetrical 7x7 filter \f$\left(\begin{array}{ccc}
* J & I & H & G & H & I & J\\
* I & F & E & D & E & F & I\\
* H & E & C & B & C & E & H\\
* G & D & B & A & B & D & G\\
* H & E & C & B & C & E & H\\
* I & F & E & D & E & F & I\\
* J & I & H & G & H & I & J\\
* \end{array}\right) \f$
* A & B & C & D & E & F & G & H & I & J & K & L & M & N & O
* \end{array}\right) \f$ that forms a symmetrical 9x9 filter \f$\left(\begin{array}{ccc}
* O & N & M & L & K & L & M & N & O\\
* N & J & I & H & G & H & I & J & N\\
* M & I & F & E & D & E & F & I & M\\
* L & H & E & C & B & C & E & H & L\\
* K & G & D & B & A & B & D & G & K\\
* L & H & E & C & B & C & E & H & L\\
* M & I & F & E & D & E & F & I & M\\
* N & J & I & H & G & H & I & J & N\\
* O & N & M & L & K & L & M & N & O\\
* \end{array}\right) \f$
* \param [in] shift right value. Shifts the result of the filter adds / multiplies by this value.
* \param [out] output image. Only type supported is VX_DF_IMAGE_U8
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Success
* \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxuSymmetricalNxNFilterIntel(
    vx_context context, vx_image input, vx_enum size, vx_array coefficients,
    vx_int32 shift_right, vx_image output);

/*! \brief [Immediate] Creates a channel separate node.
 * \param [in] graph The graph reference.
 * \param [in] output The output image. The format of the image must be defined, even if the image is virtual.
 * \param [out] plane0 The plane that forms channel 0. Must be <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [out] plane1 The plane that forms channel 1. Must be <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [out] plane2 [optional] The plane that forms channel 2. Must be <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [out] plane3 [optional] The plane that forms channel 3. Must be <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuChannelSeparateIntel(vx_context context,
        vx_image input, vx_image plane0, vx_image plane1, vx_image plane2,
        vx_image plane3);


/*! \brief [Graph] Creates a node for computing Gradient vectors of an image using the Sobel, Scharr, or Prewitt operator.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] kernel A <tt>\ref VX_TYPE_ENUM</tt> of the <tt>\ref vx_operator_e</tt> enumeration.
 * \param [in] size Supported mask size: 3 or 5 (for Sobel operator only).
 * \param [in] norm Normalization <tt>\ref VX_TYPE_ENUM</tt> type, either <tt>\ref VX_NORM_L1</tt> or <tt>\ref VX_NORM_L2</tt>.
 * \param [out] gx The output x component of gradient vector in <tt>\ref VX_DF_IMAGE_S16</tt> (in case of <tt>\ref VX_DF_IMAGE_U8</tt> input) or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [out] gy The output y component of gradient vector in <tt>\ref VX_DF_IMAGE_S16</tt> (in case of <tt>\ref VX_DF_IMAGE_U8</tt> input) or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [out] mag The magnitude of gradient vector in <tt>\ref VX_DF_IMAGE_U16</tt> (in case of <tt>\ref VX_DF_IMAGE_U8</tt> input) or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [out] ang The angle of gradient vector in <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxGradientVectorNodeIntel(vx_graph graph,
        vx_image input, vx_enum kernel, vx_int32 size, vx_enum norm, vx_image gx,
        vx_image gy, vx_image mag, vx_image ang);

/*! \brief [Immediate] Compute Gradient vectors of an image using the Sobel, Scharr, or Prewitt operator.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] kernel A <tt>\ref VX_TYPE_ENUM</tt> of the <tt>\ref vx_operator_e</tt> enumeration.
 * \param [in] size Supported mask size: 3 or 5 (for Sobel operator only)
 * \param [in] norm Normalization <tt>\ref VX_TYPE_ENUM</tt> type, either <tt>\ref VX_NORM_L1</tt> or <tt>\ref VX_NORM_L2</tt>.
 * \param [out] gx The output x component of gradient vector in <tt>\ref VX_DF_IMAGE_S16</tt> (in case of <tt>\ref VX_DF_IMAGE_U8</tt> input) or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [out] gy The output y component of gradient vector in <tt>\ref VX_DF_IMAGE_S16</tt> (in case of <tt>\ref VX_DF_IMAGE_U8</tt> input) or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [out] mag The magnitude of gradient vector in <tt>\ref VX_DF_IMAGE_U16</tt> (in case of <tt>\ref VX_DF_IMAGE_U8</tt> input) or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [out] ang The angle of gradient vector in <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success.
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuGradientVectorIntel(vx_context context,
        vx_image input, vx_enum kernel, vx_int32 size, vx_enum norm, vx_image gx,
        vx_image gy, vx_image mag, vx_image ang);

/*! \brief [Graph] Creates a Haar Classifier cascade node.
 * \param [in] graph The reference to the graph
 * \param [in] input The input image
 * \param [in] init Enumeration that specifies way to interpret string with classifier data. Possible values: <tt>\ref VX_HAAR_FROM_FILE</tt> - path to file with trained data, or <tt>\ref VX_HAAR_FROM_STRING</tt> - trained data
 * \param [in] decstage The parameter for haar thresholds tuning. Usually 0.
 * \param [out] num The number of positive decisions
 * \param [out] output The output image. Pixels corresponding to detected objects set to 1, all other to 0
 * \return <tt>\ref vx_node</tt>
 * \retval 0 Node could not be created
 * \retval * Node handle
 */
VX_API_ENTRY vx_node VX_API_CALL vxHaarClassifierNodeIntel(vx_graph graph,
        vx_image input, vx_enum init, vx_float32 decstage, vx_scalar num,
        vx_image output);

/*! \brief [Immediate] Apply Haar Classifier cascade to the input data.
 * \param [in] context The reference to the context
 * \param [in] input The input image
 * \param [in] init Enumeration that specifies way to interpret string with classifier data. Possible values: <tt>\ref VX_HAAR_FROM_FILE</tt> - path to file with trained data, or <tt>\ref VX_HAAR_FROM_STRING</tt> - trained data
 * \param [in] decstage The parameter for haar thresholds tuning. Usually 0.
 * \param [out] num The number of positive decisions
 * \param [out] output The output image. Pixels corresponding to detected objects set to 1, all other to 0
 * \return A <tt>\ref vx_status_e</tt> enumeration
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>
 */
VX_API_ENTRY vx_status VX_API_CALL vxuHaarClassifierIntel(vx_context context,
        vx_image input, vx_enum init, vx_float32 decstage, const char *str,
        size_t str_size, vx_scalar num, vx_image output);

/*! \brief [Graph] Applies Local Binary Patterns operator to an input image.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image VX_DF_IMAGE_U8 or VX_DF_IMAGE_F32_INTEL.
 * \param [in] mask_size Kernel size: 3 or 5.
 * \param [out] output The output image VX_DF_IMAGE_U8 or VX_DF_IMAGE_U16.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxLBPImageNodeIntel(vx_graph graph,
        vx_image input, vx_int32 mask_size, vx_image output);
/*! \brief [Immediate] Applies Local Binary Patterns operator to an input image.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image VX_DF_IMAGE_U8 or VX_DF_IMAGE_F32_INTEL.
 * \param [in] mask_size Kernel size: 3 or 5.
 * \param [out] output The output image VX_DF_IMAGE_U8 or VX_DF_IMAGE_U16.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuLBPImageIntel(vx_context context,
        vx_image input, vx_int32 mask_size, vx_image output);

/*! \brief [Graph] Creates a MaskedCopy node.
 * \param [in] graph The reference to the graph.
 * \param [in] backgound Input background image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>, or <tt>\ref VX_DF_IMAGE_S32</tt> format.
 * \param [in] foreground Input foreground image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>, or <tt>\ref VX_DF_IMAGE_S32</tt> format.
 * \param [in] mask Input mask image in <tt>\ref VX_DF_IMAGE_U8</tt> format. Non-zero values mark pixels that should be taken from foreground image, zero values - from background image.
 * \param [out] output The output image in the same format as the input images.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxMaskedCopyNodeIntel(vx_graph graph,
        vx_image foreground, vx_image background, vx_image mask, vx_image output);

/*! \brief [Immediate] Provide masked copy operation.
 * \param [in] context The reference to the overall context.
 * \param [in] backgound Input background image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>, or <tt>\ref VX_DF_IMAGE_S32</tt> format.
 * \param [in] foreground Input foreground image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>, or <tt>\ref VX_DF_IMAGE_S32</tt> format.
 * \param [in] mask Input mask image in <tt>\ref VX_DF_IMAGE_U8</tt> format. Non-zero values mark pixels that should be taken from foreground image, zero values - from background image.
 * \param [out] output The output image in the same format as the input images.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success.
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuMaskedCopyIntel(vx_context context,
        vx_image foreground, vx_image background, vx_image mask, vx_image output);

/*! \brief [Immediate] Perform halftone operation
 * \param [in] context The reference to the context
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format
 * \param [in] thresholds The image of VX_DF_IMAGE_U8 type with threshold values. This values must be set in the moment of graph compilation.
 * \param [in] shift Horizontal shift for threshold blocks. Calculated as (y / ThresholdHeight) * shift
 * \param [out] output The output image in the same format as input image
 * \return A <tt>\ref vx_status_e</tt> enumeration
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>
 */
VX_API_ENTRY vx_status VX_API_CALL vxuHalftoneIntel(vx_context context,
        vx_image input, vx_image thresholds, vx_int32 shift, vx_image output);

/*! \brief [Graph] Fast color space conversion using tetrahedral interpolation.
 * \param [in] graph The reference to the graph.
 * \param [in] input1 The input image for 1st channel.
 * \param [in] input2 The input image for 2nd channel.
 * \param [in] input3 The input image for 3rd channel.
 * \param [in] map Maps source channels from (0..255) to (0..15) with 4.7 precision. Array element type <tt>\ref vx_coordinates3d_t</tt>.
 * \param [in] values Set of 4096 samples of output channels. Array element type <tt>\ref vx_coordinates4d_intel_t</tt>.
 * \param [out] output1 The output image for 1st channel.
 * \param [out] output2 The output image for 2nd channel.
 * \param [out] output3 The output image for 3rd channel.
 * \param [out] output4 The output image fir 4th channel.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTetrahedralInterpolationNodeIntel(vx_graph graph, vx_image input1, vx_image input2, vx_image input3,
                                                                vx_array map, vx_array values,
                                                                vx_image output1, vx_image output2, vx_image output3, vx_image output4);


/*! \brief [Graph] Computes Integral and square integral image.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image VX_DF_IMAGE_U8.
 * \param [in] val Constant value to be added to all output pixels.
 * \param [in] valSqr Constant value to be added to all output sqr pixels.
* \param [out] output The output integral image VX_DF_IMAGE_F32_INTEL. This image must be of the same size as input image.
 * \param [out] outputSqr The output integral square image VX_DF_IMAGE_F64_INTEL.
 The size of this image must be 1 pixel larger than input image in both dimensions.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxSqrIntegralNodeIntel(vx_graph graph,
        vx_image input, vx_float64 val, vx_float64 valSqr, vx_image output,
        vx_image outputSqr);
/*! \brief [Immediate] Computes Integral and square integral image.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image VX_DF_IMAGE_U8.
 * \param [in] val Constant value to be added to all output pixels.
 * \param [in] valSqr Constant value to be added to all output sqr pixels.
 * \param [out] output The output integral image VX_DF_IMAGE_F32_INTEL. This image must be of the same size as input image.
 * \param [out] outputSqr The output integral square image VX_DF_IMAGE_F64_INTEL.
 The size of this image must be 1 pixel larger than input image in both dimensions.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuSqrIntegralIntel(vx_context context,
        vx_image input, vx_float64 val, vx_float64 valSqr, vx_image output,
        vx_image outputSqr);


/*! \brief [Immediate] Performs non-maximum suppression in given radius for each point of image.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] threshold Threshold, as <tt>\ref VX_TYPE_FLOAT32</tt>, all values below it set to zero in output image.
 * \param [in] radius Suppression radius.
 * \param [out] output The output image in the same format as the input image.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success.
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuEuclideanNonMaxSuppressionIntel(vx_context,
        vx_image input, vx_scalar threshold, vx_float32 radius, vx_image output);

/*! \brief [Graph] Creates a Euclidean non-max suppression node.
* \param [in] graph The reference to the graph.
* \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
* \param [in] threshold Threshold, as <tt>\ref VX_TYPE_FLOAT32</tt>, all values below it set to zero in output image.
* \param [in] radius Suppression radius.
* \param [out] output The output image in the same format as the input image.
* \return <tt>\ref vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
*/
VX_API_ENTRY vx_node VX_API_CALL vxEuclideanNonMaxSuppressionNodeIntel(
    vx_graph graph, vx_image input, vx_scalar threshold, vx_float32 radius,
    vx_image output);


/*! \brief [Graph] Creates peak-to-sidelobe ratio node:
 * \ PSR=(p - mu)/sigma, Where p is a peak intensity value, mu and sigma are mean and standard deviation of intensity values in a rectangular frame around the peak:
 *! Please find the details here:
   *  \image html psr.jpg.
 * \param [in] graph The reference to the graph.
 * \param [in] input An input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format
 * \param [in] peakTop Peak top margin
 * \param [in] peakBot Peak bottom margin
 * \param [in] peakLef Peak left margin
 * \param [in] peakRig Peak right margin
 * \param [in] winTop Window (sidelobe) top margin
 * \param [in] winBot Window (sidelobe) bottom margin
 * \param [in] winLef Window (sidelobe) left margin
 * \param [in] winRig Window (sidelobe) right margin
 * \param [out] out The output image in the same format as the input image
 * \ingroup group_vision_function_psr
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxPSRNodeIntel(vx_graph graph, vx_image input,
        vx_uint32 peakTop, vx_uint32 peakBot, vx_uint32 peakLef,
        vx_uint32 peakRig, vx_uint32 winTop, vx_uint32 winBot, vx_uint32 winLef,
        vx_uint32 winRig, vx_image output);

/*! \brief [Immediate] Compute peak-to-sidelobe ratio at every pixel location of input image:
 * \ PSR=(p - mu)/sigma, Where p is a peak intensity value, mu and sigma are mean and standard deviation of intensity values in a rectangular frame around the peak:
 *! Please find the details here:
    *  \image html psr.jpg
 * \param [in] context The reference to the overall context.
 * \param [in] input An input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format
 * \param [in] peakTop Peak top margin
 * \param [in] peakBot Peak bottom margin
 * \param [in] peakLef Peak left margin
 * \param [in] peakRig Peak right margin
 * \param [in] winTop Window (sidelobe) top margin
 * \param [in] winBot Window (sidelobe) bottom margin
 * \param [in] winLef Window (sidelobe) left margin
 * \param [in] winRig Window (sidelobe) right margin
 * \param [out] out The output image in the same format as the input image
 * \ingroup group_vision_function_psr
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuPSRIntel(vx_context context, vx_image input,
        vx_uint32 peakTop, vx_uint32 peakBot, vx_uint32 peakLef,
        vx_uint32 peakRig, vx_uint32 winTop, vx_uint32 winBot, vx_uint32 winLef,
        vx_uint32 winRig, vx_image output);

/*! \brief [Immediate] Performs pixel-wise division on pixel values of input images
 * \param [in] context The reference to the overall context.
 * \param [in] divident Dividend image of <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [in] divisor Divisor image of <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [out] quotient Quotient image of <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [out] reminder Reminder image of <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \ingroup group_vision_function_devide
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuDivideIntel(vx_context context,
        vx_image input1, vx_image input2, vx_image quotient, vx_image reminder);

/*! \brief [Graph] Creates a Non-max suppression node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_U16</tt> or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] conv The reference to <tt>\ref vx_convolution</tt> object which holds the structural element's coefficients
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxNonMaxSuppressionNodeIntel(vx_graph graph,
        vx_image input, vx_convolution conv, vx_image output);

/*! \brief [Immediate] Perform a non-maximum suppression with the given structural element.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_U16</tt> or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] conv The reference to <tt>\ref vx_convolution</tt> object which holds the structural element's coefficients
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success.
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuNonMaxSuppressionIntel(vx_context context,
        vx_image input, vx_convolution conv, vx_image output);

/*! \brief [Graph] Creates a Peak Lister node.
 * \param [in] graph The reference to the graph.
 * \param [in] input An input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format
 * \param [out] out Output array of keypoints
 * \ingroup group_vision_function_peaklister
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxPeakListerNodeIntel(vx_graph graph,
        vx_image image, vx_array peaks);

/*! \brief [Immediate] Finds all clusters of non-zero pixels in the input image, produces a list of these clusters (represented as keypoints).
 * \param [in] graph The reference to the graph.
 * \param [in] input An input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format
 * \param [out] out Output array of keypoints
 * \ingroup group_vision_function_peaklister
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuListPeaksIntel(vx_context context,
        vx_image image, vx_array peaks);

/*! \brief [Graph] Create Hough lines detection node.
 * \param [in] graph The reference to the graph.
 * \param [in] input An Input image in <tt>\ref VX_DF_IMAGE_U8</tt> format
 * \param [in] deltaRho Non-negative <tt>\ref VX_TYPE_FLOAT32</tt> step of discretization of radial coordinate
 * \param [in] deltaTheta Non-negative <tt>\ref VX_TYPE_FLOAT32</tt> step of discretization of angular coordinate
 * \param [in] threshold Input <tt>\ref VX_TYPE_INT32</tt> minimum number of points that are required to detect the line
 * \param [in] maxcount Input <tt>\ref VX_TYPE_INT32</tt> maximum number of lines to be stored
 * \param [out] lines Detected lines stored in <tt>\ref vx_array</tt> of type <tt>\ref vx_coordinates_polar_t</tt>.
 * \param [out] count Output <tt>\ref VX_TYPE_INT32</tt> number of detected lines [optional]
 * \ingroup group_vision_function_houghlines
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxHoughLinesNodeIntel(vx_graph graph,
        vx_image input, vx_float32 deltaRho, vx_float32 deltaTheta,
        vx_int32 threshold, vx_int32 maxcount, vx_array lines, vx_scalar count);

/*! \brief [Immediate] Detects straight lines using Hough transform.
 * \param [in] context The reference to the overall context.
 * \param [in] input An Input image in <tt>\ref VX_DF_IMAGE_U8</tt> format
 * \param [in] deltaRho Non-negative <tt>\ref VX_TYPE_FLOAT32</tt> step of discretization of radial coordinate
 * \param [in] deltaTheta Non-negative <tt>\ref VX_TYPE_FLOAT32</tt> step of discretization of angular coordinate
 * \param [in] threshold Input <tt>\ref VX_TYPE_INT32</tt> minimum number of points that are required to detect the line
 * \param [in] maxcount Input <tt>\ref VX_TYPE_INT32</tt> maximum number of lines to be stored
 * \param [out] lines Detected lines stored in <tt>\ref vx_array</tt> of type <tt>\ref vx_coordinates_polar_t</tt>.
 * \param [out] count Output <tt>\ref VX_TYPE_INT32</tt> number of detected lines [optional]
 * \ingroup group_vision_function_houghlines
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuHoughLinesIntel(vx_context context,
        vx_image input, vx_float32 deltaRho, vx_float32 deltaTheta,
        vx_int32 threshold, vx_int32 maxcount, vx_array lines, vx_scalar count);
/*! \brief [Immediate] Detects straight lines using probabilistic Hough transform.
 * \param [in] context The reference to the overall context.
 * \param [in] input An Input image in <tt>\ref VX_DF_IMAGE_U8</tt> format
 * \param [in] deltaRho Non-negative <tt>\ref VX_TYPE_FLOAT32</tt> step of discretization of radial coordinate
 * \param [in] deltaTheta Non-negative <tt>\ref VX_TYPE_FLOAT32</tt> step of discretization of angular coordinate
 * \param [in] threshold Input <tt>\ref VX_TYPE_INT32</tt> minimum number of points that are required to detect the line
 * \param [in] lineLen Minimum length of the line
 * \param [in] lineGap Maximum length of the gap between lines.
 * \param [in] maxcount Input <tt>\ref VX_TYPE_INT32</tt> maximum number of lines to be stored
 * \param [in] algorithm <tt>\ref vx_enum</tt> which specifies algorithm to be used. Possible values are: VX_HOUGH_LINES_P_NONE, VX_HOUGH_LINES_P_FAST, VX_HOUGH_LINES_P_ACCURATE
 * \param [out] lines Detected lines stored in <tt>\ref vx_array</tt> of type <tt>\ref vx_rectangle_t<\tt>.
 * \param [out] count Output <tt>\ref VX_TYPE_INT32</tt> number of detected lines [optional]
 * \ingroup group_vision_function_houghlines_p
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuHoughLinesPIntel(vx_context context,
        vx_image input, vx_float32 deltaRho, vx_float32 deltaTheta,
        vx_int32 threshold, vx_int32 lineLen, vx_int32 lineGap, vx_int32 maxcount,
        vx_array lines, vx_scalar count);

/*! \brief [Immediate] Detects Hough circles.
 * \param [in] graph The reference to the graph.
 * \param [in] input An Input binary image of edges in <tt>\ref VX_DF_IMAGE_U8</tt> format
 * \param [in] input An Input image of dx gradients in <tt>\ref VX_DF_IMAGE_S16</tt> format
 * \param [in] input An Input image of dy gradients in <tt>\ref VX_DF_IMAGE_S16</tt> format
 * \param [in] minDist Non-negative <tt>\ref VX_TYPE_INT32</tt> minimum distance between centers of detected circles
 * \param [in] minRadius Non-negative <tt>\ref VX_TYPE_INT32</tt> minimum radius of detected circle
 * \param [in] maxRadius Non-negative <tt>\ref VX_TYPE_INT32</tt> maximum radius of detected circle
 * \param [in] maxcount Input <tt>\ref VX_TYPE_INT32</tt> maximum number of circles to be stored
 * \param [in] minEvidencePointsThreshold <tt>\ref VX_TYPE_INT32</tt> minimum points count in the suitable neighborhood to propose center
 * \param [in] minCircleFilledThreshold <tt>\ref VX_TYPE_FLOAT32</tt> in range [0..1] minimum percent of circle line filling to detect circle *
 * \param [out] circles Detected circles stored in <tt>\ref vx_array</tt> of type <tt>\ref vx_coordinates3d_t</tt>. *
 * \param [out] count Output <tt>\ref VX_TYPE_INT32</tt> number of detected circles [optional]
 * \ingroup group_vision_function_houghcircles
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuHoughCirclesIntel(vx_context context,
        vx_image input, vx_image dx, vx_image dy, vx_scalar minDistance,
        vx_scalar minRadius, vx_scalar maxRadius,
        vx_scalar minEvidencePointsThreshold, vx_scalar minCircleFilledThreshold,
        vx_array circles, vx_scalar count);

/*! \brief [Graph] Creates DCT node
 * \param [in] graph The reference to the graph.
 * \param [in] input An input image in <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] is_reverse false - forward DCT, true - inverse DCT
 * \param [in] per_line false - perform DCT on whole image, true - perform DCT per image row.
 * \param [out] out The output image in the same format as the input image
 * \ingroup group_vision_function_dct
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxDctNodeIntel(vx_graph graph, vx_image input,
        vx_bool is_inverse, vx_bool per_line, vx_image output);

/*! \brief [Immediate] Performs DCT of input image.
 * \param [in] context The reference to the overall context.
 * \param [in] input An input image in <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] is_reverse false - forward DCT, true - inverse DCT
 * \param [in] per_line false - perform DCT on whole image, true - perform DCT per image row.
 * \param [out] out The output image in the same format as the input image
 * \ingroup group_vision_function_dct
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuDctIntel(vx_context context, vx_image input,
        vx_bool is_inverse, vx_bool per_line, vx_image output);

/*! \brief [Graph] Creates crop node
 * \param [in] graph The reference to the graph.
 * \param [in] input An input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_S32</tt>, <tt>\ref VX_DF_IMAGE_U32</tt>, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>, <tt>\ref VX_DF_IMAGE_RGB</tt>, or <tt>\ref VX_DF_IMAGE_RGBX</tt> format.
 * \param [in] left distance from left border in pixels
 * \param [in] right distance from right border in pixels
 * \param [in] top distance from top border in pixels
 * \param [in] bottom distance from bottom border in pixels
 * \param [out] out The output image in the same format as the input image
 * \ingroup group_vision_function_crop
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxCropNodeIntel(vx_graph graph, vx_image input,
        vx_int32 left, vx_int32 right, vx_int32 top, vx_int32 bottom,
        vx_image out);

/*! \brief [Immediate] Performs crop of input images
 * \param [in] context The reference to the overall context.
 * \param [in] input An input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_S32</tt>, <tt>\ref VX_DF_IMAGE_U32</tt>, or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] left distance from left border in pixels
 * \param [in] right distance from right border in pixels
 * \param [in] top distance from top border in pixels
 * \param [in] bottom distance from bottom border in pixels
 * \param [out] out The output image in the same format as the input image
 * \ingroup group_vision_function_crop
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuCropIntel(vx_context context,
        vx_image input, vx_int32 left, vx_int32 right, vx_int32 top,
        vx_int32 bottom, vx_image out);

/*! \brief [Graph] Creates pad node
* \param [in] graph The reference to the graph.
* \param [in] input An input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_RGB</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt> format.
* \param [in] left distance from left border in pixels
* \param [in] right distance from right border in pixels
* \param [in] top distance from top border in pixels
* \param [in] bottom distance from bottom border in pixels
* \param [out] out The output image in the same format as the input image
* \ingroup group_vision_function_crop
* \return <tt>\ref vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
*/
VX_API_ENTRY vx_node VX_API_CALL vxPadNodeIntel(vx_graph graph, vx_image input,
    vx_int32 left, vx_int32 right, vx_int32 top, vx_int32 bottom,
    vx_image out);

/*! \brief [Immediate] Performs padding of input images
* \param [in] context The reference to the overall context.
* \param [in] input An input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_RGB</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt> format.
* \param [in] left distance from left border in pixels
* \param [in] right distance from right border in pixels
* \param [in] top distance from top border in pixels
* \param [in] bottom distance from bottom border in pixels
* \param [out] out The output image in the same format as the input image
* \ingroup group_vision_function_crop
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Success
* \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxuPadIntel(vx_context context,
    vx_image input, vx_int32 left, vx_int32 right, vx_int32 top,
    vx_int32 bottom, vx_image out);

/*! \brief [Immediate] Computes square roots of pixel values of a source image and writes them into the destination image.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image.
 * \param [out] output The output image.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuSqrtIntel(vx_context context,
        vx_image input, vx_image output);

/*! \brief [Immediate] Calculate a log of the pixel values of a source image and writes them into the destination image.
* \param [in] src The input image of type VX_DF_IMAGE_S16.
* \param [in] scale A Scale const parameter. The result will be multiplied by the scale
* factor before converted to S16.
* \param [in] overflow_policy policy to use when the result overflowed of type vx_convert_polict_e.
* \param [in] rounding_policy policy to use in truncation. of type vx_round_policy_e.
* \param [out] output log of the input image of type VX_DF_IMAGE_S16.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Success
* \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxuLogIntel(vx_context context, vx_image input,
        vx_int32 scale, vx_enum convert_policy, vx_enum round_policy, vx_image output);

/*! \brief [Graph] Creates an True Distance Transform node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image, <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [out] output The output image, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>.
 * \return <tt>\ref vx_node</tt>
 * \retval vx_node A node reference.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTrueDistanceTransformNodeIntel(vx_graph graph,
        vx_image input, vx_image output);

/*! \brief [Immediate] Computes the Eucledian distance to the closest zero pixel for all non-zero pixels.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image, <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [out] output The output image, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuTrueDistanceTransformIntel(
        vx_context context, vx_image input, vx_image output);

/*! \brief [Immediate] Compute an absolute difference norm of a given image(images) or relative difference norm of two given images
 * \param [in] context The reference to the overall context.
 * \param [in] input1 The first input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_U16</tt> or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] input2 The second input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_U16</tt> or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format [optional].
 * \param [in] metric A norm type from the <tt>\ref vx_norm_intel_e</tt> enumeration. L1, L2, INF norm types are supported.
 * \param [out] output The output norm value.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success.
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuNormIntel(vx_context context,
        vx_image input1, vx_image input2, vx_enum metric, vx_scalar output);

/*! \brief [Graph] Creates a Color to Gray Convert node.
 * \param [in] graph The reference to the graph.
 * \param [in] input0 The input channel 0.
 * \param [in] input1 The input channel 1 [optional].
 * \param [in] input2 The input channel 2 [optional].
 * \param [in] coeffs The conversion coefficients [optional].
 * \param [out] output The output image.
 * \return <tt>\ref vx_node</tt>
 * \retval vx_node A node reference.
 */
VX_API_ENTRY vx_node VX_API_CALL vxColorToGrayNodeIntel(vx_graph graph,
        vx_image input0, vx_image input1, vx_image input2, vx_array coeffs,
        vx_image output);

/*! \brief [Immediate] Performs color to gray conversion.
 * \param [in] context The reference to the overall context.
 * \param [in] input0 The input channel 0.
 * \param [in] input1 The input channel 1 [optional].
 * \param [in] input2 The input channel 2 [optional].
 * \param [in] coeffs The conversion coefficients [optional].
 * \param [out] output The output image.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuColorToGrayIntel(vx_context context,
        vx_image input0, vx_image input1, vx_image input2, vx_array coeffs,
        vx_image output);

/*! \brief [Graph] Creates a transpose node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image.
 * \param [out] output The output image.
 * \return <tt>\ref vx_node</tt>
 * \retval vx_node A node reference.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTransposeNodeIntel(vx_graph graph,
        vx_image input, vx_image output);

/*! \brief [Immediate] Performs the transpose operation.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image.
 * \param [out] output The output image.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuTransposeIntel(vx_context context,
        vx_image input, vx_image output);

/*! \brief [Graph] Performs per-element comparison between two images.
* \param [in] context The reference to the overall context.
* \param [in] input1 The first input image.
* Supported input data types: \n
*  VX_DF_IMAGE_RGBX \n
*  VX_DF_IMAGE_RGB \n
*  VX_DF_IMAGE_U8 \n
*  VX_DF_IMAGE_U16 \n
*  VX_DF_IMAGE_S16 \n
* \param [in] input2 The second input image of the same type as input1.
* \param [in] type A comparison type from the <tt>\ref vx_compare_intel_e</tt> enumeration.
* \param [out] output The output image.
* Output data type is either VX_DF_IMAGE_RGBX, VX_DF_IMAGE_RGB, or VX_DF_IMAGE_U8 for all other input data types.
* Output pixels values equal to 255 if input pixels meet the comparison condition, and 0 otherwise.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Success
* \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxuCompareIntel(vx_context context,
        vx_image input1, vx_image input2, vx_enum type, vx_image output);

/*! \brief [Graph] Creates a spectrum multiplication node.
 * \param [in] graph The reference to the graph.
 * \param [in] input1 The first input image , <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>.
 * \param [in] input2 The second input image, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>.
 * \param [out] output The output image, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>.
 * \return <tt>\ref vx_node</tt>
 * \retval vx_node A node reference.
 */
VX_API_ENTRY vx_node VX_API_CALL vxMulSpectrumsNodeIntel(vx_graph graph,
        vx_image input1, vx_image input2, vx_image output);

/*! \brief [Immediate] Performs the spectrums multiplication operation.
 * \param [in] context The reference to the overall context.
 * \param [in] input1 The first input image , <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>.
 * \param [in] input2 The second input image, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>.
 * \param [out] output The output image, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuMulSpectrumsIntel(vx_context context,
        vx_image input1, vx_image input2, vx_image output);

/*! \brief [Immediate] Converts to Cartesian coordinates.
 * \param [in] context The reference to the overall context.
 * \param [in] inputMagn The input image which stores the magnitude (radius) components of the elements in polar coordinate form, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> or <tt>\ref VX_DF_IMAGE_F64_INTEL</tt>.
 * \param [in] inputPh The input image which stores the phase (angle) components of the elements in polar coordinate form in radians, <tt>\ref VX_DF_IMAGE_F32_INTEL or <tt>\ref VX_DF_IMAGE_F64_INTEL</tt>.</tt>.
 * \param [out] outputRe The output image which stores the real components of Cartesian X,Y pairs, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt>.
 * \param [out] outputIm The output image which stores the imaginary components of Cartesian X,Y pairs, <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> or <tt>\ref VX_DF_IMAGE_F64_INTEL</tt>.</tt>.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuPolarToCartIntel(vx_context context,
        vx_image inputMagn, vx_image inputPh, vx_image outputRe,
        vx_image outputIm);

/*! \brief [Immediate] Applies an adaptive threshold to an image.
 * * In case of adaptive threshold box type the following equation apply
 * * \f$ intermediate(y,x)=\frac{1}{N^{2}}\sum_{i=\frac{N-1}{2}}^{\frac{{N-1}}{2}}\sum_{j=\frac{{N-1}}{2}}^{\frac{{N-1}}{2}}input(y+j,x+i)\\
 * *  \f$
 * * In case of adaptive threshold Gaussian type the following equation apply
 * * \f$ intermediate(y,x)=\frac{1}{N^{2}}\sum_{i=\frac{N-1}{2}}^{\frac{{N-1}}{2}}\sum_{j=\frac{{N-1}}{2}}^{\frac{{N-1}}{2}}G(i,j)input(y+j,x+i)\\
 * * \f$
 * * Where \f$G(i,j) \f$ is the Gaussian filter. Currently only Gaussian of 3x3 and 5x5 are supported.
 * * \param [in] graph The reference to the graph.
 * * \param [in] input The input image <tt>\ref VX_DF_IMAGE_U8</tt>.
 * * \param [in] filter A filtering type from <tt>\ref vx_adaptive_threshold_filter_intel_e</tt> enumeration. Possible values: <tt>\ref VX_ADAPTIVE_THRESHOLD_FILTER_BOX_INTEL</tt>, <tt>\ref VX_ADAPTIVE_THRESHOLD_FILTER_GAUSS_INTEL</tt>.
 * * \param [in] ksize Linear size of the filtering kernel. Supported values are 3 and 5 for <tt>\ref VX_ADAPTIVE_THRESHOLD_FILTER_GAUSS_INTEL</tt> and any odd nuber from 3 to 31 for <tt>\ref VX_ADAPTIVE_THRESHOLD_FILTER_BOX_INTEL</tt>.
 * * \param [in] threshold <tt>\ref vx_threshold</tt> of type <tt>\ref VX_THRESHOLD_TYPE_BINARY</tt>.
 * * \param [out] output The output image <tt>\ref VX_DF_IMAGE_U8</tt>.
 * * \return <tt>\ref vx_node</tt>.
 * * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 * */

VX_API_ENTRY vx_status VX_API_CALL vxuAdaptiveThresholdIntel(vx_context context,
        vx_image input, vx_enum filter, vx_uint32 ksize, vx_threshold threshold,
        vx_image output);



/*! \brief [Graph] Creates a node for advanced morphology operations.
 * \param [in] graph The reference to the graph.
 * \param [in] input The first input image in <tt>\ref VX_DF_IMAGE_RGB</tt>, <tt>\ref VX_DF_IMAGE_RGBX</tt>, <tt>\ref VX_DF_IMAGE_U8</tt>, or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] kernel The reference to <tt>\ref vx_convolution</tt> object which holds the morphology operation kernel
 * \param [in] type The morphology operation type as <tt>\ref VX_TYPE_ENUM</tt> <tt>\ref vx_enum_morph_e</tt>.
 * \param [in] iterations Number of iterations to be done.
 * \param [out] output The output image in the same format as input image.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxMorphologyExNodeIntel(vx_graph graph,
        vx_image input, vx_convolution kernel, vx_enum type, vx_int32 iterations,
        vx_image output);

/*! \brief [Graph] Performs advanced morphology operations.
 * \param [in] context The reference to the overall context.
 * \param [in] input The first input image in <tt>\ref VX_DF_IMAGE_RGB</tt>, <tt>\ref VX_DF_IMAGE_RGBX</tt>, <tt>\ref VX_DF_IMAGE_U8</tt>, or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] kernel The reference to <tt>\ref vx_convolution</tt> object which holds the morphology operation kernel
 * \param [in] type The morphology operation type as <tt>\ref VX_TYPE_ENUM</tt> <tt>\ref vx_enum_morph_e</tt>.
 * \param [in] iterations Number of iterations to be done.
 * \param [out] output The output image in the same format as input image.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success.
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuMorphologyExIntel(vx_context context,
        vx_image input, vx_convolution kernel, vx_enum type, vx_int32 iterations,
        vx_image output);

/*! \brief [Graph] Creates a normalization node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] norm The norm value to be obtained.
 * \param [in] type The normalization type as <tt>\ref VX_TYPE_ENUM</tt>, either <tt>\ref VX_NORM_L1</tt>, or <tt>\ref VX_NORM_L2</tt>, or <tt>\ref VX_NORM_INF</tt>.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxNormalizeNodeIntel(vx_graph graph,
        vx_image input, vx_float32 norm, vx_enum type, vx_image output);

/*! \brief [Immediate] Performs normalization of an image to obtain image norm of required type equal to given value.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] norm The norm value to be obtained.
 * \param [in] type The normalization type as <tt>\ref VX_TYPE_ENUM</tt>, either <tt>\ref VX_NORM_L1</tt>, or <tt>\ref VX_NORM_L2</tt>, or <tt>\ref VX_NORM_INF</tt>.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuNormalizeIntel(vx_context context,
        vx_image input, vx_float32 norm, vx_enum type, vx_image output);

/*! \brief [Immediate] Performs the template matching operation.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image in either <tt>\ref VX_DF_IMAGE_U8</tt>,  <tt>\ref VX_DF_IMAGE_U16</tt>, or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] templ The template image in the same format as the input image.
 * \param [in] match_algorithm The match template algorithm to use. <tt>\ref VX_TM_SQDIFF_INTEL</tt>, <tt>\ref VX_TM_CRCORR_INTEL</tt> and <tt>\ref VX_TM_CRCOEFF_INTEL</tt> are supported
 * \param [in] impl_algorithm Implementation variant of algorithm to use for calculation. <tt>\ref VX_MATCH_TEMPLATE_DIRECT_INTEL</tt> and <tt>\ref VX_MATCH_TEMPLATE_FFT_INTEL</tt> are supported
 * \param [in] normalize If true, output normalization will be used (not used in case of VX_TM_CRCOEFF).
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \ingroup group_vision_function_match_template
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuMatchTemplateIntel(vx_context context,
        vx_image input, vx_image templ, vx_enum match_algorithm,
        vx_enum impl_algorithm, vx_bool normalize, vx_image output);

/*! \brief [Graph] Creates a CornerMinEigenVal node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in either <tt>\ref VX_DF_IMAGE_U8</tt>,  <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] kernelType Specifies the type of kernel used to compute derivatives, possible values are:<tt>\ref VX_OPERATOR_SOBEL </tt> or <tt>\ref VX_OPERATOR_SCHARR<//tt>.
 * \param [in] apertureSize Size of the derivative operator in pixels, possible values are 3 or 5. Only 3x3 size is available for the Scharr kernel.
 * \param [in] avgWindow Size of the blurring window in pixels, possible values are 3 or 5.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxCornerMinEigenValNodeIntel(vx_graph graph,
        vx_image input, vx_enum kernelType, vx_uint32 apertureSize,
        vx_uint32 avgWindow, vx_image output);

/*! \brief [Immediate] Performs the minimal eigenvalue calculation of image blocks.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image in either <tt>\ref VX_DF_IMAGE_U8</tt>,  <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] kernelType Specifies the type of kernel used to compute derivatives, possible values are:<tt>\ref VX_OPERATOR_SOBEL </tt> or <tt>\ref VX_OPERATOR_SCHARR<//tt>.
 * \param [in] apertureSize Size of the derivative operator in pixels, possible values are 3 or 5. Only 3x3 size is available for the Scharr kernel.
 * \param [in] avgWindow Size of the blurring window in pixels, possible values are 3 or 5.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuCornerMinEigenValIntel(vx_context context,
        vx_image input, vx_enum kernelType, vx_uint32 apertureSize,
        vx_uint32 avgWindow, vx_image output);


/*! \brief [Graph] Creates a RANSACLineFitting node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input array of <tt>\ref vx_coordinates_2dt_t</tt> determines the set of input points.
 * \param [in] dist_type Specifies which distance function will be used to weight point. Values are: <tt>\ref VX_CURVE_DISTANCE_L1 </tt>, <tt>\ref VX_CURVE_DISTANCE_L12 </tt>, <tt>\ref VX_CURVE_DISTANCE_FAIR </tt>, <tt>\ref VX_CURVE_DISTANCE_WELSCH </tt>, <tt>\ref VX_CURVE_DISTANCE_HUBER </tt>
 * \param [in] iters_count Number of RANSAC iterations.
 * \param [out] output The output array of detected curve koefficients, three for line or six for ellipse.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxRANSACLineFittingNodeIntel(vx_graph graph,
        vx_array input, vx_enum dist_type, vx_int32 iters_count, vx_array output,
        vx_float32 dist_param);

/*! \brief [Immediate] Performs the RANSAC line fitting.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input array of <tt>\ref vx_coordinates_2dt_t</tt> determines the set of input points.
 * \param [in] dist_type Specifies which distance function will be used to weight point. Values are: <tt>\ref VX_CURVE_DISTANCE_L1 </tt>, <tt>\ref VX_CURVE_DISTANCE_L12 </tt>, <tt>\ref VX_CURVE_DISTANCE_FAIR </tt>, <tt>\ref VX_CURVE_DISTANCE_WELSCH </tt>, <tt>\ref VX_CURVE_DISTANCE_HUBER </tt>
 * \param [in] iters_count Number of RANSAC iterations.
 * \param [out] output The output array of detected curve koefficients, three for line or six for ellipse.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuRANSACLineFittingIntel(vx_context context,
        vx_array input, vx_enum dist_type, vx_int32 iters_count, vx_array output,
        vx_float32 dist_param);

/*! \brief [Graph] Creates StereoBM node.
 * \param [in] graph The reference to the graph.
 * \param [in] left The left rectified image of stereo pair in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [in] right The right rectified image of stereo pair in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [in] minDisp Minimum disparity range.
 * \param [in] maxDisp Maximum disparity range.
 * \param [in] windowSize Size of matching window. Must be odd.
 * \param [out] output The output image with depth map.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxStereoBMNodeIntel(vx_graph graph,
        vx_image left, vx_image right, vx_int32 minDisp, vx_int32 maxDisp,
        vx_int32 windowSize, vx_image output);

/*! \brief [Immediate] Performs the Stereo Block Matching.
 * \param [in] graph The reference to the graph.
 * \param [in] left The left rectified image of stereo pair in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [in] right The right rectified image of stereo pair in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [in] minDisp Minimum disparity range.
 * \param [in] maxDisp Maximum disparity range.
 * \param [in] windowSize Size of matching window. Must be odd.
 * \param [out] output The output image with depth map.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuStereoBMIntel(vx_context context,
        vx_image left, vx_image right, vx_int32 minDisp, vx_int32 maxDisp,
        vx_int32 windowSize, vx_image output);

/*! \brief Start printing debug messages.
 *  \param [in] numZones The number of debug zones
 *  \param [in] zones The array of debug zones from \ref vx_debug_zone_e.
 *  \ingroup group_int_debug
 */
VX_API_ENTRY void VX_API_CALL vxStartTraceIntel(vx_size numZones,
        vx_enum zones[]);

/*! \brief Stop printing debug messages.
 *  \ingroup group_int_debug
 */
VX_API_ENTRY void VX_API_CALL vxStopTraceIntel();

/*! \brief Set zone trace level.
 *  \param [in] zone The debug zone from \ref vx_debug_zone_e.
 *  \param [in] level New trace level
 *  \ingroup group_int_debug
 */
VX_API_ENTRY void VX_API_CALL vxSetZoneTraceLevelIntel(vx_enum zone,
        vx_int32 level);

/*! \brief The trace callback function.
 * \ingroup group_int_debug
 */
typedef void (*vx_trace_callback_intel_f)(void* opaque, vx_enum zone,
        const vx_char* string);

/*! \brief Set trace callback.
 *  \param [in] opaque User data passed to callback, it can be NULL
 *  \param [in] callback The callback function. If NULL, the previous callback is removed.
 *  \ingroup group_int_debug
 */
VX_API_ENTRY void VX_API_CALL vxRegisterTraceCallbackIntel(void* opaque,
        vx_trace_callback_intel_f callback);
/*! \brief Save profile info from specific graph to file.
 *  \param [in] graph Graph.
 *  \param [in] filename Profile info filename.
 *  \ingroup group_int_debug
 */
VX_API_ENTRY vx_status VX_API_CALL vxSaveProfileInfoIntel(vx_graph graph,
        const vx_char* filename);

/*! \brief [Graph] Creates a spherical warper node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in either <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_RGB</tt>, <tt>\ref VX_DF_IMAGE_RGBX</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] radius Radius of a sphere in pixels.
 * \param [in] interpolation Interpolation type. Supported values are: <tt>\ref VX_INTERPOLATION_NEAREST_NEIGHBOR</tt> or <tt>\ref VX_INTERPOLATION_BILINEAR</tt>
 * \param [out] output The output image in the same format as the input image.
 * \ingroup group_vision_function_sphericalwarper
 */
VX_API_ENTRY vx_node VX_API_CALL vxSphericalWarperNodeIntel(vx_graph graph,
        vx_image input, vx_uint32 radius, vx_enum interpolation, vx_image output);

/*! \brief [Immediate] Performs spherical warp of an input image.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image in either <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_RGB</tt>, <tt>\ref VX_DF_IMAGE_RGBX</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] radius Radius of a sphere in pixels.
 * \param [in] interpolation Interpolation type. Supported values are: <tt>\ref VX_INTERPOLATION_NEAREST_NEIGHBOR</tt> or <tt>\ref VX_INTERPOLATION_BILINEAR</tt>
 * \param [out] output The output image in the same format as the input image.
 * \ingroup group_vision_function_sphericalwarper
 */
VX_API_ENTRY vx_status VX_API_CALL vxuSphericalWarperIntel(vx_context context,
        vx_image input, vx_uint32 radius, vx_enum interpolation, vx_image output);

/*! \brief [Graph] Creates a cylindrical warper node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in either <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_RGB</tt>, <tt>\ref VX_DF_IMAGE_RGBX</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] radius Radius of a sphere in pixels.
 * \param [in] interpolation Interpolation type. Supported values are: <tt>\ref VX_INTERPOLATION_NEAREST_NEIGHBOR</tt> or <tt>\ref VX_INTERPOLATION_BILINEAR</tt>
 * \param [out] output The output image in the same format as the input image.
 * \ingroup group_vision_function_cylindricalwarper
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxCylindricalWarperNodeIntel(vx_graph graph,
        vx_image input, vx_uint32 radius, vx_enum interpolation, vx_image output);

/*! \brief [Immediate] Performs cylindrical warp of an input image.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image in either <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_RGB</tt>, <tt>\ref VX_DF_IMAGE_RGBX</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_S16</tt>, or <tt>\ref VX_DF_IMAGE_F32_INTEL</tt> format.
 * \param [in] radius Radius of a sphere in pixels.
 * \param [in] interpolation Interpolation type. Supported values are: <tt>\ref VX_INTERPOLATION_NEAREST_NEIGHBOR</tt> or <tt>\ref VX_INTERPOLATION_BILINEAR</tt>
 * \param [out] output The output image in the same format as the input image.
 * \ingroup group_vision_function_cylindricalwarper
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuCylindricalWarperIntel(vx_context context,
        vx_image input, vx_uint32 radius, vx_enum interpolation, vx_image output);

/*! \brief Creates producer image from externally allocated memory.
 * \details Any access via vxCommitImagePatch marks region of the image as valid so graphs can use it immediately.
 * It is possible to mark region of image as invalid via vxInvalidateProducerImageRect.
 * Any access to image via vxAccessImagePatch, vxCommitImagePatch and vxInvalidateProducerImageRect must be aligned to the tile size.
 * This image can't be used as argument to vxCreateImageFromROI
 * \param [in] context The reference to the implementation context.
 * \param [in] color See the <tt>\ref vx_df_image_e</tt> codes. This mandates the
 * number of planes needed to be valid in the \a addrs and \a ptrs arrays based on the format given.
 * \param [in] addrs[] The array of image patch addressing structures that
 * define the dimension and stride of the array of pointers.
 * \param [in] ptrs[] The array of platform-defined references to each plane.
 * \param [in] import_type <tt>\ref vx_import_type_e</tt>. When giving <tt>\ref VX_IMPORT_TYPE_HOST</tt>
 * the \a ptrs array is assumed to be HOST accessible pointers to memory.
 * \param [in] tileWidth The image tile width in pixels.
 * \param [in] tileHeight The image tile height in pixels.
 * \returns An image reference <tt>\ref vx_image</tt>. Any possible errors preventing a successful
 * creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \see vxAccessImagePatch to obtain direct memory access to the image data.
 * \ingroup group_image
 */
VX_API_ENTRY vx_image VX_API_CALL vxCreateProducerImageFromHandleIntel(
        vx_context context, vx_df_image color, vx_imagepatch_addressing_t addrs[],
        void *ptrs[], vx_enum import_type, vx_uint32 tileWidth,
        vx_uint32 tileHeight);

/*! \brief Creates the producer image.
 * \details Any access via vxCommitImagePatch marks region of the image as valid so graphs can use it immediately.
 * It is possible to mark region of image as invalid via vxInvalidateProducerImageRect.
 * Any access to image via vxAccessImagePatch, vxCommitImagePatch and vxInvalidateProducerImageRect must be aligned to the tile size.
 * This image can't be used as argument to vxCreateImageFromROI
 * \param [in] context The reference to the implementation context.
 * \param [in] width The image width in pixels.
 * \param [in] height The image height in pixels.
 * \param [in] format The VX_DF_IMAGE (<tt>\ref vx_df_image_e</tt>) code that represents the format of the image and the color space.
 * \param [in] tileWidth The image tile width in pixels.
 * \param [in] tileHeight The image tile height in pixels.
 * \returns An image reference <tt>\ref vx_image</tt>. Any possible errors preventing a successful
 * creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \see vxAccessImagePatch to obtain direct memory access to the image data.
 * \ingroup group_image
 */
VX_API_ENTRY vx_image VX_API_CALL vxCreateProducerImageIntel(vx_context context,
        vx_uint32 width, vx_uint32 height, vx_df_image format,
        vx_uint32 tileWidth, vx_uint32 tileHeight);

/*! \brief Invalidates specific regions the producer image.
 * \details If there are executing graphs using image in time of call, behavior is undefined.
 * Only images created via vxCreateProducerImage can be used as input to this function.
 * \param [in] image The reference to image.
 * \param [in] rect The region of interest rectangle. Must contain points within
 * the image pixel space. Can be NULL to invalidate entire image.
 * \returns A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \ingroup group_image
 */
VX_API_ENTRY vx_status VX_API_CALL vxInvalidateProducerImageRectIntel(
        vx_image image, const vx_rectangle_t* rect);

/*! \brief The consumer callback
 */
typedef vx_status (*vx_consumer_callback_intel_f)(void* opaque,
        const vx_rectangle_t* rect);

/*! \brief Creates consumer image from externally allocated memory.
 * \details This image can't be used as argument to vxCreateImageFromROI
 * Callback will be called from unspecified thread and must be synchronized properly
 * \param [in] context The reference to the implementation context.
 * \param [in] color See the <tt>\ref vx_df_image_e</tt> codes. This mandates the
 * number of planes needed to be valid in the \a addrs and \a ptrs arrays based on the format given.
 * \param [in] addrs[] The array of image patch addressing structures that
 * define the dimension and stride of the array of pointers.
 * \param [in] ptrs[] The array of platform-defined references to each plane.
 * \param [in] import_type <tt>\ref vx_import_type_e</tt>. When giving <tt>\ref VX_IMPORT_TYPE_HOST</tt>
 * the \a ptrs array is assumed to be HOST accessible pointers to memory.
 * \param [in] tileWidth The image tile width in pixels.
 * \param [in] tileHeight The image tile height in pixels.
 * \param [in] opaque The opaque pointer which will be passed to callback, can be null
 * \param [in] callback The callback which will be called for every part when it ready
 * \returns An image reference <tt>\ref vx_image</tt>. Any possible errors preventing a successful
 * creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \see vxAccessImagePatch to obtain direct memory access to the image data.
 * \ingroup group_image
 */
VX_API_ENTRY vx_image VX_API_CALL vxCreateConsumerImageFromHandleIntel(
        vx_context context, vx_df_image color, vx_imagepatch_addressing_t addrs[],
        void *ptrs[], vx_enum import_type, vx_uint32 tileWidth,
        vx_uint32 tileHeight, void* opaque, vx_consumer_callback_intel_f callback);

/*! \brief Creates the consumer image.
 * \details This image can't be used as argument to vxCreateImageFromROI
 * Callback will be called from unspecified thread and must be synchronized properly
 * \param [in] context The reference to the implementation context.
 * \param [in] width The image width in pixels.
 * \param [in] height The image height in pixels.
 * \param [in] format The VX_DF_IMAGE (<tt>\ref vx_df_image_e</tt>) code that represents the format of the image and the color space.
 * \param [in] tileWidth The image tile width in pixels.
 * \param [in] tileHeight The image tile height in pixels.
 * \param [in] opaque The opaque pointer which will be passed to callback, can be null
 * \param [in] callback The callback which will be called for every part when it ready
 * \returns An image reference <tt>\ref vx_image</tt>. Any possible errors preventing a successful
 * creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \see vxAccessImagePatch to obtain direct memory access to the image data.
 * \ingroup group_image
 */
VX_API_ENTRY vx_image VX_API_CALL vxCreateConsumerImageIntel(vx_context context,
        vx_uint32 width, vx_uint32 height, vx_df_image format,
        vx_uint32 tileWidth, vx_uint32 tileHeight, void* opaque,
        vx_consumer_callback_intel_f callback);

/*! \brief [Graph] Creates an improved adaptive background subtractor with mixture of gaussians model node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_RGB</tt> format.
 * \param [in] state The state of subtractor with foreground model.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \ingroup group_vision_function_backgroundsubmog2
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxBackgroundSubMOG2NodeIntel(vx_graph graph,
        vx_image input, vx_bg_state_intel state, vx_image output);

/*! \brief [Immediate] Performs background subtraction of input image and updates the model of foreground.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_RGB</tt> format.
 * \param [in] state The state of subtractor with foreground model.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \ingroup group_vision_function_backgroundsubmog2
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuBackgroundSubMOG2Intel(vx_context context,
        vx_image input, vx_bg_state_intel state, vx_image output);

/*! \brief [Immediate] Computes a Bilateral filter on the image.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [in] diameter The diameter of pixel neighborhood.
 * \param [in] sigma_color Filter sigma in the color space.
 * \param [in] sigma_space Filter sigma in the coordinate space.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \ingroup group_vision_function_bilateral_image
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuBilateralIntel(vx_context context,
        vx_image input, vx_uint32 diameter, vx_float32 sigma_color,
        vx_float32 sigma_space, vx_image output);

/*! \brief [Immediate] Detects objects by scanning and classifying sub-windows
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format. Size must be less than 2048x2048.
 * \param [in] classifier_feature Specifies the type of classifier to be used <tt>\ref vx_classifier_feature_type_e</tt>
 * \param [in] object_width  Positive <tt>\ref VX_TYPE_UINT32</tt> width of the sliding window size in pixels
 * \param [in] object_height Positive <tt>\ref VX_TYPE_UINT32</tt> height of the sliding window size in pixels
 * \param [in] stage_sizes The size (number of features) per each cascade stage as <tt>\ref VX_TYPE_UINT32</tt> values array.
 * \param [in] stage_thresholds The threshold per each cascade stage as <tt>\ref VX_TYPE_FLOAT32</tt> values array.
 * \param [in] weak_classifiers The Array of weak classifier structures. The type of structure depends on the <tt>\ref classifier_feature</tt>
 * \param [out] object_positions An output array of object positions as <tt>\ref VX_TYPE_COORDINATES2D</tt>.
 * \param [out] object_confidences [Optional] An output array of object confidences in a <tt>\ref VX_TYPE_FLOAT32</tt>.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuCascadeClassifierIntel(vx_context context,
        vx_image input, vx_enum classifier_feature, vx_uint32 object_width,
        vx_uint32 object_height, vx_array stage_sizes, vx_array stage_thresholds,
        vx_array weak_classifiers, vx_array object_positions,
        vx_array object_confidences);

/*! \brief [Graph] Creates a Box NxN filter node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [in] N The convolution window size in both dimensions (NxN).
 * \param [in] border The structure with border mode used to extrapolate pixels outside of the image.
 * \param [out] output The output image in the same format as the input image.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxBoxNxNNodeIntel(vx_graph graph, vx_image input,
        vx_size N, vx_image output);

/*! \brief [Immediate] Performs a Box NxN filtering.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [in] N The convolution window size in both dimensions (NxN).
 * \param [in] border The structure with border mode used to extrapolate pixels outside of the image.
 * \param [out] output The output image in the same format as the input image.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuBoxNxNIntel(vx_context context,
        vx_image input, vx_size N, vx_image output);

/*! \brief [Graph] Performs a RGB to YCbCr color space conversion.
* \param [in] graph The reference to the graph.
* \param [in] input The input image for R channel in <tt>\ref VX_DF_IMAGE_U8</tt> format.
* \param [in] input The input image for G channel in <tt>\ref VX_DF_IMAGE_U8</tt> format.
* \param [in] input The input image for B channel in <tt>\ref VX_DF_IMAGE_U8</tt> format.
* \param [in] 3x3 matrix coefficients (9 elements). Array element type <tt>\ref VX_TYPE_FLOAT32</tt>
* \param [in] Color conversion offsets (3 elements). Range {0.0, 255.0}. Array element type <tt>\ref VX_TYPE_FLOAT32</tt>
* \param [out] output The output Y channel image in the same format as the input image.
* \param [out] output The output Cb channel image in the same format as the input image.
* \param [out] output The output Cr channel image in the same format as the input image.
* \return <tt>\ref vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
*/
VX_API_ENTRY vx_node VX_API_CALL vxRgbToYCbCrNodeIntel(vx_graph graph, vx_image r,
        vx_image g, vx_image b, vx_array coefficients, vx_array offsets,
        vx_image y, vx_image cb, vx_image cr);

/*! \brief [Immediate] Performs a RGB to YCbCr color space conversion.
* \param [in] graph The reference to the graph.
* \param [in] input The input image for R channel in <tt>\ref VX_DF_IMAGE_U8</tt> format.
* \param [in] input The input image for G channel in <tt>\ref VX_DF_IMAGE_U8</tt> format.
* \param [in] input The input image for B channel in <tt>\ref VX_DF_IMAGE_U8</tt> format.
* \param [in] 3x3 matrix coefficients (9 elements). Array element type <tt>\ref VX_TYPE_FLOAT32</tt>
* \param [in] Color conversion offsets (3 elements). Range {0.0, 255.0}. Array element type <tt>\ref VX_TYPE_FLOAT32</tt>
* \param [out] output The output Y channel image in the same format as the input image.
* \param [out] output The output Cb channel image in the same format as the input image.
* \param [out] output The output Cr channel image in the same format as the input image.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Success
* \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxuRgbToYCbCrIntel(vx_context context,
        vx_image r, vx_image g, vx_image b, vx_array coefficients,
        vx_array offsets, vx_image y, vx_image cb, vx_image cr);

/*! \brief [Graph] Performs a RGB to Lab color space conversion.
* \param [in] graph The reference to the graph.
* \param [in] input The input image for R channel in <tt>\ref VX_DF_IMAGE_U8</tt> format.
* \param [in] input The input image for G channel in <tt>\ref VX_DF_IMAGE_U8</tt> format.
* \param [in] input The input image for B channel in <tt>\ref VX_DF_IMAGE_U8</tt> format.
* \param [in] 3x3 matrix coefficients (9 elements). Array element type <tt>\ref VX_TYPE_FLOAT32</tt>
* \param [in] Reciprocal tristimulus values for X,Y & Z (3 elements). Array element type <tt>\ref VX_TYPE_FLOAT32</tt>
* \param [out] output The output L channel image in the same format as the input image.
* \param [out] output The output a channel image in the same format as the input image.
* \param [out] output The output b channel image in the same format as the input image.
* \return <tt>\ref vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
*/
VX_API_ENTRY vx_node VX_API_CALL vxRgbToLabNodeIntel(vx_graph graph, vx_image r,
        vx_image g, vx_image b, vx_array coefficients, vx_array recip,
        vx_image d_l, vx_image d_a, vx_image d_b);

/*! \brief [Immediate] Performs a RGB to Lab color space conversion.
* \param [in] graph The reference to the graph.
* \param [in] input The input image for R channel in <tt>\ref VX_DF_IMAGE_U8</tt> format.
* \param [in] input The input image for G channel in <tt>\ref VX_DF_IMAGE_U8</tt> format.
* \param [in] input The input image for B channel in <tt>\ref VX_DF_IMAGE_U8</tt> format.
* \param [in] 3x3 matrix coefficients (9 elements). Array element type <tt>\ref VX_TYPE_FLOAT32</tt>
* \param [in] Reciprocal tristimulus values for X,Y & Z color (3 elements). Array element type <tt>\ref VX_TYPE_FLOAT32</tt>
* \param [out] output The output L channel image in the same format as the input image.
* \param [out] output The output a channel image in the same format as the input image.
* \param [out] output The output b channel image in the same format as the input image.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Success
* \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxuRgbToLabIntel(vx_context context,
        vx_image r, vx_image g, vx_image b, vx_array coefficients, vx_array recip,
        vx_image d_l, vx_image d_a, vx_image d_b);

/*==============================================================================
 SVM PARAMETERS
 =============================================================================*/

/*! \brief Creates a reference to a set of parameters for SVM classifier
 * with C_SVC mode and RBF kernel.
 * \param [in] context The reference to the overall context.
 * \return A svm_params reference <tt>\ref vx_bg_state_intel</tt>.
 * \ingroup group_svm_params
 */
VX_API_ENTRY vx_svm_params_intel VX_API_CALL vxCreateSVMParamsDefaultIntel(
        vx_context context);

/*! \brief Creates a reference to a set of parameters for SVM classifier
 * with linear kernel.
 * \param [in] context The reference to the overall context.
 * \param [in] svm_mode The mode  of classifier's working.
 * \param [in] classes_num The number of classes.
 * \return A svm_params reference <tt>\ref vx_bg_state_intel</tt>.
 * \ingroup group_svm_params
 */
VX_API_ENTRY vx_svm_params_intel VX_API_CALL vxCreateSVMParamsForLinearKernelIntel(
        vx_context context, vx_enum svm_mode, vx_uint32 classes_num);

/*! \brief Creates a reference to a set of parameters for SVM classifier
 * with polynomial kernel.
 * \param [in] context The reference to the overall context.
 * \param [in] svm_mode The mode  of classifier's working.
 * \param [in] classes_num The number of classes.
 * \param [in] degree The degree of polynomial.
 * \param [in] gamma The parameter gamma.
 * \param [in] coef0 The parameter gamma0.
 * \return A svm_params reference <tt>\ref vx_bg_state_intel</tt>.
 * \ingroup group_svm_params
 */
VX_API_ENTRY vx_svm_params_intel VX_API_CALL vxCreateSVMParamsForPolyKernelIntel(
        vx_context context, vx_enum svm_mode, vx_uint32 classes_num,
        vx_float64 degree, vx_float64 gamma, vx_float64 coef0);

/*! \brief Creates a reference to a set of parameters for SVM classifier
 * with RBF kernel.
 * \param [in] context The reference to the overall context.
 * \param [in] svm_mode The mode  of classifier's working.
 * \param [in] classes_num The number of classes.
 * \param [in] gamma The parameter gamma.
 * \return A svm_params reference <tt>\ref vx_bg_state_intel</tt>.
 * \ingroup group_svm_params
 */
VX_API_ENTRY vx_svm_params_intel VX_API_CALL vxCreateSVMParamsForRBFKernelIntel(
        vx_context context, vx_enum svm_mode, vx_uint32 classes_num,
        vx_float64 gamma);

/*! \brief Creates a reference to a set of parameters for SVM classifier
 * with sigmoid kernel.
 * \param [in] context The reference to the overall context.
 * \param [in] svm_mode The mode  of classifier's working.
 * \param [in] classes_num The number of classes.
 * \param [in] gamma The parameter gamma.
 * \param [in] coef0 The parameter gamma0.
 * \return A svm_params reference <tt>\ref vx_bg_state_intel</tt>.
 * \ingroup group_svm_params
 */
VX_API_ENTRY vx_svm_params_intel VX_API_CALL vxCreateSVMParamsForSigmoidKernelIntel(
        vx_context context, vx_enum svm_mode, vx_uint32 classes_num,
        vx_float64 gamma, vx_float64 coef0);

/*! \brief Creates a reference to a set of parameters for SVM classifier
 * with specified types of svm and kernel.
 * \param [in] context The reference to the overall context.
 * \param [in] svm_mode The mode  of classifier's working.
 * \param [in] kernel_type The type of kernel.
 * \param [in] classes_num The number of classes.
 * \param [in] degree The degree of polynomial.
 * \param [in] gamma The parameter gamma.
 * \param [in] coef0 The parameter gamma0.
 * \return A svm_params reference <tt>\ref vx_bg_state_intel</tt>.
 * \ingroup group_svm_params
 */
VX_API_ENTRY vx_svm_params_intel VX_API_CALL vxCreateSpecificSVMParamsIntel(
        vx_context context, vx_enum svm_mode, vx_enum kernel_type,
        vx_uint32 classes_num, vx_float64 degree, vx_float64 gamma,
        vx_float64 coef0);

/*! \brief Releases a reference to a SVM`s parameters object.
 * \param [in] svm_params The pointer to the object to release.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE If svm_params is not a <tt>\ref svm_params</tt>.
 * \ingroup group_svm_params
 */
VX_API_ENTRY vx_status VX_API_CALL vxReleaseSVMParamsIntel(
        vx_svm_params_intel *svm_params);

/*! \brief [Graph] Creates support vector machine classifier.
 * \param [in] graph The reference to the graph.
 * \param [in] samples The set of objects for classification (each row is characteristic vector of one object).
 * \param [in] support_vectors The set of support vectors (each row is one support vector).
 * \param [in] dual_vars The set of dual variables needed for support vectors.
 * \param [in] w0 Displacement of the hyperplane from the origin.
 * \param [in] labels The array of classes' names.
 * \param [in] params The structure which contains work parameters of SVM.
 * \param [out] output The array where will be written labels of classes for samples.
 * \ingroup group_svm_classifier
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxSVMClassifierNodeIntel(vx_graph graph,
        vx_image samples, vx_image support_vectors, vx_array dual_vars,
        vx_float64 w0, vx_array labels, vx_svm_params_intel params, vx_array output);

/*! \brief [Immediate] Performs classification using support vector machine.
 * \param [in] context The reference to the overall context.
 * \param [in] samples The set of objects for classification (each row is characteristic vector of one object).
 * \param [in] support_vectors The set of support vectors (each row is one support vector).
 * \param [in] dual_vars The set of dual variables needed for support vectors.
 * \param [in] w0 Displacement of the hyperplane from the origin.
 * \param [in] labels The array of classes' names.
 * \param [in] params The structure which contains work parameters of SVM.
 * \param [out] output The array where will be written labels of classes for samples.
 * \ingroup group_svm_classifier
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuSVMClassifierIntel(vx_context context,
        vx_image samples, vx_image support_vectors, vx_array dual_vars,
        vx_float64 w0, vx_array labels, vx_svm_params_intel params, vx_array output);

/*==============================================================================
 SEPFILTER2D
 =============================================================================*/

/*! \brief Creates a reference to a sepfilter2d object.
 * \param [in] context The reference to the overall context.
 * \param [in] rows The dimension of the kernel for filtering rows.
 * \param [in] columns The dimension of the kernel for filtering columns.
 * \returns A sepfilter2d reference <tt>\ref vx_sepfilter2d_intel</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_sepfilter2d
 */
VX_API_ENTRY vx_sepfilter2d_intel VX_API_CALL vxCreateSepFilter2DIntel(
        vx_context context, vx_size rows, vx_size columns);

/*! \brief Releases the reference to a sepfilter2d object.
 * The object may not be garbage collected until its total reference count is zero.
 * \param [in] sepFilter2D The pointer to the sepfilter2d object to release.
 * \post After returning from this function the reference is zeroed.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE If sepFilter2D is not a <tt>\ref vx_sepfilter2d_intel</tt>.
 * \ingroup group_sepfilter2d
 */
VX_API_ENTRY vx_status VX_API_CALL vxReleaseSepFilter2DIntel(
        vx_sepfilter2d_intel* sepFilter2D);

/*! \brief Queries an attribute on the sepfilter2d object.
 * \param [in] sepFilter2D The sepfilter2d object to query.
 * \param [in] attribute The attribute to query. Use a <tt>\ref vx_sepfilter2d_attribute_e</tt> enumeration.
 * \param [out] ptr The location at which to store the resulting value.
 * \param [in] size The size in bytes of the container to which \a ptr points.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \ingroup group_sepfilter2d
 */
VX_API_ENTRY vx_status VX_API_CALL vxQuerySepFilter2DIntel(
        vx_sepfilter2d_intel sepFilter2D, vx_enum attribute, void *ptr,
        vx_size size);

/*! \brief Sets attributes on the sepfilter2d object.
 * \param [in] sepFilter2D The reference to the sepFilter2D.
 * \param [in] attribute The attribute to modify. Use a <tt>\ref vx_sepfilter2d_attribute_e</tt> enumeration.
 * \param [in] ptr The pointer to the value to which to set the attribute.
 * \param [in] size The size in bytes of the data pointed to by \a ptr.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \ingroup group_sepfilter2d
 */
VX_API_ENTRY vx_status VX_API_CALL vxSetSepFilter2DAttributeIntel(
        vx_sepfilter2d_intel sepFilter2D, vx_enum attribute, const void* ptr,
        vx_size size);

/*! \brief Gets the sepFilter2D kernel data for filtering rows (copy)
 * \param [in] sepFilter2D The reference to the sepFilter2D.
 * \param [in] array The array to place the kernel.
 * \see <tt>\ref vxQuerySepFilter2D</tt> and <tt>\ref VX_SEPFILTER2D_ATTRIBUTE_ROW_KERNEL_SIZE</tt> to get the
 * needed number of bytes of the array.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \ingroup group_sepfilter2d
 */
VX_API_ENTRY vx_status VX_API_CALL vxReadSepFilter2DRowKernelIntel(
        vx_sepfilter2d_intel sepFilter2D, vx_int16* array);


/*! \brief Gets the sepFilter2D kernel data for filtering columns (copy)
 * \param [in] sepFilter2D The reference to the sepFilter2D.
 * \param [in] array The array to place the kernel.
 * \see <tt>\ref vxQuerySepFilter2D</tt> and <tt>\ref VX_SEPFILTER2D_ATTRIBUTE_COLUMN_KERNEL_SIZE</tt> to get the
 * needed number of bytes of the array.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \ingroup group_sepfilter2d
 */
VX_API_ENTRY vx_status VX_API_CALL vxReadSepFilter2DColumnKernelIntel(
        vx_sepfilter2d_intel sepFilter2D, vx_int16* array);


/*! \brief Sets the sepFilter2D kernel for filtering rows (copy)
 * \param [in] sepFilter2D The reference to the sepFilter2D.
 * \param [in] array The array containing the kernel to be written.
 * \see <tt>\ref vxQuerySepFilter2D</tt> and <tt>\ref VX_SEPFILTER2D_ATTRIBUTE_ROW_KERNEL_SIZE</tt> to get the
 * needed number of bytes of the array.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \ingroup group_sepfilter2d
 */
VX_API_ENTRY vx_status VX_API_CALL vxWriteSepFilter2DRowKernelIntel(
        vx_sepfilter2d_intel sepFilter2D, const vx_int16* array);


/*! \brief Sets the sepFilter2D kernel for filtering columns (copy)
 * \param [in] sepFilter2D The reference to the sepFilter2D.
 * \param [in] array The array containing the kernel to be written.
 * \see <tt>\ref vxQuerySepFilter2D</tt> and <tt>\ref VX_SEPFILTER2D_ATTRIBUTE_COLUMN_KERNEL_SIZE</tt> to get the
 * needed number of bytes of the array.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \ingroup group_sepfilter2d
 */
VX_API_ENTRY vx_status VX_API_CALL vxWriteSepFilter2DColumnKernelIntel(
        vx_sepfilter2d_intel sepFilter2D, const vx_int16* array);


/*==============================================================================
 BACKGROUND SUBTRACTOR`S STATE FROM IPP9
 =============================================================================*/

/*! \brief Creates a reference to a background subtractor state.
 * \param [in] context The reference to the overall context.
 * \param [in] width The width of source image.
 * \param [in] height The height of source image.
 * \param [in] numFrames The length of the history.
 * \param [in] numGauss The maximum number of gaussian components per pixel.
 * \param [in] varInit Initial value of variance for new gaussian component.
 * \param [in] varMin Minimal bound of variance.
 * \param [in] varMax Maximal bound of variance.
 * \param [in] varWBRatio Background threshold.
 * \param [in] bckgThr Background total weights sum threshold.
 * \param [in] varNGRatio Threshold for adding new gaussian component to list.
 * \param [in] reduction Speed of reduction non-active gaussian components.
 * \param [in] shadowFlag Search shadows flag.
 * \param [in] shadowValue Returned shadow value.
 * \param [in] shadowRatio Shadow threshold.
 * \return A bg_state reference <tt>\ref vx_bg_state_intel</tt>.
 * \ingroup group_bg_state
 */
VX_API_ENTRY vx_bg_state_intel VX_API_CALL vxCreateBGStateIntel(vx_context context,
        vx_uint32 width, vx_uint32 height, vx_uint32 numFrames, /* length of history */
        vx_uint32 numGauss, /* maximal number of gaussian components per pixel */
        /* (numGaussPerPixel<=numGauss) */
        vx_float32 varInit, /* initial value of variance for new gaussian component */
        vx_float32 varMin, /* minimal bound of variance */
        vx_float32 varMax, /* maximal bound of variance */
        vx_float32 varWBRatio, /* background threshold */
        vx_float32 bckgThr, /* background total weights sum threshold */
        vx_float32 varNGRatio, /* threshold for adding new gaussian component to list */
        vx_float32 reduction, /* speed of reduction non-active gaussian components */
        vx_char shadowFlag, /* search shadows flag */
        vx_char shadowValue, /* returned shadow value */
        vx_float32 shadowRatio); /* shadow threshold */

/*! \brief Creates a reference to a background subtractor state
 * with parameters of model by default.
 * \param [in] context The reference to the overall context.
 * \param [in] width The width of source image.
 * \param [in] height The height of source image.
 * \return A bg_state reference <tt>\ref vx_bg_state_intel</tt>.
 * \ingroup group_bg_state
 */
VX_API_ENTRY vx_bg_state_intel VX_API_CALL vxCreateBGStateDefaultIntel(
        vx_context context, vx_uint32 width, vx_uint32 height);

/*! \brief Resets foreground model of subtractor`s state.
 * \param [in] bg_state The state needed to reset.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_NOT_ALLOCATED This bg_state has not been created before resetting.
 * \retval VX_FAILURE Unknown error.
 * \ingroup group_bg_state
 */
VX_API_ENTRY vx_status VX_API_CALL vxResetBGStateIntel(vx_bg_state_intel bg_state);

/*! \brief Checks readiness of subtractor`s state.
 * \param [in] bg_state The vx_bg_state_intel which is needed checking.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_NOT_SUFFICIENT If bg_state was not created.
 * \ingroup group_bg_state
 */
VX_API_ENTRY vx_status VX_API_CALL vxCheckBGStateIntel(vx_bg_state_intel bg_state);

/*! \brief Releases a reference to a subtractor`s state object.
 * \param [in] bg_state The pointer to the state to release.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE If bg_state is not a <tt>\ref vx_bg_state_intel</tt>.
 * \ingroup group_bg_state
 */
VX_API_ENTRY vx_status VX_API_CALL vxReleaseBGStateIntel(vx_bg_state_intel *bg_state);

/*! \brief Calculates an absolute difference norm, or a relative difference norm.
 * \param [in] graph The reference to the graph.
 * \param [in] mag The input image which stores the magnitude (radius) components of the elements
 *  in polar coordinate form,VX_DF_IMAGE_S16.
 * \param [in] orientation The input image which stores the phase (angle) components of the
 *  elements in polar coordinate form in radians, VX_DF_IMAGE_U8 or VX_DF_IMAGE_S16.
 * \param [out] grad_x The output image which stores grad_x, VX_DF_IMAGE_S16.
 * \param [out] grade_y The output image which stores grad_y, VX_DF_IMAGE_S16.
 * \return vx node A node reference.
 * \Any possible errors preventing a successful creation should be checked using vxGetStatus
 */

VX_API_ENTRY vx_node VX_API_CALL vxPolarToCartNodeIntel(vx_graph graph,
        vx_image mag, vx_image orientation, vx_image grade_x, vx_image grade_y);

/*! \brief Performs stereo matching using the block matching algorithm.
 * \param [in] graph The reference to the graph.
 * \param [in] left_image The input left image VX_DF_IMAGE_U8
 * \param [in] right_image The input right image VX_DF_IMAGE_U8.
 * \param [in] minimum_disparity The starting disparity in search for the minimum distance between
 * corresponding pixels.
 * \param [in] maximum_disparity The end disparity in search for the minimum distance between
 * corresponding pixels.
 * \param [in] block_size The height and width of the block around the pixel (centred) over which the
 * distance (norm) is computed for each disparity calculation.
 * \param [in] norm_type The norm type use for norm computation: VX_HAMMING_INTEL/VX_L1_INTEL/
 * VX_L2_INTEL/VX_CCORR_INTEL. see vx_comp_metric_intel_e.
 * \param [out] disparity The output disparity map VX_DF_IMAGE_U8 , maximum value of disparity is
 * 256 , or VX_DF_IMAGE_S16
 * \param [out] distance The output distance of the minimum disparity reported in disparity
 * VX_DF_IMAGE_S16 . If Null distance image is not generated.
 * \return vx node A node reference.
 * \Any possible errors preventing a successful creation should be checked using vxGetStatus
 */

VX_API_ENTRY vx_node VX_API_CALL vxBlockMatchingNodeIntel(vx_graph graph,
        vx_image left_image, vx_image right_image, vx_uint8 minimum_disparity,
        vx_uint8 maximum_disparity, vx_uint8 block_size, vx_enum norm_type,
        vx_image disparity, vx_image distance);

/*! \brief [Graph] Halftone CMYK channels of the input image using error diffusion dithering pattern.
 * \param [in] graph The reference to the graph.
 * \param [in] inputCMYK The input CMYK image (<tt>\ref VX_DF_IMAGE_RGBX</tt>)
 * \param [out] outputCMYK The output CMYK image (<tt>\ref VX_DF_IMAGE_RGBX</tt>)
 * \note Since OpenVX does not support images of CMYK type, you can use any 4-channel image like a <tt>\ref VX_DF_IMAGE_RGBX</tt> as a proxy.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxErrorDiffusionCMYKNodeIntel(vx_graph graph,
        vx_image inputCMYK, vx_image outputCMYK);

/*! \brief [Graph] Halftone CMYK channels of the input image using error diffusion dithering pattern.
 * \param [in] graph The reference to the graph.
 * \param [in] inputCMYK The input CMYK image (<tt>\ref VX_DF_IMAGE_RGBX</tt>)
 * \param [out] outputC The output CMYK image C channel (<tt>\ref VX_DF_IMAGE_U8</tt>)
 * \param [out] outputM The output CMYK image M channel (<tt>\ref VX_DF_IMAGE_U8</tt>)
 * \param [out] outputY The output CMYK image Y channel (<tt>\ref VX_DF_IMAGE_U8</tt>)
 * \param [out] outputK The output CMYK image K channel (<tt>\ref VX_DF_IMAGE_U8</tt>)
 * \note Since OpenVX does not support images of CMYK type, you can use any 4-channel image like a <tt>\ref VX_DF_IMAGE_RGBX</tt> as a proxy.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxErrorDiffusionCMYKPlanarNodeIntel(vx_graph graph,
        vx_image inputCMYK, vx_image outputC, vx_image outputM, vx_image outputY, vx_image outputK);

/*! \brief [Graph] Performs Error Diffusion on the input image.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_U16</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [in] levels The number of output levels for halftoning (dithering).
 * \param [in] noise The number specifying the amount of noise added (0..100).
 * \param [in] dither An dithering type from <tt>\ref vx_dither_pi_intel_e</tt> enumeration.
 * \param [out] output The output image of the same format as input image.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxErrorDiffusionNodeIntel(vx_graph graph,
        vx_image input, vx_int32 levels, vx_int32 noise,
        vx_dither_pi_intel_e dither, vx_image output);

/*! \brief [Immediate] Performs Error Diffusion on the input image.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_U16</tt>, <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [in] levels The number of output levels for halftoning (dithering).
 * \param [in] noise The number specifying the amount of noise added (0..100).
 * \param [in] dither Dithering type <tt>\ref VX_TYPE_ENUM</tt> of the <tt>\ref vx_ed_dither</tt> enumeration.
 * \param [out] output The output image in the same format as input image.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success.
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
 */
VX_API_ENTRY vx_status VX_API_CALL vxuErrorDiffusionIntel(vx_context context,
        vx_image input, vx_int32 levels, vx_int32 noise,
        vx_dither_pi_intel_e dither, vx_image output);

/*! \brief [Graph] Makes halftoning of input image.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image of <tt>\ref VX_DF_IMAGE_U8</tt> format
 * \param [in] thresholds The image of VX_DF_IMAGE_U8 type with threshold values. This values must be set at the moment of graph compilation.
 * \param [in] shift Horizontal shift for threshold blocks. Calculated as (y / ThresholdHeight) * shift
 * \param [out] output The output image of the same format as input image.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxHalftoneNodeIntel(vx_graph graph,
        vx_image input, vx_image thresholds, vx_int32 shift, vx_image output);

/*! \brief [Graph] Applies 3-dimensional NxNxN look-up table to the image.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image of <tt>\ref VX_DF_IMAGE_RGB</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt> format.
 * \param [in] interpolation Interpolation type. Supported values are: <tt>\ref VX_INTERPOLATION_NEAREST_NEIGHBOR</tt> or <tt>\ref VX_INTERPOLATION_TRILINEAR_INTEL</tt> or <tt>\ref VX_INTERPOLATION_TETRAHEDRAL_INTEL</tt>
 * \param [in] nlatticepoints The number of lattice points defined in the 3d lut. Must be in the range [2,33]
 * \param [in] xmap Optional 256-entry array of type <tt>\ref VX_TYPE_FLOAT32</tt>. Used to map input image channel 0 to the lattice point index range (0:nlatticepoints-1)
 *             A default identity mapping will be used if xmap is 0.
 * \param [in] ymap Optional 256-entry array of type <tt>\ref VX_TYPE_FLOAT32</tt>. Used to map input image channel 1 to the lattice point index range (0:nlatticepoints-1)
 *             A default identity mapping will be used if ymap is 0.
 * \param [in] zmap Optional 256-entry array of type <tt>\ref VX_TYPE_FLOAT32</tt>. Used to map input image channel 2 to the lattice point index range (0:nlatticepoints-1)
 *             A default identity mapping will be used if zmap is 0.
 * \param [in] lut3d The packed lookup table containing values of <tt>\ref VX_TYPE_UINT8</tt> type. If the output image type is VX_DF_IMAGE_RGB, the number
 *  of elements contained within this array depends on the output format of the image:
 *   If output format is VX_DF_IMAGE_U8, number of elements is expected to be nlatticepoints*nlatticepoints*nlatticepoints.
 *   If output format is VX_DF_IMAGE_RGB, number of elements is expected to be nlatticepoints*nlatticepoints*nlatticepoints*3.
 *   If output format is VX_DF_IMAGE_RGBX, number of elements is expected to be nlatticepoints*nlatticepoints*nlatticepoints*4.
 * \param [out] output The output image of <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_RGB</tt>, <tt>\ref VX_DF_IMAGE_RGBX</tt> format.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxLUT3DNodeIntel(vx_graph graph, vx_image input,
        vx_enum interpolation, vx_int32 nlatticepoints,
        vx_array xmap, vx_array ymap, vx_array zmap,
        vx_array lut3d, vx_image output);

/*! \brief [Graph] Applies 3-dimensional NxNxN look-up table to the image.
 * \param [in] context The reference to the overall context.
 * \param [in] input The input image of <tt>\ref VX_DF_IMAGE_RGB</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt> format.
 * \param [in] interpolation Interpolation type. Supported values are: <tt>\ref VX_INTERPOLATION_NEAREST_NEIGHBOR</tt> or <tt>\ref VX_INTERPOLATION_TRILINEAR_INTEL</tt> or <tt>\ref VX_INTERPOLATION_TETRAHEDRAL_INTEL</tt>
 * \param [in] nlatticepoints The number of lattice points defined in the 3d lut. Must be in the range [2,33]
 * \param [in] xmap Optional 256-entry array of type <tt>\ref VX_TYPE_FLOAT32</tt>. Used to map input image channel 0 to the lattice point index range (0:nlatticepoints-1)
 *             A default identity mapping will be used if xmap is 0.
 * \param [in] ymap Optional 256-entry array of type <tt>\ref VX_TYPE_FLOAT32</tt>. Used to map input image channel 1 to the lattice point index range (0:nlatticepoints-1)
 *             A default identity mapping will be used if ymap is 0.
 * \param [in] zmap Optional 256-entry array of type <tt>\ref VX_TYPE_FLOAT32</tt>. Used to map input image channel 2 to the lattice point index range (0:nlatticepoints-1)
 *             A default identity mapping will be used if zmap is 0.
 * \param [in] lut3d The packed lookup table containing values of <tt>\ref VX_TYPE_UINT8</tt> type. If the output image type is VX_DF_IMAGE_RGB, the number
 *  of elements contained within this array depends on the output format of the image:
 *   If output format is VX_DF_IMAGE_U8, number of elements is expected to be nlatticepoints*nlatticepoints*nlatticepoints.
 *   If output format is VX_DF_IMAGE_RGB, number of elements is expected to be nlatticepoints*nlatticepoints*nlatticepoints*3.
 *   If output format is VX_DF_IMAGE_RGBX, number of elements is expected to be nlatticepoints*nlatticepoints*nlatticepoints*4.
 * \param [out] output The output image of <tt>\ref VX_DF_IMAGE_U8</tt>, <tt>\ref VX_DF_IMAGE_RGB</tt>, <tt>\ref VX_DF_IMAGE_RGBX</tt> format.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Success.
 * \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxuLUT3DIntel(vx_context context, vx_image input,
        vx_enum interpolation, vx_int32 nlatticepoints,
        vx_array xmap, vx_array ymap, vx_array zmap,
        vx_array lut3d, vx_image output);

/*! \brief [Graph] Creates a cascade classifier node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format. Size must be less than 2048x2048.
 * \param [in] classifier_feature Specifies the type of classifier to be used <tt>\ref vx_classifier_feature_type_e</tt>
 * \param [in] object_width  Positive <tt>\ref VX_TYPE_UINT32</tt> width of the sliding window size in pixels
 * \param [in] object_height Positive <tt>\ref VX_TYPE_UINT32</tt> height of the sliding window size in pixels
 * \param [in] stage_sizes The size (number of features) per each cascade stage as <tt>\ref VX_TYPE_UINT32</tt> values array.
 * \param [in] stage_thresholds The threshold per each cascade stage as <tt>\ref VX_TYPE_FLOAT32</tt> values array.
 * \param [in] weak_classifiers The Array of weak classifier structures. The type of structure depends on the <tt>\ref classifier_feature</tt>
 * \param [out] object_positions An output array of object positions as <tt>\ref VX_TYPE_COORDINATES2D</tt>.
 * \param [out] object_confidences [Optional] An output array of object confidences in a <tt>\ref VX_TYPE_FLOAT32</tt>.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxCascadeClassifierNodeIntel(vx_graph  graph,
                                                         vx_image  input,
                                                         vx_enum   classifier_feature,
                                                         vx_uint32 object_width,
                                                         vx_uint32 object_height,
                                                         vx_array  stage_sizes,
                                                         vx_array  stage_thresholds,
                                                         vx_array  weak_classifiers,
                                                         vx_array  object_positions,
                                                         vx_array  object_confidences);

/*! \brief [Graph] Create probabilistic Hough lines detection node.
 * \param [in] graph The reference to the graph.
 * \param [in] input An Input image in <tt>\ref VX_DF_IMAGE_U8</tt> format
 * \param [in] deltaRho Non-negative <tt>\ref VX_TYPE_FLOAT32</tt> step of discretization of radial coordinate
 * \param [in] deltaTheta Non-negative <tt>\ref VX_TYPE_FLOAT32</tt> step of discretization of angular coordinate
 * \param [in] threshold Input <tt>\ref VX_TYPE_INT32</tt> minimum number of points that are required to detect the line
 * \param [in] lineLen Minimum length of the line
 * \param [in] lineGap Maximum length of the gap between lines.
 * \param [in] maxcount Input <tt>\ref VX_TYPE_INT32</tt> maximum number of lines to be stored
 * \param [out] lines Detected lines stored in <tt>\ref vx_array</tt> of type <tt>\ref vx_rectangle_t<\tt>.
 * \param [out] count Output <tt>\ref VX_TYPE_INT32</tt> number of detected lines [optional]
 * \ingroup group_vision_function_houghlines_p
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxHoughLinesPNodeIntel(vx_graph graph,
        vx_image input, vx_float32 deltaRho, vx_float32 deltaTheta,
        vx_int32 threshold, vx_int32 lineLen, vx_int32 lineGap,
        vx_int32 maxcount, vx_array lines, vx_scalar count);

/*! \brief [Graph] Creates a match template node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in either <tt>\ref VX_DF_IMAGE_U8</tt>,  <tt>\ref VX_DF_IMAGE_U16</tt>, or <tt>\ref VX_DF_IMAGE_F32</tt> format.
 * \param [in] templ The template image in the same format as the input image.
 * \param [in] match_algorithm The match template algorithm to use. <tt>\ref VX_TM_SQDIFF_INTEL</tt>, <tt>\ref VX_TM_CRCORR_INTEL</tt> and <tt>\ref VX_TM_CRCOEFF_INTEL</tt> are supported
 * \param [in] impl_algorithm Implementation variant of algorithm to use for calculation. <tt>\ref VX_MATCH_TEMPLATE_DIRECT_INTEL</tt> and <tt>\ref VX_MATCH_TEMPLATE_FFT_INTEL</tt> are supported
 * \param [in] normalize If true, output normalization will be used (not used in case of VX_TM_CRCOEFF_INTEL).
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_F32</tt> format.
 * \ingroup group_vision_function_match_template
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxMatchTemplateNodeIntel(vx_graph graph,
            vx_image input, vx_image templ, vx_enum match_algorithm,
            vx_enum impl_algorithm, vx_bool normalize, vx_image output);

/*! \brief [Graph] Creates a Bilateral Filter Node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [in] diameter The diameter of pixel neighborhood.
 * \param [in] sigma_color Filter sigma in the color space.
 * \param [in] sigma_space Filter sigma in the coordinate space.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \ingroup group_vision_function_bilateral_image
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxBilateralNodeIntel(vx_graph graph,
         vx_image input, vx_uint32 diameter, vx_float32 sigma_color,
         vx_float32 sigma_space, vx_image output);

/*! \brief [Graph] Creates a 8 bpp to 1 bpp bit depth conversion node.
* \param [in] graph The reference to the graph.
* \param [in] input The  input image , <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \param [out] output The output image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \return <tt>\ref vx_node</tt>
* \retval vx_node A node reference.
*/
VX_API_ENTRY vx_node VX_API_CALL vxPack8to1NodeIntel(vx_graph graph,
    vx_image input, vx_image output);

/*! \brief [Immediate] Performs the 8 bpp to 1 bpp bit depth conversion operation.
* \param [in] context The reference to the overall context.
* \param [in] input The  input image , <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \param [out] output The output image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Success
* \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxuPack8to1Intel(vx_context context,
    vx_image input, vx_image output);

/*! \brief [Graph] Creates a 8 bpp to 2 bpp bit depth conversion node.
* \param [in] graph The reference to the graph.
* \param [in] input The  input image , <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \param [out] output The output image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \return <tt>\ref vx_node</tt>
* \retval vx_node A node reference.
*/
VX_API_ENTRY vx_node VX_API_CALL vxPack8to2NodeIntel(vx_graph graph,
    vx_image input, vx_image output);

/*! \brief [Immediate] Performs the 8 bpp to 2 bpp bit depth conversion operation.
* \param [in] context The reference to the overall context.
* \param [in] input The  input image , <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \param [out] output The output image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Success
* \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxuPack8to2Intel(vx_context context,
    vx_image input, vx_image output);

/*! \brief [Graph] Creates a 8 bpp to 4 bpp bit depth conversion node.
* \param [in] graph The reference to the graph.
* \param [in] input The  input image , <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \param [out] output The output image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \return <tt>\ref vx_node</tt>
* \retval vx_node A node reference.
*/
VX_API_ENTRY vx_node VX_API_CALL vxPack8to4NodeIntel(vx_graph graph,
    vx_image input, vx_image output);

/*! \brief [Immediate] Performs the 8 bpp to 4 bpp bit depth conversion operation.
* \param [in] context The reference to the overall context.
* \param [in] input The  input image , <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \param [out] output The output image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Success
* \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxuPack8to4Intel(vx_context context,
    vx_image input, vx_image output);

/*! \brief [Graph] Creates a 1 bpp to 8 bpp bit depth conversion node.
* \param [in] graph The reference to the graph.
* \param [in] input The  input image , <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \param [out] output The output image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \return <tt>\ref vx_node</tt>
* \retval vx_node A node reference.
*/
VX_API_ENTRY vx_node VX_API_CALL vxUnpack1to8NodeIntel(vx_graph graph,
    vx_image input, vx_image output);

/*! \brief [Immediate] Performs the 1 bpp to 8 bpp bit depth conversion operation.
* \param [in] context The reference to the overall context.
* \param [in] input The  input image , <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \param [out] output The output image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Success
* \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxuUnpack1to8Intel(vx_context context,
    vx_image input, vx_image output);

/*! \brief [Graph] Creates a 2 bpp to 8 bpp bit depth conversion node.
* \param [in] graph The reference to the graph.
* \param [in] input The  input image , <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \param [out] output The output image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \return <tt>\ref vx_node</tt>
* \retval vx_node A node reference.
*/
VX_API_ENTRY vx_node VX_API_CALL vxUnpack2to8NodeIntel(vx_graph graph,
    vx_image input, vx_image output);

/*! \brief [Immediate] Performs the 2 bpp to 8 bpp bit depth conversion operation.
* \param [in] context The reference to the overall context.
* \param [in] input The  input image , <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \param [out] output The output image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Success
* \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxuUnpack2to8Intel(vx_context context,
    vx_image input, vx_image output);

/*! \brief [Graph] Creates a 4 bpp to 8 bpp bit depth conversion node.
* \param [in] graph The reference to the graph.
* \param [in] input The  input image , <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \param [out] output The output image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \return <tt>\ref vx_node</tt>
* \retval vx_node A node reference.
*/
VX_API_ENTRY vx_node VX_API_CALL vxUnpack4to8NodeIntel(vx_graph graph,
    vx_image input, vx_image output);

/*! \brief [Immediate] Performs the 4 bpp to 8 bpp bit depth conversion operation.
* \param [in] context The reference to the overall context.
* \param [in] input The  input image , <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \param [out] output The output image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGBX</tt>.
* \return A <tt>\ref vx_status_e</tt> enumeration.
* \retval VX_SUCCESS Success
* \retval * An error occurred. See <tt>\ref vx_status_e</tt>.
*/
VX_API_ENTRY vx_status VX_API_CALL vxuUnpack4to8Intel(vx_context context,
    vx_image input, vx_image output);

/*! \brief Used to retrieve a target reference by the index of the target.
 * \param [in] context The reference to the overall context.
 * \param [in] index The index of the target to get a reference to.
 * \return <tt>\ref vx_target_intel</tt>
 * \retval 0 Invalid index.
 * \retval * A target reference.
 * \note Use <tt>\ref vxQueryContext</tt> with <tt>\ref VX_CONTEXT_ATTRIBUTE_NUMTARGETS</tt> to retrieve the upper limit of targets.
 * \ingroup group_target
 */
VX_API_ENTRY vx_target_intel VX_API_CALL vxGetTargetByIndexIntel(vx_context context, vx_uint32 index);

/*! \brief Used to get a reference to named target when the name is known beforehand.
 * \param [in] context The reference to the overall context.
 * \param [in] name The target string name.
 * \return <tt>\ref vx_target_intel</tt>
 * \retval 0 Invalid index.
 * \retval * A target reference.
 * \ingroup group_target
 */
VX_API_ENTRY vx_target_intel VX_API_CALL vxGetTargetByNameIntel(vx_context context, const vx_char *name);

/*! \brief Releases a reference to a target object.
 * The object may not be garbage collected until its total reference count is zero.
 * \param [in] target The pointer to the target to release.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE If target is not a <tt>\ref vx_target_intel</tt>.
 * \note After returning from this function the reference will be zeroed.
 * \ingroup group_target
 */
VX_API_ENTRY vx_status VX_API_CALL vxReleaseTargetIntel(vx_target_intel *target);

/*! \brief Used to query the target about it's properties.
 * \param [in] target The reference to the target.
 * \param [in] attribute The <tt>\ref vx_target_attribute_e</tt> value to query for.
 * \param [out] ptr The location at which the resulting value will be stored.
 * \param [in] size The size of the container to which ptr points.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \pre <tt>\ref vxGetTargetByNameIntel</tt> or <tt>\ref vxGetTargetByIndexIntel</tt>
 * \ingroup group_target
 */
VX_API_ENTRY vx_status VX_API_CALL vxQueryTargetIntel(vx_target_intel target, vx_enum attribute, void *ptr, vx_size size);

/*! \brief Used to assign target affinity to a node.
 * \note This assignment overrides implementation chosen behavior.
 * \param [in] node The node reference to assign affinity to.
 * \param [in] target The reference to the target to execute the Node on.
 * \pre <tt>\ref vxGetTargetByNameIntel</tt> or <tt>\ref vxGetTargetByIndex</tt>
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \ingroup group_target
 * \pre <tt>vxCreateGenericNode</tt> or some other node creation function.
 * \retval VX_ERROR_INVALID_REFERENCE Either node or target was not a valid reference.
 * \retval VX_ERROR_NOT_SUPPORTED The node can not be executed on that target.
 */
VX_API_ENTRY vx_status VX_API_CALL vxAssignNodeAffinityIntel(vx_node node, vx_target_intel target);

/*! \brief The User Tiling Function tile block size declaration.
 * \details The author of a User Tiling Kernel will use this structure to define
 * the dimensionality of the tile block.
 * \ingroup group_tiling
 */
typedef struct _vx_tile_block_size_intel_t {
    vx_int32 width; /*!< \brief Tile block width in pixels. */
    vx_int32 height; /*!< \brief Tile block height in pixels. */
} vx_tile_block_size_intel_t;

/*! \brief The User Tiling Function Neighborhood declaration.
 * \details The author of a User Tiling Kernel will use this structure to define
 * the neighborhood surrounding the tile block.
 * \ingroup group_tiling
 */
typedef struct _vx_neighborhood_size_intel_t {
    vx_int32 left;   /*!< \brief Left of the tile block. */
    vx_int32 right;  /*!< \brief Right of the tile block. */
    vx_int32 top;    /*!< \brief Top of the tile block. */
    vx_int32 bottom; /*!< \brief Bottom of the tile block. */
} vx_neighborhood_size_intel_t;

/*! \brief The tile attributes data structure that is shared with the users.
 * \ingroup group_tiling
 */
typedef struct _vx_tile_t_attributes_intel_t {
   vx_int32 x;                /*!< \brief The X coordinate of the tile */
   vx_int32 y;                /*!< \brief The Y coordinate of the tile */
   vx_tile_block_size_intel_t tile_block; /*!< \brief The size of the tile*/
} vx_tile_t_attributes_intel_t;

/*! \brief A structure which describes the tile's parent image.
 * \ingroup group_tiling
 */
typedef struct _vx_image_description_intel_t {
    vx_uint32 width;  /*!< \brief Width of the image */
    vx_uint32 height; /*!< \brief Height of the image */
    vx_df_image format; /*!< \brief The <tt>\ref vx_df_image_e</tt> of the image */
    vx_uint32 planes; /*!< \brief The number of planes in the image */
    vx_enum range;    /*!< \brief The <tt>\ref vx_channel_range_e</tt> enumeration. */
    vx_enum space;    /*!< \brief The <tt>\ref vx_color_space_e</tt> enumeration. */
} vx_image_description_intel_t;

/*! \brief The maximum number of planes in a tiled image.
 * \ingroup group_tiling
 */
#define VX_MAX_TILING_PLANES_INTEL (4)

/*! \brief The tile structure declaration.
 * \ingroup group_tiling
 */
typedef struct _vx_tile_intel_t {
    /*! \brief The array of pointers to the tile's image plane. */
    vx_uint8 * base[VX_MAX_TILING_PLANES_INTEL];
    /*! \brief The top left X pixel index within the width dimension of the image. */
    vx_uint32 tile_x;
    /*! \brief The top left Y pixel index within the height dimension of the image. */
    vx_uint32 tile_y;
    /*! \brief The array of addressing structure to describe each plane. */
    vx_imagepatch_addressing_t addr[VX_MAX_TILING_PLANES_INTEL];
    /*! \brief The output block size structure. */
    vx_tile_block_size_intel_t tile_block;
    /*! \brief The neighborhood definition. */
    vx_neighborhood_size_intel_t neighborhood;
    /*! \brief The description and attributes of the image. */
    vx_image_description_intel_t image;
} vx_tile_intel_t;

/*!
 * \brief The full height of the tile's parent image in pixels.
 * \param [in] ptile The pointer to the \ref vx_tile_intel_t structure.
 * \ingroup group_tiling
 */
#define vxImageHeightIntel(ptile)    ((ptile))->image.height)

/*!
 * \brief The full width of the tile's parent image in pixels.
 * \param [in] ptile The pointer to the \ref vx_tile_intel_t structure.
 * \ingroup group_tiling
 */
#define vxImageWidthIntel(ptile)   ((ptile))->image.width)

/*!
 * \brief The offset between the left edge of the image and the left edge of the tile, in pixels.
 * \param [in] ptile The pointer to the \ref vx_tile_intel_t structure.
 * \ingroup group_tiling
 */
#define vxTileXIntel(ptile)        ((ptile)->tile_x)

/*!
 * \brief The offset between the top edge of the image and the top edge of the tile, in pixels.
 * \param [in] ptile The pointer to the \ref vx_tile_intel_t structure.
 * \ingroup group_tiling
 */
#define vxTileYIntel(ptile)        ((ptile)->tile_y)

/*!
 * \brief The width of the tile in pixels.
 * \param [in] ptile The pointer to the \ref vx_tile_intel_t structure.
 * \param [in] index The plane index.
 * \ingroup group_tiling
 */
#define vxTileWidthIntel(ptile, index)    ((ptile)->addr[index].dim_x)

/*!
 * \brief The height of the tile in pixels.
 * \param [in] ptile The pointer to the \ref vx_tile_intel_t structure.
 * \param [in] index The plane index.
 * \ingroup group_tiling
 */
#define vxTileHeightIntel(ptile, index)   ((ptile)->addr[index].dim_y)

/*!
 * \brief The tile block height.
 * \param [in] ptile The pointer to the \ref vx_tile_intel_t structure.
 * \ingroup group_tiling
 */
#define vxTileBlockHeightIntel(ptile)     ((ptile)->tile_block.height)

/*!
 * \brief The tile block width.
 * \param [in] ptile The pointer to the \ref vx_tile_intel_t structure.
 * \ingroup group_tiling
 */
#define vxTileBlockWidthIntel(ptile)      ((ptile)->tile_block.width)

/*!
 * \brief The simple wrapper to access each image's neighborhood -X value.
 * \param [in] ptile The pointer to the \ref vx_tile_intel_t structure.
 * \ingroup group_tiling
 */
#define vxNeighborhoodLeftIntel(ptile)    ((ptile)->neighborhood.left)

/*!
 * \brief The simple wrapper to access each image's neighborhood +X value.
 * \param [in] ptile The pointer to the \ref vx_tile_intel_t structure.
 * \ingroup group_tiling
 */
#define vxNeighborhoodRightIntel(ptile)   ((ptile)->neighborhood.right)

/*!
 * \brief The simple wrapper to access each image's neighborhood -Y value.
 * \param [in] ptile The pointer to the \ref vx_tile_intel_t structure.
 * \ingroup group_tiling
 */
#define vxNeighborhoodTopIntel(ptile)     ((ptile)->neighborhood.top)

/*!
 * \brief The simple wrapper to access each image's neighborhood +Y value.
 * \param [in] ptile The pointer to the \ref vx_tile_intel_t structure.
 * \ingroup group_tiling
 */
#define vxNeighborhoodBottomIntel(ptile)  ((ptile)->neighborhood.bottom)

/*! \brief The tiling border mode extensions
 * \ingroup group_tiling
 */
enum vx_border_mode_tiling_intel_e {
    /*! \brief This value indicates that the author of the tiling kernel wrote
     * code to handle border conditions into the kernel itself. If this mode
     * is set, it can not be overriden by a call to the \ref vxSetNodeAttribute
     * with \ref VX_NODE_BORDER.
     */
    VX_BORDER_SELF_INTEL = VX_ENUM_BASE(VX_ID_INTEL, VX_ENUM_BORDER) + 0x0,
};

/*! \def vxImageOffsetIntel
 * \brief Computes the offset within an image.
 * \param [in] ptile The pointer to the \ref vx_tile_intel_t structure.
 * \param [in] i The plane index.
 * \param [in] x The Width Coordinates.
 * \param [in] y The Height Coordinates.
 * \param [in] ox The X offset.
 * \param [in] oy The Y offset.
 * \ingroup group_tiling
 */
#define vxPixelOffsetIntel(ptile, i, x, y) \
   (((ptile)->addr[i].stride_y * ((vx_int32)(((vx_int32)(y) * (vx_int32)(ptile)->addr[i].scale_y)/(vx_int32)VX_SCALE_UNITY))) +\
   ((ptile)->addr[i].stride_x * ((vx_int32)(((vx_int32)(x) * (vx_int32)(ptile)->addr[i].scale_x)/(vx_int32)VX_SCALE_UNITY))))

#define vxImageOffsetIntel(ptile, i, x, y, ox, oy) \
   vxPixelOffsetIntel(ptile, i, x + ox + (ptile)->tile_x, y + oy + (ptile)->tile_y) - vxPixelOffsetIntel(ptile, i, (ptile)->tile_x, (ptile)->tile_y)

/*! \def vxImagePixelIntel
 * \brief Accesses an image pixel as a type-cast indexed pointer dereference.
 * \param [in] type The type of the image pixel. Example values are <tt>\ref vx_uint8</tt>, <tt>\ref vx_uint16</tt>, <tt>\ref vx_uint32</tt>, etc.
 * \param [in] ptile The pointer to the \ref vx_tile_intel_t structure.
 * \param [in] i The plane index.
 * \param [in] x The Center Pixel in Width Coordinates.
 * \param [in] y The Center Pixel in Height Coordinates.
 * \param [in] ox The X offset.
 * \param [in] oy The Y offset.
 * \ingroup group_tiling
 */
#define vxImagePixelIntel(type, ptile, i, x, y, ox, oy) \
    *((type *)(&((vx_uint8 *)(ptile)->base[i])[vxImageOffsetIntel(ptile, i, x, y, ox, oy)]))

/*! \typedef vx_advanced_tiling_kernel_intel_f
 * \brief Advanced Tiling Kernel function typedef for User AdvancedTiling Kernels.
 * \note Tiles may come in any dimension and are not guaranteed to be delivered in
 * any particular order.
 * \param [in] node The handle to the node that contains this kernel.
 * \param [in] parameters The array abstract pointers to parameters.
 * \param [in] num The number of parameters
 * \param [in] tile_memory The local tile memory pointer if requested, otherwise NULL.
 * \param [in] tile_memory_size The size of the local tile memory, if not requested, 0.
 * \retval VX_SUCCESS No errors.
 * \ingroup group_tiling
 */
typedef vx_status (*vx_advanced_tiling_kernel_intel_f)(vx_node node,
                                                       void * parameters[],
                                                       vx_uint32 num,
                                                       void * tile_memory,
                                                       vx_size tile_memory_size);


/*! \typedef vx_mapping_intel_f
 * \brief Mapping function typedef for User Advanced Tiling Kernels.
 * \note Map rectangle in the destination image back to a rectangle in the source image
 * \param [in] node The handle to the node that contains this node's tile mapping.
 * \param [in] parameters The array abstract pointers to parameters.
 * \param [in] dstRectIn The pointer to rectangle of destination image.
 * \param [in] param_num Parameter index of destination image.
 * \param [out] srcRectOut The pointer to rectangle of source image.
 * \retval VX_SUCCESS No errors.
 * \ingroup group_tiling
 */

typedef vx_status (*vx_mapping_intel_f) (vx_node node,
                                         vx_reference parameters[],
                                         const vx_tile_t_attributes_intel_t* dstRectIn,
                                         vx_tile_t_attributes_intel_t* srcRectOut,
                                         vx_uint32 param_num);

/*! \typedef vx_inplace_params_intel_f
 * \brief for User Advanced Tiling Kernels.
 * \note Function specifies which parameters can be processed in in-place mode
 * \param [in] parameters The array abstract pointers to parameters.
 * \param [in] output_image_index parameter index of output vx_image.
 * \retval parameter index of input vx_image which can share a buffer with
 *         the given output image. If the given output image cannot share
 *         a buffer with any input image, -1 should be returned.
 * \ingroup group_tiling
 */

typedef vx_int32 (*vx_inplace_params_intel_f) (vx_reference parameters[],
                                    vx_uint32 output_image_index);


/*!
 * \brief The pointer to the kernel pre-process function. If the host code requires a call
 * to initialize data at the start of each call to vxProcessGraph, this function is called
 * if not NULL.
 * \param [in] node The handle to the node that contains this kernel.
 * \param [in] parameters The array of parameter references.
 * \param [in] num The number of parameters.
 * \param [in] tile_memory An array of pointers to local tile memory if requested, otherwise NULL.
 * \param [in] num_tile_memory_indices Number of elements in tile_memory array.
 * \param [in] tile_memory_size Size in bytes of the chunk for which each element of tile_memory points to.
 * \ingroup group_user_kernels
 */
typedef vx_status (*vx_kernel_preprocess_intel_f)(vx_node node,
                                            const vx_reference *parameters,
                                            vx_uint32 num_parameters,
                                            void * tile_memory[],
                                            vx_uint32 num_tile_memory_elements,
                                            vx_size tile_memory_size);

/*!
 * \brief The pointer to the kernel pre-process function. If the host code requires a call
 * to initialize data at the end of each call to vxProcessGraph, this function is called
 * if not NULL.
 * \param [in] node The handle to the node that contains this kernel.
 * \param [in] parameters The array of parameter references.
 * \param [in] num The number of parameters.
 * \param [in] tile_memory An array of pointers to local tile memory if requested, otherwise NULL.
 * \param [in] num_tile_memory_indices Number of elements in tile_memory array.
 * \param [in] tile_memory_size Size in bytes of the chunk for which each element of tile_memory points to.
 * \ingroup group_user_kernels
 */
typedef vx_status (*vx_kernel_postprocess_intel_f)(vx_node node,
                                             const vx_reference *parameters,
                                             vx_uint32 num_parameters,
                                             void * tile_memory[],
                                             vx_uint32 num_tile_memory_indices,
                                             vx_size tile_memory_size);

/*!
 * \brief The pointer to the kernel 'set tile dimensions' function. If the host code requires a call
 * to set a custom tile dimension, this function is called  within vxVerifyGraph
 * if not NULL.
 * \param [in] node The handle to the node that contains this kernel.
 * \param [in] parameters The array of parameter references.
 * \param [in] param_num The number of parameters.
 * \param [in] current_tile_dimensions The currently set tile dimensions.
 * \param [out] updated_tile_dimensions The updated tile dimensions.
 * \ingroup group_user_kernels
 */
typedef vx_status (*vx_kernel_set_tile_dimensions_intel_f)(vx_node node,
                                                     const vx_reference *parameters,
                                                     vx_uint32 param_num,
                                                     const vx_tile_block_size_intel_t *current_tile_dimensions,
                                                     vx_tile_block_size_intel_t *updated_tile_dimensions);

/*!
 * \brief The pointer to the kernel 'tile dimensions initialize' function. If the host code requires a call
 * to perform some initialization and/or to set specific node attributes as a function of the tile
 * dimensions, this function is called within vxVerifyGraph, after all calls to vx_kernel_set_tile_dimensions_intel_f
 * if not NULL.
 * \param [in] node The handle to the node that contains this kernel.
 * \param [in] parameters The array of parameter references.
 * \param [in] param_num The number of parameters.
 * \param [in] tile_dimensions The currently set tile dimensions.
 * \ingroup group_user_kernels
 */
typedef vx_status (*vx_kernel_tile_dimensions_initialize_intel_f)(vx_node node,
                                                           const vx_reference *parameters,
                                                           vx_uint32 param_num,
                                                           const vx_tile_block_size_intel_t *tile_dimensions);

VX_API_ENTRY vx_kernel VX_API_CALL vxAddAdvancedTilingKernelIntel(vx_context context,
                            vx_char name[VX_MAX_KERNEL_NAME],
                            vx_enum enumeration,
                            vx_advanced_tiling_kernel_intel_f kernel_func_ptr,
                            vx_mapping_intel_f mapping_func_ptr,
                            vx_uint32 num_params,
                            vx_kernel_validate_f validate,
                            vx_kernel_initialize_f initialize,
                            vx_kernel_deinitialize_f deinitialize,
                            vx_kernel_preprocess_intel_f preprocess,
                            vx_kernel_postprocess_intel_f postprocess,
                            vx_kernel_set_tile_dimensions_intel_f settiledimensions,
                            vx_kernel_tile_dimensions_initialize_intel_f tiledimensionsinitialize );


#ifdef  __cplusplus
} // extern "C"
#endif
#endif /*VX_INTEL_VOLATILE_H */
