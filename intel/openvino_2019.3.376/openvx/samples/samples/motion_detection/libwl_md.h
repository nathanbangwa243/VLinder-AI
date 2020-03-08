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
#ifndef __LIBWL_MD_H__
#define __LIBWL_MD_H__

#include <intel/workload_api/cvsdkworkload.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IOTG_MD_VER_MAJOR 0x01
#define IOTG_MD_VER_MINOR 0x00

enum
{
  WL_CFG_MOTIONDETECTION = WL_CFG_VENDOR_EXT + 2,
};

typedef struct _WL_MDConfig
{
  unsigned int threshold;   // The threshold to filter out little components(noise), it's the number of pixels, here 0 is the default number
  unsigned int mergeBoxes;  // A flag to indicate if it's necessary to merge overlapped boxes, or those boxes very close to each other
  unsigned int scaleImage;  // A flag to indicate if it's necessary to scale larger input images into smaller images to improve performance
  unsigned int sceneAdaption;   // The scene adaption vaule
} WL_MDConfig;

#ifdef __cplusplus
}
#endif

#endif /* __LIBWL_MD_H__ */
