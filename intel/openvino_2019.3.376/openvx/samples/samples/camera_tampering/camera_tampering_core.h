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

#ifndef _MOTION_DETECTION_CORE_H_
#define _MOTION_DETECTION_CORE_H_

#include <stdio.h>
#include <time.h>
#include <malloc.h>
#include <memory.h>
#include <assert.h>
#include <VX/vx.h>
#include <VX/vxu.h>

//////////////////////////////////////////////////////////////////////////
typedef unsigned int    uint32;
typedef int             int32;
typedef unsigned short  uint16;
typedef short           int16;
typedef unsigned char   uint8;
typedef char            int8;

// The step to get bounding rectangle in SizeFilter(). Ideally, the value should be 1 to get best accuracy.
// Tests show that with the value as 8, performance is better with negligible loss in accuracy.
// Higher value can have better performance and worse accuracy, customers can tune this value for their use.
#define BOUND_RECT_STEP                         (8)

typedef struct ConnectedComponentLabelingConfig
{
    uint32  threshold;      // The threshold used by size filter to filter out component with small size

    ConnectedComponentLabelingConfig()
    {
        threshold   = 0;
    }
} ConnectedComponentLabelingConfig;

//////////////////////////////////////////////////////////////////////////
class ConnectedComponentLabelingClass
{
public:
  // set image dimension
  void init(int32 width, int32 height, int32 srcImgStep, int32 dstImgStep);

  // set configuration
  void setConfig(ConnectedComponentLabelingConfig *config)
  {
    if (config != NULL)
    {
        m_nThreshold    = config->threshold;
    }
  }

  /// release memory
  void release();

  // connected component labeling and size filtering
  int32 Do(uint8 *pSrcImg, uint32 *pDstImg, vx_rectangle_t *rectList, uint32 rectListLen);

private:
  int32     m_nWidth;           // Image width
  int32     m_nHeight;          // Image height
  uint32    m_nThreshold;       // The threshold used by size filter to filter out component with small size
  int32     m_nSrcImgStep;      // Line step of source image
  int32     m_nDstImgStep;      // Line step of destination image
  uint32    m_nBoundingBoxCnt;  // Counter of bounding boxes after size filter processing
};

#endif


