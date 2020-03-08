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

#ifndef __CVSDK_WORKLOAD_H__
#define __CVSDK_WORKLOAD_H__

/*=====================================================================
  Workload internal object operations API and structure definitions
  =====================================================================
*/

#define WL_MAX_PLANES 4

#define WL_MAX_IMAGES 4

// ROI Rectangle data structure
typedef struct _WL_Roi
{
  int x, y;
  int width, height;
}WL_Roi;

typedef struct _WL_Rois
{
  int count;
  WL_Roi rois[128];
}WL_Rois;

// enums for all workload functions integer return status
typedef enum
{
  WL_OK                = 0,
  WL_BAD_VALUE         = -1,
  WL_BAD_INDEX         = -2,
  WL_NO_MEMORY         = -3,
  WL_NOT_SUPPORTED     = -4,
  WL_INVALID_OPERATION = -5,
  WL_OPERATION_FAIL    = -6,
  WL_UNKNOWN_ERROR     = -255,
}WL_Status;

// configuration index enums for WL_Config
typedef enum
{
  WL_CFG_INPUT,               //Workload config input format
  WL_CFG_OUTPUT,              //Workload config output format
  WL_CFG_HETEROGENEITY,       //Workload config heterogeneity preference
  WL_CFG_ROI,                 //Workload config region of interest

  WL_CFG_VENDOR_EXT = 128,    //workload config vendor extension
}WL_Cfg_Index;

// color enums for WL_Buffer_Format configuration
typedef enum
{
  WL_COLOR_UNKNOWN = -1,
  WL_COLOR_RGB,
  WL_COLOR_NV12,
  WL_COLOR_I420,
  WL_COLOR_NUM
}WL_Color;


// colorspace enums for WL_Image_Info configuration
typedef enum
{
  WL_COLORSPACE_NONE,            //Use to indicate that no color space is used
  WL_COLORSPACE_BT601_525,
  WL_COLORSPACE_BT601_625,
  WL_COLORSPACE_BT709,
  WL_COLORSPACE_BT2020,
}WL_Colorspace;

// data structure for WL_CFG_INPUT/OUTPUT configuration
typedef struct _WL_Image_Info {
  int width;
  int height;
  WL_Color color;
  WL_Colorspace colorspace;
} WL_Image_Info;

// enums for heterogeneity preference
typedef enum
{
  WL_HETER_UNKNOWN = -1,    //Invalid value of heterogeneity configurations
  WL_HETER_CPU_ONLY = 0,    //CPU used only
  WL_HETER_GPU_ONLY,        //GPU used only
  WL_HETER_IPU_ONLY,        //IPU used only
  WL_HETER_CPU_PREF,        //CPU used preferred
  WL_HETER_GPU_PREF,        //GPU used preferred
  WL_HETER_IPU_PREF,        //IPU used preferred
  WL_HETER_SOC_OPTI,        //SOC based optimization
  WL_HETER_CUSTOM,          //Customized for each node
  WL_HETER_MAX_NUM          //Number of heterogeneity configurations
  //Architecture number
}WL_Heterogeneity;

// data structure for WL_CFG_HETEROGENEITY configuration
typedef struct _WL_Heterogeneity_Info {
  int heterogeneity_pref;
  char *heterogeneity_config_file;
} WL_Heterogeneity_Info;

// data structure for input / output image
//
// buf means the start address of image buffer.
//
// For the plane data start address, they are:
// 1st Plane = buf + offset[0];
// 2nd Plane = buf + offset[1];
// 3rd Plane = buf + offset[2];
// 4th Plane = buf + offset[3];
typedef struct _WL_Image {
  char *buf;

  int width;
  int height;
  int bufsize;
  int numofplanes;
  int offset[WL_MAX_PLANES];
  int stride[WL_MAX_PLANES];
  WL_Color color;
} WL_Image;

typedef struct _WL_Images {
  WL_Image imgs[WL_MAX_IMAGES];
  int count;
}WL_Images;

// Function pointer definition for WL_Config
typedef WL_Status (* WL_Config_Func) (void *wl, WL_Cfg_Index index, void *config);
// Function pointer definition for WL_Init
typedef WL_Status (* WL_Init_Func) (void *wl);
// Function pointer definition for WL_Process
typedef WL_Status (* WL_Process_Func) (void *wl, WL_Images *inimgs, WL_Images *outimgs, void *outmetadata);
// Function pointer definition for WL_Deinit
typedef WL_Status (* WL_Deinit_Func) (void *wl);

//MARCO to call WL_Config for one workload context
#define WL_Config(ctx, index, config)  (*ctx->Config)(ctx->wl, index, config)
//MARCO to call WL_Init for one workload context
#define WL_Init(ctx)  (*ctx->Init)(ctx->wl)
//MARCO to call WL_Process for one workload context
#define WL_Process(ctx, in, out, data)  (*ctx->Process)(ctx->wl, in, out, data)
//MARCO to call WL_Deinit for one workload context
#define WL_Deinit(ctx)  (*ctx->Deinit)(ctx->wl)


/*=====================================================================
  Workload context related API and structure definitions
  =====================================================================
*/
//WL_API version information
#define WL_API_VER_MAJOR 0x01
#define WL_API_VER_MINOR 0x00

#define WL_API_VER                  ((WL_API_VER_MAJOR << 8) + WL_API_VER_MINOR)
#define WL_VER(fmajor, fminor)      ((WL_API_VER << 16) + ((fmajor << 8) + fminor) )

#define WL_API_MAJOR_VER(Ver)   (((Ver) >> 24) & 0xFF)
#define WL_API_MINOR_VER(Ver)   (((Ver) >> 16) & 0xFF)
#define WL_LIB_MAJOR_VER(Ver)   (((Ver) >> 8) & 0xFF)
#define WL_LIB_MINOR_VER(Ver)   ((Ver) & 0xFF)

// Workload context definition
typedef struct _WLContext {
  void *wl;                     // private internal workload object

  WL_Config_Func  Config;       // WL_Config function pointer
  WL_Init_Func    Init;         // WL_Init function pointer
  WL_Process_Func Process;      // WL_Process function pointer
  WL_Deinit_Func  Deinit;       // WL_Deinit function pointer
}WLContext;

// enums for workload type definition
typedef enum
{
  WL_TYPE_ROIDETECTOR   = 1,
  WL_TYPE_ROIRECOGNIZER = 2,
  WL_TYPE_ROIPROCESSOR  = 4,
}WLType;

typedef int WLVersion;

// Function pointer definition for WLCREATE
typedef WLContext* (* WLCREATE_Func) ();
// Function pointer definition for WLDESTROY
typedef void (* WLDESTROY_Func) (WLContext *wl);
// Function pointer definition for WLTYPES
typedef WLType (* WLTYPES_Func) ();
// Function pointer definition for WLVERSION
typedef WLVersion (* WLVERSION_Func) ();

// For workload library users, if you want to dynamically link the workload library, you need to define Marco
// WORKLOAD_USER_LINK in your Makefile.
//
// If no those Marcos defined, it means workload library is used in dlopen mode.
#ifdef WORKLOAD_USER_LINK

// Create a workload context implemented in workload library
WLContext*      WLCREATE();

// Destroy workload instance created before
void            WLDESTROY(WLContext *wl);

// Get workload type declaration implemented in workload library, it is
// used to identify the general purpose and usage for workload, type is
// enumerated with WL_TYPE_xxx, and can be multiplied.
WLType          WLTYPES();

// Get version of workload library.
// The version consists of API version and feature version.
// API version is used to check the API interfaces compatibility.
// Refer to the Workload library materials for workload feature version compatibility.
//
// xx  |          31  -  24        |         23  -  16         |            15  -  8           |           7  -  0             |
//     | 8bit WL API MAJOR version | 8bit WL API MINOR version | 8bit WL feature MAJOR version | 8bit WL feature MINOR version |
WLVersion       WLVERSION();

#endif

#endif /* __CVSDK_WORKLOAD_H__ */
