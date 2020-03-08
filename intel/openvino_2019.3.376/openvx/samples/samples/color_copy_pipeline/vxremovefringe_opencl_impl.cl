/* ////////////////////////////////////////////////////////////////////////////////////
//
//                   INTEL CORPORATION PROPRIETARY INFORMATION
//      This software is supplied under the terms of a license agreement or
//      nondisclosure agreement with Intel Corporation and may not be copied
//      or disclosed except in accordance with the terms of that agreement.
//        Copyright(c) 2015-2016 Intel Corporation. All Rights Reserved.
//
*/

//These are the parameters from the OpenVX standpoint. Using the OpenVX->OpenCL ABI 
// (which is defined in the user guide), each of the below objects is expanded into
// 1 or more CL objects.
//For example, a vx_image of type U8 is expanded into:
// uchar *pSrc, int x, int y, int tileWidth, int tileHeight, int imageWidth, int imageHeight, int xStride, int yStride
//vx_image inputCMYK,
//vx_image inputK,
//vx_image inputNeutralEdgeMask,
//vx_image outputCMYK,
__kernel void _RemoveFringeKernel(__global const uchar *pInputCMYK, int xInputCMYK, int yInputCMYK, 
                                 int tileWidthInputCMYK, int tileHeightInputCMYK, 
                                 int imageWidthInputCMYK, int imageHeightInputCMYK,
                                 int inputCMYKPixelStride, int inputCMYKRowStride,
                                 
                                 __global const uchar *pInputL, int xInputL, int yInputL, 
                                 int tileWidthInputL, int tileHeightInputL, 
                                 int imageWidthInputL, int imageHeightInputL,
                                 int inputLPixelStride, int inputLRowStride,
                                 
                                 __global const uchar *pInputNEdgeMask, int xInputNEdgeMask, int yInputNEdgeMask, 
                                 int tileWidthInputNEdgeMask, int tileHeightInputNEdgeMask, 
                                 int imageWidthInputNEdgeMask, int imageHeightInputNEdgeMask,
                                 int inputNEdgeMaskPixelStride, int inputNEdgeMaskRowStride,
                                 
                                 __global uchar *pOutputCMYK, int xOutputCMYK, int yOutputCMYK, 
                                 int tileWidthOutputCMYK, int tileHeightOutputCMYK, 
                                 int imageWidthOutputCMYK, int imageHeightOutputCMYK,
                                 int outputCMYKPixelStride, int outputCMYKRowStride,
                                 
                                 __global uchar *pOutputK, int xOutputK, int yOutputK, 
                                 int tileWidthOutputK, int tileHeightOutputK, 
                                 int imageWidthOutputK, int imageHeightOutputK,
                                 int outputKPixelStride, int outputKRowStride,
                                 
                                 __global uchar *pLtoKLUT, uint nltok_items, uint ltok_itemSize, uint ltok_stride                
                                 )
{
  int x = get_global_id(0) * WORK_ITEM_XSIZE;
  int y = get_global_id(1) * WORK_ITEM_YSIZE;
  
  __global const uchar* srcCMYK = pInputCMYK + y*inputCMYKRowStride + x*4;
  __global const uchar* srcL = pInputL + y*inputLRowStride + x;
  __global const uchar* srcNEMask = pInputNEdgeMask + y*inputNEdgeMaskRowStride + x;
  __global       uchar* dstCMYK = pOutputCMYK + y*outputCMYKRowStride + x*4;
  __global       uchar* dstK = pOutputK + y*outputKRowStride + x;


  #pragma unroll
  for (int cy = 0; cy < WORK_ITEM_YSIZE; ++cy) 
  {
    if (y + cy < tileHeightInputCMYK) 
    {
      __global const uchar* tmpSrcCMYK = srcCMYK;
      __global const uchar* tmpSrcL = srcL;
      __global const uchar* tmpSrcNEMask = srcNEMask;
      __global uchar* tmpDstCMYK = dstCMYK;
      __global uchar* tmpDstK = dstK;

      if(x + 4 <= tileWidthInputCMYK )
      {
        const uchar16 srcCMYKPixel = vload16(0, tmpSrcCMYK);
        const uchar4 srcLPixel = vload4(0, tmpSrcL);
        const uchar4 srcNEMaskPixel = vload4(0, tmpSrcNEMask);

        uchar4 koutPixel;
        koutPixel.s0 = pLtoKLUT[srcLPixel.s0];
        koutPixel.s1 = pLtoKLUT[srcLPixel.s1];
        koutPixel.s2 = pLtoKLUT[srcLPixel.s2];
        koutPixel.s3 = pLtoKLUT[srcLPixel.s3];
        ((__global uchar4 *)(tmpDstK))[0] = koutPixel;

        uchar4 edgeResult;
        edgeResult.s0 = 0;
	    edgeResult.s1 = 0;
	    edgeResult.s2 = 0;
	    edgeResult.s3 = koutPixel.s0;

        uchar16 cmykOut;
        cmykOut.s0123 = srcNEMaskPixel.s0 ? edgeResult : srcCMYKPixel.s0123;
        edgeResult.s3 = koutPixel.s1;
        cmykOut.s4567 = srcNEMaskPixel.s1 ? edgeResult : srcCMYKPixel.s4567;
        edgeResult.s3 = koutPixel.s2;
        cmykOut.s89ab = srcNEMaskPixel.s2 ? edgeResult : srcCMYKPixel.s89ab;
        edgeResult.s3 = koutPixel.s3;
        cmykOut.scdef = srcNEMaskPixel.s3 ? edgeResult : srcCMYKPixel.scdef;

        ((__global uchar16 *)(tmpDstCMYK))[0] = cmykOut;

      }
      else
      {
          while( x < tileWidthInputCMYK )
          {
	        const uchar4 srcCMYKPixel = vload4(0, tmpSrcCMYK); 
	        const uchar srcLPixel = *tmpSrcL; 
	        const uchar srcNEMaskPixel = *tmpSrcNEMask;
	        
	        uchar koutPixel = pLtoKLUT[srcLPixel];
	        *tmpDstK = koutPixel;
	        
	        uchar4 edgeResult;
	        edgeResult.s0 = 0;
	        edgeResult.s1 = 0;
	        edgeResult.s2 = 0;
	        edgeResult.s3 = koutPixel;
	        
	        uchar4 cmykOut = srcNEMaskPixel ? edgeResult : srcCMYKPixel;
	        ((__global uchar4 *)(tmpDstCMYK))[0] = cmykOut;
	        
	        tmpSrcCMYK += 4;
	        tmpSrcL++;
	        tmpSrcNEMask++;
	        tmpDstCMYK += 4;
	        tmpDstK++;
            x++;
          }
      }
        
      srcCMYK += inputCMYKRowStride;
      srcL += inputLRowStride;
      srcNEMask += inputNEdgeMaskRowStride;
      dstCMYK += outputCMYKRowStride;
      dstK += outputKRowStride;
      
    }
  
  }
  
}

//These are the parameters from the OpenVX standpoint. Using the OpenVX->OpenCL ABI
// (which is defined in the user guide), each of the below objects is expanded into
// 1 or more CL objects.
//For example, a vx_image of type U8 is expanded into:
// uchar *pSrc, int x, int y, int tileWidth, int tileHeight, int imageWidth, int imageHeight, int xStride, int yStride
//vx_image inputCMYK,
//vx_image inputK,
//vx_image inputNeutralEdgeMask,
//vx_image outputCMYK,
__kernel void _RemoveFringePlanarKernel(__global const uchar *pInputCMYK, int xInputCMYK, int yInputCMYK,
                                         int tileWidthInputCMYK, int tileHeightInputCMYK,
                                         int imageWidthInputCMYK, int imageHeightInputCMYK,
                                         int inputCMYKPixelStride, int inputCMYKRowStride,

                                         __global const uchar *pInputL, int xInputL, int yInputL,
                                         int tileWidthInputL, int tileHeightInputL,
                                         int imageWidthInputL, int imageHeightInputL,
                                         int inputLPixelStride, int inputLRowStride,

                                         __global const uchar *pInputNEdgeMask, int xInputNEdgeMask, int yInputNEdgeMask,
                                         int tileWidthInputNEdgeMask, int tileHeightInputNEdgeMask,
                                         int imageWidthInputNEdgeMask, int imageHeightInputNEdgeMask,
                                         int inputNEdgeMaskPixelStride, int inputNEdgeMaskRowStride,

                                         __global uchar *pOutputC, int xOutputC, int yOutputC,
                                         int tileWidthOutputC, int tileHeightOutputC,
                                         int imageWidthOutputC, int imageHeightOutputC,
                                         int outputCPixelStride, int outputCRowStride,

                                         __global uchar *pOutputM, int xOutputM, int yOutputM,
                                         int tileWidthOutputM, int tileHeightOutputM,
                                         int imageWidthOutputM, int imageHeightOutputM,
                                         int outputMPixelStride, int outputMRowStride,

                                         __global uchar *pOutputY, int xOutputY, int yOutputY,
                                         int tileWidthOutputY, int tileHeightOutputY,
                                         int imageWidthOutputY, int imageHeightOutputY,
                                         int outputYPixelStride, int outputYRowStride,

                                         __global uchar *pOutputK, int xOutputK, int yOutputK,
                                         int tileWidthOutputK, int tileHeightOutputK,
                                         int imageWidthOutputK, int imageHeightOutputK,
                                         int outputKPixelStride, int outputKRowStride,

                                         __global uchar *pOutputKEdge, int xOutputKEdge, int yOutputKEdge,
                                         int tileWidthOutputKEdge, int tileHeightOutputKEdge,
                                         int imageWidthOutputKEdge, int imageHeightOutputKEdge,
                                         int outputKEdgePixelStride, int outputKEdgeRowStride,

                                         __global uchar *pLtoKLUT, uint nltok_items, uint ltok_itemSize, uint ltok_stride
                                 )
{
  int x = get_global_id(0) * WORK_ITEM_XSIZE;
  int y = get_global_id(1) * WORK_ITEM_YSIZE;
  
  __global const uchar* srcCMYK = pInputCMYK + y*inputCMYKRowStride + x*4;
  __global const uchar* srcL = pInputL + y*inputLRowStride + x;
  __global const uchar* srcNEMask = pInputNEdgeMask + y*inputNEdgeMaskRowStride + x;
  __global       uchar* dstC = pOutputC + y*outputCRowStride + x;
  __global       uchar* dstM = pOutputM + y*outputMRowStride + x;
  __global       uchar* dstY = pOutputY + y*outputYRowStride + x;
  __global       uchar* dstK = pOutputK + y*outputKRowStride + x;
  __global       uchar* dstKEdge = pOutputKEdge + y*outputKEdgeRowStride + x;


  #pragma unroll
  for (int cy = 0; cy < WORK_ITEM_YSIZE; ++cy)
  {
    if (y + cy < tileHeightInputCMYK)
    {
      __global const uchar* tmpSrcCMYK = srcCMYK;
      __global const uchar* tmpSrcL = srcL;
      __global const uchar* tmpSrcNEMask = srcNEMask;
      __global uchar* tmpDstC = dstC;
      __global uchar* tmpDstM = dstM;
      __global uchar* tmpDstY = dstY;
      __global uchar* tmpDstK = dstK;
      __global uchar* tmpDstKEdge = dstKEdge;

      if(x + 4 <= tileWidthInputCMYK )
      {
        const uchar16 srcCMYKPixel = vload16(0, tmpSrcCMYK);
        const uchar4 srcLPixel = vload4(0, tmpSrcL);
        const uchar4 srcNEMaskPixel = vload4(0, tmpSrcNEMask);

        uchar4 kedgeoutPixel;
        kedgeoutPixel.s0 = pLtoKLUT[srcLPixel.s0];
        kedgeoutPixel.s1 = pLtoKLUT[srcLPixel.s1];
        kedgeoutPixel.s2 = pLtoKLUT[srcLPixel.s2];
        kedgeoutPixel.s3 = pLtoKLUT[srcLPixel.s3];
        ((__global uchar4 *)(tmpDstKEdge))[0] = kedgeoutPixel;


        uchar4 Cout =  (uchar4)(srcCMYKPixel.s0, srcCMYKPixel.s4, srcCMYKPixel.s8, srcCMYKPixel.sc);
               Cout = Cout & ~srcNEMaskPixel;
        ((__global uchar4 *)(tmpDstC))[0] = Cout;

        uchar4 Mout =  (uchar4)(srcCMYKPixel.s1, srcCMYKPixel.s5, srcCMYKPixel.s9, srcCMYKPixel.sd);
               Mout = Mout & ~srcNEMaskPixel;
        ((__global uchar4 *)(tmpDstM))[0] = Mout;

        uchar4 Yout =  (uchar4)(srcCMYKPixel.s2, srcCMYKPixel.s6, srcCMYKPixel.sa, srcCMYKPixel.se);
               Yout = Yout & ~srcNEMaskPixel;
        ((__global uchar4 *)(tmpDstY))[0] = Yout;

        uchar4 Kout =  (uchar4)(srcCMYKPixel.s3, srcCMYKPixel.s7, srcCMYKPixel.sb, srcCMYKPixel.sf);
               Kout = (srcNEMaskPixel & kedgeoutPixel) | (Kout & ~srcNEMaskPixel);
        ((__global uchar4 *)(tmpDstK))[0] = Kout;

      }
      else
      {
          while( x < tileWidthInputCMYK )
          {
	        const uchar4 srcCMYKPixel = vload4(0, tmpSrcCMYK);
	        const uchar srcLPixel = *tmpSrcL;
	        const uchar srcNEMaskPixel = *tmpSrcNEMask;

	        uchar kedgeoutPixel = pLtoKLUT[srcLPixel];
	        *tmpDstKEdge = kedgeoutPixel;

            uchar Cout = srcNEMaskPixel ? 0 : srcCMYKPixel.s0;
            uchar Mout = srcNEMaskPixel ? 0 : srcCMYKPixel.s1;
            uchar Yout = srcNEMaskPixel ? 0 : srcCMYKPixel.s2;
            uchar Kout = srcNEMaskPixel ? kedgeoutPixel : srcCMYKPixel.s3;

            *tmpDstC = Cout;
            *tmpDstM = Mout;
            *tmpDstY = Yout;
            *tmpDstK = Kout;

	        tmpSrcCMYK += 4;
	        tmpSrcL++;
	        tmpSrcNEMask++;
	        tmpDstC++;
            tmpDstM++;
            tmpDstY++;
            tmpDstK++;
            tmpDstKEdge++;
            x++;
          }
      }

      srcCMYK += inputCMYKRowStride;
      srcL += inputLRowStride;
      srcNEMask += inputNEdgeMaskRowStride;
      dstC += outputCRowStride;
      dstM += outputMRowStride;
      dstY += outputYRowStride;
      dstK += outputKRowStride;
      dstKEdge += outputKEdgeRowStride;
    }
  }
}



