/*
        Copyright 2018 Intel Corporation.
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


kernel void symm7x7_opt(

	// The kernel signature is completely defined by param_types array passed to
	// vxAddDeviceKernelIntel function in host C code. Each OpenVX parameter is translated
	// to one or multiple arguments of OpenCL kernel.
	// Even if the kernel's body doesn't use all these arguments, they should be defined here
	// anyway, because OpenVX run-time relies on the order and specific number of parameters,
	// to set them correctly when calling this kernel and translating OpenVX parameters.

	// OpenVX kernel 0-th parameter has type vx_image, it is mapped to these 5 OpenCL kernel arguments
	// This is input VX_DF_IMAGE_U8 image

	global const uchar* inImgPtr,
	unsigned int        widthInImg,        // width of the input image
	unsigned int        heightInImg,       // height of the input image
	unsigned int        pixelStrideInImg,  // pixel stride in bytes
	unsigned int        rowPitchInImg,     // row stride in bytes

	// OpenVX kernel 1-st parameter has type vx_image, it is mapped to these 5 OpenCL kernel arguments
	// This is output VX_DF_IMAGE_U8 image

	global uchar*       outImgPtr,
	unsigned int        widthOutImg,       // width of the output image
	unsigned int        heightOutImg,      // height of the output image
	unsigned int        pixelStrideOutImg, // pixel stride in bytes
	unsigned int        rowPitchOutImg     // row stride in bytes
	)
{
	int x = get_global_id(0);   // 0..(widthInImg/VX_OPENCL_WORK_ITEM_XSIZE-1)
	int y = get_global_id(1);   // 0..(heightInImg/VX_OPENCL_WORK_ITEM_YSIZE-1)


	// coordinates of a top-left pixel in output image area for processing by one WI
	// some of the pixels from WORK_ITEM_XSIZE x WORK_ITEM_YSIZE area are not valid on the border
	// they will not be processed (look at if's in the code below)
	x *= WORK_ITEM_XSIZE;
	y *= WORK_ITEM_YSIZE;

	int xSize = WORK_ITEM_XSIZE;
	int ySize = WORK_ITEM_YSIZE;

	//Symm7x7 layout
	//   JIHGHIJ
	//   IFEDEFI
	//   HECBCEH
	//   GDBABGD
	//   HECBCEH
	//   IFEDEFI
	//   JIHGHIJ

	//Symm7x7 coefficients
	const short A = 1140; const short B = -118; const short C = 526; const short D = 290; const short E = -236;
	const short F = 64; const short G = -128; const short H = -5; const short I = -87; const short J = -7;

	const short shift = 10;
	const short border = 255;

	if (
		(x > 2) &&
		(x < widthOutImg - 3) &&
		(y > 2) &&
		(y < heightOutImg - 3)
		)
	{
		//non border pixels processing
		for (int yi = y; yi < y + ySize; ++yi)
		{
			for (int xi = x; xi < x + xSize; ++xi)
			{
				//Start src offset
				int srcOffset0 = (yi - 3)*rowPitchInImg + (xi - 3)*pixelStrideInImg;
				int srcOffset1 = (yi - 3)*rowPitchInImg + (xi - 2)*pixelStrideInImg;
				int srcOffset2 = (yi - 3)*rowPitchInImg + (xi - 1)*pixelStrideInImg;
				int srcOffset3 = (yi - 3)*rowPitchInImg + (xi)*pixelStrideInImg;
				int srcOffset4 = (yi - 3)*rowPitchInImg + (xi + 1)*pixelStrideInImg;
				int srcOffset5 = (yi - 3)*rowPitchInImg + (xi + 2)*pixelStrideInImg;
				int srcOffset6 = (yi - 3)*rowPitchInImg + (xi + 3)*pixelStrideInImg;

				int dstOffset = yi*rowPitchOutImg + xi*pixelStrideOutImg;

				int intermediate;
				short intermediateA = 0;
				short intermediateB = 0;
				short intermediateC = 0;
				short intermediateD = 0;
				short intermediateE = 0;
				short intermediateF = 0;
				short intermediateG = 0;
				short intermediateH = 0;
				short intermediateI = 0;
				short intermediateJ = 0;

				//1st line JIHGHIJ
				intermediateJ = inImgPtr[srcOffset0] + inImgPtr[srcOffset6];
				intermediateI = inImgPtr[srcOffset1] + inImgPtr[srcOffset5];
				intermediateH = inImgPtr[srcOffset2] + inImgPtr[srcOffset4];
				intermediateG = inImgPtr[srcOffset3];

				srcOffset0 = srcOffset0 + rowPitchInImg;
				srcOffset1 = srcOffset1 + rowPitchInImg;
				srcOffset2 = srcOffset2 + rowPitchInImg;
				srcOffset3 = srcOffset3 + rowPitchInImg;
				srcOffset4 = srcOffset4 + rowPitchInImg;
				srcOffset5 = srcOffset5 + rowPitchInImg;
				srcOffset6 = srcOffset6 + rowPitchInImg;

				//2nd line IFEDEFI
				intermediateI = intermediateI + inImgPtr[srcOffset0] + inImgPtr[srcOffset6];
				intermediateF = inImgPtr[srcOffset1] + inImgPtr[srcOffset5];
				intermediateE = inImgPtr[srcOffset2] + inImgPtr[srcOffset4];
				intermediateD = inImgPtr[srcOffset3];

				srcOffset0 = srcOffset0 + rowPitchInImg;
				srcOffset1 = srcOffset1 + rowPitchInImg;
				srcOffset2 = srcOffset2 + rowPitchInImg;
				srcOffset3 = srcOffset3 + rowPitchInImg;
				srcOffset4 = srcOffset4 + rowPitchInImg;
				srcOffset5 = srcOffset5 + rowPitchInImg;
				srcOffset6 = srcOffset6 + rowPitchInImg;

				//3rd line HECBCEH
				intermediateH = intermediateH + inImgPtr[srcOffset0] + inImgPtr[srcOffset6];
				intermediateE = intermediateE + inImgPtr[srcOffset1] + inImgPtr[srcOffset5];
				intermediateC = inImgPtr[srcOffset2] + inImgPtr[srcOffset4];
				intermediateB = inImgPtr[srcOffset3];

				srcOffset0 = srcOffset0 + rowPitchInImg;
				srcOffset1 = srcOffset1 + rowPitchInImg;
				srcOffset2 = srcOffset2 + rowPitchInImg;
				srcOffset3 = srcOffset3 + rowPitchInImg;
				srcOffset4 = srcOffset4 + rowPitchInImg;
				srcOffset5 = srcOffset5 + rowPitchInImg;
				srcOffset6 = srcOffset6 + rowPitchInImg;

				//4th line GDBABDG
				intermediateG = intermediateG + inImgPtr[srcOffset0] + inImgPtr[srcOffset6];
				intermediateD = intermediateD + inImgPtr[srcOffset1] + inImgPtr[srcOffset5];
				intermediateB = intermediateB + inImgPtr[srcOffset2] + inImgPtr[srcOffset4];
				intermediateA = inImgPtr[srcOffset3];

				srcOffset0 = srcOffset0 + rowPitchInImg;
				srcOffset1 = srcOffset1 + rowPitchInImg;
				srcOffset2 = srcOffset2 + rowPitchInImg;
				srcOffset3 = srcOffset3 + rowPitchInImg;
				srcOffset4 = srcOffset4 + rowPitchInImg;
				srcOffset5 = srcOffset5 + rowPitchInImg;
				srcOffset6 = srcOffset6 + rowPitchInImg;

				//5th line HECBCEH
				intermediateH = intermediateH + inImgPtr[srcOffset0] + inImgPtr[srcOffset6];
				intermediateE = intermediateE + inImgPtr[srcOffset1] + inImgPtr[srcOffset5];
				intermediateC = intermediateC + inImgPtr[srcOffset2] + inImgPtr[srcOffset4];
				intermediateB = intermediateB + inImgPtr[srcOffset3];

				srcOffset0 = srcOffset0 + rowPitchInImg;
				srcOffset1 = srcOffset1 + rowPitchInImg;
				srcOffset2 = srcOffset2 + rowPitchInImg;
				srcOffset3 = srcOffset3 + rowPitchInImg;
				srcOffset4 = srcOffset4 + rowPitchInImg;
				srcOffset5 = srcOffset5 + rowPitchInImg;
				srcOffset6 = srcOffset6 + rowPitchInImg;

				//6th line IFEDEFI
				intermediateI = intermediateI + inImgPtr[srcOffset0] + inImgPtr[srcOffset6];
				intermediateF = intermediateF + inImgPtr[srcOffset1] + inImgPtr[srcOffset5];
				intermediateE = intermediateE + inImgPtr[srcOffset2] + inImgPtr[srcOffset4];
				intermediateD = intermediateD + inImgPtr[srcOffset3];

				srcOffset0 = srcOffset0 + rowPitchInImg;
				srcOffset1 = srcOffset1 + rowPitchInImg;
				srcOffset2 = srcOffset2 + rowPitchInImg;
				srcOffset3 = srcOffset3 + rowPitchInImg;
				srcOffset4 = srcOffset4 + rowPitchInImg;
				srcOffset5 = srcOffset5 + rowPitchInImg;
				srcOffset6 = srcOffset6 + rowPitchInImg;

				//7th line JIHGHIJ
				intermediateJ = intermediateJ + inImgPtr[srcOffset0] + inImgPtr[srcOffset6];
				intermediateI = intermediateI + inImgPtr[srcOffset1] + inImgPtr[srcOffset5];
				intermediateH = intermediateH + inImgPtr[srcOffset2] + inImgPtr[srcOffset4];
				intermediateG = intermediateG + inImgPtr[srcOffset3];

				//Multiply + accumulate
				intermediate = intermediateA * A + intermediateB * B + intermediateC * C +
					intermediateD * D + intermediateE * E + intermediateF * F +
					intermediateG * G + intermediateH * H + intermediateI * I +
					intermediateJ * J;

				//shift right
				intermediate = intermediate >> shift;

				//saturate
				if (intermediate < 0)
				{
					intermediate = 0;
				}
				else if (intermediate > 255)
				{
					intermediate = 255;
				}

				outImgPtr[dstOffset] = intermediate;
			}
		}
	}
	else
	{
		//border pixels processing (constant=255 border mode)
		for (int yi = y; yi < y + ySize; ++yi)
		{
			for (int xi = x; xi < x + xSize; ++xi)
			{
				//Start src offset
				int srcOffset0;
				int srcOffset1;
				int srcOffset2;
				int srcOffset3;
				int srcOffset4;
				int srcOffset5;
				int srcOffset6;

				if ((xi - 3) < 0 || (xi - 3) >= widthInImg || (yi - 3) < 0 || (yi - 3) >= heightInImg)
					srcOffset0 = -1;
				else
					srcOffset0 = (yi - 3)*rowPitchInImg + (xi - 3)*pixelStrideInImg;

				if ((xi - 2) < 0 || (xi - 2) >= widthInImg || (yi - 3) < 0 || (yi - 3) >= heightInImg)
					srcOffset1 = -1;
				else
					srcOffset1 = (yi - 3)*rowPitchInImg + (xi - 2)*pixelStrideInImg;

				if ((xi - 1) < 0 || (xi - 1) >= widthInImg || (yi - 3) < 0 || (yi - 3) >= heightInImg)
					srcOffset2 = -1;
				else
					srcOffset2 = (yi - 3)*rowPitchInImg + (xi - 1)*pixelStrideInImg;

				if ((xi) < 0 || (xi) >= widthInImg || (yi - 3) < 0 || (yi - 3) >= heightInImg)
					srcOffset3 = -1;
				else
					srcOffset3 = (yi - 3)*rowPitchInImg + (xi)*pixelStrideInImg;

				if ((xi + 1) < 0 || (xi + 1) >= widthInImg || (yi - 3) < 0 || (yi - 3) >= heightInImg)
					srcOffset4 = -1;
				else
					srcOffset4 = (yi - 3)*rowPitchInImg + (xi + 1)*pixelStrideInImg;

				if ((xi + 2) < 0 || (xi + 2) >= widthInImg || (yi - 3) < 0 || (yi - 3) >= heightInImg)
					srcOffset5 = -1;
				else
					srcOffset5 = (yi - 3)*rowPitchInImg + (xi + 2)*pixelStrideInImg;

				if ((xi + 3) < 0 || (xi + 3) >= widthInImg || (yi - 3) < 0 || (yi - 3) >= heightInImg)
					srcOffset6 = -1;
				else
					srcOffset6 = (yi - 3)*rowPitchInImg + (xi + 3)*pixelStrideInImg;


				int dstOffset = yi*rowPitchOutImg + xi*pixelStrideOutImg;

				int intermediate;
				short intermediateA = 0;
				short intermediateB = 0;
				short intermediateC = 0;
				short intermediateD = 0;
				short intermediateE = 0;
				short intermediateF = 0;
				short intermediateG = 0;
				short intermediateH = 0;
				short intermediateI = 0;
				short intermediateJ = 0;

				//1st line JIHGHIJ
				intermediateJ = ((srcOffset0 < 0) ? border : inImgPtr[srcOffset0]) + ((srcOffset6 < 0) ? border : inImgPtr[srcOffset6]);
				intermediateI = ((srcOffset1 < 0) ? border : inImgPtr[srcOffset1]) + ((srcOffset5 < 0) ? border : inImgPtr[srcOffset5]);
				intermediateH = ((srcOffset2 < 0) ? border : inImgPtr[srcOffset2]) + ((srcOffset4 < 0) ? border : inImgPtr[srcOffset4]);
				intermediateG = ((srcOffset3 < 0) ? border : inImgPtr[srcOffset3]);

				if ((xi - 3) < 0 || (xi - 3) >= widthInImg || (yi - 2) < 0 || (yi - 2) >= heightInImg)
					srcOffset0 = -1;
				else
					srcOffset0 = (yi - 2)*rowPitchInImg + (xi - 3)*pixelStrideInImg;

				if ((+xi - 2) < 0 || (+xi - 2) >= widthInImg || (yi - 2) < 0 || (yi - 2) >= heightInImg)
					srcOffset1 = -1;
				else
					srcOffset1 = (yi - 2)*rowPitchInImg + (xi - 2)*pixelStrideInImg;

				if ((xi - 1) < 0 || (xi - 1) >= widthInImg || (yi - 2) < 0 || (yi - 2) >= heightInImg)
					srcOffset2 = -1;
				else
					srcOffset2 = (yi - 2)*rowPitchInImg + (xi - 1)*pixelStrideInImg;

				if ((xi) < 0 || (xi) >= widthInImg || (yi - 2) < 0 || (yi - 2) >= heightInImg)
					srcOffset3 = -1;
				else
					srcOffset3 = (yi - 2)*rowPitchInImg + (xi)*pixelStrideInImg;

				if ((xi + 1) < 0 || (xi + 1) >= widthInImg || (yi - 2) < 0 || (yi - 2) >= heightInImg)
					srcOffset4 = -1;
				else
					srcOffset4 = (yi - 2)*rowPitchInImg + (xi + 1)*pixelStrideInImg;

				if ((xi + 2) < 0 || (xi + 2) >= widthInImg || (yi - 2) < 0 || (yi - 2) >= heightInImg)
					srcOffset5 = -1;
				else
					srcOffset5 = (yi - 2)*rowPitchInImg + (xi + 2)*pixelStrideInImg;

				if ((xi + 3) < 0 || (xi + 3) >= widthInImg || (yi - 2) < 0 || (yi - 2) >= heightInImg)
					srcOffset6 = -1;
				else
					srcOffset6 = (yi - 2)*rowPitchInImg + (xi + 3)*pixelStrideInImg;

				//2nd line IFEDEFI
				intermediateI = intermediateI + ((srcOffset0 < 0) ? border : inImgPtr[srcOffset0]) + ((srcOffset6 < 0) ? border : inImgPtr[srcOffset6]);
				intermediateF = ((srcOffset1 < 0) ? border : inImgPtr[srcOffset1]) + ((srcOffset5 < 0) ? border : inImgPtr[srcOffset5]);
				intermediateE = ((srcOffset2 < 0) ? border : inImgPtr[srcOffset2]) + ((srcOffset4 < 0) ? border : inImgPtr[srcOffset4]);
				intermediateD = ((srcOffset3 < 0) ? border : inImgPtr[srcOffset3]);

				if ((xi - 3) < 0 || (xi - 3) >= widthInImg || (yi - 1) < 0 || (yi - 1) >= heightInImg)
					srcOffset0 = -1;
				else
					srcOffset0 = (yi - 1)*rowPitchInImg + (xi - 3)*pixelStrideInImg;

				if ((xi - 2) < 0 || (xi - 2) >= widthInImg || (yi - 1) < 0 || (yi - 1) >= heightInImg)
					srcOffset1 = -1;
				else
					srcOffset1 = (yi - 1)*rowPitchInImg + (xi - 2)*pixelStrideInImg;

				if ((xi - 1) < 0 || (xi - 1) >= widthInImg || (yi - 1) < 0 || (yi - 1) >= heightInImg)
					srcOffset2 = -1;
				else
					srcOffset2 = (yi - 1)*rowPitchInImg + (xi - 1)*pixelStrideInImg;

				if ((xi) < 0 || (xi) >= widthInImg || (yi - 1) < 0 || (yi - 1) >= heightInImg)
					srcOffset3 = -1;
				else
					srcOffset3 = (yi - 1)*rowPitchInImg + (xi)*pixelStrideInImg;

				if ((xi + 1) < 0 || (xi + 1) >= widthInImg || (yi - 1) < 0 || (yi - 1) >= heightInImg)
					srcOffset4 = -1;
				else
					srcOffset4 = (yi - 1)*rowPitchInImg + (xi + 1)*pixelStrideInImg;

				if ((xi + 2) < 0 || (xi + 2) >= widthInImg || (yi - 1) < 0 || (yi - 1) >= heightInImg)
					srcOffset5 = -1;
				else
					srcOffset5 = (yi - 1)*rowPitchInImg + (xi + 2)*pixelStrideInImg;

				if ((xi + 3) < 0 || (xi + 3) >= widthInImg || (yi - 1) < 0 || (yi - 1) >= heightInImg)
					srcOffset6 = -1;
				else
					srcOffset6 = (yi - 1)*rowPitchInImg + (xi + 3)*pixelStrideInImg;

				//3rd line HECBCEH
				intermediateH = intermediateH + ((srcOffset0 < 0) ? border : inImgPtr[srcOffset0]) + ((srcOffset6 < 0) ? border : inImgPtr[srcOffset6]);
				intermediateE = intermediateE + ((srcOffset1 < 0) ? border : inImgPtr[srcOffset1]) + ((srcOffset5 < 0) ? border : inImgPtr[srcOffset5]);
				intermediateC = ((srcOffset2 < 0) ? border : inImgPtr[srcOffset2]) + ((srcOffset4 < 0) ? border : inImgPtr[srcOffset4]);
				intermediateB = ((srcOffset3 < 0) ? border : inImgPtr[srcOffset3]);


				if ((xi - 3) < 0 || (xi - 3) >= widthInImg || (yi) < 0 || (yi) >= heightInImg)
					srcOffset0 = -1;
				else
					srcOffset0 = (yi)*rowPitchInImg + (xi - 3)*pixelStrideInImg;

				if ((xi - 2) < 0 || (xi - 2) >= widthInImg || (yi) < 0 || (yi) >= heightInImg)
					srcOffset1 = -1;
				else
					srcOffset1 = (yi)*rowPitchInImg + (xi - 2)*pixelStrideInImg;

				if ((xi - 1) < 0 || (xi - 1) >= widthInImg || (yi) < 0 || (yi) >= heightInImg)
					srcOffset2 = -1;
				else
					srcOffset2 = (yi)*rowPitchInImg + (xi - 1)*pixelStrideInImg;

				if ((xi) < 0 || (xi) >= widthInImg || (yi) < 0 || (yi) >= heightInImg)
					srcOffset3 = -1;
				else
					srcOffset3 = (yi)*rowPitchInImg + (xi)*pixelStrideInImg;

				if ((xi + 1) < 0 || (xi + 1) >= widthInImg || (yi) < 0 || (yi) >= heightInImg)
					srcOffset4 = -1;
				else
					srcOffset4 = (yi)*rowPitchInImg + (xi + 1)*pixelStrideInImg;

				if ((xi + 2) < 0 || (xi + 2) >= widthInImg || (yi) < 0 || (yi) >= heightInImg)
					srcOffset5 = -1;
				else
					srcOffset5 = (yi)*rowPitchInImg + (xi + 2)*pixelStrideInImg;

				if ((xi + 3) < 0 || (xi + 3) >= widthInImg || (yi) < 0 || (yi) >= heightInImg)
					srcOffset6 = -1;
				else
					srcOffset6 = (yi)*rowPitchInImg + (xi + 3)*pixelStrideInImg;

				//4th line GDBABDG
				intermediateG = intermediateG + ((srcOffset0 < 0) ? border : inImgPtr[srcOffset0]) + ((srcOffset6 < 0) ? border : inImgPtr[srcOffset6]);
				intermediateD = intermediateD + ((srcOffset1 < 0) ? border : inImgPtr[srcOffset1]) + ((srcOffset5 < 0) ? border : inImgPtr[srcOffset5]);
				intermediateB = intermediateB + ((srcOffset2 < 0) ? border : inImgPtr[srcOffset2]) + ((srcOffset4 < 0) ? border : inImgPtr[srcOffset4]);
				intermediateA = ((srcOffset3 < 0) ? border : inImgPtr[srcOffset3]);


				if ((xi - 3) < 0 || (xi - 3) >= widthInImg || (yi + 1) < 0 || (yi + 1) >= heightInImg)
					srcOffset0 = -1;
				else
					srcOffset0 = (yi + 1)*rowPitchInImg + (xi - 3)*pixelStrideInImg;

				if ((xi - 2) < 0 || (xi - 2) >= widthInImg || (yi + 1) < 0 || (yi + 1) >= heightInImg)
					srcOffset1 = -1;
				else
					srcOffset1 = (yi + 1)*rowPitchInImg + (xi - 2)*pixelStrideInImg;

				if ((xi - 1) < 0 || (xi - 1) >= widthInImg || (yi + 1) < 0 || (yi + 1) >= heightInImg)
					srcOffset2 = -1;
				else
					srcOffset2 = (yi + 1)*rowPitchInImg + (xi - 1)*pixelStrideInImg;

				if ((xi) < 0 || (xi) >= widthInImg || (yi + 1) < 0 || (yi + 1) >= heightInImg)
					srcOffset3 = -1;
				else
					srcOffset3 = (yi + 1)*rowPitchInImg + (xi)*pixelStrideInImg;

				if ((xi + 1) < 0 || (xi + 1) >= widthInImg || (yi + 1) < 0 || (yi + 1) >= heightInImg)
					srcOffset4 = -1;
				else
					srcOffset4 = (yi + 1)*rowPitchInImg + (xi + 1)*pixelStrideInImg;

				if ((xi + 2) < 0 || (xi + 2) >= widthInImg || (yi + 1) < 0 || (yi + 1) >= heightInImg)
					srcOffset5 = -1;
				else
					srcOffset5 = (yi + 1)*rowPitchInImg + (xi + 2)*pixelStrideInImg;

				if ((xi + 3) < 0 || (xi + 3) >= widthInImg || (yi + 1) < 0 || (yi + 1) >= heightInImg)
					srcOffset6 = -1;
				else
					srcOffset6 = (yi + 1)*rowPitchInImg + (xi + 3)*pixelStrideInImg;

				//5th line HECBCEH
				intermediateH = intermediateH + ((srcOffset0 < 0) ? border : inImgPtr[srcOffset0]) + ((srcOffset6 < 0) ? border : inImgPtr[srcOffset6]);
				intermediateE = intermediateE + ((srcOffset1 < 0) ? border : inImgPtr[srcOffset1]) + ((srcOffset5 < 0) ? border : inImgPtr[srcOffset5]);
				intermediateC = intermediateC + ((srcOffset2 < 0) ? border : inImgPtr[srcOffset2]) + ((srcOffset4 < 0) ? border : inImgPtr[srcOffset4]);
				intermediateB = intermediateB + ((srcOffset3 < 0) ? border : inImgPtr[srcOffset3]);


				if ((xi - 3) < 0 || (xi - 3) >= widthInImg || (yi + 2) < 0 || (yi + 2) >= heightInImg)
					srcOffset0 = -1;
				else
					srcOffset0 = (yi + 2)*rowPitchInImg + (xi - 3)*pixelStrideInImg;

				if ((xi - 2) < 0 || (xi - 2) >= widthInImg || (yi + 2) < 0 || (yi + 2) >= heightInImg)
					srcOffset1 = -1;
				else
					srcOffset1 = (yi + 2)*rowPitchInImg + (xi - 2)*pixelStrideInImg;

				if ((xi - 1) < 0 || (xi - 1) >= widthInImg || (yi + 2) < 0 || (yi + 2) >= heightInImg)
					srcOffset2 = -1;
				else
					srcOffset2 = (yi + 2)*rowPitchInImg + (xi - 1)*pixelStrideInImg;

				if ((xi) < 0 || (xi) >= widthInImg || (yi + 2) < 0 || (yi + 2) >= heightInImg)
					srcOffset3 = -1;
				else
					srcOffset3 = (yi + 2)*rowPitchInImg + (xi)*pixelStrideInImg;

				if ((xi + 1) < 0 || (xi + 1) >= widthInImg || (yi + 2) < 0 || (yi + 2) >= heightInImg)
					srcOffset4 = -1;
				else
					srcOffset4 = (yi + 2)*rowPitchInImg + (xi + 1)*pixelStrideInImg;

				if ((xi + 2) < 0 || (xi + 2) >= widthInImg || (yi + 2) < 0 || (yi + 2) >= heightInImg)
					srcOffset5 = -1;
				else
					srcOffset5 = (yi + 2)*rowPitchInImg + (xi + 2)*pixelStrideInImg;

				if ((xi + 3) < 0 || (xi + 3) >= widthInImg || (yi + 2) < 0 || (yi + 2) >= heightInImg)
					srcOffset6 = -1;
				else
					srcOffset6 = (yi + 2)*rowPitchInImg + (xi + 3)*pixelStrideInImg;

				//6th line IFEDEFI
				intermediateI = intermediateI + ((srcOffset0 < 0) ? border : inImgPtr[srcOffset0]) + ((srcOffset6 < 0) ? border : inImgPtr[srcOffset6]);
				intermediateF = intermediateF + ((srcOffset1 < 0) ? border : inImgPtr[srcOffset1]) + ((srcOffset5 < 0) ? border : inImgPtr[srcOffset5]);
				intermediateE = intermediateE + ((srcOffset2 < 0) ? border : inImgPtr[srcOffset2]) + ((srcOffset4 < 0) ? border : inImgPtr[srcOffset4]);
				intermediateD = intermediateD + ((srcOffset3 < 0) ? border : inImgPtr[srcOffset3]);


				if ((xi - 3) < 0 || (xi - 3) >= widthInImg || (yi + 3) < 0 || (yi + 3) >= heightInImg)
					srcOffset0 = -1;
				else
					srcOffset0 = (yi + 3)*rowPitchInImg + (xi - 3)*pixelStrideInImg;

				if ((xi - 2) < 0 || (xi - 2) >= widthInImg || (yi + 3) < 0 || (yi + 3) >= heightInImg)
					srcOffset1 = -1;
				else
					srcOffset1 = (yi + 3)*rowPitchInImg + (xi - 2)*pixelStrideInImg;

				if ((xi - 1) < 0 || (xi - 1) >= widthInImg || (yi + 3) < 0 || (yi + 3) >= heightInImg)
					srcOffset2 = -1;
				else
					srcOffset2 = (yi + 3)*rowPitchInImg + (xi - 1)*pixelStrideInImg;

				if ((xi) < 0 || (xi) >= widthInImg || (yi + 3) < 0 || (yi + 3) >= heightInImg)
					srcOffset3 = -1;
				else
					srcOffset3 = (yi + 3)*rowPitchInImg + (xi)*pixelStrideInImg;

				if ((xi + 1) < 0 || (xi + 1) >= widthInImg || (yi + 3) < 0 || (yi + 3) >= heightInImg)
					srcOffset4 = -1;
				else
					srcOffset4 = (yi + 3)*rowPitchInImg + (xi + 1)*pixelStrideInImg;

				if ((xi + 2) < 0 || (xi + 2) >= widthInImg || (yi + 3) < 0 || (yi + 3) >= heightInImg)
					srcOffset5 = -1;
				else
					srcOffset5 = (yi + 3)*rowPitchInImg + (xi + 2)*pixelStrideInImg;

				if ((xi + 3) < 0 || (xi + 3) >= widthInImg || (yi + 3) < 0 || (yi + 3) >= heightInImg)
					srcOffset6 = -1;
				else
					srcOffset6 = (yi + 3)*rowPitchInImg + (xi + 3)*pixelStrideInImg;

				//7th line JIHGHIJ
				intermediateJ = intermediateJ + ((srcOffset0 < 0) ? border : inImgPtr[srcOffset0]) + ((srcOffset6 < 0) ? border : inImgPtr[srcOffset6]);
				intermediateI = intermediateI + ((srcOffset1 < 0) ? border : inImgPtr[srcOffset1]) + ((srcOffset5 < 0) ? border : inImgPtr[srcOffset5]);
				intermediateH = intermediateH + ((srcOffset2 < 0) ? border : inImgPtr[srcOffset2]) + ((srcOffset4 < 0) ? border : inImgPtr[srcOffset4]);
				intermediateG = intermediateG + ((srcOffset3 < 0) ? border : inImgPtr[srcOffset3]);


				//Multiply + accumulate
				intermediate = intermediateA * A + intermediateB * B + intermediateC * C +
					intermediateD * D + intermediateE * E + intermediateF * F +
					intermediateG * G + intermediateH * H + intermediateI * I +
					intermediateJ * J;

				//shift right
				intermediate = intermediate >> shift;

				//saturate
				if (intermediate < 0)
				{
					intermediate = 0;
				}
				else if (intermediate > 255)
				{
					intermediate = 255;
				}

				outImgPtr[dstOffset] = intermediate;
			}
		}
	}
}

kernel void symm7x7tiled_opt(

	// The kernel signature is completely defined by param_types array passed to
	// vxAddDeviceKernelIntel function in host C code. Each OpenVX parameter is translated
	// to one or multiple arguments of OpenCL kernel.
	// Even if the kernel's body doesn't use all these arguments, they should be defined here
	// anyway, because OpenVX run-time relies on the order and specific number of parameters,
	// to set them correctly when calling this kernel and translating OpenVX parameters.

	// OpenVX kernel 0-th parameter has type vx_image, it is mapped to these 5 OpenCL kernel arguments
	// This is input VX_DF_IMAGE_U8 image tile

	global const uchar* inImgPtr,

	int in_tile_x,        // x coordinate of the tile
	int in_tile_y,        // y coordinate of the tile
	int in_tile_width,    // width of tile
	int in_tile_height,   // height of tile

	unsigned int        widthInImg,        // width of the input image
	unsigned int        heightInImg,       // height of the input image
	unsigned int        pixelStrideInImg,  // pixel stride in bytes
	unsigned int        rowPitchInImg,     // row stride in bytes

	// OpenVX kernel 1-st parameter has type vx_image, it is mapped to these 5 OpenCL kernel arguments
	// This is output VX_DF_IMAGE_U8 image

	global uchar*       outImgPtr,

	int out_tile_x,        // x coordinate of the tile
	int out_tile_y,        // y coordinate of the tile
	int out_tile_width,    // width of tile
	int out_tile_height,   // height of tile

	unsigned int        widthOutImg,       // width of the output image
	unsigned int        heightOutImg,      // height of the output image
	unsigned int        pixelStrideOutImg, // pixel stride in bytes
	unsigned int        rowPitchOutImg     // row stride in bytes
	)
{
	int x = get_global_id(0);   // 0..(widthInImg/VX_OPENCL_WORK_ITEM_XSIZE-1)
	int y = get_global_id(1);   // 0..(heightInImg/VX_OPENCL_WORK_ITEM_YSIZE-1)


	// coordinates of a top-left pixel in output image area for processing by one WI
	// some of the pixels from WORK_ITEM_XSIZE x WORK_ITEM_YSIZE area are not valid on the border
	// they will not be processed (look at if's in the code below)
	x *= WORK_ITEM_XSIZE;
	y *= WORK_ITEM_YSIZE;

	int xSize = WORK_ITEM_XSIZE;
	int ySize = WORK_ITEM_YSIZE;

	//Symm7x7 layout
	//   JIHGHIJ
	//   IFEDEFI
	//   HECBCEH
	//   GDBABGD
	//   HECBCEH
	//   IFEDEFI
	//   JIHGHIJ

	//Symm7x7 coefficients
	const short A = 1140; const short B = -118; const short C = 526; const short D = 290; const short E = -236;
	const short F = 64; const short G = -128; const short H = -5; const short I = -87; const short J = -7;

	const short shift = 10;
	const short border = 255;
	int y_shift = out_tile_y - in_tile_y;

	if (
		((out_tile_x + x) > 2) &&
		((out_tile_x + x) < widthOutImg - 3) &&
		((out_tile_y + y) > 2) &&
		((out_tile_y + y) < heightOutImg - 3)
		)
	{
		//non border pixels processing
		for (int yi = y; yi < y + ySize; ++yi)
		{
			for (int xi = x; xi < x + xSize; ++xi)
			{
				//Start src offset
				int srcOffset0 = (yi - 3 + y_shift)*rowPitchInImg + (xi - 3)*pixelStrideInImg;
				int srcOffset1 = (yi - 3 + y_shift)*rowPitchInImg + (xi - 2)*pixelStrideInImg;
				int srcOffset2 = (yi - 3 + y_shift)*rowPitchInImg + (xi - 1)*pixelStrideInImg;
				int srcOffset3 = (yi - 3 + y_shift)*rowPitchInImg + (xi)*pixelStrideInImg;
				int srcOffset4 = (yi - 3 + y_shift)*rowPitchInImg + (xi + 1)*pixelStrideInImg;
				int srcOffset5 = (yi - 3 + y_shift)*rowPitchInImg + (xi + 2)*pixelStrideInImg;
				int srcOffset6 = (yi - 3 + y_shift)*rowPitchInImg + (xi + 3)*pixelStrideInImg;

				int dstOffset = yi*rowPitchOutImg + xi*pixelStrideOutImg;

				int intermediate;
				short intermediateA = 0;
				short intermediateB = 0;
				short intermediateC = 0;
				short intermediateD = 0;
				short intermediateE = 0;
				short intermediateF = 0;
				short intermediateG = 0;
				short intermediateH = 0;
				short intermediateI = 0;
				short intermediateJ = 0;

				//1st line JIHGHIJ
				intermediateJ = inImgPtr[srcOffset0] + inImgPtr[srcOffset6];
				intermediateI = inImgPtr[srcOffset1] + inImgPtr[srcOffset5];
				intermediateH = inImgPtr[srcOffset2] + inImgPtr[srcOffset4];
				intermediateG = inImgPtr[srcOffset3];

				srcOffset0 = srcOffset0 + rowPitchInImg;
				srcOffset1 = srcOffset1 + rowPitchInImg;
				srcOffset2 = srcOffset2 + rowPitchInImg;
				srcOffset3 = srcOffset3 + rowPitchInImg;
				srcOffset4 = srcOffset4 + rowPitchInImg;
				srcOffset5 = srcOffset5 + rowPitchInImg;
				srcOffset6 = srcOffset6 + rowPitchInImg;

				//2nd line IFEDEFI
				intermediateI = intermediateI + inImgPtr[srcOffset0] + inImgPtr[srcOffset6];
				intermediateF = inImgPtr[srcOffset1] + inImgPtr[srcOffset5];
				intermediateE = inImgPtr[srcOffset2] + inImgPtr[srcOffset4];
				intermediateD = inImgPtr[srcOffset3];

				srcOffset0 = srcOffset0 + rowPitchInImg;
				srcOffset1 = srcOffset1 + rowPitchInImg;
				srcOffset2 = srcOffset2 + rowPitchInImg;
				srcOffset3 = srcOffset3 + rowPitchInImg;
				srcOffset4 = srcOffset4 + rowPitchInImg;
				srcOffset5 = srcOffset5 + rowPitchInImg;
				srcOffset6 = srcOffset6 + rowPitchInImg;

				//3rd line HECBCEH
				intermediateH = intermediateH + inImgPtr[srcOffset0] + inImgPtr[srcOffset6];
				intermediateE = intermediateE + inImgPtr[srcOffset1] + inImgPtr[srcOffset5];
				intermediateC = inImgPtr[srcOffset2] + inImgPtr[srcOffset4];
				intermediateB = inImgPtr[srcOffset3];

				srcOffset0 = srcOffset0 + rowPitchInImg;
				srcOffset1 = srcOffset1 + rowPitchInImg;
				srcOffset2 = srcOffset2 + rowPitchInImg;
				srcOffset3 = srcOffset3 + rowPitchInImg;
				srcOffset4 = srcOffset4 + rowPitchInImg;
				srcOffset5 = srcOffset5 + rowPitchInImg;
				srcOffset6 = srcOffset6 + rowPitchInImg;

				//4th line GDBABDG
				intermediateG = intermediateG + inImgPtr[srcOffset0] + inImgPtr[srcOffset6];
				intermediateD = intermediateD + inImgPtr[srcOffset1] + inImgPtr[srcOffset5];
				intermediateB = intermediateB + inImgPtr[srcOffset2] + inImgPtr[srcOffset4];
				intermediateA = inImgPtr[srcOffset3];

				srcOffset0 = srcOffset0 + rowPitchInImg;
				srcOffset1 = srcOffset1 + rowPitchInImg;
				srcOffset2 = srcOffset2 + rowPitchInImg;
				srcOffset3 = srcOffset3 + rowPitchInImg;
				srcOffset4 = srcOffset4 + rowPitchInImg;
				srcOffset5 = srcOffset5 + rowPitchInImg;
				srcOffset6 = srcOffset6 + rowPitchInImg;

				//5th line HECBCEH
				intermediateH = intermediateH + inImgPtr[srcOffset0] + inImgPtr[srcOffset6];
				intermediateE = intermediateE + inImgPtr[srcOffset1] + inImgPtr[srcOffset5];
				intermediateC = intermediateC + inImgPtr[srcOffset2] + inImgPtr[srcOffset4];
				intermediateB = intermediateB + inImgPtr[srcOffset3];

				srcOffset0 = srcOffset0 + rowPitchInImg;
				srcOffset1 = srcOffset1 + rowPitchInImg;
				srcOffset2 = srcOffset2 + rowPitchInImg;
				srcOffset3 = srcOffset3 + rowPitchInImg;
				srcOffset4 = srcOffset4 + rowPitchInImg;
				srcOffset5 = srcOffset5 + rowPitchInImg;
				srcOffset6 = srcOffset6 + rowPitchInImg;

				//6th line IFEDEFI
				intermediateI = intermediateI + inImgPtr[srcOffset0] + inImgPtr[srcOffset6];
				intermediateF = intermediateF + inImgPtr[srcOffset1] + inImgPtr[srcOffset5];
				intermediateE = intermediateE + inImgPtr[srcOffset2] + inImgPtr[srcOffset4];
				intermediateD = intermediateD + inImgPtr[srcOffset3];

				srcOffset0 = srcOffset0 + rowPitchInImg;
				srcOffset1 = srcOffset1 + rowPitchInImg;
				srcOffset2 = srcOffset2 + rowPitchInImg;
				srcOffset3 = srcOffset3 + rowPitchInImg;
				srcOffset4 = srcOffset4 + rowPitchInImg;
				srcOffset5 = srcOffset5 + rowPitchInImg;
				srcOffset6 = srcOffset6 + rowPitchInImg;

				//7th line JIHGHIJ
				intermediateJ = intermediateJ + inImgPtr[srcOffset0] + inImgPtr[srcOffset6];
				intermediateI = intermediateI + inImgPtr[srcOffset1] + inImgPtr[srcOffset5];
				intermediateH = intermediateH + inImgPtr[srcOffset2] + inImgPtr[srcOffset4];
				intermediateG = intermediateG + inImgPtr[srcOffset3];

				//Multiply + accumulate
				intermediate = intermediateA * A + intermediateB * B + intermediateC * C +
					intermediateD * D + intermediateE * E + intermediateF * F +
					intermediateG * G + intermediateH * H + intermediateI * I +
					intermediateJ * J;

				//shift right
				intermediate = intermediate >> shift;

				//saturate
				if (intermediate < 0)
				{
					intermediate = 0;
				}
				else if (intermediate > 255)
				{
					intermediate = 255;
				}

				outImgPtr[dstOffset] = intermediate;
			}
		}
	}
	else
	{
		//border pixels processing (constant=255 border mode)
		for (int yi = y; yi < y + ySize; ++yi)
		{
			for (int xi = x; xi < x + xSize; ++xi)
			{
				//Start src offset
				int srcOffset0;
				int srcOffset1;
				int srcOffset2;
				int srcOffset3;
				int srcOffset4;
				int srcOffset5;
				int srcOffset6;

				if ((out_tile_x + xi - 3) < 0 || (out_tile_x + xi - 3) >= widthOutImg || (out_tile_y + yi - 3) < 0 || (out_tile_y + yi - 3) >= heightOutImg)
					srcOffset0 = -1;
				else
					srcOffset0 = (yi - 3 + y_shift)*rowPitchInImg + (xi - 3)*pixelStrideInImg;

				if ((out_tile_x + xi - 2) < 0 || (out_tile_x + xi - 2) >= widthOutImg || (out_tile_y + yi - 3) < 0 || (out_tile_y + yi - 3) >= heightOutImg)
					srcOffset1 = -1;
				else
					srcOffset1 = (yi - 3 + y_shift)*rowPitchInImg + (xi - 2)*pixelStrideInImg;

				if ((out_tile_x + xi - 1) < 0 || (out_tile_x + xi - 1) >= widthOutImg || (out_tile_y + yi - 3) < 0 || (out_tile_y + yi - 3) >= heightOutImg)
					srcOffset2 = -1;
				else
					srcOffset2 = (yi - 3 + y_shift)*rowPitchInImg + (xi - 1)*pixelStrideInImg;

				if ((out_tile_x + xi) < 0 || (out_tile_x + xi) >= widthOutImg || (out_tile_y + yi - 3) < 0 || (out_tile_y + yi - 3) >= heightOutImg)
					srcOffset3 = -1;
				else
					srcOffset3 = (yi - 3 + y_shift)*rowPitchInImg + (xi)*pixelStrideInImg;

				if ((out_tile_x + xi + 1) < 0 || (out_tile_x + xi + 1) >= widthOutImg || (out_tile_y + yi - 3) < 0 || (out_tile_y + yi - 3) >= heightOutImg)
					srcOffset4 = -1;
				else
					srcOffset4 = (yi - 3 + y_shift)*rowPitchInImg + (xi + 1)*pixelStrideInImg;

				if ((out_tile_x + xi + 2) < 0 || (out_tile_x + xi + 2) >= widthOutImg || (out_tile_y + yi - 3) < 0 || (out_tile_y + yi - 3) >= heightOutImg)
					srcOffset5 = -1;
				else
					srcOffset5 = (yi - 3 + y_shift)*rowPitchInImg + (xi + 2)*pixelStrideInImg;

				if ((out_tile_x + xi + 3) < 0 || (out_tile_x + xi + 3) >= widthOutImg || (out_tile_y + yi - 3) < 0 || (out_tile_y + yi - 3) >= heightOutImg)
					srcOffset6 = -1;
				else
					srcOffset6 = (yi - 3 + y_shift)*rowPitchInImg + (xi + 3)*pixelStrideInImg;


				int dstOffset = yi*rowPitchOutImg + xi*pixelStrideOutImg;

				int intermediate;
				short intermediateA = 0;
				short intermediateB = 0;
				short intermediateC = 0;
				short intermediateD = 0;
				short intermediateE = 0;
				short intermediateF = 0;
				short intermediateG = 0;
				short intermediateH = 0;
				short intermediateI = 0;
				short intermediateJ = 0;

				//1st line JIHGHIJ
				intermediateJ = ((srcOffset0 < 0) ? border : inImgPtr[srcOffset0]) + ((srcOffset6 < 0) ? border : inImgPtr[srcOffset6]);
				intermediateI = ((srcOffset1 < 0) ? border : inImgPtr[srcOffset1]) + ((srcOffset5 < 0) ? border : inImgPtr[srcOffset5]);
				intermediateH = ((srcOffset2 < 0) ? border : inImgPtr[srcOffset2]) + ((srcOffset4 < 0) ? border : inImgPtr[srcOffset4]);
				intermediateG = ((srcOffset3 < 0) ? border : inImgPtr[srcOffset3]);

				if ((out_tile_x + xi - 3) < 0 || (out_tile_x + xi - 3) >= widthOutImg || (out_tile_y + yi - 2) < 0 || (out_tile_y + yi - 2) >= heightOutImg)
					srcOffset0 = -1;
				else
					srcOffset0 = (yi - 2 + y_shift)*rowPitchInImg + (xi - 3)*pixelStrideInImg;

				if ((out_tile_x + xi - 2) < 0 || (out_tile_x + xi - 2) >= widthOutImg || (out_tile_y + yi - 2) < 0 || (out_tile_y + yi - 2) >= heightOutImg)
					srcOffset1 = -1;
				else
					srcOffset1 = (yi - 2 + y_shift)*rowPitchInImg + (xi - 2)*pixelStrideInImg;

				if ((out_tile_x + xi - 1) < 0 || (out_tile_x + xi - 1) >= widthOutImg || (out_tile_y + yi - 2) < 0 || (out_tile_y + yi - 2) >= heightOutImg)
					srcOffset2 = -1;
				else
					srcOffset2 = (yi - 2 + y_shift)*rowPitchInImg + (xi - 1)*pixelStrideInImg;

				if ((out_tile_x + xi) < 0 || (out_tile_x + xi) >= widthOutImg || (out_tile_y + yi - 2) < 0 || (out_tile_y + yi - 2) >= heightOutImg)
					srcOffset3 = -1;
				else
					srcOffset3 = (yi - 2 + y_shift)*rowPitchInImg + (xi)*pixelStrideInImg;

				if ((out_tile_x + xi + 1) < 0 || (out_tile_x + xi + 1) >= widthOutImg || (out_tile_y + yi - 2) < 0 || (out_tile_y + yi - 2) >= heightOutImg)
					srcOffset4 = -1;
				else
					srcOffset4 = (yi - 2 + y_shift)*rowPitchInImg + (xi + 1)*pixelStrideInImg;

				if ((out_tile_x + xi + 2) < 0 || (out_tile_x + xi + 2) >= widthOutImg || (out_tile_y + yi - 2) < 0 || (out_tile_y + yi - 2) >= heightOutImg)
					srcOffset5 = -1;
				else
					srcOffset5 = (yi - 2 + y_shift)*rowPitchInImg + (xi + 2)*pixelStrideInImg;

				if ((out_tile_x + xi + 3) < 0 || (out_tile_x + xi + 3) >= widthOutImg || (out_tile_y + yi - 2) < 0 || (out_tile_y + yi - 2) >= heightOutImg)
					srcOffset6 = -1;
				else
					srcOffset6 = (yi - 2 + y_shift)*rowPitchInImg + (xi + 3)*pixelStrideInImg;

				//2nd line IFEDEFI
				intermediateI = intermediateI + ((srcOffset0 < 0) ? border : inImgPtr[srcOffset0]) + ((srcOffset6 < 0) ? border : inImgPtr[srcOffset6]);
				intermediateF = ((srcOffset1 < 0) ? border : inImgPtr[srcOffset1]) + ((srcOffset5 < 0) ? border : inImgPtr[srcOffset5]);
				intermediateE = ((srcOffset2 < 0) ? border : inImgPtr[srcOffset2]) + ((srcOffset4 < 0) ? border : inImgPtr[srcOffset4]);
				intermediateD = ((srcOffset3 < 0) ? border : inImgPtr[srcOffset3]);

				if ((out_tile_x + xi - 3) < 0 || (out_tile_x + xi - 3) >= widthOutImg || (out_tile_y + yi - 1) < 0 || (out_tile_y + yi - 1) >= heightOutImg)
					srcOffset0 = -1;
				else
					srcOffset0 = (yi - 1 + y_shift)*rowPitchInImg + (xi - 3)*pixelStrideInImg;

				if ((out_tile_x + xi - 2) < 0 || (out_tile_x + xi - 2) >= widthOutImg || (out_tile_y + yi - 1) < 0 || (out_tile_y + yi - 1) >= heightOutImg)
					srcOffset1 = -1;
				else
					srcOffset1 = (yi - 1 + y_shift)*rowPitchInImg + (xi - 2)*pixelStrideInImg;

				if ((out_tile_x + xi - 1) < 0 || (out_tile_x + xi - 1) >= widthOutImg || (out_tile_y + yi - 1) < 0 || (out_tile_y + yi - 1) >= heightOutImg)
					srcOffset2 = -1;
				else
					srcOffset2 = (yi - 1 + y_shift)*rowPitchInImg + (xi - 1)*pixelStrideInImg;

				if ((out_tile_x + xi) < 0 || (out_tile_x + xi) >= widthOutImg || (out_tile_y + yi - 1) < 0 || (out_tile_y + yi - 1) >= heightOutImg)
					srcOffset3 = -1;
				else
					srcOffset3 = (yi - 1 + y_shift)*rowPitchInImg + (xi)*pixelStrideInImg;

				if ((out_tile_x + xi + 1) < 0 || (out_tile_x + xi + 1) >= widthOutImg || (out_tile_y + yi - 1) < 0 || (out_tile_y + yi - 1) >= heightOutImg)
					srcOffset4 = -1;
				else
					srcOffset4 = (yi - 1 + y_shift)*rowPitchInImg + (xi + 1)*pixelStrideInImg;

				if ((out_tile_x + xi + 2) < 0 || (out_tile_x + xi + 2) >= widthOutImg || (out_tile_y + yi - 1) < 0 || (out_tile_y + yi - 1) >= heightOutImg)
					srcOffset5 = -1;
				else
					srcOffset5 = (yi - 1 + y_shift)*rowPitchInImg + (xi + 2)*pixelStrideInImg;

				if ((out_tile_x + xi + 3) < 0 || (out_tile_x + xi + 3) >= widthOutImg || (out_tile_y + yi - 1) < 0 || (out_tile_y + yi - 1) >= heightOutImg)
					srcOffset6 = -1;
				else
					srcOffset6 = (yi - 1 + y_shift)*rowPitchInImg + (xi + 3)*pixelStrideInImg;

				//3rd line HECBCEH
				intermediateH = intermediateH + ((srcOffset0 < 0) ? border : inImgPtr[srcOffset0]) + ((srcOffset6 < 0) ? border : inImgPtr[srcOffset6]);
				intermediateE = intermediateE + ((srcOffset1 < 0) ? border : inImgPtr[srcOffset1]) + ((srcOffset5 < 0) ? border : inImgPtr[srcOffset5]);
				intermediateC = ((srcOffset2 < 0) ? border : inImgPtr[srcOffset2]) + ((srcOffset4 < 0) ? border : inImgPtr[srcOffset4]);
				intermediateB = ((srcOffset3 < 0) ? border : inImgPtr[srcOffset3]);


				if ((out_tile_x + xi - 3) < 0 || (out_tile_x + xi - 3) >= widthOutImg || (out_tile_y + yi) < 0 || (out_tile_y + yi) >= heightOutImg)
					srcOffset0 = -1;
				else
					srcOffset0 = (yi + y_shift)*rowPitchInImg + (xi - 3)*pixelStrideInImg;

				if ((out_tile_x + xi - 2) < 0 || (out_tile_x + xi - 2) >= widthOutImg || (out_tile_y + yi) < 0 || (out_tile_y + yi) >= heightOutImg)
					srcOffset1 = -1;
				else
					srcOffset1 = (yi + y_shift)*rowPitchInImg + (xi - 2)*pixelStrideInImg;

				if ((out_tile_x + xi - 1) < 0 || (out_tile_x + xi - 1) >= widthOutImg || (out_tile_y + yi) < 0 || (out_tile_y + yi) >= heightOutImg)
					srcOffset2 = -1;
				else
					srcOffset2 = (yi + y_shift)*rowPitchInImg + (xi - 1)*pixelStrideInImg;

				if ((out_tile_x + xi) < 0 || (out_tile_x + xi) >= widthOutImg || (out_tile_y + yi) < 0 || (out_tile_y + yi) >= heightOutImg)
					srcOffset3 = -1;
				else
					srcOffset3 = (yi + y_shift)*rowPitchInImg + (xi)*pixelStrideInImg;

				if ((out_tile_x + xi + 1) < 0 || (out_tile_x + xi + 1) >= widthOutImg || (out_tile_y + yi) < 0 || (out_tile_y + yi) >= heightOutImg)
					srcOffset4 = -1;
				else
					srcOffset4 = (yi + y_shift)*rowPitchInImg + (xi + 1)*pixelStrideInImg;

				if ((out_tile_x + xi + 2) < 0 || (out_tile_x + xi + 2) >= widthOutImg || (out_tile_y + yi) < 0 || (out_tile_y + yi) >= heightOutImg)
					srcOffset5 = -1;
				else
					srcOffset5 = (yi + y_shift)*rowPitchInImg + (xi + 2)*pixelStrideInImg;

				if ((out_tile_x + xi + 3) < 0 || (out_tile_x + xi + 3) >= widthOutImg || (out_tile_y + yi) < 0 || (out_tile_y + yi) >= heightOutImg)
					srcOffset6 = -1;
				else
					srcOffset6 = (yi + y_shift)*rowPitchInImg + (xi + 3)*pixelStrideInImg;

				//4th line GDBABDG
				intermediateG = intermediateG + ((srcOffset0 < 0) ? border : inImgPtr[srcOffset0]) + ((srcOffset6 < 0) ? border : inImgPtr[srcOffset6]);
				intermediateD = intermediateD + ((srcOffset1 < 0) ? border : inImgPtr[srcOffset1]) + ((srcOffset5 < 0) ? border : inImgPtr[srcOffset5]);
				intermediateB = intermediateB + ((srcOffset2 < 0) ? border : inImgPtr[srcOffset2]) + ((srcOffset4 < 0) ? border : inImgPtr[srcOffset4]);
				intermediateA = ((srcOffset3 < 0) ? border : inImgPtr[srcOffset3]);


				if ((out_tile_x + xi - 3) < 0 || (out_tile_x + xi - 3) >= widthOutImg || (out_tile_y + yi + 1) < 0 || (out_tile_y + yi + 1) >= heightOutImg)
					srcOffset0 = -1;
				else
					srcOffset0 = (yi + 1 + y_shift)*rowPitchInImg + (xi - 3)*pixelStrideInImg;

				if ((out_tile_x + xi - 2) < 0 || (out_tile_x + xi - 2) >= widthOutImg || (out_tile_y + yi + 1) < 0 || (out_tile_y + yi + 1) >= heightOutImg)
					srcOffset1 = -1;
				else
					srcOffset1 = (yi + 1 + y_shift)*rowPitchInImg + (xi - 2)*pixelStrideInImg;

				if ((out_tile_x + xi - 1) < 0 || (out_tile_x + xi - 1) >= widthOutImg || (out_tile_y + yi + 1) < 0 || (out_tile_y + yi + 1) >= heightOutImg)
					srcOffset2 = -1;
				else
					srcOffset2 = (yi + 1 + y_shift)*rowPitchInImg + (xi - 1)*pixelStrideInImg;

				if ((out_tile_x + xi) < 0 || (out_tile_x + xi) >= widthOutImg || (out_tile_y + yi + 1) < 0 || (out_tile_y + yi + 1) >= heightOutImg)
					srcOffset3 = -1;
				else
					srcOffset3 = (yi + 1 + y_shift)*rowPitchInImg + (xi)*pixelStrideInImg;

				if ((out_tile_x + xi + 1) < 0 || (out_tile_x + xi + 1) >= widthOutImg || (out_tile_y + yi + 1) < 0 || (out_tile_y + yi + 1) >= heightOutImg)
					srcOffset4 = -1;
				else
					srcOffset4 = (yi + 1 + y_shift)*rowPitchInImg + (xi + 1)*pixelStrideInImg;

				if ((out_tile_x + xi + 2) < 0 || (out_tile_x + xi + 2) >= widthOutImg || (out_tile_y + yi + 1) < 0 || (out_tile_y + yi + 1) >= heightOutImg)
					srcOffset5 = -1;
				else
					srcOffset5 = (yi + 1 + y_shift)*rowPitchInImg + (xi + 2)*pixelStrideInImg;

				if ((out_tile_x + xi + 3) < 0 || (out_tile_x + xi + 3) >= widthOutImg || (out_tile_y + yi + 1) < 0 || (out_tile_y + yi + 1) >= heightOutImg)
					srcOffset6 = -1;
				else
					srcOffset6 = (yi + 1 + y_shift)*rowPitchInImg + (xi + 3)*pixelStrideInImg;

				//5th line HECBCEH
				intermediateH = intermediateH + ((srcOffset0 < 0) ? border : inImgPtr[srcOffset0]) + ((srcOffset6 < 0) ? border : inImgPtr[srcOffset6]);
				intermediateE = intermediateE + ((srcOffset1 < 0) ? border : inImgPtr[srcOffset1]) + ((srcOffset5 < 0) ? border : inImgPtr[srcOffset5]);
				intermediateC = intermediateC + ((srcOffset2 < 0) ? border : inImgPtr[srcOffset2]) + ((srcOffset4 < 0) ? border : inImgPtr[srcOffset4]);
				intermediateB = intermediateB + ((srcOffset3 < 0) ? border : inImgPtr[srcOffset3]);


				if ((out_tile_x + xi - 3) < 0 || (out_tile_x + xi - 3) >= widthOutImg || (out_tile_y + yi + 2) < 0 || (out_tile_y + yi + 2) >= heightOutImg)
					srcOffset0 = -1;
				else
					srcOffset0 = (yi + 2 + y_shift)*rowPitchInImg + (xi - 3)*pixelStrideInImg;

				if ((out_tile_x + xi - 2) < 0 || (out_tile_x + xi - 2) >= widthOutImg || (out_tile_y + yi + 2) < 0 || (out_tile_y + yi + 2) >= heightOutImg)
					srcOffset1 = -1;
				else
					srcOffset1 = (yi + 2 + y_shift)*rowPitchInImg + (xi - 2)*pixelStrideInImg;

				if ((out_tile_x + xi - 1) < 0 || (out_tile_x + xi - 1) >= widthOutImg || (out_tile_y + yi + 2) < 0 || (out_tile_y + yi + 2) >= heightOutImg)
					srcOffset2 = -1;
				else
					srcOffset2 = (yi + 2 + y_shift)*rowPitchInImg + (xi - 1)*pixelStrideInImg;

				if ((out_tile_x + xi) < 0 || (out_tile_x + xi) >= widthOutImg || (out_tile_y + yi + 2) < 0 || (out_tile_y + yi + 2) >= heightOutImg)
					srcOffset3 = -1;
				else
					srcOffset3 = (yi + 2 + y_shift)*rowPitchInImg + (xi)*pixelStrideInImg;

				if ((out_tile_x + xi + 1) < 0 || (out_tile_x + xi + 1) >= widthOutImg || (out_tile_y + yi + 2) < 0 || (out_tile_y + yi + 2) >= heightOutImg)
					srcOffset4 = -1;
				else
					srcOffset4 = (yi + 2 + y_shift)*rowPitchInImg + (xi + 1)*pixelStrideInImg;

				if ((out_tile_x + xi + 2) < 0 || (out_tile_x + xi + 2) >= widthOutImg || (out_tile_y + yi + 2) < 0 || (out_tile_y + yi + 2) >= heightOutImg)
					srcOffset5 = -1;
				else
					srcOffset5 = (yi + 2 + y_shift)*rowPitchInImg + (xi + 2)*pixelStrideInImg;

				if ((out_tile_x + xi + 3) < 0 || (out_tile_x + xi + 3) >= widthOutImg || (out_tile_y + yi + 2) < 0 || (out_tile_y + yi + 2) >= heightOutImg)
					srcOffset6 = -1;
				else
					srcOffset6 = (yi + 2 + y_shift)*rowPitchInImg + (xi + 3)*pixelStrideInImg;

				//6th line IFEDEFI
				intermediateI = intermediateI + ((srcOffset0 < 0) ? border : inImgPtr[srcOffset0]) + ((srcOffset6 < 0) ? border : inImgPtr[srcOffset6]);
				intermediateF = intermediateF + ((srcOffset1 < 0) ? border : inImgPtr[srcOffset1]) + ((srcOffset5 < 0) ? border : inImgPtr[srcOffset5]);
				intermediateE = intermediateE + ((srcOffset2 < 0) ? border : inImgPtr[srcOffset2]) + ((srcOffset4 < 0) ? border : inImgPtr[srcOffset4]);
				intermediateD = intermediateD + ((srcOffset3 < 0) ? border : inImgPtr[srcOffset3]);


				if ((out_tile_x + xi - 3) < 0 || (out_tile_x + xi - 3) >= widthOutImg || (out_tile_y + yi + 3) < 0 || (out_tile_y + yi + 3) >= heightOutImg)
					srcOffset0 = -1;
				else
					srcOffset0 = (yi + 3 + y_shift)*rowPitchInImg + (xi - 3)*pixelStrideInImg;

				if ((out_tile_x + xi - 2) < 0 || (out_tile_x + xi - 2) >= widthOutImg || (out_tile_y + yi + 3) < 0 || (out_tile_y + yi + 3) >= heightOutImg)
					srcOffset1 = -1;
				else
					srcOffset1 = (yi + 3 + y_shift)*rowPitchInImg + (xi - 2)*pixelStrideInImg;

				if ((out_tile_x + xi - 1) < 0 || (out_tile_x + xi - 1) >= widthOutImg || (out_tile_y + yi + 3) < 0 || (out_tile_y + yi + 3) >= heightOutImg)
					srcOffset2 = -1;
				else
					srcOffset2 = (yi + 3 + y_shift)*rowPitchInImg + (xi - 1)*pixelStrideInImg;

				if ((out_tile_x + xi) < 0 || (out_tile_x + xi) >= widthOutImg || (out_tile_y + yi + 3) < 0 || (out_tile_y + yi + 3) >= heightOutImg)
					srcOffset3 = -1;
				else
					srcOffset3 = (yi + 3 + y_shift)*rowPitchInImg + (xi)*pixelStrideInImg;

				if ((out_tile_x + xi + 1) < 0 || (out_tile_x + xi + 1) >= widthOutImg || (out_tile_y + yi + 3) < 0 || (out_tile_y + yi + 3) >= heightOutImg)
					srcOffset4 = -1;
				else
					srcOffset4 = (yi + 3 + y_shift)*rowPitchInImg + (xi + 1)*pixelStrideInImg;

				if ((out_tile_x + xi + 2) < 0 || (out_tile_x + xi + 2) >= widthOutImg || (out_tile_y + yi + 3) < 0 || (out_tile_y + yi + 3) >= heightOutImg)
					srcOffset5 = -1;
				else
					srcOffset5 = (yi + 3 + y_shift)*rowPitchInImg + (xi + 2)*pixelStrideInImg;

				if ((out_tile_x + xi + 3) < 0 || (out_tile_x + xi + 3) >= widthOutImg || (out_tile_y + yi + 3) < 0 || (out_tile_y + yi + 3) >= heightOutImg)
					srcOffset6 = -1;
				else
					srcOffset6 = (yi + 3 + y_shift)*rowPitchInImg + (xi + 3)*pixelStrideInImg;

				//7th line JIHGHIJ
				intermediateJ = intermediateJ + ((srcOffset0 < 0) ? border : inImgPtr[srcOffset0]) + ((srcOffset6 < 0) ? border : inImgPtr[srcOffset6]);
				intermediateI = intermediateI + ((srcOffset1 < 0) ? border : inImgPtr[srcOffset1]) + ((srcOffset5 < 0) ? border : inImgPtr[srcOffset5]);
				intermediateH = intermediateH + ((srcOffset2 < 0) ? border : inImgPtr[srcOffset2]) + ((srcOffset4 < 0) ? border : inImgPtr[srcOffset4]);
				intermediateG = intermediateG + ((srcOffset3 < 0) ? border : inImgPtr[srcOffset3]);


				//Multiply + accumulate
				intermediate = intermediateA * A + intermediateB * B + intermediateC * C +
					intermediateD * D + intermediateE * E + intermediateF * F +
					intermediateG * G + intermediateH * H + intermediateI * I +
					intermediateJ * J;

				//shift right
				intermediate = intermediate >> shift;

				//saturate
				if (intermediate < 0)
				{
					intermediate = 0;
				}
				else if (intermediate > 255)
				{
					intermediate = 255;
				}

				outImgPtr[dstOffset] = intermediate;
			}
		}
	}
}


