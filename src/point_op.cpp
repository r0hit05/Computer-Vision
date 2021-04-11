#include <iostream>
#include "Timer.h"
#include <opencv2/core.hpp>
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>
using namespace cv;
 
const double TIME = 1000;
const uchar max_bin = 255;


void threshold_naive(const Mat& src, Mat& dst, uchar threshold_val, int threshold_type = 0)
{
	Timer time(TIME);
	int rows = src.rows, cols = src.cols, channels = src.channels();

	dst.create(src.size(), src.type());

	for (int i = 0; i < rows; i++)
	{
		const uchar* p = src.ptr<uchar>(i);
		uchar* pdst = dst.ptr<uchar>(i);
		for (int j = 0; j < cols*channels; j++)
		{
			*(pdst + j) = (*(p + j)) < threshold_val? 0 :  max_bin;
		}
	}
}

void threshold_intrinsics(const Mat& src, Mat& dst, uchar threshold_val, int threshold_type = 0)
{
	Timer time(TIME);
	int rows = src.rows, cols = src.cols, channels = src.channels();

	dst.create(src.size(), src.type());

	for (int i = 0; i < rows; i++)
	{
		const uchar* p = src.ptr<uchar>(i);
		uchar* pdst = dst.ptr<uchar>(i);
		v_uint8 al = vx_setall_u8(threshold_val);
		for (int j = 0; j < cols*channels / 64; j++)
		{
			
		}
	}
}


void threshold_built_in(const Mat& src, Mat& dst, uchar threshold_val, int threshold_type = 0)
{
	Timer time(TIME);
	threshold(src, dst, threshold_val, max_bin, threshold_type);
}

int main()
{
	Mat img = imread("../images/sample_4856x3238.jpeg", 1);

	Mat dst1, dst2, dst3;
	uchar threshold_val = 128;
	int threshold_type = 0;
	imshow("Input", img);

	threshold_naive(img, dst1, threshold_val, threshold_type);
	namedWindow("Output", 1);
	imshow("Output", dst1);
	waitKey(0);

	threshold_intrinsics(img, dst2, threshold_val, threshold_type);
	namedWindow("Output", 1);
	imshow("Output", dst2);
	waitKey(0);

	threshold_built_in(img, dst3, threshold_val, threshold_type);
	namedWindow("Output", 1);
	imshow("Output", dst3);
	waitKey(0);
	return 0;
}
