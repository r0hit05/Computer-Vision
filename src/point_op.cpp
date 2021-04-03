#include <iostream>
#include "Timer.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>

using namespace cv;

int Timer::cnt = 0;
const double TIME = 1000;

void apply(const Mat& src, Mat& dst, float alpha, float beta)
{
	Timer time(TIME);
	int rows = src.rows, cols = src.cols, channels = src.channels();

	dst.create(src.size(), src.type());

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			for (int c = 0; c < channels; c++)
			{
				dst.at<Vec3b>(i, j)[c] = saturate_cast<uchar>(alpha * src.at<Vec3b>(i, j)[c] + beta);
			}
		}
	}
}

void apply_fast(const Mat& src, Mat& dst, float alpha, float beta)
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
			*(pdst + j) = saturate_cast<uchar>(alpha * (*(p + j)) + beta);
		}
	}
}

void apply(const Mat& src, Mat& dst, float gamma)
{
	Timer time(TIME);
	int rows = src.rows, cols = src.cols, channels = src.channels();

	dst.create(src.size(), src.type());

	for (int i = 0; i < rows; i++)
	{
		const uchar* p = src.ptr<uchar>(i);
		uchar* pdst = dst.ptr<uchar>(i);
		for (int j = 0; j < cols * channels; j++)
		{
			*(pdst + j) = saturate_cast<uchar>(pow((float)*(p + j)/255, gamma)*255);
		}
	}
}



void apply_memo(const Mat& src, Mat& dst, float gamma)
{
	Timer time(TIME);
	int rows = src.rows, cols = src.cols, channels = src.channels();

	dst.create(src.size(), src.type());

	for (int i = 0; i < rows; i++)
	{
		int table[256];
		memset(table, -1, sizeof(int) * 256);
		const uchar* p = src.ptr<uchar>(i);
		uchar* pdst = dst.ptr<uchar>(i);
		for (int j = 0; j < cols * channels; j++)
		{
			if (table[*(p + j)] == -1)
				table[*(p + j)] = saturate_cast<uchar>(pow((float)*(p + j) / 255, gamma) * 255);
			*(pdst + j) = table[*(p + j)];
		}
	}
}

void apply_built_in(const Mat& src, Mat& dst, float alpha, float beta)
{
	Timer time(TIME);
	src.convertTo(dst, -1, alpha, beta);
}

//int main()
//{
//	Mat img = imread("Lena.png", 1);
//
//	Mat dst;
//	float alpha = 0.5, beta = 0.5, gamma = 0.5;
//	imshow("Input", img);
//	apply(img, dst, alpha, beta);
//	namedWindow("Output", 1);
//	imshow("Output", dst);
//	waitKey(0);
//
//	apply_fast(img, dst, alpha, beta);
//	namedWindow("Output", 1);
//	//imshow("Input", img);
//	imshow("Output", dst);
//	waitKey(0);
//
//	apply_memo(img, dst, gamma);
//	namedWindow("Output", 1);
//	//imshow("Input", img);
//	imshow("Output", dst);
//	waitKey(0);
//
//	apply(img, dst, gamma);
//	namedWindow("Output", 1);
//	//imshow("Input", img);
//	imshow("Output", dst);
//	waitKey(0);
//
//	apply_built_in(img, dst, alpha, beta);
//	namedWindow("Output", 1);
//	//imshow("Input", img);
//	imshow("Output", dst);
//	waitKey(0);
//	return 0;
//}