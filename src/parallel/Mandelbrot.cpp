#include <iostream>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace cv;

#include "Timer.h"

const int TIME = 1;
void draw_coordinate(Mat& dst, double scalex = 1, double scaley = 1, double originx = -1, double originy = -1)
{
	Timer time(TIME);
	if (originx == -1)
		originx = scalex / 2;
	if (originy == -1)
		originy = scaley / 2;

	int rows = dst.rows, cols = dst.cols;
	double xmul = scalex / cols, ymul = scaley / rows;

	int originx_screen = originx / xmul, originy_screen = originy / ymul;
	for (int i = 0; i < rows; i++)
	{
		dst.at<uchar>(i, originx_screen) = 255;
	}
	for (int i = 0; i < cols; i++)
	{
		dst.at<uchar>(originy_screen, i) = 255;
	}

}


void Mandelbrot_fast(Mat& dst, int n, double scalex = 1, double scaley = 1, double originx = -1, double originy = -1)
{
	Timer time(TIME);
	if (originx == -1)
		originx = scalex / 2;
	if (originy == -1)
		originy = scaley / 2;

	int rows = dst.rows, cols = dst.cols;
	double xmul = scalex / (double)cols, ymul = scaley / (double)rows;



	for (int r = 0; r < rows * cols; r++)
	{
		int i = r / cols, j = r % cols;

		double y0 = ((double)i * ymul) - originy;
		double x0 = ((double)j * xmul) - originx;

		std::complex<double> z(0, 0);
		std::complex<double> c(x0, y0);
		int iteration = 0;
		while (z.real()*z.real() + z.imag()*z.imag() < 4 && iteration != n)
		{
			z = z * z + c;
			iteration++;
		}

		//std::cout << r << " " << std::endl;
		if (iteration == n)
			dst.ptr<uchar>(i)[j] = 0;
		else
			dst.ptr<uchar>(i)[j] = (sqrt((double)iteration / n)) * 255;
	}

	std::cout << "Naive Implementation: ";
	
}

void Mandelbrot_fast_threads(Mat& dst, int n, double scalex = 1, double scaley = 1, double originx = -1, double originy = -1)
{
	Timer time(TIME);
	if (originx == -1)
		originx = scalex / 2;
	if (originy == -1)
		originy = scaley / 2;

	int rows = dst.rows, cols = dst.cols;
	double xmul = scalex / (double)cols, ymul = scaley / (double)rows;

	const int n_threads = 56;
	int r_max =rows*cols, r_begin = 0, r_interval = r_max / n_threads, r_end = r_interval;

	std::thread t[n_threads];

	for (int i = 0; i < n_threads; i++)
	{
		t[i] = std::thread([&](int r_b, int r_e) {
			for (int r = r_b; r <= r_e; r++)
			{
				int i = r / cols, j = r % cols;

				double y0 = ((double)i * ymul) - originy;
				double x0 = ((double)j * xmul) - originx;

				std::complex<double> z(0, 0);
				std::complex<double> c(x0, y0);
				int iteration = 0;
				while (z.real() * z.real() + z.imag() * z.imag() < 4 && iteration != n)
				{
					z = z * z + c;
					iteration++;
				}

				if (iteration == n)
					dst.ptr<uchar>(i)[j] = 0;
				else
					dst.ptr<uchar>(i)[j] = (sqrt((double)iteration / n)) * 255;
			}
			}, r_begin, r_end);
		
		r_begin = r_end + 1;
		r_end += r_interval + 1;
		r_end = min(r_max - 1, r_end);
	}

	for (int i = 0; i < n_threads; i++)
	{
		t[i].join();
	}

	std::cout << "Threaded Implementation: ";
}

void Mandelbrot_parallel_for(Mat& dst, int n, double scalex = 1, double scaley = 1, double originx = -1, double originy = -1)
{
	Timer time(TIME);
	if (originx == -1)
		originx = scalex / 2;
	if (originy == -1)
		originy = scaley / 2;

	int rows = dst.rows, cols = dst.cols;
	double xmul = scalex / (double)cols, ymul = scaley / (double)rows;

	parallel_for_(Range(0, rows * cols), [&](const Range& range) {
		for (int r = range.start; r < range.end; r++)
		{
			int i = r / cols, j = r % cols;

			double y0 = ((double)i * ymul) - originy;
			double x0 = ((double)j * xmul) - originx;

			std::complex<double> z(0, 0);
			std::complex<double> c(x0, y0);
			int iteration = 0;
			while (z.real() * z.real() + z.imag() * z.imag() < 4 && iteration != n)
			{
				z = z * z + c;
				iteration++;
			}

			if (iteration == n)
				dst.ptr<uchar>(i)[j] = 0;
			else
				dst.ptr<uchar>(i)[j] = (sqrt((double)iteration / n)) * 255;
		}
		});

	std::cout << "Parallel_for_ Implementation: ";

}


int main()
{
	float scalex = 3, scaley = 2, originx = 2, originy = 1;
	int iter = 500;
	Mat img(1080, 1920, CV_8U, Scalar(0));

	Mandelbrot_fast(img, iter, scalex, scaley, originx, originy);
	imshow("Output", img);
	waitKey(0);
	destroyWindow("Output");

	img = Mat(1080, 1920, CV_8U, Scalar(0));
	Mandelbrot_fast_threads(img, iter, scalex, scaley, originx, originy);
	imshow("Output", img);
	waitKey(0);
	destroyWindow("Output");
	
	img = Mat(1080, 1920, CV_8U, Scalar(0));
	Mandelbrot_parallel_for(img, iter, scalex, scaley, originx, originy);
	imshow("Output", img);
	waitKey(0);
	return 0;
}
