#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

#include "Timer.h"

const int TIME = 1;
int res_x = 1920, res_y = 1080, scale = 50, originx = res_x/2, originy = res_y/2, iterations = 200;
Mat img(res_y, res_x, CV_8UC3, Scalar(0, 0, 0));

void drawMandelbrot();

void callback(int event, int x, int y, int flags, void* param)
{
	static int flag = 0;
	static int xi, yi, og_x, og_y;
	if (event == EVENT_LBUTTONDOWN)
	{
		flag = 1;
		xi = x;
		yi = y;
		og_x = originx;
		og_y = originy;
	}

	if (event == EVENT_LBUTTONUP)
	{
		flag = 0;
	}
	
	if (flag == 1)
	{
		originx = og_x;
		originy = og_y;
		originx = og_x + (x - xi);
		originy = og_y + (y - yi);
		drawMandelbrot();
 	}
	imshow("Output", img);

}

const int mx_iterations = 2000;
Vec3b color[mx_iterations][mx_iterations];

void precompute()
{
	for (int i = 100; i < mx_iterations; i++)
	{
		for (int j = 0; j < mx_iterations; j++)
		{
			double pos = (double)j / i;
			if (pos <= 0.16)
			{
				color[i][j][2] = 32 * (pos / 0.16);
				color[i][j][1] = 100 * (pos / 0.16) + 7;
				color[i][j][0] = 103 * (pos / 0.16) + 100;
			}
			else if (pos <= .42)
			{
				color[i][j][2] = 205 * ((pos - 0.16) / (0.42 - 0.16)) + 32;
				color[i][j][1] = 148 * ((pos - 0.16)/ (0.42 - 0.16)) + 107;
				color[i][j][0] = 52 * ((pos - 0.16)/ (0.42 - 0.16)) + 203;
			}
			else if (pos <= .6425)
			{
				color[i][j][2] = 18 * ((pos - 0.42) / (0.6425 - 0.42)) + 237;
				color[i][j][1] = -85 * ((pos - 0.42)/ (0.6425 - 0.42)) + 255;
				color[i][j][0] = -255 * ((pos - 0.42) / (0.6425 - 0.42)) + 255;
			}
			else if (pos <= 0.8575)
			{
				color[i][j][2] = -255 * ((pos - 0.6425) / (0.8575 - 0.6425)) + 255;
				color[i][j][1] = -168 * ((pos - 0.6425)/ (0.8575 - 0.6425)) + 170;
				color[i][j][0] = 0;
			}
		}
	}
}

class ParallelMandelbrot : public ParallelLoopBody
{
public:
	virtual void operator()(const Range& range) const CV_OVERRIDE
	{
		
		for (int r = range.start; r < range.end; r++)
		{
			int i = r / res_x, j = r % res_x;
			double cx = (double)(j - originx)/scale, cy = (double)(originy - i)/scale, x = 0, y = 0;
			int n = 0;
			while (n != iterations && (x * x + y * y) <= 4)
			{
				double x_temp = x * x - y * y + cx;
				double y_temp = 2 * x * y + cy;
				x = x_temp;
				y = y_temp;
				n++;
			}

			
			img.ptr<uchar>(i)[j*3] =  color[iterations][n][0];
			img.ptr<uchar>(i)[j*3 + 1] =  color[iterations][n][1];
			img.ptr<uchar>(i)[j*3 + 2] =  color[iterations][n][2];
		}
	}
};

void drawMandelbrot()
{
	Timer timer(TIME);
	img = Mat(res_y, res_x, CV_8UC3, Scalar(0, 0, 0));
	ParallelMandelbrot obj;
	parallel_for_(Range(0, res_x * res_y), obj);
}




int main()
{
	namedWindow("Output");
	int q = 0;
	precompute();
	drawMandelbrot();
	imshow("Output", img);
	setMouseCallback("Output", callback);
	do
	{
		if (q == 'q')
		{
			originx = res_x / 2 - (scale + 0.1*scale)*((double)res_x / 2 - originx) / scale;
			originy = res_y / 2 + (scale + 0.1*scale) * (originy - (double)res_y / 2) / scale;
			scale += 0.1*scale;
			drawMandelbrot();
			imshow("Output", img);
		}
		else if (q == 'e')
		{
			originx = res_x / 2 - (scale - 0.1*scale) * ((double)res_x / 2 - originx) / scale;
			originy = res_y / 2 + (scale - 0.1*scale) * (originy - (double)res_y / 2) / scale;
			scale -= 0.1*scale;
			drawMandelbrot();
			imshow("Output", img);
		}
		else if (q == 'n')
		{
			iterations -= 15;
			drawMandelbrot();
			imshow("Output", img);
		}
		else if (q == 'm')
		{
			iterations += 15;
			drawMandelbrot();
			imshow("Output", img);
		}
		cout << "Current Values: \nScale: " << scale << "\nIterations: " << iterations << "\nOrigin: (" << originx << ", " << originy << ")\n";
		q = waitKey(0);
	} while (q != 13);
	return 0;
}
