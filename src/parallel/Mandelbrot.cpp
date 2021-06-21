#pragma GCC optimize("Ofast")
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/simd_intrinsics.hpp>


using namespace cv;
using namespace std;

#include "Timer.h"

const int TIME = 1;
const int res_x = 1920, res_y = 1080;
long long int scale = 350, originx = res_x / 2, originy = res_y / 2, iterations = 600;
Mat img(res_y, res_x, CV_8UC3, Scalar(0, 0, 0));

const int mx_iterations = 2000;
Vec3b color[mx_iterations][mx_iterations];

void precompute()
{
  for (int i = 1; i < mx_iterations; i++)
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
        color[i][j][1] = 148 * ((pos - 0.16) / (0.42 - 0.16)) + 107;
        color[i][j][0] = 52 * ((pos - 0.16) / (0.42 - 0.16)) + 203;
      }
      else if (pos <= .6425)
      {
        color[i][j][2] = 18 * ((pos - 0.42) / (0.6425 - 0.42)) + 237;
        color[i][j][1] = -85 * ((pos - 0.42) / (0.6425 - 0.42)) + 255;
        color[i][j][0] = -255 * ((pos - 0.42) / (0.6425 - 0.42)) + 255;
      }
      else if (pos <= 0.8575)
      {
        color[i][j][2] = -255 * ((pos - 0.6425) / (0.8575 - 0.6425)) + 255;
        color[i][j][1] = -168 * ((pos - 0.6425) / (0.8575 - 0.6425)) + 170;
        color[i][j][0] = 0;
      }
    }
  }
}

class ParallelMandelbrot : public ParallelLoopBody
{
public:
  virtual void operator()(const Range &range) const CV_OVERRIDE
  {

    for (int r = range.start; r < range.end; r++)
    {
      int i = r / res_x, j = r % res_x;
      double cx = (double)(j - originx) / scale, cy = (double)(originy - i) / scale, x = 0, y = 0;
      int n = 0;
      while (n != iterations && (x * x + y * y) <= 4)
      {
        double x_temp = x * x - y * y + cx;
        double y_temp = 2 * x * y + cy;
        x = x_temp;
        y = y_temp;
        n++;
      }

      img.ptr<uchar>(i)[j * 3] = color[iterations][n][0];
      img.ptr<uchar>(i)[j * 3 + 1] = color[iterations][n][1];
      img.ptr<uchar>(i)[j * 3 + 2] = color[iterations][n][2];
    }
  }
};

void drawMandelbrot()
{
  Timer timer(TIME);
  ParallelMandelbrot obj;
  parallel_for_(Range(0, res_x * res_y), obj);
}





void drawMandelbrotsimdhighres()
{
  Timer timer(TIME);
  parallel_for_(Range(0, res_x * res_y), [&](const Range &range)
                {

#if (CV_SIMD256 == 1)
                  int step = 256 / 64;
                  v_float64 ox_wide, oy_wide, scale_wide, x_wide, y_wide, n_wide, x2, y2, xwide_temp, ywide_temp, four_wide, iter_wide, ones_wide, two_wide;
                  ox_wide = vx_setall_f64(originx);
                  oy_wide = vx_setall_f64(originy);
                  scale_wide = vx_setall_f64((double)1 / scale);
                  four_wide = vx_setall_f64(4);
                  iter_wide = vx_setall_f64(iterations);
                  ones_wide = vx_setall_f64(1);
                  two_wide = vx_setall_f64(2);
                  for (long long r = range.start; r < range.end; r += step)
                  {

                    // make register for i
                    // |r / res_x|(r + 1)/res_X|(r + 2)/res_x|(r + 3)/res_x|
                    v_float64 i_wide(r / res_x, (r + 1) / res_x, (r + 2) / res_x, (r + 3) / res_x);
                    v_float64 j_wide(r % res_x, (r + 1) % res_x, (r + 2) % res_x, (r + 3) % res_x);

                    // helper registers

                    // cx, cy, x, y registers. x2, y2 registers for comparison
                    v_float64 cx_wide = (j_wide - ox_wide) * scale_wide, cy_wide = (oy_wide - i_wide) * scale_wide;

                    // since the cpu can handle a maximum limit of registers at a time without sending them to the cache, it is advisable to precompute some of the regsiters to avoid extra time due to cache miss.
                    // v_float64 cx_wide((double)((r % res_x) - originx)/scale, (double)(((r + 1) % res_x) - originx)/scale, (double)(((r + 2) % res_x) - originx)/scale, (double)(((r + 3) % res_x) - originx)/scale);
                    // v_float64 cy_wide((double)(originy - r / res_x)/scale, (double)(originy - (r + 1) / res_x)/scale, (double)(originy - (r + 2) / res_x)/scale, (double)(originy - (r + 3) / res_x)/scale);

                    x_wide = vx_setzero_f64();
                    y_wide = vx_setzero_f64();
                    n_wide = vx_setzero_f64();


                    x2 = x_wide * x_wide;
                    y2 = y_wide * y_wide;

                    double ptr[4] = {1};
                    while (ptr[0] || ptr[1] || ptr[2] || ptr[3])
                    {
                      xwide_temp = x2 - y2 + cx_wide;
                      ywide_temp = two_wide * x_wide * y_wide + cy_wide;

                      x_wide = xwide_temp;
                      y_wide = ywide_temp;

                      x2 = x_wide * x_wide;
                      y2 = y_wide * y_wide;

                      v_float64 cmp = ((x2 + y2) <= four_wide) & (n_wide != iter_wide);
                      v_store(ptr, cmp);

                      v_float64 add_one(ptr[0] != 0, ptr[1] != 0, ptr[2] != 0, ptr[3] != 0);
                      n_wide += add_one;
                    }

                    double n[4];
                    v_store(n, n_wide);
                    for (int i = 0; i < 4; i++)
                    {
                      if (r + i < range.end)
                      {
                        img.ptr<uchar>((r + i) / res_x)[((r + i) % res_x) * 3] = color[iterations][(int)n[i]][0];
                        img.ptr<uchar>((r + i) / res_x)[((r + i) % res_x) * 3 + 1] = color[iterations][(int)n[i]][1];
                        img.ptr<uchar>((r + i) / res_x)[((r + i) % res_x) * 3 + 2] = color[iterations][(int)n[i]][2];
                      }
                    }
                  }
#else
        for (int r = range.start; r < range.end; r++)
        {
          int i = r / res_x, j = r % res_x;

          double cx = (double)(j - originx) / scale, cy = (double)(originy - i) / scale, x = 0, y = 0;
          int n = 0;
          while (n != iterations && (x * x + y * y) <= 4)
          {
            double x_temp = x * x - y * y + cx;
            double y_temp = 2 * x * y + cy;
            x = x_temp;
            y = y_temp;
            n++;
          }

          img.ptr<uchar>(i)[j * 3] = color[iterations][n][0];
          img.ptr<uchar>(i)[j * 3 + 1] = color[iterations][n][1];
          img.ptr<uchar>(i)[j * 3 + 2] = color[iterations][n][2];
          // nr_normal[r] = n;
        }
#endif
                });
}

void drawMandelbrotsimdlowres()
{
  Timer timer(TIME);
  parallel_for_(Range(0, res_x * res_y), [&](const Range &range)
                {

#if (CV_SIMD256 == 1)
                  int step = 256 / 32;
                  v_float32 x_wide, y_wide, n_wide, x2, y2;
                  v_float32 xwide_temp, ywide_temp, four_wide, iter_wide, ones_wide, two_wide;
                  four_wide = vx_setall_f32(4);
                  iter_wide = vx_setall_f32(iterations);
                  ones_wide = vx_setall_f32(1);
                  two_wide = vx_setall_f32(2);
                  for (int r = range.start; r < range.end; r += step)
                  {
                    v_float32 cx_wide((float)(r % res_x - originx) / scale, (float)((r + 1) % res_x - originx) / scale, (float)((r + 2) % res_x - originx) / scale, (float)((r + 3) % res_x - originx) / scale, (float)((r + 4) % res_x - originx) / scale, (float)((r + 5) % res_x - originx) / scale, (float)((r + 6) % res_x - originx) / scale, (float)((r + 7) % res_x - originx) / scale);
                    v_float32 cy_wide((float)(originy - r / res_x) / scale, (float)(originy - (r + 1) / res_x) / scale, (float)(originy - (r + 2) / res_x) / scale, (float)(originy - (r + 3) / res_x) / scale, (float)(originy - (r + 4) / res_x) / scale, (float)(originy - (r + 5) / res_x) / scale, (float)(originy - (r + 6) / res_x) / scale, (float)(originy - (r + 7) / res_x) / scale);

                    x_wide = vx_setzero_f32();
                    y_wide = vx_setzero_f32();
                    n_wide = vx_setzero_f32();

                    x2 = x_wide * x_wide;
                    y2 = y_wide * y_wide;

                    float ptr[8] = {1};
                    while (ptr[0] || ptr[1] || ptr[2] || ptr[3] || ptr[4] || ptr[5] || ptr[6] || ptr[7])
                    {
                      xwide_temp = x_wide * x_wide - y_wide * y_wide + cx_wide;
                      ywide_temp = two_wide * x_wide * y_wide + cy_wide;

                      x_wide = xwide_temp;
                      y_wide = ywide_temp;

                      x2 = x_wide * x_wide;
                      y2 = y_wide * y_wide;

                      v_float32 cmp = ((x2 + y2) <= four_wide) & (n_wide != iter_wide);
                      v_store(ptr, cmp);

                      v_float32 add_one(ptr[0] != 0, ptr[1] != 0, ptr[2] != 0, ptr[3] != 0, ptr[4] != 0, ptr[5] != 0, ptr[6] != 0, ptr[7] != 0);
                      n_wide += add_one;
                    }

                    float n[8];
                    v_store(n, n_wide);
                    for (int i = 0; i < 8; i++)
                    {
                      if (r + i < range.end)
                      {
                        img.ptr<uchar>((r + i) / res_x)[((r + i) % res_x) * 3] = color[iterations][(int)n[i]][0];
                        img.ptr<uchar>((r + i) / res_x)[((r + i) % res_x) * 3 + 1] = color[iterations][(int)n[i]][1];
                        img.ptr<uchar>((r + i) / res_x)[((r + i) % res_x) * 3 + 2] = color[iterations][(int)n[i]][2];
                      }
                    }
                  }
#else
        for (int r = range.start; r < range.end; r++)
        {
          int i = r / res_x, j = r % res_x;

          double cx = (double)(j - originx) / scale, cy = (double)(originy - i) / scale, x = 0, y = 0;
          int n = 0;
          while (n != iterations && (x * x + y * y) <= 4)
          {
            double x_temp = x * x - y * y + cx;
            double y_temp = 2 * x * y + cy;
            x = x_temp;
            y = y_temp;
            n++;
          }

          img.ptr<uchar>(i)[j * 3] = color[iterations][n][0];
          img.ptr<uchar>(i)[j * 3 + 1] = color[iterations][n][1];
          img.ptr<uchar>(i)[j * 3 + 2] = color[iterations][n][2];
        }
#endif
                });
}

void drawMandelbrotslow()
{
  Timer timer(TIME);
  for (int r = 0; r < res_x * res_y; r++)
  {
    int i = r / res_x, j = r % res_x;
    double cx = (double)(j - originx) / scale, cy = (double)(originy - i) / scale, x = 0, y = 0;
    int n = 0;

    while (n != iterations && (x * x + y * y) <= 4)
    {
      double x_temp = x * x - y * y + cx;
      double y_temp = 2 * x * y + cy;
      x = x_temp;
      y = y_temp;
      n++;
    }

    img.ptr<uchar>(i)[j * 3] = color[iterations][n][0];
    img.ptr<uchar>(i)[j * 3 + 1] = color[iterations][n][1];
    img.ptr<uchar>(i)[j * 3 + 2] = color[iterations][n][2];
  }
}

auto func = drawMandelbrotslow;
int flagfunc = 1;

void callback(int event, int x, int y, int flags, void *param)
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
    func();
    imshow("Output", img);
  }
}

int main()
{
  namedWindow("Output");
  int q = 0;
  char a[50] = "Naive";
  precompute();
  func();
  imshow("Output", img);
  setMouseCallback("Output", callback);

  do
  {
    if (q == 'q')
    {
      originx = res_x / 2 - (scale + 0.1 * scale) * ((double)res_x / 2 - originx) / scale;
      originy = res_y / 2 + (scale + 0.1 * scale) * (originy - (double)res_y / 2) / scale;
      scale += 0.1 * scale;
      func();
      imshow("Output", img);
    }
    else if (q == 'e')
    {
      originx = res_x / 2 - (scale - 0.1 * scale) * ((double)res_x / 2 - originx) / scale;
      originy = res_y / 2 + (scale - 0.1 * scale) * (originy - (double)res_y / 2) / scale;
      scale -= 0.1 * scale;
      func();
      imshow("Output", img);
    }
    else if (q == 'n')
    {
      if (iterations > 15)
        iterations -= 15;
      func();
      imshow("Output", img);
    }
    else if (q == 'm')
    {
      iterations += 15;
      func();
      imshow("Output", img);
    }
    else if (q == 't')
    {
      if (flagfunc == 0)
      {
        func = drawMandelbrotslow;
        strcpy(a, "Naive");
        flagfunc = 1;
      }
      else if (flagfunc == 1)
      {
        func = drawMandelbrot;
        strcpy(a, "Parallel");
        flagfunc = 2;
      }
      else if (flagfunc == 2)
      {
        func = drawMandelbrotsimdhighres;
        strcpy(a, "SIMD + Parallel (high precision)");
        flagfunc = 3;
      }
      else if (flagfunc == 3)
      {
        func = drawMandelbrotsimdlowres;
        strcpy(a, "SIMD + Parallel (low precision)");
        flagfunc = 0;
      }
    }
    printf("Current Values: \nScale: %lld\nIterations: %lld\nMethod: %s\n", scale, iterations, a);
    q = waitKey(0);
  } while (q != 13);
  return 0;
}
