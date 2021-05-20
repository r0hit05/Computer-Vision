/*
The code is still buggy. I wrote this quite a while ago and will have to dig a lot
fix the bug.  
The program takes a black image with starting positions of cue ball and the stick 
direction. The ball then proceeds to move and collide with the walls.
1.png is working
2.png is buggy
3.png is working
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

#define CV_THRESH_BINARY 0
#define CV_BGR2GRAY 6
#define CV_RETR_TREE RETR_TREE

Mat img = imread("../images/3.png",1);
int rows = img.rows, cols = img.cols;
Mat imggray(rows,cols,CV_8UC1,Scalar(0));
Mat imgline(rows,cols,CV_8UC1,Scalar(0));
Mat newimg(rows,cols,CV_8UC3,Scalar(0,0,0));
int maxx=0,maxy=0,minx=cols,miny=rows,flag=0;

float distance(Point a, Point b)
{
  return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y));
}

void linefind()
{
  for(int i=0;i<rows;i++)
    for(int j=0;j<cols;j++)
    {
      if(img.at<Vec3b>(i,j)[2]==255 && img.at<Vec3b>(i,j)[1]==0 && img.at<Vec3b>(i,j)[0]==0 )
          {
            if(i>maxy)
              {
                maxy=i;
                maxx=j;
              }
            if(i<miny)
              {
                miny=i;
                minx=j;
              }
          }
      else
        {
          img.at<Vec3b>(i,j)[0]=0;
          img.at<Vec3b>(i,j)[1]=0;
          img.at<Vec3b>(i,j)[2]=0;
        }
    }
}

struct myClass
{
  bool operator()(Point pt1,Point pt2){return(pt1.x<pt2.x);}
}obj;

int main()
{
  cvtColor(img,imggray,CV_BGR2GRAY);
  threshold(imggray,imggray,127,255,CV_THRESH_BINARY);
  Canny(imggray,imggray,50,200,3);
  linefind();
  vector<vector<Point>>contours;
  vector<Vec4i>hierarchy;
  findContours(imggray,contours,hierarchy,CV_RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0,0));
  int n = contours.size();
  vector <Moments> mu(n);
  for(int i=0;i<n;i++)
    mu[i]=moments(contours[i],true);
  vector <Point> mc(n);
  for(int i=0;i<n;i++)
  {
    mc[i].x = mu[i].m10/mu[i].m00;
    mc[i].y = mu[i].m01/mu[i].m00;
  }

  vector <float> rad(n);                            //Finding Radius
  vector <Point2f> cen(n);
  for(int i=0;i<n;i++)
    minEnclosingCircle(contours[i],cen[i],rad[i]);
  sort(rad.begin(), rad.end());
  float radius = rad[n-1];

  vector <Point> mcc;
  Point curr,prev;
  for(int i=0;i<n;i++)                          //Eliminating Duplicates
  {
    if(i==0)
      {
        curr = mc[i];
        mcc.push_back(curr);
      }
    else
    {
      prev = curr;
      curr = mc[i];
      float d = distance(prev,curr);
      if(d>radius+1)
        mcc.push_back(curr);
    }
  }
  vector <int> v(mcc.size());
  for(int i=0;i<mcc.size();i++)
    v[i]=0;
  //finding the cue ball
  //cout<<radius<<endl;
  //cout<<maxx<<" "<<maxy<<" "<<minx<< " "<< miny<<endl;
  // circle(newimg,Point(maxx,maxy),5,Scalar(0,0,255),-1);
  // circle(newimg,Point(minx,miny),5,Scalar(0,0,255),-1);
  int cue=0;
  for(int i=0;i<mcc.size();i++)
  {
    float d1 = distance(Point(maxx,maxy),mcc[i]);
    float d2 = distance(Point(minx,miny),mcc[i]);
    //cout<<d1<<" "<<d2<<endl;
    if(d1<=radius+5 && d1>= radius - 1 )
      {
        cue=i;
        flag=2;                       //reps the side closer to the ball
        break;
      }
    else if(d2<=radius+5 && d2>= radius - 1)
      {
        cue=i;
        flag=1;
        break;
      }
  }
  //cout<<cue<<endl;
  //cout<<mcc.size()<<endl;
  for(int i=0; i< mcc.size();i++)
  {
    if(i == cue)
      {
        //cout<<i<<" "<<mcc[i]<<endl;
        circle(newimg,mcc[i],radius,Scalar(0,0,255),-1);
        v[i]=1;
      }
    else
    {
      //cout<<i<<" "<<mcc[i]<<endl;
      circle(newimg,mcc[i],radius,Scalar(255,255,255),-1);
    }
  }
  //cout<<v[0]<<" "<<v[1]<<" "<<v[2]<<" "<<v[3]<<endl;
  //finding line of motion
  float slope = (float)(maxy-miny)/(maxx-minx);
  float theta = atan(slope);
  int coll = 0,col1 = 5,slopeflag=0;
  cout<<slope<<" "<<flag<<" "<<cos(theta)<<" "<<sin(theta)<< endl;
  // circle(newimg,Point(0+radius,0+radius),radius,Scalar(0,0,255),-1);
  // circle(newimg,Point(0+radius,rows-radius),radius,Scalar(0,0,255),-1);
  // circle(newimg,Point(cols-radius,0+radius),radius,Scalar(0,0,255),-1);
  // circle(newimg,Point(cols-radius,rows-radius),radius,Scalar(0,0,255),-1);
  //moving along the line
  // cout<<"Enter no. of collisions"<<endl;
  // cin>>col1;
  int x2 = mcc[cue].x, y2 = mcc[cue].y;
  for(int i=0;i<2000 && coll<col1;i++)
  {
    for(int i=0;i<rows;i++)
    {
      for(int j=0;j<cols;j++)
      {
        newimg.at<Vec3b>(i,j)[0]=0;
        newimg.at<Vec3b>(i,j)[1]=0;
        newimg.at<Vec3b>(i,j)[2]=0;
      }
    }
    float x,y;
    if(slope<0 && slopeflag==0)
    {
      if(flag==1)
      {
        x = x2 + i*cos(theta);
        y = y2 + i*sin(theta);
      }
      else
      {
        x = x2 - i*cos(theta);
        y = y2 - i*sin(theta);
      }
    }
    else if(slope>0 && slopeflag==0)
    {
      if(flag==1)
      {
        x = x2 - i*cos(theta);
        y = y2 - i*sin(theta);
      }
      else
      {
        x = x2 + i*cos(theta);
        y = y2 + i*sin(theta);
      }
    }
    else
    {
      if(flag==1)
        y = y2 + i;
      else
        y = y2 - i;
    }
    Point p(x,y);
    //cout<<p<<endl;
    // line(newimg,)
    mcc[cue]=p;
    for(int i=0;i<mcc.size();i++)
    {
      if(i!=cue)
      circle(newimg,mcc[i],radius,Scalar(255,255,255),-1);
      else
      circle(newimg,mcc[i],radius,Scalar(0,0,255),-1);
    }
    int c=0;
    float d;
    int min = sqrt(rows*rows + cols*cols);
    if(x<=radius+1 || x>=cols-radius-1)
    {
      i=1;
      x2=x;
      y2=y;
      theta = -theta;
      circle(newimg,p,radius,Scalar(255,255,255),-1);
      if(flag==1)
      flag=2;
      else
      flag=1;
    }
    else if(y<=radius+1 || y>=rows-radius-1)
    {
      i=1;
      x2=x;
      y2=y;
      theta = CV_PI - theta;
      circle(newimg,p,radius,Scalar(255,255,255),-1);
      if(flag==1)
      flag=2;
      else
      flag=1;
    }
    while(mcc.size() - c)
    {
      if(v[c]==0)
      {
        d = distance(p,mcc[c]);
        if(d<=2*radius+1 && d>=2*radius-1)
        {
          coll++;
          circle(newimg,p,radius,Scalar(255,255,255),-1);
          v[cue]=0;
          // mcc[cue]=p;
          if(mcc[c].y<mcc[cue].y && mcc[c].x>mcc[cue].x)
            {
              if(flag==1)
                flag=2;
              else
                flag=1;
            }
          cue = c;
          v[cue]=1;
          x2 = mcc[cue].x;
          y2 = mcc[cue].y;
          if(x2==p.x)
            slopeflag=1;
          slope=(float)(y2-p.y)/(x2-p.x);
          //if(slope)
          theta = atan(slope);
          //cout<<slope<<" "<<flag<<" "<<theta<<" "<<cos(theta)<<" "<<sin(theta)<<endl;
          // if(slope*slopetemp<0)
          // {
          //   if(flag==1)
          //   flag=2;
          //   else
          //   flag=1;
          // }
          //cout<<v[0]<<" "<<v[1]<<" "<<v[2]<<" "<<v[3]<<" "<<cue<<endl<<endl;
          cout<<"Collision "<<coll<<endl;
          i=2;
          break;
        }
      }
      c++;
    }
    namedWindow("Output",0);
    imshow("Output",newimg);
    waitKey(3);
  }
  for(int i=0;i<mcc.size();i++)
  {
    cout<<mcc[i]<<" ";
  }
  cout<<endl;
  cout<<v[0]<<" "<<v[1]<<" "<<v[2]<<" "<<v[3]<<" "<<cue<<endl;

  namedWindow("Output",0);
  imshow("Output",newimg);
  waitKey(0);
  return 0;
}
