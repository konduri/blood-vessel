#include <iostream>
#include <stdio.h>
#include "highgui.h"
#include "cxcore.h"
#include "cv.h"
#include <math.h>
#include <cstdlib>

#define PI 3.1416

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <ardrone_autonomy/Navdata.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h> //for sonar 
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "dynamixel_msgs/JointState.h"
#include "std_msgs/Float64.h"
#include "geometry_msgs/Twist.h"

#include "data.h"

using namespace std;
using namespace cv;

cv_bridge::CvImagePtr cv_ptr,cv_ptr2;

bttnState bttn;


/////////////////////////////////////////
int slider1;
int slider2;
int slider3;
int slider4;
int slider5;
int slider6;
int slider7;
int slider8;
//Whenever the trackbar is adjusted, "reset" the reset status
void on_trackbar1( int, void* ){}
void on_trackbar2( int, void* ){}
void on_trackbar3( int, void* ){}
void on_trackbar4( int, void* ){}
void on_trackbar5( int, void* ){}
void on_trackbar6( int, void* ){}
void on_trackbar7( int, void* ){}
void on_trackbar8( int, void* ){}
////////////////////////////////////////

//sonar positions(placeholders)
float sonar_front[8][2];
float sonar_back[8][2];

//directions(placeholders)
int R_L = -1;
int F_B = 0;
int U_D = -1;
int ROT = 0;

//battery(placeholders)
int battery_uav = 100;

void uavBatteryCallback(const ardrone_autonomy::NavdataConstPtr& nav_msg)
{
	battery_uav = nav_msg->batteryPercent;
}

void imageCb(const sensor_msgs::ImageConstPtr& msg)
{ cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8); }

void imageCb2(const sensor_msgs::ImageConstPtr& msg)
{ cv_ptr2 = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8); }

//z-axis is ignored
//if it's needed in the future, just expand the array
//from 8x2 to 8x3
void sonarCallback(const sensor_msgs::PointCloud::ConstPtr& msg){
  for(int i=0; i<8; i++){
    sonar_front[i][0] = msg->points[i].x;
    sonar_front[i][1] = msg->points[i].y;
  }
  for(int j=8; j<16; j++){
    sonar_back[j-8][0] = msg->points[j].x;
    sonar_back[j-8][1] = msg->points[j].y;
  }
}

//positions of all the joints
//
float pos_shoulder;
float pos_elbow;
float pos_base;
float pos_wrist;
float pos_jaw;
void shoulderCallback(const dynamixel_msgs::JointStateConstPtr& msg){
  pos_shoulder = (msg->current_pos)*180/PI;
}
void elbowCallback(const dynamixel_msgs::JointStateConstPtr& msg){
  pos_elbow = (msg->current_pos)*180/PI;
}
void baseCallback(const dynamixel_msgs::JointStateConstPtr& msg){
  pos_base = (msg->current_pos)*180/PI;
}
void wristCallback(const dynamixel_msgs::JointStateConstPtr& msg){
  pos_wrist = (msg->current_pos)*180/PI;
}
void jawCallback(const dynamixel_msgs::JointStateConstPtr& msg){
  pos_jaw = (msg->current_pos)*180/PI;
}




int main(int argc, char** argv)
{   ros::init(argc, argv, "image_converter");
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_(nh_);
    image_transport::Subscriber image_sub_ , image_sub_2;

    image_sub_  = it_.subscribe("/ardrone/image_raw", 1, imageCb);
    image_sub_2 = it_.subscribe("/ugv_camera_node/image_raw", 1, imageCb2);
    
    ros::Subscriber uav_battery_sub = nh_.subscribe("/ardrone/navdata", 1, uavBatteryCallback);

    initSonar(sonar_front);
    initSonar(sonar_back); //initialize sonar positions to zero first
    ros::Subscriber sonar_sub = nh_.subscribe("/ugv_node/sonar", 1, sonarCallback);


    //Subscribe to the 5 joints
    ros::Subscriber arm_shoulder_sub = nh_.subscribe("shoulder/state", 1000, shoulderCallback);
    ros::Publisher arm_shoulder_pub = nh_.advertise<std_msgs::Float64>("shoulder/command", 1000);
    std_msgs::Float64  shoulder_val;
    ros::Subscriber arm_elbow_sub = nh_.subscribe("elbow/state", 1000, elbowCallback);
    ros::Publisher arm_elbow_pub = nh_.advertise<std_msgs::Float64>("elbow/command", 1000);
    std_msgs::Float64  elbow_val;
    ros::Subscriber arm_base_sub = nh_.subscribe("base/state", 1000, baseCallback);
    ros::Publisher arm_base_pub = nh_.advertise<std_msgs::Float64>("base/command", 1000);
    std_msgs::Float64  base_val;
    ros::Subscriber arm_wrist_sub = nh_.subscribe("wrist/state", 1000, wristCallback);
    ros::Publisher arm_wrist_pub = nh_.advertise<std_msgs::Float64>("wrist/command", 1000);
    std_msgs::Float64  wrist_val;
    ros::Subscriber arm_jaw_sub = nh_.subscribe("jaw/state", 1000, jawCallback);
    ros::Publisher arm_jaw_pub = nh_.advertise<std_msgs::Float64>("jaw/command", 1000);
    std_msgs::Float64  jaw_val;
    

    //base joint won't move with 30!!!!!!!!!!!!!!!!
    ros::Rate loop_rate(10);
    //ros::Rate loop_rate(30);
    
    imgMaterial img; //all the images are stored here
    
    Mat bg = img.bg;
    Mat temp0 = imread("/home/vamshi/fuerte_workspace/tryout1/icons/temp1.jpg");
    Mat temp1 = imread("/home/vamshi/fuerte_workspace/tryout1/icons/temp1.jpg");

    double slope_ratio = 0.0;//for adjusting Y relative to X

    bool uav_video_prev_state = 0;//for switching the video on the uav
    int snapshot_count = 0;
    char snapshot_title[255];
    


    /////////////////////////////////////////////////////////////////////////////////////
    Mat pink = img.arm[6];
    namedWindow("Arm", 1);
    imshow("Arm", pink);
    slider1 = 50;
    slider2 = 50;
    slider3 = 60;
    slider4 = 50;
    slider5 = 50;
    slider6 = 30;
    slider7 = 10;
    slider8 = 0;
    createTrackbar( "Shoulder", "Arm", &slider1, 130, on_trackbar1 );
    createTrackbar( "Elbow"   , "Arm", &slider2, 360, on_trackbar2 );
    createTrackbar( "Base"    , "Arm", &slider3, 120, on_trackbar3 );
    createTrackbar( "Wrist"   , "Arm", &slider4, 360, on_trackbar4 );
    createTrackbar( "Jaw"     , "Arm", &slider5,  70, on_trackbar5 );
    createTrackbar( "X"       , "Arm", &slider6,  60, on_trackbar6 );
    createTrackbar( "Y"       , "Arm", &slider7,  60, on_trackbar7 );
    createTrackbar( "Protrude", "Arm", &slider8,  60, on_trackbar8 );
    on_trackbar1( slider1, 0 );
    on_trackbar2( slider2, 0 );
    on_trackbar3( slider3, 0 );
    on_trackbar4( slider4, 0 );
    on_trackbar5( slider5, 0 );
    on_trackbar6( slider6, 0 );
    on_trackbar7( slider7, 0 );
    on_trackbar8( slider8, 0 );
    Point2f pt;//for the second arm
    ///////////////////////////////////////////////////////////////////////////////////////
    

	try
	{
		
		while(true)
		{ if(cv_ptr != NULL)
		  { temp0 = cv_ptr->image;}
		  if(cv_ptr2 != NULL)
		  { temp1 = cv_ptr2->image; }
		  bg = overlayImage(bg, img.bttn_main[bttn.main_state], Point(0,0));
		  bg = overlayImage(bg, img.bttn_full[bttn.full_state], Point(0,80));
		  bg = overlayImage(bg, img.bttn_arm[bttn.arm_state], Point(0,160));
		  bg = overlayImage(bg, img.bttn_setting[bttn.setting_state], Point(0,240));
		  bg = overlayImage(bg, img.bttn_quit[bttn.quit_state], Point(0,720));
		  bg = overlayImage(bg, img.title[bttn.window_state], Point(80,0));
		  bg = overlayImage(bg, img.battery[0], Point(1054,15));
		  bg = overlayImage(bg, img.battery[1], Point(1054-25-192,15));
		  bg = overlayImage(bg, displayBattery(battery_uav), Point(1054+75+6-5,15+10-5));
		  bg = overlayImage(bg, displayBattery(80), Point(1054+75+6-5-25-192,15+10-5));
		  Mat arm_bg_blank = img.arm[4];//this is the temporary background of the manipulator arm

		  
		  cvSetMouseCallback("Zaphod", mouseEvent , (void*)&bttn);
		  cvSetMouseCallback("Arm"   , mouseEvent2, (void*)&bttn);
		  
		  vector<int> dirVector = getDirection(R_L, F_B, U_D, ROT);



		  int L1 = 36;
		  int L2 = 36;
		  double x, y;
		  double theta1, theta2;
		  x = slider6; 
		  y = slider7-30;

		  //protruding works with p1 cm at a button press
		  
		  double p1 = 1;
		  if(bttn.pro_forward==1){
			x += p1;
			cout << "y: " << y << endl;
			y = y - p1*slope_ratio;
			cout << "p1*slope: " << p1*slope_ratio << endl;
			cout << "y: " << y << endl;
			bttn.pro_forward = 0;
			setTrackbarPos("X", "Arm", (int)x);
			setTrackbarPos("Y", "Arm", (int)y+30);
		  }
		  if(bttn.pro_backward==1){
			x -= p1;
			y = y + p1*slope_ratio;
			//cout << "p1*slope:" << p1*slope_ratio << endl;
			bttn.pro_backward = 0;
			setTrackbarPos("X", "Arm", (int)x);
			setTrackbarPos("Y", "Arm", (int)y+30);
		  }





		  theta2 = (-1)*acos((x*x+y*y-L1*L1-L2*L2)/(2*L1*L2));
		  theta1 = atan2(y,x)-atan2(L2*sin(theta2),L1+L2*cos(theta2));
		  theta2 = theta2* 180 / PI + 180;
		  theta1 = theta1* 180 / PI + 30;
		  //theta2 = theta2* 180 / PI + 180 + (180-pos_elbow-pos_shoulder);
		  //theta1 = theta1* 180 / PI + 30 + (180-pos_elbow-pos_shoulder);
		   
		  //printf("theta1:%f, theta2:%f\n",theta1,theta2);


		  //PUBLISH ARM STATUS
		  //shoulder_val.data = slider1*PI/180;
		  //elbow_val.data    = slider2*PI/180;
		  shoulder_val.data = theta1*PI/180;  
		  elbow_val.data    = theta2*PI/180;  
		  base_val.data     = slider3*PI/180;
		  wrist_val.data    = slider4*PI/180; 
		  jaw_val.data      = slider5*PI/180; 

		  //this slope ratio is for the next while loop
		  slope_ratio = tan((180-theta1-theta2+25)*PI/180);
		  //slope_ratio = tan((180-pos_elbow-pos_shoulder+25)*PI/180);
		  //cout << "angle: " << 180-pos_elbow-pos_shoulder+25 << endl;
		  //cout << "slope: " << slope_ratio << endl;
		  
		  //cout << shoulder_val.data << ", " << pos_shoulder << endl;

		  // cout << "Shoulder: " << pos_shoulder << endl;
		  // cout << "Elbow: "    << pos_elbow    << endl;
		  // cout << "Base: "     << pos_base     << endl;
		  // cout << "Wrist: "    << pos_wrist    << endl;
		  // cout << "Jaw: "      << pos_jaw      << endl;

		  //remove this when putText is not needed
		  Point pt2;


		  // Switching the camera on the UAV
		  if(bttn.uav_video!=uav_video_prev_state){
			  system( "rosservice call /ardrone/togglecam" );
			  uav_video_prev_state = bttn.uav_video;
			}      

		  //different windows
		  switch(bttn.window_state)
		  { case 0:
			resize(temp0, temp0, img.temp[0].size());
			resize(temp1, temp1, img.temp[0].size());
			bg = overlayImage(bg, img.main_tag[0], Point(620,80));
			bg = overlayImage(bg, img.main_tag[1], Point(620,440));
			bg = overlayImage(bg, temp0, Point(620,80));
			bg = overlayImage(bg, temp1, Point(620,440));
			// bg = overlayImage(bg, img.arrow[dirVector[0]], Point(620+576,80+95));
			// bg = overlayImage(bg, img.arrow[dirVector[1]], Point(620+576,80+95+50));
			// bg = overlayImage(bg, img.arrow[dirVector[2]], Point(620+576,80+95+100));
			// bg = overlayImage(bg, img.arrow[dirVector[3]], Point(620+576,80+95+150));


			bg = overlayImage(bg, img.sonar[0], Point(110,80));
			for(int i=0; i<8; i++){
			  bg = overlayImage(bg, img.sonar[1], pointOnRadar(sonar_front[i][0],sonar_front[i][1]));
			}

			bg = overlayImage(bg, img.main[0], Point(110,580));
			bg = overlayImage(bg, img.main[1], Point(365,580));


			// Auto-approach the target
			if(bttn.auto_approach==1){
			  //do something
			  bttn.auto_approach = 0;
			}

			// Take snapshot
			if(bttn.main_snapshot==1){
			  sprintf(snapshot_title, "%d.png", snapshot_count);
			  imwrite(snapshot_title,temp0);
			  snapshot_count++;
			  bttn.main_snapshot = 0;
			}
			

			break;
			
			case 1:
			resize(temp0, temp0, img.temp[1].size());
			bg = overlayImage(bg, temp0, Point(120,80));
			bg = overlayImage(bg, img.full_uav[1-bttn.full_LR], Point(520,730));
			bg = overlayImage(bg, img.full_ugv[bttn.full_LR], Point(690,730));
			break;
			
			case 2:
			resize(temp0, temp0, img.temp[0].size());
			resize(temp1, temp1, img.temp[0].size());
			bg = overlayImage(bg, img.main_tag[0], Point(620,80));
			bg = overlayImage(bg, img.main_tag[1], Point(620,440));
			bg = overlayImage(bg, temp0, Point(620,80));
			bg = overlayImage(bg, temp1, Point(620,440));
			bg = overlayImage(bg, img.arm[0], Point(110,80));

			//warpAffine(img.arm[1], img.arm[2], getRotationMatrix2D(Point(250,250), pos_shoulder, 1.0), Size(img.arm[1].cols,img.arm[1].rows));
			//warpAffine(img.arm[1], img.arm[3], getRotationMatrix2D(Point(250,250), pos_shoulder-(180-pos_elbow), 1.0), Size(img.arm[1].cols,img.arm[1].rows));
			//pt = Point(170*cos(pos_shoulder*PI/180)+250, -170*sin(pos_shoulder*PI/180)+250);
			warpAffine(img.arm[1], img.arm[2], getRotationMatrix2D(Point(250,250), theta1, 1.0), Size(img.arm[1].cols,img.arm[1].rows));
			warpAffine(img.arm[1], img.arm[3], getRotationMatrix2D(Point(250,250), theta1-(180-theta2), 1.0), Size(img.arm[1].cols,img.arm[1].rows));
			pt = Point(170*cos(theta1*PI/180)+250, -170*sin(theta1*PI/180)+250);

			arm_bg_blank = overlayImage(arm_bg_blank, img.arm[2], Point(250,250));
			arm_bg_blank = overlayImage(arm_bg_blank, img.arm[3], pt);

			bg = overlayImage(bg, arm_bg_blank(Rect(430,240,480,480)), Point(110,80));
			bg = overlayImage(bg, img.arm[5], Point(425,95));
			bg = overlayImage(bg, img.arm[7], Point(110,580));
			bg = overlayImage(bg, img.arm[8], Point(365,580));
			bg = overlayImage(bg, img.arm[9], Point(110,688));
			bg = overlayImage(bg, img.arm[10], Point(365,688));

			if(bttn.arm_setPosition==1){
			  setTrackbarPos("X"    , "Arm", 30);
			  setTrackbarPos("Y"    , "Arm", 10);
			  setTrackbarPos("Base" , "Arm", 60);
			  setTrackbarPos("Wrist", "Arm", 50);
			  setTrackbarPos("Jaw"  , "Arm", 50);
			  bttn.arm_setPosition = 0;
			}
			if(bttn.arm_setPosition==2){
			  setTrackbarPos("X", "Arm", 32);
			  setTrackbarPos("Y", "Arm", 49);
			  bttn.arm_setPosition = 0;
			}
			if(bttn.arm_setPosition==3){
			  setTrackbarPos("X"    , "Arm", 30);
			  setTrackbarPos("Y"    , "Arm", 10);
			  setTrackbarPos("Base" , "Arm", 60);
			  setTrackbarPos("Wrist", "Arm", 50);
			  setTrackbarPos("Jaw"  , "Arm", 50);
			  bttn.arm_setPosition = 0;
			}
			if(bttn.arm_setPosition==4){
			  setTrackbarPos("Jaw"  , "Arm", 20);
			  bttn.arm_setPosition = 0;
			}
			if(bttn.arm_setPosition==5){
			  setTrackbarPos("Jaw"  , "Arm", 50);
			  bttn.arm_setPosition = 0;
			}

			///////Temporary text/////////////////
			char text_shoulder[255];
			char text_base[255];
			char text_elbow[255];
			char text_wrist[255];
			char text_jaw[255];
			sprintf(text_shoulder, "Shoulder: %d", (int)theta1);
			sprintf(text_base, "Base: %d", (int)slider3);
			sprintf(text_elbow, "Elbow: %d", (int)theta2);
			sprintf(text_wrist, "Wrist: %d", (int)slider4);
			sprintf(text_jaw, "Jaw: %d", (int)slider5);
			putText(bg, text_shoulder, Point(175,345), 1, 1.2, Scalar(255,255,0));
			putText(bg, text_base, Point(175,360), 1, 1.2, Scalar(0,255,255));
			pt2 = Point(170*cos(theta1*PI/180)+250-80, -170*sin(theta1*PI/180)+250+110);
			putText(bg, text_elbow, pt2, 1, 1.2, Scalar(0,0,255));
			putText(bg, text_wrist, pt2+Point(1,15), 1, 1.2, Scalar(0,0,255));
			putText(bg, text_jaw, pt2+Point(1,30), 1, 1.2, Scalar(0,0,255));
			//////////////////////////////////////

			break;
			
			case 3: 

			break;
		  }

		  arm_elbow_pub.publish(elbow_val);
		  arm_shoulder_pub.publish(shoulder_val);
		  arm_base_pub.publish(base_val);
		  arm_wrist_pub.publish(wrist_val);
		  arm_jaw_pub.publish(jaw_val);


		  imshow("Zaphod", bg);
		  int c = cvWaitKey(10);
		  if(c==27||bttn.quit_GUI==1) { break; }
		  
		  bg = overlayImage(bg, img.bg_blank, Point(0,0));// refresh the entire screen
		  
		  ros::spinOnce();
		  loop_rate.sleep();
		}
	}
	catch(const cv::Exception& e)
	{
		cout << "Nothing happened. Whatsoever... Really." << endl;
	}
    
	return 0;
}
