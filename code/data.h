#include <iostream>
#include "highgui.h"
#include <stdio.h>
#include "cv.h"
#include "cxcore.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;  

Mat overlayImage(const Mat&, const Mat&, Point2i);
//Mat overlayImageROI(const Mat&, const Mat&, Point2i);

void mouseEvent(int, int, int, int, void*);
void mouseEvent2(int, int, int, int, void*);
std::vector<int> getDirection(int, int, int, int);
Mat displayBattery(int);
void initSonar(float[8][2]);
Point pointOnRadar(float, float);


struct imgMaterial
{ Mat bttn_main[3];
  Mat bttn_full[3];
  Mat bttn_arm[3];
  Mat bttn_setting[3];
  Mat bttn_quit[2];
  Mat title[4];
  Mat bg;
  Mat bg_blank;
  Mat temp[2];
  Mat full_uav[2];
  Mat full_ugv[2];
  Mat main_tag[2];
  Mat arrow[8];
  Mat battery[2];
  Mat arm[10];  // subject to change
  Mat sonar[10]; // subject to change
  string currLocation;
  
 
  imgMaterial()
  { currLocation = "/home/vamshi/fuerte_workspace/tryout1/";
    bg = imread(currLocation+"icons/bg1-1.jpg");
    bg_blank = imread(currLocation+"icons/bg1-1.jpg");
    bttn_main[0] = imread(currLocation+"icons/main_normal.jpg");
    bttn_main[1] = imread(currLocation+"icons/main_light.jpg");
    bttn_main[2] = imread(currLocation+"icons/main_using.jpg");
    bttn_full[0] = imread(currLocation+"icons/full_normal.jpg");
    bttn_full[1] = imread(currLocation+"icons/full_light.jpg");
    bttn_full[2] = imread(currLocation+"icons/full_using.jpg");
    bttn_arm[0] = imread(currLocation+"icons/arm_normal.jpg");
    bttn_arm[1] = imread(currLocation+"icons/arm_light.jpg");
    bttn_arm[2] = imread(currLocation+"icons/arm_using.jpg");
    bttn_setting[0] = imread(currLocation+"icons/setting_normal.jpg");
    bttn_setting[1] = imread(currLocation+"icons/setting_light.jpg");
    bttn_setting[2] = imread(currLocation+"icons/setting_using.jpg");
    bttn_quit[0] = imread(currLocation+"icons/quit_normal.jpg");
    bttn_quit[1] = imread(currLocation+"icons/quit_light.jpg");
    title[0] = imread(currLocation+"icons/title_main.jpg");
    title[1] = imread(currLocation+"icons/title_full.jpg");
    title[2] = imread(currLocation+"icons/title_arm.jpg");
    title[3] = imread(currLocation+"icons/title_setting.jpg");
    temp[0] = imread(currLocation+"icons/temp1.jpg");
    temp[1] = imread(currLocation+"icons/temp1-2.jpg");
    full_uav[0] = imread(currLocation+"icons/full_uav.jpg");
    full_uav[1] = imread(currLocation+"icons/full_uav_light.jpg");
    full_ugv[0] = imread(currLocation+"icons/full_ugv.jpg");
    full_ugv[1] = imread(currLocation+"icons/full_ugv_light.jpg");
    main_tag[0] = imread(currLocation+"icons/tag_uav.jpg");
    main_tag[1] = imread(currLocation+"icons/tag_ugv.jpg");
    arrow[0] = imread(currLocation+"icons/arrow_move_forward.jpg");
    arrow[1] = imread(currLocation+"icons/arrow_move_right.jpg");
    arrow[2] = imread(currLocation+"icons/arrow_move_backward.jpg");
    arrow[3] = imread(currLocation+"icons/arrow_move_left.jpg");
    arrow[4] = imread(currLocation+"icons/arrow_hover_up.jpg");
    arrow[5] = imread(currLocation+"icons/arrow_hover_down.jpg");
    arrow[6] = imread(currLocation+"icons/arrow_turn_cw.jpg");
    arrow[7] = imread(currLocation+"icons/arrow_turn_ccw.jpg");
    battery[0] = imread(currLocation+"icons/battery_uav.jpg");
    arm[0] = imread(currLocation+"icons/arm_bg.jpg");
    arm[1] = imread(currLocation+"icons/arm_link.png",-1);
    //arm[2] & arm[3] will be rotated arm[1]
    arm[4] = imread(currLocation+"icons/arm_bg_empty.png",-1);
    arm[5] = imread(currLocation+"icons/arm_reset.jpg");
    arm[6] = imread(currLocation+"icons/button_temp.png",-1);

    sonar[0] = imread(currLocation+"icons/sonar_bg.jpg");
    sonar[1] = imread(currLocation+"icons/sonar_dot.png",-1);
  }
  
  ~imgMaterial() {}

};

struct bttnState
{ int main_state;
  int full_state;
  int arm_state;
  bool arm_reset;
  int setting_state;
  int quit_state;
  bool quit_GUI;
  int window_state;
  int full_LR;
  //these two are for protruding the arm
  bool pro_forward;
  bool pro_backward;
  
  bttnState()
  { main_state = 2;
    full_state = 0;
    arm_state = 0;
    arm_reset = 0;
    setting_state = 0;
    quit_state = 0;
    quit_GUI = 0;
    window_state = 0;
    full_LR = 0;
    pro_forward = 0;
    pro_backward = 0;
  }
  ~bttnState() {}
};
