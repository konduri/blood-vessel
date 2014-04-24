#include <iostream>
#include "highgui.h"
#include <stdio.h>
#include "cv.h"
#include "cxcore.h"
#include "math.h"

#include "data.h"

using namespace std;
using namespace cv;


Mat overlayImage(const Mat &background, const Mat &foreground, Point2i location){
  Mat output;
  background.copyTo(output);
  //1. PNG with transparent background
  if(foreground.channels()==4)
  {
  for(int y = std::max(location.y , 0); y < background.rows; ++y)
  {
    int fY = y - location.y;
    if(fY >= foreground.rows)
      break;
    for(int x = std::max(location.x, 0); x < background.cols; ++x)
    {
      int fX = x - location.x;
      if(fX >= foreground.cols)
        break;
      double opacity =
        ((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3])/ 255.;
      for(int c = 0; opacity > 0 && c < output.channels(); ++c)
      {
        unsigned char foregroundPx =
          foreground.data[fY * foreground.step + fX * foreground.channels() + c];
        unsigned char backgroundPx =
          background.data[y * background.step + x * background.channels() + c];
        output.data[y*output.step + output.channels()*x + c] =
          backgroundPx * (1.-opacity) + foregroundPx * opacity;
      }
    }
  }
  }
  //2. JPG 
  else
  { 
    foreground.copyTo(background(Rect(location.x, location.y, foreground.cols,  
    foreground.rows)));
    output = background;
  }
  return output;
}

void mouseEvent(int event, int x, int y, int flags, void *param){
  bttnState tempState = *(bttnState*)param;
//0. Display cursor position (for debugging)
  if(x>0 && x<1280 && y>0 && y<800)
    { //cout << "x: " << x << "," << "y: " << y << endl;
    }
//1. Main 
  if(event == EVENT_LBUTTONDOWN && x>0 && x<80 && y>0 && y<80)
    { tempState.main_state = 2;    tempState.full_state = 0; 
      tempState.setting_state = 0; tempState.arm_state = 0;
      tempState.window_state = 0;
    }
  else if(tempState.main_state != 2 && x>0 && x<80 && y>0 && y<80)
    { tempState.main_state = 1; }
  else if(tempState.main_state != 2 && !(x>0 && x<80 && y>0 && y<80))
    { tempState.main_state = 0; }
  if(event == EVENT_LBUTTONDOWN && x>620 && x<1246 && y>80 && y<404)
    { tempState.uav_video = !(tempState.uav_video);
      cout << tempState.uav_video << endl;
    }
//2. Full Screen
  if(event == EVENT_LBUTTONDOWN && x>0 && x<80 && y>80 && y<160)
    { tempState.main_state = 0;    tempState.full_state = 2; 
      tempState.setting_state = 0; tempState.arm_state = 0;
      tempState.window_state = 1;
    }
  else if(tempState.full_state != 2 && x>0 && x<80 && y>80 && y<160)
    { tempState.full_state = 1; }
  else if(tempState.full_state != 2 && !(x>0 && x<80 && y>80 && y<160))
    { tempState.full_state = 0; }
//3. Manipulator Arm   
  if(event == EVENT_LBUTTONDOWN && x>0 && x<80 && y>160 && y<240)
    { tempState.main_state = 0;    tempState.full_state = 0; 
      tempState.setting_state = 0; tempState.arm_state = 2;
      tempState.window_state = 2;
    }
  else if(tempState.arm_state != 2 && x>0 && x<80 && y>160 && y<240)
    { tempState.arm_state = 1; }
  else if(tempState.arm_state != 2 && !(x>0 && x<80 && y>160 && y<240))
    { tempState.arm_state = 0; }

//4. Setting    
  if(event == EVENT_LBUTTONDOWN && x>0 && x<80 && y>240 && y<320)
    { tempState.main_state = 0;    tempState.full_state = 0; 
      tempState.setting_state = 2; tempState.arm_state = 0;
      tempState.window_state = 3;
    }
  else if(tempState.setting_state != 2 && x>0 && x<80 && y>240 && y<320)
    { tempState.setting_state = 1; }
  else if(tempState.setting_state != 2 && !(x>0 && x<80 && y>240 && y<320))
    { tempState.setting_state = 0; }
//5. Quit 
  if(event == EVENT_LBUTTONDOWN && x>0 && x<80 && y>720 && y<800)
    { tempState.quit_GUI = 1; cout << "clicked on quit" << endl; }
  else if(event != EVENT_LBUTTONDOWN && x>0 && x<80 && y>720 && y<800)
    { tempState.quit_state = 1; }
  else
    { tempState.quit_state = 0;}
//Switch between UAV/UGV full screen 
  if(tempState.window_state==1)
    { if(event == EVENT_LBUTTONDOWN && x>520 && x<670 && y>730 && y<780)
        { tempState.full_LR = 0; cout << "switched to full_uav" << endl; }
      if(event == EVENT_LBUTTONDOWN && x>690 && x<840 && y>730 && y<780)
        { tempState.full_LR = 1; cout << "swtiched to full_ugv" << endl; }
    }
//Reset arm position   
  if(tempState.window_state==2)
    { if(event == EVENT_LBUTTONDOWN && x>425 && x<575 && y>95 && y<145)
        { tempState.arm_setPosition = 1; cout << "reset arm" << endl; }
      if(event == EVENT_LBUTTONDOWN && x>110 && x<335 && y>580 && y<660)
        { tempState.arm_setPosition = 2; cout << "survey position" << endl; }
      if(event == EVENT_LBUTTONDOWN && x>365 && x<590 && y>580 && y<660)
        { tempState.arm_setPosition = 3; cout << "grasp position" << endl; }
      if(event == EVENT_LBUTTONDOWN && x>110 && x<335 && y>688 && y<768)
        { tempState.arm_setPosition = 4; cout << "gripper open" << endl; }
      if(event == EVENT_LBUTTONDOWN && x>365 && x<590 && y>688 && y<768)
        { tempState.arm_setPosition = 5; cout << "gripper close" << endl; }
    }
//Buttons on the main page  
  if(tempState.window_state==0)
    { if(event == EVENT_LBUTTONDOWN && x>110 && x<335 && y>580 && y<660)
        { tempState.auto_approach = 1; cout << "auto approach" << endl; }
      if(event == EVENT_LBUTTONDOWN && x>365 && x<590 && y>580 && y<660)
        { tempState.main_snapshot = 1; cout << "take snapshot" << endl; }
    }
    
  *(bttnState*)param = tempState;
}

void mouseEvent2(int event, int x, int y, int flags, void *param){
  bttnState tempState = *(bttnState*)param;
  if(event == EVENT_LBUTTONDOWN && x>0 && x<190 && y>0 && y<40){
    tempState.pro_backward = 1;
  }
  if(event == EVENT_LBUTTONDOWN && x>190 && x<385 && y>0 && y<40){
    tempState.pro_forward = 1;
  }
  *(bttnState*)param = tempState;
}

vector<int> getDirection(int R_L, int F_B, int U_D, int ROT){
  vector<int> allDir(4,0);
  allDir[0] = (R_L>=0)? 1 : 3;
  allDir[1] = (F_B>=0)? 0 : 2;
  allDir[2] = (U_D>=0)? 4 : 5;
  allDir[3] = (ROT>=0)? 6 : 7;
  return allDir;
}

Mat displayBattery(int battery_reading){
  Mat battery_bar(Size(battery_reading,40),CV_8UC3);
  if(battery_reading>75) 
    { battery_bar = Scalar(76,207,81); }
  if(battery_reading<75 && battery_reading>50) 
    { battery_bar = Scalar(43,234,255); }
  if(battery_reading<50 && battery_reading>25) 
    { battery_bar = Scalar(43,146,255); }
  if(battery_reading<25 && battery_reading>0) 
    { battery_bar = Scalar(47,43,255); }
  return battery_bar;
}

void initSonar(float input[8][2]){
  for(int i=0; i<8; i++){
    for(int j=0; j<2; j++){
      input[i][j]=0;
    }
  }
}

Point pointOnRadar(float y, float x){
  //remember to remove the dots that go beyond the range
  int radius = 6; //3.42m for now
  int radar_x = int(-240*x/radius)+350-11;
  int radar_y = int(-240*y/radius)+320-11;

  if(radar_x<110||radar_x>590||radar_y<80||radar_y>560){
    radar_x = 1;
    radar_y = 1;
  }

  return Point(radar_x,radar_y);
}
