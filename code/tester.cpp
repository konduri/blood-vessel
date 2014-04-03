#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <ardrone_autonomy/Navdata.h>
#include <sensor_msgs/Joy.h>
#include <dynamic_reconfigure/server.h>
#include <ardrone_tagfollow/dynamicConfig.h>
#include <math.h>

#define errorIdxX 	0
#define errorIdxY 	1
#define errorIdxYaw	2
#define errorIdxZ	3

#define movementPID	0

#define constMovePosX	1
#define constMoveNegX	2
#define constMovePosY	3
#define constMoveNegY	4
#define constMovePosW	5
#define constMoveNegW	6
#define constMovePosZ	7
#define constMoveNegZ	8

#define PI 3.141592653589793

//#define SET_HEIGHT 1700 		- Moved to the class as a variable in order to implement landing functionality

using namespace std;

class Tag_follow_class
{
	private:
		ros::NodeHandle nh;		// so that we dont make a copy while passing this as reference, and work on this one directly
		ros::Subscriber joy_sub;		// joystick button and potentiometer
		ros::Subscriber nav_sub;		// AR Drone navigation data
		ros::Subscriber joy_drone_sub;	// Drone velocities generated from joystick controls (manual control modde velocity)
		ros::Subscriber ugv_yaw_sub;	// TODO: Here we need to subscribe to the rosaria_pose message
		ros::Publisher vel_pub;			// this will be publishing the command velocity to the drone
		ros::Publisher err_pub;			// publish error in PID controller - TODO : Block this in final code
 		//ros::Publisher vel_directions;// for later purposes, when i want to show markers on opencv using this array; will define messagetype later
		
		geometry_msgs::Twist 	twist_auto, twist_manual, twist_error;		// the message we use to send command to the drone
		
		double_t tnow, t_last,tdiff, t_prev_tag;     // current time as per roscore initialization, convert to double using .toSec() fxn
		//int tdiff;
		//ros::Duration t_diff;          // duration track of t_diff.  we can sleep and run the loop at the rate we want
		
		bool joy_automode;		// the flag we use to keep track of the state we are in ,{Manual or Autonomous}
		bool tag_in_sight;		// keeps track of whether the tag is visible in the 'navdata' message coming in from AR Drone
		bool LANDING_FLAG;		// TODO: Verify why this fails. Checked when landing command is to be given
		
		float goal_x, goal_y, goal_yaw,SET_HEIGHT;		// the goal positon for our PID, in our case the center of image (320,160,0)
		
		void joy_callback(const sensor_msgs::Joy::ConstPtr& msg);
		void nav_callback(const ardrone_autonomy::NavdataConstPtr& nav_msg);
		void joy_drone_callback(const geometry_msgs::Twist::ConstPtr& msg);
		void Rosaria_pose_callback(const geometry_msgs::Twist::ConstPtr& msg);	// TODO: call back for the rosaria pose

		void publishtwist();
		
		double vel_x_kp, vel_x_kd, vel_y_kp, vel_y_kd, yaw_kp, yaw_kd, thrust_kp, thrust_kd; // parameters for the pid controls
		void pidcontroller();

		//void landing();
		float  error[4],error_prev[4];
 
		//I am defining a global variable to keep track of the angle of the UGV 
		float ugv_yaw;// *************************************************

		uint8_t isConstVel;		// Used for debugging - TODO: Remove for final code
  
	public:
		Tag_follow_class();		//only the constructor is public
		void configCallback(ardrone_tagfollow::dynamicConfig &config, uint32_t level);
};

Tag_follow_class::Tag_follow_class():joy_automode(true)
{
	ROS_INFO("Tag_follow start");
    cout  << "We are in the constructor of the class \n";
    this->SET_HEIGHT  = 1700;
	this->t_last =ros::Time::now().toSec(); //initialized the time_last to the time at which we start the system.
	//we assume that the tag is in the view of the bottom camera since the begining so this should not be a problem

	this->joy_sub       = this->nh.subscribe("joy", 1, &Tag_follow_class::joy_callback, this);
	this->nav_sub       = this->nh.subscribe("/ardrone/navdata", 1, &Tag_follow_class::nav_callback, this);
	this->joy_drone_sub = this->nh.subscribe("/cmd_vel_drone", 1, &Tag_follow_class::joy_drone_callback, this);
	this->ugv_yaw_sub   = this->nh.subscribe("/ugv_node/euler_yaw",1,&Tag_follow_class::Rosaria_pose_callback,this);		// TODO: change to 'pose'
	
	this->vel_pub       = this->nh.advertise<geometry_msgs::Twist>("/cmd_vel_tester", 1);
	this->err_pub       = this->nh.advertise<geometry_msgs::Twist>("/pid_tune_err", 1);
	
	/******************   For Testing Purpose   **********************/
	isConstVel = movementPID;
	/*****************************************************************/

	//this->t_last = ros::Time::now();
	t_prev_tag = -10;
	tag_in_sight = false;

	LANDING_FLAG = false;
	
	this->twist_auto.linear.x  = this->twist_auto.linear.y  = this->twist_auto.linear.z  = 0;
	this->twist_auto.angular.x = this->twist_auto.angular.y = this->twist_auto.angular.z = 0; 
	
	this->error[0]      = this->error[1]      = this->error[2]      = this->error[3]      = 0;
	this->error_prev[0] = this->error_prev[1] = this->error_prev[2] = this->error_prev[3] = 0;  
	
	publishtwist();
}
 
void Tag_follow_class::joy_callback(const sensor_msgs::Joy::ConstPtr& msg) 
{
	if (msg->buttons[4] == 1) 		// TODO : Give a nice name to this button index
	{
		if (this->joy_automode) 
		{
			this->joy_automode = false;
			//cout << "We have now entered manual mode" << endl;
		}
	}

	// write here the part for the landing funtion FLAG
	if(msg->buttons[5] == 1)  		// RB button on the joystick. TODO : Verify the button functionality
	{ 
		if(!this-> LANDING_FLAG)
		{
			this->LANDING_FLAG = true;
			cout << "landing button pressed \n";
		}
	}

	if(msg->buttons[0] == 1)  		// RB button on the joystick. TODO : Verify the button functionality
	{ 
		if(this-> LANDING_FLAG)
		{
			this->SET_HEIGHT = 1700;
			this->LANDING_FLAG = false;
			cout << "takeoff button pressed \n";
		}
	}
}
 
void Tag_follow_class::nav_callback(const ardrone_autonomy::NavdataConstPtr& nav_msg)
{
	// we are checking all the errors and wether we are detecting tag in this call back
	this->error[errorIdxZ]  = this -> SET_HEIGHT - nav_msg -> altd;	// Distance in millimeters
	cout << " erroer of height value is " << this->error[errorIdxZ] << "and the set_height is " << this->SET_HEIGHT << "\n";
	this->twist_error.linear.z  = this -> error[errorIdxZ];  
	
	if(!nav_msg->tags_width.empty()) 
	{
		//******************************************	
		this->ugv_yaw = this->thrust_kd;
		//******************************************
		this->tag_in_sight       = true;
		this->joy_automode       = true;
		this->error[errorIdxX]   = 500 - float_t(nav_msg->tags_yc[0]);		// y in image is along x-direction of motion
		this->error[errorIdxY]   = 500 - float_t(nav_msg->tags_xc[0]);		// x in image is along y-direction of motion
		//	this->error[errorIdxYaw] = 270 - float_t(nav_msg->tags_orientation[0]);//*******************changed to
		this->error[errorIdxYaw] = 0;// (180*(PI+this->ugv_yaw)/PI) - (nav_msg->rotZ+180);  //****************************************** i am also assuming the data from navdata is set in range of 0 to 360
		// cout <<"value of data are "<<180*(PI+this->ugv_yaw)/PI<< " and "<<nav_msg->rotZ+180 << "and error is"<<this->error[errorIdxYaw] <<"\n";

		this->twist_error.linear.x  = this ->error[errorIdxX];
		this->twist_error.linear.y  = this ->error[errorIdxY];
		this->twist_error.angular.x = 0;
		this->twist_error.angular.y = 0;
		this->twist_error.angular.z = this ->error[errorIdxYaw];

		this->tnow			= nav_msg->header.stamp.toSec() ;
		pidcontroller();
		this->t_last		= this->tnow; 
		this->t_prev_tag	= this->tnow;
		memcpy(this->error_prev, this->error,sizeof(this->error));
	}
	
	else
	{
		this->tag_in_sight = false;
		this->error[errorIdxX] = this->error[errorIdxY] = this->error[errorIdxYaw] = 0;	
		this->twist_error.linear.x  = this ->error[errorIdxX];
		this->twist_error.linear.y  = this ->error[errorIdxY];
		this->twist_error.linear.z  = this ->error[errorIdxZ];
		this->twist_error.angular.x = 0;//this ->error[errorIdxY]
		this->twist_error.angular.y = 0;//this ->error[errorIdxX]
		this->twist_error.angular.z = this ->error[errorIdxYaw];

		this->tnow     = nav_msg->header.stamp.toSec() ;
		pidcontroller();	
		this->t_last = this->tnow;
		memcpy(this->error_prev, this->error, sizeof(this->error));
	}
}

void Tag_follow_class::joy_drone_callback(const geometry_msgs::Twist::ConstPtr& msg) 
{
    this->twist_manual.linear.x  = msg->linear.x ; 
    this->twist_manual.linear.y  = msg->linear.y ; 
    this->twist_manual.linear.z  = msg->linear.z ;//this->twist_auto.linear.z ;
    this->twist_manual.angular.x = msg->angular.x;
    this->twist_manual.angular.y = msg->angular.y;
    this->twist_manual.angular.z = msg->angular.z;
    this->twist_auto.angular.z   = msg->angular.z;
}

// *************************************************
// call back funtion for the ar_pose data
void Tag_follow_class::Rosaria_pose_callback(const geometry_msgs::Twist::ConstPtr& msg)
{
	this->ugv_yaw = msg->angular.z;
	//    cout << "the yaw angle that we observe is  " << this->ugv_yaw << "\n" ;
}
//*****************************************************	


void Tag_follow_class::publishtwist()
{
	if(joy_automode)
	{
		vel_pub.publish(twist_auto);
		err_pub.publish(twist_error);
		//cout<< "auto mode " << endl;
	}
	else if(!joy_automode)
	{
		vel_pub.publish(twist_manual);
		err_pub.publish(twist_error);
		//cout<<"manual mode" <<  endl;
	}
}
 
void Tag_follow_class::pidcontroller()
{
	if (this->tag_in_sight == true || (this->tag_in_sight == false && this->tnow - this->t_prev_tag > 1.0)) 
	{
		//cout << "we are in the pid loop" << endl;
		this->twist_auto.linear.x = this->vel_x_kp*this->error[errorIdxX]+(this->vel_x_kd)*(this->error[errorIdxX]-this->error_prev[errorIdxX])   / (this->tnow - this->t_last);// +/b;
		this->twist_auto.linear.y = this->vel_y_kp  	* this->error[errorIdxY]	+ (this-> vel_y_kd)	* ( this->error[errorIdxY]   - this-> error_prev[errorIdxY])   / (this->tnow - this->t_last);
		//this->twist_auto.angular.z  	= this->yaw_kp	   	* this->error[errorIdxYaw]	+ (this->yaw_kd)	* ( this->error[errorIdxYaw] - this-> error_prev[errorIdxYaw]) / (this->tnow - this->t_last);

		//cout << "the error value of yaw is " << error[errorIdxYaw]  << " and the difference of error is " <<this->error[errorIdxYaw] - this-> error_prev[errorIdxYaw]  << "and the velocity is " << this->twist_auto.angular.z <<"\n" ;
		if (this->LANDING_FLAG)  
		{
			if (this->SET_HEIGHT > 200) 
			{
				this->SET_HEIGHT = this->SET_HEIGHT - 2;
			}
			// cout << this->SET_HEIGHT;
			//cout << "Decreasing the height\n";
		}

		if ((this->error[errorIdxZ] < 200) && (LANDING_FLAG ==true) && this->tag_in_sight == true)  
		{
			cout << "Emergency land loop \n";
			//	sensor_msgs::Joy landing_message;
			//		ros::Publisher joy_land_msg;
			//	landing_message.buttons[1] = 1;
			//joy_land_msg  = this->nh.advertise<sensor_msgs::Joy>("/joy", 1); 
			// 	joy_land_msg.publish(landing_message);
		}


		this->twist_auto.linear.z = this->thrust_kp	* this->error[errorIdxZ] + /*(this->thrust_kd)*/0 * ( this->error[errorIdxZ] - this->error_prev[errorIdxZ]) / (this->tnow - this->t_last);
		// cout << "pid value of the angular z" << this->twist_auto.linear.z;
		this->twist_auto.angular.x 	= 0;
		this->twist_auto.angular.y 	= 0;
	
	}
	
	//	cout << "vX : " << this->twist_auto.linear.x << " vY : " << this->twist_auto.linear.y << " wZ : " << this->twist_auto.angular.z << " vZ : " << this->twist_auto.linear.z << endl;
	//	cout  << " vxkp: "<<this->vel_x_kp<< " vxkd: "<<this->vel_x_kd<< " vykp: "<<this->vel_y_kp<< " vykd: "<<this->vel_y_kd << endl;
	publishtwist();
}


void Tag_follow_class::configCallback(ardrone_tagfollow::dynamicConfig &config, uint32_t level)
{
	//for the dynamic reconfigure part of the code
	this->vel_x_kp  = config.vel_x_kp;
	this->vel_x_kd  = config.vel_x_kd;
	this->vel_y_kp  = config.vel_x_kp;
	this->vel_y_kd  = config.vel_x_kd;
	this->yaw_kp    = config.yaw_kp;
	this->yaw_kd    = config.yaw_kd;
	this->thrust_kp = config.thrust_kp;
	this->thrust_kd = config.thrust_kd;
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "tagfollow");                                                 //we are initializing the node
    cout << "We have begun the main program" << "\n" ;  
	Tag_follow_class  tagfollower;                                                    //the class the we have defined above

	dynamic_reconfigure::Server<ardrone_tagfollow::dynamicConfig> dr_srv;				// belongs to dynamic reconfigure {changes kp kd online}
	dynamic_reconfigure::Server<ardrone_tagfollow::dynamicConfig>::CallbackType dr_cb;	// you can ignore this for the program above
	dr_cb = boost::bind(&Tag_follow_class::configCallback, &tagfollower, _1, _2);		// dynamic recongif 
	dr_srv.setCallback(dr_cb);															// dynamic reconfig

	ros::spin(); 
	return 0;
}
