#!/usr/bin/env python

# Import the ROS libraries, and load the manifest file which through <depend package=... /> will give us access to the project dependencies
import roslib; roslib.load_manifest('ardrone_tutorials')
import rospy

# Import the messages we're interested in sending and receiving
from geometry_msgs.msg import Twist  	 # for sending commands to the drone

# Import exponential stuff
from math import expm1
from math import copysign

COMMAND_PERIOD = 20 #ms

exponential_control = True
exponential_max     = expm1(4)

class BasicPioneerController(object):
	def __init__(self):
		# Holds the current drone status
		self.status = -1
		
		# Allow the controller to publish to the /cmd_vel topic and thus control the drone
		self.pubCommand = rospy.Publisher('/cmd_vel_pioneer',Twist)

		# Setup regular publishing of control packets
		self.command = Twist()
		self.commandTimer = rospy.Timer(rospy.Duration(COMMAND_PERIOD/1000.0),self.SendCommand)

	def SetCommand(self,linear=0,angular=0):
		# TODO: Implement expoenetial control and / or thresholded control strategy in here:
		if exponential_control == True:
			vel_linear  = expm1(abs(4*linear) ) / exponential_max
			vel_angular = expm1(abs(4*angular)) / exponential_max
			self.command.linear.x  = copysign(vel_linear , linear )
			self.command.angular.z = -1 * copysign(vel_angular, angular)
		else:
			self.command.linear.x  = linear
			self.command.angular.z = angular


	def SendCommand(self,event):
		self.pubCommand.publish(self.command)
