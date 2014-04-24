#!/usr/bin/env python

# Import the ROS libraries, and load the manifest file which through <depend package=... /> will give us access to the project dependencies
import roslib; roslib.load_manifest('ardrone_tutorials')
import rospy
import math

# Load the DroneController class, which handles interactions with the drone, and the DroneVideoDisplay class, which handles video display
from pioneer_controller import BasicPioneerController
from drone_video_display import DroneVideoDisplay

# Import the joystick message
from sensor_msgs.msg import Joy

# Finally the GUI libraries
from PySide import QtCore, QtGui

# define the default mapping between joystick axes and their corresponding directions
AxisLinear        = 1
AxisAngular       = 0

# define the default scaling to apply to the axis inputs. useful where an axis is inverted
ScaleLinear       = 1.0
ScaleAngular      = 1.0

# handles the reception of joystick packets
def ReceiveJoystickMessage(data):
	controller.SetCommand(data.axes[AxisLinear]/ScaleLinear, data.axes[AxisAngular]/ScaleAngular)


# Setup the application
if __name__=='__main__':
	import sys
	# Firstly we setup a ros node, so that we can communicate with the other packages
	rospy.init_node('pioneer_joystick_controller')

	AxisLinear        = int ( rospy.get_param("~AxisLinear",AxisLinear) )
	AxisAngular       = int ( rospy.get_param("~AxisAngular",AxisAngular) )

	ScaleLinear       = float ( rospy.get_param("~ScaleLinear",ScaleLinear) )
	ScaleAngular      = float ( rospy.get_param("~ScaleAngular",ScaleAngular) )

	# Now we construct our Qt Application and associated controllers and windows
	app = QtGui.QApplication(sys.argv)
	#display = DroneVideoDisplay()
	controller = BasicPioneerController()

	# subscribe to the /joy topic and handle messages of type Joy with the function ReceiveJoystickMessage
	subJoystick = rospy.Subscriber('/joy', Joy, ReceiveJoystickMessage)
	
	# executes the QT application
	#display.show()
	status = app.exec_()

	# and only progresses to here once the application has been shutdown
	#rospy.signal_shutdown('Great Flying!')
	sys.exit(status)
