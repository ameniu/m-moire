

from robodk import robolink    # RoboDK API
from robodk import robomath    # Robot toolbox
RDK = robolink.Robolink()


from robodk import *      # RoboDK API
from robolink import *    # Robot toolbox


tool  = RDK.Item('OnRobot VGC10 Vacuum Gripper')
tool  = RDK.Item('OnRobot Eyes Camera')

robot = RDK.Item('Doosan Robotics M1013', ITEM_TYPE_ROBOT)
home = RDK.Item('home', ITEM_TYPE_TARGET)
pick = RDK.Item('pick', ITEM_TYPE_TARGET)
place = RDK.Item('place', ITEM_TYPE_TARGET)

robot.RunCode('Postmainprog.py')

