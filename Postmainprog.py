
import sys
import os
sys.path.append(os.path.abspath(r"""C:/RoboDK/Nouveau dossier/RoboDK1/Posts/""")) # temporarily add path to POSTS folder

from Doosan_Robotics import *

try:
  from robodk.robomath import PosePP as p
except:
  # This will be removed in future versions of RoboDK
  from robodk import PosePP as p


print('Total instructions: 6')
r = RobotPost(r"""Doosan_Robotics""",r"""Doosan Robotics M1013""",6, axes_type=['R','R','R','R','R','R'], native_name=r"""Doosan Robotics M1013""", ip_com=r"""127.0.0.1""", api_port=20500, prog_ptr=2152440530064, robot_ptr=2152325758832, lines_x_prog=5000)

r.ProgStart(r"""mainprog""")
r.RunMessage(r"""Program generated by RoboDK v5.6.0 for Doosan Robotics M1013 on 13/06/2023 11:29:59""",True)
r.RunMessage(r"""Using nominal kinematics.""",True)
r.RunCode(r"""Prog3""", True)
r.RunCode(r"""Prog4""", True)
r.ProgFinish(r"""mainprog""")
r.ProgStart(r"""Prog3""")
r.setFrame(p(0,0,0,0,0,0),-1,r"""home""")
r.setTool(p(0,0,117.401,0,0,0),-1,r"""OnRobot VGC10 Vacuum Gripper""")
r.MoveJ(None,[6.04545,-8.97254,-78.4319,0.145923,-93.9702,6.05381],None)
r.MoveJ(None,[-6.44069,-28.2831,-100.919,-0.196255,-52.171,-6.31847],None)
r.RunMessage(r"""Attacher à OnRobot VGC10 Vacuum Gripper""",True)
r.ProgFinish(r"""Prog3""")
r.ProgStart(r"""Prog4""")
r.setFrame(p(-728.524,-297.705,6.95441,0,0,0),4,r"""Frame 4""")
r.setTool(p(0,0,117.401,0,0,0),-1,r"""OnRobot VGC10 Vacuum Gripper""")
r.MoveJ(None,[24.2462,-28.1761,-101.468,0.723937,-51.6184,23.7904],None)
r.RunMessage(r"""Détachez de OnRobot VGC10 Vacuum Gripper""",True)
r.MoveJ(None,[6.04545,-8.97254,-78.4319,0.145923,-93.9702,6.05381],None)
r.ProgFinish(r"""Prog4""")
r.ProgSave(r"""C:/Users/user/Documents/RoboDK/Programs/doosan/""",r"""mainprog""",False,r"""C:/RoboDK/Nouveau dossier/RoboDK/Other/VSCodium/VSCodium.exe""")
