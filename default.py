#!/usr/bin/env python3

# ! /usr/bin/env morseexec
import logging
import math

import sys
from morse.builder import *
from math import pi
import xmlrpc.client


morse_logger = logging.getLogger("pymorse")
morse_logger.setLevel(logging.ERROR)


with xmlrpc.client.ServerProxy("http://localhost:8000/") as proxy:
    val, xe, ye, xp, yp, ori_1, ori_2 = proxy.world_sel()
    print(val, xe, ye, xp, yp, ori_1, ori_2)


laser_pose = [-0.3, 0, 0.8]
laser_range = 4.0 #10.0
laser_resolution = 2.0 #3.0
laser_layers = 1
laser_scan_window = 60.0
laser_freq = 1.0 #hz
laser_arc_visible = False

evader = ATRV()
evader.set_color(0.6, 0.0, 0.0)
evader.translate(x=xe, y=ye, z=0.9)
evader.rotate(z=ori_1)
evader.properties(Object=True, Graspable=False, Label="evader")


motion_r = Waypoint()
motion_r.properties(ObstacleAvoidance=False)
motion_r.add_interface('socket')
evader.append(motion_r)

motion_r_vw = MotionVW()
evader.append(motion_r_vw)
motion_r_vw.add_interface('socket')

pose_evader = Pose()
evader.append(pose_evader)
pose_evader.add_interface('socket')

sick_r = SickLDMRS()
sick_r.properties(Visible_arc=laser_arc_visible)
sick_r.properties(resolution=laser_resolution)
sick_r.properties(scan_window=laser_scan_window)
# sick_r.properties(scan_window=45)
sick_r.properties(laser_range=laser_range)
sick_r.properties(layers=laser_layers)
# sick_r.frequency(laser_freq)
sick_r.translate(laser_pose[0], laser_pose[1], laser_pose[2])
evader.append(sick_r)
sick_r.add_interface('socket')

# PURSUER ------------------------------------------------------------------------------
pursuer = ATRV()
pursuer.set_color(0.0, 0.6, 0.0)
pursuer.translate(x=xp, y=yp, z=0.9)
pursuer.rotate(z=ori_2)
# vader.rotate(z = -pi/2)

motion_v = Waypoint()
motion_v.properties(ObstacleAvoidance=False)
motion_v.add_interface('socket')
pursuer.append(motion_v)

v_cam = SemanticCamera()
v_cam.add_interface('socket')
v_cam.rotate(x=0.0, y=0.0, z=pi)
v_cam.translate(x=0., y=0., z=1.0)
pursuer.append(v_cam)

pose_pursuer = Pose()
pursuer.append(pose_pursuer)
pose_pursuer.add_interface('socket')

sick_v = SickLDMRS()
sick_v.properties(Visible_arc=True)
sick_v.properties(resolution=laser_resolution)
sick_v.properties(scan_window=laser_scan_window)
# sick_v.properties(scan_window=30)
sick_v.properties(laser_range=laser_range)
sick_v.properties(layers=laser_layers)
# sick_v.frequency(laser_freq)
sick_v.translate(laser_pose[0], laser_pose[1], laser_pose[2])
pursuer.append(sick_v)
sick_v.add_interface('socket')

motion_v_vw = MotionVW()
pursuer.append(motion_v_vw)
motion_v_vw.add_interface('socket')

vel_v = Velocity()
pursuer.append(vel_v)
vel_v.add_interface('socket')


# ------------------------------------------------------------------------------------------
# Environment
env = Environment(val)

env.set_camera_location([0.0, 0.8, 50])
env.set_camera_rotation([0.0, 0.0, math.pi/2])
env.select_display_camera(v_cam)

# env.set_time_scale(accelerate_by = 5.0) #10x
# env.use_internal_syncer()
