#!/usr/bin/env python3
import _thread
import copy
import logging
import os
import sys
import threading
import time
import traceback
from collections import deque
import random
import atexit, signal

import psutil
from filterpy.kalman import KalmanFilter
from pymorse import Morse
import math
import numpy as np
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical
from utils import one_hot_decode, predict_sequence, setup_my_logger
import pandas as pd


path = './results/'
spawn_file = 'spawnFiles/spawn_wall_worlds.csv'
exp_type = 'TAG'
log_file_pkl = path + 'results_' + exp_type + '.pkl'
log_file_csv = path + 'results_' + exp_type + '.csv'

logger = setup_my_logger('run_pursuer', 1)
columns = ['world', 'evader_spawn', 'pursuer_spawn', 'evader_ori', 'pursuer_ori', 'dist_profile','steps_count']

# taking out variables
distance_profile = []
ctr = 0
is_evader_visible = None
spawn_df = None
log_data = None


def handle_exit(a, b):
    global log_data, distance_profile, ctr, save_df
    logger.info("............................EXITED..................................................")
    logger.info("............................SAVING LOGS......................................................")
    
    temp = np.append(log_data,[distance_profile, ctr])
    a_df = pd.DataFrame(temp.reshape(1, len(temp)), columns=columns)
    
    save_df = save_df.append(a_df, ignore_index = True)    
    save_df.to_csv(log_file_csv, index = False)
    save_df.to_pickle(log_file_pkl)
    time.sleep(0.1)
    cur_pid = os.getpid()
    os.kill(cur_pid, signal.SIGKILL)
    # time.sleep(2.0)
signal.signal(signal.SIGTERM, handle_exit)


class PidEvader:
    pe_obj = None
    def run(self):
        global ctr
        try:
            while True:
                self.pe_obj = MorseEnv.getInstance()
                # if not self.pe_obj.is_done:
                if self.pe_obj.evader_pose and self.pe_obj.evader_scan:
                    # logger.debug('goal_id: {}, len: {}'.format(self.pe_obj.goal_id, len(self.pe_obj.destinations)))
                    if not self.pe_obj.evader_goal:
                        self.pe_obj.evader_goal = [self.pe_obj.destinations[self.pe_obj.goal_id]['x'],
                                                   self.pe_obj.destinations[self.pe_obj.goal_id]['y']]
                        self.pe_obj.motion_r.goto(self.pe_obj.evader_goal[0],
                                                  self.pe_obj.evader_goal[1], 0.9, 0.1, self.pe_obj.max_linear_speed)

                    if ctr % 50 == 0:
                        logger.info(' ################### Evader:{} ##############'.format(ctr))
                        logger.info(' cur:{}, goal: {}'.format(self.pe_obj.evader_pose, self.pe_obj.evader_goal))

                    if self.pe_obj.obstacle_r:
                        self.pe_obj.cleared_r = False
                        self.pe_obj.motion_r.stop()
                        sign = - np.sign(self.pe_obj.angle_obs_r)

                        self.pe_obj.motion_r_vw.set_speed(0.0, sign * self.pe_obj.evader_rotation_speed)

                        while not self.pe_obj.cleared_r:
                            if not self.pe_obj.obstacle_r:
                                self.pe_obj.cleared_r = True

                        # logger.debug()("Cleared")
                        if self.pe_obj.trap_danger_evader:
                            self.pe_obj.motion_r_vw.set_speed(self.pe_obj.trap_escape_vel, self.pe_obj.trap_angular_vel)
                            self.pe_obj.simu.sleep(self.pe_obj.trap_duration_time)
                        else:
                            self.pe_obj.motion_r_vw.set_speed(0.0, 0.0)

                        self.pe_obj.motion_r.resume()

                    if self.pe_obj.simu.evader.motion_r.get_status() == "Arrived":
                        self.pe_obj.goal_id += 1
                        # logger.debug()("Here we are at goal ID ", goal_id - 1)
                        if self.pe_obj.goal_id >= len(self.pe_obj.destinations):
                            self.pe_obj.goal_id = self.pe_obj.goal_id % len(self.pe_obj.destinations)

                        self.pe_obj.evader_goal = [self.pe_obj.destinations[self.pe_obj.goal_id]['x'],
                                                   self.pe_obj.destinations[self.pe_obj.goal_id]['y']]
                        self.pe_obj.motion_r.goto(self.pe_obj.evader_goal[0],
                                                  self.pe_obj.evader_goal[1], 0.9, 0.1, self.pe_obj.max_linear_speed)
                    ctr += 1
                time.sleep(0.01)
        except:
            logger.exception('Evader Exception in script!!!')
            traceback.print_exc()
            _thread.interrupt_main()


class MorseEnv:
    __instance = None

    pursuer_type = 2
    
    if pursuer_type == 1: #------------------------------------------camera only
        is_camera_based, use_pred, pred_type = True, False, 0        
    elif pursuer_type == 2: #------------------------------------------kalman filtering
        is_camera_based, use_pred, pred_type = False, True, 0
    else: #------------------------------------------------------------lstm model
        is_camera_based, use_pred, pred_type = False, True, 1

    n_steps_out = 20
    goal_id = 1
    evader_thread = None
    mult_val = 100.0
    is_done = False

    obstacle_r, angle_obs_r = False, None
    obstacle_v, angle_obs_v = False, None

    pursuer_pose, evader_pose = None, None
    pursuer_goal, evader_goal = None, None

    camera_evader_pose = None
    camera_evader_last_pose = None

    evader_scan, pursuer_scan = None, None
    cleared_v, cleared_r = True, True
    trap_danger_pursuer, trap_danger_evader = False, False
    des_yaw_pursuer, des_yaw_evader = None, None
    pursuer_sign, evader_sign = 1, 1
    curr_pred_x, curr_pred_y = None, None

    pursuer_rotation_speed = math.pi / 2
    evader_rotation_speed = 2 / 3 * math.pi
    max_linear_speed = 3.0
    collision_distance = 2.5
    capture_distance = 3.0
    trap_escape_vel = 5
    trap_duration_time = 0.15
    trap_angular_vel = 0  # math.pi/6

    destinations = [{'x': -1.8787, 'y': 16.013},  # spawn
                    {'x': -13.529, 'y': 16.59},
                    {'x': -15.656, 'y': 13.49},
                    {'x': -16.508, 'y': -14.72},

                    {'x': 12.508, 'y': -10.72},  # Added
                    {'x': 8.508, 'y': 10.72},
                    {'x': 12.508, 'y': 9.72},
                    {'x': -2.508, 'y': 8.72},
                    {'x': 13.508, 'y': -2.72},

                    {'x': -11.554, 'y': -20.548},
                    {'x': 11.254, 'y': -21.202},
                    {'x': 14.306, 'y': -15.251},
                    {'x': 15.447, 'y': 12.368},
                    {'x': 12.479, 'y': 16.582}]  # Path 0

    if n_steps_out == 20:
        path = './pred_models_20/'
    # else:
    #     path = './pred_models/'
    enc_x = load_model(path + 'enc_x.h5', compile=False)
    dec_x = load_model(path + 'dec_x.h5', compile=False)
    enc_y = load_model(path + 'enc_y.h5', compile=False)
    dec_y = load_model(path + 'dec_y.h5', compile=False)

    shift_val = 0.28000000000000114
    n_features_x = int(56.00000000000023) + 1
    n_features_y = int(53.000000000000114) + 1

    pose_x_deque = deque(maxlen=n_steps_out + 1)
    pose_y_deque = deque(maxlen=n_steps_out + 1)

    dt = 0.05
    pred = 5* dt #changed here
    f = KalmanFilter(dim_x=4, dim_z=4)
    f.F = np.array([[1., 0., dt, 0.],  # State transition matrix
                    [0., 1., 0., dt],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])

    # Measurement function
    f.H = np.array([[1., 0., 1., 0.],
                    [0., 1., 0., 1.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])

    f.P *= 1000.  # Current state covariance matrix

    f.R = 0.1  # Measurement noise covariance

    f.Q = np.array([[0., 0., 0., 0.],  # Process noise covariance
                    [0., 0., 0., 0.],
                    [0., 0., 0.1, 0.],
                    [0., 0., 0., 0.1]])

    lock = _thread.allocate_lock()
    

    @staticmethod
    def getInstance():
        """ Static access method. """
        if MorseEnv.__instance == None:
            MorseEnv()
        return MorseEnv.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if MorseEnv.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            MorseEnv.__instance = self

    def predictor(self):
        if self.pred_type == 1:
            if len(self.pose_x_deque) == self.n_steps_out + 1:
                pose_x_source = np.asarray(self.pose_x_deque)
                diff_x = np.diff(pose_x_source)
                pose_y_source = np.asarray(self.pose_y_deque)
                diff_y = np.diff(pose_y_source)

                # preprocessing
                pose_x_source = np.multiply(np.add(diff_x, self.shift_val), self.mult_val)
                pose_y_source = np.multiply(np.add(diff_y, self.shift_val), self.mult_val)

                pose_x_src_encoded = to_categorical([pose_x_source], num_classes=self.n_features_x)
                pose_y_src_encoded = to_categorical([pose_y_source], num_classes=self.n_features_y)

                X1_x = np.array(pose_x_src_encoded)
                X1_y = np.array(pose_y_src_encoded)

                target_x = predict_sequence(self.enc_x, self.dec_x, X1_x, self.n_steps_out, self.n_features_x)
                target_y = predict_sequence(self.enc_y, self.dec_y, X1_y, self.n_steps_out, self.n_features_y)
                # flip and reconstruct yhat
                target_x = np.flipud(np.subtract(np.divide(np.asarray(one_hot_decode(target_x)), self.mult_val),
                                                 self.shift_val))
                target_y = np.flipud(np.subtract(np.divide(np.asarray(one_hot_decode(target_y)), self.mult_val), 
                                                 self.shift_val))

                if len(self.pose_x_deque) > 0:
                    # last_pos_x, last_pos_y = pose_x_source[-1], pose_y_source[-1]
                    last_pos_x, last_pos_y = self.pose_x_deque[-1], self.pose_y_deque[-1]
    
                    target_x_new = np.hstack((last_pos_x, target_x)).cumsum()[1:]
                    target_y_new = np.hstack((last_pos_y, target_y)).cumsum()[1:]
                    
                    # print(target_x_new, target_y_new,  +"#####################################################")
                    # distance filter
                    # if not distance((target_x[-1], target_y[-1]), (pose_x_deque[-1], pose_y_deque[-1])) > 2.0:
                    self.curr_pred_x = target_x_new[-1]
                    self.curr_pred_y = target_y_new[-1]
        else:
            # if kalman filter
            new_f = copy.copy(self.f)
            self.f.predict()
            ctr = 0
            while ctr <= self.pred / self.dt:
                new_f.predict()
                ctr += 1
            self.curr_pred_x, self.curr_pred_y = new_f.x[0][0], new_f.x[1][0]

    @staticmethod
    def get_distance(a, b):
        return np.linalg.norm(np.subtract(a, b))

    def laser_data_evader(self, data):
        # logger.debug()('########  laser_data_evader ##############')
        min_val = min(data['range_list'])
        self.evader_scan = data['range_list']
        if min_val < self.collision_distance:
            self.obstacle_r = True
            point = data['point_list'][data['range_list'].index(min_val)]
            self.angle_obs_r = math.degrees(math.atan2(point[1], point[0]))
        else:
            self.obstacle_r = False

    def distance_update_evader(self, pose_r):
        # logger.debug()('######## distance_update_evader #################')
        r2_x, r2_y, current_yaw = round(pose_r['x'], 2), round(pose_r['y'], 2), round(pose_r['yaw'], 3)
        self.evader_pose = (r2_x, r2_y, current_yaw)
        # logger.debug()('evader_go_to: {}'.format(self.evader_goal))

        if self.obstacle_r:
            if self.evader_pose and self.evader_goal:
                self.des_yaw_evader = math.atan2(self.evader_goal[1] - self.evader_pose[1],
                                                 self.evader_goal[0] - self.evader_pose[0])

                if not np.sign(self.des_yaw_evader) == np.sign(self.evader_pose[2]):
                    self.trap_danger_evader = True
                else:
                    self.trap_danger_evader = False

    @staticmethod
    def is_evader_semantic(semantic_camera_data):
        timestamp = semantic_camera_data["timestamp"]
        visible_objects = semantic_camera_data['visible_objects']
        for visible_object in visible_objects:
            if visible_object['name'] == "evader":
                return (visible_object['position'], timestamp)
        return ([None, None, None], timestamp)

    def laser_data_pursuer(self, data):
        # logger.debug()('####  laser_data_pursuer #####')
        # logger.debug()(obstacle_v)
        min_val = min(data['range_list'])
        self.pursuer_scan = data['range_list']
        if min_val < self.collision_distance:
            self.obstacle_v = True
            point = data['point_list'][data['range_list'].index(min_val)]
            self.angle_obs_v = math.degrees(math.atan(point[1] / point[0]))
        else:
            self.obstacle_v = False

    def distance_update_pursuer(self, pose_v):
        # logger.debug()('###### distance_update_pursuer ################')
        curr_pose_pursuer_x = round(pose_v['x'], 2)
        curr_pose_pursuer_y = round(pose_v['y'], 2)
        yaw_v = round(pose_v['yaw'], 3)
        self.pursuer_pose = (curr_pose_pursuer_x, curr_pose_pursuer_y, yaw_v)

        if self.pursuer_goal:
            self.des_yaw_pursuer = math.atan2(self.pursuer_goal[1] - self.pursuer_pose[1],
                                              self.pursuer_goal[0] - self.pursuer_pose[0])

            if not np.sign(self.des_yaw_pursuer) == np.sign(self.pursuer_pose[2]):
                self.trap_danger_pursuer = True
            else:
                self.trap_danger_pursuer = False

        D = self.get_distance(self.pursuer_pose[:2], self.evader_pose[:2])
        distance_profile.append(D)
        self.end_time = time.time()
        

    def derive_pursuer_goal(self, cam):
        global is_evader_visible
        self.beg_ctr += 1
        # logger.debug()('#### derive_pos ###########')
        ([pose_a, pose_b, _], ts) = self.is_evader_semantic(cam)
        if pose_a and pose_b:
            is_evader_visible = True
            self.camera_evader_pose = (pose_a, pose_b)
            if self.use_pred:
                if self.pred_type == 1:
                    self.pose_x_deque.append(round(pose_a, 2))
                    self.pose_y_deque.append(round(pose_b, 2))
                else:
                    if self.beg_ctr == 1:
                        self.f.x = np.array([[pose_a], [pose_b], [0.], [0.]])
                    else:
                        # update z here
                        if self.camera_evader_last_pose[0] and self.camera_evader_last_pose[1]:
                            z = [[pose_a], [pose_b], [(pose_a - self.camera_evader_last_pose[0]) / self.dt],
                                 [(pose_b - self.camera_evader_last_pose[1]) / self.dt]]
                            self.f.update(z)
        else:
            is_evader_visible = False
            if self.use_pred:
                if self.pred_type == 1:
                    
                    # print("CLEARING CLEARING DEQUE..................")
                    self.pose_x_deque.clear()
                    self.pose_y_deque.clear()

                    self.curr_pred_x, self.curr_pred_y = None, None

        self.camera_evader_last_pose = [pose_a, pose_b]
        # sys.exit(-1)
        # ToDO: Add if/else to get goal based on pursuer-type

        if self.curr_pred_x:
            goal_diff = np.linalg.norm(np.subtract((self.curr_pred_x, self.curr_pred_y),
                                                   (self.pursuer_pose[0], self.pursuer_pose[1])))
            if goal_diff < self.capture_distance - 0.5:
                self.curr_pred_x, self.curr_pred_y = None, None

        if self.use_pred is True:
            if self.curr_pred_x and not self.obstacle_v:
                self.pursuer_goal = (self.curr_pred_x, self.curr_pred_y)                
            elif self.camera_evader_pose and not self.obstacle_v:
                self.pursuer_goal = (self.camera_evader_pose[0], self.camera_evader_pose[1])
        else:
            # logger.debug('####### No Predictor #################3')
            if self.is_camera_based is True:
                if self.camera_evader_pose and not self.obstacle_v:
                    self.pursuer_goal = self.camera_evader_pose
            else:
                if self.evader_pose and not self.obstacle_v:
                    self.pursuer_goal = self.camera_evader_pose


        if self.pursuer_goal and not self.obstacle_v:
            # self.ctr += 1
            logger.debug('Pursuer going to: {}, from: {}'.format(self.pursuer_goal, self.pursuer_pose))
            self.motion_v.goto(self.pursuer_goal[0], self.pursuer_goal[1], 0.9, 0.05, self.max_linear_speed)

        time.sleep(0.01)

    def connect_to_morse(self):
        try:
            self.simu = Morse()
            return True
        except ConnectionRefusedError:
            return False
        except RuntimeError:
            return False


    def run(self, cur_run, spawn_file, log_file_csv, log_file_pkl):
        global log_data, ctr, spawn_df
        # logger.debug(' ###################### inside run() ##############################')
        # with Morse() as self.simu:
        try:
            while not self.connect_to_morse():
                continue
            # logger.debug()('#######  connected in baseline ################')
            spawn_df = pd.read_csv(spawn_file)
            log_data = np.array(spawn_df.loc[cur_run])

            self.goal_id = log_data[5]
            log_data = log_data[:5]

            self.motion_v = self.simu.pursuer.motion_v
            self.motion_v_vw = self.simu.pursuer.motion_v_vw
            self.sick_v = self.simu.pursuer.sick_v
            self.v_cam = self.simu.pursuer.v_cam

            self.evader_pose_sensor = self.simu.evader.pose_evader
            self.pursuer_pose_sensor = self.simu.pursuer.pose_pursuer

            self.motion_r = self.simu.evader.motion_r
            self.motion_r_vw = self.simu.evader.motion_r_vw
            self.sick_r = self.simu.evader.sick_r

            self.beg_ctr = 0

            # logger.debug()(' ########## sensors set #################')

            self.sick_r.subscribe(self.laser_data_evader)
            self.sick_v.subscribe(self.laser_data_pursuer)

            self.evader_pose_sensor.subscribe(self.distance_update_evader)
            self.pursuer_pose_sensor.subscribe(self.distance_update_pursuer)
            self.v_cam.subscribe(self.derive_pursuer_goal)

            self.episode_counter = cur_run

            self.log_file_csv = log_file_csv
            self.log_file_pkl = log_file_pkl

            self.start_time = time.time()
            pid_e = PidEvader()
            self.evader_thread = threading.Thread(target=pid_e.run, daemon=True)
            self.evader_thread.start()


            while True:
                self.predictor()
                    
                logger.debug(' ##################### Pursuer: {} #################'.format(ctr))
                #if self.pursuer_scan and self.pursuer_pose:
                if self.obstacle_v:
                    self.cleared_v = False
                    self.motion_v.stop()
                    # logger.debug()('### HALTING HERE #####')
                    sign = - np.sign(self.angle_obs_v)
                    self.motion_v_vw.set_speed(0.0, sign * self.pursuer_rotation_speed)

                    while not self.cleared_v:
                        # logger.debug()("NOT CLEARED V")
                        if not self.obstacle_v:
                            self.cleared_v = True
                    # logger.debug()("Cleared")
                    if self.trap_danger_pursuer:
                        self.motion_v_vw.set_speed(self.trap_escape_vel, self.trap_angular_vel)
                        time.sleep(self.trap_duration_time)
                    else:
                        self.motion_v_vw.set_speed(0.0, 0.0)

                #time.sleep(0.01)

        except ConnectionRefusedError:
            logger.exception('Could not connect to simulator.')
            traceback.print_exc()
            sys.exit()
        except FileNotFoundError:
            logger.exception('Could not connect to simulator.')
            traceback.print_exc()
            sys.exit()
        except Exception:
            logger.exception('Unknown Exception')
            traceback.print_exc()
            sys.exit()


if __name__ == '__main__':
    logger.debug('#################### Baseline Start #########################')

    print(sys.argv)
    # cur_run, spawn_file, log_file_csv, log_file_pkl = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    cur_run = int(sys.argv[1])

    if os.path.isfile(log_file_csv):
        save_df = pd.read_csv(log_file_csv)
    else:
        save_df = pd.DataFrame(columns=columns)

    __instance = MorseEnv().getInstance()
    __instance.run(cur_run, spawn_file, log_file_csv, log_file_pkl)
