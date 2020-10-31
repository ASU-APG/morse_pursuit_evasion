import traceback
from xmlrpc.server import SimpleXMLRPCServer

from pymorse import Morse
import subprocess
from threading import Thread
import sys
import time
import math
import pandas as pd
import numpy as np
from utils import setup_my_logger

logger = setup_my_logger('run_simulation', 1)

connection = None
pursuer_pid = None

# spawn locations file
spawn_file = 'spawnFiles/spawn_wall_worlds.csv'
script_name = 'run_pursuer.py'

# total runs for the experiment
max_runs = 200
cur_run = 0

timeout_append = False


def is_running():
    global connection
    try:
        # with Morse() as morse:
        connection = Morse()
        return True
    except ConnectionRefusedError:
        return False
    except RuntimeError:    
        return False


def world_sel():
    data_frame = pd.read_csv(spawn_file)
    world_sel.counter += 1
    check = world_sel.counter
    count = len(data_frame)
    if check == count:
        print('All simulations completed')
        exit()


    world = data_frame.world[check]
    (xe, ye) = eval(str(data_frame.evader_spawn[check]))
    (xp, yp) = eval(str(data_frame.pursuer_spawn[check]))
    ori_1 = float(data_frame.evader_ori[check])
    ori_2 = float(data_frame.pursuer_ori[check])
    logger.info('{}, {}, {}, {}, {}, {}, {}'.format(type(world), type(xe), type(ye), type(xp), type(yp), type(ori_1), type(ori_2)))

    return world, xe, ye, xp, yp, ori_1, ori_2


def create_server():
    server = SimpleXMLRPCServer(("localhost", 8000))
    print("Listening on port 8000...")
    server.register_function(world_sel, "world_sel")
    server.serve_forever()
    print('in server')


if __name__ == '__main__':
    try:
        t = Thread(target=create_server)
        t.start()

        df = pd.read_csv(spawn_file)
        columns = ['world', 'evader_spawn', 'pursuer_spawn', 'evader_ori', 'pursuer_ori', 'dist_profile', 'sim_time',
                   'steps_count',
                   'is_Captured', 'is_Visible']

        # Loop Simulation
        step_count = 0
        world_sel.counter = cur_run - 1
        proc_id = None
        time_count = 0
        row_id = cur_run
        distance = []
        start_time = 0
        is_captured = False
        connection = None
        script_start_time = time.time()
        logs = []

        while cur_run < max_runs:
            log_data = np.array(df.loc[cur_run])
            logger.debug('ld: {}'.format(log_data))
            try:
                proc_id = subprocess.Popen(['morse', 'run', 'default.py'])
                time.sleep(1)
                while not is_running():
                    continue
                logger.info('connected')
                start_time = time.time()
                # pursuer_pid = subprocess.Popen(['python', script_name])
                # time.sleep(0.01)
                con = Morse()
                while connection:
                    # con = connection
                    # poll = pursuer_pid.poll()

                    logger.debug('### inside morse #####')
                    cur_time = time.time()
                    pose_e = con.rpc('evader.pose_evader', 'get_local_data')
                    pose_p = con.rpc('pursuer.pose_pursuer', 'get_local_data')
                    is_visible = False
                    semantic_camera_data = con.rpc('pursuer.v_cam', 'get_local_data')
                    visible_objects = semantic_camera_data['visible_objects']
                    for visible_object in visible_objects:
                        if visible_object['name'] == "evader":
                            is_visible = True
                    if time_count == 0 and not is_visible:
                        # log_data = np.append(log_data, [None, None, None, None, None])
                        # logs.append(log_data)
                        is_captured = False
                        distance = []
                        logger.debug('Initially not visible; so moving to the next episode')
                        row_id += 1
                        time_count = 0
                        con.quit()
                        connection = None
                        out = None
                    else:
                        if time_count == 0:
                            # kill = lambda p:p.terminate()
                            command = ['python', script_name, str(row_id)]
                            logger.info('command: {}'.format(command))
                            start = time.time()
                            #print("############ time exceeeded ################")
                            pursuer_pid = subprocess.Popen(command)

                            time.sleep(0.01)
                            time_count +=1
                        else:
                            dist = math.sqrt(pow((pose_p['x'] - pose_e['x']), 2) + pow((pose_p['y'] - pose_e['y']), 2))
                            #time_count += 1
                            time_out = time.time() - start > 55.0
                            close = dist < 3.0
                            if time_out or close:
                                logger.info('time_out' if time_out else 'distance')
                                # if not timeout_append:
                                # start = 0.0
                                # logger.error("############ time exceeeded ################")
                                time_count = 0
                                cur_run += 1
                                row_id += 1
                                con.quit()
                                connection = None
                                p = pursuer_pid
                                p.terminate()

                            else:
                                # distance.append(dist)
                                time_count += 1
                                # continue

            except KeyboardInterrupt:
                sys.exit()
            except:
                logger.exception('Error in morse_simulation.py')
                traceback.print_exc()

        script_end_time = time.time()
        logger.info('Total time taken: {} sec'.format(script_end_time - script_start_time))
    except KeyboardInterrupt:
        subprocess.Popen(['killall', '-9' 'python'])
    except:
        logger.exception('Error in morse_simulation.py')
        traceback.print_exc()
        sys.exit(0)
