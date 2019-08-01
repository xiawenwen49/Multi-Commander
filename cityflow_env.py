import cityflow
import pandas as pd
import os
import json
import math
# from sim_setting import sim_setting_control

class CityFlowEnv(object):
    def __init__(self, config):
        # cityflow_config['rlTrafficLight'] = rl_control # use RL to control the light or not
        self.eng = cityflow.Engine(config['cityflow_config_file'], thread_num=config['thread_num'])

        self.config = config
        self.num_step = config['num_step']
        self.state_size = config['state_size']
        self.lane_phase_info = config['lane_phase_info'] # "intersection_1_1"

        self.intersection_id = list(self.lane_phase_info.keys())[0]
        self.start_lane = self.lane_phase_info[self.intersection_id]['start_lane']
        self.phase_list = self.lane_phase_info[self.intersection_id]["phase"]
        self.phase_startLane_mapping = self.lane_phase_info[self.intersection_id]["phase_startLane_mapping"]

        self.current_phase = self.phase_list[0]
        self.current_phase_time = 0
        self.yellow_time = 5
        self.state_store_i = 0

        self.phase_log = []


    def reset(self):
        self.eng.reset()

    def step(self, next_phase):
        if self.current_phase == next_phase:
            self.current_phase_time += 1
        else:
            self.current_phase = next_phase
            self.current_phase_time = 1

        self.eng.set_tl_phase(self.intersection_id, self.current_phase) # set phase of traffic light
        self.eng.next_step()
        self.phase_log.append(self.current_phase)

        return self.get_state(), self.get_reward() # return next_state and reward

    def get_state(self):
        state = {}
        state['lane_vehicle_count'] = self.eng.get_lane_vehicle_count()  # {lane_id: lane_count, ...}
        state['start_lane_vehicle_count'] = {lane: self.eng.get_lane_vehicle_count()[lane] for lane in self.start_lane}
        state['lane_waiting_vehicle_count'] = self.eng.get_lane_waiting_vehicle_count()  # {lane_id: lane_waiting_count, ...}
        state['lane_vehicles'] = self.eng.get_lane_vehicles()  # {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
        state['vehicle_speed'] = self.eng.get_vehicle_speed()  # {vehicle_id: vehicle_speed, ...}
        state['vehicle_distance'] = self.eng.get_vehicle_distance() # {vehicle_id: distance, ...}
        state['current_time'] = self.eng.get_current_time()
        state['current_phase'] = self.current_phase
        state['current_phase_time'] = self.current_phase_time

        # return_state = np.array(list(state['start_lane_vehicle_count'].values()) + [state['current_phase']] )
        # return_state = np.reshape(return_state, [1, self.state_size])
        # if self.state_store_i <= 30:
        #     with open("test/state/state-{}".format(self.state_store_i+1), 'w') as file:
        #         json.dump(state, file)
        #     self.state_store_i  += 1
            
        return state

    
    def state_transform(self):
        pass


    def get_reward(self):
        # a sample reward function which calculates the total of waiting vehicles
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        lane_waiting_vehicle_count_list = list(lane_waiting_vehicle_count.values())
        reward = -1 * ( sum(lane_waiting_vehicle_count_list) + max(lane_waiting_vehicle_count_list) )
        return reward

    def log(self):
        if not os.path.exists(self.config['replay_data_path']):
            os.makedirs(self.config["replay_data_path"])
        
        # self.eng.print_log(self.config['replay_data_path'] + "/replay_roadnet.json",
        #                    self.config['replay_data_path'] + "/replay_flow.json")

        df = pd.DataFrame({self.intersection_id: self.phase_log[:self.num_step]})
        df.to_csv(os.path.join(self.config['replay_data_path'], 'signal_plan.txt'), index=None)
<<<<<<< HEAD
=======

class CityFlowEnvR1(CityFlowEnv):
    def __init__(self, config):
        super(CityFlowEnvR1, self).__init__(config)
    
    def get_reward(self):
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        reward = (100000 - lane_vehicle_count)/1 + math.exp(-1 * lane_vehicle_count)
        return reward
>>>>>>> ab8dec1bbb6699d23bad3eac393eca66241cef47
