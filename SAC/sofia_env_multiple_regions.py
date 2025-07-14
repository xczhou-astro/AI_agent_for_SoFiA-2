from joblib import Parallel, delayed
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
from astropy.io import fits
from astropy.wcs import WCS
import os
import sys
import json
import subprocess
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(project_root, "scripts"))

from utils import (sofia_par, sofia_score, count_truth, custom_serialier,
                   plot_summary_multiple_regions, plot_summary_region_no_benchmarks)

class SoFiAEnv(gym.Env):
    """
    Custom Environment for SoFiA parameter optimization using RL
    """
    def __init__(self, input_cube, temp_par_file, workspace, 
                 truth_catalogue, input_regions=None, 
                 max_steps=100, plot_interval=50,
                 benchmarks=[], param_configs=None):
        super(SoFiAEnv, self).__init__()
        
        self.input_cube = input_cube
        self.temp_par_file = temp_par_file
        self.workspace = workspace
        self.truth_catalogue = truth_catalogue
        self.input_regions = input_regions
        self.plot_interval = plot_interval
        self.benchmarks = benchmarks if len(benchmarks) > 0 else None
        
        self.num_of_regions = len(input_regions)
        
        self.base_path = os.getcwd()
        
        os.makedirs(self.workspace, exist_ok=True) # base
        
        # print(self.input_regions)
        
        self.region_paths = []
        self.region_names = []
        for region in self.input_regions:

            path = '_'.join(map(str, region))
            region_path = os.path.join(self.workspace, path)
            self.region_paths.append(region_path)
            os.makedirs(region_path, exist_ok=True)
            self.region_names.append(path)
        
        self.data_params = {
            'input.data': input_cube,
        }
        
        self.max_steps = max_steps
        
        with fits.open(self.input_cube) as hdu:
            self.wcs = WCS(hdu[0].header)
        
        if param_configs is None:
        
            self.param_configs = {
                'scfind.threshold': {'low': 3.5, 'high': 4.0, 'type': 'continuous'},
                'reliability.threshold': {'low': 0.0, 'high': 0.6, 'type': 'continuous'},
                'reliability.minSNR': {'low': 0.0, 'high': 2.0, 'type': 'continuous'},
                'reliability.scaleKernel': {'low': 0.1, 'high': 0.7, 'type': 'continuous'},
            }
        else:
            self.param_configs = param_configs
            
        with open(f'{self.workspace}/config.json', 'w') as f:
            json.dump(self.param_configs, f, indent=4, default=custom_serialier)
            
        for key, value in self.param_configs.items():
            print(key, value['low'], value['high'])
        
        # Define action space (normalized between -1 and 1)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(len(self.param_configs),), dtype=np.float32
        )
        
        # Define observation space (current parameters + performance metrics)
        # augmentated state
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(len(self.param_configs) + 3,), dtype=np.float32
        # )
        # original state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.param_configs) + 1,), dtype=np.float32
        )
        
        self.episode_count = -1
        self.step_count = 0 # step in an episode
        
        self.best_score_recorder = -np.inf
        self.best_score_ratio_recorder = -np.inf
        os.makedirs(f'{self.workspace}/best_params', exist_ok=True)
        
        self.step_recorder = 0 # total step
        
        self.reward_scale_factor = 5
        self.reward_exp_factor = 3
        self.improvement_threshold = 0.005
        
        self._setup_logging()
        
        # Initialize state
        self.reset()
    
    def _setup_logging(self):
        
        self.reward_recorder = os.path.join(self.workspace, 'reward_recorder.txt')
        
        with open(self.reward_recorder, 'w') as f:
            
            columns = 'episode,step,scfind.threshold,reliability.threshold,reliability.minSNR,reliability.scaleKernel'
            for i in range(self.num_of_regions):
                columns += f',score_{i},score_ratio_{i},num_of_matched_sources_{i}'
            columns += ',total_score,total_score_ratio,total_num_of_matched_sources'
            columns += ',reward\n'
            
            f.write(columns)
        
                
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.episode_count += 1
        self.step_count = 0
        
        # Initialize parameters with random values within bounds
        self.current_params = {}
        for param, config in self.param_configs.items():
            if config['type'] == 'discrete':
                self.current_params[param] = np.random.randint(config['low'], config['high'] + 1)
            elif config['type'] == 'continuous':
                self.current_params[param] = np.random.uniform(config['low'], config['high'])
            else:
                raise ValueError(f"Invalid parameter type: {config['type']}")
        
        # Initialize performance metrics
        self.performance_metrics = {
            'score_ratio': 0,
        }
        
        self.previous_score_ratio = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        
        self.step_count += 1
        
        for i, (param, config) in enumerate(self.param_configs.items()):
            low = config['low']
            high = config['high']
            scaled_action = (action[i] + 1) / 2  # Scale to [0, 1]
            if config['type'] == 'discrete':
                new_value = low + int(scaled_action * (high - low))
                self.current_params[param] = max(low, min(high, new_value))
            elif config['type'] == 'continuous':
                new_value = low + scaled_action * (high - low)
                self.current_params[param] = new_value
            else:
                raise ValueError(f"Invalid parameter type: {config['type']}")
        
        score_regions, score_ratio_regions, num_of_matched_sources_regions, num_of_sources_regions, total_recorder = self.pipline()
        
        self.performance_metrics['score_ratio'] = total_recorder['score_ratio']
        
        is_max_step = self.step_count == self.max_steps
        
        is_initial_step = self.step_count == 1
        
        done = self._check_done(is_max_step)
        
        reward = self._calculate_reward(total_recorder['score_ratio'], is_max_step, is_initial_step)

        with open(self.reward_recorder, 'a') as f:
            
            data = f'{self.episode_count}, {self.step_count},'
            data += f'{self.current_params["scfind.threshold"]}, {self.current_params["reliability.threshold"]},'
            data += f'{self.current_params["reliability.minSNR"]}, {self.current_params["reliability.scaleKernel"]}'
            
            for i in range(self.num_of_regions):
                data += f',{score_regions[i]}, {score_ratio_regions[i]}, {num_of_matched_sources_regions[i]}'
            
            data += f',{total_recorder["score"]}, {total_recorder["score_ratio"]}, {total_recorder["num_of_matched_sources"]}'
            data += f',{reward}\n'
            
            f.write(data)
        
        
        if self.step_recorder % self.plot_interval == 0 and self.step_recorder > 0:
            if self.benchmarks is not None:
                plot_summary_multiple_regions(self.reward_recorder, self.num_of_regions, self.workspace, self.benchmarks)
            else:
                plot_summary_region_no_benchmarks(self.reward_recorder, self.num_of_regions, self.workspace)
        
        self.step_recorder += 1
        
        return self._get_observation(), reward, done, False, {}
    
    def _get_observation(self):
        # Combine current parameters and performance metrics into observation
        obs = []
        for param in self.param_configs.keys():
            obs.append(self.current_params[param])
        obs.append(self.performance_metrics['score_ratio'])
    
        return np.array(obs, dtype=np.float32)


    def run_sofia_and_score(self, idx, episode_count, step_count):
        
        par_file = self.temp_par_file
        
        region_path = self.region_paths[idx]
        sofia_output_path = f'{region_path}/episode_{episode_count}/step_{step_count}'
        
        os.makedirs(sofia_output_path, exist_ok=True)
        
        self.data_params['output.directory'] = sofia_output_path
        self.data_params['input.region'] = ', '.join(map(str, self.input_regions[idx]))
        
        params = self.current_params | self.data_params
        
        par_file = sofia_par(par_file, params, sofia_output_path)
        
        os.chdir(sofia_output_path)
        
        subprocess.run(f'sofia sofia_par.par > sofia_log.out 2>&1', shell=True,
                       capture_output=True, text=True)
        
        os.chdir(self.base_path)
        
        # num_of_sources = self.benchmarks[idx]['num_of_sources']
        num_of_sources = count_truth(self.truth_catalogue, self.wcs, self.input_regions[idx])
        
        if os.path.exists(f'{sofia_output_path}/sofia_output_cat.txt'):
            result = 0
        else:
            result = 1
            
        if result == 0:
            score_ratio, score, num_of_matched_sources, num_of_sources = \
                sofia_score(sofia_output_path, self.truth_catalogue, self.wcs,
                           input_region=self.input_regions[idx])
        
        elif result == 1:
            score = -num_of_sources
            score_ratio = -1.0
            num_of_matched_sources = 0
            
        logging = {}
        logging['params'] = self.current_params.copy()
        logging['result'] = result
        logging['score'] = score
        logging['score_ratio'] = score_ratio
        logging['num_of_matched_sources'] = num_of_matched_sources
        logging['num_of_sources'] = num_of_sources
        
        with open(f'{sofia_output_path}/logging.json', 'w') as f:
            json.dump(logging, f, indent=4, default=custom_serialier)
        
        return result, score, score_ratio, num_of_matched_sources, num_of_sources
        
    
    def pipline(self):
        
        output = Parallel(n_jobs=self.num_of_regions)(delayed(self.run_sofia_and_score)(i, self.episode_count, self.step_count) for i in range(self.num_of_regions))
        results = [output[i][0] for i in range(self.num_of_regions)]
        scores = [output[i][1] for i in range(self.num_of_regions)]
        score_ratios = [output[i][2] for i in range(self.num_of_regions)]
        num_of_matched_sources = [output[i][3] for i in range(self.num_of_regions)]
        num_of_sources = [output[i][4] for i in range(self.num_of_regions)]
        
        total_score = sum(scores)
        total_sources = sum(num_of_sources)
        
        total_score_ratio = total_score / total_sources
        total_num_of_matched_sources = sum(num_of_matched_sources)
        
        score_regions = {}
        score_ratio_regions = {}
        num_of_matched_sources_regions = {}
        num_of_sources_regions = {}
        
        for i in range(self.num_of_regions):
            score_regions[i] = scores[i]
            score_ratio_regions[i] = score_ratios[i]
            num_of_matched_sources_regions[i] = num_of_matched_sources[i]
            num_of_sources_regions[i] = num_of_sources[i]
        
        total_recorder = {}
        total_recorder['score'] = total_score
        total_recorder['score_ratio'] = total_score_ratio
        total_recorder['num_of_matched_sources'] = total_num_of_matched_sources
        total_recorder['num_of_sources'] = total_sources
        
        if total_score > self.best_score_recorder:
            self.best_score_recorder = total_score
            self.best_score_ratio_recorder = total_score_ratio
            best_params = self.current_params.copy()
            best_params['episode'] = self.episode_count
            best_params['step'] = self.step_count
            
            params = {}
            for param in self.param_configs.keys():
                params[param] = self.current_params[param]
                
            best_params['params'] = params
            
            recorder_regions = {}
            for i in range(self.num_of_regions):
                
                recorder = {}
                recorder['score'] = scores[i]
                recorder['score_ratio'] = score_ratios[i]
                recorder['num_of_matched_sources'] = num_of_matched_sources[i]
                recorder['num_of_sources'] = num_of_sources[i]
                
                recorder_regions[f'region_{i}'] = recorder
                
            best_params['recorder_regions'] = recorder_regions
            
            total_recorder = {}
            total_recorder['score'] = total_score
            total_recorder['score_ratio'] = total_score_ratio
            total_recorder['num_of_matched_sources'] = total_num_of_matched_sources
            total_recorder['num_of_sources'] = total_sources
            
            best_params['total_recorder'] = total_recorder
            
            with open(f'{self.workspace}/best_params/best_params.json', 'w') as f:
                
                json.dump(best_params, f, indent=4, default=custom_serialier)
        
        return score_regions, score_ratio_regions, num_of_matched_sources_regions, num_of_sources_regions, total_recorder
        
    
    def _check_done(self, is_max_steps):
        
        done = False
        if is_max_steps:
            done = True
        
        return done
    
    # def _calculate_reward(self, score_ratio, is_max_step, is_initial_step):
    #     """
    #     Improved reward function with better stability and learning dynamics
    #     """
    #     # Handle failed runs
    #     if score_ratio < 0:
    #         return -5.0
        
    #     # More moderate exponential scaling for better stability
    #     reward = self.reward_scale_factor * (np.exp(self.reward_exp_factor * score_ratio) - 1.0)
        
    #     # Calculate improvement with threshold to reduce noise
    #     improvement = score_ratio - self.previous_score_ratio
        
    #     # Only reward significant improvements to reduce fluctuations
    #     if improvement > self.improvement_threshold:
    #         reward += 1.0
    #     elif improvement < -self.improvement_threshold:
    #         reward -= 1.0
        
    #     # Reward for achieving new best score (not just improvement)
    #     if not is_initial_step and score_ratio > self.best_score_ratio_recorder:
    #         reward += 2.0
            
    #     # Small penalty for episode end to encourage efficiency
    #     if is_max_step:
    #         reward -= 1.0
        
    #     self.previous_score_ratio = score_ratio
        
    #     return reward 
    
    def _calculate_reward(self, score_ratio, is_max_step, is_initial_step):
        # Handle failed runs
        if score_ratio < 0:
            return -5.0
        
        scale_factor = 10
        exp_factor = 5
        reward = scale_factor * (np.exp(exp_factor * score_ratio) - 1.0)
        
        improvement = score_ratio - self.previous_score_ratio
        # improvement_threshold = 0.008
        improvement_threshold = 0
            
        if not is_initial_step:
            
            if improvement > improvement_threshold:
                reward += 2.0
            elif improvement < -improvement_threshold:
                reward -= 2.0
                

            if score_ratio > self.best_score_recorder:
                self.best_score_recorder = score_ratio
                reward += 3.0
                
        if is_max_step:
            reward -= 3.0
        
        self.previous_score_ratio = score_ratio
        
        return reward
