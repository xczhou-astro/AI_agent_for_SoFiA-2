import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
import gymnasium as gym
from sofia_env_multiple_regions import SoFiAEnv
import torch
import argparse

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def make_env(input_cube, temp_par_file, workspace, 
             truth_catalogue, input_regions,
             max_steps, plot_interval,
             benchmarks):
    def _init():
        return SoFiAEnv(input_cube, temp_par_file, workspace, 
                        truth_catalogue, input_regions,
                        max_steps, plot_interval,
                        benchmarks)
        
    return _init

def train_agent(
    total_timesteps=10000,
    learning_rate=3e-4,
    buffer_size=1000,
    learning_starts=64,
    batch_size=64,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    save_path=None,
    input_cube=None,
    temp_par_file=None,
    workspace=None,
    truth_catalogue=None,
    input_regions=None,
    max_steps=100,
    checkpoint_freq=100,
    benchmarks=None,
    plot_interval=100,
):
        
    env = DummyVecEnv([make_env(input_cube, temp_par_file, workspace, truth_catalogue,
                   input_regions, max_steps, plot_interval, benchmarks)])
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        verbose=1,
        device=device,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=os.path.join(save_path, "checkpoints/"),
        name_prefix="sofia_model",
        save_replay_buffer=True,
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=False,
        log_interval=1
    )
    
    model.save(os.path.join(save_path, "final_model"))
    
    env.close()
    
    return model

if __name__ == "__main__":
    
    workspace = "workspace"
    os.makedirs(workspace, exist_ok=True)
    
    workspace = os.path.abspath(workspace)
    
    save_path = f"{workspace}/models"
    
    input_cube = f'/home/xczhou/share/xczhou/agent/dataset/development/sky_dev_v2.fits'
    temp_par_file = f'/home/xczhou/share/xczhou/agent/dataset/template_par_file.par'
    truth_catalogue = f'/home/xczhou/share/xczhou/agent/dataset/development/sky_dev_truthcat_v2.txt'
    
    
    region_0 = [0, 322, 0, 322, 0, 6668]
    region_1 = [0, 322, 322, 643, 0, 6668]
    region_2 = [322, 643, 0, 322, 0, 6668]
    region_3 = [322, 643, 322, 643, 0, 6668]
    
    benchmark = {}
    benchmark['region_0'] = {}
    benchmark['region_1'] = {}
    benchmark['region_2'] = {}
    benchmark['region_3'] = {}

    benchmark['region_0']['score'] = 38.69
    benchmark['region_0']['score_ratio'] = 0.068
    benchmark['region_0']['num_of_sources'] = 572


    benchmark['region_1']['score'] = 44.37
    benchmark['region_1']['score_ratio'] = 0.062
    benchmark['region_1']['num_of_sources'] = 717

    benchmark['region_2']['score'] = 36.42
    benchmark['region_2']['score_ratio'] = 0.055
    benchmark['region_2']['num_of_sources'] = 660

    benchmark['region_3']['score'] = 60.2
    benchmark['region_3']['score_ratio'] = 0.082
    benchmark['region_3']['num_of_sources'] = 734
    
    train_regions = [region_0, region_2, region_3]
    train_benchmarks = [benchmark['region_0'], benchmark['region_2'], benchmark['region_3']]
    
        
    model = train_agent(
        total_timesteps=10000,
        learning_rate=1e-4,
        buffer_size=1000,
        learning_starts=64,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        save_path=save_path,
        input_cube=input_cube,
        temp_par_file=temp_par_file,
        workspace=workspace,
        truth_catalogue=truth_catalogue,
        input_regions=train_regions,
        max_steps=100,
        checkpoint_freq=1,
        benchmarks=train_benchmarks,
        plot_interval=10,
    )
    
    