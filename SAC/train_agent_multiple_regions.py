import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sofia_env_multiple_regions import SoFiAEnv
import torch

# replace with your available GPU id, otherwise use cpu
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


def make_env(input_cube, temp_par_file, workspace, 
             truth_catalogue, input_regions, input_region_names,
             max_steps, plot_interval,
             benchmarks, sofia_num_threads):
    '''
    Create the environment for agent.
    
    '''
    def _init():
        return SoFiAEnv(input_cube=input_cube, temp_par_file=temp_par_file, workspace=workspace, 
                        truth_catalogue=truth_catalogue, input_regions=input_regions,
                        input_region_names=input_region_names,
                        max_steps=max_steps, plot_interval=plot_interval,
                        benchmarks=benchmarks, sofia_num_threads=sofia_num_threads)
        
    return _init

def train_agent(
    total_timesteps=10000, # total number of training steps
    learning_rate=3e-4, # learning rate for the networks
    buffer_size=1000, # size of the replay buffer
    learning_starts=64, # number of steps before training starts
    batch_size=64, # batch size for training
    tau=0.005, # target network update rate
    gamma=0.99, # discount factor
    train_freq=1, # frequency of training
    gradient_steps=1, # number of gradient steps per training
    save_path=None, # path to save the model
    input_cube=None, # path to the input cube
    temp_par_file=None, # path to the template sofia config file
    workspace=None, # path to workspace
    truth_catalogue=None, # path to the truth catalogue
    input_regions=None, # list of regions to train on
    input_region_names=None, # list of region names. Names will be numbers if None
    max_steps=100, # maximum number of steps per episode
    checkpoint_freq=100, # frequency of saving checkpoints
    benchmarks=None, # list of benchmark scores
    plot_interval=100, # frequency of plotting
    sofia_num_threads=32, # number of threads for sofia
):
    '''
    Train agent
    '''
        
    env = DummyVecEnv([make_env(input_cube=input_cube, temp_par_file=temp_par_file, workspace=workspace, 
                                truth_catalogue=truth_catalogue, input_regions=input_regions,
                                input_region_names=input_region_names,
                                max_steps=max_steps, plot_interval=plot_interval,
                                benchmarks=benchmarks, sofia_num_threads=sofia_num_threads)])
    
    model = SAC(
        "MlpPolicy", # use mlp network
        env, # environment
        learning_rate=learning_rate, # learning rate for the networks
        buffer_size=buffer_size, # size of the replay buffer
        learning_starts=learning_starts, # number of steps before training starts
        batch_size=batch_size, # batch size for training
        tau=tau, # target network update rate
        gamma=gamma, # discount factor
        train_freq=train_freq, # frequency of training
        gradient_steps=gradient_steps, # number of gradient steps per training
        verbose=1, # verbosity level
        device=device, # device to use
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq, # frequency of saving checkpoints
        save_path=os.path.join(save_path, "checkpoints/"), # path to save the checkpoints
        name_prefix="sofia_model", # prefix for the checkpoint files
        save_replay_buffer=True, # save the replay buffer for restoring the checkpoint
    )
    
    model.learn(
        total_timesteps=total_timesteps, # total number of training steps
        callback=checkpoint_callback, # callback for saving checkpoints
        progress_bar=False, # if show progress bar
        log_interval=1 # log interval
    )
    
    model.save(os.path.join(save_path, "final_model")) # save the final model
    
    env.close() # close the environment
    
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
    
    train_regions = [region_0, region_1, region_2]
    train_benchmarks = [benchmark['region_0']['score'], benchmark['region_1']['score'], benchmark['region_2']['score']]
    train_num_of_sources = [benchmark['region_0']['num_of_sources'], benchmark['region_1']['num_of_sources'],
                            benchmark['region_2']['num_of_sources']]
    
    train_region_names = ['region_0', 'region_1', 'region_2']
    
    for i in range(len(train_regions)):
        print(train_regions[i], train_num_of_sources[i], train_benchmarks[i])
    
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
        input_region_names=train_region_names,
        max_steps=100,
        checkpoint_freq=1,
        benchmarks=train_benchmarks,
        plot_interval=2,
        sofia_num_threads=16,
    )
    
    