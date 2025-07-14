import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from stable_baselines3 import SAC
from sofia_env_multiple_regions import SoFiAEnv
from joblib import Parallel, delayed
import shutil

def run_evaluation(model_path, step):
    
    eval_workspace = f'eval_models/model_{step:05d}'
    if os.path.exists(eval_workspace) and len(os.listdir(os.path.join(eval_workspace, '0_322_322_643_0_6668/episode_1'))) == 100:
        print(f'Evaluation for step {step} with model {model_path} already exists')
        return
    
    os.makedirs(eval_workspace, exist_ok=True)
    
    logs_path = os.path.join(eval_workspace, 'logs.txt')
    
    with open(logs_path, 'w') as f:
        f.write(f'Running evaluation for step {step} with model {model_path}\n')

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

    eval_regions = [region_1]
    eval_benchmarks = [benchmark['region_1']]
    
    with open(logs_path, 'a') as f:
        f.write(f'eval_regions: {eval_regions}\n')
        f.write(f'eval_benchmarks: {eval_benchmarks}\n')
    
    buffer_path = os.path.join(os.path.dirname(model_path), f'sofia_model_replay_buffer_{step}_steps.pkl')
    
    shutil.copy(buffer_path, os.path.join(eval_workspace, os.path.basename(buffer_path)))
    shutil.copy(model_path, os.path.join(eval_workspace, os.path.basename(model_path)))
    
    eval_workspace = os.path.abspath(eval_workspace)
    
    eval_env = SoFiAEnv(
        input_cube=input_cube,
        temp_par_file=temp_par_file,
        workspace=eval_workspace,
        truth_catalogue=truth_catalogue,
        input_regions=eval_regions,
        max_steps=100,
        plot_interval=10,
        benchmarks=eval_benchmarks,
    )
    
    model = SAC.load(model_path, env=eval_env, device='cuda:4')
    
    obs, _ = eval_env.reset()
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, _, _ = eval_env.step(action)
        if done:
            break
        
    with open(logs_path, 'a') as f:
        f.write(f'Evaluation for step {step} with model {model_path} completed\n')

if __name__ == '__main__':

    model_dir = 'workspace/models/checkpoints'
    model_names = [name for name in os.listdir(model_dir) if name.endswith('.zip')]
    num_step = [int(re.findall(r'\d+', model_name)[0]) for model_name in model_names]

    model_paths = [os.path.join(model_dir, model_name) for model_name in model_names]

    idx_sort = np.argsort(np.array(num_step))
    model_paths = [model_paths[i] for i in idx_sort]
    num_steps = [num_step[i] for i in idx_sort]
    
    begin = 3999
    model_paths = model_paths[begin:]
    num_steps = num_steps[begin:]
    
    interval = 100
    model_paths = model_paths[::interval]
    num_steps = num_steps[::interval]
    
    n_jobs = 4
    Parallel(n_jobs=n_jobs)(delayed(run_evaluation)(model_path, num_steps[i]) 
                            for i, model_path in enumerate(model_paths))
    