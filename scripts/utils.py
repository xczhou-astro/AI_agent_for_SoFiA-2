import pandas as pd
import numpy as np
import astropy.constants as const
import astropy.units as u
import subprocess
import os
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import shutil
import sys
from physical_parameter_conversion import physical_parameter_conversion
from ska_sdc.sdc2.sdc2_scorer import Sdc2Scorer
import re
import time

class TeeLogger:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.terminal = sys.stdout
        
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.terminal.flush()
        self.file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()

def standardize_sofia_catalogue(output_path, method='SoFiA_Team', 
                                options={'no_corr': True, 
                                         'drop_neg_nan': True,
                                         'skew_fill_cut': True, 
                                         'SDC2_cosmo': True}):
    
    '''
    Method:
        SoFiA_Team:
            - Use the method in SoFiA_Team, need VOTable file
        Tianlai_Team:
            - Use the method in Tianlai_Team, need text file
    '''
    
    filenames = os.listdir(output_path)
    
    if method == 'Tianlai_Team':
        
        sofia_catalogue = next((f for f in filenames if 'cat.txt' in f), None)
        sofia_catalogue = os.path.join(output_path, sofia_catalogue)
    
        with open(sofia_catalogue, 'r') as f:
            sofia_data = f.readlines()
            
        header = sofia_data[18].strip('#').strip('\n').split()
        df_detected = pd.read_csv(sofia_catalogue, sep='\s+', skiprows=21,
                            names=header)
        
        df_detected.dropna(inplace=True)
        
        df_detected = df_detected[df_detected['kin_pa'] >= -1.0]

        # ell_maj (pix -> arcsec)
        pixelScale = 2.8 # in arcsec
        df_detected['ell_maj_arcsec'] = df_detected['ell_maj'] * pixelScale

        # Line width (Hz -> km/s)
        rest_freq = 1420.40575177E6  # Hz
        # chan_size = 3.0 * 10**4 # Hz
        df_detected['w20_velocity'] = df_detected['w20'] * const.c.value / rest_freq * 1e-3

        # Inclination 
        df_detected['inclination'] = np.arccos(df_detected['ell_min'] / df_detected['ell_maj']) * 180 / np.pi
        
        # bmaj = 1.94444449153e-03
        # bmin = 1.94444449153e-03
        # pixel_size = 7.77777777778E-04
        # beam_area = np.pi * bmaj * bmin / (4 * np.log(2) * pixel_size * pixel_size)
        # df_detected['f_sum_2'] = df_detected['f_sum'] * chan_size / beam_area

        selected_columns = ['id', 'ra', 'dec', 'ell_maj_arcsec',
                            'f_sum', 'freq', 'kin_pa', 'inclination',
                            'w20_velocity']

        rename_columns = ['id', 'ra', 'dec', 'hi_size',
                        'line_flux_integral', 'central_freq',
                        'pa', 'i', 'w20']

        df_detected = df_detected[selected_columns]
        df_detected.columns = rename_columns # match columns in df_truth and df_detected
    
    elif method == 'SoFiA_Team':
        
        # print(output_path)
        
        sofia_catalogue = next((f for f in filenames if 'cat.xml' in f), None)
        # print(sofia_catalogue)
        
        sofia_catalogue = os.path.join(output_path, sofia_catalogue)
        
        df_detected = physical_parameter_conversion(sofia_catalogue, **options)
    
    return df_detected

    
def plot_coordinates(df_detected, df_truth, score, output_dir):
    plt.figure(figsize=(10, 10))
    plt.scatter(df_truth['ra'], df_truth['dec'], label='Truth', facecolor='none', edgecolor='blue', s=8)
    plt.scatter(df_detected['ra'], df_detected['dec'], c='red', label='Detected', s=2)
    plt.title(f'Matched: {df_detected.shape[0]} / Truth: {df_truth.shape[0]} | Score: {np.around(score, 2)}')
    plt.legend()
    
    imgname = os.path.join(output_dir, 'coordinates.png')
    plt.savefig(imgname)
    plt.close()
    
    
def plot_properties(df_detected, df_truth, score, output_dir):
    
    column_names = ['hi_size', 'line_flux_integral', 'central_freq', 'pa', 'i', 'w20']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i in range(2):
        for j in range(3):
            column_name = column_names[i * 3 + j]
            detected_values = df_detected[column_name].values
            truth_values = df_truth[column_name].values
            
            axes[i, j].hist(detected_values, bins=30, histtype='step', label='Detected', density=True)
            axes[i, j].hist(truth_values, bins=30, histtype='step', label='Truth', density=True)
            axes[i, j].legend()
            axes[i, j].set_title(column_name)
    
    plt.suptitle(f'Matched: {df_detected.shape[0]} / Truth: {df_truth.shape[0]} | Score: {np.around(score, 2)}')
    plt.tight_layout()
    imgname = os.path.join(output_dir, 'properties.png')
    plt.savefig(imgname)
    plt.close()
    
def plot_reward(reward_file, output_dir):
    
    reward = np.loadtxt(reward_file)
    num = len(reward)
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(num), reward)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Max Episode: {num}')
    plt.savefig(os.path.join(output_dir, 'reward.png'))
    plt.close()
    
def count_truth(truth_catalogue, wcs, input_region=None):
    
    df_truth = pd.read_csv(truth_catalogue, sep='\s+')
    
    if input_region is not None:
        x_range = [input_region[0], input_region[1]]
        y_range = [input_region[2], input_region[3]]
        z_range = [input_region[4], input_region[5]]
        
        ra_range, dec_range, freq_range = wcs.pixel_to_world_values(x_range, y_range, z_range)
        
        idx = (df_truth['ra'] <= ra_range[0]) & (df_truth['ra'] >= ra_range[1]) & \
              (df_truth['dec'] >= dec_range[0]) & (df_truth['dec'] <= dec_range[1])
        
        df_truth = df_truth[idx]
        df_truth.reset_index(drop=True, inplace=True)
        
    num_of_sources = df_truth.shape[0]
    
    return num_of_sources
        
    
def sofia_score(sofia_output_path, truth_catalogue, 
                wcs, input_region=None, method='SoFiA_Team', 
                options={'no_corr': True, 
                         'drop_neg_nan': True,
                         'skew_fill_cut': True, 
                         'SDC2_cosmo': True}):
    
    df_truth = pd.read_csv(truth_catalogue, sep='\s+')
    
    # Get truth catalogue in the input region
    if input_region is not None:
        x_range = [input_region[0], input_region[1]]
        y_range = [input_region[2], input_region[3]]
        z_range = [input_region[4], input_region[5]]
        
        ra_range, dec_range, freq_range = wcs.pixel_to_world_values(x_range, y_range, z_range)
        
        idx = (df_truth['ra'] <= ra_range[0]) & (df_truth['ra'] >= ra_range[1]) & \
              (df_truth['dec'] >= dec_range[0]) & (df_truth['dec'] <= dec_range[1])
        
        df_truth = df_truth[idx]
        df_truth.reset_index(drop=True, inplace=True)
        
    num_of_sources = df_truth.shape[0]
    
    df_detected = standardize_sofia_catalogue(sofia_output_path, method, options)
    
    sdc_scorer = Sdc2Scorer(df_detected, df_truth)
    run = sdc_scorer.run()
    score = run[0].value
    candidates = run[1]
    
    num_of_matched_sources = candidates.shape[0]
    
    score_ratio = score / num_of_sources

    columns = ['id', 'ra', 'dec'] + ['hi_size', 'line_flux_integral', 'central_freq', 'pa', 'i', 'w20']
    candidates = candidates[columns]
    
    path = sofia_output_path
    candidates.to_csv(os.path.join(path, 'candidates.csv'), index=False)
    
    plot_coordinates(candidates, df_truth, score, path)
    plot_properties(candidates, df_truth, score, path)
    
    return score_ratio, score, num_of_matched_sources, num_of_sources
    
    
def sofia_par(temp_par_file, params, save_path):
    
    with open(temp_par_file, 'r') as f:
        lines = f.readlines()
        
    keys, values = zip(*params.items())
        
    for key, value in zip(keys, values):
        
        if value is True:
            value = 'true'
        elif value is False:
            value = 'false'
            
        for i, line in enumerate(lines):
            if line.startswith(f'{key}'):
                parts = line.split('=')
                parts[1] = f'  {value}'
                lines[i] = '='.join(parts)
                lines[i] += '\n'
    
    new_par_file = f'{save_path}/sofia_par.par'
    with open(new_par_file, 'w') as f:
        f.writelines(lines)
        
    return new_par_file
    

def submit_sofia_job(par_file, slurm_template, episode, step, job_id=None):
    
    path = os.path.dirname(par_file) # path to the directory of the parameter file
        
    with open(slurm_template, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if line.startswith('#SBATCH --job-name='):
            lines[i] = f'#SBATCH --job-name=sofia_job_{episode}_{step}_{job_id}\n'
            
    with open(f'{path}/sofia_job.sh', 'w') as f:
        f.writelines(lines)
            
    base_path = os.getcwd()
    
    os.chdir(path)
    
    while True:
        # sofia_job.sh has --wait
        # wait for the job to finish
        result = subprocess.run(['sbatch', 'sofia_job.sh'], 
                                capture_output=True, text=True)
        
        if os.path.exists(f'log.out') and result.returncode == 0:
            break
        else:
            time.sleep(1)
            
    os.chdir(base_path)
        
    if os.path.exists(f'{path}/sofia_output_cat.txt'):
        return 0  # Success
    else:
        return 2  # Error
    
def submit_job(par_file, slurm_template, output_path, episode, step, env_id=0, 
               eval_flag=False):
    
    with open(slurm_template, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if line.startswith('#SBATCH --job-name='):
            lines[i] = f'#SBATCH --job-name=sofia_run_{env_id}_{episode}_{step}\n'

            if eval_flag:
                lines[i] = f'#SBATCH --job-name=sofia_eval_{env_id}_{episode}_{step}\n'
            
    with open(f'{output_path}/sofia_job.sh', 'w') as f:
        f.writelines(lines)
        
    base_path = os.getcwd()
    
    os.chdir(output_path)
    
    while True:
        result = subprocess.run(['sbatch', 'sofia_job.sh'], 
                                capture_output=True, text=True)
        
        if os.path.exists(f'log.out') and result.returncode == 0:
            break
        else:
            time.sleep(1)
            
    os.chdir(base_path)
    
    if os.path.exists(f'{output_path}/sofia_output_cat.txt'):
        return 0  # Success
    else:
        return 1  # Error
    
def submit_single_job(par_file, slurm_template, output_path, episode, step, job_id=None):
    
    with open(slurm_template, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if line.startswith('#SBATCH --job-name='):
            lines[i] = f'#SBATCH --job-name=sofia_job_{episode}_{step}\n'
        
        if job_id is not None:
            lines[i] = f'#SBATCH --job-name=sofia_job_{episode}_{step}_{job_id}\n'
            
    with open(f'{output_path}/sofia_job.sh', 'w') as f:
        f.writelines(lines)
        
    base_path = os.getcwd()
    
    os.chdir(output_path)
    
    while True:
        result = subprocess.run(['sbatch', 'sofia_job.sh'], 
                                capture_output=True, text=True)
        
        if os.path.exists(f'log.out') and result.returncode == 0:
            break
        else:
            time.sleep(1)
            
    os.chdir(base_path)
    
    if os.path.exists(f'{output_path}/sofia_output_cat.txt'):
        return 0  # Success
    else:
        return 2  # Error
 
def run_sofia(par_file):
    
    path = os.path.dirname(par_file) # path to the directory of the parameter file
    # file = os.path.basename(par_file)
    
    log_file = f'{path}/sofia_output_log.txt'
    
    # Write parameters to log file before running SoFiA
    with open(log_file, 'w') as f:
        f.write("=== SoFiA Parameters ===\n")
        f.write("Parameters:\n")
        
        # Read and write parameter file contents
        with open(par_file, 'r') as par:
            par_contents = par.read()
            f.write(par_contents)
            f.write("\n\n=== SoFiA Output ===\n\n")
            
    sofia_cmd = f'sofia {par_file}'

    with open(log_file, 'a') as f:
        result = subprocess.run(
            sofia_cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
        
def plot_episode_summary(output_path, reward_history, score_history):
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(reward_history)), reward_history)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward History')
    plt.savefig(os.path.join(output_path, 'reward_history.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(score_history)), score_history)
    plt.xlabel('Step')
    plt.ylabel('Score')
    plt.title('Score History')
    plt.savefig(os.path.join(output_path, 'score_history.png'))
    plt.close()
        
def plot_summary(workspace):
    
    recorder = pd.read_csv(os.path.join(workspace, 'reward_recorder.txt'))
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(recorder)), recorder['Score'])
    plt.xlabel('Step')
    plt.ylabel('Score')
    plt.title('Summary')
    plt.savefig(os.path.join(workspace, 'score_history.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(recorder)), recorder['Reward'])
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Summary')
    plt.savefig(os.path.join(workspace, 'reward_history.png'))
    plt.close()
    
    steps_per_episode = recorder.groupby('Episode').size()
    plt.figure(figsize=(10, 5))
    plt.plot(steps_per_episode.index, steps_per_episode.values)
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps')
    plt.title('Steps per Episode')
    plt.savefig(os.path.join(workspace, 'steps_per_episode.png'))
    plt.close()

def custom_serialier(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, object):
        return str(obj)
    else:
        return obj
    
def regularly_plots(scores, score_ratios, rewards, benchmark, output_path):
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    labels = ['Score', 'Score Ratio', 'Reward']
    threshold = [benchmark['score'], benchmark['score_ratio'], None]
    
    for i, values in enumerate([scores, score_ratios, rewards]):
        axs[i].plot(np.arange(len(values)), values)
        axs[i].set_xlabel('Step')
        axs[i].set_ylabel(labels[i])
        axs[i].set_title(labels[i])
        axs[i].set_ylim(0, None)
        
        if threshold[i] is not None:
            axs[i].axhline(y=threshold[i], color='r', linestyle='--')
        
    plt.savefig(os.path.join(output_path, 'regularly_plots.png'))
    plt.close()

def plot_summary_multiple_regions(recorder, num_of_regions, region_names, output_path, benchmarks):
    
    recorder = pd.read_csv(recorder)
    
    for i in range(num_of_regions):
        
        scores = recorder[f'score_{region_names[i]}']
        score_ratios = recorder[f'score_ratio_{region_names[i]}']
        
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        
        axs.plot(np.arange(len(scores)), scores)
        axs.set_xlabel('Step')
        axs.set_ylabel('Score')
        axs.set_title('Score')
        axs.set_ylim(0, None)
        axs.axhline(y=benchmarks[i], color='r', linestyle='--')
        axs.set_title(region_names[i])
        
        plt.savefig(os.path.join(output_path, f'summary_{region_names[i]}.png'))
        plt.close()
    
    
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    
    scores = recorder['total_score']
    
    total_benchmark = sum(benchmarks)
    
    axs.plot(np.arange(len(scores)), scores)
    axs.set_xlabel('Step')
    axs.set_ylabel('Score')
    axs.set_title('Score')
    axs.set_ylim(0, None)
    axs.axhline(y=total_benchmark, color='r', linestyle='--')
    
    plt.savefig(os.path.join(output_path, 'summary_total.png'))
    plt.close()
    
    
def plot_summary_region_no_benchmarks(recorder, num_of_regions, region_names, output_path):
    
    recorder = pd.read_csv(recorder)
    
    for i in range(num_of_regions):
        
        scores = recorder[f'score_{region_names[i]}']
        score_ratios = recorder[f'score_ratio_{region_names[i]}']
        
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        
        axs[0].plot(np.arange(len(scores)), scores)
        axs[0].set_xlabel('Step')
        axs[0].set_ylabel('Score')
        axs[0].set_title('Score')
        axs[0].set_ylim(0, None)
        
        plt.savefig(os.path.join(output_path, f'summary_{region_names[i]}.png'))
        plt.close()
    
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    
    scores = recorder['total_score']
    
    axs.plot(np.arange(len(scores)), scores)
    axs.set_xlabel('Step')
    axs.set_ylabel('Score')
    axs.set_ylim(0, None)
        
    plt.savefig(os.path.join(output_path, 'summary_total.png'))
    plt.close()