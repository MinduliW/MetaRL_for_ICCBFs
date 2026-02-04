import sys

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
import time
import os
import warnings
import shutil
import datetime
import gc
import pandas as pd
import copy
from stable_baselines3.common.policies import ActorCriticPolicy
from datetime import datetime
from stable_baselines3 import PPO
from scipy import integrate
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Type
from stable_baselines3.common.callbacks import BaseCallback
from scipy.io import savemat
import scipy as sp
import math
from math import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from typing import Callable
from iccbfs import ICCBF
from daceypy import DA  # DACEyPy DA type (Differential Algebra)

# import stable_baselines3
from stable_baselines3 import PPO

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

import torch
from torch.optim import Adam
from torch.nn.modules import activation

import multiprocessing


from RLCBF import RLCBFcontrol



def make_env(env_instance, rank, model=None, seed=0):
    def _init_():
        env = env_instance
        
        DA.init(int(4), int(2))
        env.action_space.seed(seed + rank)
        env.model = model 
        env = Monitor(env)
        return env
    return _init_

class ConstantSchedule:
    def __init__(self, initial_value: float):
        self.initial_value = initial_value

    def __call__(self, progress_remaining: float) -> float:
        return self.initial_value


class LinearSchedule:
    def __init__(self, initial_value: float, min_value: float):
        self.initial_value = initial_value
        self.min_value = min_value

    def __call__(self, progress_remaining: float) -> float:
        return max(progress_remaining * self.initial_value, self.min_value)


def is_debug_mode():
    return sys.gettrace() is not None


if __name__ == "__main__":
    
    # Clear command line3
    os.system('cls' if os.name == 'nt' else 'clear')

    multiprocessing.freeze_support()

    # Set the warning filter to "ignore" for UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Collect garbage
    gc.collect() 
    
    #### DEFINE SIMULATION ####
    trainON         = False      # TRAINING flag
    trainLoad       = True      # Load a trained model to use in Training
    modelLoad       = True      # Load a trained model to use in Validation

    totalEpisodes   = int(100*100*20); # *150
    
    if is_debug_mode():
        num_env = 1
    else:
        num_env = 64

   
    #num_env = 1
    if not trainON: num_env = 1
    if trainON: defaultActions = False

    dt = 0.1
    stoch_env = RLCBFcontrol(dt = dt, deterministic=False)
    deter_env = RLCBFcontrol(dt = dt,deterministic=True)
    

    
    #Create Parallel Environments
    if num_env > 1:
        train_env = SubprocVecEnv([make_env(stoch_env, rank=i) for i in range(num_env)])
    else:
        train_env = stoch_env
        
    approxEpisode   = 200
  
    total_timesteps = int(totalEpisodes*approxEpisode)
    learning_rate   = 1e-4
    lr_type         = 'C' # C = constant, W=warmup, D=decreasing
    gamma           = 0.999
    gae_lambda      = 0.99
    clip_range      = 0.1
    ent_coef        = 0.01 #.001 #0.01# 0.01->0.1
    target_kl=0.02 
    n_epochs        = 10# 10
    use_sde         = False
    
    
    # LR Schedules
    if lr_type == 'D':
        lr_schedule = LinearSchedule(learning_rate, learning_rate / 100)
    else:
        lr_schedule = ConstantSchedule(learning_rate)

    custom_objects = {};
    
    # Stcohasticity
    std             = 0.2
    std_tensor      = torch.tensor(std)

    layers          = 3
    nodes           = 64
    activation_fn   = activation.Tanh
    
   
    policy_kwargs   = dict(activation_fn=activation_fn, ortho_init=True, log_std_init=np.log(std_tensor), share_features_extractor=False, 
                           optimizer_class=Adam,net_arch=[dict(pi=[nodes]*layers, vf=[nodes]*layers)])
    
    
    batch_size      = 512
    n_steps         = 256
    
    MLPtype = 'PPO'

    
    # ---- build training name ----
    act_name  = getattr(activation_fn, "__name__", activation_fn.__class__.__name__)
    time_str  = datetime.now().strftime("%Y%m%d_%H%M%S")

    arch_str   = f"{MLPtype}_L{layers}_N{nodes}_{act_name}"
    hyper_str  = (
        f"lr{learning_rate:g}"
        f"_g{gamma:.3f}"
        f"_gae{gae_lambda:.3f}"
        f"_ent{ent_coef:g}"
    )

    training_name = f"MetaCNNCruiseControlMargin_{arch_str}_{hyper_str}"

    log_dir       = f"TrainedModels/{training_name}"
    print("Logging to", log_dir)
        
    # Tensorboard
    tensorboard_log_dir_source = 'TrainedModels/'
    os.makedirs(tensorboard_log_dir_source, exist_ok=True)
    

    eval_log_dir = os.path.join(tensorboard_log_dir_source, training_name)
    os.makedirs(eval_log_dir, exist_ok=True)
    if trainON and os.path.exists(tensorboard_log_dir_source+training_name+'_1'): shutil.rmtree('TrainedModels/'+training_name+'_1')
    # Plot storage
    plot_log_dir = os.path.join('Results/', training_name)
    os.makedirs(plot_log_dir, exist_ok=True)
    
    # Define Model
    if MLPtype == 'PPO':
         model = PPO("MlpPolicy", train_env, verbose=1, 
                    learning_rate=lr_schedule, tensorboard_log=tensorboard_log_dir_source, normalize_advantage = True,
                    n_steps=n_steps,batch_size=batch_size,n_epochs=n_epochs,gamma=gamma,use_sde=use_sde,
                    gae_lambda=gae_lambda,clip_range=clip_range, ent_coef=ent_coef, policy_kwargs=policy_kwargs,target_kl=target_kl)

    else:
       model = RecurrentPPO("MlpLstmPolicy", train_env, verbose=1, 
                    learning_rate=lr_schedule, tensorboard_log=tensorboard_log_dir_source, normalize_advantage = True,
                    n_steps=n_steps,batch_size=batch_size,n_epochs=n_epochs,gamma=gamma,use_sde=use_sde,
                    gae_lambda=gae_lambda,clip_range=clip_range, ent_coef=ent_coef, policy_kwargs=policy_kwargs)
 
    
    if trainLoad:
        
        model = PPO.load(os.path.join(eval_log_dir, 'best_model.zip'),env=train_env,tensorboard_log=tensorboard_log_dir_source,custom_objects=custom_objects)
        
        
    
    if trainON:
        print('Training: ',total_timesteps, 'Batch: ',batch_size * num_env,' n_steps: ',n_steps * num_env)

        # Call Back
        callbacks = EvalCallback(eval_env=deter_env,
                        n_eval_episodes=10, eval_freq=n_steps, 
                        best_model_save_path=eval_log_dir,
                        log_path=eval_log_dir,
                        deterministic=True, verbose=1)

      
        # Train Model
        print("--- STARTING LEARNING ---")
        model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name=training_name, progress_bar=True)
        print("--- DONE LEARNING ---")
        
        # Save and Load Model
        model.save(os.path.join(eval_log_dir, 'final_model.zip'))
        del model # remove to demonstrate saving and loading
    
