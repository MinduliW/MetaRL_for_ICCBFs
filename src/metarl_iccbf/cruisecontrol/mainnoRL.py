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


from CBFnoRL import RLCBFcontrol



def make_env(env_instance, rank, model=None, seed=0):
    def _init_():
        env = env_instance
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
        num_env = 20 

   
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

    training_name = f"MetaCNNCruiseControl_{arch_str}_{hyper_str}"
    # training_name = 'NoiseCruiseControl_PPO_L3_N64_Tanh_lr0.0001_g0.995_gae0.990_ent0.01_20251201_214227'
    # training_name = 'CruiseControl_PPO_L4_N64_Tanh_lr5e-05_g0.990_gae0.950_ent0.01_20251201_153135'
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
    
    if modelLoad:
        try:
            # Load the best one
            print('Loading Best Model...')
            print(eval_log_dir)
            model = PPO.load(os.path.join(eval_log_dir, 'best_model'),env=deter_env,tensorboard_log=tensorboard_log_dir_source)
        except:
            print('Unable to load model')
        
    # Deterministic Case
    ('Validating...')
   
    TOF = 40.0
    lenvec = int(TOF / dt) 
    
    points1 = [] # deter_env.getLinePoints(100)      # shape (N1, 2)
    points = deter_env.getValidPoints()        # shape (N2, 2)

    nSamples = points.shape[0]

    # idx = np.linspace(0, nSamples - 1, 10, dtype=int)
    # points = points[idx]
    # nSamples = 10; 



    print(nSamples)
    # nSamples = 5

    nobs = 2
    tvec_full = np.arange(0, TOF , dt)

    # Data buffers
    uOpts = np.zeros((nSamples, lenvec, 1))
    observations = np.zeros((nSamples, lenvec, 2))
    uOptmag = np.zeros((nSamples, lenvec))  # magnitude at each timestep
    uTotal = np.zeros((nSamples, 1)) 
    actionStore =  np.zeros((nSamples, lenvec, 4))
    rewards = np.zeros((nSamples, 1))
    hs = np.zeros((nSamples, lenvec, 1))
    hLearnt = np.zeros((nSamples, lenvec, 1))
    Vs = np.zeros((nSamples, lenvec, 1))
    comptimes    = np.zeros((nSamples, lenvec))
      
    for i in range(nSamples):
        print(i)
        t = 0.0
        step = 0

        obs, _ = deter_env.reset(postProcess=False)
        
        
        deter_env.x0 = points[i, :]
        obs = deter_env.scaleObservation(deter_env.x0)
        

        while t <= TOF and step < lenvec:
            step += 1

            action, _ = model.predict(obs, deterministic=True)
            
            t0 = time.perf_counter()
            obs, reward_temp, done, _, _ = deter_env.step(action)
            comp_dt = time.perf_counter() - t0
            comptimes[i, step-1] = comp_dt   # s
                

            
            observations[i, step-1, :] = deter_env.x0
            actionStore[i, step-1, :] = action
            uOpts[i, step-1, :] = deter_env.u
            mag = np.linalg.norm(deter_env.u)
            uOptmag[i, step-1] = mag
          
            hs[i, step-1, :] = deter_env.x0[0] - 1.8*deter_env.x0[1]
            
            # Lgb11 = deter_env.Lgb1_1_func(deter_env.x0[0],deter_env.x0[1])
     
            # if Lgb11 > 0.0:
            #     u1inf = -deter_env.ulim
            # else: 
            #     u1inf = deter_env.ulim
                    


            # hICCBF = deter_env.b2_func(deter_env.x0[0],deter_env.x0[1],u1inf)

            # hLearnt[i, step-1, :] = action[0]*hICCBF
            Vs[i, step-1, :]  = (deter_env.x0[1] - deter_env.vmax) ** 2


            t += dt
            
        uTotal[i] = np.sum(uOptmag[i, :step]) * dt

    print(f"Completed {nSamples} trajectories")
    
    print("Mean:" ,np.mean(uTotal[1:]))
    
    quartiles = np.percentile(uTotal[1:], [25, 50, 75])
    q1, q2, q3 = quartiles

    print("Q1 (25th percentile):", q1)
    print("Q2 (median):", q2)
    print("Q3 (75th percentile):", q3)


    
    savemat("NoiseMetaICCBF.mat", {
    "uOpts": uOpts,
    "states": observations,
    "rewards": rewards,
    "hs": hs,
    "hLearnt": hLearnt,
    "Vs": Vs,
    "actionStore": actionStore, 
    "uTotal": uTotal,
    "comptimes":    comptimes, 
    "tvec_full": tvec_full})



    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    # axs = axs.flatten()

    colors = plt.cm.viridis(np.linspace(0, 1, nSamples))

    # 1. x(t) vs v(t)
    for i in range(nSamples):
        axs[0, 0].plot(observations[i, :, 0], observations[i, :, 1], color=colors[i], alpha=0.8)
    axs[0, 0].set_title('$x(t)$ vs $v(t)$')
    axs[0, 0].set_xlabel('$x(t)$ (m)')
    axs[0, 0].set_ylabel('$v(t)$ (m/s)')
    axs[0, 0].grid(True)

    # Add reference line x = 1.8 * v
    v_line = np.linspace(0, 20, 300)
    x_line = 1.8 * v_line
    axs[0, 0].plot(x_line, v_line, 'r--', label='$x = 1.8 v$')
    axs[0, 0].legend()



    for i in range(nSamples):
        axs[0, 1].plot(tvec_full, uOpts[i, :, 0], color=colors[i], alpha=0.8)
    axs[0, 1].set_title('$u(N)$ vs Time')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('$u(N)$')
    axs[0, 1].grid(True)
    
        # 3. V(t) vs Time
    for i in range(nSamples):
        axs[1,0].plot(tvec_full, Vs[i, :, 0], color=colors[i], alpha=0.8)
    axs[1,0].set_title('$V(t)$ vs Time')
    axs[1,0].set_xlabel('Time (s)')
    axs[1,0].set_ylabel('$V(t)$')
    axs[1,0].grid(True)
    

    # 2. h(t) vs Time
    for i in range(nSamples):
        axs[1,1].plot(tvec_full, hs[i, :, 0], color=colors[i], alpha=0.8)
    axs[1,1].set_title('$h(t)$ original vs Time')
    axs[1,1].set_xlabel('Time (s)')
    axs[1,1].set_ylabel('$h(t)$')
    axs[1,1].grid(True)

    for i in range(nSamples):
        axs[2,0].plot(tvec_full, (observations[i, :, 1]), color=colors[i], alpha=0.8)
    axs[2,0].set_title('Velocity magnitude over time')
    axs[2,0].set_xlabel('Time (s)')
    axs[2,0].set_ylabel('$velocity(m/s)$')
    axs[2,0].grid(True)

    for i in range(nSamples):
        axs[2,1].plot(tvec_full, (observations[i, :, 0]), color=colors[i], alpha=0.8)
    axs[2,1].set_title('Position magnitude over time')
    axs[2,1].set_xlabel('Time (s)')
    axs[2,1].set_ylabel('$Position (m)$')
    axs[2,1].grid(True)





    plt.tight_layout()

    plt.show()
    
    TOF = 20 
    
    tvec = np.arange(0, 20, dt);
    
    t = 0.0
    stoch_env.x0 = np.array([stoch_env.xinit ,stoch_env.vinit])  
    obs = stoch_env.scaleObservation(stoch_env.x0)
    step =0
    observations = np.zeros(( len(tvec), 2))
    hs = np.zeros(( len(tvec), 1));
    uOpts = np.zeros(( len(tvec), 1));
    
    # Define ranges for x1 and x2
    x1_range = np.linspace(0, 100, 100)  # e.g. position
    x2_range = np.linspace(0, 30, 100)   # e.g. velocity

    X1, X2 = np.meshgrid(x1_range, x2_range)
    H = np.zeros_like(X1)

    # Compute h(x1, x2) over the grid
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = np.array([X1[i, j], X2[i, j]])  # state [x1, x2]

            # Scale observation as expected by the model
            obs_scaled = stoch_env.scaleObservation(x)
            state_tensor = torch.tensor(obs_scaled, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action_tensor, _, _ = stoch_env.model.policy.forward(state_tensor, deterministic=True)
                h_tensor = action_tensor * 70  # same as used in main sim
                h_val = h_tensor[0].item()
                
                h_val =  x[0] - 1.8 * x[1]-h_val
                
                if  x[0] - 1.8 * x[1] < 0 and h_val > 0:
                    h_val = -h_val

            H[i, j] = h_val

    # Plot contour of h

    plt.figure(figsize=(10, 6))

    # Filled contour plot of h(x1, x2)
    contour = plt.contourf(X1, X2, H, levels=50, cmap='viridis')
    plt.colorbar(contour)

    # Add the h = 0 contour line
    zero_contour = plt.contour(X1, X2, H, levels=[0], colors='red', linewidths=2)
    plt.clabel(zero_contour, fmt='h=0', colors='red')

    # Overlay the line x0 = 1.8 * x1 → x1 = x-axis, x0 = vertical axis
    x2_line = np.linspace(X2.min(), X2.max(), 300)
    x1_line = 1.8 * x2_line
    plt.plot(x1_line, x2_line, 'w--', label='$x_1 = 1.8 x_2$')  # dashed white line

    # Labels and title
    plt.title('Contour Plot of $h(x_1, x_2)$ with $x_1 = 1.8 x_2$ Line')
    plt.xlabel('$x_1$ (m)')
    plt.ylabel('$x_2$ (m/s)')
    plt.grid(True)
    plt.legend()
    plt.show()