import sys
sys.path.append('/opt/homebrew/lib/python3.12/site-packages')
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages')
sys.path.append('/Users/minduli/miniconda3/lib/python3.12/site-packages/')

import os
import warnings
import shutil
import datetime
import gc
import pandas as pd
import copy

from scipy import integrate
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Type

# from daceypy import DA, array,  RK,integrator
# from daceypy.op import cos, sin, sqr, sqrt, vnorm

import math
from math import *

import numpy as np
from typing import Callable

from scipy.linalg import solve_discrete_are
from scipy.linalg import solve_continuous_are
from scipy.linalg import solve_continuous_lyapunov
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

import torch
from torch.optim import Adam
from torch.nn.modules import activation

# from ya import YA

import multiprocessing


class supportFcn:
    
    def __init__(self):
        self.muUnScl = 3.98600435436e14;  # km^3 / s^2
        self.r_scale = 6.378136300000000e6  # km / LU  = 1 Re
        self.v_scale = np.sqrt(self.muUnScl / self.r_scale)  # km/s / LU/TU
        self.t_scale = self.r_scale / self.v_scale  # s / TU
        self.mu      = self.muUnScl / (self.r_scale ** 3) * (self.t_scale**2) 
        
    