import numpy as np
from kfsims.drone_tracker3d import run_sim
from kfsims.kftracker3d import KalmanFilterModel

# Simulation Options
sim_options = {'time_step': 0.02,
               'end_time': 200,
               'measurement_rate': 1.5,
               'measurement_noise_std': 10,
               'motion_type': 'circle',
               'start_at_origin': True,
               'start_at_random_speed': False,
               'start_at_random_heading': False,
               'draw_plots': True,
               'draw_animation': False} 

kf_options = {'accel_std':0.5, # Q Matrix Param
              'meas_std':10, # R Matrix  
              'init_on_measurement':True}

# Run the Simulation
run_sim(KalmanFilterModel, sim_options, kf_options)
