import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from .drone_model_3d import DroneModel3D
import seaborn as sns
sns.set_theme(style="darkgrid")


def run_sim(KalmanFilterModel, sim_option, kf_options):
    dt = sim_option['time_step']
    end_time = sim_option['end_time']
    measurement_rate = sim_option['measurement_rate']
    measurement_noise_std = sim_option['measurement_noise_std']
    motion_type = sim_option['motion_type']
    draw_plots = sim_option['draw_plots']
    draw_animation = sim_option['draw_animation']
    start_at_origin = sim_option['start_at_origin']
    start_at_random_speed = sim_option['start_at_random_speed']
    start_at_random_heading = sim_option['start_at_random_heading']

    initial_x_position = 0
    initial_y_position = 0
    initial_z_position = 0
    if start_at_origin is False:
        initial_x_position = 1000 * (np.random.rand() - 0.5)
        initial_y_position = 1000 * (np.random.rand() - 0.5)
        initial_z_position = 1000 * (np.random.rand() - 0.5)

    initial_speed = 3
    if start_at_random_speed is True:
        initial_speed = np.random.rand() * 5

    initial_phi_heading = 0
    initial_theta_heading = 0
    initial_psi_heading = 0
    if start_at_random_heading is True:
        initial_phi_heading = 1 * (np.random.rand() - 0.5)
        initial_theta_heading = 1 * (np.random.rand() - 0.5)
        initial_psi_heading = 1 * (np.random.rand() - 0.5)

    vehicle_params = {'initial_x_position': initial_x_position,
                      'initial_y_position': initial_y_position,
                      'initial_z_position': initial_z_position,
                      'initial_phi_heading': np.deg2rad(initial_phi_heading),
                      'initial_theta_heading': np.deg2rad(initial_theta_heading),
                      'initial_psi_heading': np.deg2rad(initial_psi_heading),
                      'initial_speed': initial_speed}

    meas_steps = np.ceil((1 / measurement_rate) / dt).astype(int)
    num_steps = np.ceil(end_time / dt).astype(int)

    # Create the Simulation Objects
    vehicle_model = DroneModel3D()
    kalman_filter = KalmanFilterModel()

    # Initialise the Components
    vehicle_model.initialise(vehicle_params)
    kalman_filter.initialise(dt, **kf_options)

    # Save the Initial States
    time_history = np.linspace(0.0, dt * num_steps, num_steps + 1)
    vehicle_position_history = [vehicle_model.get_position()]
    vehicle_velocity_history = [vehicle_model.get_velocity()]
    measurement_history = [None]
    estimated_state_history = [kalman_filter.get_current_state()]
    estimated_covariance_history = [kalman_filter.get_current_covariance()]
    estimated_error_history = [None]

    measurement_innovation_history = [None]
    measurement_innovation_covariance_history = [None]

    # Run the Simulation
    for k in range(1, num_steps + 1):

        # Update the Vehicle Model
        vehicle_model.update_vehicle(dt, np.random.randn() * 2, np.random.randn() * 2, np.random.randn() * 2, np.random.randn() * 2)

        vehicle_position = vehicle_model.get_position()
        vehicle_velocity = vehicle_model.get_velocity()

        # KF Prediction
        kalman_filter.prediction_step()

        # KF Measurement
        measurement = None
        if (k % meas_steps) == 0:
            x_meas = vehicle_position[0] + np.random.randn() * measurement_noise_std
            y_meas = vehicle_position[1] + np.random.randn() * measurement_noise_std
            z_meas = vehicle_position[2] + np.random.randn() * measurement_noise_std
            measurement = np.array([x_meas[0], y_meas[0], z_meas[0]])
            kalman_filter.update_step(measurement)
            measurement_innovation_history.append(kalman_filter.get_last_innovation())
            measurement_innovation_covariance_history.append(kalman_filter.get_last_innovation_covariance())

        # Estimation Error
        estimation_error = None
        estimated_state = kalman_filter.get_current_state()
        if estimated_state is not None:
            estimation_error = [estimated_state[0] - vehicle_position[0],
                                estimated_state[1] - vehicle_position[1],
                                estimated_state[2] - vehicle_position[2],
                                estimated_state[3] - vehicle_velocity[0],
                                estimated_state[4] - vehicle_velocity[1],
                                estimated_state[5] - vehicle_velocity[2]]

        # Save Data
        vehicle_position_history.append(vehicle_model.get_position())
        vehicle_velocity_history.append(vehicle_model.get_velocity())
        measurement_history.append(measurement)
        estimated_state_history.append(kalman_filter.get_current_state())
        estimated_covariance_history.append(kalman_filter.get_current_covariance())
        estimated_error_history.append(estimation_error)

    # Calculate Stats
    x_innov_std = np.std([v[0] for v in measurement_innovation_history if v is not None])
    y_innov_std = np.std([v[1] for v in measurement_innovation_history if v is not None])
    z_innov_std = np.std([v[2] for v in measurement_innovation_history if v is not None])

    pos_mse = np.mean([(v[0] ** 2 + v[1] ** 2) for v in estimated_error_history if v is not None])
    vel_mse = np.mean([(v[2] ** 2 + v[3] ** 2) for v in estimated_error_history if v is not None])
    print('X Position Measurement Innovation Std: {} (m)'.format(x_innov_std))
    print('Y Position Measurement Innovation Std: {} (m)'.format(y_innov_std))
    print('Z Position Measurement Innovation Std: {} (m)'.format(z_innov_std))

    print('Position Mean Squared Error: {} (m)^2'.format(pos_mse))
    print('Velocity Mean Squared Error: {} (m/s)^2'.format(vel_mse))

    if draw_plots is True:
        # ----------------------------------------
        df = pd.DataFrame(columns=['X_pos', 'Y_pos', 'Z_pos',
                                   'X_pos_mes', 'Y_pos_mes', 'Z_pos_mes',
                                   'X_pos_ref', 'Y_pos_ref', 'Z_pos_ref',
                                   # ------------------------------------
                                   'X_vel', 'Y_vel', 'Z_vel',
                                   'X_vel_ref', 'Y_vel_ref', 'Z_vel_ref',
                                   # ------------------------------------
                                   'X_pos_err', 'X_pos_err_upper', 'X_pos_err_lower',
                                   'Y_pos_err', 'Y_pos_err_upper', 'Y_pos_err_lower',
                                   'Z_pos_err', 'Z_pos_err_upper', 'Z_pos_err_lower',
                                   # --------------------------------------
                                   'X_vel_err', 'X_vel_err_upper', 'X_vel_err_lower',
                                   'Y_vel_err', 'Y_vel_err_upper', 'Y_vel_err_lower',
                                   'Z_vel_err', 'Z_vel_err_upper', 'Z_vel_err_lower',
                                   # ---------------------------------------
                                   'X_mes', 'X_mes_upper', 'X_mes_lower',
                                   'Y_mes', 'Y_mes_upper', 'Y_mes_lower',
                                   'Z_mes', 'Z_mes_upper', 'Z_mes_lower',
                                   ]
                          )
        # Plot Analysis
        fig1 = plt.figure(constrained_layout=True)
        fig1_gs = gridspec.GridSpec(ncols=2, nrows=3, figure=fig1)

        fig1_ax1 = fig1.add_subplot(fig1_gs[0, 0])
        fig1_ax2 = fig1.add_subplot(fig1_gs[1, 0])
        fig1_ax3 = fig1.add_subplot(fig1_gs[2, 0])

        fig1_ax4 = fig1.add_subplot(fig1_gs[0, 1])
        fig1_ax5 = fig1.add_subplot(fig1_gs[1, 1])
        fig1_ax6 = fig1.add_subplot(fig1_gs[2, 1])

        fig1_ax1.grid(True)
        fig1_ax2.grid(True)
        fig1_ax3.grid(True)
        fig1_ax4.grid(True)
        fig1_ax5.grid(True)
        fig1_ax6.grid(True)

        fig1_ax3.set_xlabel('t(sec)')
        fig1_ax6.set_xlabel('t(sec)')
        fig1_ax1.set_ylabel('X-Pos(m)')
        fig1_ax2.set_ylabel('Y-Pos(m)')
        fig1_ax3.set_ylabel('Z-Pos(m)')

        # Plot Vehicle State
        fig1_ax1.plot(time_history, [v[0] for v in vehicle_position_history], 'k',label='X Estimate')
        fig1_ax1.legend(loc='upper right',fontsize='medium')
        fig1_ax2.plot(time_history, [v[1] for v in vehicle_position_history], 'k',label='Y Estimate')
        fig1_ax2.legend(loc='upper right',fontsize='medium')
        fig1_ax3.plot(time_history, [v[2] for v in vehicle_position_history], 'k',label='Z Estimate')
        fig1_ax3.legend(loc='upper right',fontsize='medium')
        # add position to table
        df.X_pos = pd.Series([v[0] for v in vehicle_position_history])
        df.Y_pos = pd.Series([v[1] for v in vehicle_position_history])
        df.Z_pos = pd.Series([v[2] for v in vehicle_position_history])
        
        # Plot Estimated States
        time_plot = [t for t, v in zip(time_history, estimated_state_history) if v is not None]
        fig1_ax1.plot(time_plot, [v[0] for v in estimated_state_history if v is not None], 'b',label='X Kalman')
        fig1_ax1.legend(loc='upper right',fontsize='medium')
        fig1_ax2.plot(time_plot, [v[1] for v in estimated_state_history if v is not None], 'b',label='Y Kalman')
        fig1_ax2.legend(loc='upper right',fontsize='medium')
        fig1_ax3.plot(time_plot, [v[2] for v in estimated_state_history if v is not None], 'b',label='Z Kalman')
        fig1_ax3.legend(loc='upper right',fontsize='medium')
        # add position reference to table
        df.X_pos_ref = pd.Series([v[0] for v in estimated_state_history if v is not None])
        df.Y_pos_ref = pd.Series([v[1] for v in estimated_state_history if v is not None])
        df.Z_pos_ref = pd.Series([v[2] for v in estimated_state_history if v is not None])

        # Plot Measurements
        time_plot = [t for t, v in zip(time_history, measurement_history) if v is not None]
        fig1_ax1.plot(time_plot, [v[0] for v in measurement_history if v is not None], 'r',label='X UAV')  # measurements
        fig1_ax1.legend(loc='upper right',fontsize='medium')
        fig1_ax2.plot(time_plot, [v[1] for v in measurement_history if v is not None], 'r',label='Y UAV')
        fig1_ax2.legend(loc='upper right',fontsize='medium')
        fig1_ax3.plot(time_plot, [v[2] for v in measurement_history if v is not None], 'r',label='Z UAV')
        fig1_ax3.legend(loc='upper right',fontsize='medium')
        
        df.X_pos_mes = pd.Series([v[0] for v in measurement_history if v is not None])
        df.Y_pos_mes = pd.Series([v[1] for v in measurement_history if v is not None])
        df.Z_pos_mes = pd.Series([v[2] for v in measurement_history if v is not None])

        # Plot Errors
        time_plot = [t for t, v in zip(time_history, estimated_error_history) if v is not None]
        fig1_ax4.plot(time_plot, [v[0] for v in estimated_error_history if v is not None], 'm',label='X Error')  # position error
        fig1_ax4.legend(loc='upper right',fontsize='medium')
        fig1_ax5.plot(time_plot, [v[1] for v in estimated_error_history if v is not None], 'm',label='Y Error')
        fig1_ax5.legend(loc='upper right',fontsize='medium')
        fig1_ax6.plot(time_plot, [v[2] for v in estimated_error_history if v is not None], 'm',label='Z Error')
        fig1_ax6.legend(loc='upper right',fontsize='medium')
        # add position error to table
        df.X_pos_err = pd.Series([v[0] for v in estimated_error_history if v is not None])
        df.Y_pos_err = pd.Series([v[1] for v in estimated_error_history if v is not None])
        df.Z_pos_err = pd.Series([v[2] for v in estimated_error_history if v is not None])

        time_plot = [t for t, v in zip(time_history, estimated_covariance_history) if v is not None]
        fig1_ax4.plot(time_plot, [3.0 * np.sqrt(v[0][0]) for v in estimated_covariance_history if v is not None], 'g')
        fig1_ax5.plot(time_plot, [3.0 * np.sqrt(v[1][1]) for v in estimated_covariance_history if v is not None], 'g')
        fig1_ax6.plot(time_plot, [3.0 * np.sqrt(v[2][2]) for v in estimated_covariance_history if v is not None], 'g')
        # add position error upper to table
        df.X_pos_err_upper = pd.Series([3.0 * np.sqrt(v[0][0]) for v in estimated_covariance_history if v is not None])
        df.Y_pos_err_upper = pd.Series([3.0 * np.sqrt(v[1][1]) for v in estimated_covariance_history if v is not None])
        df.Z_pos_err_upper = pd.Series([3.0 * np.sqrt(v[2][2]) for v in estimated_covariance_history if v is not None])

        fig1_ax4.plot(time_plot, [-3.0 * np.sqrt(v[0][0]) for v in estimated_covariance_history if v is not None], 'g')
        fig1_ax5.plot(time_plot, [-3.0 * np.sqrt(v[1][1]) for v in estimated_covariance_history if v is not None], 'g')
        fig1_ax6.plot(time_plot, [-3.0 * np.sqrt(v[2][2]) for v in estimated_covariance_history if v is not None], 'g')
        # add position error upper to table
        df.X_pos_err_lower = pd.Series([-3.0 * np.sqrt(v[0][0]) for v in estimated_covariance_history if v is not None])
        df.Y_pos_err_lower = pd.Series([-3.0 * np.sqrt(v[1][1]) for v in estimated_covariance_history if v is not None])
        df.Z_pos_err_lower = pd.Series([-3.0 * np.sqrt(v[2][2]) for v in estimated_covariance_history if v is not None])


        #Velocity
        fig2 = plt.figure(constrained_layout=False)
        fig2_gs = gridspec.GridSpec(ncols=2, nrows=3, figure=fig2)
        
        fig2_ax1 = fig2.add_subplot(fig1_gs[0, 0])
        fig2_ax2 = fig2.add_subplot(fig1_gs[1, 0])
        fig2_ax3 = fig2.add_subplot(fig1_gs[2, 0])

        fig2_ax4 = fig2.add_subplot(fig1_gs[0, 1])
        fig2_ax5 = fig2.add_subplot(fig1_gs[1, 1])
        fig2_ax6 = fig2.add_subplot(fig1_gs[2, 1])

        fig2_ax1.grid(True)
        fig2_ax2.grid(True)
        fig2_ax3.grid(True)
        fig2_ax4.grid(True)
        fig2_ax5.grid(True)
        fig2_ax6.grid(True)

        fig2_ax3.set_xlabel('t(sec)')
        fig2_ax6.set_xlabel('t(sec)')
        fig2_ax1.set_ylabel('X-Vel(m/s)')
        fig2_ax2.set_ylabel('Y-Vel(m/s)')
        fig2_ax3.set_ylabel('Z-Vel(m/s)')
        
        #Velocity
        fig2_ax1.plot(time_history, [v[0] for v in vehicle_velocity_history], 'r',label='X Vel')
        fig2_ax1.legend(loc='upper right',fontsize='medium')
        fig2_ax2.plot(time_history, [v[1] for v in vehicle_velocity_history], 'r',label='Y Vel')
        fig2_ax2.legend(loc='upper right',fontsize='medium')
        fig2_ax3.plot(time_history, [v[2] for v in vehicle_velocity_history], 'r',label='Z Vel')
        fig2_ax3.legend(loc='upper right',fontsize='medium')
        # add velocity to table
        df.X_vel = pd.Series([v[0] for v in vehicle_velocity_history])
        df.Y_vel = pd.Series([v[1] for v in vehicle_velocity_history])
        df.Z_vel = pd.Series([v[2] for v in vehicle_velocity_history])
        
        #Velocity Estimate
        fig2_ax1.plot(time_plot, [v[3] for v in estimated_state_history if v is not None], 'b',label='X Vel Est')
        fig2_ax1.legend(loc='upper right',fontsize='medium')
        fig2_ax2.plot(time_plot, [v[4] for v in estimated_state_history if v is not None], 'b',label='Y Vel Est')
        fig2_ax2.legend(loc='upper right',fontsize='medium')
        fig2_ax3.plot(time_plot, [v[5] for v in estimated_state_history if v is not None], 'b',label='Y Vel Est')
        fig2_ax3.legend(loc='upper right',fontsize='medium')
        # add velocity reference to table
        df.X_vel_ref = pd.Series([v[3] for v in estimated_state_history if v is not None])
        df.Y_vel_ref = pd.Series([v[4] for v in estimated_state_history if v is not None])
        df.Z_vel_ref = pd.Series([v[5] for v in estimated_state_history if v is not None])

        # Plot Measurements
        '''time_plot = [t for t, v in zip(time_history, measurement_history) if v is not None]
        fig2_ax1.plot(time_plot, [v[3] for v in measurement_history if v is not None], 'b',label='X-Vel_Mea_Est')  # measurements
        fig2_ax1.legend(loc='upper right',fontsize='small')
        fig2_ax2.plot(time_plot, [v[4] for v in measurement_history if v is not None], 'b',label='Y-Vel_Mea_Est')
        fig2_ax2.legend(loc='upper right',fontsize='small')
        fig2_ax3.plot(time_plot, [v[5] for v in measurement_history if v is not None], 'b',label='Z-Vel_Mea_Est')
        fig2_ax3.legend(loc='upper right',fontsize='small')
        df.X_vel_mes = pd.Series([v[3] for v in measurement_history if v is not None])
        df.Y_vel_mes = pd.Series([v[4] for v in measurement_history if v is not None])
        df.Z_vel_mes = pd.Series([v[5] for v in measurement_history if v is not None])'''

        #vel Error
        fig2_ax4.plot(time_plot, [v[3] for v in estimated_error_history if v is not None], 'm',label='X Vel Error')  # velocity error
        fig2_ax4.legend(loc='upper right',fontsize='medium')
        fig2_ax5.plot(time_plot, [v[4] for v in estimated_error_history if v is not None], 'm',label='Y Vel Error')
        fig2_ax5.legend(loc='upper right',fontsize='medium')
        fig2_ax6.plot(time_plot, [v[5] for v in estimated_error_history if v is not None], 'm',label='Z Vel Error')
        fig2_ax6.legend(loc='upper right',fontsize='medium')
        # add position error to table
        df.X_vel_err = pd.Series([v[3] for v in estimated_error_history if v is not None])
        df.Y_vel_err = pd.Series([v[4] for v in estimated_error_history if v is not None])
        df.Z_vel_err = pd.Series([v[5] for v in estimated_error_history if v is not None])

        fig2_ax4.plot(time_plot, [3.0 * np.sqrt(v[3][3]) for v in estimated_covariance_history if v is not None], 'g')
        fig2_ax5.plot(time_plot, [3.0 * np.sqrt(v[4][4]) for v in estimated_covariance_history if v is not None], 'g')
        fig2_ax6.plot(time_plot, [3.0 * np.sqrt(v[5][5]) for v in estimated_covariance_history if v is not None], 'g')
        # add position error lower to table
        df.X_vel_err_upper = pd.Series([3.0 * np.sqrt(v[3][3]) for v in estimated_covariance_history if v is not None])
        df.Y_vel_err_upper = pd.Series([3.0 * np.sqrt(v[4][4]) for v in estimated_covariance_history if v is not None])
        df.Z_vel_err_upper = pd.Series([3.0 * np.sqrt(v[5][5]) for v in estimated_covariance_history if v is not None])

        fig2_ax4.plot(time_plot, [-3.0 * np.sqrt(v[3][3]) for v in estimated_covariance_history if v is not None], 'g')
        fig2_ax5.plot(time_plot, [-3.0 * np.sqrt(v[4][4]) for v in estimated_covariance_history if v is not None], 'g')
        fig2_ax6.plot(time_plot, [-3.0 * np.sqrt(v[5][5]) for v in estimated_covariance_history if v is not None], 'g')
        # add position error lower to table
        df.X_vel_err_lower = pd.Series([-3.0 * np.sqrt(v[3][3]) for v in estimated_covariance_history if v is not None])
        df.Y_vel_err_lower = pd.Series([-3.0 * np.sqrt(v[4][4]) for v in estimated_covariance_history if v is not None])
        df.Z_vel_err_lower = pd.Series([-3.0 * np.sqrt(v[5][5]) for v in estimated_covariance_history if v is not None])
        #fig1_ax10.plot(time_plot, [-3.0 * np.sqrt(v[3][3]) for v in estimated_covariance_history if v is not None], 'g')

        fig3 = plt.figure(constrained_layout=False)
        fig3_gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig3)

        fig3_ax1 = fig3.add_subplot(fig2_gs[0, 0])
        fig3_ax2 = fig3.add_subplot(fig2_gs[1, 0])
        fig3_ax3 = fig3.add_subplot(fig2_gs[2, 0])
        fig3_ax1.grid(True)
        fig3_ax2.grid(True)
        fig3_ax3.grid(True)
        fig3_ax3.set_xlabel('t(sec)')
        fig3_ax1.set_ylabel('X inno (m)')
        fig3_ax2.set_ylabel('Y inno (m)')
        fig3_ax3.set_ylabel('Z inno (m)')

        time_plot = [t for t, v in zip(time_history, measurement_innovation_history) if v is not None]
        fig3_ax1.plot(time_plot, [v[0] for v in measurement_innovation_history if v is not None], 'm',label='X-Measure')
        fig3_ax1.legend(loc='upper right',fontsize='medium')
        fig3_ax2.plot(time_plot, [v[1] for v in measurement_innovation_history if v is not None], 'm',label='Y-Measure')
        fig3_ax2.legend(loc='upper right',fontsize='medium')
        fig3_ax3.plot(time_plot, [v[2] for v in measurement_innovation_history if v is not None], 'm',label='Z-Measure')  # measure bound
        fig3_ax3.legend(loc='upper right',fontsize='medium')

        df.X_mes = pd.Series([v[0] for v in measurement_innovation_history if v is not None])
        df.Y_mes = pd.Series([v[1] for v in measurement_innovation_history if v is not None])
        df.Z_mes = pd.Series([v[2] for v in measurement_innovation_history if v is not None])
        
        time_plot = [t for t, v in zip(time_history, measurement_innovation_covariance_history) if v is not None]
        fig3_ax1.plot(time_plot,
                      [1.0 * np.sqrt(v[0][0]) for v in measurement_innovation_covariance_history if v is not None],
                      'g--')
        fig3_ax1.plot(time_plot,
                      [-1.0 * np.sqrt(v[0][0]) for v in measurement_innovation_covariance_history if v is not None],
                      'g--')
        df.X_mes_upper = pd.Series([1.0 * np.sqrt(v[0][0]) for v in measurement_innovation_covariance_history if v is not None])
        df.X_mes_lower = pd.Series([-1.0 * np.sqrt(v[0][0]) for v in measurement_innovation_covariance_history if v is not None])

        fig3_ax2.plot(time_plot,
                      [1.0 * np.sqrt(v[1][1]) for v in measurement_innovation_covariance_history if v is not None],
                      'g--')
        fig3_ax2.plot(time_plot,
                      [-1.0 * np.sqrt(v[1][1]) for v in measurement_innovation_covariance_history if v is not None],
                      'g--')
        df.Y_mes_upper = pd.Series([1.0 * np.sqrt(v[1][1]) for v in measurement_innovation_covariance_history if v is not None])
        df.Y_mes_lower = pd.Series([-1.0 * np.sqrt(v[1][1]) for v in measurement_innovation_covariance_history if v is not None])

        fig3_ax3.plot(time_plot,
                      [1.0 * np.sqrt(v[2][2]) for v in measurement_innovation_covariance_history if v is not None],
                      'g--')
        fig3_ax3.plot(time_plot,
                      [-1.0 * np.sqrt(v[2][2]) for v in measurement_innovation_covariance_history if v is not None],
                      'g--')
        df.Z_mes_upper = pd.Series([1.0 * np.sqrt(v[2][2]) for v in measurement_innovation_covariance_history if v is not None])
        df.Z_mes_lower = pd.Series([-1.0 * np.sqrt(v[2][2]) for v in measurement_innovation_covariance_history if v is not None])

        df.to_csv('KALMAN-DATA.csv', index=False)

    if draw_animation is True:

        # Plot Animation
        fig2 = plt.figure(constrained_layout=True)
        fig_ax2 = fig2.add_subplot(111, title='3D Position',
                                   aspect='equal')  # , autoscale_on=True, xlim=(0, 1000), ylim=(0, 1000))
        fig_ax2.set_xlabel('X Position (m)')
        fig_ax2.set_ylabel('Y Position (m)')
        # fig_ax2.set_zlabel('Z Position (m)')

        fig_ax2.grid(True)
        vehicle_plot, = fig_ax2.plot([], [], 'bo-', lw=1)
        meas_plot, = fig_ax2.plot([], [], '+k')
        estimate_plot, = fig_ax2.plot([], [], 'ro-')
        estimate_var_plot, = fig_ax2.plot([], [], 'g-')

        def update_plot(i):
            # Plot Vehicle
            vehicle_plot.set_data(vehicle_position_history[i])

            # Plot Measurements
            x_data = [meas[0] for meas in measurement_history[1:i] if meas is not None]
            y_data = [meas[1] for meas in measurement_history[1:i] if meas is not None]
            z_data = [meas[2] for meas in measurement_history[1:i] if meas is not None]
            meas_plot.set_data(x_data, y_data, z_data)

            # Plot Estimates
            if estimated_state_history[i] is not None:
                est_xpos = estimated_state_history[i][0]
                est_ypos = estimated_state_history[i][1]
                est_zpos = estimated_state_history[i][2]
                estimate_plot.set_data(est_xpos, est_ypos)
                if estimated_covariance_history[i] is not None:
                    cov = estimated_covariance_history[i]
                    cov_mat = np.array([[cov[0][0], cov[0][1]], [cov[1][0], cov[1][1]]])
                    U, S, V = np.linalg.svd(cov_mat)
                    theta = np.linspace(0, 2 * np.pi, 100)
                    theta_mat = np.array([np.cos(theta), np.sin(theta)])
                    D = np.matmul(np.matmul(U, np.diag(3.0 * np.sqrt(S))), theta_mat)
                    estimate_var_plot.set_data([x + est_xpos for x in D[0]], [y + est_ypos for y in D[1]])

            fig_ax2.relim()
            fig_ax2.autoscale_view()

            return vehicle_plot, meas_plot, estimate_plot, estimate_var_plot

        # # Create the Animation
        plot_animation = animation.FuncAnimation(fig2, update_plot, frames=range(0, num_steps, 50), interval=1,
                                                 repeat=False, blit=False)

    # Show Animation
    plt.show()

    return (pos_mse, vel_mse)
