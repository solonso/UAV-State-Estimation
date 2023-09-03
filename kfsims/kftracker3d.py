import numpy as np
from .kfmodels import KalmanFilterBase

# Kalman Filter Model
class KalmanFilterModel(KalmanFilterBase):
    
    def initialise(self, time_step, accel_std, meas_std, init_on_measurement=False, init_pos_std = 3, init_vel_std = 0.3):
        dt = time_step
        
        # Set Model F and H Matrices
        self.F = np.array([[1,0,0,dt,0,0],
                           [0,1,0,0,dt,0],
                           [0,0,1,0,0,dt],
                           [0,0,0,1,0,0],
                           [0,0,0,0,1,0],
                           [0,0,0,0,0,1]])

        self.H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])

        # Set R and Q Matrices
        self.Q = np.diag(np.array([(0.5*dt*dt),(0.5*dt*dt), (0.5*dt*dt),dt,dt,dt]) * (accel_std*accel_std))
        self.R = np.diag([meas_std*meas_std, meas_std*meas_std, meas_std*meas_std])

        # Set Initial State and Covariance 
        if init_on_measurement is False:
            self.state = np.array([0,0,0,0,0,0]) # Assume we are at zero position and velocity
            self.covariance = np.diag(np.array([init_pos_std*init_pos_std, init_pos_std*init_pos_std,init_pos_std*init_pos_std,
                                                init_vel_std*init_vel_std, init_pos_std*init_pos_std, init_pos_std*init_pos_std
                                                ]))
        
        return
    
    def prediction_step(self):
        # Make Sure Filter is Initialised
        if self.state is not None:
            x = self.state
            P = self.covariance

            # Calculate Kalman Filter Prediction
            x_predict = np.matmul(self.F, x) 
            P_predict = np.matmul(self.F, np.matmul(P, np.transpose(self.F))) + self.Q

            # Save Predicted State
            self.state = x_predict
            self.covariance = P_predict

        return

    def update_step(self, measurement):

        # Make Sure Filter is Initialised
        if self.state is not None and self.covariance is not None:
            x = self.state
            P = self.covariance
            H = self.H
            R = self.R

            # Calculate Kalman Filter Update
            z = np.array([measurement[0],measurement[1], measurement[2]])
            z_hat = np.matmul(H, x)

            y = z - z_hat
            S = np.matmul(H,np.matmul(P,np.transpose(H))) + R
            K = np.matmul(P,np.matmul(np.transpose(H),np.linalg.inv(S)))

            x_update = x + np.matmul(K, y)
            P_update = np.matmul( (np.eye(6) - np.matmul(K,H)), P)

            # Save Updated State
            self.innovation = y
            self.innovation_covariance = S
            self.state = x_update
            self.covariance = P_update

        else:

            # Set Initial State and Covariance 
            self.state = np.array([measurement[0],measurement[1],measurement[2],0,0,0])
            self.covariance = np.diag(np.array([self.R[0,0],self.R[1,1],self.R[2,2],10,10,10])) # Assume we don't know our velocity

        return 
