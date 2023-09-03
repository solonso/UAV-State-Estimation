import numpy as np


class DroneModel3D:
    def __init__(self):
        self.x_pos = 0
        self.y_pos = 0
        self.z_pos = 0
        self.phi = 0
        self.psi = 0
        self.theta = 0
        self.vel = 0

    def initialise(self, vehicle_params):
        self.x_pos = vehicle_params['initial_x_position']
        self.y_pos = vehicle_params['initial_y_position']
        self.z_pos = vehicle_params['initial_z_position']
        self.vel = vehicle_params['initial_speed']
        self.phi = vehicle_params['initial_phi_heading']
        self.psi = vehicle_params['initial_psi_heading']
        self.theta = vehicle_params['initial_theta_heading']

    def update_vehicle(self, time_step, accel, phi_rate, theta_rate, psi_rate):
        self.vel = self.vel + accel * time_step
        self.phi = self.phi + phi_rate * time_step
        self.theta = self.theta + theta_rate * time_step
        self.psi = self.psi + psi_rate * time_step

        R_x = np.array([[1, 0, 0], [0, np.cos(self.phi), -np.sin(self.phi)], [0, np.sin(self.phi), np.cos(self.phi)]])
        R_y = np.array([[np.cos(self.theta), 0, np.sin(self.theta)], [0, 1, 0], [-np.sin(self.theta), 0, np.cos(self.theta)]])
        R_z = np.array([[np.cos(self.psi), -np.sin(self.psi), 0], [np.sin(self.psi), np.cos(self.psi), 0], [0, 0, 1]])
        R_matrix = np.matmul(R_z, np.matmul(R_y, R_x))

        pos_vel_body = np.array([[self.vel], [self.vel], [self.vel]])
        pos_vel_fixed = np.matmul(R_matrix, pos_vel_body)
        x_vel = pos_vel_fixed[0]
        y_vel = pos_vel_fixed[1]
        z_vel = pos_vel_fixed[2]

        self.x_pos = self.x_pos + x_vel * time_step
        self.y_pos = self.y_pos + y_vel * time_step
        self.z_pos = self.z_pos + z_vel * time_step
        return

    def get_position(self):
        return [self.x_pos, self.y_pos, self.z_pos]

    def get_velocity(self):
        R_x = np.array([[1, 0, 0], [0, np.cos(self.phi), -np.sin(self.phi)], [0, np.sin(self.phi), np.cos(self.phi)]])
        R_y = np.array([[np.cos(self.theta), 0, np.sin(self.theta)], [0, 1, 0], [-np.sin(self.theta), 0, np.cos(self.theta)]])
        R_z = np.array([[np.cos(self.psi), -np.sin(self.psi), 0], [np.sin(self.psi), np.cos(self.psi), 0], [0, 0, 1]])
        R_matrix = np.matmul(R_z, np.matmul(R_y, R_x))

        pos_vel_body = np.array([[self.vel], [self.vel], [self.vel]])
        pos_vel_fixed = np.matmul(R_matrix, pos_vel_body)
        x_vel = pos_vel_fixed[0]
        y_vel = pos_vel_fixed[1]
        z_vel = pos_vel_fixed[2]
        return [x_vel, y_vel, z_vel]

    def get_speed(self):
        return self.vel

    def get_heading(self):
        return self.phi, self.theta, self.psi

