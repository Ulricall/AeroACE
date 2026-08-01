import torch
import numpy as np
import random
import json
import rowan
from tqdm import tqdm
import pkg_resources
import copy

DEFAULT_PARAMETER_FILE = pkg_resources.resource_filename(__name__, 'params/quadrotor.json')

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def readparamfile(filename, params=None):
    if params is None:
        params = {}
    with open(filename) as file:
        params.update(json.load(file))
    return params

class Quadrotor():
    def __init__(self, paramsfile=DEFAULT_PARAMETER_FILE, **kwargs):
        self.params = readparamfile(paramsfile)
        self.params.update(kwargs)
        self.B = np.array([self.params['C_T'] * np.ones(4), 
                           self.params['C_T'] * self.params['l_arm'] * np.array([-1., -1., 1., 1.]),
                           self.params['C_T'] * self.params['l_arm'] * np.array([-1., 1., 1., -1.]),
                           self.params['C_q'] * np.array([-1., 1., -1., 1.])])
        self.params['J'] = np.array(self.params['J'])
        self.Jinv = np.linalg.inv(self.params['J'])
        l_arm = self.params['l_arm']
        h = self.params['h']
        self.r_arms = np.array(((l_arm, -l_arm, h),
                                (-l_arm, -l_arm, h),
                                (-l_arm, l_arm, h),
                                (l_arm, l_arm, h)))
        self.Vwind = np.zeros(3)
        self.last_wind_force = np.zeros(3)
        self.state = 'train'
        # t_last_wind_update

    def get_wind_v(self, t):
        dt = t - self.t_last_wind_update
        if dt > self.params['wind_update_period']:
            self.Vwind += self.VwindList[self.wind_count] * dt
            self.wind_count += 1
            self.t_last_wind_update = t
        self.Vwind = np.clip(self.Vwind, -25, 25)
        return self.Vwind
    
    def get_wind_effect(self, X, Z, t):
        Vwind = self.get_wind_v(t)
        v = X[7:10]
        R_wtob = rowan.to_matrix(X[3:7]).transpose()
        Vinf = Vwind - v
        Vinf_B = R_wtob @ Vinf
        Vz_B = np.array((0., 0., Vinf_B[2]))  
        Vs_B = Vinf_B - Vz_B
        if np.linalg.norm(Vs_B) > 1e-4:
            aoa = np.arcsin(np.linalg.norm(Vz_B)/np.linalg.norm(Vinf_B))
            n = np.sqrt(Z / self.params['C_T']) 
            Fs_per_prop = self.params['C_s'] * self.params['rho'] * (n ** self.params['k1']) \
                     * (np.linalg.norm(Vinf) ** (2 - self.params['k1'])) * (self.params['D'] \
                     ** (2 + self.params['k1'])) * ((np.pi / 2) ** 2 - aoa ** 2) \
                     * (aoa + self.params['k2'])
            Fs_B = (Vs_B/np.linalg.norm(Vs_B)) * sum(Fs_per_prop)

            tau_s = np.zeros(3)
            for i in range(4):
                tau_s += np.cross(self.r_arms[i], (Vs_B/np.linalg.norm(Vs_B)) * Fs_per_prop[i])
        else:
            Fs_B = np.zeros(3)
            tau_s = np.zeros(3)
        Fs = R_wtob.transpose() @ Fs_B
        return Fs, tau_s
    
    def f(self, X, Z, t, test=False):
        p = X[0:3]
        X[3:7] = rowan.normalize(X[3:7])
        q = X[3:7]
        R = rowan.to_matrix(q)
        v = X[7:10]
        w = X[10:]

        #print("Z", Z)
        T, *tau_mec = self.B @ (Z ** 2)
        Xdot = np.empty(13)
        Xdot[0:3] = v
        Xdot[3:7] = rowan.calculus.derivative(q, w)

        F_mec = np.empty(3)
        F_mec = T * (R @ np.array([0., 0., 1.])) - np.array([0., 0., self.params['g']*self.params['m']])
        F_wind, tau_wind = self.get_wind_effect(X, Z, t)
        self.last_wind_force = F_wind.copy()
        Xdot[7:10] = (F_mec + F_wind) / self.params['m']
        Xdot[10:] = np.linalg.solve(self.params['J'], np.cross(self.params['J'] @ w, w) + tau_mec + tau_wind)
        return Xdot
    
    def update_motor_speed(self, u, dt, Z=None):
        if Z is None:
            Z = u
        else:
            alpha_m = 1 - np.exp(-self.params['w_m'] * dt)
            Z = alpha_m*u + (1 - alpha_m) * Z
        return np.maximum(np.minimum(Z, self.params['motor_max_speed']), self.params['motor_min_speed'])

    def step(self, X, u, t, dt, Z=None):
        if self.params['integration_method'] == 'rk4':
            # RK4 method
            Z = self.update_motor_speed(Z=Z, u=u, dt=0.0)
            k1 = dt * self.f(X, Z, t)
            Z = self.update_motor_speed(Z=Z, u=u, dt=dt/2)
            k2 = dt * self.f(X + k1/2, Z, t+dt/2)
            k3 = dt * self.f(X + k2/2, Z, t+dt/2)
            Z = self.update_motor_speed(Z=Z, u=u, dt=dt/2)
            k4 = dt * self.f(X + k3, Z, t+dt)
            
            Xdot = (k1 + 2*k2 + 2*k3 + k4)/6 / dt
            X = X + (k1 + 2*k2 + 2*k3 + k4)/6
        elif self.params['integration_method'] == 'euler':
            Z = self.update_motor_speed(Z=Z, u=u, dt=dt)
            Xdot = self.f(X, Z, t)
            X = X + dt * Xdot
        else:
            raise NotImplementedError
        X[3:7] = rowan.normalize(X[3:7])
        
        return (X, t+dt, Z, Xdot)
    
    def runiter(self, trajectory, controller, X0=None, Z0=None):
        X = np.zeros(13)
        if trajectory.name == 'terrain':
            pd, vd, ad = trajectory(0)
            X[2] = pd[2]
        X[3] = 1.
        if Z0 is None:
            hover_motor_speed = np.sqrt(self.params['m'] * self.params['g'] / (4 * self.params['C_T']))
            Z = hover_motor_speed * np.ones(4)
        else:
            Z = Z0
        
        t = self.params['t_start'] # simulation time
        t_posctrl = t_attctrl = t_angratectrl = t_readout = -0.0
        logentry = {}
        X_meas = X
        imu_meas = np.zeros(3)
        if (self.state == 'test'): t_stop = self.params['test_t_stop'] 
        elif self.state == 'train': t_stop = self.params['train_t_stop']
        else: t_stop = self.params['train_t_stop_1']
        while t < t_stop:
            if t >= t_posctrl:
                pd, vd, ad = trajectory(t)
                wind_force, wind_force_tau = self.get_wind_effect(X, Z, t)
                wind_gt_for_controller = (
                    wind_force
                    if bool(self.params.get('expose_wind_gt_to_controller', True))
                    else None
                )
                T_sp, q_sp = controller.position(X=X_meas, Z=Z, imu=imu_meas, pd=pd, vd=vd, ad=ad, \
                    last_wind_update=self.t_last_wind_update, t=t, wind_gt=wind_gt_for_controller)
                t_posctrl += controller.params['dt_posctrl']
            if t >= t_attctrl:
                w_sp = controller.attitude(q=X[3:7], q_sp=q_sp)
                t_attctrl += controller.params['dt_attctrl']
            if t >= t_angratectrl:
                torque_sp = controller.angrate(w=X[10:], w_sp=w_sp, dt=controller.params['dt_angratectrl'])
                u = controller.mixer(torque_sp=torque_sp, T_sp=T_sp)
                t_angratectrl += controller.params['dt_angratectrl']
            (X, t, Z, Xdot) = self.step(X=X, u=u, t=t, dt=self.params['dt'], Z=Z)
            X_meas = X
            X_meas[:3] += np.random.normal(scale=self.params['X_meas_noise'])
            imu_noise = np.random.normal(scale=self.params['imu_meas_noise'], size=3)
            outlier_probability = float(self.params.get('imu_outlier_probability', 0.0))
            if outlier_probability > 0.0 and np.random.random() < outlier_probability:
                outlier_scale = float(self.params.get('imu_outlier_scale', 0.0))
                imu_noise += np.random.normal(scale=outlier_scale, size=3)
            imu_meas = Xdot[7:10] + imu_noise

            if t>=t_readout:
                logentry['X'] = X
                logentry['pd'] = pd
                logentry['u'] = u
                logentry['wind_force'] = self.last_wind_force
                logentry['wind_velocity'] = np.asarray(self.Vwind, dtype=float).copy()
                t_readout += self.params['dt_readout']
                yield copy.deepcopy(logentry)
        
    def reset_status(self, wind_list=None):
        self.t_last_wind_update = 0
        self.wind_count = 0
        self.VwindList = wind_list
        self.Vwind = np.zeros(3)
        self.last_wind_force = np.zeros(3)

    def run(self, controller, trajectory=None, seed=None, wind_velocity_list=None, 
            reset_control=True, Name=""):
        self.name = Name
        self.reset_status(wind_velocity_list)
        if (reset_control):
            controller.reset_controller()
        if (self.state == 'test'): t_stop = self.params['test_t_stop'] 
        elif self.state == 'train': t_stop = self.params['train_t_stop']
        else: t_stop = self.params['train_t_stop_1']
        log = list(tqdm(
            self.runiter(trajectory=trajectory, controller=controller),
            total=(t_stop-self.params['t_start'])/self.params['dt_readout'],
            disable=bool(self.params.get('disable_progress', False)),
        ))
        log2 = {k: np.array([logentry[k] for logentry in log]) for k in log[0]}
        return log2
