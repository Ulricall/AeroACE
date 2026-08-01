import torch
import numpy as np
import random
import pkg_resources
import rowan
import json
import time
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
from powerformer_model import PowerFormerForceModel
from pi_transformer_model import (
    PiTransformerForceModel,
    symmetric_kl_divergence,
    prior_regularization_terms,
)
from pinnsformer_model import PINNsFormerDynamicsModel
from cluster_causal_attention_model import ClusterCausalForceModel
from causal_transformer_uav_model import CausalTransformerForceModel
from aeroace_deployment_signal import residual_force_from_onboard

torch.set_default_tensor_type('torch.DoubleTensor')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

device = "cpu"

DEFAULT_CONTROL_PARAM_FILE = pkg_resources.resource_filename(__name__, 'params/controller.json')
DEFAULT_PX4_PARAM_FILE = pkg_resources.resource_filename(__name__, 'params/px4.json')
DEFAULT_QUAD_PARAMETER_FILE = pkg_resources.resource_filename(__name__, 'params/quadrotor.json')

def readparamfile(filename, params=None):
    if params is None:
        params = {}
    with open(filename) as file:
        params.update(json.load(file))
    return params

class Controller():
    def __init__(self, quadparamfile=DEFAULT_QUAD_PARAMETER_FILE, 
                 px4paramfile=DEFAULT_PX4_PARAM_FILE):
        self.params = readparamfile(quadparamfile)

        self.px4_params = readparamfile(px4paramfile) 
        self.px4_params['angrate_max'] = np.array((self.px4_params['MC_ROLLRATE_MAX'],
                                                  self.px4_params['MC_PITCHRATE_MAX'],
                                                  self.px4_params['MC_YAWRATE_MAX']))
        self.px4_params['angrate_gain_P'] = np.diag((self.px4_params['MC_ROLLRATE_P'],
                                                  self.px4_params['MC_PITCHRATE_P'],
                                                  self.px4_params['MC_YAWRATE_P']))
        self.px4_params['angrate_gain_I'] = np.diag((self.px4_params['MC_ROLLRATE_I'],
                                                  self.px4_params['MC_PITCHRATE_I'],
                                                  self.px4_params['MC_YAWRATE_I']))
        self.px4_params['angrate_gain_D'] = np.diag((self.px4_params['MC_ROLLRATE_D'],
                                                  self.px4_params['MC_PITCHRATE_D'],
                                                  self.px4_params['MC_YAWRATE_D']))
        self.px4_params['angrate_gain_K'] = np.diag((self.px4_params['MC_ROLLRATE_K'],
                                                  self.px4_params['MC_PITCHRATE_K'],
                                                  self.px4_params['MC_YAWRATE_K']))
        self.px4_params['angrate_int_lim'] = np.array((self.px4_params['MC_RR_INT_LIM'],
                                                   self.px4_params['MC_PR_INT_LIM'],
                                                   self.px4_params['MC_YR_INT_LIM']))
        self.px4_params['attitude_gain_P'] = np.diag((self.px4_params['MC_ROLL_P'],
                                                  self.px4_params['MC_PITCH_P'],
                                                  self.px4_params['MC_YAW_P']))
        self.px4_params['angacc_max'] = np.array(self.px4_params['angacc_max'])
        self.px4_params['J'] = np.array(self.px4_params['J'])
        self.B = None
        #self.reset_controller()
    
    def reset_controller(self):
        self.w_error_int = np.zeros(3)
        self.w_filtered = np.zeros(3)
        self.w_filtered_last = np.zeros(3)

    def limit(self, array, upper_limit, lower_limit=None):
        if lower_limit is None:
            lower_limit = - upper_limit
        array[array > upper_limit] = upper_limit[array > upper_limit]
        array[array < lower_limit] = lower_limit[array < lower_limit]
    
    def mixer(self, torque_sp, T_sp):
        omega_squared = np.linalg.solve(self.B, np.concatenate(((T_sp,), torque_sp)))
        omega = np.sqrt(np.maximum(omega_squared, self.params['motor_min_speed']))
        omega = np.minimum(omega, self.params['motor_max_speed'])
        return omega 
    
    def attitude(self, q, q_sp):
        q_error = rowan.multiply(rowan.inverse(q), q_sp)
        omega_sp = 2 * self.px4_params['attitude_gain_P'] @ (np.sign(q_error[0]) * q_error[1:])
        self.limit(omega_sp, self.px4_params['angrate_max'])
        return omega_sp
    
    def angrate(self, w, w_sp, dt):
        w_error = w_sp - w
        #print("w_error", w_error)
        #print("w", w)
        self.w_error_int += dt * w_error
        self.limit(self.w_error_int, self.px4_params['angrate_int_lim'])

        const_w_filter = np.exp(- dt / self.px4_params['w_filter_time_const'])
        self.w_filtered *= const_w_filter
        self.w_filtered += (1 - const_w_filter) * w
        
        w_filtered_derivative = (self.w_filtered - self.w_filtered_last) / dt
        self.w_filtered_last[:] = self.w_filtered[:]

        alpha_sp = self.px4_params['angrate_gain_K'] \
                    @ (self.px4_params['angrate_gain_P'] @ w_error 
                       + self.px4_params['angrate_gain_I'] @ self.w_error_int
                       - self.px4_params['angrate_gain_D'] @ w_filtered_derivative)
        self.limit(alpha_sp, self.px4_params['angacc_max'])
        return alpha_sp

class PIDController(Controller):
    def __init__(self, quadparamfile=DEFAULT_QUAD_PARAMETER_FILE, 
                 ctrlparamfile=DEFAULT_CONTROL_PARAM_FILE, 
                 given_pid=False, p=0, i=0, d=0):
        super().__init__(quadparamfile=quadparamfile)
        self.params = readparamfile(filename=ctrlparamfile, params=self.params)
        self.given_pid = given_pid
        if (given_pid):
            self.p = [p,p,p]
            self.i = [i,i,i]
            self.d = [d,d,d]

    def calculate_gains(self):
        if (self.given_pid):
            self.params['K_i'] = np.diag(self.i)
            self.params['K_p'] = np.diag(self.p)
            self.params['K_d'] = np.diag(self.d)

        self.params['K_i'] = np.array(self.params['K_i'])
        self.params['K_p'] = np.diag([self.params['Lam_xy']*self.params['K_xy'],
                       self.params['Lam_xy']*self.params['K_xy'],
                       self.params['Lam_z']*self.params['K_z']])
        self.params['K_d'] = np.diag([self.params['K_xy'], self.params['K_xy'], self.params['K_z']])
        self.B = np.array([self.params['C_T'] * np.ones(4), 
                           self.params['C_T'] * self.params['l_arm'] * np.array([-1., -1., 1., 1.]),
                           self.params['C_T'] * self.params['l_arm'] * np.array([-1., 1., 1., -1.]),
                           self.params['C_q'] * np.array([-1., 1., -1., 1.])])

    def reset_controller(self):
        super().reset_controller()
        self.calculate_gains()
        self.F_r_dot = None
        self.F_r_last = None
        self.t_last = None
        self.t_last_wind_update = -self.params['wind_update_period']
        self.p_error = np.zeros(3)
        self.v_error = np.zeros(3)
        self.int_error = np.zeros(3)
        self.dt = 0.
        self.dt_inv = 0.

    def get_q(self, F_r, yaw=0., max_angle=np.pi):
        q_world_to_yaw = rowan.from_euler(0., 0., yaw, 'xyz')
        rotation_axis = np.cross((0, 0, 1), F_r)
        if np.allclose(rotation_axis, (0., 0., 0.)):
            unit_rotation_axis = np.array((1., 0., 0.,))
        else:
            unit_rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_axis /= np.linalg.norm(F_r)
        rotation_angle = np.arcsin(np.linalg.norm(rotation_axis))
        if F_r[2] < 0:
            rotation_angle = np.pi - rotation_angle
        if rotation_angle > max_angle:
            rotation_angle = max_angle
        q_yaw_to_body = rowan.from_axis_angle(unit_rotation_axis, rotation_angle)

        q_r = rowan.multiply(q_world_to_yaw, q_yaw_to_body)
        return rowan.normalize(q_r)
    
    def get_Fr(self, X, Z, imu, pd, vd, ad, meta_adapt_trigger, wind_gt):
        p_error = X[0:3] - pd
        v_error = X[7:10] - vd
        self.int_error += self.dt * p_error
        a_r = - self.params['K_p'] @ p_error - self.params['K_d'] @ v_error - \
                self.params['K_i'] @ self.int_error + ad
        F_r = (a_r * self.params['m']) + np.array([0., 0., self.params['m'] * self.params['g']])

        if self.F_r_last is None:
            self.F_r_dot = np.zeros(3)
        else:
            lam = np.exp(-self.dt / self.params['force_filter_time_const'])
            self.F_r_dot *= lam
            self.F_r_dot += (1-lam) * (F_r - self.F_r_last) / self.dt
        self.F_r_last = F_r.copy()
        return F_r, self.F_r_dot

    def position(self, X, Z, imu, pd, vd, ad, last_wind_update, t, wind_gt):
        if self.t_last is None:
            self.t_last = t
        else:
            self.dt = t - self.t_last
        if (self.t_last_wind_update < last_wind_update):
            self.t_last_wind_update = last_wind_update
            meta_adapt_trigger = True
        else:
            meta_adapt_trigger = False
        
        yaw = 0.
        self.t_last = t
        F_r, F_r_dot = self.get_Fr(X, Z=Z, imu=imu, pd=pd, vd=vd, ad=ad,  
                    meta_adapt_trigger=meta_adapt_trigger, wind_gt=wind_gt)
        T_r_prime = np.linalg.norm(F_r + self.params['thrust_delay'] * F_r_dot)
        q_r_prime = self.get_q(F_r + self.params['attitude_delay'] * F_r_dot, yaw)
        F_r_prime = rowan.to_matrix(q_r_prime) @ np.array((0, 0, T_r_prime))

        T_r_prime = np.linalg.norm(F_r_prime)
        q_r_prime = self.get_q(F_r_prime, yaw)
        return T_r_prime, q_r_prime


class MLMPCController(PIDController):
    """Meta-learning augmented MPC controller adapted to the repo position loop.

    The controller follows the paper's decomposition
    f_d(x, x_r) ~= phi(x, x_r) a_hat: a fixed nonlinear basis represents the
    disturbance model, while the final linear layer and covariance are adapted
    online. The receding-horizon part uses the repo's translational position
    dynamics and returns the first acceleration command of a finite-horizon
    disturbance-aware LQR/MPC problem.
    """

    class BasisNet(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=64, num_hidden_layers=4, use_spectral_norm=True):
            super().__init__()
            layers = []
            in_dim = int(input_dim)
            for _ in range(max(1, int(num_hidden_layers))):
                linear = nn.Linear(in_dim, int(hidden_dim))
                if use_spectral_norm:
                    linear = spectral_norm(linear)
                layers.append(linear)
                layers.append(nn.ReLU())
                in_dim = int(hidden_dim)
            out = nn.Linear(in_dim, int(output_dim))
            if use_spectral_norm:
                out = spectral_norm(out)
            layers.append(out)
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    def __init__(
        self,
        given_pid=False,
        p=0,
        i=0,
        d=0,
        basis_dim=32,
        basis_hidden_dim=64,
        basis_hidden_layers=4,
        basis_seed=2410,
        horizon=10,
        mpc_dt=None,
        q_pos=12.0,
        q_vel=4.0,
        r_acc=0.4,
        terminal_scale=4.0,
        force_bound=80.0,
        accel_bound=15.0,
        mpc_blend=0.35,
        covariance_init=25.0,
        process_noise=1e-3,
        measurement_noise=4.0,
        sigma=1e-3,
        basis_clip=10.0,
        compensation_gain=1.0,
    ):
        super().__init__(given_pid=given_pid, p=p, i=i, d=d)
        self.basis_out_dim = int(max(1, basis_dim))
        self.basis_dim = self.basis_out_dim + 1  # leading constant feature
        self.basis_hidden_dim = int(basis_hidden_dim)
        self.basis_hidden_layers = int(basis_hidden_layers)
        self.basis_seed = int(basis_seed)
        self.horizon = int(max(1, horizon))
        self.mpc_dt_override = mpc_dt
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.r_acc = float(r_acc)
        self.terminal_scale = float(terminal_scale)
        self.force_bound = float(force_bound)
        self.accel_bound = float(accel_bound)
        self.mpc_blend = float(np.clip(mpc_blend, 0.0, 1.0))
        self.covariance_init = float(covariance_init)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.sigma = float(sigma)
        self.basis_clip = float(basis_clip)
        self.compensation_gain = float(compensation_gain)
        self.basis_input_dim = 28

        rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.basis_seed)
        self.basis_net = self.BasisNet(
            input_dim=self.basis_input_dim,
            output_dim=self.basis_out_dim,
            hidden_dim=self.basis_hidden_dim,
            num_hidden_layers=self.basis_hidden_layers,
            use_spectral_norm=True,
        ).to(device).double()
        torch.random.set_rng_state(rng_state)
        for param in self.basis_net.parameters():
            param.requires_grad_(False)
        self.basis_net.eval()

        self.reset_controller()

    def reset_controller(self):
        super().reset_controller()
        self.A_hat = np.zeros((3, self.basis_dim), dtype=np.float64)
        self.P_cov = self.covariance_init * np.eye(self.basis_dim, dtype=np.float64)
        self.last_f_hat = np.zeros(3, dtype=np.float64)
        self.last_residual = np.zeros(3, dtype=np.float64)
        self.last_basis = np.zeros(self.basis_dim, dtype=np.float64)
        self.last_mpc_accel = np.zeros(3, dtype=np.float64)
        self.force_err_list = []
        self.cov_trace_hist = []
        self.fhat_norm_hist = []
        self.mpc_accel_norm_hist = []

    def _basis_input(self, X, pd, vd, ad):
        p = np.asarray(X[0:3], dtype=np.float64)
        q = np.asarray(X[3:7], dtype=np.float64)
        v = np.asarray(X[7:10], dtype=np.float64)
        w = np.asarray(X[10:13], dtype=np.float64)
        pd = np.asarray(pd, dtype=np.float64)
        vd = np.asarray(vd, dtype=np.float64)
        ad = np.asarray(ad, dtype=np.float64)
        e_p = p - pd
        e_v = v - vd
        feat = np.concatenate((
            p / 5.0,
            v / 5.0,
            q,
            w / 10.0,
            pd / 5.0,
            vd / 5.0,
            ad / 20.0,
            e_p / 5.0,
            e_v / 5.0,
        ))
        if feat.shape[0] != self.basis_input_dim:
            raise ValueError(f"MLMPC basis input dim {feat.shape[0]} != {self.basis_input_dim}")
        feat = np.nan_to_num(feat, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(feat, -10.0, 10.0)

    def _basis(self, X, pd, vd, ad):
        feat = self._basis_input(X, pd, vd, ad)
        feat_t = torch.from_numpy(feat).to(device).double().unsqueeze(0)
        with torch.no_grad():
            z = self.basis_net(feat_t).cpu().numpy().reshape(-1)
        z = np.nan_to_num(z, nan=0.0, posinf=self.basis_clip, neginf=-self.basis_clip)
        z = np.clip(z, -self.basis_clip, self.basis_clip)
        z = np.concatenate(([1.0], z))
        self.last_basis = z.copy()
        return z

    def _predict_force_from_basis(self, z):
        f_hat = self.A_hat @ z
        f_hat = np.nan_to_num(f_hat, nan=0.0, posinf=self.force_bound, neginf=-self.force_bound)
        return np.clip(f_hat, -self.force_bound, self.force_bound)

    def _measured_residual_force(self, X, Z, imu):
        if imu is None:
            return None
        q = np.asarray(X[3:7], dtype=np.float64)
        q = q / max(np.linalg.norm(q), 1e-12)
        R = rowan.to_matrix(q)
        z = np.asarray(Z, dtype=np.float64)
        z = np.nan_to_num(z, nan=0.0, posinf=self.params['motor_max_speed'], neginf=0.0)
        motor_sq = np.clip(
            z ** 2,
            self.params['motor_min_speed'] ** 2,
            self.params['motor_max_speed'] ** 2,
        )
        thrust_world = self.params['C_T'] * float(np.sum(motor_sq)) * (R @ np.array([0.0, 0.0, 1.0]))
        y = self.params['m'] * np.asarray(imu[0:3], dtype=np.float64)
        y += np.array([0.0, 0.0, self.params['m'] * self.params['g']])
        y -= thrust_world
        if not np.all(np.isfinite(y)):
            return None
        return np.clip(y, -self.force_bound, self.force_bound)

    def _adapt_last_layer(self, z, y):
        if y is None:
            return
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(3)
        if z.shape[0] != self.basis_dim or not np.all(np.isfinite(z)) or not np.all(np.isfinite(y)):
            return

        dt = self.dt if self.dt > 0 else self.params['dt_posctrl']
        decay = max(0.0, 1.0 - 2.0 * self.sigma * dt)
        P_pred = decay * self.P_cov + self.process_noise * dt * np.eye(self.basis_dim)
        P_pred = 0.5 * (P_pred + P_pred.T)
        Pz = P_pred @ z
        denom = self.measurement_noise + float(z @ Pz)
        if denom <= 1e-12 or not np.isfinite(denom):
            return
        gain = Pz / denom
        pred = self._predict_force_from_basis(z)
        err = np.clip(y - pred, -self.force_bound, self.force_bound)
        self.A_hat = (1.0 - self.sigma * dt) * self.A_hat + np.outer(err, gain)
        self.A_hat = np.nan_to_num(self.A_hat, nan=0.0, posinf=self.force_bound, neginf=-self.force_bound)
        self.A_hat = np.clip(self.A_hat, -self.force_bound, self.force_bound)
        self.P_cov = P_pred - np.outer(gain, z) @ P_pred
        self.P_cov = 0.5 * (self.P_cov + self.P_cov.T)
        diag = np.clip(np.diag(self.P_cov), 1e-8, self.covariance_init * 100.0)
        self.P_cov[np.diag_indices_from(self.P_cov)] = diag

    def _solve_mpc_accel(self, X, pd, vd, ad, f_hat):
        dt_mpc = self.params['dt_posctrl'] if self.mpc_dt_override is None else float(self.mpc_dt_override)
        dt_mpc = max(dt_mpc, 1e-3)
        I3 = np.eye(3)
        A = np.block([[I3, dt_mpc * I3], [np.zeros((3, 3)), I3]])
        B = np.vstack([0.5 * (dt_mpc ** 2) * I3, dt_mpc * I3])
        Q = np.diag([self.q_pos, self.q_pos, self.q_pos, self.q_vel, self.q_vel, self.q_vel])
        Rm = self.r_acc * I3
        S = self.terminal_scale * Q
        s = np.zeros(6)
        d_acc = self.compensation_gain * np.asarray(f_hat, dtype=np.float64) / max(self.params['m'], 1e-12)
        b = B @ d_acc

        K0 = np.zeros((3, 6))
        k0 = np.zeros(3)
        for k in reversed(range(self.horizon)):
            S_next = S
            s_next = s
            H = Rm + B.T @ S_next @ B
            G = B.T @ S_next @ A
            h = B.T @ (S_next @ b + s_next)
            try:
                H_inv_G = np.linalg.solve(H, G)
                H_inv_h = np.linalg.solve(H, h)
            except np.linalg.LinAlgError:
                H_inv_G = np.linalg.pinv(H) @ G
                H_inv_h = np.linalg.pinv(H) @ h
            S = Q + A.T @ S_next @ A - G.T @ H_inv_G
            s = A.T @ (S_next @ b + s_next) - G.T @ H_inv_h
            if k == 0:
                K0 = H_inv_G
                k0 = H_inv_h

        e = np.concatenate((np.asarray(X[0:3]) - np.asarray(pd), np.asarray(X[7:10]) - np.asarray(vd)))
        e = np.nan_to_num(e, nan=0.0, posinf=10.0, neginf=-10.0)
        u_acc = -K0 @ e - k0
        u_acc = np.nan_to_num(u_acc, nan=0.0, posinf=self.accel_bound, neginf=-self.accel_bound)
        norm_u = np.linalg.norm(u_acc)
        if norm_u > self.accel_bound:
            u_acc = u_acc * (self.accel_bound / (norm_u + 1e-12))
        a_cmd = np.asarray(ad, dtype=np.float64) + u_acc
        norm_a = np.linalg.norm(a_cmd)
        if norm_a > self.accel_bound:
            a_cmd = a_cmd * (self.accel_bound / (norm_a + 1e-12))
        return a_cmd

    def _limit_force_vector(self, F_cmd):
        F = np.asarray(F_cmd, dtype=np.float64).copy()
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        min_fz = 0.15 * self.params['m'] * self.params['g']
        if F[2] < min_fz:
            F[2] = min_fz
        max_angle = float(self.params.get('max_zenith_angle', np.pi / 4.0))
        h_norm = float(np.linalg.norm(F[0:2]))
        h_max = abs(F[2]) * np.tan(max_angle)
        if h_norm > h_max > 0:
            F[0:2] *= h_max / (h_norm + 1e-12)
        max_force = self.params['m'] * (self.params['g'] + self.accel_bound)
        f_norm = float(np.linalg.norm(F))
        if f_norm > max_force:
            F *= max_force / (f_norm + 1e-12)
        return F

    def _force_derivative(self, F_cmd):
        if self.F_r_last is None or self.dt <= 0:
            self.F_r_dot = np.zeros(3)
        else:
            lam = np.exp(-self.dt / self.params['force_filter_time_const'])
            self.F_r_dot *= lam
            self.F_r_dot += (1.0 - lam) * (F_cmd - self.F_r_last) / self.dt
        self.F_r_last = F_cmd.copy()
        return self.F_r_dot

    def position(self, X, Z, imu, pd, vd, ad, last_wind_update, t, wind_gt):
        if self.t_last is None:
            self.t_last = t
            self.dt = self.params['dt_posctrl']
        else:
            self.dt = t - self.t_last
            if self.dt <= 0:
                self.dt = self.params['dt_posctrl']
        self.t_last = t

        z = self._basis(X, pd, vd, ad)
        f_hat = self._predict_force_from_basis(z)
        if wind_gt is not None and np.all(np.isfinite(wind_gt)):
            self.force_err_list.append(float(np.linalg.norm(f_hat - wind_gt)))

        F_pid, F_pid_dot = super().get_Fr(
            X, Z=Z, imu=imu, pd=pd, vd=vd, ad=ad,
            meta_adapt_trigger=False, wind_gt=wind_gt,
        )
        base_acc = F_pid / max(self.params['m'], 1e-12) - np.array([0.0, 0.0, self.params['g']])
        compensated_base_acc = base_acc - self.compensation_gain * f_hat / max(self.params['m'], 1e-12)
        mpc_acc = self._solve_mpc_accel(X, pd, vd, ad, f_hat)
        a_cmd = (1.0 - self.mpc_blend) * compensated_base_acc + self.mpc_blend * mpc_acc
        norm_a = np.linalg.norm(a_cmd)
        if norm_a > self.accel_bound:
            a_cmd = a_cmd * (self.accel_bound / (norm_a + 1e-12))
        self.last_mpc_accel = a_cmd.copy()
        F_cmd = self.params['m'] * (a_cmd + np.array([0.0, 0.0, self.params['g']]))
        F_cmd = self._limit_force_vector(F_cmd)
        if not np.all(np.isfinite(F_cmd)) or np.linalg.norm(F_cmd) < 1e-9:
            F_cmd = F_pid

        F_cmd_dot = F_pid_dot
        T_r_prime = np.linalg.norm(F_cmd + self.params['thrust_delay'] * F_cmd_dot)
        q_r_prime = self.get_q(
            F_cmd + self.params['attitude_delay'] * F_cmd_dot,
            yaw=0.0,
            max_angle=self.params.get('max_zenith_angle', np.pi / 4.0),
        )
        F_r_prime = rowan.to_matrix(q_r_prime) @ np.array((0.0, 0.0, T_r_prime))
        T_r_prime = np.linalg.norm(F_r_prime)
        q_r_prime = self.get_q(
            F_r_prime,
            yaw=0.0,
            max_angle=self.params.get('max_zenith_angle', np.pi / 4.0),
        )

        residual = self._measured_residual_force(X, Z, imu)
        if residual is not None:
            self.last_residual = residual.copy()
            self._adapt_last_layer(z, residual)
        self.last_f_hat = f_hat.copy()
        self.cov_trace_hist.append(float(np.trace(self.P_cov)))
        self.fhat_norm_hist.append(float(np.linalg.norm(f_hat)))
        self.mpc_accel_norm_hist.append(float(np.linalg.norm(a_cmd)))
        return T_r_prime, q_r_prime


class RTNMPCController(PIDController):
    """Real-time neural MPC controller in the repo position-control interface.

    This adapts Salzmann et al.'s RTN-MPC idea to the current simulator by
    using nominal translational dynamics plus a neural residual acceleration
    model. At each position-control tick the residual model is evaluated and
    linearized in a batched PyTorch preparation phase, then a single
    RTI-style finite-horizon LQR/QP approximation returns the first control.
    """

    class ResidualNet(nn.Module):
        def __init__(self, input_dim, output_dim=3, hidden_dim=128, hidden_layers=4, use_spectral_norm=False):
            super().__init__()
            layers = []
            in_dim = int(input_dim)
            for _ in range(max(1, int(hidden_layers))):
                linear = nn.Linear(in_dim, int(hidden_dim))
                if use_spectral_norm:
                    linear = spectral_norm(linear)
                layers.append(linear)
                layers.append(nn.ReLU())
                in_dim = int(hidden_dim)
            out = nn.Linear(in_dim, int(output_dim))
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
            if use_spectral_norm:
                out = spectral_norm(out)
            layers.append(out)
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    def __init__(
        self,
        given_pid=False,
        p=0,
        i=0,
        d=0,
        horizon=10,
        mpc_dt=None,
        hidden_dim=128,
        hidden_layers=4,
        use_spectral_norm=False,
        q_pos=12.0,
        q_vel=4.0,
        r_acc=0.35,
        terminal_scale=4.0,
        accel_bound=15.0,
        residual_bound=8.0,
        mpc_blend=0.10,
        use_jacobian=True,
        online_adapt=True,
        online_lr=5e-4,
        online_batch_size=32,
        online_buffer_size=512,
        online_train_steps=1,
        online_min_samples=16,
        online_update_every=1,
        grad_clip=5.0,
        fallback_margin=1.05,
    ):
        super().__init__(given_pid=given_pid, p=p, i=i, d=d)
        self.horizon = int(max(1, horizon))
        self.mpc_dt_override = mpc_dt
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.use_spectral_norm = bool(use_spectral_norm)
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.r_acc = float(r_acc)
        self.terminal_scale = float(terminal_scale)
        self.accel_bound = float(accel_bound)
        self.residual_bound = float(residual_bound)
        self.mpc_blend = float(np.clip(mpc_blend, 0.0, 1.0))
        self.use_jacobian = bool(use_jacobian)
        self.online_adapt = bool(online_adapt)
        self.online_lr = float(online_lr)
        self.online_batch_size = int(max(1, online_batch_size))
        self.online_buffer_size = int(max(1, online_buffer_size))
        self.online_train_steps = int(max(0, online_train_steps))
        self.online_min_samples = int(max(1, online_min_samples))
        self.online_update_every = int(max(1, online_update_every))
        self.grad_clip = float(max(0.0, grad_clip))
        self.fallback_margin = float(max(1.0, fallback_margin))

        # Feature is [position error, velocity error, acceleration correction].
        self.feature_dim = 9
        self.feature_scale = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 15.0, 15.0, 15.0], dtype=np.float64)
        self.residual_model = self.ResidualNet(
            input_dim=self.feature_dim,
            output_dim=3,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            use_spectral_norm=self.use_spectral_norm,
        ).to(device).double()
        self.residual_optimizer = optim.Adam(self.residual_model.parameters(), lr=self.online_lr)
        self.reset_controller()

    def reset_controller(self):
        super().reset_controller()
        self.u_warm = np.zeros((self.horizon, 3), dtype=np.float64)
        self.prev_feature = None
        self.replay_buffer = deque(maxlen=self.online_buffer_size)
        self.rtn_force_last = None
        self.rtn_force_dot = np.zeros(3, dtype=np.float64)
        self.step_counter = 0
        self.last_residual_acc = np.zeros(3, dtype=np.float64)
        self.last_pred_residual_acc = np.zeros(3, dtype=np.float64)
        self.last_mpc_accel = np.zeros(3, dtype=np.float64)
        self.last_u_corr = np.zeros(3, dtype=np.float64)
        self.last_effective_blend = self.mpc_blend
        self.last_f_hat = np.zeros(3, dtype=np.float64)
        self.force_err_list = []
        self.rtnmpc_loss_hist = []
        self.rtnmpc_residual_norm_hist = []
        self.rtnmpc_accel_norm_hist = []

    def _dt_mpc(self):
        dt = self.params['dt_posctrl'] if self.mpc_dt_override is None else float(self.mpc_dt_override)
        return max(float(dt), 1e-3)

    def _feature_raw(self, e_state, u_corr):
        e_state = np.asarray(e_state, dtype=np.float64).reshape(6)
        u_corr = np.asarray(u_corr, dtype=np.float64).reshape(3)
        feat = np.concatenate((e_state, u_corr))
        feat = np.nan_to_num(feat, nan=0.0, posinf=20.0, neginf=-20.0)
        return feat

    def _model_output(self, feature_norm):
        return self.residual_bound * torch.tanh(self.residual_model(feature_norm))

    def _residual_and_jacobian_batch(self, e_nodes, u_nodes):
        e_nodes = np.asarray(e_nodes, dtype=np.float64).reshape(-1, 6)
        u_nodes = np.asarray(u_nodes, dtype=np.float64).reshape(-1, 3)
        raw = np.concatenate((e_nodes, u_nodes), axis=1)
        raw = np.nan_to_num(raw, nan=0.0, posinf=20.0, neginf=-20.0)
        norm = raw / self.feature_scale.reshape(1, -1)
        z = torch.from_numpy(norm).to(device).double()
        z.requires_grad_(self.use_jacobian)

        with torch.enable_grad():
            pred = self._model_output(z)
            if self.use_jacobian:
                jac_norm = []
                for axis in range(3):
                    grad_axis = torch.autograd.grad(
                        pred[:, axis].sum(),
                        z,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=False,
                    )[0]
                    jac_norm.append(grad_axis.detach().cpu().numpy())
                jac_norm = np.stack(jac_norm, axis=1)
            else:
                jac_norm = np.zeros((raw.shape[0], 3, self.feature_dim), dtype=np.float64)

        residual = pred.detach().cpu().numpy()
        residual = np.nan_to_num(residual, nan=0.0, posinf=self.residual_bound, neginf=-self.residual_bound)
        residual = np.clip(residual, -self.residual_bound, self.residual_bound)
        jac = jac_norm / self.feature_scale.reshape(1, 1, -1)
        jac = np.nan_to_num(jac, nan=0.0, posinf=0.0, neginf=0.0)
        jac_e = jac[:, :, :6]
        jac_u = jac[:, :, 6:]
        return residual, jac_e, jac_u

    def _prepare_iterate(self, e0, pid_u_corr):
        pid_u_corr = np.asarray(pid_u_corr, dtype=np.float64).reshape(3)
        pid_u_corr = np.clip(pid_u_corr, -self.accel_bound, self.accel_bound)
        if self.u_warm.shape != (self.horizon, 3):
            self.u_warm = np.zeros((self.horizon, 3), dtype=np.float64)
        if np.linalg.norm(self.u_warm) < 1e-12:
            u_nodes = np.tile(pid_u_corr.reshape(1, 3), (self.horizon, 1))
        else:
            u_nodes = np.vstack((self.u_warm[1:], self.u_warm[-1:]))
            u_nodes[0] = 0.5 * u_nodes[0] + 0.5 * pid_u_corr
        u_nodes = np.clip(u_nodes, -self.accel_bound, self.accel_bound)

        dt = self._dt_mpc()
        A = np.block([[np.eye(3), dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
        B = np.vstack([0.5 * (dt ** 2) * np.eye(3), dt * np.eye(3)])
        e_nodes = np.zeros((self.horizon, 6), dtype=np.float64)
        e = np.asarray(e0, dtype=np.float64).reshape(6)
        for k in range(self.horizon):
            e_nodes[k] = e
            e = A @ e + B @ u_nodes[k]
            e = np.nan_to_num(e, nan=0.0, posinf=20.0, neginf=-20.0)
        return e_nodes, u_nodes

    def _solve_rti_lqr(self, e0, e_nodes, u_nodes, residuals, jac_e, jac_u):
        dt = self._dt_mpc()
        A_nom = np.block([[np.eye(3), dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
        B_nom = np.vstack([0.5 * (dt ** 2) * np.eye(3), dt * np.eye(3)])
        Q = np.diag([self.q_pos, self.q_pos, self.q_pos, self.q_vel, self.q_vel, self.q_vel])
        Rm = self.r_acc * np.eye(3)
        S = self.terminal_scale * Q
        s = np.zeros(6, dtype=np.float64)
        K_seq = [np.zeros((3, 6), dtype=np.float64) for _ in range(self.horizon)]
        k_seq = [np.zeros(3, dtype=np.float64) for _ in range(self.horizon)]
        A_seq = []
        B_seq = []
        c_seq = []

        for k in range(self.horizon):
            Jx = np.asarray(jac_e[k], dtype=np.float64).reshape(3, 6)
            Ju = np.asarray(jac_u[k], dtype=np.float64).reshape(3, 3)
            rk = np.asarray(residuals[k], dtype=np.float64).reshape(3)
            xbar = np.asarray(e_nodes[k], dtype=np.float64).reshape(6)
            ubar = np.asarray(u_nodes[k], dtype=np.float64).reshape(3)
            A_k = A_nom + B_nom @ Jx
            B_k = B_nom @ (np.eye(3) + Ju)
            c_k = B_nom @ (rk - Jx @ xbar - Ju @ ubar)
            A_seq.append(A_k)
            B_seq.append(B_k)
            c_seq.append(c_k)

        for k in reversed(range(self.horizon)):
            A_k = A_seq[k]
            B_k = B_seq[k]
            c_k = c_seq[k]
            S_next = S
            s_next = s
            H = Rm + B_k.T @ S_next @ B_k
            G = B_k.T @ S_next @ A_k
            h = B_k.T @ (S_next @ c_k + s_next)
            try:
                K = np.linalg.solve(H, G)
                kff = np.linalg.solve(H, h)
            except np.linalg.LinAlgError:
                H_pinv = np.linalg.pinv(H)
                K = H_pinv @ G
                kff = H_pinv @ h
            K_seq[k] = K
            k_seq[k] = kff
            S = Q + A_k.T @ S_next @ A_k - G.T @ K
            s = A_k.T @ (S_next @ c_k + s_next) - G.T @ kff
            S = 0.5 * (S + S.T)

        x = np.asarray(e0, dtype=np.float64).reshape(6)
        u_sol = np.zeros((self.horizon, 3), dtype=np.float64)
        for k in range(self.horizon):
            u = -K_seq[k] @ x - k_seq[k]
            u = np.nan_to_num(u, nan=0.0, posinf=self.accel_bound, neginf=-self.accel_bound)
            norm_u = np.linalg.norm(u)
            if norm_u > self.accel_bound:
                u = u * (self.accel_bound / (norm_u + 1e-12))
            u_sol[k] = u
            x = A_seq[k] @ x + B_seq[k] @ u + c_seq[k]
            x = np.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)
        return u_sol

    def _accepted_blend(self, e0, pid_u_corr, rtn_u_corr, residual_acc):
        dt = self._dt_mpc()
        A = np.block([[np.eye(3), dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
        B = np.vstack([0.5 * (dt ** 2) * np.eye(3), dt * np.eye(3)])
        Q = np.diag([self.q_pos, self.q_pos, self.q_pos, self.q_vel, self.q_vel, self.q_vel])
        residual_acc = np.asarray(residual_acc, dtype=np.float64).reshape(3)
        e0 = np.asarray(e0, dtype=np.float64).reshape(6)

        def score(u_corr):
            u_corr = np.asarray(u_corr, dtype=np.float64).reshape(3)
            u_corr = np.nan_to_num(u_corr, nan=0.0, posinf=self.accel_bound, neginf=-self.accel_bound)
            e_next = A @ e0 + B @ (u_corr + residual_acc)
            e_next = np.nan_to_num(e_next, nan=0.0, posinf=20.0, neginf=-20.0)
            return float(e_next @ Q @ e_next + self.r_acc * (u_corr @ u_corr))

        pid_score = score(pid_u_corr)
        rtn_score = score(rtn_u_corr)
        if not np.isfinite(rtn_score) or rtn_score > self.fallback_margin * max(pid_score, 1e-9):
            return 0.0
        if rtn_score > pid_score:
            return 0.5 * self.mpc_blend
        return self.mpc_blend

    def _measured_residual_accel(self, X, Z, imu):
        if imu is None:
            return None
        q = np.asarray(X[3:7], dtype=np.float64)
        q = q / max(np.linalg.norm(q), 1e-12)
        R = rowan.to_matrix(q)
        z = np.asarray(Z, dtype=np.float64)
        z = np.nan_to_num(z, nan=0.0, posinf=self.params['motor_max_speed'], neginf=0.0)
        motor_sq = np.clip(
            z ** 2,
            self.params['motor_min_speed'] ** 2,
            self.params['motor_max_speed'] ** 2,
        )
        thrust_world = self.params['C_T'] * float(np.sum(motor_sq)) * (R @ np.array([0.0, 0.0, 1.0]))
        residual_force = self.params['m'] * np.asarray(imu[0:3], dtype=np.float64)
        residual_force += np.array([0.0, 0.0, self.params['m'] * self.params['g']])
        residual_force -= thrust_world
        if not np.all(np.isfinite(residual_force)):
            return None
        residual_acc = residual_force / max(self.params['m'], 1e-12)
        return np.clip(residual_acc, -self.residual_bound, self.residual_bound)

    def _observe_residual(self, feature_raw, target_acc):
        if not self.online_adapt or self.online_train_steps <= 0:
            return
        feature_raw = np.asarray(feature_raw, dtype=np.float64).reshape(self.feature_dim)
        target_acc = np.asarray(target_acc, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(feature_raw)) or not np.all(np.isfinite(target_acc)):
            return
        target_acc = np.clip(target_acc, -self.residual_bound, self.residual_bound)
        self.replay_buffer.append((feature_raw.copy(), target_acc.copy()))
        if len(self.replay_buffer) < self.online_min_samples:
            return
        if self.step_counter % self.online_update_every != 0:
            return

        batch_size = min(self.online_batch_size, len(self.replay_buffer))
        for _ in range(self.online_train_steps):
            idx = np.random.choice(len(self.replay_buffer), size=batch_size, replace=False)
            x = np.stack([self.replay_buffer[i][0] for i in idx], axis=0)
            y = np.stack([self.replay_buffer[i][1] for i in idx], axis=0)
            x_t = torch.from_numpy(x / self.feature_scale.reshape(1, -1)).to(device).double()
            y_t = torch.from_numpy(y).to(device).double()
            self.residual_optimizer.zero_grad()
            pred = self._model_output(x_t)
            loss = F.mse_loss(pred, y_t)
            if not torch.isfinite(loss):
                return
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.residual_model.parameters(), self.grad_clip)
            self.residual_optimizer.step()
            self.rtnmpc_loss_hist.append(float(loss.detach().cpu().item()))

    def _limit_force_vector(self, F_cmd):
        F_cmd = np.asarray(F_cmd, dtype=np.float64).copy()
        F_cmd = np.nan_to_num(F_cmd, nan=0.0, posinf=0.0, neginf=0.0)
        min_fz = 0.15 * self.params['m'] * self.params['g']
        if F_cmd[2] < min_fz:
            F_cmd[2] = min_fz
        max_angle = float(self.params.get('max_zenith_angle', np.pi / 4.0))
        h_norm = float(np.linalg.norm(F_cmd[0:2]))
        h_max = abs(F_cmd[2]) * np.tan(max_angle)
        if h_norm > h_max > 0:
            F_cmd[0:2] *= h_max / (h_norm + 1e-12)
        max_force = self.params['m'] * (self.params['g'] + self.accel_bound)
        f_norm = float(np.linalg.norm(F_cmd))
        if f_norm > max_force:
            F_cmd *= max_force / (f_norm + 1e-12)
        return F_cmd

    def _command_force_derivative(self, F_cmd):
        if self.rtn_force_last is None or self.dt <= 0:
            self.rtn_force_dot = np.zeros(3, dtype=np.float64)
        else:
            lam = np.exp(-self.dt / self.params['force_filter_time_const'])
            self.rtn_force_dot *= lam
            self.rtn_force_dot += (1.0 - lam) * (F_cmd - self.rtn_force_last) / self.dt
        self.rtn_force_last = F_cmd.copy()
        return self.rtn_force_dot

    def save(self, path):
        torch.save({
            'residual_model': self.residual_model.state_dict(),
            'horizon': self.horizon,
            'hidden_dim': self.hidden_dim,
            'hidden_layers': self.hidden_layers,
            'residual_bound': self.residual_bound,
            'feature_scale': self.feature_scale,
        }, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.residual_model.load_state_dict(ckpt['residual_model'])
        self.residual_model.eval()

    def position(self, X, Z, imu, pd, vd, ad, last_wind_update, t, wind_gt):
        if self.t_last is None:
            self.t_last = t
            self.dt = self.params['dt_posctrl']
        else:
            self.dt = t - self.t_last
            if self.dt <= 0:
                self.dt = self.params['dt_posctrl']
        self.t_last = t
        self.step_counter += 1

        measured_residual = self._measured_residual_accel(X, Z, imu)
        if measured_residual is not None:
            self.last_residual_acc = measured_residual.copy()
            if self.prev_feature is not None:
                self._observe_residual(self.prev_feature, measured_residual)

        F_pid, _ = super().get_Fr(
            X, Z=Z, imu=imu, pd=pd, vd=vd, ad=ad,
            meta_adapt_trigger=False, wind_gt=wind_gt,
        )
        pid_acc = F_pid / max(self.params['m'], 1e-12) - np.array([0.0, 0.0, self.params['g']])
        pid_u_corr = pid_acc - np.asarray(ad, dtype=np.float64)
        e0 = np.concatenate((np.asarray(X[0:3], dtype=np.float64) - np.asarray(pd, dtype=np.float64),
                             np.asarray(X[7:10], dtype=np.float64) - np.asarray(vd, dtype=np.float64)))
        e0 = np.nan_to_num(e0, nan=0.0, posinf=20.0, neginf=-20.0)

        e_nodes, u_nodes = self._prepare_iterate(e0, pid_u_corr)
        residuals, jac_e, jac_u = self._residual_and_jacobian_batch(e_nodes, u_nodes)
        u_sol = self._solve_rti_lqr(e0, e_nodes, u_nodes, residuals, jac_e, jac_u)
        self.u_warm = u_sol.copy()

        u_corr = u_sol[0]
        effective_blend = self._accepted_blend(e0, pid_u_corr, u_corr, residuals[0])
        rtn_acc = np.asarray(ad, dtype=np.float64) + u_corr
        a_cmd = (1.0 - effective_blend) * pid_acc + effective_blend * rtn_acc
        a_cmd = np.nan_to_num(a_cmd, nan=0.0, posinf=self.accel_bound, neginf=-self.accel_bound)
        norm_a = np.linalg.norm(a_cmd)
        if norm_a > self.accel_bound:
            a_cmd = a_cmd * (self.accel_bound / (norm_a + 1e-12))

        F_cmd = self.params['m'] * (a_cmd + np.array([0.0, 0.0, self.params['g']]))
        F_cmd = self._limit_force_vector(F_cmd)
        F_dot = self._command_force_derivative(F_cmd)
        if not np.all(np.isfinite(F_cmd)) or np.linalg.norm(F_cmd) < 1e-9:
            F_cmd = F_pid
            F_dot = np.zeros(3, dtype=np.float64)

        T_r_prime = np.linalg.norm(F_cmd + self.params['thrust_delay'] * F_dot)
        q_r_prime = self.get_q(
            F_cmd + self.params['attitude_delay'] * F_dot,
            yaw=0.0,
            max_angle=self.params.get('max_zenith_angle', np.pi / 4.0),
        )
        F_r_prime = rowan.to_matrix(q_r_prime) @ np.array((0.0, 0.0, T_r_prime))
        T_r_prime = np.linalg.norm(F_r_prime)
        q_r_prime = self.get_q(
            F_r_prime,
            yaw=0.0,
            max_angle=self.params.get('max_zenith_angle', np.pi / 4.0),
        )

        actual_u_corr = a_cmd - np.asarray(ad, dtype=np.float64)
        self.prev_feature = self._feature_raw(e0, actual_u_corr)
        self.last_pred_residual_acc = residuals[0].copy()
        self.last_u_corr = u_corr.copy()
        self.last_effective_blend = float(effective_blend)
        self.last_mpc_accel = a_cmd.copy()
        self.last_f_hat = self.params['m'] * self.last_pred_residual_acc
        if wind_gt is not None and np.all(np.isfinite(wind_gt)):
            self.force_err_list.append(float(np.linalg.norm(self.last_f_hat - wind_gt)))
        self.rtnmpc_residual_norm_hist.append(float(np.linalg.norm(self.last_pred_residual_acc)))
        self.rtnmpc_accel_norm_hist.append(float(np.linalg.norm(a_cmd)))
        return T_r_prime, q_r_prime


class AgileMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_hidden_layers=2, output_dim=6):
        super().__init__()
        layers = []
        in_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        for _ in range(max(1, int(num_hidden_layers))):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class AgileAdaptiveController(PIDController):
    """Agile real-world-style adaptation baseline integrated in the repo PID/quadsim loop."""

    def __init__(
        self,
        trajectory_obj=None,
        use_residual=True,
        use_anchor=True,
        use_ats=True,
        online_update_in_test=False,
        residual_hidden_dim=128,
        residual_hidden_layers=2,
        policy_hidden_dim=128,
        policy_hidden_layers=2,
        residual_buffer_size=5000,
        residual_batch_size=256,
        residual_lr=1e-3,
        residual_weight_decay=0.0,
        residual_train_steps=2,
        residual_grad_clip=1.0,
        policy_lr=1e-4,
        bptt_horizon=15,
        discount=0.98,
        policy_update_every=5,
        policy_update_steps=1,
        policy_grad_clip=1.0,
        w_pos=1.0,
        w_vel=0.1,
        w_u=0.01,
        w_du=0.05,
        control_delta_scale=0.3,
        omega_guidance_gain=4.0,
        omega_bound=6.0,
        residual_accel_clip=25.0,
        residual_omega_clip=8.0,
        use_action_history=True,
        alpha_init=1.0,
        alpha_min=0.6,
        alpha_max=1.8,
        alpha_lr=1e-2,
        lambda_speed=0.1,
        lambda_safe=1.0,
        error_threshold=0.8,
        reset_alpha_each_episode=True,
        reset_buffer_each_episode=True,
        given_pid=False,
        p=0,
        i=0,
        d=0,
    ):
        super().__init__(given_pid=given_pid, p=p, i=i, d=d)
        self.trajectory_obj = trajectory_obj

        self.use_residual = bool(use_residual)
        self.use_anchor = bool(use_anchor)
        self.use_ats = bool(use_ats)
        self.online_update_in_test = bool(online_update_in_test)
        self.use_action_history = bool(use_action_history)

        self.residual_input_dim = 19  # [p(3), v(3), vec(R)(9), u(4)]
        self.policy_input_dim = 13 + 9 + (4 if self.use_action_history else 0)  # [x, x_ref, h]

        self.residual_model = AgileMLP(
            input_dim=self.residual_input_dim,
            hidden_dim=int(residual_hidden_dim),
            num_hidden_layers=int(residual_hidden_layers),
            output_dim=6,
        ).to(device).double()
        self.policy_model = AgileMLP(
            input_dim=self.policy_input_dim,
            hidden_dim=int(policy_hidden_dim),
            num_hidden_layers=int(policy_hidden_layers),
            output_dim=4,  # [delta_c, delta_omega(3)]
        ).to(device).double()

        self.residual_optimizer = optim.Adam(
            self.residual_model.parameters(),
            lr=float(residual_lr),
            weight_decay=float(residual_weight_decay),
        )
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=float(policy_lr))

        self.replay_buffer = deque(maxlen=int(residual_buffer_size))
        self.residual_batch_size = int(residual_batch_size)
        self.residual_train_steps = int(max(0, residual_train_steps))
        self.residual_grad_clip = float(max(0.0, residual_grad_clip))

        self.bptt_horizon = int(max(1, bptt_horizon))
        self.discount = float(np.clip(discount, 0.0, 1.0))
        self.policy_update_every = int(max(1, policy_update_every))
        self.policy_update_steps = int(max(0, policy_update_steps))
        self.policy_grad_clip = float(max(0.0, policy_grad_clip))

        self.w_pos = float(w_pos)
        self.w_vel = float(w_vel)
        self.w_u = float(w_u)
        self.w_du = float(w_du)
        self.control_delta_scale = float(max(0.0, control_delta_scale))
        self.omega_guidance_gain = float(max(0.0, omega_guidance_gain))
        self.omega_bound = float(max(1e-6, omega_bound))
        self.residual_accel_clip = float(max(0.0, residual_accel_clip))
        self.residual_omega_clip = float(max(0.0, residual_omega_clip))

        self.alpha_init = float(alpha_init)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.alpha_lr = float(max(0.0, alpha_lr))
        self.lambda_speed = float(lambda_speed)
        self.lambda_safe = float(lambda_safe)
        self.error_threshold = float(error_threshold)
        self.reset_alpha_each_episode = bool(reset_alpha_each_episode)
        self.reset_buffer_each_episode = bool(reset_buffer_each_episode)

        self.alpha = float(np.clip(self.alpha_init, self.alpha_min, self.alpha_max))

        self.last_residual_loss = np.nan
        self.last_policy_loss = np.nan
        self.last_ats_objective = np.nan
        self.last_ats_grad = np.nan
        self.last_u_cmd = np.array([self.params['g'], 0.0, 0.0, 0.0], dtype=np.float64)
        self.last_f_hat = np.zeros(3, dtype=np.float64)
        self.force_err_list = []
        self.step_counter = 0
        self.prev_transition_state = None
        self.prev_transition_u = None

        self.reset_controller()

    @staticmethod
    def _normalize_quat_np(q):
        q = np.asarray(q, dtype=np.float64).reshape(4)
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return q / n

    @staticmethod
    def _quat_derivative_torch(q, omega):
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        wx, wy, wz = omega[..., 0], omega[..., 1], omega[..., 2]
        qdot = torch.stack(
            (
                -x * wx - y * wy - z * wz,
                w * wx + y * wz - z * wy,
                w * wy + z * wx - x * wz,
                w * wz + x * wy - y * wx,
            ),
            dim=-1,
        )
        return 0.5 * qdot

    @staticmethod
    def _quat_normalize_torch(q):
        return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-12)

    @staticmethod
    def _quat_to_rotmat_torch(q):
        q = AgileAdaptiveController._quat_normalize_torch(q)
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        r00 = 1.0 - 2.0 * (yy + zz)
        r01 = 2.0 * (xy - wz)
        r02 = 2.0 * (xz + wy)
        r10 = 2.0 * (xy + wz)
        r11 = 1.0 - 2.0 * (xx + zz)
        r12 = 2.0 * (yz - wx)
        r20 = 2.0 * (xz - wy)
        r21 = 2.0 * (yz + wx)
        r22 = 1.0 - 2.0 * (xx + yy)
        return torch.stack(
            (
                torch.stack((r00, r01, r02), dim=-1),
                torch.stack((r10, r11, r12), dim=-1),
                torch.stack((r20, r21, r22), dim=-1),
            ),
            dim=-2,
        )

    def _allow_online_updates(self):
        if self.state == 'train':
            return True
        if self.state == 'test' and self.online_update_in_test:
            return True
        return False

    def _state13_from_X(self, X):
        q = self._normalize_quat_np(X[3:7])
        return np.concatenate((X[0:3], X[7:10], q, X[10:13])).astype(np.float64)

    def _split_state13_np(self, x):
        p = x[0:3]
        v = x[3:6]
        q = x[6:10]
        w = x[10:13]
        return p, v, q, w

    def _scaled_reference(self, t, pd, vd, ad, alpha_override=None):
        if self.trajectory_obj is None:
            return pd, vd, ad
        if not self.use_ats:
            pd_u, vd_u, ad_u = self.trajectory_obj(t)
            return np.asarray(pd_u), np.asarray(vd_u), np.asarray(ad_u)
        alpha = float(self.alpha if alpha_override is None else alpha_override)
        alpha = float(np.clip(alpha, self.alpha_min, self.alpha_max))
        tau = t / max(alpha, 1e-6)
        pd_s, vd_s, ad_s = self.trajectory_obj(tau)
        inv_a = 1.0 / max(alpha, 1e-6)
        return np.asarray(pd_s), np.asarray(vd_s) * inv_a, np.asarray(ad_s) * (inv_a ** 2)

    def _nominal_u_np(self, x, ref):
        p, v, q, _ = self._split_state13_np(x)
        pd, vd, ad = ref[0:3], ref[3:6], ref[6:9]
        p_err = p - pd
        v_err = v - vd
        a_cmd = -self.params['K_p'] @ p_err - self.params['K_d'] @ v_err + ad
        F_nom = self.params['m'] * (a_cmd + np.array([0.0, 0.0, self.params['g']]))
        F_norm = max(np.linalg.norm(F_nom), 1e-6)
        c_nom = F_norm / self.params['m']

        R = rowan.to_matrix(q)
        b3 = R[:, 2]
        f_dir = F_nom / F_norm
        omega_nom = self.omega_guidance_gain * np.cross(b3, f_dir)
        u_nom = np.concatenate(([c_nom], omega_nom))
        return self._bound_u_np(u_nom)

    def _bound_u_np(self, u):
        u = np.asarray(u, dtype=np.float64).reshape(4)
        c_min = 0.2 * self.params['g']
        c_max = 3.0 * self.params['g']
        out = u.copy()
        out[0] = float(np.clip(out[0], c_min, c_max))
        out[1:4] = np.clip(out[1:4], -self.omega_bound, self.omega_bound)
        return out

    def _refresh_torch_consts(self):
        self.Kp_t = torch.from_numpy(np.asarray(self.params['K_p'], dtype=np.float64)).to(device).double()
        self.Kd_t = torch.from_numpy(np.asarray(self.params['K_d'], dtype=np.float64)).to(device).double()
        self.g_t = torch.tensor([0.0, 0.0, self.params['g']], dtype=torch.double, device=device).view(1, 3)
        self.mass_t = torch.tensor(self.params['m'], dtype=torch.double, device=device)

    def _build_residual_feature_np(self, x, u):
        p, v, q, _ = self._split_state13_np(x)
        R = rowan.to_matrix(q).reshape(-1)
        zeta = np.concatenate((p, v, R, u)).astype(np.float64)
        return zeta

    def _build_residual_feature_torch(self, x, u):
        p = x[:, 0:3]
        v = x[:, 3:6]
        q = x[:, 6:10]
        R = self._quat_to_rotmat_torch(q).reshape(x.shape[0], 9)
        return torch.cat((p, v, R, u), dim=-1)

    def _predict_residual_np(self, x, u):
        if not self.use_residual:
            return np.zeros(3), np.zeros(3)
        zeta = self._build_residual_feature_np(x, u)
        zeta_t = torch.from_numpy(zeta).double().unsqueeze(0)
        self.residual_model.eval()
        with torch.no_grad():
            out = self.residual_model(zeta_t).cpu().numpy().reshape(-1)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        a_res = np.clip(out[0:3], -self.residual_accel_clip, self.residual_accel_clip)
        omega_res = np.clip(out[3:6], -self.residual_omega_clip, self.residual_omega_clip)
        return a_res, omega_res

    def _build_policy_input_np(self, x, ref):
        if self.use_action_history:
            return np.concatenate((x, ref, self.last_u_cmd)).astype(np.float64)
        return np.concatenate((x, ref)).astype(np.float64)

    def _infer_policy_u_np(self, x, ref):
        u_nom = self._nominal_u_np(x, ref)
        if not self.use_anchor:
            return u_nom
        pol_in = self._build_policy_input_np(x, ref)
        pol_t = torch.from_numpy(pol_in).double().unsqueeze(0)
        self.policy_model.eval()
        with torch.no_grad():
            raw = self.policy_model(pol_t).cpu().numpy().reshape(-1)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        delta = self.control_delta_scale * np.tanh(raw)
        return self._bound_u_np(u_nom + delta)

    def _nominal_u_torch(self, x, ref):
        p = x[:, 0:3]
        v = x[:, 3:6]
        q = x[:, 6:10]
        pd = ref[:, 0:3]
        vd = ref[:, 3:6]
        ad = ref[:, 6:9]

        p_err = p - pd
        v_err = v - vd
        a_cmd = -torch.matmul(p_err, self.Kp_t.T) - torch.matmul(v_err, self.Kd_t.T) + ad
        F_nom = self.mass_t * (a_cmd + self.g_t)
        F_norm = torch.linalg.norm(F_nom, dim=-1, keepdim=True).clamp_min(1e-6)
        c_nom = F_norm / self.mass_t

        R = self._quat_to_rotmat_torch(q)
        b3 = R[:, :, 2]
        f_dir = F_nom / F_norm
        omega_nom = self.omega_guidance_gain * torch.cross(b3, f_dir, dim=-1)
        u_nom = torch.cat((c_nom, omega_nom), dim=-1)
        return self._bound_u_torch(u_nom)

    def _bound_u_torch(self, u):
        c_min = 0.2 * self.params['g']
        c_max = 3.0 * self.params['g']
        c = torch.clamp(u[:, 0:1], c_min, c_max)
        w = torch.clamp(u[:, 1:4], -self.omega_bound, self.omega_bound)
        return torch.cat((c, w), dim=-1)

    def _hybrid_step_torch(self, x, u, dt, enable_residual=True):
        if isinstance(dt, float) or isinstance(dt, int):
            dt_t = torch.full((x.shape[0], 1), float(dt), dtype=x.dtype, device=x.device)
        else:
            dt_t = dt
            if dt_t.ndim == 1:
                dt_t = dt_t.unsqueeze(-1)

        p = x[:, 0:3]
        v = x[:, 3:6]
        q = x[:, 6:10]

        c = u[:, 0:1]
        omega_cmd = u[:, 1:4]

        if enable_residual and self.use_residual:
            zeta = self._build_residual_feature_torch(x, u)
            res = self.residual_model(zeta)
            a_res = torch.clamp(res[:, 0:3], -self.residual_accel_clip, self.residual_accel_clip)
            omega_res = torch.clamp(res[:, 3:6], -self.residual_omega_clip, self.residual_omega_clip)
        else:
            a_res = torch.zeros((x.shape[0], 3), dtype=x.dtype, device=x.device)
            omega_res = torch.zeros((x.shape[0], 3), dtype=x.dtype, device=x.device)

        R = self._quat_to_rotmat_torch(q)
        b3 = R[:, :, 2]
        v_dot = c * b3 - self.g_t + a_res
        omega_eff = torch.clamp(omega_cmd + omega_res, -self.omega_bound, self.omega_bound)
        q_dot = self._quat_derivative_torch(q, omega_eff)

        p_next = p + dt_t * v
        v_next = v + dt_t * v_dot
        q_next = self._quat_normalize_torch(q + dt_t * q_dot)
        w_next = omega_eff
        x_next = torch.cat((p_next, v_next, q_next, w_next), dim=-1)
        return x_next, a_res, omega_res

    def _push_transition(self, x_prev, u_prev, x_next, dt):
        if (not np.all(np.isfinite(x_prev))) or (not np.all(np.isfinite(u_prev))) or (not np.all(np.isfinite(x_next))):
            return
        dt = float(dt)
        if (not np.isfinite(dt)) or dt <= 0:
            return
        self.replay_buffer.append((x_prev.copy(), u_prev.copy(), x_next.copy(), dt))

    def _train_residual_once(self):
        if (not self.use_residual) or len(self.replay_buffer) < self.residual_batch_size:
            return np.nan
        batch = random.sample(self.replay_buffer, self.residual_batch_size)
        x_b = np.asarray([b[0] for b in batch], dtype=np.float64)
        u_b = np.asarray([b[1] for b in batch], dtype=np.float64)
        xn_b = np.asarray([b[2] for b in batch], dtype=np.float64)
        dt_b = np.asarray([b[3] for b in batch], dtype=np.float64)

        x_t = torch.from_numpy(x_b).to(device).double()
        u_t = torch.from_numpy(u_b).to(device).double()
        xn_t = torch.from_numpy(xn_b).to(device).double()
        dt_t = torch.from_numpy(dt_b).to(device).double().unsqueeze(-1)

        self.residual_model.train()
        self.residual_optimizer.zero_grad()
        xn_hat, _, _ = self._hybrid_step_torch(x_t, u_t, dt_t, enable_residual=True)
        loss_p = F.mse_loss(xn_hat[:, 0:3], xn_t[:, 0:3])
        loss_v = F.mse_loss(xn_hat[:, 3:6], xn_t[:, 3:6])
        q_hat = self._quat_normalize_torch(xn_hat[:, 6:10])
        q_tgt = self._quat_normalize_torch(xn_t[:, 6:10])
        dot = torch.sum(q_hat * q_tgt, dim=-1).abs().clamp(max=1.0)
        loss_q = torch.mean(1.0 - dot ** 2)
        loss_w = F.mse_loss(xn_hat[:, 10:13], xn_t[:, 10:13])
        loss = loss_p + loss_v + 0.5 * loss_q + 0.1 * loss_w
        if not torch.isfinite(loss):
            return np.nan
        loss.backward()
        if self.residual_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.residual_model.parameters(), self.residual_grad_clip)
        self.residual_optimizer.step()
        self.residual_model.eval()
        return float(loss.item())

    def _rollout_tracking_cost(self, x0_np, t0, alpha_value):
        x = torch.from_numpy(np.asarray(x0_np, dtype=np.float64)).to(device).double().unsqueeze(0)
        prev_u = torch.from_numpy(np.asarray(self.last_u_cmd, dtype=np.float64)).to(device).double().unsqueeze(0)
        dt_roll = max(self.params['dt_posctrl'], 1e-4)
        gamma = 1.0
        cost = torch.zeros(1, dtype=torch.double, device=device)
        barrier = torch.zeros(1, dtype=torch.double, device=device)
        for k in range(self.bptt_horizon):
            tk = t0 + k * dt_roll
            pd_k, vd_k, ad_k = self._scaled_reference(
                tk,
                pd=np.zeros(3),
                vd=np.zeros(3),
                ad=np.zeros(3),
                alpha_override=alpha_value,
            )
            ref_np = np.concatenate((pd_k, vd_k, ad_k)).astype(np.float64)
            ref_t = torch.from_numpy(ref_np).to(device).double().unsqueeze(0)
            u_nom = self._nominal_u_torch(x, ref_t)
            if self.use_anchor:
                if self.use_action_history:
                    pol_in = torch.cat((x, ref_t, prev_u), dim=-1)
                else:
                    pol_in = torch.cat((x, ref_t), dim=-1)
                du = self.control_delta_scale * torch.tanh(self.policy_model(pol_in))
                u = self._bound_u_torch(u_nom + du)
            else:
                u = u_nom

            x, _, _ = self._hybrid_step_torch(x, u, dt_roll, enable_residual=True)
            pos_err = x[:, 0:3] - ref_t[:, 0:3]
            vel_err = x[:, 3:6] - ref_t[:, 3:6]
            step_cost = (
                self.w_pos * torch.sum(pos_err ** 2, dim=-1)
                + self.w_vel * torch.sum(vel_err ** 2, dim=-1)
                + self.w_u * torch.sum(u ** 2, dim=-1)
                + self.w_du * torch.sum((u - prev_u) ** 2, dim=-1)
            )
            cost = cost + gamma * torch.mean(step_cost)
            tracking_err = torch.linalg.norm(pos_err, dim=-1)
            barrier = barrier + torch.mean(F.softplus(tracking_err - self.error_threshold))
            gamma *= self.discount
            prev_u = u
        return cost.squeeze(0), barrier.squeeze(0)

    def _update_policy_from_anchor(self, x_now, t_now):
        if (not self.use_anchor) or self.policy_update_steps <= 0:
            return np.nan
        self.policy_model.train()
        last_loss = np.nan
        for _ in range(self.policy_update_steps):
            self.policy_optimizer.zero_grad()
            cost, _ = self._rollout_tracking_cost(x_now, t_now, self.alpha)
            if not torch.isfinite(cost):
                continue
            cost.backward()
            if self.policy_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.policy_grad_clip)
            self.policy_optimizer.step()
            last_loss = float(cost.item())
        self.policy_model.eval()
        return last_loss

    def _ats_objective(self, x_now, t_now, alpha_value):
        with torch.no_grad():
            _, barrier = self._rollout_tracking_cost(x_now, t_now, alpha_value)
            return float(self.lambda_speed * alpha_value + self.lambda_safe * float(barrier.item()))

    def _update_alpha(self, x_now, t_now):
        if (not self.use_ats) or self.alpha_lr <= 0:
            return
        alpha = float(np.clip(self.alpha, self.alpha_min, self.alpha_max))
        eps = max(1e-3, 0.02 * alpha)
        a_plus = float(np.clip(alpha + eps, self.alpha_min, self.alpha_max))
        a_minus = float(np.clip(alpha - eps, self.alpha_min, self.alpha_max))
        if abs(a_plus - a_minus) < 1e-9:
            return
        j_plus = self._ats_objective(x_now, t_now, a_plus)
        j_minus = self._ats_objective(x_now, t_now, a_minus)
        grad = (j_plus - j_minus) / (a_plus - a_minus)
        if np.isfinite(grad):
            alpha_new = alpha - self.alpha_lr * grad
            self.alpha = float(np.clip(alpha_new, self.alpha_min, self.alpha_max))
            self.last_ats_grad = float(grad)
            self.last_ats_objective = self._ats_objective(x_now, t_now, self.alpha)

    def reset_controller(self):
        super().reset_controller()
        self._refresh_torch_consts()
        self.step_counter = 0
        self.prev_transition_state = None
        self.prev_transition_u = None
        if self.reset_alpha_each_episode:
            self.alpha = float(np.clip(self.alpha_init, self.alpha_min, self.alpha_max))
        else:
            self.alpha = float(np.clip(self.alpha, self.alpha_min, self.alpha_max))
        if self.reset_buffer_each_episode:
            self.replay_buffer.clear()
        self.last_residual_loss = np.nan
        self.last_policy_loss = np.nan
        self.last_ats_objective = np.nan
        self.last_ats_grad = np.nan
        self.last_u_cmd = np.array([self.params['g'], 0.0, 0.0, 0.0], dtype=np.float64)
        self.last_f_hat = np.zeros(3, dtype=np.float64)
        self.force_err_list = []

    def position(self, X, Z, imu, pd, vd, ad, last_wind_update, t, wind_gt):
        if self.t_last is None:
            self.dt = self.params['dt_posctrl']
        else:
            self.dt = max(t - self.t_last, 1e-6)
        self.t_last = t

        x_now = self._state13_from_X(X)
        if self.prev_transition_state is not None and self.prev_transition_u is not None:
            self._push_transition(self.prev_transition_state, self.prev_transition_u, x_now, self.dt)

        self.step_counter += 1
        if self._allow_online_updates() and (self.step_counter % self.policy_update_every == 0):
            if self.use_residual and self.residual_train_steps > 0 and len(self.replay_buffer) >= self.residual_batch_size:
                losses = []
                for _ in range(self.residual_train_steps):
                    lv = self._train_residual_once()
                    if np.isfinite(lv):
                        losses.append(lv)
                if len(losses) > 0:
                    self.last_residual_loss = float(np.mean(losses))
            if self.use_anchor:
                self.last_policy_loss = self._update_policy_from_anchor(x_now, t)
            if self.use_ats:
                self._update_alpha(x_now, t)

        pd_use, vd_use, ad_use = self._scaled_reference(t, pd, vd, ad)
        Fr_nom, Fr_dot = super().get_Fr(
            X, Z=Z, imu=imu, pd=pd_use, vd=vd_use, ad=ad_use, meta_adapt_trigger=False, wind_gt=wind_gt
        )
        ref_vec = np.concatenate((pd_use, vd_use, ad_use)).astype(np.float64)
        u_cmd = self._infer_policy_u_np(x_now, ref_vec)
        a_res, omega_res = self._predict_residual_np(x_now, u_cmd)

        Fr_dir = Fr_nom / max(np.linalg.norm(Fr_nom), 1e-6)
        F_policy = self.params['m'] * u_cmd[0] * Fr_dir
        F_cmd = F_policy - self.params['m'] * a_res
        F_cmd = np.nan_to_num(F_cmd, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(F_cmd)) or np.linalg.norm(F_cmd) < 1e-9:
            F_cmd = Fr_nom.copy()

        self.last_f_hat = self.params['m'] * a_res
        self.last_u_cmd = u_cmd.copy()
        if wind_gt is not None and np.all(np.isfinite(wind_gt)):
            self.force_err_list.append(float(np.linalg.norm(self.last_f_hat - wind_gt)))

        T_cmd = np.linalg.norm(F_cmd + self.params['thrust_delay'] * Fr_dot)
        q_cmd = self.get_q(F_cmd + self.params['attitude_delay'] * Fr_dot, yaw=0.0)
        omega_eff = np.clip(u_cmd[1:4] + omega_res, -self.omega_bound, self.omega_bound)
        omega_norm = np.linalg.norm(omega_eff)
        if omega_norm > 1e-8:
            axis = omega_eff / omega_norm
            dtheta = float(np.clip(omega_norm * max(self.dt, 1e-4), 0.0, 0.5))
            dq = rowan.from_axis_angle(axis, dtheta)
            q_cmd = rowan.normalize(rowan.multiply(q_cmd, dq))

        self.prev_transition_state = x_now.copy()
        self.prev_transition_u = u_cmd.copy()
        return T_cmd, q_cmd

    def save(self, path):
        state = {
            'residual_model': self.residual_model.state_dict(),
            'policy_model': self.policy_model.state_dict(),
            'alpha': float(self.alpha),
            'use_residual': bool(self.use_residual),
            'use_anchor': bool(self.use_anchor),
            'use_ats': bool(self.use_ats),
        }
        torch.save(state, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        if 'residual_model' in ckpt:
            self.residual_model.load_state_dict(ckpt['residual_model'])
        if 'policy_model' in ckpt:
            self.policy_model.load_state_dict(ckpt['policy_model'])
        if 'alpha' in ckpt:
            self.alpha = float(np.clip(float(ckpt['alpha']), self.alpha_min, self.alpha_max))
        self.residual_model.eval()
        self.policy_model.eval()

class CausalTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.pad,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv(x)
        if self.pad > 0:
            y = y[:, :, :-self.pad]
        y = self.bn(y)
        y = self.relu(y)
        y = self.dropout(y)
        return y

class PITCNDynamicsModel(nn.Module):
    def __init__(
        self,
        input_dim=14,
        tcn_hidden_dim=16,
        tcn_num_layers=4,
        tcn_dropout=0.1,
        mlp_hidden_dims=(64, 32, 32),
        encoder_type='tcn',
        predict_residual=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.encoder_type = encoder_type
        self.predict_residual = predict_residual
        self.output_dim = 6

        if encoder_type == 'tcn':
            blocks = []
            in_ch = input_dim
            for i in range(tcn_num_layers):
                dilation = 2 ** i
                blocks.append(
                    CausalTemporalBlock(
                        in_channels=in_ch,
                        out_channels=tcn_hidden_dim,
                        kernel_size=3,
                        dilation=dilation,
                        dropout=tcn_dropout,
                    )
                )
                in_ch = tcn_hidden_dim
            self.encoder = nn.Sequential(*blocks)
            decoder_in = tcn_hidden_dim
        elif encoder_type == 'mlp':
            self.encoder = nn.Identity()
            decoder_in = input_dim
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        dims = [decoder_in] + list(mlp_hidden_dims) + [self.output_dim]
        decoder = []
        for i in range(len(dims) - 2):
            decoder.append(nn.Linear(dims[i], dims[i+1]))
            decoder.append(nn.ReLU())
        decoder.append(nn.Linear(dims[-2], dims[-1]))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, hist, dyn_nom=None):
        if hist.dim() != 3:
            raise ValueError(f"Expected hist shape (B,T,F), got {tuple(hist.shape)}")
        if hist.size(-1) != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {hist.size(-1)}")

        if self.encoder_type == 'tcn':
            x = hist.transpose(1, 2)  # (B,F,T)
            z = self.encoder(x)
            feat = z[:, :, -1]
        else:
            feat = hist[:, -1, :]

        dyn_raw = self.decoder(feat)
        if self.predict_residual:
            if dyn_nom is None:
                raise ValueError("dyn_nom is required when predict_residual=True")
            dyn_pred = dyn_nom + dyn_raw
        else:
            dyn_pred = dyn_raw

        if not torch.all(torch.isfinite(dyn_pred)):
            raise ValueError("Non-finite PI-TCN forward output detected.")

        return {
            'accel_pred': dyn_pred[:, 0:3],
            'ang_accel_pred': dyn_pred[:, 3:6],
            'dyn_pred': dyn_pred,
            'dyn_raw': dyn_raw,
        }

    def save(self, path):
        state = {
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'encoder_type': self.encoder_type,
            'predict_residual': self.predict_residual,
        }
        torch.save(state, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.load_state_dict(ckpt['state_dict'])

class PITCNController(PIDController):
    def __init__(
        self,
        pitcn_model=None,
        history_len=20,
        compensation_gain=0.05,
        force_bound=20.0,
        use_compensation=True,
        given_pid=False,
        p=0,
        i=0,
        d=0,
    ):
        super().__init__(given_pid=given_pid, p=p, i=i, d=d)
        self.pitcn_model = pitcn_model
        self.history_len = int(max(1, history_len))
        self.compensation_gain = float(compensation_gain)
        self.force_bound = float(force_bound)
        self.use_compensation = bool(use_compensation) and (pitcn_model is not None)
        self.motor_speed = np.zeros(4)
        self.force_err_list = []
        if self.pitcn_model is not None:
            self.pitcn_model.eval()

    def reset_controller(self):
        super().reset_controller()
        self.motor_speed = np.zeros(4)
        self.hist_feat = []
        self.last_f_hat = np.zeros(3)
        self.force_err_list = []

    def mixer(self, torque_sp, T_sp):
        self.motor_speed = super().mixer(torque_sp, T_sp)
        return self.motor_speed

    def _build_step_feature(self, X, Z):
        # [v(3), q(4), omega(3), u(4)] -> 14D
        return np.concatenate((X[7:10], X[3:7], X[10:13], Z))

    def _nominal_dyn(self, X, Z):
        motor_sq = np.clip(
            Z ** 2,
            self.params['motor_min_speed'] ** 2,
            self.params['motor_max_speed'] ** 2,
        )
        eta = self.B @ motor_sq
        T = eta[0]
        tau = eta[1:4]
        q = X[3:7]
        q = q / max(np.linalg.norm(q), 1e-12)
        R = rowan.to_matrix(q)
        e3 = np.array([0., 0., 1.])
        m = self.params['m']
        g = self.params['g']
        v_dot_nom = (T * (R @ e3) - np.array([0., 0., m * g])) / m
        w = X[10:13]
        J = np.array(self.params['J'])
        alpha_nom = np.linalg.solve(J, np.cross(J @ w, w) + tau)
        return np.concatenate((v_dot_nom, alpha_nom))

    def _predict_force(self, X, Z):
        if (not self.use_compensation) or (self.pitcn_model is None):
            return np.zeros(3)

        step_feat = self._build_step_feature(X, Z)
        self.hist_feat.append(step_feat.copy())
        if len(self.hist_feat) > self.history_len:
            self.hist_feat = self.hist_feat[-self.history_len:]
        if len(self.hist_feat) < self.history_len:
            return np.zeros(3)

        hist = np.asarray(self.hist_feat, dtype=float)
        dyn_nom = self._nominal_dyn(X, Z)
        hist_t = torch.from_numpy(hist).double().unsqueeze(0)

        with torch.no_grad():
            if getattr(self.pitcn_model, 'predict_residual', False):
                out = self.pitcn_model(hist_t, dyn_nom=torch.from_numpy(dyn_nom).double().unsqueeze(0))
            else:
                out = self.pitcn_model(hist_t)
            dyn_pred = out['dyn_pred'].cpu().numpy().reshape(-1)

        if dyn_pred.shape[0] != 6 or (not np.all(np.isfinite(dyn_pred))):
            return np.zeros(3)
        a_pred = dyn_pred[0:3]
        a_nom = dyn_nom[0:3]
        f_hat = self.params['m'] * (a_pred - a_nom)
        f_hat = np.nan_to_num(f_hat, nan=0.0, posinf=0.0, neginf=0.0)
        f_hat = np.clip(f_hat, -self.force_bound, self.force_bound)
        return f_hat

    def get_Fr(self, X, Z, imu, pd, vd, ad, meta_adapt_trigger, wind_gt):
        Fr, Fr_dot = super().get_Fr(
            X, Z=Z, imu=imu, pd=pd, vd=vd, ad=ad,
            meta_adapt_trigger=meta_adapt_trigger, wind_gt=wind_gt
        )
        f_hat = self._predict_force(X, Z)
        self.last_f_hat = f_hat.copy()
        if wind_gt is not None and np.all(np.isfinite(wind_gt)):
            self.force_err_list.append(float(np.linalg.norm(f_hat - wind_gt)))
        return Fr - self.compensation_gain * f_hat, Fr_dot


class PINNsFormerController(PIDController):
    """PID base + PINNsFormer-inspired physics-informed dynamics compensator."""

    def __init__(
        self,
        seq_len=5,
        d_model=32,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ff_dim=128,
        dropout=0.1,
        use_layernorm=False,
        use_time_feature=True,
        force_bound=200.0,
        compensation_gain=0.01,
        given_pid=False,
        p=0,
        i=0,
        d=0,
    ):
        """Initialize PINNsFormer controller.

        Input sequence shape: (B, T, 14), with per-step feature [v(3), q(4), w(3), u(4)].
        Model output shape: (B, T, 6), where last token is current dynamics prediction.
        """
        super().__init__(given_pid=given_pid, p=p, i=i, d=d)
        self.device = device
        self.dtype = torch.double
        self.seq_len = int(max(1, seq_len))
        self.feature_dim = 14
        self.output_dim = 6
        self.force_bound = float(force_bound)
        self.compensation_gain = float(compensation_gain)

        self.model = PINNsFormerDynamicsModel(
            input_dim=self.feature_dim,
            seq_len=self.seq_len,
            d_model=int(d_model),
            n_heads=int(n_heads),
            n_encoder_layers=int(n_encoder_layers),
            n_decoder_layers=int(n_decoder_layers),
            ff_dim=int(ff_dim),
            dropout=float(dropout),
            out_dim=self.output_dim,
            use_layernorm=bool(use_layernorm),
            use_time_feature=bool(use_time_feature),
        ).to(self.device).double()

        self.feature_mean = np.zeros(self.feature_dim, dtype=np.float64)
        self.feature_std = np.ones(self.feature_dim, dtype=np.float64)
        self.reset_controller()

    def reset_controller(self):
        """Reset controller runtime buffers."""
        super().reset_controller()
        self.hist_features = []
        self.last_pred = np.zeros(self.output_dim, dtype=np.float64)
        self.last_f_hat = np.zeros(3, dtype=np.float64)
        self.last_motor_speed = np.zeros(4, dtype=np.float64)
        self.force_err_list = []

    def mixer(self, torque_sp, T_sp):
        """Run base mixer and cache latest motor command."""
        self.last_motor_speed = super().mixer(torque_sp, T_sp)
        return self.last_motor_speed

    def build_feature(self, X, Z, imu=None, pd=None, vd=None, ad=None):
        """Build one token feature at position-loop tick.

        Returns:
            feature_t: np.ndarray with shape (14,).
        """
        u_feat = self.last_motor_speed if self.last_motor_speed is not None else Z
        return np.concatenate((X[7:10], X[3:7], X[10:13], u_feat)).astype(np.float64, copy=False)

    def _normalize_hist(self, hist):
        """Normalize history features using train split statistics."""
        std = np.where(self.feature_std > 1e-9, self.feature_std, 1.0)
        return (hist - self.feature_mean) / std

    def _nominal_dyn(self, X, Z):
        """Compute nominal 6D dynamics [v_dot_nom(3), w_dot_nom(3)] for current state."""
        motor_sq = np.clip(
            Z ** 2,
            self.params['motor_min_speed'] ** 2,
            self.params['motor_max_speed'] ** 2,
        )
        eta = self.B @ motor_sq
        T = eta[0]
        tau = eta[1:4]

        q = X[3:7]
        q = q / max(np.linalg.norm(q), 1e-12)
        R = rowan.to_matrix(q)
        e3 = np.array([0., 0., 1.])
        m = self.params['m']
        g = self.params['g']
        v_dot_nom = (T * (R @ e3) - np.array([0., 0., m * g])) / m

        w = X[10:13]
        J = np.array(self.params['J'])
        w_dot_nom = np.linalg.solve(J, np.cross(J @ w, w) + tau)
        return np.concatenate((v_dot_nom, w_dot_nom))

    def predict_current(self, X, Z):
        """Predict current dynamics and force compensation from history.

        Returns:
            dyn_pred_cur: np.ndarray, shape (6,)
            f_hat: np.ndarray, shape (3,)
        """
        if len(self.hist_features) < self.seq_len:
            return np.zeros(self.output_dim, dtype=np.float64), np.zeros(3, dtype=np.float64)

        hist = np.asarray(self.hist_features[-self.seq_len:], dtype=np.float64)
        hist = self._normalize_hist(hist)
        hist_t = torch.from_numpy(hist).to(self.device).double().unsqueeze(0)  # (1,T,F)

        self.model.eval()
        with torch.no_grad():
            out = self.model(hist_t)
            dyn_pred = out['dyn_pred'].cpu().numpy().reshape(-1)

        if dyn_pred.shape[0] != self.output_dim or (not np.all(np.isfinite(dyn_pred))):
            return np.zeros(self.output_dim, dtype=np.float64), np.zeros(3, dtype=np.float64)

        dyn_nom = self._nominal_dyn(X, Z)
        a_pred = dyn_pred[0:3]
        a_nom = dyn_nom[0:3]
        f_hat = self.params['m'] * (a_pred - a_nom)
        f_hat = np.nan_to_num(f_hat, nan=0.0, posinf=0.0, neginf=0.0)
        f_hat = np.clip(f_hat, -self.force_bound, self.force_bound)
        return dyn_pred, f_hat

    def get_Fr(self, X, Z, imu, pd, vd, ad, meta_adapt_trigger, wind_gt):
        """Force compensation at position-loop update."""
        Fr_nom, Fr_dot = super().get_Fr(
            X, Z=Z, imu=imu, pd=pd, vd=vd, ad=ad,
            meta_adapt_trigger=meta_adapt_trigger, wind_gt=wind_gt,
        )

        feat = self.build_feature(X, Z=Z, imu=imu, pd=pd, vd=vd, ad=ad)
        self.hist_features.append(feat.copy())
        if len(self.hist_features) > self.seq_len:
            self.hist_features = self.hist_features[-self.seq_len:]

        self.last_pred, self.last_f_hat = self.predict_current(X, Z)

        if wind_gt is not None and np.all(np.isfinite(wind_gt)):
            self.force_err_list.append(float(np.linalg.norm(self.last_f_hat - wind_gt)))

        Fr_comp = Fr_nom - self.compensation_gain * self.last_f_hat
        return Fr_comp, Fr_dot

    def fit_dynamics_model(
        self,
        train_hist,
        train_label_cur,
        train_nom_seq,
        val_hist=None,
        val_label_cur=None,
        val_nom_seq=None,
        epochs=100,
        batch_size=64,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        lambda_data=1.0,
        lambda_phys=1.0,
        lambda_anchor=1.0,
    ):
        """Train PINNsFormer with sequential physics-informed loss.

        Args:
            train_hist: (N, T, F)
            train_label_cur: (N, 6)
            train_nom_seq: (N, T, 6)
        """
        if len(train_hist) == 0:
            raise RuntimeError("Empty PINNsFormer training set.")
        if train_hist.ndim != 3 or train_hist.shape[1] != self.seq_len or train_hist.shape[2] != self.feature_dim:
            raise ValueError(f"Expected train_hist shape (N,{self.seq_len},{self.feature_dim}), got {train_hist.shape}")
        if train_label_cur.ndim != 2 or train_label_cur.shape[1] != self.output_dim:
            raise ValueError(f"Expected train_label_cur shape (N,{self.output_dim}), got {train_label_cur.shape}")
        if train_nom_seq.ndim != 3 or train_nom_seq.shape[1] != self.seq_len or train_nom_seq.shape[2] != self.output_dim:
            raise ValueError(f"Expected train_nom_seq shape (N,{self.seq_len},{self.output_dim}), got {train_nom_seq.shape}")

        flat = train_hist.reshape(-1, self.feature_dim)
        self.feature_mean = np.mean(flat, axis=0).astype(np.float64)
        self.feature_std = np.std(flat, axis=0).astype(np.float64)
        self.feature_std = np.where(self.feature_std > 1e-9, self.feature_std, 1.0)

        train_hist_n = self._normalize_hist(train_hist)

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        mse = nn.MSELoss()
        n = len(train_hist_n)

        for epoch in range(int(epochs)):
            perm = np.random.permutation(n)
            total_loss = 0.0
            total_data = 0.0
            total_phys = 0.0
            total_anchor = 0.0
            total_count = 0

            for st in range(0, n, int(batch_size)):
                idx = perm[st:st + int(batch_size)]
                x = torch.from_numpy(train_hist_n[idx]).to(self.device).double()
                y_cur = torch.from_numpy(train_label_cur[idx]).to(self.device).double()
                y_nom_seq = torch.from_numpy(train_nom_seq[idx]).to(self.device).double()

                out = self.model(x)
                y_seq = out['dyn_seq_pred']   # (B,T,6)
                y_last = out['dyn_pred']      # (B,6)

                l_data_last = mse(y_last, y_cur)
                l_phys_seq = mse(y_seq, y_nom_seq)
                l_anchor = mse(y_last, y_cur)
                loss = float(lambda_data) * l_data_last + float(lambda_phys) * l_phys_seq + float(lambda_anchor) * l_anchor

                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss encountered in PINNsFormer training.")

                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None and grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                optimizer.step()

                bsz = len(idx)
                total_loss += float(loss.item()) * bsz
                total_data += float(l_data_last.item()) * bsz
                total_phys += float(l_phys_seq.item()) * bsz
                total_anchor += float(l_anchor.item()) * bsz
                total_count += bsz

            train_loss = total_loss / max(total_count, 1)
            train_data = total_data / max(total_count, 1)
            train_phys = total_phys / max(total_count, 1)
            train_anchor = total_anchor / max(total_count, 1)
            msg = (
                f"[PINNsFormer][Epoch {epoch+1}/{epochs}] "
                f"loss={train_loss:.6f} data={train_data:.6f} phys={train_phys:.6f} anchor={train_anchor:.6f}"
            )
            if val_hist is not None and val_label_cur is not None and val_nom_seq is not None and len(val_hist) > 0:
                val_metrics = self.eval_dynamics_model(val_hist, val_label_cur, val_nom_seq)
                msg += f" val_rmse_total={val_metrics['rmse_total']:.6f}"
            print(msg)

        self.model.eval()

    def eval_dynamics_model(self, hist, label_cur, nom_seq):
        """Evaluate current-step dynamics prediction."""
        if len(hist) == 0:
            return {'rmse_lin': np.nan, 'rmse_ang': np.nan, 'rmse_total': np.nan}

        hist_n = self._normalize_hist(hist)
        x = torch.from_numpy(hist_n).to(self.device).double()
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
            pred = out['dyn_pred'].cpu().numpy()

        err = pred - label_cur
        rmse_lin = float(np.sqrt(np.mean(err[:, 0:3] ** 2)))
        rmse_ang = float(np.sqrt(np.mean(err[:, 3:6] ** 2)))
        rmse_total = float(np.sqrt(np.mean(err ** 2)))
        return {'rmse_lin': rmse_lin, 'rmse_ang': rmse_ang, 'rmse_total': rmse_total}

    def save(self, path):
        """Save PINNsFormer checkpoint and normalizer stats."""
        state = {
            'pinnsformer_model': self.model.state_dict(),
            'seq_len': self.seq_len,
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
        }
        torch.save(state, path)

    def load(self, path, map_location=None):
        """Load PINNsFormer checkpoint and optional normalizer stats."""
        ckpt = torch.load(path, map_location=map_location)
        if isinstance(ckpt, dict) and 'pinnsformer_model' in ckpt:
            state_dict = ckpt['pinnsformer_model']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()

        if isinstance(ckpt, dict):
            if 'feature_mean' in ckpt:
                self.feature_mean = np.asarray(ckpt['feature_mean'], dtype=np.float64)
            if 'feature_std' in ckpt:
                std = np.asarray(ckpt['feature_std'], dtype=np.float64)
                self.feature_std = np.where(std > 1e-9, std, 1.0)

class DMRACFeatureNet(nn.Module):
    def __init__(self, input_dim, feature_dim=20, hidden_dim=64, num_hidden_layers=3, dropout_p=0.0):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * num_hidden_layers + [feature_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MANNFeatureNet(nn.Module):
    """Two-layer sigmoid MLP used for MANN hidden features."""

    def __init__(self, input_dim, hidden_dim=64, feature_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        h = torch.sigmoid(self.fc1(x))
        return torch.sigmoid(self.fc2(h))

class DMRACReplayBuffer:
    def __init__(
        self,
        capacity,
        state_dim,
        action_dim,
        feature_dim,
        zeta_tol=1e-2,
        novelty_mode='feature',
        prune_mode='fifo',
    ):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.feature_dim = int(feature_dim)
        self.zeta_tol = float(zeta_tol)
        self.novelty_mode = novelty_mode
        self.prune_mode = prune_mode
        self.states = []
        self.labels = []
        self.features = []
        self.w_snapshots = []

    def __len__(self):
        return len(self.states)

    def _distance_to_set(self, vec, vec_set):
        if len(vec_set) == 0:
            return np.inf
        mat = np.stack(vec_set, axis=0)
        return float(np.min(np.linalg.norm(mat - vec[None, :], axis=1)))

    def _should_add(self, x, phi):
        if len(self.states) == 0:
            return True
        if self.novelty_mode == 'feature' and phi is not None:
            d = self._distance_to_set(phi, self.features)
        else:
            d = self._distance_to_set(x, self.states)
        return d > self.zeta_tol

    def _prune_once(self):
        if len(self.states) <= self.capacity:
            return
        if self.prune_mode == 'redundant_nn' and len(self.states) >= 3:
            feats = np.stack(self.features if len(self.features) == len(self.states) else self.states, axis=0)
            dmat = np.linalg.norm(feats[:, None, :] - feats[None, :, :], axis=2)
            np.fill_diagonal(dmat, np.inf)
            idx = int(np.argmin(np.min(dmat, axis=1)))
        else:
            idx = 0
        self.states.pop(idx)
        self.labels.pop(idx)
        if len(self.features) > idx:
            self.features.pop(idx)
        if len(self.w_snapshots) > idx:
            self.w_snapshots.pop(idx)

    def add(self, x, y, phi=None, w_snapshot=None):
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        if x.shape[0] != self.state_dim or y.shape[0] != self.action_dim:
            return False
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return False
        if phi is not None:
            phi = np.asarray(phi).reshape(-1)
            if phi.shape[0] != self.feature_dim or not np.all(np.isfinite(phi)):
                phi = None

        if not self._should_add(x, phi):
            return False

        self.states.append(x.copy())
        self.labels.append(y.copy())
        self.features.append(phi.copy() if phi is not None else x.copy())
        if w_snapshot is not None:
            self.w_snapshots.append(np.asarray(w_snapshot).copy())
        else:
            self.w_snapshots.append(None)
        self._prune_once()
        return True

    def sample(self, batch_size):
        if len(self.states) == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        n = min(int(batch_size), len(self.states))
        idx = np.random.randint(0, len(self.states), size=n)
        x = np.stack([self.states[i] for i in idx], axis=0)
        y = np.stack([self.labels[i] for i in idx], axis=0)
        w_list = [self.w_snapshots[i] for i in idx]
        has_w = all(w is not None for w in w_list)
        if has_w:
            w = np.stack(w_list, axis=0)
        else:
            w = None
        return x, y, w

    def export_npz(self, path):
        if len(self.states) == 0:
            np.savez(path, x=np.empty((0, self.state_dim)), y=np.empty((0, self.action_dim)))
            return
        x = np.stack(self.states, axis=0)
        y = np.stack(self.labels, axis=0)
        has_w = all(w is not None for w in self.w_snapshots)
        if has_w:
            w = np.stack(self.w_snapshots, axis=0)
            np.savez(path, x=x, y=y, w=w)
        else:
            np.savez(path, x=x, y=y)

class DMRACQuadController(PIDController):
    def __init__(
        self,
        given_pid=False,
        p=0,
        i=0,
        d=0,
        model_name='dmrac',
        feature_dim=20,
        hidden_dim=64,
        hidden_layers=3,
        dropout_p=0.0,
        gamma_adapt=5.0,
        projection_type='elementwise_clip',
        w_bound=50.0,
        q_diag=None,
        wn=1.4,
        zeta=0.9,
        buffer_capacity=5000,
        zeta_tol=0.0,
        novelty_mode='feature',
        buffer_prune='fifo',
        inner_lr=1e-3,
        inner_update_every=20,
        inner_sgd_steps=5,
        min_buffer_to_train=256,
        batch_size_inner=128,
        record_every_n_steps=1,
        enable_inner_training=False,
        phi_bound=100.0,
        error_bound=200.0,
        update_bound=10.0,
        mann_num_slots=4,
        mann_cw=1.0,
        mann_dt_mem=0.05,
        mann_alpha_mem=1.0,
        mann_memory_clip=50.0,
        mann_reset_memory_each_episode=True,
        mann_qmu_bound=200.0,
    ):
        super().__init__(given_pid=given_pid, p=p, i=i, d=d)
        self.model_name = model_name
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.dropout_p = float(dropout_p)
        self.gamma_adapt = float(gamma_adapt)
        self.projection_type = projection_type
        self.w_bound = float(w_bound)
        self.wn = float(wn)
        self.zeta = float(zeta)
        self.buffer_capacity = int(buffer_capacity)
        self.zeta_tol = float(zeta_tol)
        self.novelty_mode = novelty_mode
        self.buffer_prune = buffer_prune
        self.inner_lr = float(inner_lr)
        self.inner_update_every = int(inner_update_every)
        self.inner_sgd_steps = int(inner_sgd_steps)
        self.min_buffer_to_train = int(min_buffer_to_train)
        self.batch_size_inner = int(batch_size_inner)
        self.record_every_n_steps = int(record_every_n_steps)
        self.enable_inner_training = bool(enable_inner_training) and (model_name in ['dmrac', 'mann'])
        self.phi_bound = float(phi_bound)
        self.error_bound = float(error_bound)
        self.update_bound = float(update_bound)
        self.mann_num_slots = int(max(1, mann_num_slots))
        self.mann_cw = float(mann_cw)
        self.mann_dt_mem = float(mann_dt_mem)
        self.mann_alpha_mem = float(mann_alpha_mem)
        self.mann_memory_clip = float(mann_memory_clip)
        self.mann_reset_memory_each_episode = bool(mann_reset_memory_each_episode)
        self.mann_qmu_bound = float(mann_qmu_bound)
        self.use_memory = (model_name == 'mann')

        if q_diag is None:
            q_diag = np.array([10., 10., 10., 1., 1., 1.], dtype=float)
        self.q_diag = np.array(q_diag, dtype=float).reshape(-1)
        if self.q_diag.shape[0] != 6:
            raise ValueError(f"DMRACQuadController expects q_diag length 6, got {self.q_diag.shape[0]}")

        self.use_adaptation = model_name in ['shallow_mrac', 'dmrac_frozen', 'dmrac', 'mann']
        self.use_deep_feature = model_name in ['dmrac_frozen', 'dmrac', 'mann']

        self.feature_net = None
        self.optimizer = None
        if self.use_deep_feature:
            if self.model_name == 'mann':
                self.feature_net = MANNFeatureNet(
                    input_dim=13,
                    hidden_dim=self.hidden_dim,
                    feature_dim=self.feature_dim,
                ).to(device).double()
            else:
                self.feature_net = DMRACFeatureNet(
                    input_dim=13,
                    feature_dim=self.feature_dim,
                    hidden_dim=self.hidden_dim,
                    num_hidden_layers=self.hidden_layers,
                    dropout_p=self.dropout_p,
                ).to(device).double()
            self.optimizer = optim.Adam(self.feature_net.parameters(), lr=self.inner_lr)

        self._build_mrac_matrices()
        self.reset_controller()

    def _build_mrac_matrices(self):
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        self.B_mrac = np.vstack([Z3, I3])
        self.A_rm = np.block([
            [Z3, I3],
            [-(self.wn ** 2) * I3, -2.0 * self.zeta * self.wn * I3],
        ])
        Q = np.diag(self.q_diag)
        n = self.A_rm.shape[0]
        lhs = np.kron(np.eye(n), self.A_rm.T) + np.kron(self.A_rm.T, np.eye(n))
        rhs = -Q.reshape(-1)
        P = np.linalg.solve(lhs, rhs).reshape(n, n)
        self.P = 0.5 * (P + P.T)
        if not np.all(np.isfinite(self.P)):
            self.P = np.eye(6)

    def set_inner_training(self, enabled):
        self.enable_inner_training = bool(enabled) and (self.model_name in ['dmrac', 'mann'])

    def save(self, path):
        if self.feature_net is None:
            return
        torch.save({'feature_net': self.feature_net.state_dict()}, path)

    def load(self, path, map_location='cpu'):
        if self.feature_net is None:
            return
        ckpt = torch.load(path, map_location=map_location)
        if isinstance(ckpt, dict) and 'feature_net' in ckpt:
            self.feature_net.load_state_dict(ckpt['feature_net'])
        else:
            self.feature_net.load_state_dict(ckpt)

    def reset_controller(self):
        prev_mu = getattr(self, 'mu', None)
        super().reset_controller()
        self.W = np.zeros((self.feature_dim, 3))
        self.adapt_step = 0
        self.last_f_hat = np.zeros(3)
        self.nonfinite_w_updates = 0
        self.e_norm_hist = []
        self.nu_norm_hist = []
        self.w_norm_hist = []
        self.inner_loss_hist = []
        self.memory_norm_hist = []
        if self.use_memory:
            keep_prev = (
                (not self.mann_reset_memory_each_episode)
                and prev_mu is not None
                and prev_mu.shape == (self.feature_dim, self.mann_num_slots)
                and np.all(np.isfinite(prev_mu))
            )
            if keep_prev:
                self.mu = prev_mu.copy()
            else:
                self.mu = np.zeros((self.feature_dim, self.mann_num_slots), dtype=float)
        else:
            self.mu = None
        self.replay = DMRACReplayBuffer(
            capacity=self.buffer_capacity,
            state_dim=13,
            action_dim=3,
            feature_dim=self.feature_dim,
            zeta_tol=self.zeta_tol,
            novelty_mode=self.novelty_mode,
            prune_mode=self.buffer_prune,
        )

    def _shallow_feature(self, x):
        base = np.concatenate([x, x ** 2, np.sin(x), np.cos(x)])
        if len(base) >= self.feature_dim:
            return base[:self.feature_dim]
        return np.concatenate([base, np.zeros(self.feature_dim - len(base))])

    def _project_W(self, W):
        if self.projection_type == 'fro_norm':
            nrm = np.linalg.norm(W)
            if nrm > self.w_bound:
                W = W * (self.w_bound / (nrm + 1e-12))
            return W
        return np.clip(W, -self.w_bound, self.w_bound)

    def _get_phi(self, X):
        x_in = np.asarray(X, dtype=float).reshape(-1)
        if self.model_name == 'linear_baseline':
            return np.zeros(self.feature_dim), x_in
        if self.model_name == 'shallow_mrac':
            phi = self._shallow_feature(x_in)
        else:
            with torch.no_grad():
                x_t = torch.from_numpy(x_in).double().unsqueeze(0)
                phi = self.feature_net(x_t).cpu().numpy().reshape(-1)
        phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        phi = np.clip(phi, -self.phi_bound, self.phi_bound)
        return phi, x_in

    def _adapt_W(self, phi, e):
        e = np.nan_to_num(e, nan=0.0, posinf=self.error_bound, neginf=-self.error_bound)
        e = np.clip(e, -self.error_bound, self.error_bound)
        s = e @ self.P @ self.B_mrac
        s = np.nan_to_num(s, nan=0.0, posinf=50.0, neginf=-50.0)
        s = np.clip(s, -50.0, 50.0)
        update = np.outer(phi, s) / (1.0 + float(np.dot(phi, phi)))
        update = np.clip(update, -self.update_bound, self.update_bound)
        dt_adapt = self.dt if self.dt > 0 else self.params['dt_posctrl']
        W_candidate = self.W + dt_adapt * update * self.gamma_adapt
        W_candidate = self._project_W(W_candidate)
        if np.all(np.isfinite(W_candidate)):
            self.W = W_candidate
        else:
            self.nonfinite_w_updates += 1

    def _safe_softmax(self, logits):
        logits = np.asarray(logits, dtype=float).reshape(-1)
        logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        logits = np.clip(logits, -50.0, 50.0)
        logits = logits - np.max(logits)
        w = np.exp(logits)
        denom = float(np.sum(w))
        if denom <= 1e-12 or (not np.isfinite(denom)):
            return np.ones_like(logits) / max(len(logits), 1)
        return w / denom

    def _memory_read(self, a):
        if (not self.use_memory) or self.mu is None:
            return np.zeros(self.feature_dim), np.ones(1)
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.shape[0] != self.feature_dim:
            return np.zeros(self.feature_dim), np.ones(self.mann_num_slots) / self.mann_num_slots
        logits = self.mu.T @ a
        z = self._safe_softmax(logits)
        m_read = self.mu @ z
        if (not np.all(np.isfinite(m_read))) or m_read.shape[0] != self.feature_dim:
            return np.zeros(self.feature_dim), np.ones(self.mann_num_slots) / self.mann_num_slots
        return m_read, z

    def _compute_q_mu(self, X, pd, vd):
        e_p = pd - X[0:3]
        e_v = vd - X[7:10]
        # q_mu is projected from existing PID tracking errors into force-compensation space.
        q_mu = self.params['m'] * (
            self.params['K_p'] @ e_p
            + self.params['K_d'] @ e_v
            - self.params['K_i'] @ self.int_error
        )
        q_mu = np.nan_to_num(q_mu, nan=0.0, posinf=self.mann_qmu_bound, neginf=-self.mann_qmu_bound)
        return np.clip(q_mu, -self.mann_qmu_bound, self.mann_qmu_bound)

    def _memory_write(self, a, z, q_mu):
        if (not self.use_memory) or self.mu is None:
            return
        a = np.asarray(a, dtype=float).reshape(-1)
        z = np.asarray(z, dtype=float).reshape(-1)
        q_mu = np.asarray(q_mu, dtype=float).reshape(-1)
        if a.shape[0] != self.feature_dim or z.shape[0] != self.mann_num_slots or q_mu.shape[0] != 3:
            return
        if (not np.all(np.isfinite(a))) or (not np.all(np.isfinite(z))) or (not np.all(np.isfinite(q_mu))):
            return

        error_update = self.W @ q_mu
        if not np.all(np.isfinite(error_update)):
            error_update = np.zeros_like(a)

        z_row = z.reshape(1, -1)
        delta = (
            - self.mu * z_row
            + self.mann_cw * np.outer(a, z)
            + np.outer(error_update, z)
        )
        self.mu = self.mu + self.mann_dt_mem * delta
        if self.mann_memory_clip > 0:
            self.mu = np.clip(self.mu, -self.mann_memory_clip, self.mann_memory_clip)
        self.mu = np.nan_to_num(self.mu, nan=0.0, posinf=0.0, neginf=0.0)

    def _inner_update(self):
        if (not self.use_deep_feature) or (not self.enable_inner_training):
            return np.nan
        if self.inner_update_every <= 0:
            return np.nan
        if len(self.replay) < self.min_buffer_to_train:
            return np.nan
        W_t = torch.from_numpy(self.W.copy()).double().detach()
        losses = []
        for _ in range(self.inner_sgd_steps):
            x_b, y_b, _ = self.replay.sample(self.batch_size_inner)
            x_t = torch.from_numpy(x_b).double()
            y_t = torch.from_numpy(y_b).double()
            phi = self.feature_net(x_t)
            y_pred = phi @ W_t
            loss = torch.mean((y_pred - y_t) ** 2)
            if not torch.isfinite(loss):
                break
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))
        if len(losses) == 0:
            return np.nan
        return float(np.mean(losses))

    def get_adapt_stats(self):
        out = {
            'w_norm_mean': float(np.mean(self.w_norm_hist)) if len(self.w_norm_hist) > 0 else 0.0,
            'nu_ad_norm_mean': float(np.mean(self.nu_norm_hist)) if len(self.nu_norm_hist) > 0 else 0.0,
            'inner_loss_mean': float(np.mean(self.inner_loss_hist)) if len(self.inner_loss_hist) > 0 else np.nan,
            'replay_size': len(self.replay),
            'nonfinite_w_updates': int(self.nonfinite_w_updates),
            'memory_norm_mean': float(np.mean(self.memory_norm_hist)) if len(self.memory_norm_hist) > 0 else 0.0,
        }
        return out

    def get_Fr(self, X, Z, imu, pd, vd, ad, meta_adapt_trigger, wind_gt):
        Fr, Fr_dot = super().get_Fr(
            X, Z=Z, imu=imu, pd=pd, vd=vd, ad=ad,
            meta_adapt_trigger=meta_adapt_trigger, wind_gt=wind_gt
        )

        phi, x_in = self._get_phi(X)
        if self.use_memory:
            m_read, z = self._memory_read(phi)
            phi_out = phi + self.mann_alpha_mem * m_read
        else:
            z = None
            phi_out = phi
        if self.use_adaptation:
            f_hat = self.W.T @ phi_out
        else:
            f_hat = np.zeros(3)
        f_hat = np.nan_to_num(f_hat, nan=0.0, posinf=0.0, neginf=0.0)
        f_hat = np.clip(f_hat, -self.w_bound, self.w_bound)

        if self.use_adaptation:
            e_p = pd - X[0:3]
            e_v = vd - X[7:10]
            e = np.concatenate([e_p, e_v])
            self._adapt_W(phi, e)
            y_pseudo = self.W.T @ phi
            if (self.adapt_step % self.record_every_n_steps) == 0:
                self.replay.add(x_in, y_pseudo, phi=phi, w_snapshot=self.W.copy())
            if self.enable_inner_training and self.inner_update_every > 0 and (self.adapt_step % self.inner_update_every == 0):
                loss_inner = self._inner_update()
                if np.isfinite(loss_inner):
                    self.inner_loss_hist.append(loss_inner)
            self.e_norm_hist.append(float(np.linalg.norm(e)))
            if self.use_memory and z is not None:
                q_mu = self._compute_q_mu(X, pd, vd)
                self._memory_write(phi, z, q_mu)

        self.nu_norm_hist.append(float(np.linalg.norm(f_hat)))
        self.w_norm_hist.append(float(np.linalg.norm(self.W)))
        if self.use_memory and self.mu is not None:
            self.memory_norm_hist.append(float(np.linalg.norm(self.mu)))
        self.last_f_hat = f_hat.copy()
        self.adapt_step += 1
        return Fr - f_hat, Fr_dot

class MetaAdapt(PIDController):
    def __init__(self, given_pid=False, p=0, i=0, d=0, use_state_lpf=False, state_lpf_alpha=1.0):
        super().__init__(given_pid=given_pid, p=p, i=i, d=d)
        self.motor_speed = np.zeros(4)
        self.use_state_lpf = bool(use_state_lpf)
        self.state_lpf_alpha = float(state_lpf_alpha)
        self.state_lpf_prev = None

    def reset_controller(self):
        super().reset_controller()
        self.state_lpf_prev = None

    def filter_state_input(self, X):
        x = np.asarray(X, dtype=float).copy()
        if (not self.use_state_lpf) or self.state_lpf_alpha >= 1.0:
            return x
        alpha = np.clip(self.state_lpf_alpha, 0.0, 1.0)
        if self.state_lpf_prev is None:
            self.state_lpf_prev = x.copy()
            return x
        x_filt = alpha * x + (1.0 - alpha) * self.state_lpf_prev
        self.state_lpf_prev = x_filt.copy()
        return x_filt
    
    def get_residual(self, X, imu):
        q = X[3:7]
        T = self.params['C_T'] * sum(self.motor_speed ** 2)
        return residual_force_from_onboard(
            acceleration_world=imu[0:3],
            attitude_wxyz=q,
            thrust_n=T,
            mass_kg=self.params['m'],
            gravity_m_s2=self.params['g'],
        )
    
    def get_Fr(self, X, Z, imu, pd, vd, ad, meta_adapt_trigger, wind_gt):
        y = self.get_residual(X, imu)
        X_in = self.filter_state_input(X)
        fhat_F = self.get_f_hat(X_in)
        self.inner_adapt(X_in, fhat_F, y)
        self.update_batch(X_in, fhat_F, y)
        if (meta_adapt_trigger and self.state=='train'):
            self.meta_adapt()
        
        Fr,Fr_dot = super().get_Fr(X, Z, imu, pd, vd, ad, meta_adapt_trigger, wind_gt)
        f_hat = self.get_f_hat(X_in)
        return Fr-f_hat, Fr_dot
    
    def mixer(self, torque_sp, T_sp):
        self.motor_speed = super().mixer(torque_sp, T_sp)
        return self.motor_speed
    
    def get_f_hat(self,X):
        raise NotImplementedError
    def inner_adapt(self, X, fhat, y):
        raise NotImplementedError
    def update_batch(self, X, fhat, y):
        raise NotImplementedError
    def meta_adapt(self, ):
        raise NotImplementedError

class MetaAdaptDeep(MetaAdapt):
    class Phi(nn.Module):
        def __init__(self, input_kernel, dim_kernel, layer_sizes):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(input_kernel, layer_sizes[0]))
            self.fc2 = spectral_norm(nn.Linear(layer_sizes[0], layer_sizes[1]))
            self.fc3 = spectral_norm(nn.Linear(layer_sizes[1], dim_kernel))
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    def __init__(self, given_pid=False, p=0, i=0, d=0, dim_a=100, layer_size=(25,30), 
                 eta_a_base=0.3, eta_A_base=0.01, use_state_lpf=False, state_lpf_alpha=1.0):
        super().__init__(
            given_pid=given_pid,
            p=p,
            i=i,
            d=d,
            use_state_lpf=use_state_lpf,
            state_lpf_alpha=state_lpf_alpha,
        )
        self.dim_a = dim_a - dim_a%3
        self.layer_sizes = layer_size
        self.eta_a_base = eta_a_base
        self.eta_A_base = eta_A_base
        self.loss = nn.MSELoss()
        self.state = 'train'
    
    def reset_controller(self):
        super().reset_controller()
        self.a = np.zeros(self.dim_a)
        self.phi = self.Phi(input_kernel=13, dim_kernel=self.dim_a//3, layer_sizes=self.layer_sizes).to(device)
        self.optimizer = optim.Adam(self.phi.parameters(), lr=self.eta_A_base)
        self.inner_adapt_count = 0
        self.batch = []
    
    def get_phi(self, X):
        with torch.no_grad():
            return np.kron(np.eye(3), self.phi(torch.from_numpy(X).to(device)).cpu().numpy())
    
    def get_f_hat(self, X):
        phi = self.get_phi(X)
        return phi @ self.a

    def inner_adapt(self, X, fhat, y):
        self.inner_adapt_count += 1
        eta_a = self.eta_a_base / np.sqrt(self.inner_adapt_count)
        self.a -= eta_a * 2 * (fhat - y).transpose() @ self.get_phi(X)

    def update_batch(self, X, fhat, y):
        self.batch.append((X, y, self.a.copy()))
    
    def meta_adapt(self):
        self.inner_adapt_count = 0
        self.optimizer.zero_grad()
        loss = 0
        for X, y, a in self.batch:
            phi = torch.kron(torch.eye(3).to(device), self.phi(torch.from_numpy(X).to(device)))
            loss += self.loss(torch.matmul(phi, torch.from_numpy(a).to(device)), torch.from_numpy(y).to(device))
        loss.backward()
        self.optimizer.step()
        self.batch = []

class MetaAdaptOoD(MetaAdaptDeep):
    def __init__(self, given_pid=False, p=0, i=0, d=0, dim_a=100, layer_size=(25,30), eta_a_base=0.05,
                 eta_A_base=0.05, noise_x=0.01, noise_a=0.01, use_state_lpf=False, state_lpf_alpha=1.0):
        super().__init__(
            given_pid=given_pid,
            p=p,
            i=i,
            d=d,
            dim_a=dim_a,
            layer_size=layer_size,
            eta_a_base=eta_a_base,
            eta_A_base=eta_A_base,
            use_state_lpf=use_state_lpf,
            state_lpf_alpha=state_lpf_alpha,
        )
        self.noise_x = noise_x
        self.noise_a = noise_a
    
    def inner_adapt(self, X, fhat, y):
        self.a -= self.eta_a_base * 2 * (fhat - y).transpose() @ self.get_phi(X)

    def meta_adapt(self):
        self.optimizer.zero_grad()

        loss = 0
        for X, y, a in self.batch:
            X = X + self.noise_x*np.random.normal(0,1,X.shape)
            a = a + self.noise_a*np.random.normal(0,1,a.shape)
            phi = torch.kron(torch.eye(3).to(device), self.phi(torch.from_numpy(X).to(device)))
            loss += self.loss(torch.matmul(phi, torch.from_numpy(a).to(device)), torch.from_numpy(y).to(device))
        loss.backward()
        self.optimizer.step()
        self.batch = []

class NeuralFly(MetaAdaptDeep):
    class H(nn.Module):
        def __init__(self, start_kernel, dim_kernel, layer_sizes):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(start_kernel, layer_sizes[0]))
            self.fc2 = spectral_norm(nn.Linear(layer_sizes[0], layer_sizes[1]))
            self.fc3 = spectral_norm(nn.Linear(layer_sizes[1], dim_kernel))
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    def __init__(self, given_pid=False, p=0, i=0, d=0, dim_a=100, layer_size=(25,30), eta_a_base=0.3, 
                 eta_A_base=0.05, use_state_lpf=False, state_lpf_alpha=1.0):
        super().__init__(
            given_pid=given_pid,
            p=p,
            i=i,
            d=d,
            dim_a=dim_a,
            layer_size=layer_size,
            eta_a_base=eta_a_base,
            eta_A_base=eta_A_base,
            use_state_lpf=use_state_lpf,
            state_lpf_alpha=state_lpf_alpha,
        )
        self.wind_idx = 0
        self.alpha = 0.1
        # self.lr = lr

    def reset_controller(self):
        super().reset_controller()
        self.h = self.H(start_kernel = self.dim_a//3, dim_kernel=10, layer_sizes=self.layer_sizes).to(device)
        self.h_optimizer = optim.Adam(params=self.h.parameters(), lr=0.1)
        self.h_loss = nn.CrossEntropyLoss()
    
    def meta_adapt(self):
        self.inner_adapt_count = 0
        self.optimizer.zero_grad()
        loss = 0
        target = torch.tensor([self.wind_idx], dtype=int).to(device)
        for X, y, a in self.batch:
            phi = torch.kron(torch.eye(3).to(device), self.phi(torch.from_numpy(X).to(device)))
            loss += self.loss(torch.matmul(phi, torch.from_numpy(a).to(device)), torch.from_numpy(y).to(device))
            loss -= self.alpha * self.h_loss(self.h(self.phi(torch.from_numpy(X).to(device))).unsqueeze(0), target).detach()
        loss.backward()
        self.optimizer.step()

        if (np.random.uniform(0,1) < 0.5):
            loss_h = 0
            self.h_optimizer.zero_grad()
            for X, y, a in self.batch:
                phi = self.phi(torch.from_numpy(X).to(device)).detach()
                h = self.h(phi)
                loss_h += self.h_loss(h.unsqueeze(0), target)
            loss_h.backward()
            self.h_optimizer.step()
        
        self.batch = []

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=4, num_layers=2, dim_feedforward=256, embed_dim=32, context_dim=16):
        super(TransformerEncoderDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        assert embed_dim % nhead == 0, "embed_dim must be divisible by num_heads"

        self.input_embedding = nn.Linear(input_dim, embed_dim).float()
        self.output_embedding = nn.Linear(output_dim, embed_dim).float()
        self.context_embedding = nn.Linear(context_dim, embed_dim).float()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward).float()
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward).float()
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_dim, output_dim).float()

    def forward(self, src, tgt, context):
        src = src.to(dtype=torch.float32)
        tgt = tgt.to(dtype=torch.float32)
        context = context.to(dtype=torch.float32)
        src = self.input_embedding(src)
        tgt = self.output_embedding(tgt)
        context_embedded = self.context_embedding(context).unsqueeze(0)
        src = src + context_embedded

        encoder_output = self.encoder(src)
        output = self.decoder(tgt, encoder_output)
        output = self.fc_out(output[-1])  # Take the last output
        return output  # Output shape: (batch_size, output_dim)

class MetaAdaptTransformer(PIDController):
    def __init__(self, given_pid=False, p=0, i=0, d=0):
        super().__init__(given_pid=given_pid, p=p, i=i, d=d)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_length = 10
        self.state_dim = 13
        self.control_dim = 4
        self.wind_force_dim = 3
        self.context_dim = 72  # Implicit environment embedding dimension
        self.input_dim = self.state_dim + self.control_dim
        self.output_dim = self.wind_force_dim

        self.transformer_model = TransformerEncoderDecoder(input_dim=self.input_dim, output_dim=self.output_dim, context_dim=self.context_dim).to(self.device)
        self.optimizer = optim.Adam(self.transformer_model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()

        self.memory = []
        self.memory_capacity = 1000
        self.batch_size = 32
        self.state_history = []
        self.control_history = []
        self.wind_force_history = []
        self.context_vector = torch.zeros(self.context_dim).to(self.device)  # Initialize context vector

    def get_f_hat(self, X, u):
        self.state_history.append(X.copy())
        self.control_history.append(u.copy())
        if len(self.wind_force_history) == 0:
            self.wind_force_history.append(np.zeros(self.wind_force_dim))

        if len(self.state_history) > self.seq_length:
            self.state_history.pop(0)
            self.control_history.pop(0)
            self.wind_force_history.pop(0)

        if len(self.state_history) == self.seq_length:
            input_sequence = []
            for s, c in zip(self.state_history, self.control_history):
                sc = np.concatenate([s, c])
                input_sequence.append(sc)
            input_sequence = np.stack(input_sequence)
            wind_force_sequence = np.stack(self.wind_force_history)

            input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(1).to(self.device)
            wind_force_tensor = torch.tensor(wind_force_sequence, dtype=torch.float32).unsqueeze(1).to(self.device)

            with torch.no_grad():
                wind_force_pred = self.transformer_model(input_tensor, wind_force_tensor, self.context_vector)
            wind_force_pred = wind_force_pred.squeeze(0).cpu().numpy()

            self.wind_force_history.append(wind_force_pred)

            return wind_force_pred
        else:
            return np.zeros(self.output_dim)

    def get_Fr(self, X, Z, imu, pd, vd, ad, meta_adapt_trigger, wind_gt):
        u = Z.copy()
        f_hat = self.get_f_hat(X, u)
        self.update_memory(X, u, wind_gt)
        self.update_context(X, u, wind_gt)
        Fr, Fr_dot = super().get_Fr(X, Z, imu, pd, vd, ad, meta_adapt_trigger, wind_gt)
        return Fr - f_hat, Fr_dot

    def update_context(self, X, u, wind_force_gt):
        X_u = np.concatenate([X, u])
        X_u_tensor = torch.tensor(X_u, dtype=torch.float32).unsqueeze(0).to(self.device)
        wind_force_gt_tensor = torch.tensor(wind_force_gt, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predicted_wind_force = self.transformer_model(X_u_tensor.unsqueeze(0), wind_force_gt_tensor.unsqueeze(0), self.context_vector)
        error = wind_force_gt_tensor - predicted_wind_force
        if error.size(1) < self.context_dim:
            error_padded = torch.zeros(self.context_dim).to(self.device)
            error_padded[:error.size(1)] = error.squeeze(0)
            self.context_vector += 0.01 * error_padded
        else:
            self.context_vector += 0.01 * error.squeeze(0)[:self.context_dim]

    def update_memory(self, X, u, wind_force_gt):
        self.memory.append((X.copy(), u.copy(), wind_force_gt.copy()))
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)

    def train_model(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        input_sequences = []
        wind_force_sequences = []
        targets = []
        for X, u, y in batch:
            for i in range(len(self.memory)):
                if np.array_equal(self.memory[i][0], X) and np.array_equal(self.memory[i][1], u) and np.array_equal(self.memory[i][2], y):
                    memory_idx = i
                    break
            if memory_idx >= self.seq_length - 1:
                seq = self.memory[memory_idx - self.seq_length + 1: memory_idx + 1]
                input_sequence = [np.concatenate([s[0], s[1]]) for s in seq]
                wind_force_sequence = [s[2] for s in seq]
                input_sequences.append(np.stack(input_sequence))
                wind_force_sequences.append(np.stack(wind_force_sequence))
                targets.append(y)

        if len(input_sequences) == 0:
            return

        input_sequences = np.stack(input_sequences)
        wind_force_sequences = np.stack(wind_force_sequences)
        targets = np.stack(targets)

        input_tensor = torch.tensor(input_sequences, dtype=torch.float32).transpose(0, 1).to(self.device)
        wind_force_tensor = torch.tensor(wind_force_sequences, dtype=torch.float32).transpose(0, 1).to(self.device)
        target_tensor = torch.tensor(targets, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        output = self.transformer_model(input_tensor, wind_force_tensor, self.context_vector)
        loss = self.criterion(output, target_tensor)
        loss.backward()
        self.optimizer.step()

class FGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FGRUCell, self).__init__()
        self.hidden_dim = hidden_dim

        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_x = nn.Linear(input_dim, hidden_dim, bias=True)
        self.gamma = 0.9
        self.C = 20.0
        self.last_g_t = 1.0  # recorded after each forward pass for analysis

    def compute_force_fluctuation(self, residual_forces):
        if residual_forces.shape[0] <= 1:
            return torch.tensor(1.0, dtype=residual_forces.dtype, device=residual_forces.device)
        diffs = torch.sum(torch.abs(residual_forces[-1] - residual_forces[:-1]), dim=1)
        weights = self.gamma ** torch.arange(len(diffs), dtype=diffs.dtype, device=diffs.device)
        fluctuation = torch.exp(-torch.sum(weights * diffs) / self.C)
        return fluctuation

    def forward(self, x_t, h_prev, residual_forces):
        # x_t: (seq_len, input_dim); h_prev: (hidden_dim,) or (seq_len, hidden_dim)
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0).expand(x_t.shape[0], -1)
        g_t = self.compute_force_fluctuation(residual_forces)
        self.last_g_t = float(g_t.item())

        h_candidate = torch.tanh(self.W_h(h_prev))
        z_t = torch.sigmoid(self.W_x(x_t))
        h_t = (1 - g_t) * h_candidate + g_t * z_t
        return h_t


class FGRUCellFixedGate(FGRUCell):
    """Ablation: FGRU with fixed g_t = 0.5 (no force-gating).

    Isolates the contribution of the physically-motivated gating mechanism
    from the general recurrent structure. If performance degrades significantly
    vs. the full FGRU, it confirms force-gating is the key driver.
    """
    def compute_force_fluctuation(self, residual_forces):
        return torch.tensor(0.5, dtype=residual_forces.dtype, device=residual_forces.device)


class VanillaGRUCell(nn.Module):
    """Standard GRU encoder with the same input and hidden dimensions as FGRU."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.last_g_t = float("nan")

    def _run_gru(self, x_t, h_prev):
        if h_prev.dim() == 2:
            h_t = h_prev[-1]
        else:
            h_t = h_prev
        outputs = []
        for x_step in x_t:
            h_t = self.gru(x_step.unsqueeze(0), h_t.unsqueeze(0)).squeeze(0)
            outputs.append(h_t)
        return torch.stack(outputs, dim=0)

    def forward(self, x_t, h_prev, residual_forces):
        del residual_forces
        self.last_g_t = float("nan")
        return self._run_gru(x_t, h_prev)


class ExpertDictionary(nn.Module):
    def __init__(
        self,
        key_dim,
        value_dim,
        max_entries=10000,
        normalize_keys=False,
        temperature=1.0,
        min_cosine_distance=0.0,
        max_entries_per_bucket=0,
        ema_alpha=0.9,
    ):
        super(ExpertDictionary, self).__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.max_entries = max_entries
        self.normalize_keys = bool(normalize_keys)
        self.temperature = float(temperature)
        self.min_cosine_distance = float(min_cosine_distance)
        self.max_entries_per_bucket = int(max_entries_per_bucket)
        self.ema_alpha = float(ema_alpha)
        
        self.keys = torch.zeros((max_entries, key_dim), requires_grad=False)  # FGRU embeddings
        self.values = torch.zeros((max_entries, value_dim), requires_grad=False)  # Environment embeddings
        self.entry_counts = torch.zeros(max_entries, dtype=torch.long)
        self.entry_buckets = []
        self.entry_metadata = []
        self.bucket_counts = {}
        self.current_size = 0
        self.num_candidates = 0
        self.num_inserted = 0
        self.num_merged = 0
        self.num_skipped_full = 0
        self.num_skipped_balance = 0
        self.num_skipped_nonfinite = 0
        self.last_store_status = None

    def _as_tensor_1d(self, data, dim, name):
        if isinstance(data, torch.Tensor):
            tensor = data.detach().to(dtype=self.keys.dtype).reshape(-1)
        else:
            tensor = torch.as_tensor(data, dtype=self.keys.dtype).reshape(-1)
        if tensor.numel() != dim:
            raise ValueError(f"{name} has dim {tensor.numel()}, expected {dim}")
        return tensor

    def _normalize(self, tensor):
        return tensor / torch.clamp(torch.linalg.norm(tensor), min=1e-12)

    def _bucket_from_metadata(self, metadata):
        if metadata is None:
            return "global"
        if "bucket" in metadata:
            return str(metadata["bucket"])
        parts = []
        for field in ("trajectory", "wind_label", "wind_mean", "wind_std", "episode"):
            if field in metadata:
                parts.append(f"{field}={metadata[field]}")
        return "|".join(parts) if parts else "global"

    def _key_distances(self, key, start_index=0):
        start_index = min(max(int(start_index), 0), self.current_size)
        if self.current_size == start_index:
            return torch.empty(0, dtype=self.keys.dtype)
        active_keys = self.keys[start_index:self.current_size]
        if self.normalize_keys:
            similarities = active_keys @ key
            return 1.0 - similarities
        return torch.linalg.norm(active_keys - key, dim=1)

    def clear(self):
        self.keys.zero_()
        self.values.zero_()
        self.entry_counts.zero_()
        self.entry_buckets = []
        self.entry_metadata = []
        self.bucket_counts = {}
        self.current_size = 0
        self.num_candidates = 0
        self.num_inserted = 0
        self.num_merged = 0
        self.num_skipped_full = 0
        self.num_skipped_balance = 0
        self.num_skipped_nonfinite = 0
        self.last_store_status = None

    def store(self, key, value, metadata=None, min_merge_index=0):
        self.num_candidates += 1
        metadata = dict(metadata or {})
        bucket = self._bucket_from_metadata(metadata)
        metadata.setdefault("bucket", bucket)

        key_t = self._as_tensor_1d(key, self.key_dim, "dictionary key")
        value_t = self._as_tensor_1d(value, self.value_dim, "dictionary value")
        if not torch.all(torch.isfinite(key_t)) or not torch.all(torch.isfinite(value_t)):
            self.num_skipped_nonfinite += 1
            self.last_store_status = "skipped_nonfinite"
            return False
        if self.normalize_keys:
            key_t = self._normalize(key_t)

        min_merge_index = min(max(int(min_merge_index), 0), self.current_size)
        if self.current_size > min_merge_index and self.min_cosine_distance > 0:
            distances = self._key_distances(key_t, start_index=min_merge_index)
            nearest_idx = min_merge_index + int(torch.argmin(distances).item())
            nearest_distance = float(distances[nearest_idx - min_merge_index].item())
            if nearest_distance < self.min_cosine_distance:
                alpha = min(max(self.ema_alpha, 0.0), 1.0)
                self.keys[nearest_idx] = alpha * self.keys[nearest_idx] + (1.0 - alpha) * key_t
                if self.normalize_keys:
                    self.keys[nearest_idx] = self._normalize(self.keys[nearest_idx])
                self.values[nearest_idx] = alpha * self.values[nearest_idx] + (1.0 - alpha) * value_t
                self.entry_counts[nearest_idx] += 1
                if nearest_idx < len(self.entry_metadata):
                    self.entry_metadata[nearest_idx]["updates"] = int(self.entry_counts[nearest_idx].item())
                self.num_merged += 1
                self.last_store_status = "merged"
                return False

        if self.max_entries_per_bucket > 0 and self.bucket_counts.get(bucket, 0) >= self.max_entries_per_bucket:
            self.num_skipped_balance += 1
            self.last_store_status = "skipped_balance"
            return False

        if self.current_size >= self.max_entries:
            self.num_skipped_full += 1
            self.last_store_status = "skipped_full"
            return False

        self.keys[self.current_size, :] = key_t
        self.values[self.current_size, :] = value_t
        self.entry_counts[self.current_size] = 1
        self.entry_buckets.append(bucket)
        metadata["updates"] = 1
        self.entry_metadata.append(metadata)
        self.bucket_counts[bucket] = self.bucket_counts.get(bucket, 0) + 1
        self.current_size += 1
        self.num_inserted += 1
        self.last_store_status = "inserted"
        return True

    def retrieve(self, query):
        query_t = self._as_tensor_1d(query, self.key_dim, "dictionary query")
        if self.current_size == 0:
            return torch.zeros(self.value_dim, dtype=query_t.dtype)
        if self.normalize_keys:
            query_t = self._normalize(query_t)
        temperature = max(self.temperature, 1e-6)
        similarities = torch.matmul(self.keys[:self.current_size], query_t) / temperature
        weights = torch.softmax(similarities, dim=0)
        retrieved_value = torch.matmul(weights, self.values[:self.current_size])
        return retrieved_value

    def max_similarity(self, query):
        query_t = self._as_tensor_1d(query, self.key_dim, "dictionary query")
        if self.current_size == 0:
            return float("-inf")
        if self.normalize_keys:
            query_t = self._normalize(query_t)
        similarities = torch.matmul(self.keys[:self.current_size], query_t)
        return float(torch.max(similarities).item())

    def load_entries(self, keys, values, metadata=None, counts=None, normalize_existing=False):
        size = int(keys.shape[0])
        size = min(size, self.max_entries)
        self.clear()
        if size <= 0:
            return
        key_t = keys[:size].detach().to(dtype=self.keys.dtype).clone()
        if normalize_existing or self.normalize_keys:
            key_t = F.normalize(key_t, p=2, dim=1, eps=1e-12)
        self.keys[:size] = key_t
        self.values[:size] = values[:size].detach().to(dtype=self.values.dtype).clone()
        if counts is not None:
            count_t = torch.as_tensor(counts[:size], dtype=torch.long)
            self.entry_counts[:size] = torch.clamp(count_t, min=1)
        else:
            self.entry_counts[:size] = 1
        self.current_size = size
        if metadata is not None and len(metadata) >= size:
            self.entry_metadata = [dict(m) for m in metadata[:size]]
        else:
            self.entry_metadata = [{"bucket": "loaded", "updates": int(self.entry_counts[i].item())} for i in range(size)]
        self.entry_buckets = [self._bucket_from_metadata(m) for m in self.entry_metadata]
        self.bucket_counts = {}
        for bucket in self.entry_buckets:
            self.bucket_counts[bucket] = self.bucket_counts.get(bucket, 0) + 1

    def describe(self):
        size = int(self.current_size)
        dtype_bytes = int(self.keys.element_size())
        storage_bytes = size * (self.key_dim + self.value_dim) * dtype_bytes
        retrieval_macs = size * (self.key_dim + self.value_dim) + 3 * self.key_dim
        bucket_sizes = list(self.bucket_counts.values())
        return {
            "size": size,
            "key_dim": int(self.key_dim),
            "value_dim": int(self.value_dim),
            "max_entries": int(self.max_entries),
            "normalize_keys": bool(self.normalize_keys),
            "temperature": float(self.temperature),
            "distance_metric": "cosine" if self.normalize_keys else "euclidean",
            "min_cosine_distance": float(self.min_cosine_distance),
            "max_entries_per_bucket": int(self.max_entries_per_bucket),
            "num_buckets": len(self.bucket_counts),
            "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
            "storage_mb": storage_bytes / (1024.0 * 1024.0),
            "retrieval_macs": int(retrieval_macs),
            "num_candidates": int(self.num_candidates),
            "num_inserted": int(self.num_inserted),
            "num_merged": int(self.num_merged),
            "num_skipped_full": int(self.num_skipped_full),
            "num_skipped_balance": int(self.num_skipped_balance),
            "num_skipped_nonfinite": int(self.num_skipped_nonfinite),
        }

    def measure_retrieval_latency_ms(self, num_trials=200):
        if self.current_size == 0:
            return float("nan")
        query = torch.randn(self.key_dim, dtype=self.keys.dtype)
        with torch.no_grad():
            for _ in range(min(20, num_trials)):
                self.retrieve(query)
            start = time.perf_counter()
            for _ in range(num_trials):
                self.retrieve(query)
            elapsed = time.perf_counter() - start
        return 1000.0 * elapsed / max(num_trials, 1)

class AeroACE(PIDController):
    def __init__(
        self,
        input_dim=17,
        hidden_dim=33,
        expert_dim=99,
        seq_len=10,
        given_pid=False,
        p=0,
        i=0,
        d=0,
        dict_max_entries=10000,
        dict_normalize_keys=False,
        dict_temperature=1.0,
        dict_min_cosine_distance=0.01,
        dict_max_entries_per_bucket=1024,
        dict_ema_alpha=0.9,
        c_refer_threshold_coefficient=0.1,
        online_update=False,
        online_anomaly_similarity_threshold=0.9,
        online_min_anomaly_steps=3,
        online_warmup_steps=20,
        online_update_interval_steps=10,
        online_residual_window=5,
        online_residual_consistency_threshold=6.0,
        online_force_clip_norm=30.0,
        online_force_reject_norm=60.0,
        online_require_anomaly=True,
    ):
        super(AeroACE, self).__init__(given_pid=given_pid, p=p, i=i, d=d)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.seq_len = int(seq_len)
        self.fgru = FGRUCell(input_dim, hidden_dim)
        self.expert_dict = ExpertDictionary(
            hidden_dim,
            expert_dim,
            max_entries=dict_max_entries,
            normalize_keys=bool(dict_normalize_keys),
            temperature=dict_temperature,
            min_cosine_distance=dict_min_cosine_distance,
            max_entries_per_bucket=dict_max_entries_per_bucket,
            ema_alpha=dict_ema_alpha,
        )
        self.c_refer_threshold_coefficient = float(
            c_refer_threshold_coefficient
        )
        if self.c_refer_threshold_coefficient < 0.0:
            raise ValueError("c_refer_threshold_coefficient must be non-negative")
        self.h = np.zeros((self.hidden_dim,))
        self.motor_speed = np.zeros(4)
        self.state = "train"
        self.state_list = []
        self.control_list = []
        self.force_list = []
        self.c = np.zeros(3 * self.hidden_dim, dtype=float)
        self.residual = np.zeros(3)
        self.count = 0
        self.force_err_list = []

        self.training_stage = None  # None | 'stage1' | 'stage2'
        self.fgru_optimizer = optim.Adam(self.fgru.parameters(), lr=1e-4)
        self.c_env = None
        self.dictionary_context = {}
        self.dictionary_step = 0

        self.online_update_enabled = bool(online_update)
        self.online_anomaly_similarity_threshold = float(online_anomaly_similarity_threshold)
        self.online_min_anomaly_steps = max(1, int(online_min_anomaly_steps))
        self.online_warmup_steps = max(0, int(online_warmup_steps))
        self.online_update_interval_steps = max(1, int(online_update_interval_steps))
        self.online_residual_window = max(1, int(online_residual_window))
        self.online_residual_consistency_threshold = max(
            0.0, float(online_residual_consistency_threshold)
        )
        self.online_force_clip_norm = max(0.0, float(online_force_clip_norm))
        self.online_force_reject_norm = max(0.0, float(online_force_reject_norm))
        self.online_require_anomaly = bool(online_require_anomaly)
        self.online_base_dictionary_size = None
        self._reset_online_update_state()

    def reset_controller(self):
        super().reset_controller()
        self.motor_speed = np.zeros(4)
        self.state_list = []
        self.control_list = []
        self.force_list = []
        self.h = np.zeros((self.hidden_dim,))
        self.c = np.zeros(3 * self.hidden_dim, dtype=float)
        self.residual = np.zeros(3)
        self.count = 0
        self.force_err_list = []
        self.dictionary_step = 0
        self.online_base_dictionary_size = None
        self._reset_online_update_state()
        if hasattr(self.fgru, "last_g_t"):
            self.fgru.last_g_t = 1.0

    def _reset_online_update_state(self):
        self.online_residual_history = deque(maxlen=self.online_residual_window)
        self.online_update_step = 0
        self.online_anomaly_run = 0
        self.online_last_update_step = -self.online_update_interval_steps
        self.online_last_max_similarity = float("nan")
        self.online_last_residual_deviation = float("nan")
        self.online_last_force_raw = np.zeros(3, dtype=float)
        self.online_last_force_used = np.zeros(3, dtype=float)
        self.online_last_update_status = "not_evaluated"
        self.online_update_history = []
        self.online_similarity_history = []
        self.online_update_stats = {
            "steps": 0,
            "anomaly_steps": 0,
            "candidates": 0,
            "accepted": 0,
            "inserted": 0,
            "merged": 0,
            "clipped": 0,
            "rejected_nonfinite": 0,
            "rejected_force_norm": 0,
            "rejected_inconsistent": 0,
            "rejected_warmup": 0,
            "rejected_anomaly_persistence": 0,
            "rejected_rate_limit": 0,
            "rejected_capacity": 0,
            "covered_steps": 0,
        }
    
    def mixer(self, torque_sp, T_sp):
        self.motor_speed = super().mixer(torque_sp, T_sp)
        return self.motor_speed
    
    def get_residual(self, X, imu):
        q = X[3:7]
        R = rowan.to_matrix(q)

        H = self.params['m'] * np.eye(3)
        G = np.array((0., 0., self.params['g'] * self.params['m']))
        T = self.params['C_T'] * sum(self.motor_speed ** 2)
        u = T * R @ np.array((0., 0., 1.))
        y = (H @ imu[0:3] + G - u)

        return y
    
    def set_dictionary_context(self, metadata=None):
        self.dictionary_context = dict(metadata or {})
        self.dictionary_step = 0

    def _dictionary_store_metadata(self, wind_force=None):
        metadata = dict(self.dictionary_context)
        metadata["step"] = int(self.dictionary_step)
        if wind_force is not None and np.all(np.isfinite(wind_force)):
            metadata["wind_force_norm"] = float(np.linalg.norm(wind_force))
        self.dictionary_step += 1
        return metadata

    @staticmethod
    def _clip_vector_norm(vector, max_norm):
        vector = np.asarray(vector, dtype=float).reshape(3)
        norm = float(np.linalg.norm(vector))
        if max_norm > 0.0 and norm > max_norm:
            return vector * (max_norm / max(norm, 1e-12)), True
        return vector.copy(), False

    def _project_environment_embedding(self, h_t, c_prior, force_target):
        h_t = np.asarray(h_t, dtype=float).reshape(self.hidden_dim)
        c_prior = np.asarray(c_prior, dtype=float).reshape(3 * self.hidden_dim)
        force_target = np.asarray(force_target, dtype=float).reshape(3)
        h3 = np.kron(np.eye(3), h_t)
        gram = h3 @ h3.T
        if not np.all(np.isfinite(gram)) or float(np.linalg.norm(h_t)) < 1e-10:
            return None
        delta = h3 @ c_prior - force_target
        try:
            correction = h3.T @ np.linalg.solve(gram, delta)
        except np.linalg.LinAlgError:
            correction = h3.T @ np.linalg.pinv(gram) @ delta
        c_new = c_prior - correction
        if not np.all(np.isfinite(c_new)):
            return None
        return c_new

    def _record_online_update(self, status, force_raw, force_used, max_similarity, deviation):
        self.online_last_update_status = status
        self.online_last_force_raw = np.asarray(force_raw, dtype=float).reshape(3).copy()
        self.online_last_force_used = np.asarray(force_used, dtype=float).reshape(3).copy()
        self.online_last_max_similarity = float(max_similarity)
        self.online_last_residual_deviation = float(deviation)
        self.online_update_history.append({
            "step": int(self.online_update_step),
            "status": str(status),
            "max_similarity": float(max_similarity),
            "residual_deviation": float(deviation),
            "force_raw": self.online_last_force_raw.tolist(),
            "force_used": self.online_last_force_used.tolist(),
            "dictionary_size": int(self.expert_dict.current_size),
        })

    def maybe_online_update(self, h_t, c_prior, residual_force):
        self.online_update_step += 1
        self.online_update_stats["steps"] += 1

        force_raw = np.asarray(residual_force, dtype=float).reshape(3)
        zero_force = np.zeros(3, dtype=float)
        max_similarity = self.expert_dict.max_similarity(h_t)
        self.online_last_max_similarity = max_similarity
        self.online_similarity_history.append(float(max_similarity))

        if not self.online_update_enabled:
            self.online_last_update_status = "disabled"
            return False
        if self.online_base_dictionary_size is None:
            self.online_base_dictionary_size = int(self.expert_dict.current_size)

        if not np.all(np.isfinite(force_raw)):
            self.online_update_stats["rejected_nonfinite"] += 1
            self._record_online_update(
                "rejected_nonfinite", zero_force, zero_force, max_similarity, float("nan")
            )
            return False

        force_norm = float(np.linalg.norm(force_raw))
        if self.online_force_reject_norm > 0.0 and force_norm > self.online_force_reject_norm:
            self.online_update_stats["rejected_force_norm"] += 1
            self._record_online_update(
                "rejected_force_norm", force_raw, zero_force, max_similarity, float("nan")
            )
            return False

        force_used, clipped = self._clip_vector_norm(force_raw, self.online_force_clip_norm)
        if clipped:
            self.online_update_stats["clipped"] += 1
        self.online_residual_history.append(force_used.copy())

        anomalous = (
            not np.isfinite(max_similarity)
            or max_similarity < self.online_anomaly_similarity_threshold
        )
        if anomalous:
            self.online_anomaly_run += 1
            self.online_update_stats["anomaly_steps"] += 1
        else:
            self.online_anomaly_run = 0
            self.online_update_stats["covered_steps"] += 1

        if self.online_update_step <= self.online_warmup_steps:
            self.online_update_stats["rejected_warmup"] += 1
            self._record_online_update(
                "rejected_warmup", force_raw, force_used, max_similarity, float("nan")
            )
            return False

        if self.online_require_anomaly and not anomalous:
            self._record_online_update(
                "covered", force_raw, force_used, max_similarity, float("nan")
            )
            return False

        if self.online_require_anomaly and self.online_anomaly_run < self.online_min_anomaly_steps:
            self.online_update_stats["rejected_anomaly_persistence"] += 1
            self._record_online_update(
                "rejected_anomaly_persistence",
                force_raw,
                force_used,
                max_similarity,
                float("nan"),
            )
            return False

        if self.online_update_step - self.online_last_update_step < self.online_update_interval_steps:
            self.online_update_stats["rejected_rate_limit"] += 1
            self._record_online_update(
                "rejected_rate_limit", force_raw, force_used, max_similarity, float("nan")
            )
            return False

        residual_stack = np.stack(tuple(self.online_residual_history), axis=0)
        residual_median = np.median(residual_stack, axis=0)
        deviation = float(np.linalg.norm(force_used - residual_median))
        if (
            self.online_residual_consistency_threshold > 0.0
            and len(self.online_residual_history) >= self.online_residual_window
            and deviation > self.online_residual_consistency_threshold
        ):
            self.online_update_stats["rejected_inconsistent"] += 1
            self._record_online_update(
                "rejected_inconsistent", force_raw, force_used, max_similarity, deviation
            )
            return False

        self.online_update_stats["candidates"] += 1
        c_new = self._project_environment_embedding(h_t, c_prior, force_used)
        if c_new is None:
            self.online_update_stats["rejected_nonfinite"] += 1
            self._record_online_update(
                "rejected_projection", force_raw, force_used, max_similarity, deviation
            )
            return False

        metadata = self._dictionary_store_metadata(force_used)
        metadata.update({
            "source": "online_residual_estimate",
            "bucket": "online",
            "max_similarity_before_update": float(max_similarity),
            "residual_deviation": float(deviation),
            "force_raw_norm": force_norm,
            "force_used_norm": float(np.linalg.norm(force_used)),
        })
        self.expert_dict.store(
            h_t,
            c_new,
            metadata=metadata,
            min_merge_index=self.online_base_dictionary_size,
        )
        store_status = self.expert_dict.last_store_status
        if store_status not in ("inserted", "merged"):
            if store_status in ("skipped_full", "skipped_balance"):
                self.online_update_stats["rejected_capacity"] += 1
            self._record_online_update(
                str(store_status), force_raw, force_used, max_similarity, deviation
            )
            return False

        self.online_last_update_step = self.online_update_step
        self.online_update_stats["accepted"] += 1
        self.online_update_stats[store_status] += 1
        self._record_online_update(
            store_status, force_raw, force_used, max_similarity, deviation
        )
        return True

    def get_online_update_report(self):
        report = dict(self.online_update_stats)
        report.update({
            "enabled": bool(self.online_update_enabled),
            "dictionary_size": int(self.expert_dict.current_size),
            "protected_dictionary_size": int(
                self.online_base_dictionary_size
                if self.online_base_dictionary_size is not None
                else self.expert_dict.current_size
            ),
            "last_status": str(self.online_last_update_status),
            "last_max_similarity": float(self.online_last_max_similarity),
            "last_residual_deviation": float(self.online_last_residual_deviation),
            "confidence_weighting": False,
            "forgetting": False,
            "pruning": False,
            "capacity_policy": "reject_new_entries",
        })
        return report

    def get_Fr(self, X, Z, imu, pd, vd, ad, meta_adapt_trigger, wind_gt):
        y = self.get_residual(X, imu)
        self.residual = y
        self.wind_gt = wind_gt
        if len(self.state_list) < self.seq_len:
            self.state_list.append(X)
            self.control_list.append(Z)
            self.force_list.append(y)
        else:
            self.state_list = self.state_list[1:]
            self.state_list.append(X)
            self.control_list = self.control_list[1:]
            self.control_list.append(Z)
            self.force_list = self.force_list[1:]
            self.force_list.append(y)

        Train = (self.state == "train")
        f_hat = self.get_f_hat(self.state_list, self.control_list, self.force_list, Train, wind_gt)
        if wind_gt is not None and np.all(np.isfinite(wind_gt)):
            self.force_err_list.append(np.linalg.norm(f_hat - wind_gt))
        else:
            self.force_err_list.append(float("nan"))

        Fr, Fr_dot = super().get_Fr(X, Z, imu, pd, vd, ad, meta_adapt_trigger, wind_gt)
        return Fr - f_hat, Fr_dot

    def get_f_hat(self, states, controls, forces, training=True, ground_truth_force=None):
        states_np = np.array(states)
        controls_np = np.array(controls)
        forces_t = torch.from_numpy(np.array(forces))
        x_seq_np = np.concatenate((states_np, controls_np), axis=1)
        h_prev_t = torch.from_numpy(self.h)
        x_seq_t = torch.from_numpy(x_seq_np)

        # Stage 1: train FGRU with fixed environment embedding
        if training and self.training_stage == 'stage1' and ground_truth_force is not None and self.c_env is not None:
            self.fgru.train()
            self.fgru_optimizer.zero_grad()
            h_seq = self.fgru(x_seq_t, h_prev_t, forces_t)
            h_last = h_seq[-1]
            # Predict force via fixed environment embedding
            h3 = torch.kron(torch.eye(3, dtype=h_last.dtype), h_last)
            c_env_t = torch.from_numpy(self.c_env)
            f_hat_t = torch.matmul(h3, c_env_t)
            y_t = torch.from_numpy(ground_truth_force)
            loss = torch.mean((f_hat_t - y_t) ** 2)
            loss.backward()
            self.fgru_optimizer.step()
            self.h = h_last.detach().numpy()
            return f_hat_t.detach().numpy()

        # Stage 2: enlarge dictionary with projected c_t
        if training and self.training_stage == 'stage2' and ground_truth_force is not None:
            self.fgru.eval()
            with torch.no_grad():
                h_seq = self.fgru(x_seq_t, h_prev_t, forces_t)
                h_last = h_seq[-1]
            h3 = torch.kron(torch.eye(3, dtype=h_last.dtype), h_last)
            if not hasattr(self, 'c') or self.c is None or self.c.shape[0] != 3 * self.hidden_dim:
                self.c = np.zeros(3 * self.hidden_dim, dtype=float)
            c_prev = torch.from_numpy(self.c)
            y_t = torch.from_numpy(ground_truth_force)
            # Project previous c onto constraint h3 c = y
            M = h3 @ h3.T
            delta = h3 @ c_prev - y_t
            update = h3.T @ torch.linalg.solve(M, delta)
            c_new = c_prev - update
            self.c = c_new.detach().numpy()
            self.h = h_last.detach().numpy()
            # Store in expert dictionary
            self.expert_dict.store(
                self.h,
                self.c,
                metadata=self._dictionary_store_metadata(ground_truth_force),
            )
            f_hat_t = (h3 @ c_new).detach().numpy()
            return f_hat_t

        # Inference or generic training fallback (inner update + retrieval)
        h_seq = self.fgru(x_seq_t, h_prev_t, forces_t)
        h_last = h_seq[-1].detach()
        self.h = h_last.numpy()

        if not training:
            c_inf = self.expert_dict.retrieve(h_last).detach().numpy()
            self.c_refer = self.get_c_refer(self.h, c_inf)
            if (
                np.linalg.norm(c_inf - self.c_refer)
                <= np.sqrt(self.hidden_dim) * self.c_refer_threshold_coefficient
            ):
                self.c = c_inf
            else:
                self.c = self.c_refer
            self.maybe_online_update(self.h, c_inf, self.residual)
        else:
            self.inner_adapt(self.h, self.c, ground_truth_force)

        opt3 = np.kron(np.eye(3), self.h)
        return np.dot(opt3, self.c)

    def inner_adapt(self, h_t, c_last, ground_truth_force):
        h3 = np.kron(np.eye(3), h_t)
        f_hat = np.dot(h3, c_last)
        error = f_hat - ground_truth_force
        gradient = np.matmul(error, h3)
        step_size = np.sum(error * error) / (np.sum(gradient * gradient) + 1e-10)
        if not np.isnan(step_size):
            self.c -= step_size * gradient
        else:
            self.c -= 0.01 * gradient
        if self.state == 'train':
            self.expert_dict.store(
                h_t,
                self.c,
                metadata=self._dictionary_store_metadata(ground_truth_force),
            )

    def get_c_refer(self, h_t, c_last):
        h3 = np.kron(np.eye(3), h_t)
        f_hat = np.dot(h3, c_last)
        error = f_hat - self.residual
        gradient = np.matmul(error, h3)
        step_size = np.sum(error * error) / (np.sum(gradient * gradient) + 1e-10)
        if not np.isnan(step_size):
            c = c_last - step_size * gradient
        else:
            c = c_last - 0.01 * gradient
        return c

    def save(self, path):
        state = {
            'fgru': self.fgru.state_dict(),
            'hidden_dim': self.hidden_dim,
            'expert_dim': self.expert_dict.value_dim,
            'expert_max_entries': self.expert_dict.max_entries,
            'expert_size': self.expert_dict.current_size,
            'expert_keys': self.expert_dict.keys[:self.expert_dict.current_size].detach().cpu().clone(),
            'expert_values': self.expert_dict.values[:self.expert_dict.current_size].detach().cpu().clone(),
            'expert_normalize_keys': self.expert_dict.normalize_keys,
            'expert_temperature': self.expert_dict.temperature,
            'expert_min_cosine_distance': self.expert_dict.min_cosine_distance,
            'expert_max_entries_per_bucket': self.expert_dict.max_entries_per_bucket,
            'expert_ema_alpha': self.expert_dict.ema_alpha,
            'expert_entry_counts': self.expert_dict.entry_counts[:self.expert_dict.current_size].detach().cpu().clone(),
            'expert_entry_metadata': self.expert_dict.entry_metadata[:self.expert_dict.current_size],
        }
        torch.save(state, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.fgru.load_state_dict(ckpt['fgru'])
        size = int(ckpt.get('expert_size', 0))
        current_dict = self.expert_dict
        max_entries = max(int(ckpt.get('expert_max_entries', current_dict.max_entries)), size)
        key_dim = ckpt.get('hidden_dim', self.hidden_dim)
        value_dim = ckpt.get('expert_dim', 99)
        self.expert_dict = ExpertDictionary(
            key_dim=key_dim,
            value_dim=value_dim,
            max_entries=max_entries,
            normalize_keys=ckpt.get('expert_normalize_keys', current_dict.normalize_keys),
            temperature=ckpt.get('expert_temperature', current_dict.temperature),
            min_cosine_distance=ckpt.get(
                'expert_min_cosine_distance', current_dict.min_cosine_distance
            ),
            max_entries_per_bucket=ckpt.get(
                'expert_max_entries_per_bucket', current_dict.max_entries_per_bucket
            ),
            ema_alpha=ckpt.get('expert_ema_alpha', current_dict.ema_alpha),
        )
        if size > 0:
            self.expert_dict.load_entries(
                ckpt['expert_keys'],
                ckpt['expert_values'],
                metadata=ckpt.get('expert_entry_metadata'),
                counts=ckpt.get('expert_entry_counts'),
                normalize_existing=bool(ckpt.get('expert_normalize_keys', False)),
            )
        self.online_base_dictionary_size = int(self.expert_dict.current_size)

    def begin_stage1_episode(self, c_env):
        self.training_stage = 'stage1'
        if isinstance(c_env, np.ndarray):
            self.c_env = c_env
        else:
            self.c_env = np.array(c_env, dtype=float)
        self.c = np.zeros(3 * self.hidden_dim, dtype=float)

    def end_stage1_episode(self):
        self.c_env = None

    def begin_stage2(self, reset_dictionary=False):
        self.training_stage = 'stage2'
        if reset_dictionary:
            self.expert_dict.clear()


class AeroACEFixedGate(AeroACE):
    """Ablation variant: AeroACE with fixed gate g_t = 0.5 (no force-gating).

    Replaces FGRUCell with FGRUCellFixedGate so the gate is always 0.5,
    isolating the contribution of the force-gating mechanism from the general
    recurrent structure.
    """
    def __init__(self, input_dim=17, hidden_dim=33, expert_dim=99, seq_len=10,
                 given_pid=False, p=0, i=0, d=0, **dict_kwargs):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, expert_dim=expert_dim,
                         seq_len=seq_len, given_pid=given_pid, p=p, i=i, d=d,
                         **dict_kwargs)
        # Replace FGRU with fixed-gate variant; re-create optimizer for new params
        self.fgru = FGRUCellFixedGate(input_dim, hidden_dim)
        self.fgru_optimizer = optim.Adam(self.fgru.parameters(), lr=1e-4)


class AeroACEVanillaGRU(AeroACE):
    """Ablation using a standard learned GRU instead of the force-driven gate."""

    def __init__(self, input_dim=17, hidden_dim=33, expert_dim=99, seq_len=10,
                 given_pid=False, p=0, i=0, d=0, **dict_kwargs):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, expert_dim=expert_dim,
                         seq_len=seq_len, given_pid=given_pid, p=p, i=i, d=d,
                         **dict_kwargs)
        self.fgru = VanillaGRUCell(input_dim, hidden_dim)
        self.fgru_optimizer = optim.Adam(self.fgru.parameters(), lr=1e-4)


class NeuralBEMController(PIDController):
    def __init__(
        self,
        input_dim=17,
        hidden_dim=33,
        expert_dim=99,
        seq_len=10,
        inner_lr=1e-4,
        use_dict=True,
        given_pid=False,
        p=0,
        i=0,
        d=0,
    ):
        super().__init__(given_pid=given_pid, p=p, i=i, d=d)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.expert_dim = int(expert_dim)
        self.seq_len = int(seq_len)
        self.inner_lr = float(inner_lr)
        self.use_dict = bool(use_dict)

        self.fgru = FGRUCell(self.input_dim, self.hidden_dim)
        self.expert_dict = ExpertDictionary(self.hidden_dim, self.expert_dim)
        self.fgru_optimizer = optim.Adam(self.fgru.parameters(), lr=self.inner_lr)
        self.force_err_list = []
        self.reset_controller()

    def reset_controller(self):
        super().reset_controller()
        self.motor_speed = np.zeros(4)
        self.state_list = []
        self.control_list = []
        self.force_list = []
        self.h = np.zeros((self.hidden_dim,))
        self.c = np.zeros((3 * self.hidden_dim,), dtype=float)
        self.residual = np.zeros(3)
        self.last_f_hat = np.zeros(3)
        self.force_err_list = []

    def mixer(self, torque_sp, T_sp):
        self.motor_speed = super().mixer(torque_sp, T_sp)
        return self.motor_speed

    def get_residual(self, X, imu):
        q = X[3:7]
        R = rowan.to_matrix(q)
        H = self.params['m'] * np.eye(3)
        G = np.array((0., 0., self.params['g'] * self.params['m']))
        T = self.params['C_T'] * sum(self.motor_speed ** 2)
        u = T * R @ np.array((0., 0., 1.))
        y = (H @ imu[0:3] + G - u)
        return y

    def _append_history(self, X, Z, y):
        self.state_list.append(X.copy())
        self.control_list.append(Z.copy())
        self.force_list.append(y.copy())
        if len(self.state_list) > self.seq_len:
            self.state_list = self.state_list[-self.seq_len:]
            self.control_list = self.control_list[-self.seq_len:]
            self.force_list = self.force_list[-self.seq_len:]

    def inner_adapt(self, h_t, c_last, target_force):
        h3 = np.kron(np.eye(3), h_t)
        f_hat = np.dot(h3, c_last)
        err = f_hat - target_force
        grad = np.matmul(err, h3)
        step = np.sum(err * err) / (np.sum(grad * grad) + 1e-10)
        if not np.isfinite(step):
            step = 0.01
        self.c -= step * grad

    def get_f_hat(self, training=True, ground_truth_force=None):
        if len(self.state_list) == 0:
            return np.zeros(3)

        states_np = np.array(self.state_list)
        controls_np = np.array(self.control_list)
        forces_np = np.array(self.force_list)
        x_seq_np = np.concatenate((states_np, controls_np), axis=1)

        x_seq_t = torch.from_numpy(x_seq_np)
        h_prev_t = torch.from_numpy(self.h)
        forces_t = torch.from_numpy(forces_np)

        if training and ground_truth_force is not None:
            self.fgru.train()
            self.fgru_optimizer.zero_grad()
            h_seq = self.fgru(x_seq_t, h_prev_t, forces_t)
            h_last = h_seq[-1]
            h3 = torch.kron(torch.eye(3, dtype=h_last.dtype), h_last)
            c_t = torch.from_numpy(self.c)
            f_hat_t = torch.matmul(h3, c_t)
            y_t = torch.from_numpy(ground_truth_force)
            loss = torch.mean((f_hat_t - y_t) ** 2)
            loss.backward()
            self.fgru_optimizer.step()
            self.h = h_last.detach().numpy()
            self.inner_adapt(self.h, self.c, ground_truth_force)
            if self.use_dict:
                self.expert_dict.store(self.h, self.c)
            return f_hat_t.detach().numpy()

        self.fgru.eval()
        with torch.no_grad():
            h_seq = self.fgru(x_seq_t, h_prev_t, forces_t)
            h_last = h_seq[-1]
        self.h = h_last.numpy()

        if self.use_dict and self.expert_dict.current_size > 0:
            c_inf = self.expert_dict.retrieve(h_last).detach().numpy()
            if np.all(np.isfinite(c_inf)):
                self.c = c_inf
        h3_np = np.kron(np.eye(3), self.h)
        f_hat = np.dot(h3_np, self.c)
        return np.nan_to_num(f_hat, nan=0.0, posinf=0.0, neginf=0.0)

    def get_Fr(self, X, Z, imu, pd, vd, ad, meta_adapt_trigger, wind_gt):
        y = self.get_residual(X, imu)
        self.residual = y
        self._append_history(X, Z, y)
        training = (self.state == 'train')
        f_hat = self.get_f_hat(training=training, ground_truth_force=wind_gt)
        self.last_f_hat = f_hat.copy()
        if wind_gt is not None and np.all(np.isfinite(wind_gt)):
            self.force_err_list.append(float(np.linalg.norm(f_hat - wind_gt)))
        Fr, Fr_dot = super().get_Fr(X, Z, imu, pd, vd, ad, meta_adapt_trigger, wind_gt)
        return Fr - f_hat, Fr_dot

    def save(self, path):
        state = {
            'fgru': self.fgru.state_dict(),
            'hidden_dim': self.hidden_dim,
            'expert_dim': self.expert_dict.value_dim,
            'expert_max_entries': self.expert_dict.max_entries,
            'expert_size': self.expert_dict.current_size,
            'expert_keys': self.expert_dict.keys[:self.expert_dict.current_size].detach().cpu().clone(),
            'expert_values': self.expert_dict.values[:self.expert_dict.current_size].detach().cpu().clone(),
            'c': self.c.copy(),
        }
        torch.save(state, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.fgru.load_state_dict(ckpt['fgru'])
        max_entries = ckpt.get('expert_max_entries', 10000)
        key_dim = ckpt.get('hidden_dim', self.hidden_dim)
        value_dim = ckpt.get('expert_dim', self.expert_dim)
        self.expert_dict = ExpertDictionary(key_dim=key_dim, value_dim=value_dim, max_entries=max_entries)
        size = ckpt.get('expert_size', 0)
        if size > 0:
            self.expert_dict.keys[:size] = ckpt['expert_keys']
            self.expert_dict.values[:size] = ckpt['expert_values']
            self.expert_dict.current_size = int(size)
        if 'c' in ckpt:
            c = np.asarray(ckpt['c']).reshape(-1)
            if c.shape[0] == 3 * self.hidden_dim:
                self.c = c.copy()


# ---------------------------------------------------------------------------
# Powerformer: Transformer with Weighted Causal Attention
# ---------------------------------------------------------------------------

class PowerformerWeightedCausalAttention(nn.Module):
    """Multi-head self-attention with per-head learned exponential decay (WCA).

    For position i attending to position j (j <= i), the attention logit is
    biased by -alpha_h * (i - j), where alpha_h > 0 is a per-head learnable
    parameter. This gives recent context higher weight while preserving causality.
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Learnable per-head log-decay; alpha = softplus(log_decay) > 0
        self.log_decay = nn.Parameter(torch.zeros(nhead))

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, _ = x.shape
        H, Dh = self.nhead, self.d_head

        Q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # (B,H,T,Dh)
        K = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        scale = Dh ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B,H,T,T)

        # Causal mask: future positions set to -inf
        causal_mask = torch.full((T, T), float('-inf'), device=x.device, dtype=x.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)  # (T,T)

        # Distance matrix: dist[i,j] = i - j (>= 0 for causal positions)
        idx = torch.arange(T, device=x.device, dtype=x.dtype)
        dist = idx.unsqueeze(0) - idx.unsqueeze(1)  # dist[i,j] = i - j

        # Decay bias: -alpha_h * (i - j) for valid positions
        alpha = F.softplus(self.log_decay)             # (H,), positive
        decay_bias = -alpha.view(H, 1, 1) * dist.unsqueeze(0)  # (H,T,T)

        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0) + decay_bias.unsqueeze(0)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)                   # (B,H,T,Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class PowerformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer using Weighted Causal Attention."""

    def __init__(self, d_model, nhead, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = PowerformerWeightedCausalAttention(d_model, nhead, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PowerformerModel(nn.Module):
    """Powerformer: sequence-to-force predictor using WCA-Transformer.

    Input:  (B, seq_len, input_dim=14)  [v, q, omega, u per step]
    Output: (B, 3)                       world-frame aerodynamic force
    """

    def __init__(self, input_dim=14, seq_len=20, d_model=64, nhead=4,
                 num_layers=2, ffn_dim=128, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)

        # Fixed sinusoidal positional encoding
        pe = torch.zeros(seq_len, d_model, dtype=torch.double)
        pos = torch.arange(seq_len, dtype=torch.double).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.double) *
                        (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, seq_len, d_model)

        self.layers = nn.ModuleList([
            PowerformerEncoderLayer(d_model, nhead, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 3)

    def forward(self, x):
        # x: (B, T, input_dim)
        z = self.input_proj(x) + self.pe[:, :x.size(1), :]
        for layer in self.layers:
            z = layer(z)
        z = self.norm(z)
        out = self.output_proj(z[:, -1, :])  # use last timestep
        return out


class PowerformerController(PIDController):
    """PID base + PowerFormer residual-force compensator."""

    def __init__(
        self,
        seq_len=16,
        patch_len=4,
        patch_stride=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        ffn_dim=128,
        dropout=0.1,
        head_dropout=0.1,
        mask_type='weight_powerlaw',
        alpha=0.5,
        force_bound=0.5,
        compensation_gain=1.0,
        given_pid=False,
        p=0,
        i=0,
        d=0,
    ):
        super().__init__(given_pid=given_pid, p=p, i=i, d=d)
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.feature_dim = 14
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.ffn_dim = int(ffn_dim)
        self.dropout_rate = float(dropout)
        self.head_dropout = float(head_dropout)
        self.mask_type = str(mask_type)
        self.alpha = float(alpha)
        self.force_bound = float(force_bound)
        self.compensation_gain = float(compensation_gain)

        self.force_model = PowerFormerForceModel(
            input_dim=self.feature_dim,
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            patch_stride=self.patch_stride,
            d_model=self.d_model,
            n_heads=self.nhead,
            n_layers=self.num_layers,
            d_ff=self.ffn_dim,
            dropout=self.dropout_rate,
            head_dropout=self.head_dropout,
            attn_dropout=0.0,
            mask_type=self.mask_type,
            alpha=self.alpha,
        ).to(device).double()

        self.position_tick_counter = 0
        self.last_motor_speed = None
        self.hist_features = []
        self.last_f_hat = np.zeros(3)
        self.force_err_list = []

    def reset_controller(self):
        super().reset_controller()
        self.position_tick_counter = 0
        self.last_motor_speed = None
        self.hist_features = []
        self.last_f_hat = np.zeros(3)
        self.force_err_list = []

    def mixer(self, torque_sp, T_sp):
        motor_speed = super().mixer(torque_sp, T_sp)
        self.last_motor_speed = motor_speed.copy()
        return motor_speed

    def build_feature(self, X, Z, pd=None, vd=None, ad=None, imu=None):
        u_feat = self.last_motor_speed if self.last_motor_speed is not None else Z
        return np.concatenate((X[7:10], X[3:7], X[10:13], u_feat))

    def predict_force_or_disturbance(self):
        if len(self.hist_features) < self.seq_len:
            return np.zeros(3)
        hist = np.asarray(self.hist_features[-self.seq_len:], dtype=np.float64)
        hist_t = torch.from_numpy(hist).to(device).double().unsqueeze(0)

        self.force_model.eval()
        with torch.no_grad():
            f_hat = self.force_model(hist_t).cpu().numpy().reshape(-1)

        if f_hat.shape[0] != 3 or (not np.all(np.isfinite(f_hat))):
            return np.zeros(3)
        return np.clip(f_hat, -self.force_bound, self.force_bound)

    def get_Fr(self, X, Z, imu, pd, vd, ad, meta_adapt_trigger, wind_gt):
        Fr_nominal, Fr_dot = super().get_Fr(
            X, Z=Z, imu=imu, pd=pd, vd=vd, ad=ad,
            meta_adapt_trigger=meta_adapt_trigger, wind_gt=wind_gt,
        )

        self.position_tick_counter += 1
        feat = self.build_feature(X, Z, pd=pd, vd=vd, ad=ad, imu=imu)
        self.hist_features.append(feat.copy())
        if len(self.hist_features) > self.seq_len:
            self.hist_features = self.hist_features[-self.seq_len:]

        if len(self.hist_features) >= self.seq_len:
            self.last_f_hat = self.predict_force_or_disturbance()
        else:
            self.last_f_hat = np.zeros(3)

        if wind_gt is not None and np.all(np.isfinite(wind_gt)):
            self.force_err_list.append(float(np.linalg.norm(self.last_f_hat - wind_gt)))

        Fr_comp = Fr_nominal - self.compensation_gain * self.last_f_hat
        return Fr_comp, Fr_dot

    def fit_force_model(
        self,
        train_hist,
        train_targets,
        val_hist=None,
        val_targets=None,
        epochs=100,
        batch_size=64,
        lr=1e-3,
        weight_decay=0.0,
    ):
        if len(train_hist) == 0:
            raise RuntimeError("Empty Powerformer training set.")
        if train_hist.ndim != 3 or train_hist.shape[1] != self.seq_len or train_hist.shape[2] != self.feature_dim:
            raise ValueError(
                f"Expected train_hist shape (N,{self.seq_len},{self.feature_dim}), got {train_hist.shape}"
            )
        if train_targets.ndim != 2 or train_targets.shape[1] != 3:
            raise ValueError(f"Expected train_targets shape (N,3), got {train_targets.shape}")

        self.force_model.train()
        optimizer = optim.Adam(self.force_model.parameters(), lr=lr, weight_decay=weight_decay)
        mse = nn.MSELoss()
        n = len(train_hist)

        for epoch in range(int(epochs)):
            perm = np.random.permutation(n)
            total_loss = 0.0
            total_count = 0
            for st in range(0, n, int(batch_size)):
                idx = perm[st:st + int(batch_size)]
                x = torch.from_numpy(train_hist[idx]).to(device).double()
                y = torch.from_numpy(train_targets[idx]).to(device).double()
                optimizer.zero_grad()
                yhat = self.force_model(x)
                loss = mse(yhat, y)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss in Powerformer training.")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(idx)
                total_count += len(idx)

            train_mse = total_loss / max(total_count, 1)
            msg = f"[Powerformer][Epoch {epoch+1}/{epochs}] train_mse={train_mse:.6f}"
            if val_hist is not None and val_targets is not None and len(val_hist) > 0:
                val_metrics = self.eval_force_model(val_hist, val_targets)
                msg += f" val_rmse={val_metrics['rmse']:.6f} val_mae={val_metrics['mae']:.6f}"
            print(msg)

        self.force_model.eval()

    def eval_force_model(self, hist, targets):
        if len(hist) == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'axis_rmse': np.array([np.nan, np.nan, np.nan])}

        x = torch.from_numpy(hist).to(device).double()
        self.force_model.eval()
        with torch.no_grad():
            pred = self.force_model(x).cpu().numpy()
        err = pred - targets
        rmse = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
        mae = float(np.mean(np.linalg.norm(err, axis=1)))
        axis_rmse = np.sqrt(np.mean(err ** 2, axis=0))
        return {'rmse': rmse, 'mae': mae, 'axis_rmse': axis_rmse}

    def save(self, path):
        state = {
            'force_model': self.force_model.state_dict(),
            'seq_len': self.seq_len,
            'patch_len': self.patch_len,
            'patch_stride': self.patch_stride,
            'feature_dim': self.feature_dim,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'ffn_dim': self.ffn_dim,
            'dropout': self.dropout_rate,
            'head_dropout': self.head_dropout,
            'mask_type': self.mask_type,
            'alpha': self.alpha,
        }
        torch.save(state, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        if 'force_model' in ckpt:
            state_dict = ckpt['force_model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        self.force_model.load_state_dict(state_dict)
        self.force_model.eval()


class PiTransformerController(PIDController):
    """PID base + Pi-Transformer-inspired residual-force compensator."""

    def __init__(
        self,
        seq_len=16,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        gamma=1.0,
        sigma=4.0,
        tau_eps=1e-3,
        alpha_prior=0.0,
        force_bound=0.25,
        compensation_gain=1.0,
        given_pid=False,
        p=0,
        i=0,
        d=0,
    ):
        """Initialize controller and Pi-Transformer force model.

        Sequence input shape: (B, T, F) with F=14 as [v(3), q(4), omega(3), u(4)].
        Model output shape: (B, 3) world-frame force residual.
        """
        super().__init__(given_pid=given_pid, p=p, i=i, d=d)
        self.device = device
        self.dtype = torch.double

        self.seq_len = int(max(1, seq_len))
        self.feature_dim = 14
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.d_ff = int(d_ff)
        self.dropout = float(dropout)
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.tau_eps = float(tau_eps)
        self.alpha_prior = float(alpha_prior)
        self.force_bound = float(force_bound)
        self.compensation_gain = float(compensation_gain)

        self.model = PiTransformerForceModel(
            input_dim=self.feature_dim,
            seq_len=self.seq_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            gamma=self.gamma,
            sigma=self.sigma,
            tau_eps=self.tau_eps,
            alpha_prior=self.alpha_prior,
        ).to(self.device).double()

        self.feature_mean = np.zeros(self.feature_dim, dtype=np.float64)
        self.feature_std = np.ones(self.feature_dim, dtype=np.float64)
        self.reset_controller()

    def reset_controller(self):
        """Reset PID and Pi-Transformer runtime buffers."""
        super().reset_controller()
        self.position_tick_counter = 0
        self.last_motor_speed = np.zeros(4, dtype=np.float64)
        self.hist_features = []
        self.last_f_hat = np.zeros(3, dtype=np.float64)
        self.last_diag = {}
        self.force_err_list = []

    def mixer(self, torque_sp, T_sp):
        """Run base mixer and cache latest motor speed command."""
        motor_speed = super().mixer(torque_sp, T_sp)
        self.last_motor_speed = motor_speed.copy()
        return motor_speed

    def build_feature(self, X, Z, imu=None, pd=None, vd=None, ad=None):
        """Build one position-tick feature vector.

        Inputs are read at current control tick; output shape is (14,).
        """
        u_feat = self.last_motor_speed if self.last_motor_speed is not None else Z
        feat = np.concatenate((X[7:10], X[3:7], X[10:13], u_feat))
        return feat.astype(np.float64, copy=False)

    def _normalize_hist(self, hist):
        """Normalize history tensor with stored train-set statistics.

        Args:
            hist: np.ndarray with shape (N, T, F) or (T, F).
        """
        eps = 1e-9
        std = np.where(self.feature_std > eps, self.feature_std, 1.0)
        return (hist - self.feature_mean) / std

    def _build_divergence_loss(self, aux, stopgrad_series=False, stopgrad_prior=False):
        """Average symmetric KL divergence over layers."""
        div_list = []
        for s, p in zip(aux['series_attn'], aux['prior_attn']):
            div_list.append(
                symmetric_kl_divergence(
                    s,
                    p,
                    stopgrad_series=stopgrad_series,
                    stopgrad_prior=stopgrad_prior,
                    eps=1e-9,
                )
            )
        return torch.stack(div_list).mean()

    def _build_regularization_loss(
        self,
        aux,
        lambda_smooth=1e-3,
        lambda_prior=1e-4,
        lambda_distill=0.0,
        tau_ref=1.0,
    ):
        """Compute Lreg = lambda_smooth*Rsmooth + lambda_prior*Rprior + lambda_distill*Rdistill."""
        terms = prior_regularization_terms(aux['H'], aux['tau'], tau_ref=tau_ref)
        reg = (
            float(lambda_smooth) * terms['smooth']
            + float(lambda_prior) * terms['prior']
            + float(lambda_distill) * terms['distill']
        )
        return reg, terms

    def predict_force(self, return_aux=False):
        """Predict force from buffered history.

        Returns:
            f_hat: np.ndarray with shape (3,).
            aux(optional): diagnostics dict.
        """
        if len(self.hist_features) < self.seq_len:
            if return_aux:
                return np.zeros(3, dtype=np.float64), {}
            return np.zeros(3, dtype=np.float64)

        hist = np.asarray(self.hist_features[-self.seq_len:], dtype=np.float64)
        hist = self._normalize_hist(hist)
        hist_t = torch.from_numpy(hist).to(self.device).double().unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            if return_aux:
                f_hat_t, aux = self.model(hist_t, return_aux=True)
            else:
                f_hat_t = self.model(hist_t, return_aux=False)
                aux = None

        f_hat = f_hat_t.cpu().numpy().reshape(-1)
        if f_hat.shape[0] != 3 or (not np.all(np.isfinite(f_hat))):
            f_hat = np.zeros(3, dtype=np.float64)
        f_hat = np.clip(f_hat, -self.force_bound, self.force_bound)

        if not return_aux:
            return f_hat

        diag = {}
        if aux is not None:
            diag = {
                'mean_mismatch': float(aux['mean_mismatch'].item()),
                'series_attn': [x.cpu().numpy() for x in aux['series_attn']],
                'prior_attn': [x.cpu().numpy() for x in aux['prior_attn']],
                'H': [x.cpu().numpy() for x in aux['H']],
                'tau': [x.cpu().numpy() for x in aux['tau']],
                'theta': [x.cpu().numpy() for x in aux['theta']],
            }
        return f_hat, diag

    def get_Fr(self, X, Z, imu, pd, vd, ad, meta_adapt_trigger, wind_gt):
        """Position-loop force command with learned residual compensation."""
        Fr_nominal, Fr_dot = super().get_Fr(
            X, Z=Z, imu=imu, pd=pd, vd=vd, ad=ad,
            meta_adapt_trigger=meta_adapt_trigger, wind_gt=wind_gt,
        )

        self.position_tick_counter += 1
        feat = self.build_feature(X, Z=Z, imu=imu, pd=pd, vd=vd, ad=ad)
        self.hist_features.append(feat.copy())
        if len(self.hist_features) > self.seq_len:
            self.hist_features = self.hist_features[-self.seq_len:]

        if len(self.hist_features) >= self.seq_len:
            self.last_f_hat, self.last_diag = self.predict_force(return_aux=True)
        else:
            self.last_f_hat = np.zeros(3, dtype=np.float64)
            self.last_diag = {}

        if wind_gt is not None and np.all(np.isfinite(wind_gt)):
            self.force_err_list.append(float(np.linalg.norm(self.last_f_hat - wind_gt)))

        Fr_comp = Fr_nominal - self.compensation_gain * self.last_f_hat
        return Fr_comp, Fr_dot

    def fit_force_model(
        self,
        train_hist,
        train_targets,
        val_hist=None,
        val_targets=None,
        epochs=100,
        batch_size=64,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        lambda_div=0.1,
        lambda_smooth=1e-3,
        lambda_prior=1e-4,
        lambda_distill=0.0,
        tau_ref=1.0,
    ):
        """Train Pi-Transformer with stop-gradient alternating symmetric-KL."""
        if len(train_hist) == 0:
            raise RuntimeError("Empty Pi-Transformer training set.")
        if train_hist.ndim != 3 or train_hist.shape[1] != self.seq_len or train_hist.shape[2] != self.feature_dim:
            raise ValueError(
                f"Expected train_hist shape (N,{self.seq_len},{self.feature_dim}), got {train_hist.shape}"
            )
        if train_targets.ndim != 2 or train_targets.shape[1] != 3:
            raise ValueError(f"Expected train_targets shape (N,3), got {train_targets.shape}")

        # Feature normalization statistics from training split only.
        flat = train_hist.reshape(-1, self.feature_dim)
        self.feature_mean = np.mean(flat, axis=0).astype(np.float64)
        self.feature_std = np.std(flat, axis=0).astype(np.float64)
        self.feature_std = np.where(self.feature_std > 1e-9, self.feature_std, 1.0)

        train_hist_n = self._normalize_hist(train_hist)

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        mse = nn.MSELoss()
        n = len(train_hist_n)

        for epoch in range(int(epochs)):
            perm = np.random.permutation(n)
            total_loss = 0.0
            total_force = 0.0
            total_div = 0.0
            total_reg = 0.0
            total_count = 0

            for st in range(0, n, int(batch_size)):
                idx = perm[st:st + int(batch_size)]
                x = torch.from_numpy(train_hist_n[idx]).to(self.device).double()
                y = torch.from_numpy(train_targets[idx]).to(self.device).double()

                # Pass 1: D(S || sg(P))
                optimizer.zero_grad()
                yhat_1, aux_1 = self.model(x, return_aux=True)
                force_loss_1 = mse(yhat_1, y)
                div_loss_1 = self._build_divergence_loss(aux_1, stopgrad_series=False, stopgrad_prior=True)
                reg_loss_1, _ = self._build_regularization_loss(
                    aux_1,
                    lambda_smooth=lambda_smooth,
                    lambda_prior=lambda_prior,
                    lambda_distill=lambda_distill,
                    tau_ref=tau_ref,
                )
                loss_1 = force_loss_1 + float(lambda_div) * div_loss_1 + reg_loss_1
                if not torch.isfinite(loss_1):
                    raise ValueError("Non-finite loss in Pi-Transformer pass-1.")
                loss_1.backward()
                if grad_clip is not None and grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                optimizer.step()

                # Pass 2: D(P || sg(S))
                optimizer.zero_grad()
                yhat_2, aux_2 = self.model(x, return_aux=True)
                force_loss_2 = mse(yhat_2, y)
                div_loss_2 = self._build_divergence_loss(aux_2, stopgrad_series=True, stopgrad_prior=False)
                reg_loss_2, _ = self._build_regularization_loss(
                    aux_2,
                    lambda_smooth=lambda_smooth,
                    lambda_prior=lambda_prior,
                    lambda_distill=lambda_distill,
                    tau_ref=tau_ref,
                )
                loss_2 = force_loss_2 + float(lambda_div) * div_loss_2 + reg_loss_2
                if not torch.isfinite(loss_2):
                    raise ValueError("Non-finite loss in Pi-Transformer pass-2.")
                loss_2.backward()
                if grad_clip is not None and grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                optimizer.step()

                total_loss += float((loss_1.item() + loss_2.item()) * 0.5) * len(idx)
                total_force += float((force_loss_1.item() + force_loss_2.item()) * 0.5) * len(idx)
                total_div += float((div_loss_1.item() + div_loss_2.item()) * 0.5) * len(idx)
                total_reg += float((reg_loss_1.item() + reg_loss_2.item()) * 0.5) * len(idx)
                total_count += len(idx)

            mean_loss = total_loss / max(total_count, 1)
            mean_force = total_force / max(total_count, 1)
            mean_div = total_div / max(total_count, 1)
            mean_reg = total_reg / max(total_count, 1)
            msg = (
                f"[PiTransformer][Epoch {epoch+1}/{epochs}] "
                f"loss={mean_loss:.6f} force={mean_force:.6f} div={mean_div:.6f} reg={mean_reg:.6f}"
            )
            if val_hist is not None and val_targets is not None and len(val_hist) > 0:
                val_metrics = self.eval_force_model(val_hist, val_targets)
                msg += f" val_rmse={val_metrics['rmse']:.6f} val_mae={val_metrics['mae']:.6f}"
            print(msg)

        self.model.eval()

    def eval_force_model(self, hist, targets):
        """Evaluate force regression metrics on a sequence dataset."""
        if len(hist) == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'axis_rmse': np.array([np.nan, np.nan, np.nan])}

        hist_n = self._normalize_hist(hist)
        x = torch.from_numpy(hist_n).to(self.device).double()
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x).cpu().numpy()

        err = pred - targets
        rmse = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
        mae = float(np.mean(np.linalg.norm(err, axis=1)))
        axis_rmse = np.sqrt(np.mean(err ** 2, axis=0))
        return {'rmse': rmse, 'mae': mae, 'axis_rmse': axis_rmse}

    def save(self, path):
        """Save model weights and normalization statistics."""
        state = {
            'pi_transformer_model': self.model.state_dict(),
            'seq_len': self.seq_len,
            'feature_dim': self.feature_dim,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
            'gamma': self.gamma,
            'sigma': self.sigma,
            'tau_eps': self.tau_eps,
            'alpha_prior': self.alpha_prior,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
        }
        torch.save(state, path)

    def load(self, path, map_location=None):
        """Load model weights and optional normalization statistics."""
        ckpt = torch.load(path, map_location=map_location)
        if 'pi_transformer_model' in ckpt:
            state_dict = ckpt['pi_transformer_model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        self.model.load_state_dict(state_dict)
        self.model.eval()

        if isinstance(ckpt, dict):
            if 'feature_mean' in ckpt:
                self.feature_mean = np.asarray(ckpt['feature_mean'], dtype=np.float64)
            if 'feature_std' in ckpt:
                std = np.asarray(ckpt['feature_std'], dtype=np.float64)
                self.feature_std = np.where(std > 1e-9, std, 1.0)


class ClusterCausalAttentionController(PIDController):
    """PID base + cluster-biased causal attention residual-force compensator."""

    def __init__(
        self,
        seq_len=12,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ff_dim=128,
        dropout=0.1,
        beta=16.0,
        center_c=4.0,
        lambda_same_cluster=0.5,
        lambda_center_token=0.3,
        lambda_other_cluster=0.2,
        lambda_far=0.2,
        far_radius=1.0,
        force_bound=200.0,
        compensation_gain=0.1,
        given_pid=False,
        p=0,
        i=0,
        d=0,
    ):
        """Initialize controller and cluster-causal force model."""
        super().__init__(given_pid=given_pid, p=p, i=i, d=d)
        self.device = device
        self.dtype = torch.double
        self.seq_len = int(max(1, seq_len))
        self.feature_dim = 14
        self.force_bound = float(force_bound)
        self.compensation_gain = float(compensation_gain)

        self.model = ClusterCausalForceModel(
            input_dim=self.feature_dim,
            seq_len=self.seq_len,
            d_model=int(d_model),
            n_heads=int(n_heads),
            n_layers=int(n_layers),
            ff_dim=int(ff_dim),
            dropout=float(dropout),
            beta=float(beta),
            center_c=float(center_c),
            lambda_same_cluster=float(lambda_same_cluster),
            lambda_center_token=float(lambda_center_token),
            lambda_other_cluster=float(lambda_other_cluster),
            lambda_far=float(lambda_far),
            far_radius=float(far_radius),
        ).to(self.device).double()

        self.feature_mean = np.zeros(self.feature_dim, dtype=np.float64)
        self.feature_std = np.ones(self.feature_dim, dtype=np.float64)
        self.reset_controller()

    def reset_controller(self):
        """Reset runtime buffers for new rollout."""
        super().reset_controller()
        self.hist_features = []
        self.last_f_hat = np.zeros(3, dtype=np.float64)
        self.last_motor_speed = np.zeros(4, dtype=np.float64)
        self.last_diag = {}
        self.force_err_list = []

    def mixer(self, torque_sp, T_sp):
        """Run base mixer and cache latest motor speed."""
        motor_speed = super().mixer(torque_sp, T_sp)
        self.last_motor_speed = motor_speed.copy()
        return motor_speed

    def build_feature(self, X, Z, imu=None, pd=None, vd=None, ad=None):
        """Build one position-tick feature vector with shape (14,)."""
        u_feat = self.last_motor_speed if self.last_motor_speed is not None else Z
        feat = np.concatenate((X[7:10], X[3:7], X[10:13], u_feat))
        return feat.astype(np.float64, copy=False)

    def _normalize_hist(self, hist):
        std = np.where(self.feature_std > 1e-9, self.feature_std, 1.0)
        return (hist - self.feature_mean) / std

    def predict_force(self, return_aux=False):
        """Predict force compensation from buffered history.

        Returns:
            f_hat: np.ndarray shape (3,)
        """
        if len(self.hist_features) < self.seq_len:
            if return_aux:
                return np.zeros(3, dtype=np.float64), {}
            return np.zeros(3, dtype=np.float64)

        hist = np.asarray(self.hist_features[-self.seq_len:], dtype=np.float64)
        hist = self._normalize_hist(hist)
        hist_t = torch.from_numpy(hist).to(self.device).double().unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            out = self.model(hist_t, return_aux=bool(return_aux))

        f_hat = out['force_pred'].cpu().numpy().reshape(-1)
        if f_hat.shape[0] != 3 or (not np.all(np.isfinite(f_hat))):
            f_hat = np.zeros(3, dtype=np.float64)
        f_hat = np.clip(f_hat, -self.force_bound, self.force_bound)

        if not return_aux:
            return f_hat

        diag = {
            'compactness': float(out['compactness'].item()) if 'compactness' in out else 0.0,
            'separation': float(out['separation'].item()) if 'separation' in out else 0.0,
            'center_attn_mass': float(out['center_attn_mass'].item()) if 'center_attn_mass' in out else 0.0,
        }
        if 'attn_aux' in out and len(out['attn_aux']) > 0:
            first = out['attn_aux'][0]
            diag.update(
                {
                    'delta': float(first['delta'].item()),
                    'center_counts': first['center_counts'].cpu().numpy(),
                    'center_indices': first['center_indices'].cpu().numpy(),
                }
            )
        return f_hat, diag

    def get_Fr(self, X, Z, imu, pd, vd, ad, meta_adapt_trigger, wind_gt):
        """Return compensated force command at position loop."""
        Fr_nom, Fr_dot = super().get_Fr(
            X, Z=Z, imu=imu, pd=pd, vd=vd, ad=ad,
            meta_adapt_trigger=meta_adapt_trigger, wind_gt=wind_gt,
        )

        feat = self.build_feature(X, Z=Z, imu=imu, pd=pd, vd=vd, ad=ad)
        self.hist_features.append(feat.copy())
        if len(self.hist_features) > self.seq_len:
            self.hist_features = self.hist_features[-self.seq_len:]

        if len(self.hist_features) >= self.seq_len:
            self.last_f_hat, self.last_diag = self.predict_force(return_aux=True)
        else:
            self.last_f_hat = np.zeros(3, dtype=np.float64)
            self.last_diag = {}

        if wind_gt is not None and np.all(np.isfinite(wind_gt)):
            self.force_err_list.append(float(np.linalg.norm(self.last_f_hat - wind_gt)))

        return Fr_nom - self.compensation_gain * self.last_f_hat, Fr_dot

    def fit_force_model(
        self,
        train_hist,
        train_targets,
        val_hist=None,
        val_targets=None,
        epochs=100,
        batch_size=64,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        lambda_main=1.0,
        lambda_phys=1.0,
        lambda_compact=1e-2,
        lambda_sep=1e-2,
        lambda_center_attn=1e-3,
    ):
        """Train model with main + cluster regularization losses."""
        if len(train_hist) == 0:
            raise RuntimeError("Empty cluster-causal training set.")
        if train_hist.ndim != 3 or train_hist.shape[1] != self.seq_len or train_hist.shape[2] != self.feature_dim:
            raise ValueError(
                f"Expected train_hist shape (N,{self.seq_len},{self.feature_dim}), got {train_hist.shape}"
            )
        if train_targets.ndim != 2 or train_targets.shape[1] != 3:
            raise ValueError(f"Expected train_targets shape (N,3), got {train_targets.shape}")

        flat = train_hist.reshape(-1, self.feature_dim)
        self.feature_mean = np.mean(flat, axis=0).astype(np.float64)
        self.feature_std = np.std(flat, axis=0).astype(np.float64)
        self.feature_std = np.where(self.feature_std > 1e-9, self.feature_std, 1.0)
        train_hist_n = self._normalize_hist(train_hist)

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        mse = nn.MSELoss()
        n = len(train_hist_n)

        for epoch in range(int(epochs)):
            perm = np.random.permutation(n)
            total_loss = 0.0
            total_main = 0.0
            total_phys = 0.0
            total_compact = 0.0
            total_sep = 0.0
            total_center = 0.0
            total_count = 0

            for st in range(0, n, int(batch_size)):
                idx = perm[st:st + int(batch_size)]
                x = torch.from_numpy(train_hist_n[idx]).to(self.device).double()
                y = torch.from_numpy(train_targets[idx]).to(self.device).double()

                out = self.model(x, return_aux=True)
                y_last = out['force_pred']
                y_seq = out['force_seq_pred']

                l_main = mse(y_last, y)
                # PITCN-style nominal bias: prefer small residual force in absence of data support.
                l_phys = mse(y_seq, torch.zeros_like(y_seq))
                l_compact = out['compactness']
                l_sep = out['separation']
                l_center_attn = -out['center_attn_mass']

                loss = (
                    float(lambda_main) * l_main
                    + float(lambda_phys) * l_phys
                    + float(lambda_compact) * l_compact
                    + float(lambda_sep) * l_sep
                    + float(lambda_center_attn) * l_center_attn
                )
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss encountered in cluster-causal training.")

                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None and grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                optimizer.step()

                bsz = len(idx)
                total_loss += float(loss.item()) * bsz
                total_main += float(l_main.item()) * bsz
                total_phys += float(l_phys.item()) * bsz
                total_compact += float(l_compact.item()) * bsz
                total_sep += float(l_sep.item()) * bsz
                total_center += float(l_center_attn.item()) * bsz
                total_count += bsz

            msg = (
                f"[ClusterCausal][Epoch {epoch+1}/{epochs}] "
                f"loss={total_loss/max(total_count,1):.6f} "
                f"main={total_main/max(total_count,1):.6f} "
                f"phys={total_phys/max(total_count,1):.6f} "
                f"compact={total_compact/max(total_count,1):.6f} "
                f"sep={total_sep/max(total_count,1):.6f} "
                f"center={total_center/max(total_count,1):.6f}"
            )
            if val_hist is not None and val_targets is not None and len(val_hist) > 0:
                val_metrics = self.eval_force_model(val_hist, val_targets)
                msg += f" val_rmse={val_metrics['rmse']:.6f} val_mae={val_metrics['mae']:.6f}"
            print(msg)

        self.model.eval()

    def eval_force_model(self, hist, targets):
        """Evaluate force prediction metrics."""
        if len(hist) == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'axis_rmse': np.array([np.nan, np.nan, np.nan])}

        hist_n = self._normalize_hist(hist)
        x = torch.from_numpy(hist_n).to(self.device).double()
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x)['force_pred'].cpu().numpy()
        err = pred - targets
        rmse = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
        mae = float(np.mean(np.linalg.norm(err, axis=1)))
        axis_rmse = np.sqrt(np.mean(err ** 2, axis=0))
        return {'rmse': rmse, 'mae': mae, 'axis_rmse': axis_rmse}

    def save(self, path):
        """Save model and feature normalization stats."""
        state = {
            'cluster_causal_model': self.model.state_dict(),
            'seq_len': self.seq_len,
            'feature_dim': self.feature_dim,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
        }
        torch.save(state, path)

    def load(self, path, map_location=None):
        """Load model and optional feature normalization stats."""
        ckpt = torch.load(path, map_location=map_location)
        if isinstance(ckpt, dict) and 'cluster_causal_model' in ckpt:
            state_dict = ckpt['cluster_causal_model']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()

        if isinstance(ckpt, dict):
            if 'feature_mean' in ckpt:
                self.feature_mean = np.asarray(ckpt['feature_mean'], dtype=np.float64)
            if 'feature_std' in ckpt:
                std = np.asarray(ckpt['feature_std'], dtype=np.float64)
                self.feature_std = np.where(std > 1e-9, std, 1.0)


class CausalTransformerController(PIDController):
    """PID base + 3-stream masked causal transformer force compensator.

    Stream definitions at position-loop ticks:
        X_t: [v(3), q(4), w(3)] -> 10D
        A_t: motor speed command -> 4D
        Y_t: previous residual-force prediction -> 3D
    """

    def __init__(
        self,
        seq_len=12,
        d_model=64,
        n_heads=4,
        n_blocks=2,
        d_qkv=16,
        ff_dim=128,
        dropout=0.1,
        attn_dropout=0.1,
        lmax=8,
        force_bound=200.0,
        compensation_gain=1.0,
        lambda_main=1.0,
        lambda_seq=0.0,
        lambda_phys=1.0,
        alpha_conf=0.0,
        ema_beta=0.99,
        use_out_proj=False,
        given_pid=False,
        p=0,
        i=0,
        d=0,
    ):
        super().__init__(given_pid=given_pid, p=p, i=i, d=d)
        self.device = device
        self.dtype = torch.double

        self.seq_len = int(max(1, seq_len))
        self.d_x = 10
        self.d_a = 4
        self.d_y = 3
        self.force_bound = float(force_bound)
        self.compensation_gain = float(compensation_gain)

        self.lambda_main = float(lambda_main)
        self.lambda_seq = float(lambda_seq)
        self.lambda_phys = float(lambda_phys)
        self.alpha_conf = float(alpha_conf)
        self.ema_beta = float(ema_beta)

        self.model = CausalTransformerForceModel(
            d_x=self.d_x,
            d_a=self.d_a,
            d_y=self.d_y,
            seq_len=self.seq_len,
            d_model=int(d_model),
            n_heads=int(n_heads),
            n_blocks=int(n_blocks),
            d_qkv=int(d_qkv),
            ff_dim=int(ff_dim),
            dropout=float(dropout),
            attn_dropout=float(attn_dropout),
            lmax=int(lmax),
            use_out_proj=bool(use_out_proj),
        ).to(self.device).double()

        self.x_mean = np.zeros(self.d_x, dtype=np.float64)
        self.x_std = np.ones(self.d_x, dtype=np.float64)
        self.a_mean = np.zeros(self.d_a, dtype=np.float64)
        self.a_std = np.ones(self.d_a, dtype=np.float64)
        self.y_mean = np.zeros(self.d_y, dtype=np.float64)
        self.y_std = np.ones(self.d_y, dtype=np.float64)

        self.reset_controller()

    def reset_controller(self):
        """Reset PID states and causal-transformer runtime buffers."""
        super().reset_controller()
        self.position_tick_counter = 0
        self.last_motor_speed = np.zeros(self.d_a, dtype=np.float64)
        self.last_f_hat = np.zeros(3, dtype=np.float64)
        self.hist_X = []
        self.hist_A = []
        self.hist_Y = []
        self.force_err_list = []

    def mixer(self, torque_sp, T_sp):
        """Run base mixer and cache latest motor speed."""
        motor_speed = super().mixer(torque_sp, T_sp)
        self.last_motor_speed = motor_speed.copy()
        return motor_speed

    def build_stream_features(self, X, Z, imu=None, pd=None, vd=None, ad=None):
        """Build one-step stream features at current position tick.

        Args:
            X: State vector (13,).
            Z: Current motor-speed state (4,).

        Returns:
            x_t: (10,)
            a_t: (4,)
            y_t: (3,) previous residual prediction used as teacher/autoregressive input.
        """
        u_feat = self.last_motor_speed if self.last_motor_speed is not None else Z
        x_t = np.concatenate((X[7:10], X[3:7], X[10:13])).astype(np.float64, copy=False)
        a_t = np.asarray(u_feat, dtype=np.float64).reshape(-1)
        y_t = self.last_f_hat.astype(np.float64, copy=True)
        return x_t, a_t, y_t

    @staticmethod
    def build_teacher_forcing_stream(y_target_seq):
        """Build shifted Y stream for teacher forcing.

        Args:
            y_target_seq: np.ndarray with shape (N, T, 3).

        Returns:
            y_teacher: np.ndarray with shape (N, T, 3),
                y_teacher[:, 1:] = y_target_seq[:, :-1].
        """
        if y_target_seq.ndim != 3 or y_target_seq.shape[-1] != 3:
            raise ValueError(f"Expected y_target_seq shape (N,T,3), got {y_target_seq.shape}")
        y_teacher = np.zeros_like(y_target_seq, dtype=np.float64)
        if y_target_seq.shape[1] > 1:
            y_teacher[:, 1:, :] = y_target_seq[:, :-1, :]
        return y_teacher

    def _normalize_np(self, data, mean, std):
        std_safe = np.where(std > 1e-9, std, 1.0)
        return (data - mean) / std_safe

    def _normalize_infer_triplet(self, x_seq, a_seq, y_seq):
        x_n = self._normalize_np(x_seq, self.x_mean, self.x_std)
        a_n = self._normalize_np(a_seq, self.a_mean, self.a_std)
        y_n = self._normalize_np(y_seq, self.y_mean, self.y_std)
        return x_n, a_n, y_n

    def predict_force(self):
        """Predict current residual force from buffered stream histories.

        Returns:
            f_hat: np.ndarray with shape (3,).
        """
        if len(self.hist_X) < self.seq_len:
            return np.zeros(3, dtype=np.float64)

        x_seq = np.asarray(self.hist_X[-self.seq_len:], dtype=np.float64)
        a_seq = np.asarray(self.hist_A[-self.seq_len:], dtype=np.float64)
        y_seq = np.asarray(self.hist_Y[-self.seq_len:], dtype=np.float64)
        x_seq, a_seq, y_seq = self._normalize_infer_triplet(x_seq, a_seq, y_seq)

        x_t = torch.from_numpy(x_seq).to(self.device).double().unsqueeze(0)
        a_t = torch.from_numpy(a_seq).to(self.device).double().unsqueeze(0)
        y_t = torch.from_numpy(y_seq).to(self.device).double().unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            out = self.model(x_t, a_t, y_t, return_attn=False)
            f_hat = out['force_pred'].cpu().numpy().reshape(-1)

        if f_hat.shape[0] != 3 or (not np.all(np.isfinite(f_hat))):
            return np.zeros(3, dtype=np.float64)
        return np.clip(f_hat, -self.force_bound, self.force_bound)

    def get_Fr(self, X, Z, imu, pd, vd, ad, meta_adapt_trigger, wind_gt):
        """Return compensated force command at position-loop update."""
        fr_nominal, fr_dot = super().get_Fr(
            X, Z=Z, imu=imu, pd=pd, vd=vd, ad=ad,
            meta_adapt_trigger=meta_adapt_trigger, wind_gt=wind_gt,
        )

        self.position_tick_counter += 1
        x_t, a_t, y_t = self.build_stream_features(X, Z=Z, imu=imu, pd=pd, vd=vd, ad=ad)

        self.hist_X.append(x_t.copy())
        self.hist_A.append(a_t.copy())
        self.hist_Y.append(y_t.copy())

        if len(self.hist_X) > self.seq_len:
            self.hist_X = self.hist_X[-self.seq_len:]
            self.hist_A = self.hist_A[-self.seq_len:]
            self.hist_Y = self.hist_Y[-self.seq_len:]

        if len(self.hist_X) >= self.seq_len:
            self.last_f_hat = self.predict_force()
        else:
            self.last_f_hat = np.zeros(3, dtype=np.float64)

        if wind_gt is not None and np.all(np.isfinite(wind_gt)):
            self.force_err_list.append(float(np.linalg.norm(self.last_f_hat - wind_gt)))

        fr_comp = fr_nominal - self.compensation_gain * self.last_f_hat
        return fr_comp, fr_dot

    def fit_force_model(
        self,
        train_x,
        train_a,
        train_y_teacher,
        train_target_seq,
        val_x=None,
        val_a=None,
        val_y_teacher=None,
        val_target_seq=None,
        epochs=100,
        batch_size=64,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        lambda_main=None,
        lambda_seq=None,
        lambda_phys=None,
    ):
        """Train causal-transformer force model.

        Args:
            train_x: (N, T, d_x)
            train_a: (N, T, d_a)
            train_y_teacher: (N, T, d_y)
            train_target_seq: (N, T, 3)
        """
        if len(train_x) == 0:
            raise RuntimeError("Empty causal-transformer training set.")
        if train_x.ndim != 3 or train_x.shape[1] != self.seq_len or train_x.shape[2] != self.d_x:
            raise ValueError(f"Expected train_x shape (N,{self.seq_len},{self.d_x}), got {train_x.shape}")
        if train_a.ndim != 3 or train_a.shape[1] != self.seq_len or train_a.shape[2] != self.d_a:
            raise ValueError(f"Expected train_a shape (N,{self.seq_len},{self.d_a}), got {train_a.shape}")
        if train_y_teacher.ndim != 3 or train_y_teacher.shape[1] != self.seq_len or train_y_teacher.shape[2] != self.d_y:
            raise ValueError(f"Expected train_y_teacher shape (N,{self.seq_len},{self.d_y}), got {train_y_teacher.shape}")
        if train_target_seq.ndim != 3 or train_target_seq.shape[1] != self.seq_len or train_target_seq.shape[2] != 3:
            raise ValueError(f"Expected train_target_seq shape (N,{self.seq_len},3), got {train_target_seq.shape}")

        lam_main = self.lambda_main if lambda_main is None else float(lambda_main)
        lam_seq = self.lambda_seq if lambda_seq is None else float(lambda_seq)
        lam_phys = self.lambda_phys if lambda_phys is None else float(lambda_phys)

        self.x_mean = np.mean(train_x.reshape(-1, self.d_x), axis=0).astype(np.float64)
        self.x_std = np.std(train_x.reshape(-1, self.d_x), axis=0).astype(np.float64)
        self.x_std = np.where(self.x_std > 1e-9, self.x_std, 1.0)

        self.a_mean = np.mean(train_a.reshape(-1, self.d_a), axis=0).astype(np.float64)
        self.a_std = np.std(train_a.reshape(-1, self.d_a), axis=0).astype(np.float64)
        self.a_std = np.where(self.a_std > 1e-9, self.a_std, 1.0)

        self.y_mean = np.mean(train_y_teacher.reshape(-1, self.d_y), axis=0).astype(np.float64)
        self.y_std = np.std(train_y_teacher.reshape(-1, self.d_y), axis=0).astype(np.float64)
        self.y_std = np.where(self.y_std > 1e-9, self.y_std, 1.0)

        train_x_n = self._normalize_np(train_x, self.x_mean, self.x_std)
        train_a_n = self._normalize_np(train_a, self.a_mean, self.a_std)
        train_y_n = self._normalize_np(train_y_teacher, self.y_mean, self.y_std)

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        mse = nn.MSELoss()
        n = len(train_x_n)

        for epoch in range(int(epochs)):
            perm = np.random.permutation(n)
            total_loss = 0.0
            total_main = 0.0
            total_seq = 0.0
            total_phys = 0.0
            total_count = 0

            for st in range(0, n, int(batch_size)):
                idx = perm[st:st + int(batch_size)]
                x = torch.from_numpy(train_x_n[idx]).to(self.device).double()
                a = torch.from_numpy(train_a_n[idx]).to(self.device).double()
                y_in = torch.from_numpy(train_y_n[idx]).to(self.device).double()
                y_seq = torch.from_numpy(train_target_seq[idx]).to(self.device).double()
                y_cur = y_seq[:, -1, :]

                out = self.model(x, a, y_in, return_attn=False)
                y_hat_seq = out['force_seq_pred']
                y_hat_cur = out['force_pred']

                l_main = mse(y_hat_cur, y_cur)
                l_seq = mse(y_hat_seq, y_seq)
                # Reuse existing force-model style PI regularizer: keep residual sequence bounded.
                l_phys = mse(y_hat_seq, torch.zeros_like(y_hat_seq))

                loss = float(lam_main) * l_main + float(lam_seq) * l_seq + float(lam_phys) * l_phys
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss encountered in Causal-Transformer training.")

                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None and grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                optimizer.step()

                bsz = len(idx)
                total_loss += float(loss.item()) * bsz
                total_main += float(l_main.item()) * bsz
                total_seq += float(l_seq.item()) * bsz
                total_phys += float(l_phys.item()) * bsz
                total_count += bsz

            msg = (
                f"[CausalTransformer][Epoch {epoch+1}/{epochs}] "
                f"loss={total_loss/max(total_count,1):.6f} "
                f"main={total_main/max(total_count,1):.6f} "
                f"seq={total_seq/max(total_count,1):.6f} "
                f"phys={total_phys/max(total_count,1):.6f}"
            )
            if (
                val_x is not None and val_a is not None and val_y_teacher is not None and
                val_target_seq is not None and len(val_x) > 0
            ):
                val_metrics = self.eval_force_model(val_x, val_a, val_y_teacher, val_target_seq[:, -1, :])
                msg += f" val_rmse={val_metrics['rmse']:.6f} val_mae={val_metrics['mae']:.6f}"
            print(msg)

        self.model.eval()

    def eval_force_model(self, x_hist, a_hist, y_hist, target_cur):
        """Evaluate current-step force prediction metrics.

        Args:
            x_hist: (N, T, d_x)
            a_hist: (N, T, d_a)
            y_hist: (N, T, d_y)
            target_cur: (N, 3)
        """
        if len(x_hist) == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'axis_rmse': np.array([np.nan, np.nan, np.nan])}

        x_n = self._normalize_np(x_hist, self.x_mean, self.x_std)
        a_n = self._normalize_np(a_hist, self.a_mean, self.a_std)
        y_n = self._normalize_np(y_hist, self.y_mean, self.y_std)

        x_t = torch.from_numpy(x_n).to(self.device).double()
        a_t = torch.from_numpy(a_n).to(self.device).double()
        y_t = torch.from_numpy(y_n).to(self.device).double()

        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_t, a_t, y_t)['force_pred'].cpu().numpy()

        err = pred - target_cur
        rmse = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
        mae = float(np.mean(np.linalg.norm(err, axis=1)))
        axis_rmse = np.sqrt(np.mean(err ** 2, axis=0))
        return {'rmse': rmse, 'mae': mae, 'axis_rmse': axis_rmse}

    def save(self, path):
        """Save model and normalizer states."""
        state = {
            'causal_transformer_model': self.model.state_dict(),
            'seq_len': self.seq_len,
            'd_x': self.d_x,
            'd_a': self.d_a,
            'd_y': self.d_y,
            'x_mean': self.x_mean,
            'x_std': self.x_std,
            'a_mean': self.a_mean,
            'a_std': self.a_std,
            'y_mean': self.y_mean,
            'y_std': self.y_std,
        }
        torch.save(state, path)

    def load(self, path, map_location=None):
        """Load model and optional normalizer states."""
        ckpt = torch.load(path, map_location=map_location)
        if isinstance(ckpt, dict) and 'causal_transformer_model' in ckpt:
            state_dict = ckpt['causal_transformer_model']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()

        if isinstance(ckpt, dict):
            if 'x_mean' in ckpt:
                self.x_mean = np.asarray(ckpt['x_mean'], dtype=np.float64)
            if 'x_std' in ckpt:
                std = np.asarray(ckpt['x_std'], dtype=np.float64)
                self.x_std = np.where(std > 1e-9, std, 1.0)
            if 'a_mean' in ckpt:
                self.a_mean = np.asarray(ckpt['a_mean'], dtype=np.float64)
            if 'a_std' in ckpt:
                std = np.asarray(ckpt['a_std'], dtype=np.float64)
                self.a_std = np.where(std > 1e-9, std, 1.0)
            if 'y_mean' in ckpt:
                self.y_mean = np.asarray(ckpt['y_mean'], dtype=np.float64)
            if 'y_std' in ckpt:
                std = np.asarray(ckpt['y_std'], dtype=np.float64)
                self.y_std = np.where(std > 1e-9, std, 1.0)


PowerFormerController = PowerformerController
