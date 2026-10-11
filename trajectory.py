import numpy as np
import torch
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class hover():
    def __init__(self, pd=np.zeros(3)):
        self.pd = pd
        self.name = 'hover'
    def __call__(self, t):
        vd = ad = np.zeros(3)
        return self.pd, vd, ad

class fig8():
    def __init__(self, T=15., dir1=(2., 2., 0.), dir2=(0., 0., 1.)):
        self.w = 2*np.pi/T
        self.dir1 = np.array(dir1)
        self.dir2 = np.array(dir2)
        self.name = 'fig8'

    def __call__(self, t):
        pd =                np.sin(self.w*t) * self.dir1 +                 np.sin(2*self.w*t) * self.dir2
        vd =   self.w     * np.cos(self.w*t) * self.dir1 +  2*self.w     * np.cos(2*self.w*t) * self.dir2
        ad = -(self.w)**2 * np.sin(self.w*t) * self.dir1 - (2*self.w)**2 * np.sin(2*self.w*t) * self.dir2
        return pd, vd, ad

class sin_forward():
    def __init__(self, T=6., A=2, Vy=0.2, Vz=0.5):
        self.w = np.pi*2/T
        self.A = A
        self.Vy = Vy
        self.Vz = Vz
        self.name = 'sin'
    
    def __call__(self,t):
        pd = np.array((self.A*np.sin(self.w*t), self.Vy*t, self.Vz*t))
        vd = np.array((self.w * self.A * np.cos(self.w*t), self.Vy, self.Vz))
        ad = np.array((-self.w**2 * self.A * np.sin(self.w*t), 0, 0))
        return pd, vd, ad

class spiral_up():
    def __init__(self, T=10., R=1., Vz=1, Vr=0.3):
        self.w = np.pi*2/T
        self.R = R
        self.Vr = Vr
        self.Vz = Vz
        self.name = 'spiral'
    
    def __call__(self, t):
        R = self.R + self.Vr * t
        pd = np.array((R*(np.cos(self.w*t)-1), R*np.sin(self.w*t), self.Vz*t))
        vd = np.array((self.Vr*(np.cos(self.w*t)-1)-R*self.w*np.sin(self.w*t), \
                        self.Vr*np.sin(self.w*t)+R*self.w*np.cos(self.w*t), self.Vz))
        ad = np.array((-2*self.Vr*self.w*np.sin(self.w*t)-R*(self.w**2)*np.cos(self.w*t), \
                        2*self.Vr*self.w*np.cos(self.w*t)-R*(self.w**2)*np.sin(self.w*t), 0))
        return pd, vd, ad

class ZigZag():
    def __init__(self, start_pos=(0.,0.,0.), speed_X=0.25, amplitude_Y=1.0, period_Y_time=10.0):
        self.start_pos = np.array(start_pos)
        self.V_x = speed_X
        self.A_y = amplitude_Y
        self.T_y = period_Y_time
        self.name = 'zigzag'

    def _triangle_wave_normalized(self, phi_normalized_period):
        t_mod_period = phi_normalized_period % 1.0
        if t_mod_period < 0.25:
            return 4 * t_mod_period
        elif t_mod_period < 0.75:
            return 1.0 - 4 * (t_mod_period - 0.25)
        else:
            return -1.0 + 4 * (t_mod_period - 0.75)
            
    def _triangle_wave_derivative_normalized(self, phi_normalized_period):
        t_mod_period = phi_normalized_period % 1.0
        if t_mod_period < 0.25:
            return 4.0
        elif t_mod_period < 0.75:
            return -4.0
        else:
            return 4.0

    def __call__(self, t):
        if t < 0: t = 0
        pd_x = self.start_pos[0] + self.V_x * t

        time_normalized_for_period = t / self.T_y
        triangle_val = self._triangle_wave_normalized(time_normalized_for_period)

        pd_y = self.start_pos[1] + self.A_y * triangle_val
        pd_z = self.start_pos[2]
        pd = np.array([pd_x, pd_y, pd_z])

        vd_x = self.V_x
        
        # d(t/T_y)/dt = 1/T_y
        # So vd_y = A_y * triangle_wave_derivative_normalized(t/T_y) * (1/T_y)
        triangle_deriv_val = self._triangle_wave_derivative_normalized(time_normalized_for_period)
        vd_y = self.A_y * triangle_deriv_val / self.T_y
        
        vd_z = 0.0
        vd = np.array([vd_x, vd_y, vd_z])
        ad = np.zeros(3)
        
        return pd, vd, ad

class LinePathXY:
    """
    A simple 2D line path planner in the XY plane.
    Provides pd_xy, vd_xy, ad_xy for the ground track.
    """
    def __init__(self, start_xy=(0.,0.), end_xy=(10.,10.), total_time=20.0):
        self.start_xy = np.array(start_xy)
        self.end_xy = np.array(end_xy)
        self.T_total = float(total_time)
        self.name = 'line_xy_base_path'
        if self.T_total <= 0:
            raise ValueError("Total time must be positive.")
        self.diff_xy = self.end_xy - self.start_xy
        self.vel_xy_const = self.diff_xy / self.T_total

    def __call__(self, t):
        if t < 0:
            u = 0.0
            current_vel_xy = np.zeros(2)
        elif t > self.T_total:
            u = 1.0
            current_vel_xy = np.zeros(2)
        else:
            u = t / self.T_total
            current_vel_xy = self.vel_xy_const
        
        pd_xy = self.start_xy + self.diff_xy * u
        pd = np.array([pd_xy[0], pd_xy[1], 0.0])
        vd = np.array([current_vel_xy[0], current_vel_xy[1], 0.0])
        ad = np.zeros(3)
        
        return pd, vd, ad

# --- Terrain Model: Sum of Random 2D Gaussian Functions ---
class RandomGaussianTerrain:
    """
    Simulates terrain as a sum of multiple 2D Gaussian functions 
    with randomized parameters. Provides elevation, gradient, and Hessian.
    """
    def __init__(self, num_gaussians=3, amplitude_range=(8, 10), 
                 sigma_range=(1.0, 3.0), center_range_x=(0, 10), 
                 center_range_y=(0, 10), rotation_enabled=True, seed=1):
        self.name = 'random_gaussian_terrain'
        self.gaussians_params = []
        setup_seed(seed)

        for _ in range(num_gaussians):
            amp = np.random.uniform(amplitude_range[0], amplitude_range[1]) * 0.1
            mux = np.random.uniform(center_range_x[0], center_range_x[1])
            # muy = np.random.uniform(center_range_y[0], center_range_y[1])
            muy = mux
            sigx = np.random.uniform(sigma_range[0], sigma_range[1])
            # sigy = np.random.uniform(sigma_range[0], sigma_range[1])
            sigy = sigx
            if rotation_enabled:
                theta = np.random.uniform(0, np.pi) # Rotation angle for this Gaussian
            else:
                theta = 0.0
            self.gaussians_params.append({
                'A': amp, 'mux': mux, 'muy': muy, 
                'sigx': sigx, 'sigy': sigy, 'theta': theta,
                'cos_th': np.cos(theta), 'sin_th': np.sin(theta)
            })

    def _single_gaussian_eval(self, x, y, params):
        A, mux, muy, sigx, sigy = params['A'], params['mux'], params['muy'], params['sigx'], params['sigy']
        cos_th, sin_th = params['cos_th'], params['sin_th']

        x_prime = (x - mux) * cos_th + (y - muy) * sin_th
        y_prime = -(x - mux) * sin_th + (y - muy) * cos_th

        u_val = (x_prime**2) / (2 * sigx**2) + (y_prime**2) / (2 * sigy**2)
        if u_val > 100:
             G_val = 0.0
             grad_G = np.zeros(2)
             H_G_xx, H_G_yy, H_G_xy = 0.0,0.0,0.0
        else:
            G_val = A * np.exp(-u_val)

            # Gradient terms (see derivation in thought process)
            term_ux = x_prime / sigx**2
            term_uy = y_prime / sigy**2

            g_common_x = term_ux * cos_th - term_uy * sin_th
            g_common_y = term_ux * sin_th + term_uy * cos_th
            
            grad_G_x = -G_val * g_common_x
            grad_G_y = -G_val * g_common_y
            grad_G = np.array([grad_G_x, grad_G_y])

            # Hessian terms (see derivation in thought process)
            # d/dx (term_ux*cos_th - term_uy*sin_th)
            d_g_common_x_dx = (cos_th**2 / sigx**2) + (sin_th**2 / sigy**2)
            # d/dy (term_ux*sin_th + term_uy*cos_th)
            d_g_common_y_dy = (sin_th**2 / sigx**2) + (cos_th**2 / sigy**2)
            # d/dy (term_ux*cos_th - term_uy*sin_th) or d/dx (term_ux*sin_th + term_uy*cos_th)
            d_g_common_x_dy = cos_th * sin_th * (1/sigx**2 - 1/sigy**2) # = d_g_common_y_dx

            H_G_xx = G_val * (g_common_x**2 - d_g_common_x_dx)
            H_G_yy = G_val * (g_common_y**2 - d_g_common_y_dy)
            H_G_xy = G_val * (g_common_x * g_common_y - d_g_common_x_dy)

        return G_val, grad_G, np.array([H_G_xx, H_G_yy, H_G_xy])

    def get_elevation(self, x, y):
        total_elevation = 0.0
        for params in self.gaussians_params:
            G_val, _, _ = self._single_gaussian_eval(x, y, params)
            total_elevation += G_val
        return total_elevation

    def get_gradient(self, x, y): # Returns [dh/dx, dh/dy]
        total_gradient = np.zeros(2)
        for params in self.gaussians_params:
            _, grad_G, _ = self._single_gaussian_eval(x, y, params)
            total_gradient += grad_G
        return total_gradient

    def get_hessian_components(self, x, y): # Returns [d2h/dx2, d2h/dy2, d2h/dxdy]
        total_hess_xx, total_hess_yy, total_hess_xy = 0.0, 0.0, 0.0
        for params in self.gaussians_params:
            _, _, H_G_comps = self._single_gaussian_eval(x, y, params)
            total_hess_xx += H_G_comps[0]
            total_hess_yy += H_G_comps[1]
            total_hess_xy += H_G_comps[2]
        return total_hess_xx, total_hess_yy, total_hess_xy

    def __call__(self, x, y): # Convenience to make it callable for elevation
        return self.get_elevation(x,y)


class TerrainFollowingPath():
    def __init__(self, base_path_planner, terrain_model, agl_offset=5.0, 
                 x_offset=0., y_offset=0., z_offset=0.):
        setup_seed(0)
        self.base_path_planner = base_path_planner
        self.terrain_model = terrain_model
        self.AGL_offset = agl_offset
        self.offsets = np.array([x_offset, y_offset, z_offset])
        self.name = 'terrain'

    def __call__(self, t):
        # Get 2D path components from the base planner (position, velocity, acceleration)
        pd_base, vd_base, ad_base = self.base_path_planner(t)
        
        xp_t, yp_t = pd_base[0], pd_base[1]
        vxp_t, vyp_t = vd_base[0], vd_base[1]
        axp_t, ayp_t = ad_base[0], ad_base[1]

        terrain_z = self.terrain_model.get_elevation(xp_t, yp_t)
        grad_h_x, grad_h_y = self.terrain_model.get_gradient(xp_t, yp_t) # [dh/dx, dh/dy]
        hess_h_xx, hess_h_yy, hess_h_xy = self.terrain_model.get_hessian_components(xp_t, yp_t) # [d2h/dx2, d2h/dy2, d2h/dxdy]

        pd = np.array([xp_t, yp_t, terrain_z + self.AGL_offset]) + self.offsets

        vd_x = vxp_t
        vd_y = vyp_t
        vd_z = grad_h_x * vxp_t + grad_h_y * vyp_t
        vd = np.array([vd_x, vd_y, vd_z])

        ad_x = axp_t
        ad_y = ayp_t
        # az = d/dt (vz)
        # vz = Gx * Vx + Gy * Vy  (where Gx=grad_h_x, Vx=vxp_t, etc.)
        # az = (dGx/dt)*Vx + Gx*(dVx/dt) + (dGy/dt)*Vy + Gy*(dVy/dt)
        # dGx/dt = (d2h/dx2)*Vx + (d2h/dxdy)*Vy = Hxx*Vx + Hxy*Vy
        # dGy/dt = (d2h/dydx)*Vx + (d2h/dy2)*Vy = Hxy*Vx + Hyy*Vy
        # dVx/dt = axp_t, dVy/dt = ayp_t
        ad_z = (hess_h_xx * vxp_t + hess_h_xy * vyp_t) * vxp_t + grad_h_x * axp_t + \
               (hess_h_xy * vxp_t + hess_h_yy * vyp_t) * vyp_t + grad_h_y * ayp_t
        # Simplified: ad_z = hess_h_xx * vxp_t**2 + hess_h_yy * vyp_t**2 + 2 * hess_h_xy * vxp_t * vyp_t + \
        #                   grad_h_x * axp_t + grad_h_y * ayp_t
        ad = np.array([ad_x, ad_y, ad_z])
        
        return pd, vd, ad