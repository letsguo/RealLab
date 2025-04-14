import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dynamics_cfg import SimpleCarDynamicsCfg

from ..maps import BEVMap

class SimpleCarDynamics(torch.nn.Module):
    """
    Class for Dynamics modeling
    """
    def __init__(
        self,
        dynamics_cfg: 'SimpleCarDynamicsCfg',
        num_envs: int,
        bevmap: BEVMap,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):
        super(SimpleCarDynamics, self).__init__()
        self.dtype = dtype
        self.d = device
        self.bevmap = bevmap
        self.cfg = dynamics_cfg

        self.wheelbase = self.cfg.wheelbase  # the distance between front and back
        self.throttle_to_wheelspeed = self.cfg.throttle_to_wheelspeed  # relation for position of throttle pedal and resulting speed
        self.steering_max = self.cfg.steering_max  # max steering
        self.dt = self.cfg.dt
        self.curvature_max = self.steering_max / self.wheelbase  # shouldn't it be tangent of steering angle (should be tangent)
        self.steering_max = self.cfg.steering_max  # max steering
        self.dt = self.cfg.dt

        #look for a1.py or Ground control configs for defining map resolution
        self.num_envs = num_envs
        self.GRAVITY = -9.81

        self.x_dim = self.cfg.x_dim
        self.concatenate_feats = self.cfg.concatenate_feats

        if self.concatenate_feats:
            self.state_dim = self.x_dim + self.cfg.feat_dim
        else:
            self.state_dim = self.x_dim

    ## remember, this function is called only once! If you have a single-step dynamics
    # function, you will need to roll it out inside this function.
    def forward(self, x0, controls):
        '''
        :param: state: torch.Tensor of shape (num_envs, state_dim)
        :param: controls: torch.Tensor of shape (num_envs, ..., nu)
        '''
        assert x0.shape[-1] == self.cfg.x_dim, f"Expected state dimension {self.cfg.x_dim}, got {x0.shape[-1]}"

        xs = x0.view((x0.shape[0], *(1,)*(controls.ndim-2), x0.shape[1]))
        xs = xs.repeat(1, *controls.shape[1:-1], 1)

        x = xs[..., 0]
        y = xs[..., 1]
        z = xs[..., 2]
        roll = xs[..., 3]
        pitch = xs[..., 4]
        yaw = xs[..., 5]
        vx = xs[..., 6]
        vy = xs[..., 7]
        vz = xs[..., 8]
        wx = xs[..., 9]
        wy = xs[..., 10]
        wz = xs[..., 11]
        throttle = controls[..., 0]
        steer = controls[..., 1]
        xs[..., 12:] = controls

        K = torch.tan(steer * self.steering_max)/self.wheelbase  # this is just a placeholder for curvature since steering correlates to curvature

        # CONTROL SAMPLING LAW: directly control in state space
        vx[...] = throttle * self.throttle_to_wheelspeed

        dS = vx * self.dt

        wz[...] = vx * K

        yaw[...] += self.dt * torch.cumsum(wz, dim=2)  # this is what the yaw will become

        cy = torch.cos(yaw)
        sy = torch.sin(yaw)

        ## Compute position
        x[...] += torch.cumsum(dS * cy, dim=-1)
        y[...] += torch.cumsum(dS * sy, dim=-1)
        map_vals = self.bevmap.get_map_xy(x, y)
        z[...] = map_vals[..., 0]

        ## Compute attitude angles
        normal = map_vals[..., 1:]
        heading = torch.stack([cy, sy, torch.zeros_like(yaw)], dim=-1) ## heading is a unit vector --ergo, all cross products will be unit vectors and don't need normalization

        # Calculate the cross product of the heading and normal vectors to get the vector perpendicular to both
        left = torch.cross(normal, heading, dim=-1)
        # Calculate the cross product of the right and normal vectors to get the vector perpendicular to both and facing upwards
        forward = torch.cross(left, normal, dim =-1)
        # Calculate the roll angle (rotation around the forward axis)
        roll[...] = torch.asin(left[...,2])
        # Calculate the pitch angle (rotation around the right axis)
        pitch[...] = -torch.asin(forward[...,2])

        wx[..., 1:] = torch.diff(roll, dim=-1)/self.dt
        wy[..., 1:] = torch.diff(pitch, dim=-1)/self.dt

        # No-slip assumption
        vy[...] = torch.zeros_like(vx)
        vz[...] = torch.zeros_like(vx)

        if self.concatenate_feats:
            return torch.cat((xs, map_vals), dim=-1)

        return xs

class SimpleCarDynamicsNoAction(SimpleCarDynamics):
    """
    Class for Car Dynamics modeling
    """

    def forward(self, x0, controls):
        '''
        :param: state: torch.Tensor of shape (num_envs, state_dim)
        :param: controls: torch.Tensor of shape (num_envs, ..., T, nu)
        '''
        assert x0.shape[-1] == self.cfg.x_dim, f"Expected state dimension {self.cfg.x_dim}, got {x0.shape[-1]}"

        xs = x0.view((x0.shape[0], *(1,)*(controls.ndim-2), x0.shape[1])) # (num_envs, 1, x_dim)
        xs = xs.repeat(1, *controls.shape[1:-1], 1)

        x = xs[..., 0]
        y = xs[..., 1]
        z = xs[..., 2]
        roll = xs[..., 3]
        pitch = xs[..., 4]
        yaw = xs[..., 5]
        vx = xs[..., 6]
        vy = xs[..., 7]
        vz = xs[..., 8]
        wx = xs[..., 9]
        wy = xs[..., 10]
        wz = xs[..., 11]
        throttle = controls[..., 0]
        steer = controls[..., 1]

        K = torch.tan(steer * self.steering_max)/self.wheelbase  # this is just a placeholder for curvature since steering correlates to curvature

        # CONTROL SAMPLING LAW: directly control in state space
        vx[...] = throttle * self.throttle_to_wheelspeed

        dS = vx * self.dt

        wz[...] = vx * K

        yaw[...] += self.dt * torch.cumsum(wz, dim=-1)  # this is what the yaw will become

        cy = torch.cos(yaw)
        sy = torch.sin(yaw)

        ## Compute position
        x[...] += torch.cumsum(dS * cy, dim=-1)
        y[...] += torch.cumsum(dS * sy, dim=-1)
        map_vals = self.bevmap.get_map_xy(x, y)
        z[...] = map_vals[..., 0]

        ## Compute attitude angles
        normal = map_vals[..., 1:]
        heading = torch.stack([cy, sy, torch.zeros_like(yaw)], dim=-1) ## heading is a unit vector --ergo, all cross products will be unit vectors and don't need normalization

        # Calculate the cross product of the heading and normal vectors to get the vector perpendicular to both
        left = torch.cross(normal, heading, dim=-1)
        # Calculate the cross product of the right and normal vectors to get the vector perpendicular to both and facing upwards
        forward = torch.cross(left, normal, dim =-1)
        # Calculate the roll angle (rotation around the forward axis)
        roll[...] = torch.asin(left[...,2])
        # Calculate the pitch angle (rotation around the right axis)
        pitch[...] = -torch.asin(forward[...,2])

        wx[..., 1:] = torch.diff(roll, dim=-1)/self.dt
        wy[..., 1:] = torch.diff(pitch, dim=-1)/self.dt

        # No-slip assumption
        vy[...] = torch.zeros_like(vx)
        vz[...] = torch.zeros_like(vx)

        if self.concatenate_feats:
            return torch.cat((xs, map_vals), dim=-1)

        return xs

