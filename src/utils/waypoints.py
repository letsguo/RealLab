import numpy as np
import torch

CORNER_IN_RADIUS = 0.5
CORNER_OUT_RADIUS = CORNER_IN_RADIUS + 1
STRAIGHT = 1.5

class Waypoints:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Waypoints, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            # Initialize your waypoints here
            self.waypoints = None
            # ...additional initialization code...

    def generate_waypoints(self, track_radius=(CORNER_IN_RADIUS+CORNER_OUT_RADIUS)/2.0, track_straight_dist=STRAIGHT, num_waypoints=10):
        dist_track = 2 * np.pi * track_radius + 4 * track_straight_dist
        dist_between_waypoints = dist_track / num_waypoints
        waypoints = []
        for i in range(num_waypoints):
            waypoint = self.make_waypoint(dist_between_waypoints*i, track_radius, track_straight_dist)
            waypoints.append(waypoint)
        self.waypoints = torch.tensor(waypoints, dtype=torch.float32)

    def make_waypoint(self, dist: int, track_radius: int, track_straight_dist: int) -> list[float]:
        if dist < 2*track_straight_dist:
            return [dist-track_straight_dist, -track_radius, 0]
        elif dist < 2*track_straight_dist + np.pi*track_radius:
            angle = (dist - 2*track_straight_dist) / track_radius
            return [track_straight_dist + track_radius*np.sin(angle), -track_radius*np.cos(angle), 0]
        elif dist < 4*track_straight_dist + np.pi*track_radius:
            remaining = dist - 2*track_straight_dist - np.pi*track_radius
            return [track_straight_dist - remaining, track_radius, 0]
        else:
            angle = (dist - 4*track_straight_dist - np.pi*track_radius) / track_radius
            return [-track_straight_dist - track_radius*np.sin(angle), track_radius*np.cos(angle), 0]