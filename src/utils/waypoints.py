import numpy as np
import torch


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

    def generate_waypoints(self, track_radius = 1, track_straight_dist = 1.5 , num_waypoints = 10):
        self.track_radius = track_radius
        self.track_straight_dist = track_straight_dist
        dist_track = 2 * np.pi * track_radius + 4 * track_straight_dist
        dist_between_waypoints = dist_track / num_waypoints
        waypoints = []
        for i in range(num_waypoints):
            waypoint, _ = self.make_waypoint(dist_between_waypoints*i)
            waypoints.append(waypoint)
        self.waypoints = torch.tensor(waypoints, dtype=torch.float32)

    def make_waypoint(self, dist):
        track_radius = self.track_radius
        track_straight_dist = self.track_straight_dist
        if dist < 2*track_straight_dist:
            return [track_radius, dist-track_straight_dist, 0], [0,0,90]
        elif dist < 2*track_straight_dist + np.pi*track_radius:
            angle = (dist - 2*track_straight_dist) / track_radius
            return [track_radius*np.cos(angle), track_straight_dist + track_radius*np.sin(angle), 0], [0,0,90 + angle*180/np.pi]
        elif dist < 4*track_straight_dist + np.pi*track_radius:
            remaining = dist - 2*track_straight_dist - np.pi*track_radius
            return [-track_radius, track_straight_dist - remaining, 0], [0,0,270]
        else:
            angle = (dist - 4*track_straight_dist - np.pi*track_radius) / track_radius
            return [-track_radius*np.cos(angle), -track_straight_dist - track_radius*np.sin(angle), 0], [0,0,270 + angle*180/np.pi]