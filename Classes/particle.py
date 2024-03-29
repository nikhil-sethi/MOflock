from Utils.vector_algebra import unit_vector, absheading_from_vec
from Utils.controls import norm
import numpy as np
import Config.defaults as df


class Particle:
    '''
    A particle class for animation and timed updates
    Params
    df: a python file with appropriate variables
        vmax= maximum velocity
        amax = maximum acceleration
        maxForce
    position : initial position, default
    '''

    def __init__(self):
        self.pos = np.zeros(2, dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.prevel = np.zeros(2, dtype=float)
        self.acc = np.zeros(2, dtype=float)
        self.gps_vel = np.zeros(2, dtype=float)
        self.v_d = np.zeros(2, dtype=float)

    def update(self, step, frame):
        self.acc = self.v_d - self.vel - self.gps_vel  # self.memory[-1][1]
        self.vel += unit_vector(self.acc)[0] * min(norm(self.acc), df.amax) * step
        self.pos += self.vel * step
        # print(self.vel,self.acc)
        # self.vel = unit_vector(self.vel)[0] * min(norm(self.vel), df.vmax)

        # self.prevel = self.vel
        self.acc = np.zeros(2)

    def update2(self, step, frame):
        pre_acc = self.acc
        self.acc  = self.v_d - self.vel - self.gps_vel  # self.memory[-1][1]
        
        int_acc = (self.acc + pre_acc)/2
        
        pre_vel = self.vel
        self.vel += int_acc * step
        # unit_vector(self.acc)[0] * min(norm(self.acc), df.amax) * step
    
        int_vel = (self.vel + pre_vel)/2
        self.pos += int_vel * step
        
    # control methods
    def push(self, force, maxForce: float):
        """apply a force in give direction on the particle"""

    def steer(self, steer, maxForce):
        steer = df.vmax * unit_vector(steer)[0]  # normalise
        steer -= self.vel  # this is the actual steering vector. The above one is the target direction(new vector)
        steer *= maxForce  # sensitivity
        return steer

    # setter methods
    def sch(self, heading: int):
        heading = np.deg2rad(heading)
        self.vel = df.vmax * np.array([np.cos(heading), np.sin(heading)])

    def scv(self, vx, vy):
        self.vel = np.array([vx, vy], dtype=float)

    def scp(self, x, y):
        self.pos = np.array([x, y], dtype=float)

    def sca(self, x, y=None):
        if y is None:
            y = x
        self.acc = np.array([x, y], dtype=float)

    # getters
    def gch(self, mode='cartesian'):
        """return absolute heading"""
        return absheading_from_vec(self.vel, mode)

    def gcp(self):
        return self.pos

    def gcv(self):
        return self.vel


def set_all(seq, method, *args, **kwargs):
    for obj in seq:
        getattr(obj, method)(*args, **kwargs)
