from Methods.vector_algebra import unit_vector, absheading_from_vec
from Methods.controls import norm
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
    def __init__(self, position=np.zeros(2), velocity=np.zeros(2), acceleration=np.zeros(2), animated=True):

        self.pos = np.array(position, dtype=float)
        self.vel = np.array(velocity, dtype=float)
        self.acc = acceleration
        self.animated = animated

    def update(self, step, frame):
        self.vel += step*unit_vector(self.acc)[0] * min(norm(self.acc), df.amax)
        #print(self.vel,self.acc)
        self.vel = unit_vector(self.vel)[0] * min(norm(self.vel), df.vmax)
        self.pos += self.vel * step
        self.acc = np.zeros(2)

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
            y=x
        self.acc = np.array([x, y], dtype=float)

    #getters
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

