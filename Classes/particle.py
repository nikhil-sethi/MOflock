from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from Methods.vector_algebra import unit_vector, absheading_from_vec
from Methods.controls import  update_corr
import numpy as np


class Particle:
    '''
    A particle class for animation and timed updates
    Params
    conf: a python file with appropriate variables
        vmax= maximum velocity
        amax = maximum acceleration
        maxForce
    position : initial position, default
    '''
    def __init__(self, position=np.zeros(2), velocity=np.zeros(2), max_vel=0.5, acceleration=np.zeros(2), max_acc=0.2, animated=True):

        self.pos = np.array(position, dtype=float)
        self.vel = np.array(velocity, dtype=float)
        self.acc = acceleration
        self.vmax = max_vel
        self.amax = max_acc
        self.animated = animated

    def update(self, interval):
        step = (interval/1000)*(1+update_corr(interval)) #correction term for cpu animation lag
        self.vel += self.acc * step
        #print(self.vel,self.acc)
        self.vel = self.vmax * unit_vector(self.vel)
        self.pos += self.vel * step
        self.acc = np.zeros(2)

    # control methods
    def push(self, force, maxForce: float):
        """apply a force in give direction on the particle"""

    def steer(self, steer, maxForce):
        steer = self.vmax * unit_vector(steer)  # normalise
        steer -= self.vel  # this is the actual steering vector. The above one is the target direction(new vector)
        steer *= maxForce  # sensitivity
        return steer

    # setter methods
    def sch(self, heading: int):
        heading = np.deg2rad(heading)
        self.vel = self.vmax*np.array([np.cos(heading), np.sin(heading)])

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
