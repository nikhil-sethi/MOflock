import numpy as np
from Methods.vector_algebra import norm
from itertools import combinations as pairs
import math

def update_corr(x):
    """correction equation for the t_update update. weight and bias will depend on CPU"""
    return math.exp(-2 * math.log10(x) + 2)


def create_adjMat(agents):
    distMat = np.zeros((len(agents), len(agents)))
    velMat = np.zeros((len(agents), len(agents)))
    for a1, a2 in pairs(agents, 2):
        distMat[a1.id, a2.id] = distMat[a2.id, a1.id] = norm(a1.pos-a2.pos)
        velMat[a1.id, a2.id] = velMat[a2.id, a1.id] = norm(a1.vel - a2.vel)
    return distMat, velMat


def brake_decay(r, a, p):
    arr = np.array(r)
    arr[r < 0] = 0
    arr[r < (a / p ** 2)] *= p
    arr[r >= (a / p ** 2)] = np.sqrt((2 * a * arr[r >= (a / p ** 2)]) - (a / p) ** 2)
    return arr

def add_innernoise(gpos, gvel, sigma, deltaT, lam =0.1):
    force = -(np.sqrt(2*lam*sigma)/3)*deltaT*gpos
    damping = -lam*deltaT*gvel
    noiseToAdd = np.random.normal(0, 1, size=len(gpos))*(np.sqrt(2*lam*sigma*deltaT)) + damping + force
    gvel += noiseToAdd
    gpos += gvel * deltaT

def add_outernoise(vel, sigma, deltaT):
    vel += np.random.normal(0, 1, size=len(vel))*np.sqrt(2*sigma*deltaT)