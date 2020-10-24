import math
from Config import defaults as df
from Methods.controls import sigmoid

def f1(phi, phi_o, d):
    """sinusoidally growing function between (phi_o-d) to phi_o"""
    return 1-sigmoid(phi, phi_o, d)

def f2(phi, sigma):
    """normal distribution"""
    return math.exp(-phi**2/sigma**2)

def f3(phi, a):
    """sharp peak"""
    return a**2/(phi + a**2)

def orderparamsTofitness(op_array):
    """apply respective transfer functions to an array of order parameters
       **order of elements matters
    """
    f_speed = f1(op_array[1], df.v_flock, df.v_tol)
    f_coll = f3(op_array[3], df.a_tol)
    f_disc = f3(op_array[4], df.num_agents/5)
    f_wall = f2(op_array[0], df.r_tol)
    if op_array[2] > 0:
        f_corr = op_array[2]
    else:
        f_corr = 0
    return -f_speed*f_coll*f_disc*f_wall*f_corr