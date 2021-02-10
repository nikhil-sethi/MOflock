import math
from Config import defaults as df
from Utils.controls import sigmoid_decay


def f1(phi, phi_o, d):
    """sinusoidally growing function between (phi_o-d) to phi_o"""
    return 1 - sigmoid_decay(phi, phi_o, d)


def f2(phi, sigma):
    """normal distribution"""
    return math.exp(-phi ** 2 / sigma ** 2)


def f3(phi, a):
    """sharp peak"""
    return a ** 2 / (phi + a) ** 2


def optofitness(op_array, n_obj=1):
    """apply respective transfer functions to an array of order parameters
       **order of elements matters
    """
    d = 5
    f_speed = f1(op_array[1], df.v_flock, df.v_tol)
    f_coll = f3(op_array[3], df.a_tol)
    f_disc = f3(op_array[4], df.num_agents / 5)
    f_wall = f2(op_array[0], df.r_tol)
    f_cluster = f1(op_array[5], df.num_agents / 5, df.num_agents / 5)
    if op_array[2] > 0:
        f_corr = op_array[2]
    else:
        f_corr = 0
    time_fit = 1  # (1-sigmoid_decay(op_array[6], df.max_sim_time-df.wait_time, 200))
    if n_obj == 2:
        # F1 = -time_fit * f_speed * f_corr * f_disc * f_cluster
        # F2 = -time_fit * f_wall * f_coll
        F2 = -time_fit *f_coll * f_corr * f_disc * f_cluster
        F1 = -time_fit * f_wall * f_speed
        return round(F1, d), round(F2, d)

    elif n_obj == 3:
        F1 = -time_fit * f_speed * f_corr * f_disc * f_cluster
        F2 = -time_fit * f_wall
        F3 = -time_fit * f_coll
        return round(F1, d), round(F2, d), round(F3, d)
    elif n_obj == 'all':
        return round(f_wall, d), round(f_speed, d), round(f_corr, d), round(f_coll, d), round(f_disc, d), round(f_cluster, d)
    F1 = -time_fit * f_speed * f_coll * f_disc * f_wall * f_corr * f_cluster
    return round(F1, d)
