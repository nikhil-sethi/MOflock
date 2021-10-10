import Config.cobot_config as cf
from Classes.environment import Env
from Classes.cobot import CoBot
import multiprocessing as mp
import random


def run_env(env, q=None):
    return env.run(q)


def setup_envs(pop, seed, q):
    envs = []
    for i in range(len(pop)):
        genome = pop[i]
        paramdict = dict(zip(cf.paramdict.keys(), genome))
        env = Env(paramdict, seed)
        env.add_agents(CoBot)
        envs.append((env, q))
    return envs


def eval_envs_parallel(envs, chunksize):
    p = mp.Pool(len(envs))
    job = p.starmap(run_env, envs, chunksize=chunksize)
    p.close()
    p.join()
    return job
