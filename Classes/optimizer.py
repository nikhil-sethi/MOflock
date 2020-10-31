import numpy as np
from Classes.environment import Env
from Classes.cobot import CoBot
import Config.defaults as df
from Config import cobot_config
from cma import CMAEvolutionStrategy
import multiprocessing as mp
from Methods.transfer_funcs import orderparamsTofitness
import math
from itertools import chain
from time import perf_counter
from copy import deepcopy


class Optimizer(CMAEvolutionStrategy):
    def __init__(self, params, agent_type=CoBot):
        super().__init__(params["init_vars"], params["sigma"], {'popsize': params["npop"]})
        if params["ngen"] is not None: self.opts.set('maxiter', params["ngen"])
        assert df.animated is False, 'Animation with optimization is not supported. Please change to headless mode'
        # assert len(params["var_lims"]) == \
        #        len(params["var_lims"]) == \
        #        len(params["var_ul"]) == \
        #        params["nvars"], \
        #     "Standard deviation and variable limits should be same as number of variables"

        self.ncpu = params["ncpu"]
        self.m = params["asyncEnvs/core"]
        self.varnames = list(params["vars"].keys())
        self.var_ll, self.var_ul = map(np.array, zip(*params["var_lims"]))
        assert len(self.varnames) == self.N, 'Number of parameters should be same as number of optimization variables.'
        self.agent_type = agent_type
        self.pops = []
        self.ops = []
        # mu = (params["var_lims"] + params["var_ul"]) / 2
        # self.pop = self.init_pop(mu, params["var_sd"])

    def init_pop(self, mu, sd):
        return np.random.normal(mu, sd, size=(self.popsize, self.N))

    def run(self):
        g = 0
        while not self.stop():
            self.env_count = 0
            start = perf_counter()
            print(f'\n **** Generation {g} started **** ')
            X = self.ask()  # popsize x N
            scaled_X = self.var_ll + (self.var_ul - self.var_ll) * X
            self.pops.append(scaled_X)
            fitness_vector = self.get_pop_fitness(scaled_X)  # popsize x 1 vector
            self.tell(X, fitness_vector)
            self.disp(1)
            print(fitness_vector)
            self.fitnesses = fitness_vector
            print(f'Total time taken by generation {g}: {(perf_counter() - start) / 60} minutes')
            g += 1

    def get_pop_fitness(self, X):
        subarrays = math.ceil(self.popsize / self.m / self.ncpu)
        genomes = np.array_split(X, subarrays)
        op_all_nested = list(map(self.eval_async, genomes))  # done for m x ncpu environments at a time
        op_all = chain.from_iterable(op_all_nested)  # unpack all order params into one iterable
        self.ops.append(np.array(list(deepcopy(op_all))))
        fitnesses = list(map(orderparamsTofitness, op_all))
        return fitnesses

    def eval_async(self, asyncGenomes):
        """ calls each parallel environment's run function after some preprocessing"""
        p = mp.Pool(len(asyncGenomes))
        # print(proc.pid, proc.cpu_affinity())
        jobs = []
        for genome in asyncGenomes:
            env = Env(self.env_count)  # new env for process
            params = dict(zip(self.varnames, genome))  # pack up for agents
            # params = cobot_config.paramdict
            env.add_agents(self.agent_type, params, seed=df.envseed)
            jobs.append(p.apply_async(env.run, args=(df.envseed,)))
            self.env_count += 1

        res = [job.get() for job in jobs]
        p.close()
        p.join()
        return res
