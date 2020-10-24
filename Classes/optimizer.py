import numpy as np
from Classes.environment import Env
from Classes.cobot import CoBot
import Config.defaults as df
from Config import cobot_config
import psutil
from cma import CMAEvolutionStrategy
import multiprocessing as mp
from multiprocessing.pool import Pool
from Methods.transfer_funcs import orderparamsTofitness
import math
from itertools import chain
from time import perf_counter
from copy import deepcopy

class NoDaemonProcess(mp.Process):  # courtesy of https://stackoverflow.com/questions/43388770/how-to-create-a-process-inside-a-process-using-multiprocessing-in-python
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(Pool):
    Process = NoDaemonProcess

class Optimizer(CMAEvolutionStrategy):
    def __init__(self, params, agent_type=CoBot):
        super().__init__(params["init_vars"], params["sigma"], {'popsize':params["npop"]})
        if params["ngen"] >0: self.opts.set('maxiter', params["ngen"])
        # assert df.animated == False, 'Animation with optimization is not supported. Please change to headless mode'
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
            start = perf_counter()
            print(f'\n **** Generation {g} started **** ')
            X = self.ask()  # popsize x N
            # scaled_X = [[18.13797864,	0.284189019,	27.66904294	,0.228751087,	2.89329558	,4.341461035,	7.310661496	,-11.00204234,	18.62826807	,8.081053292,	4.75371955]]*100

            scaled_X = self.var_ll + (self.var_ul - self.var_ll) * X
            self.pops.append(scaled_X)
            fitness_vector = self.get_pop_fitness(scaled_X)   # popsize x 1 vector
            self.tell(X, fitness_vector)
            self.disp(1)
            print(fitness_vector)
            self.fitnesses = fitness_vector
            print(f'Total time taken by generation {g}: {(perf_counter()-start)/60} minutes')
            g += 1


    def get_pop_fitness(self, X):
        subarrays = math.ceil(self.popsize / self.m)
        cpus = (list(range(self.ncpu))*math.ceil(subarrays/self.ncpu))
        temp = list(zip(range(self.popsize), X))
        genomes = np.array_split(temp, subarrays)
        p = MyPool(self.ncpu)
        op_per_core_eval = p.starmap(self.eval_core, zip(cpus, genomes), chunksize=1)  #done for each cpu
        op_all = chain.from_iterable(op_per_core_eval)  # unpack all order params into one iterable
        self.ops.append(np.array(list(deepcopy(op_all))))
        fitnesses = list(map(orderparamsTofitness, op_all))
        p.close()
        p.join()
        return fitnesses

    def eval_core(self, whichcpu, asyncGenomes):
        """ calls each parallel environment's run function after some preprocessing"""
        proc = psutil.Process()
        if df.mp_affinity:
            proc.cpu_affinity([whichcpu])
        p = mp.Pool(len(asyncGenomes))
        # print(proc.pid, proc.cpu_affinity())
        jobs = []
        for whichenv, genome in asyncGenomes:

            env = Env(whichenv)  # new env for process
            params = dict(zip(self.varnames, genome))  # pack up for agents
            # params = cobot_config.paramdict
            env.add_agents(self.agent_type, params, seed=whichenv+1)
            jobs.append(p.apply_async(env.run, args=(whichenv+1,)))

        res = [job.get() for job in jobs]
        p.close()
        p.join()
        return res

    def set_popsize(self, value):
        self._popsize = value


'''
pseudocode

def start():
    while g <num_gens
        while t < gen_time
            get model parameters
            get order parameters from opt config file
            calculate fitness from transfer functions
            if current fitness> max fitness
                evolution step
        write best fitness model params to cobot config
                
                
            
        
        

'''
