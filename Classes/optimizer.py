import numpy as np
from Classes.environment import Env
from Classes.cobot import CoBot
import Config.defaults as df

import multiprocessing as mp
from Utils.transfer_funcs import optofitness
from itertools import chain
from time import perf_counter
from copy import deepcopy
from Utils.opt_utils import eval_envs_parallel, setup_envs


if df.optimizer == 'CMA-ES':
    from cma import CMAEvolutionStrategy

    class CMA(CMAEvolutionStrategy):

        def __init__(self, params):
            self.var_ll, self.var_ul = map(np.array, zip(*params["var_lims"]))
            super().__init__(params["init_vars"], params["sigma"], {'popsize': params["npop"]})#, 'bounds': [[0] * len(params["init_vars"]), [1] * len(params["init_vars"])]})
            if params["ngen"] is not None: self.opts.set('maxiter', params["ngen"])
            assert df.animated is False, 'Animation with optimization is not supported. Please change to headless mode'

            self.maxiters = params["ngen"]
            self.ncpu = params["ncpu"]
            self.m = params["asyncEnvs/core"]
            self.chunksize = params["chunksize"]
            self.varnames = list(params["vars"].keys())
            assert len(self.varnames) == self.N, 'Number of parameters should be same as number of optimization variables.'

            self.data = []
            self.op_history=[]
            self.env_count = 0

        def normal_pop(self, mu, sd):
            return np.random.normal(mu, sd, size=(self.popsize, self.N))

        def uniform_pop(self):
            np.random.seed()
            return np.random.uniform(size=(self.popsize, self.N))

        def run(self):
            g=0
            print("Starting CMA-ES with ", self.maxiters, "generations")
            while not self.stop(): #g<self.maxiters
                self.env_count = 0
                start = perf_counter()
                # print(f'\n **** Generation {g} started **** ')
                X = self.ask()  # popsize x N self.uniform_pop() #
                scaled_X = self.var_ll + (self.var_ul - self.var_ll) * X
                fitness_vector = self.get_pop_fitness(scaled_X, n_obj=1)  # popsize x 1 vector
                self.tell(X, fitness_vector)
                self.disp(1)
                print(fitness_vector)
                self.fitnesses = fitness_vector
                # print(f'Total time taken by generation {g}: {(perf_counter() - start) / 60} minutes')
                g += 1

        def get_pop_fitness(self, X, n_obj):
            envs = setup_envs(X, seed=df.envseed)
            op_history = eval_envs_parallel(envs, self.chunksize)
            ops = [op[0][-1] for op in op_history]
            fits = [optofitness(op, n_obj=n_obj) for op in ops]
            self.op_history.append(op_history)
            self.data.append(np.hstack((X, ops, np.expand_dims(fits,1))))
            return fits


elif df.optimizer == 'PSO':
    from pyswarms.single import GlobalBestPSO


    class PSO(GlobalBestPSO):
        def __init__(self, params, agent_type=CoBot):
            bounds = tuple(map(np.array, zip(*params["var_lims"])))
            super().__init__(params["npop"], len(params["vars"]), params["opts"], bounds=bounds)

            assert df.animated is False, 'Animation with optimization is not supported. Please change to headless mode'

            self.maxiters = params["ngen"]
            self.ncpu = params["ncpu"]
            self.m = params["asyncEnvs/core"]
            self.varnames = list(params["vars"].keys())
            assert len(
                self.varnames) == self.dimensions, 'Number of parameters should be same as number of optimization variables.'
            self.agent_type = agent_type

            self.fits = []
            self.times = []
            self.ops = []
            self.env_count = 0
            # mu = (params["var_lims"] + params["var_ul"]) / 2
            # self.pop = self.init_pop(mu, params["var_sd"])

        def init_pop(self, mu, sd):
            return np.random.normal(mu, sd, size=(self.n_particles, self.dimensions))

        def run(self):
            self.optimize(self.get_pop_fitness, self.maxiters)

        def get_pop_fitness(self, X):
            subarrays = self.n_particles / self.m / self.ncpu
            genomes = np.array_split(X, subarrays)
            op_all_nested = list(map(self.eval_async, genomes))  # done for m x ncpu environments at a time
            op_all = chain.from_iterable(op_all_nested)  # unpack all order params into one iterable
            self.ops.append(np.array(list(deepcopy(op_all))))
            fitnesses = list(map(optofitness, op_all))
            self.fits.append(deepcopy(fitnesses))
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

            ops = [job.get() for job in jobs]

            p.close()
            p.join()
            return ops
elif df.optimizer == 'NSGA2':
    from pymoo.algorithms.nsga2 import NSGA2
    from pymoo.factory import get_sampling, get_crossover, get_mutation
    from pymoo.model.problem import Problem
    from pymoo.factory import get_termination
    from pymoo.optimize import minimize

    class MOProblem(Problem):
        def __init__(self, params, agent_type=CoBot):
            bounds = tuple(map(np.array, zip(*params["var_lims"])))
            super().__init__(n_var=params["n_vars"],
                             n_obj=2,
                             xl=np.array(bounds[0]),
                             xu=np.array(bounds[1]))
            self.ncpu = params["ncpu"]
            self.m = params["asyncEnvs/core"]
            self.chunksize = params["chunksize"]
            self.varnames = list(params["vars"].keys())
            assert len(
                self.varnames) == self.n_var, 'Number of parameters should be same as number of optimization variables.'

            self.op_history = []
            self.data = []

            self.env_count = 0
            self.algorithm = NSGA2(pop_size=params["npop"],
                                   n_offsprings=params["npop"],
                                   sampling=get_sampling("real_random"),
                                   crossover=get_crossover("real_sbx", prob=0.9, eta=15),
                                   mutation=get_mutation("real_pm", eta=20),
                                   eliminate_duplicates=True,
                                   seed=1)
            self.termination = get_termination("n_gen", params["ngen"])

        def run(self):
            return minimize(self,
                            self.algorithm,
                            self.termination,
                            seed=1,
                            save_history=True,
                            verbose=True)

        def _evaluate(self, X, out, *args, **kwargs):
            envs = setup_envs(X, seed=df.envseed)
            op_history = eval_envs_parallel(envs, self.chunksize)
            ops = [op[0][-1] for op in op_history]
            fits = [optofitness(op, n_obj=2) for op in ops]
            self.op_history.append(op_history)
            self.data.append(np.hstack((X, ops, fits)))
            f1, f2 = zip(*fits)
            out["F"] = np.column_stack([np.array(f1), np.array(f2)])








