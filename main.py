'''
Version: 3.3
features/corrections added
- Particle swarm optimizer
- new order parameter for minimum clusters
- corrected order parameter calculations for wall, disconnected, clusters
- corrected cluster radius calculation via brake_decay_inverse function


problems

'''

from Config import defaults, cobot_config, opt_config

from Classes.environment import Env
from Classes.cobot import CoBot  # a collaborative aerial bot
from Classes.optimizer import Optimizer_PSO
import pickle
import multiprocessing as mp
import numpy as np
from Methods.transfer_funcs import orderparamsTofitness
from time import perf_counter

if __name__ == '__main__':
    if defaults.opt_flag:
        # parameters = {
        #     "npop": opt_config.population_size,
        #     "ngen": opt_config.number_of_generations,
        #     "ncpu": opt_config.number_of_cpus,
        #     "asyncEnvs/core": opt_config.parallelEnvs_per_core,
        #     "vars": cobot_config.paramdict,
        #     "var_lims": np.array(opt_config.var_lims),
        #     "init_vars": opt_config.init_vars,
        #     "sigma": opt_config.sigma
        # }

        parameters = {
            "npop": opt_config.population_size,
            "ngen": opt_config.number_of_generations,
            "ncpu": opt_config.number_of_cpus,
            "asyncEnvs/core": opt_config.parallelEnvs_per_core,
            "vars": cobot_config.paramdict,
            "var_lims": np.array(opt_config.var_lims),
            "opts": opt_config.pso_opts
        }
        start = perf_counter()
        opt = Optimizer_PSO(parameters)
        opt.run()
        if defaults.save_data:
            f = open("opt_pickle", "ab")
            s = {"times": opt.times, "fits": opt.fits, "pops": opt.pos_history, "ops": opt.ops}
            pickle.dump(s, f)
            #np.savetxt("opt.csv", np.hstack((np.array(opt["pops"]).reshape(3000,11),np.array(opt["ops"]).reshape(3000,6),np.array(opt["fits"]).reshape(3000,1))), delimiter=',')
            f.close()
        print(f"Total time taken for optimization = {(perf_counter() - start) / 60} minutes")

    else:

        num_envs = defaults.num_envs
        if num_envs <= 1:
            num_envs = 1
        elif num_envs > 1:
            assert defaults.animated is False, '!!Animation with multiple environments is not recommended. Please change to headless mode.!!'

        pool = mp.Pool(num_envs)
        jobs = []
        for i in range(num_envs):
            env = Env(i)
            genome = [34.1342,	0.2866,	20.366	,0.2657	,3.6581	,7.4869	,9.8308	,-13.838,	15.8349,	8.8114,	5.7284]
            paramdict = dict(zip(cobot_config.paramdict.keys(), genome))
            # paramdict = cobot_config.paramdict  # change this if you want for each environment
            env.add_agents(CoBot, paramdict, seed=defaults.envseed)
            jobs.append(pool.apply_async(env.run, args=(defaults.envseed,)))

        # Wait for children to finnish
        pool.close()
        pool.join()

        order_paramlist = [job.get() for job in jobs]
        fitnesses = list(map(orderparamsTofitness, order_paramlist))
        print(fitnesses)
