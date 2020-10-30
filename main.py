'''
Version: 3.1
features/corrections added
- optimization
- multiprocessing
- collective target tracking with mouse input
- configuration, API changes
- wall avoidance algo changed (repulsive corner wasnt the best)
- animation capability changed to custom loop instead of funcanimation

problems
- A lot of stuff might not be optimized for least complexity

TODO-other branches
- RL
- Cooperation + target detection
- SITL, ROS etc.
> paper implementaion essentially ends here. New features will be added in other branches

'''

from Config import defaults, cobot_config, opt_config

from Classes.environment import Env
from Classes.cobot import CoBot  # a collaborative aerial bot
from Classes import optimizer

import multiprocessing as mp
import numpy as np
from Methods.transfer_funcs import orderparamsTofitness
from time import perf_counter

# agents[0].scp(0,0)
# agents[1].scp(-20,0)
# agents[0].scv(1,1)
# agents[1].scv(1,0)
# agents[0].memory.append(agents[0].get_state())
# agents[1].memory.append(agents[1].get_state())

if __name__ == '__main__':
    if defaults.opt_flag:

        parameters = {
            "npop": opt_config.population_size,
            "ngen": opt_config.number_of_generations,
            "ncpu": opt_config.number_of_cpus,
            "asyncEnvs/core": opt_config.parallelEnvs_per_core,
            "vars": cobot_config.paramdict,
            "var_lims": np.array(opt_config.var_lims),
            "init_vars": opt_config.init_vars,
            "sigma": opt_config.sigma
        }
        start = perf_counter()
        opt = optimizer.Optimizer(parameters)
        opt.run()
        pops = opt.pops
        ops = opt.ops

        print(f"Total time taken for optimization = {(perf_counter() - start)/60} minutes")

    else:

        num_envs = defaults.num_envs
        if num_envs <= 1:
            num_envs = 1
        # elif num_envs > 1:
        #     assert defaults.animated == False, '!!Animation with multiple environments is not supported. Please change to headless mode.!!'

        pool = mp.Pool()
        jobs = []
        for i in range(num_envs):
            env = Env(i)
            genome = [18.13797864,	0.284189019,	27.66904294	,0.228751087,	2.89329558	,4.341461035,	7.310661496	,-11.00204234,	18.62826807	,8.081053292,	4.75371955]
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
