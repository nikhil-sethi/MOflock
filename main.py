'''
Version: 3.4
features/corrections added
- multiobjective optimizer: pymoo/NSGA2 is now supported
- order parameter history is now stored
- live graph for order parameters
- corrected a bug in collision order parameter: self.phi_coll=0 in cobot/get_nbors()
- added sliders to control params in real time
- some major refactoring in environment.py- two seperate loops for wait and measure
- choice between grid and random initialisation
- changed the disconnection calculation under r_cluster instead of r_comm

Problems/known issues:
- architecture to choose optimizer is pretty bad. Need to work on it but not urgent
- live update sliders will not update r_cluster as of now
- Things are slow during animation. thinking of changing the engine to pygame


'''

from Config import defaults, cobot_config, opt_config

from Classes.environment import Env
from Classes.cobot import CoBot  # a collaborative aerial bot
import pickle
import multiprocessing as mp
import numpy as np
from Methods.transfer_funcs import optofitness
from time import perf_counter

# if __name__ == '__main__':
if defaults.opt_flag:
    parameters = {
        "npop": opt_config.population_size,
        "ngen": opt_config.number_of_generations,
        "ncpu": opt_config.number_of_cpus,
        "n_vars": opt_config.number_of_variables,
        "asyncEnvs/core": opt_config.parallelEnvs_per_core,
        "vars": cobot_config.paramdict,
        "var_lims": np.array(opt_config.var_lims),
    }
    if defaults.optimizer == "CMA-ES":
        from Classes.optimizer import CMA
        parameters.update(init_vars=opt_config.init_vars,
                          sigma=opt_config.sigma)
        opt = CMA(parameters)
    elif defaults.optimizer == "PSO":
        from Classes.optimizer import PSO
        parameters.update(opts=opt_config.pso_opts)
        opt = PSO(parameters)
    elif defaults.optimizer == "NSGA2":
        from Classes.optimizer import MOProblem
        opt = MOProblem(parameters)

    start = perf_counter()
    # try:
    result = opt.run()
    # except:
    #     "Please provide a valid optimizer from the following options: \n" \
    #     "CMA-ES, PSO, NSGA2"

    if defaults.save_data:
        f = open("opt_pickle", "ab")
        s = {"times": opt.times,
            "fits": opt.fits,
            "pops": opt.pops,
            "ops": opt.ops,
            "res": result}
        pickle.dump(s, f)
        fevals = opt_config.population_size * opt_config.number_of_generations
        # np.savetxt("opt.csv", np.hstack((np.concatenate(opt["pops"]),np.concatenate(opt["ops"]),np.concatenate(opt["fits"]))), delimiter=',')
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

    # var_ll, var_ul = map(np.array, zip(*opt_config.var_lims))
    # pop = var_ll+(var_ul-var_ll)*np.random.uniform(size=(num_envs, opt_config.number_of_variables))
    # pop = [[ 1.31563090e+01,  3.99545570e-01,  7.47478539e+01,  3.57726493e+00,
    #     2.69535304e+00,  1.51287003e-01,  3.21752617e-02, -1.49772819e+01,
    #     1.76489941e+01,  5.97210480e+00,  6.10416763e+00]]

    for i in range(num_envs):
        env = Env(i)
    #     genome = opt["pops"][24][i]
    #     genome = pop[i]
        genome = [12.1438	,0.3996	,73.8804,	3.3078	,2.9063,	0.1338	,0.0309	,-12.3612,17.7703,	5.9838,	6.2384
]
        paramdict = dict(zip(cobot_config.paramdict.keys(), genome))
        # paramdict = cobot_config.paramdict  # change this if you want for each environment
        seed=defaults.envseed
        env.add_agents(CoBot, paramdict, seed=seed)
        if num_envs>1:
            jobs.append(pool.apply_async(env.run, args=(seed,)))
        else:
            order_paramlist=[env.run(seed=seed)]
    # Wait for children to finnish
    # pool.join()
    if num_envs>1:
        pool.close()
        order_paramlist = [job.get() for job in jobs]
    # print("ops= ",order_paramlist[0])
    fitnesses= [optofitness(i[-1], n_obj=2) for i in order_paramlist]  # list(map(optofitness, order_paramlist))
    print("Fitness= ",fitnesses)
