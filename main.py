'''
Version 4.0 (Breaking)
features/corrections added
- New visualisation engine with pygame
- Changed parallel optimization to pool methods
- New opt uitls file for easier handling of multiprocessing
- Better and cleaner implementation of order parameter calculations
- support for steady state order parameter added
- New target order parameter added
- support for simultaneous simulations of different parameter sets in single environment(multisim.py)
- Refactored bot, particle and cobot classes
- added new parameter c_shill
- simulator has been made closer to the original work

Problems/known issues:
- obstacle avoidnace is broken. Do not use!
- architecture to choose optimizer is pretty bad. Need to work on it but not urgent
- Sliders removed in this version as they are not that important

'''

import Config.cobot_config as cf
import Config.defaults as df
import Config.opt_config as of
import pickle
import numpy as np
from Utils.transfer_funcs import optofitness
from time import perf_counter
from Utils.opt_utils import setup_envs, eval_envs_parallel

# if __name__ == '__main__':
if df.opt_flag:
    parameters = {
        "npop": of.population_size,
        "ngen": of.number_of_generations,
        "ncpu": of.number_of_cpus,
        "n_vars": of.number_of_variables,
        "asyncEnvs/core": of.parallelEnvs_per_core,
        "vars": cf.paramdict,
        "var_lims": of.var_lims.values(),
        "chunksize": of.chunksize
    }
    if df.optimizer == "CMA-ES":
        from Classes.optimizer import CMA

        parameters.update(init_vars=of.init_vars,
                          sigma=of.sigma)

        opt = CMA(parameters)
    elif df.optimizer == "PSO":
        from Classes.optimizer import PSO

        parameters.update(opts=of.pso_opts)
        opt = PSO(parameters)
    elif df.optimizer == "NSGA2":
        from Classes.optimizer import MOProblem

        opt = MOProblem(parameters)

    start = perf_counter()
    # try:

    print(f"Optimization started with {df.optimizer}"
          f"**** Configuration **** \n"
          f"Popsize: {of.population_size} \n "
          f"Generations: {of.number_of_generations} \n"
          f"Chunksize: {of.chunksize} \n"
          f"Bounds: {of.var_lims} \n"
          f"****")
    result = opt.run()
    # except:
    #     "Please provide a valid optimizer from the following options: \n" \
    #     "CMA-ES, PSO, NSGA2"

    if df.save_data:
        f = open("opt_pickle", "ab")
        s = {"data": opt.data,
             "op_history": opt.op_history,
             # "res_history": [(result.history[i].opt.get("X"), result.history[i].opt.get("F")) for i in
             #                 range(len(result.history))]
             }
        pickle.dump(s, f)
        fevals = of.population_size * of.number_of_generations
        # np.savetxt("opt.csv", np.hstack((np.concatenate(opt["pops"]),np.concatenate(opt["ops"]),np.concatenate(opt["fits"]))), delimiter=',')
        f.close()
    print(f"Total time taken for optimization = {(perf_counter() - start) / 60} minutes")

else:

    var_ll, var_ul = map(np.array, zip(*of.var_lims.values()))
    # pop = var_ll + (var_ul - var_ll) * np.random.uniform(size=(df.num_envs, len(var_ul)))
    # pop = [list(cf.paramdict.values())]*df.num_envs
    pop = [[  3.34531599e+01,  2.84441546e-02,  5.89520177e+01,  8.22330460e+00,
        2.67682789e+00,  3.00404844e-01,  1.84166142e-01, -2.13483170e-01,
        1.29389626e+01,  2.57143904e+00,  1.30148574e+00,  9.31371072e-01]] * df.num_envs
    # pop = np.array([[  3.36971443e+01,  2.36834330e-02,  5.92608031e+01,  5.38039876e+00,
    #     4.62063792e+00,  1.73348260e+00,  3.52457821e-02, -2.45024732e+00,
    #     1.29383823e+01,  4.84342556e+00,  4.83207547e+00,  5.55573940e-01],
    #                [  3.34531599e+01,  2.84441546e-02,  5.89520177e+01,  8.22330460e+00,
    #     2.67682789e+00,  3.00404844e-01,  1.84166142e-01, -2.13483170e-01,
    #     1.29389626e+01,  2.57143904e+00,  1.30148574e+00,  9.31371072e-01]]*1)    # 82 9394
    # pop = np.array([[3.13791389e+01, 5.88404323e-02, 6.19661687e+01,
    #                  6.94702705e+00, 1.81474582e+00, 3.30122333e-01,
    #                  1.77654583e-01, -1.94089572e+00, 1.29068772e+01,
    #                  3.54015072e+00, 5.43638404e+00, 7.48881247e-01],
    #                 [3.26168002e+01, 2.25248339e-02, 5.91091630e+01,
    #                  8.18629996e+00, 3.30108761e+00, 1.66213931e+00,
    #                  3.72514451e-02, -6.58828690e+00, 1.29394850e+01,
    #                  3.54767023e+00, 5.43828556e+00, 7.41399626e-01]])
    #

    envs = setup_envs(pop, seed=df.envseed, q=None)

    if df.num_envs > 1:
        # assert defaults.animated is False, '!!Animation with multiple environments is not recommended. Please change to headless mode.!!'
        start = perf_counter()
        op_all = eval_envs_parallel(envs, df.chunksize)
        print(f"Total time taken for parallel run = {(perf_counter() - start) / 60} minutes")

    else:
        op_all = [envs[0][0].run()]

    ops = np.array([op[0][-1] for op in op_all])
    fits = np.array([optofitness(op, n_obj='all') for op in ops])
    print("Fitness =", fits)

    # import pickle
    # import numpy as np
    # f = open("random",'ab')
    #
    # d = {"data": np.hstack((pop, ops, fits)), "op_history":op_all}
    # pickle.dump(d, f)
    # f.close()
# x_fav=0.72784593, -0.07781396,  0.37246379,  0.5320078 ,  0.09619003,
#        0.52643428,  0.87741975,  1.11721408,  1.34201872, -0.05435581,
#        0.45408709,  0.80462036



