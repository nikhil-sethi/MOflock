# optimization configuration file

population_size = 50  # = None for default
number_of_generations = 40  # = None for default
number_of_variables = 12
parallelEnvs_per_core = 5  # number of environments running asynchronously per core per serial iteration
number_of_cpus = 12
n_obj = 2
chunksize = 2
#            LL      UL
var_lims = {'r0_rep': (30.8, 51),  # 0  r0_rep
            'p_rep': (0.02, 0.10),  # 1  p_rep
            'r0_frict': (58.5, 100),  # 2  r0_frict
            'a_frict': (5.04, 10.0),  # 3  a_frict
            'p_frict': (0.38, 9.67),  # 4  p_frict
            'v_frict': (0.3, 2.7),  # 5  v_frict
            'c_frict': (0.03, 0.22),  # 6  c_frict
            'r0_shill': (-10, 0),  # 7  r0_shill
            'v_shill': (10.0, 15.0),  # 8  v_shill
            'a_shill': (1.54, 6.55),  # 9  a_shill
            'p_shill': (0.48, 9.96),  # 10  p_shill
            'c_shill': (0.3, 1)}

init_vars = [0.5] * number_of_variables
sigma = 0.3
pso_opts = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
