# optimization configuration file

population_size = 50   # = None for default
number_of_generations = 1   # = None for default
number_of_variables = 11
parallelEnvs_per_core = 3  # number of environments running asynchronously per core per serial iteration
number_of_cpus = 4
#            LL      UL
var_lims = [(15   ,  50),    # 0  r0_rep
            (0.03 ,  0.3),   # 1  p_rep
            (20 ,    70),    # 2  r0_frict
            (0.13 ,  0.32),  # 3  a_frict
            (0.5  ,  3.7),   # 4  p_frict
            (2.38 ,  7.67),  # 5  v_frict
            (5.04 ,  10.0),  # 6  c_frict
            (-20    ,5.3),   # 7  r0_shill
            (15   ,  20.0),  # 8  v_shill
            (3.48 ,  10.96),  # 9  a_shill
            (2.53 ,  6.12)]  # 10  p_shill

init_vars = [0.5]*number_of_variables
sigma = 0.3
pso_opts= {'c1': 0.5, 'c2': 0.3, 'w': 0.9}