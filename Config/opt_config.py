# optimization configuration file
#set any values to None to use default optimizer value

population_size = 100
number_of_generations = 10
number_of_variables = 11
parallelEnvs_per_core = 3  # number of environments running asynchronously per core per serial iteration
number_of_cpus = 4
#            LL      UL
var_lims = [(20   ,  50),     # 0  r0_rep
            (0.03 ,  0.16),  # 1  p_rep
            (58.5 ,  100),   # 2  r0_frict
            (0.03 ,  0.22),  # 3  a_frict
            (0.3  ,  2.7),   # 4  p_frict
            (0.38 ,  9.67),  # 5  v_frict
            (5.04 ,  10.0),  # 6  c_frict
            (0    ,  9.3),   # 7  r0_shill
            (15   ,  20.0),  # 8  v_shill
            (0.48 ,  9.96),  # 9  a_shill
            (2.53 ,  5.12)]  # 10  p_shill

init_vars = [0.5]*number_of_variables
sigma = 0.3
