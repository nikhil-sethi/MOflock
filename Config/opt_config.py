# optimization configuration file

population_size = 48   # = None for default
number_of_generations = 2   # = None for default
number_of_variables = 11
parallelEnvs_per_core = 4  # number of environments running asynchronously per core per serial iteration
number_of_cpus = 12
#            LL      UL
var_lims = {'r0_rep':(15   ,  50),    # 0  r0_rep
            'p_rep':(0.03 ,  0.4),   # 1  p_rep
            'r0_frict':(20 ,    70),    # 2  r0_frict
            'a_frict':(4.04 ,  11.0),  # 3  a_frict
            'p_frict':(0.13 ,  9.67),  # 4  p_frict
            'v_frict':(0.1  ,  3.7),   # 5  v_frict
            'c_frict':(0.03 ,  2.22),   # 6  c_frict
            'r0_shill':(-15    ,5.3),   # 7  r0_shill
            'v_shill':(13   ,  20.0),  # 8  v_shill
            'a_shill':(1.53 ,  6.12),  # 9  a_shill
            'p_shill':(0.48 ,  10.96)}  # 10  p_shill

init_vars = [0.5]*number_of_variables
sigma = 0.3
pso_opts = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}