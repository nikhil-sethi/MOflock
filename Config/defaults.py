# [simulation]
num_agents = 10
num_envs = 1  # = maximum number of processors if 0 or greater than max processors ;  =1 if -1
max_sim_time = 600  # seconds
speedup = 7  # BEWARE! Using a high value will result in loss of simulation time. depends on animation, cpu usage etc.
wait_time = 0  # seconds to wait before doing any order parameter calculation
interval = 0.1  # time in seconds between each update
envseed = 'id'  # None: seed=random ; 'id': seed=env.id ; k: seed=k
optimizer = "CMA-ES"  # ("CMA-ES","PSO", "NSGA2")
pg_scale = 1  # pixels per unit length    pg_scale=
bound_tol = 100 # metres
n_clusters = 1
start_loc = 0, 0
start_sep = 10
start_type = 'random'
chunksize = 1

obs_flag = False
wall_flag = True
sep_flag = True
align_flag = True
wp_flag = False
animated = True
opt_flag = False  # flag for optimization
mp_affinity = False
warning_flag = False  # shows earnings every 5 seconds
save_data = True

# [Agent]

# physics
vmax = 6.0  # m/s
amax = 6.0  # m2/s
v_flock = 6.0  # m/s
v_target = 2.0
# sensors
gps_del = 0.2  # seconds
sigma_inner = 0.005  # m2/s2
memory_len = 5
comm_del = 1  # s
comm_radius = 100.0  # m
coll_radius = 3.0

# [environment]
# weather
weather = {'temperature': 37,
           'wind_speed': 0,
           'wind_direction': None}
sigma_outer = 0.2  # m2/s3

# [optimization]
v_tol = 1.5 * v_flock / 4
a_tol = 0.0003
r_tol = 5
