
# [simulation]
num_agents = 30
num_envs = 1  # = maximum number of processors if 0 or greater than max processors ;  =1 if -1
max_sim_time = 660  # seconds
wait_time = 60  # seconds to wait before doing any order parameter calculation
interval = 0.1  # time in seconds between each update
envseed = 1  # Keep >0 if you want to seed. Else 0 or none. seed value to make all environments independent of randomness


obs_flag = False
wall_flag = True
sep_flag = True
align_flag = True
wp_flag = False
animated = False
opt_flag = True  # flag for optimization
mp_affinity = False
warning_flag = False  # shows earnings every 5 seconds
save_data = False

# [Agent]

# physics
vmax = 6.0  # m/s
amax = 6.0  # m2/s
v_flock = 6.0  # m/s
v_target = 2.0
# sensors
gps_del = 0.2  # seconds
sigma_inner = 0.005  # m2/s2
comm_del = 1  # s
comm_radius = 80.0   # m
coll_radius = 3.0

# [environment]
# weather
weather = {'temperature': 37,
            'wind_speed': 0,
            'wind_direction': None}
sigma_outer = 0.2  # m2/s3

# [optimization]
v_tol = 1.5*v_flock/4
a_tol = 0.03
r_tol = 2


