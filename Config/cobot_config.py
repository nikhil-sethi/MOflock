# CoBot configuration file

v_flock = 6

# Collaboration parameters
r0_rep = 30  # metres   # take more than double of your 'safe' distance (depends on gain + noise though)
p_rep = 0.07
r0_frict = 80   # best to keep this around comm range(do the best alignment possible under comm range)
'''Stopping point offset of alignment. The distance of the stopping point in front of
agents according to the optimal velocity reduction curve. Below this value alignment reduces all velocity difference above the given small velocity slack threshold.
Optimization tends to increase this value above intuitive levels to maximize interagent alignment in the whole communication range without spatial dependence.'''
a_frict = 9.94
'''Acceleration of braking curve. The maximal allowed acceleration in the optimal
braking curve used for determining the maximal allowed velocity difference between agents. 
Higher values assume that agents can brake quicker and thus make
aligment more local. Too high values result in the inability of agents to react to too
large velocity differences in time and thus lead to collisions.
'''
p_frict = 5.32
'''Gain of braking curve
Linear gain of the optimal braking curve used in determining the maximal allowed 
velocity difference. Note that linearity is only expressed inthe v-x plane, while the parameter has a nonlinear effect in behaviour. Large values
approximate the braking curve to the curve of constant acceleration. Small values
elongate the final part of braking (at small speeds) with decreasing acceleration and
smoother stops'''
v_frict = 0.93
'''Velocity slack of alignment. This parameter sets the velocity difference level
agents are allowed to have at all times. Having a non-zero value reduces local
overdamped dynamics a bit and thus increases the information spreading capability
within the flock during turns, which results in increased flock-level agility around
obstacles and at walls. Some velocity slack also helps eliminating roll and pitch oscillations arising in real systems due to strong alignment and the delayed response
between tilting and velocity change. Should be kept at small levels as too large
values disable alignment completely.'''
c_frict = 0.05
'''Coefficient of velocity alignment. Linear coefficient of the velocity difference error reduction in the velocity alignment term. Higher values create stronger damping between agents which helps reducing repulsion-induced oscillations but makes
motion more sluggish. Optimization tends to decrease this value below intuitive
levels.'''

# Bot parameters
r0_shill = 0.5
v_shill = 14    # higher means faster shilling
a_shill = 4  # higher means delayed but sharper shilling (a is the slope), lower is smoother
p_shill = 2  # higher means delayed and sharper and linear curve below a/p^2
charge = 95

# Particle parameters
vmax = 8 # m/s
amax = 6 # m2/s

# Sensors
gps_del = 200  # ms
sigma_inner = 0.005  # m2/s2

comm_del = 1000  # ms
comm_radius = 80  # m