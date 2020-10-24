# CoBot configuration file


# Collaboration parameters
# 0
r0_rep = 41.1  # metres   # take more than double of your 'safe' distance (depends on gain + noise though)
# 1
p_rep = 0.07
# 2
r0_frict = 88.5  # best to keep this around comm range(do the best alignment possible under comm range)
'''Stopping point offset of alignment. The distance of the stopping point in front of
agents according to the optimal velocity reduction curve. Below this value alignment reduces all velocity difference above the given small velocity slack threshold.
Optimization tends to increase this value above intuitive levels to maximize interagent alignment in the whole communication range without spatial dependence.'''
# 3
a_frict = 9.94
'''Acceleration of braking curve. The maximal allowed acceleration in the optimal
braking curve used for determining the maximal allowed velocity difference between agents. 
Higher values assume that agents can brake quicker and thus make
aligment more local. Too high values result in the inability of agents to react to too
large velocity differences in time and thus lead to collisions.
'''
# 4
p_frict = 5.32
'''Gain of braking curve
Linear gain of the optimal braking curve used in determining the maximal allowed 
velocity difference. Note that linearity is only expressed inthe v-x plane, while the parameter has a nonlinear effect in behaviour. Large values
approximate the braking curve to the curve of constant acceleration. Small values
elongate the final part of braking (at small speeds) with decreasing acceleration and
smoother stops'''
# 5
v_frict = 0.93
'''Velocity slack of alignment. This parameter sets the velocity difference level
agents are allowed to have at all times. Having a non-zero value reduces local
overdamped dynamics a bit and thus increases the information spreading capability
within the flock during turns, which results in increased flock-level agility around
obstacles and at walls. Some velocity slack also helps eliminating roll and pitch oscillations arising in real systems due to strong alignment and the delayed response
between tilting and velocity change. Should be kept at small levels as too large
values disable alignment completely.'''
# 6
c_frict = 0.05
'''Coefficient of velocity alignment. Linear coefficient of the velocity difference error reduction in the velocity alignment term. Higher values create stronger damping between agents which helps reducing repulsion-induced oscillations but makes
motion more sluggish. Optimization tends to decrease this value below intuitive
levels.'''

# Bot parameters
r0_shill = 0.5
v_shill = 13 #14  # higher means faster shilling
a_shill = 5.44  # higher means delayed but sharper shilling (a is the slope), lower is smoother
p_shill = 3.32  # higher means delayed and sharper and linear curve below a/p^2

paramdict = dict(
    r0_rep=r0_rep,
    p_rep=p_rep,
    r0_frict=r0_frict,
    a_frict=a_frict,
    p_frict=p_frict,
    v_frict=v_frict,
    c_frict=c_frict,
    r0_shill=r0_shill,
    v_shill=v_shill,
    a_shill=a_shill,
    p_shill=p_shill
)

# {
# "r0_rep": r0_rep,
# "p_rep": p_rep,
# "r0_frict": r0_frict,
# "a_frict": a_frict,
# "p_frict": p_frict,
# "v_frict": v_frict,
# "c_frict": c_frict,
# "r0_shill": r0_shill,
# "v_shill": v_shill,
# "a_shill": a_shill,
# "p_shill": p_shill}
