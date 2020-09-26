'''
Version 1.0 @ Paper_implementaion
features/corrections
- Basic architecture and file structure
- obstacle avoidance
- geofence avoidance
-
problems
- drones escape the geofence sometimes
- uneven speeds in long unobstructed patches
- untuned params
TODO
- flocking
- simulate realistic features(noise etc.)
- optimization

commit video: https://www.youtube.com/watch?v=SSjn7sUVycI
'''
from Config import defaults, cobot_config, env_config
from Classes.environment import Env
from Classes.cobot import CoBot  # a collaborative aerial bot
import numpy as np
import time
env = Env(env_config)
agents = list()

for i in range(defaults.num_agents):
    agents.append(CoBot(cobot_config, env))
np.random.seed(1)
for agent in agents:
    #agent.scp(-50+50*np.random.rand(), -50+100*np.random.rand())
    agent.scv(-1 + 2 * np.random.rand(), -1 + 2 * np.random.rand())

    agent.scp(0,0)
    #agent.scv(-1,-1)
if __name__ == '__main__':
    env.add_agents(agents)
    time.sleep(5)
    env.start()

'''
psuedocode

import libs

load env_config
load CoBot_config

create environment(env_config)
for number of UAVs
    create CoBots(CoBot_config, env) 
    CoBots.start()

'''