'''
Version: 1.1
features/corrections added
- centralised control by environment
- repulsion
- alignment
- corrected acceleration calculation acc= desired - current velocity-> 2nd problem in 1.0 resolved
- added descriptions in cobot config
- normalised velocity to min(current, vmax)-> 1st problem in 1.0 resolved

problems
- centralised
- distance matrix non optimal

TODO-next version
- decentralised + comm range
- inner + outer noise
- comm delays
- maximal acceleration/ inertia

commit video: https://www.youtube.com/watch?v=xShuYQ1tOzY (Only added for major and minor commit)
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
#np.random.seed(1)
for i, agent in enumerate(agents):
    agent.id = i
    agent.scp(-40+80*np.random.rand(), -40+80*np.random.rand())

    agent.scv(-1 + 2 * np.random.rand(), -1 + 2 * np.random.rand())
    #agent.scp(0,0)
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