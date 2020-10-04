'''
Version: 1.2
features/corrections added
- GPS(inner noise)
- Wind (outer noise)
- added comm range in a dirty manner
- added comm delays by making a memory
- added maximum acceleration/ inertia
- changed a lot of design with the update functions. Almost all centralised

problems
- centralised
- some shady behaviour on corners. Repulsion doesnt work around corners.
- very dirty. needs to be changed so committed anyway

TODO-next version
- decentralised
- cleaned up

commit video: none. sorry
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
    agent.scp(-100+200*np.random.rand(), -100+200*np.random.rand())

    agent.scv(-1 + 2 * np.random.rand(), -1 + 2 * np.random.rand())
    #agent.scp(0,0)
    #agent.scv(-1,-1)
if __name__ == '__main__':
    env.add_agents(agents)
    #time.sleep(5)
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