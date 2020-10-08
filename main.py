'''
Version: 2.0
features/corrections added

- Completely decentralised(at least i think so) and noised.
- Fixed the corner problem by adding a repulsive corner vector
- Added + tested some avoidance methods
- Much cleaner than before
- Added some flags in default config to toggle features in real time


problems
- A lot of stuff might not be optimized for least complexity
- API could be better in cobot


TODO-next version
- optimization

commit video: https://www.youtube.com/watch?v=6dMFWlXTj9U
'''

from Config import defaults, cobot_config, env_config   # import all configs even if not used to change real time params
from Classes.environment import Env
from Classes.cobot import CoBot  # a collaborative aerial bot
import numpy as np
import time
env = Env()
agents = list()

for i in range(defaults.num_agents):
    agents.append(CoBot(env))
#np.random.seed(1)
for i, agent in enumerate(agents):
    agent.id = i
    agent.scp(-100+200*np.random.rand(), -100+200*np.random.rand())
    agent.scv(-1 + 2 * np.random.rand(), -1 + 2 * np.random.rand())
    agent.memory.append(agent.get_state())

    #agent.scp(0,0)
    #agent.scv(-1,-1)
# agents[0].scp(0,0)
# agents[1].scp(-20,0)
# agents[0].scv(1,1)
# agents[1].scv(1,0)
# agents[0].memory.append(agents[0].get_state())
# agents[1].memory.append(agents[1].get_state())
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