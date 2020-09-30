'''
environment class
- arena + g
- obstacles
- GPS accuracy
- Weather
'''
import pdb
import numpy as np
from Methods.controls import create_adjMat
import matplotlib.patches as p
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Env():
    def __init__(self, econf):
        # unpack
        #self.econf = econf
        self.fig, self.ax = plt.subplots()
        gmin = np.min(econf.geofence, axis=0)    #for plotting more than geofence
        gmax = np.max(econf.geofence, axis=0)
        self.ax.set(xlim=(gmin[0], gmax[0]), ylim=(gmin[1], gmax[1]), aspect='equal')
        self.arena = p.Polygon(econf.geofence * 1, fill=False)
        self.obstacles = [p.Polygon(obs) for obs in econf.obstacles]
        self.temp = econf.weather['temperature']
        self.wind_speed, self.wind_dir = econf.weather['wind_speed'], econf.weather['wind_direction']
        self.agents = list()    # everyone on the current map
        self.interval = econf.update_interval
#        self.state = np.zeros(len(self.agents), 2)
        self.plot_static()

    def add_agents(self, agents):
        self.agents = agents
        for agent in agents:
            agent.artist, = plt.gca().plot([], [], 'ro', markersize=2)

    def start(self):

        self.ani = FuncAnimation(self.fig, self.update, interval=self.interval)


    def update(self, frame):
        #create adjacency matrix
        # this code is pretty shitty rn. Please change pliss
        pos, r_ij, vel, v_ij = self.get_state()
        for agent in self.agents:
            r_j = (r_ij[:, agent.id]).reshape(len(self.agents),1)    # all other distances
            v_j = (v_ij[:, agent.id]).reshape(len(self.agents),1)    # all other velocities

            agent.update(self.interval, pos, r_j, vel, v_j)  #this will happen on each UAV on its computer as a separate process

            if agent.animated:
                agent.artist.set_data(agent.pos[0], agent.pos[1])

    def get_state(self):
        r_ij, v_ij = create_adjMat(self.agents)
        pos = np.array([agent.pos for agent in self.agents])
        vel = np.array([agent.vel for agent in self.agents])
        return pos, r_ij, vel, v_ij

    def pause(self):
        self.ani.event_source.stop()
    def play(self):
        self.ani.event_source.start()

    def plot_static(self):
        '''plot static things on the axes'''
        for obs in self.obstacles:
            self.ax.add_patch(obs)
        self.ax.add_patch(self.arena)