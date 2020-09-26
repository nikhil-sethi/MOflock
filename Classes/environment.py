'''
environment class
- arena + g
- obstacles
- GPS accuracy
- Weather
'''
import pdb
import numpy as np
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
        self.arena = p.Polygon(econf.geofence * 0.8, fill=False)
        self.obstacles = [p.Polygon(obs) for obs in econf.obstacles]
        self.temp = econf.weather['temperature']
        self.wind_speed, self.wind_dir = econf.weather['wind_speed'], econf.weather['wind_direction']
        self.agents = list()    # everyone on the current map
        self.interval = econf.update_interval

        self.plot_static()

    def add_agents(self, agents):
        self.agents = agents
        for agent in agents:
            agent.artist, = plt.gca().plot([], [], 'ro', markersize=2)

    def start(self):

        self.ani = FuncAnimation(self.fig, self.update, interval=self.interval)


    def update(self, frame):
        for agent in self.agents:
            agent.update(self.interval)  #this will happen on each UAV on its computer as a separate process
            if agent.animated:
                agent.artist.set_data(agent.pos[0], agent.pos[1])

    def pause(self):
        self.ani.event_source.stop()
    def play(self):
        self.ani.event_source.start()

    def plot_static(self):
        '''plot static things on the axes'''
        for obs in self.obstacles:
            self.ax.add_patch(obs)
        self.ax.add_patch(self.arena)