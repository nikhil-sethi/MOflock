'''
environment class
- arena + g
- obstacles
- GPS accuracy
- Weather
'''
import pdb
import numpy as np
from Methods.controls import update_corr
import matplotlib.patches as p
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import Config.env_config as econf
import Config.defaults as df
from itertools import combinations as pairs


class Env:
    def __init__(self):
        # unpack
        # self.econf = econf
        plt.close()
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(7, 7)
        self.fig.canvas.manager.window.wm_geometry("+%d+%d" % (3150, 100))
        gmin = np.min(econf.geofence, axis=0)  # for plotting more than geofence
        gmax = np.max(econf.geofence, axis=0)
        self.ax.set(xlim=(gmin[0], gmax[0]), ylim=(gmin[1], gmax[1]), aspect='equal')
        self.arena = p.Polygon(econf.geofence * 1, fill=False)
        self.obstacles = [p.Polygon(obs) for obs in econf.obstacles]
        self.temp = econf.weather['temperature']
        self.wind_speed, self.wind_dir = econf.weather['wind_speed'], econf.weather['wind_direction']
        self.agents = list()  # everyone on the current map
        self.interval = econf.interval
        self.t_update = (econf.interval/ 1000) * (1 + update_corr(econf.interval))  # correction term for cpu animation lag

        self.sigma_outer = econf.sigma_outer

        #        self.state = np.zeros(len(self.agents), 2)
        self.plot_static()

    def add_agents(self, agents):
        self.agents = agents
        for agent in agents:
            agent.artist, = plt.gca().plot([], [], 'ro', markersize=2)

    def start(self):

        self.ani = FuncAnimation(self.fig, self.update, interval=self.interval)

    def update(self, frame):
        for agent in self.agents:
            if df.animated:
                agent.artist.set_data(agent.pos[0], agent.pos[1])
                # agent.ln.set_data(agent.pos + agent.v_d)

        for agent in self.agents:
            agent.update(self.interval, frame)

    def move_obstacle(self, i, vec):
        self.obstacles[i].set_xy(self.obstacles[i].get_xy()+vec)

    def pause(self):
        self.ani.event_source.stop()

    def play(self):
        self.ani.event_source.start()

    def plot_static(self):
        '''plot static things on the axes'''
        for obs in self.obstacles:
            self.ax.add_patch(obs)
        self.ax.add_patch(self.arena)
