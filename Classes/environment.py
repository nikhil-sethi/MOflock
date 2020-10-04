'''
environment class
- arena + g
- obstacles
- GPS accuracy
- Weather
'''
import pdb
import numpy as np
from Methods.controls import create_adjMat, add_innernoise, add_outernoise, update_corr
import matplotlib.patches as p
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Methods.vector_algebra import unit_vector,norm, absheading_from_vec

from itertools import combinations as pairs


class Env():
    def __init__(self, econf):
        # unpack
        # self.econf = econf
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(4.8, 4.8)
        #self.fig.canvas.manager.window.wm_geometry("+%d+%d" % (500, 500))
        gmin = np.min(econf.geofence, axis=0)  # for plotting more than geofence
        gmax = np.max(econf.geofence, axis=0)
        self.ax.set(xlim=(gmin[0], gmax[0]), ylim=(gmin[1], gmax[1]), aspect='equal')
        self.arena = p.Polygon(econf.geofence * 1, fill=False)
        self.obstacles = [p.Polygon(obs) for obs in econf.obstacles]
        self.temp = econf.weather['temperature']
        self.wind_speed, self.wind_dir = econf.weather['wind_speed'], econf.weather['wind_direction']
        self.agents = list()  # everyone on the current map
        self.t_update = econf.update_interval
        self.t_gps = econf.gps_refresh
        self.t_del = econf.comm_refresh
        self.sigma_inner = econf.sigma_inner
        self.sigma_outer = econf.sigma_outer

        self.memory = []
        #        self.state = np.zeros(len(self.agents), 2)
        self.plot_static()

    def add_agents(self, agents):
        self.agents = agents
        for agent in agents:
            agent.artist, = plt.gca().plot([], [], 'ro', markersize=2)

    def start(self):

        self.ani = FuncAnimation(self.fig, self.update, interval=self.t_update)

    def update(self, frame):
        step = (self.t_update / 1000) * (1 + update_corr(self.t_update))  # correction term for cpu animation lag
        for agent in self.agents:
            agent.pos += agent.vel * step
            if agent.animated:
                agent.artist.set_data(agent.pos[0], agent.pos[1])
                #agent.ln.set_data(agent.pos + agent.v_d)

        gpos = []
        gvel = []
        for agent in self.agents:
            if frame % (self.t_gps / self.t_update) == 0:
                add_innernoise(agent.gps_pos, agent.gps_vel, self.sigma_inner,
                               self.t_gps / 1000)  # add gps noise to delayed distances and velocities
                agent.gps_pos
            gpos.append(agent.gps_pos)
            gvel.append(agent.gps_vel)
        # create adjacency matrix
        # this code is pretty shitty rn. Please change pliss
        state = self.get_state()
        if frame >= (self.t_del / self.t_update):
            self.memory = self.memory[1:]
        self.memory.append(state)
        pos_del, vel_del = self.memory[0]  # t_del older state wrt current fram


        for agent in self.agents:
            d_j = norm(agent.pos- pos_del + agent.gps_pos - gpos, axis=1)   # all other delayed +gps error distances

            a_j = norm(agent.vel - vel_del + agent.gps_vel - gvel, axis=1)

            nbors = (np.arange(len(self.agents)) != agent.id) & (d_j < 80)
            d_j = (d_j[nbors]).reshape(len(nbors[nbors]), 1)  # delayed distances from neighbours
            a_j = (a_j[nbors]).reshape(len(nbors[nbors]), 1)  # delayed vel differences from neighbours
            r_j = (pos_del +gpos)[nbors]  # delayed + gps error positions of neighbours
            v_j = (vel_del +gvel)[nbors]  # delayed + gps error velocities of neighbours


            if frame % (self.t_gps / self.t_update) == 0:
                agent.update(step, r_j, d_j, v_j, a_j)  # this will happen on each UAV on its computer as a separate process
            agent.acc += agent.v_d - agent.vel - agent.gps_vel # deltaV

            agent.vel +=  unit_vector(agent.acc) * min(norm(agent.acc), agent.amax) * step
            #print(agent.vel,agent.acc)
            agent.vel = unit_vector(agent.vel) * min(norm(agent.vel), agent.vmax)
            # agent.ln.set_data(agent.pos + agent.v_d)
            #print(agent.v_d,agent.vel)# self.pos += self.vel * step
            agent.acc = np.zeros(2)
            add_outernoise(agent.vel, self.sigma_outer, self.t_update / 1000)

    def get_state(self):

        pos = np.array([agent.pos for agent in self.agents])
        vel = np.array([agent.vel for agent in self.agents])
        return pos,vel

    def pause(self):
        self.ani.event_source.stop()

    def play(self):
        self.ani.event_source.start()

    def plot_static(self):
        '''plot static things on the axes'''
        for obs in self.obstacles:
            self.ax.add_patch(obs)
        self.ax.add_patch(self.arena)
