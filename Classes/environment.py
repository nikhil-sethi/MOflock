import numpy as np
from Methods.controls import update_corr
import matplotlib.patches as p
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import Config.env_config as econf
import Config.defaults as df
from Config.opt_config import var_lims
import time
import psutil


class Env:
    def __init__(self, id=1):
        self.gmin = np.min(econf.geofence, axis=0)  # for plotting more than geofence
        self.gmax = np.max(econf.geofence, axis=0)
        self.arena = p.Polygon(econf.geofence, fill=False)
        self.obstacles = [p.Polygon(obs) for obs in econf.obstacles]
        self.temp = df.weather['temperature']
        self.wind_speed, self.wind_dir = df.weather['wind_speed'], df.weather['wind_direction']
        self.agents = list()  # everyone on the current map
        self.order_params = np.zeros(6)
        self.t_update = (df.interval) * (1 + update_corr(df.interval))  # correction term for cpu animation lag
        self.id = id
        self.wall_count = 0
        self.wait_frames = df.wait_time / df.interval
        self.op_all = [np.zeros(6)]  # np.zeros([int(df.max_sim_time), len(self.order_params)])

    def add_agents(self, cls, params, seed=None):

        if seed:
            np.random.seed(seed)
        self.agents = list()
        self.params = params
        for _ in range(df.num_agents):
            self.agents.append(cls(self, params))

        self.place_agents('grid', gridparams=(-100, -100, 20))

    def place_agents(self, type, **kwargs):

        for i, agent in enumerate(self.agents):
            agent.id = i
            # spawn in bottom left grid
            if type == 'grid':
                x = kwargs["gridparams"][0] + kwargs["gridparams"][2] * (agent.id % 5)
                y = kwargs["gridparams"][1] + kwargs["gridparams"][2] * int(agent.id / 5)
                u = 1
                v = 2
            elif type == "random":
                x = self.gmin[0] + 2 * self.gmax[1] * np.random.rand()
                # y = self.gmin[0] + 2 * self.gmax[1] * np.random.rand()
                u = -1 + 2 * np.random.rand()
                v = -1 + 2 * np.random.rand()
            agent.scp(x, y)

            if df.obs_flag:
                for obs in self.obstacles:
                    if agent.inPolygon(obs):
                        agent.pos = np.zeros(2)
            agent.scv(u, v)

            if df.wp_flag:
                agent.scw(0, 0)  # default waypoint. use mouseclick to change in real time
            agent.memory.append(agent.get_state())

    def sl_update(self, val):
        for sl in self.sliders:
            self.params[sl.label.get_text()] = sl.val

    def run(self, seed=None):
        if seed:
            np.random.seed(seed)
        p = psutil.Process()
        if df.mp_affinity:
            p.cpu_affinity([self.id % psutil.cpu_count()])
        print(f'Env-{self.id}: Started on cpu {p.cpu_affinity()} at {time.ctime()[11:-5]}')

        if df.animated:
            self.fig, (self.ax, self.op_ax) = plt.subplots(ncols=2)
            self.fig.set_size_inches(10.82, 5.61)
            self.fig.canvas.manager.window.wm_geometry(f"+{2650}+{0}")
            self.ax.set(xlim=(self.gmin[0] - 20, self.gmax[0] + 20), ylim=(self.gmin[1] - 20, self.gmax[1] + 20),
                        aspect='equal')
            self.ax.set_position([0.089, 0.27, 0.32, 0.62])
            self.ax.margins(x=0,y=0)
            self.op_ax.margins(x=0,y=0)
            self.op_ax.set_position([0.484, 0.27, 0.415, 0.62])
            self.op_ax.set_prop_cycle(color=['r', 'g', 'b', 'y', 'c', 'k'])
            self.op_ln = self.op_ax.plot(self.op_all)
            self.op_ax.legend(['wall', 'speed', 'corr', 'coll', 'dis', 'conn'])
            if df.wp_flag:
                self.wp_artist, = self.ax.plot([], [], 'bo', markersize='4')
                self.ax.figure.canvas.mpl_connect('button_release_event', self.change_waypoint)
            for agent in self.agents:
                agent.artist, = self.ax.plot(agent.pos[0], agent.pos[1], 'ro', markersize=2)
                # agent.ln, = self.ax.plot([], [], 'bo', markersize='1')
                # agent.v, = self.ax.plot([], [], 'ko', markersize='1')
            self.sliders = [Slider(
                plt.axes([0.05 + (i % 4) * (0.043 + 0.8 / 4), 0.15 - int(i / 4) * (0.015 + 0.03), 0.15, 0.03],
                         facecolor='lightgoldenrodyellow'), key, var_lims[key][0], var_lims[key][1],
                valinit=self.params[key]) for i, key in enumerate(self.params)]
            for sl in self.sliders:
                sl.on_changed(self.sl_update)
            self.plot_static()

        # MAIN LOOP
        current_frame = 0
        start = time.time()
        act_time = 0

        # wait loop
        while current_frame < self.wait_frames:
            current_frame += 1
            for agent in self.agents:
                agent.update(df.interval, current_frame)
                if df.animated:
                    agent.artist.set_data(agent.pos[0], agent.pos[1])
            plt.pause(0.001)
            act_time = time.time() - start  # actual time since start of simulation
            time.sleep(df.interval - act_time % df.interval)

        # performance loop
        while act_time < df.max_sim_time:
            current_frame += 1
            self.update(current_frame)
            plt.pause(0.001)
            act_time = time.time() - start  # actual time since start of simulation
            time.sleep(df.interval - act_time % df.interval)

            # if :  # or (self.order_params[3] / ((df.num_agents - 1) / df.num_agents / (current_frame+0.00001-self.wait_frames))) > 0.01: #or (self.order_params[0] / wall_count) > 0.3:
        # POST PROCESS
        print(
            f'Env-{self.id}: Ended at at {time.ctime()[11:-5]}: simulation time= {current_frame * df.interval} actual time taken= {act_time}')
        # plt.close()
        self.op_all = np.array(self.op_all)
        # if self.wall_count:
        #     self.op_all[:, 0] /= self.wall_count
        #
        # all_times = np.append(np.ones(df.wait_time),
        #                       np.arange(df.interval * 100, (len(self.op_all) - df.wait_time + 1) * 10, 10))
        # self.op_all[:, 1:4] = self.op_all[:, 1:4] / (all_times[:, np.newaxis] * df.num_agents)
        # self.op_all[:, 3] /= df.num_agents - 1
        # self.op_all[:, 4] = self.op_all[:, 4] / all_times
        # self.op_all[:, 5] = self.op_all[: 5] / all_times
        # # self.op_all[:,6] = current_frame * df.interval
        return self.op_all

    def update(self, frame):

        n_min = 1000  # just a random large number
        n_disc = 0
        for agent in self.agents:
            agent.update(df.interval, frame)
            # calculate order parameters for optimization
            self.order_params[0] += agent.phi_wall
            self.wall_count += agent.outside
            self.order_params[1] += agent.phi_vel
            self.order_params[2] += agent.phi_corr
            self.order_params[3] += agent.phi_coll
            n_disc += agent.disc
            n_min = min(agent.cluster_count, n_min)
            if df.animated:
                agent.artist.set_data(agent.pos[0], agent.pos[1])
                # agent.ln.set_data(agent.pos + agent.v_d)
                # agent.ln.set_data(agent.waypoint)
            if df.warning_flag:
                if agent.warnings:
                    if frame % 10 == 0:
                        print(f'Env-{self.id}: Agent-{agent.id} warnings:', *agent.warnings,
                              f'at {time.ctime()[11:-5]}')

        self.order_params[4] += n_disc
        self.order_params[5] += n_min

        # else:
        #     for agent in self.agents:
        #         agent.update(df.interval, frame)

        if frame % (1 / df.interval) == 0:
            temp = self.order_params.copy()
            if self.wall_count:
                temp[0] /= self.wall_count
            temp[1:4] /= ((frame - self.wait_frames + 0.0001) * df.num_agents)
            temp[3] /= df.num_agents - 1
            temp[4] /= (frame - self.wait_frames + 0.0001)
            temp[5] /= (frame - self.wait_frames + 0.0001)
            self.op_all.append(temp)

            self.op_ax.lines = self.op_ax.plot(self.op_all)

    def move_obstacle(self, i, vec):
        self.obstacles[i].set_xy(self.obstacles[i].get_xy() + vec)

    def pause(self):
        self.ani.event_source.stop()

    def play(self):
        self.ani.event_source.start()

    def plot_static(self):
        '''plot static things on the axes'''
        if df.obs_flag:
            for obs in self.obstacles:
                self.ax.add_patch(obs)
        self.ax.add_patch(self.arena)

    def change_waypoint(self, event):
        waypoint = np.array([event.xdata, event.ydata])
        for agent in self.agents:
            agent.waypoint = waypoint
        self.wp_artist.set_data(waypoint[0], waypoint[1])
