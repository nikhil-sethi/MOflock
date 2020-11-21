import numpy as np
from Methods.controls import update_corr
import matplotlib.patches as p
import matplotlib.pyplot as plt
import Config.env_config as econf
import Config.defaults as df
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
        self.order_params = np.zeros(7)
        self.t_update = (df.interval) * (1 + update_corr(df.interval))  # correction term for cpu animation lag
        self.id = id
        self.wall_count = 0
        self.wait_frames = df.wait_time / df.interval

    def add_agents(self, cls, params, seed=None):
        if seed:
            np.random.seed(seed)
        agents = list()

        for i in range(df.num_agents):
            agents.append(cls(self, params))

        for i, agent in enumerate(agents):
            agent.id = i
            #spawn in bottom left grid
            x = -100 + 20 * (agent.id % 5)
            y = -100 + 20 * int(agent.id / 5)
            agent.scp(x, y)
            # agent.scp(self.gmin[0] + 2*self.gmax[1] * np.random.rand(), self.gmin[0] + 2*self.gmax[1] * np.random.rand())
            if df.obs_flag:
                for obs in self.obstacles:
                    if agent.inPolygon(obs):
                        agent.pos = np.zeros(2)
            agent.scv(1, 2)
            # agent.scv(-1 + 2 * np.random.rand(), -1 + 2 * np.random.rand())
            if df.wp_flag:
                agent.scw(0, 0)  # default waypoint. use mouseclick to change in real time
            agent.memory.append(agent.get_state())

        self.agents = agents

    def run(self, seed=None):
        if seed:
            np.random.seed(seed)
        p = psutil.Process()
        if df.mp_affinity:
            p.cpu_affinity([self.id % psutil.cpu_count()])
        print(f'Env-{self.id}: Started on cpu {p.cpu_affinity()} at {time.ctime()[11:-5]}')
        current_frame = 0

        if df.animated:
            self.fig, self.ax = plt.subplots()
            self.fig.set_size_inches(6, 6)
            self.fig.canvas.manager.window.wm_geometry(f"+{200 + self.id * 550}+{200}")
            self.ax.set(xlim=(self.gmin[0] - 20, self.gmax[0] + 20), ylim=(self.gmin[1] - 20, self.gmax[1] + 20),
                        aspect='equal')
            if df.wp_flag:
                self.wp_artist, = self.ax.plot([], [], 'bo', markersize='4')
                self.ax.figure.canvas.mpl_connect('button_release_event', self.change_waypoint)
            for agent in self.agents:
                agent.artist, = self.ax.plot(agent.pos[0], agent.pos[1], 'ro', markersize=2)
                # agent.ln, = self.ax.plot([], [], 'bo', markersize='1')
                # agent.v, = self.ax.plot([], [], 'ko', markersize='1')
            self.plot_static()

        start = time.time()
        while True:
            self.update(current_frame)
            plt.pause(0.001)
            current_frame += 1
            act_time = time.time() - start  # actual time since start of simulation

            if act_time > df.max_sim_time:  # or (self.order_params[3] / ((df.num_agents - 1) / df.num_agents / (current_frame+0.00001-self.wait_frames))) > 0.01: #or (self.order_params[0] / wall_count) > 0.3:
                print(
                    f'Env-{self.id}: Ended at at {time.ctime()[11:-5]}: simulation time= {current_frame * df.interval}  actual time taken= {act_time}')
                # plt.close()
                if self.wall_count:
                    self.order_params[0] /= self.wall_count
                self.order_params[1:4] /= (current_frame - self.wait_frames) * df.num_agents
                self.order_params[3] /= df.num_agents - 1
                self.order_params[4] /= (current_frame - self.wait_frames)
                self.order_params[5] -= 1000
                self.order_params[5] /= (current_frame - self.wait_frames)
                self.order_params[6] = current_frame * df.interval
                return self.order_params

            time.sleep(df.interval - act_time % df.interval)

    def update(self, frame):

        if df.animated:
            for agent in self.agents:
                agent.artist.set_data(agent.pos[0], agent.pos[1])
                # agent.ln.set_data(agent.pos + agent.v_d)
                # agent.ln.set_data(agent.waypoint)

        if frame > self.wait_frames:  # give 15 seconds to sort shit out
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
                n_disc += int(agent.disc)
                n_min = min(agent.cluster_count, n_min)
                if frame % 10 == 0 and agent.warnings and df.warning_flag:
                    print(f'Env-{self.id}: Agent-{agent.id} warnings:', *agent.warnings, f'at {time.ctime()[11:-5]}')
            if n_disc > 0:
                print(frame, n_disc)
            self.order_params[4] += n_disc
            self.order_params[5] += n_min
        else:
            for agent in self.agents:
                agent.update(df.interval, frame)

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
