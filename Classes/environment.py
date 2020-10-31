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
        self.order_params = np.zeros(5)
        self.t_update = (df.interval) * (1 + update_corr(df.interval))  # correction term for cpu animation lag
        self.id = id


    def add_agents(self, cls, params, seed=None):
        if seed:
            np.random.seed(seed)
        agents = list()

        for i in range(df.num_agents):
            agents.append(cls(self, params))

        for i, agent in enumerate(agents):
            agent.id = i
            agent.scp(self.gmin[0] + 2*self.gmax[1] * np.random.rand(), self.gmin[0] + 2*self.gmax[1] * np.random.rand())
            for obs in self.obstacles:
                if agent.inPolygon(obs):
                    agent.pos = np.zeros(2)
            agent.scv(-1 + 2 * np.random.rand(), -1 + 2 * np.random.rand())
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
        wait_frames = df.wait_time/df.interval
        wall_count = 0.000001  # number of cumulative collisions throughout the simulation

        if df.animated:
            self.fig, self.ax = plt.subplots()
            self.fig.set_size_inches(6, 6)
            self.fig.canvas.manager.window.wm_geometry(f"+{200 + self.id * 550}+{200}")
            self.ax.set(xlim=(self.gmin[0]-20, self.gmax[0]+20), ylim=(self.gmin[1]-20, self.gmax[1]+20), aspect='equal')
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
            wall_count = self.update(current_frame, wall_count)
            plt.pause(0.0001)
            current_frame += 1
            act_time = time.time() - start  # actual time since start of simulation

            if act_time > df.max_sim_time or (self.order_params[3] / ((df.num_agents - 1) / df.num_agents / (current_frame+0.00001-wait_frames))) > 0.01: #or (self.order_params[0] / wall_count) > 0.3:
                print(f'Env-{self.id}: Ended at at {time.ctime()[11:-5]}: simulation time= {current_frame * df.interval}  actual time taken= {act_time}')
                # plt.close()
                if wall_count:
                    self.order_params[0] /= wall_count
                self.order_params[1:4] /= (current_frame-wait_frames) * df.num_agents
                self.order_params[4] /= (current_frame-wait_frames)
                self.order_params[3] /= df.num_agents - 1
                return self.order_params

            time.sleep(df.interval - act_time % df.interval)

    def update(self, frame, count):
        wait_frames = df.wait_time/df.interval
        if df.animated:
            for agent in self.agents:
                agent.artist.set_data(agent.pos[0], agent.pos[1])
                # agent.ln.set_data(agent.pos + agent.v_d)
                # agent.ln.set_data(agent.waypoint)

        for agent in self.agents:
            agent.update(df.interval, frame)

            # calculate order parameters for optimization
            if frame > wait_frames:  # give 15 seconds to sort shit out
                self.order_params[0] += agent.phi_wall
                if not agent.phi_wall:
                    count += 1
                self.order_params[1] += agent.phi_vel
                self.order_params[2] += agent.phi_corr
                self.order_params[3] += agent.phi_coll
                self.order_params[4] += int(agent.disc)
                if frame % 50 == 0 and agent.warnings and df.warning_flag:
                    print(f'Env-{self.id}: Agent-{agent.id} warnings:', *agent.warnings, f'at {time.ctime()[11:-5]}')
        return count

    def move_obstacle(self, i, vec):
        self.obstacles[i].set_xy(self.obstacles[i].get_xy() + vec)

    def pause(self):
        self.ani.event_source.stop()

    def play(self):
        self.ani.event_source.start()

    def plot_static(self):
        '''plot static things on the axes'''
        for obs in self.obstacles:
            self.ax.add_patch(obs)
        self.ax.add_patch(self.arena)

    def change_waypoint(self, event):
        waypoint = np.array([event.xdata, event.ydata])
        for agent in self.agents:
            agent.waypoint = waypoint
        self.wp_artist.set_data(waypoint[0], waypoint[1])

