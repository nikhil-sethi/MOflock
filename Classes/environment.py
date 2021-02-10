import numpy as np
from Utils.controls import update_corr
import matplotlib.patches as p
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import Config.env_config as econf
import Config.defaults as df
from Config.opt_config import var_lims
import time
import psutil
import pygame
import os
from sklearn.preprocessing import minmax_scale

class Env:
    _id = 0
    def __init__(self,params, seed=None, q=None):
        self.id = Env._id
        Env._id += 1
        self.out_queue=q
        self.gmin = np.min(econf.geofence, axis=0)  # for plotting more than geofence
        self.gmax = np.max(econf.geofence, axis=0)
        self.arena = p.Polygon(econf.geofence, fill=False)
        self.obstacles = [p.Polygon(obs) for obs in econf.obstacles]
        self.temp = df.weather['temperature']
        self.wind_speed, self.wind_dir = df.weather['wind_speed'], df.weather['wind_direction']
        self.agents = list()  # everyone on the current map
        self.op_cum = np.zeros(7)
        self.t_update = (df.interval) * (1 + update_corr(df.interval))  # correction term for cpu animation lag
        self.wc_cum = 0
        self.wait_frames = df.wait_time / df.interval
        self.op_all_cum = []  # np.zeros([int(df.max_sim_time), len(self.op_cum)])
        self.op_all_curr = []
        self.op_ranges = np.array([df.r_tol, df.v_flock, 1, df.a_tol, df.num_agents / 5, df.num_agents])
        self.map_layer = np.zeros(2*self.gmax)
        self.params=params
        if seed is None:
            seed = np.random.randint(0, 10000)
        elif seed == 'id':
            seed = self.id
        np.random.seed(seed)
        self.seed = seed
        # start out thread
        # import threading
        # threading.Thread(target=self.thread_out).start()

    def thread_out(self):
        while True:
            for _ in range(df.speedup):
                self.out_queue.put(self.agents)
            time.sleep(df.interval)

    def add_agents(self, cls, seed=None):
        self.agents = []
        self.clusters = [[] for _ in range(df.n_clusters)]
        for i in range(df.num_agents):
            agent = cls(self)
            agent.id = i
            # spawn in bottom left grid
            if df.start_type == 'grid':
                x = np.array(df.start_loc[0]) + df.start_sep * (agent.id % 5)
                y = np.array(df.start_loc[1]) + df.start_sep * int(agent.id / 5)

            elif df.start_type == "random":
                x = self.gmin[0] + 2 * self.gmax[1] * np.random.rand()
                y = self.gmin[0] + 2 * self.gmax[1] * np.random.rand()
                u = -1 + 2 * np.random.rand()
                v = -1 + 2 * np.random.rand()
            if df.obs_flag:
                for obs in self.obstacles:
                    if agent.inPolygon(obs):
                        agent.pos = np.zeros(2)

            if df.wp_flag:
                agent.scw(-220, -220)  # default waypoint. use mouseclick to change in real time
                if self.id==1:
                    agent.scw(220, 220)
            agent.scp(x, y)
            self.clusters[i % df.n_clusters].append(agent)

        for cluster in self.clusters:
            if df.start_type =='grid':
                u,v = 1,0
            color = np.random.randint(40, 205, size=(3))
            for agent in cluster:
                agent.cluster = cluster
                # print(agent.pos)
                agent.scv(u, v)
                agent.color = color
                agent.memory.append(agent.get_state())
                self.agents.append(agent)

    def assign_cluster(self):
        labels =KMeans(n_clusters=df.n_clusters).fit([agent.pos for agent in self.agents]).labels_
        print(labels)
        self.clusters=[]
        for i in range(df.n_clusters):
            self.clusters.append(np.array(self.agents)[np.argwhere(labels==i).flatten()])

        for cluster in self.clusters:
            cluster_color = np.random.randint(0,255, size=(3))
            print(cluster)
            for agent in cluster:
                agent.cluster = cluster
                agent.color = cluster_color

    def sl_update(self, val):
        for sl in self.sliders:
            self.params[sl.label.get_text()] = sl.val

    def run(self,q=None):


        p = psutil.Process()

        if df.mp_affinity:
            p.cpu_affinity([self.id % psutil.cpu_count()])
        print(f'Env-{self.id}: Started on cpu {p.cpu_affinity()} at {time.ctime()[11:-5]} with seed {self.seed}')

        if df.animated:
            pygame.init()
            # pygame.display.set
            os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0+self.id*(df.pg_scale * (self.gmax[0] - self.gmin[0] + 2 * df.bound_tol)), 0)
            # self.window = window
            self.window = pygame.display.set_mode((df.pg_scale * (self.gmax - self.gmin + 2 * df.bound_tol)))

            # self.dispay_surf=pygame.display.set_mode((1000,1000), flags=pygame.RESIZABLE)#|pygame.OPENGL)
            self.bound_rect = (df.pg_scale * df.bound_tol, df.pg_scale * df.bound_tol, *(df.pg_scale * (self.gmax - self.gmin)))
            self.drone_img = pygame.image.load('drone_15.png')
            self.drone_ghost_img = pygame.image.load('drone_15_ghost.png')
            pygame.display.set_caption(f"Environment {self.id}")

        # MAIN LOOP
        current_frame = 0
        start = time.time()
        act_time = 0
        max_actual_time = df.max_sim_time/df.speedup
        actual_wait_time = df.wait_time/df.speedup
        speedup = df.speedup
        interval = df.interval
        clock = pygame.time.Clock()
        # wait loop
        self.transform = df.pg_scale * np.array([[1, 0], [0, -1]])
        while act_time < actual_wait_time:
            for _ in range(speedup):
                current_frame += 1
                self.update(interval, current_frame)
                if df.animated:
                    self.render(self.window)
            act_time = time.time() - start  # actual time since start of simulation
            clock.tick(1 / interval)
        # performance loop
        while act_time < max_actual_time:
            for _ in range(speedup):
                current_frame += 1
                self.update(interval, current_frame)
                self.calcOrderParams(interval, current_frame)
                try:
                    q.put(self.agents)
                except:
                    pass
                # q.close()
                # q.join_thread()
                if df.animated:
                    self.render(self.window)
                # time.sleep(0.00001)
            act_time = time.time() - start  # actual time since start of simulation
            # print(p.cpu_percent())
            clock.tick(1 / interval)

        # POST PROCESS
        print(
            f'Env-{self.id}: Ended at at {time.ctime()[11:-5]}: simulation time= {current_frame * df.interval} actual time taken= {act_time}')
        self.op_all_cum = np.array(self.op_all_cum)
        self.op_all_curr = np.array(self.op_all_curr)
        all_times = np.arange(1 / df.interval, (len(self.op_all_cum) + 1) * (1 / df.interval), 1 / df.interval)

        self.op_all_cum[:, 1:4] = self.op_all_cum[:, 1:4] / (all_times[:, np.newaxis] * df.num_agents)
        self.op_all_cum[:, 3] /= (df.num_agents - 1)*2
        self.op_all_cum[:, 4] = self.op_all_cum[:, 4] / all_times
        self.op_all_cum[:, 5] = self.op_all_cum[:, 5] / all_times
        self.op_all_cum[:, 6] = self.op_all_cum[:, 6] / all_times*df.num_agents
        # print(f"Env-{self.id} op=",self.op_all_cum[-1])

        return self.op_all_cum, self.op_all_curr

    def update(self, interval, frame):
        # wc_curr = 0
        # op_curr = np.array([0, 0, 0, 0, 0, 1000.])
        for agent in self.agents:
            agent.update(interval, frame)

        for agent in self.agents:
            agent.calcDesiredVelocity(interval, frame)

    def calcOrderParams(self, interval, frame):
        wc_curr = 0
        op_curr = np.array([0, 0, 0, 0, 0, 1000.,0])
        for agent in self.agents:
            op_curr[0] += agent.phi_wall
            wc_curr += agent.outside
            op_curr[1] += agent.phi_vel
            op_curr[2] += agent.phi_corr
            op_curr[3] += agent.phi_coll
            op_curr[4] += agent.disc
            op_curr[5] = min(agent.cluster_count, op_curr[5])
            op_curr[6] += agent.phi_target
            if df.warning_flag:
                if agent.warnings:
                    if frame % 10 == 0:
                      print(f'Env-{self.id}: Agent-{agent.id} warnings:', *agent.warnings,
                              f'at {time.ctime()[11:-5]}')

        self.op_cum += op_curr
        self.wc_cum += wc_curr

        if frame % (1 / interval) == 0:
            n = df.num_agents
            temp = self.op_cum.copy()
            op_curr /= np.array([wc_curr + 0.00001, n, n, n * (n - 1)*2, 1., 1., n])   # 2 is here because of the decentralised nature
            temp[0] /= (self.wc_cum + 0.00001)  # this` is done here to avoid storing all wall counts in a list
            # print(op_curr[-1])
            self.op_all_cum.append(temp)
            self.op_all_curr.append(op_curr)
            # print("\n")
            # self.op_ax.lines = self.op_ax.plot(self.op_all_cum/self.op_ranges)

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

    def render(self, surface):
        e=pygame.event.get()
        surface.fill((230, 230, 230))
        pygame.draw.rect(surface, (100, 100, 100), self.bound_rect, 1)
        for agent in self.agents:
            pygame.draw.circle(surface,(255//(self.id+1), 0, 0),
                               (self.transform @ agent.pos + df.pg_scale * (self.gmax[0] + df.bound_tol)).astype('int'), 3)
            # surface.blit(self.drone_img,tuple(self.transform @ agent.pos + df.pg_scale * (self.gmax[0] + df.bound_tol)))
            # surface.blit(self.drone_ghost_img,
            #                  tuple(self.transform @ agent.memory[0][0] + df.pg_scale * (self.gmax[0] + df.bound_tol)))
            try:
                pygame.draw.circle(surface, (80, 80, 80),
                                   self.transform @ agent.memory[5][0] + df.pg_scale * (self.gmax[0] + df.bound_tol), 2)
                pygame.draw.circle(surface, (40, 40, 40), self.transform @ agent.memory[0][0] + df.pg_scale * (self.gmax[0] + df.bound_tol), 1 )
            except:
                pass
            # pygame.draw.circle(surface=surface,color=(255,255,255), center=self.transform @ agent.pos+3*agent.gps_pos + df.pg_scale * (self.gmax[0] + df.bound_tol), radius=20, width=2)
        # pygame.transform.scale(surface, (1000,1000), self.dispay_surf)
        pygame.display.update()
