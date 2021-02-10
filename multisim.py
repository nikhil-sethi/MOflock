import Config.defaults as df
import Config.cobot_config as cf
from multiprocessing import Queue
import multiprocessing as mp
import numpy as np
import pygame
from Config.env_config import geofence
import time
import os
from Utils.opt_utils import setup_envs, eval_envs_parallel
from Utils.transfer_funcs import optofitness

WHITE = np.array([255, 255, 255])
GRAY = WHITE - 50
BLACK = WHITE - 255
RED = np.array([255, 0, 0])
BLUE = np.array([0, 0, 255])

REDp = np.clip(RED-50,0,None)
REDpp = np.clip(RED-80,0,None)
REDppp = np.clip(RED-105,0,None)
REDpppp= np.clip(RED-130,0,None)
REDm = np.clip(RED+50,None,255)
REDmm = np.clip(RED+80,None,255)
REDmmm = np.clip(RED+135,None,255)
REDmmmm= np.clip(RED+200,None,255)

BLUEp = np.clip(BLUE - 50,0,None)
BLUEpp = np.clip(BLUE - 100,0,None)
BLUEm = np.clip(BLUE + 50,None,255)
BLUEmm = np.clip(BLUE + 100,None,255)
BLUEmmm = np.clip(BLUE+135,None,255)
BLUEmmmm= np.clip(BLUE+200,None,255)

m = mp.Manager()
# q1 = mp.SimpleQueue()
# q2 = mp.SimpleQueue()
qs = [m.Queue() for _ in range(2)]
color = [RED, BLUE]
colormm = [REDmm, BLUEmm]
colorm = [REDm, BLUEm]
colorp = [REDp, BLUEp]
colorpp = [REDpp, BLUEpp]
# colorppp = [REDppp, BLUEppp]
# colorpppp = [REDpppp, BLUEpppp]
colormmm = [REDmmm, BLUEmmm]
colormmmm = [REDmmmm, BLUEmmmm]
# p.close()

clock = pygame.time.Clock()

gmax = geofence.max()
bound_rect = (
    df.pg_scale * df.bound_tol, df.pg_scale * df.bound_tol,
    *(df.pg_scale * (geofence.max(axis=0) - geofence.min(axis=0))))


def run_game(q):
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (2500, 0)
    surface = pygame.display.set_mode((df.pg_scale * (geofence.max(axis=0) - geofence.min(axis=0) + 2 * df.bound_tol)))

    start = time.time()
    act_time = 0
    max_actual_time = df.max_sim_time / df.speedup
    actual_wait_time = df.wait_time / df.speedup
    speedup = df.speedup
    interval = df.interval
    current_frame = 0
    # wait loop
    transform = df.pg_scale * np.array([[1, 0], [0, -1]])
    while act_time < max_actual_time:
        # if current_frame % 5 == 0:
        # surface.fill((0, 0, 0))
        for e in pygame.event.get():
            if e.type==pygame.MOUSEBUTTONUP:
                pygame.image.save(surface, f"screenshot{act_time}.png")
        for _ in range(speedup):
            agents = []
            surface.fill(WHITE)
            pygame.draw.rect(surface, (170, 170, 170), bound_rect, 1)
            current_frame += 1
            try:
                agents.extend(q[0].get())
                agents.extend(q[1].get())
            except:
                break
            # q[0].task_done()
            # q[1].task_done()
            # agents = q.get()
            for agent in agents:
                eid = agent.env.id
                pygame.draw.circle(surface, tuple(color[eid]),
                                   (transform @ agent.pos + df.pg_scale * (gmax + df.bound_tol)).astype('int'),
                                   4)
                # surface.blit(self.drone_img,tuple(self.transform @ agent.pos + df.pg_scale * (self.gmax[0] + df.bound_tol)))
                # surface.blit(self.drone_ghost_img,
                #                  tuple(self.transform @ agent.memory[0][0] + df.pg_scale * (self.gmax[0] + df.bound_tol)))

                try:
                    pygame.draw.circle(surface, tuple(colorm[eid]),
                                       (transform @ agent.memory[-5][0] + df.pg_scale * (gmax + df.bound_tol)).astype(
                                           'int'), 3)
                    pygame.draw.circle(surface, tuple(colormm[eid]),
                                       (transform @ agent.memory[-10][0] + df.pg_scale * (gmax + df.bound_tol)).astype(
                                           'int'),
                                       3)
                    pygame.draw.circle(surface, tuple(colormm[eid]),
                                       (transform @ agent.memory[-15][0] + df.pg_scale * (gmax + df.bound_tol)).astype(
                                           'int'),
                                       2)
                    pygame.draw.circle(surface, tuple(colormmm[eid]),
                                       (transform @ agent.memory[-20][0] + df.pg_scale * (gmax + df.bound_tol)).astype(
                                           'int'),
                                       2)
                    pygame.draw.circle(surface, tuple(colormmmm[eid]),
                                       (transform @ agent.memory[-25][0] + df.pg_scale * (gmax + df.bound_tol)).astype(
                                           'int'),
                                       2)
                    pygame.draw.circle(surface, tuple(colormmmm[eid]),
                                       (transform @ agent.memory[-30][0] + df.pg_scale * (gmax + df.bound_tol)).astype(
                                           'int'),
                                       1)
                    pygame.draw.circle(surface, tuple(colormmmm[eid]),
                                       (transform @ agent.memory[-35][0] + df.pg_scale * (gmax + df.bound_tol)).astype(
                                           'int'),
                                       1)
                    pygame.draw.circle(surface, tuple(colormmmm[eid]),
                                       (transform @ agent.memory[-40][0] + df.pg_scale * (gmax + df.bound_tol)).astype(
                                           'int'),
                                       1)
                    pygame.draw.circle(surface, tuple(colormmmm[eid]),
                                       (transform @ agent.memory[-45][0] + df.pg_scale * (gmax + df.bound_tol)).astype(
                                           'int'),
                                       1)
                    pygame.draw.circle(surface, tuple(colormmmm[eid]),
                                       (transform @ agent.memory[-50][0] + df.pg_scale * (gmax + df.bound_tol)).astype(
                                           'int'),
                                       1)
                except IndexError:
                    pass
            pygame.display.flip()
        act_time = time.time() - start  # actual time since start of simulation
        clock.tick(1 / interval)

    print(act_time)


import threading

# pop = np.array([list(cf.paramdict.values())
pop = np.array([[3.13791389e+01, 5.88404323e-02, 6.19661687e+01,
                 6.94702705e+00, 1.81474582e+00, 3.30122333e-01,
                 1.77654583e-01, -1.94089572e+00, 1.29068772e+01,
                 3.54015072e+00, 5.43638404e+00, 3.48881247e-01],
                [3.26168002e+01, 2.25248339e-02, 5.91091630e+01,
                 8.18629996e+00, 3.30108761e+00, 1.66213931e+00,
                 3.72514451e-02, -6.58828690e+00, 1.29394850e+01,
                 3.54767023e+00, 5.43828556e+00, 7.41399626e-01]])
seed = 264
envs = setup_envs(pop, seed, qs)
t = threading.Thread(target=run_game, args=(qs,))
t.start()

# jobs = []
# for env in envs:
#     p = mp.Process(target=env.run, args=(qs[env.id],))
#     # p = threading.Thread(target=env.run, args=(qs[env.id],))
#     jobs.append(p)
#
# for p in jobs:
#     p.start()
#     # p.join() 264

op_all = eval_envs_parallel(envs, df.chunksize)
# q.join()
t.join()
fits = [optofitness(i[0][-1], n_obj='all') for i in op_all]
print("Fitness =", fits)

'''
ax1.plot(-0.6799131249859199,-0.8618850204, 'g*')
ax1.plot(-0.87505968,-0.14104066, 'g*')
ax1.plot(-0.89083400,-0.774698, 'g*')
ax1.plot(-0.93177,-0.933657, 'g*')
ax1.plot(-0.99414442* 0.851038,-0.91072122*0.77119648*0.94300563*0.81817796, 'g*')
ax.plot(-0.8012,-0.73827, 'b*')
ax.plot(-0.8014,-0.699752, 'r*')
'''
