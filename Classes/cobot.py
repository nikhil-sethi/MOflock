import numpy as np
from Classes import bot
from Methods.controls import brake_decay, add_innernoise, add_outernoise
from Methods.vector_algebra import unit_vector
import Config.cobot_config as conf
import Config.defaults as df

class CoBot(bot.Bot):

    def __init__(self, env):
        super().__init__(env)
        self.memory = []
        # unpack configuration

    def update(self, step, frame):

        if frame % (conf.gps_del / step) == 0:
            add_innernoise(self.gps_pos, self.gps_vel, conf.sigma_inner,
                           conf.gps_del / 1000)

        self.nbors = self.get_nbors()
        # print(self.id, len(self.nbors))
        if frame >= (conf.comm_del / step):
            self.memory = self.memory[1:]

        if frame % (conf.gps_del / step) == 0:
            self.v_d = np.zeros(2)
            super().calcDesiredVelocity()
            if any(self.nbors):
                rji, rji_mag, vji, vji_mag = self.get_noisedState()
                self.calcDesiredVelocity(rji, rji_mag, vji, vji_mag)

        # print(self.id, self.v_d)

        super().update(step/1000, frame)

        self.memory.append(self.get_state())

    def calcDesiredVelocity(self, *args):
        v_rep = 0
        v_frict = 0
        if df.sep_flag:
            v_rep = self.separate(args[0], args[1])
        if df.align_flag:
            v_frict = self.align(args[2], args[3], args[1])
        # print(self.id, v_rep, v_frict)
        # pr int(self.id,self.v_d, self.vel)
        self.v_d += v_rep + v_frict


    def get_nbors(self):
        rji = np.array([agent.pos for agent in self.env.agents]) - self.pos
        rji_mag = np.sqrt(rji[:, 0] ** 2 + rji[:, 1] ** 2)
        nbors_bool = (np.arange(df.num_agents) != self.id) & (rji_mag < conf.comm_radius)
        return np.array(self.env.agents)[nbors_bool]

    def get_noisedState(self):
        rji = np.array([agent.memory[0][0] + agent.gps_pos for agent in self.nbors]) - self.pos - self.gps_pos
        rji_mag = np.sqrt(rji[:, 0] ** 2 + rji[:, 1] ** 2)

        vji = np.array([agent.memory[0][1] + agent.gps_vel for agent in self.nbors]) - self.vel - self.gps_vel
        vji_mag = np.sqrt(vji[:, 0] ** 2 + vji[:, 1] ** 2)

        return rji, rji_mag.reshape(len(self.nbors), 1), vji, vji_mag.reshape(len(self.nbors),1)

    def separate(self, rji, rji_mag):
        temp = np.copy(rji_mag)  # create new reference
        temp[rji_mag > conf.r0_rep] = conf.r0_rep
        v_reps = conf.p_rep * (conf.r0_rep - temp) * (-rji / temp)

        return np.sum(v_reps, axis=0)

    def align(self, vji, vji_mag, rji_mag):

        v_frict_max = brake_decay(rji_mag - conf.r0_frict, conf.a_frict, conf.p_frict)
        v_frict_max[v_frict_max < conf.v_frict] = conf.v_frict
        temp = np.copy(vji_mag)
        temp[vji_mag < v_frict_max] = np.squeeze(v_frict_max[vji_mag < v_frict_max])
        v_fricts = conf.c_frict * (temp - v_frict_max) * (vji / temp)

        return np.sum(v_fricts, axis=0)

    '''
pseudocode
def update(self):
    for other in self.others :
        if other.id <self.id
            continue
        get distance from self
        if dist< cluster dist
            add to own flockmates
    
    v_rep = self.separate(self.flockmates)
    v_frict = self.align(self.flockmates)
    
    v_d += v_rep + v_frict
    self.acc += v_flock * unit_vector(v_d) + v_d


def separate():
    
'''
