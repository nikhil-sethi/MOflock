import numpy as np
from Classes import bot
from Methods.controls import brake_decay
from Methods.vector_algebra import unit_vector


class CoBot(bot.Bot):

    def __init__(self, conf, env):
        super().__init__(conf, env)
        # unpack configuration

    def update(self, interval, *args):
        #self.flockmates = self.env.agents(np.argwhere(d_j <self.conf.r_cluster))

        v_rep = self.separate(args[0], args[1])
        v_frict = self.align(args[2], args[3], args[1])
        #print(self.id,self.v_d, self.vel)
        self.v_d = v_rep + v_frict + self.conf.v_flock * unit_vector(self.vel)
        #print(self.id,self.v_d, self.vel,'\n')

        #print(self.vel)
        super().update(interval)

    def get_nbors(self):
        r_j = np.array([agent.pos-self.pos for agent in self.env.agents])
        return r_j

    def separate(self, r, r_j):
        temp = np.copy(r_j)  # create new reference
        temp[r_j > self.conf.r0_rep] = self.conf.r0_rep
        v_reps = self.conf.p_rep *(self.conf.r0_rep - temp) * ((self.pos-r)/temp)

        return np.nansum(v_reps, axis=0)

    def align(self, v, v_j, r_j):

        v_frict_max = brake_decay(r_j-self.conf.r0_frict, self.conf.a_frict, self.conf.p_frict)
        v_frict_max[v_frict_max < self.conf.v_frict] = self.conf.v_frict
        temp = np.copy(v_j)
        temp[v_j < v_frict_max] = np.squeeze(v_frict_max[v_j < v_frict_max])
        v_fricts = self.conf.c_frict *(temp-v_frict_max) * ((v-self.vel)/temp)

        return np.nansum(v_fricts, axis=0)

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