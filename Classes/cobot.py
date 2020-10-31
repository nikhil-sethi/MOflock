import numpy as np
from Classes import bot
from Methods.controls import brake_decay, add_innernoise, packet_lost
from Methods.vector_algebra import norm
import Config.defaults as df


class CoBot(bot.Bot):

    def __init__(self, env, paramdict):
        self.memory = []
        self.phi_coll = 0
        self.phi_corr = 0
        self.phi_vel = 0
        super().__init__(env, paramdict)
        self.r_cluster = max(self.conf["r0_rep"], self.conf["r0_frict"] + float(brake_decay(df.v_flock, self.conf["a_frict"], self.conf["p_frict"])))

    def update(self, step, frame):
        self.warnings = []
        if frame % (df.gps_del / step) == 0:
            add_innernoise(self.gps_pos, self.gps_vel, df.sigma_inner,
                           df.gps_del)

        self.nbors = self.get_nbors()
        if not self.nbors:
            self.disc = True  # used for the disconnection order parameter
            self.warnings.append("Disconnected,  ")
            # print(f'Env-{self.env.id}: Agent {self.id} disconnected')
        else:
            self.disc = False

        # print(self.id, len(self.nbors))
        if frame >= (df.comm_del / step):
            self.memory = self.memory[1:]

        if frame % (df.gps_del / step) == 0:
            self.v_d = np.zeros(2)
            super().calcDesiredVelocity()   # for Bot
            if any(self.nbors):
                rji, rji_mag, vji, vji_mag = self.get_noisedState()
                self.calcDesiredVelocity(rji, rji_mag, vji, vji_mag)

        # print(self.id, self.pos)

        super().update(step, frame)

        self.phi_vel = norm(self.vel)
        self.memory.append(self.get_state())

    def calcDesiredVelocity(self, *args):
        v_rep = 0
        v_frict = 0
        if df.sep_flag:
            v_rep = self.separate(args[0], args[1])
        if df.align_flag:
            v_frict = self.align(args[2], args[3], args[1])
        self.v_d += v_rep + v_frict

    def get_nbors(self):
        # rji = np.array([agent.pos for agent in self.env.agents]) - self.pos
        # rji_mag = np.sqrt(rji[:, 0] ** 2 + rji[:, 1] ** 2)
        # self_bool = np.arange(df.num_agents) != self.id
        # nbors_bool = self_bool & (rji_mag < df.comm_radius)
        # return np.array(self.env.agents)[nbors_bool]
        nbors = []
        cluster_count = 0
        self.phi_corr = 0
        self.phi_coll = 0
        for agent in self.env.agents:
            dist = norm(agent.pos - self.pos)
            if 0 < dist < self.r_cluster:
                self.phi_corr += self.vel.dot(agent.vel) / norm(self.vel) / norm(agent.vel)
                cluster_count += 1
            if 0 < dist < df.comm_radius and not packet_lost(dist):
                nbors.append(agent)
                if dist < df.coll_radius:
                    self.phi_coll += 1
                    self.warnings.append(f'Collision with agent {agent.id},  ')
        if cluster_count:
            self.phi_corr /= cluster_count
        return nbors

    def get_noisedState(self):
        rj = np.array([agent.memory[0][0] + agent.gps_pos for agent in self.nbors])
        if df.wp_flag:
            self.localcom = np.sum(rj, axis=0)/len(self.nbors)
        rji = rj - self.pos - self.gps_pos
        rji_mag = np.sqrt(rji[:, 0] ** 2 + rji[:, 1] ** 2)

        vji = np.array([agent.memory[0][1] + agent.gps_vel for agent in self.nbors]) - self.vel - self.gps_vel
        vji_mag = np.sqrt(vji[:, 0] ** 2 + vji[:, 1] ** 2)

        return rji, rji_mag.reshape(len(self.nbors), 1), vji, vji_mag.reshape(len(self.nbors), 1)   #reshape is faster than expand dims

    def separate(self, rji, rji_mag):
        temp = np.copy(rji_mag)  # create new reference
        temp[rji_mag > self.conf["r0_rep"]] = self.conf["r0_rep"]
        v_reps = self.conf["p_rep"] * (self.conf["r0_rep"] - temp) * (-rji / temp)

        return np.sum(v_reps, axis=0)

    def align(self, vji, vji_mag, rji_mag):

        v_frict_max = brake_decay(rji_mag - self.conf["r0_frict"], self.conf["a_frict"], self.conf["p_frict"])
        v_frict_max[v_frict_max < self.conf["v_frict"]] = self.conf["v_frict"]
        temp = np.copy(vji_mag)
        temp[vji_mag < v_frict_max] = np.squeeze(v_frict_max[vji_mag < v_frict_max])
        v_fricts = self.conf["c_frict"] * (temp - v_frict_max) * (vji / temp)

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
