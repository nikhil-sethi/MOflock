import numpy as np
from Classes import bot
from Utils.controls import brake_decay, sdl, add_innernoise, packet_lost, brake_decay_inverse
from Utils.vector_algebra import norm, unit_vector
import Config.defaults as df


class CoBot(bot.Bot):

    def __init__(self, env):
        # self.memory = []
        self.phi_coll = 0
        self.phi_corr = 0
        self.phi_vel = 0
        super().__init__(env)
        # self.r_cluster = max(self.conf["r0_rep"], self.conf["r0_frict"] + brake_decay_inverse(df.v_flock, self.conf["a_frict"], self.conf["p_frict"]))
        # paper version
        self.r_cluster = self.conf["r0_rep"] + self.conf["r0_frict"] + sdl(df.v_flock, self.conf["a_frict"],
                                                                           self.conf["p_frict"])


    def calcDesiredVelocity(self, step, frame):
        self.warnings = []
        if frame % (df.gps_del / step) == 0:
            add_innernoise(self.gps_pos, self.gps_vel, df.sigma_inner,
                           df.gps_del)
            # print(frame, self.gps_pos)
        self.nbors = self.get_nbors()
        # print(self.id, `,len(self.nbors))

        if frame % (df.gps_del / step) == 0:
            self.v_d = np.zeros(2)
            # super().calcDesiredVelocity()  # for Bot
            if any(self.nbors):
                rji, rji_mag, vji, vji_mag = self.get_noisedState()
                v_rep = 0
                v_frict = 0
                if df.sep_flag:
                    v_rep = self.separate(rji, rji_mag)
                if df.align_flag:
                    v_frict = self.align(vji, vji_mag, rji_mag)
                self.v_d += v_rep + v_frict
                # print(self.id, "v_rep=", v_rep, "mag=", norm(v_rep))
                # print(self.id, "v_rep=", norm(v_rep), " v_align= ", norm(v_frict))
            super().calcDesiredVelocity()  # for Bot
            # self.v_d = unit_vector(self.v_d)[0] *min(norm(self.v_d), df.vmax)#df.v_flock  #

    def get_nbors(self):
        # rji = np.array([agent.pos for agent in self.env.agents]) - self.pos
        # rji_mag = np.sqrt(rji[:, 0] ** 2 + rji[:, 1] ** 2)
        # self_bool = np.arange(df.num_agents) != self.id
        # nbors_bool = self_bool & (rji_mag < df.comm_radius)
        # return np.array(self.env.agents)[nbors_bool]
        nbors = []
        self.cluster_count = 0
        self.phi_corr = 0
        self.phi_coll = 0
        for agent in self.env.agents:  # in self.cluster
            dist = norm(agent.pos - self.pos)
            if 0 < dist < self.r_cluster:
                self.phi_corr += self.vel.dot(agent.vel) / norm(self.vel) / norm(agent.vel)
                self.cluster_count += 1
            if 0 < dist < df.comm_radius and not packet_lost(dist):
                nbors.append(agent)
                if dist < df.coll_radius:
                    self.phi_coll += 1
                    self.warnings.append(f'Collision with agent {agent.id},  ')
        if self.cluster_count:
            self.phi_corr /= self.cluster_count
            self.disc = 0
        else:
            self.disc = 1
            self.warnings.append("Disconnected,  ")
        return nbors

    def get_noisedState(self):
        # rj = np.array([agent.memory[0][0] + agent.gps_pos for agent in self.nbors])
        # vji = np.array([agent.memory[0][1] + agent.gps_vel for agent in
        #                 self.nbors]) - self.vel - self.gps_vel

        try:
            rj = np.array([agent.memory[-int(df.comm_del/df.interval)][0] + agent.gps_pos for agent in self.nbors])
            vji = np.array([agent.memory[-int(df.comm_del / df.interval)][1] + agent.gps_vel for agent in
                            self.nbors]) - self.vel - self.gps_vel
        except:
            rj = np.array([agent.memory[0][0] + agent.gps_pos for agent in self.nbors])
            vji = np.array([agent.memory[0][1] + agent.gps_vel for agent in
                            self.nbors]) - self.vel - self.gps_vel
        if df.wp_flag:
            self.localcom = np.sum(rj, axis=0) / len(self.nbors)
        rji = rj - self.pos - self.gps_pos
        # print(self.id, "rji=", rji)
        rji_mag = np.sqrt(rji[:, 0] ** 2 + rji[:, 1] ** 2)

        # print(self.id, "vji", vji)
        vji_mag = np.sqrt(vji[:, 0] ** 2 + vji[:, 1] ** 2)
        # print(vji_mag)
        return rji, rji_mag.reshape(len(self.nbors), 1), vji, vji_mag.reshape(len(self.nbors),
                                                                              1)  # reshape is faster than expand dims

    def separate(self, rji, rji_mag):
        temp = np.clip(rji_mag, a_min=None, a_max=self.conf["r0_rep"])
        v_rep_max = self.conf["p_rep"] * (self.conf["r0_rep"] - temp)
        v_reps = np.clip(v_rep_max, a_min=None, a_max=6.0) * (-rji / temp)
        return np.sum(v_reps, axis=0)

    def align(self, vji, vji_mag, rji_mag):
        v_frict_max = brake_decay(rji_mag - self.conf["r0_frict"] - self.conf['r0_rep'], self.conf["a_frict"],
                                  self.conf["p_frict"], v_m=vji_mag)
        v_frict_max = np.clip(v_frict_max, a_min=self.conf['v_frict'], a_max=None)
        temp = np.clip(vji_mag, a_min=v_frict_max, a_max=None)
        v_fricts = self.conf["c_frict"] * (temp - v_frict_max) * (vji / temp)

        return np.sum(v_fricts, axis=0)

