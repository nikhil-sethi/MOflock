# TODO

import numpy as np
from Classes import particle
from Methods.controls import brake_decay, add_outernoise, sigmoid_brake
from Methods.vector_algebra import vectorFromPolygon, unit_vector, norm, relu
import Config.defaults as df


class Bot(particle.Particle):

    def __init__(self, env, paramdict):
        super().__init__()
        self.id = None
        self.warnings = []

        self.v_d = np.zeros(2)
        self.gps_pos = np.zeros(2)
        self.gps_vel = np.zeros(2)
        self.waypoint = None
        self.localcom = self.pos
        self.env = env
        self.conf = paramdict
        self.phi_wall = 0
        self.wall_count = 0

    def get_state(self):
        return self.pos, self.vel

    def update(self, step, frame):

        add_outernoise(self.vel, df.sigma_outer, step)
        self.acc += self.v_d - self.vel - self.gps_vel

        super().update(step, frame)

        # self.ln.set_data(self.pos + self.acc)

    def calcDesiredVelocity(self):
        v_shill_obstacle = 0
        v_shill_wall = 0
        v_wp = np.zeros(2)
        v_wp_mag = 0
        self.phi_wall = 0
        self.outside = 0
        if df.obs_flag:
            v_shill_obstacle = self.avoid(self.env.obstacles, 'obstacle')
        if df.wall_flag:
            v_shill_wall = self.avoid([self.env.arena], 'arena', 'center_square')
            # if not self.inArena():  # come back to arena if you're out. Bad Bot!
            #     v_shill_wall = -2 * v_shill_wall
        if df.wp_flag:
            v_wp, v_wp_mag = unit_vector(self.goto(self.waypoint))
        self.v_d += min(v_wp_mag, df.v_target) * v_wp + v_shill_obstacle + v_shill_wall + df.v_flock * \
                    unit_vector(self.vel)[0]

    def sense(self, obstacle, obstype, method, index=1):
        if method == 'perpendicular':
            v_s = vectorFromPolygon(self.pos, obstacle)  # shill vector from obstacle to position
            # this condition below smooths the repulsion at corners avoiding the conflict at equal distance walls
            if obstype is 'arena' and np.all(
                    abs(self.pos) > 0.8 * obstacle[1][0]):  # TODO: change second condition to obstacle.get_xy()?
                com = np.sum(obstacle[:-1, :], axis=0) / len(obstacle)
                v_s = 1.5 * (com - self.pos)

            v_s, r_is = unit_vector(v_s)  # distance from obstacle
            return r_is, self.conf["v_shill"] * v_s

        elif method == 'center_square':
            # com = np.sum(obstacle[:-1, :], axis=0) / len(obstacle)
            toCenter = -self.pos
            r_si = obstacle[1, 0] - abs(toCenter[index])
            v_s = np.zeros(2)
            v_s[index] = toCenter[index]
            return r_si, self.conf["v_shill"] * unit_vector(v_s)[0]

        elif method is 'center_square_approx':
            assert obstype != 'obstacle', "'center square' method is only valid for arena type objects. Select the 'center circle' or 'perpendicular' method"
            if not self.inPolygon(obstacle, factor=1.2):
                com = np.sum(obstacle[:-1, :], axis=0) / len(obstacle)
                v_s = com - self.pos
                r_i_com = norm(v_s)  # distance from COM
                r_obs = 1.2 * obstacle[1, 0]
                r_is = abs(r_i_com - r_obs)
                return abs(r_is), self.conf["v_shill"] * v_s / r_i_com
            return 0, 0

        elif method is 'center_circle':
            com = np.sum(obstacle[:-1, :], axis=0) / len(obstacle)
            v_s = com - self.pos
            r_i_com = norm(v_s)  # distance from COM
            if obstype == 'arena':
                r_obs = 1.2 * obstacle[1, 0]
                r_is = abs(r_i_com - r_obs)
                return abs(r_is), self.conf["v_shill"] * v_s / r_i_com

            r_obs = np.max(np.linalg.norm(obstacle[:-1, :] - com, axis=1))  # obstacle radius
            r_is = abs(r_i_com - r_obs)
            return r_is, -self.conf["v_shill"] * v_s / r_i_com

    def avoid(self, obstacles, obstype, method='perpendicular'):
        """obstacle/wall collision avoidance through shilling"""
        # this is a more exact method which loops over all obstacles
        # can be modeled as a discrete points as well
        v_si = np.zeros(2)
        if obstype is 'arena':
            temp = 10000
            for i in range(2):  # for each component
                r_si, v_s = self.sense(obstacles[0].get_xy(), obstype, method, i)
                if r_si < temp:
                    temp = r_si

                v_smax = brake_decay(r_si - self.conf["r0_shill"], self.conf["a_shill"], self.conf["p_shill"])
                v_si_mag = norm(v_s - self.vel)
                if v_si_mag > v_smax:
                    v_si += (v_si_mag - v_smax) * (v_s - self.vel) / v_si_mag

            if not self.inPolygon(obstacles[0]):
                self.phi_wall += temp
                self.outside = 1
                self.warnings.append(f'Collision with {obstype}, ')
        else:
            r0_shill = relu(self.conf["r0_shill"]) + 1  # just some extra safety
            for obs in obstacles:
                r_si, v_s = self.sense(obs.get_xy(), obstype, method)

                if self.inPolygon(obs):
                    self.phi_wall += r_si  # order parameter
                    self.warnings.append(f'Collision with {obstype}, ')
                    # print(f"Env-{self.env.id}:")
                v_smax = brake_decay(r_si - r0_shill, self.conf["a_shill"], self.conf["p_shill"])
                v_si_mag = norm(v_s - self.vel)
                if v_si_mag > v_smax:
                    v_si += (v_si_mag - v_smax) * (v_s - self.vel) / v_si_mag

        # self.v.set_data(self.pos + v_si)

        return v_si  # all possible shilling from walls and geofence

    def scw(self, *args):
        """ set current waypoint"""
        if len(args) == 0:
            self.waypoint = None
        else:
            self.waypoint = np.array([args[0], args[1]])

    def goto(self, waypoint):
        toWP = waypoint - self.localcom
        unit_to_WP, dist_to_WP = unit_vector(toWP)
        # target_mag = brake_decay(dist_to_WP - 30, 5.54, 3.32)
        target_flag = sigmoid_brake(dist_to_WP, 0, 2)
        toCOM = self.localcom - self.pos
        unit_to_COM, dist_to_COM = unit_vector(toCOM)
        # com_mag = brake_decay(dist_to_COM - 20, 5.54, 3.32)
        com_flag = sigmoid_brake(dist_to_COM, 40, 4)

        if waypoint is None:
            return 0
        return df.v_target * (target_flag * unit_to_WP + com_flag * unit_to_COM)

    def inPolygon(self, polygon, factor=1.):
        return polygon.get_path().contains_point(factor * self.pos)

    '''
    def land(self):
        pass
    
    def arm_and_takeoff(altitude):
        vehicle = connect("127.0.0.1:14%d1" % (55 + sysid - 1), wait_ready=True)
        time.sleep(1)
        while vehicle.is_armable is False:
            print
            "Initialsing.."
            time.sleep(5)
        while vehicle.mode != 'GUIDED':
            vehicle.mode = VehicleMode('GUIDED')
            time.sleep(0.5)
        while not vehicle.armed:
            vehicle.armed = True
            time.sleep(0.5)
        print
        "Taking Off"
        vehicle.simple_takeoff(altitude)
        while True:
            # Break and return from function just below target altitude.
            if vehicle.location.global_relative_frame.alt >= target_alt * 0.95:
                print("Reached target altitude")
                break
            time.sleep(1)

    def send_ned_velocity(velcoity_x, velocity_y, velocity_z, sysid):

        msg = vehicle[sysid].message_factory.set_position_target_local_ned_encode(
            0,
            0, 0,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,
            0, 0, 0,
            velcoity_x, velocity_y, velocity_z,  # velocities in m/s
            0, 0, 0,  # accelerations
            0, 0)  # yaw,yaw_rate
        vehicle[sysid].send_mavlink(msg)

    def send_attitude_target(roll_angle=0.0, pitch_angle=0.0,
                             yaw_angle=None, sysid=1.0, yaw_rate=0.0, use_yaw_rate=False,
                             thrust=0.5):
        """
        use_yaw_rate: the yaw can be controlled using yaw_angle OR yaw_rate.
                      When one is used, the other is ignored by Ardupilot.
        thrust: 0 <= thrust <= 1, as a fraction of maximum vertical thrust.
                Note that as of Copter 3.5, thrust = 0.5 triggers a special case in
                the code for maintaining current altitude.
        """
        if yaw_angle is None:
            # this value may be unused by the vehicle, depending on use_yaw_rate
            yaw_angle = vehicle.attitude.yaw
        # Thrust >  0.5: Ascend
        # Thrust == 0.5: Hold the altitude
        # Thrust <  0.5: Descend
        msg = vehicle[sysid].message_factory.set_attitude_target_encode(
            0,  # time_boot_ms
            1,  # Target system
            1,  # Target component
            0b00000000 if use_yaw_rate else 0b00000100,
            to_quaternion(roll_angle, pitch_angle, yaw_angle),  # Quaternion
            0,  # Body roll rate in radian
            0,  # Body pitch rate in radian
            math.radians(yaw_rate),  # Body yaw rate in radian/second
            thrust  # Thrust
        )
        vehicle[sysid].send_mavlink(msg)

    def set_attitude(roll_angle=0.0, pitch_angle=0.0,
                     yaw_angle=None, sysid=1.0, yaw_rate=0.0, use_yaw_rate=False,
                     thrust=0.5, duration=1):
        """
        Note that from AC3.3 the message should be re-sent more often than every
        second, as an ATTITUDE_TARGET order has a timeout of 1s.
        In AC3.2.1 and earlier the specified attitude persists until it is canceled.
        The code below should work on either version.
        Sending the message multiple times is the recommended way.
        """
        send_attitude_target(roll_angle, pitch_angle,
                             yaw_angle, sysid, yaw_rate, False,
                             thrust)
        start = time.time()
        while time.time() - start < duration:
            send_attitude_target(roll_angle, pitch_angle,
                                 yaw_angle, sysid, yaw_rate, False,
                                 thrust)
            time.sleep(0.1)
        # Reset attitude, or it will persist for 1s more due to the timeout
        send_attitude_target(0, 0,
                             0, sysid, 0, True,
                             thrust)

    
    '''
