#TODO

import numpy as np
from Classes import particle  # needs to be after figure definition as it gets the current figure
from Methods.controls import brake_decay
from Methods.vector_algebra import vectorFromPolygon, unit_vector, norm

class Bot(particle.Particle):

    def __init__(self, conf, env):
        super().__init__()
        self.id = None
        self.conf = conf
        self.vmax=conf.vmax
        self.env = env
        self._state = {"position":self.gcp(), "velocity":self.gcv(), "charge":self.conf.charge}
        self.waypoint = None
        #self.sensor= patch.Wedge((self.gcp()[0], self.gcp[1]), self.r, self.gch()-self.phi/2, self.gch()+self.phi/2,edgecolor='b', facecolor=None)
        self.ln,= self.env.ax.plot([],[], 'bo', markersize='2')

    def get_state(self):
        return self._state

    def update(self, interval, *args):
        v_shill_obstacle = self.avoid(self.env.obstacles)
        v_shill_wall = self.avoid([self.env.arena])
        if not self.inArena():  # come back to arena if you're out. Bad Bot!
             v_shill_wall = -2*v_shill_wall
        v_wp = self.goto(self.waypoint)
        self.acc += (v_wp + v_shill_obstacle + v_shill_wall)-self.vel

        #self.ln.set_data(self.pos + self.acc)
        #print(self.vel, self.acc)
        # decay charge
        super().update(interval)

    def sense(self, obstacle):
        # this is a more exact method which loops over all obstacles
        v_s = vectorFromPolygon(self.pos, obstacle) # shill vector from obstacle to position
        r_is = norm(v_s)    #distance from obstacle

        return r_is, self.conf.v_shill*unit_vector(v_s)

    def avoid(self, obstacles):
        """obstacle/wall collision avoidance through shilling"""
        v_si = np.zeros(2)
        for obs in obstacles:
            r_si, v_s = self.sense(obs.get_xy())
            v_smax = brake_decay(r_si - self.conf.r0_shill, self.conf.a_shill, self.conf.p_shill)
            v_si_mag = norm(v_s-self.vel)
            if v_si_mag > v_smax:
                v_si += (v_si_mag - v_smax)* unit_vector(v_s-self.vel)
        return v_si  # all possible shilling from walls and geofence

    def scw(self, *args):
        """ set current waypoint"""
        if len(args) == 0:
            self.waypoint = None
        else:
            self.waypoint = self.vmax*unit_vector(np.array([args[0], args[1]]))

    def goto(self, waypoint):
        if waypoint is None:
            return 0
        return waypoint-self.pos

    def inArena(self):
        return self.env.arena.get_path().contains_point(self.pos)

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
