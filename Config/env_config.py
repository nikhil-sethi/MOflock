# environment configuration file
import numpy as np
# [geofence]
geofence = np.array([[-125, -125],  #bottom left
                    [125, -125],
                    [125, 125],
                    [-125, 125]])     # top left

# [obstacles]
'''is a dictionary of obstacles with each obstacle as matrix of points[x,y] in CCW manner'''

obstacles = [np.array([[20, 0],
                    [50, -20],
                    [50, 20]]),
             np.array([[-34, -62],
                        [-50, -40],
                        [-45, -10],
                        [-34, -50]])]
             # #np.array([[],
    #                 [],
    #                 [],
    #                 []]), }

# [weather]
weather = {'temperature': 37,
            'wind_speed': 0,
            'wind_direction': None}
sigma_outer = 0.2  #m2/s3

# [animation]
interval = 100  # time in miliseconds between each frame
# this does not update in real time with this value. Probably slowed down due to cpu.
# takes about exp(-2*log10(update_interval)+2)*100 percent longer on an i3 8th gen cpu

