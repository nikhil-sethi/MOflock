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
