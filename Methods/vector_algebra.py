import numpy as np
norm =np.linalg.norm

def confidence_ellipse(x, y, chi=5.991):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    chi : float
    confidence of enclosing ellipse(from chi square distribution table)
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm
    95% confidence: 5.991(2D),7.815(3D)
    95% confidence: 7.378(2D),9.348(3D)
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    E, V = np.linalg.eig(np.cov(x, y))  # this might be expensive. replace with above pearson method if possible
    # chi = 5.991  #
    ell_radius_x = np.sqrt(chi * max(E))
    ell_radius_y = np.sqrt(chi * min(E))
    v = V[:, np.argmax(E)]
    angle = np.arctan(v[1] / v[0])
    if angle < 0:
        angle += np.pi
    return (np.mean(x), np.mean(y)), ell_radius_x, ell_radius_y, angle


def rotmat(th):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def unit_vector(vec):
    if np.any(vec):
        return vec / np.linalg.norm(vec)
    return np.zeros(2)


def absheading_from_vec(vec: np.array, mode='cartesian') -> float:
    """return absolute heading"""
    if mode is 'azimuth':
        y = vec[0]
        x = vec[1]
    elif mode is 'cartesian':
        y = vec[1]
        x = vec[0]
    head = np.arctan2(y, x) * 180 / np.pi
    if head < 0: head += 360
    return head


def vec_from_absheading(head: float, mode: str) -> np.array:
    '''returns a vector in cartesian form from any absolute heading(aziumth/cartesian)'''
    head = np.deg2rad(head)
    if mode is 'cartesian':
        return np.array([np.cos(head), np.sin(head)])
    elif mode is 'azimuth':
        return np.array([np.sin(head), np.cos(head)])


def to_quaternion(roll=0.0, pitch=0.0, yaw=0.0):
    """
        Convert current attitude to quaternions
        """
    t0 = math.cos(math.radians(yaw * 0.5))
    t1 = math.sin(math.radians(yaw * 0.5))
    t2 = math.cos(math.radians(roll * 0.5))
    t3 = math.sin(math.radians(roll * 0.5))
    t4 = math.cos(math.radians(pitch * 0.5))
    t5 = math.sin(math.radians(pitch * 0.5))

    w = t0 * t2 * t4 + t1 * t3 * t5
    x = t0 * t3 * t4 - t1 * t2 * t5
    y = t0 * t2 * t5 + t1 * t3 * t4
    z = t1 * t2 * t4 - t0 * t3 * t5

    return [w, x, y, z]

def distFromLine(point, line):
    pass
def InShadow(point, edge_startPoint, edge_endPoint):
    '''
    returns: actual vector from edge to point, distance, bool for shadow check
    '''
    edge=unit_vector(edge_endPoint-edge_startPoint)
    v1 = edge_startPoint - point
    v2 = edge_endPoint - point
    proj = v1.dot(edge)*edge
    edgeToPoint = proj-v1
    if v1.dot(edge)*v2.dot(edge)<0:
        return edgeToPoint, norm(edgeToPoint), True
    return edgeToPoint, norm(edgeToPoint), False

def vectorFromPolygon(point, polygon):
    """
    :param point is a 1x2 numpy array [x,y] of concerned point
    :param polygon is an Nx2 matrix of CCW ordered points
    :returns a vector from closest point on polygon towards 'position'"""
    cp_index = np.argmin(norm(point - polygon, axis=1))
    edge2ToPoint, dist2, e2_shadow = InShadow(point, polygon[cp_index], polygon[cp_index + 1])
    if cp_index==0:
        cp_index = -1   # point -1 and 0 are same.
    edge1ToPoint, dist1, e1_shadow = InShadow(point,polygon[cp_index-1], polygon[cp_index])

    if e1_shadow:
        ans = edge1ToPoint
        if e2_shadow:
            if dist2<dist1:
                ans= edge2ToPoint
    elif e2_shadow:
        ans = edge2ToPoint
    else:
        return point - polygon[cp_index]
    return ans


