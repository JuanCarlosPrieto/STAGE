from scipy.optimize import brentq


def find_intersection(phi, a, b):
    '''
    Find the intersection of a line going from a to b with the curve defined by the function phi.

    Parameters
    ----------
    phi : callable
        A function that takes a point and returns a scalar value
    a : array-like
        The starting point of the line segment
    b : array-like
        The ending point of the line segment
    '''

    if phi(a) * phi(b) > 0:
        raise ValueError("The function must have opposite signs at the endpoints a and b.")

    def f(t):
        return phi(a + t * (b - a)), t

    try:
        t_intersection = brentq(f, 0, 1)
        intersection_point = a + t_intersection * (b - a)
        return intersection_point
    
    except ValueError:
        return None