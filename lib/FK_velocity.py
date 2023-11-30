import sys
sys.path.append('../../meam520_labs')

import numpy as np 
from lib.calcJacobian import calcJacobian
from math import *


def FK_velocity(q_in, dq):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param dq: 1 x 7 vector corresponding to the joint velocities.
    :return:
    velocity - 6 x 1 vector corresponding to the end effector velocities.    
    """

    ## STUDENT CODE GOES HERE

    velocity = np.zeros((6, 1))
    velocity = calcJacobian(q_in) @ dq


    return velocity

if __name__ == '__main__':
    q = np.array([ 0,    0,     0, 0, 0, 0, 0 ])
    dq = np.array([0, 0, 0, 1, 0, 0, 0])
    v = FK_velocity(q, dq)

    print("v = ", np.round(v, 3))

    