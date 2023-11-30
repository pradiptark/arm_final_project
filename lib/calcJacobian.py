import sys
sys.path.append('../../meam520_labs')

import numpy as np
from lib.calculateFK import FK
from math import *

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))
    fk = FK()

    jointPositions, T0e = fk.forward(q_in)
    n_joints = len(jointPositions)
    for i in range(n_joints):
        if i > 0:
            d_origin = T0e[:3, -1] - T0i[:3, -1]
            J[:3,i-1] = np.cross(Rz, d_origin)
            J[3:,i-1] = Rz
        if i < n_joints-1:
            T0i = fk.compute_T0i(q_in, i)
            Rz = T0i[:3, 2]

    return J


if __name__ == '__main__':

    # q = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    # q = np.array([ 0,    0,     0, -0.0698, 0, 0, 0 ])
    # q = np.array([ 0,    0,     0, 0, 0, 0, 0 ])
    # q = np.array([ 0,    0,     0, pi, 0, 0, 0 ])
    q = np.array([ pi,    0,     0, 0, pi/2, pi, 0 ])
#     q =  np.array([ 2.39440460e-01, -1.30909691e-02,  2.36914434e-01, -3.42488372e-01,
#   7.70812199e-02,  2.51799186e+00, -1.43777163e-03])
#     q = np.array([1.95732384e-01, -4.07308228e-02 , 1.95268631e-01 ,-1.60060992e-01,
#   8.34576426e-02 , 1.89073753e+00 ,-1.43779118e-03])
#     q = np.array([ 1.58504729e-01,  1.49157936e-02,  1.59362814e-01, -3.67399694e-01,
#   7.97911566e-02 , 2.76077419e+00, -1.16144012e-03])

    J = np.round(calcJacobian(q),3)

    print("J = ", J)
    print("rank J = ", np.linalg.matrix_rank(J))
