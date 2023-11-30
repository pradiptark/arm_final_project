import numpy as np
from math import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout

        self.l = [0.141, 0.192, 0.195, 0.121, 0.0825, 0.0825, 0.125, 0.259, 0.088, 0.051, 0.159]
        self.joint_5_offset = 0.015

        pass

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here

        jointPositions = np.zeros((8,3))
        T0e = np.identity(4)

        A = np.eye(4)
        for i in range(7):
            j = i + 1
            joint_pos = jointPositions[i]
            joint_ee = jointPositions[-1]
            if i == 0:
                joint_pos[2] = self.l[i]
            elif i == 2:
                joint_pos[2] = self.l[i]
            elif i == 4:
                joint_pos[2] = self.l[6]
            elif i == 5:
                joint_pos[2] = -self.joint_5_offset
            elif i == 6:
                joint_pos[2] = self.l[9]
                
            joint_pos = A @ np.append(joint_pos, 1)
            jointPositions[i] = joint_pos[:3]

            A = A @ self.compute_Ai(q, i)
            
            if j == 7:
                joint_ee = A @ np.append(joint_ee, 1)
                jointPositions[j] = joint_ee[:3]

        T0e = A

        return jointPositions, T0e
    
    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        Z = np.empty([0, 3])
        A = np.eye(4)

        for i in range(len(q)):
            A = A @ self.compute_Ai(q, i)
            Az = A[:3, 2].reshape((1,3))
            Z = np.append(Z, Az, axis = 0)
        return Z
    
    def compute_Ai(self, q, i):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """

        a = [0, 0, self.l[4], -self.l[5], 0, self.l[8], 0]
        alpha = [-pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2, 0]
        d = [self.l[0]+self.l[1], 0, self.l[2]+self.l[3], 0, self.l[6]+self.l[7], 0, self.l[9]+self.l[10]]
        theta = [q[0], q[1], q[2], q[3], q[4], q[5], q[6]-pi/4]

        A = [[cos(theta[i]), -sin(theta[i])*cos(alpha[i]), sin(theta[i])*sin(alpha[i]), a[i]*cos(theta[i])],
             [sin(theta[i]), cos(theta[i])*cos(alpha[i]), -cos(theta[i])*sin(alpha[i]), a[i]*sin(theta[i])],
             [0, sin(alpha[i]), cos(alpha[i]), d[i]],
             [0, 0, 0, 1]]

        return A
    
    def compute_T0i(self, q, i):
        T0i = np.eye(4)
        for i in range(i):
            T0i = T0i @ self.compute_Ai(q, i)
        
        return T0i

if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    # q = np.array([0,0,0,-pi/2,0,pi/2,0])
    # q = np.array([ 0,    0, -pi/2, -pi/4,  pi/2, pi,   pi/4 ])

    # q = np.array([0,0,0,0,0,0,0])

    # q = np.array([-0.01779206, -0.76012354, 0.01978261, -2.34205014, 0.02984053, 3.14159265, 0.75344866])
    # q = np.array([ 0,    0,     0, 0, 0, pi/2, 0 ])

    q =  np.array([ 2.39440460e-01, -1.30909691e-02,  2.36914434e-01, -3.42488372e-01,
  7.70812199e-02,  2.51799186e+00, -1.43777163e-03])


    joint_positions, T0e = fk.forward(q)
    z = fk.get_axis_of_rotation(q)

    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
    print("Z = ", z)

   