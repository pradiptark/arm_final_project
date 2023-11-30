import numpy as np
from pydrake.autodiffutils import AutoDiffXd

def cos(theta):
    return AutoDiffXd.cos(theta)
def sin(theta):
    return AutoDiffXd.sin(theta)

def AddFinalLandingPositionConstraint(prog, xf, d, t_catch, plant):
    '''
    Impose a constraint such that if the ball is released at final state xf, 
    it will land a distance d from the base of the robot 
    '''
    l = 1
    g = 9.81
    # constraint_eval = np.zeros((3,), dtype=AutoDiffXd)

    plant_autodiff = plant.ToAutoDiffXd()
    n_x = plant_autodiff.num_positions() + plant_autodiff.num_velocities()
    context = plant_autodiff.CreateDefaultContext()


    # TODO: Express the landing constraint as a function of q, qdot, and t_catch

    # 1. find ball final xyz
    # 2. calc final ee pos from (1) using FK using q as vars
    # 3. set final q position as constraints

    def EndEffectorFinalPosHelper(xf):
        return EndEffectorFinalPos(plant_autodiff, context, xf)

    def CalcCatchPos(q0, v0, t_catch):
        y_final = q0[1] + v0[1] * t_catch
        x_final = q0[0] + v0[0] * t_catch
        z_final = q0[2] + v0[2] * t_catch - 0.5 * g * t_catch**2
        return np.array([x_final, y_final, z_final])    

    q0_ball = np.array([0, 0, 3])
    v0_ball = np.array([0, 0, 0])

    # lb = np.zeros(3)
    # ub = np.zeros(3)

    lb = CalcCatchPos(q0_ball, v0_ball, t_catch)
    ub = CalcCatchPos(q0_ball, v0_ball, t_catch)

    prog.AddConstraint(EndEffectorFinalPosHelper, lb, ub, xf)


def EndEffectorFinalPos(plant, context, xf):
    context.SetContinuousState(xf)
    ee_frame = plant.GetBodyByName("panda_link8").body_frame()
    ee_point_tracked = np.zeros(3)
    ee_pos = plant.CalcPointsPositions(context, ee_frame, ee_point_tracked, plant.world_frame()).ravel()

    return ee_pos


