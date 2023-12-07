import numpy as np
from pydrake.autodiffutils import AutoDiffXd

def cos(theta):
    return AutoDiffXd.cos(theta)
def sin(theta):
    return AutoDiffXd.sin(theta)

def AddFinalLandingPositionConstraint(prog, q0_ball, v0_ball, xf, d, t_catch, plant):
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

    def EndEffectorFinalPosHelper(vars):
        tf = vars[0]
        xf = vars[1:]
        # print('ee pos', EndEffectorFinalPos(plant_autodiff, context, xf))
        # print('ball pos', CalcCatchPos(q0_ball, v0_ball, tf))
        return EndEffectorFinalPose(plant_autodiff, context, xf) - CalcCatchPose(q0_ball, v0_ball, tf)

    def CalcCatchPose(q0, v0, t_catch):
        y_final = q0[1] + v0[1] * t_catch
        x_final = q0[0] + v0[0] * t_catch
        z_final = q0[2] + v0[2] * t_catch - 0.5 * g * t_catch**2
        pos_final = np.array([x_final, y_final, z_final])

        vz_final = v0[2] - g * t_catch
        vel_final = np.array([v0[0], v0[1], vz_final])
        vel_final_unit = vel_final / np.linalg.norm(vel_final)

        pos_vel_final = np.append(pos_final, vel_final_unit)

        print("pos_vel_final = ", pos_vel_final)

        return pos_vel_final
        # return pos_final
        # return vel_final

    lb = np.zeros(6) # CalcCatchPos(q0_ball, v0_ball, t_catch)
    ub = np.zeros(6) # CalcCatchPos(q0_ball, v0_ball, t_catch)

    # lb = np.zeros(3) # CalcCatchPos(q0_ball, v0_ball, t_catch)
    # ub = np.zeros(3) # CalcCatchPos(q0_ball, v0_ball, t_catch)

    # lb[3:] = np.array([-0.2,-0.2,-0.2])
    # ub[3:] = np.array([0.2,0.2,0.2])

    prog.AddConstraint(EndEffectorFinalPosHelper, lb, ub, [*t_catch, *xf])

    z_final = CalcCatchPose(q0_ball, v0_ball, t_catch[0])[2]
    prog.AddConstraint(z_final, 0.2, np.inf)

def EndEffectorFinalPose(plant, context, xf):
    context.SetContinuousState(xf)
    ee_align = plant.GetBodyByName("panda_link7").body_frame()
    ee_body = plant.GetBodyByName("panda_link9")
    net_frame = ee_body.body_frame()
    ee_point_tracked = np.zeros(3)
    ee_pos = plant.CalcPointsPositions(context, ee_frame, ee_point_tracked, plant.world_frame()).ravel()

    # Get ee orientation
    ee_align = plant.GetBodyByName("panda_link7")
    body_pose = plant.EvalBodyPoseInWorld(context, ee_align)
    ee_rot = body_pose.GetAsMatrix4()[:-1, 2]
    ee_pos_rot = np.append(ee_pos, ee_rot)

    print("ee pos rot = ", ee_pos_rot)

    return ee_pos_rot
    # return ee_pos
    # return ee_rot


