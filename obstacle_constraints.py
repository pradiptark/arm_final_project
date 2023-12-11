import numpy as np
from pydrake.geometry import GeometryInstance

def AddObstacleConstraint(prog, obstacles, xf, plant):
    """
        obstacles - 4x1 [radius, x, y, z] = x, y, z in world frame
    """
    plant_autodiff = plant.ToAutoDiffXd()
    context = plant_autodiff.CreateDefaultContext()

    sphere_origin = np.array([
        [0, -0.04, 0.0, 0.07],
        [1, 0.0, -0.03, -0.06],
        [2, 0.0, -0.06, 0.03],
        [3, 0.06, 0.024, -0.06],
        [4, -0.05, 0.01, 0.02],
        [5, 0.0, 0.027, -0.21],
        [5, 0.0, 0.07, -0.04],
        [6, 0.04, 0.0, 0.0],
        [7, 0.02, 0.02, 0.07],
        [8, 0.08, 0.08, 0.0],
        [9, 0, 0, 0]
    ])

    sphere_radius = np.array([
        0.14, 0.15, 0.15, 0.14, 0.14, 0.1, 0.115, 0.11, 0.08, 0.03, 0.085 
    ])

    n_sphere = len(sphere_origin)
    n_obs = len(obstacles)

    def JointSpherePosHelper(xf):
        sphere_origin_world = JointSpherePos(plant_autodiff, context, xf, sphere_origin)

        # sphere_obs_dist_total = np.zeros((n_obs, n_sphere))
        sphere_obs_dist_total = np.array([])

        if len(obstacles) > 0:
            for obs in obstacles:

                obs_margin = 0.1
                obs_radius = np.tile(obs[0] + obs_margin, (n_sphere))
                obs_origin = np.tile(obs[1:], (n_sphere, 1))

                sphere_obs_diff = sphere_origin_world - obs_origin
                sphere_obs_dist = np.array([np.linalg.norm(s) for s in sphere_obs_diff]) - obs_radius - sphere_radius
                sphere_obs_dist_total = np.append(sphere_obs_dist_total, sphere_obs_dist)

                # print("sphere_obs_diff = \n", sphere_obs_diff.shape)
                # print("sphere_obs_dist = ", sphere_obs_dist.shape)
                # print("sphere_obs_dist_inv = ", sphere_obs_dist_inv.shape)

        return sphere_obs_dist_total.flatten()

    lb = np.ones(n_sphere * n_obs) * 0
    ub = np.ones(n_sphere * n_obs) * np.inf

    prog.AddConstraint(JointSpherePosHelper, lb, ub, xf)


def JointSpherePos(plant, context, xf, sphere_origin):
    context.SetContinuousState(xf)

    sphere_origin_world = np.empty((0, 3))
    
    for i, ori in enumerate(sphere_origin):
        link_name = "panda_link" + str(int(ori[0]))
        joint_body = plant.GetBodyByName(link_name)
        joint_frame = joint_body.body_frame()

        sphere_origin_i = plant.CalcPointsPositions(context, joint_frame,
                                              np.array(ori[1:]), plant.world_frame()).ravel()

        sphere_origin_world = np.vstack((sphere_origin_world, sphere_origin_i))

    return sphere_origin_world