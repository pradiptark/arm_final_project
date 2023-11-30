import matplotlib.pyplot as plt
import numpy as np
import importlib
from random import uniform

from pydrake.all import (
    DiagramBuilder, Simulator, FindResourceOrThrow, MultibodyPlant, PiecewisePolynomial, SceneGraph,
    Parser, JointActuatorIndex, MathematicalProgram, Solve
)
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph


import kinematic_constraints
import dynamics_constraints
importlib.reload(kinematic_constraints)
importlib.reload(dynamics_constraints)
from kinematic_constraints import (
    AddFinalLandingPositionConstraint
)
from dynamics_constraints import (
AddCollocationConstraints,
    EvaluateDynamics
)

def find_throwing_trajectory(N, initial_state, final_configuration, distance, tf):
    '''
    Parameters:
        N - number of knot points
        initial_state - starting configuration
        distance - target distance to throw the ball

    '''


    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.)
    parser = Parser(plant)
    parser.AddModels(url="package://drake/manipulation/models/franka_description/urdf/panda_arm.urdf")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0"))

    plant.Finalize()
    planar_arm = plant.ToAutoDiffXd()

    plant_context = plant.CreateDefaultContext()
    context = planar_arm.CreateDefaultContext()

    # Dimensions specific to the planar_arm
    n_q = planar_arm.num_positions()
    n_v = planar_arm.num_velocities()
    n_x = n_q + n_v
    n_u = planar_arm.num_actuators()

    print("n_q = ", n_q)
    print("n_v = ", n_v)
    print("n_x = ", n_x)
    print("n_u = ", n_u)

    # Store the actuator limits here
    effort_limits = np.zeros(n_u)
    for act_idx in range(n_u):
        effort_limits[act_idx] = \
        planar_arm.get_joint_actuator(JointActuatorIndex(act_idx)).effort_limit()
    joint_limits = np.pi * np.ones(n_q)
    vel_limits = 15 * np.ones(n_v)

    # Create the mathematical program
    prog = MathematicalProgram()
    x = np.zeros((N, n_x), dtype="object")
    u = np.zeros((N, n_u), dtype="object")
    for i in range(N):
        x[i] = prog.NewContinuousVariables(n_x, "x_" + str(i))
        u[i] = prog.NewContinuousVariables(n_u, "u_" + str(i))

    # t_catch = prog.NewContinuousVariables(1, "t_catch")
    t_catch = tf

    t0 = 0.0
    timesteps = np.linspace(t0, tf, N)
    x0 = x[0]

    xf = x[-1]

    # DO NOT MODIFY THE LINES ABOVE

    # Add the kinematic constraints (initial state, final state)
    # TODO: Add constraints on the initial state
    prog.AddLinearEqualityConstraint(x[0], initial_state)

    # Add the kinematic constraint on the final state
    AddFinalLandingPositionConstraint(prog, xf, distance, t_catch, plant)

    # Add the collocation aka dynamics constraints
    AddCollocationConstraints(prog, planar_arm, context, N, x, u, timesteps)

    # TODO: Add the cost function here
    cost = 0
    dt = tf/N
    for i in range(N-2):
        cost += dt/2 * (u[i].T @ u[i] + u[i+1].T @ u[i+1])
    prog.AddQuadraticCost(cost)
    # print("cost = ", cost)

    # TODO: Add bounding box constraints on the inputs and qdot 
    v = np.append(x.flatten(), u.flatten())
    x_limits = np.tile(np.append(joint_limits, vel_limits), N)
    u_limits = np.tile(effort_limits, N)
    ub = np.append(x_limits, u_limits)
    lb = np.append(-x_limits, -u_limits)
    prog.AddBoundingBoxConstraint(lb, ub, v)
    # print("v shape = ", v.shape)
    # print("ub shape = ", ub.shape)
    # print("lb shape = ", lb.shape)

    # TODO: give the solver an initial guess for x and u using prog.SetInitialGuess(var, value)
    x_init_guess = np.zeros((N, n_x))
    x_init_guess[0] = initial_state
    for i in range(N):
        x_init_guess[i][0] = i*np.pi/N
    
    u_init_guess0 = np.zeros(n_u)
    u_init_guess0[0] = uniform(-effort_limits[0], effort_limits[0])
    u_init_guess0[1] = uniform(-effort_limits[1], effort_limits[1])
    u_init_guess = np.tile(u_init_guess0, (N,1))

    # print("x = ", x)
    # print("u = ", u)
    # print("x_init_guess = ", x_init_guess)
    # print("u_init_guess = ", u_init_guess)

    prog.SetInitialGuess(x, x_init_guess)
    prog.SetInitialGuess(u, u_init_guess)

    #DO NOT MODIFY THE LINES BELOW

    # Set up solver
    result = Solve(prog)
    
    x_sol = result.GetSolution(x)
    u_sol = result.GetSolution(u)
    t_catch_sol = result.GetSolution(t_catch)

    print('optimal cost: ', result.get_optimal_cost())
    print('x_sol: ', x_sol)
    print('u_sol: ', u_sol)
    print('t_catch: ', t_catch_sol)

    print(result.get_solution_result())

    # Reconstruct the trajectory
    xdot_sol = np.zeros(x_sol.shape)
    for i in range(N):
        xdot_sol[i] = EvaluateDynamics(plant, plant_context, x_sol[i], u_sol[i])
    
    x_traj = PiecewisePolynomial.CubicHermite(timesteps, x_sol.T, xdot_sol.T)
    u_traj = PiecewisePolynomial.ZeroOrderHold(timesteps, u_sol.T)

    return x_traj, u_traj, prog, prog.GetInitialGuess(x), prog.GetInitialGuess(u)
    
if __name__ == '__main__':
    N = 5
    initial_state = np.zeros(4)
    final_configuration = np.array([np.pi, 0])
    tf = 3.0
    distance = 15.0
    x_traj, u_traj, prog, _, _ = find_throwing_trajectory(N, initial_state, final_configuration, distance, tf)