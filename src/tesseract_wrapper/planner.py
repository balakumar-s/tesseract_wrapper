from dataclasses import dataclass
from tesseract_robotics.tesseract_common import (
    ManipulatorInfo,
    Isometry3d,
    Translation3d,
    Quaterniond,
    FilesystemPath,
)
from tesseract_wrapper.util_file import get_content_path, join_path
from tesseract_robotics.tesseract_environment import (
    Environment,
    AddLinkCommand,
    ChangeCollisionMarginsCommand,
)
from tesseract_robotics.tesseract_scene_graph import (
    Link,
    Joint,
    JointType_FIXED,
)

import numpy as np
import traceback
import os
import re
import time

from tesseract_robotics.tesseract_common import (
    SimpleResourceLocator,
    SimpleResourceLocatorFn,
    CollisionMarginData,
    CollisionMarginOverrideType_OVERRIDE_DEFAULT_MARGIN,
)
from tesseract_robotics.tesseract_command_language import (
    CartesianWaypoint,
    Waypoint,
    PlanInstructionType_FREESPACE,
    PlanInstructionType_START,
    PlanInstruction,
    Instruction,
    CompositeInstruction,
    flatten,
    JointWaypoint,
    StateWaypoint,
    ProfileDictionary,
)
from tesseract_robotics.tesseract_process_managers import (
    ProcessPlanningServer,
    ProcessPlanningRequest,
    FREESPACE_PLANNER_NAME,
    
)
from tesseract_robotics.tesseract_motion_planners_trajopt import (
    TrajOptDefaultCompositeProfile,
    TrajOptMotionPlanner,
    ProfileDictionary_addProfile_TrajOptPlanProfile,
    TrajOptDefaultPlanProfile,
    TrajOptDefaultCompositeProfile,
    ProfileDictionary_addProfile_TrajOptCompositeProfile,
    CollisionEvaluatorType_CAST_CONTINUOUS,
)

from tesseract_robotics.tesseract_motion_planners import (
    generateSeed,
    PlannerResponse,
    PlannerRequest,
    OMPL_DEFAULT_NAMESPACE,
    TRAJOPT_DEFAULT_NAMESPACE,
)
from tesseract_robotics.tesseract_motion_planners_ompl import (
    OMPLMotionPlanner,
    OMPLDefaultPlanProfile,
    ProfileDictionary_addProfile_OMPLPlanProfile,
    AITstarConfigurator,
)

from tesseract_wrapper.world import get_tesseract_obstacle_commands


@dataclass
class TesseractConfig:
    robot_urdf: str
    robot_srdf: str

    def __post_init__(self):
        if isinstance(self.robot_urdf, str):
            self.robot_urdf = FilesystemPath(self.robot_urdf)
        if isinstance(self.robot_srdf, str):
            self.robot_srdf = FilesystemPath(self.robot_srdf)


def _locate_support(url):
    # todo: pass this
    tesseract_support_dir = join_path(
        get_content_path(), "franka_description"
    )  # get_tesseract_ws_path()

    url_match = re.match(r"^package:\/\/tesseract_support\/(.*)$", url)
    if url_match is None:
        return url
    return os.path.join(tesseract_support_dir, os.path.normpath(url_match.group(1)))


def _locate_resource(url):
    try:
        url_match = re.match(r"^meshes(.*)$", url)
        if url_match is None:
            return _locate_support(url)

        tesseract_support = join_path(get_content_path(), "franka_description")
        new_path = os.path.join(tesseract_support, os.path.normpath(url_match.group(0)))
        return new_path
    except:
        traceback.print_exc()


def get_trajectory_from_tesseract_instructions(instruction):
    p_v_a = [
        np.ravel(
            [
                r.as_MoveInstruction().getWaypoint().as_StateWaypoint().position,
                # r.as_MoveInstruction().getWaypoint().as_StateWaypoint().velocity,
                # r.as_MoveInstruction().getWaypoint().as_StateWaypoint().acceleration,
                # r.as_MoveInstruction().getWaypoint().as_StateWaypoint().time,
            ]
        )
        for r in instruction
    ]

    traj = np.array(p_v_a)
    # traj_points = traj[:,:3]
    # time_from_start = traj[:,3]
    time_from_start = np.zeros((traj.shape[0]))
    return traj, time_from_start


def get_ompl_trajectory_from_tesseract_instructions(instruction):
    return get_trajectory_from_tesseract_instructions(instruction)


class TesseractPlanner(TesseractConfig):
    def __init__(self, tesseract_config: TesseractConfig):
        super().__init__(**vars(tesseract_config))
        self.env = Environment()
        self.init_ompl()
        self.init_trajopt()
        # self.init_planning_server()

    def load_robot(self):
        locator_fn = SimpleResourceLocatorFn(_locate_resource)

        # create a new environment
        self.env.init(
            self.robot_urdf, self.robot_srdf, SimpleResourceLocator(locator_fn)
        )
        link = Link("world")
        j = Joint("j_w_b")
        j.parent_link_name = "panda_link0"
        j.child_link_name = "world"
        j.type = JointType_FIXED
        cmd = AddLinkCommand(link, j)
        self.env.applyCommand(cmd)
        self.j_group = self.env.getJointGroup("manipulator")

        self.joint_names = list(self.j_group.getJointNames())
        self.manip_info = ManipulatorInfo()
        self.manip_info.tcp_frame = "right_gripper"
        self.manip_info.manipulator = "manipulator"
        self.manip_info.working_frame = "panda_link0"
        self.manip_info.manipulator_ik_solver = "KDLInvKinChainLMA"

    def init_ompl(self):
        self.ompl_planner = OMPLMotionPlanner()

    def init_trajopt(self):
        self.trajopt_planner = TrajOptMotionPlanner()

    def init_planning_server(self):
        self.planning_server = ProcessPlanningServer(self.env, 1)
        self.planning_server.loadDefaultProcessPlanners()

    def clear_environment(self):
        # clear environment and add back robot:
        self.env.clear()
        self.env = Environment()
        self.load_robot()

    def load_world(self, world_config, collision_margin: float = -0.01):
        self.clear_environment()
        cmd_list = get_tesseract_obstacle_commands(world_config)
        self.env.applyCommands(cmd_list)
        # change margin data:
        margin_data = CollisionMarginData(collision_margin)
        # self.env.setCollisionMarginData(margin_data)
        cmd = ChangeCollisionMarginsCommand(margin_data)
        self.env.applyCommand(cmd)

        # margin_data.incrementMargins(-0.1)

    def plan_composite_cartesian(self, start_q, goal_position, goal_quat):
        self.env.setState(self.joint_names, start_q)
        program = self._create_cartesian_program(start_q, goal_position, goal_quat)
        result = self._plan(program)
        if result is None:
            return None
        traj = get_trajectory_from_tesseract_instructions(result)
        return traj

    def plan_cartesian(self, start_q, goal_position, goal_quat):
        self.env.setState(self.joint_names, start_q)
        program = self._create_cartesian_program(start_q, goal_position, goal_quat)

        seed = generateSeed(program, self.env.getState(), self.env)
        ompl_response, request = self._ompl_plan(program, seed)

        if ompl_response is None:
            return None
        # print(ompl_response.data)
        ompl_result = ompl_response.results
        # trajopt_response = None
        trajopt_response = self._trajopt_plan(program, ompl_result, request)
        if trajopt_response is None:
            return None
            result = flatten(ompl_result)
        else:
            result = flatten(trajopt_response.results)
        traj = get_ompl_trajectory_from_tesseract_instructions(result)

        traj = self.interpolate_trajectory(traj)
        return traj

    def _create_cartesian_program(self, start_q, goal_pos, goal_quat):
        g_pos = goal_pos
        g_q = goal_quat
        jp1 = JointWaypoint(self.joint_names, np.ravel(start_q))

        wp1 = CartesianWaypoint(
            Isometry3d.Identity()
            * Translation3d(g_pos[0], g_pos[1], g_pos[2])
            * Quaterniond(g_q[0], g_q[1], g_q[2], g_q[3])
        )
        start_instruction = PlanInstruction(
            Waypoint(jp1), PlanInstructionType_START, "DEFAULT"
        )
        plan_f1 = PlanInstruction(
            Waypoint(wp1), PlanInstructionType_FREESPACE, "DEFAULT"
        )
        program = CompositeInstruction("DEFAULT")
        program.setStartInstruction(Instruction(start_instruction))
        program.setManipulatorInfo(self.manip_info)
        program.append(Instruction(plan_f1))
        return program

    def plan_cspace(self, start_q, goal_q, run_trajopt=True, run_ompl=True):
        self.env.setState(self.joint_names, start_q)
        # print(goal_q)
        program = self._create_plan_program(start_q, goal_q)
        request = None
        # result = self._plan(program)
        seed = generateSeed(program, self.env.getState(), self.env)
        debug_result = {
            "status": "Graph Fail",
            "graph_time": 0,
            "trajopt_time": 0,
            "total_time": 0,
        }
        ompl_time = 0
        if run_ompl:
            ompl_response, request, ompl_time = self._ompl_plan(program, seed)
            debug_result["graph_time"] = ompl_time
            debug_result["total_time"] = ompl_time
            if ompl_response is None:
                return None, debug_result
            result = flatten(ompl_response.results)
            traj, time_from_start = get_ompl_trajectory_from_tesseract_instructions(
                result
            )
            debug_result["graph_raw_traj"] = traj
            debug_result["graph_time_from_start"] = time_from_start
            debug_result["graph_traj"] = traj

            ompl_result = ompl_response.results
        else:
            ompl_result = seed
        trajopt_response = None
        trajopt_time = 0
        if run_trajopt:
            trajopt_response, trajopt_time = self._trajopt_plan(
                program, ompl_result, request, constraint=False
            )
            if trajopt_response is not None and False:
                trajopt_response, trajopt_time_2 = self._trajopt_plan(
                    program,
                    trajopt_response.results,
                    request,
                    constraint=False,
                    smooth_jerk=True,
                )
                trajopt_time += trajopt_time_2
        debug_result["trajopt_time"] = trajopt_time

        debug_result["total_time"] = ompl_time + trajopt_time
        debug_result["status"] = None
        if run_trajopt and trajopt_response is None:
            debug_result["status"] = "Opt Fail"
            return None, debug_result
        elif run_trajopt:
            result = flatten(trajopt_response.results)
            traj, time_from_start = get_ompl_trajectory_from_tesseract_instructions(
                result
            )
            debug_result["trajopt_raw_traj"] = traj
            debug_result["trajopt_time_from_start"] = time_from_start
            # traj = self.interpolate_trajectory(traj)
        return traj, debug_result

    def plan_composite_cspace(self, start_q, goal_q):
        self.env.setState(self.joint_names, start_q)

        program = self._create_plan_program(start_q, goal_q)

        # result = self._plan(program)
        seed = generateSeed(program, self.env.getState(), self.env)
        result = self._ompl_plan(program, seed)
        if result is None:
            return None
        traj = get_trajectory_from_tesseract_instructions(result)
        return traj

    def _create_plan_program(self, start_q, goal_q):
        jp1 = StateWaypoint(self.joint_names, np.ravel(start_q))
        # jp1.velocity = np.float64(np.zeros(len(start_q)))
        # jp1 =
        jp2 = StateWaypoint(self.joint_names, np.ravel(goal_q))
        # jp2.velocity = np.zeros(len(goal_q))
        start_instruction = PlanInstruction(
            Waypoint(jp1), PlanInstructionType_START, "DEFAULT", "DEFAULT"
        )
        plan_f1 = PlanInstruction(
            Waypoint(jp2), PlanInstructionType_FREESPACE, "FREESPACE", "FREESPACE"
        )
        program = CompositeInstruction("DEFAULT")
        program.setStartInstruction(Instruction(start_instruction))
        program.setManipulatorInfo(self.manip_info)
        program.append(Instruction(plan_f1))
        return program

    def configure_ompl(self):
        #
        ompl_config = AITstarConfigurator()

        return ompl_config

    def _trajopt_plan(
        self,
        program,
        seed,
        request=None,
        constraint=False,
        smooth_jerk=False,
        smooth_velocity=True,
    ):
        # ompl_profile.optimize = False
        profiles = ProfileDictionary()
        plan_profile = TrajOptDefaultPlanProfile()
        composite_profile = TrajOptDefaultCompositeProfile()
        # change composite profile
        if smooth_velocity:
            composite_profile.smooth_velocities = True
            # print("vel:", composite_profile.velocity_coeff)
            composite_profile.velocity_coeff = np.zeros(7) + 100.0
        else:
            composite_profile.smooth_velocities = False
        # composite_profile.
        composite_profile.smooth_accelerations = True
        # composite_profile.acceleration_coeff = np.zeros(7) + 10.0
        composite_profile.smooth_jerks = False
        # composite_profile.jerk_coeff = np.float64(np.zeros(7)) + 100.0

        if smooth_jerk:
            composite_profile.smooth_jerks = True
            composite_profile.jerk_coeff = np.float64(np.zeros(7)) + 100.0
            composite_profile.smooth_accelerations = False

        if constraint:
            # composite_profile.smooth_velocities = False

            # composite_profile.collision_cost_config.enabled = True

            composite_profile.collision_constraint_config.enabled = True
            composite_profile.collision_constraint_config.type = (
                CollisionEvaluatorType_CAST_CONTINUOUS
            )
            composite_profile.collision_constraint_config.safety_margin = 0.005
            composite_profile.collision_constraint_config.safety_margin_buffer = 0.005
            composite_profile.collision_cost_config.enabled = True
            composite_profile.collision_cost_config.type = (
                CollisionEvaluatorType_CAST_CONTINUOUS
            )
            composite_profile.collision_cost_config.safety_margin = 0.005
            composite_profile.collision_cost_config.safety_margin_buffer = 0.01

        else:
            composite_profile.collision_cost_config.enabled = True
            composite_profile.collision_cost_config.type = (
                CollisionEvaluatorType_CAST_CONTINUOUS
            )
            composite_profile.collision_cost_config.safety_margin = 0.001
            composite_profile.collision_cost_config.safety_margin_buffer = 0.005
            composite_profile.collision_constraint_config.enabled = False

        composite_profile.longest_valid_segment_length = 0.025
        ProfileDictionary_addProfile_TrajOptPlanProfile(
            profiles, TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", plan_profile
        )
        ProfileDictionary_addProfile_TrajOptCompositeProfile(
            profiles, TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", composite_profile
        )
        if request is None:
            request = PlannerRequest()
            curr_state = self.env.getState()
            request.env = self.env
            request.env_state = curr_state
            request.instructions = program

        request.profiles = profiles
        request.seed = seed
        response = PlannerResponse()
        start_time = time.time()
        status = self.trajopt_planner.solve(request, response, False)
        plan_time = time.time() - start_time
        if status.value() != 0:
            return None, plan_time

        return response, plan_time

    def _get_ompl_profile(self):
        ompl_profile = OMPLDefaultPlanProfile()
        ompl_profile.simplify = False
        ompl_profile.optimize = False
        ompl_profile.planning_time = 30.0
        ompl_profile.collision_check_config.contact_manager_config.margin_data_override_type = (
            CollisionMarginOverrideType_OVERRIDE_DEFAULT_MARGIN
        )
        ompl_profile.collision_check_config.contact_manager_config.margin_data.setDefaultCollisionMargin(
            -0.02
        )  # -0.06
        ompl_profile.collision_check_config.type = (
            CollisionEvaluatorType_CAST_CONTINUOUS
        )
        ompl_profile.collision_check_config.longest_valid_segment_length = 0.025
        # new_config = self.configure_ompl()
        # ompl_profile.planners.clear()
        # ompl_profile.planners.append(new_config)
        return ompl_profile

    def _get_request(self, program, seed):
        request = PlannerRequest()
        curr_state = self.env.getState()
        request.env = self.env
        request.env_state = curr_state
        request.profiles = profiles
        request.seed = seed
        request.instructions = program
        return request

    def _ompl_plan(self, program, seed):
        ompl_profile = self._get_ompl_profile()
        profiles = ProfileDictionary()
        ProfileDictionary_addProfile_OMPLPlanProfile(
            profiles, OMPL_DEFAULT_NAMESPACE, "DEFAULT", ompl_profile
        )

        request = PlannerRequest()
        curr_state = self.env.getState()
        request.env = self.env
        request.env_state = curr_state
        request.profiles = profiles
        request.seed = seed
        request.instructions = program
        # check if input is good?:

        ompl_response = PlannerResponse()
        start_time = time.time()
        status = self.ompl_planner.solve(request, ompl_response, False)
        plan_time = time.time() - start_time

        if status.value() == -2:
            return None, request, 0.0
        if status.value() != 0:
            return None, request, plan_time

        return ompl_response, request, plan_time

    def _plan(self, program):
        request = ProcessPlanningRequest()
        request.name = FREESPACE_PLANNER_NAME
        request.instructions = Instruction(program)
        response = self.planning_server.run(request)
        self.planning_server.waitForAll()
        if response.interface.isSuccessful():
            results = flatten(response.getResults().as_CompositeInstruction())
        else:
            results = None
        return results

    def interpolate_trajectory(self, traj):
        return traj

    def init_aux(self):
        # self.time_optimal = TimeOptimalTrajectoryGeneration()
        pass

    def timeoptimal_trajectory(self, traj):
        pass
