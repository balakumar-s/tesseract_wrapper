from robometrics.statistics import Statistic, TrajectoryGroupMetrics, TrajectoryMetrics

from robometrics.datasets import demo_raw, motion_benchmaker_raw, mpinets_raw
from tqdm import tqdm

from tesseract_wrapper.planner import TesseractPlanner, TesseractConfig

from tesseract_wrapper.util_file import (
    get_content_path,
    join_path,
    load_yaml,
    write_yaml,
)
import time
import numpy as np
import matplotlib.pyplot as plt


from trajectory_smoothing import TrajectorySmoother
import torch
import argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional
import torch

def fd_tensor(p: torch.Tensor, dt: torch.Tensor):
    out = ((torch.roll(p, -1, -2) - p) * (1 / dt).unsqueeze(-1))[..., :-1, :]
    return out

@dataclass
class JointState:
    position: torch.Tensor
    velocity: Optional[torch.Tensor] = None
    acceleration: Optional[torch.Tensor] = None
    jerk: Optional[torch.Tensor] = None

    def calculate_fd_from_position(self,dt: torch.Tensor):
        self.velocity = fd_tensor(self.position, dt)
        self.acceleration = fd_tensor(self.velocity, dt)
        self.jerk = fd_tensor(self.acceleration, dt)
        return self
    @staticmethod
    def from_numpy(position):
        return JointState(torch.as_tensor(position, device="cuda:0"))


def smooth_totg_trajectory(position_trajectory: np.ndarray, interpolation_dt):
    dof = position_trajectory.shape[-1]
    n = position_trajectory.shape[-2]
    max_velocity = (
        np.ravel([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]) - 0.02
    )
    max_acceleration = np.zeros((dof), dtype=np.float32) + 15.0
    max_deviation = 0.01

    trajectory_sm = TrajectorySmoother(
        dof, max_velocity, max_acceleration, max_deviation
    )
    # trajectory_sm.test()
    # return
    st_time = time.time()
    result = trajectory_sm.smooth_interpolate(
        position_trajectory, interpolation_dt=interpolation_dt, traj_dt=0.001
    )
    end_time = time.time() - st_time
    if result.success:
        return {
            "position": result.position,
            "velocity": result.velocity,
            "acceleration": result.acceleration,
            "jerk": result.jerk,
            "dt": interpolation_dt,
            "solve_time": end_time,
        }


def fd_trajectory(position_trajectory: np.ndarray, interpolation_dt: float = 0.025):
    max_velocity = (
        np.ravel([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]) - 0.02
    )

    js = JointState.from_numpy(position=position_trajectory)
    js.calculate_fd_from_position(torch.as_tensor(interpolation_dt, device="cuda"))

    vel = torch.max(torch.abs(js.velocity.view(-1, 7)), dim=0)[0] / torch.as_tensor(
        max_velocity, device="cuda"
    )
    raw_dt = interpolation_dt * torch.max(vel)
    js = JointState.from_numpy(position=position_trajectory)
    js.calculate_fd_from_position(raw_dt)

    return {
        "position": js.position.cpu().squeeze().numpy().tolist(),
        "velocity": js.velocity.cpu().squeeze().numpy().tolist(),
        "acceleration": js.acceleration.cpu().squeeze().numpy().tolist(),
        "jerk": js.jerk.cpu().squeeze().numpy().tolist(),
        "dt": raw_dt.item(),
        "solve_time": 0.0,
    }


def plot_traj(act_seq: JointState, dt=0.25, title="", save_path="plot.png"):
    fig, ax = plt.subplots(4, 1, figsize=(5, 8), sharex=True)
    t_steps = np.linspace(0, act_seq.position.shape[0] * dt, act_seq.position.shape[0])
    for i in range(act_seq.position.shape[-1]):
        ax[0].plot(t_steps, act_seq.position[:, i].cpu(), "-", label=str(i))
        ax[1].plot(
            t_steps[: act_seq.velocity.shape[0]], act_seq.velocity[:, i].cpu(), "-"
        )
        ax[2].plot(
            t_steps[: act_seq.acceleration.shape[0]],
            act_seq.acceleration[:, i].cpu(),
            "-",
        )
        ax[3].plot(t_steps[: act_seq.jerk.shape[0]], act_seq.jerk[:, i].cpu(), "-")
    ax[0].set_title(title + "{:.3f}".format(dt))
    ax[3].set_xlabel("Time(s)")
    ax[3].set_ylabel("Jerk rad. s$^{-3}$")
    ax[0].set_ylabel("Position rad.")
    ax[1].set_ylabel("Velocity rad. s$^{-1}$")
    ax[2].set_ylabel("Acceleration rad. s$^{-2}$")
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()
    # ax[0].legend(loc="upper right")
    ax[0].legend(bbox_to_anchor=(0.5, 1.6), loc="upper center", ncol=4)
    plt.tight_layout()


def plot_position_traj(act_seq: JointState, dt=0.25, title="", save_path="plot.png"):
    fig, ax = plt.subplots(4, 1, figsize=(5, 8), sharex=True)
    t_steps = np.linspace(0, act_seq.position.shape[0] * dt, act_seq.position.shape[0])
    for i in range(act_seq.position.shape[-1]):
        ax[0].plot(t_steps, act_seq.position[:, i].cpu(), "-", label=str(i))
        ax[1].plot(
            t_steps[: act_seq.velocity.shape[0]], act_seq.velocity[:, i].cpu(), "-"
        )
        ax[2].plot(
            t_steps[: act_seq.acceleration.shape[0]],
            act_seq.acceleration[:, i].cpu(),
            "-",
        )
        ax[3].plot(t_steps[: act_seq.jerk.shape[0]], act_seq.jerk[:, i].cpu(), "-")
    ax[0].set_title(title + "_" + "{:.3f}".format(dt))
    ax[3].set_xlabel("Time(s)")
    ax[3].set_ylabel("Jerk rad. s$^{-3}$")
    ax[0].set_ylabel("Position rad.")
    ax[1].set_ylabel("Velocity rad. s$^{-1}$")
    ax[2].set_ylabel("Acceleration rad. s$^{-2}$")
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()
    # ax[0].legend(loc="upper right")
    ax[0].legend(bbox_to_anchor=(0.5, 1.6), loc="upper center", ncol=4)
    plt.tight_layout()


def load_tesseract_planner(use_25=True):
    if use_25:
        robot_urdf = join_path(get_content_path(), "franka_description/franka_p25.urdf")
    else:
        robot_urdf = join_path(get_content_path(), "franka_description/franka.urdf")

    robot_srdf = join_path(get_content_path(), "franka_description/franka.srdf")
    tess_config = TesseractConfig(robot_urdf, robot_srdf)
    planner = TesseractPlanner(tess_config)
    planner.load_robot()
    return planner


def call_planner(
    planner, start_q, goal_ik, n_attempts=100, timeout=60, interpolation_dt=0.025,
    graph_mode: bool = False,
):
    time_dict = {"graph_time": 0, "trajopt_time": 0, "total_time": 0}
    start_time = time.time()
    ik_i = 0
    for i in range(n_attempts):
        goal_q = goal_ik[ik_i]
        ik_i += 1

        result, debug = planner.plan_cspace(
            np.float64(start_q), np.float64(goal_q), run_trajopt=not graph_mode
        )

        time_elapsed = time.time() - start_time
        debug["attempts"] = i
        time_dict["graph_time"] += debug["graph_time"]
        time_dict["total_time"] += debug["total_time"]
        time_dict["trajopt_time"] += debug["trajopt_time"]
        if result is not None:
            debug["smooth_graph"] = smooth_totg_trajectory(
                debug["graph_raw_traj"], interpolation_dt
            )
            if "trajopt_raw_traj" not in debug:
                debug["trajopt_raw_traj"] = debug["graph_raw_traj"]
            debug["smooth_trajopt"] = smooth_totg_trajectory(
                debug["trajopt_raw_traj"], interpolation_dt
            )
            debug["fd_graph"] = fd_trajectory(debug["graph_raw_traj"], interpolation_dt)
            debug["fd_trajopt"] = fd_trajectory(
                debug["trajopt_raw_traj"], interpolation_dt
            )

            break
        if time_elapsed > timeout:
            break
        if ik_i >= len(goal_ik):
            ik_i = 0
    debug["trajopt_time"] = time_dict["trajopt_time"]
    debug["total_time"] = time_dict["total_time"]
    debug["graph_time"] = time_dict["graph_time"]

    return result, debug


def run_tesseract(demo=True, collision_distance=-0.015, args=None):
    plot = False
    dataset = [motion_benchmaker_raw, mpinets_raw][:]
    if demo:
        dataset = [demo_raw]
    # dataset = [get_mpinets_dataset()]
    dataset_combined = []
    for k_f in dataset:
        mpinets_data = False
        k = k_f()

        if "dresser_task_oriented" in list(k.keys()):
            mpinets_data = True

        planner = load_tesseract_planner(use_25=mpinets_data)

        a_list = []

        for key, v in tqdm(k.items()):
            # if "cubby_task_oriented" not in key:
            #    continue
            # if key not in ["table_under_pick_panda"]:
            #    continue
            scene_problems = k[key]#[:6]  # [:2]
            i = 0
            m_list = []
            coll_distance = collision_distance
            for problem in tqdm(scene_problems, leave=False):
                if "goal_ik" not in problem.keys():
                    continue
                if problem["collision_buffer_ik"] < 0.0:
                    continue

                obstacles = deepcopy(problem["obstacles"])
                planner.load_world(obstacles, coll_distance)
                start_q = problem["start"]
                goal_q = problem["goal_ik"]
                result, debug = call_planner(planner, start_q, goal_q, graph_mode=args.graph)
                problem_name = key + "_" + str(i)
                i += 1
                problem["solution"] = None
                if result is not None:
                    m_list.append(
                        TrajectoryMetrics(
                            skip=False,
                            success=True,
                            solve_time=debug["total_time"],
                            collision=False,
                            joint_limit_violation=False,
                            self_collision=False,
                            physical_violation=False,
                            position_error=0.0,
                            orientation_error=0.0,
                            eef_orientation_path_length=0.0,
                            eef_position_path_length=0.0,
                            motion_time=0.0, #(result.shape[0]-1) * 0.02
                        )
                    )

                    problem["solution"] = result.tolist()
                    problem["solution_debug"] = debug
                    if plot:
                        graph_traj = debug["smooth_graph"]
                        # run through smoothing:
                        graph_js = JointState(
                            position=torch.as_tensor(graph_traj["position"]),
                            velocity=torch.as_tensor(graph_traj["velocity"]),
                            acceleration=torch.as_tensor(graph_traj["acceleration"]),
                            jerk=torch.as_tensor(graph_traj["jerk"]),
                        )

                        plot_traj(graph_js, graph_traj["dt"], title="graph dt=")
                        plt.savefig("graph_totg.pdf")
                        plt.close()
                        graph_traj = debug["fd_graph"]
                        # run through smoothing:
                        graph_js = JointState(
                            position=torch.as_tensor(graph_traj["position"]),
                            velocity=torch.as_tensor(graph_traj["velocity"]),
                            acceleration=torch.as_tensor(graph_traj["acceleration"]),
                            jerk=torch.as_tensor(graph_traj["jerk"]),
                        )

                        plot_traj(graph_js, graph_traj["dt"], title="graph dt=")
                        plt.savefig("graph_fd.pdf")
                        plt.close()
                        trajopt_traj = debug["fd_trajopt"]

                        trajopt_js = JointState(
                            position=torch.as_tensor(trajopt_traj["position"]),
                            velocity=torch.as_tensor(trajopt_traj["velocity"]),
                            acceleration=torch.as_tensor(trajopt_traj["acceleration"]),
                            jerk=torch.as_tensor(trajopt_traj["jerk"]),
                        )

                        plot_traj(trajopt_js, trajopt_traj["dt"], title="trajopt dt=")
                        plt.savefig("trajopt_fd.pdf")
                        plt.close()
                        trajopt_traj = debug["smooth_trajopt"]

                        trajopt_js = JointState(
                            position=torch.as_tensor(trajopt_traj["position"]),
                            velocity=torch.as_tensor(trajopt_traj["velocity"]),
                            acceleration=torch.as_tensor(trajopt_traj["acceleration"]),
                            jerk=torch.as_tensor(trajopt_traj["jerk"]),
                        )

                        plot_traj(trajopt_js, trajopt_traj["dt"], title="trajopt dt=")
                        plt.savefig("trajopt_totg.pdf")
                        plt.close()
                else:
                    print("fail")
                    m_list.append(TrajectoryMetrics())
            t = TrajectoryGroupMetrics.from_list(m_list)
            a_list += m_list
            #print(key, t.solve_time, t.success)

        t = TrajectoryGroupMetrics.from_list(a_list)
        #print(t.solve_time, t.success, t.motion_time)
        dataset_combined += a_list
        if mpinets_data:
            write_yaml(k, join_path(args.save_path, args.file_name + "_mpinets.yaml"))
        else:
            write_yaml(k, join_path(args.save_path,args.file_name + "_mb.yaml"))
    t = TrajectoryGroupMetrics.from_list(dataset_combined)

    if args.kpi:
        kpi_data = {"Success": t.success, "Planning Time": float(t.solve_time.mean)}
        write_yaml(kpi_data, join_path(args.save_path, args.file_name + ".yml"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default=".",
        help="path to save file",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="tesseract_",
        help="File name prefix to use to save benchmark results",
    )
    parser.add_argument(
        "--collision_buffer",
        type=float,
        default=-0.015,
        help="Robot collision buffer",
    )

    parser.add_argument(
        "--graph",
        action="store_true",
        help="When True, runs only geometric planner",
        default=False,
    )
    parser.add_argument(
        "--kpi",
        action="store_true",
        help="When True, saves minimal metrics",
        default=False,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="When True, runs only on small dataaset",
        default=False,
    )
    args = parser.parse_args()
    print(args)
    run_tesseract(args=args,demo=args.demo, collision_distance=args.collision_buffer)
