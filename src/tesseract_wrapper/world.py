from tesseract_robotics.tesseract_common import (
    Isometry3d,
    Translation3d,
    Quaterniond,
)
import yaml
from tesseract_robotics.tesseract_environment import (
    AddLinkCommand,
    Commands,
)
from tesseract_robotics.tesseract_scene_graph import (
    Link,
    Joint,
    Visual,
    Collision,
    JointType_FIXED,
)
from tesseract_robotics.tesseract_geometry import Sphere, Box, Cylinder

# from tesseract_robotics_viewer import TesseractViewer
import numpy as np


def get_link(primitive, params, reference_frame, l_name):
    # create a link and return add link command:
    if primitive == "cuboid":
        return create_cuboid(
            params["pose"], params["dims"], reference_frame, link_name="env_" + l_name
        )
    elif primitive == "sphere":
        return create_sphere(
            params["position"],
            params["radius"],
            reference_frame,
            link_name="env_" + l_name,
        )
    else:
        raise NotImplementedError


def create_sphere(position, radius, ref_frame, link_name):
    link = Link(link_name)
    visual = Visual()
    pose = (
        Isometry3d.Identity()
        * Translation3d(position[0], position[1], position[2])
        * Quaterniond(1.0, 0, 0.0, 0)
    )
    visual.origin = pose
    visual.geometry = Sphere(radius)
    link.visual.append(visual)
    collision = Collision()
    collision.origin = pose
    collision.geometry = Sphere(radius)
    link.collision.append(collision)
    joint = Joint("j_" + link_name)
    joint.parent_link_name = ref_frame
    joint.child_link_name = link_name
    joint.type = JointType_FIXED
    cmd = AddLinkCommand(link, joint)
    return cmd


def create_cuboid(pose, dims, ref_frame, link_name):
    link = Link(link_name)
    visual = Visual()
    pose = (
        Isometry3d.Identity()
        * Translation3d(pose[0], pose[1], pose[2])
        * Quaterniond(pose[3], pose[4], pose[5], pose[6])
    )
    visual.origin = pose
    visual.geometry = Box(dims[0], dims[1], dims[2])
    visual.material.color = np.ravel([0.6, 0.6, 0.0, 1.0])
    link.visual.append(visual)
    collision = Collision()
    collision.origin = pose
    collision.geometry = Box(dims[0], dims[1], dims[2])
    link.collision.append(collision)
    joint = Joint("j_" + link_name)
    joint.parent_link_name = ref_frame
    joint.child_link_name = link_name
    joint.type = JointType_FIXED
    cmd = AddLinkCommand(link, joint)
    return cmd


def create_cylinder(pose, height, radius, ref_frame, link_name):
    link = Link(link_name)
    visual = Visual()
    pose = (
        Isometry3d.Identity()
        * Translation3d(pose[0], pose[1], pose[2])
        * Quaterniond(pose[3], pose[4], pose[5], pose[6])
    )
    visual.origin = pose
    visual.geometry = Cylinder(radius, height)
    visual.material.color = np.ravel([0.6, 0.6, 0.0, 1.0])
    link.visual.append(visual)
    collision = Collision()
    collision.origin = pose
    collision.geometry = Cylinder(radius, height)
    link.collision.append(collision)
    joint = Joint("j_" + link_name)
    joint.parent_link_name = ref_frame
    joint.child_link_name = link_name
    joint.type = JointType_FIXED
    cmd = AddLinkCommand(link, joint)
    return cmd


def get_tesseract_obstacle_commands(world_params):
    # create a set of commands:
    reference_frame = "world"
    cmd_list = Commands()

    if "cuboid" in world_params:
        for k in world_params["cuboid"].keys():
            prim = world_params["cuboid"][k]
            cmd = create_cuboid(prim["pose"], prim["dims"], reference_frame, k)
            cmd_list.append(cmd)
    if "cylinder" in world_params:
        for k in world_params["cylinder"].keys():
            prim = world_params["cylinder"][k]
            cmd = create_cylinder(
                prim["pose"], prim["height"], prim["radius"], reference_frame, k
            )
            cmd_list.append(cmd)

    # return commands
    return cmd_list
