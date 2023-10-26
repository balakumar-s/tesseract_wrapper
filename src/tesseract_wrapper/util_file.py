# Standard Library
import os
from typing import Dict

# Third Party
import yaml
from yaml import SafeLoader


def join_path(path1, path2):
    if isinstance(path2, str):
        return os.path.join(path1, path2)
    else:
        return path2


# get paths
def get_module_path():
    path = os.path.dirname(__file__)
    return path


def get_content_path():
    root_path = get_module_path()
    path = os.path.join(root_path, "content")
    return path


def load_yaml(file_path):
    if isinstance(file_path, str):
        with open(file_path) as file_p:
            yaml_params = yaml.load(file_p, Loader=SafeLoader)
    else:
        yaml_params = file_path
    return yaml_params


def write_yaml(data: Dict, file_path):
    with open(file_path, "w") as file:
        yaml.dump(data, file)
