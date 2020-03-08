"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


__version__ = '0.4'

import os
import argparse
import tarfile
import tempfile
from deployman.logger import init_logger
from deployman.config import ConfigReader, ComponentFactory, Component
from deployman.ui import UserInterface

logger = init_logger('WARNING')


# main class
class DeploymentManager:
    def __init__(self, args, selected_targets, components):
        self.args = args
        self.selected_targets = selected_targets
        self.components = components
        self.dependencies = []
        self.mandatory_components = []

    def get_dependencies(self):
        dependencies_names = []
        logger.debug("Updating dependencies...")
        for target in self.selected_targets:
            if hasattr(target, 'dependencies'):
                dependencies_names.extend(target.dependencies)
        # remove duplications
        dependencies_names = list(dict.fromkeys(dependencies_names))
        for dependency in dependencies_names:
            for _target in self.components:
                _target: Component
                if _target.name == dependency:
                    if _target.is_exist():
                        self.dependencies.append(_target)
                    else:
                        FileNotFoundError("Dependency {} not available.".format(_target.name))

    def get_mandatory_component(self):
        for _target in self.components:
            _target: Component
            if hasattr(_target, 'mandatory'):
                if _target.is_exist():
                    self.mandatory_components.append(_target)
                else:
                    FileNotFoundError("Mandatory component {} not available.".format(_target.name))

    @staticmethod
    def make_tarfile(out_file_name, target_dir, source_dir):
        with tarfile.open(os.path.join(target_dir, out_file_name), "w:gz") as tar:
            tar.add(source_dir, arcname="openvino")
        logger.setLevel('INFO')
        logger.info("Deployment archive is ready."
                    "You can find it here:\n\t{}".format(os.path.join(target_dir, out_file_name)))

    def process(self):
        # get dependencies if have
        self.get_dependencies()
        # get mandatory components
        self.get_mandatory_component()

        with tempfile.TemporaryDirectory() as tmpdirname:
            for target in self.selected_targets:
                target: Component
                target.copy_files(tmpdirname)
            if len(self.dependencies) > 0:
                for dependency in self.dependencies:
                    dependency: Component
                    dependency.copy_files(tmpdirname)
            if len(self.mandatory_components) > 0:
                for target in self.mandatory_components:
                    target: Component
                    target.copy_files(tmpdirname)
            if self.args.user_data and os.path.exists(self.args.user_data):
                from shutil import copytree
                copytree(self.args.user_data,
                         os.path.join(tmpdirname, os.path.basename(self.args.user_data)),
                         symlinks=True)
            self.make_tarfile(self.args.archive_name + ".tar.gz",
                              self.args.output_dir, tmpdirname)


def main():
    # read main config
    cfg = ConfigReader(logger)

    # here we store all components
    components = []

    for component in cfg.components:
        components.append(ComponentFactory.create_component(component,
                                                            cfg.components[component],
                                                            logger))

    # list for only available components
    available_targets = []
    help_msg = ''

    for component in components:
        if component.is_exist() and not component.invisible:
            available_targets.append(component)
            help_msg += "{} - {}\n".format(component.name, component.ui_name)

    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--targets", nargs="+", help="List of targets."
                                                     "Possible values: \n{}".format(help_msg))
    parser.add_argument("--user_data", type=str, help="Path to user data that will be added to "
                                                      "the deployment package", default=None)
    parser.add_argument("--output_dir", type=str, help="Output directory for deployment archive",
                        default=os.getenv("HOME", os.path.join(os.path.join(
                            os.path.dirname(__file__), os.pardir))))
    parser.add_argument("--archive_name", type=str, help="Name for deployment archive",
                        default="openvino_deploy_package", )
    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)
    args = parser.parse_args()

    selected_targets = []
    ui = UserInterface(__version__, args, available_targets, logger)
    if not args.targets:
        ui.run()
        selected_targets = ui.get_selected_targets()
        args = ui.args
    else:
        for target in args.targets:
            if not any(target == _target.name for _target in available_targets):
                raise ValueError("You input incorrect target. {} is not available.".format(target))
            for _target in available_targets:
                if _target.name == target:
                    selected_targets.append(_target)

    _manager = DeploymentManager(args, selected_targets, components)
    _manager.process()

