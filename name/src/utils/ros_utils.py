"""

    ROS関係のコード

    go_home.py + replay.pyを統合したもの

"""

import subprocess
import time

import numpy as np
import rospy
from open_manipulator_msgs.msg import JointPosition
from open_manipulator_msgs.srv import SetJointPosition


def torque_on(port=0):
    """ トルクをONにする

    トルクを切るときはprocess.terminate()で

    Args:
        port(int): USBのポート番号

    """
    msg = 'roslaunch open_manipulator_controller '
    msg += f'open_manipulator_controller.launch usb_port:=/dev/ttyUSB{port}'
    process = subprocess.Popen("exec " + msg, shell=True)
    return process


def move_to_home_position():
    """ アームをホームポジションに移動させる関数

    Args:
        test(bool): M1のコードだと，データ取得時とテスト時でコードが異なるので合せるため

    """

    HOME_POSITION = np.array([-0.017, 0.05, 0.443, 1.131])
    replay = Replay()
    replay.call_service(HOME_POSITION)


class Replay:
    """

    関節に命令を送るクラス

    """

    def __init__(self, rate: float=9.0, path_time: float=0.05, split: int=10):
        self.rate = rate
        self.path_time = path_time
        self.split = split
        self.joint_state = []
        self.gripper_state = []

        self.arm_service = rospy.ServiceProxy(
            "goal_joint_space_path_from_present", SetJointPosition)

        self.previous_joint_state = np.array([-0.02607767, 0.240835, -0.11811652, 1.34223318])

        self.name = ['joint1', 'joint2', 'joint3', 'joint4']
        self.max_velocity_scaling_factor = 0.0
        self.max_acceleration_scaling_factor = 0.0
    
    def home(self):
        arm_service = rospy.ServiceProxy("/goal_joint_space_path", SetJointPosition)
        gripper_service = rospy.ServiceProxy("/goal_tool_control", SetJointPosition)
        state = np.array([-0.02607767, 0.240835, -0.11811652, 1.34223318, 0.01000])
        self.joint_state = state[:4]
        self.gripper_state = [state[-1]]
        # print(len(self.gripper_state))
        arm, gripper = self.create_msg()
        # gripper_service('', gripper, 0.3)
        arm_service('', arm, 1.5)
        
    def move_to_up(self):
        arm_service = rospy.ServiceProxy("goal_joint_space_path", SetJointPosition)
        self.joint_state = np.array([-0.01073787, 0.43258259, -1.04310691, 2.00000])
        arm, _ = self.create_msg()
        arm_service('', arm, 1.5)
        
    def close_gripper(self):
        gripper_service = rospy.ServiceProxy("/goal_tool_control", SetJointPosition)
        self.gripper_state = np.array([-0.01000])
        # print(len(self.gripper_state))
        _, gripper = self.create_msg()
        gripper_service('', gripper, 0.3)
        
    def open_gripper(self):
        gripper_service = rospy.ServiceProxy("/goal_tool_control", SetJointPosition)
        self.gripper_state = np.array([0.01000])
        # print(len(self.gripper_state))
        _, gripper = self.create_msg()
        gripper_service('', gripper, 0.3)

    def create_msg(self):
        return JointPosition(self.name, self.joint_state, self.max_velocity_scaling_factor, self.max_acceleration_scaling_factor),\
               JointPosition(['gripper'], self.gripper_state, self.max_velocity_scaling_factor, self.max_acceleration_scaling_factor)
        # return \
        #     JointPosition(
        #         self.name,
        #         self.joint_state,
        #         self.max_velocity_scaling_factor,
        #         self.max_acceleration_scaling_factor
        #     )

    def call_service(self, state):
        step = (state - self.previous_joint_state) / self.split #前にいた位置と次の目標の差をn分割したもの
        step.tolist()

        for i in range(self.split):
            try:
                self.joint_state = step
                self.arm_service('', self.create_msg(), self.path_time / self.split)

            except rospy.ServiceException as e:
                print(f'service call failed: {e}')

            rospy.sleep(1.0 / self.rate / self.split)
            
        self.previous_joint_state = state
       
if __name__ == '__main__':
    move_to_home_position()
