import datetime
import os
import time
from glob import glob
from digit_interface import Digit

import numpy as np
import rospy
from sensor_msgs.msg import Image, JointState

from src.utils.camera_utils import get_image, get_pipeline
from src.utils.digit_utils import get_tactile_image

class Saver:
    """ データの記録・保存を行うクラス

    Args:
        fps(int): 1秒間に取得する画像の枚数
        seq_len(int): 画像を撮影する総枚数

    触覚センサは右側を使用（予定）

    """

    def __init__(self, fps, seq_len):
        self.seq_len = seq_len
        self.current_pose = None
        self.rate = rospy.Rate(fps)
        self.pipeline_of_camera = get_pipeline()
        self.joint_pose = np.empty([seq_len, 5])#joint1-4, gripper
        self.image_states = np.empty([seq_len, 32, 32, 3])
        # self.left_tactile_states = np.empty([seq_len, 320, 240, 3])
        self.right_tactile_states = np.empty([seq_len, 320, 240, 3])

    def callback(self, data):
        self.current_pose = data.position

    def get_array(self, verbose=True):
        counter = 0
        # d_l = Digit("D20537") # left tactile data
        d_r = Digit("D20542") # right tactile data #new
        # d_l.connect()
        d_r.connect()
        
        while counter < self.seq_len:
            rospy.Subscriber(
                '/joint_states', JointState, self.callback, queue_size=1)
            if self.current_pose is None:
                continue

            # self.left_tactile_states[counter] = get_tactile_image(d_l)
            self.right_tactile_states[counter] = get_tactile_image(d_r)
            self.image_states[counter] = get_image(self.pipeline_of_camera)
            self.joint_pose[counter] = np.array(self.current_pose)

            if verbose:
                print(f'step: {counter}', end=' ')
                print(f'current pose: {self.joint_pose[-1]}', end=' ')
                print(f'color image: {self.image_states[counter]}')

            self.rate.sleep()
            counter += 1
        # d_l.disconnect()
        d_r.disconnect()
        self.pipeline_of_camera.stop()

    def save_array(self, joint_save_dir, image_save_dir, right_tactile_save_dir):
    # def save_array(self, joint_save_dir, image_save_dir, left_tactile_save_dir, right_tactile_save_dir):
        self.joint_pose = np.array(self.joint_pose)


        np.save(joint_save_dir, self.joint_pose)
        print(f'array shape: {self.joint_pose.shape}')
        print(f'saved data at: {joint_save_dir}')

        np.save(image_save_dir, self.image_states)
        print(f"array shape: {self.image_states.shape}")
        print(f"saved data at: {image_save_dir}")
        
        # np.save(left_tactile_save_dir, self.left_tactile_states)
        # print(f"array shape: {self.left_tactile_states.shape}")
        # print(f"saved data at: {left_tactile_save_dir}")
        
        np.save(right_tactile_save_dir, self.right_tactile_states)
        print(f"array shape: {self.right_tactile_states.shape}")
        print(f"saved data at: {right_tactile_save_dir}")


def run_saver(dirname):

    rospy.init_node("saver")

    while True:
        saver = Saver(fps=5, seq_len=80)

        start_flag = input("press ENTER to start / q to quit: ")
        if start_flag.lower() == "q":
            print("quitting")
            break

        saver.get_array()

        save_flag = input("save data? 1~20 / n : ")
        if save_flag.lower() == 'n':
            continue

        elif 1 <= int(save_flag) <= 20:
            img_num = len(glob(f'{dirname}/*{save_flag}_[0-9]*.npy')) // 3 #取得するデータの種類によって変える
            joint_save_dir = f'{dirname}/raw_joint_state_{save_flag}_{img_num}.npy'
            image_save_dir = f'{dirname}/raw_image_state_{save_flag}_{img_num}.npy'
            # left_tactile_save_dir = f'{dirname}/raw_left_tactile_state_{save_flag}_{img_num}.npy'
            right_tactile_save_dir = f'{dirname}/raw_right_tactile_state_{save_flag}_{img_num}.npy'
            saver.save_array(joint_save_dir, image_save_dir, right_tactile_save_dir)
            # saver.save_array(joint_save_dir, image_save_dir, left_tactile_save_dir, right_tactile_save_dir)

        else:
            print(save_flag + "is incorrect")

if __name__ == "__main__":

    username = 'name'

    now = datetime.datetime.now()
    dirname = os.path.join(f'{username}/data/raw_data', f'{now.month}_{now.day}')
    os.makedirs(dirname, exist_ok=True)

    run_saver(dirname)
