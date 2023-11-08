import cv2
import numpy as np
import pyrealsense2 as rs


def get_pipeline():
    """

    カメラの設定とpipelineの取得

    reference:
        https://qiita.com/tom_eng_ltd/items/ae5f27b2d17edb1d47e5

    """
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline = rs.pipeline()
    pipeline.start(config)
    return pipeline


def get_image(pipeline, trim: bool = True, out_size: tuple = (32, 32)):
    """

    pipelineから画像を取得

    """
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        return None

    color_image = np.asanyarray(color_frame.get_data())

    if trim: 
        x, y = 85, 40
        h, w = 475, 445
        color_image = color_image[y:y + h, x:x + w]

    color_image = cv2.resize(color_image, dsize=(out_size[0], out_size[1]))

    # 色配置の変換 BGR→RGB
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    color_image = color_image.reshape([1, out_size[0], out_size[1], 3])

    return color_image
