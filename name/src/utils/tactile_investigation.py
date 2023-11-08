import numpy as np
import cv2
from catkin_ws.src.liftobj.script.utils.digit_utils import get_tactile_image

# x:87, y:109, R:250, G:85, B:100
# x:212, y: 152, R:60, G170, B:60

# X = 212

# Y = 152


while True:
    image_data = get_tactile_image() #(320, 240, 3)
    cv2.imshow(f"DV", image_data)
    

# a = image_data[Y, X, [0, 1, 2]]

# print(a)

