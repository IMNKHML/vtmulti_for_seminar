import numpy as np
from digit_interface import Digit


def get_tactile_image(d):
    
    # d = Digit("D20537") # Unique serial number
    # d.connect()
    frame = d.get_frame()
    # d.disconnect()
    
    #print(type(frame))
    #print(frame.shape)
    
    return frame
