import face_recognition_extra
import os, sys
import cv2

import numpy as np
import math

def face_confidence(face_distance, face_match_theshold =  0.4):
    range = (1.0 - face_match_theshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)
    
    if face_distance > face_match_theshold:
       return str(round(linear_val * 100)) + "%"
    else:
        value = (linear_val  + ((1.0 - linear_val)  * math.pow((linear_val - 0.5) * 2,  0.2))) * 100
        return str(round(value)) + "%"