import face_recognition 
import os, sys
import cv2
from face_recognition_extra import FaceRecognition
import numpy as np
import math

        
if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
