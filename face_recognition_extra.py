from face_confidence import face_confidence
import face_recognition
import os, sys
import cv2

import numpy as np
import math

class FaceRecognition:
    face_locations = []
    face_encodings =  []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    
    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('/Users/trongphan/Desktop/Facial_recognition/faces'):
            face_image =  face_recognition.load_image_file(f'/Users/trongphan/Desktop/Facial_recognition/faces/{image}')
            face_encoding = face_recognition.api.face_encodings(face_image)[0]
            
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        
        print(self.known_face_names)
        
        
    def run_recognition(self):
        video_capture = cv2.VideoCapture(1)

        if not video_capture.isOpened():
            sys.exit('Video where????')
            
        while True:
            ret, frame = video_capture.read()
            if self.process_current_frame:
                small_frame = cv2.resize(frame , (0, 0), fx = 0.25, fy = 0.25)
                # rgb_small_frame = small_frame[:, :, ::-1]
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
                
                # print("RGB SMALL FRAME")
                # print(rgb_small_frame)
                
                #find all faces in frame
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.api.face_encodings(rgb_small_frame, self.face_locations)
                
                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = "Unknown"
                    
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                        
                    self.face_names.append(f'{name} ({str(confidence)})')
                    
            self.process_current_frame = not self.process_current_frame
            
            #Display annotations
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                
                top *= 4
                right *= 4 
                left *=  4
                bottom *= 4
                
                cv2.rectangle(frame,  (left, top), (right,bottom), (0,0,255), 2)
                cv2.rectangle(frame,  (left, bottom - 35), (right,bottom), (0,0,255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
                # print("Here is the name")
                # print(name)

            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) ==  ord('q'):
                break
            
        video_capture.release()
        cv2.destroyAllWindows()
        