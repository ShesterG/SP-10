#30 September 2022
#frames from videos; https://github.com/Ai-Genome-Saksham/OpenCV/blob/main/OpenCV/%239%20Extracting%20Images%20from%20Video.py
#text from images;   https://github.com/bhadreshpsavani/ExploringOCR/blob/master/OCRusingTesseract.ipynb

#!sudo apt install tesseract-ocr # TODO : INSERT INSIDE REQUIREMENTS ?
#!pip install pytesseract        # TODO : INSERT INSIDE REQUIREMENTS ?

#import pytesseract
import random
import logging
import argparse
from pathlib import Path
import os
import cv2
import re
import shutil
import glob
import io
import pandas as pd
import numpy as np
import math
import sys
import pickle
import gzip
#from keras.preprocessing import image
#from PIL import Image
#from PIL import ImageChops
#from PIL import ImageEnhance
import tensorflow as tf
import torch
import json
from PIL import Image
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tarfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

#changes the frame rate of all the videos to 25
def main(args):
    lan = "xki"
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    mp_pose = mp.solutions.pose
    root_path = Path(args.save_path)
    types = ('*.gz','*.zip') # the tuple of file types
    verses_zip_files = []   
    for files in types:
        verses_zip_files.extend(root_path.glob(files))        
    print(len(verses_zip_files))    
    verses_list=[]
    verses_count = 0    
    for verse_zip_file in verses_zip_files:
        if str(verse_zip_file).endswith("tar.gz"):
            #print("found!")
            tar = tarfile.open(verse_zip_file, "r:gz")
            tar.extractall()
            tar.close()
        elif verse_zip_file.endswith("tar.gz"):
            tar = tarfile.open(verse_zip_file, "r:gz")
            tar.extractall()
            tar.close()
        #!tar -xvzf $verse_zip_file -C /content/Dataset 
        for root11, dirs11, files11 in os.walk(Path("/content/Dataset/content")):
          dirs1 = dirs11
          break   
        for root0, dirs0, files0 in os.walk(Path(f"/content/Dataset/content/{dirs1[0]}")):
          dirs = dirs0
          break          
        for d in dirs:          
          num_images = len(os.listdir(f"/content/Dataset/content/{dirs1[0]}/{d}"))
          mean_distance = 0
          verse_dict = {}
          verse_features = []
          verse_dict["name"] = d
          for i in range(1,num_images+1):
            frame_name = 'images' + str("{:05d}".format(i)) + '.png'
            frame = cv2.imread(f"/content/Dataset/content/{dirs1[0]}/{d}/{frame_name}")
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
              # Make detections
                  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
                  frame.flags.writeable = False                  # Image is no longer writeable
                  results = holistic.process(frame)              # Make prediction                  
                  pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
                  face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
                  lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
                  rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
                  

                  # a variant of your normalise method here   
                  left_shoulder = np.array([pose[44],pose[45],pose[46]])
                  right_shoulder = np.array([pose[48],pose[49],pose[50]])
                  shoulder_width = (((right_shoulder - left_shoulder)**2).sum())**0.5
                  mean_distance = mean_distance + shoulder_width
                  center_pose = np.array([(left_shoulder[0]+right_shoulder[0])/2,
                                    (left_shoulder[1]+right_shoulder[1])/2,
                                      0,0])
                  center_pose = np.tile(center_pose,33)
                  pose = pose - center_pose 
                  center = np.array([(left_shoulder[0]+right_shoulder[0])/2,
                                    (left_shoulder[1]+right_shoulder[1])/2,
                                      0])
                  center_face = np.tile(center,468)
                  face = face - center_face

                  center_hands = np.tile(center,21)
                  lh = lh - center_hands
                  rh = rh - center_hands
                  


                  image_feature = np.concatenate([pose, face, lh, rh])
                  image_feature = torch.from_numpy(image_feature)
                  image_feature = torch.flatten(image_feature)
                  verse_features.append(image_feature)  
          #mean_distance = mean_distance / num_images
          #verse_features = verse_features * (1 / mean_distance)
          verse_dict["sign"]  = verse_features
          verses_list.append(verse_dict)
          verses_count = verses_count + 1
          print(f"Verse {verses_count} done.")
        shutil.rmtree(f"/content/Dataset/content")  

    file = gzip.GzipFile(f"/content/drive/MyDrive/Sign_Language_Videos/dataset/{lan}240.dataset", 'wb')
    file.write(pickle.dumps(verses_list,0))
    file.close()

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--save_path", help="Path to save the video", default="./videos", type=str
    )
    args = parse.parse_args()
    logging.info("Start converting dataset")
    main(args)
    logging.info("Converting dataset finished")
