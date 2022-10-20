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
from keras.preprocessing import image
#from keras.layers import merge, Input
from PIL import Image
from PIL import ImageChops
from PIL import ImageEnhance
import tensorflow as tf
import torch
import json
from PIL import Image
#from torchvision import transforms
#from efficientnet_pytorch import EfficientNet


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
#changes the frame rate of all the videos to 25
def main(args):

    lan = "ase"  
    root_path = Path(args.save_path)
    videos_path = Path(args.save_path + f"/{lan}_videos")
    verses_num = 0
    
    if Path(f"{root_path}/{lan}_videospath.txt").exists():
        print(f"{lan}_videospath.txt exists.")
        files_grabbed = []
        with open(f"{root_path}/{lan}_videospath.txt", 'r') as filehandle:
            for line in filehandle:
                # Remove linebreak which is the last character of the string
                curr_place = line[:-1]
                # Add item to the list
                files_grabbed.append(curr_place)
    else:
        files_grabbed = sorted(videos_path.glob('*.mp4'), key=os.path.getmtime)
        with open(f"{root_path}/{lan}_videospath.txt", 'w') as filehandle:
            for listitem in files_grabbed:
                filehandle.write(f'{listitem}\n')

    #Emodel = EfficientNet.from_pretrained('efficientnet-b0')
    verses_list=[]
    video_i = 1
    
    step = 1/25    
    for videopath in files_grabbed[video_i-1:video_i+100]:
        #vid = cv2.VideoCapture(str(video_path))
        refB = str(videopath).split('/')[-1].split('.')[0]
        os.system(f"ffprobe -i {videopath} -print_format default -show_chapters -loglevel error > {videos_path}/{refB}.json 2>&1")
        
        with open(f"{videos_path}/{refB}.json", "r") as infile:
            data = infile.read()
        data = data.replace("\n", "|")
        data = data.replace("|[/CHAPTER]|", "\n")
        colnames=["CHAPTER","id","time_base","start","start_time","end","end_time","title"]
        dataIO = io.StringIO(data)
        df = pd.read_csv(dataIO, sep="|", names=colnames, header=None)
        df.drop(['CHAPTER', 'id', 'time_base', 'start', 'end'], axis=1, inplace=True)
        try:
            df['LastDigit'] = [x.strip()[-1] for x in df['title']]
        except AttributeError:
            print(f"SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED Video {video_i} - {refB} done.")
            continue
        df = df[df['LastDigit'].str.isdigit()]
        df.drop(['LastDigit'], axis=1, inplace=True)
        df["start_time"] = df["start_time"].str.replace("start_time=", "")
        df["end_time"] = df["end_time"].str.replace("end_time=", "")
        df["title"] = df["title"].str.replace("TAG:title=", "")
        #df['title'] = df['title'].str.replace('.', '_') 
        #df['title'] = df['title'].str.replace(':', '_')
        #df['title'] = df['title'].str.replace(' ', '_')        
        verse_dict = {}
        video = cv2.VideoCapture(str(videopath))
        for index, row in df.iterrows():           
            #print(row["Name"], row["Age"])
            #opencv_method
            """
            verse_dict["video_name"] = refB
            verse_dict["name"] = row["title"]
            verse_dict["signer"] = "Signer8"            
            verse_dict["duration"] = float(float(row["end_time"]) - float(row["start_time"]))
            verse_dict["text"] = "Verse Text"
            """
            #video = cv2.VideoCapture(str(videopath))
            currentframe = 1
            #step = 1/25
            #enet_feature_list=[]
            for current_second in np.arange(math.ceil(float(row["start_time"])), math.floor(float(row["end_time"])), step):
              t_msec = 1000*(current_second)
              video.set(cv2.CAP_PROP_POS_MSEC, t_msec)
              success, frame = video.read()
              if success:
                #image_save_start
                verse_path = f'{root_path}/{lan}_verses/{row["title"]}'
                # creates folder with verse name if it doesn't yet exists. 
                try:
                    # creates folder with verse name if it doesn't yet exists. 
                    verse_path = Path(verse_path)  # TODO FY
                    if not os.path.exists(verse_path):
                        os.makedirs(verse_path)           
                # if not created then raise error
                except OSError:
                    print('Error: Creating directory of data')     
                                                      
                name = 'images' + str("{:05d}".format(currentframe)) + '.png' #TODO SHESTER : NOT VERY SURE OF THE f"{verse}
                #print('Creating...' + name)
                image_path = f"{verse_path}/{name}"
                # writing the extracted images
                cv2.imwrite(image_path,cv2.resize(frame,(320, 240)))
                currentframe += 1
                #image_save_end
            verses_num += 1
    
        print(f"Video {video_i} - {refB} done.")
        video_i += 1
        video.release()
        cv2.destroyAllWindows() 
    print(f"COMPLETED. Total of {verses_num} verses folders generated. ")
    #vfile = gzip.GzipFile("/content/SP-10/dataset/GSL240.dataset", 'wb')
    #vfile.write(pickle.dumps(verses_list,0))
    #vfile.close()
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--save_path", help="Path to save the video", default="./videos", type=str
    )
    args = parse.parse_args()
    logging.info("Start converting dataset")
    main(args)
    logging.info("Converting dataset finished")
