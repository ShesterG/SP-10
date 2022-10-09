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
from keras.preprocessing import image
from keras.layers import merge, Input
#from PIL import Image
#from PIL import ImageChops
#from PIL import ImageEnhance
import tensorflow as tf
import torch
import json
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

#changes the frame rate of all the videos to 25
def main(args):
    root_path = Path(args.save_path)
    #vidnum = 1    
    types = ('*.mp4','*.m4v') # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(root_path.glob(files))
    Emodel = EfficientNet.from_pretrained('efficientnet-b0')
    verses_list=[]
    video_i = 1    
    for videopath in files_grabbed:
        #vid = cv2.VideoCapture(str(video_path))
        refB = str(videopath).split('/')[-1].split('.')[0]
        os.system(f"ffprobe -i {videopath} -print_format default -show_chapters -loglevel error > {root_path}/{refB}.json 2>&1")
        
        with open(f"{root_path}/{refB}.json", "r") as infile:
            data = infile.read()
        data = data.replace("\n", "|")
        data = data.replace("|[/CHAPTER]|", "\n")
        colnames=["CHAPTER","id","time_base","start","start_time","end","end_time","title"]
        dataIO = io.StringIO(data)
        df = pd.read_csv(dataIO, sep="|", names=colnames, header=None)
        df.drop(['CHAPTER', 'id', 'time_base', 'start', 'end'], axis=1, inplace=True)
        df['LastDigit'] = [x.strip()[-1] for x in df['title']]
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
            verse_dict["video_name"] = refB
            verse_dict["name"] = row["title"]
            verse_dict["signer"] = "Signer8"            
            verse_dict["duration"] = float(float(row["end_time"]) - float(row["start_time"]))
            verse_dict["text"] = "Verse Text"
            #video = cv2.VideoCapture(str(videopath))
            ##currentframe = 1
            step = 1/25
            enet_feature_list=[]
            for current_second in np.arange(math.ceil(float(row["start_time"])), math.floor(float(row["end_time"])), step):
              t_msec = 1000*(current_second)
              video.set(cv2.CAP_PROP_POS_MSEC, t_msec)
              success, frame = video.read()
              if success:
                """
                verse_path = f'{root_path}/{row["title"]}'
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
                """ 

                frame = cv2.resize(frame,(320, 240))
                frame = Image.fromarray(frame)

                with torch.no_grad():
                  tfms = transforms.Compose([transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
                  #img = image.load_img(frame, target_size=(320, 240))
                  img = tfms(frame).unsqueeze(0)
                  #print(img.shape) # torch.Size([1, 3, 224, 224])
                  enet_feature = Emodel.extract_features(img)
                  enet_feature = torch.flatten(enet_feature)
                  enet_feature_list.append(enet_feature)                
                              
                  #cv2.imwrite(image_path,cv2.resize(frame,(320, 240)))
                  # increasing counter so that it will
                  # show how many frames are created
                  ##currentframe += 1
            with torch.no_grad():
              try:
                verse_torch = torch.stack(enet_feature_list,0)
                dense_model = torch.nn.Linear(in_features=enet_feature_list[0].numel(), out_features=1024, bias=True)
                dense_verse_torch = dense_model(verse_torch)
                verse_dict["sign"] = dense_verse_torch
                

                verses_list.append(verse_dict)
              except RuntimeError:
                print(f"ERROR: Video {video_i} - {refB}.") 
        print(f"Video {video_i} done.")
        video_i += 1
        video.release()
        cv2.destroyAllWindows() 
    file = gzip.GzipFile("/content/SP-10/dataset/GSL240.dataset", 'wb')
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