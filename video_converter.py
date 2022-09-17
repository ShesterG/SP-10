#frames from videos; https://github.com/Ai-Genome-Saksham/OpenCV/blob/main/OpenCV/%239%20Extracting%20Images%20from%20Video.py
#text from images;   https://github.com/bhadreshpsavani/ExploringOCR/blob/master/OCRusingTesseract.ipynb

#!sudo apt install tesseract-ocr # TODO : INSERT INSIDE REQUIREMENTS ?
#!pip install pytesseract        # TODO : INSERT INSIDE REQUIREMENTS ?

import pytesseract
import random
import logging
import argparse
from pathlib import Path
import os
import cv2
import re
import shutil
import glob
from PIL import Image
from PIL import ImageChops
from PIL import ImageEnhance


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

#changes the frame rate of all the videos to 25
def main(args):
    root_path = Path(args.save_path)
    #vidnum = 1    
    types = ('*.mp4', '*.m4v') # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(root_path.glob(files))
    for video_path in files_grabbed:
        vid = cv2.VideoCapture(str(video_path))
        frame_rate = int(vid.get(cv2.CAP_PROP_FPS))
        #if frame_rate == 25:
        #    continue
        #logging.info(f"Start converting {video_path}, its frame rate is {frame_rate}")
        video_path = video_path.resolve()
        os.system(f"ffmpeg -i '{video_path}' -r 25 /tmp/sign.mp4 -y")
        framed_vid = cv2.VideoCapture("/tmp/sign.mp4")
        
        # frame
        #currentframe = 1
        
        while (True):

            # reading from frame
            success, frame = framed_vid.read()

            if success:
                frame = cv2.resize(frame,(640, 480))
                frameT = Image.fromarray(frame)
                #w,h = frameT.size
                #cropped_frame = frame[h//3:w//2]
                rePattern = re.compile("^([0-9a-zA-Z][^0-9]+)([0-9]{1,}(\:[0-9]{1,})?)$")
                #cropped_frame = frameT.crop((10, 10, 110, 40))
                #verse = pytesseract.image_to_string(frameT.crop((10, 10, 110, 40))).strip()
                #print(verse)
                #matchesPattern = False 
                verse = pytesseract.image_to_string(ImageEnhance.Contrast(ImageChops.invert(frameT.crop((50, 25, 215, 75)))).enhance(2))
                #if bool(re.search(rePattern, verse)):                    
                #    print(verse)
                #    matchesPattern = True
                """
                elif bool(re.search(rePattern, pytesseract.image_to_string(ImageEnhance.Contrast(ImageChops.invert(frameT.crop((10, 10, 100, 40)))).enhance(1.5)).strip())):
                    verse = pytesseract.image_to_string(ImageEnhance.Contrast(ImageChops.invert(frameT.crop((10, 10, 100, 40)))).enhance(1.5))
                    print(verse)
                    matchesPattern = True
                elif bool(re.search(rePattern, pytesseract.image_to_string(ImageEnhance.Contrast(ImageChops.invert(frameT.crop((10, 10, 90, 40)))).enhance(1.5)).strip())):
                    verse = pytesseract.image_to_string(ImageEnhance.Contrast(ImageChops.invert(frameT.crop((10, 10, 90, 40)))).enhance(1.5))
                    print(verse)
                    matchesPattern = True                    
                else:
                    matchesPattern = False   
                """          
                           





                #temp_name = 'temp_image' + '.png' #TODO SHESTER : NOT VERY SURE OF THE f"{verse}
                #print('Creating...' + name)
                #temp_im_path = f"/tmp/{temp_name}"                
                #cv2.imwrite(temp_im_path, frame)

                #(Rect(0, 0, frame.cols/2, frame.rows/3));
                #verse = pytesseract.image_to_string(temp_im_path).strip()
                #print(verse)
                #os.remove(temp_im_path)
                #TODO FY
                #rePattern = re.compile("^([0-9a-zA-Z][^0-9]+)([0-9]{1,}(\:[0-9]{1,})?)$")
                #matchesPattern = bool(re.search(rePattern, verse)) 
                if bool(re.search(rePattern, verse)): #TODO FY                                  
                    # continue creating images until video remains  
                    refinedPattern = re.compile(":|\s")                 
                    refinedVerse = re.sub(refinedPattern, '_', verse) 
                    verse_path = f"/content/SP-10/dataset/gse/{refinedVerse}"
                    # creates folder with verse name if it doesn't yet exists. 
                    try:
                        # creates folder with verse name if it doesn't yet exists. 
                        verse_path = Path(verse_path)  # TODO FY
                        if not os.path.exists(verse_path):
                            os.makedirs(verse_path)
                            currentframe = 1
                    # if not created then raise error
                    except OSError:
                        print('Error: Creating directory of data')     
                                                            
                    name = 'images' + str("{:04d}".format(currentframe)) + '.png' #TODO SHESTER : NOT VERY SURE OF THE f"{verse}
                    print('Creating...' + name)
                    image_path = f"{verse_path}/{name}"

                    # writing the extracted images
                    cv2.imwrite(image_path,cv2.resize(frame,(320, 240)))

                    # increasing counter so that it will
                    # show how many frames are created
                    currentframe += 1
                    
            else:
                break
        # Release all space and windows once done
        #vidnum +=1
        vid.release()
        cv2.destroyAllWindows()        
        
        shutil.move("/tmp/sign.mp4", str(video_path)) #TODO SHESTER: MOVE TO A PLACE WHERE ALL FRAMED VIDEOS WILL BE. 


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--save_path", help="Path to save the video", default="./videos", type=str
    )
    args = parse.parse_args()
    logging.info("Start converting dataset")
    main(args)
    logging.info("Converting dataset finished")
