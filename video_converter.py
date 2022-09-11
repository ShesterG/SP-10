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
import shutil
try:
    from PIL import Image
except ImportError:
    import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

#changes the frame rate of all the videos to 25
def main(args):
    root_path = Path(args.save_path)
    for video_path in root_path.rglob("*.mp4"):
        vid = cv2.VideoCapture(str(video_path))
        frame_rate = int(vid.get(cv2.CAP_PROP_FPS))
        #if frame_rate == 25:
        #    continue
        #logging.info(f"Start converting {video_path}, its frame rate is {frame_rate}")
        video_path = video_path.resolve()
        os.system(f"ffmpeg -i '{video_path}' -r 25 /tmp/sign.mp4 -y")
        framed_vid = cv2.VideoCapture("/tmp/sign.mp4")
        
        # frame
        currentframe = 1
        
        while (True):

            # reading from frame
            success, frame = framed_vid.read()

            if success:
                h, w, channels = frame.shape
                cropped_frame = frame[h//3:w//2]
                #(Rect(0, 0, frame.cols/2, frame.rows/3));
                verse = pytesseract.image_to_string(Image.open(cropped_frame))
                 #TODO FY
                if VERSE_EQUALS_TO_THE_STANDARD_FORMAT: #TODO FY                                  
                    # continue creating images until video remains                   
                    verse = verse.replace(k)
                    # creates folder with verse name if it doesn't yet exists. 
                    try:
                        # creates folder with verse name if it doesn't yet exists. 
                        verse_path = # TODO FY
                        if not os.path.exists(verse_path):
                            os.makedirs(verse_path)
                            currentframe = 1
                    # if not created then raise error
                    except OSError:
                        print('Error: Creating directory of data')     
                                                            
                    name = './f"{verse}"/images' + str("{:04d}".format(currentframe)) + '.png' #TODO SHESTER : NOT VERY SURE OF THE f"{verse}
                    print('Creating...' + name)

                    # writing the extracted images
                    cv2.imwrite(name, frame)

                    # increasing counter so that it will
                    # show how many frames are created
                    currentframe += 1
                    
            else:
                break
        # Release all space and windows once done
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
