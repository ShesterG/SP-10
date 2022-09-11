
#TO BE ADDED Shester ; ocr start
!sudo apt install tesseract-ocr
!pip install pytesseract

import pytesseract
import shutil
import os
import random
try:
    from PIL import Image
except ImportError:
    import Image

extractedInformation = pytesseract.image_to_string(Image.open('2.jpg'))

#TO BE ADDED Shester ; ocr end

import logging
import argparse
from pathlib import Path
import os
import cv2
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

#changes the frame rate of all the videos to 25
def main(args):
    root_path = Path(args.save_path)
    for video_path in root_path.rglob("*.mp4"):
        cap = cv2.VideoCapture(str(video_path))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        if frame_rate == 25:
            continue
        logging.info(f"Start converting {video_path}, its frame rate is {frame_rate}")
        video_path = video_path.resolve()
        os.system(f"ffmpeg -i '{video_path}' -r 25 /tmp/sign.mp4 -y")
        shutil.move("/tmp/sign.mp4", str(video_path))


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--save_path", help="Path to save the video", default="./videos", type=str
    )
    args = parse.parse_args()
    logging.info("Start converting dataset")
    main(args)
    logging.info("Converting dataset finished")
