import urllib.request
import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def main(args):
    # user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.74 Safari/537.36'
    # headers = {'User-Agent': user_agent}
    #parts = ["train","dev", "test"]
    #for part in parts:
    lan = "gse"
    logging.info(f"Downloading {lan} set")
    with open(f"dataset/{lan}.json", "r") as f:
        data = json.load(f)
    i=1
    for obj in data:
        #video_name = obj["video_name"]
        #sign_list = obj["sign_list"]
        #for sign_obj in sign_list:
        video_url = obj["videoUrl"]
        if video_url is None:
            continue
        
        file_path = Path(args.save_path + f"/{lan}/video{i}.mp4")
        if file_path.exists():
            logging.info(f"{file_path} already exists")
            continue
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading {file_path}")
        logging.info(f"Requesting {video_url}")
        urllib.request.urlretrieve(video_url, file_path)
        i = i+1 
    logging.info(f"Downloading {lan} set finished")





if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--save_path", help="Path to save the video",default="./videos",type=str)
    args = parse.parse_args()
    logging.info("Start downloading dataset")
    main(args)
    logging.info("Downloading dataset finished")
