import cv2
import multiprocessing
from multiprocessing import Pool


def process_depth_calculations(video_name):
    print(video_name)
    # read the whole video file and find per frame the deepest zone
    # pseudo code more stuff before doing real coding!!!!!!!!


if __name__ == "__main__":
    input_videos = ["good_path", "floor", "right_wall"]
    
    # multiprocessing of the videos - calulating the Deepest Zones
    pool = multiprocessing.Pool(3)
    zip(pool.map(process_depth_calculations, input_videos))