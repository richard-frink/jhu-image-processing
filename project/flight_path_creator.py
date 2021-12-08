import multiprocessing
from multiprocessing import Pool


def process_video(name):
    print(name)


if __name__ == "__main__":
    input_videos = ["good_path", "ceilings", "floor", "right_wall"]
    
    # multiprocessing of the remaining videos
    pool = multiprocessing.Pool(4)
    zip(pool.map(process_video, input_videos))