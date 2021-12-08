import multiprocessing
from multiprocessing import Pool


def process_depth_calculations(video_name):
    print(video_name)


if __name__ == "__main__":
    input_videos = ["good_path", "floor", "right_wall"]
    
    # multiprocessing of the remaining videos
    pool = multiprocessing.Pool(3)
    zip(pool.map(process_depth_calculations, input_videos))