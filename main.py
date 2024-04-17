"""
date: 2024.4.16(GMT+8)
author: CookMaker(www.github/Net-Maker.com)
describe: a script which include latest pose estinmation, human segmentation and object detection. 
          make them more easy to use. 
"""
import subprocess
import shlex
import os
from models.segment import process_img_with_sam

def get_frame_count(video_path):
    """ 使用ffprobe获取视频的总帧数 """
    command = f"ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 {shlex.quote(video_path)}"
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        raise Exception(f"Error in ffprobe: {result.stderr}")
    return int(result.stdout.strip())

def extract_frames(video_path, output_folder):
    """ 使用ffmpeg按帧抽取视频 """
    #frame_count = get_frame_count(video_path)
    os.makedirs(output_folder, exist_ok=True)
    command = f"ffmpeg -i {shlex.quote(video_path)} -vf fps=1 {shlex.quote(output_folder)}/frame_%04d.png"
    subprocess.run(command, shell=True, check=True)
    #print(f"Frames extracted to {output_folder}. Total frames: {frame_count}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process a video file to extract frames.')
    parser.add_argument('--video_path', type=str, default="/home/cookmaker/Codes/humanseg/data/yunque_001_03_1a_546_clip.mp4", help='Path to the video file.')
    args = parser.parse_args()
    
    video_path = args.video_path
    output_folder = os.path.splitext(video_path)[0] + "_frames" + "/images"
    
    extract_frames(video_path, output_folder)
    process_img_with_sam_keypoints(data_dir=os.path.splitext(video_path)[0] + "_frames",sam_model_type="S")

if __name__ == "__main__":
    main()
    
