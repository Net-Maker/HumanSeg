import subprocess
import os
import json
from final_remove import process_images_in_parallel
import cv2
import numpy as np
from rembg import new_session, remove
import glob
import shutil

class VideoMatting:
    def __init__(self, infer_video_path=None, slient_video_path=None, train_video_path=None, output_folder='./output'):
        """
        获取视频的path，包括原视频的path和经过裁剪后视频的path
        获取视频的分辨率，视频的帧率，为后续去除背景做准备
        需要父类（音频处理提供的信息：开始帧数，结束帧数，）
        """
        self.slient_video_path = slient_video_path
        self.infer_video_path = infer_video_path
        self.train_video_path = train_video_path
        self.output_folder = output_folder

        # self.video_name = slient_video_path.split('/')[-1][:-4]
        # self.output_path = f"./output/{self.video_name}"

        # self.width,self.height,self.fps = self.get_video_info(video_path=self.infer_video_path)
        # print('Video Matting Initialization Finished')

    def beautify_video(self,target_image=None):
        """
        调用akool的api，美化输入的素材，需要输入一张target_image用来美化
        target_image就是用来美化的图片
        api document : https://d21ksh0k4smeql.cloudfront.net/API/faceswap-web-api-document-V2.pdf
        postman example : https://documenter.getpostman.com/view/32241598/2s9YsKgsTM#a2352945-6e18-4aba-838b-740145877b77
        token : eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY1ZTk3MGVjNmQxYzhkOGEzMWMzYTZlOSIsInVpZCI6MTQ3NzM4MiwidHlwZSI6InVzZXIiLCJmcm9tIjoidG9iIiwiZW1haWwiOiJ3dzEwMTUwMDUzMTNAZ21haWwuY29tIiwiZmlyc3ROYW1lIjoi5L2zIiwiaWF0IjoxNzExMzYwMDkzLCJleHAiOjE3NDM1MDA4OTN9.ktYMiYe25fGsHWQkk20uwp5KhdAy2-S8HztNYWGJNMI
        """
        pass


    def get_video_info(self,video_path):
        """
        使用ffprobe获取视频的分辨率和帧率。

        参数:
        - video_path: 视频文件的路径。

        返回:
        - 分辨率和帧率的字典。
        """
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate',
            '-of', 'json',
            video_path
        ]

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            video_info = json.loads(result.stdout)

            # 提取分辨率和帧率
            width = video_info['streams'][0]['width']
            height = video_info['streams'][0]['height']
            frame_rate_str = video_info['streams'][0]['r_frame_rate']
            # 帧率是以分数形式给出的，比如"30000/1001"，这里将其转换为浮点数
            frame_rate = eval(frame_rate_str)

            return width,height,frame_rate
        except subprocess.CalledProcessError as e:
            print("获取视频信息失败:", e)
            exit()


    def remove_bg(self,start_time=0,video_path=None):
        """
        先对视频进行抽帧操作，然后调用process_image函数，处理所有图像
        """
        # 确保输出文件夹存在
        video_name = video_path.split('/')[-1][:-4]
        output_path = f"{self.output_folder}/{video_name}"
        frame_folder = os.path.join(output_path, 'frames')
        print(frame_folder)
        os.makedirs(frame_folder, exist_ok=True)

        # 构建ffmpeg命令
        ffmpeg_cmd = [
            'ffmpeg', 
            '-i', video_path,
            '-vf', f"fps={self.fps}",
            '-ss', f"{start_time}",
            '-vsync', 'vfr', '-q:v', '2',  # 调整输出质量
            os.path.join(frame_folder, 'frame_%04d.png')  # 输出文件命名
        ]
        
        
        # 执行ffmpeg命令抽帧
        subprocess.run(ffmpeg_cmd, check=True)
        
        image = cv2.imread(os.path.join(frame_folder,os.listdir(frame_folder)[0]))
        bg_image = red_background = np.zeros_like(image)
        red_background[:, :] = [0, 177, 64]
        bg_output_path = os.path.join(frame_folder,"nobg")


        process_images_in_parallel(frame_folder, bg_output_path, bg_image)

        # 获取视频对应的音频
        audio_path = os.path.join(output_path, f'{video_name}.wav')
        audio_cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn','-acodec','pcm_s16le',
            '-ar', ' 44100',
            '-ac', '2',
            audio_path
        ]
        subprocess.run(audio_cmd, check=True)

        self.generate_no_bg_video(bg_output_path,audio_path,output_path,video_name)


        

    def generate_no_bg_video(self,bg_path,audio_path,output_path,video_name):
        """
        将无背景的帧合成为一个透明背景的视频。
        """
        no_bg_frame_path = bg_path
        output_video_path = os.path.join(output_path, f'nobg_{video_name}.mp4')

        # 确保使用带有alpha通道的编码格式，比如libvpx
        ffmpeg_command = [
            'ffmpeg',
            '-framerate', f'{self.fps}',  # 或者其他你用于生成帧的帧率
            '-i', os.path.join(no_bg_frame_path, 'frame_%04d_processed.png'),
            '-i', audio_path,
            '-c:v', 'libx264',  
            '-c:a', 'copy','-shortest',
            '-pix_fmt', 'yuva420p',  # 使用支持透明的像素格式
            '-b:v', '50000k',
            output_video_path
        ]

        # 执行ffmpeg命令
        try:
            subprocess.run(ffmpeg_command, check=True)
            print("无背景视频已生成：", output_video_path)
            return output_video_path
        except subprocess.CalledProcessError as e:
            print("生成无背景视频时出错：", e)
            exit()


    def generate_poster(self):
        """
        生成封面图，一般为静默视频的第一帧
        """
        cap = cv2.VideoCapture(self.slient_video_path)
        # 检查视频是否成功打开
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        # 读取第一帧
        ret, frame = cap.read()
        # 检查是否成功读取到帧
        if ret:
            # 将第一帧的背景移除作为封面
            poster = remove(frame)
            cv2.imwrite(f"{self.output_folder}/poster.png",poster)
            print("封面图已保存.")
        else:
            print("生成封面图失败，无法读取静默视频.")

        # 释放视频对象
        cap.release()
    def clear_temp(self,result_folder='./results'):
        """
        收集产生的结果，删除中间的帧图片等等
        逻辑:删除outputfolder中除了poster.png和mp4文件之外的所有文件
        """
        # 确保result文件夹存在
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # 查找所有符合条件的mp4文件和poster.png
        mp4_files = glob.glob(os.path.join(self.output_folder, '*/*.mp4'))
        poster_file = os.path.join(self.output_folder, 'poster.png')
        if os.path.exists(poster_file):
            mp4_files.append(poster_file)

        # 移动文件并收集信息
        file_info = {}
        for file_path in mp4_files:
            # 生成新路径
            new_path = os.path.join(result_folder, os.path.basename(file_path))
            # 移动文件
            shutil.move(file_path, new_path)
            # 收集文件信息
            file_info[os.path.basename(file_path)] = new_path

        # 生成JSON文件
        json_path = os.path.join(result_folder, 'file_info.json')
        with open(json_path, 'w') as json_file:
            json.dump(file_info, json_file, indent=4)

        # 删除output文件夹
        shutil.rmtree(output_folder)
        print("All tasks completed successfully.")
        


        


infer_video_path = './data/output5_inference.MP4'
slient_video_path = './data/output5_silent.MP4'
train_video_path = './data/output5.MP4'
output_folder = './output'
test = VideoMatting(infer_video_path,slient_video_path,train_video_path,output_folder)
test.remove_bg(video_path=test.slient_video_path)
# test.remove_bg(video_path=test.train_video_path)
# test.remove_bg(video_path=test.infer_video_path)
test.generate_poster()
# test.generate_no_bg_video(output_path='./output/output5_inference/frames/nobg')
test.clear_temp()
