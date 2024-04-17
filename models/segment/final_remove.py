import cv2
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from rembg import new_session, remove

def process_image(image_path, output_path,bg_image):
    """
    先使用传统算法，针对绿幕进行移除，然后再使用深度学习算法针对边缘进行进一步优化。
    """

    # 读取图像
    image = cv2.imread(image_path)
    
    blue, green, red = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # 检查绿色通道是否为RGB中的最大值且大于30的条件
    mask_condition = (green > red) & (green > blue) & (green * 2 - red - blue > 30)
    #mask = np.where(mask_condition, 0, 255).astype(np.uint8)

    # 根据条件创建mask，满足条件的像素为0，不满足的为255
    mask = np.where(mask_condition, 0, 255).astype(np.uint8)
    
    # 这里是加上神经网络的部分
    model_name = "u2net"
    session = new_session(model_name)
    seg_mask = remove(image,
                      only_mask=True,
                      session=session,
                      alpha_matting=True)  
    final_mask = (seg_mask > 150) * 255
    # cv2.imwrite('final_mask1.png',mask)
    # cv2.imwrite('final_mask2.png',seg_mask)
    mask = np.bitwise_or(mask,final_mask).astype(np.uint8)




    mask_inv = cv2.bitwise_not(mask)
    # cv2.imwrite('final_mask.png',mask)
    #print(mask)

    background = cv2.bitwise_and(bg_image, bg_image, mask=mask_inv)
    # 将处理好的mask应用到原图像以生成前景图像
    rgb_foreground = cv2.bitwise_and(image, image, mask=mask)
    final_image = cv2.add(rgb_foreground, background)

    # 构建输出文件路径
    base_name = os.path.basename(image_path)
    foreground_file_path = os.path.join(output_path, base_name.replace('.png', '_processed.png'))

    # 保存处理后的图像
    # cv2.imwrite(mask_file_path, mask)  # 假设mask已经定义
    cv2.imwrite(foreground_file_path, final_image)  # 假设rgb_foreground已经定义



def process_images_in_parallel(image_folder, output_folder, bg_image):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 构建一个包含所有图像路径的列表
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]
    output_paths = [os.path.join(output_folder) for path in image_paths]

    
    # 使用ProcessPoolExecutor来并行处理
    with ProcessPoolExecutor() as executor:
        # 提交所有任务
        future_to_image = {executor.submit(process_image, image_path, output_path,bg_image): image_path for image_path, output_path in zip(image_paths, output_paths)}
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(future_to_image), total=len(future_to_image), desc="Processing Images"):
            image_path = future_to_image[future]
            try:
                # 尝试获取执行结果
                result = future.result()
            except Exception as exc:
                print(f"{image_path} generated an exception: {exc}")

# # 使用示例
# input_folder = './kiki_sit'
# output_folder = './kiki_sit_result'
# image = cv2.imread(os.path.join(input_folder,os.listdir(input_folder)[0]))
# #image = cv2.imread(input_folder)
# bg_image = red_background = np.zeros_like(image)
# red_background[:, :] = [255, 255, 255]
# process_images_in_parallel(input_folder, output_folder, bg_image)
