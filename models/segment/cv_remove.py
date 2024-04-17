import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
import numpy as np
from rembg import new_session, remove
import mediapipe as mp
import matplotlib.pyplot as plt

def detect_objects(image_path):
    # 加载预训练的模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # 读取图片
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 转换图像数据为模型输入格式
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image_rgb)

    # 检测图片中的物体
    with torch.no_grad():
        prediction = model([image_tensor])

    # 获取置信度和标签
    scores = prediction[0]['scores'].numpy()
    labels = prediction[0]['labels'].numpy()
    
    # 过滤出置信度大于0.8且标签不等于1的检测框和对应的标签
    high_conf_indices = np.where((scores > 0.1) & (labels != 1))[0]
    high_conf_boxes = prediction[0]['boxes'].numpy()[high_conf_indices]
    
    low_box = np.array([0,image.shape[0]-100,image.shape[1],image.shape[0]])
    high_conf_boxes = np.append(high_conf_boxes, [low_box], axis=0)
    #print(high_conf_boxes)
    
    return image, high_conf_boxes
    

def convert_rgb_to_rgba(rgb_image):
    """将RGB图像转换为RGBA图像，增加全不透明的Alpha通道"""
    height, width = rgb_image.shape[:2]
    rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
    rgba_image[:, :, :3] = rgb_image
    rgba_image[:, :, 3] = 0  # 设置Alpha通道为255（全不透明）
    return rgba_image

def Add_bounded(x1,x2,width,shift=50):
    """
    为边界框增加一些范围，因为目标检测的框并不是很准确
    """
    if x1-shift > 0:
        x1 -= shift
    else:
        x1 = 0
    if x2+shift < width:
        x2 += shift
    else:
        x2 = width
    return x1,x2

def save_alpha_channel_visualization(image, save_path):
    """
    提取给定图像的Alpha通道，并将可视化结果保存到指定路径。
    
    :param image: 一个包含RGBA通道的图像，已经通过cv2.imread读取。
    :param save_path: 可视化结果保存的文件路径。
    """
    alpha_channel = image[:, :, 3]
    mask = alpha_channel < 200
    
    # 创建一个全黑的图像用于可视化
    visualization = np.zeros_like(image[:, :, :3])
    
    # 在掩码标记为True的位置上将可视化图像设置为白色
    visualization[mask] = [255, 255, 255]  # 白色

    # 使用matplotlib保存可视化结果
    plt.imshow(visualization, cmap='gray')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def segment_and_recombine(image, boxes):
    """
    获取目标检测后所有目标的mask，并使用or操作全部拼在一起，最后用此mask去筛原图像
    """
    if boxes.shape[0] == 0:
        print("没有其他物体")
        return remove(image)
    masks = []
    for box in boxes:
        # 分割图像
        x1, y1, x2, y2 = map(int, box)
        x1,x2 = Add_bounded(x1,x2,image.shape[1])
        y1,y2 = Add_bounded(y1,y2,image.shape[0])
        obj_image = image[y1:y2, x1:x2]
        # 假设remove函数处理后的图像是RGBA格式
        model_name = "u2net"
        session = new_session(model_name)
        seg_mask = remove(obj_image,
                          only_mask=True,
                          session=session,
                          alpha_matting=True)  # 这里的remove函数应该返回RGBA格式的图像
        #cv2.imwrite(f"object_mask{x1}_{x2}.png",seg_mask)
        final_mask = (seg_mask > 150) * 255
        masks.append((final_mask, (x1, y1, x2, y2)))
    
    # 所有mask全部放在一起，由于mask是通过局部检测出来的，所以要还原到原图中
    bg = convert_rgb_to_rgba(image)
    result = remove(image,only_mask=True)  # 这里的remove函数应该返回RGBA格式的图像

    prople = remove(image)
    #save_alpha_channel_visualization(prople,"./fig.png")
    #cv2.imwrite("people_mask.png",result)
    #cv2.imwrite("people.png",prople)
    bg[:,:,3] = (result > 180) * 255


    for mask, (x1, y1, x2, y2) in masks:
        # org_img=bg.copy()
        #print(result.shape,mask.shape)
        bg[y1:y2, x1:x2,3] = np.bitwise_or(bg[y1:y2, x1:x2,3], mask)
        #cv2.imwrite(f"bitwise{x1}_{x2}.png",bg)
    mask = bg[:, :, 3] < 200
    bg[mask, 0] = 0  # B通道
    bg[mask, 1] = 0  # G通道
    bg[mask, 2] = 255  # R通道
    return bg



def process_image(image_path,output_path):
    image, boxes = detect_objects(image_path)
    output = segment_and_recombine(image, boxes)
    cv2.imwrite(output_path,output)
    print("image saved at ",output_path)


# process_image("./test.jpg","./combine.jpg")