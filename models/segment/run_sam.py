import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from segment_anything import SamPredictor, sam_model_registry
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from torchvision import transforms
import zipfile
import argparse
import torch
from tqdm import tqdm
from PIL import Image

# old sam
CHECKPOINT = os.path.expanduser("~/third_party/segment-anything/ckpts/sam_vit_h_4b8939.pth")
MODEL = "vit_h"

def visualize_and_save_keypoints(image, keypoints, output_folder,output_name):
    # 检查输出文件夹是否存在，如果不存在，则创建
    os.makedirs(output_folder, exist_ok=True)
    
    # 绘制关键点
    output_image = image.copy()
    for x, y, visibility in keypoints:
        if visibility > 0.5:  # 只绘制可见的关键点
            cv2.circle(output_image, (int(x), int(y)), 5, (255, 255, 255), -1)
    
    # 保存图像
    #output_path = os.path.join(output_folder, output_name)
    Image.fromarray(output_image).save(output_name)
    return output_name


def detect_keypoints(image,output_folder,key_point_output_name):
    # Mediapipe 初始化
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.8)
    # 关键点预测
    results = pose.process(image)
    if not results.pose_landmarks:
        return None
    # 筛选特定关键点
    # 看这个网站选点：https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
    selected_indices = [0, 7, 8, 11, 12, 13, 14, 19, 20, 23, 24, 25, 26, 29, 30]  # 全身的点要稀疏，太过密集会导致SAM忽略旁边的点
    keypoints = np.array([(lmk.x * image.shape[1], lmk.y * image.shape[0], lmk.visibility) 
                 for i, lmk in enumerate(results.pose_landmarks.landmark) if i in selected_indices])
    # keypoints = np.array([(lmk.x * image.shape[1], lmk.y * image.shape[0], lmk.visibility) for lmk in results.pose_landmarks.landmark])
    output_path = visualize_and_save_keypoints(image, keypoints, output_folder, key_point_output_name)
    print(f"Keypoints visualized and saved to {output_path}")
    return keypoints


def process_img_with_sam_keypoints(data_dir,sam_model_type):
    # SAM模型初始化
    if sam_model_type == "Old":
        sam = sam_model_registry[MODEL](checkpoint=CHECKPOINT)
        sam.to("cuda")
        predictor = SamPredictor(sam)
    elif sam_model_type == "S":
        with zipfile.ZipFile("/home/cookmaker/Codes/humanseg/models/segment/EfficientSAM/weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
            zip_ref.extractall("/home/cookmaker/Codes/humanseg/models/segment/EfficientSAM/weights")
            # Build the EfficientSAM-S model.
            predictor = build_efficient_sam_vits(checkpoint_path="/home/cookmaker/Codes/humanseg/models/segment/EfficientSAM/weights/efficient_sam_vits.pt")
    elif sam_model_type == "Ti":
        predictor = build_efficient_sam_vitt()

    root = data_dir
    img_lists = sorted(glob.glob(f"{root}/images/*.png"))
    #print(img_lists)
    os.makedirs(f"{root}/masks_sam", exist_ok=True)
    os.makedirs(f"{root}/masked_sam_images", exist_ok=True)
    os.makedirs(f"{root}/keypoints_images", exist_ok=True)

    for fn in tqdm(img_lists):
        #print(img_lists)
        img_np = np.array(Image.open(fn))
        keypoints = detect_keypoints(img_np,f"{root}/keypoints_images",fn.replace("images", "keypoints_images"))
        img = transforms.ToTensor()(img_np)
        # print(img[None, ...].shape)
        if keypoints is not None and sam_model_type == "Old":
            m = keypoints[..., -1] > 0.5
            pts = keypoints[m][:,:2]
            pts = transforms.ToTensor()(pts)
            masks, _, _ = predictor.predict(pts[:, :2], np.ones_like(pts[:, 0]))
            mask = masks.sum(axis=0) > 0
            cv2.imwrite(fn.replace("images", "masks_sam"), mask.astype(np.uint8) * 255)

            img[~mask] = 0
            cv2.imwrite(fn.replace("images", "masked_sam_images"), img)
        else:
            m = keypoints[..., -1] > 0.5
            pts = keypoints[m][:,:2]
            
            pts = transforms.ToTensor()(pts)
            label = torch.ones((1,1,pts.shape[1]))
            print(pts.shape)
            width = img.shape[2]
            height = img.shape[1]
            corners = torch.tensor([[[0, 0],
                         [width - 1, 0],
                         [0, height - 1],
                         [width - 1, height - 1]]])
            new_pts = torch.cat((pts, corners), dim=1)
            zeros = torch.zeros((1, 1, 4), dtype=label.dtype)
            new_label = torch.cat((label, zeros), dim=2)
            # print(pts[None, ...].shape)
            # print(pts.shape)
            # masks, _, _ = predictor.predict(pts[:, :2], np.ones_like(pts[:, 0]))
            predicted_logits, predicted_iou = predictor(
            img[None, ...],
            new_pts[None, ...],
            new_label,
            )
            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
            predicted_logits = torch.take_along_dim(
                predicted_logits, sorted_ids[..., None, None], dim=2
            )
            # The masks are already sorted by their predicted IOUs.
            # The first dimension is the batch size (we have a single image. so it is 1).
            # The second dimension is the number of masks we want to generate (in this case, it is only 1)
            # The third dimension is the number of candidate masks output by the model.
            # For this demo we use the first mask.
            # print(predicted_logits)
            # print(predicted_logits.shape)
            # break
            masks = torch.ge(predicted_logits[0, 0, :, :, :], 0).cpu().detach().numpy()
            print(masks.shape)
            mask = np.logical_or.reduce(masks, axis=0)
            print(mask.shape)
            cv2.imwrite(fn.replace("images", "masks_sam"), np.uint8(mask)*255)
            masked_image_np = img_np.copy().astype(np.uint8) * mask[:,:,None]
            #print(masked_image_np)
            Image.fromarray(masked_image_np).save(fn.replace("images", "masked_sam_images"))
            # cv2.imwrite(fn.replace("images", "masked_sam_images"), masked_image_np)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--sam_model_type", type=str, default="S",help="[S,Ti,Old],分别代表EfficientSAM的S，Ti和老的SAM",required=False)
    args = parser.parse_args()
    process_img_with_sam_keypoints(args.data_dir,args.sam_model_type)