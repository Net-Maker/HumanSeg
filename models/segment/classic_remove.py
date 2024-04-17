import numpy as np
import cv2

image_path = "./test2.png"
image = cv2.imread(image_path)


# 将图像数据转换为float，便于计算
# image_float = image.astype(np.float32)

# 分别获取BGR通道
# blue, green, red = image_float[:, :, 0], image_float[:, :, 1], image_float[:, :, 2]
blue, green, red = image[:, :, 0], image[:, :, 1], image[:, :, 2]

# 检查绿色通道是否为RGB中的最大值且大于30的条件
mask_condition = (green * 2 - red - blue > 30) & (green == np.maximum(np.maximum(red, green), blue))

# 根据条件创建mask，满足条件的像素为0，不满足的为255
mask = np.where(mask_condition, 0, 255).astype(np.uint8)
mask_inv = cv2.bitwise_not(mask)


# 创建一个与原图像大小相同的全红色背景
red_background = np.zeros_like(image)
red_background[:, :] = [255, 255, 255]  # 设置为红色，注意OpenCV使用BGR格式

# 使用掩码将原图像的前景与背景合并
# 前景部分
foreground = cv2.bitwise_and(image, image, mask=mask)
# 背景部分
background = cv2.bitwise_and(red_background, red_background, mask=mask_inv)
# 合并前景和背景
final_image = cv2.add(foreground, background)
#使用形态学操作改进mask
# kernel = np.ones((5,5),np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算去噪点
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # 闭运算填充小洞


#mask = cv2.GaussianBlur(mask, (5, 5), 0)
#mask = cv2.medianBlur(mask, 5)
cv2.imwrite("rgb_mask.png",mask)


#rgb_foreground = cv2.bitwise_and(image, image, mask=mask)

cv2.imwrite("rgb_result.png", final_image)



# _______________________________________________________hsv_____________________________________________________

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# print(image_hsv[:,:,1])
# print(image_hsv[200,200])
lower_green = np.array([40, 60, 60])
upper_green = np.array([80, 255, 255])
    
    # 根据颜色范围创建掩码
mask = cv2.inRange(image_hsv, lower_green, upper_green)

# 使用形态学操作改进mask
# kernel = np.ones((5,5),np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算去噪点
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # 闭运算填充小洞


mask_inv = cv2.bitwise_not(mask)

foreground = cv2.bitwise_and(image, image, mask=mask_inv)


cv2.imwrite("hsv1_image.jpg",image_hsv[:,:,0])
cv2.imwrite("hsv2_image.jpg",image_hsv[:,:,1])
cv2.imwrite("hsv3_image.jpg",image_hsv[:,:,2])
cv2.imwrite("hsv_mask.png",mask)
cv2.imwrite("hsv_result.png", foreground)
    


# 选择要进行Sobel边缘检测的通道，这里以V通道为例
channel_v = image_hsv[:, :, 1]

# 使用Sobel算子进行边缘检测
sobelx = cv2.Sobel(channel_v, cv2.CV_64F, 1, 0, ksize=3)  # 对x方向进行边缘检测
sobely = cv2.Sobel(channel_v, cv2.CV_64F, 0, 1, ksize=3)  # 对y方向进行边缘检测


# 计算梯度幅度
sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)

# 转换为8位图像显示
sobel_magnitude = np.uint8(sobel_magnitude / np.max(sobel_magnitude) * 255)

# 显示结果
cv2.imwrite('Sobel.jpg', sobel_magnitude)

_, mask = cv2.threshold(sobel_magnitude, 40, 255, cv2.THRESH_BINARY)

# 可选：使用膨胀操作来强化mask边缘
kernel = np.ones((5,5), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)

# 显示结果
cv2.imwrite('SMask.jpg', mask)