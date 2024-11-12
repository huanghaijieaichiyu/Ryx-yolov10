import cv2
import os
from tqdm import tqdm


def vedio_writer(
        img_folder='runs/vedio'):

    # 图片文件夹路径
    image_folder = img_folder
    # 视频输出路径
    path = 'runs/pred'

    os.mkdir(path)
    video_output = os.path.join(path, 'result.avi')
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # 图片文件名列表
    images = os.listdir(image_folder)
    # images.sort()  # 不排序会乱
    # 假设所有图片尺寸相同，这里我们只读取第一张图片的尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(
        video_output, fourcc, 20, (width, height))
    pbar = tqdm(images, total=len(
        images), colour='#8762A5', ncols=200)
    # 将图片逐一写入视频
    for image in pbar:
        img = cv2.imread(os.path.join(img_folder, image))
        '''cv2.rectangle(img, (5, 5), (10, 30), (0, 0, 255), -1)
        cv2.putText(img, '检测到的周围车辆', (7, 35),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.75, (255, 255, 255), 0)
        cv2.rectangle(img, (25, 5), (30, 30), (0, 255, 0), -1)
        cv2.putText(img, '驾驶车辆', (27, 35),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.75, (255, 255, 255), 0)
        cv2.rectangle(img, (45, 5), (150, 30), (255, 0, 255), -1)
        cv2.putText(img, '地图', (47, 35),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.75, (255, 255, 255), 0)'''
        video.write(img)

    # 释放VideoWriter对象
    video.release()
    pbar.close()


if __name__ == '__main__':
    vedio_writer()
