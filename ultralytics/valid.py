from datetime import datetime

import torch

from ultralytics import YOLOv10


if __name__ == '__main__':
    # init
    time = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    config = 'valid'
    datasets = 'KITTI'
    optimizer = 'AdamW'
    lr = 1e-3
    epochs = 100
    seed = 1999
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = YOLOv10(config+'.yaml')
    model = YOLOv10('/home/huang/Ryx-yolov10/runs/detect/iRMB_KITTI_CIoU_24-08-19_11-05-38/weights/best.pt')
    model.val(
        data=datasets+'.yaml',
        resume=True,
        optimizer=optimizer,
        lr0=lr,
        epochs=epochs,
        seed=seed,
        dnn=True,
        name=config+'_'+datasets+'_'+time,
        device=device,
        visualize=False
    )
    #  model.export()