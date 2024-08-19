from datetime import datetime

import torch

from ultralytics import YOLOv10


if __name__ == '__main__':
    # init
    time = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    config = 'iRMB'
    datasets = 'KITTI'
    optimizer = 'AdamW'
    lr = 1e-3
    epochs = 100
    batch_size = 16
    num_workers = 8
    seed = 1999
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_iou = 'CIoU'

    model = YOLOv10(config+'.yaml')
    # model = YOLOv10('runs/detect/iRMB_KITTI_24-08-18_13-12-32/weights/last.pt')
    model.train(
        data=datasets+'.yaml',
        resume=True,
        optimizer=optimizer,
        lr0=lr,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        dnn=True,
        name=config+'_'+datasets+'_'+loss_iou+'_'+time,
        device=device,
        visualize=False
    )
    #  model.export()