from datetime import datetime
from ultralytics import YOLOv10


if __name__ == '__main__':
    # init
    time = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    config = 'iRMB'
    datasets = 'KITTI'
    optimizer = 'AdamW'
    lr = 1e-3
    epochs = 100
    seed = 1999

    model = YOLOv10(config+'.yaml')
    model.train(
        data=datasets+'.yaml',
        optimizer=optimizer,
        lr0=lr,
        epochs=epochs,
        seed=seed,
        dnn=True,
        name=config+'_'+datasets+'_'+time,
        device=0,
        visualize=False
    )
    #  model.export()