from tensorflow import keras
import losses, train

data_path = "datasets/faces_emore_112x112_folders"
eval_paths = [
    "datasets/faces_emore/lfw.bin",
    "datasets/faces_emore/cfp_fp.bin",
    "datasets/faces_emore/agedb_30.bin",
]
tt = train.Train(
    data_path,
    "ghostnetv1_w1.3_s2.h5",
    eval_paths,
    model="./checkpoints/ghostnetv1_w1.3_s2.h5",
    batch_size=128,
    random_status=0,
    lr_base=0.1,
    lr_decay=0.5,
    lr_decay_steps=45,
    lr_min=1e-5,
    eval_freq=1,
    output_weight_decay=1,
)

sch = [
    # {"loss": losses.ArcfaceLoss(scale=32), "epoch": 1, "optimizer": optimizer},
    {"loss": losses.ArcfaceLoss(scale=64), "epoch": 35},
]
tt.train(sch, initial_epoch=15)
