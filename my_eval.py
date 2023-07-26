import evals
import keras


basic_model = keras.models.load_model("checkpoints/ghostnetv1_w1.3_s2.h5", compile=False)
ee = evals.EvalCallback(
    basic_model,
    "datasets/faces_emore/lfw.bin",
    batch_size=256,
    PCA_acc=True,
)
ee.on_epoch_end(0)

# >>>> lfw evaluation max accuracy: 0.996833, thresh: 0.223459, previous max accuracy: 0.000000, PCA accuray = 0.996000 ± 0.002494
# >>>> Improved = 0.996833
