GPUS = 2
EPOCHS = 50
ECHO_LAG = 3
SEQ_LEN = 15

BATCH_SIZE = 32  # Batch size on EACH GPU (will not be divided by GPUS)
GRADIENT_CLIP = 1.0 # Gradients will be scaled to have a magnitude

FILTERS =      [10, 10, 10, 10]
KERNEL_SIZES = [ 7,  7,  7,  7]              # 3x3 kernels
INPUTS = [28]                                # 28x28 input
STATES = [[INPUTS[0], n] for n in FILTERS]   # featuremaps are same size as input
CLASSES = [10]
TRAIN_DIR = './mnist_png/training'
VALIDATION_DIR = './mnist_png/testing'
