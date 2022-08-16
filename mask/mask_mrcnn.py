import os
import sys
import numpy as np
import skimage

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import visualize
from mrcnn import model as modellib

# Matilda Class names
# Index of the class in the list is its ID. For example, to get ID of
class_names = ['BG', 'ring', 'shirts', 'pants', 'hat', 'shoes']
outdir = './mask/test_imgs'

############################################################
#  Configurations
############################################################

class MatildaConfig(Config):
    NUM_CLASSES = 1 + 5  # Background + [ring, shirts, pants, hat, shoes]
    NAME = "matilda"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

############################################################
#  Functions
############################################################

def detect_mask(model, images):
    masks = []
    for i, image in enumerate(images):
        if (image.shape[2] != 3):
            image = image[:, :, 0:3]
        # Run detection
        results = model.detect([image], verbose=1)
        # Save results
        r = results[0]
        mask = r['masks'][:, :, 0]

        ## test masked_image
        colors = visualize.random_colors(len(class_names))
        masked_image = image.astype(np.uint32).copy()
        masked_image = visualize.apply_mask(masked_image, mask, colors[0])

        mask = np.expand_dims(mask, axis=2)

        # Save output
        file_name = f"{outdir}/{i}_rgb.png"
        skimage.io.imsave(file_name, masked_image.astype(np.uint8))

        masks.append(mask)

    return masks

def get_mask_model(weights_path):
    config = MatildaConfig()
    config.display()
    model = modellib.MaskRCNN(config=config)
    model.load_weights(weights_path, by_name=True)
    return model

