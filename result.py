import pandas as pd
from mrcnn import model as modellib, utils
from mrcnn.config import Config
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from skimage import io
import cv2

# Root directory of the project
ROOT_DIR = ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

    args = parser.parse_args()

    class InferenceConfig(CustomConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=args.logs)
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
    df = pd.read_csv('names.csv')

    for _, row in df.iterrows():
        try:
            img = io.imread(os.path.join('scratch', 'train', row['filename']))
        except:
            img = io.imread(os.path.join('scratch', 'train', row['filename']))

        rig = model.detect([img], verbose=1)[0]

        for i in range(rig['rois'].shape[0]):
            x1, y1, x2, y2 = rig['rios'][i][0], rig['rios'][i][1], rig['rios'][i][2], rig['rios'][i][3]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        cv2.imwrite(os.path.join('output', row['filename']), img)
