# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time
import glob 
import pandas as pd
from tqdm import tqdm
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"


def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        return False
    else:
        return True


def test(image_list, model_dir, device_id):
    arr = []
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    for image_name in tqdm(glob.glob(image_list)):
        image = cv2.imread(image_name)
        result = check_image(image)
        image_bbox = model_test.get_bbox(image)
        prediction = np.zeros((1, 3))
        test_speed = 0
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            start = time.time()
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
            test_speed += time.time()-start

        # draw result of prediction
        label = np.argmax(prediction)
        value = prediction[0][label]/3
        # if label == 1:
        #     print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name.split("\\")[-1], value))
        # else:
        #     print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name.split("\\")[-1], value))
        arr.append([image_name.split("\\")[-1], label, value])
    arr = np.array(arr)
    df = pd.DataFrame(arr, columns=["fname","class","score"])
    df.to_csv("result.csv", index =False)


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models/",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default="./cropped_frame/*",
        help="image used to test")
    args = parser.parse_args()
    test(args.image_name, args.model_dir, args.device_id)
