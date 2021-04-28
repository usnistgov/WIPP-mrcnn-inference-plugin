# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys, glob, warnings, os
import time
import os

import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage.color import label2rgb

import argparse
import skimage.io
import numpy as np
from os import listdir
from os.path import isfile, join

sys.path.insert(1, '../inference/mrcnn/src');
sys.path.insert(1, '../visualization')

from mrcnn_infer import *
from download_util import *


def main():
    # print(os.getcwd())
    # os.chdir(os.getcwd())
    # os.chdir(os.getcwd()+"/..")
    # time.sleep(80)

    global tic
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    # time.sleep(30)

    parser = argparse.ArgumentParser(prog='threshold',
                                     description='Create a binary image from a grayscale image and threshold value')

    # Define arguments
    parser.add_argument('--inputImages', dest='input_images', type=str,
                        help='filepath to the directory containing the images', required=True)
    # parser.add_argument('--thresholdValue', dest='threshold_value', type=int, required=True)
    parser.add_argument('--output', dest='output_folder', type=str, required=True)

    # Parse arguments
    args = parser.parse_args()
    input_images = args.input_images
    # threshold_value = args.threshold_value
    output_folder = args.output_folder

    # MRCNN_MODEL_URL = 'https://ndownloader.figshare.com/files/22280580?private_link=dd27a1ea28ce434aa7d4'
    # MRCNN_MODEL_PATH = 'MRCNN_pretrained.zip'
    # download_and_unzip_datasets(MRCNN_MODEL_URL, MRCNN_MODEL_PATH)
    mrcnn_model_path = "./mrcnn_pretrained.h5"

    config_file_path = "./demo.ini"
    with open(config_file_path, 'r') as fin:
        print(fin.read())

    images = listdir(input_images)

    print(input_images)

    print(images)

    img = np.zeros((len(images), 1040, 1392))

    for i in range(len(img)):
        tic=0
        toc=0

        print('Start processing image ' + images[i])
        tic = time.perf_counter()
        # image_resized = img_as_ubyte(resize(np.array(Image.open(input_images + "/" + images[i]).convert("L")), (1040, 1392)))
        image_resized = img_as_ubyte(resize(np.array(Image.open(input_images + "/" + images[i])), (1040, 1392)))
        img[i, :, :] = image_resized

        tac = time.perf_counter()
        binary = mrcnn_infer(img, mrcnn_model_path, config_file_path)

        print(binary.shape)

        plt.rcParams['figure.figsize'] = [15, 15]

        # plt.imsave(output_folder + os.path.splitext(images[i])[0] + '.png', binary[i])
        # plt.imsave(output_folder + os.path.splitext(images[i])[0] + '.png', label2rgb(binary[i], bg_color=(0, 0, 0), bg_label=0))

        skimage.io.imsave(output_folder + "/" + images[i], binary[i].astype(np.uint16), 'tifffile', False,
                          tile=(1024, 1024))
        # skimage.io.imsave(output_folder + images[i], label2rgb(binary[i].astype(np.uint8), bg_color=(0, 0, 0), bg_label=0), 'tifffile', False, tile=(1024, 1024))

        print('End processing image' + images[i])

        toc = time.perf_counter()

        # print(f"before mrcnn_infer {tac - tic:0.4f} seconds")
        # print(f"after mrcnn_infer {toc - tac:0.4f} seconds")

        print(f"Processing the image" + images[i] + f"took {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()
