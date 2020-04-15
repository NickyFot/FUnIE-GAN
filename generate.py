from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import argparse
import time
import ntpath
import numpy as np
from scipy import misc
from tensorflow.python.keras.models import model_from_json
from utils.data_utils import getPaths, read_and_resize, preprocess, deprocess

parser = argparse.ArgumentParser(description='Enhance underwater images')
parser.add_argument('--inputdir', type=str, required=True, help='Image folder to enhance')
parser.add_argument('--outputdir', type=str, required=True, help='Output directory')
args = parser.parse_args(sys.argv[1:])

test_paths = args.inputdir
samples_dir = args.outputdir
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

checkpoint_dir = 'saved_models/gen_p/'
model_name_by_epoch = "model_15320_"
model_h5 = checkpoint_dir + model_name_by_epoch + ".h5"
model_json = checkpoint_dir + model_name_by_epoch + ".json"

with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()
funie_gan_generator = model_from_json(loaded_model_json)
funie_gan_generator.load_weights(model_h5)
print("\nLoaded data and model")

times = []
s = time.time()
for root, dirs, files in os.walk(test_paths):
    for img_path in files:
        if not img_path.lower().endswith('.jpg'):
            continue
        img_name = ntpath.basename(img_path).split('.')[0]
        im, shape = read_and_resize(os.path.join(root, img_path), (256, 256))
        im = preprocess(im)
        s = time.time()
        gen = funie_gan_generator.predict(im)
        gen = deprocess(gen, shape)
        tot = time.time()-s
        times.append(tot)
        misc.imsave(os.path.join(root, img_name+'_gen.png'), gen[0])

    # some statistics
    num_test = len(test_paths)
    if (num_test==0):
        print ("\nFound no images for test")
    else:
        print ("\nTotal images: {0}".format(num_test))

