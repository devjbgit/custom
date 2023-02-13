# [INFO] '20.10.14 Update
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from object_detection.utils import label_map_util
# from keras.models import load_model # lenet
import efficientnet.tfkeras
from tensorflow.keras.models import load_model # efficient
import keras.backend as K
import tensorflow as tf
import numpy as np
import math as mt
import shutil
import imutils
import argparse
import pickle
import time
import cv2
from tqdm import tqdm
import os
import gc
from imutils import paths
import matplotlib.cm as cm



ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required = True,
                help = 'image directory')
args = vars(ap.parse_args())


# Hyper parameters #####################################################################
EFFI_MODEL = f'model/epoch_20.hdf5'
EFFI_PICKLE = f'model/model.pickle'
#############2###########################################################################

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

last_conv_layer_name = "top_activation"

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(image, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + image
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img
    



class EfficientNetIMG:
    def __init__(self):
        pass
    
    def load_model(self, model_path, label_path):
        self.model = load_model(model_path)
        self.lb = pickle.loads(open(label_path, 'rb').read())
        
        
    def inspection(self, img):	
        img = cv2.resize(img, (300, 300))
        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        
        # with self.graph.as_default():	
        proba = self.model.predict(img)[0]
        idx = np.argmax(proba)
        amount = proba[idx]
        label = self.lb.classes_[idx]
        return label, amount, img

    def save_img(self, path, img):
        now = str(time.time()).replace('.', '_')
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(os.path.join(path, f'TRASH_{now}.jpg'), img)

img_paths = sorted(list(paths.list_images(args['input'])))
EFFI = EfficientNetIMG()



model = EFFI.load_model(EFFI_MODEL, EFFI_PICKLE)
cnt = 0
ok_cnt = 0
ng_cnt = 0
for idx , path in enumerate(img_paths):
    cnt+=1
    img = cv2.imread(path)
    output = img.copy()
    og_label = path.split(os.path.sep)[-2]
    filename = path.split(os.path.sep)[-1]
    st = time.time()
    label, score, np_img = EFFI.inspection(img)
    et = time.time()
    if label == og_label:
        print('ok!!!')
    else:
        print('ng!!!')
        checkFile = f'output/cls/{og_label}_{label}'
        os.makedirs(checkFile, exist_ok = True)
        cv2.imwrite(f"{checkFile}/{filename}",output )
    grad_np = make_gradcam_heatmap(np_img,EFFI.model,last_conv_layer_name)
    superimposed_img = save_and_display_gradcam(output, grad_np)
    grad_path = f'output/grad/{og_label}'
    os.makedirs(grad_path, exist_ok = True)
    superimposed_img.save(f"{grad_path}/{filename}")
