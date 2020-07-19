#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------

# Imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import pandas as pd
import csv
import time
from packaging import version

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util

# initialize .csv
with open('traffic_measurement.csv', 'w') as f:
    writer = csv.writer(f)
    csv_line = \
        'Vehicle Type/Size, Vehicle Color, Vehicle Movement Direction, Vehicle Speed (km/h)'
    writer.writerows([csv_line.split(',')])

if version.parse(tf.__version__) < version.parse('1.4.0'):
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!'
                      )

# input video
# cap = cv2.VideoCapture('sub-1504619634606.mp4')

# cap = cv2.imread('jeep.jpg')
# cap = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
# Variables
total_passed_vehicle = 0  # using it to count vehicles

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = \
    'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Download Model
# uncomment if you have not download the model yet
# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



# Detection
def object_detection_function():
    total_passed_vehicle = 0
    speed = 'waiting...'
    direction = 'waiting...'
    size = 'waiting...'
    color = 'waiting...'
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            car_data = pd.read_csv("../car_image_data.csv")
            tmp_list = {}
            for col in car_data.columns:
                tmp_list[col] = []
            tmp_list['is_car'] = []
            start = time.time()
            for i in range(len(car_data)):
                input_frame = cv2.imread("../Myauto_data/Car_Images/Car_Images/{}".format(car_data["img_path"][i]))

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = \
                    sess.run([detection_boxes, detection_scores,
                             detection_classes, num_detections],
                             feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    0,
                    input_frame,
                    boxes[0],
                    classes[0].astype(np.int32),
                    scores[0],
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4
                    )

                for col in car_data.columns:
                    tmp_list[col].append(car_data[col][i])

                is_car = 0
                for i in range(len(classes[0])):
                    if classes[0][i] == 3 or classes[0][i] == 8:
                        if scores[0][i] > 0.7:
                            is_car = 1
                            break

                tmp_list["is_car"].append(is_car)

            tmp_list = pd.DataFrame(tmp_list)
            tmp_list.to_csv('whole_data.csv', index = False)  
            end = time.time()
            print("time: ", end - start)
object_detection_function()		
