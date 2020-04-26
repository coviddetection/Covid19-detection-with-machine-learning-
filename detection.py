import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow.compat.v2 as tf
import tarfile
import tensorflow as tf
import zipfile
from tensor import tensordict as tensor_dict
from distutils.version import StrictVersion
from collections import defaultdict
from tensorflow import data
import utils
from tkinter import *
from io import StringIO
from tkinter import Frame
from matplotlib import pyplot as plt
from PIL import Image
from tk import *
from object_detection.utils import ops as utils_ops
class Application(Frame):
    tf.compat.v1.GraphDef()   # -> instead of tf.GraphDef()
    PATH_TO_FROZEN_GRAPH = "covid19.model" 
    graph = "plot.png"
    image=input("Dosya seciniz:") 
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        tf.compat.v2.io.gfile.GFile(PATH_TO_FROZEN_GRAPH)
        od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name= '')
    def run_inference_for_single_image(self: image):
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                 output_dict['detection_masks'] = output_dict['detection_masks'][0]
            f = open(graph, "w")
            f.write(output_dict)
            return output_dict



    
    def aaa(self):
        run_inference_for_single_image(image, graph)
    def say_hi(self):
        print ("hi there, everyone!")

    def createWidgets(self):
        self.QUIT = Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.quit

        self.QUIT.pack({"side": "left"})

        self.hi_there = Button(self)
        self.hi_there["text"] = "Korona tespit et "
        self.hi_there["command"] = self.run_inference_for_single_image

        self.hi_there.pack({"side": "left"})

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()
        
root = Tk()
        
app = Application(master=root)
app.mainloop()
root.destroy()




    
