"""Run inference a DeepLab v3 model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

import gaze_estimate_model
from utils import preprocessing
from utils import dataset_util
import numpy as np
import timeit

from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default="/home/insfan/insfan-git/data-sets/MPIIFaceGaze_fem64_p00.npz",
                    help='The directory containing the image data.')

parser.add_argument('--output_dir', type=str, default='./dataset/inference_output',
                    help='Path to the directory to generate the inference results')

parser.add_argument('--infer_data_list', type=str, default='./dataset/sample_images_list.txt',
                    help='Path to the file listing the inferring images.')

parser.add_argument('--model_dir', type=str, default='./dataset/export_output/1563027569/',
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

_NUM_CLASSES = 21

_MEAN_LE = 0.47839513
_MEAN_RE = 0.2806009
_MEAN_F = 0.3976589

# Import data
def load_data(file):
    npzfile = np.load(file)
    # face = np.array(f['faceData'])
    train_eye_left = np.array(npzfile["train_eye_left"])
    train_eye_right = np.array(npzfile["train_eye_right"])
    train_face = np.array(npzfile["train_face"])
    train_face_mask = np.array(npzfile["train_face_mask"])
    train_y = np.array(npzfile["train_y"])/10.0
    #train_y = np.squeeze(train_y)
    return [train_eye_left, train_eye_right, train_face, train_face_mask, train_y]

def normalize(data):
    shape = data.shape
    data = np.reshape(data, (shape[0], -1))
    data = data.astype('float32') / 255. # scaling
    data = data - np.mean(data) # normalizing
    return np.reshape(data, shape)

def prepare_data_mp2(data):
    eye_left, eye_right, face, face_mask, y = data
    eye_left = eye_left.astype('float32') / 255. - _MEAN_LE
    eye_right = eye_right.astype('float32') / 255. - _MEAN_RE
    face = face.astype('float32') / 255. - _MEAN_F
    face_mask = face_mask.astype('float32')
    # face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')
    y = y.astype('float32')
    return [eye_left, eye_right, face, face_mask, y]

def eval_input_fn(dataset, batch_size):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  nums = dataset[0].shape[0]
  dataset0 = tf.convert_to_tensor(dataset[0])
  dataset1 = tf.convert_to_tensor(dataset[1])
  dataset2 = tf.convert_to_tensor(dataset[2])
  dataset3 = tf.convert_to_tensor(dataset[3])
  dataset4 = tf.convert_to_tensor(dataset[4])
  dataset = tf.data.Dataset.from_tensor_slices(((dataset0, dataset1, dataset2, dataset3), dataset4))

  # dataset = dataset.map(parse_record)
  # dataset = dataset.map(
  #     lambda image, label: preprocess_image(image, label, is_training))
  dataset = dataset.prefetch(batch_size)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels


def next_batch(data, batch_size):
    for i in np.arange(0, data[0].shape[0], batch_size):
        # yield a tuple of the current batched data
        batch_data = [each[i: i + batch_size] for each in data]
        yield batch_data

def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  dataset = load_data(FLAGS.data_dir)
  # dataset = shuffle_data(dataset)
  nums = dataset[-1].shape[0]
  val_data = [dataset[0][int(0.8*nums):,:,:,:], dataset[1][int(0.8*nums):,:,:,:], dataset[2][int(0.8*nums):,:,:,:],\
              dataset[3][int(0.8*nums):,:,:,:], dataset[4][int(0.8*nums):,:,:,:]]

  with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], FLAGS.model_dir)
    graph = tf.get_default_graph()

    eye_left = sess.graph.get_tensor_by_name('eye_left:0')
    eye_right = sess.graph.get_tensor_by_name('eye_right:0')
    face = sess.graph.get_tensor_by_name('face:0')
    face_mask = sess.graph.get_tensor_by_name('face_mask:0')
    pred = sess.graph.get_tensor_by_name('gaze_estimate/conv3/BiasAdd:0')
    # summaryWriter = tf.summary.FileWriter('log/', graph)
    gen_bacth = next_batch(val_data, 1)
    while True:
      batch_xs = next(gen_bacth)
      preds = sess.run(pred,feed_dict={eye_left: batch_xs[0], eye_right: batch_xs[1],face: batch_xs[2],face_mask: batch_xs[3]})
      print("predict: ", (np.squeeze(preds)), np.squeeze(batch_xs[-1]))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)






