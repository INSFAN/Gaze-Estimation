"""Train a DeepLab v3 plus model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import gaze_estimate_model
from utils import preprocessing
from tensorflow.python import debug as tf_debug
import numpy as np
from sklearn.model_selection import KFold

import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='./model',
                    help='Base directory for the model.')

parser.add_argument('--clean_model_dir', action='store_true',
                    help='Whether to clean up the model directory if present.')

parser.add_argument('--train_epochs', type=int, default=26,
                    help='Number of training epochs: '
                         'For 30K iteration with batch size 6, train_epoch = 17.01 (= 30K * 6 / 10,582). '
                         'For 30K iteration with batch size 8, train_epoch = 22.68 (= 30K * 8 / 10,582). '
                         'For 30K iteration with batch size 10, train_epoch = 25.52 (= 30K * 10 / 10,582). '
                         'For 30K iteration with batch size 11, train_epoch = 31.19 (= 30K * 11 / 10,582). '
                         'For 30K iteration with batch size 15, train_epoch = 42.53 (= 30K * 15 / 10,582). '
                         'For 30K iteration with batch size 16, train_epoch = 45.36 (= 30K * 16 / 10,582).')

parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of training epochs to run between evaluations.')

parser.add_argument('--tensorboard_images_max_outputs', type=int, default=6,
                    help='Max number of batch elements to generate for Tensorboard.')

parser.add_argument('--batch_size', type=int, default=64,
                    help='Number of examples per batch.')

parser.add_argument('--learning_rate_policy', type=str, default='poly',
                    choices=['poly', 'piecewise'],
                    help='Learning rate policy to optimize loss.')

parser.add_argument('--max_iter', type=int, default=50000,
                    help='Number of maximum iteration used for "poly" learning rate policy.')

parser.add_argument('--data_dir', type=str, default="/home/insfan/insfan-git/data-sets/gazecaputer_mini/",
                    help='Path to the directory containing the PASCAL VOC data tf record.')

parser.add_argument('--freeze_batch_norm', action='store_true',
                    help='Freeze batch normalization parameters during the training.')

parser.add_argument('--initial_learning_rate', type=float, default=1e-3,
                    help='Initial learning rate for the optimizer.')

parser.add_argument('--end_learning_rate', type=float, default=1e-6,
                    help='End learning rate for the optimizer.')

parser.add_argument('--initial_global_step', type=int, default=0,
                    help='Initial global step for controlling learning rate when fine-tuning model.')

parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='The weight decay to use for regularizing the model.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

_HEIGHT = 64
_WIDTH = 64
_DEPTH = 3
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255

_POWER = 0.9
_MOMENTUM = 0.9

_BATCH_NORM_DECAY = 0.9997

_NUM_IMAGES = {
    'train': 2927,
    'validation': 1449,
}


def get_filenames(is_training, data_dir):
  """Return a list of filenames.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: path to the the directory containing the input data.

  Returns:
    A list of file names.
  """
  if is_training:
    return [os.path.join(data_dir, 'Gaze_train.record')]
  else:
    return [os.path.join(data_dir, 'Gaze_val.record')]


def parse_record(raw_record):
  """Parse PASCAL image and label from a tf record."""
  keys_to_features = {
      # 'image/height':
      # tf.FixedLenFeature((), tf.int64),
      # 'image/width':
      # tf.FixedLenFeature((), tf.int64),
      # 'image/encoded':
      # tf.FixedLenFeature((), tf.string, default_value=''),
      # 'image/format':
      # tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      # 'label/encoded':
      # tf.FixedLenFeature((), tf.string, default_value=''),
      # 'label/format':
      # tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/en_eye_left': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/en_eye_right': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/en_face': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/en_face_mask': tf.FixedLenFeature((), tf.string, default_value=''),
      'label/en_xy': tf.FixedLenFeature((), tf.string),
  }

  parsed = tf.parse_single_example(raw_record, keys_to_features)

  # eye_left = tf.image.decode_image(
  #         tf.reshape(parsed['image/en_eye_left'], shape=[64, 64, 3]), _DEPTH)
  # eye_left = tf.to_float(tf.image.convert_image_dtype(eye_left, dtype=tf.uint8))
  # eye_left.set_shape([None, None, 3])
  #
  # eye_right = tf.image.decode_image(
  #         tf.reshape(parsed['image/en_eye_right'], shape=[64, 64, 3]), _DEPTH)
  # eye_right = tf.to_float(tf.image.convert_image_dtype(eye_right, dtype=tf.uint8))
  # eye_right.set_shape([None, None, 3])
  #
  # face = tf.image.decode_image(
  #         tf.reshape(parsed['image/en_face'], shape=[64, 64, 3]), _DEPTH)
  # face = tf.to_float(tf.image.convert_image_dtype(face, dtype=tf.uint8))
  # face.set_shape([None, None, 3])
  #
  # face_mask = tf.image.decode_image(
  #         tf.reshape(parsed['image/en_face_mask'], shape=[25, 25, 1]), 1)
  # face_mask = tf.to_float(tf.image.convert_image_dtype(face_mask, dtype=tf.uint8))
  #
  # face_mask.set_shape([None, None, 1])
  #
  # label = tf.reshape(parsed['image/en_xy'], shape=[])



  eye_left = tf.decode_raw(bytes=parsed['image/en_eye_left'], out_type=tf.float32)
  eye_right = tf.decode_raw(parsed['image/en_eye_right'], tf.float32)
  face = tf.decode_raw(parsed['image/en_face'], tf.float32)
  face_mask = tf.decode_raw(parsed['image/en_face_mask'], tf.float32)
  label = tf.decode_raw(parsed['label/en_xy'], tf.float32)


  eye_left = tf.reshape(eye_left, [64, 64, 3])
  eye_right = tf.reshape(eye_right, [64, 64, 3])
  face = tf.reshape(face, [64, 64, 3])
  face_mask = tf.reshape(face_mask, [25, 25, 1])
  label = tf.reshape(label, [1, 1, 2])


  # image = tf.image.decode_image(
  #     tf.reshape(parsed['image/encoded'], shape=[]), _DEPTH)
  # image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
  # image.set_shape([None, None, 3])
  #
  # label = tf.image.decode_image(
  #     tf.reshape(parsed['label/encoded'], shape=[]), 1)
  # label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
  # label.set_shape([None, None, 1])

  return (eye_left, eye_right, face, face_mask), label


def get_record_parser():
    def parse(example):
        features = tf.parse_single_example(
                   example,features={
                                    'img':tf.FixedLenFeature([], tf.string),
                                    'label':tf.FixedLenFeature([], tf.int64)
                                    }
                    )
        image = tf.reshape(tf.decode_raw(features['img'],tf.int32),[784])
        label = features['label']
        return image,label
    return parse

def preprocess_image(image, label, is_training=False):
  """Preprocess a single image of layout [height, width, depth]."""
  # if is_training:
  #   # Randomly scale the image and label.
  #   image, label = preprocessing.random_rescale_image_and_label(
  #       image, label, _MIN_SCALE, _MAX_SCALE)
  #
  #   # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
  #   image, label = preprocessing.random_crop_or_pad_image_and_label(
  #       image, label, _HEIGHT, _WIDTH, _IGNORE_LABEL)
  #
  #   # Randomly flip the image and label horizontally.
  #   image, label = preprocessing.random_flip_left_right_image_and_label(
  #       image, label)
  #
  #   image.set_shape([_HEIGHT, _WIDTH, 3])
  #   label.set_shape([_HEIGHT, _WIDTH, 1])

  image = preprocessing.mean_image_subtraction(image)

  return image, label

# Import data
def load_data(file):
    npzfile = np.load(file)
    # face = np.array(f['faceData'])
    train_eye_left = np.array(npzfile["train_eye_left"])
    train_eye_right = np.array(npzfile["train_eye_right"])
    train_face = np.array(npzfile["train_face"])
    train_face_mask = np.array(npzfile["train_face_mask"])
    train_y = np.array(npzfile["train_y"])*100
    #train_y = np.squeeze(train_y)
    return [train_eye_left, train_eye_right, train_face, train_face_mask, train_y]

def load_data_from_npz(file):

    print("Loading dataset from npz file...", end='')
    npzfile = np.load(file)
    train_eye_left = npzfile["train_eye_left"]
    train_eye_right = npzfile["train_eye_right"]
    train_face = npzfile["train_face"]
    train_face_mask = npzfile["train_face_mask"]
    train_y = npzfile["train_y"]
    val_eye_left = npzfile["val_eye_left"]
    val_eye_right = npzfile["val_eye_right"]
    val_face = npzfile["val_face"]
    val_face_mask = npzfile["val_face_mask"]
    val_y = npzfile["val_y"]
    print("Done.")

    return [train_eye_left, train_eye_right, train_face, train_face_mask, train_y], [val_eye_left, val_eye_right, val_face, val_face_mask, val_y]


def normalize(data):
    shape = data.shape
    data = np.reshape(data, (shape[0], -1))
    data = data.astype('float32') / 255. # scaling
    data = data - np.mean(data, axis=0) # normalizing
    return np.reshape(data, shape)

def prepare_data_mp2(data):
    eye_left, eye_right, face, face_mask, y = data
    eye_left = normalize(eye_left)
    eye_right = normalize(eye_right)
    face = normalize(face)
    face_mask = face_mask.astype('float32')
    # face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')
    y = y.astype('float32')
    return [eye_left, eye_right, face, face_mask, y]

def shuffle_data(data):
    idx = np.arange(data[0].shape[0])
    np.random.shuffle(idx)
    for i in range(len(data)):
        data[i] = data[i][idx]
    return data

def next_batch(data, batch_size):
    for i in np.arange(0, data[0].shape[0], batch_size):
        # yield a tuple of the current batched data
        batch_data = [each[i: i + batch_size] for each in data]
        yield batch_data[:-1], batch_data[-1]

def input_fn_(is_training, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training, data_dir))
  dataset = dataset.flat_map(tf.data.TFRecordDataset)

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

  dataset = dataset.map(parse_record)
  dataset = dataset.map(
      lambda image, label: preprocess_image(image, label))
  dataset = dataset.prefetch(batch_size)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()


  return images, labels

def input_fn(is_training, dataset, batch_size, num_epochs=1):
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

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=2000)

  # dataset = dataset.map(parse_record)
  # dataset = dataset.map(
  #     lambda image, label: preprocess_image(image, label, is_training))
  dataset = dataset.prefetch(batch_size)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  if is_training:
    dataset = dataset.repeat(None)
  else:
    dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels




def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  if FLAGS.clean_model_dir:
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig(
    model_dir=FLAGS.model_dir,
    tf_random_seed=None,
    save_summary_steps=50,
    save_checkpoints_steps=1000,
    save_checkpoints_secs=None,
    session_config=None,
    keep_checkpoint_max=3,
    keep_checkpoint_every_n_hours=10000,
    log_step_count_steps=100)
  model = tf.estimator.Estimator(
      model_fn=gaze_estimate_model.gaze_estimate_model_fn,    # Model function
      model_dir=FLAGS.model_dir,  # Storage catalogue
      config=run_config,         # Setting parameter object
      # hyper parameter, which will be passed to model_fn for use
      params={
          'batch_size': FLAGS.batch_size,
          'batch_norm_decay': _BATCH_NORM_DECAY,
          'tensorboard_images_max_outputs': FLAGS.tensorboard_images_max_outputs,
          'weight_decay': FLAGS.weight_decay,
          'learning_rate_policy': FLAGS.learning_rate_policy,
          'num_train': _NUM_IMAGES['train'],
          'initial_learning_rate': FLAGS.initial_learning_rate,
          'max_iter': FLAGS.max_iter,
          'end_learning_rate': FLAGS.end_learning_rate,
          'power': _POWER,
          'momentum': _MOMENTUM,
          'freeze_batch_norm': FLAGS.freeze_batch_norm,
          'initial_global_step': FLAGS.initial_global_step
      })

  tensors_to_log = {
    'learning_rate': 'learning_rate',
    'mse_loss': 'mse_loss',
    'train_px_error': 'train_px_error',
   }
  logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
  train_hooks = [logging_hook]

  val_tensors_to_log = {
    'val_mse_loss': 'mse_loss',
    'val_px_error': 'val_px_error',
   }
  val_logging_hook = tf.train.LoggingTensorHook(
        tensors=val_tensors_to_log, every_n_iter=10)
  val_hooks = [val_logging_hook]

  if FLAGS.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    train_hooks.append(debug_hook)
    eval_hooks = [debug_hook]
  tf.logging.info("Start training.")
  train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn_(True, FLAGS.data_dir, FLAGS.batch_size),
                                      hooks=train_hooks,
                                      max_steps=60000)
  eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn_(False, FLAGS.data_dir, FLAGS.batch_size, 1),
                                    steps=None,
                                    name=None,
                                    hooks=val_hooks,
                                    exporters=None,
                                    start_delay_secs=120,
                                    throttle_secs=60)

  tf.estimator.train_and_evaluate(model, train_spec, eval_spec)






if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
