"""Converts PASCAL dataset to TFRecords file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import os
import sys

import PIL.Image
import tensorflow as tf
import numpy as np
from utils import dataset_util

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='/home/insfan/insfan-git/data-sets/gazecaputer_mini/eye_tracker_train_and_val.npz',
                    help='Path to the directory containing the PASCAL VOC data.')

parser.add_argument('--output_path', type=str, default='/home/insfan/insfan-git/data-sets/gazecaputer_mini/',
                    help='Path to the directory to create TFRecords outputs.')

parser.add_argument('--train_data_list', type=str, default='./dataset/train.txt',
                    help='Path to the file listing the training data.')

parser.add_argument('--valid_data_list', type=str, default='./dataset/val.txt',
                    help='Path to the file listing the validation data.')

parser.add_argument('--image_data_dir', type=str, default='JPEGImages',
                    help='The directory containing the image data.')

parser.add_argument('--label_data_dir', type=str, default='SegmentationClassAug/SegmentationClassAug',
                    help='The directory containing the augmented label data.')

# load data directly from the npz file (small dataset, 48k and 5k for train and test)
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


def dict_to_tf_example(images,
                       label):
  """Convert image and label to tf.Example proto.

  Args:
    image: left_eye, right_eye, face, face_mask image.
    label: (x,y)label.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by image_path is not a valid JPEG or
                if the label pointed to by label_path is not a valid PNG or
                if the size of image does not match with that of label.
  """

  example = tf.train.Example(features=tf.train.Features(feature={
    # 'image/height64': dataset_util.int64_feature(height64),
    # 'image/width64': dataset_util.int64_feature(width64),
    # 'image/height25': dataset_util.int64_feature(height25),
    # 'image/width25': dataset_util.int64_feature(width25),
    # 'image/en_eye_left': dataset_util.bytes_feature([images[0].tostring()]),
    # 'image/en_eye_right': dataset_util.bytes_feature([images[1].tostring()]),
    # 'image/en_face': dataset_util.bytes_feature([images[2].tostring()]),
    # 'image/en_face_mask': dataset_util.bytes_feature([images[3].tostring()]),
    # # 'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    # 'label/en_xy': dataset_util.float_list_feature([label]),
    # # 'label/format': dataset_util.bytes_feature('png'.encode('utf8')),
    'image/en_eye_left': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images[0].tostring()])),
    'image/en_eye_right': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images[1].tostring()])),
    'image/en_face': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images[2].tostring()])),
    'image/en_face_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images[3].tostring()])),
    # 'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    'label/en_xy': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()])),
    # 'label/format': dataset_util.bytes_feature('png'.encode('utf8')),
  }))
  return example


def create_tf_record(output_filename,
                     dataset,
                     ):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    image_dir: Directory where image files are stored.
    label_dir: Directory where label files are stored.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  nums = dataset[0].shape[0]
  for i in range(nums):
    eye_left = dataset[0][i]
    eye_right = dataset[1][i]
    face = dataset[2][i]
    face_mask = dataset[3][i]
    label = dataset[4][i]
    tf_example = dict_to_tf_example([eye_left, eye_right, face,face_mask], label)
    writer.write(tf_example.SerializeToString())

  writer.close()

def save_tfrecords(data, label, desfile):
    with tf.python_io.TFRecordWriter(desfile) as writer:
      for i in range(len(data)):
        features = tf.train.Features(
          feature={
            "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data[i].astype(np.float64).tostring()])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label[i]]))
          }
        )
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)




def main(unused_argv):
  if not os.path.exists(FLAGS.output_path):
    os.makedirs(FLAGS.output_path)

  tf.logging.info("Reading from GazeCapture dataset")
  train_data, val_data = load_data_from_npz(FLAGS.data_dir)

  train_output_path = os.path.join(FLAGS.output_path, 'Gaze_train.record')
  val_output_path = os.path.join(FLAGS.output_path, 'Gaze_val.record')

  create_tf_record(train_output_path, train_data)
  create_tf_record(val_output_path, val_data)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
