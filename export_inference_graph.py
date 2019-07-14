"""Export inference graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

import gaze_estimate_model
from utils import preprocessing


parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='./model',
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")

parser.add_argument('--export_dir', type=str, default='dataset/export_output',
                    help='The directory where the exported SavedModel will be stored.')



_NUM_CLASSES = 21


_MEAN_LE = 0.47839513
_MEAN_RE = 0.2806009
_MEAN_F = 0.3976589

def prepare_data_mp2(data):
    eye_left, eye_right, face, face_mask = data
    eye_left = eye_left / 255. - _MEAN_LE
    eye_right = eye_right / 255. - _MEAN_RE
    face = face / 255. - _MEAN_F

    features = tf.concat([eye_left, eye_right, face, face_mask], axis=-1)
    # face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')
    return features


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  model = tf.estimator.Estimator(
      model_fn=gaze_estimate_model.gaze_estimate_model_fn,
      model_dir=FLAGS.model_dir,
      params={
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
          'batch_norm_decay': None,
      })

  # Export the model
  def serving_input_receiver_fn():
    eye_left = tf.placeholder(tf.float32, [None, 64, 64, 3], name='eye_left')
    eye_right = tf.placeholder(tf.float32, [None, 64, 64, 3], name='eye_right')
    face = tf.placeholder(tf.float32, [None, 64, 64, 3], name='face')
    face_mask = tf.placeholder(tf.float32, [None, 64, 64, 1], name='face_mask')

    receiver_tensors = {'eye_left': eye_left, 'eye_right': eye_right,'face': face,'face_mask': face_mask}
    features = prepare_data_mp2([eye_left, eye_right, face, face_mask])
    return tf.estimator.export.ServingInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors)

  model.export_savedmodel(FLAGS.export_dir, serving_input_receiver_fn)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
