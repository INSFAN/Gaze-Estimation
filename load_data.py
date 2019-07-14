import numpy as np
import cv2
import os
import glob
from os.path import join
import json
from utils import dataset_util
import data_utility

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

# Import data
def load_data(file):
    npzfile = np.load(file)
    # face = np.array(f['faceData'])
    train_eye_left = np.array(npzfile["train_eye_left"])
    train_eye_right = np.array(npzfile["train_eye_right"])
    train_face = np.array(npzfile["train_face"])
    train_face_mask = np.array(npzfile["train_face_mask"])
    train_y = np.array(npzfile["train_y"])
    #train_y = np.squeeze(train_y)
    return [train_eye_left, train_eye_right, train_face, train_face_mask, train_y]

# load a batch with data loaded from the npz file
def load_batch(data, img_cols, img_rows, img_ch):

    # useful for debug
    save_images = False

    # if save images, create the related directory
    img_dir = "images"
    if save_images:
        if not os.path.exists(img_dir):
            os.makedir(img_dir)

    right_eye_batch = data[1]
    left_eye_batch = data[0]
    face_batch = data[2]
    face_grid_batch = np.expand_dims(data[3], axis=3)
    y_batch = data[4]

    return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch


# create a list of all names of images in the dataset
def load_data_names(path):

    seq_list = []
    seqs = sorted(glob.glob(join(path, "0*")))

    for seq in seqs:

        file = open(seq, "r")
        content = file.read().splitlines()
        for line in content:
            seq_list.append(line)

    return seq_list

def dict_to_tf_example(image_path,
                       label_path):
  """Convert image and label to tf.Example proto.

  Args:
    image_path: Path to a single PASCAL image.
    label_path: Path to its corresponding label.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by image_path is not a valid JPEG or
                if the label pointed to by label_path is not a valid PNG or
                if the size of image does not match with that of label.
  """
  with tf.gfile.GFile(image_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')

  with tf.gfile.GFile(label_path, 'rb') as fid:
    encoded_label = fid.read()
  encoded_label_io = io.BytesIO(encoded_label)
  label = PIL.Image.open(encoded_label_io)
  if label.format != 'PNG':
    raise ValueError('Label format not PNG')

  if image.size != label.size:
    raise ValueError('The size of image does not match with that of label.')

  width, height = image.size

  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/encoded': dataset_util.bytes_feature(encoded_jpg),
    'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    'label/encoded': dataset_util.bytes_feature(encoded_label),
    'label/format': dataset_util.bytes_feature('png'.encode('utf8')),
  }))
  return example


def create_tf_record(output_filename,
                     image_dir,
                     label_dir,
                     examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    image_dir: Directory where image files are stored.
    label_dir: Directory where label files are stored.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
    if idx % 500 == 0:
      tf.logging.info('On image %d of %d', idx, len(examples))
    image_path = os.path.join(image_dir, example + '.jpg')
    label_path = os.path.join(label_dir, example + '.png')

    if not os.path.exists(image_path):
      tf.logging.warning('Could not find %s, ignoring example.', image_path)
      continue
    elif not os.path.exists(label_path):
      tf.logging.warning('Could not find %s, ignoring example.', label_path)
      continue

    try:
      tf_example = dict_to_tf_example(image_path, label_path)
      writer.write(tf_example.SerializeToString())
    except ValueError:
      tf.logging.warning('Invalid example: %s, ignoring.', example)

  writer.close()

if __name__ == "__main__":

    # debug
    seq_list = load_data_names("/cvgl/group/GazeCapture/test")

    batch_size = len(seq_list)
    dataset_path = "/media/insfan/00028D8D000E9194/MPIIFaceGaze/MPIIFaceGaze/MPIIFaceGaze_fem64.npz"

    train_data= load_data(dataset_path)
    data_utility.prepare_data(train_data)

    # img_ch = 3
    # img_cols = 64
    # img_rows = 64
    #
    # test_batch = load_batch_from_names_random(seq_list, dataset_path, batch_size, 64, 64, 3)
    #
    # print("Loaded: {} data".format(len(test_batch[0][0])))
