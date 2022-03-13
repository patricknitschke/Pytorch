import tensorflow as tf
import torch
import numpy as np
import skimage.io as io
from utilities import get_files_ending_with

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def collate_batch(batch):
    image = batch[0][0]
    image_filtered = batch[0][1]
    height = batch[0][2]
    width = batch[0][3]
    depth = batch[0][4]
    
    return torch.Tensor(np.array(image)).to(device), torch.Tensor(np.array(image_filtered)).to(device), torch.Tensor(np.array(height)).to(device), torch.Tensor(np.array(width)).to(device), torch.Tensor(np.array(depth)).to(device)

class DepthImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, tfrecord_folder, batch_size=32, shuffle=True, one_tfrecord=False):
        super(DepthImageDataset).__init__()
        self.tfrecord_folder = tfrecord_folder
        self.dataset, self.data_len = self.load_tfrecords(is_shuffle_and_repeat=shuffle, batch_size=batch_size, one_tfrecord=one_tfrecord)

    def read_tfrecord(self, serialized_example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'image_filtered': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(serialized_example, feature_description)

        image = tf.cast(tf.io.parse_tensor(example['image'], out_type = tf.uint8), tf.float32) / 255.
        image_filtered = tf.cast(tf.io.parse_tensor(example['image_filtered'], out_type = tf.uint8), tf.float32) / 255.
        height = example['height']
        width = example['width']
        depth = example['depth']
        
        return image, image_filtered, height, width, depth

    def load_tfrecords(self, is_shuffle_and_repeat=True, shuffle_buffer_size=5000, prefetch_buffer_size_multiplier=2, batch_size=32, one_tfrecord=False):
        print('Loading tfrecords... ', end="\t")
        tfrecord_fnames = get_files_ending_with(self.tfrecord_folder, '.tfrecords')
        assert len(tfrecord_fnames) > 0
        if is_shuffle_and_repeat:
            np.random.shuffle(tfrecord_fnames)
        else:
            tfrecord_fnames = sorted(tfrecord_fnames) # 176 tfrecords for train, 20 for test

        if one_tfrecord:
            tfrecord_fnames = tfrecord_fnames[:1]
            print(tfrecord_fnames, end="\t")

        dataset = tf.data.TFRecordDataset(tfrecord_fnames)
        dataset = dataset.map(self.read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if is_shuffle_and_repeat: 
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size_multiplier * batch_size)

        print('Iterating length... ', end="\t")
        data_len = sum(1 for _ in dataset)
        print('Done:', data_len)
        
        return dataset, data_len
    
    def __iter__(self):
        print("gotcha")
        return self.dataset.__iter__()

    def __len__(self):
        return self.data_len


if __name__ == "__main__":
    dataset = DepthImageDataset('/home/patricknit/rl_data/tfrecord')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_batch)
    data = next(iter(loader))
    image = data[0]
    img_idx = 20
    io.imshow(image[img_idx,...,0].numpy())
    io.show()