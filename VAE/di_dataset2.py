import sys
import tensorflow as tf
import torch
import numpy as np
import skimage.io as io
import os

sys.path.append('.')

def get_files_ending_with(folder_or_folders, ext):
    if isinstance(folder_or_folders, str):
        folder = folder_or_folders
        assert os.path.exists(folder)

        fnames = []
        for fname in os.listdir(folder):
            if fname.endswith(ext):       
                fnames.append(os.path.join(folder, fname))
        return sorted(fnames)
    else:
        assert hasattr(folder_or_folders, '__iter__')
        print('folder_or_folders:', folder_or_folders)
        return list(itertools.chain(*[get_files_ending_with(folder, ext) for folder in folder_or_folders]))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def collate_batch(batch):
    # image, actions, robot_state, collision_label, info_label, height, width, depth, action_horizon = [], [], [], [], [], [], [], [], []
    # for _image, _actions, _robot_state, _collision_label, _info_label, _height, _width, _depth, _action_horizon in batch:
    #     image.append(_image.numpy())
    #     actions.append(_actions.numpy())
    #     robot_state.append(_robot_state.numpy())
    #     collision_label.append(_collision_label.numpy())
    #     info_label.append(_info_label.numpy())
    #     height.append(_height.numpy())
    #     width.append(_width.numpy())
    #     depth.append(_depth.numpy())
    #     action_horizon.append(_action_horizon.numpy())
    # print('batch[0]:', batch[0])
    image = batch[0][0]
    height = batch[0][1]
    width = batch[0][2]
    depth = batch[0][3]
    
    return torch.Tensor(np.array(image)).to(device), torch.Tensor(np.array(height)).to(device), torch.Tensor(np.array(width)).to(device), torch.Tensor(np.array(depth)).to(device)


class DepthImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, tfrecord_folder, batch_size=32):
        super(DepthImageDataset).__init__()
        self.tfrecord_folder = tfrecord_folder
        self.itr = self.load_tfrecords(batch_size=batch_size)

    def read_tfrecord(self, serialized_example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(serialized_example, feature_description)

        image = tf.transpose(tf.cast(tf.io.parse_tensor(example['image'], out_type = tf.uint8), tf.float32),  perm=[2, 0, 1]) / 256
        height = example['height']
        width = example['width']
        depth = example['depth']
        return image, height, width, depth

    def load_tfrecords(self, is_shuffle_and_repeat=True, shuffle_buffer_size=5000, prefetch_buffer_size_multiplier=2, batch_size=32):
        print('Loading tfrecords...')
        tfrecord_fnames = get_files_ending_with(self.tfrecord_folder, '.tfrecords')
        assert len(tfrecord_fnames) > 0
        if is_shuffle_and_repeat:
            np.random.shuffle(tfrecord_fnames)
        else:
            tfrecord_fnames = sorted(tfrecord_fnames) # 176 tfrecords for train, 20 for test

        tfrecord_fnames = tfrecord_fnames[:1]

        dataset = tf.data.TFRecordDataset(tfrecord_fnames)
        dataset = dataset.map(self.read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if is_shuffle_and_repeat: 
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size_multiplier * batch_size)

        iterator = dataset.__iter__()
        print('Done.')
        return iterator
    
    def __iter__(self):
        return self.itr  



if __name__ == "__main__":
    dataset = DepthImageDataset('/home/patricknit/rl_data/tfrecord')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_batch)
    data = next(iter(loader))
    image = data[0]
    img_idx = 20
    io.imshow(image[img_idx,...,0].numpy())
    io.show()