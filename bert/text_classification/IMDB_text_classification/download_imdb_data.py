# -*- encoding: utf-8 -*-
'''
@File    :   download_imdb_data.py
@Time    :   2021/03/18 11:35:52
@Author  :   Liang Xiaoguang
@Contact :   hplxg@hotmail.com
--**--
--**--
'''


import os
import shutil

import tensorflow as tf

tf.get_logger().setLevel('ERROR')

# Download the IMDB dataset
url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

train_dir = os.path.join(dataset_dir, 'train')

# remove unused folders to make it easier to load the data
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
