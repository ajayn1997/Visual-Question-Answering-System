import numpy as np
from keras.utils.np_utils import to_categorical
import json
import h5py
import os
from constants import *

def right_align(seq, lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N - lengths[i]:N] = seq[i][0: lengths[i]]
    return v

def read_data(data_limit):
    print('Reading Data......')
    img_data = h5py.File(data_img)
    ques_data = h5py.File(data_prepo)

    img_data = np.array(img_data['images_train'])
    img_pos_train = ques_data['img_pos_train'][:data_limit]
    train_img_data = np.array([img_data[_-1, :] for _ in img_pos_train])
    # Normalizing images
    tem = np.sqrt(np.sum(np.multiply(train_img_data, train_img_data), axis=1))
    train_img_data = np.divide(train_img_data, np.transpose(np.tile(tem, (4096,1))))

    # Shifting padding to left side
    ques_data = np.array(ques_data['ques_train'])[:data_limit, :]
    ques_length_train = np.array(ques_data['ques_length_train'])[:data_limit]
    ques_train = right_align(ques_train, ques_length_train)

    train_X = [train_img_data, ques_train]
    # NOTE should've consturcted one-hots using exhausitve list of answers, cause some answers may not be in dataset
    # To temporarily rectify this, all those answer indices is set to 1 in validation set
    train_y = to_categorical(ques_data['answers'])[:data_limit, :]
    
    return train_X, train_y

def get_val_data():
    img_data = h5py.File(data_img)
    ques_data = h5py.File(data_prepo)
    metadata = get_metadata()
    with open(val_annotations_path, 'r') as an_file:
        annotations = json.loads(an_file.read())
    
    img_data = np.array(img_data['images_test'])
    img_pos_test = ques_data['img_pos_test']
    test_img_data = np.array([img_data[_-1, :] for _ in img_pos_test])
    tem = np.sqrt(np.sum(np.multiply(test_img_data, test_img_data), axis=1))
    test_img_data = np.divide(test_img_data, np.transpose(np.tile(tem, (4096, 1))))

    ques_test = np.array(ques_data['ques_test'])
    ques_length_test = np.array(ques_data['ques_length_test'])
    ques_test = right_align(ques_test, ques_length_test)
    
    # Convert all last index to 0, coz embeddings were made that way
    


def get_metadata():
    meta_data = json.load(open(data_prepo_meta, 'r'))
    meta_data['ix_to_word'] = {str(word):int(i) for i,word in meta_data['ix_to_word'].items()}
    return meta_data