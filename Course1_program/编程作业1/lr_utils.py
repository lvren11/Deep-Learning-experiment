import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")  #加载训练集
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features (209,64,64,3)
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r") #加载测试集
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes 保存的是以bytes类型保存的两个字符串数据
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # 维度(1，209) 0 | 1
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0])) # 维度(1，209) 0 | 1
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes