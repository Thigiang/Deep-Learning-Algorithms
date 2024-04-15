import numpy as np
import h5py
def load_dataset():
    path = "/Users/gabati/Documents/GitHub/"
    file_name_train, file_name_test = 'train_catvnoncat.h5', 'test_catvnoncat.h5'
    file_path_train = path +"Datasets/"+file_name_train
    file_path_test = path + "Datasets/"+ file_name_train
    print(file_path_train)
    
    train_dataset = h5py.File(file_path_train, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    
    test_dataset = h5py.File(file_path_test, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    
    classes = np.array(test_dataset["list_classes"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes