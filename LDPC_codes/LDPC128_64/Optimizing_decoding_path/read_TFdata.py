import tensorflow as tf
from functools import partial
import globalmap as GL

def feature_selection(feature_indicator):
    if feature_indicator == 'zero':
        code = GL.get_map('code_parameters')
        code_length = code.n
        feature_description = {
            'feature': tf.io.FixedLenFeature([code_length], tf.float32),
            'label': tf.io.FixedLenFeature([code_length], tf.int64), 
            'shape': tf.io.FixedLenFeature([], tf.int64)
        }
    if feature_indicator == 'one':  
        code = GL.get_map('code_parameters')  # Define code_length here
        code_length = code.n
        list_length = GL.get_map('num_iterations') + 2
        feature_description = {
            'feature': tf.io.FixedLenFeature([list_length], tf.float32),
            'label': tf.io.FixedLenFeature([code_length], tf.int64),
            'shape': tf.io.FixedLenFeature([], tf.int64)
        }
    return feature_description

# Function to get keys of feature_description
def get_feature_keys(feature_description):
    return list(feature_description.keys())

def _parse_function(serial_exmp, feature_description, feature_keys):    
    feats = tf.io.parse_single_example(serial_exmp, feature_description)
    feat_list = [feats[key] for key in feature_keys]    
    return feat_list

def get_dataset(fname, feature_description):
    dataset = tf.data.TFRecordDataset(fname)  
    feature_keys = get_feature_keys(feature_description)
    parse_fn = partial(_parse_function, feature_description=feature_description, feature_keys=feature_keys)  
    dataset = dataset.map(parse_fn)
    return dataset

def data_handler(file_name, feature_indicator='zero', batch_size=1): 
    feature_description = feature_selection(feature_indicator)
    dataset_train = get_dataset(file_name, feature_description)  
    #dataset_test = dataset_test.shuffle(buffer_size=10000)  # Shuffle the dataset      
    dataset_train = dataset_train.batch(batch_size, drop_remainder=False)
    dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset_train