import os
import yaml

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def data_generator(config):
  train_datagen = ImageDataGenerator(rescale = 1./config['rescale'],
                                   shear_range = float(config['shear_range']),
                                   zoom_range = float(config['zoom_range']),
                                   horizontal_flip = bool(config['horizontal_flip']))

  test_datagen = ImageDataGenerator(rescale = config['rescale'])

  return train_datagen, test_datagen

def load_data(path):
  with open(f'{path}/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

  train_datagen, test_datagen = data_generator(config)

  training_set = train_datagen.flow_from_directory(f'{path}/{config['data_path']}/{config['training_set_path']}',
                                                  target_size = (config['target_size'], config['target_size']),
                                                  batch_size = int(config['batch_size']),
                                                  class_mode = 'categorical')

  test_set = test_datagen.flow_from_directory(f'{path}/{config['data_path']}/{config['test_set_path']}',
                                                  target_size = (config['target_size'], config['target_size']),
                                                  batch_size = int(config['batch_size']),
                                                  class_mode = 'categorical')

  return training_set, test_set

if __name__ == '__main__':
  load_data('../../')