import yaml

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generate_data(path):
  with open(f'{path}/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

  train_datagen = ImageDataGenerator(rescale = 1./config['rescale'],
                                   shear_range = float(config['shear_range']),
                                   zoom_range = float(config['zoom_range']),
                                   horizontal_flip = bool(config['horizontal_flip']))

  test_datagen = ImageDataGenerator(rescale = config['rescale'])

  return train_datagen, test_datagen