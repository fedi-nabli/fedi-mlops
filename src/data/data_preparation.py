from tensorflow.keras.preprocessing.image import ImageDataGenerator

def data_generator():
  train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

  test_datagen = ImageDataGenerator(rescale = 1./255)

  return train_datagen, test_datagen

def load_data(path):
  train_datagen, test_datagen = data_generator()

  training_set = train_datagen.flow_from_directory(f'{path}/training_set',
                                                  target_size = (124, 124),
                                                  batch_size = 16,
                                                  class_mode = 'categorical')

  test_set = test_datagen.flow_from_directory(f'{path}/test_set',
                                                  target_size = (124, 124),
                                                  batch_size = 16,
                                                  class_mode = 'categorical')

  return training_set, test_set