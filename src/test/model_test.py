import yaml
import numpy as np
import keras
from tensorflow.keras.utils import load_img
import matplotlib.pyplot as plt

def test_model(model, class_names, path):
  with open(f'{path}/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

  test_image = load_img(f'{path}/data{config['case_path']}', target_size = (124,124))
  plt.imshow(test_image, interpolation = 'spline16')
  plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
  plt.show()

  test_image = np.expand_dims(test_image, axis = 0)
  result= model.predict(test_image)
  t=0
  i=0
  for label in class_names:
    print("\t%s ==> %.2f %%" % (label, result[t][i]*100))
    i = i + 1