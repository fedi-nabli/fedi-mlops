from src.data.data_generation import generate_data
from src.data.data_preparation import load_data
from src.model.model_create import create_model
from src.model.model_train import train_model
from src.model.model_validate import validate_model
from src.test.model_test import test_model

def main():
  train_datagen, test_datagen = generate_data('./')
  training_set, test_set = load_data('./', train_datagen, test_datagen)
  num_classes = training_set.num_classes
  class_names = list(training_set.class_indices.keys())
  print(num_classes , class_names)

  model = create_model(num_classes)
  history = train_model('./', model, training_set, test_set)
  validate_model(model, history, training_set, test_set)
  test_model(model, class_names, './')

if __name__ =='__main__':
  main()