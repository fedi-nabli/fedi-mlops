import os.path

from src.data.data_preparation import load_data
from src.model.model_create import create_model
from src.model.model_train import train_model
from src.model.model_validate import validate_model
from src.test.model_test import test_model

training_set, test_set = load_data('./data')
num_classes = training_set.num_classes
class_names = list(training_set.class_indices.keys())
print(num_classes , class_names)

model = create_model()
history = train_model(model, training_set, test_set)
validate_model(model, training_set, test_set)
test_model(model, class_names)