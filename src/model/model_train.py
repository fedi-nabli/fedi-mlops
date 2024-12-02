import yaml

def train_model(path, model, training_set, test_set):
  with open(f'{path}/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

  model.compile(optimizer = config['optimizer'], loss = config['loss'], metrics = ['accuracy'])
  history= model.fit(training_set,
                            epochs = int(config['epochs']),
                            validation_data = test_set)
  # list all data in history
  print(history.history.keys())
  
  return history