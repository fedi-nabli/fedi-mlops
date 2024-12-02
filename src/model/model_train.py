def train_model(model, training_set, test_set):
  model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  history= model.fit(training_set,
                            epochs = 5,
                            validation_data = test_set)
  # list all data in history
  print(history.history.keys())
  
  return history