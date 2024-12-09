import os
import yaml
import shutil

def load_data(path, train_datagen, test_datagen):
  with open(f'{path}/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

  data_gen_path = f'{path}/{config['data_gen_path']}'
  shutil.rmtree(data_gen_path)
  os.makedirs(data_gen_path, exist_ok=True)
  os.makedirs(f'{data_gen_path}/training', exist_ok=True)
  os.makedirs(f'{data_gen_path}/test', exist_ok=True)
  
  training_set = train_datagen.flow_from_directory(f'{path}/{config['data_path']}/{config['training_set_path']}',
                                                  target_size = (config['target_size'], config['target_size']),
                                                  batch_size = int(config['batch_size']),
                                                  class_mode = 'categorical',
                                                  save_to_dir=f'{path}/{config['data_gen_path']}/training',
                                                  save_prefix='aug',
                                                  save_format='jpeg')

  test_set = test_datagen.flow_from_directory(f'{path}/{config['data_path']}/{config['test_set_path']}',
                                                  target_size = (config['target_size'], config['target_size']),
                                                  batch_size = int(config['batch_size']),
                                                  class_mode = 'categorical',
                                                  save_to_dir=f'{path}/{config['data_gen_path']}/test',
                                                  save_prefix='aug',
                                                  save_format='jpeg')

  return training_set, test_set

if __name__ == '__main__':
  load_data('../../')