import subprocess

def update_date_dvc(data_version):
  subprocess.run(['dvc', 'add', 'data'])
  subprocess.run(['git', 'add', 'data.dvc'])
  subprocess.run(['git', 'commit', '-m', 'create new data version'])
  subprocess.run(['git', 'tag', '-a', f'{data_version}', '-m', f'create new data version {data_version}'])
  subprocess.run(['git', 'push'])

if __name__ == '__main__':
  import yaml

  with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

  data_version = config['data_version']
  data_version = int(data_version) + 1
  config['data_version'] = data_version
  
  with open('config.yaml', 'w') as file:
    yaml.dump(config, file)

  update_date_dvc(data_version)