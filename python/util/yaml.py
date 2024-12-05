import yaml

def load_yaml(config_path):
  with open(config_path, 'r') as yml:
    config = yaml.safe_load(yml)
  
  return config