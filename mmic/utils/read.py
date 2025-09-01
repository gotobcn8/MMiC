import yaml

def yaml_read(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
        result = yaml.load(data, Loader=yaml.FullLoader)
    return result
