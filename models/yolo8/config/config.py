import yaml

def load_config(path="../../config/yolo_8.yaml"):
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config