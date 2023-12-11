import yaml

yaml_path = '../data/data_loading/data_sources.yaml'

with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)

sorted_data = {}
for key, value in sorted(data.items()):
    if isinstance(value['urls'], list):
        value['urls'] = sorted(value['urls'])
    sorted_data[key] = value

with open(yaml_path, 'w') as file:
    yaml.dump(sorted_data, file, default_flow_style=False)
