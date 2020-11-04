import json


def dump_json(file_path, value, method='w'):
    with open(file_path, method) as f:
        f.write(json.dumps(value, sort_keys=True, indent=4, separators=(',', ': ')))
        if method == 'a':
            f.write('\n')

def load_json(file_path):
    return json.load(file_path)
