import json

RESULTS_PATH = './metrics.json'

def read_data(path):
    with open(path, 'r') as file:
        return json.load(file)

def write_results(results):
    with open(RESULTS_PATH, 'w') as file:
        file.write(json.dumps(results))