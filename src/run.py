from metrics import add_metrics
from jsonHandler import write_results, read_data

results = {}

data = read_data('test.json')
for reference, prediction in data.items():
    results[reference] = { 'generated_text': prediction }
    add_metrics(prediction, reference, results)

write_results(results)