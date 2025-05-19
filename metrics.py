import evaluate
from external.moverscore.moverscore_v2 import word_mover_score
from collections import defaultdict

def add_bleu(prediction, reference, data):
    metric_name = 'bleu'
    metric = evaluate.load(metric_name)
    results = metric.compute(predictions=[prediction], references=[reference])
    data[reference][metric_name] = results[metric_name]

def add_meteor(prediction, reference, data):
    metric_name = 'meteor'
    metric = evaluate.load(metric_name)
    results = metric.compute(predictions=[prediction], references=[reference])
    data[reference][metric_name] = results[metric_name]

def add_rouge(prediction, reference, data):
    metric_name = 'rouge'
    metric = evaluate.load(metric_name)
    results = metric.compute(predictions=[prediction], references=[reference])
    data[reference][metric_name] = results[metric_name]

def add_bertscore(prediction, reference, data, lang):
    metric_name = 'bertscore'
    metric = evaluate.load(metric_name)
    results = metric.compute(predictions=[prediction], references=[reference], lang=lang)
    data[reference][metric_name] = results['f1'][0]

def add_moverscore(prediction, reference, data):
    metric_name = 'moverscore'
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    results = word_mover_score([reference], [prediction], idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    data[reference][metric_name] = results[0]

def add_metrics(prediction, reference, data, lang='en'):
    add_bleu(prediction, reference, data)
    #add_rouge(prediction, reference, data)
    add_meteor(prediction, reference, data)
    add_bertscore(prediction, reference, data, lang)
    add_moverscore(prediction, reference, data)


