import evaluate
from rouge import Rouge 
from external.moverscore.moverscore_v2 import word_mover_score
from collections import defaultdict
from external.BARTScore.bart_score import BARTScorer


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
    metric = Rouge()
    results = metric.get_scores(hyps=prediction, refs=reference)
    data[reference][metric_name+'-1'] = results[0][metric_name+'-1']['f']
    data[reference][metric_name+'-2'] = results[0][metric_name+'-2']['f']
    data[reference][metric_name+'-l'] = results[0][metric_name+'-l']['f']

def add_bertscore(prediction, reference, data, lang):
    metric_name = 'bertscore'
    metric = evaluate.load(metric_name)
    results = metric.compute(predictions=[prediction], references=[reference], lang=lang)
    data[reference][metric_name+'-p']  = results['precision'][0]
    data[reference][metric_name+'-r']  = results['recall'][0]
    data[reference][metric_name+'-f1'] = results['f1'][0]

def add_moverscore(prediction, reference, data):
    metric_name = 'moverscore'
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    results = word_mover_score([reference], [prediction], idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    data[reference][metric_name] = results[0]

def add_bartscore(prediction, reference, data):
    metric_name = 'bartscore'
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    results = bart_scorer.score([prediction], [reference], batch_size=4)
    data[reference][metric_name] = results[0]

def add_metrics(prediction, reference, data, lang='en'):
    add_bleu(prediction, reference, data)
    add_rouge(prediction, reference, data)
    add_meteor(prediction, reference, data)
    add_bertscore(prediction, reference, data, lang)
    add_moverscore(prediction, reference, data)
    add_bartscore(prediction, reference, data)


