# importing module
import sys

# appending a path
sys.path.append('external/SUPERT')

# importing required module
from ref_free_metrics.supert import Supert
from utils.data_reader import CorpusReader

# read docs and summaries
reader = CorpusReader('data/topic_1')
source_docs = reader()
summaries = reader.readSummaries() 

print(source_docs)
print(summaries)

# compute the Supert scores
supert = Supert(source_docs) 
scores = supert(summaries)

print(scores)

