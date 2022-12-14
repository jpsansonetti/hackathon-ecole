import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from tqdm.notebook import tqdm
import os, sys, pickle, glob, gc
from collections import Counter
#import cudf, itertools
import pyarrow.parquet as pq
import numpy as np
#print('We will use RAPIDS version',cudf.__version__)

# pyarrow.parquet IS IMPORTED AS pq.
train_parquet_file = pq.ParquetFile('train.parquet')
test_parquet_file = pq.ParquetFile('test.parquet')
df_batches = []
# YOU CAN TUNE batch_size TO YOUR NEEDS. IT REPRESENTS THE ROW NUMBER FOR EACH BATCH.
for batch in train_parquet_file.iter_batches(batch_size=65536*100):
    df_batches.append(batch.to_pandas())
for batch in test_parquet_file.iter_batches(batch_size=65536*100):
    df_batches.append(batch.to_pandas())
    
len(df_batches) # NOW WE HAVE DATASET DIVIDED INTO MULTIPLE PARTS
print(df_batches)
