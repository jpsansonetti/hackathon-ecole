import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from tqdm.notebook import tqdm
import os, sys, pickle, glob, gc
from collections import Counter
#import cudf, itertools
import pyarrow.parquet as pq
import numpy as np
#print('We will use RAPIDS version',cudf.__version__)

from surprise import prediction_algorithms
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt
import time



# pyarrow.parquet IS IMPORTED AS pq.
train_parquet_file = pq.ParquetFile('train.parquet')
test_parquet_file = pq.ParquetFile('test.parquet')
df_batches = []
# YOU CAN TUNE batch_size TO YOUR NEEDS. IT REPRESENTS THE ROW NUMBER FOR EACH BATCH.
for batch in train_parquet_file.iter_batches(batch_size=65536*100):
    df_batches.append(batch.to_pandas())
for batch in test_parquet_file.iter_batches(batch_size=65536*100):
    df_batches.append(batch.to_pandas())
    
algo = prediction_algorithms.KNNBasic()
print('start')
i = 0
for df in df_batches:
    print(i)
    i+=1

    n_folds = 5

    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader()

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[['session', 'aid', 'type']], reader)

    # We can now use this dataset as we please, e.g. calling cross_validate
    res_KNNBasic = cross_validate(algo, data, cv=n_folds, n_jobs=-1)


