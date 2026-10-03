import redis
import json
import tqdm

from p_tqdm import p_umap
from functools import partial

import multiprocessing
from multiprocessing import Queue

with open("clustering.json") as fp:
    data = json.load(fp)

def chunk_data(data, num_chunks):
    keys = list(data.items())
    chunk_size = len(keys) // num_chunks
    chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
    return chunks

r = redis.Redis(host='localhost', port=7777, db=0)

data_chunks = chunk_data(data, len(data) // 1000)
for chunk in tqdm.tqdm(data_chunks):
    pipeline = r.pipeline()

    for key, value in chunk:
        pipeline.set(key, value)

    # Execute the pipeline
    pipeline.execute()

# process_data(data, num_processes=4)
r.save()