import redis
import json
import tqdm
import argparse
import os
import pickle

def chunk_data(data, num_chunks):
    keys = list(data.items())
    num_chunks = max(1, num_chunks)
    chunk_size = len(keys) // num_chunks
    chunk_size = max(1, chunk_size)
    chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
    return chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Load a dictionary into a redis database file")
    parser.add_argument("--dict", type=str)
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument(
        "--pickle-values",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Serialize values with pickle before writing to Redis.",
    )
    args = parser.parse_args()

    ext = os.path.splitext(args.dict)[-1]

    if ext == '.pkl':
        with open(args.dict, 'rb') as fp:
            data = pickle.load(fp)
    elif ext == '.json':
        with open(args.dict) as fp:
            data = json.load(fp)
    else:
        raise ValueError("we only support input dictionaries in .pkl or .json format")

    r = redis.Redis(host='localhost', port=args.port, db=0)

    data_chunks = chunk_data(data, len(data) // args.chunk_size)
    for chunk in tqdm.tqdm(data_chunks):
        pipeline = r.pipeline()

        for key, value in chunk:
            if args.pickle_values:
                pipeline.set(key, pickle.dumps(value))
            else:
                pipeline.set(key, value)

        # Execute the pipeline
        pipeline.execute()

    r.save()