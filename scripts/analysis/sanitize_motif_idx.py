import json

with open("samples_metadata.json") as fp:
    metadata = json.load(fp)

faulty_metadata = []
for name, entry in metadata.items():
    fixed_res = entry['fixed_bb_res_idx']
    fixed_seq = entry['fixed_seq_res_idx']

    if len(set(fixed_res)) < len(fixed_res):
        faulty_metadata.append(name)
    elif len(set(fixed_seq)) < len(fixed_seq):
        faulty_metadata.append(name)

print(len(faulty_metadata), faulty_metadata)
with open("exclude_samples.json", 'w') as fp:
    json.dump(faulty_metadata, fp)