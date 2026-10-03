import json
import os

import pandas as pd

cluster_tsv = os.environ.get(
    "AFDB_CLUSTER_TSV", "1-AFDBClusters-entryId_repId_taxId.tsv"
)
df = pd.read_csv(cluster_tsv, sep='\t', header=None)
# yea its code golf but it works LOL
mapping_df = df.groupby(1)[0].apply(list)
mapping = mapping_df.to_dict()

df[1] = "AF-".lower() + df[1].str.lower() + "-F1-model_v4_1".lower()
df[0] = "AF-".lower() + df[0].str.lower() + "-F1-model_v4_1".lower()
mapping_df = df.groupby(1)[0].apply(list)
mapping = mapping_df.to_dict()

clustering = {}
for key, values in mapping.items():
    for item in values:
        clustering[item] = key

with open("clustering.json", 'w') as fp:
    json.dump(clustering, fp)