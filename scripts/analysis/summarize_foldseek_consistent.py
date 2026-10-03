import pandas as pd

df = pd.read_csv("novelty/aln_PDB_consistent.tsv", sep="\t", header=None)

mean_novelty_tm = df.groupby(0).max(2).mean()[2]
min_novelty_tm = df.groupby(0).max(2).min()[2]

with open("pdb_novelty_summary_consistent.txt", "w") as fp:
    fp.write(f"Mean of max TM: {mean_novelty_tm}\n")
    fp.write(f"Min of max TM: {min_novelty_tm}\n")