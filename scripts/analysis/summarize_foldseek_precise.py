import pandas as pd

df = pd.read_csv("novelty/aln_PDB_precise.tsv", sep="\t", header=None)

mean_novelty_tm = df.groupby(0).max(2).mean()[2]
min_novelty_tm = df.groupby(0).max(2).min()[2]

with open("pdb_novelty_summary_precise.txt", "w") as fp:
    fp.write(f"Mean of max TM: {mean_novelty_tm}\n")
    fp.write(f"Min of max TM: {min_novelty_tm}\n")

df_list = []
for i in ['uniprot50', 'proteome', 'swiss_prot']:
    df_list.append(pd.read_csv(f"novelty/aln_AFDB_{i}_precise.tsv", sep="\t", header=None))

df = pd.concat(df_list)
mean_novelty_tm = df.groupby(0).max(2).mean()[2]
min_novelty_tm = df.groupby(0).max(2).min()[2]

with open(f"afdb_novelty_summary_precise.txt", "w") as fp:
    fp.write(f"Mean of max TM: {mean_novelty_tm}\n")
    fp.write(f"Min of max TM: {min_novelty_tm}\n")