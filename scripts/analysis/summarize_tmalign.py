import pandas as pd

df = pd.read_csv("usalign.tsv", sep="\t")

tm1 = df["TM1"].mean()
tm2 = df["TM2"].mean()

with open("tm_diversity.txt", "w") as fp:
    fp.write(f"Average pairwise TM: {(tm1 + tm2)/2}")