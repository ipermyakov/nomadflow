import pandas as pd

from nomadflow.benchmark.success_criteria import filter_one_shot, filter_pmpnn_esmfold

sc_df = pd.read_csv("esmfold/sc_rmsd.csv")
folding_df = pd.read_csv("esmfold/folding_rmsd.csv")

all_tasks = sorted(folding_df['task'].unique().tolist())

pmpnn_pass = filter_pmpnn_esmfold(sc_df)
pmpnn_pass = pmpnn_pass.groupby("name").sample(1)
pmpnn_task_pass = pmpnn_pass.groupby("task").count()

designable = sc_df[(sc_df['motif_all_atom_rmsd'] < 1.5) & (sc_df['global_bb_rmsd'] < 2)]

# print(pmpnn_task_pass)
# print(pmpnn_task_pass.index)
# pmpnn_passed_tasks = pmpnn_task_pass.sample(1)['task'].unique().tolist()

pz_pass = filter_one_shot(folding_df)
pz_task_pass = pz_pass.groupby("task").count()
# print(pz_task_pass)
# pz_passed_tasks = pz_task_pass.sample(1)['task'].unique().tolist()

with open("tasks_passed.txt", 'w') as fp:
    fp.write("PMPNN pass\n")
    for task in all_tasks:
        if task in pmpnn_task_pass.index:
            pass_rate = int(pmpnn_task_pass.loc[task]['name']) / 100
            fp.write(f"{task}: {pass_rate}\n")
        else:
            fp.write(f"{task}: 0.0\n")
    fp.write("\n")
    fp.write("PZ pass\n")
    for task in all_tasks:
        if task in pz_task_pass.index:
            pass_rate = int(pz_task_pass.loc[task]['name']) / 100
            fp.write(f"{task}: {pass_rate}\n")
        else:
            fp.write(f"{task}: 0\n")