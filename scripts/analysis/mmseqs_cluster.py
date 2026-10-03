import os
import subprocess
from biotite.sequence.io import fasta

# from jason yim multiflow
def run_easy_cluster(precise_fasta, output_dir):
    # designable_dir should be a directory with individual PDB files in it that we want to cluster
    # output_dir is where we are going to save the easy cluster output files

    # Returns the number of clusters

    easy_cluster_args = [
        'mmseqs',
        'easy-cluster',
        precise_fasta,
        os.path.join(output_dir, 'res'),
        output_dir,
    ]
    # print(easy_cluster_args)
    process = subprocess.Popen(
        easy_cluster_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    # print(stdout.decode())
    # print(stderr.decode())
    del stdout # We don't actually need the stdout, we will read the number of clusters from the output files
    del stderr
    rep_seq_fasta = fasta.FastaFile.read(os.path.join(output_dir, 'res_rep_seq.fasta'))
    return len(rep_seq_fasta)


if __name__ == '__main__':
    os.system("~/software/bin/pdb2fasta precise_samples_folded/* > precise_samples.fa")
    os.makedirs("precise_seq_clusters", exist_ok=True)
    num_clusters = run_easy_cluster("precise_samples.fa", "precise_seq_clusters/")
    with open("num_mmseqs_precise_clusters.txt", "w") as fp:
        fp.write(f"Num clusters: {num_clusters}")

    os.system("~/software/bin/pdb2fasta consistent_samples_folded/* > consistent_samples.fa")
    os.makedirs("consistent_seq_clusters", exist_ok=True)
    num_clusters = run_easy_cluster("consistent_samples.fa", "consistent_seq_clusters/")
    with open("num_mmseqs_consistent_clusters.txt", "w") as fp:
        fp.write(f"Num clusters: {num_clusters}")