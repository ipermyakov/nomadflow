from omegaconf import OmegaConf

# start = 60
# end = 128
# step = 1
# num_samples = 10
#
# tasks = []
# for length in range(start, end+1, step):
#     tasks.append({
#         "task": "unconditional",
#         "sample_length": length,
#         "num_samples": num_samples
#     })
num_samples = 5
tasks = []
for length in [60, 80, 100]:
    tasks.append({
        "task": "unconditional",
        "sample_length": length,
        "num_samples": num_samples
    })

conf = OmegaConf.create({
    "tasks": tasks
})
with open("config.yaml", 'w') as fp:
    OmegaConf.save(config=conf, f=fp)