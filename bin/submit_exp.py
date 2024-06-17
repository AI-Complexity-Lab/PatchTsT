import os

# exp_ids = [10, 11, 12, 13, 14, 15, 16, 17]
exp_ids = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# exp_ids = [10, 11, 12]

for exp_id in exp_ids:
    os.system(f'sbatch gl_job.sh {exp_id}')
