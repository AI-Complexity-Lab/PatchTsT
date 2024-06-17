import os

input_param_file_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# get pis
for idx in input_param_file_idxs:
    os.system(f'python aci.py -i {idx}')

seeds = [1, 2, 3, 4, 1, 2, 1, 2, 3, 1, 2, 3]
exp_id_base = 4800
# get viz
for i in range(len(input_param_file_idxs)):
    exp_id = exp_id_base + input_param_file_idxs[i]
    seed = seeds[i]
    os.system(f'python eval.py --exp_id {exp_id} --seed {seed} --viz')