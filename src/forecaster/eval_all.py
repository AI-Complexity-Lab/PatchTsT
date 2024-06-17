import os

exp_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
seeds = [1, 2, 3, 4, 1, 2, 1, 2, 3, 1, 2, 3]
# exp_ids = [11]
# seeds = [2]


if __name__ == '__main__':
    for i, exp_id in enumerate(exp_ids):
        try:
            seed = seeds[i]
            input_file_id = exp_id
            print(input_file_id, seed, exp_id)
            os.system(f'python aci.py --input {input_file_id}')
            os.system(f'python eval.py --exp_id {exp_id} --seed {seed} --viz')
        except:
            print(exp_id)