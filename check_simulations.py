import os

output_folder = 'output/ex2_hartmann'

folder_list = ['num_bees', 'ML_approaches', 'memory', 'max_trials', 'levy_proba', 'BO_steps']

subfolder_list = {
    'BO_steps': ['0', '1', '3', '5'],
    'levy_proba': ['0.1', '0.05', '0.01', '0.2'],
    'max_trials': ['3', '5', '10', '15', '30'],
    'memory': ['local', 'global'],
    'ML_approaches': ['none_none', 'none_local', 'none_global', 'local_none', 'local_local', 'local_global', 'global_none', 'global_local', 'global_global'],
    'num_bees': ['10_160', '20_80', '40_40', '80_20', '160_10']
}

repetitions = 30

for folder in folder_list:
    for subfolder in subfolder_list[folder]:
        for idx in range(repetitions):
            run_name = f'abc_run{idx+1}'
            act_folder = os.path.join(output_folder, folder, subfolder, run_name)

            if not os.path.isfile(os.path.join(act_folder, "summary.json")):
                print(os.path.join(act_folder, "summary.json"))
        