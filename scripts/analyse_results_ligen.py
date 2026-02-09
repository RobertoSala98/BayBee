from statistics import mean
import pandas as pd
from numpy import inf
import csv
import os

maxs = {
    2.0: 567.312555400384,
    2.1: 567.312555400384,
    2.2: 464.349441178703,
    2.45: 424.355184049556,
    2.75: 340.099251789615
        }

print_single_stats = False

header = [['Constraint value', 'Type', 'MLC', 'MLT', 'MAPR [%]', 'feas_rate [%]']]
ml_values = ['global', 'local', 'none']
data = []

for constraint_value in [2.0, 2.1, 2.2, 2.45, 2.75]:
    global_opt = maxs[constraint_value]

    for algo in ['ABC-MLCO-aLMem', 'ABC-MLCO-aGMem', 'ABC-MLCO-aLMem-BO', 'ABC-MLCO-aGMem-BO']:
        for xxx1 in ml_values:
            for xxx2 in ml_values:
                algo_name = f'Async ABC - {algo}-{xxx1}-{xxx2}'
                print('Algorithm:', algo_name, 'Constraint:', constraint_value)

                MAPRs = []
                unique_pts = []
                feasible_pts = []

                if algo == 'ABC-MLCO-aLMem':
                    for run in range(10):
                        filename = f'output/output_noBO/{constraint_value}/{xxx1}/{xxx2}/noBO/localMemory_Cache/abc_run{run + 1}/bee_logrun_{run + 1}.csv'
                        df = pd.read_csv(filename)

                        valid_df = df[df['valid'] >= 1.0]
                        min_OF = valid_df['objective_function'].min()
                        min_regret = (min_OF - global_opt)/abs(global_opt)*100
                        if print_single_stats: print(f"Minimum regret: {round(min_regret, 2)} %, Best OF: {min_OF}")
                        MAPRs.append(min_regret)

                        unique_positions = df['position'].nunique()
                        total_rows = len(df)
                        if print_single_stats: print(f"Percentage of unique points: {round(unique_positions/total_rows*100,2)} %")
                        unique_pts.append(unique_positions/total_rows*100)

                        feasible_points = valid_df['position'].nunique()
                        if print_single_stats: print(f"Percentage of feasible points: {round(feasible_points/ unique_positions*100,2)} %")
                        feasible_pts.append(feasible_points/ unique_positions*100)

                if algo == 'ABC-MLCO-aGMem':

                    for run in range(10):
                        filename = f'output/output_noBO/{constraint_value}/{xxx1}/{xxx2}/noBO/globalMemory_noCache/abc_run{run + 1}/bee_logrun_{run + 1}.csv'
                        df = pd.read_csv(filename)

                        valid_df = df[df['valid'] >= 1.0]
                        min_OF = valid_df['objective_function'].min()
                        min_regret = (min_OF - global_opt)/abs(global_opt)*100
                        if print_single_stats: print(f"Minimum regret: {round(min_regret, 2)} %, Best OF: {min_OF}")
                        MAPRs.append(min_regret)

                        unique_positions = df['position'].nunique()
                        total_rows = len(df)
                        if print_single_stats: print(f"Percentage of unique points: {round(unique_positions/total_rows*100,2)} %")
                        unique_pts.append(unique_positions/total_rows*100)

                        feasible_points = valid_df['position'].nunique()
                        if print_single_stats: print(f"Percentage of feasible points: {round(feasible_points/ unique_positions*100,2)} %")
                        feasible_pts.append(feasible_points/ unique_positions*100)

                if algo == 'ABC-MLCO-aLMem-BO':

                    for run in range(10):
                        filename = f'output/output_BO/{constraint_value}/{xxx1}/{xxx2}/BO/localMemory_Cache/abc_run{run + 1}/bee_logrun_{run + 1}.csv'
                        df = pd.read_csv(filename)

                        valid_df = df[df['valid'] >= 1.0]
                        min_OF = valid_df['objective_function'].min()
                        min_regret = (min_OF - global_opt)/abs(global_opt)*100
                        if print_single_stats: print(f"Minimum regret: {round(min_regret, 2)} %, Best OF: {min_OF}")
                        MAPRs.append(min_regret)

                        unique_positions = df['position'].nunique()
                        total_rows = len(df)
                        if print_single_stats: print(f"Percentage of unique points: {round(unique_positions/total_rows*100,2)} %")
                        unique_pts.append(unique_positions/total_rows*100)

                        feasible_points = valid_df['position'].nunique()
                        if print_single_stats: print(f"Percentage of feasible points: {round(feasible_points/ unique_positions*100,2)} %")
                        feasible_pts.append(feasible_points/ unique_positions*100)

                if algo == 'ABC-MLCO-aGMem-BO':

                    for run in range(10):
                        filename = f'output/output_BO/{constraint_value}/{xxx1}/{xxx2}/BO/globalMemory_noCache/abc_run{run + 1}/bee_logrun_{run + 1}.csv'
                        df = pd.read_csv(filename)

                        valid_df = df[df['valid'] >= 1.0]
                        min_OF = valid_df['objective_function'].min()
                        min_regret = (min_OF - global_opt)/abs(global_opt)*100
                        if print_single_stats: print(f"Minimum regret: {round(min_regret, 2)} %, Best OF: {min_OF}")
                        MAPRs.append(min_regret)

                        unique_positions = df['position'].nunique()
                        total_rows = len(df)
                        if print_single_stats: print(f"Percentage of unique points: {round(unique_positions/total_rows*100,2)} %")
                        unique_pts.append(unique_positions/total_rows*100)

                        feasible_points = valid_df['position'].nunique()
                        if print_single_stats: print(f"Percentage of feasible points: {round(feasible_points/ unique_positions*100,2)} %")
                        feasible_pts.append(feasible_points/ unique_positions*100)

                print("-------------------------------------------------------")
                print(f"MAPR: {round(mean(MAPRs),2)} %")
                print(f"Percentage of unique points: {round(mean(unique_pts),2)} %")
                print(f"Feasibility rate: {round(mean(feasible_pts),2)} %")
                print("-------------------------------------------------------")
                print("-------------------------------------------------------")

                data.append([constraint_value, algo, xxx1, xxx2, float(round(mean(MAPRs),2)), round(mean(feasible_pts),2)])

with open('data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(header)
    writer.writerows(data)