import os
import pandas as pd
import numpy as np

output_dir = f"output/gemm_20_BO_3_mt10_async_constr_ridge_target_nn"

rep = 30

end_times_all = set()
run_dfs = []
final_values = []

for r in range(1, rep + 1):

    dir = os.path.join(output_dir, f"abc_run{r}")
    if not os.path.exists(os.path.join(dir, "summary.json")):
        continue

    csv_path = os.path.join(output_dir, f"abc_run{r}", f"bee_logrun_{r}.csv")

    df = pd.read_csv(csv_path)

    df = df[["fitness", "valid", "objective_function",
                "start_clock_time", "execution_time"]]

    df = df[df["valid"] == 1.0].copy()

    df["end_time"] = df["start_clock_time"] + df["execution_time"]

    df["best_OF"] = df["objective_function"].cummin()

    run_dfs.append(df[["end_time", "best_OF"]].copy())

    end_times_all.update(df["end_time"].values)

    final_values.append(df["best_OF"].iloc[-1])

mean_final = np.mean(final_values)

print("Mean final OF value:", mean_final)