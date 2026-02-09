import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys


def load_and_validate(files):
    dataframes = {}
    bees_list = []
    iterations_list = []

    for file in files:
        df = pd.read_csv(file)
        bees_list.append(df['bee_index'].iloc[0])
        iterations_list.append(df['index'].max())
        dataframes[file] = df

    return dataframes, bees_list, iterations_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Paths to log run files")
    args = parser.parse_args()

    dataframes, bees_list, iterations_list = load_and_validate(args.files)

    if len(set(bees_list)) > 1:
        bees = pd.Series(bees_list).mode()[0]
        for key in dataframes:
            dataframes[key] = dataframes[key][dataframes[key]['bee_index'] == bees]

    if len(set(iterations_list)) > 1:
        iterations = min(iterations_list)
        for key in dataframes:
            dataframes[key] = dataframes[key][dataframes[key]['index'] <= iterations]

    plt.figure(figsize=(10, 6))
    for file, df in dataframes.items():
        plt.plot(df["index"], df["regret"], alpha=0.6, linewidth=2, label=f"Run: {file}")
    plt.xlabel("Iteration #")
    plt.ylabel("Absolute Regret")
    plt.title(f"Comparative Average Best")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
