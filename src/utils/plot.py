import matplotlib.pyplot as plt
import pandas as pd
import os


def convergence_plot(cost, title, num_runs):
    """
    Plot the convergence of the optimization algorithm over iterations.

    Parameters:
        :param dict cost    : mean and best cost over cycles/generations as returned by an optimiser.
        :param str title    : title of the plot.
        :param int num_runs : number of runs.
    """
    plot_title = title + " - " + str(num_runs) + " algorithm runs"

    plt.figure(figsize=(12, 4))

    # plot the absolute regret and mean fitness over iterations
    plt.plot(range(len(cost["abs_regret"])), cost["abs_regret"], label="Average Absolute Regret", color="red", linewidth=2)
    plt.scatter(range(len(cost["mean"])), cost["mean"], label="Average Mean Fitness", color="gray", linestyle="--", s=5, alpha=0.2)

    # configure labels, title, and legend
    plt.xlabel("Iteration #")
    plt.ylabel("Absolute Regret")
    plt.title(plot_title)
    plt.legend(loc="best", fontsize="large")

    # set x-axis limits and add a grid for readability
    plt.xlim([0, len(cost["mean"])])
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # define output path and save the figure
    output_path = f"output/{title}"
    fig = os.path.join(output_path, f"convergence_plot_{num_runs}_runs.png")
    plt.savefig(fig)
    plt.clf()

def bee_plot(pos_data, title, output_path):
    """
    Create histograms of bee positions for all roles combined (Worker, Onlooker, Scout) for each dimension.

    Parameters:
        :param pd.DataFrame pos_data: dataframe with positions and roles.
        :param str title            : title of the plot.
        :param str output_path      : path to save the plot.
    """
    # extract all dimensions from the dataset
    dimensions = [col for col in pos_data.columns if col.startswith("dim_")]
    roles = pos_data["bee_Role"].unique()

    # generate histograms for each dimension
    for dim in dimensions:
        plt.figure(figsize=(10, 6))
        # plot histograms for each role
        for role in roles:
            data = pos_data[pos_data["bee_Role"] == role]
            plt.hist(data[dim], bins=30, alpha=0.5, label=role)

        # configure labels, title, and legend
        plt.title(f"{title} - Dimension {dim}")
        plt.xlabel("Position in Dimension i")
        plt.ylabel("Total Bee Occurrences")
        plt.legend()
        plt.grid(True)

        # save plot
        fig = os.path.join(output_path, f"{dim}_all_roles.png")
        plt.savefig(fig)
        plt.clf()
        plt.close()

def bee_plot_detail(pos_data, title, output_path):
    """
    Create separate histograms of bee positions for each role individually, for each dimension.

    Parameters:
        :param pd.DataFrame pos_data: dataframe with positions and roles.
        :param str title            : title of the plot.
        :param str output_path      : path to save the plot.
    """
    # extract all dimensions from the dataset
    dimensions = [col for col in pos_data.columns if col.startswith("dim_")]
    roles = pos_data["bee_Role"].unique()

    # generate histograms for each dimension
    for dim in dimensions:
        plt.figure(figsize=(12, 4 * len(roles)))
        # plot histograms for each role
        for i, role in enumerate(roles):
            data = pos_data[pos_data["bee_Role"] == role]
            plt.subplot(len(roles), 1, i + 1)
            plt.hist(data[dim], bins=50, alpha=0.7, label=role, color="C" + str(i))

            # configure labels, title, and legend
            plt.title(f"{title} - {role} Positions in Dimension {dim}")
            plt.xlabel("Position in Dimension i")
            plt.ylabel("Total Bee Occurrences")
            plt.legend(loc="upper right")
            plt.grid(True)

        # save plot
        plt.tight_layout()
        fig = os.path.join(output_path, f"{dim}_by_role_and_dimension.png")
        plt.savefig(fig)
        plt.clf()
        plt.close()

def bee_plot_behaviour(pos_data, title, output_path):
    """
    Plot how bee positions change over iterations for each dimension and role dividing by role.
    Visualize how the swarm converges over time, and how different roles cluster in the search space during the run.

    Parameters:
        :param pd.DataFrame pos_data: dataframe with positions and roles.
        :param str title            : title of the plot.
        :param str output_path      : path to save the plot.
    """
    unique_roles = pos_data["bee_Role"].unique()
    dimensions = [col for col in pos_data.columns if col.startswith("dim_")]

    for dim in dimensions:
        plt.figure(figsize=(10, 6))
        # scatter plot for each role
        for i, role in enumerate(unique_roles):
            role_data = pos_data[pos_data["bee_Role"] == role]
            plt.scatter(role_data["index"], role_data[dim], alpha=0.2 if role == "Worker" else 0.7,
                        label=role, color="C" + str(i), s=10)

        # configure labels, title, and legend
        plt.title(f"Bee Positions in {dim} Over Iterations - {title}")
        plt.xlabel("Iteration #")
        plt.ylabel("Position in Dimension i")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()

        # save plot
        fig_path = os.path.join(output_path, f"bee_position_{dim}.png")
        plt.savefig(fig_path)
        plt.clf()
        plt.close()

def plot_utilization(foldername, filename):
    """
    Plot the utilization schedule of bees based on logged timing data and save the plot.

    Parameters:
        :param str foldername   : folder where the plot will be saved
        :param str filename     : path (relative to the output folder) of the CSV log file.
    """
    df = pd.read_csv(f"output/{filename}")

    required_columns = {'bee_index', 'start_clock_time', 'update_time', 'execution_time'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    df['start_time'] = df['start_clock_time'] + df['update_time']
    df['end_time'] = df['start_time'] + df['execution_time']

    colors = {'Initialization': 'lime',
              'Worker_BO': 'purple',
              'Worker': 'red',
              'Onlooker': 'orange',
              'Onlooker_Levy': 'gold',
              'Scout': 'blue'}

    first_occurence = {'Initialization': True,
                       'Worker_BO': True,
                       'Worker': True,
                       'Onlooker': True,
                       'Onlooker_Levy': True,
                       'Scout': True}

    plt.figure(figsize=(10, 5))
    for _, row in df.iterrows():
        role = row['bee_Role']
        if row['index'] >= 0:
            if first_occurence[role]:
                plt.plot([row['start_time'], row['end_time']], [row['bee_index']] * 2, color=colors[role], label=role)
                first_occurence[role] = False
            else:
                plt.plot([row['start_time'], row['end_time']], [row['bee_index']] * 2, color=colors[role])

    plt.xlabel('Time [s]')
    plt.ylabel('Bee ID')
    plt.xlim([None, max(df['end_time']) * 1.25])
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"output/{foldername}/utilization.pdf", dpi=300, format='pdf')