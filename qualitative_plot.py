import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("TkAgg")  # comment out if not needed
import matplotlib.pyplot as plt
import argparse
import os
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 18,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


FIG_WIDTH = 12.0
LEFT_MARGIN = 0.15
RIGHT_MARGIN = 0.95

def parse_action(action_str):

    if isinstance(action_str, str) and action_str.strip():
        return [int(x) for x in action_str.split(";")]
    return []

def parse_latencies_json(lat_json_str):

    if not isinstance(lat_json_str, str):
        return {}
    txt = lat_json_str.replace('""', '"')
    try:
        d = json.loads(txt)
        # keys may be strings; normalize to int
        return {int(k): float(v) for k, v in d.items()}
    except Exception:
        return {}
def build_qualitative_view(raw_path, algo_name="CAR-UCB", trial_id=0, top_n_paths=3):

    df = pd.read_csv(raw_path)


    df_env = df[(df["algorithm"] == "FT-Flood") & (df["trial_id"] == trial_id)].copy()
    df_env = df_env.sort_values("t")

    #parse latencies_json for FT-Flood
    df_env["lat_dict"] = df_env["latencies_json"].apply(parse_latencies_json)

    #determine set of all paths from FT-Flood
    all_paths = sorted({p for d in df_env["lat_dict"] for p in d.keys()})
    K = len(all_paths)


    env_t = df_env["t"].to_numpy()


    path_index = {p: i for i, p in enumerate(all_paths)}
    ground_truth_lat = np.full((len(env_t), K), np.nan)

    for i, d in enumerate(df_env["lat_dict"]):
        for p, lat in d.items():
            j = path_index[p]
            ground_truth_lat[i, j] = lat


    df_car = df[(df["algorithm"] == algo_name) & (df["trial_id"] == trial_id)].copy()
    df_car = df_car.sort_values("t")

    df_car["action_list"] = df_car["action"].apply(parse_action)
    car_t = df_car["t"].to_numpy()
    car_action_size = df_car["action_size"].to_numpy()


    from collections import Counter
    c = Counter()
    for acts in df_car["action_list"]:
        c.update(acts)


    top_paths = [p for p, _ in c.most_common(top_n_paths)]
    top_paths = [p for p in top_paths if p in path_index]  # ensure present in env


    common_ts = np.intersect1d(env_t, car_t)
    env_idx = {t: i for i, t in enumerate(env_t)}
    car_idx = {t: i for i, t in enumerate(car_t)}

    # Build aligned arrays
    aligned_lat = []
    aligned_sel_matrix = []
    aligned_action_size = []

    for t in common_ts:
        i_env = env_idx[t]
        i_car = car_idx[t]


        lat_row = []
        for p in top_paths:
            j = path_index[p]
            lat_row.append(ground_truth_lat[i_env, j])
        aligned_lat.append(lat_row)


        acts = df_car.iloc[i_car]["action_list"]
        sel_row = [1 if p in acts else 0 for p in top_paths]
        aligned_sel_matrix.append(sel_row)


        aligned_action_size.append(car_action_size[i_car])

    aligned_lat = np.array(aligned_lat)
    aligned_sel_matrix = np.array(aligned_sel_matrix)
    aligned_action_size = np.array(aligned_action_size)

    return {
        "timesteps": common_ts,
        "top_paths": top_paths,
        "latencies": aligned_lat,
        "selection_matrix": aligned_sel_matrix,
        "action_size": aligned_action_size,
    }

def plot_latency_only(data, scenario_label="K = 50", T_MAX=2000):

    t = data["timesteps"][:T_MAX]
    lat = data["latencies"][:T_MAX]
    top_paths = data["top_paths"]


    fig, ax1 = plt.subplots(
        1, 1,
        figsize=(FIG_WIDTH+ 0.5, 4.3),
    )
    ax1.set_xlim(0, T_MAX)

    for i, p in enumerate(top_paths):
        ax1.plot(t, lat[:, i], label=f"Path {p}", linewidth=1.8)


    deadline = 100  # ms
    ax1.axhline(deadline, color='red', linestyle='--', linewidth=1.5)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_xlabel("Time step $t$")
    ax1.grid(True, alpha=0.3)


    ax1.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=6,
        frameon=False,
    )

    plt.tight_layout()
    #plt.show()
    fig.savefig("plot/new_latency_shared.pdf", bbox_inches="tight")


def plot_algorithm_details(data, algo_name="CAR-UCB", T_MAX=2000):

    t = data["timesteps"][:T_MAX]
    sel = data["selection_matrix"][:T_MAX]
    action_size = data["action_size"][:T_MAX]
    top_paths = data["top_paths"]
    num_top = len(top_paths)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(FIG_WIDTH, 3.9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.3, 0.5]}
    )

    ax2, ax3 = axes

    sorted_paths = sorted(top_paths)


    path_to_old_idx = {p: i for i, p in enumerate(top_paths)}
    row_order = [path_to_old_idx[p] for p in sorted_paths]


    sel_sorted = sel[:, row_order]

    im = ax2.imshow(
        sel_sorted[:T_MAX].T,
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        extent=[t.min(), t.max(), -0.5, num_top - 0.5],
        origin="lower",
    )
    ax2.set_yticks(range(num_top))
    ax2.set_yticklabels([f"Path {p}" for p in sorted_paths])
    ax2.set_ylabel("Selected paths")
    ax2.set_title(f"Path Selection (black = selected)")  # Added algo_name to title
    ax2.grid(False)


    ax3.plot(t, action_size, linewidth=1.8, color='tab:blue')  # Explicit color is safer
    ax3.set_xlabel("Time step $t$")
    ax3.set_ylabel("Action size\n$|A_t|$")
    ax3.grid(True, alpha=0.3)

    ax2.set_xlim(0, T_MAX)
    #plt.show()
    fig.savefig(f"plot/new_{algo_name}_details.pdf", bbox_inches="tight")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-trial network path selection experiments"
    )

    parser.add_argument("--num-interfaces", type=int, default=NUM_INTERFACES,
                        help="Number of network interfaces")
    parser.add_argument("--num-servers", type=int, default=NUM_SERVERS,
                        help="Number of servers")
    parser.add_argument("--trial-id", type=int, default=TRIAL_ID,
                        help="trial id in the data when run with multiple trial/seeds")

    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs("plot", exist_ok=True)
    NUM_INTERFACES = 2
    NUM_SERVERS = 3
    TRIAL_ID = 10
    args = parse_args()

    # Override global configuration
    NUM_INTERFACES = args.num_interfaces
    NUM_SERVERS = args.num_servers
    TRIAL_ID = args.trial_id
    K = NUM_INTERFACES * NUM_SERVERS
    print("=== Experiment Configuration ===")
    print(f"NUM_INTERFACES      = {NUM_INTERFACES}")
    print(f"NUM_SERVERS         = {NUM_SERVERS}")
    print(f"TRIAL_ID         = {TRIAL_ID}")

    algorithms = ["CAR-UCB", "PLR", "Oracle", "FT-Flood"]
    raw_path = f"data_output/final_raw_steps_all_{NUM_INTERFACES}_{NUM_SERVERS}.csv"

    chose_alg_ref = algorithms[0]
    data_ref = build_qualitative_view(raw_path, algo_name=chose_alg_ref, trial_id=TRIAL_ID, top_n_paths=6)

    #plot the shared Latency Panel
    plot_latency_only(
        data_ref,
        scenario_label=f"Inter={NUM_INTERFACES}, Serv={NUM_SERVERS}",
        T_MAX=2000
    )

    #plot the Specific Algorithm Panels
    #for CAR-UCB
    target_alg = "CAR-UCB"
    data_car = build_qualitative_view(raw_path, algo_name=target_alg, trial_id=TRIAL_ID, top_n_paths=6)

    plot_algorithm_details(
        data_car,
        algo_name=target_alg,
        T_MAX=2000
    )

    #for PLR
    target_alg_2 = "PLR"
    data_plr = build_qualitative_view(raw_path, algo_name=target_alg_2, trial_id=TRIAL_ID, top_n_paths=6)

    plot_algorithm_details(
        data_plr,
        algo_name=target_alg_2,
        T_MAX=2000
    )

    # for Oracle
    target_alg_2 = "Oracle"
    data_plr = build_qualitative_view(raw_path, algo_name=target_alg_2, trial_id=TRIAL_ID, top_n_paths=6)

    plot_algorithm_details(
        data_plr,
        algo_name=target_alg_2,
        T_MAX=2000
    )

    # for FT-Flood
    target_alg_2 = "FT-Flood"
    data_plr = build_qualitative_view(raw_path, algo_name=target_alg_2, trial_id=TRIAL_ID, top_n_paths=6)

    plot_algorithm_details(
        data_plr,
        algo_name=target_alg_2,
        T_MAX=2000
    )

    print("Done... All figures are saved in /plot")
