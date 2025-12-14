import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import argparse
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR_MAP = {
    "CAR-UCB" : "#1f77b4",   # blue
    "PLR"     : "#ff7f0e",   # orange

    "FT-Flood": "#9467bd",   # purple
    "Oracle"  : "#d62728"    # red
}

def compute_min_latency_column(raw_path):
    df = pd.read_csv(raw_path)

    def compute_min_latency(lat_json_str):
        try:
            if not isinstance(lat_json_str, str):
                return np.nan
            txt = lat_json_str.replace('""', '"')
            lat_dict = json.loads(txt)
            return float(min(lat_dict.values()))
        except Exception:
            return np.nan

    df["min_latency"] = df["latencies_json"].apply(compute_min_latency)
    df = df.dropna(subset=["min_latency"])
    df["min_latency"] = df["min_latency"].astype(float)
    return df



def compute_mean_regret_per_algorithm(df):

    mean_regret = (
        df.groupby("algorithm")["inst_regret"]
        .mean()
        .sort_values()
    )
    return mean_regret


def compute_mean_regret_stats(df_regret):
    stats = (
        df_regret
        .groupby(["algorithm", "trial_id"])["inst_regret"]
        .mean()
        .reset_index()
        .groupby("algorithm")["inst_regret"]
        .agg(["mean", "std"])
    )
    return stats




def compute_cumulative_regret(df_regret, algorithms=None):
    if algorithms is None:
        algorithms = sorted(df_regret["algorithm"].unique())

    cumreg_by_alg = {}

    for alg in algorithms:
        df_a = df_regret[df_regret["algorithm"] == alg]
        runs = []
        for trial_id in sorted(df_a["trial_id"].unique()):
            df_t = df_a[df_a["trial_id"] == trial_id].sort_values("t")
            reg = df_t["inst_regret"].to_numpy()
            cumreg = np.cumsum(reg)
            runs.append(cumreg)
        cumreg_by_alg[alg] = runs

    return cumreg_by_alg
def plot_cumulative_regret(cumreg_by_alg, T=None, title=None, ax = None):

    if ax is None:
        ax = plt.gca()
    #plt.figure(figsize=(7,5))

    for alg, runs in cumreg_by_alg.items():
        # pad to same length if needed
        max_len = max(len(r) for r in runs)
        arr = np.array([np.pad(r, (0, max_len - len(r)), mode="edge") for r in runs])

        mean_cum = arr.mean(axis=0)
        std_cum = arr.std(axis=0)

        x = np.arange(len(mean_cum))
        if T is not None:
            x = x[:T]
            mean_cum = mean_cum[:T]
            std_cum = std_cum[:T]

        color = COLOR_MAP.get(alg, "black")

        ax.plot(x, mean_cum, label=alg, linewidth=2, color= color)
        ax.fill_between(x, mean_cum - std_cum, mean_cum + std_cum, alpha=0.2, color = color)

    ax.set_xlabel("Time step t", fontsize=12)
    ax.set_ylabel("Cumulative \nCost-Weighted Regret", fontsize=12)
    #plt.title(title if title is not None else "Cumulative Regret vs. Oracle", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-trial network path selection experiments"
    )

    parser.add_argument("--num-interfaces", type=int, default=NUM_INTERFACES,
                        help="Number of network interfaces")
    parser.add_argument("--num-servers", type=int, default=NUM_SERVERS,
                        help="Number of servers")


    return parser.parse_args()





if __name__ == "__main__":
    os.makedirs("plot", exist_ok=True)
    NUM_INTERFACES = 2
    NUM_SERVERS = 3
    args = parse_args()

    # Override global configuration
    NUM_INTERFACES = args.num_interfaces
    NUM_SERVERS = args.num_servers
    K = NUM_INTERFACES * NUM_SERVERS
    print("=== Experiment Configuration ===")
    print(f"NUM_INTERFACES      = {NUM_INTERFACES}")
    print(f"NUM_SERVERS         = {NUM_SERVERS}")
    print(f"TOTAL_PATHS         = {K}")

    raw_path = f"data_output/final_raw_steps_all_{NUM_INTERFACES}_{NUM_SERVERS}.csv"

    algorithms_to_plot = ["CAR-UCB", "PLR", "FT-Flood", "Oracle"]

    df = compute_min_latency_column(raw_path)

    df = df[df["algorithm"].isin(algorithms_to_plot)]

    #cost-weighted regret
    f_star = 1.0  #benchmark reliability

    df["inst_regret"] = df["cost"] * (f_star - df["outcome"])

    regret_stats = compute_mean_regret_stats(df)
    print(regret_stats)
    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    cumreg_by_alg = compute_cumulative_regret(df, algorithms=["CAR-UCB", "PLR", "Oracle"])
    plot_cumulative_regret(cumreg_by_alg, title=f"Cumulative Regret (K = {NUM_SERVERS * NUM_INTERFACES})", ax=ax)
    plt.savefig(f"plot/new_cumulative_regret_k{K}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Done... see figure in plot/new_cumulative_regret_k{K}.pdf")
