import os
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import argparse

NUM_INTERFACES = 2
NUM_SERVERS = 3
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
def load_latency_by_algorithm(raw_path, algorithms=None,
                              filter_alg=None, threshold=None, replacement=None):

    df = pd.read_csv(raw_path)

    #parse latencies_json and compute min latency per row
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

    #apply filtering rule for one algorithm
    if filter_alg is not None and threshold is not None and replacement is not None:
        mask = df["algorithm"] == filter_alg
        df.loc[mask, "min_latency"] = df.loc[mask, "min_latency"].apply(
            lambda x: x if x <= threshold else replacement
        )

    if algorithms is None:
        algorithms = sorted(df["algorithm"].unique())

    latency_by_alg = {}

    for alg in algorithms:
        runs = []
        for trial_id in sorted(df["trial_id"].unique()):
            mask = (df["algorithm"] == alg) & (df["trial_id"] == trial_id)
            lat = df.loc[mask, "min_latency"].to_numpy(dtype=float)
            if len(lat) == 0:
                continue
            runs.append(lat)
        latency_by_alg[alg] = runs

    return latency_by_alg



#compute mean CDF + variability across trials
def compute_mean_cdf(latencies_list, num_points=400):
    all_lat = np.concatenate(latencies_list)
    x = np.linspace(all_lat.min(), all_lat.max(), num_points)

    cdfs = []
    for lat in latencies_list:
        lat_sorted = np.sort(lat)
        cdf = np.searchsorted(lat_sorted, x, side="right") / len(lat_sorted)
        cdfs.append(cdf)

    cdfs = np.array(cdfs)
    mean_cdf = cdfs.mean(axis=0)
    std_cdf = cdfs.std(axis=0)

    return x, mean_cdf, std_cdf


#plot CDF with shaded variability
def plot_latency_cdf_with_shading(latency_by_alg, deadline=None, ax = None):

    if ax is None:
        ax = plt.gca()
    for alg, runs in latency_by_alg.items():
        if len(runs) == 0:
            continue

        x, mean_cdf, std_cdf = compute_mean_cdf(runs)

        ax.plot(x, mean_cdf, label=alg, linewidth=2)

        lower = np.clip(mean_cdf - std_cdf, 0.0, 1.0)
        upper = np.clip(mean_cdf + std_cdf, 0.0, 1.0)
        ax.fill_between(x, lower, upper, alpha=0.2)

    #vertical deadline line
    if deadline is not None:
        ax.axvline(deadline, color="red", linestyle="--", linewidth=1)

        ax.text(
            deadline,
            0.3,
            rf"$\tau$ = {deadline} ms",
            rotation=90,
            verticalalignment='center',
            horizontalalignment='left',
            color='red',
            fontsize=12
        )

    ax.set_xlabel("Latency (ms)", fontsize=14)
    ax.set_ylabel("CDF", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend( framealpha=0.5)
    #plt.tight_layout()
    #plt.show()



#calculate tail metrics: P95, CVaR95, P99
def compute_latency_metrics(latency_by_alg):
    metrics = {}

    for alg, runs in latency_by_alg.items():
        if len(runs) == 0:
            continue

        all_lat = np.concatenate(runs)

        p95 = np.percentile(all_lat, 95)
        p99 = np.percentile(all_lat, 99)
        cvar95 = all_lat[all_lat >= p95].mean()

        trial_p95 = []
        trial_p99 = []
        trial_cvar95 = []

        for lat in runs:
            p95_t = np.percentile(lat, 95)
            p99_t = np.percentile(lat, 99)
            cvar95_t = lat[lat >= p95_t].mean()

            trial_p95.append(p95_t)
            trial_p99.append(p99_t)
            trial_cvar95.append(cvar95_t)

        metrics[alg] = {
            "combined_p95": p95,
            "combined_cvar95": cvar95,
            "combined_p99": p99,
            "mean_p95": np.mean(trial_p95),
            "std_p95": np.std(trial_p95),
            "mean_cvar95": np.mean(trial_cvar95),
            "std_cvar95": np.std(trial_cvar95),
            "mean_p99": np.mean(trial_p99),
            "std_p99": np.std(trial_p99),
        }

    return metrics



def compute_average_latency_stats(latency_by_alg):
    stats = {}
    for alg, runs in latency_by_alg.items():
        trial_means = [lat.mean() for lat in runs]
        stats[alg] = {
            "mean": np.mean(trial_means),
            "std": np.std(trial_means)
        }
    return stats


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
    K = NUM_INTERFACES * NUM_SERVERS

    args = parse_args()
    print("=== Experiment Configuration ===")
    print(f"NUM_INTERFACES      = {NUM_INTERFACES}")
    print(f"NUM_SERVERS         = {NUM_SERVERS}")
    print(f"TOTAL_PATHS         = {K}")

    raw_path = f"data_output/final_raw_steps_all_{NUM_INTERFACES}_{NUM_SERVERS}.csv"

    algorithms_to_plot = ["CAR-UCB", "PLR", "FT-Flood", "Oracle"]


    latency_by_alg = load_latency_by_algorithm(
        raw_path,
        algorithms=algorithms_to_plot

    )

    results = compute_latency_metrics(latency_by_alg)

    for alg, m in results.items():
        print(f"\n=== {alg} ===")
        print(f"P95 (combined):      {m['combined_p95']:.2f} ms")
        print(f"CVaR95 (combined):   {m['combined_cvar95']:.2f} ms")
        print(f"P99 (combined):      {m['combined_p99']:.2f} ms")
        print(f"P95  mean±std:       {m['mean_p95']:.2f} ± {m['std_p95']:.2f}")
        print(f"CVaR95 mean±std:     {m['mean_cvar95']:.2f} ± {m['std_cvar95']:.2f}")
        print(f"P99  mean±std:       {m['mean_p99']:.2f} ± {m['std_p99']:.2f}")

    fig, ax = plt.subplots(figsize=(3.8, 2.5))

    plot_latency_cdf_with_shading(latency_by_alg, deadline=100, ax=ax)
    plt.savefig(f"plot/latency_cdf_k{K}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Done! Plot saved at plot/latency_cdf_k{K}.pdf")
    #avg_stats = compute_average_latency_stats(latency_by_alg)
    #for alg, s in avg_stats.items():
    #    print(f"{alg}: Mean = {s['mean']:.2f} ms, Std = {s['std']:.2f} ms")


