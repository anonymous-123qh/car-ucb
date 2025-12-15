import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("TkAgg")  # comment out if not needed
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

        return {int(k): float(v) for k, v in d.items()}
    except Exception:
        return {}



def choose_p_star_from_ftflood(df_env):
    """
    Choose path with lowest mean latency under FT-Flood as p*.
    Replace this with a fixed id (e.g., p_star = 0).
    """
    df_env = df_env.copy()
    df_env["lat_dict"] = df_env["latencies_json"].apply(parse_latencies_json)

    # collect latencies per path
    per_path = {}
    for d in df_env["lat_dict"]:
        for p, L in d.items():
            per_path.setdefault(p, []).append(L)

    mean_lat = {p: np.mean(vals) for p, vals in per_path.items()}
    p_star = min(mean_lat, key=mean_lat.get)  # smallest mean latency
    return p_star


def build_multi_agent_usage(raw_path, algo_names, trial_id=0, p_star=None, T_MAX=2000):
    """
    Build data for herd avoidance style plots:
      - uses FT-Flood rows as environment ground truth
      - picks a single path p_star to track
      - for each algorithm in algo_names, records whether p_star is in its action at each t
    """
    df = pd.read_csv(raw_path)

    #Ground truth from FT-Flood
    df_env = df[(df["algorithm"] == "FT-Flood") & (df["trial_id"] == trial_id)].copy()
    df_env = df_env.sort_values("t")
    df_env["lat_dict"] = df_env["latencies_json"].apply(parse_latencies_json)

    env_t = df_env["t"].to_numpy()

    if p_star is None:
        p_star = choose_p_star_from_ftflood(df_env)

    lat_p_star = []
    for d in df_env["lat_dict"]:

        lat = d.get(p_star, np.nan)
        lat_p_star.append(lat)
    lat_p_star = np.array(lat_p_star)

    T = min(T_MAX, len(env_t))
    t = env_t[:T]
    lat_p_star = lat_p_star[:T]

    algo_names = list(algo_names)
    usage_matrix = np.zeros((len(algo_names), T), dtype=int)
    df["action_list"] = df["action"].apply(parse_action)
    df_trial = df[df["trial_id"] == trial_id].copy()

    #for each algorithm
    for i, alg in enumerate(algo_names):
        df_a = df_trial[df_trial["algorithm"] == alg].copy()
        df_a = df_a.sort_values("t")

        action_map = dict(zip(df_a["t"].to_numpy(), df_a["action_list"]))

        row = []
        for tt in t:
            acts = action_map.get(tt, [])
            row.append(1 if p_star in acts else 0)
        usage_matrix[i, :] = np.array(row, dtype=int)

    return {
        "t": t,
        "p_star": p_star,
        "lat_p_star": lat_p_star,
        "usage_matrix": usage_matrix,
        "algo_names": algo_names,
    }

def plot_path_usage_multi_agent(data, title=None):

    t = data["t"]
    lat = data["lat_p_star"]
    usage = data["usage_matrix"]
    algo_names = data["algo_names"]
    sort_idx = sorted(
        range(len(algo_names)),
        key=lambda k: int(algo_names[k].split("CAR-UCB")[-1]),
        reverse= True
    )
    # Apply sorted order
    algo_names = [algo_names[k] for k in sort_idx]
    usage = usage[sort_idx, :]
    p_star = data["p_star"]

    num_agents = usage.shape[0]

    ##change name the algorithm to agent UCB3, UCB2, UCB1
    algo_names = ["Agent3", "Agent2", "Agent1"]

    fig, axes = plt.subplots(
        num_agents + 1, 1,
        figsize=(12, 2.0 * (num_agents + 1)),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0] + [1.0] * num_agents},
    )

    ax_lat = axes[0]

    # (1) latency of p*
    ax_lat.plot(t, lat, linewidth=2.0, color="C1")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title(f"Path  {p_star}")

    #add deadline line at 100 ms
    DEADLINE = 100
    ax_lat.axhline(DEADLINE, color="red", linestyle="--", linewidth=2)
    ax_lat.text(
        t[0], DEADLINE + 3,
        rf"$\tau$ = {DEADLINE} ms",
        color="red",
        fontsize=12,
        va="bottom"
    )
    ax_lat.grid(True, alpha=0.3)

    #usage per agent
    for i, alg in enumerate(algo_names):
        ax = axes[i + 1]
        ax.step(t, usage[i, :], where="post", linewidth=1.8)
        ax.set_ylabel(alg)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time step $t$")

    fig.tight_layout()
    fig.savefig("plot/path_usage_multi_agent.pdf", bbox_inches="tight")
    print("Done... Plot agent's usage of a specific path is saved at plot/path_usage_multi_agent.pdf")
    #plt.show()

def plot_path_usage_heatmap(data, title=None):

    t = data["t"]
    usage = data["usage_matrix"]
    algo_names = data["algo_names"]
    p_star = data["p_star"]

    num_agents, T = usage.shape

    fig, ax = plt.subplots(figsize=(12, 0.5 * num_agents + 2))

    im = ax.imshow(
        usage,
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        extent=[t.min(), t.max(), -0.5, num_agents - 0.5],
        origin="lower",
    )

    ax.set_yticks(range(num_agents))
    ax.set_yticklabels(algo_names)
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("Agent / Algorithm")

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Usage of Path $p^* = {p_star}$ Across Agents")

    fig.tight_layout()
    fig.savefig("plot/path_usage_heatmap.pdf", bbox_inches="tight")
    plt.show()

def build_multi_agent_primary_paths(raw_path, algo_names, trial_id=0, T_MAX=2000):

    df = pd.read_csv(raw_path)
    df_trial = df[df["trial_id"] == trial_id].copy()

    #parse action and latencies_json once
    df_trial["action_list"] = df_trial["action"].apply(parse_action)
    df_trial["lat_dict"] = df_trial["latencies_json"].apply(parse_latencies_json)


    all_t = np.sort(df_trial["t"].unique())
    if T_MAX is not None:
        all_t = all_t[:T_MAX]
    T = len(all_t)

    algo_names = list(algo_names)
    num_agents = len(algo_names)


    primary_paths = np.full((num_agents, T), np.nan)


    for i, alg in enumerate(algo_names):
        df_a = df_trial[df_trial["algorithm"] == alg].copy()
        df_a = df_a.sort_values("t")


        action_map = dict(zip(df_a["t"].to_numpy(), df_a["action_list"]))
        lat_map = dict(zip(df_a["t"].to_numpy(), df_a["lat_dict"]))

        row_vals = []
        for tt in all_t:
            acts = action_map.get(tt, [])
            latd = lat_map.get(tt, {})

            if not acts:
                row_vals.append(np.nan)
                continue


            candidates = [(p, latd.get(p, np.nan)) for p in acts]

            candidates = [(p, L) for p, L in candidates if not np.isnan(L)]

            if not candidates:
                row_vals.append(np.nan)
            else:

                p_best = min(candidates, key=lambda x: x[1])[0]
                row_vals.append(p_best)

        primary_paths[i, :] = np.array(row_vals, dtype=float)

    return {
        "t": all_t,
        "primary_paths": primary_paths,
        "algo_names": algo_names,
    }
def plot_primary_paths_panels(data, title=None):

    t = data["t"]
    primary_paths = data["primary_paths"]
    algo_names = data["algo_names"]
    num_agents, T = primary_paths.shape


    unique_paths = sorted(set(primary_paths[~np.isnan(primary_paths)].astype(int)))
    num_paths = len(unique_paths)


    cmap = plt.get_cmap("tab10", num_paths)
    path_to_idx = {p: i for i, p in enumerate(unique_paths)}

    # Convert primary_paths -> color index matrix
    color_idx_matrix = np.full_like(primary_paths, np.nan)
    for i in range(num_agents):
        for t_i in range(T):
            p = primary_paths[i, t_i]
            if not np.isnan(p):
                color_idx_matrix[i, t_i] = path_to_idx[int(p)]


    fig, axes = plt.subplots(
        num_agents, 1,
        figsize=(6.0, 1.6 * num_agents + 1.0),
        sharex=True
    )
    if num_agents == 1:
        axes = [axes]

    #plot each agent as a 1-row image strip
    for i, alg in enumerate(algo_names):
        ax = axes[i]
        row = color_idx_matrix[i, :].reshape(1, -1)

        im = ax.imshow(
            row,
            aspect="auto",
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
            vmin=0,
            vmax=num_paths - 1,
            extent=[t.min(), t.max(), -0.5, 0.5],
        )

        ax.set_yticks([])
        ax.set_ylabel(alg, rotation=0, labelpad=35)
        ax.grid(False)

        if i == 0 and title:
            ax.set_title(title)

    axes[-1].set_xlabel("Time step $t$")


    legend_handles = []
    for p in unique_paths:
        idx = path_to_idx[p]
        color = cmap(idx)
        patch = Patch(facecolor=color, edgecolor="black", label=f"Path {p}")
        legend_handles.append(patch)


    fig.subplots_adjust(right=0.80)


    axes[0].legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=False,
        title="Paths"
    )
    fig.savefig("plot/new_primary_paths_per_agent_with_legend.pdf", bbox_inches="tight")
    print("Done... Plot saved at plot/new_primary_paths_per_agent_with_legend.pdf")
    #plt.show()

def plot_latency_and_primary_paths(latency_data,
                                   primary_data,
                                   scenario_label="K = 50",
                                   T_MAX=2000,
                                   deadline=100):


    #unpack latency data
    t_lat = latency_data["timesteps"]
    lat = latency_data["latencies"]
    top_paths = latency_data["top_paths"]   # list of path IDs

    #unpack primary-path data
    t_prim = primary_data["t"]
    primary_paths = primary_data["primary_paths"]   # shape (num_agents, T_prim)
    algo_names = primary_data["algo_names"]
    num_agents, T_prim = primary_paths.shape

    #align time horizon
    T_lat = len(t_lat)
    T_common = min(T_lat, T_prim)
    if T_MAX is not None:
        T_common = min(T_common, T_MAX)

    t = t_lat[:T_common]
    lat = lat[:T_common, :]
    primary_paths = primary_paths[:, :T_common]

    if len(t) == 0:
        raise ValueError("No timesteps after alignment / T_MAX; check inputs.")


    paths_from_latency = set(top_paths)
    paths_from_primary = set(primary_paths[~np.isnan(primary_paths)].astype(int))
    all_paths = sorted(paths_from_latency.union(paths_from_primary))


    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        # Fallback
        default_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    path_to_color = {}
    color_idx = 0


    for p in top_paths:
        if p not in path_to_color:
            path_to_color[p] = default_colors[color_idx % len(default_colors)]
            color_idx += 1

    #
    for p in all_paths:
        if p not in path_to_color:
            path_to_color[p] = default_colors[color_idx % len(default_colors)]
            color_idx += 1


    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.5]},
        constrained_layout=True
    )


    for i, p in enumerate(top_paths):
        ax1.plot(
            t,
            lat[:, i],
            label=f"Path {p}",
            linewidth=1.3,
            color=path_to_color[p]
        )

    ax1.axhline(deadline, color='red', linestyle='--', linewidth=1.5)
    #ax1.text(
    #    t[0], deadline + 3,
    #    rf"$\tau$ = {deadline} ms",
    #    color='red',
     #   fontsize=11,
     #   va="bottom"
    #)

    ax1.set_ylabel("Latency (ms)")
    #ax1.set_title(f"Latency and Primary Path Usage ({scenario_label})", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(t[0], t[-1])


    ax1.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=min(len(top_paths), 6),
        frameon=False,
    )

    #BOTTOM PANEL: primary path usage per agent
    for i, alg in enumerate(algo_names):
        ax = ax2


        y_base = i
        for tt in range(T_common - 1):
            p = primary_paths[i, tt]
            if np.isnan(p):
                continue
            p = int(p)
            color = path_to_color.get(p, "black")

            ax.fill_between(
                [t[tt], t[tt + 1]],
                y_base - 0.4,
                y_base + 0.4,
                color=color
            )

    ax2.set_yticks(range(num_agents))
    ax2.set_yticklabels(algo_names) #["Agent1", "Agent2", "Agent3"])
    ax2.set_xlabel("Time step $t$")
    ax2.set_ylabel("Agent")
    ax2.grid(False)

    #fig.tight_layout()
    fig.savefig("plot/new_avoidance_herd_analyse.pdf", bbox_inches="tight", pad_inches=0.02)
    print("Plot for all agents behaviors saved at plot/new_avoidance_herd_analyse.pdf")
    #plt.show()

def build_qualitative_view(raw_path, algo_name="CAR-UCB", trial_id=0, top_n_paths=3):

    df = pd.read_csv(raw_path)

    #filter FT-Flood for environment ground truth
    df_env = df[(df["algorithm"] == "FT-Flood") & (df["trial_id"] == trial_id)].copy()
    df_env = df_env.sort_values("t")

    # Parse latencies_json for FT-Flood
    df_env["lat_dict"] = df_env["latencies_json"].apply(parse_latencies_json)

    # Determine set of all paths from FT-Flood
    all_paths = sorted({p for d in df_env["lat_dict"] for p in d.keys()})
    K = len(all_paths)


    env_t = df_env["t"].to_numpy()


    path_index = {p: i for i, p in enumerate(all_paths)}
    ground_truth_lat = np.full((len(env_t), K), np.nan)

    for i, d in enumerate(df_env["lat_dict"]):
        for p, lat in d.items():
            j = path_index[p]
            ground_truth_lat[i, j] = lat

    #filter CAR-UCB decisions for the same trial
    df_car = df[(df["algorithm"] == algo_name) & (df["trial_id"] == trial_id)].copy()
    df_car = df_car.sort_values("t")

    df_car["action_list"] = df_car["action"].apply(parse_action)
    car_t = df_car["t"].to_numpy()
    car_action_size = df_car["action_size"].to_numpy()

    #identify top-N most frequently used paths by CAR-UCB
    from collections import Counter
    c = Counter()
    for acts in df_car["action_list"]:
        c.update(acts)


    top_paths = [p for p, _ in c.most_common(top_n_paths)]
    top_paths = [p for p in top_paths if p in path_index]  # ensure present in env


    common_ts = np.intersect1d(env_t, car_t)


    env_idx = {t: i for i, t in enumerate(env_t)}
    car_idx = {t: i for i, t in enumerate(car_t)}

    #build aligned arrays
    aligned_lat = []
    aligned_sel_matrix = []
    aligned_action_size = []

    for t in common_ts:
        i_env = env_idx[t]
        i_car = car_idx[t]

        #ground truth latencies at t
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

def plot_latency_only(data, scenario_label="K = 50", T_MAX=2000, deadline=100):


    t = data["timesteps"]
    lat = data["latencies"]
    top_paths = data["top_paths"]

    if T_MAX is not None and T_MAX < len(t):
        t = t[:T_MAX]
        lat = lat[:T_MAX, :]


    fig, ax1 = plt.subplots(
        1, 1,
        figsize=(6.0, 3.0),
    )


    for i, p in enumerate(top_paths):
        ax1.plot(t, lat[:, i], label=f"Path {p}", linewidth=1.8)


    ax1.axhline(deadline, color='red', linestyle='--', linewidth=1.5)

    ax1.text(
        t[0], deadline + 3,
        rf"$\tau$ = {deadline} ms",
        color='red',
        fontsize=12,
        verticalalignment="bottom"
    )

    ax1.set_ylabel("Latency (ms)")
    ax1.set_xlabel("Time step $t$")
    ax1.grid(True, alpha=0.3)


    ax1.set_xlim(t[0], t[-1])


    ax1.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=min(len(top_paths), 6),
        frameon=False,
        title=None,
    )

    plt.tight_layout()
    fig.savefig("plot/new_latency_shared.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    NUM_INTERFACES =2
    NUM_SERVERS = 3
    os.makedirs("plot", exist_ok=True)


    #override global configuration
    K = NUM_INTERFACES * NUM_SERVERS
    print("=== Experiment Configuration ===")
    print(f"NUM_INTERFACES      = {NUM_INTERFACES}")
    print(f"NUM_SERVERS         = {NUM_SERVERS}")
    print(f"TOTAL_PATHS         = {K}")
    raw_path = f"data_output/multiple_robot_raw_steps_all_{NUM_INTERFACES}_{NUM_SERVERS}.csv"
    algo_names = ["CAR-UCB1", "CAR-UCB2", "CAR-UCB3"]  # adjust to actual names

    #data for path 1 (set at p_star)
    data_multi = build_multi_agent_usage(
        raw_path,
        algo_names=algo_names,
        trial_id=1,
        p_star=1,     # or p_star=0 if want a fixed path
        T_MAX=1000
    )
    #then draw plot for path 1
    plot_path_usage_multi_agent(
        data_multi,
        title=f"Path usage for $p^*$ (K = {NUM_INTERFACES * NUM_SERVERS})"
    )

    # heatmap view
    #plot_path_usage_heatmap(
    #    data_multi,
    #    title=f"Usage of $p^*$ across agents (K = {NUM_INTERFACES * NUM_SERVERS})"
    #)

    #data for all path
    data_primary = build_multi_agent_primary_paths(
        raw_path,
        algo_names=algo_names,
        trial_id=1,
        T_MAX=1000,
    )

    data = build_qualitative_view(raw_path, algo_name="CAR-UCB1", trial_id=1, top_n_paths=6)
    #then draw for all agents
    plot_latency_and_primary_paths(
        data,
        data_primary,
        scenario_label=f"K = {NUM_INTERFACES * NUM_SERVERS}",
        T_MAX=1000,
        deadline=100,
    )
