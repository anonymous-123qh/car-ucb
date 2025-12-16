import time
import json
import argparse
import numpy as np
import random
import itertools

from math import sqrt, log
import csv

from collections import Counter
plr_action_counts = Counter()
plr_pp_action_counts = Counter()
car_ucb_action_counts = Counter()
topK_ucb_action_counts = Counter()


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

# System scale
NUM_INTERFACES =2      #(e.g., WiFi + 5G)5
NUM_SERVERS =  3 #number of cloud/edge servers15
TOTAL_PATHS = NUM_INTERFACES * NUM_SERVERS  #path = interface–server pair
CONGESTION_PENALTY = 30
ENV_MODE = "dynamic"
#Deadline and simulation length
DEADLINE_TAU = 100.0      #ms (hard deadline)
SIMULATION_STEPS = 5000   # number of decision

# Experiment Settings
NUM_TRIALS = 20         #Run 20 independent trials for statistics

# CAR-UCB parameters (from paper style)
CAR_UCB_HISTORY_WINDOW = 100   #W
CAR_UCB_LOCAL_K = 2            #k (top-k centers)
CAR_UCB_GAMMA = 2           # gamma
MAX_SUBSET_SIZE = 2

#PLR parameters
PLR_RELIABILITY_TARGET = 0.90

#FT parameters – replicate to N servers each request
FT_NUM_REPLICAS = 2             #FogROS2-FT experiments



#Environment: Trace-based Latency Generator
class TraceEnvironment:
    """
    Simulates per-path latency traces with non-stationarity.
    """
    def __init__(self, num_paths, mode="dynamic"):
        self.num_paths = num_paths
        self.mode = mode
        self.costs = {}
        for p in range(num_paths):
            r = random.random()
            if r < 0.3:
                c = 1.0
            elif r < 0.8:
                c = 2.0
            else:
                c = 3.0
            self.costs[p] = c

        #generate traces
        self.phase_good_sets = None
        self.phase_len = None

        if mode == "static":
            self.traces = self.static_trace()
        elif mode == "dynamic":
            self.traces = self.dynamic_trace()
        else:
            raise ValueError("Unknown environment mode")

        self.pointers = {}

    def static_trace(self):
        traces = {}
        T = SIMULATION_STEPS + 100
        for p in range(self.num_paths):
            base_mean = 50 if p in [0, 4] else 120
            data = np.random.normal(base_mean, 10, T)
            if p % 2 == 0:
                for burst_start in range(200, SIMULATION_STEPS, 300):
                    data[burst_start:burst_start + 50] += 30
            traces[p] = data
        return traces

    def dynamic_trace(self):
        traces = {}
        T = SIMULATION_STEPS + 100
        PHASE_LEN = 500
        NUM_GOOD = max(1, self.num_paths // 5)

        num_phases = (T + PHASE_LEN - 1) // PHASE_LEN
        all_paths = list(range(self.num_paths))

        phase_good_sets = [
            set(random.sample(all_paths, NUM_GOOD))
            for _ in range(num_phases)
        ]
        self.phase_good_sets = phase_good_sets
        self.phase_len = PHASE_LEN

        for p in range(self.num_paths):
            data = np.zeros(T)
            for t in range(T):
                phase = t // PHASE_LEN
                good_paths = phase_good_sets[phase]
                if p in good_paths:
                    base = 50.0
                else:
                    base = 120.0
                data[t] = np.random.normal(base, 8.0)

            #independent random bursts
            for _ in range(3):
                s = random.randint(0, T - 50)
                data[s:s + 50] += 30
            traces[p] = data
        return traces

    def get_reader(self, agent_name):
        if agent_name not in self.pointers:
            self.pointers[agent_name] = {p: 0 for p in range(self.num_paths)}

        def read_observation(action):
            if not action: return 0, {}
            latencies = {}
            for p in action:
                trace = self.traces[p]
                ptr = self.pointers[agent_name][p]
                latencies[p] = trace[ptr]

            #Congestion Penalty
            iface_counts = {}
            for p in action:
                iface = p // NUM_SERVERS
                iface_counts[iface] = iface_counts.get(iface, 0) + 1


            for p in action:
                iface = p // NUM_SERVERS
                k = iface_counts[iface]
                if k > 1:
                    extra = (k - 1) * CONGESTION_PENALTY
                    latencies[p] += extra

            min_latency = min(latencies.values())
            outcome = 1 if min_latency <= DEADLINE_TAU else 0
            return outcome, latencies

        return read_observation

    def step_time(self):
        for agent in self.pointers:
            for p in range(self.num_paths):
                self.pointers[agent][p] += 1

    def get_action_cost(self, action):
        return sum(self.costs[p] for p in action)

    def get_latencies(self, t):
        current_latencies = {}
        for p in range(self.num_paths):
            path_trace = self.traces[p]
            idx = t if t < len(path_trace) else t % len(path_trace)
            current_latencies[p] = path_trace[idx]
        return current_latencies


################ALgorithm and baselines##############################

#CAR-UCB
class CAR_UCB_Agent:

    def __init__(self, num_paths, costs, window_size, local_k, gamma=0.1, max_subset_size= 2):
        self.K = num_paths
        self.costs = costs
        self.W = window_size
        self.k = local_k
        self.gamma = gamma
        self.all_paths = list(range(num_paths))

        self.max_subset_size = max_subset_size
        #Candidate budget B ≈ 1.5 * K
        self.B = int(1.5 * self.K)

        self.t = 0
        self.history = []
        self.mu = {}
        self.sigma = {}
        self.best_action_so_far = None

        #log
        self.debug_logs = []

    #Surrogate model:,approximation of GPC on actions
    def _update_model(self):
        #sliding-window dataset
        recent = self.history[-self.W:]
        counts = {}
        successes = {}

        for act, out in recent:
            if act not in counts:
                counts[act] = 0
                successes[act] = 0
            counts[act] += 1
            successes[act] += out

        for act, c in counts.items():
            s = successes[act]
            #Bernoulli mle with small smoothing
            p_hat = (s + 1) / (c + 2)

            #uncertainty shrinks
            std = 1.0 / sqrt(c + 1.0)

            self.mu[act] = p_hat
            self.sigma[act] = std

        #update best_action_so_far
        if self.mu:
            self.best_action_so_far = max(self.mu.keys(), key=lambda a: self.mu[a])

    def _predict(self, action):
        if action not in self.mu:
            return 0.5, 1.0
        return self.mu[action], self.sigma[action]


    def _get_top_k_actions(self):
        if not self.mu:
            return []
        #sort actions in descending order
        sorted_actions = sorted(self.mu.keys(), key=lambda a: self.mu[a], reverse=True)
        unique_actions = []
        for a in sorted_actions:
            if a not in unique_actions:
                unique_actions.append(a)
            if len(unique_actions) >= self.k:
                break
        return unique_actions

    #Implemenation of Algorithm 1: Generate_Candidates
    def generate_candidates(self):
        Acand = set()

        #step 1: Historical exploitation
        if self.best_action_so_far is not None:
            Acand.add(self.best_action_so_far)
        if self.history:
            last_action = self.history[-1][0]
            Acand.add(last_action)

        #step 2: Local search (expansion and contraction around top-k)
        Htop = self._get_top_k_actions()
        for A in Htop:
            A_set = set(A)
            for p in self.all_paths:
                if p not in A_set:
                    #Expand
                    newA = tuple(sorted(list(A_set | {p})))
                    Acand.add(newA)
                elif len(A_set) > 1:
                    #Contract
                    newA = tuple(sorted(list(A_set - {p})))
                    if newA:
                        Acand.add(newA)

        #step 3: Cost-weighted random exploration
        Nneed = self.B - len(Acand)
        if Nneed > 0:
            inc_prob = min(0.2, 1.0)  # tune if needed

            for _ in range(Nneed):
                Arand = []
                for p in self.all_paths:
                    if random.random() < inc_prob:
                        Arand.append(p)
                if Arand:
                    Acand.add(tuple(sorted(Arand)))

        #limit to budget B
        Acand = list(Acand)
        Acand = [A for A in Acand if 1 <= len(A) <= self.max_subset_size]
        Acand = list(Acand)[:self.B]
        return Acand


    def select_action(self):
        self.t += 1

        #update surrogate from history
        self._update_model()

        #set exploration and tolerance schedules
        beta_t = sqrt(2 * log(1 + self.t))
        epsilon_t = self.gamma / sqrt(log(1 + self.t) + 1e-9)

        #Step 1: Generate candidate set, call algorithm 1
        Acand = self.generate_candidates()
        Acand = [A for A in Acand if 1 <= len(A) <= self.max_subset_size]
        if not Acand:
            #Fallback
            return (0,)

        #Step 2: optimistic estimates
        u_scores = {}
        for A in Acand:
            m, s = self._predict(A)
            u_scores[A] = m + beta_t * s

        #Step 3: best optimistic value
        g_t = max(u_scores.values())

        #Step 4: near-optimal set S1
        S1 = [A for A in Acand if u_scores[A] + epsilon_t >= g_t]
        if not S1:
            S1 = Acand

        #Step 5: cheapest near-optimal actions S2
        sizes_s1 = {A: len(A) for A in S1}

        #Check if have any safe options (size >= 2) in the candidate pool
        has_redundant_option = any(s >= 2 for s in sizes_s1.values())

        if has_redundant_option:
            target_size = min(s for s in sizes_s1.values() if s >= 2)
        else:
            #Fallback: If only size 1 exists, we have to take it.
            target_size = min(sizes_s1.values())

        S2 = [A for A in S1 if len(A) == target_size]


        #Step 6: pick action with highest u within S2
        A_t = max(S2, key=lambda A: u_scores[A])
        return A_t

    def update(self, action, outcome, latencies):
        #Append to history
        self.history.append((action, outcome))


#FT-Flood baseline
class FT_Flood_Agent:

    def __init__(self, num_paths, costs):
        self.paths = list(range(num_paths))
        self.costs = costs

    def select_action(self):
        return tuple(self.paths)

    def update(self, action, outcome, latencies):
        pass  # no learningmalways flood



#PLR baseline
class PLR_Agent:
    """
    - Tracks per-path P(deadline hit) empirically.
    - Assumes independent failures.
    - Chooses smallest subset with success prob >= target.
    """

    def __init__(self, num_paths, costs,
                 deadline_tau=DEADLINE_TAU,
                 reliability_target=PLR_RELIABILITY_TARGET):
        self.paths = list(range(num_paths))
        self.costs = costs
        self.deadline = deadline_tau
        self.reliability_target = reliability_target

        #Laplace-smoothed success estimator per path
        self.success = {p: 1 for p in self.paths}
        self.total = {p: 2 for p in self.paths}

    def _path_reliability(self, p):
        return self.success[p] / self.total[p]

    def _set_reliability(self, A):
        prob_all_fail = 1.0
        for p in A:
            prob_all_fail *= (1 - self._path_reliability(p))
        return 1 - prob_all_fail

    def select_action(self):
        best = None
        best_size = float('inf')
        #keep track of the overall best reliability
        best_overall = None
        best_overall_r = -1.0

        for size in range(1,3):
            for A in itertools.combinations(self.paths, size):
                r = self._set_reliability(A)

                #track global best
                if r > best_overall_r:
                    best_overall_r = r
                    best_overall = A

                #track best meets target
                if r >= self.reliability_target:
                    if size < best_size:
                        best_size = size
                        best = A

        #If something meets target then use smallest such set
        if best is not None:
            return best

        #If no subset hits the reliability target, fallback to the set with highest estimated reliabilty
        if best_overall is not None:
            return best_overall

        #safety fallbac
        return (self.paths[0],)

    def update(self, action, outcome, latencies):
        # Update per-path success
        for p in action:
            self.total[p] += 1
            if p in latencies:
                succ_p = 1 if latencies[p] <= self.deadline else 0
            else:
                succ_p = outcome
            self.success[p] += succ_p


#Oracle baseline
class OracleAgent:

    def __init__(self, num_paths, path_costs, deadline_ms):

        self.num_paths = num_paths
        self.costs = path_costs
        self.deadline = deadline_ms
        self.t = 0


        self.history = []

    def select_action(self, ground_truth_latencies):

        self.t += 1


        feasible_paths = [
            p for p, latency in ground_truth_latencies.items()
            if latency <= self.deadline
        ]

        #best path
        if feasible_paths:
            #pick the one with the lowest cost
            best_path = min(feasible_paths, key=lambda p: (self.costs[p], ground_truth_latencies[p]))
            return [best_path]
        else:
            #fallback
            best_path = min(ground_truth_latencies, key=ground_truth_latencies.get)
            return [best_path]

    def update(self, action, outcome):
        #track history for metrics
        self.history.append((action, outcome))

######################Multi-Trial Execution and Aggregation

def run_single_trial(trial_seed, trial_id = None, step_writer = None):

    if trial_id == None:
        trial_id = trial_seed

    #seed for reproducibility
    random.seed(trial_seed)
    np.random.seed(trial_seed)

    #create environment
    env = TraceEnvironment(TOTAL_PATHS, mode=ENV_MODE)
    #print(f"=== Path costs for trial {trial_id} ===")
    #for p in range(TOTAL_PATHS):
    #    print(f"path {p}: cost {env.costs[p]}")

    #initialize agents
    agents = {
        "FT-Flood": FT_Flood_Agent(TOTAL_PATHS, env.costs),

        "CAR-UCB": CAR_UCB_Agent(
            TOTAL_PATHS,
            env.costs,
            window_size=CAR_UCB_HISTORY_WINDOW,
            local_k=CAR_UCB_LOCAL_K,
            gamma=CAR_UCB_GAMMA,
            max_subset_size= MAX_SUBSET_SIZE
        ),


        "PLR": PLR_Agent(TOTAL_PATHS, env.costs),

        "Oracle": OracleAgent(TOTAL_PATHS, env.costs, deadline_ms=DEADLINE_TAU),

    }

    metrics = {
        name: {
            "success": 0,
            "cost": 0.0,
            "switches": 0,
            "last_act": None,
            "total_size": 0,
            "dec_times": [],
            "latencies": []  #per-step min latency
        }
        for name in agents
    }


    for t in range(SIMULATION_STEPS):
        gt_latencies = env.get_latencies(t)

        for name, agent in agents.items():
            t0 = time.perf_counter()
            if name == "Oracle":
                action = agent.select_action(gt_latencies)
            else:
                action = agent.select_action()

            reader = env.get_reader(name)
            outcome, latencies = reader(action)

            if name == "Oracle":
                agent.update(action, outcome)
            else:
                agent.update(action, outcome, latencies)

            dt = time.perf_counter() - t0

            # Update Metrics
            metrics[name]["success"] += outcome
            metrics[name]["cost"] += env.get_action_cost(action)
            metrics[name]["total_size"] += len(action)
            metrics[name]["dec_times"].append(dt)

            if metrics[name]["last_act"] != action and metrics[name]["last_act"] is not None:
                metrics[name]["switches"] += 1
            metrics[name]["last_act"] = action

            if latencies:
                min_lat = min(latencies.values())
                metrics[name]["latencies"].append(min_lat)

            #RAW step logging
            if step_writer is not None:

                action_str = ";".join(str(p) for p in action)

                latencies_clean = {
                    int(k): float(v) for k, v in latencies.items()
                }
                latencies_json = json.dumps(latencies_clean)  # {path: latency}

                step_writer.writerow([
                    trial_id,
                    t,
                    name,
                    action_str,
                    outcome,
                    env.get_action_cost(action),
                    len(action),
                    latencies_json
                ])
        env.step_time()

    #Summarize of this trial
    summary = {}
    for name, data in metrics.items():
        #decision times
        avg_dec_time = (
            sum(data["dec_times"]) / len(data["dec_times"])
            if data["dec_times"] else 0.0
        )

        #latency statistics
        if data["latencies"]:
            lat_arr = np.array(data["latencies"], dtype=float)
            avg_latency = float(np.mean(lat_arr))
            p95_latency = float(np.percentile(lat_arr, 95))

            #CVaR95 = mean over tail above VaR95
            tail = lat_arr[lat_arr >= p95_latency]
            if tail.size > 0:
                cvar95_latency = float(np.mean(tail))
            else:
                cvar95_latency = p95_latency
        else:
            avg_latency = 0.0
            p95_latency = 0.0
            cvar95_latency = 0.0

        summary[name] = {
            "success_rate": data["success"] / SIMULATION_STEPS,
            "avg_cost": data["cost"] / SIMULATION_STEPS,
            "avg_size": data["total_size"] / SIMULATION_STEPS,
            "total_switches": data["switches"],
            "avg_latency": avg_latency,
            "p95_latency": p95_latency,
            "cvar95_latency": cvar95_latency,
            "avg_dec_time": avg_dec_time,
        }
    return summary

import os
def run_experiment_suite():
    print(f"Starting Experiment Suite: {NUM_TRIALS} Trials, {SIMULATION_STEPS} Steps each, Environment with #Network Interface={NUM_INTERFACES}, #Server={NUM_SERVERS}...")
    os.makedirs("data_output", exist_ok=True)

    #Store all results in memory (for printing mean/std)
    algo_names = ["FT-Flood", "CAR-UCB", "PLR", "Oracle"]

    #Store all results
    all_results = {
        name: {
            "success_rate": [],
            "avg_cost": [],
            "avg_size": [],
            "total_switches": [],
            "avg_latency": [],
            "p95_latency": [],
            "cvar95_latency": [],
            "avg_dec_time": []
        }
        for name in algo_names
    }

    #CSV file for per-trial summaries
    raw_path = f"data_output/final_raw_steps_all_{NUM_INTERFACES}_{NUM_SERVERS}.csv"
    summary_path = f"data_output/final_trial_summary_{NUM_INTERFACES}_{NUM_SERVERS}.csv"
    with open(raw_path, "w", newline="") as f_raw, \
            open(summary_path, "w", newline="") as f_sum:

        raw_writer = csv.writer(f_raw)
        sum_writer = csv.writer(f_sum)
        # header for raw per-step data
        raw_writer.writerow([
            "trial_id",
            "t",
            "algorithm",
            "action",
            "outcome",
            "cost",
            "action_size",
            "latencies_json"
        ])

        #header for per-trial summary
        sum_writer.writerow([
            "trial_id",
            "algorithm",
            "success_rate",
            "avg_cost",
            "avg_size",
            "total_switches",
            "avg_latency",
            "p95_latency",
            "cvar95_latency",
            "avg_dec_time"
        ])
        for i in range(NUM_TRIALS):
            trial_seed = 2024 + i
            print(f"  -> Running Trial {i + 1}/{NUM_TRIALS} (seed={trial_seed})...")
            # seeds 2024, 2025...
            #now run for single trial
            trial_res = run_single_trial(trial_seed=trial_seed, trial_id=i,
                                         step_writer=raw_writer)

            # Save per-trial results to CSV
            for name, res in trial_res.items():
                sum_writer.writerow([
                    i,  #trial_id
                    name,  #algorithm
                    res["success_rate"],
                    res["avg_cost"],
                    res["avg_size"],
                    res["total_switches"],
                    res["avg_latency"],
                    res["p95_latency"],
                    res["cvar95_latency"],
                    res["avg_dec_time"],
                ])

                for key, val in res.items():
                    all_results[name][key].append(val)



    print("\n\n=== FINAL AGGREGATED RESULTS (Mean +/- Std Dev) ===")

    print(
        f"{'Algorithm':<12} | "
        f"{'Success Rate':<20} | "
        f"{'Avg Cost':<12} | "
        f"{'Avg Lat(ms)':<14} | "
        f"{'P95 Lat(ms)':<14} | "
        f"{'CVaR95(ms)':<14} | "
        f"{'DecTime(ms)':<14} | "
        f"{'Switches':<10}"
    )
    print("-" * 130)

    for name, metrics in all_results.items():
        sr_mean = np.mean(metrics["success_rate"])
        sr_std = np.std(metrics["success_rate"])

        cost_mean = np.mean(metrics["avg_cost"])
        cost_std = np.std(metrics["avg_cost"])

        lat_mean = np.mean(metrics["avg_latency"])
        lat_std = np.std(metrics["avg_latency"])

        p95_mean = np.mean(metrics["p95_latency"])
        p95_std = np.std(metrics["p95_latency"])

        cvar_mean = np.mean(metrics["cvar95_latency"])
        cvar_std = np.std(metrics["cvar95_latency"])

        # avg_dec_time is in seconds, convert to ms
        dt_mean = np.mean(metrics["avg_dec_time"]) * 1000.0
        dt_std = np.std(metrics["avg_dec_time"]) * 1000.0

        sw_mean = np.mean(metrics["total_switches"])
        sw_std = np.std(metrics["total_switches"])

        print(
            f"{name:<12} | "
            f"{sr_mean * 100:6.2f}% +/- {sr_std * 100:4.2f}% | "
            f"{cost_mean:6.2f} +/- {cost_std:4.2f} | "
            f"{lat_mean:6.2f} +/- {lat_std:4.2f} | "
            f"{p95_mean:6.2f} +/- {p95_std:4.2f} | "
            f"{cvar_mean:6.2f} +/- {cvar_std:4.2f} | "
            f"{dt_mean:8.3f} +/- {dt_std:6.3f} | "
            f"{sw_mean:6.0f} +/- {sw_std:3.0f}"
        )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-trial network path selection experiments"
    )

    parser.add_argument("--num-interfaces", type=int, default=NUM_INTERFACES,
                        help="Number of network interfaces")
    parser.add_argument("--num-servers", type=int, default=NUM_SERVERS,
                        help="Number of servers")
    parser.add_argument("--congestion-penalty", type=float, default=CONGESTION_PENALTY,
                        help="Congestion penalty per extra path on same interface")
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS,
                        help="Number of independent trials")

    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()

    # Override global configuration
    NUM_INTERFACES = args.num_interfaces
    NUM_SERVERS = args.num_servers
    CONGESTION_PENALTY = args.congestion_penalty
    NUM_TRIALS = args.num_trials

    TOTAL_PATHS = NUM_INTERFACES * NUM_SERVERS

    print("=== Experiment Configuration ===")
    print(f"NUM_INTERFACES      = {NUM_INTERFACES}")
    print(f"NUM_SERVERS         = {NUM_SERVERS}")
    print(f"TOTAL_PATHS         = {TOTAL_PATHS}")
    print(f"CONGESTION_PENALTY  = {CONGESTION_PENALTY}")
    print(f"NUM_TRIALS          = {NUM_TRIALS}")
    print("===============================\n")
    run_experiment_suite()