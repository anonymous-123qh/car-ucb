# car-ucb

## Installation

### Prerequisites
- Python 3.10 
- pip

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/<repository>.git
   cd <car-ucb>
2. Install packages:
   ```bash
    pip install -r requirements.txt
    ```
## Usage and Reproduce Results
### Running the Simulator

Execute the main simulation script, passing the critical parameters for the system configuration.

| Parameter | Description |
| :--- | :--- |
| `--num-interfaces` | The number of network interfaces available. |
| `--num-servers` | The number of processing servers. |

```
python main_simulator.py \
    --num-interfaces 2 \
    --num-servers 3
```
### Plotting Results
After successfully running the simulator, the data will be saved to the data_output directory.

#### CDF latency plot: 
   Run, e.g., to draw a shaded latency plot for experiments with #interface 2, #server 3
  ```
  python shaded_latency_plot.py \
    --num-interfaces 2 \
    --num-servers 3
  ```
          
### Regret plot:
  Run, e.g., to draw a regret plot for experiments with #interface 2, #server 3
  ```
  python regret_plot.py \
    --num-interfaces 2 \
    --num-servers 3
  ```
### Ground-truth network dynamics and algorithms'decision plot:
  Run , e.g., to draw  for experiments with #interface 2, #server 3
  
  ```
  python qualitative_plot.py \
    --num-interfaces 2 \
    --num-servers 3
  ```
### Multiple-agent decision plot:
1. Run the experiment with
 ```
 python multiple_car_ucb_multiple_seed.py \
   --num-interfaces 2 \
   --num-servers 3
 ```
3. Draw the plot result
```
python herd_avoidance_analyse.py 
```

  
       
