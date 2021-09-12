#Robust Learning to Simulate (RL2S)
## Requirements
1. Install [MuJoCo 1.50](https://www.roboti.us/index.html) at `~/.mujoco/mjpro150` and copy your license key to `~/.mujoco/mjkey.txt`
2. `pip install -r requirements.txt`

## Data
All the data that RL2S needed is saved in `./l2s_dataset`.
- Training data are saved as trajectories in `./l2s_dataset/hopper/train_data/dataset_policy_{index}.pkl`
- Train policies, test policies and corresponding datasets are saved in `./l2s_dataset/hopper`. You can specify the data used to train and evaluate the simuator through `train_data_index` and `test_data_index` in `l2s_demos_listing.yaml`.
- The learned simulators which will be used for policy ranking and offline policy improvement are saved in `./l2s_dataset/leaned_dynamic/hopper`
- All the states that have appeared in the expert datasets are saved in `./l2s_dataset/end_data/hopper.pkl`

##Running
### Policy Value Difference Evaluation
To run RL2S, please use a command like this, and the `use_robust` in `l2s_hopper.yaml` should be set to true. During training, the AVD, MVD will be logged in `./l2s_logs/RL2S/.../progress.csv`

`python3 run_l2s.py  -e exp_specs/l2s_hopper.yaml  --nosrun -c 0`

For GAIL, just set the `use_robust` to false.

### Policy Ranking
Please use a command like this to get the performance of the policy in the learned simulator.

`python3 utils_script.py -d hopper -t 0 -g 0`

Please use a command like this to compute the kendall rank correlation coefficient and nDCG.

` python3 utils_script.py -d hopper -t 1`

### Policy Improvement
For policy improvement, run the command below.

`python3 run_l2s_downstream.py  -e exp_specs/l2s_downstream_hopper.yaml  --nosrun -c 2`