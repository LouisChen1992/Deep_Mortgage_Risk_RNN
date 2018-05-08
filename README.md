# Deep_Mortgage_Risk_RNN

### Data Processing
Run the following commands to create randomized buckets: 
```
$ python3 python3 run_data_process.py --dataset subprime --bucket 5 # bucket size 5 for subprime data
$ python3 python3 run_data_process.py --dataset prime --bucket 5 # bucket size 5 for prime data
$ python3 python3 run_data_process.py --dataset all # merge subprime data with prime data
```

### Train, Validation & Test
- Train & Validation
```
$ python3 run.py --config=configs/baseline/config_baseline_ff.json --logdir=output/all/baseline/baseline_ff --mode=valid --dataset=all --num_epochs=20 --summary_frequency=1000
```

The table below reports test loss for the best fully-connected model (on validation set):

| Epoch | Train Loss | Validation Loss | Test Loss |
|:-----:|:----------:|:---------------:|:---------:|
| 8     | 0.1754     | 0.2152          | 0.1903    |
