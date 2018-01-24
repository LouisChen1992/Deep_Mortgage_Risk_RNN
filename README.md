# Deep_Mortgage_Risk_RNN

### Data Processing
Run the following commands to create randomized buckets: 
```
$ python3 python3 run_data_process.py --dataset subprime --bucket 5 # bucket size 5 for subprime data
$ python3 python3 run_data_process.py --dataset prime --bucket 5 # bucket size 5 for prime data
$ python3 python3 run_data_process.py --dataset all # merge subprime data with prime data
```