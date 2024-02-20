# Adversarial Feature Map Pruning for Backdoor

## How to run FMT for different attack?

### First, generate injected model

Taking badnet and cifar10 dataset as a example.
```
python ./attack/badnet_attack.py --yaml_path ../config/attack/badnet/cifar10.yaml --dataset cifar10 --dataset_path ../data --save_folder_name badnet_cifar10
```

### Second, use FMT to repair DNN model.

```
python ./defense/feature.py  --yaml_path ./config/defense/feature/cifar10.yaml --dataset cifar10 --result_file badnet_cifar10
```

### Then, run baseline approaches~(e.g., FT).

```
python ./defense/ft/ft.py --result_file badnet_cifar10 --yaml_path ./config/defense/ft/cifar10.yaml --dataset cifar10
```
