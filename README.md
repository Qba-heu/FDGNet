# FDGNet: Frequency Disentanglement and Data Geometry for Domain Generalization in Cross-Scene Hyperspectral Image Classification

This repository contains the `PyTorch` implementation for the early access FDGNet, which has been accpted by TNNLS.


* The detailed citation and implementation will be updated when the paper is published.


The default dataset folder is `./Datasets/`.

### Running
* running FDGNet by
```
python train_manifold.py --source_name Houston13 --target_name Houston18 --data_path ./Datasets/Houston/ --re_ratio 5 --training_sample_ratio 0.8  --d_se 64 --lambda_1 1.0 --lambda_2 1.0 --low_freq True --patch_size 13;
python train_manifold.py --source_name paviaU --target_name paviaC --data_path ./Datasets/Pavia/ --re_ratio 1 --training_sample_ratio 0.8  --d_se 64 --lambda_1 1.0 --lambda_2 1.0 --low_freq True --patch_size 13;
python train_manifold.py --source_name whu071 --target_name whu078 --data_path ./Datasets/WUH_15-71/ --re_ratio 1 --training_sample_ratio 0.1  --d_se 64 --lambda_1 1.0 --lambda_2 1.0 --low_freq True --patch_size 17
```

### Shell
- [x] sh file: `sh RUN_FDGNet.sh`
