# SHIFT-Continual_Test_Time_Adaptation-Semantic_Segmentaion_Benchmark

## enviromnet setup
create conda envrioment
```
conda create -n SHIFT_CTTA python=3.8
conda activate SHIFT_CTTA
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```
install mmcv-full
```
wget https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/mmcv_full-1.2.7-cp38-cp38-manylinux1_x86_64.whl
pip install mmcv_full-1.2.7-cp38-cp38-manylinux1_x86_64.whl
```

install MMSegmentaion
```
pip install -v -e .
```

install other dependencies
```
pip install -r requirements.txt
```


## Run
- download pretrained source model [iter_40000.pth](https://drive.google.com/file/d/1J7a8k-XBi9LGNhciOw5xzmQ-GAb-tff6/view?usp=sharing), then place it in 'deeplabv3_r50_shift_800x500' folder
- set dataset path in mmseg config in '/local_configs/', there are two settings for different dataset split: shift_val_800x500.py, shift_train_800x500.py 
- the following commands test on val split.
```
bash run_base.sh
bash run_tent.sh
bash run_cotta.sh
# Example rerun logs are included in ./example_logs/base.log, tent.log, and cotta.log.
```
## Benchmark

|           | building | fence | pedestrian | pole | road line | road | sidewalk | vegetation | vehicle | wall | traffic sign | sky | traffic light | terrain | *Average* |
|-----------|----------|-------|------------|------|-----------|------|----------|------------|---------|------|--------------|-----|---------------|---------|-----------|
| Source model | 50.98    | 26.92 | 41.03      | 36.10 | 61.93     | 93.38| 69.29    | 52.68      | 79.44   | 39.14| 31.91        | 54.44 | 26.51         | 35.63   | 49.96     |
| Cotta     | 44.01    | 30.16 | 46.68      | 44.92 | 65.98     | 90.65| 69.62    | 54.93      | 69.78   | 41.13| 37.96        | 52.27 | 32.81         | 35.20   | 51.15     |
| Tent      | 43.59    | 27.87 | 42.63      | 39.88 | 62.33     | 91.23| 68.27    | 55.68      | 68.12   | 41.15| 34.42        | 51.75 | 30.32         | 34.64   | 49.42     |

|           | building | fence | pedestrian | pole | road line | road | sidewalk | vegetation | vehicle | wall | traffic sign | sky | traffic light | terrain | *Average* |
|-----------|----------|-------|------------|------|-----------|------|----------|------------|---------|------|--------------|-----|---------------|---------|-----------|
| Source model | 84.39    | 33.36 | 46.83      | 41.01 | 69.09     | 96.22| 74.49    | 61.02      | 95.24   | 50.68| 35.32        | 57.53 | 28.92         | 42.59   | 58.34     |
| Cotta     | 83.66    | 39.04 | 53.66      | 55.03 | 74.36     | 93.26| 73.85    | 65.03      | 96.65   | 54.50| 42.18        | 55.94 | 36.61         | 46.04   | 62.13     |
| Tent      | 81.20    | 38.11 | 50.20      | 51.73 | 72.35     | 93.92| 73.55    | 66.50      | 96.46   | 55.14| 39.92        | 55.35 | 35.00         | 46.60   | 61.14     |

## Toturial
## License
Non-commercial. Code is heavily based on Cotta, MMSegmentaion. 

