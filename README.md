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

### Source model
| Class         | IoU   | Acc   |
|---------------|-------|-------|
| building      | 50.98 | 84.39 |
| fence         | 26.92 | 33.36 |
| pedestrian    | 41.03 | 46.83 |
| pole          | 36.1  | 41.01 |
| road line     | 61.93 | 69.09 |
| road          | 93.38 | 96.22 |
| sidewalk      | 69.29 | 74.49 |
| vegetation    | 52.68 | 61.02 |
| vehicle       | 79.44 | 95.24 |
| wall          | 39.14 | 50.68 |
| traffic sign  | 31.91 | 35.32 |
| sky           | 54.44 | 57.53 |
| traffic light | 26.51 | 28.92 |
| terrain       | 35.63 | 42.59 |
| *Average*     | 49.96 | 58.34 |


### Cotta

| Class         | IoU   | Acc   |
|---------------|-------|-------|
| building      | 44.01 | 83.66 |
| fence         | 30.16 | 39.04 |
| pedestrian    | 46.68 | 53.66 |
| pole          | 44.92 | 55.03 |
| road line     | 65.98 | 74.36 |
| road          | 90.65 | 93.26 |
| sidewalk      | 69.62 | 73.85 |
| vegetation    | 54.93 | 65.03 |
| vehicle       | 69.78 | 96.65 |
| wall          | 41.13 | 54.5  |
| traffic sign  | 37.96 | 42.18 |
| sky           | 52.27 | 55.94 |
| traffic light | 32.81 | 36.61 |
| terrain       | 35.2  | 46.04 |
| *Average*     | 51.15 | 62.13 |


### Tent

| Class         | IoU   | Acc   |
|---------------|-------|-------|
| building      | 43.59 | 81.2  |
| fence         | 27.87 | 38.11 |
| pedestrian    | 42.63 | 50.2  |
| pole          | 39.88 | 51.73 |
| road line     | 62.33 | 72.35 |
| road          | 91.23 | 93.92 |
| sidewalk      | 68.27 | 73.55 |
| vegetation    | 55.68 | 66.5  |
| vehicle       | 68.12 | 96.46 |
| wall          | 41.15 | 55.14 |
| traffic sign  | 34.42 | 39.92 |
| sky           | 51.75 | 55.35 |
| traffic light | 30.32 | 35.0  |
| terrain       | 34.64 | 46.6  |
| *Average*     | 49.42 | 61.14 |

|           | Source model | Cotta   | Tent    |
|-----------|--------------|---------|---------|
| building  | 50.98        | 44.01   | 43.59   |
| fence     | 26.92        | 30.16   | 27.87   |
| pedestrian| 41.03        | 46.68   | 42.63   |
| pole      | 36.1         | 44.92   | 39.88   |
| road line | 61.93        | 65.98   | 62.33   |
| road      | 93.38        | 90.65   | 91.23   |
| sidewalk  | 69.29        | 69.62   | 68.27   |
| vegetation| 52.68        | 54.93   | 55.68   |
| vehicle   | 79.44        | 69.78   | 68.12   |
| wall      | 39.14        | 41.13   | 41.15   |
| traffic sign | 31.91     | 37.96   | 34.42   |
| sky       | 54.44        | 52.27   | 51.75   |
| traffic light | 26.51  | 32.81   | 30.32   |
| terrain   | 35.63        | 35.2    | 34.64   |
| *Average* | 49.96        | 51.15   | 49.42   |


|           | Source model | Cotta   | Tent    |
|-----------|--------------|---------|---------|
| building  | 84.39        | 83.66   | 81.2    |
| fence     | 33.36        | 39.04   | 38.11   |
| pedestrian| 46.83        | 53.66   | 50.2    |
| pole      | 41.01        | 55.03   | 51.73   |
| road line | 69.09        | 74.36   | 72.35   |
| road      | 96.22        | 93.26   | 93.92   |
| sidewalk  | 74.49        | 73.85   | 73.55   |
| vegetation| 61.02        | 65.03   | 66.5    |
| vehicle   | 95.24        | 96.65   | 96.46   |
| wall      | 50.68        | 54.5    | 55.14   |
| traffic sign | 35.32     | 42.18   | 39.92   |
| sky       | 57.53        | 55.94   | 55.35   |
| traffic light | 28.92  | 36.61   | 35.0    |
| terrain   | 42.59        | 46.04   | 46.6    |
| *Average* | 58.34        | 62.13   | 61.14   |
## License
Non-commercial. Code is heavily based on Cotta, MMSegmentaion. 

