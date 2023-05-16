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
download pretrained source model iter_40000.pth, then place it in 'deeplabv3_r50_shift_800x500' folder
- [iter_40000.pth](https://drive.google.com/file/d/1J7a8k-XBi9LGNhciOw5xzmQ-GAb-tff6/view?usp=sharing)
set dataset path in mmseg config in '/local_configs/'
there are two settings for different dataset split: shift_val_800x500.py, shift_train_800x500.py 
the following commands test on val split.
```
bash run_base.sh
bash run_tent.sh
bash run_cotta.sh
# Example rerun logs are included in ./example_logs/base.log, tent.log, and cotta.log.
```


## License
Non-commercial. Code is heavily based on Cotta, MMSegmentaion. 

