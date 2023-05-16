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
- set dataset path in mmseg config in '/local_configs/', there are two settings for different dataset split: 'shift_val_800x500.py', 'shift_train_800x500.py'
- the following scripts are for *SHIFT_continuous_videos_1x_val_front* split test. 
```
bash run_base.sh
bash run_tent.sh
bash run_cotta.sh
# Example rerun logs are included in ./example_logs/base.log, tent.log, and cotta.log.
```
- for *SHIFT_continuous_videos_1x_train_front*,
modify config path 'shift_train_800x500.py' to 'shift_val_800x500.py' 
## Eval on SHIFT_continuous_videos_1x_val_front

### [Source model](https://github.com/zwbx/SHIFT-Continual_Test_Time_Adaptation/blob/master/work_dirs_train/source_model_eval/evluation.txt)

| Class         | IoU   | Acc   |
|---------------|-------|-------|
| building      | 43.81 | 80.69 |
| fence         | 18.89 | 27.51 |
| pedestrian    | 38.59 | 48.23 |
| pole          | 34.07 | 41.22 |
| road line     | 60.2  | 71.2  |
| road          | 87.21 | 89.5  |
| sidewalk      | 66.33 | 72.23 |
| vegetation    | 46.45 | 52.19 |
| vehicle       | 63.69 | 95.78 |
| wall          | 31.5  | 50.64 |
| sky           | 51.0  | 55.48 |
| traffic light | 25.06 | 30.05 |
| terrain       | 32.67 | 41.75 |
| traffic sign  | 16.69 | 18.5  |
| *Average*     | 44.01 | 55.35 |



### [Cotta](https://github.com/zwbx/SHIFT-Continual_Test_Time_Adaptation/blob/master/work_dirs_train/cotta_eval/evluation.txt)

| Class         | IoU   | Acc   |
|---------------|-------|-------|
| building      | 46.93 | 84.33 |
| fence         | 25.02 | 32.73 |
| pedestrian    | 49.46 | 57.1  |
| pole          | 48.93 | 58.55 |
| road line     | 66.92 | 75.0  |
| road          | 91.38 | 94.1  |
| sidewalk      | 72.34 | 76.64 |
| vegetation    | 51.24 | 61.39 |
| vehicle       | 73.06 | 96.35 |
| wall          | 43.88 | 57.31 |
| sky           | 54.99 | 59.52 |
| traffic light | 37.99 | 42.76 |
| terrain       | 40.59 | 48.94 |
| traffic sign  | 37.0  | 41.3  |
| *Average*     | 52.84 | 63.29 |


### [Tent](https://github.com/zwbx/SHIFT-Continual_Test_Time_Adaptation/blob/master/work_dirs_train/tent_eval/evluation.txt)
| Class         | IoU   | Acc   |
|---------------|-------|-------|
| building      | 52.1  | 83.95 |
| fence         | 22.11 | 27.97 |
| pedestrian    | 44.23 | 50.56 |
| pole          | 39.23 | 43.93 |
| road line     | 62.96 | 69.49 |
| road          | 93.41 | 96.43 |
| sidewalk      | 72.38 | 77.73 |
| vegetation    | 49.05 | 57.84 |
| vehicle       | 81.02 | 95.05 |
| wall          | 43.24 | 53.3  |
| sky           | 56.3  | 61.02 |
| traffic light | 30.51 | 33.87 |
| terrain       | 39.69 | 46.25 |
| traffic sign  | 30.47 | 33.68 |
| *Average*     | 51.19 | 59.36 |


## Toturial: least changes to create your own codebase
Our demo is based on mmsegmentaion. To conduct evaluation on SHIFT dataset, there are two main fuctions to be implemented:
- dataloader
- evaluator

### dataloader
We need to define each sequence as an independent dataset in mmsegmentation. 
- tools/{tent,test,cotta}.py

```
# select sequence
    with open(seq_info_path, 'r') as file:
        reader = csv.reader(file)
        next(reader) # skip header row
        for row in reader: 
            if True:# condition that filters seq. e.g. row[0]=='.....'
                seq_id_list.append(row[0])

# define config for each sequence
    seq_cfg_list =[]
    for i, seq in enumerate(os.listdir(os.path.join(cfg.data.test.data_root,cfg.data.test.img_dir))):
        if seq in seq_id_list:
            globals()["cfg.data.test{}".format(i)] = deepcopy(cfg.data.test)
            globals()["cfg.data.test{}".format(i)].img_dir = os.path.join(cfg.data.test.img_dir,seq)
            globals()["cfg.data.test{}".format(i)].ann_dir = os.path.join(cfg.data.test.ann_dir,seq)
            seq_cfg_list.append(globals()["cfg.data.test{}".format(i)])

# build dataset and dataloader
    datasets = [build_dataset(seq) for seq in seq_cfg_list]#, build_dataset(cfg.data.test1), build_dataset(cfg.data.test2),build_dataset(cfg.data.test3)]
    data_loaders = [build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False) for dataset in datasets]
```

- mmseg/datasets/shift.py
create shift.py
```
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        # note that sort the squence to conduct contintual test-time adaptation
        img_infos_sorted = sorted(img_infos, key=lambda x: x['filename'])[:2]
        return img_infos_sorted 
```
### evaluator
Our evaluation follows the following steps: 
1. Test each sequence independently to obtain IoU and Acc on each category. 
2. Calculate the average of all sequences for each category. 
3. Calculate the overall average.

- tools/{tent,test,cotta}.py
after *dataset.evaluate* on each sequence, we store the results into a json file. Finally, *res_process()* aggregate all the results of each sequence and outoput the last evaluation results.
```
            if args.eval:
                _, eval_res,_ = dataset.evaluate(outputs, args.eval, **kwargs)
                out_dir = './Test_on_{}/tent_eval/'.format(cfg.data_split_type)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir + 'res')
                mmcv.dump(eval_res, out_dir + 'res/{}.json'.format(seq_name), indent=4)
    res_process(out_dir,cfg.csv_root)
```

Note that evaluation for each sequence requires reloading pretrained parameters of source model.


- tools/res_process.py
```
def res_process(res_path,csv_root,in_domain=False):
        # Initialize the dictionary
    json_dict = {}
    seq_path = os.path.join(res_path,'res') 
    # Loop over all files in the path
    for file_name in os.listdir(seq_path):
        # Check if the file is a JSON file
        if file_name.endswith(".json"):
            # Extract the sequence ID from the file name
            seq_id = file_name.split(".")[0]
            # Read the JSON file
            with open(os.path.join(seq_path, file_name), "r") as f:
                content = json.load(f)
            # Append the content to the dictionary
            json_dict[seq_id] = content # json dict 
            ...
```

## License
Non-commercial. Code is heavily based on Cotta, MMSegmentaion. 

