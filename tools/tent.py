import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction
import csv
from mmseg.apis import multi_gpu_test, single_gpu_test, single_gpu_tent
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from IPython import embed
from copy import deepcopy
from res_process import res_process

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', default='./res.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='mIoU',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if 'None' in args.eval:
        args.eval = None
    if args.eval and args.format_only:

        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    print(cfg)

    seq_info_path = cfg.csv_root
    seq_id_list = []

    # select seq
    with open(seq_info_path, 'r') as file:
        reader = csv.reader(file)
        next(reader) # skip header row
        for row in reader: 
            if True:# condition that filters seq. e.g. row[0]=='.....'
                seq_id_list.append(row[0])

    seq_cfg_list =[]
    for i, seq in enumerate(os.listdir(os.path.join(cfg.data.test.data_root,cfg.data.test.img_dir))):
        if seq in seq_id_list:
            globals()["cfg.data.test{}".format(i)] = deepcopy(cfg.data.test)
            globals()["cfg.data.test{}".format(i)].img_dir = os.path.join(cfg.data.test.img_dir,seq)
            globals()["cfg.data.test{}".format(i)].ann_dir = os.path.join(cfg.data.test.ann_dir,seq)
            seq_cfg_list.append(globals()["cfg.data.test{}".format(i)])
        
    datasets = [build_dataset(seq) for seq in seq_cfg_list]#, build_dataset(cfg.data.test1), build_dataset(cfg.data.test2),build_dataset(cfg.data.test3)]
    data_loaders = [build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False) for dataset in datasets]

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    efficient_test = True #False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    for dataset, data_loader in zip(datasets, data_loaders):
        seq_name = dataset.img_dir.split('/')[-1]
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_tent(model, data_loader, args.show, args.show_dir,
                                efficient_test)

        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset.format_results(outputs, **kwargs)
            if args.eval:
                _, eval_res,_ = dataset.evaluate(outputs, args.eval, **kwargs)
                out_dir = './Test_on_{}/tent_eval/'.format(cfg.data_split_type)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir + 'res')
                mmcv.dump(eval_res, out_dir + 'res/{}.json'.format(seq_name), indent=4)
    res_process(out_dir,cfg.csv_root)

if __name__ == '__main__':
    main()
