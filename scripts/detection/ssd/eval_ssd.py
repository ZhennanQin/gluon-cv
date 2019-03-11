from __future__ import division
from __future__ import print_function

import argparse
import logging
logging.basicConfig(level=logging.INFO)
import time
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet import nd
from mxnet import gluon
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric

def parse_args():
    parser = argparse.ArgumentParser(description='Eval SSD networks.')
    parser.add_argument('--model-prefix', type=str, default='converted',
                        help='Prefix of converted model.')
    parser.add_argument('--data-shape', type=int, default=300,
                        help="Input data shape")
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    args = parser.parse_args()
    return args

def get_dataset(dataset, data_shape):
    if dataset.lower() == 'voc':
        val_dataset = gdata.VOCDetection(splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False, root='./data/')
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(data_shape, data_shape))
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric

def get_dataloader(val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)), batchify_fn=batchify_fn,
        batch_size=batch_size, shuffle=False, last_batch='keep', num_workers=num_workers)
    return val_loader

def validate(net, val_data, ctx, classes, size, metric):
    """Test on validation dataset."""
    metric.reset()

    with tqdm(total=size) as pbar:
        i = 0
        for ib, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            dry_run = 2
            for x, y in zip(data, label):
                if i == dry_run:
                    tic = time.time()
                    batch_size = batch[0].shape[0]
                i += 1
                net.forward(mx.io.DataBatch([x]))
                ids, scores, bboxes = net.get_outputs()
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
            metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
            pbar.update(batch[0].shape[0])
    total_time = time.time() - tic
    num_img = (i - dry_run) * batch_size
    fps = num_img / total_time
    latency = 1000 * total_time / (i - dry_run)
    print("inference completed. thoughput %f img/s, latency %f ms." % (fps, latency))

    return metric.get()

if __name__ == '__main__':

    CHANNEL_COUNT = 3

    args = parse_args()

    # training contexts
    #ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    #ctx = ctx if ctx else [mx.cpu()]
    ctx = [mx.cpu()]
    # network
    sym, arg_params, aux_params = mx.model.load_checkpoint('./model/' + args.model_prefix, 0)
    #graph = mx.viz.plot_network(sym, save_format='png')
    #graph.render(args.model_prefix)

    net = mx.mod.Module(sym, label_names=[], context=ctx)
    data_shape = [args.batch_size, CHANNEL_COUNT, args.data_shape, args.data_shape]
    net.bind(data_shapes=[("data", data_shape)], inputs_need_grad=False, for_training=False)
    net.set_params(arg_params=arg_params, aux_params=aux_params)

    # training data
    val_dataset, val_metric = get_dataset(args.dataset, args.data_shape)
    val_data = get_dataloader(
        val_dataset, args.data_shape, args.batch_size, args.num_workers)
    classes = val_dataset.classes  # class names

    # training
    names, values = validate(net, val_data, ctx, classes, len(val_dataset), val_metric)
    for k, v in zip(names, values):
        print(k, v)
