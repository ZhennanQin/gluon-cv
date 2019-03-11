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
from mxnet.contrib.quantization import *

def parse_args():
    parser = argparse.ArgumentParser(description='Eval SSD networks.')
    parser.add_argument('--model-prefix', type=str, default='converted',
                        help='Prefix of converted model.')
    parser.add_argument('--data-shape', type=int, default=300,
                        help="Input data shape")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=1, help='Number of data workers')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--num-calib-batches', type=int, default=4,
                        help='number of batches for calibration')
    parser.add_argument('--calib-mode', type=str, default='naive',
                        help='calibration mode used for generating calibration table for the quantized symbol; supports'
                             ' 1. naive: simply take min and max values of layer outputs as thresholds for'
                             ' quantization. In general, the inference accuracy worsens with more examples used in'
                             ' calibration. It is recommended to use `entropy` mode as it produces more accurate'
                             ' inference results.'
                             ' 2. entropy: calculate KL divergence of the fp32 output and quantized output for optimal'
                             ' thresholds. This mode is expected to produce the best inference accuracy of all three'
                             ' kinds of quantized models if the calibration dataset is representative enough of the'
                             ' inference dataset.')
    args = parser.parse_args()
    return args


def save_symbol(fname, sym, logger=None):
    if logger is not None:
        logger.info('Saving symbol into file at %s' % fname)
    sym.save(fname)


def save_params(fname, arg_params, aux_params, logger=None):
    if logger is not None:
        logger.info('Saving params into file at %s' % fname)
    save_dict = {('arg:%s' % k): v.as_in_context(cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(cpu()) for k, v in aux_params.items()})
    mx.nd.save(fname, save_dict)


def get_dataset(dataset, data_shape):
    if dataset.lower() == 'voc':
        val_dataset = gdata.VOCDetection(splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False, root='data')
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
        batch_size=batch_size, shuffle=True, last_batch='keep', num_workers=num_workers)
    return val_loader

class COCOIter(mx.io.DataIter):
    def __init__(self, val_data, ctx, data_shape, batch_size):
        super(COCOIter, self).__init__(batch_size)
        self.data_shape = (batch_size, 3,) + (data_shape, data_shape)
        self.label_shape = (batch_size,)
        self.provide_data = [('data', self.data_shape)]
        self.provide_label = []
        self.val_data = val_data

    def __iter__(self):
        for ib, batch in enumerate(self.val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=[ctx], batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=[ctx], batch_axis=0, even_split=False)
            yield mx.io.DataBatch(data=data, label=label)

if __name__ == '__main__':

    CHANNEL_COUNT = 3

    args = parse_args()

    # training contexts
    #ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    #ctx = ctx if ctx else [mx.cpu()]
    ctx = mx.cpu()

    # network
    sym, arg_params, aux_params = mx.model.load_checkpoint('model/' + args.model_prefix, 0)
    sym = sym.get_backend_symbol('MKLDNN')
    # training data
    val_dataset, val_metric = get_dataset(args.dataset, args.data_shape)
    val_data = get_dataloader(
        val_dataset, args.data_shape, args.batch_size, args.num_workers)
    classes = val_dataset.classes  # class names

    data = COCOIter(val_data, ctx, args.data_shape, args.batch_size)
    excluded_sym_names = ["ssd0_flatten0", "ssd0_flatten1", "ssd0_flatten2", "ssd0_flatten3", "ssd0_flatten4", "ssd0_flatten5",
                          "ssd0_flatten6", "ssd0_flatten7", "ssd0_flatten8",  "ssd0_flatten9", "ssd0_flatten10", "ssd0_flatten11",
                          "ssd0_concat1", "ssd0_concat0", "ssd0_concat3", "ssd0_bboxcentertocorner0_concat0"]
    calib_layer = lambda name: ((name.startswith("sg_mkldnn_conv") and name.endswith('_output'))
                                or name == "data" or name.endswith("mul0_0"))
    qsym, qarg_params, aux_params = quantize_model(
        sym=sym,
        arg_params=arg_params,
        aux_params=aux_params,
        ctx=ctx,
        excluded_sym_names=excluded_sym_names,
        calib_mode=args.calib_mode,
        calib_data=data,
        num_calib_examples=args.num_calib_batches * args.batch_size,
        calib_layer=calib_layer,
        quantized_dtype='auto',
        label_names=())
    prefix = 'quantized'
    sym_name = 'model/%s-symbol.json' % (prefix)
    qsym = qsym.get_backend_symbol('MKLDNN_POST_QUANTIZE')
    save_symbol(sym_name, qsym)
    param_name = 'model/%s-%04d.params' % (prefix, 0)
    save_params(param_name, qarg_params, aux_params)
