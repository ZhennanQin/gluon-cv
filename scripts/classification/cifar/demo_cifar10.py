import argparse

import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt

from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model

from mxnet.contrib import amp

#amp.init(target_dtype='bfloat16')

parser = argparse.ArgumentParser(description='Predict CIFAR10 classes from a given image')
parser.add_argument('--model', type=str, required=True,
                    help='name of the model to use')
parser.add_argument('--saved-params', type=str, default='',
                    help='path to the saved model parameters')
parser.add_argument('--batch-size', type=int, default=128,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=2, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are imperative, hybrid')
opt = parser.parse_args()

classes = 10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

ctx = [mx.cpu()]

# Load Model
model_name = opt.model
pretrained = True if opt.saved_params == '' else False
kwargs = {'classes': classes, 'pretrained': pretrained}
net = get_model(model_name, **kwargs)

if opt.mode == 'hybrid':
    net.hybridize(static_alloc=True, static_shape=True)

if not pretrained:
    net.load_parameters(opt.saved_params, ctx = ctx)

def test(ctx, val_data):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
        break
    return metric.get()


transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

val_data = gluon.data.DataLoader(
           gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
           batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

name, val_acc = test(ctx, val_data)

print('val=%f' % val_acc)

