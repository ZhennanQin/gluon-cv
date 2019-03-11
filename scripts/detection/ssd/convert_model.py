import argparse
import os
import mxnet as mx
import gluoncv as gcv


def main():

	DEFAULT_NETWORK = 'vgg16_atrous'
	DEFAULT_DATA_SET = 'coco'
	DEFAULT_DATA_SHAPE = 300
	DEFAULT_BATCH_SIZE = 64
	DEFAULT_MODEL_PATH = 'model'
	DEFAULT_MODEL_PREFIX = 'converted'
	NMS_THRESHOLD = 0.45
	NMS_TOP_K = 400
	CHANNEL_COUNT = 3

	parser = argparse.ArgumentParser()

	parser.add_argument('--network', type = str, default = DEFAULT_NETWORK, help = 'name of the base network (default = ' + DEFAULT_NETWORK + ')')
	parser.add_argument('--dataset', type = str, default = DEFAULT_DATA_SET, help = 'name of the data set (default = ' + DEFAULT_DATA_SET + ')')
	parser.add_argument('--data_shape', type = int, default = DEFAULT_DATA_SHAPE, help = 'shape of the input data (default = ' + str(DEFAULT_DATA_SHAPE) + ')')
	parser.add_argument('--batch_size', type = int, default = DEFAULT_BATCH_SIZE, help = 'size of the batch (default = ' + str(DEFAULT_BATCH_SIZE) + ')')
	parser.add_argument('--model_path', type = str, default = DEFAULT_MODEL_PATH, help = 'root path of the model (default = ' + DEFAULT_MODEL_PATH + ')')
	parser.add_argument('--model_prefix', type = str, default = DEFAULT_MODEL_PREFIX, help = 'prefix of the model (default = ' + DEFAULT_MODEL_PREFIX + ')')

	args = parser.parse_args()

	# Retrieve original model
	net_name = '_'.join(('ssd', str(args.data_shape), args.network, args.dataset))
	net = gcv.model_zoo.get_model(net_name, pretrained = True, root = args.model_path)
	net.set_nms(nms_thresh = NMS_THRESHOLD, nms_topk = NMS_TOP_K)

	# Convert model
	print('Converting model...')
	net.hybridize(static_alloc = True, static_shape = True)
	data_shape = [args.batch_size, CHANNEL_COUNT, args.data_shape, args.data_shape]
	dummy_data = mx.nd.ones(data_shape).as_in_context(mx.cpu())
	net.forward(dummy_data)

	# Save model
	print('Saving model...')
	net.export(os.path.join(args.model_path, args.model_prefix))


if __name__ == '__main__':

	main()
