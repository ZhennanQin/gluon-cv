# SSD-VGG16

## Reference
https://arxiv.org/abs/1512.02325

## Implementation

https://gitlab.devtools.intel.com/zhennanq/gluon-cv.git

## Prepare Data Set
COCO 2017 (http://cocodataset.org/)

From the root directory of gluon-cv, set the python path using the following command.
```
export PYTHONPATH=$PYTHONPATH:`pwd`
```

Enter work directory.

```
cd scripts/detection/ssd
```

Create a `data` directory using the following command.
```
mkdir ./data
```

Download the validation images file to the `./temp` directory using the following command.
```
wget http://images.cocodataset.org/zips/val2017.zip -P ./data/
```

Decompress the validation images file using the following command.
```
unzip ./data/val2017.zip -d ./data/
```

Download the validation annotations file to the `./data` directory using the following command.
```
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ./data/
```

Decompress the validation annotations file using the following command.
```
unzip ./data/annotations_trainval2017.zip -d ./data/
```


#Prepare model

Create a `model` directory using the following command.
```
mkdir ./model
```

Convert gluon model into symbolic.
```
python convert_model.py
```

Quantize model.
```
python quantize_ssd.py
```

Setup running environment
```
export OMP_NUM_THREADS=27
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
```


Validate the fp32 model using the following command.
```
numactl --physcpubind=0-27 --membind=0 python eval_ssd.py --model-prefix converted --batch-size=64 --num-workers=1
```
Upon completion of this command, the validation COCO mAP will be displayed and should closely match the validation COCO mAP shown below.

~~~~ MeanAP @ IoU=[0.50,0.95] ~~~~
 24.059


Validate the quantized model using the following command.
```
numactl --physcpubind=0-27 --membind=0 python eval_ssd.py --model-prefix quantized --batch-size=64 --num-workers=1
```
Upon completion of this command, the validation COCO mAP will be displayed and should closely match the validation COCO mAP shown below.

~~~~ MeanAP @ IoU=[0.50,0.95] ~~~~
 23.879
