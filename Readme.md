# Data generator

`python data_generator.py $dataset_size $batch_size --gpus $gpus`

`dataset_size`: Dataset size. This argument determines the number of data samples to be generated by the data generator.

`batch_size`: Batch size. This argument specifies the number of data samples to be processed at once during training or model evaluation.

`--gpus`: Number of GPUs. This argument indicates the number of Graphics Processing Units (GPUs) to be used for data processing.

# Train

## Start train
`python train.py configs/base.py` 

### Wandb (logging)
https://wandb.ai/misterrendal/text_ai_classification

# Production
`python predict.py $onnx_model --batch-size $batch-size`

`onnx_model` : Path to the ONNX model file.

`--batch-size 16`: Set batch size for prediction.
