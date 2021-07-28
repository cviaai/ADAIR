## Usage

### Configurations

You should create `configs.yaml` configuration file in the `root` directory like this with the following fields:

```
paths:
    data:
        train_data: ../Datasets/Vein_data # path to folder relative to the root with train data
    dataset_table: dataset.csv # path to dataset csv table
    log_dir: logs # path to folder relative to the root where the results of experiments from the tensorboard will be recorded

data_parameters:
    image_size: 512, 512 # model image input size (height, width)
    batch_size: 4 # batch size

model_parameters:
    DnCNN:
        num_features: 32 # number of channels (features) in initial conv
        num_layers: 20 # number of layers
    Fourier:
        fourier_layer: None # None, linear, linear_log initial adaptive Fourier layer

train_parameters:
    lr: 0.001 # learning rate
    epochs: 25 # number of epochs
```

### Models Training

To train the model run this command:

`python train_denoising.py --root=<path to the project>`

* In the `train_denoising.py` file, you can choose the type of noise by comment out or vice versa rows with noise overlaying transforms.
