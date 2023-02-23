Self-Supervised Learning via Image Reconstruction
================================================================================

This repository hosts a cleaned up implementation of Image Reconstruction w/ 
Coarse Cutout based SSL pretraining in association with my thesis work.

How To Train
--------------------------------------------------------------------------------

Launching the training scripts is fairly automatic (thanks to pytorch lightning).

```sh
# Will train the SSL model with Image Reconstruction w/ Coarse Cutout
py -m train --seed 3741 --config ./experiments/voc_reconstruction_resunet.yml
```

After that completes, you can run:

```sh
# Will train the segmentaton model with weights initialized from the model above
py -m train --seed 3741 --config ./experiments/voc_resunet_pretrain_reconstruction.yml
```

**Note:** When training with weight intialization from an SSL checkpoint, we expect a mismatched parameters warning. This is due to different class counts in the final layer of the encoder.

```
Found 2 missing or mismatched parameters. The model will still have original values for these parameters.
 -> ['underlying_model.mask.conv.weight', 'underlying_model.mask.conv.bias']
Overwriting target model state!
```

Additionally, for baseline comparisons:

```sh
# Trains the segmentation model with default random initialization
py -m train --seed 3741 --config ./experiments/voc_resunet_pretrain_none.yml
```

```sh
# Trains the segmentation model with torchvision provided ImageNet pretraining
py -m train --seed 3741 --config ./experiments/voc_resunet_pretrain_library.yml
```

Requirements
--------------------------------------------------------------------------------

See `requirements.txt` for details.

Example Experiment YML
--------------------------------------------------------------------------------

Example: `voc_reconstruction_resunet.yml` for SSL pretraining.

```yml
# Dataset Parameters
dataset: pascal_voc_unlabelled
dataset_args:
  batch_size: 32 # per-gpu

# Augmentation Parameters
augmentation: reconstruction_coarse_cutout
augmentation_args:
  augmentation_steps:
    - HorizontalFlip
    - ColorJitter
  normalization:
    # Normalization values generated from pascal_voc_unlabelled
    mean: [0.45286129, 0.43170348, 0.39989259]
    std: [0.44426655, 0.46648413, 0.48871749]
  resize: 256

# Model Parameters
model: reconstruction
model_args:
  underlying_model: resunet34
  underlying_model_args:
    num_classes: 3 # rgb image
    ignore_class: null
    pretrain: false

# Optimizer Parameters (Adam)
optimizer_args:
  lr: 3.0e-3

# Scheduler Parameters (Cosine Annealing)
scheduler_args:
  min_multiplier: 1.0e-2
  warmup_epochs: 10

# Training Duration
epochs: 1000
```

Example: `voc_resunet_pretrain_reconstruction.yml` for training the segmentation 
model initialized with SSL weights.

```yml
# Dataset Parameters
dataset: pascal_voc_segmentation
dataset_args:
  batch_size: 32 # per-gpu

# Augmentation Parameters
augmentation: custom
augmentation_args:
  augmentation_steps:
    - HorizontalFlip
    - ColorJitter
  normalization:
    # Normalization values generated from pascal_voc_unlabelled
    mean: [0.45286129, 0.43170348, 0.39989259]
    std: [0.44426655, 0.46648413, 0.48871749]
  resize: 256

# Model Parameters
model: resunet34
model_args:
  num_classes: 22
  ignore_class: 21
  pretrain: voc_reconstruction_resunet/3741

# Optimizer Parameters (Adam)
optimizer_args:
  lr: 3.0e-3

# Scheduler Parameters (Cosine Annealing)
scheduler_args:
  min_multiplier: 1.0e-2
  warmup_epochs: 10

# Training Duration
epochs: 500
```
