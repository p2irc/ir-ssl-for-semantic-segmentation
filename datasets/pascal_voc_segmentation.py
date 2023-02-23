import os
from typing import List, Optional, Tuple

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import config
from augmentations import get_augmentation
from utilities.augmentations import AugmentedDataset
from utilities.helpers import (
    STAGE_TEST,
    STAGE_TRAIN,
    STAGE_VALIDATION,
    get_sufficient_num_workers,
)
from utilities.io import load_image

CLASS_NAMES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
    "void",
]

CLASS_COLORS = [
    np.array(x)
    for x in [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
        [217, 217, 187],  # void
    ]
]


class PascalVocSegmentation(LightningDataModule):

    identifier = "pascal_voc_segmentation"

    def __init__(self, batch_size: int):
        super().__init__()

        self.batch_size = batch_size

        def get_samples(era, path):

            # Get the directory of the specific year
            directory = os.path.join(config.DATASET_DIRECTORY, "VOCdevkit", era)

            # Load sample names in each group
            with open(os.path.join(directory, "ImageSets/Segmentation", path), "r") as f:
                names = f.read().splitlines()

            def get_label_pair(name):

                img = os.path.join(directory, "JPEGImages", name + ".jpg")
                img_exist = os.path.exists(img)

                lbl = os.path.join(directory, "SegmentationClass", name + ".png")
                lbl_exist = os.path.exists(lbl)

                # print(f"path: '{path}'", f"'{name}'", img_exist, lbl_exist)
                return (img, lbl) if (img_exist and lbl_exist) else None

            # Map samples to image pairs
            return list(filter(None, [get_label_pair(name) for name in names]))

        # Map samples to stage.
        # We use the 2012 dataset for training and validation as it is the most recent.
        # We use the 2007 dataset for testing as labels are publically available.
        self._datasets = {
            STAGE_TRAIN: get_samples("VOC2012", "train.txt"),
            STAGE_VALIDATION: get_samples("VOC2012", "val.txt"),
            STAGE_TEST: get_samples("VOC2007", "test.txt"),
        }

        # Instantiate desired augmentation class
        self.augmentations = get_augmentation(config.AUGMENTATION_NAME, config.AUGMENTATION_ARGS)

    @property
    def train_samples(self) -> List[Tuple[str, str]]:
        return self._datasets[STAGE_TRAIN]

    @property
    def val_samples(self) -> List[Tuple[str, str]]:
        return self._datasets[STAGE_VALIDATION]

    @property
    def test_samples(self) -> List[Tuple[str, str]]:
        return self._datasets[STAGE_TEST]

    def get_dataset(self, stage: str) -> Optional[Dataset]:
        return PascalVocSegmentation._Dataset(self._datasets[stage])

    def train_dataloader(self):
        return self._get_dataloader(STAGE_TRAIN)

    def val_dataloader(self):
        return self._get_dataloader(STAGE_VALIDATION)

    def test_dataloader(self):
        return self._get_dataloader(STAGE_TEST)

    def get_class_color(self, i: int) -> np.ndarray:
        return CLASS_COLORS[i]

    def get_class_name(self, i: int) -> str:
        return CLASS_NAMES[i]

    def _get_dataloader(self, stage: str):

        return DataLoader(
            AugmentedDataset(self, self.get_dataset(stage), self.augmentations, stage),
            batch_size=(1 if stage == STAGE_TEST else self.batch_size),
            persistent_workers=True,
            num_workers=get_sufficient_num_workers(),
            shuffle=(stage == STAGE_TRAIN),
            pin_memory=True,
            drop_last=True,
        )

    class _Dataset(Dataset):
        """"""

        def __init__(self, samples: List[Tuple[str, str]]):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):

            # Get image and label paths
            image_path, label_path = self.samples[idx]

            # Read the image and label
            image = load_image(image_path)

            # Load color label
            raw_label = load_image(label_path)
            raw_label[raw_label == 255] = 21

            return image, raw_label
