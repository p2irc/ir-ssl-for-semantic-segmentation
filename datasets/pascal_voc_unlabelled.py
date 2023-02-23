import glob
import os
from typing import List, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

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


class PascalVocUnlabelled(LightningDataModule):

    identifier = "pascal_voc_unlabelled"

    def __init__(self, batch_size: int):
        super().__init__()

        self.batch_size = batch_size

        # Full path to VOC directory
        voc_directory = os.path.join(config.DATASET_DIRECTORY, "VOCdevkit")

        def get_samples(era, path):

            # Get the directory of the specific year
            directory = os.path.join(voc_directory, era)

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

        # Get all known segmentation samples
        train_samples = get_samples("VOC2012", "train.txt")
        test_samples = get_samples("VOC2007", "test.txt")
        val_samples = get_samples("VOC2012", "val.txt")

        # We only want images without a segmentation label
        def is_labelled_image(path):
            name = os.path.splitext(os.path.basename(path))[0]
            return (name in train_samples) or (name in val_samples) or (name in test_samples)

        # Finds all pascal voc images without a segmentation annotation
        unlabelled_samples = [
            img_path
            for img_path in glob.glob(
                os.path.join(voc_directory, "VOC2012/JPEGImages/**/*.jpg"),
                recursive=True,
            )
            if not is_labelled_image(img_path)
        ]

        # Count how many samples we have
        num_samples = len(unlabelled_samples)

        # Compute the size of each split (70/15/15)
        test_size = int(num_samples * 0.15)
        val_size = int(num_samples * 0.15)
        train_size = num_samples - val_size - test_size

        # Split samples into a training and validation splits
        train_samples, val_samples, test_samples = random_split(
            lengths=[train_size, val_size, test_size],
            dataset=unlabelled_samples,
            # We forcefully set a generator here with our random seed to ensure
            # a deterministic split of the unlabelled images.
            generator=torch.Generator().manual_seed(config.RANDOM_SEED),
        )

        self._datasets = {
            STAGE_TRAIN: train_samples,
            STAGE_VALIDATION: val_samples,
            STAGE_TEST: test_samples,
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
        return PascalVocUnlabelled._Dataset(self._datasets[stage])

    def train_dataloader(self):
        return self._get_dataloader(STAGE_TRAIN)

    def val_dataloader(self):
        return self._get_dataloader(STAGE_VALIDATION)

    def test_dataloader(self):
        return self._get_dataloader(STAGE_TEST)

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

        def __init__(self, samples: List[str]):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):

            # Get image path
            image_path = self.samples[idx]

            # Read the image
            image = load_image(image_path)

            return image, None
