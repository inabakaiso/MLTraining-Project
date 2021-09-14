import torch

import numpy as np
from PIL import Image
from PIL import ImageFile

class ClassficationDataset:
    """
    二値分類
    """
    def __init__(self, image_path, targets, resize=None, augmentations=None):
        self.image_path = image_path
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        """
        指定されたindexに対してmodelの学習に必要なすべての要素を返す
        :param item:
        :return:
        """
        image = Image.open(self.image_path[item])
        image = image.convert("RGB")
        targets = self.targets[item]
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[2]),
                resample=Image.BILINEAR
            )
        image = np.array(image)
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        ## Pytorchで期待される形に変換 (チャンネル, 高さ, 幅)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        ## 回帰問題の場合はtorch.float, 目的変数は一列なのでtorch.long
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long)
        }
