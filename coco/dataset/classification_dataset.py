import os
import json

import torch
from PIL import Image

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from transformers import AutoProcessor

image_processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")


class COCOClassificationDataset(Dataset):
    def __init__(self, data_dir, mode="train"):
        super().__init__()
        """
        Data format
        data_dir
           |
           |----annotations
           |
           |----train2017
           |
           |----val2017
        """
        # get annotation file path
        if mode == "train":
            self.image_dir = os.path.join(data_dir, "train2017")
            annotation_file = os.path.join(data_dir, "annotations", "instances_train2017.json")
        elif mode == "test":
            self.image_dir = os.path.join(data_dir, "val2017")
            annotation_file = os.path.join(data_dir, "annotations", "instances_val2017.json")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # load annotations
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)

        # load id to image name mapping
        self.ids_to_names = {e['id']: e['name'] for e in self.annotations["categories"]}
        self.ids_to_image_paths = {e['id']: e['file_name'] for e in self.annotations["images"]}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        class_id = self.annotations["annotations"][idx]["category_id"]
        image_path = os.path.join(self.image_dir, self.ids_to_image_paths[self.annotations["annotations"][idx]["image_id"]])
        image = Image.open(image_path)
        return image, class_id


def collate_fn(batch):
    """convert raw data to pytorch tensors"""
    images, class_ids = zip(*batch)
    image_tensor = image_processor(images=images, return_tensors="pt")["pixel_values"]
    labels = torch.tensor(class_ids, dtype=torch.long)
    return image_tensor, labels


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=0, drop_last=True):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, collate_fn=collate_fn)


if __name__ == "__main__":
    # dataset = COCOClassificationDataset(data_dir="C:/Users/22969/OneDrive/文档/application/大语言模型科研/data", mode="test")
    dataset = COCOClassificationDataset(data_dir="C:\\Users\\61556\\Downloads\\data\\coco\\", mode="test")
    image, class_id = dataset[0]
    print(f"image: {image}")
    print(f"class_id: {class_id}")

    dataloader = get_dataloader(dataset, batch_size=4, num_workers=0)
    for image_tensor, labels in dataloader:
        print(f"image_tensor shape: {image_tensor.shape}")
        print(f"labels shape: {labels.shape}")
        print(f"labels: {labels}")
        break
