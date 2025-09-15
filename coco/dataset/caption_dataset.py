import os
import json

from PIL import Image
from torch.utils.data.dataset import Dataset


class CustomImageDataset(Dataset):
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
            annotation_file = os.path.join(data_dir, "annotations", "captions_train2017.json")
        elif mode == "test":
            self.image_dir = os.path.join(data_dir, "val2017")
            annotation_file = os.path.join(data_dir, "annotations", "captions_val2017.json")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # load annotations
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)

        # load id to image name mapping
        self.ids_to_image_paths = {e['id']: e['file_name'] for e in self.annotations["images"]}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        caption = self.annotations["annotations"][idx]["caption"]
        image_path = os.path.join(self.image_dir, self.ids_to_image_paths[self.annotations["annotations"][idx]["image_id"]])
        image = Image.open(image_path)
        return image, caption


if __name__ == "__main__":
    dataset = CustomImageDataset(data_dir="C:/Users/61556/Downloads/data/coco", mode="test")
    image, caption = dataset[0]
    print(f"image: {image}")
    print(f"caption: {caption}")
