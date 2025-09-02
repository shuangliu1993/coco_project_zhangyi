import torch

from coco.dataset.classification_dataset import COCOClassificationDataset, get_dataloader
from coco.model.classification_model import COCOClassificationModel

DATA_DIR = "C:\\Users\\61556\\Downloads\\data\\coco\\"
BATCH_SIZE = 16
NUM_EPOCHS = 5


def train():
    # initialize dataloader
    dataset = COCOClassificationDataset(data_dir=DATA_DIR, mode="train")
    dataloader = get_dataloader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # initialize model
    model = COCOClassificationModel(num_classes=len(dataset.ids_to_names))

    # initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # training epoch (https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop)


if __name__ == "__main__":
    train()
