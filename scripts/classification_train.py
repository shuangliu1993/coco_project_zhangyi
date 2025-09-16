import torch

from coco.dataset.classification_dataset import COCOClassificationDataset, get_dataloader
from coco.model.classification_model import COCOClassificationModel

from tqdm import tqdm

DATA_DIR = "C:\\Users\\61556\\Downloads\\data\\coco\\"
BATCH_SIZE = 16
NUM_EPOCHS = 5


def train_one_epoch(dataloader, model, optimizer, loss_fn, epoch_index):
    running_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(tqdm(dataloader)):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        # if i % 10 == 0:
        last_loss = running_loss / 10  # loss per batch
        print('  epoch {}, step {}, loss: {}'.format(epoch_index, i + 1, last_loss))
        running_loss = 0.0
    return


def train():
    # initialize dataloader
    dataset = COCOClassificationDataset(data_dir=DATA_DIR, mode="test")
    dataloader = get_dataloader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # initialize model
    model = COCOClassificationModel(num_classes=len(dataset.names_to_ids))

    # initialize loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # training epoch (https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop)
    for epoch in range(NUM_EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_one_epoch(dataloader, model, optimizer, loss_fn, epoch)

        # save the model's state
        model_path = 'ckpt_{}.pt'.format(epoch)
        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    train()
