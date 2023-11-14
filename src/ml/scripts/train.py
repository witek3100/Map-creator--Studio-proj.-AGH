import os
import sys
import torch
from models.model1 import Model
from datasets import dataset, transforms
from utils.metrics import accuracy
from utils.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from config import BASE_DIR


def train(model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training using: {device}')

    model = model.to(device)

    NUM_EPOCHS = 15
    BATCH_SIZE = 64

    train_df = os.path.join(BASE_DIR, 'data/processed/images/train.csv')
    train_loader = DataLoader(
        dataset.Dataset(train_df, transforms.train_transforms),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_df = os.path.join(BASE_DIR, 'data/processed/images/validation.csv')
    val_loader = DataLoader(
        dataset.Dataset(val_df, transforms.test_val_transforms),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    es = EarlyStopping(2, 0.01)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(NUM_EPOCHS):
        print('\n ----------------------------------')
        print(f'EPOCH: {epoch + 1}/{NUM_EPOCHS}')

        train_loss = 0.0
        train_acc = 0.0
        validation_loss = 0.0
        val_acc = 0.0

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(device)

            optimizer.zero_grad()
            out = model(images)
            labels = labels.unsqueeze(1)
            labels = labels.float()

            loss = criterion(out, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy(out, labels)

        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        print(f'train loss: {round(train_loss, 3)}')

        train_acc = train_acc / len(train_loader)
        train_accs.append(train_acc)
        print(f'train accuracy: {round(train_acc, 3)}\n')

        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                out = model(images)

                labels = labels.unsqueeze(1)
                labels = labels.float()

                loss = criterion(out, labels)

                validation_loss += loss.item()
                val_acc += accuracy(out, labels)

        validation_loss = validation_loss / len(val_loader)
        val_losses.append(validation_loss)
        print(f'validation loss: {round(validation_loss, 3)}')

        val_acc = val_acc / len(val_loader)
        val_accs.append(val_acc)
        print(f'validation accuracy: {round(val_acc, 3)}')

        if es.early_stop(validation_loss):
            print('validation loss is not decresing - training stopped')
            break

    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'src/ml/models/model.pth'))

if __name__ == '__main__':

    try:
        model_name = sys.argv[1]
        try:
            model = Model()
            model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'src/ml/models/model1.pth'), map_location=torch.device('cpu')))
        except FileNotFoundError:
            print('Model not found')
            sys.exit()
    except IndexError:
        print('No model name provided')
        sys.exit()

    train(model)
