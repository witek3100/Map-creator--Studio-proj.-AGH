import os
import torch
import sys
import pandas as pd
import importlib
from datasets import dataset, transforms
from torch.utils.data import DataLoader
from config import BASE_DIR
from utils.metrics import accuracy


def test(model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'testing using {device}')

    criterion = torch.nn.BCELoss()

    test_df = pd.read_csv(os.path.join(BASE_DIR, 'data/processed/test.csv'))
    test_loader = DataLoader(
        dataset.Dataset(test_df, transforms.test_val_transforms),
        batch_size=64,
        shuffle=True,
    )

    test_loss = 0.0
    test_acc = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            out = model(images)

            labels = labels.unsqueeze(1)
            labels = labels.float()

            loss = criterion(out, labels)

            test_loss += loss.item()
            test_acc += accuracy(out, labels)

    test_loss = test_loss / len(test_loader)
    test_acc = test_acc / len(test_loader)

    print("TEST")
    print(f'Loss: {round(test_loss, 5)}')
    print(f'Accuracy: {round(test_acc, 5)}')

if __name__ == '__main__':

    try:
        model_name = sys.argv[1]
        try:
            module = importlib.import_module(f'models.{model_name}')
            model = module.Model()
            model.load_state_dict(torch.load(os.path.join(BASE_DIR, f'src/ml/models/{model_name}.pth'), map_location=torch.device('cpu')))
            model.eval()
        except (FileNotFoundError, ModuleNotFoundError):
            raise Exception('Model not found')
            sys.exit()

    except IndexError:
        raise Exception('No model name provided')
        sys.exit()

    test(model)