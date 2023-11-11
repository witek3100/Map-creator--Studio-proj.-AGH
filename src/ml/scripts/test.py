import os
import torch
from ..models.model import Model
from ..datasets import dataset, transforms
from torch.utils.data import DataLoader
from config import BASE_DIR
from ..engines.metrics import accuracy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model()
model = model.load_state_dict(torch.load('model/model.pth', map_location=torch.device(DEVICE)))

criterion = torch.nn.BCELoss()

test_df = os.path.join(BASE_DIR, 'data/processed/images/test.csv')
test_loader = DataLoader(
    dataset.Dataset(test_df, transforms.test_val_transforms),
    batch_size=64,
    shuffle=True,
)


def test():
    test_loss = 0.0
    test_acc = 0.0

    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        out = model(images)

        labels = labels.unsqueeze(1)
        labels = labels.float()

        loss = criterion(out, labels)

        test_loss += loss.item()
        test_acc += accuracy(out, labels)

    test_loss = test_loss / len(test_loader)
    test_acc = test_acc / len(test_loader)

    print("TEST")
    print(f'Loss: {test_loss}')
    print(f'Accuracy: {test_acc}')

if __name__ == '__main__':
    test()