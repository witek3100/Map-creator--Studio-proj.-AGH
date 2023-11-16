from torchvision import transforms
from PIL import Image


train_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.8, 1.2], saturation=[0.8, 1.2]),
    transforms.Resize((224, 224), interpolation=Image.NEAREST),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224), interpolation=Image.NEAREST),
    transforms.ToTensor(),
])