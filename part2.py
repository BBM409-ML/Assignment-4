import os

import numpy as np
from glob import glob

import torch
from torch import nn
from torch import optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

from torchvision import transforms
from torchvision.models.vgg import vgg19_bn



def provide_determinism(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(Dataset):
    translate = {
        "cane": "dog",
        "cavallo": "horse",
        "elefante": "elephant",
        "farfalla": "butterfly",
        "gallina": "chicken",
        "gatto": "cat",
        "mucca": "cow",
        "pecora": "sheep",
        "ragno": "spider",
        "scoiattolo": "squirrel",
        "dog": "cane",
        "horse": "cavallo",
        "elephant" : "elefante",
        "butterfly": "farfalla",
        "chicken": "gallina",
        "cat": "gatto",
        "cow": "mucca",
        "sheep": "pecora",
        "spider": "ragno",
        "squirrel": "scoiattolo"
    }

    def __init__(self, data_parent='data/'):
        super().__init__()

        self.data_path = os.path.join(data_parent, 'raw-img')

        self.org_names = list(self.translate.keys())[:10]
        self.eng_names = list(self.translate.keys())[10:]

        self.data_paths = []
        for i in range(len(self.org_names)):
            img_paths = glob(
                os.path.join(self.data_path, self.org_names[i], '*')
            )

            self.data_paths.extend(list(zip(img_paths, [i] * len(img_paths))))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        img_path, label = self.data_paths[index]
        img = Image.open(img_path).convert('RGB')

        x = self.transform(img)

        return x, label

def train(train_set, test_set, only_train_fc=False):
    train_loader = DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
        num_workers=32,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        pin_memory=True,
        num_workers=32,
    )

    # VGG is 1000 class network, changing the last layer is a solution
    # but it would change weight distribution so I am not changing it.
    # Model should learn to output only the first 10 class.
    model = vgg19_bn(pretrained=True).to(device)

    if only_train_fc:
        for param in model.parameters():
            param.requires_grad = False

        # Train only last two linear layers
        for param in model.classifier[3].parameters():
            param.requires_grad = True
        for param in model.classifier[6].parameters():
            param.requires_grad = True

    lr = 0.0002 if only_train_fc else 0.0005
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    best_acc = -100
    for epoch in range(1, 11):
        model.train()

        train_correct = 0
        loss_acc = 0
        epoch_tqdm = tqdm(
            train_loader, total=len(train_loader), leave=True, desc="Train"
        )
        for images, labels in epoch_tqdm:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            output = model(images)
            output = output.squeeze(1)
            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            loss_acc += loss.item()

            output = output.argmax(dim=1)
            train_correct += sum(output==labels)

        optimizer.zero_grad(set_to_none=True)

        model.eval()

        test_correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                output = model(images)
                output = output.squeeze(1)
                output = output.argmax(dim=1)

                test_correct += sum(output==labels)

        test_acc = (float(test_correct) / len(test_set)) * 100

        if test_acc >= best_acc:
            torch.save(
                {'model': model},
                'vgg_' + ('fc' if only_train_fc else 'full') + '.pth'
            )

            best_acc = test_acc

        print(
            f"Epoch {str(epoch).zfill(3)} | "
            + f"Best Test Acc. = {best_acc:.2f}%"      # Best test. acc.
            + f", Last Test Acc. = {test_acc:.2f}%"     # Last test. acc.
        )

        scheduler.step()

    return model

def test(model, test_set):
    test_loader = DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        pin_memory=True,
        num_workers=32,
    )

    model.eval()

    conf_mat = torch.zeros(10, 10)
    test_correct = 0
    with torch.no_grad():
        epoch_tqdm = tqdm(
            test_loader, total=len(test_loader), leave=True, desc="Test"
        )
        for images, labels in epoch_tqdm:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            output = output.squeeze(1)
            output = output.argmax(dim=1)

            for t, p in zip(labels.view(-1), output.view(-1)):
                conf_mat[t.long(), p.long()] += 1

            test_correct += sum(output==labels)

    test_acc = (float(test_correct) / len(test_set)) * 100

    print(f"Test Acc. = {test_acc:.2f}%")

    print("Confusion matrix:")
    print(conf_mat)


if __name__ == "__main__":
    print("Device:", device, "\n")

    provide_determinism(123)

    dataset = Dataset()

    test_set_size = len(dataset)//10
    train_set_size = len(dataset)-test_set_size

    train_set, test_set = torch.utils.data.random_split(
        dataset, [train_set_size, test_set_size]
    )

    model_path = "models/vgg_full.pth"
    if model_path != "" or model_path != None:
        model = torch.load(model_path, map_location=torch.device('cpu'))['model']

        test(model, test_set)
    else:
        model_fc = train(train_set, test_set, only_train_fc=True)
        test(model_fc, test_set)

        model_full = train(train_set, test_set, only_train_fc=False)
        test(model_full, test_set)
