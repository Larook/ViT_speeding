# CODE FROM https://towardsdatascience.com/a-demonstration-of-using-vision-transformers-in-pytorch-mnist-handwritten-digit-recognition-407eafbc15b0

import os
import time

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn, optim
import einops
from torchvision import datasets, transforms, models

import vit_pytorch



class ViT_classification(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cpu')
        # self.device = torch.device('cuda:0')


        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = vit_pytorch.Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

    def load_split_train_test(self, data_dir, valid_size=.2):
        train_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.Grayscale(num_output_channels=1),
                                               transforms.RandomVerticalFlip(),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               ])

        test_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                              transforms.Grayscale(num_output_channels=1),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              ])

        train_data = datasets.ImageFolder(data_dir, transform=train_transforms)
        test_data = datasets.ImageFolder(data_dir, transform=test_transforms)


        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)

        from torch.utils.data.sampler import SubsetRandomSampler

        train_idx, test_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        self.trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=64)
        self.testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=64)



def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()

        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())


def evaluate(model, data_loader, loss_history):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')


if __name__ == "__main__":

    torch.manual_seed(42)

    DOWNLOAD_PATH = os.getcwd() + '/data/mnist'
    BATCH_SIZE_TRAIN = 10 # 100
    BATCH_SIZE_TEST = 100 # 1000

    # transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    # train_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True,
    #                                        transform=transform_mnist)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    #
    # test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True,
    #                                       transform=transform_mnist)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

    N_EPOCHS = 1 #25

    start_time = time.time()
    model = ViT_classification(image_size=256, patch_size=8, num_classes=6, channels=1,
                               dim=64, depth=1, heads=2, mlp_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # training function
    data_dir = os.getcwd() + '/data/try_smart_fast_p50'
    model.load_split_train_test(data_dir)

    train_loss_history, test_loss_history = [], []
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        # train_epoch(model, optimizer, train_loader, train_loss_history)
        # evaluate(model, test_loader, test_loss_history)

        for inputs, labels in model.trainloader:
            # steps += 1
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            print('labels', labels)
            # translate labels to regression task
            # labels_to_target = dict(east=1, grasp=2, north=3, release=4, south=5, west=6)
            # target = labels_to_target[labels]

            print('labels', labels)

            optimizer.zero_grad()
            # TODO: Check the sizes to find whats wrong
            output = F.log_softmax(model(inputs), dim=1)
            print('output', output)

            loss = F.nll_loss(output, labels)
            print('loss', loss)
            loss.backward()
            optimizer.step()

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')