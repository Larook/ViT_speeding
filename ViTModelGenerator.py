# CODE FROM https://towardsdatascience.com/a-demonstration-of-using-vision-transformers-in-pytorch-mnist-handwritten-digit-recognition-407eafbc15b0

import os
import pprint
import time

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from einops import rearrange
from torch import nn, optim
import einops
from tqdm import tqdm

from SimulationData import SimulationData
from SimulationImageDataset import SimulationImageDataset
from model_training import vit_pytorch


class ViTRegression(nn.Module):
    def __init__(self, wandb_config, *, image_size, patch_size, num_outputs, dim, depth, heads, mlp_dim, channels=3):
        torch.manual_seed(42)

        super().__init__()

        self.wandb_config = wandb_config
        print('self.wandb_config', self.wandb_config)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('cpu')
        # self.device = torch.device(wandb_config['device'])

        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)).to(self.device)
        self.patch_to_embedding = nn.Linear(patch_dim, dim).to(self.device)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)).to(self.device)
        self.transformer = vit_pytorch.Transformer(dim, depth, heads, mlp_dim).to(self.device)

        self.to_cls_token = nn.Identity().to(self.device)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_outputs)
        ).to(self.device)
        # self.optimizer = optim.Adam(self.parameters(), lr=self.wandb_config['learning_rate'])  #0.003
        print("self.wandb_config")
        pprint.pprint(self.wandb_config)
        self.optimizer = self.build_optimizer(self.wandb_config['optimizer'])
        self.criterion = F.smooth_l1_loss

    def build_optimizer(self, optimizer):
        if optimizer == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.wandb_config['learning_rate'], momentum=0.9)
        elif optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.wandb_config['learning_rate'])
        return optimizer

    def load_dataloaders(self, pickle_df_path):
        """ reads the pickles from the directory and updates dfs, SimulationData and dataloaders """
        # load training data
        self.data = SimulationData(create=False)
        self.train_df, self.test_df = self.data.load_dfs_from_pickles(create=False, training_percentage=0.8,
                                                                      pickle_df_path=pickle_df_path, shuffle=True)

        train_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.Grayscale(num_output_channels=1),
                                               # transforms.RandomVerticalFlip(),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor()
                                               ])

        dataset_train = SimulationImageDataset(main_df=self.train_df, transform=train_transforms)
        dataset_test = SimulationImageDataset(main_df=self.test_df, transform=train_transforms)

        # create dataloaders
        self.train_dataloader = DataLoader(dataset_train, batch_size=self.wandb_config['batch_size'], shuffle=True)
        self.test_dataloader = DataLoader(dataset_test, batch_size=self.wandb_config['batch_size'], shuffle=True)
        pass

    # def load_split_train_test(self, train_data, test_data, valid_size=.2):
    #     train_transforms = transforms.Compose([transforms.Resize((256, 256)),
    #                                            transforms.Grayscale(num_output_channels=1),
    #                                            transforms.RandomVerticalFlip(),
    #                                            transforms.RandomHorizontalFlip(),
    #                                            transforms.ToTensor(),
    #                                            ])
    #
    #     test_transforms = transforms.Compose([transforms.Resize((256, 256)),
    #                                           transforms.Grayscale(num_output_channels=1),
    #                                           transforms.RandomVerticalFlip(),
    #                                           transforms.RandomHorizontalFlip(),
    #                                           transforms.ToTensor(),
    #                                           ])
    #
    #     # train_data = datasets.ImageFolder(data_dir, transform=train_transforms)
    #     # test_data = datasets.ImageFolder(data_dir, transform=test_transforms)
    #
    #
    #     num_train = len(train_data)
    #     indices = list(range(num_train))
    #     split = int(np.floor(valid_size * num_train))
    #     np.random.shuffle(indices)
    #
    #     from torch.utils.data.sampler import SubsetRandomSampler
    #
    #     train_idx, test_idx = indices[split:], indices[:split]
    #     train_sampler = SubsetRandomSampler(train_idx)
    #     test_sampler = SubsetRandomSampler(test_idx)
    #
    #     self.trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=64)
    #     self.testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=64)


    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p).to(self.device)  # height weight channel batch
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        # print('cls_tokens', cls_tokens)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        # print('x', x)
        return self.mlp_head(x)

    # def load_simulation_data(self, main_df, shuffle=True):
    #     # row = dict(time=time.time()-self.start_time, steering_angle=angle, velocity_y=v_y, image=img)
    #     main_df = main_df.sample(frac=1)
    #
    #     # inputs are just images or maybe images + time
    #     # outputs should be steering angle + velocity
    #
    #     self.inputs = torch.tensor(main_df['image'].values)
    #     self.outputs = torch.tensor(main_df['steering_angle'].values)
    #     pass

    def train_epochs(self):

        def train_log(loss, example_ct, epoch, train=True):
            # Where the magic happens
            if train:
                wandb.log({"epoch": epoch, "loss_train": loss, 'step_train': example_ct}, step=example_ct)
            else:
                wandb.log({"epoch": epoch, "loss_test": loss, 'step_test': example_ct}, step=example_ct)
            # print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

        with wandb.init(config=self.wandb_config):
            config = wandb.config
            print("train_epochs, config", config)
            # pprint(config)

            self.max_epochs = config['epochs']
            self.loss_train_history = []
            self.loss_test_history = []
            example_ct_train = 0  # number of examples seen
            batch_ct_train = 0

            example_ct_test = 0  # number of examples seen
            batch_ct_test = 0

            # total_samples = len(self.train_dataloader.dataset)
            wandb.watch(self, self.criterion, log="all", log_freq=1000)

            # Loop over epochs
            for epoch in tqdm(range(self.max_epochs)):
                running_loss_train = []
                running_loss_test = []

                self.train()
                for i, [imgs, angles, vels] in enumerate(self.train_dataloader):
                    imgs, angles, vels = imgs.to(self.device), angles.to(self.device), vels.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.forward(imgs)
                    target = angles.unsqueeze(1).float()
                    loss = self.criterion(output, target)  # L1 loss for regression applications
                    loss.backward()
                    self.optimizer.step()
                    running_loss_train.append(loss.item())

                    example_ct_train += len(imgs)
                    batch_ct_train += 1
                    # train_log(loss.item(), example_ct_train, epoch, train=True)

                self.loss_train_history.append(np.mean(running_loss_train))

                self.eval()
                for j, [imgs, angles, vels] in enumerate(self.test_dataloader):
                    imgs, angles, vels = imgs.to(self.device), angles.to(self.device), vels.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.forward(imgs)
                    target = angles.unsqueeze(1).float()
                    loss = F.smooth_l1_loss(output, target)  # L1 loss for regression applications
                    # loss.backward()
                    # self.optimizer.step()
                    running_loss_test.append(loss.item())
                    example_ct_test += len(imgs)
                    batch_ct_test += 1
                    # train_log(loss.item(), example_ct_test, epoch, train=False)

                wandb.log({"epoch": epoch, "avg_loss_train": np.mean(running_loss_train),
                           "avg_loss_test": np.mean(running_loss_test)}, step=epoch)

                # wandb.log({"loss": np.mean(running_loss_test), "epoch": epoch})
                self.loss_test_history.append(np.mean(running_loss_test))

        self.save_training_model_statistics()

        model_save_path = 'model_training/trained_models/model_' + str(self.max_epochs) + '.pth'
        torch.save(self.state_dict(), model_save_path)

    def save_training_model_statistics(self):

        float_formatter = "{:.4f}".format

        final_loss_train = float_formatter(self.loss_train_history[-1])
        min_loss_train = float_formatter(np.min(self.loss_train_history))

        final_loss_test = float_formatter(self.loss_test_history[-1])
        min_loss_test = float_formatter(np.min(self.loss_test_history))

        self.training_model_statistics_dict = dict(final_loss_train=final_loss_train, min_loss_train=min_loss_train,
                                                   final_loss_test=final_loss_test, min_loss_test=min_loss_test)

    def plot_training_history(self):
        plotname = 'model_training/training_plots/model_' + str(self.max_epochs) + '.png'
        f = plt.figure(figsize=(9, 4.8))

        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Check losses for overfitting")
        plt.suptitle(str(self.training_model_statistics_dict), fontsize=9)
        plt.plot(self.loss_train_history, label='loss_train_history')
        plt.plot(self.loss_test_history, label='loss_test_history')
        plt.legend()
        plt.savefig(plotname)
        plt.show()

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

    DOWNLOAD_PATH = os.getcwd() + '/data/mnist'
    BATCH_SIZE_TRAIN = 10 # 100
    BATCH_SIZE_TEST = 100 # 1000

    N_EPOCHS = 5 #25

    start_time = time.time()
    model = ViTRegression(image_size=256, patch_size=8, num_outputs=1, channels=1,
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
            target = labels * 10

            print('target', target)

            optimizer.zero_grad()
            # TODO: Check the sizes to find whats wrong
            output = model.forward(inputs)
            print('output', output)

            loss = F.smooth_l1_loss(output, target.float())  # L1 loss for regression applications
            loss.backward()
            print('loss', loss)
            optimizer.step()

    torch.save(model, 'KEYVIBE_model.pth')
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')