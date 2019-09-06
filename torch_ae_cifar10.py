import os
import torch
import numpy as np
import torchvision
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torch.autograd import Variable
from alisuretool.Tools import Tools
import torchvision.transforms as transforms
import torch.utils.data as torch_utils_data


class AutoEncoder1(nn.Module):

    def __init__(self):
        super(AutoEncoder1, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            # nn.Conv2d(48, 96, 4, stride=2, padding=1),  # [batch, 96, 2, 2]
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            # nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    pass


class AutoEncoder2(nn.Module):

    def __init__(self):
        super(AutoEncoder2, self).__init__()
        # Input size: [batch, 3, 32, 32] Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),  # [batch, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # [batch, 128, 8, 8]
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # [batch, 256, 4, 4]
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # [batch, 512, 2, 2]
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # [batch, 512, 1, 1]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),  # [batch, 512, 2, 2]
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # [batch, 256, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # [batch, 128, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # [batch, 64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )
        pass

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    pass


class AutoEncoder(nn.Module):

    def __init__(self, low_dim=512):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),  # [batch, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # [batch, 128, 8, 8]
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # [batch, 256, 4, 4]
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # [batch, 512, 2, 2]
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2)  # [batch, 512, 1, 1]
        )

        self.linear = nn.Linear(512, low_dim)
        self.softmax = nn.Softmax(dim=-1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(low_dim, 512, 4, stride=2, padding=1),  # [batch, 512, 2, 2]
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # [batch, 256, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # [batch, 128, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # [batch, 64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )
        pass

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        linear = self.linear(encoded)
        softmax = self.softmax(linear)
        decoded = self.decoder(linear.view(linear.size(0), -1, 1, 1))
        return encoded, linear, softmax, decoded

    pass


class Data(object):

    @staticmethod
    def data(data_root, batch_size):
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        train_loader = torch_utils_data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        test_loader = torch_utils_data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return train_loader, test_loader, classes

    pass


class Runner(object):

    def __init__(self, data_root='./data', batch_size=32, low_dim=512, checkpoint_path="./checkpoint/ckpt.t7"):
        self.checkpoint_path = Tools.new_dir(checkpoint_path)

        self.auto_encoder = AutoEncoder(low_dim=low_dim).cuda()

        self.train_loader, self.test_loader, self.classes = Data.data(data_root, batch_size=batch_size)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.auto_encoder.parameters())

        pass

    def train(self, max_epoch=100):
        for epoch in range(max_epoch):
            running_loss = 0.0
            for i, (inputs, _) in enumerate(self.train_loader):
                inputs = Variable(inputs.cuda())

                encoded, linear, softmax, decoded = self.auto_encoder(inputs)
                loss = self.criterion(decoded, inputs)
                running_loss += loss.data

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pass
            Tools.print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(self.train_loader)))
            pass
        Tools.print('Finished Training')

        Tools.print('Saving Model...')
        torch.save(self.auto_encoder.state_dict(), self.checkpoint_path)
        pass

    def inference(self):
        Tools.print("Loading checkpoint...")
        self.auto_encoder.load_state_dict(torch.load(self.checkpoint_path))

        test_iter = iter(self.test_loader)
        images, labels = next(test_iter)

        print('GroundTruth: ', ' '.join('%5s' % self.classes[labels[j]] for j in range(16)))
        show_data_1 = np.asarray(np.transpose(torchvision.utils.make_grid(images).cpu().numpy(),
                                              (1, 2, 0)) * 255, np.uint8)
        shape = show_data_1.shape

        images = Variable(images.cuda())
        encoded, linear, softmax, decoded_img = self.auto_encoder(images)
        show_data_2 = np.asarray(np.transpose(torchvision.utils.make_grid(decoded_img.data).cpu().numpy(),
                                              (1, 2, 0)) * 255, np.uint8)

        padding = 5
        show = np.zeros(shape=(shape[0] + padding * 2, shape[1] * 2 + padding * 3, shape[2]), dtype=np.uint8)
        show[padding:-padding, padding:padding + shape[1], :] = show_data_1
        show[padding:-padding, padding * 2 + shape[1]:-padding, :] = show_data_2
        Image.fromarray(show).show()
        pass

    def print_model(self):
        Tools.print()
        Tools.print("============== Encoder ==============")
        print(self.auto_encoder.encoder)
        Tools.print("============== Decoder ==============")
        print(self.auto_encoder.decoder)
        Tools.print()
        pass

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    _data_root = './data'
    _batch_size = 64
    _max_epoch = 100
    _checkpoint_path = "./checkpoint/ckpt.t7"

    runner = Runner(data_root=_data_root, batch_size=_batch_size, checkpoint_path=_checkpoint_path)
    runner.print_model()
    runner.train(max_epoch=_max_epoch)
    runner.inference()
