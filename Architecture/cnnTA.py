import torch
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

imageList = np.load("Images.npy").astype(np.float64)
labelList = np.load("Labels.npy").astype(np.int64)

imageList_copy = imageList[:]
imageList_copy = imageList_copy.reshape(4960, -1)

mean = np.mean(imageList_copy, axis=0)
std = np.std(imageList_copy, axis=0)

imageList_copy = (imageList_copy - mean) / std
imageList = imageList_copy.reshape(4960, 11, 11, 1)
class CustomDataset(Dataset):
    """ Custom dataset for flattened 11x11 csv dataset """

    # Initialize data
    def __init__(self, labelList, imageList, transform=None):
        self.transform = transform
        self.labelList = labelList
        self.imageList = imageList

    def __getitem__(self, index):
        x = self.imageList[index]
        y = self.labelList[index]
        y = torch.as_tensor(y, dtype=torch.long)
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return self.labelList.shape[0]

transform = transforms.Compose([
    transforms.ToTensor(),
])

## Configure the hyperparameters
# torch parameters
SEED = 60  # reproducability
# NN Parameters
EPOCHS = 200  # number of epochs
LR = 0.001  # learning rate
MOMENTUM = 0.9  # momentum for the SGD optimizer (how much of the past gradients)
GAMMA = 0.1  # learning rate scheduler (how much to decrease learning rate)
BATCH_SIZE = 64  # number of images to load per iteration

num_classes = 3
input_shape = (11, 11, 1)

x_train, x_test, y_train, y_test = train_test_split(imageList, labelList, test_size=0.1, random_state=100)

train_dataset = CustomDataset(y_train, x_train, transform=transform)
test_dataset = CustomDataset(y_test, x_test, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()

        ## [(W - Kernelw + 2*padding)/stride] + 1
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2,
                                 stride=2)

        self.dropout1 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(in_features=64 * 5 * 5,  ## 64 * 7 * 7
                             out_features=128)

        self.dropout2 = nn.Dropout(0.50)

        self.fc2 = nn.Linear(in_features=128,
                             out_features=3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
net = Conv().to(device, dtype=torch.double)

loss_fn = nn.CrossEntropyLoss()
# specify the optimizer to update the weights during backward pass
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
# change learning rate over time
scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=GAMMA)

def train_net():
    # put the network in training mode
    net.train()
    # keep record of the loss value
    epoch_loss = 0.0
    # use training data as batches
    for xt, rt in train_loader:
        # move training instances and corresponding labels into gpu if cuda is available
        xt, rt = xt.to(device), rt.to(device)

        # clear the previously accumulated gradients
        optimizer.zero_grad()
        # forward the network
        yt = net(xt)
        # calculate loss
        rt = rt.squeeze_()
        loss = loss_fn(yt, rt)

        # make a backward pass, calculate gradients
        loss.backward()
        # update weights
        optimizer.step()
        # accumulate loss
        epoch_loss += loss.item()
    return epoch_loss

## Define test function
def eval_net(loader):
    # put the network in evaluation mode
    net.eval()
    # keep record of the loss value
    total_loss = 0.0
    # number of correctly classified instances
    correct = 0
    # disable gradient tracking
    with torch.no_grad():
        for xt, rt in loader:
            xt, rt = xt.to(device), rt.to(device)

            yt = net(xt)
            # calculate loss
            rt = rt.squeeze_()
            loss = loss_fn(yt, rt)
            # accumulate loss
            total_loss += loss.item()
            # get predicted classes
            pred = yt.argmax(dim=1)
            # accumulate correctly classified image counts
            correct += (pred == rt).sum().item()
            # correct += pred.eq(rt.view_as(pred)).sum().item()
    return correct / len(loader.dataset), total_loss

writer = SummaryWriter()
# train the network
for epoch in range(1, EPOCHS + 1):
    # train network for one epoch
    train_net()
    # get accuracy and loss on the training dataset
    tr_ac, tr_loss = eval_net(train_loader)
    # get accuracy and loss on the test dataset
    tt_ac, tt_loss = eval_net(test_loader)
    # save stats
    writer.add_scalars("Loss", {"tr_loss": tr_loss, "tt_loss": tt_loss}, epoch)
    writer.add_scalars("Accuracy", {"tr_acc": tr_ac, "tt_acc": tt_ac}, epoch)

    if (epoch - 1) % 10 == 0:
        print("Epoch", epoch, "Tr Acc:", tr_ac, "Tt_Ac", tt_ac)

    writer.flush()
writer.close()