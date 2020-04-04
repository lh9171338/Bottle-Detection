import os
import glob
import numpy as np
from datetime import datetime
import cv2
import torch
import torch.nn as nn
import torch.utils.data as Data
from tensorboardX import SummaryWriter


# Load data
class BottleDataset(Data.Dataset):
    def __init__(self, path, pattern, augment=None):
        self.file_list = glob.glob(os.path.join(path, pattern))
        self.augment = augment

    def __getitem__(self, index):
        filename = self.file_list[index]
        image = cv2.imread(filename)
        image = cv2.resize(image, (64, 64))
        image = np.transpose(image, [2, 0, 1])
        label = np.array([0])
        if filename.find('Pos') >= 0:
            label = np.array([1])
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        return image, label

    def __len__(self):
        return len(self.file_list)


# Define network
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Sequential(  # input shape (1, 64, 64)
            nn.Conv2d(3, 12, 5, padding=2),
            nn.ReLU(),              # activation
            nn.MaxPool2d(2),        # output shape (12, 32, 32)
            nn.Conv2d(12, 48, 5, padding=2),
            nn.ReLU(),              # activation
            nn.MaxPool2d(2),        # output shape (48, 16, 16)
        )
        self.fc = nn.Linear(48 * 16 * 16, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y


def train_net(train_path, pattern, model_filename, log_path, batch_size, num_epochs, step_size, lr, device):
    # Create model
    model = Model().to(device)
    print(model)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)
    loss_func = torch.nn.BCEWithLogitsLoss()

    # Summary
    writer = SummaryWriter(log_path)
    model_input = torch.rand(batch_size, 3, 64, 64).to(device)
    writer.add_graph(model, (model_input,))

    # Creating data loader
    dataset = BottleDataset(train_path, pattern)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    total_correct = 0
    total_loss = 0
    total_num = 0
    step = 0
    for epoch in range(num_epochs):
        for image_batch, label_batch in loader:
            image_batch = torch.FloatTensor(image_batch).to(device)
            label_batch = torch.FloatTensor(label_batch).to(device)
            outputs = model(image_batch)
            outputs = torch.squeeze(outputs, -1)
            labels = torch.squeeze(label_batch, -1)

            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logical_outputs = outputs.cpu().detach().numpy() > 0
            logical_labels = labels.cpu().detach().numpy() > 0.5
            total_correct += np.sum(logical_outputs == logical_labels)
            total_loss += loss.cpu().detach().numpy()
            total_num += logical_labels.shape[0]
            if step % 50 == 0:
                accuracy = float(total_correct) / float(total_num)
                avg_loss = float(total_loss) / float(total_num)
                total_correct = 0
                total_loss = 0
                total_num = 0
                print('Epoch: ', epoch + 1, '| train loss: %.6f' % avg_loss, '| train accuracy: %.4f' % accuracy)

                writer.add_images('image', image_batch[0:3], step)
                writer.add_scalar('loss', avg_loss, step)
                writer.add_scalar('accuracy', accuracy, step)
            step += 1

        scheduler.step()

    writer.close()

    # Save model
    torch.save(model.cpu(), model_filename)


def test_net(test_path, pattern, model_filename, batch_size, device):
    # Load model
    model = torch.load(model_filename).to(device)
    print(model)

    # Creating data loader
    dataset = BottleDataset(test_path, pattern)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False
    )

    pos_error = 0
    neg_error = 0
    total_num = 0
    for image_batch, label_batch in loader:
        image_batch = torch.FloatTensor(image_batch).to(device)
        label_batch = torch.FloatTensor(label_batch).to(device)
        outputs = model(image_batch)
        outputs = torch.squeeze(outputs, -1)
        labels = torch.squeeze(label_batch, -1)

        logical_outputs = outputs.cpu().detach().numpy() > 0
        logical_labels = labels.cpu().detach().numpy() > 0.5
        pos_error += np.sum(np.logical_and(logical_outputs != logical_labels, logical_labels == True))
        neg_error += np.sum(np.logical_and(logical_outputs != logical_labels, logical_labels == False))
        total_num += logical_labels.shape[0]

    total_error = pos_error + neg_error
    accuracy = 1 - float(total_error) / float(total_num)
    print('positive error: ', pos_error)
    print('negative error: ', neg_error)
    print('accuracy: %.4f' % accuracy)


if __name__ == '__main__':
    # Parameter
    train_path = '../Dataset/train'
    test_path = '../Dataset/test'
    pattern = '*.jpg'
    model_filename = '../Model/bottleNet.pkl'
    log_path = 'log/{}'.format(datetime.now().strftime("%Y%m%d-%H%M"))
    batch_size = 128
    num_epochs = 10
    step_size = 5
    lr = 1e-4
    train_flag = True
    test_flag = True

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print('use_gpu: ', use_gpu)

    # Training network
    if train_flag:
        train_net(train_path, pattern, model_filename, log_path, batch_size, num_epochs, step_size, lr, device)

    # Test
    if test_flag:
        test_net(test_path, pattern, model_filename, batch_size, device)
