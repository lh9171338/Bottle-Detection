import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.data as Data

# Parameter
trainPath = '../Dataset/train'
testPath = '../Dataset/test'
modelFilename = '../Model/bottleNet.pkl'
batchSize = 128
numEpochs = 10
stepSize = 5
LR = 1e-4
trainFlag = False
testFlag = True


# Load data
class BottleDataset(Data.Dataset):
    def __init__(self, root, augment=None):
        self.fileList = glob.glob(os.path.join(root, '*.jpg'))
        self.augment = augment

    def __getitem__(self, index):
        filename = self.fileList[index]
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
        return len(self.fileList)


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


if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    print('use_gpu: ', use_gpu)

    # Training network
    if trainFlag:
        # Create model
        model = Model()
        if use_gpu:
            model = model.cuda()
        print(model)
        # optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, alpha=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, stepSize, gamma=0.1)
        lossFunc = torch.nn.BCEWithLogitsLoss()

        # Creating data loader
        trainDataset = BottleDataset(trainPath)
        trainLoader = Data.DataLoader(
            dataset=trainDataset,
            batch_size=batchSize,
            shuffle=True
        )

        totalCorrect = 0
        totalLoss = 0
        totalNum = 0
        for epoch in range(numEpochs):
            for step, (batchX, batchY) in enumerate(trainLoader):
                batchX = torch.FloatTensor(batchX)
                batchY = torch.FloatTensor(batchY)
                if use_gpu:
                    batchX = batchX.cuda()
                    batchY = batchY.cuda()
                output = model(batchX)
                output = torch.squeeze(output, -1)
                batchY = torch.squeeze(batchY, -1)
                loss = lossFunc(output, batchY)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predY = output.cpu().detach().numpy() > 0
                trueY = batchY.cpu().detach().numpy() > 0.5
                totalCorrect += np.sum(predY == trueY)
                totalLoss += loss.cpu().detach().numpy()
                totalNum += batchX.shape[0]
                if step % 50 == 0:
                    accuracy = float(totalCorrect) / float(totalNum)
                    avgLoss = float(totalLoss) / float(totalNum)
                    totalCorrect = 0
                    totalLoss = 0
                    totalNum = 0
                    print('Epoch: ', epoch + 1, '| train loss: %.6f' % avgLoss, '| train accuracy: %.4f' % accuracy)
            scheduler.step()

        # Save model
        torch.save(model.cpu(), modelFilename)

    # Test
    if testFlag:
        # Load model
        model = torch.load(modelFilename)
        if use_gpu:
            model = model.cuda()
        print(model)

        # Creating data loader
        testDataset = BottleDataset(testPath)
        testLoader = Data.DataLoader(
            dataset=testDataset,
            batch_size=batchSize,
            shuffle=False
        )

        posError = 0
        negError = 0
        totalError = 0
        totalNum = 0
        for batchX, batchY in testLoader:
            batchX = torch.FloatTensor(batchX)
            batchY = torch.FloatTensor(batchY)
            if use_gpu:
                batchX = batchX.cuda()
                batchY = batchY.cuda()
            output = model(batchX)
            output = torch.squeeze(output, -1)
            batchY = torch.squeeze(batchY, -1)
            predY = output.cpu().detach().numpy() > 0
            trueY = batchY.cpu().detach().numpy() > 0.5
            posError += np.sum(np.logical_and(predY != trueY, trueY == True))
            negError += np.sum(np.logical_and(predY != trueY, trueY == False))
            totalNum += batchX.shape[0]

        totalError = posError + negError
        accuracy = 1 - float(totalError) / float(totalNum)
        print('posError: ', posError)
        print('negError: ', negError)
        print('accuracy: %.4f' % accuracy)
