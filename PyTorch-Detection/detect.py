import os
import glob
import time
import numpy as np
import cv2
import torch
import torch.nn as nn

ImageWidth = 64
ImageHeight = 64


# Define network
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Sequential(  # input shape (3, 64, 64)
            nn.Conv2d(3, 12, 5),    # output shape (12, 60, 60)
            nn.ReLU(),              # activation
            nn.MaxPool2d(2),        # output shape (12, 30, 30)
            nn.Conv2d(12, 48, 5),   # output shape (48, 26, 26)
            nn.ReLU(),              # activation
            nn.MaxPool2d(2),        # output shape (48, 13, 13)
        )
        self.fc = nn.Linear(48 * 13 * 13, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y


if __name__ == '__main__':
    # Parameter
    srcPath = '../Image/TestImage/'
    dstPath = '../Image/PyTorch/'
    if not os.path.exists(dstPath):
        os.mkdir(dstPath)
    pattern = '*.jpg'
    modelFilename = '../Model/model.yml'
    netFilename = '../Model/bottleNet.pkl'
    showFlag = False
    saveFlag = True
    use_gpu = torch.cuda.is_available()
    startTime = time.time()

    # Initialize
    pDollar = cv2.ximgproc.createStructuredEdgeDetection(modelFilename)
    edgeboxes = cv2.ximgproc.createEdgeBoxes()
    edgeboxes.setMaxBoxes(100)
    model = torch.load(netFilename)
    if use_gpu:
        model = model.cuda()
    print(model)

    # Processing images
    fileList = glob.glob(os.path.join(srcPath, pattern))
    numFiles = len(fileList)
    for i, srcFilename in enumerate(fileList):
        # Read image
        srcImage = cv2.imread(srcFilename)
        if srcImage is None:
            print('Read image failed!')
            continue
        dstImage = srcImage.copy()

        # Extract structured edge
        image = np.float32(srcImage) / 255.0
        edge = pDollar.detectEdges(image)
        orientation = pDollar.computeOrientation(edge)
        edge = pDollar.edgesNms(edge, orientation, 2, 0, 1, True)

        # Extract candidates
        candidates = edgeboxes.getBoundingBoxes(edge, orientation)[0]

        # Classify
        numCandidates = candidates.shape[0]
        images = np.zeros((numCandidates, 3, ImageHeight, ImageWidth), np.float32)
        for j, bbox in enumerate(candidates):
            x1 = bbox[0]
            x2 = x1 + bbox[2] - 1
            y1 = bbox[1]
            y2 = y1 + bbox[3] - 1
            image = srcImage[y1:y2, x1:x2]
            image = cv2.resize(image, (ImageWidth, ImageHeight))
            image = np.transpose(image, [2, 0, 1])
            images[j, :, :, :] = np.float32(image)
        images = torch.FloatTensor(images)
        if use_gpu:
            images = images.cuda()
        outputs = torch.sigmoid(model(images))
        outputs = torch.squeeze(outputs, -1)
        outputs = outputs.cpu().detach().numpy()
        bboxes = []
        scores = []
        for j, bbox in enumerate(candidates):
            output = float(outputs[j])
            if output > 0.5:
                bboxes.append(bbox)
                scores.append(output)
        indices = cv2.dnn.NMSBoxes(bboxes, scores, 0.8, 0.01)
        if len(indices) > 0:
            for idx in indices.flatten():
                bbox = bboxes[idx]
                pt1 = (bbox[0], bbox[1])
                pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                dstImage = cv2.rectangle(dstImage, pt1, pt2, (0, 0, 255), 2)

        # Show and save
        if showFlag:
            cv2.namedWindow("srcImage", 0)
            cv2.namedWindow("dstImage", 0)
            cv2.imshow("srcImage", srcImage)
            cv2.imshow("dstImage", dstImage)
            cv2.waitKey()
        if saveFlag:
            pos = srcFilename.find('\\') + 1
            dstFilename = os.path.join(dstPath, srcFilename[pos:])
            cv2.imwrite(dstFilename, dstImage)
        print('Progress: %d / %d' % (i, numFiles))

    endTime = time.time()
    totalTime = endTime - startTime
    averageTime = totalTime / numFiles
    print('Average time: %.4fs' % averageTime)
