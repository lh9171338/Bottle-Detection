import os
import glob
import time
import numpy as np
import cv2
import torch
import torch.nn as nn


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
    scr_path = '../Image/TestImage/'
    dst_path = '../Image/PyTorch/'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    pattern = '*.jpg'
    model_filename = '../Model/model.yml'
    net_filename = '../Model/bottleNet.pkl'
    show_flag = True
    save_flag = False
    start_time = time.time()

    # Initialize
    structured_edge = cv2.ximgproc.createStructuredEdgeDetection(model_filename)
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(100)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(net_filename).to(device)
    print(model)

    # Processing images
    file_list = glob.glob(os.path.join(scr_path, pattern))
    num_files = len(file_list)
    for i, src_filename in enumerate(file_list):
        # Read image
        src_image = cv2.imread(src_filename)
        if src_image is None:
            print('Read image failed!')
            continue
        dst_image = src_image.copy()

        # Extract structured edge
        image = np.float32(src_image) / 255.0
        edge = structured_edge.detectEdges(image)
        orientation = structured_edge.computeOrientation(edge)
        edge = structured_edge.edgesNms(edge, orientation, 2, 0, 1, True)

        # Extract candidates
        candidates = edge_boxes.getBoundingBoxes(edge, orientation)[0]

        # Classify
        num_candidates = candidates.shape[0]
        images = np.zeros((num_candidates, 3, 64, 64), np.float32)
        for j, bbox in enumerate(candidates):
            x1 = bbox[0]
            x2 = x1 + bbox[2] - 1
            y1 = bbox[1]
            y2 = y1 + bbox[3] - 1
            image = src_image[y1:y2, x1:x2]
            image = cv2.resize(image, (64, 64))
            image = np.transpose(image, [2, 0, 1])
            images[j, :, :, :] = np.float32(image)
        images = torch.FloatTensor(images).to(device)
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
                dst_image = cv2.rectangle(dst_image, pt1, pt2, (0, 0, 255), 2)

        # Show and save
        if show_flag:
            cv2.namedWindow("dst_image", 0)
            cv2.imshow("dst_image", dst_image)
            cv2.waitKey()
        if save_flag:
            pos = src_filename.find('\\') + 1
            dst_filename = os.path.join(dst_path, src_filename[pos:])
            cv2.imwrite(dst_filename, dst_image)
        print('Progress: %d / %d' % (i, num_files))

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_files
    print('Average time: %.4fs' % avg_time)
