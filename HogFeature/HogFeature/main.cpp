#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <time.h> 

using namespace std;
using namespace cv;
using namespace cv::ml;

#define ImageWidht	64		
#define ImageHeight	64

int main()
{
	/******************* 参数 *******************/
	string trainPathFilename = "../../Dataset/train/label.txt";
	string testPathFilename = "../../Dataset/test/label.txt";
	string HogFeatureFilename = "../../Dataset/HogFeature.xml";

	/******************* 初始化 *******************/
	HOGDescriptor hog(Size(ImageWidht, ImageHeight), Size(16, 16), Size(8, 8), Size(8, 8), 9);

	/******************* 训练数据 *******************/
	// 加载数据 
	vector<string> trainFilenames;
	vector<int> trainLabels;
	ifstream trainFin(trainPathFilename);
	if (!trainFin.is_open()) {
		cout << "Failed to open " << trainPathFilename << endl;
		return -1;
	}
	string buffer;
	while (getline(trainFin, buffer)) {
		int pos = buffer.find(" ");
		string imageFilename = buffer.substr(0, pos);
		string strLabel = buffer.substr(pos);
		int imageLabel;
		sscanf_s(strLabel.c_str(), "%d", &imageLabel);
		trainFilenames.push_back(imageFilename);
		trainLabels.push_back(imageLabel);
	}
	trainFin.close();
	int numTrainImages = (int)trainFilenames.size();

	// 计算HOG特征 
	int dimension = 0;
	Mat trainFeatureMat, trainLabelMat;
	for (int i = 0; i < numTrainImages; i++) {
		string imageFilename = trainFilenames[i];
		int imageLabel = trainLabels[i];
		Mat srcImage = imread(imageFilename);
		if (srcImage.empty()) {
			cout << "Failed to open " << imageFilename << endl;
			return -1;
		}
		resize(srcImage.clone(), srcImage, Size(ImageWidht, ImageHeight), 0, 0, INTER_CUBIC);
		vector<float> descriptors;
		hog.compute(srcImage, descriptors, Size(8, 8));
		if (i == 0) {
			dimension = (int)descriptors.size();
			trainFeatureMat = Mat::zeros(numTrainImages, dimension, CV_32FC1);
			trainLabelMat = Mat::zeros(numTrainImages, 1, CV_32SC1);
		}
		for (int j = 0; j < dimension; j++)
			trainFeatureMat.at<float>(i, j) = descriptors[j];
		trainLabelMat.at<int>(i, 0) = imageLabel;

		cout << "Progress:\t" << (i + 1) << "/" << numTrainImages << endl;
	}

	/******************* 测试数据 *******************/
	// 加载数据
	vector<string> testFilenames;
	vector<int> testLabels;
	ifstream testFin(testPathFilename);
	if (!testFin.is_open()) {
		cout << "Failed to open " << testPathFilename << endl;
		return -1;
	}
	while (getline(testFin, buffer)) {
		int pos = buffer.find(" ");
		string imageFilename = buffer.substr(0, pos);
		string strLabel = buffer.substr(pos);
		int imageLabel;
		sscanf_s(strLabel.c_str(), "%d", &imageLabel);
		testFilenames.push_back(imageFilename);
		testLabels.push_back(imageLabel);
	}
	testFin.close();
	int numTestImages = (int)testFilenames.size();

	// 计算HOG特征 
	Mat testFeatureMat = Mat::zeros(numTestImages, dimension, CV_32FC1);
	Mat	testLabelMat = Mat::zeros(numTestImages, 1, CV_32SC1);
	for (int i = 0; i < numTestImages; i++) {
		string imageFilename = testFilenames[i];
		int imageLabel = testLabels[i];
		Mat srcImage = imread(imageFilename);
		if (srcImage.empty()) {
			cout << "Failed to open " << imageFilename << endl;
			return -1;
		}
		resize(srcImage.clone(), srcImage, Size(ImageWidht, ImageHeight), 0, 0, INTER_CUBIC);
		vector<float> descriptors;
		hog.compute(srcImage, descriptors, Size(8, 8));
		for (int j = 0; j < dimension; j++)
			testFeatureMat.at<float>(i, j) = descriptors[j];
		testLabelMat.at<int>(i, 0) = imageLabel;

		cout << "Progress:\t" << (i + 1) << "/" << numTestImages << endl;
	}

	/******************* 保存结果 *******************/
	FileStorage fs(HogFeatureFilename, FileStorage::WRITE);
	if (fs.isOpened()) {
		fs << "TrainFeatures" << trainFeatureMat;
		fs << "TrainLabels" << trainLabelMat;
		fs << "TestFeatures" << testFeatureMat;
		fs << "TestLabels" << testLabelMat;
	}
	fs.release();

	return 0;
}