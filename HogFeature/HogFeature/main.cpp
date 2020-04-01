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
	string trainPath = "../../Dataset/train/";
	string testPath = "../../Dataset/test/";
	string pattern = "*.jpg";
	string HogFeatureFilename = "../../Dataset/HogFeature.xml";
	
	/******************* 初始化 *******************/
	HOGDescriptor hog(Size(ImageWidht, ImageHeight), Size(16, 16), Size(8, 8), Size(8, 8), 9);

	/******************* 训练数据 *******************/
	// 加载数据 
	vector<string> trainFileList;
	glob(trainPath + pattern, trainFileList, false);
	int numTrainImages = (int)trainFileList.size();

	// 计算HOG特征 
	int dimension = 0;
	Mat trainFeatureMat, trainLabelMat;
	for (int i = 0; i < numTrainImages; i++) {
		string imageFilename = trainFileList[i];
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
		int imageLabel = (imageFilename.find("Pos") != string::npos) ? 1 : 0;
		trainLabelMat.at<int>(i, 0) = imageLabel;

		cout << "Progress:\t" << (i + 1) << "/" << numTrainImages << endl;
	}

	/******************* 测试数据 *******************/
	// 加载数据
	vector<string> testFileList;
	glob(testPath + pattern, testFileList, false);
	int numTestImages = (int)testFileList.size();

	// 计算HOG特征 
	Mat testFeatureMat = Mat::zeros(numTestImages, dimension, CV_32FC1);
	Mat	testLabelMat = Mat::zeros(numTestImages, 1, CV_32SC1);
	for (int i = 0; i < numTestImages; i++) {
		string imageFilename = testFileList[i];
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
		int imageLabel = (imageFilename.find("Pos") != string::npos) ? 1 : 0;
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