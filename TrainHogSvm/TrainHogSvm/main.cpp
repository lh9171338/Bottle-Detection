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
	string trainLabelFilename = "E:/BottleDataset/classification/train/label.txt";
	string testLabelFilename = "E:/BottleDataset/classification/test/label.txt";
	string modelFilename = "svm-rbf.xml";
	bool trainFlag = true;
	bool testFlag = true;

	/******************* 初始化 *******************/
	HOGDescriptor hog(Size(ImageWidht, ImageHeight), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setC(1.0);
	svm->setClassWeights(Mat(Vec2d(1.0, 1.0)));
	svm->setKernel(SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-7));

	/******************* 训练 *******************/
	if (trainFlag) {
		// 加载训练数据 
		vector<string> trainFilenames;
		vector<int> trainLabels;
		ifstream trainFin(trainLabelFilename);
		if (!trainFin.is_open()) {
			cout << "Failed to open " << trainLabelFilename << endl;
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
		int dimension;
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

			//cout << "Progress:\t" << (i + 1) << "/" << numTrainImages << endl;
		}
		cout << "******************* Training sample parameters *******************" << endl;
		cout << "Image width:\t" << ImageWidht << endl;
		cout << "Image height:\t" << ImageHeight << endl;
		cout << "Image number:\t" << numTrainImages << endl << endl;

		// 训练SVM分类器 
		clock_t startTime, endTime, totalTime;
		startTime = clock();
		bool trainResult = svm->train(trainFeatureMat, ROW_SAMPLE, trainLabelMat);
		endTime = clock();
		totalTime = (endTime - startTime) / CLOCKS_PER_SEC + 0.5;

		cout << "******************* SVM training parameters *******************" << endl;
		cout << "Feature number:\t" << trainFeatureMat.rows << endl;
		cout << "Feature dimension\t" << trainFeatureMat.cols << endl;
		cout << "Training result:\t" << trainResult << endl;
		cout << "Training time:\t" << totalTime << "s" << endl << endl;
		if (!trainResult)
			return -1;
		svm->save(modelFilename);
	}

	/******************* 测试 *******************/
	if (testFlag) {
		svm = SVM::load(modelFilename);
		int dimension = svm->getVarCount();

		// 加载测试数据
		vector<string> testFilenames;
		vector<int> testLabels;
		ifstream testFin(testLabelFilename);
		if (!testFin.is_open()) {
			cout << "Failed to open " << testLabelFilename << endl;
			return -1;
		}
		string buffer;
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

			//cout << "Progress:\t" << (i + 1) << "/" << numTestImages << endl;
		}
		cout << "******************* Test sample parameters *******************" << endl;
		cout << "Image width:\t" << ImageWidht << endl;
		cout << "Image height:\t" << ImageHeight << endl;
		cout << "Image number:\t" << numTestImages << endl << endl;

		// 测试精度
		Mat predictLabelMat;
		svm->predict(testFeatureMat, predictLabelMat, 0);
		int totalError = 0;
		int posError = 0;
		int negError = 0;
		float correct;
		for (int i = 0; i < numTestImages; i++) {
			int imageLabel = testLabelMat.at<int>(i, 0);
			float predictLabel = predictLabelMat.at<float>(i, 0);
			if (imageLabel != predictLabel) {
				if (imageLabel == 1)
					posError++;
				else
					negError++;
			}
		}
		totalError = posError + negError;
		correct = 1.0 - totalError * 1.0 / numTestImages;

		cout << "******************* Test results *******************" << endl;
		cout << "Image number:\t" << numTestImages << endl;
		cout << "Total error number:\t" << totalError << endl;
		cout << "Positive error number:\t" << posError << endl;
		cout << "Negitive error number:\t" << negError << endl;
		cout << "Correct ratio:\t" << correct << endl << endl;
	}

	return 0;
}