#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <time.h> 

using namespace std;
using namespace cv;
using namespace cv::ml;

int main()
{
	/******************* 参数 *******************/
	string HogFeatureFilename = "../../Dataset/HogFeature.xml";
	string modelFilename = "../../Model/svm-linear.xml";
	bool trainFlag = true;
	bool testFlag = true;
	int numTrain = 1000;
	clock_t startTime, endTime, totalTime;

	/******************* 初始化 *******************/
	Ptr<SVM> model = SVM::create();
	model->setType(SVM::C_SVC);
	model->setC(1.0);
	model->setClassWeights(Mat(Vec2d(1.0, 1.0)));
	model->setKernel(SVM::LINEAR);
	model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-7));

	/******************* 获取数据 *******************/
	cout << "******************* Loading data *******************" << endl;
	cout << "Loading..." << endl;
	startTime = clock();

	FileStorage fs(HogFeatureFilename, FileStorage::READ);
	Mat trainFeatureMat, trainLabelMat, testFeatureMat, testLabelMat;
	if (fs.isOpened()) {
		if (trainFlag) {
			fs["TrainFeatures"] >> trainFeatureMat;
			fs["TrainLabels"] >> trainLabelMat;
			if (numTrain < trainFeatureMat.rows) {
				trainFeatureMat = trainFeatureMat.rowRange(0, numTrain).clone();
				trainLabelMat = trainLabelMat.rowRange(0, numTrain).clone();
			}
		}
		if (testFlag) {
			fs["TestFeatures"] >> testFeatureMat;
			fs["TestLabels"] >> testLabelMat;
		}
	}
	fs.release();

	endTime = clock();
	totalTime = (endTime - startTime) / CLOCKS_PER_SEC + 0.5;
	cout << "Loading time:\t" << totalTime << "s" << endl << endl;

	/******************* 训练SVM分类器 *******************/
	if (trainFlag) {
		cout << "******************* Training SVM *******************" << endl;
		cout << "Feature number:\t" << trainFeatureMat.rows << endl;
		cout << "Feature dimension\t" << trainFeatureMat.cols << endl;
		cout << "Training..." << endl;
		startTime = clock();

		bool trainResult = model->train(trainFeatureMat, ROW_SAMPLE, trainLabelMat);

		endTime = clock();
		totalTime = (endTime - startTime) / CLOCKS_PER_SEC + 0.5;
		cout << "Training result:\t" << trainResult << endl;
		cout << "Training time:\t" << totalTime << "s" << endl << endl;
		if (!trainResult)
			return -1;
		model->save(modelFilename);
	}

	/******************* 测试SVM分类器 *******************/
	if (testFlag) {
		cout << "******************* Testing SVM *******************" << endl;
		cout << "Image number:\t" << testFeatureMat.rows << endl;
		cout << "Testing..." << endl;
		startTime = clock();

		model = SVM::load(modelFilename);
		Mat predictLabelMat;
		model->predict(testFeatureMat, predictLabelMat, 0);
		int numTestImages = (int)predictLabelMat.rows;
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

		endTime = clock();
		totalTime = (endTime - startTime) / CLOCKS_PER_SEC + 0.5;
		cout << "Total error number:\t" << totalError << endl;
		cout << "Positive error number:\t" << posError << endl;
		cout << "Negitive error number:\t" << negError << endl;
		cout << "Correct ratio:\t" << correct << endl;
		cout << "Testing time:\t" << totalTime << "s" << endl << endl;
	}

	return 0;
}