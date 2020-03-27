#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <time.h> 

using namespace std;
using namespace cv;
using namespace cv::ximgproc;
using namespace cv::ml;

#define ImageWidht	64		
#define ImageHeight	64
#define MinNumImage	100

int main()
{
	/******************* 参数 *******************/
	string srcPath = "src/";
	string dstPath = "dst/";
	String modelFilename = "E:/opencv/HogSvm/EdgeBoxes/EdgeBoxes/model.yml";
	string svmFilename = "svm-rbf.xml";
	string pattern = srcPath + "*.jpg";
	bool showFlag = false;
	bool saveFlag = true;
	clock_t startTime, endTime;
	float totalTime, averageTime;
	startTime = clock();

	/******************* 初始化 *******************/
	Ptr<StructuredEdgeDetection> pDollar = createStructuredEdgeDetection(modelFilename);
	Ptr<EdgeBoxes> edgeboxes = createEdgeBoxes();
	edgeboxes->setMaxBoxes(100);
	HOGDescriptor hog(Size(ImageWidht, ImageHeight), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	Ptr<SVM> svm = SVM::load(svmFilename);
	int dimension = svm->getVarCount();

	/******************* 处理所有图片 *******************/
	vector<string> fileList;
	glob(pattern, fileList, false);
	int numImages = min(MinNumImage, (int)fileList.size());
	for (int i = 0; i < numImages; i++) {

		// 读取图片
		string srcFilename = fileList[i];
		Mat srcImage = imread(srcFilename);
		if (srcImage.empty()) {
			cout << "Read image Failed!" << endl;
			continue;
		}
		Mat dstImage = srcImage.clone();

		// 提取结构边缘
		Mat _src, _edge, _orientation;
		srcImage.convertTo(_src, CV_32F, 1 / 255.0);
		pDollar->detectEdges(_src, _edge);
		pDollar->computeOrientation(_edge, _orientation);
		pDollar->edgesNms(_edge.clone(), _orientation, _edge, 2, 0, 1, true);

		// 提取候选框 
		vector<Rect> candidates;
		edgeboxes->getBoundingBoxes(_edge, _orientation, candidates);
		int numCandidates = (int)candidates.size();

		// 计算HOG特征 
		Mat featureMat = Mat::zeros(numCandidates, dimension, CV_32FC1);
		for (int j = 0; j < numCandidates; j++) {
			Mat image = srcImage(candidates[j]).clone();
			resize(image.clone(), image, Size(ImageWidht, ImageHeight), 0, 0, INTER_CUBIC);
			vector<float> descriptors;
			hog.compute(image, descriptors, Size(8, 8));
			for (int k = 0; k < dimension; k++)
				featureMat.at<float>(j, k) = descriptors[k];
		}

		// SVM分类
		vector<Rect> bboxes;
		Mat labelMat;
		svm->predict(featureMat, labelMat, 0);
		for (int j = 0; j < numCandidates; j++) {
			float label = labelMat.at<float>(j, 0);
			if (label > 0)
				bboxes.push_back(candidates[j]);
		}
		groupRectangles(bboxes, 2, 0.5);

		// 显示和保存
		int numBoxes = (int)bboxes.size();
		for (int j = 0; j < numBoxes; j++) {
			Point pt1(bboxes[j].x, bboxes[j].y);
			Point pt2(bboxes[j].x + bboxes[j].width, bboxes[j].y + bboxes[j].height);
			rectangle(dstImage, pt1, pt2, Scalar(0, 0, 255), 2);
		}
		if (showFlag) {
			namedWindow("srcImage", 0);
			namedWindow("dstImage", 0);
			imshow("srcImage", srcImage);
			imshow("dstImage", dstImage);
			waitKey();
		}
		if (saveFlag) {
			int pos = (int)srcFilename.find_last_of("\\") + 1;
			string dstFilename = dstPath + srcFilename.substr(pos);
			imwrite(dstFilename, dstImage);
		}
		cout << "Progress: " << i + 1 << "/" << numImages << endl;
	}

	endTime = clock();
	totalTime = (endTime - startTime) / CLOCKS_PER_SEC + 0.5;
	averageTime = totalTime / numImages;
	cout << "Average time:\t" << averageTime << "s" << endl << endl;

	return 0;
}