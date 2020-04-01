#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <time.h> 

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

int main()
{
	/******************* 参数 *******************/
	string srcPath = "../../Image/TestImage/";
	string dstPath = "../../Image/EdgeBoxes/";
	String modelFilename = "../../Model/model.yml";
	string pattern = "*.jpg";
	bool showFlag = true;
	bool saveFlag = false;
	clock_t startTime, endTime;
	float totalTime, averageTime;
	startTime = clock();

	/******************* 初始化 *******************/
	Ptr<StructuredEdgeDetection> pDollar = createStructuredEdgeDetection(modelFilename);
	Ptr<EdgeBoxes> edgeboxes = createEdgeBoxes();
	edgeboxes->setMaxBoxes(100);

	/******************* 处理所有图片 *******************/
	vector<string> fileList;
	glob(srcPath + pattern, fileList, false);
	int numImages = (int)fileList.size();
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
		vector<Rect> bboxes;
		edgeboxes->getBoundingBoxes(_edge, _orientation, bboxes);
		int numBboxes = (int)bboxes.size();
		for (int j = 0; j < numBboxes; j++)
			rectangle(dstImage, bboxes[j], Scalar(0, 0, 255), 2);

		// 显示和保存
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