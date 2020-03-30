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
	/******************* ���� *******************/
	string srcPath = "../../Image/TestImage/";
	string dstPath = "../../Image/EdgeBoxes/";
	String modelFilename = "model.yml";
	string pattern = srcPath + "*.jpg";
	bool showFlag = false;
	bool saveFlag = true;
	clock_t startTime, endTime;
	float totalTime, averageTime;
	startTime = clock();

	/******************* ��ʼ�� *******************/
	Ptr<StructuredEdgeDetection> pDollar = createStructuredEdgeDetection(modelFilename);
	Ptr<EdgeBoxes> edgeboxes = createEdgeBoxes();
	edgeboxes->setMaxBoxes(100);

	/******************* ��������ͼƬ *******************/
	vector<string> fileList;
	glob(pattern, fileList, false);
	int numImages = (int)fileList.size();
	for (int i = 0; i < numImages; i++) {
		// ��ȡͼƬ
		string srcFilename = fileList[i];
		Mat srcImage = imread(srcFilename);
		if (srcImage.empty()) {
			cout << "Read image Failed!" << endl;
			continue;
		}
		Mat dstImage = srcImage.clone();

		// ��ȡ�ṹ��Ե
		Mat _src, _edge, _orientation;
		srcImage.convertTo(_src, CV_32F, 1 / 255.0);
		pDollar->detectEdges(_src, _edge);
		pDollar->computeOrientation(_edge, _orientation);
		pDollar->edgesNms(_edge.clone(), _orientation, _edge, 2, 0, 1, true);

		// ��ȡ��ѡ�� 
		vector<Rect> bboxes;
		edgeboxes->getBoundingBoxes(_edge, _orientation, bboxes);
		int numBboxes = (int)bboxes.size();
		for (int j = 0; j < numBboxes; j++)
			rectangle(dstImage, bboxes[j], Scalar(0, 0, 255), 2);

		// ��ʾ�ͱ���
		if (showFlag) {
			namedWindow("srcImage", 0);
			namedWindow("dstImage", 0);
			imshow("srcImage", srcImage);
			imshow("dstImage", dstImage);
			waitKey(1);
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