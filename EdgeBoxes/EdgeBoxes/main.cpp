#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;


int main()
{
	/******************* ���� *******************/
	string srcPath = "src/";
	string dstPath = "dst/";
	String modelFilename = "model.yml";
	string pattern = srcPath + "*.jpg";
	bool showFlag = false;
	bool saveFlag = true;

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
		for (int j = 0; j < numBboxes; j++) {
			Point pt1(bboxes[j].x, bboxes[j].y);
			Point pt2(bboxes[j].x + bboxes[j].width, bboxes[j].y + bboxes[j].height);
			rectangle(dstImage, pt1, pt2, Scalar(0, 0, 255), 2);
		}

		// ��ʾ�ͱ���
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

	return 0;
}