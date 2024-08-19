#include "../inc/dnn.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

#define USE_CUDA false // use opencv-cuda

using namespace std;
using namespace cv;
using namespace dnn;

int main()
{
	cv::VideoCapture cap(2);
	cv::Mat img;
	string model_path3 = "/home/ddxy/下载/dnn/best.onnx";
	Yolov8 yolov8;
	Net net3;
	bool isOK = yolov8.readModel(net3, model_path3, USE_CUDA);
	if (isOK)
	{
		cout << "read net ok!" << endl;
	}
	else
	{
		cout << "read onnx model failed!";
		return -1;
	}
	while (1)
	{
		cap.read(img);

		vector<Scalar> color;
		srand(time(0));
		for (int i = 0; i < 80; i++)
		{
			int b = rand() % 256;
			int g = rand() % 256;
			int r = rand() % 256;
			color.push_back(Scalar(b, g, r));
		}

		vector<Detection> result3 = yolov8.Detect(img, net3);
		yolov8.drawPred(img, result3, color);
		Mat dst = img({0, 0, img.cols, img.rows});

		cv::imshow("aaa", img);
		cv::waitKey(1);
	}
	return 0;
}