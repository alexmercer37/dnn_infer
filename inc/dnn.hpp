#pragma once
#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace cv::dnn;
struct Detection
{
    int class_id{0};
    float confidence{0.0};
    cv::Rect box{};
};
class Yolo
{
public:
    bool readModel(cv::dnn::Net &net, std::string &netPath, bool isCuda);
    void drawPred(cv::Mat &img, std::vector<Detection> result, std::vector<cv::Scalar> color);
    virtual vector<Detection> Detect(cv::Mat &SrcImg, cv::dnn::Net &net) = 0;
    float sigmoid_x(float x) { return static_cast<float>(1.f / (1.f + exp(-x))); }
    Mat formatToSquare(const cv::Mat &source)
    {
        int col = source.cols;
        int row = source.rows;
        int _max = MAX(col, row);
        cv::Mat result = cv::Mat::zeros(source.rows, source.cols, CV_8UC3);
        source.copyTo(result(cv::Rect(0, 0, col, row)));
        return result;
    }
    const int netWidth = 640;
    const int netHeight = 640;

    float modelConfidenceThreshold{0.0};
    float modelScoreThreshold{0.0};
    float modelNMSThreshold{0.0};

    std::vector<std::string> classes = {"red_ball", "blue_ball"};
};

class Yolov8 : public Yolo
{
public:
    vector<Detection> Detect(Mat &SrcImg, Net &net);

private:
    float confidenceThreshold{0.25};
    float nmsIoUThreshold{0.70};
};