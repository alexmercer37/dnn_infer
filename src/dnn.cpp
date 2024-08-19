#include "../inc/dnn.hpp"

bool Yolo::readModel(Net &net, string &netPath, bool isCuda = false)
{
    try
    {
        net = readNetFromONNX(netPath);
    }
    catch (const std::exception &)
    {
        return false;
    }
    if (isCuda)
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    return true;
}

void Yolo::drawPred(Mat &img, vector<Detection> results, vector<Scalar> colors)
{

    for (auto result : results)
    {
        cv::Rect box = result.box;
        cv::Scalar color = colors[result.class_id];

        cv::rectangle(img, box, color, 2);

        std::string classString = classes[result.class_id] + ' ' + std::to_string(result.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(img, textBox, color, cv::FILLED);
        cv::putText(img, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
}

vector<Detection> Yolov8::Detect(Mat &modelInput, Net &net)
{
    modelInput = formatToSquare(modelInput);
    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0 / 255.0, Size(netWidth, netHeight), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    int rows = outputs[0].size[2];
    int dimensions = outputs[0].size[1];

    outputs[0] = outputs[0].reshape(1, dimensions);
    cv::transpose(outputs[0], outputs[0]);

    float *data = (float *)outputs[0].data;
    // Mat detect_output(8400, 84, CV_32FC1, data);// 8400 = 80*80+40*40+20*20
    float x_factor = (float)modelInput.cols / netWidth;
    float y_factor = (float)modelInput.rows / netHeight;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {
        cv::Mat scores(1, classes.size(), CV_32FC1, data + 4);
        cv::Point class_id;
        double maxClassScore;

        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
        if (maxClassScore > modelConfidenceThreshold)
        {
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);

            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
        data += dimensions;
    }

    // 执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
    vector<int> nms_result;
    NMSBoxes(boxes, confidences, confidenceThreshold, nmsIoUThreshold, nms_result);
    vector<Detection> detections{};
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        detections.push_back(result);
    }
    return detections;
}