#ifndef NETWORKPREDICTOR_H
#define NETWORKPREDICTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <QtCore>
#include <QDebug>

using namespace std;
using namespace cv;
using namespace dnn;

class networkPredictor
{
public:
    networkPredictor();


    void loadNetwork();
    void predict();


private:
    Net m_net;
    std::string m_config_path = "D:/inputs/yolo/yolov4-tiny.cfg";
    std::string m_weights_path = "D:/inputs/yolo/yolov4-tiny.weights";
    Size m_size = Size(416, 416);
    std::vector<std::string> m_class_names;
    double m_conf_threshold = 0.2;
    double m_NMS_threshold = 0.4;

    vector<String> getOutputsNames(const Net &net);
};

#endif // NETWORKPREDICTOR_H
