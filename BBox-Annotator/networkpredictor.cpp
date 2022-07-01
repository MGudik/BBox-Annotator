#include "networkpredictor.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <QtCore>
#include <QDebug>

using namespace std;
using namespace cv;
using namespace dnn;

networkPredictor::networkPredictor()
{

}

void networkPredictor::loadNetwork()
{
    ifstream ifs(string("D:/inputs/yolo/coco.names").c_str());
    string line;

    while (getline(ifs, line))
       m_class_names.push_back(line);

    // load the neural network model
    m_net = readNetFromDarknet("D:/inputs/yolo/yolov4-tiny.cfg", "D:/inputs/yolo/yolov4-tiny.weights");
    m_net.setPreferableBackend(DNN_BACKEND_OPENCV);
    m_net.setPreferableTarget(DNN_TARGET_CPU);
}

void networkPredictor::predict()
{
    Mat image = imread("D:/inputs/img3.jpg");
    Mat resized;
    resize(image, resized, Size(416, 416), 0, 0);

    Mat src_blob;
    blobFromImage(image, src_blob, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);

    m_net.setInput(src_blob);

    std::vector<Mat> output_blobs;
    m_net.forward(output_blobs);

    // Initialize vectors to hold respective outputs while unwrapping     detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;
    // Resizing factor.
    float x_factor = image.cols;
    float y_factor = image.rows;
    float *data = (float *)output_blobs[0].data;

    const int rows = 2028;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= m_conf_threshold)
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, m_class_names.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > m_conf_threshold)
            {
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }
        }
        // Jump to the next row.
        data += 85;
    }

    vector<int> indices;
    NMSBoxes(boxes, confidences, 0.3, 0.45, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(image, Point(left, top), Point(left + width, top + height), Scalar(0, 255, 0), 1);
        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = m_class_names[class_ids[idx]] + ":" + label;
    }

    imshow("image", image);
    imwrite("image_result.jpg", image);
    waitKey(0);
    destroyAllWindows();
}

// Get the names of the output layers
vector<String> networkPredictor::getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}
