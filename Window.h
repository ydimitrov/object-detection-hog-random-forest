#ifndef RF_WINDOW_H
#define RF_WINDOW_H


#include <opencv2/opencv.hpp>
#include <vector>

class Window {

public:

    Window();

    Window(cv::String origImg, cv::Rect bndrs, cv::Mat mtrx, int lbl, double cnf);

    ~Window();

    void setOriginImg(cv::String origImg);
    void setBoundaries(cv::Rect bndrs);
    void setMatrix(cv::Mat matrix);
    void setLabel(int lbl);
    void setConfidence(double confidence);

    cv::String getOriginImg();
    cv::Rect getBoundaries();
    cv::Mat getMatrix();
    int getLabel();
    double getConfidence();


private:
    cv::String originImg;
    cv::Rect boundaries;
    cv::Mat matrix;
    int label;
    double confidence;
};

#endif //RF_WINDOW_H
