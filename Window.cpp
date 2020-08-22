#include "Window.h"


Window::Window(){
}

Window::Window (cv::String origImg, cv::Rect bndrs, cv::Mat mtrx, int lbl, double cnf) {
    originImg = origImg;
    boundaries = bndrs;
    matrix = mtrx;
    label = lbl;
    confidence = cnf;
}

Window::~Window(){}


void Window::setOriginImg(cv::String origImg) {
    originImg = origImg;
}

void Window::setBoundaries(cv::Rect bndrs) {
    boundaries = bndrs;
}
void Window::setMatrix(cv::Mat mtrx) {
    matrix = mtrx;
}
void Window::setLabel(int lbl) {
    label = lbl;
}
void Window::setConfidence(double cnf) {
    confidence = cnf;
}

cv::String Window::getOriginImg() {
    return originImg;
}
cv::Rect Window::getBoundaries() {
    return boundaries;
}
cv::Mat Window::getMatrix() {
    return matrix;
}
int Window::getLabel() {
    return label;
}
double Window::getConfidence() {
    return confidence;
}