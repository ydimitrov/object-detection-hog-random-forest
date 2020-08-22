#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "HOGDescriptor.h"
#include "Window.h"

std::vector<cv::String> loadTestData(cv::String testFolderPath);
std::vector<cv::String> getRandomTrainData (std::vector<int> &labels);
std::vector<Window> createWindows(cv::Mat img) ;
HOGDescriptor hogDescpObj();
std::vector<float> calculateHog(cv::Mat);
cv::Mat convertHogToTestData(std::vector<float> descriptor);
bool sortByConf(Window &lhs, Window &rhs);
void drawBoundingBox(Window windowObj0, Window windowObj1, Window windowObj2, cv::String path, int nameName);
void drawBoundingBox2(std::vector<std::vector<Window>> container, cv::String path, int nameName);
void extractBoundingCoords(std::vector<std::vector<Window>> giantContainer);
std::vector<Window> nonMaxSuppress(std::vector<Window> windows, float threshold);

#endif //HELPERFUNCTIONS_H