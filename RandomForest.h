#ifndef RF_RANDOMFOREST_H
#define RF_RANDOMFOREST_H

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <algorithm>
#include <math.h>
#include "HelperFunctions.h"
#include "HOGDescriptor.h"

class RandomForest
{
public:
	RandomForest();

    // You can create the forest directly in the constructor or create an empty forest and use the below methods to populate it
	RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories);
    
    ~RandomForest();

    void setTreeCount(int treeCount);
    void setMaxDepth(int maxDepth);
    void setCVFolds(int cvFols);
    void setMinSampleCount(int minSampleCount);
    void setMaxCategories(int maxCategories);


    void train();

    void predict(cv::Mat query, double &confidence, int &_class);


private:
    int mTreeCount;
    int mMaxDepth;
    int mCVFolds;
    int mMinSampleCount;
    int mMaxCategories;

private:
    // these methods are applied to each tree
    std::vector<cv::Mat> augmentImages(cv::Mat img); 
    HOGDescriptor hogDescpObj();
    cv::Ptr<cv::ml::TrainData> createTrainData();    

    // M-Trees for constructing the forest
    std::vector<cv::Ptr<cv::ml::DTrees> > mTrees;
};

#endif //RF_RANDOMFOREST_H
