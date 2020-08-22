#include "RandomForest.h"

#define ROWS 128 //average height
#define COLS 136 //average width

// #define COLS 104 //average width
// #define ROWS 104 //average height

// #define COLS 120 //average width
// #define ROWS 140 //average height

RandomForest::RandomForest(){
}

RandomForest::RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories)
    :mTreeCount(treeCount), mMaxDepth(maxDepth), mCVFolds(CVFolds), mMinSampleCount(minSampleCount), mMaxCategories(maxCategories) {
   /*
     construct a forest with given number of trees and initialize all the trees with the
     given parameters
   */
    mTreeCount = treeCount;
    for(int treeIdx = 0; treeIdx < treeCount; treeIdx++){
        cv::Ptr<cv::ml::DTrees> tree = cv::ml::DTrees::create();
        tree->setCVFolds(CVFolds);
        tree->setMaxCategories(maxCategories);
        tree->setMaxDepth(maxDepth);
        tree->setMinSampleCount(minSampleCount);
        mTrees.push_back(tree);
    }
}

RandomForest::~RandomForest(){}


void RandomForest::setTreeCount(int treeCount) {
    // Fill
    mTreeCount = treeCount;

}

void RandomForest::setMaxDepth(int maxDepth) {
    mMaxDepth = maxDepth;
    for(int treeIdx = 0; treeIdx < mTreeCount; treeIdx++)
        mTrees[treeIdx]->setMaxDepth(mMaxDepth);
}

void RandomForest::setCVFolds(int cvFolds) {
    // Fill
    mCVFolds = cvFolds;
    for(int treeIdx = 0; treeIdx < mTreeCount; treeIdx++)
        mTrees[treeIdx]->setCVFolds(mCVFolds);

}

void RandomForest::setMinSampleCount(int minSampleCount) {
    // Fill
    mMinSampleCount = minSampleCount;
    for(int treeIdx = 0; treeIdx < mTreeCount; treeIdx++)
        mTrees[treeIdx]->setMinSampleCount(mMinSampleCount);
}

void RandomForest::setMaxCategories(int maxCategories) {
    // Fill
    mMaxCategories = maxCategories;
    for(int treeIdx = 0; treeIdx < mTreeCount; treeIdx++)
        mTrees[treeIdx]->setMaxCategories(mMaxCategories);   
}

HOGDescriptor RandomForest::hogDescpObj() {

    HOGDescriptor hog_obj;
    hog_obj.setWinSize(cv::Size(COLS, ROWS));                       // cv::Size(136, 128)
    hog_obj.setBlockSize(cv::Size((COLS / 8) * 2, (ROWS / 8) * 2)); // cv::Size(34, 32)
    hog_obj.setBlockStep(cv::Size((COLS / 8), (ROWS / 8)));         // cv::Size(17, 16)
    hog_obj.setCellSize(cv::Size((COLS / 8), (ROWS / 8)));          // cv::Size(17, 16)
    hog_obj.setNbins(9);

    return hog_obj;
}

std::vector<cv::Mat> RandomForest::augmentImages(cv::Mat img) {

    std::vector<cv::Mat> augm_img;
    cv::Mat gray_scale, rotated180, rotated90, flipped, skewed1, skewed2, skewed3, skewed4;

    cv::cvtColor(img, gray_scale, cv::COLOR_BGR2GRAY);
    cv::flip(img, rotated180, -1); // rotate 180 degrees
    cv::rotate(img, rotated90, cv::ROTATE_90_CLOCKWISE); // rotate 90 degrees
    cv::flip(img, flipped, 1);


    float skew1[6] = {1, 0.26, 0, 0, 1, 0};
    auto R = cv::Mat(2, 3, CV_32F, skew1);
    cv::warpAffine(img, skewed1, R, img.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);

    float skew2[6] = {1, 0, 0, 0.26, 1, 0};
    R = cv::Mat(2, 3, CV_32F, skew2);
    cv::warpAffine(img, skewed2, R, img.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);

    float skew3[6] = {1, -0.4, 0, 0, 1, 0};
    R = cv::Mat(2, 3, CV_32F, skew3);
    cv::warpAffine(img, skewed3, R, img.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);

    float skew4[6] = {1, 0, 0, -0.4, 1, 0};
    R = cv::Mat(2, 3, CV_32F, skew4);
    cv::warpAffine(img, skewed4, R, img.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);


    augm_img.push_back(img);
    augm_img.push_back(gray_scale);
    augm_img.push_back(rotated180);
    augm_img.push_back(rotated90);
    augm_img.push_back(flipped);
    augm_img.push_back(skewed1);
    augm_img.push_back(skewed2);
    augm_img.push_back(skewed3);
    augm_img.push_back(skewed4);

    return augm_img;
}

cv::Ptr<cv::ml::TrainData> RandomForest::createTrainData() {
    // Fill
    std::vector<std::vector<float>> samples;
    std::vector<int> responses, labels;
    cv::Mat file;   
    HOGDescriptor hog_obj = RandomForest::hogDescpObj();

    std::vector<cv::String> trainDataNames = getRandomTrainData(labels);

    for (auto file_name = trainDataNames.begin(); file_name != trainDataNames.end(); ++file_name) {
        file = cv::imread(*file_name, cv::IMREAD_COLOR);
        std::vector<cv::Mat> augm_img = RandomForest::augmentImages(file);
        for (auto img = augm_img.begin(); img != augm_img.end(); ++img) {
            samples.push_back(hog_obj.detectHOGDescriptor(*img, cv::Size(COLS, ROWS), false));
            responses.push_back(labels[file_name - trainDataNames.begin()]); // push the value of the index of the label into responses from labels
        }
    }

    int numberOfSamples = samples.size();
    int featureSize = samples[0].size();

    cv::Mat samples_mat(numberOfSamples, featureSize, CV_32FC1); // number of samples, feature size, float
    cv::Mat labels_mat(numberOfSamples, 1, CV_32SC1); //number of samples, 1 column, int

    for(int i = 0; i < numberOfSamples; i++) {
        for(int j = 0; j < featureSize; j++) {
            samples_mat.ptr<float>(i)[j] = samples[i][j];
        }
    }
    
    for(int i = 0; i < numberOfSamples; i++)
        labels_mat.ptr<int>(i)[0] = responses[i];

    cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(samples_mat, cv::ml::ROW_SAMPLE, labels_mat); // !!!
    
    return train_data;
    
}

void RandomForest::train() {
    // Fill
    
    std::cout << "Training forest..." << std::endl;
    for(int treeIdx = 0; treeIdx < mTreeCount; treeIdx++){
        std::cout << "Training tree: " << treeIdx + 1;
        mTrees[treeIdx]->train(RandomForest::createTrainData());
        std::cout << "...done" << std::endl;
    }
    std::cout << "Training...completed" << std::endl;

}

void RandomForest::predict(cv::Mat query, double &confidence, int &label) {
    // Fill

    cv::Mat prediction;
    std::vector<int> counter(mMaxCategories);
    std::fill(counter.begin(), counter.end(), 0);
    std::vector<cv::Mat> setOfPredictions;
    
    for(int treeIdx = 0; treeIdx < mTreeCount; treeIdx++) {
        mTrees[treeIdx]->predict(query, prediction);
        setOfPredictions.push_back(prediction);
        prediction.release();
    }

    for(int i = 0; i < setOfPredictions.size(); i++) { //cols

        if (setOfPredictions[i].at<float>(0, 0) == 0) counter[0]++;
        if (setOfPredictions[i].at<float>(0, 0) == 1) counter[1]++;
        if (setOfPredictions[i].at<float>(0, 0) == 2) counter[2]++;
        if (setOfPredictions[i].at<float>(0, 0) == 3) counter[3]++;
    }

    auto max = std::max_element(counter.begin(), counter.end());
    label = std::distance(counter.begin(), max);

    confidence = (double) *max / (double) mTreeCount;
}