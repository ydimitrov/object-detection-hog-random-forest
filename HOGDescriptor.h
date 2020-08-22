#ifndef RF_HOGDESCRIPTOR_H
#define RF_HOGDESCRIPTOR_H


#include <opencv2/opencv.hpp>
#include <vector>

class HOGDescriptor {

public:

    HOGDescriptor() {
        //initialize default parameters(win_size, block_size, block_step,....)
        win_size = cv::Size(64, 64);
        block_size = cv::Size(16, 16);
        block_step = cv::Size(8, 8);
        cell_size = cv::Size(8, 8);
        nbins = 9;

        //Fill other parameters here
        
        // parameter to check if descriptor is already initialized or not
        is_init = false;
    };

    void setWinSize(cv::Size ws) {
        win_size = ws; 
    }

    void setBlockSize(cv::Size bsize) {
        block_size = bsize;
    }

    void setBlockStep(cv::Size bstep) {
       block_step = bstep;
    }

    void setCellSize(cv::Size csize) {
        cell_size = csize;
    }

    void setPadSize(cv::Size sz) {
        pad_size = sz;
    }

    void setNbins(int bins) {
        nbins = bins;
    }

    cv::Size getWinSize() {
        return win_size;
    }

    cv::Size getBlockSize() {
        return block_size;
    }
    
    cv::Size getBlockStep() {
        return block_step;
    }

    cv::Size getCellSize() {
        return cell_size;
    }

    cv::Size getPadSize() {
        return pad_size;
    }

    int getNbins() {
        return nbins;
    }

    void initDetector(); // set parameters for already initialised object

    void visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor);

    std::vector<float> detectHOGDescriptor(cv::Mat &im, cv::Size sz, bool show);

    ~HOGDescriptor() {};


private:
    cv::Size win_size;
    cv::Size pad_size;
    cv::Size block_size;
    cv::Size block_step;
    cv::Size cell_size;
    int nbins;

    /*
        Fill other parameters here
    */

    // cv::HOGDescriptor hog_detector;
public:
    cv::HOGDescriptor hog_detector;
    cv::HOGDescriptor getHog_detector();
    void setHog_detector(cv::HOGDescriptor hog);

private:
    bool is_init;
};

#endif //RF_HOGDESCRIPTOR_H
