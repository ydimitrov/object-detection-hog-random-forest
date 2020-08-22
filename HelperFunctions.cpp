#include "HelperFunctions.h"
#include <string>	

#define COLS 136 //average width
#define ROWS 128 //average height

// #define COLS 104 //average width
// #define ROWS 104 //average height

// #define COLS 120 //average width
// #define ROWS 140 //average height

std::vector<cv::String> loadTestData(cv::String testFolderPath) { // "/home/yordan/Documents/Courses/TDCV/hw_2/data/task3/test"

	std::cout << "Loading test data...";
	std::vector<cv::String> filenames;

	cv::glob(testFolderPath, filenames);

	std::cout << "completed" << std::endl;
	return filenames;
}


std::vector<cv::String> getRandomTrainData (std::vector<int> &labels) {
	
	std::vector<cv::String> randomData, allTrainData;

	cv::glob("data/train", allTrainData, true);

	std::random_device rd; // obtain a random number from hardware
	std::mt19937 eng(rd()); // seed the generator
	std::uniform_int_distribution<> distr(0, allTrainData.size() - 1); // define the range
	std::vector<int> randomIdx;
	int randomNumber = distr(eng);

	for(int i = 0; i < allTrainData.size(); ++i)
		randomIdx.push_back(distr(eng));

	for (auto idx = 0; idx < randomIdx.size(); idx++)
		randomData.push_back(allTrainData[randomIdx[idx]]);

	for (auto idx = 0; idx < randomIdx.size(); idx++)
		labels.push_back(allTrainData[randomIdx[idx]][12] - '0');
	
	return randomData;
}



std::vector<Window> createWindows(cv::Mat img) {
	
	std::vector<Window> windows;

	int windowRows = 104; // width
	int windowCols = 104; // height
	int StepSlide = 10;   // step of each window

	for (int row = 0; row <= img.rows - windowRows; row += StepSlide) {
		for (int col = 0; col <= img.cols - windowCols; col += StepSlide) {
			Window windowObj;

			cv::Rect windowRect(col, row, windowRows, windowCols);
			cv::Mat roi = img(windowRect);

			windowObj.setBoundaries(windowRect);
			windowObj.setMatrix(roi);

			windows.push_back(windowObj);
		}
	}

	return windows;
}


HOGDescriptor hogDescpObj() {

	HOGDescriptor hog_obj;
	hog_obj.setWinSize(cv::Size(COLS, ROWS));                       // cv::Size(136, 128)
	hog_obj.setBlockSize(cv::Size((COLS / 8) * 2, (ROWS / 8) * 2)); // cv::Size(34, 32)
	hog_obj.setBlockStep(cv::Size((COLS / 8), (ROWS / 8)));         // cv::Size(17, 16)
	hog_obj.setCellSize(cv::Size((COLS / 8), (ROWS / 8)));          // cv::Size(17, 16)
	hog_obj.setNbins(9);

	return hog_obj;
}

std::vector<float> calculateHog(cv::Mat file) {

	HOGDescriptor hog_obj = hogDescpObj();
	return hog_obj.detectHOGDescriptor(file, cv::Size(COLS, ROWS), false);
}


cv::Mat convertHogToTestData(std::vector<float> descriptor) {
	
	int featureSize = descriptor.size();
	cv::Mat query(1, featureSize, CV_32FC1);
	
	for(int i = 0; i < featureSize; i++)
		query.at<float>(i) = descriptor[i];

	return query;
}


bool sortByConf(Window &lhs, Window &rhs) { 
	return lhs.getConfidence() > rhs.getConfidence(); 
}

void drawBoundingBox(Window windowObj0, Window windowObj1, Window windowObj2, cv::String path, int nameName) {

	// std::cout << "HERE2" << std::endl;
	cv::Rect box0 = windowObj0.getBoundaries();
	cv::Rect box1 = windowObj1.getBoundaries();
	cv::Rect box2 = windowObj2.getBoundaries();

	cv::String resultFolder = "data/results/";

	cv::Mat img = cv::imread(path);

	cv::rectangle(img, box0, cv::Scalar(255, 0, 0)); // class 0, blue, jaw
	cv::rectangle(img, box1, cv::Scalar(0, 255, 0)); // class 1, green, camera
	cv::rectangle(img, box2, cv::Scalar(0, 0, 255)); // class 2, red, plastic piece

	cv::putText(img, "class: " + std::to_string(windowObj0.getLabel()) + " " + std::to_string(windowObj0.getConfidence()), cv::Point(box0.x, box0.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 0, 0), 1); //class 0, blue text
	cv::putText(img, "class: " + std::to_string(windowObj1.getLabel()) + " " + std::to_string(windowObj1.getConfidence()), cv::Point(box1.x, box1.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 1); //class 1, green text
	cv::putText(img, "class: " + std::to_string(windowObj2.getLabel()) + " " + std::to_string(windowObj2.getConfidence()), cv::Point(box2.x, box2.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255), 1); //class 2, red  text
	cv::imwrite(resultFolder + std::to_string(nameName) + ".jpg", img);
}

void drawBoundingBox2(std::vector<std::vector<Window>> container, cv::String path, int nameName) {

	cv::String resultFolder = "data/results/";
	cv::Mat img = cv::imread(path);

	for (auto & class_ : container){
		for (auto & window : class_){
			cv::Rect box = window.getBoundaries();
			int label = window.getLabel();
			if (label == 0) {
				cv::rectangle(img, box, cv::Scalar(255, 0, 0));
				cv::putText(img, "class: " + std::to_string(window.getLabel()) + " " + std::to_string(window.getConfidence()), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 0, 0), 1); //class 0, blue text
			}
			if (label == 1) {
				cv::rectangle(img, box, cv::Scalar(0, 255, 0));
				cv::putText(img, "class: " + std::to_string(window.getLabel()) + " " + std::to_string(window.getConfidence()), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 1); //class 1, green text
			}
			if (label == 2) {
				cv::rectangle(img, box, cv::Scalar(0, 0, 255));
				cv::putText(img, "class: " + std::to_string(window.getLabel()) + " " + std::to_string(window.getConfidence()), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255), 1); //class 2, red  text
			}
		}
	}

	imwrite(resultFolder + std::to_string(nameName) + ".jpg", img);

}



void extractBoundingCoords(std::vector<std::vector<Window>> giantContainer) {

	cv::String path = "data/gt_results";

	for (auto & perImageBoxes : giantContainer){
		for (auto & window : perImageBoxes){
			cv::Rect box = window.getBoundaries();
			int label = window.getLabel();

			int top_left_x = box.x;
			int top_left_y = box.y;
			int bottom_right_x = box.x + box.width;
			int bottom_right_ = box.y - box.height;
		}
	}
}		


std::vector<Window> nonMaxSuppress(std::vector<Window> windows, float threshold) {

  std::vector<Window> initial = windows;
  std::vector<Window> result;

	while (!initial.empty()) {
		
		sort(initial.begin(), initial.end(), sortByConf);
		auto best = initial.begin();

		result.push_back(*best);

		initial.erase(best);
		auto it = initial.begin();

		while (it != initial.end()) {

			cv::Rect rectIntersection = best->getBoundaries() & it->getBoundaries();
			cv::Rect rectUnion = best->getBoundaries() | it->getBoundaries();

			float iou = float(rectIntersection.area()) / float(rectUnion.area());

			if (iou > threshold) {
				it = initial.erase(it);
			} else {
				++it;
			}
		}
	}

	return result;
}