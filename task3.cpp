#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <math.h>
#include <random>
#include <algorithm>

#include "HOGDescriptor.h"
#include "RandomForest.h"
#include "HelperFunctions.h"
#include "Window.h"

std::vector<std::vector<Window>> testForest(int treeCount, int treesMaxDepth) {

	// int treeCount = 10, treesMaxDepth = 10000;
	// float confThresh = 0.15;

	int num_classes = 4, label;
	double confidence = 0.0;
	std::vector<cv::String> testData;
	bool sorting = true;

	std::vector<Window> class0, class1, class2, windows;

	std::vector<std::vector<Window>> perImageBoxes;
	std::vector<std::vector<Window>> giantContainer;

	RandomForest forest = RandomForest(treeCount, treesMaxDepth, 1, 3, num_classes); //number of trees, max depth, cv folds, min sample count, max categories;
	forest.train();

	testData = loadTestData("data/test");
	
	int nameName = 1;
	std::cout << "Detecting images:" << std::endl;
	for (auto & path : testData) {
		
		std::cout << "image: " << nameName << std::endl;
		cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
		windows = createWindows(img);
		for (auto & window : windows){

			forest.predict(convertHogToTestData(calculateHog(window.getMatrix())), confidence, label);

			window.setLabel(label);
			window.setConfidence(confidence);

			/* Here you place the "if" for the threshold, i.e. store only those windows which have higher confidence than threshold */
			if (confidence >= 0.5) {
				if (label == 0) class0.push_back(window);
				if (label == 1) class1.push_back(window);
				if (label == 2) class2.push_back(window);
			}
		}

		if (sorting) {
		
			sort(class0.begin(), class0.end(), sortByConf);
			sort(class1.begin(), class1.end(), sortByConf);
			sort(class2.begin(), class2.end(), sortByConf);
			// std::cout << "HERE1" << std::endl;
			drawBoundingBox(class0[0], class1[0], class2[0], path, nameName);

		} else {

			class0 = nonMaxSuppress(class0, 0.1);
			class1 = nonMaxSuppress(class1, 0.1);
			class2 = nonMaxSuppress(class2, 0.1);

			perImageBoxes.push_back(class0);
			perImageBoxes.push_back(class1);
			perImageBoxes.push_back(class2);
		
			drawBoundingBox2(perImageBoxes, path, nameName);
			perImageBoxes.clear();
		}


		// giantContainer.push_back(perImageBoxes);

		class0.clear();
		class1.clear();
		class2.clear();
		windows.clear();

		nameName++;
	}

	std::cout << "completed" << std::endl;

	return giantContainer;
}

int main(int argc, char** argv) {

	std::vector<std::vector<Window>> wow = testForest(atoi(argv[1]), atoi(argv[2])); // treeCount, treesMaxDepth
	extractBoundingCoords(wow);
	return 0;
}
