# Object detection and classification on calculated Histogram of Oriented Gradients via Random Forests

The algorithm is trained on 3+1 different classes of objects, where the fourth one is the background. To train the random forest, each tree is trained with an ensemble of images (bagging). For each training image several augmentations are created in order to impose more diversity. After the training is complete, the algorithm analyses a set of test images, by utisiling the sliding window approach. Each window is ran through the forest and assigned a confidence factor regarding which class it belongs to. Thereafter, the algorithm has two modes of drawing the bounding boxes. First one is to pick the bounding box of each class with the highest score, while the other one is non maximum supression. Only the bounding boxes of the objects are drawn. The results can be seen in 'data/results'

### Prerequisites
- C++11
- OpenCV 4.1+

### Parameters

- k: number of trees of a given forest

- d: maximum depth of each tree in forest

- folds: number of CV folds (can only be set through .cpp files)

- min_count: minimum training samples for each tree (can only be set through .cpp files)

Typical parameters are k = 50, d = 1000.

## Getting Started
To run algorithm:

1) Type 

```
make
```
to compile "./main"

2) Run 

> ./main <*number of trees*> <*maximum depth*>

- training data is in 'data/train/\*'

- results can be found in 'data/results'

- ground truth bounding boxes are located in 'data/gt'

## Example segmentations

![example](https://github.com/ydimitrov/object-detection-hog-random-forest/blob/master/data/example.png?raw=true)

## Authors
Yordan Dimitrov
