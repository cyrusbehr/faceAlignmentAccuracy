//#define TF_DLIB
#define OPENCV_DEBUG
//#define TF_MTCNN_CPP_ORIGINAL
#define TF_MTCNN_CPP_NEW

#include <iostream>
#include <experimental/filesystem>
#include <functional>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>

#include "opencv2/opencv.hpp"
#ifdef TF_MTCNN_CPP_ORIGINAL
#include "mtcnn_original/mtcnn.hpp"
#endif

#ifdef TF_MTCNN_CPP_NEW
#include "mtcnn_new/mtcnn.h"
#endif

#ifdef TF_DLIB
#include "dlib/opencv/cv_image.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
#endif

namespace fs = std::experimental::filesystem;
typedef std::chrono::high_resolution_clock Clock;

template <typename T>
T getLargest(const std::vector<T>& tVec, std::function<int(T)> getArea) {
    auto maxItr = std::max_element(tVec.begin(), tVec.end(), [getArea](const T& a, const T& b){
        return getArea(a) < getArea(b);
    });

    return *maxItr;
}

int main() {
    cv::VideoCapture cap("../data/vid.mp4");
    if (!cap.isOpened()) {
        std::cout << "Unable to open video file\n";
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);

    // Load the model files
#ifdef TF_DLIB
    // Load the dlib model
    std::string modelPath = "../models/dlib/shape_predictor_5_face_landmarks.dat";
    dlib::shape_predictor shapePredictor;
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::deserialize(modelPath) >> shapePredictor;
    std::function<int(dlib::rectangle rect)> byArea = [](dlib::rectangle rect){return static_cast<int>(rect.area());};
#endif

#if defined(TF_MTCNN_CPP_ORIGINAL) || defined(TF_MTCNN_CPP_NEW)
    MTCNN mtcnn;
#endif

    // Compare for 10,000 images
    const size_t numIts = 10000;
    size_t numImgProcessed = 0;
    size_t numFaceFound = 0;
    double totalTime = 0;
    double totalAngleAbs = 0;
    double totalTxAbs = 0;
    double totalTyAbs = 0;
    double totalScale = 0;

    while(true) {
        cv::Mat img;
        auto success = cap.read(img);
        if (!success) {
            break;
        }

        std::cout << ++numImgProcessed << std::endl;

#ifdef TF_DLIB
        dlib::cv_image<dlib::rgb_pixel> cvImg(img);
        auto t1 = Clock::now();

        // Get the bounding box
        std::vector<dlib::rectangle> dets = detector(cvImg);

        if (dets.empty()) {
            continue;
        }

        // Get the landmarks
        const auto largestBBox = getLargest(dets, byArea);
        dlib::full_object_detection shape = shapePredictor(cvImg, largestBBox);
        auto t2 = Clock::now();

        if (!shape.num_parts()) {
            continue;
        }

        // Copy over the landmarks to a vector of cv points for getting the affine transform matrix
        std::vector<cv::Point2f> landmarkVec;

        for (size_t j = 0; j < shape.num_parts(); ++j) {
            landmarkVec.emplace_back(shape.part(j).x(), shape.part(j).y());
        }

#endif

#if defined(TF_MTCNN_CPP_ORIGINAL) || defined(TF_MTCNN_CPP_NEW)
        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);
        std::vector<Bbox> bboxVec;

        auto t1 = Clock::now();
#endif

#ifdef TF_MTCNN_CPP_ORIGINAL
        mtcnn.detectMaxFace(ncnn_img, bboxVec);
#endif

#ifdef TF_MTCNN_CPP_NEW
        mtcnn.detect(ncnn_img, bboxVec);
#endif

#if defined(TF_MTCNN_CPP_ORIGINAL) || defined(TF_MTCNN_CPP_NEW)
        auto t2 = Clock::now();

        if (bboxVec.empty())
            continue;

        Bbox bbox = bboxVec[0];

        if (bboxVec.size() > 1) {
            std::function<int(Bbox)> byArea = [](Bbox box){return static_cast<int>(box.area);};
            bbox = getLargest(bboxVec, byArea);
        }

        std::vector<cv::Point2f> landmarkVec;
        for (auto j = 0; j < 5; ++j) {
            landmarkVec.emplace_back(cv::Point2f(bbox.ppoint[j], bbox.ppoint[j + 5]));
        }

#endif
        ++numFaceFound;

        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        totalTime += time;


        // Display the landmarks in debug mode
#ifdef OPENCV_DEBUG
        for (auto & j : landmarkVec) {
            cv::Point pt(j.x, j.y);
            circle(img, pt, 1, cv::Scalar(0, 255, 0), 3, 3);
        }

        cv::imshow("Original", img);
        cv::waitKey();
#endif
    }

    std::cout << "Total number of images processed: " << numImgProcessed << std::endl;
    std::cout << "Total number of images with faces detected: " << numFaceFound << std::endl;
    std::cout << "Average time: " << totalTime / numFaceFound << " ms" << std::endl;
    return 0;
}