#include <iostream>
#include <experimental/filesystem>
#include <functional>
#include <algorithm>
#include <chrono>

#include "opencv2/opencv.hpp"
#include "dlib/opencv/cv_image.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
// Provide the system with aligned faces from MS1M dataset
// Measure the number of faces detected using the different alignmnet methods
// Record the average rotation, scale, and translation for the different alignment methods
// Which method has the least rotation, scale, and translation?

#define TF_DLIB
#define OPENCV_DEBUG

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
    // Set the reference landmark locations
#ifdef TF_DLIB
    std::vector<cv::Point2f> coord5points = {{79.8561, 51.75}, {65.6543, 52.5136}, {30.3598, 51.9508}, {44.4134, 52.6071}, {54.8819, 79.7481}};
#endif

    const std::string dataPath = "/home/nchafni/Cyrus/python/readRecFile/data"; // Aligned
//    const std::string dataPath = "/home/nchafni/Cyrus/data/lfw"; // Unaligned
    // Get all the image files in the directory
    fs::recursive_directory_iterator iter(dataPath);
    fs::recursive_directory_iterator end;
    std::vector<std::string> listOfFiles;
    while (iter != end)
    {
        if (!fs::is_directory(iter->path())) {
            listOfFiles.push_back(iter->path().string());
        }
        std::error_code ec;
        iter.increment(ec);
        if (ec) {
            std::cerr << "Error While Accessing : " << iter->path().string() << " :: " << ec.message() << '\n';
        }
    }

#ifdef TF_DLIB
    // Load the dlib model
    std::string modelPath = "/home/nchafni/Cyrus/nist-frvt/truefaceCommon/config/shape_predictor_5_face_landmarks.dat";
    dlib::shape_predictor shapePredictor;
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::deserialize(modelPath) >> shapePredictor;
    std::function<int(dlib::rectangle rect)> byArea = [](dlib::rectangle rect){return static_cast<int>(rect.area());};

    // Compare for 10,000 images
    const size_t numIts = 10000;
    size_t numImgProcessed = 0;
    size_t numFaceFound = 0;
    double totalTime = 0;

    const size_t upperBounds = (numIts < listOfFiles.size() ? numIts : listOfFiles.size());
    for (size_t i = 0; i < upperBounds; ++i) {
        std::cout << i << "/" << upperBounds << std::endl;
        ++numImgProcessed;

        cv::Mat img = cv::imread(listOfFiles[i]);
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

        if (!shape.num_parts()) {
            continue;
        }

        ++numFaceFound;

        auto t2 = Clock::now();
        totalTime += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        // Copy over the landmarks to a vector of cv points for getting the affine transform matrix
        std::vector<cv::Point2f> landmarkVec;

        for (size_t j = 0; j < shape.num_parts(); ++j) {
            landmarkVec.emplace_back(shape.part(j).x(), shape.part(j).y());
        }

        // Display the landmarks in debug mode
#ifdef OPENCV_DEBUG
        for (size_t i = 0; i < landmarkVec.size(); ++i) {
            cv::Point pt(landmarkVec[i].x, landmarkVec[i].y);
            circle(img, pt, 1, cv::Scalar(0, 255, 0), 3, 3);
        }

        cv::imshow("Original", img);
#endif
        auto warpMat = cv::estimateAffinePartial2D(landmarkVec, coord5points);
        warpMat.convertTo(warpMat, CV_32FC1);
        cv::Mat alignedFace = cv::Mat::zeros(112, 112, img.type());
        cv::warpAffine(img, alignedFace, warpMat, alignedFace.size());

#ifdef OPENCV_DEBUG
        cv::imshow("Aligned", alignedFace);
        cv::waitKey();
#endif

#endif
    }

    std::cout << "Total number of images processed: " << numImgProcessed << std::endl;
    std::cout << "Total number of images with faces detected: " << numFaceFound << std::endl;
    std::cout << "Average time: " << totalTime / numFaceFound << " ms" << std::endl;

    return 0;
}