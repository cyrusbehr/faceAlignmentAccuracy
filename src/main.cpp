//#define TF_DLIB
//#define OPENCV_DEBUG
//#define TF_MTCNN_CPP_ORIGINAL
#define TF_MTCNN_CPP_NEW

#include <iostream>
#include <experimental/filesystem>
#include <functional>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

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

// Provide the system with images from Megaface
// Compute the landmark locations
// Record the difference between the compute landmark and the ground truth (from the json file).

namespace fs = std::experimental::filesystem;
typedef std::chrono::high_resolution_clock Clock;

// Util class to write csv to file
class FileWriter {
public:
    explicit FileWriter(std::string fileName) {
        m_file.open(fileName);
    }
    ~FileWriter() {
        m_file.close();
    }
    void write(std::string txt) {
        std::string writeStr = txt + "\n";
        m_file << writeStr;
    }
private:
    std::ofstream m_file;
};

// Util function to compare ending of string
bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

// Template function for returning largest item in vector
template <typename T>
T getLargest(const std::vector<T>& tVec, std::function<int(T)> getArea) {
    auto maxItr = std::max_element(tVec.begin(), tVec.end(), [getArea](const T& a, const T& b){
        return getArea(a) < getArea(b);
    });

    return *maxItr;
}

int main() {
    // Select the data we want to use
//    const std::string dataPath = "/home/nchafni/Cyrus/python/readRecFile/data"; // Aligned
    const std::string dataPath = "/home/nchafni/Cyrus/data/facescrub_aligned"; // Megaface
//    const std::string dataPath = "/home/nchafni/Cyrus/data/lfw"; // Unaligned

    // Get all the image files in the directory
    fs::recursive_directory_iterator iter(dataPath);
    fs::recursive_directory_iterator end;
    std::vector<std::string> listOfFiles;
    while (iter != end)
    {
        if (!fs::is_directory(iter->path())) {
            const std::string path = iter->path().string();
            const std::string jsonSuffix = ".json";
            // Only add image files to our list, and not the json files
            if (!hasEnding(path, jsonSuffix)) {
                listOfFiles.push_back(iter->path().string());
            }
        }
        std::error_code ec;
        iter.increment(ec);
        if (ec) {
            std::cerr << "Error While Accessing : " << iter->path().string() << " :: " << ec.message() << '\n';
        }
    }

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

    const size_t numIts = 1000;
    size_t numImgProcessed = 0;
    size_t numFaceFound = 0;
    double totalTime = 0;
    double totalDx1 = 0;
    double totalDx2 = 0;
    double totalDx3 = 0;
    double totalDy1 = 0;
    double totalDy2 = 0;
    double totalDy3 = 0;

    const size_t upperBounds = (numIts < listOfFiles.size() ? numIts : listOfFiles.size());
    for (size_t i = 0; i < upperBounds; ++i) {
        std::cout << i << "/" << upperBounds << std::endl;
        ++numImgProcessed;

        cv::Mat img = cv::imread(listOfFiles[i]);

#ifdef TF_DLIB
        dlib::cv_image<dlib::rgb_pixel> cvImg(img);
        auto t1 = Clock::now();

        // Get the bounding box
        std::vector<dlib::rectangle> dets = detector(cvImg);

        if (dets.empty()) {
            std::cout << "No bbox: " << listOfFiles[i] << std::endl;
            continue;
        }

        // Get the landmarks
        const auto largestBBox = getLargest(dets, byArea);
        dlib::full_object_detection shape = shapePredictor(cvImg, largestBBox);
        auto t2 = Clock::now();

        if (!shape.num_parts()) {
            std::cout << "No landmarks: " << listOfFiles[i] << std::endl;
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

        // Compare the computed landmark locations to the provided locations
        const std::string jsonPath = listOfFiles[i] + ".json";
        std::ifstream jsonFile(jsonPath, std::ifstream::binary);
        json j = json::parse(jsonFile);

        totalDx1 += std::abs(static_cast<double>(j["landmarks"]["0"]["x"]) - landmarkVec[1].x);
        totalDx2 += std::abs(static_cast<double>(j["landmarks"]["1"]["x"]) - landmarkVec[0].x);
        totalDx3 += std::abs(static_cast<double>(j["landmarks"]["2"]["x"]) - landmarkVec[2].x);
        totalDy1 += std::abs(static_cast<double>(j["landmarks"]["0"]["y"]) - landmarkVec[1].y);
        totalDy2 += std::abs(static_cast<double>(j["landmarks"]["1"]["y"]) - landmarkVec[0].y);
        totalDy3 += std::abs(static_cast<double>(j["landmarks"]["2"]["y"]) - landmarkVec[2].y);

#ifdef OPENCV_DEBUG
        cv::Point pt(j["landmarks"]["2"]["x"], j["landmarks"]["2"]["y"]);
        circle(img, pt, 1, cv::Scalar(0, 255, 0), 3, 3);
        cv::Point pt1(j["landmarks"]["0"]["x"], j["landmarks"]["0"]["y"]);
        circle(img, pt1, 1, cv::Scalar(0, 255, 0), 3, 3);
        cv::Point pt2(j["landmarks"]["1"]["x"], j["landmarks"]["1"]["y"]);
        circle(img, pt2, 1, cv::Scalar(0, 255, 0), 3, 3);

        cv::Point pt4(landmarkVec[1].x, landmarkVec[1].y);
        circle(img, pt4, 1, cv::Scalar(255, 0, 0), 3, 3);
        cv::Point pt5(landmarkVec[2].x, landmarkVec[2].y);
        circle(img, pt5, 1, cv::Scalar(255, 0, 0), 3, 3);
        cv::Point pt6(landmarkVec[0].x, landmarkVec[0].y);
        circle(img, pt6, 1, cv::Scalar(255, 0, 0), 3, 3);
        cv::imshow("Megaface Landmarks", img);
        cv::waitKey();
#endif
    }

    std::cout << "Total number of images processed: " << numImgProcessed << std::endl;
    std::cout << "Total number of images with faces detected: " << numFaceFound << std::endl;
    std::cout << "Average time: " << totalTime / numFaceFound << " ms" << std::endl;
    std::cout << "Right eye: " << totalDx1 / numFaceFound << ", " << totalDy1 / numFaceFound << std::endl;
    std::cout << "Left eye: " << totalDx2 / numFaceFound << ", " << totalDy2 / numFaceFound << std::endl;
    std::cout << "Nose: " << totalDx3 / numFaceFound << ", " << totalDy3 / numFaceFound << std::endl;
    return 0;
}