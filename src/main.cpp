//#define TF_DLIB
#define OPENCV_DEBUG
#define TF_MTCNN_CPP_ORIGINAL
//#define TF_MTCNN_CPP_NEW

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

// Provide the system with aligned faces from MS1M dataset
// Measure the number of faces detected using the different alignment methods
// Record the average rotation, scale, and translation for the different alignment methods
// Which method has the least rotation, scale, and translation?


namespace fs = std::experimental::filesystem;
typedef std::chrono::high_resolution_clock Clock;


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
    FileWriter writer("DLIB_accuracy");
    writer.write("time, angle, tx, ty, scale");
    std::vector<cv::Point2f> coord5points = {{79.8561, 51.75}, {65.6543, 52.5136}, {30.3598, 51.9508}, {44.4134, 52.6071}, {54.8819, 79.7481}};
#endif

#ifdef TF_MTCNN_CPP_ORIGINAL
    FileWriter writer("MTCNN_CPP_ORIGINAL_accuracy");
    writer.write("time, angle, tx, ty, scale");
    std::vector<cv::Point2f> coord5points = {{38.2946, 51.6963}, {73.5318, 51.5014}, {56.0252, 71.7366}, {41.5493, 92.3655}, {70.7299, 92.2041}};
#endif

#ifdef TF_MTCNN_CPP_NEW
    FileWriter writer("MTCNN_CPP_NEW_accuracy");
    writer.write("time, angle, tx, ty, scale");
    std::vector<cv::Point2f> coord5points = {{38.2946, 51.6963}, {73.5318, 51.5014}, {56.0252, 71.7366}, {41.5493, 92.3655}, {70.7299, 92.2041}};
#endif

    // Select the data we want to use
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

    // Load the model files
#ifdef TF_DLIB
    // Load the dlib model
    std::string modelPath = "/home/nchafni/Cyrus/archive-projects/faceAlignmentAccuracy/models/dlib/shape_predictor_5_face_landmarks.dat";
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


        // Display the landmarks in debug mode
#ifdef OPENCV_DEBUG
        for (auto & j : landmarkVec) {
            cv::Point pt(j.x, j.y);
            circle(img, pt, 1, cv::Scalar(0, 255, 0), 3, 3);
        }

        cv::imshow("Original", img);
#endif
        // output of estimateAffinePartial2D described here:
        // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gad767faff73e9cbd8b9d92b955b50062d
        auto warpMat = cv::estimateAffinePartial2D(landmarkVec,
                coord5points,
                cv::noArray(),
                cv::RANSAC,
                100,
                20,
                0.90,
                100);

        const auto tx = warpMat.at<double>(0, 2);
        const auto ty = warpMat.at<double>(1, 2);
        const auto theta = std::atan2(-1 * warpMat.at<double>(0, 1), warpMat.at<double>(0, 0));
        const auto scale = warpMat.at<double>(0, 0) / std::cos(theta);

        totalScale += scale;
        totalTxAbs += std::abs(tx);
        totalTyAbs += std::abs(ty);
        totalAngleAbs += std::abs(theta);

        std::string writeStr = std::to_string(time) + "," + std::to_string(theta) + "," + std::to_string(tx) + "," + std::to_string(ty) + "," + std::to_string(scale);
        writer.write(writeStr);

#ifdef OPENCV_DEBUG
        warpMat.convertTo(warpMat, CV_32FC1);
        cv::Mat alignedFace = cv::Mat::zeros(112, 112, img.type());
        cv::warpAffine(img, alignedFace, warpMat, alignedFace.size());
        cv::imshow("Aligned", alignedFace);
        cv::waitKey();
#endif

    }

    std::cout << "Total number of images processed: " << numImgProcessed << std::endl;
    std::cout << "Total number of images with faces detected: " << numFaceFound << std::endl;
    std::cout << "Average time: " << totalTime / numFaceFound << " ms" << std::endl;
    std::cout << "Average absolute angle [rads]: " << totalAngleAbs / numFaceFound << std::endl;
    std::cout << "Average tx abs: " << totalTxAbs / numFaceFound << " Average ty abs: " << totalTyAbs / numFaceFound << std::endl;
    std::cout << "Average scale: " << totalScale / numFaceFound << std::endl;
    return 0;
}