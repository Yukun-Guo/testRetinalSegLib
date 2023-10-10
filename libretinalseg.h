#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

/**
 * @brief read OCT file and return a vector of cv::Mat
 * param: octpath: path to the oct file
 * return: vector of cv::Mat
*/
std::vector<cv::Mat> readOCT(std::string);

/**
 * @brief load NN model from modelpath and return a session object 
 * param: modelpath: path to the model file
 * param: threadnum: number of threads to use when running the model on CPU
 * param: gpuindex: index of gpu to use when running the model on GPU
 * return: Ort::Session object
*/
Ort::Session loadNNModel(std::string modelpath, int threadnum=2, int gpuindex=0);
/**
 * @brief read OCT file and return a vector of cv::Mat
 * param: octpath: path to the oct file
 * return: vector of cv::Mat
*/
std::vector<cv::Mat> runSegmentation(Ort::Session& session, std::vector<cv::Mat>& oct);