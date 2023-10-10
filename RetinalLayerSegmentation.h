#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <sys/sysinfo.h>
#include <unistd.h>

Ort::Session loadModel(std::string, int t=2, int gpu=0);
std::vector<cv::Mat> runModel(Ort::Session&, std::vector<cv::Mat>&);

// helper functions
std::vector<cv::Mat> readOCT(std::string);
cv::Mat getCategoricalColormap();
std::vector<cv::Mat> generateCurve(std::vector<cv::Mat>);
void normalizeMat(std::vector<cv::Mat> &,std::vector<cv::Mat> &);
std::vector<cv::Mat> drawCurve(std::vector<cv::Mat> &,std::vector<cv::Mat> &);
int retinalLayerSegmentation(std::vector<cv::Mat> &);

cv::Mat argMax(cv::Mat &);
std::vector<unsigned long> getRAMStatus();
cv::Mat myApplyColormap(cv::Mat &, int);
std::vector<char> readONNXFile(const std::string& );
void writeONNXFile(const std::string& , std::vector<char>& );
std::vector<char> encodeONNXFile(std::vector<char>&);
std::vector<char> decodeONNXFile(std::vector<char>&);

std::vector<char> encodeONNXFile_v2(std::vector<char>&, double lv=1);
std::vector<char> decodeONNXFile_v2(std::vector<char>&);
