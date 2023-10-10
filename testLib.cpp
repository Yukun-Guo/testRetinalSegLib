#include <iostream>
#include <vector>
#include <string>
#include "libretinalseg.h"
using namespace std;

int main(){

    auto start = std::chrono::high_resolution_clock::now();
    auto octs = readOCT("3f7ad01ba326194ea6c90a44fd74de24_M_FX.foct");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "load oct time: " << elapsed.count() << " s\n";
   
    start = std::chrono::high_resolution_clock::now();
    auto session = loadNNModel("model.enc", 6,1); 
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "load model time: " << elapsed.count() << " s\n";
    
    start = std::chrono::high_resolution_clock::now();
    auto curves = runSegmentation(session, octs);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    // print 
    std::cout << "runSegmentation time: " << elapsed.count() << " s\n";
    return 0;
}

