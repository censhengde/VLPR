//
// Created by 德先生 on 2018/11/25.
//

#ifndef VLPR_CARPLATELOCATION_H
#define VLPR_CARPLATELOCATION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "base/BasePlateLocation.h"

using namespace cv;
using namespace std;

class SobelLocation: public BasePlateLocation {

public:
    SobelLocation();

    ~SobelLocation();

    // 1、要定位的图片 2、引用类型 作为定位结果
    void location(const Mat src, vector< Mat> &dst);



};

#endif //VLPR_CARPLATELOCATION_H
