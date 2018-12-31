//
// Created by 德先生 on 2018/11/28.
//

#ifndef VLPR_COLORLOCATION_H
#define VLPR_COLORLOCATION_H
#include"base/BasePlateLocation.h"
class ColorLocation:public BasePlateLocation{


public:
    ColorLocation();

    virtual ~ColorLocation();

    void location(Mat src, vector<Mat> &dst);


};

#endif //VLPR_COLORLOCATION_H
