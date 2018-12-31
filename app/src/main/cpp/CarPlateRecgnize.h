//
// Created by 德先生 on 2018/11/28.
//

#ifndef VLPR_CARPLATERECGNIZE_H
#define VLPR_CARPLATERECGNIZE_H

#include "SobelLocation.h"
#include "ColorLocation.h"
#include <string>
#include <opencv2/ml.hpp>

using namespace std;
using namespace ml;


class CarPlateRecgnize {


public:
    CarPlateRecgnize(const char *svm_model,const char* ann_ch_path, const char* ann_path);

    ~CarPlateRecgnize();

    /*
     * 识别车牌，返回结果给调用者
     * 1.定位
     * 2.识别
     * */
    string plateRegnize(Mat src);

private:
    void getHogFeatures(HOGDescriptor *svmHog, Mat src, Mat &out);
    void clearFixPoint(Mat& src);

    int verityCharSize(Mat src);

    int getCityIndex(vector<Rect> src);
    void getChineseRect(Rect city,Rect& chineseRect);

    void predict(vector<Mat> plateChar,string& result);
private:
    SobelLocation *sobelLocation = 0;
    ColorLocation *colorLocation = 0;
    Ptr<SVM> svm;
    HOGDescriptor *svmHog = 0;
    HOGDescriptor *annHog = 0;

    Ptr<ANN_MLP> annCh;
    Ptr<ANN_MLP> ann;
    static string ZHCHARS[];
    static char CHARS[];

};


#endif //VLPR_CARPLATERECGNIZE_H
