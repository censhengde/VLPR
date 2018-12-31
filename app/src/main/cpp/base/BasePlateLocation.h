//
// Created by 德先生 on 2018/11/27.
//

#ifndef VLPR_BASEPLATELOCATION_H
#define VLPR_BASEPLATELOCATION_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class BasePlateLocation{

public:
    BasePlateLocation();

    virtual ~BasePlateLocation();

protected:
    int verifySizes(RotatedRect rotatedRect);

    //矫正
    void tortuosity(Mat src, vector<RotatedRect> &rects, vector<Mat> &dst_plates);

    void safeRect(Mat src, RotatedRect rect, Rect2f &dst_rect);

    void rotation(Mat src, Mat &dst, Size rect_size, Point2f center, double angle);



};

#endif //VLPR_BASEPLATELOCATION_H
