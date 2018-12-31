//
// Created by 德先生 on 2018/11/28.
//

#include "ColorLocation.h"

ColorLocation::ColorLocation() {}

ColorLocation::~ColorLocation() {

}

void ColorLocation::location(Mat src, vector<Mat> &dst) {
    //1、转成HSV
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

//获取hsv三个通道数
    int channels = hsv.channels();

    /*
     * hsv模型数据存储方式:内存足够下会连续存储为一行，此时高度=1，宽=数据长度
     * 内存不够下存储为多行多列。
     * H，S，V一个分量占一个字节
     * */

    int h = hsv.rows;//数据高度=行数
    int w = hsv.cols * 3;//数据宽度=列数x3个字
    //如果是一行存储·
    if (hsv.isContinuous()) {
        w = w * h;
        h = 1;
    }

    //遍历整张图片（一个像素一个像素遍历，一个hsv=一个像素）
    for (size_t i = 0; i < h; i++) {
        //第i行的数据 一行 hsv的数据 uchar = java byte
        uchar *p = hsv.ptr<uchar>(i);
        //因为在HSV模型中，一行中数据是h s v  h s v.....形式存储的
        //所以 每隔3位是相同分量
        for (size_t j = 0; j < w; j += 3) {
            int h = int(p[j]);
            int s = int(p[j + 1]);
            int v = int(p[j + 2]);

            //是否为蓝色像素点的标记
            bool isBlue = false;
            //若各分量取值在范围内则是蓝色
            if (h >= 100 && h <= 124 && s >= 43 && s <= 255 && v >= 46 && v <= 255) {
                isBlue = true;
            }

            //遍历到蓝色像素就只为白色，凸显出来
            if (isBlue) {
                p[j] = 0;//h
                p[j + 1] = 0;//s
                p[j + 2] = 255;//v

            } else {//不是蓝色像素就置为黑色
                p[j] = 0;//h
                p[j + 1] = 0;//s
                p[j + 2] = 0;//v


            }

        }

    }
    //经过遍历整张图片变为一张黑白图
    //把v分量数据抽出来
    //把h、s、v分离出来
    vector<Mat> hsv_split;
    split(hsv, hsv_split);


//=============往下步骤与sobel定位相同==========================
    //二值化
    Mat shold;
    threshold(hsv_split[2], shold, 0, 255, THRESH_OTSU + THRESH_BINARY);

    //闭操作
    Mat close;
    Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));
    morphologyEx(shold, close, MORPH_CLOSE, element);

    //6、查找轮廓
    //获得初步筛选车牌轮廓================================================================
    //轮廓检测
    vector< vector<Point> > contours;
    //查找轮廓 提取最外层的轮廓  将结果变成点序列放入 集合
    findContours(close, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    //遍历
    vector<RotatedRect> vec_color_roi;
    for (vector<Point> point : contours) {
        RotatedRect rotatedRect = minAreaRect(point);
        //rectangle(src, rotatedRect.boundingRect(), Scalar(255, 0, 255));
        //进行初步的筛选 把完全不符合的轮廓给排除掉 ( 比如：1x1，5x1000 )
        if (verifySizes(rotatedRect)) {
            vec_color_roi.push_back(rotatedRect);
        }
    }

    tortuosity(src, vec_color_roi, dst);
    /*for (Mat s : dst) {
        imshow("候选2", s);
        waitKey();
    }*/


}