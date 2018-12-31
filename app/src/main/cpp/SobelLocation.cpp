//
// Created by 德先生 on 2018/11/25.
//

#include "SobelLocation.h"

SobelLocation::SobelLocation() {}

SobelLocation::~SobelLocation() {

}

void SobelLocation::location(const Mat src, vector<Mat> &dst) {
    //预处理 ：去噪 让车牌区域更加突出
    //==========  1.高斯模糊========
    Mat blur;

    /*src:输入图片
     * blur:输出图片
     * Size(5,5):将一张m*n的图片分为一个个5*5的区域，再每个区域中取高斯平均值，
     * 理论上Size(，)区域越大图片越模糊，EasyPR取5*5.其他参数默认即可。
     * */
    GaussianBlur(src, blur, Size(5, 5), 0);

    //=========  2.灰度化(颜色对于计算机来说没用，降噪)=============
    Mat gray;
    //opencv读取到的图片默认是BGR
    cvtColor(blur, gray, COLOR_BGR2GRAY);

    //======3.边缘检测（图片轮廓更清晰，车牌更突出）==============
    Mat sobel_16;
    //CV_16S:一张图片是一个byte数组数据，一个byte即8位，这里要16位是为了给sobel运算预留空间。其他参数默认。
    Sobel(gray, sobel_16, CV_16S, 1, 0);
    //最终仍要转为8位才能显示
    Mat sobel_8;
    convertScaleAbs(sobel_16, sobel_8);

    //=======  4.二值化  =============
    /*灰度图是0到255之间色值的图片
     * 二值化以后只有0和255两种色值，0代表极白，255代表极黑，图片最后是由极白和极黑两种颜色构成。
     * */
    Mat shold;
    /*大律法（算法）:假设阈值为100，大律法大概作用是让色值小于100的都=0，大于100的都等于255.
     *
     * */
    threshold(sobel_8, shold, 0, 255, THRESH_OTSU + THRESH_BINARY);

    //=============  5.闭操作  ============
    // 将相邻的白色区域扩大 连接成一个整体
    Mat close_mat;
    /*element:是Size(17,3)宽17，高3的矩形区域，相邻两个白色区域间隔·如果小于element则连为一体。
     * */
    Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));
    morphologyEx(shold, close_mat, MORPH_CLOSE, element);

    //========  6.查找轮廓  =========

    vector<vector<Point>> contours;//此时contours还不是矩形
    //查找轮廓 提取最外层的轮廓  将结果变成点序列放入 集合
    findContours(close_mat, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE);
    //遍历
    vector<RotatedRect> vec_sobel_roi;//装载矫正后的矩形
    for (vector<Point> point:contours) {
        //将点集合组装成一个个带角度的矩形
        RotatedRect rotatedRect = minAreaRect(point);
        //rectangle(src, rotatedRect.boundingRect(), Scalar(255, 0, 255));
        //进行初步的筛选 把完全不符合的轮廓给排除掉 ( 比如：1x1，5x1000 )
        if (verifySizes(rotatedRect)) {
            vec_sobel_roi.push_back(rotatedRect);

        }

    }

    //==========7.矫正 ==================
    //因为可能斜的，处理扭曲
    //获得候选车牌
//    vector<Mat> plates;
    //1 整个图片 2经过初步赛选的车牌  3 得到的候选车牌
    tortuosity(src, vec_sobel_roi, dst);


    //更进一步的筛选
    //借助svm 进一步筛选


    //imshow("找到轮廓",src);
    blur.release();
    gray.release();
    //......
    sobel_16.release();
    sobel_8.release();
    shold.release();
    close_mat.release();


}

