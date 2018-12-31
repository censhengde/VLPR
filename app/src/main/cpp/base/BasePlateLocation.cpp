//
// Created by 德先生 on 2018/11/27.
//

#include "BasePlateLocation.h"

BasePlateLocation::BasePlateLocation() {}

BasePlateLocation::~BasePlateLocation() {

}

/*verify:核实，查证。
 * */
int BasePlateLocation::verifySizes(RotatedRect rotatedRect) {
    //容错率
    float error = 0.75f;

    //训练时候模型的宽高 136 * 32
    //获得宽高比
    float aspect = float(136) / float(32);

    //最小 最大面积 不符合的丢弃
    //给个大概就行 随时调整
    //尽量给大一些没关系， 这还是初步筛选。
    int min = 20 * 20 * aspect;
    int max = 180 * 180 * aspect;

    //比例浮动 error认为也满足
    //最小宽、高比
    float rmin = aspect - aspect * error;
    //最大的宽高比
    float rmax = aspect + aspect * error;
    //矩形的面积
    float area = rotatedRect.size.height * rotatedRect.size.width;
    //矩形的宽高比例
    float r = (float) rotatedRect.size.width / (float) rotatedRect.size.height;
    if ((area < min || area > max) || (r < rmin || r > rmax));
    {
        return 0;
    }
    return 1;

}

/*矫正
 *
 * */
void BasePlateLocation::tortuosity(Mat src, vector<RotatedRect> &rects, vector<Mat> &dst_plates) {
    //循环要处理的矩形
    for (RotatedRect roi_rect:rects) {
        //矩形宽高比
        float r = (float) roi_rect.size.width / (float) roi_rect.size.height;
        //矩形角度
        float roi_angle = roi_rect.angle;
        //矩形大小
        Size roi_rect_size = roi_rect.size;

        //让rect在一个安全的范围(不能超过src)
        Rect2f rect2f;
        safeRect(src, roi_rect, rect2f);


        //候选车牌
        //抠图  这里不是产生一张新图片 而是在src身上定位到一个Mat 让我们处理
        //数据和src是同一份
        Mat src_rect = src(rect2f);
        //真正的候选车牌
        Mat dst;
        //不需要旋转的 旋转角度小没必要旋转了
        if (roi_angle - 5 < 0 && roi_angle + 5 > 0) {
            dst = src_rect.clone();//直接克隆
        } else {
            //获取src_rect的中心点（相对于它左上角坐标的中心点坐标）
            //相对于roi的中心点 不减去左上角坐标是相对于整个src图的坐标
            //减去左上角则是相对于候选车牌的中心点 坐标
            Point2f roi_ref_center = roi_rect.center - rect2f.tl();
            //校正后的图片
            Mat rotated_mat;
            rotation(src_rect, rotated_mat, roi_rect_size, roi_ref_center, roi_angle);
            dst = rotated_mat;


        }


        //重定义dst大小
        Mat plare_mat;
        plare_mat.create(32, 136, CV_8UC3);
        resize(dst, plare_mat, plare_mat.size());

        //将重定义好的车牌矩阵放入集合
        dst_plates.push_back(plare_mat);
        dst.release();


    }

}

/*获取合法矩形
 * 矩形可能越出src，越界部分切割掉*/

//参数1、图片 2、要处理的矩形(可能超出有效范围) 3、处理之后在有效范围的矩形
void BasePlateLocation::safeRect(Mat src, RotatedRect rect, Rect2f &dst_rect) {

    //转为正常的带坐标的边框(Rect2f:带坐标的矩形)
    Rect2f boundRect = rect.boundingRect2f();
    //左上角坐标(tl_x,tl_y)必须再src内
    float tl_x = boundRect.x > 0 ? boundRect.x : 0;//(类似于坦克大战项目中坦克碰撞判断）
    float tl_y = boundRect.y > 0 ? boundRect.y : 0;

    //右下角坐标(br_x,br_y)
    float br_x = boundRect.x + boundRect.width < src.cols
                 ? boundRect.x + boundRect.width - 1
                 : src.cols - 1;

    float br_y = boundRect.y + boundRect.height < src.rows
                 ? boundRect.y + boundRect.height - 1
                 : src.rows - 1;

    float w = br_x - tl_x;
    float h = br_y - tl_y;
    if (w <= 0 || h <= 0) { return; }

    dst_rect = Rect2f(tl_x, tl_y, w, h);


}

/*矫正
 * */
//参数1、矫正前 2、矫正后 3、矩形的大小 4、矩形中心点坐标  5、角度
void BasePlateLocation::rotation(Mat src, Mat &dst, Size rect_size, Point2f center, double angle) {

    //获得旋转矩阵(参数1源图像中心，2，旋转角度。角度为正值表示向逆时针旋转（坐标原点是左上角）3.缩放比例系数)
    Mat rot_mat = getRotationMatrix2D(center, angle, 1);//该矩阵可理解为是封装了一些旋转参数的一个对象。

    //运用仿射变换
    Mat mat_dst;
    //矫正后 大小会不一样，但是对角线肯定能容纳
    int max = sqrt(pow(src.rows, 2) + pow(src.cols, 2));//宽平方+高平方再开平方=对角线长度
    //参数1源图像 2.输出图像 3旋转参数对象 4输出图像尺寸 5.输出图像的插值方法
    warpAffine(src, mat_dst, rot_mat, Size(max, max), CV_INTER_CUBIC);

    //imshow("旋转前", src);
    //imshow("旋转", mat_rotated);
    //截取 尽量把车牌多余的区域截取掉
    //参数1 源图像 2 获取矩形的大小 3 获取的矩形在源图像中的位置 4 输出图像
    getRectSubPix(mat_dst, Size(rect_size.width, rect_size.height), center, dst);

    //释放
    mat_dst.release();
    rot_mat.release();
}