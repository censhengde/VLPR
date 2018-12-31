//
// Created by 德先生 on 2018/11/28.
//

#include "CarPlateRecgnize.h"

string CarPlateRecgnize::ZHCHARS[] = {"川", "鄂", "赣", "甘", "贵", "桂", "黑", "沪", "冀", "津", "京", "吉",
                                      "辽", "鲁", "蒙", "闽", "宁", "青", "琼", "陕", "苏", "晋", "皖", "湘",
                                      "新", "豫", "渝", "粤", "云", "藏", "浙"};
char CarPlateRecgnize::CHARS[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
                                  'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
                                  'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};


//参数：训练模型
CarPlateRecgnize::CarPlateRecgnize(const char *svm_model, const char *ann_ch_path,
                                   const char *ann_path) {
    sobelLocation = new SobelLocation();
    colorLocation = new ColorLocation();
//加载车牌训练模型
    svm = SVM::load(svm_model);
    //参数1的宽-参数2的宽 结果与参数3的余数为0  高也一样
    svmHog = new HOGDescriptor(Size(128, 64), Size(16, 16), Size(8, 8), Size(8, 8), 3);

    //创建特征提取对象
    annHog = new HOGDescriptor(Size(32, 32), Size(16, 16), Size(8, 8), Size(8, 8), 3);
    //加载汉字训练模型
    annCh = ANN_MLP::load(ann_ch_path);
    //加载数字，字母训练模型
    ann = ANN_MLP::load(ann_path);


}

CarPlateRecgnize::~CarPlateRecgnize() {
    //释放
    if (!sobelLocation) {
        delete sobelLocation;
        sobelLocation = 0;
    }

    if (!colorLocation) {
        delete colorLocation;
        colorLocation = 0;
    }
    svm->clear();
    svm.release();
    if (svmHog) {
        delete svmHog;
        svmHog = 0;
    }
    if (annHog) {
        delete annHog;
        annHog = 0;
    }
    annCh->clear();
    annCh.release();
    ann->clear();
    ann.release();

}



//入口：传入车图片，返回车牌信息
string CarPlateRecgnize::plateRegnize(Mat src) {
    //sobel定位结果
    vector<Mat> sobel_plates;
    sobelLocation->location(src, sobel_plates);

    //颜色定位
    vector<Mat> color_plates;
    colorLocation->location(src, color_plates);

    //整合两种定位的结果
    vector<Mat> plates;
    plates.insert(plates.end(), sobel_plates.begin(), sobel_plates.end());
    plates.insert(plates.end(), color_plates.begin(), color_plates.end());


//=========== 用SVM进行评测（分两步：一，提取HOG特征矩阵，二，将特征矩阵交给SVM评测） ===============
    int index = -1;
    float minScore = FLT_MAX;//svm内置的最大值评分，做对比用。
    for (int i = 0; i < plates.size(); ++i) {

        Mat plate = plates[i];
        //抽取车牌特征 HOG
        // 灰度化
        Mat gray;
        cvtColor(plate, gray, COLOR_BGR2GRAY);
        //二值化
        Mat shold;
        threshold(gray, shold, 0, 255, THRESH_OTSU + THRESH_BINARY);

        //提取HOG特征
        Mat features;
        getHogFeatures(svmHog, shold, features);

        //把特征置为一行
        Mat samples = features.reshape(1, 1);
        //RAW_OUTPUT:让svm 给出一个评分
        float score = svm->predict(samples, noArray(), StatModel::Flags::RAW_OUTPUT);
        //比较评测分数，取最小值对象
        if (score < minScore) {
            minScore = score;//刷新minScore.
            index = i;
        }

        //释放
        gray.release();
        shold.release();
        features.release();
        samples.release();
    }

    //将评测结果克隆给dst;
    Mat dst;
    if (index >= 0) {
        dst = plates[index].clone();
    }

    //释放
    for (Mat p : plates) {
        p.release();
    }


//============== 识别 （ann:神经网络） ======================

    //灰度化
    Mat plate_gray;
    cvtColor(dst, plate_gray, COLOR_BGR2GRAY);

    //二值化
    Mat plate_shold;
    threshold(plate_gray, plate_shold, 0, 255, THRESH_OTSU + THRESH_BINARY);

    //去掉车牌上的两颗钉子
    clearFixPoint(plate_shold);

    //字符分割
    vector<vector<Point> > contours;
    //查找轮廓 提取最外层的轮廓  将结果变成点序列放入 集合
    findContours(plate_shold, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    vector<Rect> charVec;
    for (vector<Point> point:contours) {
        //将point组装得到轮廓清晰的矩形
        Rect rect = boundingRect(point);
        //抠图
        Mat p = plate_shold(rect);

        // 进行初步的筛选
        if (verityCharSize(p)) {

            charVec.push_back(rect);
        }
    }
        //集合中仍然会存在 非字符矩形
        //对集合中的矩形按照x进行一下排序，保证它们是从左到右的顺序
        sort(charVec.begin(), charVec.end(), [](const Rect &r1, const Rect &r2) {
            return r1.x < r2.x;
        });

        //汉字比较特殊
        // 如何拿汉字的矩形：获取城市字符的轮廓所在集合的下标 比如湘A ，那么A就是城市字符 代表长沙
        int cityIndex = getCityIndex(charVec);

        //通过城市的下标 判断获取汉字轮廓
        Rect chineseRect;
        getChineseRect(charVec[cityIndex], chineseRect);

        //包含了所有待识别的字符图片
        vector<Mat> plateChar;
        //抠出汉字放入plateChar
        plateChar.push_back(plate_shold(chineseRect));

        //遍历抠出汉字后面的字符
        int cout = 0;
        for (size_t i = cityIndex; i < charVec.size(), cout < 6; cout++, ++i) {
            plateChar.push_back(plate_shold(charVec[i]));

        }




    //将得到的plateChar集合拿去识别
    string plate_str;
    predict(plateChar, plate_str);

    plate_gray.release();
    plate_shold.release();

    return plate_str;


}







/*
 * 提取HOG特征
*/

void CarPlateRecgnize::getHogFeatures(HOGDescriptor *svmHog, Mat src, Mat &out) {

    //重新定义大小 缩放 提取特征的时候数据需要为  ：CV_32S 有符号的32位数据
    Mat trainImg = Mat(svmHog->winSize, CV_32S);
    resize(src, trainImg, svmHog->winSize);


    //计算特征 获得float集合
    vector<float> d;
    svmHog->compute(trainImg, d, Size(8, 8));

    //得出HOG特征矩阵
    Mat features(d);
    //将特征矩阵复制给输出矩阵
    features.copyTo(out);

    //释放
    features.release();
    trainImg.release();

}

/*
 * 去牌钉
 * */
void CarPlateRecgnize::clearFixPoint(Mat& src) {
    //经调整，预留最大眺变次数是10最合适（因为两颗牌钉之间本来最大跳变次数是4，但可能由于掉漆出现若干个白点）
    int maxChange = 10;
    //一个集合统计每一行的跳变次数
    vector<int> c;
    for (size_t i = 0; i < src.rows; i++)
    {
        //记录这一行的改变次数
        int change = 0;
        for (size_t j = 0; j < src.cols - 1; j++)
        {
            //根据坐标获得像素值
            char p = src.at<char>(i, j);
            //当前的像素点与下一个像素点值是否相同（换句话说就是颜色是否相同）
            if (p != src.at<char>(i, j + 1)) {//如果不同
                change++;
            }
        }
        c.push_back(change);
    }
    for (size_t i = 0; i < c.size(); i++)
    {
        //取出每一行的改变次数
        int change = c[i];
        //如果小与max ，则可能就是干扰点所在的行
        if (change <= maxChange) {
            //把这一行都抹黑
            for (size_t j = 0; j < src.cols; j++)
            {
                src.at<char>(i, j) = 0;
            }
        }
    }
}

/*
 * 初步筛选字符轮廓
 * */
int CarPlateRecgnize::verityCharSize(Mat src) {
    //最理想情况 车牌字符的标准宽高比
    float aspect = 45.0f / 90;
    // 当前获得矩形的真实宽高比
    float realAspect = (float)src.cols / (float)src.rows;
    //最小的字符高
    float minHeight = 10.0f;
    //最大的字符高
    float maxHeight = 35.0f;
    //1、判断高符合范围  2、宽、高比符合范围
    //最大宽、高比 最小宽高比
    float error = 0.7f;
    float maxAspect = aspect + aspect * error;
    float minAspect = aspect - aspect * error;

    if (realAspect  >= minAspect && realAspect <= maxAspect && src.rows >= minHeight && src.rows <= maxHeight)
    {
        return 1;
    }
    return 0;
}

/*
 * 获取城市字符的索引
 * */
int CarPlateRecgnize::getCityIndex(vector<Rect> src) {
    int cityIndex = 0;
    //循环集合
    for (size_t i = 0; i < src.size(); i++)
    {
        Rect rect = src[i];
        //获得矩形
        //把车牌区域划分为7个字符
        //如果当前获得的矩形 它的中心点 比 1/7 大，比2/7小，那么就是城市的轮廓
        int midX = rect.x + rect.width / 2;
        if (midX < 136 / 7 * 2 && midX > 136 / 7) {
            cityIndex = i;
            break;
        }
    }

    return cityIndex;
}

/*
 * 根据城市字符矩形获取它前面的汉字字符矩形（计算坐标）
 * */
void CarPlateRecgnize::getChineseRect(Rect city, Rect& chineseRect) {
    //把宽度稍微扩大一点
    float width = city.width * 1.15f;
    //城市轮廓的x坐标
    int x = city.x;

    //x ：当前汉字后面城市轮廓的x坐标
    //减去城市的宽
    int newX = x - width;
    chineseRect.x = newX >= 0 ? newX : 0;
    chineseRect.y = city.y;
    chineseRect.width = width;
    chineseRect.height = city.height;
}

/*
 * 识别
 * */

void CarPlateRecgnize::predict(vector<Mat> vec,string& result) {
    for (size_t i = 0; i < vec.size(); i++)
    {
        //提取图片特征去进行识别
        Mat src = vec[i];
        Mat features;
        //提取hog特征
        getHogFeatures(annHog,src,features);
        Mat response;
        Point maxLoc;
        Point minLoc;
        //特征置为一行
        Mat samples = features.reshape(1, 1);
        //i=0，即=false，是汉字矩形，i！=0即=true，是字母或者数字矩形
        if (i) {
            //识别字母与数字
            //ann已经在ann = ANN_MLP::load(ann_path)语句中加载了模型，现在是拿samples去跟模型比对;
            ann->predict(samples,response);
            //获取最大可信度 匹配度最高的属于10个数字+26个大写字母=36种中的哪一个。
            minMaxLoc(response, 0, 0, &minLoc, &maxLoc);
            //跟你训练时候有关（即模型样本顺序要跟数组元素顺序要一致）
            int index = maxLoc.x;
            //拼接字符
            result += CHARS[index];
        } else {
            //识别汉字
            annCh->predict(samples, response);
            //获取最大可信度 匹配度最高的属于31种中的哪一个。
            minMaxLoc(response,0,0,&minLoc, &maxLoc);
            //跟你训练时候有关
            int index = maxLoc.x;
            //识别出来的汉字 拼到string当中去
            result += ZHCHARS[index];
        }
//        cout << "匹配度:" << minLoc.x << endl;
    }
}
