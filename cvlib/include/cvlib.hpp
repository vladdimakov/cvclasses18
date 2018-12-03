/* Computer Vision Functions.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#ifndef __CVLIB_HPP__
#define __CVLIB_HPP__

#include <opencv2/opencv.hpp>

namespace cvlib
{
/// \brief Split and merge algorithm for image segmentation
/// \param image, in - input image
/// \param stddev, in - threshold to treat regions as homogeneous
/// \return segmented image
void split_and_merge(const cv::Mat& image, cv::Mat& splitImage, cv::Mat& mergeImage, double stddev, int minSquare, double meanDeviation,
                     double scaleFactor);

/// \brief Segment texuture on passed image according to sample in ROI
/// \param image, in - input image
/// \param roi, in - region with sample texture on passed image
/// \param eps, in - threshold parameter for texture's descriptor distance
/// \return binary mask with selected texture
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps);

/// \brief Motion Segmentation algorithm
class motion_segmentation : public cv::BackgroundSubtractor
{
    public:
    /// \brief ctor
    motion_segmentation();

    void setThreshold(double threshold);

    /// \see cv::BackgroundSubtractor::apply
    void apply(cv::InputArray image, cv::OutputArray foregroundMask, double learningRate) override;

    /// \see cv::BackgroundSubtractor::BackgroundSubtractor
    void getBackgroundImage(cv::OutputArray backgroundImage) const override;

    private:
    bool m_isInitialized;
    double m_threshold;
    cv::Mat m_grayImage;
    cv::Mat m_mu;
    cv::Mat m_muTmp;
    cv::Mat m_sigma;
    cv::Mat m_sigmaQuad;
    cv::Mat m_sigmaQuadTmp;
    cv::Mat m_distance;
    cv::Mat m_foregroundMask;
};

/// \brief FAST corner detection algorithm
class corner_detector_fast : public cv::Feature2D
{
    public:
    corner_detector_fast();

    /// \brief Fabrique method for creating FAST detector
    static cv::Ptr<corner_detector_fast> create();

    /// \see Feature2d::detect
    virtual void detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask = cv::noArray()) override;

    void setThreshold(int threshold);

    /// \see Feature2d::compute
    virtual void compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) override;

    /// \see Feature2d::detectAndCompute
    virtual void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors,
                                  bool useProvidedKeypoints = false) override;

    /// \see Feature2d::getDefaultName
    virtual cv::String getDefaultName() const override;

    private:
    bool isCorner(const cv::Point2i& point, int step, int pointNumThreshold);
    void generateTestPoints();
    void calcDescriptor(const cv::Point2i& keypoint, cv::Mat& descriptor);

    cv::Mat m_imageForDetector;
    int m_threshold = 30;
    const cv::Point2i m_template[16] = {cv::Point2i(0, -3), cv::Point2i(1, -3),  cv::Point2i(2, -2),  cv::Point2i(3, -1),
                                        cv::Point2i(3, 0),  cv::Point2i(3, 1),   cv::Point2i(2, 2),   cv::Point2i(1, 3),
                                        cv::Point2i(0, 3),  cv::Point2i(-1, 3),  cv::Point2i(-2, 2),  cv::Point2i(-3, 1),
                                        cv::Point2i(-3, 0), cv::Point2i(-3, -1), cv::Point2i(-2, -2), cv::Point2i(-1, -3)};
    cv::Mat m_imageForDescriptor;
    int m_testAreaSize;
    int m_testPointsNum;
    int m_descriptorBytesNum;
    double m_sigma;
    std::vector<std::pair<cv::Point2i, cv::Point2i>> m_testPoints;
};

/// \brief Descriptor matched based on ratio of SSD
class descriptor_matcher : public cv::DescriptorMatcher
{
    public:
    /// \brief ctor
    descriptor_matcher(float ratio = 1.5) : ratio_(ratio)
    {
    }

    /// \brief setup ratio threshold for SSD filtering
    void set_ratio(float r);

    protected:
    /// \see cv::DescriptorMatcher::knnMatchImpl
    virtual void knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k,
                              cv::InputArrayOfArrays masks = cv::noArray(), bool compactResult = false) override;

    /// \see cv::DescriptorMatcher::radiusMatchImpl
    virtual void radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
                                 cv::InputArrayOfArrays masks = cv::noArray(), bool compactResult = false) override;

    /// \see cv::DescriptorMatcher::isMaskSupported
    virtual bool isMaskSupported() const override;

    /// \see cv::DescriptorMatcher::isMaskSupported
    virtual cv::Ptr<cv::DescriptorMatcher> clone(bool emptyTrainData = false) const override;

    private:
    float ratio_;
};

/// \brief Stitcher for merging images into big one
class Stitcher
{
    /// \todo design and implement
};

class AdvancedMotionSegmentation
{
    public:
    AdvancedMotionSegmentation(float refreshRate, float deviationFactor, float targetsFactor, int maxCornersNum, int minCornersNum);
    void setNeedToInit(bool needToInit);
    void process(const cv::Mat& frame);
    void getDeviationImage(cv::Mat& deviationImage);
    void getBackgroundImage(cv::Mat& backgroundImage);
    void getBinaryImage(cv::Mat& binaryImage);

    private:
    void init();
    float getVectMedian(std::vector<float> value);
    cv::Point2f getPointsOffset();
    cv::Point2f getFrameOffset();
    void translateFrame(cv::Mat inputFrame, cv::Mat& outputFrame, cv::Point2f offset);
    void translateAverageBackAndDeviationImg(cv::Point2f frameOffset);
    int getBackgroundFactor();

    bool m_needToInit;
    float m_refreshRate;
    float m_deviationFactor;
    float m_targetsFactor;
    int m_maxCornersNum;
    int m_minCornersNum;
    float m_deviationImgInitValue;

    std::vector<cv::Point2f> m_currPoints;
    std::vector<cv::Point2f> m_prevPoints;

    cv::Mat m_currFrame8u;
    cv::Mat m_currFrame32f;
    cv::Mat m_prevFrame8u;

    cv::Mat m_currDeviationImg;
    cv::Mat m_frameStaticPartMask;
    cv::Mat m_backgroundMask;
    cv::Mat m_currFrameStaticPart;
    cv::Mat m_currDeviationImgStaticPart;
    cv::Mat m_hist;

    cv::Mat m_deviationImg;
    cv::Mat m_translatedDeviationImg;
    cv::Mat m_averageBackImg;
    cv::Mat m_translatedAverageBackImg;
    cv::Mat m_binaryFrame;
};

class Object
{
    public:
    std::vector<cv::Point> contour;
    cv::Rect boundingRect;
    cv::Point centerPosition;
    cv::Point Position;

    Object(std::vector<cv::Point> _contour);
};

void Count(cv::Mat Inp_image, std::vector<cvlib::Object>& objects_, int& Number_);
} // namespace cvlib

#endif // __CVLIB_HPP__
