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

class Detector
{
public:
	Detector();
	std::vector<cv::Point2f> findCorners(cv::Mat grayFrame, int maxCornersNum);
	void calcOpticalFlow(cv::Mat prevGrayFrame, cv::Mat currentGrayFrame, std::vector<cv::Point2f> prevPoints, std::vector<cv::Point2f>& currentPoints, std::vector<uchar>& status);
	void translateFrame(cv::Mat inputFrame, cv::Mat& outputFrame, cv::Point2f offset);
	cv::Mat subPixTranslateFrameOpenCV(cv::Mat inputFrame, cv::Point2f subPixOffset);
	float findMedian(std::vector<float> value);
	cv::Point2f findOffsetMedian(std::vector<cv::Point2f> prevPoints, std::vector<cv::Point2f> currentPoints);
	void makeInitialFrame(cv::Mat prevGrayFrame, std::vector<cv::Point2f>& prevPoints);
	cv::Point2f calcFrameOffset(cv::Mat& currentGrayFrame);
	void translateAverageBackAndDeviationImg(cv::Mat currentFrame, cv::Point2f currentOffset);
	void calcFrameStaticPartMask(cv::Mat currentFrame, float deviationFactor);
	void calcAverageBackAndDeviationImg(cv::Mat currentFrame, float refreshRate);
	int getBackgroundBoundOpenCV(cv::Mat frame);
	void calcTargetsBinaryFrame(cv::Mat currentFrame, float targetsFactor);
	
	void getDeviationImage(cv::Mat &deviationImage);
	void getBackgroundImage(cv::Mat &backgroundImage);
	void getBinaryImage(cv::Mat &binaryImage);
	
	cv::Mat frameStaticPartMask, averageBackImg, deviationImg, targetsBinaryFrame;
	bool needToInit;
	float deviationImgFillValue;

	cv::Mat prevGrayFrame, currentDeviationImg;
	std::vector<cv::Point2f> prevPoints, currentPoints;
};
} // namespace cvlib

#endif // __CVLIB_HPP__
