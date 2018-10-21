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
    /// \brief Fabrique method for creating FAST detector
    static cv::Ptr<corner_detector_fast> create();

    /// \see Feature2d::detect
    virtual void detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask = cv::noArray()) override;
};
} // namespace cvlib

#endif // __CVLIB_HPP__
