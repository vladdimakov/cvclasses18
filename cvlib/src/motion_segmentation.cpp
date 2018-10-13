/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <iostream>

namespace cvlib
{
motion_segmentation::motion_segmentation()
{
	m_isInitialized = false;
}

void motion_segmentation::setThreshold(double threshold)
{
	m_threshold = threshold;
}

void motion_segmentation::apply(cv::InputArray image, cv::OutputArray foregroundMask, double learningRate)
{
	cv::cvtColor(image, m_grayImage, cv::COLOR_BGR2GRAY);
	m_grayImage.convertTo(m_grayImage, CV_32FC1);

	if (!m_isInitialized)
	{
		m_grayImage.copyTo(m_mu);

		cv::Size windowSize(11, 11);

		cv::Mat localMu, localQuadMu;
		cv::blur(m_grayImage, localMu, windowSize);
		cv::blur(m_grayImage.mul(m_grayImage), localQuadMu, windowSize);

		m_sigmaQuad = localQuadMu - localMu.mul(localMu);

		m_isInitialized = true;

		foregroundMask.assign(cv::Mat(m_grayImage.size(), CV_8U, cv::Scalar(0)));
	}
	else
	{
		m_mu = learningRate * m_grayImage + (1 - learningRate) * m_mu;
		m_distance = cv::abs(m_grayImage - m_mu);
		m_sigmaQuad = learningRate * m_distance.mul(m_distance) + (1 - learningRate) * m_sigmaQuad;
		
		cv::sqrt(m_sigmaQuad, m_sigma);

		m_backgroundModel = m_distance / m_sigma;

		foregroundMask.assign(m_backgroundModel > m_threshold);
	}
}

void motion_segmentation::getBackgroundImage(cv::OutputArray backgroundImage) const
{
	backgroundImage.assign(m_backgroundModel);
}
} // namespace cvlib
