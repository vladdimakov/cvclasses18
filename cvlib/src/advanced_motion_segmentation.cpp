#include "cvlib.hpp"
#include <functional>

namespace cvlib
{
	Detector::Detector(float refreshRate, float deviationFactor, float targetsFactor, int maxCornersNum, int minCornersNum) :
		m_refreshRate(refreshRate), m_deviationFactor(deviationFactor), m_targetsFactor(targetsFactor), m_maxCornersNum(maxCornersNum),	m_minCornersNum(minCornersNum)
	{
		m_needToInit = true;
		m_deviationImgInitValue = 256.0f / targetsFactor;
	}

	void Detector::init()
	{
		m_prevFrame8u.convertTo(m_averageBackImg, CV_32F);
		m_deviationImg = cv::Mat(m_prevFrame8u.size(), CV_32F, cv::Scalar(m_deviationImgInitValue));

		m_prevPoints.clear();
		goodFeaturesToTrack(m_prevFrame8u, m_prevPoints, m_maxCornersNum, 0.25, 10, cv::Mat(), 3, false, 0.04);
	}

	float Detector::getVectMedian(std::vector<float> vect)
	{
		std::sort(vect.begin(), vect.end(), std::greater_equal<float>());
		return (vect.size() % 2 == 1) ? vect[int(vect.size() / 2)] : (vect[vect.size() / 2 - 1] + vect[vect.size() / 2]) / 2;
	}

	cv::Point2f Detector::getPointsOffset()
	{
		if (m_currPoints.size() != 0)
		{
			std::vector<float> xOffset, yOffset;
			for (int i = 0; i < m_currPoints.size(); i++)
			{
				xOffset.push_back(m_prevPoints[i].x - m_currPoints[i].x);
				yOffset.push_back(m_prevPoints[i].y - m_currPoints[i].y);
			}

			return cv::Point2f(getVectMedian(xOffset), getVectMedian(yOffset));
		}

		return cv::Point2f(0, 0);
	}

	cv::Point2f Detector::getFrameOffset()
	{
		if (m_needToInit)
		{
			m_currFrame8u.copyTo(m_prevFrame8u);
			init();
			m_needToInit = false;
			return cv::Point2f(0, 0);
		}

		std::vector<uchar> status;
		std::vector<float> error;
		cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
		calcOpticalFlowPyrLK(m_prevFrame8u, m_currFrame8u, m_prevPoints, m_currPoints, status, error, cv::Size(21, 21), 3, criteria, 0, 0.0001);

		size_t k = 0;
		for (size_t i = 0; i < m_currPoints.size(); i++)
		{
			if (status[i])
			{
				m_currPoints[k] = m_currPoints[i];
				m_prevPoints[k] = m_prevPoints[i];
				k++;
			}
		}
		m_currPoints.resize(k);
		m_prevPoints.resize(k);

		if (m_currPoints.size() < m_minCornersNum)
			m_needToInit = true;

		cv::Point2f frameOffset = getPointsOffset();

		std::swap(m_prevPoints, m_currPoints);
		cv::swap(m_prevFrame8u, m_currFrame8u);

		return frameOffset;
	}
	
	void Detector::translateFrame(cv::Mat inputFrame, cv::Mat& outputFrame, cv::Point2f offset)
	{
		cv::Point2i frameSize(inputFrame.size().width, inputFrame.size().height);

		cv::Point2i intOffset = cv::Point2i(offset);
		cv::Point2f subPixOffset = offset - cv::Point2f(intOffset);

		if (subPixOffset.x != 0.0f || subPixOffset.y != 0.0f)
		{
			if (subPixOffset.x > 0.5f)
			{
				subPixOffset.x -= 1.0f;
				intOffset.x += 1;
			}
			else if (subPixOffset.x < -0.5f)
			{
				subPixOffset.x += 1.0f;
				intOffset.x -= 1;
			}
			if (subPixOffset.y > 0.5f)
			{
				subPixOffset.y -= 1.0f;
				intOffset.y += 1;
			}
			else if (subPixOffset.y < -0.5f)
			{
				subPixOffset.y += 1.0f;
				intOffset.y -= 1;
			}

			cv::Point2f center = cv::Point2f(frameSize) / 2 + subPixOffset - cv::Point2f(0.5, 0.5);
			getRectSubPix(inputFrame, inputFrame.size(), center, inputFrame);
		}

		if (intOffset.x != 0 || intOffset.y != 0)
		{
			int xOld[2] = { 0, frameSize.x };
			int yOld[2] = { 0, frameSize.y };
			int xNew[2] = { 0, frameSize.x };
			int yNew[2] = { 0, frameSize.y };

			if (intOffset.x > 0)
			{
				xOld[0] = intOffset.x;
				xNew[1] -= intOffset.x;
			}
			else if (intOffset.x < 0)
			{
				xOld[1] += intOffset.x;
				xNew[0] = -intOffset.x;
			}

			if (intOffset.y > 0)
			{
				yOld[0] = intOffset.y;
				yNew[1] -= intOffset.y;
			}
			else if (intOffset.y < 0)
			{
				yOld[1] += intOffset.y;
				yNew[0] = -intOffset.y;
			}

			inputFrame.rowRange(yOld[0], yOld[1]).colRange(xOld[0], xOld[1]).copyTo(outputFrame.rowRange(yNew[0], yNew[1]).colRange(xNew[0], xNew[1]));
		}
		else
		{
			inputFrame.copyTo(outputFrame);
		}
	}

	void Detector::translateAverageBackAndDeviationImg(cv::Point2f frameOffset)
	{
		m_currFrame32f.copyTo(m_translatedAverageBackImg);
		translateFrame(m_averageBackImg, m_translatedAverageBackImg, frameOffset);

		m_translatedDeviationImg = cv::Mat(m_currFrame32f.size(), CV_32F, cv::Scalar(m_deviationImgInitValue));
		translateFrame(m_deviationImg, m_translatedDeviationImg, frameOffset);

		m_translatedAverageBackImg.copyTo(m_averageBackImg);
		m_translatedDeviationImg.copyTo(m_deviationImg);
	}

	int Detector::getBackgroundFactor()
	{
		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };

		calcHist(&m_currDeviationImg, 1, 0, cv::Mat(), m_hist, 1, &histSize, &histRange);

		int startInd = 1;
		if (m_hist.at<float>(0) <= m_hist.at<float>(1))
		{
			for (startInd = 1; startInd < 256; startInd++)
			{
				if (m_hist.at<float>(startInd + 1) < m_hist.at<float>(startInd))
					break;
			}
		}

		int endInd = 1;
		for (endInd = startInd; endInd < 256; endInd++)
		{
			if (m_hist.at<float>(endInd + 1) >= m_hist.at<float>(endInd))
				break;
		}

		return endInd;
	}

	void Detector::setNeedToInit(bool needToInit)
	{
		m_needToInit = needToInit;
	}

	void Detector::process(const cv::Mat &frame)
	{
		if (frame.empty())
			return;
		else if (frame.channels() == 1)
			frame.copyTo(m_currFrame8u);
		else if (frame.channels() == 3)
			cv::cvtColor(frame, m_currFrame8u, cv::COLOR_BGR2GRAY);

		m_currFrame8u.convertTo(m_currFrame32f, CV_32F);

		cv::Point2f frameOffset = getFrameOffset();
		translateAverageBackAndDeviationImg(frameOffset);
		
		m_currDeviationImg = cv::abs(m_currFrame32f - m_averageBackImg);
		m_frameStaticPartMask = (m_deviationFactor * m_deviationImg) > m_currDeviationImg;

		m_averageBackImg.copyTo(m_currFrameStaticPart);
		m_currFrame32f.copyTo(m_currFrameStaticPart, m_frameStaticPartMask);
		
		m_deviationImg.copyTo(m_currDeviationImgStaticPart);
		m_currDeviationImg.copyTo(m_currDeviationImgStaticPart, m_frameStaticPartMask);

		m_averageBackImg = (1 - m_refreshRate) * m_averageBackImg + m_refreshRate * m_currFrameStaticPart;
		m_deviationImg = (1 - m_refreshRate) * m_deviationImg + m_refreshRate * m_currDeviationImgStaticPart;

		m_frameStaticPartMask = (m_targetsFactor * m_deviationImg) > m_currDeviationImg;

		m_currDeviationImg.convertTo(m_currDeviationImg, CV_8U);
		m_backgroundMask = m_currDeviationImg <= getBackgroundFactor();

		m_targetsBinaryFrame = cv::Mat(m_currFrame8u.size(), CV_8U, cv::Scalar(255));
		m_targetsBinaryFrame.setTo(cv::Scalar(0), m_frameStaticPartMask + m_backgroundMask);
	}

	void Detector::getDeviationImage(cv::Mat &deviationImage)
	{
		deviationImage = m_deviationImg * 20;
		deviationImage.convertTo(deviationImage, CV_8U);
	}

	void Detector::getBackgroundImage(cv::Mat &backgroundImage)
	{
		m_averageBackImg.convertTo(backgroundImage, CV_8U);
	}

	void Detector::getBinaryImage(cv::Mat &binaryImage)
	{
		m_targetsBinaryFrame.assignTo(binaryImage);
	}
}