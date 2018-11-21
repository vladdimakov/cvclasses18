#include "cvlib.hpp"

namespace cvlib
{
	const int CAP_FRAME_WIDTH = 1280;
	const int CAP_FRAME_HEIGHT = 960;

	const int MAX_CORNERS_NUM = 64;
	const int MIN_CORNERS_NUM = 16;

	Detector::Detector(float refreshRate, float deviationFactor, float targetsFactor)
	{
		m_needToInit = true;

		m_refreshRate = refreshRate;
		m_deviationFactor = deviationFactor;
		m_targetsFactor = targetsFactor;
		m_deviationImgFillValue = 256.0f / targetsFactor;
	}
	
	void Detector::calcOpticalFlow(cv::Mat prevGrayFrame, cv::Mat currentGrayFrame, std::vector<cv::Point2f> prevPoints, std::vector<cv::Point2f>& currentPoints, std::vector<uchar>& status)
	{
		std::vector<float> err; // Вектор погрешностей. Тип меры погрешности может быть установлен соответсвующим флагом
		cv::Size winSize(21, 21); // Размер окна при поиске
		int maxLevel = 3; // Максимальное число уровней пирамид
		cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01); // Параметр, указывающий критерии завершения алгоритма итеративного поиска сдвига
		int flags = 0; // Флаги
		double minEigThreshold = 0.0001; // Пороговое значение градиента, ниже которого матрица считается вырожденной

		calcOpticalFlowPyrLK(prevGrayFrame, currentGrayFrame, prevPoints, currentPoints, status, err, winSize, maxLevel, criteria, flags, minEigThreshold);
	}

	void Detector::translateFrame(cv::Mat inputFrame, cv::Mat& outputFrame, cv::Point2f offset)
	{
		cv::Point2i intOffset;
		intOffset.x = (int)offset.x;
		intOffset.y = (int)offset.y;

		cv::Point2f subPixOffset;
		subPixOffset.x = offset.x - (float)intOffset.x;
		subPixOffset.y = offset.y - (float)intOffset.y;

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

			inputFrame = subPixTranslateFrameOpenCV(inputFrame, subPixOffset);
		}

		if (intOffset.x != 0 || intOffset.y != 0)
		{
			int xOld[2] = { 0, CAP_FRAME_WIDTH };
			int yOld[2] = { 0, CAP_FRAME_HEIGHT };
			int xNew[2] = { 0, CAP_FRAME_WIDTH };
			int yNew[2] = { 0, CAP_FRAME_HEIGHT };

			if (intOffset.x > 0)
			{
				xOld[0] = intOffset.x;
				xNew[1] = CAP_FRAME_WIDTH - intOffset.x;
			}
			else if (intOffset.x < 0)
			{
				xOld[1] = CAP_FRAME_WIDTH + intOffset.x;
				xNew[0] = -intOffset.x;
			}

			if (intOffset.y > 0)
			{
				yOld[0] = intOffset.y;
				yNew[1] = CAP_FRAME_HEIGHT - intOffset.y;
			}
			else if (intOffset.y < 0)
			{
				yOld[1] = CAP_FRAME_HEIGHT + intOffset.y;
				yNew[0] = -intOffset.y;
			}

			inputFrame.rowRange(yOld[0], yOld[1]).colRange(xOld[0], xOld[1]).copyTo(outputFrame.rowRange(yNew[0], yNew[1]).colRange(xNew[0], xNew[1]));
		}
		else
		{
			inputFrame.copyTo(outputFrame);
		}
	}

	cv::Mat Detector::subPixTranslateFrameOpenCV(cv::Mat inputFrame, cv::Point2f subPixOffset)
	{
		if (subPixOffset.x == 0.0f && subPixOffset.y == 0.0f)
			return inputFrame;

		cv::Mat outputFrame;
		cv::Point2f center;

		center.x = (float)CAP_FRAME_WIDTH / 2 - 0.5f + subPixOffset.x;
		center.y = (float)CAP_FRAME_HEIGHT / 2 - 0.5f + subPixOffset.y;

		getRectSubPix(inputFrame, cv::Size(CAP_FRAME_WIDTH, CAP_FRAME_HEIGHT), center, outputFrame);

		return outputFrame;
	}

	float Detector::findMedian(std::vector<float> value)
	{
		bool exit = false;
		size_t size = value.size();

		while (!exit)
		{
			exit = true;
			for (size_t i = 0; i < size - 1; i++)
			{
				if (value[i] > value[i + 1])
				{
					std::swap(value[i], value[i + 1]);
					exit = false;
				}
			}
		}

		if (size % 2 == 1)
		{
			return value[int(size / 2)];
		}
		else
		{
			return (value[size / 2 - 1] + value[size / 2]) / 2;
		}
	}

	cv::Point2f Detector::findOffsetMedian(std::vector<cv::Point2f> prevPoints, std::vector<cv::Point2f> currentPoints)
	{
		if (currentPoints.size() != 0)
		{
			std::vector<float> xOffset, yOffset;

			for (int i = 0; i < currentPoints.size(); i++)
			{
				xOffset.push_back(prevPoints[i].x - currentPoints[i].x);
				yOffset.push_back(prevPoints[i].y - currentPoints[i].y);
			}

			return cv::Point2f(findMedian(xOffset), findMedian(yOffset));
		}
		else
		{
			return cv::Point2f(0, 0);
		}
	}

	void Detector::init()
	{
		m_prevGrayFrame.convertTo(m_averageBackImg, CV_32F);

		m_deviationImg = cv::Mat(CAP_FRAME_HEIGHT, CAP_FRAME_WIDTH, CV_32F, cv::Scalar(m_deviationImgFillValue));
	
		m_prevPoints.clear();
		goodFeaturesToTrack(m_prevGrayFrame, m_prevPoints, MAX_CORNERS_NUM, 0.25, 10, cv::Mat(), 3, false, 0.04);
	}

	cv::Point2f Detector::calcFrameOffset(cv::Mat& currentGrayFrame)
	{
		if (m_needToInit)
		{
			currentGrayFrame.copyTo(m_prevGrayFrame);
			init();
			m_needToInit = false;
			return cv::Point2f(0, 0);
		}

		std::vector<uchar> status;
		cv::Point2f frameOffset;

		calcOpticalFlow(m_prevGrayFrame, currentGrayFrame, m_prevPoints, m_currentPoints, status);

		size_t k = 0;
		for (size_t i = 0; i < m_currentPoints.size(); i++)
		{
			if (status[i])
			{
				m_currentPoints[k] = m_currentPoints[i];
				m_prevPoints[k] = m_prevPoints[i];
				k++;
			}
		}
		m_currentPoints.resize(k);
		m_prevPoints.resize(k);

		if (m_currentPoints.size() < MIN_CORNERS_NUM)
			m_needToInit = true;

		frameOffset = findOffsetMedian(m_prevPoints, m_currentPoints);

		std::swap(m_prevPoints, m_currentPoints);
		cv::swap(m_prevGrayFrame, currentGrayFrame);

		return frameOffset;
	}

	void Detector::translateAverageBackAndDeviationImg(cv::Mat currentFrame, cv::Point2f currentOffset)
	{
		cv::Mat translatedAverageBackImg, translatedDeviationImg;

		currentFrame.copyTo(translatedAverageBackImg);
		translateFrame(m_averageBackImg, translatedAverageBackImg, currentOffset);

		translatedDeviationImg = cv::Mat(CAP_FRAME_HEIGHT, CAP_FRAME_WIDTH, CV_32F, cv::Scalar(m_deviationImgFillValue));
		translateFrame(m_deviationImg, translatedDeviationImg, currentOffset);

		translatedAverageBackImg.copyTo(m_averageBackImg);
		translatedDeviationImg.copyTo(m_deviationImg);
	}

	void Detector::calcFrameStaticPartMask(cv::Mat currentFrame, float deviationFactor)
	{
		m_currentDeviationImg = abs(currentFrame - m_averageBackImg);

		m_frameStaticPartMask = deviationFactor * m_deviationImg - m_currentDeviationImg;
		m_frameStaticPartMask.convertTo(m_frameStaticPartMask, CV_8U);
	}

	void Detector::calcAverageBackAndDeviationImg(cv::Mat currentFrame, float refreshRate)
	{
		cv::Mat currentDeviationImgStaticPart, currentFrameStaticPart;

		m_averageBackImg.copyTo(currentFrameStaticPart);
		currentFrame.copyTo(currentFrameStaticPart, m_frameStaticPartMask);

		m_averageBackImg = (1 - refreshRate) * m_averageBackImg + refreshRate * currentFrameStaticPart;

		m_deviationImg.copyTo(currentDeviationImgStaticPart);
		m_currentDeviationImg.copyTo(currentDeviationImgStaticPart, m_frameStaticPartMask);

		m_deviationImg = (1 - refreshRate) * m_deviationImg + refreshRate * currentDeviationImgStaticPart;
	}

	int Detector::getBackgroundBoundOpenCV(cv::Mat frame)
	{
		cv::Mat histogram;

		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };

		calcHist(&frame, 1, 0, cv::Mat(), histogram, 1, &histSize, &histRange);

		int startInd = 1;
		if (histogram.at<float>(0) <= histogram.at<float>(1))
		{
			for (startInd = 1; startInd < 256; startInd++)
			{
				if (histogram.at<float>(startInd + 1) < histogram.at<float>(startInd))
				{
					break;
				}
			}
		}

		int endInd = 1;
		for (endInd = startInd; endInd < 256; endInd++)
		{
			if (histogram.at<float>(endInd + 1) >= histogram.at<float>(endInd))
			{
				break;
			}
		}

		return endInd;
	}

	void Detector::calcTargetsBinaryFrame(cv::Mat currentFrame, float targetsFactor)
	{
		cv::Mat backgroundBoundMask = cv::Mat(CAP_FRAME_HEIGHT, CAP_FRAME_WIDTH, CV_8U, cv::Scalar(255));

		m_currentDeviationImg = abs(currentFrame - m_averageBackImg);

		m_frameStaticPartMask = targetsFactor * m_deviationImg - m_currentDeviationImg;
		m_frameStaticPartMask.convertTo(m_frameStaticPartMask, CV_8U);

		m_currentDeviationImg.convertTo(m_currentDeviationImg, CV_8U);
		int backgroundBound = getBackgroundBoundOpenCV(m_currentDeviationImg);

		cv::Mat(CAP_FRAME_HEIGHT, CAP_FRAME_WIDTH, CV_8U, cv::Scalar(0)).copyTo(backgroundBoundMask, m_currentDeviationImg - backgroundBound);
		cv::Mat(CAP_FRAME_HEIGHT, CAP_FRAME_WIDTH, CV_8U, cv::Scalar(255)).copyTo(m_frameStaticPartMask, backgroundBoundMask);

		m_targetsBinaryFrame.setTo(cv::Scalar(255));
		cv::Mat(CAP_FRAME_HEIGHT, CAP_FRAME_WIDTH, CV_8U, cv::Scalar(0)).copyTo(m_targetsBinaryFrame, m_frameStaticPartMask);
	}

	void Detector::setNeedToInit(bool needToInit)
	{
		m_needToInit = needToInit;
	}

	void Detector::process(const cv::Mat &frame)
	{
		cv::Mat grayFrame8U, grayFrame32F;

		cv::cvtColor(frame, grayFrame8U, CV_RGB2GRAY);
		grayFrame8U.convertTo(grayFrame32F, CV_32F);

		cv::Point2f currentOffset = calcFrameOffset(grayFrame8U);
		translateAverageBackAndDeviationImg(grayFrame32F, currentOffset);
		calcFrameStaticPartMask(grayFrame32F, m_deviationFactor);
		calcAverageBackAndDeviationImg(grayFrame32F, m_refreshRate);
		calcTargetsBinaryFrame(grayFrame32F, m_targetsFactor);
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