#include "cvlib.hpp"

namespace cvlib
{
	const int CAP_FRAME_WIDTH = 1280;
	const int CAP_FRAME_HEIGHT = 960;

	const int MAX_CORNERS_NUM = 64;
	const int MIN_CORNERS_NUM = 16;

	Detector::Detector()
	{
		needToInit = true;
	}
	
	std::vector<cv::Point2f> Detector::findCorners(cv::Mat grayFrame, int maxCornersNum)
	{
		// Shi-Tomasi Corner Detector
		double qualityLevel = 0.25;//0.01; // ћера "качества" особых точек
		double minDistance = 10; // ћинимальное рассто€ние между особыми точками (в евклидовой мере)
		int blockSize = 3; // –азмер блока дл€ вычислени€ производной ковариационной матрицы в окрестности каждого пиксел€
		bool useHarrisDetector = false; // ѕараметр, указывающий, следует ли использовать детектор ’арриса
		double k = 0.04; // Ёмпирическа€ константа дл€ подсчета собственных значений ([0.04; 0.06]) (дл€ детектора ’арриса)

		std::vector<cv::Point2f> corners;
		goodFeaturesToTrack(grayFrame, corners, maxCornersNum, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);

		return corners;
	}

	void Detector::calcOpticalFlow(cv::Mat prevGrayFrame, cv::Mat currentGrayFrame, std::vector<cv::Point2f> prevPoints, std::vector<cv::Point2f>& currentPoints, std::vector<uchar>& status)
	{
		std::vector<float> err; // ¬ектор погрешностей. “ип меры погрешности может быть установлен соответсвующим флагом
		cv::Size winSize(21, 21); // –азмер окна при поиске
		int maxLevel = 3; // ћаксимальное число уровней пирамид
		cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01); // ѕараметр, указывающий критерии завершени€ алгоритма итеративного поиска сдвига
		int flags = 0; // ‘лаги
		double minEigThreshold = 0.0001; // ѕороговое значение градиента, ниже которого матрица считаетс€ вырожденной

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

	void Detector::makeInitialFrame(cv::Mat prevGrayFrame, std::vector<cv::Point2f>& prevPoints)
	{
		prevGrayFrame.convertTo(averageBackImg, CV_32F);

		deviationImg = cv::Mat(CAP_FRAME_HEIGHT, CAP_FRAME_WIDTH, CV_32F, cv::Scalar(deviationImgFillValue));

		prevPoints = findCorners(prevGrayFrame, MAX_CORNERS_NUM);
	}

	cv::Point2f Detector::calcFrameOffset(cv::Mat& currentGrayFrame)
	{
		std::vector<uchar> status;
		cv::Point2f frameOffset;

		if (needToInit)
		{
			currentGrayFrame.copyTo(prevGrayFrame);
			makeInitialFrame(prevGrayFrame, prevPoints);
			needToInit = false;
			return cv::Point2f(0, 0);
		}

		calcOpticalFlow(prevGrayFrame, currentGrayFrame, prevPoints, currentPoints, status);

		size_t k = 0;
		for (size_t i = 0; i < currentPoints.size(); i++)
		{
			if (status[i])
			{
				currentPoints[k] = currentPoints[i];
				prevPoints[k] = prevPoints[i];
				k++;
			}
		}
		currentPoints.resize(k);
		prevPoints.resize(k);

		if (currentPoints.size() < MIN_CORNERS_NUM)
			needToInit = true;

		frameOffset = findOffsetMedian(prevPoints, currentPoints);

		std::swap(prevPoints, currentPoints);
		cv::swap(prevGrayFrame, currentGrayFrame);

		return frameOffset;
	}

	void Detector::translateAverageBackAndDeviationImg(cv::Mat currentFrame, cv::Point2f currentOffset)
	{
		cv::Mat translatedAverageBackImg, translatedDeviationImg;

		currentFrame.copyTo(translatedAverageBackImg);
		translateFrame(averageBackImg, translatedAverageBackImg, currentOffset);

		translatedDeviationImg = cv::Mat(CAP_FRAME_HEIGHT, CAP_FRAME_WIDTH, CV_32F, cv::Scalar(deviationImgFillValue));
		translateFrame(deviationImg, translatedDeviationImg, currentOffset);

		translatedAverageBackImg.copyTo(averageBackImg);
		translatedDeviationImg.copyTo(deviationImg);
	}

	void Detector::calcFrameStaticPartMask(cv::Mat currentFrame, float deviationFactor)
	{
		currentDeviationImg = abs(currentFrame - averageBackImg);

		frameStaticPartMask = deviationFactor * deviationImg - currentDeviationImg;
		frameStaticPartMask.convertTo(frameStaticPartMask, CV_8U);
	}

	void Detector::calcAverageBackAndDeviationImg(cv::Mat currentFrame, float refreshRate)
	{
		cv::Mat currentDeviationImgStaticPart, currentFrameStaticPart;

		averageBackImg.copyTo(currentFrameStaticPart);
		currentFrame.copyTo(currentFrameStaticPart, frameStaticPartMask);

		averageBackImg = (1 - refreshRate) * averageBackImg + refreshRate * currentFrameStaticPart;

		deviationImg.copyTo(currentDeviationImgStaticPart);
		currentDeviationImg.copyTo(currentDeviationImgStaticPart, frameStaticPartMask);

		deviationImg = (1 - refreshRate) * deviationImg + refreshRate * currentDeviationImgStaticPart;
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

		currentDeviationImg = abs(currentFrame - averageBackImg);

		frameStaticPartMask = targetsFactor * deviationImg - currentDeviationImg;
		frameStaticPartMask.convertTo(frameStaticPartMask, CV_8U);

		currentDeviationImg.convertTo(currentDeviationImg, CV_8U);
		int backgroundBound = getBackgroundBoundOpenCV(currentDeviationImg);

		cv::Mat(CAP_FRAME_HEIGHT, CAP_FRAME_WIDTH, CV_8U, cv::Scalar(0)).copyTo(backgroundBoundMask, currentDeviationImg - backgroundBound);
		cv::Mat(CAP_FRAME_HEIGHT, CAP_FRAME_WIDTH, CV_8U, cv::Scalar(255)).copyTo(frameStaticPartMask, backgroundBoundMask);

		targetsBinaryFrame.setTo(cv::Scalar(255));
		cv::Mat(CAP_FRAME_HEIGHT, CAP_FRAME_WIDTH, CV_8U, cv::Scalar(0)).copyTo(targetsBinaryFrame, frameStaticPartMask);
	}
	
}