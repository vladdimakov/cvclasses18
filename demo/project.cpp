/* Demo application for Computer Vision Library.
* @file
* @date 2018-11-05
* @author Anonymous
*/

#include <cvlib.hpp>
#include "utils.hpp"

int project_demo(int argc, char* argv[])
{
	const cv::String keys = "{video          |      | video file           }";
	cv::CommandLineParser parser(argc, argv, keys);
	auto videoFile = parser.get<cv::String>("video");
	
	const auto origin_wnd = "Origin";
	const auto deviation_image_wnd = "Deviation image";
	const auto background_image_wnd = "Background image";
	const auto binary_image_wnd = "Binary image";

	const auto wnd_width = 640;
	const auto wnd_height = 480;
	
	cv::namedWindow(origin_wnd, CV_WINDOW_NORMAL);
	cv::namedWindow(deviation_image_wnd, CV_WINDOW_NORMAL);
	cv::namedWindow(background_image_wnd, CV_WINDOW_NORMAL);
	cv::namedWindow(binary_image_wnd, CV_WINDOW_NORMAL);

	cv::resizeWindow(origin_wnd, wnd_width, wnd_height);
	cv::resizeWindow(deviation_image_wnd, wnd_width, wnd_height);
	cv::resizeWindow(background_image_wnd, wnd_width, wnd_height);
	cv::resizeWindow(binary_image_wnd, wnd_width, wnd_height);
	
	cvlib::Detector detector;
	cv::Mat frame, grayFrame8U, grayFrame32F, deviationImage, backgroundImage, binaryImage;
	cv::Point2f currentOffset;

	const float refreshRate = 0.02f;
	const float deviationFactor = 5.5f;
	const float targetsFactor = 15.0f;
	const float scalingFactor = 20.0f;
	detector.deviationImgFillValue = 256.0f / targetsFactor;

	cv::VideoCapture cap(videoFile);
	if (!cap.isOpened())
		return -1;

	utils::fps_counter fps;
	while (true)
	{
		cap >> frame;
		
		if (frame.empty())
			break;

		cv::cvtColor(frame, grayFrame8U, CV_RGB2GRAY);
		grayFrame8U.convertTo(grayFrame32F, CV_32F);

		currentOffset = detector.calcFrameOffset(grayFrame8U);
		detector.translateAverageBackAndDeviationImg(grayFrame32F, currentOffset);
		detector.calcFrameStaticPartMask(grayFrame32F, deviationFactor);
		detector.calcAverageBackAndDeviationImg(grayFrame32F, refreshRate);
		detector.calcTargetsBinaryFrame(grayFrame32F, targetsFactor);
		
		detector.getDeviationImage(deviationImage);
		detector.getBackgroundImage(backgroundImage);
		detector.getBinaryImage(binaryImage);

		cv::line(frame, cv::Point(frame.cols / 2, 200), cv::Point(frame.cols / 2, frame.rows - 200), cv::Scalar(0, 0, 255), 2, 8);
		utils::put_fps_text(frame, fps);
		
		cv::imshow(origin_wnd, frame);
		cv::imshow(deviation_image_wnd, deviationImage);
		cv::imshow(background_image_wnd, backgroundImage);
		cv::imshow(binary_image_wnd, binaryImage);

		char key = (char)cv::waitKey(1); 
		if (key == 27)
		{
			break;
		}
		else if (key == ' ')
		{
			detector.needToInit = true;
			continue;
		}
	}

	cv::destroyWindow(origin_wnd);
	cv::destroyWindow(deviation_image_wnd);
	cv::destroyWindow(background_image_wnd);
	cv::destroyWindow(binary_image_wnd);

	return 0;
}
