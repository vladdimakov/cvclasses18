/* Demo application for Computer Vision Library.
* @file
* @date 2018-11-05
* @author Anonymous
*/

#include <cvlib.hpp>
#include "utils.hpp"

int project_demo(int argc, char* argv[])
{
	const cv::String keys = // clang-format off
		"{help h usage ? |      | print this message   }"
		"{video          |      | video file           }";
	// clang-format on

	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("Application name v1.0.0");
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	auto video = parser.get<cv::String>("video");
	cv::VideoCapture cap(video);
	if (!cap.isOpened())
		return -1;

	const auto origin_wnd = "origin";
	const auto demo_wnd = "demo";

	const auto wnd_width = 640;
	const auto wnd_height = 480;
	
	cv::namedWindow(origin_wnd, CV_WINDOW_NORMAL);
	cv::namedWindow(demo_wnd, CV_WINDOW_NORMAL);

	cv::resizeWindow(origin_wnd, wnd_width, wnd_height);
	cv::resizeWindow(demo_wnd, wnd_width, wnd_height);
	
	cvlib::Detector detector;
	cv::Mat frame, grayFrame8U, grayFrame32F;
	cv::Point2f currentOffset;

	const float refreshRate = 0.02f;
	const float deviationFactor = 5.5f;
	const float targetsFactor = 15.0f;
	const float scalingFactor = 20.0f;
	detector.deviationImgFillValue = 256.0f / targetsFactor;

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

		cv::line(frame, cv::Point(frame.cols / 2, 200), cv::Point(frame.cols / 2, frame.rows - 200), cv::Scalar(0, 0, 255), 2, 8);
		utils::put_fps_text(frame, fps);
		cv::imshow(origin_wnd, frame);

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
	cv::destroyWindow(demo_wnd);

	return 0;
}
