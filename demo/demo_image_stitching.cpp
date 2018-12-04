/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

int demo_image_stitching(int argc, char* argv[])
{
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
		return -1;

	const auto main_wnd = "orig";
	const auto demo_wnd = "demo";

	cv::namedWindow(main_wnd);
	cv::namedWindow(demo_wnd);

	auto detector = cvlib::corner_detector_fast::create();
	auto matcher = cvlib::descriptor_matcher();

	int detector_threshold = 30;
	int ratio = 12;
	int max_distance = 20;

	detector->setThreshold(detector_threshold);
	cv::createTrackbar(
		"Threshold", main_wnd, &detector_threshold, 255,
		[](int threshold, void* detector) { ((cvlib::corner_detector_fast*)detector)->setThreshold(threshold); }, (void*)detector);

	matcher.set_ratio(ratio / 10.0f);
	cv::createTrackbar(
		"Ratio * 10", main_wnd, &ratio, 20, [](int ratio, void* matcher) { ((cvlib::descriptor_matcher*)matcher)->set_ratio(ratio / 10.0f); },
		(void*)&matcher);

	cv::createTrackbar("MaxDist", main_wnd, &max_distance, 500);

	/// \brief helper struct for tidy code
	struct img_features
	{
		cv::Mat img;
		std::vector<cv::KeyPoint> corners;
		cv::Mat descriptors;
	};

	img_features ref;
	img_features test;
	std::vector<std::vector<cv::DMatch>> pairs;

	cv::Mat main_frame;
	cv::Mat demo_frame;
	utils::fps_counter fps;
	int pressed_key = 0;

	bool flag = false;
	while (pressed_key != 27) // ESC
	{
		cap >> test.img;

		detector->detect(test.img, test.corners);
		//cv::drawKeypoints(test.img, test.corners, main_frame);

		pressed_key = cv::waitKey(30);
		if (pressed_key == ' ') // space
		{
			ref.img = test.img.clone();
			detector->detectAndCompute(ref.img, cv::Mat(), ref.corners, ref.descriptors);
			flag = true;
		}

		if (ref.corners.empty())
		{
			continue;
		}

		detector->compute(test.img, test.corners, test.descriptors);

		matcher.radiusMatch(test.descriptors, ref.descriptors, pairs, max_distance);

		ref.img.copyTo(main_frame);
		test.img.copyTo(demo_frame);

		std::vector<cv::Point2f> src, dst;

		for (int i = 0; i < pairs.size(); i++)
		{
			if (pairs[i].size() > 0)
			{
				cv::Point2f srcPoint = ref.corners[pairs[i][0].trainIdx].pt;
				cv::Point2f dstPoint = test.corners[i].pt;
				dst.push_back(dstPoint);
				src.push_back(srcPoint);
			}
		}
	

		if (src.size() == 0)
		{
			continue;
		}

		cv::Mat homography = cv::findHomography(src, dst, CV_RANSAC);

		if (homography.empty())
		{
			continue;
		}

		if (homography.at<double>(0, 2) > 0 && homography.at<double>(1, 2) > 0)
		{
			cv::Size sz = ref.img.size();
			sz.width += homography.at<double>(0, 2);
			sz.height += homography.at<double>(1, 2);
			cv::warpPerspective(ref.img, demo_frame, homography, sz);
			test.img.copyTo(demo_frame.rowRange(0, test.img.rows).colRange(0, test.img.cols));
		}
		else if (homography.at<double>(0, 2) <= 0 && homography.at<double>(1, 2) <= 0)
		{
			cv::Size sz = ref.img.size();
			sz.width -= homography.at<double>(0, 2);
			sz.height -= homography.at<double>(1, 2);
			cv::warpPerspective(test.img, demo_frame, homography.inv(), sz);
			ref.img.copyTo(demo_frame.rowRange(0, test.img.rows).colRange(0, test.img.cols));
		}	

		utils::put_fps_text(demo_frame, fps);

		cv::imshow(main_wnd, main_frame);
		cv::imshow(demo_wnd, demo_frame);
	}

	cv::destroyWindow(main_wnd);
	cv::destroyWindow(demo_wnd);

	return 0;
}
