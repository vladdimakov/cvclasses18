/* Car counter.
 * @file
 * @date 2018-09-17
 * @author Pavel Egorov
 */

#include "cvlib.hpp"
#include <vector>

namespace cvlib
{
	Object::Object(std::vector<cv::Point> _contour)
	{
		contour = _contour;

		boundingRect = cv::boundingRect(contour);

		centerPosition.x = (boundingRect.x + boundingRect.x + boundingRect.width) / 2;
		centerPosition.y = (boundingRect.y + boundingRect.y + boundingRect.height) / 2;
		Position.x = boundingRect.x;
		Position.y = centerPosition.y;
	}

	void Count(cv::Mat im_, std::vector<Object>& objects, int& number_)
	{
		std::vector<Object> found_objects;
		cv::Mat structuringElement5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		cv::Mat im = im_;
		cv::dilate(im, im, structuringElement5);
		cv::dilate(im, im, structuringElement5);

		std::vector<std::vector<cv::Point> > contours;

		cv::findContours(im, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		std::vector<std::vector<cv::Point> > convexHulls(contours.size());

		for (unsigned int i = 0; i < contours.size(); i++)
		{
			cv::convexHull(contours[i], convexHulls[i]);
		}
		cv::drawContours(im, convexHulls, -1, cv::Scalar(255.0, 255.0, 255.0), -1);
		for (auto& hull : convexHulls)
		{
			Object new_obj(hull);

			if (new_obj.Position.x < (im.cols) && new_obj.Position.x >(im.cols / 32))
			{
				int flag = 0;
				for (size_t i = 0; i < objects.size(); ++i)
				{
					if (cv::pointPolygonTest(hull, objects[i].centerPosition, false) >= 0)
					{
						objects[i] = new_obj;
						flag = 1;
						found_objects.push_back(objects[i]);
						objects.erase(objects.begin() + i);
					}
					if (flag)
						break;
				}
				if (flag == 0 && new_obj.boundingRect.height > 50 && new_obj.boundingRect.width > 100 && new_obj.Position.x < (im.cols / 2))
					found_objects.push_back(new_obj);
			}
		}
		std::swap(found_objects, objects);

		for (size_t i = 0; i < objects.size(); ++i)
		{
			if (objects[i].Position.x >= (im.cols / 2))
			{
				/*if (objects[i].boundingRect.height > 150) // при "150" 3 из 4 - распознано, 1 - ложно распознано (на ~100 машинах)
				number_ += 2;
				else*/
				number_++;
				printf(" %d \n", number_);
				objects.erase(objects.begin() + i);
			}
		}
	}
} // namespace cvlib
