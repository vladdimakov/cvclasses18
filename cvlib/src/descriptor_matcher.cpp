/* Descriptor matcher algorithm implementation.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace cvlib
{
void descriptor_matcher::set_ratio(float r)
{
    ratio_ = r;
}

void descriptor_matcher::erase_matches_by_ratio(std::vector<cv::DMatch>& matches)
{
    std::vector<bool> needToErase(matches.size(), false);
    for (int i = 1; i < matches.size(); i++)
    {
        if (matches[i - 1].distance / matches[i].distance <= ratio_)
        {
            needToErase[i - 1] = true;
            needToErase[i] = true;
        }
    }

    auto it = matches.begin();
    int i = 0;
    while (it != matches.end())
    {
        it = (needToErase[i]) ? matches.erase(it) : it + 1;
        i++;
    }
}

void descriptor_matcher::knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k /*unhandled*/,
                                      cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    for (auto& curr_matches : matches)
    {
        if (curr_matches.size() > 1)
        {
            std::sort(curr_matches.begin(), curr_matches.end(),
                      [](const cv::DMatch& obj1, const cv::DMatch& obj2) -> bool { return obj1.distance >= obj2.distance; });

            erase_matches_by_ratio(curr_matches);
        }
    }
}

void descriptor_matcher::radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
                                         cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    if (trainDescCollection.empty())
        return;

    auto& ref_desc = trainDescCollection[0];
    auto test_desc = queryDescriptors.getMat();

    matches.resize(test_desc.rows);

    double distance;
    for (int i = 0; i < test_desc.rows; i++)
    {
        for (int j = 0; j < ref_desc.rows; j++)
        {
            distance = cv::norm(test_desc.row(i) - ref_desc.row(j), cv::NORM_L1);
            if (distance <= maxDistance)
            {
                matches[i].emplace_back(i, j, distance);
            }
        }
    }

    knnMatchImpl(queryDescriptors, matches, 1, masks, compactResult);
}

bool descriptor_matcher::isMaskSupported() const
{
    return false;
}

cv::Ptr<cv::DescriptorMatcher> descriptor_matcher::clone(bool emptyTrainData) const
{
    cv::Ptr<cv::DescriptorMatcher> copy = new descriptor_matcher(*this);
    if (emptyTrainData)
    {
        copy->clear();
    }
    return copy;
}
} // namespace cvlib
