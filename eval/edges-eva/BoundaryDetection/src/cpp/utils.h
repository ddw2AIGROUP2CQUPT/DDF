/*
 * Copyright 2016 Arash Akbarinia arash.akbarinia@cvc.uab.es
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This is the supplementary material of our article presented at the
 * IJCV'17 "Feedback and Surround Modulated Boundary Detection
 */

/* inclusion guard */
#ifndef __UTILS_H__
#define __UTILS_H__

#include <opencv2/core/core.hpp>

/*
 * Method definitions.
 */

cv::Mat CentreCircularZero ( cv::Mat SurroundMat, double CentreRadius );

cv::Mat CircularAverage ( double radius );

cv::Mat CircularLocalStdContrast ( cv::Mat InputImage, double SurroundRadius = 2.5 );

cv::Mat NonMaximumSuppression ( cv::Mat InputImage, cv::Mat thetas, double radius = 2.5 );

#endif /* __UTILS_H__ */
