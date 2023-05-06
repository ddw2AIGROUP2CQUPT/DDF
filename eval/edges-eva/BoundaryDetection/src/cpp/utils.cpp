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

#include "utils.h"

#include <opencv2/imgproc/imgproc.hpp>

cv::Mat CentreCircularZero ( cv::Mat SurroundMat, double CentreRadius )
{
  double KernelSize = SurroundMat.rows;
  double centre = std::ceil ( KernelSize / 2 );
  for ( int i = 0; i < KernelSize; i++ )
    {
      for ( int j = 0; j < KernelSize; j++ )
        {
          if ( sqrt ( pow ( i - centre, 2 ) + pow ( j - centre, 2 ) ) <= CentreRadius )
            {
              SurroundMat.at<float> ( i, j ) = 0;
            }
        }
    }

  return SurroundMat;
}

cv::Mat CircularAverage ( double radius )
{
  int KernelSize = radius * 2;
  if ( KernelSize % 2 == 0 )
    {
      KernelSize = KernelSize + 1;
    }

  cv::Mat kernel ( KernelSize, KernelSize, CV_32F );

  double centre = std::ceil ( KernelSize / 2 );
  for ( int i = 0; i < KernelSize; i++ )
    {
      for ( int j = 0; j < KernelSize; j++ )
        {
          if ( sqrt ( pow ( i - centre, 2 ) + pow ( j - centre, 2 ) ) <= radius )
            {
              kernel.at<float> ( i, j ) = 1;
            }
          else
            {
              kernel.at<float> ( i, j ) = 0;
            }
        }
    }

  return kernel / cv::sum ( kernel ) [0];
}

cv::Mat CircularLocalStdContrast ( cv::Mat InputImage, double SurroundRadius )
{
  cv::Mat CircularAvgKernel = CircularAverage ( SurroundRadius );

  cv::Mat ImageContrast;
  cv::filter2D ( InputImage, ImageContrast, InputImage.depth(), CircularAvgKernel );

  cv::pow ( InputImage - ImageContrast, 2, ImageContrast );
  cv::filter2D ( ImageContrast, ImageContrast, ImageContrast.depth(), CircularAvgKernel );

  cv::sqrt ( ImageContrast, ImageContrast );
  return ImageContrast;
}

cv::Mat NonMaximumSuppression ( cv::Mat InputImage, cv::Mat thetas, double radius )
{

}
