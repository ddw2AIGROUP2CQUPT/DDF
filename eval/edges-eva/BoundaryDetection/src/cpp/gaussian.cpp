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

#include "gaussian.h"

cv::Mat GaussianFilter2 ( double sigma )
{
  return GaussianFilter2 ( sigma, sigma );
}

cv::Mat GaussianFilter2 ( double sigmax, double sigmay, double meanx, double meany, double theta )
{
  double sigma = std::max ( sigmax, sigmay );
  int KernelSize = CalculateGaussianWidth ( sigma );

  cv::Mat kernel ( KernelSize, KernelSize, CV_32F );

  double centrex = ( KernelSize + 1 ) / 2;
  double centrey = ( KernelSize + 1 ) / 2;
  centrex = centrex + ( meanx * centrex );
  centrey = centrey + ( meany * centrey );

  double SigmaSqrX = pow ( sigmax, 2 );
  double SigmaSqrY = pow ( sigmay, 2 );

  double a =  pow ( cos ( theta ), 2 ) / 2 / SigmaSqrX + pow ( sin ( theta ), 2 ) / 2 / SigmaSqrY;
  double b = -sin ( 2 * theta ) / 4 / SigmaSqrX + sin ( 2 * theta ) / 4 / SigmaSqrY;
  double c =  pow ( sin ( theta ), 2 ) / 2 / SigmaSqrX + pow ( cos ( theta ), 2 ) / 2 / SigmaSqrY;

  for ( int i = 0; i < kernel.rows; i++ )
    {
      for ( int j = 0; j < kernel.cols; j++ )
        {
          float x = i - centrex + 1;
          float y = j - centrey + 1;
          kernel.at<float> ( i, j ) = exp ( - ( a * pow ( x, 2 ) + 2 * b * x * y + c * pow ( y, 2 ) ) );
        }
    }
  return kernel / cv::sum ( kernel ) [0];
}

cv::Mat Gaussian2Gradient1 ( double sigma, double theta, double seta )
{
  int KernelSize = CalculateGaussianWidth ( sigma );

  cv::Mat kernel ( KernelSize, KernelSize, CV_32F );

  double SigmaSqr = pow ( sigma, 2 );
  int width = ( kernel.rows - 1 ) / 2;
  for ( int i = 0; i < kernel.rows; i++ )
    {
      for ( int j = 0; j < kernel.cols; j++ )
        {
          float x1 = j - width;
          float y1 = i - width;
          float x = x1 * cos ( theta ) + y1 * sin ( theta );
          float y = -x1 * sin ( theta ) + y1 * cos ( theta );
          kernel.at<float> ( i, j ) = -x * exp ( - ( x * x + y * y * seta * seta ) / ( 2 * SigmaSqr ) ) / ( CV_PI * SigmaSqr );
        }
    }
  return kernel;
}

int CalculateGaussianWidth ( double sigma, int MaxWidth )
{
  int KernelSize = MaxWidth;

  double threshold = 1e-4;

  double SigmaSqr = pow ( sigma, 2 );

  for ( int i = 1; i < MaxWidth; i++ )
    {
      double current = exp ( - ( i * i ) / ( 2 * SigmaSqr ) );
      if ( current < threshold )
        {
          KernelSize = i - 1;
          break;
        }
    }

  return KernelSize * 2 + 1;
}
