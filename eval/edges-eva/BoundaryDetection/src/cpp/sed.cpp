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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <omp.h>

#include "gaussian.h"
#include "utils.h"

cv::Mat GetOpponentImage ( cv::Mat InputImage )
{
  cv::Mat OpponentImage;

  // gamma correction
  cv::Mat InputImageSqr;
  cv::sqrt ( InputImage, InputImageSqr );

  std::vector<cv::Mat> channels;

  if ( InputImage.channels() == 3 )
    {
      channels.resize ( 6 );
      cv::Mat bgr[3];

      // equilibrium single-opponent cells
      cv::split ( InputImageSqr, bgr );
      channels[0] = ( bgr[2] - bgr[1] );
      channels[1] = ( bgr[0] - 0.5 * ( bgr[1] + bgr[2] ) );
      channels[2] = ( ( bgr[0] + bgr[1] + bgr[2] ) / 3 );

      // imbalanced single-opponent cells
      cv::split ( InputImage, bgr );
      channels[3] = ( bgr[2] - 0.7 * bgr[1] );
      channels[4] = ( bgr[0] - 0.7 * 0.5 * ( bgr[1] + bgr[2] ) );

      // equivalent of the feedback channel to speed up the process
      cv::Mat ImageContrastSqr;
      cv::cvtColor ( InputImageSqr, ImageContrastSqr, CV_BGR2GRAY );
      cv::sqrt ( CircularLocalStdContrast ( ImageContrastSqr ), ImageContrastSqr );
      channels[5] = ( ImageContrastSqr );
    }
  else
    {
      channels.resize ( 2 );
      cv::Mat ImageContrastSqr;
      cv::sqrt ( CircularLocalStdContrast ( InputImageSqr ), ImageContrastSqr );
      channels[0] = ( InputImageSqr );
      channels[1] = ( ImageContrastSqr );
    }

  cv::merge ( channels, OpponentImage );
  return OpponentImage;
}

void SurroundModulation ( cv::Mat InputImage, std::vector<cv::Mat> rfresponse, double GaussianSize, double sigma )
{
  double MaxOrients[rfresponse.size()];
  #pragma omp parallel for
  for ( unsigned int t = 0; t < rfresponse.size(); t++ )
    {
      cv::minMaxLoc ( rfresponse[t], 0, &MaxOrients[t] );
    }
  double MaxAll = *std::max_element ( MaxOrients, MaxOrients + rfresponse.size() );

  double AverageSize = GaussianSize / 2.0;
  cv::Mat SurroundContrast = CircularLocalStdContrast ( InputImage, AverageSize );
  double MaxVal;
  cv::minMaxLoc ( SurroundContrast, 0, &MaxVal );
  SurroundContrast = SurroundContrast.mul ( 1 / MaxVal );
  #pragma omp parallel for
  for ( unsigned int t = 0; t < rfresponse.size(); t++ )
    {
      rfresponse[t] = rfresponse[t].mul ( 1 / MaxVal );
    }

  cv::Mat w11 = 1 - SurroundContrast;
  cv::Mat w12 = -SurroundContrast;
  cv::Mat w21 = -SurroundContrast;
  cv::Mat w22 = 1 - SurroundContrast;

  double ysigma = 0.1;
  double xsigma = 3 * sigma;
  double AxesFactor = 4;

  cv::Mat AverageFilter = CircularAverage ( AverageSize );
  AverageFilter = CentreCircularZero ( AverageFilter, AverageSize / 5 );
  AverageFilter = AverageFilter / cv::sum ( AverageFilter ) [0];

  std::vector<cv::Mat> outresponse ( rfresponse.size() );
  std::vector<cv::Mat> SameOrienResponses ( rfresponse.size() );
  std::vector<cv::Mat> OrthOrienResponses ( rfresponse.size() );

  double nThetas = rfresponse.size();
  double centre;

  #pragma omp parallel for
  for ( unsigned int t = 0; t < rfresponse.size(); t++ )
    {
      cv::Mat tresponse = rfresponse[t];

      double theta1 = double ( t ) * CV_PI / nThetas;
      double theta2 = theta1 + ( CV_PI / 2 );

      cv::Mat SameOrientationGaussian = GaussianFilter2 ( xsigma, ysigma, 0, 0, theta1 );
      centre = std::ceil ( SameOrientationGaussian.rows / 2 );
      SameOrientationGaussian.at<float> ( centre, centre ) = 0;
      cv::Mat OrthogonalOrientationGaussian = GaussianFilter2 ( xsigma / AxesFactor, ysigma, 0, 0, theta2 );
      centre = std::ceil ( OrthogonalOrientationGaussian.rows / 2 );
      OrthogonalOrientationGaussian.at<float> ( centre, centre ) = 0;

      cv::filter2D ( tresponse, SameOrienResponses[t], tresponse.depth(), SameOrientationGaussian );
      cv::filter2D ( tresponse, OrthOrienResponses[t], tresponse.depth(), OrthogonalOrientationGaussian );
    }

  #pragma omp parallel for
  for ( unsigned int t = 0; t < rfresponse.size(); t++ )
    {
      cv::Mat tresponse = rfresponse[t];

      unsigned int o = t + ( nThetas / 2 );
      if ( o >= nThetas )
        {
          o = t - ( nThetas / 2 );
        }

      cv::Mat oresponse = rfresponse[o];

      cv::Mat FullSurroundOrientation;
      cv::filter2D ( tresponse, FullSurroundOrientation, tresponse.depth(), AverageFilter );

      cv::Mat axis11 = SameOrienResponses[t];
      cv::Mat axis21 = OrthOrienResponses[t];
      cv::Mat axis12 = SameOrienResponses[o];
      cv::Mat axis22 = SameOrienResponses[o];

      cv::multiply ( w11, axis11, axis11 );
      cv::multiply ( w12, axis12, axis12 );
      cv::multiply ( w21, axis21, axis21 );
      cv::multiply ( w22, axis22, axis22 );
      outresponse[t] = cv::max ( tresponse + axis11 + axis12 + axis21 + axis22 + 0.5 * FullSurroundOrientation, 0 );
      cv::minMaxLoc ( outresponse[t], 0, &MaxOrients[t] );
    }

  MaxAll = *std::max_element ( MaxOrients, MaxOrients + outresponse.size() );
  #pragma omp parallel for
  for ( unsigned int t = 0; t < rfresponse.size(); t++ )
    {
      cv::Mat tresponse = outresponse[t];
      tresponse = tresponse.mul ( 1 / MaxAll );
      tresponse.copyTo ( rfresponse[t] );
    }
}

std::vector<std::vector<cv::Mat> > DoV1 ( cv::Mat OpponentImage, double LgnSigma, int nangles, int FarSurroundLevels )
{
  // the neurons in V1 are 2 times larger than LGN.
  double lgn2v1 = 2.7;
  double v1sigma = LgnSigma * lgn2v1;


  std::vector<double> thetas ( nangles );
  std::vector<cv::Mat> d1gs ( nangles );
  #pragma omp parallel for
  for ( int t = 0; t < nangles; t++ )
    {
      thetas[t] = double ( t ) * CV_PI / nangles;
      d1gs[t] = Gaussian2Gradient1 ( v1sigma, thetas[t], 0.5 );
    }

  cv::Mat channels[OpponentImage.channels()];
  cv::split ( OpponentImage, channels );

  std::vector<std::vector<cv::Mat> > EdgeImageResponse ( OpponentImage.channels(), std::vector<cv::Mat> ( FarSurroundLevels ) );
  #pragma omp parallel for
  for ( int c = 0; c < OpponentImage.channels(); c++ )
    {
      cv::Mat cim = channels[c];
      cv::Mat gresize;
      for ( int l = 0; l < FarSurroundLevels; l++ )
        {
          cv::Mat clim;
          if ( l > 0 )
            {
              gresize = GaussianFilter2 ( 0.3 * float ( l ) );
              double ResizeFactor = 1 / pow ( 2, float ( l ) );

              cv::filter2D ( cim, clim, cim.depth(), gresize );
              cv::resize ( clim, clim, cv::Size(), ResizeFactor, ResizeFactor );
            }
          else
            {
              cim.copyTo ( clim );
            }

          std::vector<cv::Mat> angles ( nangles );
          for ( int t = 0; t < nangles; t++ )
            {
              cv::Mat cltim;
              cv::filter2D ( clim, cltim, clim.depth(), d1gs[t] );
              cltim = cv::abs ( cltim );

              angles[t] = cltim;
            }
          SurroundModulation ( clim, angles, d1gs[0].rows, v1sigma );
          // resizing them back to the original size
          for ( int t = 0; t < nangles; t++ )
            {
              if ( l > 0 )
                {
                  cv::resize ( angles[t], angles[t], cim.size() );
                  cv::filter2D ( angles[t], angles[t], angles[t].depth(), gresize );
                }
            }
          cv::Mat LevelImage;
          cv::merge ( angles, LevelImage );

          double maxpix;
          cv::minMaxLoc ( LevelImage, 0, &maxpix );
          LevelImage = LevelImage.mul ( 1 / maxpix );

          EdgeImageResponse[c][l] = LevelImage;
        }
    }

  return EdgeImageResponse;
}

cv::Mat DoV2 ( std::vector<std::vector<cv::Mat> > v1response, cv::Mat OpponentImage, int nangles, int FarSurroundLevels )
{
  cv::Mat EdgeImageResponse = cv::Mat::zeros ( v1response[0][0].size(), CV_32F );

  // collapse planes
  std::vector<cv::Mat > planes ( v1response.size() );
  #pragma omp parallel for
  for ( unsigned int c = 0; c < v1response.size(); c++ )
    {
      planes[c] = cv::Mat::zeros ( v1response[c][0].size(), v1response[c][0].type() );
      for ( int l = 0; l < FarSurroundLevels; l++ )
        {
          planes[c] = planes[c] + v1response[c][l].mul ( 1 / double ( l + 1 ) );
        }
      planes[c] = planes[c] / double ( FarSurroundLevels );
    }

  // collapse orientation
  double v2sigma = 0.5 * 2.7 * 2.7;
  double ysigma = v2sigma / 8.0;
  double SurroundEnlarge = 5.0;
  std::vector<cv::Mat > v2response ( v1response.size() );
  std::vector<cv::Mat > v1orients ( v1response.size() );
  #pragma omp parallel for
  for ( unsigned int c = 0; c < v1response.size(); c++ )
    {
      std::vector<cv::Mat > orthogonals ( nangles );
      std::vector<cv::Mat > angles ( nangles );
      cv::split ( planes[c], angles );
      v2response[c] = cv::Mat::zeros ( angles[0].size(), CV_32F );
      v1orients[c] = cv::Mat::zeros ( angles[0].size(), CV_8UC1 );

      for ( int t = 0; t < nangles; t++ )
        {
          double theta = double ( t ) * CV_PI / double ( nangles );
          theta = theta + ( CV_PI / 2.0 );
          cv::Mat v2c;
          cv::filter2D ( angles[t], v2c, angles[t].depth(), GaussianFilter2 ( v2sigma, ysigma, 0, 0, theta ) );
          cv::Mat v2s;
          cv::filter2D ( angles[t], v2s, angles[t].depth(), GaussianFilter2 ( v2sigma * SurroundEnlarge, ysigma * SurroundEnlarge, 0, 0, theta ) );

          cv::Mat v2rf = cv::max ( v2c - 1.0 * v2s, 0 );
          orthogonals[t] = v2rf;
        }

      for ( int i = 0; i < v2response[c].rows; i++ )
        {
          uint8_t *pd = v1orients[c].ptr<uint8_t> ( i );
          float *pv = v2response[c].ptr<float> ( i );
          float *pa[nangles];
          float *po[nangles];
          for ( int t = 0; t < nangles; t++ )
            {
              pa[t] = angles[t].ptr<float> ( i );
              po[t] = orthogonals[t].ptr<float> ( i );
            }

          for ( int j = 0; j < v2response[c].cols; j++ )
            {
              uint8_t MaxInd = 0;
              double MaxVal = pa[0][j];
              for ( int t = 1; t < nangles; t++ )
                {
                  if ( pa[t][j] > MaxVal )
                    {
                      MaxInd = t;
                      MaxVal = pa[t][j];
                    }
                }
              pd[j] = MaxInd;
              pv[j] = po[MaxInd][j];
            }
        }
    }

  // collapse channels
  #pragma omp parallel for
  for ( unsigned int c = 0; c < v1response.size(); c++ )
    {
      EdgeImageResponse += v2response[c];
    }

  double maxpix;
  cv::minMaxLoc ( EdgeImageResponse, 0, &maxpix );
  EdgeImageResponse = EdgeImageResponse.mul ( 1 / maxpix );

  return EdgeImageResponse;
}

int main ( int argc, char **argv )
{
  cv::Mat InputImage = cv::imread ( argv[1], CV_LOAD_IMAGE_COLOR );
  InputImage.convertTo ( InputImage, CV_32F );

  // size of RF in LGN
  double LgnSigma = 0.5;
  cv::filter2D ( InputImage, InputImage, InputImage.depth(), GaussianFilter2 ( LgnSigma ) );

  cv::Mat OpponentImage = GetOpponentImage ( InputImage );

  int nangles = 6;
  int FarSurroundLevels = 4;
  std::vector<std::vector<cv::Mat> > v1response = DoV1 ( OpponentImage, LgnSigma, nangles, FarSurroundLevels );
  cv::Mat v2response = DoV2 ( v1response, OpponentImage, nangles, FarSurroundLevels );


  cv::Mat VisualImage;
  cv::normalize ( v2response, VisualImage, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
  cv::imshow ( "Feedback and Surround Modulation Edge Detection Demo", VisualImage );
  cv::waitKey();

  return 0;
}
