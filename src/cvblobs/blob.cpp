/************************************************************************
  			Blob.cpp
  			
- FUNCIONALITAT: Implementació de la classe CBlob
- AUTOR: Inspecta S.L.
MODIFICACIONS (Modificació, Autor, Data):

 
FUNCTIONALITY: Implementation of the CBlob class and some helper classes to perform
			   some calculations on it
AUTHOR: Inspecta S.L.
MODIFICATIONS (Modification, Author, Date):

**************************************************************************/


#include "blob.h"
#include <opencv/cxcore.h>
#include <opencv/cv.h>

CBlob::CBlob()
{
	m_area = m_perimeter = -1;
	m_externPerimeter = m_meanGray = m_stdDevGray = -1;
	m_boundingBox.width = -1;
	m_ellipse.size.width = -1;
	m_storage = NULL;
	m_id = -1;
}
CBlob::CBlob( t_labelType id, std::vector<cv::Point2i> const & contour, CvSize originalImageSize )
{
	m_id = id;
	m_area = m_perimeter = -1;
	m_externPerimeter = m_meanGray = m_stdDevGray = -1;
	m_boundingBox.width = -1;
	m_ellipse.size.width = -1;
	m_storage = cvCreateMemStorage();
    m_externalContour = CBlobContour(contour);
	m_originalImageSize = originalImageSize;
}
//! Copy constructor
CBlob::CBlob( const CBlob &src )
{
	m_storage = NULL;
	*this = src;
}

CBlob::CBlob( const CBlob *src )
{
	if (src != NULL )
	{
		m_storage = NULL;
		*this = *src;
	}
}

CBlob& CBlob::operator=(const CBlob &src )
{
	if( this != &src )
	{
		m_id = src.m_id;
		m_area = src.m_area;
		m_perimeter = src.m_perimeter;
		m_externPerimeter = src.m_externPerimeter;
		m_meanGray = src.m_meanGray;
		m_stdDevGray = src.m_stdDevGray;
		m_boundingBox = src.m_boundingBox;
		m_ellipse = src.m_ellipse;
		m_originalImageSize = src.m_originalImageSize;
		
		// clear all current blob contours
		ClearContours();
		
		if( m_storage )
			cvReleaseMemStorage( &m_storage );

		m_storage = cvCreateMemStorage();

        m_externalContour = CBlobContour(src.m_externalContour );

        if( !src.m_externalContour.m_contour.empty() )
        {
            m_externalContour.m_contour = src.m_externalContour.m_contour;
        }
		m_internalContours.clear();

		// copy all internal contours
		if( src.m_internalContours.size() )
		{
			m_internalContours = t_contourList( src.m_internalContours.size() );
			t_contourList::const_iterator itSrc;
			t_contourList::iterator it;

			itSrc = src.m_internalContours.begin();
			it = m_internalContours.begin();

			while (itSrc != src.m_internalContours.end())
			{
                *it = CBlobContour((*itSrc).GetContourPoints());
                if( !(*itSrc).m_contour.empty() )
                    (*it).m_contour = (*itSrc).m_contour;

				it++;
				itSrc++;
			}
		}
	}

	return *this;
}

CBlob::~CBlob()
{
	ClearContours();
	
	if( m_storage )
		cvReleaseMemStorage( &m_storage );
}

void CBlob::ClearContours()
{
	t_contourList::iterator it;

	it = m_internalContours.begin();

	while (it != m_internalContours.end())
	{
		(*it).ResetChainCode();
		it++;
	}	
	m_internalContours.clear();

	m_externalContour.ResetChainCode();
		
}
void CBlob::AddInternalContour( const CBlobContour &newContour )
{
	m_internalContours.push_back(newContour);
}

bool CBlob::IsEmpty()
{
    return GetExternalContour()->m_contour.empty();
}

double CBlob::Area()
{
	double area;
	t_contourList::iterator itContour; 

	area = m_externalContour.GetArea();

	itContour = m_internalContours.begin();
	
	while (itContour != m_internalContours.end() )
	{
		area -= (*itContour).GetArea();
		itContour++;
	}
	return area;
}

double CBlob::Perimeter()
{
	double perimeter;
	t_contourList::iterator itContour; 

	perimeter = m_externalContour.GetPerimeter();

	itContour = m_internalContours.begin();
	
	while (itContour != m_internalContours.end() )
	{
		perimeter += (*itContour).GetPerimeter();
		itContour++;
	}
	return perimeter;

}

int	CBlob::Exterior(IplImage *mask, bool xBorder /* = true */, bool yBorder /* = true */)
{
	if (ExternPerimeter(mask, xBorder, yBorder ) > 0 )
	{
		return 1;
	}
	
	return 0;	 
}

double CBlob::ExternPerimeter( IplImage *maskImage, bool xBorder /* = true */, bool yBorder /* = true */)
{
    std::vector<cv::Point2i> externalPoints;
    cv::Point2i actualPoint, previousPoint;

	bool find = false;
	int i,j;
	int delta = 0;
	
	// it is calculated?
    if( m_externPerimeter != -1 || m_externalContour.GetContourPoints().empty())
	{
		return m_externPerimeter;
	}

	// get contour pixels
    std::vector<cv::Point2i> hull;
    cv::convexHull( m_externalContour.GetContourPoints(), hull);

    // Compute the perimeter
    m_externPerimeter = 0.;
    for(int i=0; i<hull.size()-1; ++i)
    {
        m_externPerimeter += cv::norm(hull[i] - hull[i+1]);
    }

    m_externPerimeter += cv::norm(hull.front() - hull.back());
	
	return m_externPerimeter;
}

//! Compute blob's moment (p,q up to MAX_CALCULATED_MOMENTS)
double CBlob::Moment(int p, int q)
{
	double moment;
	t_contourList::iterator itContour; 

	moment = m_externalContour.GetMoment(p,q);

	itContour = m_internalContours.begin();
	
	while (itContour != m_internalContours.end() )
	{
		moment -= (*itContour).GetMoment(p,q);
		itContour++;
	}
	return moment;
}

double CBlob::Mean( IplImage *image )
{
	// Create a mask with same size as blob bounding box
    cv::Point2i offset;

	GetBoundingBox();
	
	if (m_boundingBox.height == 0 ||m_boundingBox.width == 0 || !CV_IS_IMAGE( image ))
	{
		m_meanGray = 0;
		return m_meanGray;
	}

	// apply ROI and mask to input image to compute mean gray and standard deviation
    cv::Mat mask( cv::Size(m_boundingBox.width, m_boundingBox.height), CV_8UC1, 1);
    mask = cv::Scalar(0);

	offset.x = -m_boundingBox.x;
	offset.y = -m_boundingBox.y;

	// draw contours on mask
    cv::drawContours( mask, m_externalContour.GetContourPoints(), -1, CV_RGB(255,255,255), -1, 8, cv::noArray(),
                      INT_MAX, offset);

	// draw internal contours
	t_contourList::iterator it = m_internalContours.begin();
	while(it != m_internalContours.end() )
	{
        cv::drawContours( mask, (*it).GetContourPoints(), -1, CV_RGB(0, 0, 0), -1, 8, cv::noArray(),
                          INT_MAX, offset);
		it++;
	}

    cv::Scalar mean, std;
	cvSetImageROI( image, m_boundingBox );
    cv::meanStdDev( cv::Mat(image), mean, std, mask );
	
	m_meanGray = mean.val[0];
	m_stdDevGray = std.val[0];

	cvResetImageROI( image );
	return m_meanGray;
}

double CBlob::StdDev( IplImage *image )
{
	// call mean calculation (where also standard deviation is calculated)
	Mean( image );

	return m_stdDevGray;
}

CvRect CBlob::GetBoundingBox() const
{
	// it is calculated?
	if( m_boundingBox.width != -1 )
	{
		return m_boundingBox;
	}
	
    return cv::boundingRect( m_externalContour.GetContourPoints() );
}

CvBox2D CBlob::GetEllipse()
{
	// it is calculated?
	if( m_ellipse.size.width != -1 )
		return m_ellipse;
	
	double u00,u11,u01,u10,u20,u02, delta, num, den, temp;

	// central moments calculation
	u00 = Moment(0,0);

	// empty blob?
	if ( u00 <= 0 )
	{
		m_ellipse.size.width = 0;
		m_ellipse.size.height = 0;
		m_ellipse.center.x = 0;
		m_ellipse.center.y = 0;
		m_ellipse.angle = 0;
		return m_ellipse;
	}
	u10 = Moment(1,0) / u00;
	u01 = Moment(0,1) / u00;

	u11 = -(Moment(1,1) - Moment(1,0) * Moment(0,1) / u00 ) / u00;
	u20 = (Moment(2,0) - Moment(1,0) * Moment(1,0) / u00 ) / u00;
	u02 = (Moment(0,2) - Moment(0,1) * Moment(0,1) / u00 ) / u00;


	// elipse calculation
	delta = sqrt( 4*u11*u11 + (u20-u02)*(u20-u02) );
	m_ellipse.center.x = u10;
	m_ellipse.center.y = u01;
	
	temp = u20 + u02 + delta;
	if( temp > 0 )
	{
		m_ellipse.size.width = sqrt( 2*(u20 + u02 + delta ));
	}	
	else
	{
		m_ellipse.size.width = 0;
		return m_ellipse;
	}

	temp = u20 + u02 - delta;
	if( temp > 0 )
	{
		m_ellipse.size.height = sqrt( 2*(u20 + u02 - delta ) );
	}
	else
	{
		m_ellipse.size.height = 0;
		return m_ellipse;
	}

	// elipse orientation
	if (u20 > u02)
	{
		num = u02 - u20 + sqrt((u02 - u20)*(u02 - u20) + 4*u11*u11);
		den = 2*u11;
	}
    else
    {
		num = 2*u11;
		den = u20 - u02 + sqrt((u20 - u02)*(u20 - u02) + 4*u11*u11);
    }
	if( num != 0 && den  != 00 )
	{
		m_ellipse.angle = 180.0 + (180.0 / CV_PI) * atan( num / den );
	}
	else
	{
		m_ellipse.angle = 0;
	}
        
	return m_ellipse;

}

void CBlob::FillBlob( IplImage *image, CvScalar color, int offsetX /*=0*/, int offsetY /*=0*/)
{
    cv::drawContours( cv::Mat(image), m_externalContour.GetContourPoints(), -1, color, -1 );
}


std::vector<cv::Point2i> CBlob::GetConvexHull()
{
    std::vector<cv::Point2i> hull;
    cv::convexHull( m_externalContour.GetContourPoints(), hull);
    return hull;
}

void CBlob::JoinBlob( CBlob *blob )
{
    // Simply append the contour points
    m_externalContour.AddContourPoints( blob->GetExternalContour()->GetContourPoints() );
}
