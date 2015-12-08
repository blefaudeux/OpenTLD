#include "BlobContour.h"
#include <opencv/cxcore.h>
#include <opencv/cvwimage.h>
#include <opencv/cv.hpp>

CBlobContour::CBlobContour()
{
	m_area = -1;
	m_perimeter = -1;
    m_moments.clear();
    m_contour.clear();
}
CBlobContour::CBlobContour( std::vector<cv::Point2i> const & contour )
{
    m_area = -1; // Lazy evaluation, only computed if necessary
	m_perimeter = -1;
    m_moments.clear();
    m_contour = contour;
}

//! Copy constructor
CBlobContour::CBlobContour( CBlobContour *source )
{
	if (source != NULL )
	{
		*this = *source;
	}
}

CBlobContour::~CBlobContour()
{
    m_contour.clear();
}

//! Copy operator
CBlobContour& CBlobContour::operator=( const CBlobContour &source )
{
	if( this != &source )
    {
        m_contour.clear();

        if (!source.m_contour.empty())
		{
            m_contour =	source.m_contour;
		}

		m_area = source.m_area;
		m_perimeter = source.m_area;
		m_moments = source.m_moments;
	}
	return *this;
}

//! Clears chain code contour and points
void CBlobContour::ResetChainCode()
{
    m_contour.clear();
}

double CBlobContour::GetPerimeter()
{
	// is calculated?
	if (m_perimeter != -1)
	{
		return m_perimeter;
	}

	if( IsEmpty() )
		return 0;

    m_perimeter = cv::arcLength( GetContourPoints(), true );

	return m_perimeter;
}

double CBlobContour::GetArea()
{
	// is calculated?
	if (m_area != -1)
	{
		return m_area;
	}

	if( IsEmpty() )
		return 0;

    m_area = fabs( cv::contourArea( GetContourPoints() ));
	
	return m_area;
}

//! Get contour moment (p,q up to MAX_CALCULATED_MOMENTS)
double CBlobContour::GetMoment(int p, int q)
{
	// is a valid moment?
	if ( p < 0 || q < 0 || p > MAX_MOMENTS_ORDER || q > MAX_MOMENTS_ORDER )
	{
		return -1;
	}

	if( IsEmpty() )
		return 0;

	// it is calculated?
    if( m_moments.empty())
	{
//        m_moments = cv::moments( GetContourPoints());
	}

    //FIXME : Ben - get something clean here
    return 0. /* m_moments */;
}

void CBlobContour::addContourPoints(std::vector<cv::Point2i> const & newPoints)
{
    m_contour.insert(m_contour.end(), newPoints.begin(), newPoints.end());
}

std::vector<cv::Point2i> const & CBlobContour::GetContourPoints() const
{
    return m_contour;
}
