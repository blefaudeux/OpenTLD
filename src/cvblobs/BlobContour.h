#ifndef BLOBCONTOUR_H_INCLUDED
#define BLOBCONTOUR_H_INCLUDED


#include "list"

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

//! Type of chain codes
typedef unsigned char t_chainCode;
//! Type of list of chain codes
typedef CvSeq* t_chainCodeList;
//! Type of list of points
typedef CvSeq* t_PointList;


//! Max order of calculated moments
#define MAX_MOMENTS_ORDER		3


//! Blob contour class (in crack code)
class CBlobContour
{
	friend class CBlob;
	friend class CBlobProperties; //AO
	
public:
	CBlobContour();
    CBlobContour(const std::vector<cv::Point2i> &contour);

    CBlobContour( CBlobContour *source );

	~CBlobContour();

    CBlobContour& operator=( const CBlobContour &source );

    void addContourPoints(std::vector<cv::Point2i> const & newPoints);

	//! Return freeman chain coded contour
    std::vector<cv::Point2i> const & GetChainCode()
	{
		return m_contour;
	}

	bool IsEmpty()
	{
        return m_contour.empty();
	}

	//! Return all contour points
    std::vector<cv::Point2i> const & GetContourPoints() const;

protected:	

	CvPoint GetStartPoint() const
	{
        return m_contour.front();
	}

	//! Clears chain code contour
	void ResetChainCode();
	

	
	//! Computes area from contour
	double GetArea();
	//! Computes perimeter from contour
	double GetPerimeter();
	//! Get contour moment (p,q up to MAX_CALCULATED_MOMENTS)
	double GetMoment(int p, int q);

private:
    std::vector<cv::Point2i> m_contour;

	double m_area;
    double m_perimeter;

    std::vector<cv::Moments> m_moments;
};

#endif	//!BLOBCONTOUR_H_INCLUDED


