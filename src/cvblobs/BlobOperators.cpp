#include <limits.h>
#include "BlobOperators.h"
#include <opencv/cv.hpp>

double CBlobGetMoment::operator()(CBlob &blob)
{
	return blob.Moment(m_p, m_q);
}

double CBlobGetHullPerimeter::operator()(CBlob &blob)
{
    return blob.Perimeter();
}

double CBlobGetHullArea::operator()(CBlob &blob)
{
    return blob.Area();
}

double CBlobGetMinXatMinY::operator()(CBlob &blob)
{
    std::vector<cv::Point2i> const & contour = blob.GetExternalContour()->GetContourPoints();

    if (contour.empty())
    {
        return LONG_MAX;
    }

    cv::Point2i minPoint = contour[0];

    for(unsigned int i=0; i<contour.size(); ++i)
    {
        if (contour[i].y < minPoint.y)
        {
            minPoint = contour[i];
        }
    }

    return minPoint.x;
}


double CBlobGetMinYatMaxX::operator()(CBlob &blob)
{
    std::vector<cv::Point2i> const & contour = blob.GetExternalContour()->GetContourPoints();

    if (contour.empty())
    {
        return LONG_MAX;
    }

    cv::Point2i maxPoint = contour[0];

    for(unsigned int i=0; i<contour.size(); ++i)
    {
        if (contour[i].x < maxPoint.x)
        {
            maxPoint = contour[i];
        }
    }

    return maxPoint.y;
}


double CBlobGetMaxXatMaxY::operator()(CBlob &blob)
{
	double result = LONG_MIN;
	
	CvSeqReader reader;
	CvPoint actualPoint;
	t_PointList externContour;
	
	externContour = blob.GetExternalContour()->GetContourPoints();
	if( !externContour ) return result;

	cvStartReadSeq( externContour, &reader);

	for( int i=0; i< externContour->total; i++)
	{
		CV_READ_SEQ_ELEM( actualPoint, reader);

		if( (actualPoint.y == blob.MaxY()) && (actualPoint.x > result) )
		{
			result = actualPoint.x;
		}	
	}

	return result;
}

double CBlobGetMaxYatMinX::operator()(CBlob &blob)
{
	double result = LONG_MIN;
	
	CvSeqReader reader;
	CvPoint actualPoint;
	t_PointList externContour;
	
	externContour = blob.GetExternalContour()->GetContourPoints();
	if( !externContour ) return result;

	cvStartReadSeq( externContour, &reader);

	
	for( int i=0; i< externContour->total; i++)
	{
		CV_READ_SEQ_ELEM( actualPoint, reader);

		if( (actualPoint.x == blob.MinX()) && (actualPoint.y > result) )
		{
			result = actualPoint.y;
		}	
	}

	return result;
}

double CBlobGetElongation::operator()(CBlob &blob)
{
	double ampladaC,longitudC,amplada,longitud;

	double tmp;

	tmp = blob.Perimeter()*blob.Perimeter() - 16*blob.Area();

	if( tmp > 0.0 )
		ampladaC = (double) (blob.Perimeter()+sqrt(tmp))/4;
	else
		ampladaC = (double) (blob.Perimeter())/4;

	if(ampladaC<=0.0) return 0;
	longitudC=(double) blob.Area()/ampladaC;

	longitud=MAX( longitudC , ampladaC );
	amplada=MIN( longitudC , ampladaC );

	return (double) longitud/amplada;
}


double CBlobGetCompactness::operator()(CBlob &blob)
{
	if( blob.Area() != 0.0 )
		return (double) pow(blob.Perimeter(),2)/(4*CV_PI*blob.Area());
	else
		return 0.0;
}


double CBlobGetRoughness::operator()(CBlob &blob)
{
	CBlobGetHullPerimeter getHullPerimeter = CBlobGetHullPerimeter();
	
	double hullPerimeter = getHullPerimeter(blob);

	if( hullPerimeter != 0.0 )
		return blob.Perimeter() / hullPerimeter;//HullPerimeter();

	return 0.0;
}

double CBlobGetLength::operator()(CBlob &blob)
{
	double ampladaC,longitudC;
	double tmp;

	tmp = blob.Perimeter()*blob.Perimeter() - 16*blob.Area();

	if( tmp > 0.0 )
		ampladaC = (double) (blob.Perimeter()+sqrt(tmp))/4;
	// error intrínsec en els càlculs de l'àrea i el perímetre 
	else
		ampladaC = (double) (blob.Perimeter())/4;

	if(ampladaC<=0.0) return 0;
	longitudC=(double) blob.Area()/ampladaC;

	return MAX( longitudC , ampladaC );
}

double CBlobGetBreadth::operator()(CBlob &blob)
{
	double ampladaC,longitudC;
	double tmp;

	tmp = blob.Perimeter()*blob.Perimeter() - 16*blob.Area();

	if( tmp > 0.0 )
		ampladaC = (double) (blob.Perimeter()+sqrt(tmp))/4;
	// error intrínsec en els càlculs de l'àrea i el perímetre 
	else
		ampladaC = (double) (blob.Perimeter())/4;

	if(ampladaC<=0.0) return 0;
	longitudC = (double) blob.Area()/ampladaC;

	return MIN( longitudC , ampladaC );
}

double CBlobGetDistanceFromPoint::operator()(CBlob &blob)
{
	double xmitjana, ymitjana;
	CBlobGetXCenter getXCenter;
	CBlobGetYCenter getYCenter;

	xmitjana = m_x - getXCenter( blob );
	ymitjana = m_y - getYCenter( blob );

	return sqrt((xmitjana*xmitjana)+(ymitjana*ymitjana));
}


double CBlobGetXYInside::operator()(CBlob &blob)
{
	if( blob.GetExternalContour()->GetContourPoints() )
	{
		return cvPointPolygonTest( blob.GetExternalContour()->GetContourPoints(), m_p,0) >= 0;
	}

	return 0;
}
