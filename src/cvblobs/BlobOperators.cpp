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
    if( !blob.GetExternalContour()->GetContourPoints().empty() )
	{
        return cv::pointPolygonTest(blob.GetExternalContour()->GetContourPoints(), m_p, false) >=0;
	}

	return 0;
}
