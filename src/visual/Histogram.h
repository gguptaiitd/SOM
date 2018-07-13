#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "../base/SVector.h"
#include "Whiteboard.h"



class Histogram
{
 public:
  Histogram() {
    x_max = 300;
    y_max = 300;
    x_offset = 20;
    y_offset = 20;
  }

  void Plot(const string & fileName,
	    const svec<double>&data, 
	    int bins, 	    
	    color c = color(0., 0., 0.7));
  
  void Plot(const string & fileName,
	    const svec<double>&data, 
	    int bins, 	    
	    double lo,
	    double hi,
	    color c = color(0., 0., 0.7));

  void Scatter(const string & fileName,
	    const svec<double>&data1, 
	    const svec<double>&data2, 	  
	    color c = color(0., 0., 0.7));
 private:
  
  void AddScaleX(ns_whiteboard::whiteboard & board, 
		 double from,
		 double to);
  void AddScaleY(ns_whiteboard::whiteboard & board, 
		 double from,
		 double to);


  double x_offset;
  double y_offset;
  double x_max;
  double y_max;

};


#endif

