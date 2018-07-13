#ifndef PIECHART_H
#define PIECHART_H

#include "../base/SVector.h"
#include "Whiteboard.h"



class PieChart
{
 public:
  PieChart() {}

  void Plot(const string & fileName,
	    const svec<double>&data);
  
 private:
  

};


#endif

