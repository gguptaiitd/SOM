#include <string>
#include "PieChart.h"
#include "Whiteboard.h"
#include "../base/CommandLineParser.h"
#include "../base/FileParser.h"

#include "../base/SVector.h"
#include "Color.h"


#include <iostream>
#include <math.h>

void PieChart::Plot(const string & fileName,
		    const svec<double> & data)
{
  int i, j;

  svec<double> norm;
  norm.resize(data.isize());

  double sum = 0;
  for (i=0; i<data.isize(); i++) {
    sum += data[i];
  }
  for (i=0; i<data.isize(); i++) {
    norm[i] = data[i]/sum;
  }
  

  string o = fileName;
  
  double x_offset = 20;
  double y_offset = 20;


  ns_whiteboard::whiteboard board;

 
  double x_max = 200.;
  double y_max = 200.;
  double rad = 100.;
  double x = x_max/2.+x_offset;
  double y = y_max/2.+y_offset;

  double last = 0.;
  for (i=0; i<norm.isize(); i++) {
  //for (i=0; i<1; i++) {
    color col = MakeUpColor(i);

    double from = 360*last;
    double to = 360*(last + norm[i]);

    //cout << x1 << " " << x2 << " - " << y1 << " " << y2 <<  " : " << counts[i] << endl;
    board.Add( new ns_whiteboard::arc( ns_whiteboard::xy_coords(x, y), 
                                     rad/2, from, to, 3*rad/4,
                                     col) );
    //board.Add( new ns_whiteboard::arc( ns_whiteboard::xy_coords(x, y), 
    //                               rad/2, 0, 150, rad/2,
    //                               col) );


    last += norm[i];

  }

  last = 0.;

  
  for (i=0; i<norm.isize(); i++) { 

    double from = last;
    double to = (last + norm[i]);

    double y1 = y + (rad-12)*sin(2*3.1415*to);
    double x1 = x + (rad-12)*cos(2*3.1415*to);
    double y2 = y + (rad-12)*sin(2*3.1415*from);
    double x2 = x + (rad-12)*cos(2*3.1415*from);

    board.Add( new ns_whiteboard::line( ns_whiteboard::xy_coords(x, y), 
					ns_whiteboard::xy_coords(x1, y1), 
					2));
    board.Add( new ns_whiteboard::line( ns_whiteboard::xy_coords(x, y), 
					ns_whiteboard::xy_coords(x2, y2), 
					2));
    //cout << "From " << from << " to " << to << endl;
    //cout << "x1 " << x1 << " y1 " << y1 << endl;
    //cout << "x2 " << x1 << " y2 " << y1 << endl;
    last += norm[i];
    //if (i == 0)
    //break;
      
  }

  board.Add( new ns_whiteboard::arc( ns_whiteboard::xy_coords(x, y), 
				     rad-12, 0, 360, 2));


  ofstream out(fileName.c_str());
  
  ns_whiteboard::ps_display display(out, x_max + 2 * x_offset, y_max + 2 * y_offset);
  board.DisplayOn(&display);
}

