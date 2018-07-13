#include <string>
#include "Histogram.h"
#include "Whiteboard.h"
#include "../base/CommandLineParser.h"
#include "../base/FileParser.h"

#include "../base/SVector.h"
#include "Color.h"


#include <iostream>
#include <math.h>


void Histogram::Plot(const string & fileName,
		     const svec<double>&data, 
		     int bins, 
		     color c)
{
  double lo = 0;
  double hi = 1.;
  if (data.isize() > 0) {
    lo = data[0];
    hi = data[0];
  }

  int i;
  for (i=0; i<data.isize(); i++) {
    if (data[i] < lo)
      lo = data[i];
    if (data[i] > hi)
      hi = data[i];
  }

  //cout << "lo=" << lo << " hi=" << hi << endl;
  Plot(fileName, data, bins, lo, hi, c);
}
void Histogram::Scatter(const string & fileName,
			const svec<double>&data1, 
			const svec<double>&data2, 	  
			color c)
{
  double hi1 = 1.;
  double hi2 = 1.;

  if (data2.isize() > 0)
    hi2 = data2[0];

  int i;
  for (i=0; i<data1.isize(); i++) {
    //if (data1[i] < lo)
    //lo = data1[i];
    if (data1[i] > hi1)
      hi1 = data1[i];
  
    //if (data2[i] < lo)
    //lo = data2[i];
    if (data2[i] > hi2)
      hi2 = data2[i];
  }
  double scale1 = 300/hi1;
  double scale2 = 300/hi2;

  string o = fileName;
  


  ns_whiteboard::whiteboard board;

  board.Add( new ns_whiteboard::rect( ns_whiteboard::xy_coords(0, 0), 
				      ns_whiteboard::xy_coords(2*x_offset+x_max, 2*y_offset+y_max),
				      color(0.99,0.99,0.99)) );
 
  double rad = 1.;

  //double scale = 12.;
  
  for (i=0; i<data1.isize(); i++) {
    double x = scale1*data1[i] + x_offset;
    double y = scale2*data2[i] + y_offset;
 
    board.Add( new ns_whiteboard::arc( ns_whiteboard::xy_coords(x, y), 
                                       rad, 0., 360., 0.5,
                                       c));
  }
  board.Add( new ns_whiteboard::line( ns_whiteboard::xy_coords(x_offset, y_offset), 
				      ns_whiteboard::xy_coords(x_offset, y_max+y_offset), 
				      1.,
				      color(0., 0., 0.)) );
  board.Add( new ns_whiteboard::line( ns_whiteboard::xy_coords(x_offset, y_offset), 
				      ns_whiteboard::xy_coords(x_max+x_offset, y_offset), 
				      1.,
				      color(0., 0., 0.)) );

  //AddScaleX(board, 0, hi1);
  AddScaleX(board, 0, hi1);

  ofstream out(o.c_str());
  
  ns_whiteboard::ps_display display(out, x_max + 2 * x_offset, y_max + 2 * y_offset);
  board.DisplayOn(&display);
 
}

void ReFormat(char * tmp, double max)
{
  int i;
  if (max < 10.) {
    for (i=0; i<(int)strlen(tmp); i++) {
      if (tmp[i] == '.' && (int)strlen(tmp) > i+2) {
	tmp[i+2] = 0;
	break;
      }
    }   
  } else {
    for (i=0; i<(int)strlen(tmp); i++) {
      if (tmp[i] == '.') {
	tmp[i] = 0;
	break;
      }
    }
  }
}

void Histogram::AddScaleX(ns_whiteboard::whiteboard & board, 
			  double from,
			  double to)
{
  int i;
  double range = to - from;
  for (i=0; i<=10; i++) {
    double v = i*range/10.;
    double x = x_offset + i*x_max/10.;
    board.Add( new ns_whiteboard::line( ns_whiteboard::xy_coords(x, y_offset), 
					ns_whiteboard::xy_coords(x, y_offset-3), 
					1.,
					color(0., 0., 0.)) );

    char mark[256];
    sprintf(mark, "%f", v);
    ReFormat(mark, to);

    board.Add( new ns_whiteboard::text( ns_whiteboard::xy_coords(x-5, y_offset-12),
					mark, black, 8., "Times-Roman", 0, true));

  }
}

void Histogram::AddScaleY(ns_whiteboard::whiteboard & board, 
			  double from,
			  double to)
{

}


void Histogram::Plot(const string & fileName,
		     const svec<double>&data, 
		     int bins, 	    
		     double lo,
		     double hi,
		     color c)


{
  int i, j;

  svec<double> counts;
  if (bins == 0)
    bins = 1;
  counts.resize(bins, 0);

  cout << "lo=" << lo << " hi=" << hi << endl;
  if (hi == lo)
    hi += 0.1;
  
  double scale = 0.999*(double)bins/(hi - lo);
  
  double max = 0.;
  for (i=0; i<data.isize(); i++) {
    //cout << lo << " " << data[i] << " " << scale << endl;
    int index = (int)(scale*(lo + data[i]));
    if (index < counts.isize()) {
      counts[index] += 1.;
      if (counts[index] >= max)
	max = counts[index];
    }
  }
  cout << "Data points: " << data.isize() << " max: " << max << endl;
  if (data.isize() == 0)
    max = 1.;

  double yscale = 300/max;

  string o = fileName;
  


  ns_whiteboard::whiteboard board;

  board.Add( new ns_whiteboard::rect( ns_whiteboard::xy_coords(0, 0), 
				      ns_whiteboard::xy_coords(2*x_offset+x_max, 2*y_offset+y_max),
				      color(0.99,0.99,0.99)) );
  

  if (max > 0.) {
    for (i=0; i<counts.isize(); i++) {
      double x1 = x_max*(double)i/(double)counts.isize();
      double x2 = x_max*(double)(i+1)/(double)counts.isize();
      double y1 = 0;
      double y2 = counts[i]*yscale;
      
      //cout << "PLOT " << x1 << " " << x2 << " - " << y1 << " " << y2 <<  " : " << yscale << endl;
      
      double edge = 1.;
      board.Add( new ns_whiteboard::rect( ns_whiteboard::xy_coords(x1 + x_offset, y1 + y_offset), 
					  ns_whiteboard::xy_coords(x2 + x_offset, y2 + y_offset),
					  color(0,0,0)) );
      board.Add( new ns_whiteboard::rect( ns_whiteboard::xy_coords(x1 + x_offset+edge, y1 + y_offset+edge), 
					  ns_whiteboard::xy_coords(x2 + x_offset-edge, y2 + y_offset-edge),
					  c) );
      
      
    }
  }
  board.Add( new ns_whiteboard::line( ns_whiteboard::xy_coords(x_offset, y_offset), 
				      ns_whiteboard::xy_coords(x_offset, y_max+y_offset), 
				      1.,
				      color(0., 0., 0.)) );
  board.Add( new ns_whiteboard::line( ns_whiteboard::xy_coords(x_offset, y_offset), 
				      ns_whiteboard::xy_coords(x_max+x_offset, y_offset), 
				      1.,
				      color(0., 0., 0.)) );

  AddScaleX(board, 0, hi-lo);

  ofstream out(fileName.c_str());
  
  ns_whiteboard::ps_display display(out, x_max + 2 * x_offset, y_max + 2 * y_offset);
  board.DisplayOn(&display);
 

}
