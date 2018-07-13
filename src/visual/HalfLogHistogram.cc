#define FORCE_DEBUG

#include "Whiteboard.h"

#include "../base/CommandLineParser.h"
#include "../base/FileParser.h"
#include "../base/SVector.h"
#include "Color.h"

#include "Compounds.h"

#include <iostream>
#include <math.h>



void TickMarks(int & howMany, int & what, double in)
{
  double l = log(in)/log(10);
  int i = (int)l;
  int f = pow(10, i);
  int max = (int)(in/f);
  howMany = max;
  cout << howMany << " " << f << endl;
  what = f;
  if (howMany < 2) {
    i /= 10;
    f = pow(10, i);
    max = (int)(l/f);
  }
  if (howMany < 5) {
    f = f/2;
    max *=2;
  }

  howMany = max;
  what = f;

  cout << in << " " << l << " " << i << " " << f << " " << max << endl;

}




int main( int argc, char** argv )
{
 
  commandArg<string> aStringI("-i","input");
  commandArg<int> colCmmd("-c","column", 0);
  commandArg<bool> sumCmmd("-s","show nucleotides in sequences, not counts", false);
  commandArg<string> aStringO("-o","outfile (post-script)");
 
  
  commandLineParser P(argc,argv);
  P.SetDescription("Scatter plot");

  P.registerArg(aStringO);
  P.registerArg(aStringI);
  P.registerArg(colCmmd);
  P.registerArg(sumCmmd);

  P.parse();

  string o = P.GetStringValueFor(aStringO);
  string fileName = P.GetStringValueFor(aStringI);
  int col = P.GetIntValueFor(colCmmd);
  bool bSum = P.GetBoolValueFor(sumCmmd);
  
  double x_offset = 40;
  double y_offset = 20;


  int i, j;
  ns_whiteboard::whiteboard board;

 
  double x_max = 0.;
  double y_max = 0.;

  FlatFileParser parser;
  
  parser.Open(fileName);

  double rad = 1.;

  double scale = 6; 

  svec<int> hist_lin;
  svec<int> hist_log;
  hist_lin.resize(24, 0);
  hist_log.resize(48, 0);
  
  double logMax = 12.;
  double sum = 0.;
  while (parser.ParseLine()) {
    if (parser.GetItemCount() == 0)
      break;
    int n = parser.AsInt(col);
    if (n < hist_lin.isize()) {
      if (bSum) {
 	hist_lin[n] += n;
     } else {
	hist_lin[n]++;
      }
    } else {
      
      //int ll = (int)(hist_log.isize()/logMax * log(n + 1 - hist_lin.isize()));
      int ll = (int)(hist_log.isize()/logMax * (log(n) - log(1+hist_lin.isize())));
      if (ll >= hist_log.isize()) {
	cout << "Error: " << ll << endl;
      } else {
	if (bSum) {
	  hist_log[ll] += n;
	} else {
	  hist_log[ll]++;
	}
      }
    }
        
  }

  double x = 0.;
  double w = 12.;
  double l = 1.;
  //scale = 400./30000.;
  scale = 100./30000.;

  if (bSum)
    scale /= 100.;

  y_max = 500.;

  double y_max_real = 0.;
  
  for (i=0; i<hist_lin.isize(); i++) {
    double h = scale * hist_lin[i];
    if (hist_lin[i] > y_max_real)
      y_max_real = hist_lin[i];

    if (h < 0.01) {
      x += w;
      if (x > x_max)
	x_max = x;
      continue;
    }
    board.Add( new ns_whiteboard::rect( ns_whiteboard::xy_coords(x_offset + x, y_offset ), 
                                        ns_whiteboard::xy_coords(x_offset + x + w, y_offset+h),
                                        color(0., 0., 0.)));

    board.Add( new ns_whiteboard::rect( ns_whiteboard::xy_coords(x_offset + x + l, y_offset + l ), 
                                        ns_whiteboard::xy_coords(x_offset + x + w - l, y_offset+h-l),
					color(0.5, 0.4, 0.2)));

    x += w;
    if (x > x_max)
      x_max = x;
    if (y_offset+h > y_max)
      y_max = y_offset+h;
   
  }

  x += w;

  for (i=0; i<hist_log.isize(); i++) {
    double h = scale * hist_log[i];
    if (hist_log[i] > y_max_real)
      y_max_real = hist_log[i];
    if (h < 0.01) {
      x += w;
      if (x > x_max)
	x_max = x;
      continue;
    }
    board.Add( new ns_whiteboard::rect( ns_whiteboard::xy_coords(x_offset + x, y_offset), 
                                        ns_whiteboard::xy_coords(x_offset + x + w, y_offset+h),
                                        color(0., 0., 0.)));

   board.Add( new ns_whiteboard::rect( ns_whiteboard::xy_coords(x_offset + x + l, y_offset + l ), 
                                        ns_whiteboard::xy_coords(x_offset + x + w - l, y_offset+h-l),
				       color(0.1, 0.1, 0.7)));

    if (y_offset+h > y_max)
      y_max = y_offset+h;

    x += w;
    if (x > x_max)
      x_max = x;
   
  }





  board.Add( new ns_whiteboard::line( ns_whiteboard::xy_coords(x_offset, y_offset), 
                                      ns_whiteboard::xy_coords(x_offset, y_max), 
                                      1.,
                                      color(0., 0., 0.)) );
  board.Add( new ns_whiteboard::line( ns_whiteboard::xy_coords(x_offset, y_offset), 
                                      ns_whiteboard::xy_coords(x_max, y_offset), 
                                      1.,
                                      color(0., 0., 0.)) );


  for (double t=0.; t<= logMax; t++) {
    double l = t*x_max/logMax;
    
    board.Add( new ns_whiteboard::line( ns_whiteboard::xy_coords(x_offset+l, y_offset), 
					ns_whiteboard::xy_coords(x_offset+l, y_offset-4), 
					1.,
					color(0., 0., 0.)) );

    char tmp[256];
    sprintf(tmp, "%2.1f", t);
    board.Add( new ns_whiteboard::text( ns_whiteboard::xy_coords(x_offset + l-8, y_offset-16),
                                        tmp, color(0,0,0), 12, "Times-Roman", 0, true));

  }

  int howMany, what;

  TickMarks(howMany, what, y_max_real);

  for (i=0; i<=howMany+1; i++) {
    double l = scale*(double)(what*i);
    
    board.Add( new ns_whiteboard::line( ns_whiteboard::xy_coords(x_offset-4, y_offset+l), 
					ns_whiteboard::xy_coords(x_offset, y_offset+l), 
					1.,
					color(0., 0., 0.)) );

    char tmp[256];
    sprintf(tmp, "%d", i*what);
    board.Add( new ns_whiteboard::text( ns_whiteboard::xy_coords(x_offset-36, y_offset+l),
                                        tmp, color(0,0,0), 12, "Times-Roman", 0, true));

  }




  cout << "x_max: " << x_max << endl;
  ofstream out(o.c_str());
  
  ns_whiteboard::ps_display display(out, x_max + 2 * x_offset, y_max + 2 * y_offset);
  board.DisplayOn(&display);
 

  return 0;
}
