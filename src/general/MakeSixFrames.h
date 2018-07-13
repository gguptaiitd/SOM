#ifndef MAKESIXFRAMES_H
#define MAKESIXFRAMES_H



#include "DNAVector.h"


class MakeSixFrames
{
 public:
  MakeSixFrames() {}

  void AllSixFrames(vecDNAVector & out, const vecDNAVector & in);
  void AllSixFrames(vecDNAVector & inout);

};



#endif 

