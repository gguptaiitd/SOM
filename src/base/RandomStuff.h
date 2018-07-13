#ifndef RANDOMSTUFF_H_
#define RANDOMSTUFF_H_

#include <math.h>
#include "SVector.h"

inline long long RandomInt(long long n)
{    
  long long r =   ((long long)( rand( ) ) << 31 ) ^ (long long)( rand( ) );
  //cout << "r=" << r;
  r = r % n;
  //cout << " " << r << endl;
  return r;
}

inline double RandomFloat(double n)
{    
  long long r =   ((long long)( rand( ) ) << 31 ) ^ (long long)( rand( ) );
  //cout << "r=" << r;
  r = r % 0x7FFFFFFF;
  //cout << " " << r << endl;
  return n*((double)r)/(double)0x7FFFFFFF;
}

inline double RandomFloatNormal(double n)
{    
  double u = RandomFloat(1.);
  double v = RandomFloat(1.);
  double x = sqrt(-2.*log(u))*cos(2.*3.1415926535897932384626433832795*v);
  return x;
}



template<class T> 
void SRandomize(svec<T>& v)
{
  random_shuffle(v.begin(), v.end());
}

template<class T> 
void Randomize(vector<T>& v)
{
  random_shuffle(v.begin(), v.end());
}







#endif //RANDOMSTUFF_H_

