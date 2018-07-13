#ifndef XCORR_H
#define XCORR_H

class XCorrelation
{
 public:
  XCorrelation();
  ~XCorrelation();
  
  void AutoCorrelate(vector<float> &out, vector<float> &in);

  void CrossCorrelate(svec<float> & o, const svec<float> & in1, const svec<float> & in2);

 private:
  FFTReal<float> * m_pFFT;
  int m_size;

};



#endif

