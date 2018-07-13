#include "src/aligns/XCorr.h"
#include <math.h>
#include "../../extern/RealFFT/FFTReal.h"

XCorrelation::XCorrelation()
{
  m_pFFT = NULL;
  m_size = 0;
}

XCorrelation::~XCorrelation()
{
  if (m_pFFT != NULL)
    delete m_pFFT;
}


void XCorrelation::AutoCorrelate(vector<float> &out, vector<float> &in)
{
  out.clear();

  int N=FFTReal_get_next_pow2((long) in.size());

  in.resize(N,0);
  out.resize(N,0);
  
  vector<float> X(N,0),tmp(N,0);

  m_pFFT = new FFTReal<float> (N);

  m_pFFT->do_fft(&X[0],&in[0]);
  
  for (int i=0; i<=N/2; ++i )
    tmp[i] = (X[i]+X[i+N/2])*(X[i]-X[i+N/2]);

  m_pFFT->do_ifft(&tmp[0],&out[0]);
  m_pFFT->rescale(&out[0]);
}


void XCorrelation::CrossCorrelate(svec<float> & o, const svec<float> & in1, const svec<float> & in2)
{
  int i;
  o.resize(in1.size(), 0);

  if (m_pFFT == NULL || m_size != in1.isize()) {
    m_size = in1.isize();
    if (m_pFFT != NULL) {
      cout << "WARNING: re-instantiating FFT object!" << endl;
      delete m_pFFT;
    }
    cout << "Initializing FFT." << endl;
    m_pFFT = new FFTReal<float>(m_size);
    cout << "done." << endl;
  }

  svec<float> tmp1;
  tmp1.resize(in1.size(), 0.);
  svec<float> tmp2;
  tmp2.resize(in2.size(), 0.);

  float * p1 = &tmp1[0];
  float * p2 = &tmp2[0];



  m_pFFT->do_fft(p1, &in1[0]);


  m_pFFT->do_fft(p2, &in2[0]);


  int N = tmp1.isize() / 2;

  tmp1[0] *= tmp2[0];
  for (i=1; i<N-1; i++) {
    float fr, fi;

 
    ComplexMult(fr, fi, tmp1[i], tmp1[i+N], tmp2[i], -tmp2[i+N]);
    tmp1[i] = fr;
    tmp1[i+N] = -fi;
  }
  m_pFFT->do_ifft(p1, p2);
  m_pFFT->rescale(p2);

  for (i=0; i<N; i++) {
    o[i] = tmp2[i+N];
  }
  for (i=N; i<2*N; i++) {
    o[i] = tmp2[i-N];
  }
 
}

