#include "NNIO.h"
#include <math.h>


void NNIO::AddElement(const string & name, double val, bool bValid)
{
  m_data.push_back(val);
  if (bValid)
    m_valid.push_back(1);
  else
    m_valid.push_back(0);
  m_name.push_back(name);
}

void NNIO::AddElement(const string & name)
{
  m_data.push_back(0.);
  m_valid.push_back(1);
  m_name.push_back(name);
}

void NNIO::SetElement(const string & name, double v)
{
  int index = Index(name);
  m_data[index] = v;
}

void NNIO::SetValid(const string & name, bool b)
{
  int index = Index(name);
  SetValid(index, b);
}

double NNIO::GetElement(const string & name) const
{
  int index = Index(name);
  return m_data[index];
}

double NNIO::IsValid(const string & name) const
{
  int index = Index(name);
  return IsValid(index);
}

double NNIO::Distance(const NNIO & n) const
{
  double d = 0.;
  double nn = 0.;

  for (int i=0; i<m_data.isize(); i++) {
    if (!IsValid(i) || n.IsValid(i))
      continue;

    double a = m_data[i];
    double b = n[i];
    nn += 1.;
    d += (a-b)*(a-b);
  }
  return sqrt(d) / nn;
}

double NNIO::Distance(const svec<double> & n) const
{
  double d = 0.;
  double nn = 0.;

  for (int i=0; i<m_data.isize(); i++) {
    if (!IsValid(i))
      continue;

    double a = m_data[i];
    double b = n[i];
    nn += 1.;
    d += (a-b)*(a-b);
  }
  return sqrt(d) / nn;
}


void NNIO::Print() const
{
  cout << "+++++ Printing IO, elements: " << m_data.isize() << endl;
  for (int i=0; i<m_data.isize(); i++) {
    cout << m_name[i] << " " << m_valid[i] << "  " << m_data[i] << endl;
  }
  cout << "+++++" << endl;
}


