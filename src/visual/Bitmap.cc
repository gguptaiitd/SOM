#include "Bitmap.h"
#include <stdio.h>

Bitmap::Bitmap() {
  m_x = 0;
  m_y = 0;


  m_fileHeader.bfType = 0x4d42;
  //m_fileHeader.bfType = 0x424d;
  m_fileHeader.bfSize = 12;
  m_fileHeader.bfReserved1 = 0;
  m_fileHeader.bfReserved2 = 0;
  m_fileHeader.bfOffBits = 0x36;
	 
	 
  // Fill the bitmap info structure
  m_infoHeader.biSize = 0x28;
  m_infoHeader.biWidth = 480;
  m_infoHeader.biHeight = 480;
  m_infoHeader.biPlanes = 1;
  m_infoHeader.biBitCount = 24;          
  m_infoHeader.biCompression = 0;     
  m_infoHeader.biSizeImage = m_x * m_y * 3;   
  m_infoHeader.biXPelsPerMeter = 0x24e7;
  m_infoHeader.biYPelsPerMeter = 0x24e7;
  m_infoHeader.biClrUsed = 0;
  m_infoHeader.biClrImportant = 0;
}


void Bitmap::Read(const string & fileName)
{
  FILE * p = fopen(fileName.c_str(), "rb");
  
  int n;
  n = fread((void*)&m_fileHeader, 12, 1, p); 
  n = fread((void*)&m_infoHeader, 0x28, 1, p); 

  int i;

  char rgb[3];
  m_x = m_infoHeader.biWidth;
  m_y = m_infoHeader.biHeight;
  
  m_data.resize(m_x * m_y);

  for (i=0; i<m_data.isize(); i++) {
    n = fread((void*)rgb, sizeof(rgb), 1, p); 
    m_data[i].Set_R(rgb[2]);
    m_data[i].Set_G(rgb[1]);
    m_data[i].Set_B(rgb[0]);
  }


  fclose(p);
}

void Bitmap::Write(const string & fileName)
{
  int n = m_x * m_y * 3; 
  int rest = n % 32;
  cout << "n: " << n << " rest " << rest << endl;
  if (rest > 0)
    rest = 32 - rest;
  cout << "Adding " << rest << endl;
  m_infoHeader.biSizeImage = n + rest;   

  m_infoHeader.biWidth = m_x;
  m_infoHeader.biHeight = m_y;

  m_fileHeader.bfSize = 12 + 0x28 + 3*m_data.isize() + 4;

  FILE * p = fopen(fileName.c_str(), "wb");
  fwrite((void*)&m_fileHeader.bfType, sizeof(m_fileHeader.bfType), 1, p); 
  fwrite((void*)&m_fileHeader.bfSize, sizeof(m_fileHeader.bfSize), 1, p); 
  fwrite((void*)&m_fileHeader.bfReserved1, sizeof(m_fileHeader.bfReserved1), 1, p); 
  fwrite((void*)&m_fileHeader.bfReserved2, sizeof(m_fileHeader.bfReserved1), 1, p); 
  fwrite((void*)&m_fileHeader.bfOffBits, sizeof(m_fileHeader.bfOffBits), 1, p); 

  fwrite((void*)&m_infoHeader.biSize, sizeof(m_infoHeader.biSize), 1, p); 
  fwrite((void*)&m_infoHeader.biWidth, sizeof(m_infoHeader.biWidth), 1, p); 
  fwrite((void*)&m_infoHeader.biHeight, sizeof(m_infoHeader.biHeight), 1, p); 
  fwrite((void*)&m_infoHeader.biPlanes, sizeof(m_infoHeader.biPlanes), 1, p); 
  fwrite((void*)&m_infoHeader.biBitCount, sizeof(m_infoHeader.biBitCount), 1, p); 
  fwrite((void*)&m_infoHeader.biCompression, sizeof(m_infoHeader.biCompression), 1, p); 
  fwrite((void*)&m_infoHeader.biSizeImage, sizeof(m_infoHeader.biSizeImage), 1, p); 
  fwrite((void*)&m_infoHeader.biXPelsPerMeter, sizeof(m_infoHeader.biXPelsPerMeter), 1, p); 
  fwrite((void*)&m_infoHeader.biYPelsPerMeter, sizeof(m_infoHeader.biYPelsPerMeter), 1, p); 
  fwrite((void*)&m_infoHeader.biClrUsed, sizeof(m_infoHeader.biClrUsed), 1, p); 
  fwrite((void*)&m_infoHeader.biClrImportant, sizeof(m_infoHeader.biClrImportant), 1, p); 
 

  int i;

  char rgb[3];

  for (i=0; i<m_data.isize(); i++) {
    rgb[0] = m_data[i].B();
    rgb[1] = m_data[i].G();
    rgb[2] = m_data[i].R();
    fwrite((void*)rgb, sizeof(rgb), 1, p); 
  }


  fclose(p);
}


