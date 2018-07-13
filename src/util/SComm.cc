#include "SComm.h"

#include <stdio.h>

#include <stdlib.h>

#include <unistd.h>

#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#include <signal.h>
#include <netdb.h>

// max number of bytes we can get at once
#define MAXDATASIZE 1024

// Make this work an a Mac.
#ifndef MSG_CONFIRM
#define MSG_CONFIRM 0
#endif 

#ifdef __APPLE_CC__    
#define MSG_NOSIGNAL 0
#endif

void w_error(const char * msg) {
  //printf("ERROR: %s\n", msg);
  //fflush(stdout);
}

class SSocketCommTransmitter : public SCommTransmitter
{
 public:
  SSocketCommTransmitter(int port = MYPORT);

  virtual ~SSocketCommTransmitter();

  virtual bool SendWait(const char * message, int size = -1);

private:
  /* listen on sock_fd, new connection on new_fd */
  int sockfd, new_fd;

  /* my address information */
  struct sockaddr_in my_addr;

  /* connector's address information */
  struct sockaddr_in their_addr;

  socklen_t sin_size;

  struct sigaction sa;

  int yes;
  int m_port;
};


class SSocketCommReceiver : public SCommReceiver
{
 public:
  SSocketCommReceiver(const char * serverName, int port = MYPORT);
  virtual ~SSocketCommReceiver() {}

  virtual bool Get(char * message, int bufSize);
  virtual bool IsFatal() {return m_bFatal;}

private:
  bool Open();
  void Close();
  char m_serverName[512];
  int m_port;
  bool m_good;
  bool m_bFatal;
  int sockfd, numbytes;
  struct hostent *he;  
  struct sockaddr_in their_addr;
  bool m_bConnected;
  int m_lastSize;
};


SCommTransmitter * GetTransmitter(int port)
{
  return new SSocketCommTransmitter(port);
}

SCommReceiver * GetReceiver(const char * serverName, int port)
{
  return new SSocketCommReceiver(serverName, port);
}

bool GetHostName(char * hostName, int bufSize)
{
  gethostname(hostName, bufSize);
  //printf("Local host: %s\n", hostName);
  return true;
}


/* how many pending connections queue will hold */
#define BACKLOG 10
void sigchld_handler(int s)
{  
  while(wait(NULL) > 0); 
}


//=========================================================================
SSocketCommTransmitter::SSocketCommTransmitter(int port)
{
  m_port = port;
  
  yes = 1;
  new_fd = -1;

  if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1)
  {
    w_error("Server-socket() error lol!");    
    //exit(1);
  } else {
    //printf("Server-socket() sockfd is OK...\n");
  }

  if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1)
  {
    w_error("Server-setsockopt() error lol!");
    //exit(1);
  } else {
    //printf("Server-setsockopt is OK...\n");
  }
  
  
  /* host byte order */
  my_addr.sin_family = AF_INET;

  /* short, network byte order */
  my_addr.sin_port = htons(m_port);
  
  /* automatically fill with my IP */
  my_addr.sin_addr.s_addr = INADDR_ANY;

  //printf("Server-Using %s and port %d...\n", inet_ntoa(my_addr.sin_addr), m_port);

     

  /* zero the rest of the struct */
  memset(&(my_addr.sin_zero), '\0', 8);

  if(bind(sockfd, (struct sockaddr *)&my_addr, sizeof(struct sockaddr)) == -1)    
  {
    w_error("Server-bind() error");
    //exit(1);
  }
  else {
    //printf("Server-bind() is OK...\n");     
  }
  if(listen(sockfd, BACKLOG) == -1)
  {
    w_error("Server-listen() error");
    //exit(1);
  }
  //printf("Server-listen() is OK...Listening...\n");

  /* clean all the dead processes */
  sa.sa_handler = sigchld_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART;

  if(sigaction(SIGCHLD, &sa, NULL) == -1)
  {
    w_error("Server-sigaction() error");
    //exit(1);
  } else {
    //printf("Server-sigaction() is OK...\n");
  }
}

SSocketCommTransmitter::~SSocketCommTransmitter() 
{
  shutdown(sockfd, 2);
  close(sockfd);
}


bool SSocketCommTransmitter::SendWait(const char * message, int size)
{
  int len = size;
  if (size < 0)
    len = strlen(message) + 1;

  //printf("Send data, size %d %d\n", len, size); 

  sin_size = sizeof(struct sockaddr_in);

  if (new_fd == -1) {
    //printf("New accept call.\n");
    new_fd = accept(sockfd, (struct sockaddr *)&their_addr, &sin_size);
    if(new_fd == -1)
    {
      w_error("Server-accept() error");
      
      return false;      
    } else {
      //printf("Accepted.\n");
    }
  }

  int ret;  

  //printf("call send\n");
  //ret = send(new_fd, (const char*)&len, sizeof(len), MSG_CONFIRM);
  ret = send(new_fd, (const char*)&len, sizeof(len), MSG_CONFIRM | MSG_NOSIGNAL);
  ret = send(new_fd, (const char*)&len, sizeof(len), MSG_CONFIRM | MSG_NOSIGNAL);
  int err = errno;
  //printf("returned %d %d\n", ret, err);
  if (ret < 0) {
    //printf("Failed, closing\n");
    close(new_fd);
    new_fd = -1;
    return false;
  }


  //printf("call send(2)\n");
  //ret = send(new_fd, message, len, MSG_CONFIRM);
  ret = send(new_fd, message, len, MSG_CONFIRM | MSG_NOSIGNAL);
  //printf("returned %d\n", ret);
  fflush(stdout);
  //ret = send(new_fd, (const char*)&len, sizeof(len), MSG_CONFIRM);
  //ret = send(new_fd, message, len, MSG_CONFIRM);
  //printf("Send returns: %d\n", ret);
  if (ret < 0) {
    //printf("Failed, closing\n");
    close(new_fd);
    new_fd = -1;
    return false;
  
    w_error("Server-send() error lol!");
  }
  //close(new_fd);
 

  
  return true;
}
//=========================================================================



SSocketCommReceiver::SSocketCommReceiver(const char * serverName, int port)
{
  strcpy(m_serverName, serverName);
  m_port = port;
  m_good = true;
  m_bFatal = false;

  sockfd = -1; 
  he = NULL;    
  m_bConnected = false;
  m_lastSize = 0;
  //printf("SSocketCommReceiver created.\n");
}

bool SSocketCommReceiver::Open()
{
  m_bConnected = false;
  // get the host info
  if (he == NULL) {
    if((he=gethostbyname(m_serverName)) == NULL) {
      w_error("gethostbyname()");
      return false;
    } else {
      //printf("Client-The remote host is: %s\n", m_serverName);
      fflush(stdout);
    }
  }
  
  
  if((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1)
  {
    w_error("socket()");
    return false;
  } else {
    //printf("Client-The socket() sockfd is OK...\n");
    fflush(stdout);
  }
   
   
  // host byte order
  
  their_addr.sin_family = AF_INET;
  
  // short, network byte order
  //printf("Server-Using %s and port %d...\n", m_serverName, m_port);
  
  their_addr.sin_port = htons(m_port);
  their_addr.sin_addr = *((struct in_addr *)he->h_addr);
  
  // zero the rest of the struct
  memset(&(their_addr.sin_zero), '\0', 8); 
    
  struct timeval tv;
  //memset(&tv, sizeof(tv), 0);
  tv.tv_sec = 5;
  tv.tv_usec = 0;
  if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (char *)&tv,  sizeof tv))
  {
    close(sockfd);
    w_error("setsockopt");
    return false;
  }


  if(connect(sockfd, (struct sockaddr *)&their_addr, sizeof(struct sockaddr)) == -1)
  {
    w_error("connect()");
    Close();
    return false;
  } else {
    //printf("Client-The connect() is OK...\n");
    m_bConnected = true;
    //printf("Connected.\n");;
    fflush(stdout);
  }
   

  return true;
}

void SSocketCommReceiver::Close()
{
  close(sockfd);
  shutdown(sockfd, 2);
  m_bConnected = false;

}

bool SSocketCommReceiver::Get(char * message, int bufSize)
{
  //printf("Enter receive, buffer: %d\n", bufSize);
  
  int numbytes = -1;

 
  /*
  if (m_bConnected) 
    printf("Get: status connected\n");
  else
    printf("Get: status NOT connected\n");
  */

  if (!m_bConnected)
    Open();
  
  //printf("call recv\n");
  int datasize = -1;
  recv(sockfd, &datasize, sizeof(datasize), 0);
  if ((numbytes = recv(sockfd, &datasize, sizeof(datasize), 0)) == -1) {
    m_bConnected = false;
    Close();
    return false;
  } else {
    //printf("Read header, bytes: %d, data size=%d\n", numbytes, datasize);  
    if (numbytes == 0) {
      //printf("Closing, no data!!\n");
      Close();
      return false;
    }
      
  }

  if (datasize > bufSize) {
    printf("ERROR in Receive!!! data: %d, buffer: %d\n", datasize, bufSize);
    return false;
  }

  int got = 0;
  int request = datasize;
  do {
    numbytes = recv(sockfd, &message[got], request, 0);
    if (numbytes == -1)
    {	
      w_error("recv()");
      m_bFatal = true;
      m_bConnected = false;
      Close();
      return false;
    } else {
      //printf("numbytes=%d, total=%d\n", numbytes, got + numbytes);      
      //printf("Client-The recv() is OK...\n");
      if (numbytes == 0) {
	//usleep(50);
	return false;
      }
    }

    got += numbytes;
    request = datasize - got;

    fflush(stdout);
  } while(got < datasize);
  //printf("done w/ recv\n");

  //message[numbytes] = '\0';
  
  //printf("Client-Received: %s", message);     
  
  //printf("Client-Closing sockfd\n");
  fflush(stdout);
  
  return true;
   
}
