#include <string>
#include "src/base/CommandLineParser.h"
#include "src/base/FileParser.h"
#include "src/general/DNAVector.h"
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <zlib.h>
#include "kseq.h"
#include <fstream>
/*
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* a,double b) { return b; }
# endif
*/

#define gpuErrorchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


using namespace std;

KSEQ_INIT(gzFile,gzread);

__device__ double Abs(double a)
{
	if (a < 0)
		return -a;
	return a;
}

inline long long RandomInt(long long n)
{
	long long r =   ((long long)( rand( ) ) << 31 ) ^ (long long)( rand( ) );
	r = r % n;
	return r;
}

double RandomFloat(double n)
{
	long long r =   ((long long)( rand( ) ) << 31 ) ^ (long long)( rand( ) );
	r = r % 0x7FFFFFFF;
	return n*((double)r)/(double)0x7FFFFFFF;
}



__global__ void Distance(double *distance,double *nn, double *inp,int neurons,int inp_size)
{
	int tid = threadIdx.x;

	double d = 0.;

	for (int i=0; i<inp_size; i++) {

		double a = inp[i];
		double b = nn[tid*inp_size+i];
		//printf("b %f\n",b);
		d += (a-b)*(a-b);
	}
	distance[tid] =  sqrt(d) / inp_size;
}

__global__ void Distance2(double *distance,double *nn, double *inp,int neurons,int inp_size)
{
	int tid = threadIdx.x;
	if(tid<neurons){
	double d = 0.;
	//printf("thread id %d\n",tid);
	//printf("mem addr %x\n",inp);
	for (int i=0; i<inp_size; i++) {
	//printf("hiiiiiiiiiiiiiiiiiiiiiii\n");

		double a = inp[i];
		//printf("A %f \t",a);
		double b = nn[tid*inp_size+i];
		//printf("b: %f",b);
		d += (a-b)*(a-b);
	}
	distance[tid] =  sqrt(d) / inp_size;
	}
	//printf("distance %f\n",distance[tid]);
}

__global__ void CalcScores(int *indices,double *nn,double *inp,int inp_size,double *scores)
{
	int tid = threadIdx.x;

	double d = 0.;

	for (int i=0; i<inp_size; i++) {

		double a = inp[i];
		double b = nn[indices[tid]*inp_size+i];
		d += (a-b)*(a-b);
	}
	scores[tid] =  sqrt(d) / inp_size;
	//    printf("%f\n",scores[tid]);

}
__device__ void Update(double *nn, double *inp,double weight,int rot,int inp_size,int tid) {

	int i;
	while (rot < 0) {
		rot += inp_size;
	}
	for (i=tid*inp_size; i<tid*inp_size+inp_size; i++) {

		int ind = (i+rot) % inp_size;

		nn[i] = nn[i]*(1.-weight) + weight*inp[ind];
		//printf("addr: %2x nn  %f tid %d index %d ind %d\n",nn,nn[i],tid,i-tid*inp_size,ind);
	}
}
__global__ void Learn(double *nn, double *inp, double *distances,int neurons, int inp_size,
		int *index, double m_distance,double *m_beta, double ext_weight,
		double m_timeShift)
{
	int tid = threadIdx.x;
	double m_decay = 0.999;
	double m_floor = 0.01;

	double dist = Abs(tid - (*index));
	int tDist = tid-(*index);

	if (dist > neurons/2) {
		dist = neurons - dist;
		tDist = -tDist;
	}

	dist *= m_distance;

	double weight = exp(-dist);
	weight *= (*m_beta);
	weight *= ext_weight;
	//printf("Beta :%f Dist: %f,weig %f, ext_we %f, tid : %d\n",*m_beta,dist,weight,ext_weight,tid);
	Update(nn, inp, weight, m_timeShift*tDist,inp_size,tid);
	(*m_beta) *= m_decay;
	if (*m_beta < m_floor)
		*m_beta = m_floor;
}


double get_dist(double *inp,int inp_size,double *nn,int best)
{

	double d = 0.;

	for (int i=0; i<inp_size; i++) {

		double a = inp[i];
		double b = nn[best*inp_size+i];
		d += (a-b)*(a-b);
	}
	double score =  sqrt(d) / inp_size;
	return score;
}
//__device__ long long randomi(long long n) {
__device__ int randomi(int n) {

	//int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	curandState state;
	curand_init((unsigned long long)clock() , 0,  0, &state);
	
	unsigned int rand1 = curand(&state);
	//curandState state1;
	//curand_init((unsigned long long)clock()+100, 0,  0, &state1);

	//long long rand2 = curand(&state1);
	//printf("rand1 %d rand2 %d\n",rand1,rand2);
	//long long r =   (rand1 << 31 ) ^ (long long)( rand2 );
	int r = rand1 % n;
	//printf("n %d, r %d\n",n,r);
	return r;
}

__device__ double d_get_dist(double *inp,int inp_size,double *nn,int best)
{

	double d = 0.;

	for (int i=0; i<inp_size; i++) {

		double a = inp[i];
		double b = nn[best*inp_size+i];
		d += (a-b)*(a-b);
	}
	double score =  sqrt(d) / inp_size;
	return score;
}

__global__ void getscoreskernel(double *nn,kstring_t *test, double *score, int niter, int inp_size,int neurons,int nreads)
{


	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;				
	int nr = blockId * blockDim.x + threadIdx.x; 	
	if(nr<nreads){
		
		//printf("NN in kernel %f\n",nn[0]);
		double *seq;
		seq = new double[inp_size];
		//printf ("mem %x \n",seq);
		//cudaMalloc((void**)&seq,sizeof(double)*inp_size);
		double *distance = new double[neurons];
		//double *distance;
		//cudaMalloc((void**)&distance,sizeof(double)*inp_size);
		for(int i=0;i<niter; i++)
		{
			int index1 = randomi(test[nr].l- (inp_size/2));
			//printf("len %d index %d\n",test[nr].l,index1);
			int k= 0;
			bool b  = true;
			
			memset(seq, 0 ,inp_size*sizeof(double));
			char *str = test[nr].s;
			//printf("str %c \n",str[2]);
			
			//for(int j= index1;j<index1+(inp_size/2);j++)
			for(int j= 0;j<10;j++)
			{
				if (str[j] == 'A'){
					seq[k] = -1.;
				}
				if (str[j] == 'T')
					seq[k] = 1.;
				k++;
				if (str[j] == 'C')
					seq[k] = -1.;
				if (str[j] == 'G')
					seq[k] = 1.;
				if (str[j] == 'N')
				{
					b = false;
					break;
				}
				k++;

			}
			/*for(int x=0;x<inp_size;x++)
				printf("seq %f \t",seq[x]);
			printf("\n");
			*/
			//printf("NN in kernel %f\n",nn[0]);
			//getDistance(distance,nn, seq,inp_size);
			Distance2<<<1,neurons>>>(distance,nn,seq,neurons,inp_size);
			//Distance2<<<1,2>>>(distance,nn,seq,neurons,inp_size);
			//daDeviceSynchronize();
			__syncthreads();

			//cudaError err = cudaGetLastError();
			/*if ( cudaSuccess != err )
			{
				printf( "cudaCheckError() failed with  %s\n",cudaGetErrorString( err ) );

			}*/
			/*for(int x=0;x<10;x++)
			{
				printf("%lf\t",distance[x]);
			}*/
			
			//get best neuron
			//cublasHandle_t handle;
			//cublasStatus_t stat;
			//cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_DEVICE);
			//cublasCreate(&handle);
			int index = 0;
			for(int di = 0 ;di < neurons;di++)
			{
				if(distance[di]<distance[index])
				{
					index = di;
				}
			}
			
			//int *index;
			/*gpuErrorchk(cudaMalloc((void**)&index,sizeof(int)));
			stat = cublasIdamin(handle, neurons, distance, 1, index);
			printf("Index %d\n",*index);*/
			// *index -= 1;

			double pscore = d_get_dist(seq,inp_size,nn,index);
			//printf("%f\n",pscore);
			//atomic add to score
			atomicAdd(score,pscore);
			//printf("%f\n",*score);
			//cublasDestroy(handle);
		}
		delete(distance);
		delete(seq);
	}
}


int main( int argc, char** argv )
{

	//read fasta and fastq
	cout<<"Reading input files......\n";

	vecDNAVector ecoli,vchol, test; 
	ecoli.Read("/home/dell/som_25Feb/sequence.fasta");
	vchol.Read("/home/dell/som_25Feb/cholera.fasta");
	

	gzFile fp;
	kseq_t *rseq;
	int l;

	vector<kstring_t>reads;

	fp = gzopen("/home/dell/som_25Feb/SRR2724094_1.fastq","r");
	rseq  = kseq_init(fp);
	int count = 0;
	
	while((l = kseq_read(rseq)) >= 0)
	{
		kstring_t r_seq ;
		string str(rseq->seq.s);
		r_seq.l = rseq->seq.l;
		char * S = new char[str.length() + 1];
		strcpy(S,str.c_str());
		r_seq.s =S;
		reads.push_back(r_seq);
		count++;
	}
	
	cout<<"num of reads "<<count<<endl;
	int nreads = count;//reads.size();
	kstring_t *dreads;
	gpuErrorchk(cudaMalloc((void**)&dreads,sizeof(kstring_t)*count));
	gpuErrorchk(cudaMemcpy(dreads,reads.data(),sizeof(kstring_t)*count,cudaMemcpyHostToDevice));
	for (int i =0;i<count;i++){
		int l = reads[i].l;
		char *h_rd = reads[i].s;
		char *d_rd;
		gpuErrorchk(cudaMalloc((void**)&d_rd,sizeof(char)*l));
		gpuErrorchk(cudaMemcpy(d_rd,h_rd,sizeof(char)*l,cudaMemcpyHostToDevice));
		//binding
		gpuErrorchk(cudaMemcpy(&(dreads[i].s),&d_rd,sizeof(dreads[i].s),cudaMemcpyHostToDevice));
	}	

	//  cout<<"Creating Networks.........\n";
	int size = 10;
	int neurons = 512;

	int nrefs = 2;

	cout<<"Creating Networks of size "<<neurons<<endl;
	//creating two networks
	double *nn[nrefs];
	double *dnn[nrefs];


	int network_size = neurons*2*size;
	int input_size = 2*size;
	const clock_t begin_time = clock();
	cudaStream_t stream[nrefs];
	for(int i = 0; i < nrefs; ++i)
	{
		cudaStreamCreate(&stream[i]);
		nn[i] = new double[network_size];
		cudaMalloc((void**)&dnn[i], network_size*sizeof(double));
	}

	//initializing both networks
	int minus = -1;
	int plus =1;
	for(int i=0; i<nrefs ; i++)
	{
		for(int j=0;j<network_size;j++)
			nn[i][j] = minus + RandomFloat(plus-minus);
	}


	//  cout<<"Copying Network to device........\n";

	//input to network
	double *seq[nrefs];

	double *dseq[nrefs];

	double *distance[nrefs];

	double *beta[nrefs],*hbeta[nrefs];

	//cout<<"Initializing learning parameters.....\n";

	double m_distance = 0.5;
	int m_timeShift = 1;
	double ext_weight =1.0;
	double m_decay = 0.999;
	double m_floor = 0.01;

	int n = ecoli[0].isize()/10;

	cublasHandle_t handle[nrefs];
	cublasStatus_t stat[nrefs];

	int *index[nrefs];
	int *dindex[nrefs];
	int rindex[nrefs];

	for(int i=0;i<nrefs;i++)
	{

		gpuErrorchk(cudaMemcpyAsync(dnn[i], nn[i], sizeof(double)*network_size, cudaMemcpyHostToDevice));

		seq[i] = new double[input_size];
		cudaMalloc((void**)&dseq[i], input_size*sizeof(double));

		gpuErrorchk(cudaMalloc((void**)&distance[i], neurons*sizeof(double)));
		gpuErrorchk(cudaMemset((void*)distance[i], 0, neurons*sizeof(double)));

		cudaMalloc((void **)&beta[i],sizeof(double));
		hbeta[i] = new double[1];
		hbeta[i][0] = 0.3;
		gpuErrorchk(cudaMemcpy(beta[i], hbeta[i], sizeof(double), cudaMemcpyHostToDevice));

		index[i] = new int[1];
		cudaMalloc((void**)&dindex[i],sizeof(int));

		cublasCreate(&handle[i]);
	}
	//n=2;
	for (int i=0; i<n; i++) {

		rindex[0] = RandomInt(ecoli[0].isize()-size);
		rindex[1] = RandomInt(vchol[0].isize()-size);
		//		if (i % 1000 == 0)
		//			cout << i << " of " << n << endl;
		//train on ecoli
		int k = 0;
		bool b1 = true;
		memset(seq[0],0,2*size*sizeof(double));
		for (int j=rindex[0]; j<rindex[0]+size; j++)
		{
			if ((ecoli[0])[j] == 'A')
				seq[0][k] = -1.;
			if ((ecoli[0])[j] == 'T')
				seq[0][k] = 1.;
			k++;
			if ((ecoli[0])[j] == 'C')
				seq[0][k] = -1.;
			if ((ecoli[0])[j] == 'G')
				seq[0][k] = 1.;
			if ((ecoli[0])[j] == 'N')
			{
				b1 = false;
				break;
			}
			k++;
		}
		k = 0;
		bool b2 = true;

		memset(seq[1],0,2*size*sizeof(double));
		for (int j=rindex[1]; j<rindex[1]+size; j++)
		{
			if ((vchol[0])[j] == 'A')
				seq[1][k] = -1.;
			if ((vchol[0])[j] == 'T')
				seq[1][k] = 1.;
			k++;
			if ((vchol[0])[j] == 'C')
				seq[1][k] = -1.;
			if ((vchol[0])[j] == 'G')
				seq[1][k] = 1.;
			if ((vchol[0])[j] == 'N')
			{
				b2 = false;
				break;
			}
			k++;
		}
		if (b1 && b2)
		{
			//cout<<"Learn\n";
			for(int s =0; s<nrefs;s++)
			{

				gpuErrorchk(cudaMemcpyAsync(dseq[s], seq[s], sizeof(double)*input_size, cudaMemcpyHostToDevice,stream[s]));
				Distance<<<1,neurons,0,stream[s]>>>(distance[s],dnn[s],dseq[s],neurons,input_size);

				cublasSetStream(handle[s],stream[s]);

				stat[s] = cublasIdamin(handle[s], neurons, distance[s], 1, index[s]);
				//cout<<*index[s]<<"\n";
				if (stat[s] != CUBLAS_STATUS_SUCCESS)
					printf("min failed\n");

			}
			index[0][0] -= 1; 
			index[1][0] -= 1; 
			for(int s =0; s<nrefs;s++)
			{
				gpuErrorchk(cudaMemcpyAsync(dindex[s], index[s], sizeof(int), cudaMemcpyHostToDevice));

				Learn<<<1,neurons,0,stream[s]>>>(dnn[s], dseq[s], distance[s],neurons,input_size,dindex[s],m_distance,beta[s],ext_weight,m_timeShift);
			}
		}


	}
	cout<<"Learned\n";
	gpuErrorchk(cudaMemcpyAsync(nn[0], dnn[0], sizeof(double)*network_size, cudaMemcpyDeviceToHost,stream[0]));
	gpuErrorchk(cudaMemcpyAsync(nn[1], dnn[1], sizeof(double)*network_size, cudaMemcpyDeviceToHost,stream[1]));
	cout<<"copying to host\n";

	cout << "Training Time "<<(float( clock () - begin_time )/CLOCKS_PER_SEC)<<endl;
	cout<<endl;
	int n_iter = 10;
	//double score1=0.;
	//double score2=0.;
	
	ofstream file("ecoli_nn.dat",ios::out|ios::binary);
        ofstream file1("vcholera_nn.dat",ios::out|ios::binary);
        if(!file||!file1){
                cout<<"Error in creating file...\n";
                return -1;
        }
	file.write(reinterpret_cast<const char*>(nn[0]),streamsize(sizeof(double)*neurons));
	file.write(reinterpret_cast<const char*>(nn[1]),streamsize(sizeof(double)*neurons));
	file.close();
	file1.close();
	cout<<"Data saved into file the file.\n";
	
	ifstream rfile("ecoli_nn.dat",ios::in|ios::binary);  
        ifstream rfile1("vcholera_nn.dat",ios::in|ios::binary);

        if(!rfile||!rfile1){
                cout<<"Error in creating file...\n";
                return -1;
        }
	if(rfile.read(reinterpret_cast<char*>(nn[0]),streamsize(sizeof(double)*neurons))){
			cout<<"Data extracted from file..\n";
	}
	else{
		cout<<"Error in reading data from file...\n";
		return -1;
	}
	if(rfile.read(reinterpret_cast<char*>(nn[1]),streamsize(sizeof(double)*neurons))){
			cout<<"Data extracted from file..\n";
	}
	else{
		cout<<"Error in reading data from file...\n";
		return -1;
	}
	rfile.close();
	rfile1.close();

	cout << "Testing." << endl;

	double *score[nrefs];
	double *dscore[nrefs];

	for(int s = 0; s < nrefs ; s++)
	{
		score[s] = new double[1];
		gpuErrorchk(cudaMalloc((void**)&dscore[s], sizeof(double)));
		gpuErrorchk(cudaMemset(dscore[s],0,sizeof(double)));
		//cudaMalloc((void**)&dscore[s], sizeof(double));
	}

	const clock_t begin_time1 = clock();

	int f= ceil(sqrt(nreads));
	dim3 dims(f,f);
	for(int s = 0; s < nrefs;s++ )
	{
		getscoreskernel<<<dims,1,0,stream[s]>>>(dnn[s],dreads,dscore[s],n_iter,input_size,neurons,nreads);
		cudaDeviceSynchronize();
		//double score;
		cudaMemcpy(score[s],dscore[s],sizeof(double),cudaMemcpyDeviceToHost);
		cout<<*score[s]<<endl;
	}
	cout << "Testing Time "<<float( clock () - begin_time1 )/ CLOCKS_PER_SEC<<endl;
	cout<<endl;
	//cout<<score1<<" "<<score2<<" dif: "<<float(score1-score2)<<endl;
	//cublasDestroy(handle);
	for (int i =0;i<nrefs;i++){
		delete(index[i]);
		cudaFree(dnn[i]);
		cudaFree(beta[i]);
		cudaFree(distance[i]);
		cudaFree(dindex[i]);
		delete(nn[i]);
		delete(hbeta[i]); 
		delete(seq[i]);
		cudaFree(dseq[i]);
		delete(score[i]);
		cudaFree(dscore[i]);
		cublasDestroy(handle[i]);
	}
	cudaFree(dreads);
	return 0;
}
