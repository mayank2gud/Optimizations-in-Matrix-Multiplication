__global__ void matrixMul(int *matA, int *matB, int *output, int N)
{
	int rowA = blockIdx.y * blockDim.y + threadIdx.y;
  	int colB = blockIdx.x * blockDim.x + threadIdx.x;
  	int i=rowA;
  	int j=colB;
  	if(i%2==0 &&j%2==1||i%2==1&&j%2==0)
  	return;
  	//first half
  	if(i%2==0)
  	{
           output[(i/2)*N+j/2]=0;
           for(int k=0;k<N;k++)
           {
               output[(i/2)*N+j/2]+=matA[i*N+k]*matB[j*N+k]; //output[(i/2)][j/2]+=matA[i][k]*matB[j][k];
           }
      
 	}

//for 2nd half
   	else
   	{
           output[(i/2)*N+N/2+j/2]=0;
           for(int k=0;k<N;k++)
           {
               output[(i/2)*N+N/2+j/2]+=matA[i*N+k]*matB[j*N+k]; //output[(i/2)][N/2+j/2]+=matA[i][k]*matB[j][k];
           }
        }  
       
   		
            
            
}
// Fill in this function
void gpuThread(int N, int *matA, int *matB, int *output)
{
	int t;
	//transpose in cpu
	for(int i = 0; i < N; ++i)
   	{
         for(int j = i+1; j < N; ++j)
      	 {
           t=matB[i*N+j];
           matB[i*N+j]=matB[j*N+i];
           matB[j*N+i]=t;
      	 }
  	}
  	//exchange
  	for(int i=0;i<N;i+=2)
  	 {
     	  for(int j=0;j<N;j+=2)
     	  {
           t=matB[i*N+j+1];//i, j+1
           matB[i*N+j+1]=matB[i*N+j]; //
           matB[i*N+j]=matB[(i+1)*(N)+j+1];
           matB[(i+1)*(N)+j+1]=matB[(i+1)*(N)+j];
           matB[(i+1)*(N)+j]=t;
      	  }
  	 }
	
	size_t bytes=N*N*sizeof(int);
	//device memory
	int *a,*b,*c;
	cudaMalloc(&a,bytes);
	cudaMalloc(&b,bytes);
	cudaMalloc(&c,bytes/2);
	
	
	/*cudaMallocManaged(&a,bytes);
	cudaMallocManaged(&b,bytes);
	cudaMallocManaged(&c,bytes/2);*/
	
	//gpuErrchk(cudaMalloc((void**)&c,sizeof(int)*N*N/2));
	/*printf("%d",a[0]);
	printf("\n...C.before...\n");
	
  	for(int i=0;i<N*N/2;i++)
	{
		if(i%N==0)cout<<endl;
		cout<<a[i]<<" ";
	}
	  	printf("\n.......\n");*/
	
	cudaMemcpy(a,matA, bytes, cudaMemcpyHostToDevice);
  	cudaMemcpy(b,matB, bytes, cudaMemcpyHostToDevice);
  	
  	int threads=16;
	
	int blocks_x=N/threads;
	int blocks_y=N/(threads);
	
	dim3 TH(threads,threads);
	dim3 BL(blocks_x,blocks_y);
	
	matrixMul<<<BL,TH>>>(a,b,c,N);
	/*printf("\n...C....\n");
  	for(int i=0;i<N*N/2;i++)
	{
		if(i%N==0)cout<<endl;
		cout<<c[i]<<" ";
	}
	  	printf("\n.......\n");*/
	
	//cudaMemcpy(c,output, bytes/2, cudaMemcpyDeviceToHost);
	cudaMemcpy(output,c, bytes/2, cudaMemcpyDeviceToHost);
	/*printf("\n...C....\n");
	for(int i=0;i<N*N/2;i++)
	{
		if(i%N==0)cout<<endl;
		cout<<output[i]<<" ";
	}
	  	printf("\n.......\n");*/
	
	cudaFree(a);
  	cudaFree(b);
 	cudaFree(c);
	
}