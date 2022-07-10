#pragma warning(disable : 4996)

#include<iostream>
#include<fstream>
#include<stdlib.h>
#include<ctime>
#include<bitset>
#include<vector>
#include<string>
#include<sstream>

#include<unordered_set>

#include<bitset>

#include<windows.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cuda_runtime_api.h>

#include<immintrin.h>  // AVX

#include<mutex>
#include<condition_variable>

using namespace std;

#define THREAD_NUM 8  // 线程数量
const int BLOCK_SIZE = 1024;


//--------------------------------------函数声明--------------------------------------


int n = 0;  // 矩阵大小
const int k = 1;

int ek_num;  // 非零消元子个数
int et_num;  // 导入被消元行行数

/* 所有数据规模：
1:  130   22    8
2:  254   106   53
3:  562   170   53
4:  1011  539   263
5:  2362  1226  453
6:  3799  2759  1953
7:  8399  6375  4535
8:  23075 18748 14325
9:  39060 23904 14921
10: 43577 39477 54274
11: 85401 5724  756
*/

//------------------------------------数据导入工具------------------------------------
const int column_num_c = 2362;
const int ek_num_c = 1226;  // 非零消元子个数
const int et_num_c = 453;  // 导入被消元行行数

string dir = "C:/Users/CCC/source/repos/GrobnerGE/GrobnerGE/data/t5/";
stringstream ss;


int bit_size = column_num_c / 32 + 1;
class MyBitSet {
public:
	int head;  // 首项
	int* content;

	MyBitSet() {
		head = -1; content = new int[bit_size];
		for (int i = 0; i < bit_size; i++) content[i] = 0;
	}

	// bool operator[](size_t index) {}

	MyBitSet& operator^=(const MyBitSet& b) {  // 默认两个输入bitset长度相同
		for (int i = 0; i < bit_size; i++) content[i] ^= b.content[i];
		for (int i = 0; i < bit_size; i++)
		{
			for (int j = 0; j < 32; j++)
			{
				if ((content[i] & (1 << j)))
				{
					head = i * 32 + j;
					return *this;
				}
			}
		}
		head = -1;
		return *this;
	}

	MyBitSet& my_xor_AVX(const MyBitSet& b) {
		__m256i v_this, v_b;
		int i = 0;
		for (i; i < bit_size - 8; i += 8) {
			v_this = _mm256_loadu_si256((__m256i*) & content[i]);
			v_b = _mm256_loadu_si256((__m256i*) & b.content[i]);
			v_this = _mm256_xor_si256(v_this, v_b);
			_mm256_storeu_si256((__m256i*) & content[i], v_this);
		}
		for (i; i < bit_size; i++)
		{
			content[i] ^= b.content[i];
		}
		for (int i = 0; i < bit_size; i++)
		{
			for (int j = 0; j < 32; j++)
			{
				if ((content[i] & (1 << j)))
				{
					head = i * 32 + j;
					return *this;
				}
			}
		}
		head = -1;
		return *this;
	}


	int test(int index) {
		return content[index / 32] & (1 << (index % 32)) ? 1 : 0;  // 寻址方式
	}

	void set(int index) {  // 置位
		content[index / 32] |= (1 << (index % 32));
	}

	bool any() {
		for (int i = 0; i < bit_size; i++) if (content[i]) return true;
		return false;
	}

private:

};


bitset<column_num_c> eks_c[column_num_c];  // 消元子，开大一些便于检索与升格
bitset<column_num_c> ets_c[et_num_c];

int lp_eks_c[column_num_c];
int lp_ets_c[et_num_c];

int** eks_bits = new int* [column_num_c];
int** ets_bits = new int* [et_num_c];




MyBitSet eks[column_num_c];
MyBitSet ets[et_num_c];



long long head, tail, freq;


//------------------------------------输出调试函数------------------------------------

void reverse_output_MyB()
{
	ofstream outp(dir + "output_MyB.txt");
	for (int i = 0; i < et_num_c; i++)
	{
		for (int j = 0; j < n; j++) if (ets[i].test(j)) outp << n - j - 1 << " ";
		outp << endl;
	}
	outp.close();
}

//------------------------------------数据读取函数------------------------------------

void readData_reverse_MyB()
{  // 倒序读入数据，读入静态位集
	string inek, inet;
	stringstream ss_inek, ss_inet;
	ifstream inElimKey(dir + "elimkey.txt");  // 消元子
	ifstream inElimTar(dir + "elimtar.txt");  // 被消元行
	int inek_loc, p_ek = 0, inet_loc, p_et = 0;  // 用于数据读入
	int lp = -1;
	while (true)  // 读取消元子
	{
		getline(inElimKey, inek);
		ss_inek = stringstream(inek);
		while (ss_inek >> inek)
		{
			inek_loc = stoi(inek);
			if (lp == -1)
			{
				lp = column_num_c - inek_loc - 1;
				eks[lp].head = lp;
			}
			//cout << inek_loc << " ";
			eks[lp].set(column_num_c - inek_loc - 1);

		}
		lp = -1;  p_ek++;
		if (inek.empty()) break;
		//cout << eks_c[p_ek] << endl;
	}
	//cout << "ek_complete" << endl;

	while (true)  // 读取被消元行
	{
		getline(inElimTar, inet);
		ss_inet = stringstream(inet);
		while (ss_inet >> inet)
		{
			inet_loc = stoi(inet);
			if (lp == -1)
			{
				lp = column_num_c - inet_loc - 1;
				ets[p_et].head = lp;

			}
			//cout << inet_loc << " ";
			ets[p_et].set(column_num_c - inet_loc - 1);
		}
		lp = -1;  p_et++;
		if (inet.empty()) break;
		//cout << ets_c[p_et] << endl;
	}
	//cout << "et_complete" << endl;
	inElimKey.close();
	inElimTar.close();
	//cout << "init_complete" << endl;

}

void init_MyB() {
	readData_reverse_MyB();  // 逆序初始化消元子和被消元行阵列
	cout << "init_complete" << endl;
}



//-----------------------------------CUDA算法核函数-----------------------------------

__global__ void upshift_kernel(int i, int** gpu_eks, int** gpu_ets, int* gpu_lp_ets, int et_num_c) {
	//主线程做
	if (blockIdx.x == 0 && threadIdx.x == 0)
	for (int j = 0; j < et_num_c; j++)
	{
		if (i == gpu_lp_ets[j])  // 说明存在对应被消元行
		{
			gpu_eks[i] = gpu_ets[j];
			gpu_lp_ets[j] = -1;
			break;
		}
	}
}


__global__ void elim_kernel(int i, int** gpu_ek, int** gpu_et, int* gpu_lp_ets, int et_num_c, int column_num_c, int bit_size) {
	//int tx = blockDim.x * blockIdx.x + threadIdx.x;
	//if (!tx) data[k * N + k] = 1.0;//对角线元素设为 1
	//int row = k + 1 + blockIdx.x;//每个块负责一行
	//float t;
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockIdx.x;//每个块负责一个被消元行

	bool find = false;
	for (int j = row; j < et_num_c; j += blockDim.x)  // 循环划分并行化
	{
		if (i == gpu_lp_ets[j])  // 说明存在对应被消元行
		{
			//gpu_et[j] ^= gpu_ek[i];

			for (int k = 0; k < bit_size; k++) gpu_et[j][k] ^= gpu_ek[i][k];
			__syncthreads();
			if (threadIdx.x == 0)
			{
				for (int k = 0; k < bit_size; k++)
				{
					for (int l = 0; l < 32; l++)
					{
						if ((gpu_et[j][k] & (1 << l)))
						{
							gpu_lp_ets[j] = k * 32 + l;
							find = true;
							break;
						}
					}
					if (find)
					{
						find ^= find;
						break;
					}
					else gpu_lp_ets[j] = -1;
				}
			}
		}
	}
}



//-------------------------------------展示所有内容-------------------------------------



//------------------------------------CUDA消去算法------------------------------------

void CUDA_GE(bitset<column_num_c>* ek, bitset<column_num_c>* et, int* lp_ets) {

	for (int i = 0; i < column_num_c; i++)
	{	
		eks_bits[i] = new int[bit_size];
		for (int j = 0; j < bit_size; j++)
		{
			eks_bits[i][j] = 0;
		}
	}
	for (int i = 0; i < et_num_c; i++)
	{
		ets_bits[i] = new int[bit_size];
		for (int j = 0; j < bit_size; j++)
		{
			ets_bits[i][j] = 0;
		}
	}
	string inek, inet;
	stringstream ss_inek, ss_inet;
	ifstream inElimKey(dir + "elimkey.txt");  // 消元子
	ifstream inElimTar(dir + "elimtar.txt");  // 被消元行
	int inek_loc, p_ek = 0, inet_loc, p_et = 0;  // 用于数据读入
	int lp = -1;
	while (true)  // 读取消元子
	{
		getline(inElimKey, inek);
		ss_inek = stringstream(inek);
		while (ss_inek >> inek)
		{
			inek_loc = stoi(inek);
			if (lp == -1)
			{
				lp = column_num_c - inek_loc - 1;
				//eks[lp].head = lp;
				lp_eks_c[lp] = lp;
			}
			//cout << inek_loc << " ";
			//eks[lp].set(column_num_c - inek_loc - 1);
			// 拆解set函数
			eks_bits[lp][(column_num_c - inek_loc - 1) / 32] |= (1 << ((column_num_c - inek_loc - 1) % 32));
		}
		lp = -1;  p_ek++;
		if (inek.empty()) break;
		//cout << eks_c[p_ek] << endl;
	}
	//cout << "ek_complete" << endl;

	while (true)  // 读取被消元行
	{
		getline(inElimTar, inet);
		ss_inet = stringstream(inet);
		while (ss_inet >> inet)
		{
			inet_loc = stoi(inet);
			if (lp == -1)
			{
				lp = column_num_c - inet_loc - 1;
				//ets[p_et].head = lp;
				lp_ets_c[p_et] = lp;
			}
			//cout << inet_loc << " ";
			//ets[p_et].set(column_num_c - inet_loc - 1);
			ets_bits[p_et][(column_num_c - inek_loc - 1) / 32] |= (1 << ((column_num_c - inek_loc - 1) % 32));
		}
		lp = -1;  p_et++;
		if (inet.empty()) break;
		//cout << ets_c[p_et] << endl;
	}
	//cout << "et_complete" << endl;
	inElimKey.close();
	inElimTar.close();
	//cout << "init_complete" << endl;

	//show_1d(temp, N);
	cudaError_t ret;//用于错误检查，当 CUDA 接口调用成功会返回 cudaSucess



	int** gpu_ets;
	int** gpu_eks;

	int** gpu_ets_bits = new int* [et_num_c];  // 存放的是指针首地址
	int** gpu_eks_bits = new int* [column_num_c];  // 存放的是指针首地址
	int* gpu_lp_ets;




	int gpu_et_size = et_num_c * sizeof(int*);
	int gpu_ek_size = column_num_c * sizeof(int*);

	int bitset_size = sizeof(int) * bit_size;
	int gpu_lp_ets_size = et_num_c * sizeof(int);

	for (int i = 0; i < et_num; i++)
	{
		int* bits;  // 暂时存放显存地址
		if (cudaMalloc(&bits, gpu_et_size) != cudaSuccess)  printf("cudaMalloc gpudata failed!\n");
		if (cudaMemcpy(bits, ets_bits[i], bitset_size, cudaMemcpyHostToDevice) != cudaSuccess) printf("cudaMemcpyHostToDevice failed!\n");
		gpu_ets_bits[i] = bits;
	}

	if (cudaMalloc(&gpu_ets, gpu_et_size) != cudaSuccess)  printf("cudaMalloc gpudata failed!\n");
	if (cudaMemcpy(gpu_ets, gpu_ets_bits, gpu_et_size, cudaMemcpyHostToDevice) != cudaSuccess) printf("cudaMemcpyHostToDevice failed!\n");

	for (int i = 0; i < column_num_c; i++)
	{
		int* bits;  // 暂时存放显存地址
		if (cudaMalloc(&bits, gpu_ek_size) != cudaSuccess)  printf("cudaMalloc gpudata failed!\n");
		if (cudaMemcpy(bits, eks_bits[i], bitset_size, cudaMemcpyHostToDevice) != cudaSuccess) printf("cudaMemcpyHostToDevice failed!\n");
		gpu_eks_bits[i] = bits;
	}

	if (cudaMalloc(&gpu_eks, gpu_ek_size) != cudaSuccess)  printf("cudaMalloc gpudata failed!\n");
	if (cudaMemcpy(gpu_eks, gpu_eks_bits, gpu_ek_size, cudaMemcpyHostToDevice) != cudaSuccess) printf("cudaMemcpyHostToDevice failed!\n");

	//分配显存空间并且进行错误检查
	if (cudaMalloc(&gpu_lp_ets, gpu_lp_ets_size) != cudaSuccess)  printf("cudaMalloc gpudata failed!\n");
	
	//将数据传输至 GPU 端并进行错误检查
	if (cudaMemcpy(gpu_lp_ets, lp_ets_c, gpu_lp_ets_size, cudaMemcpyHostToDevice) != cudaSuccess) printf("cudaMemcpyHostToDevice failed!\n");

	dim3 dimBlock(BLOCK_SIZE, 1), dimGrid(1, 1); //线程块、线程网格

	cudaEvent_t start, stop;  //计时器
	float elapsedTime = 0.0;
	cudaEventCreate(&start), cudaEventCreate(&stop);
	cudaEventRecord(start, 0);  //开始计时

	cudaError_t exec;
	for (int i = 0; i < column_num_c; i++)  // 取每个消元子，对被消元行进行操作，便于并行化
	{
		if (!(eks_bits[i][i / 32] & (1 << (i % 32))))  // 消元子被逆序初始化时满足“行号” = “首项”的条件
		{
			upshift_kernel << <1, 1 >> > (i, gpu_eks, gpu_ets, gpu_lp_ets, et_num_c);
		}
		cudaDeviceSynchronize();//CPU 与 GPU 之间的同步函数
		exec = cudaGetLastError();
		if (exec != cudaSuccess) printf("upshift_kernel failed, %s\n", cudaGetErrorString(exec));

		elim_kernel << <dimGrid, dimBlock >> > (i, gpu_eks, gpu_ets, gpu_lp_ets, et_num_c, column_num_c, bit_size);//负责消去任务的核函数

		cudaDeviceSynchronize();//CPU 与 GPU 之间的同步函数
		exec = cudaGetLastError();
		if (exec != cudaSuccess) printf("elim_kernel failed, %s\n", cudaGetErrorString(exec));
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);//停止计时
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("CUDA_GE:%f ms\n", elapsedTime);

	cudaError_t cudaStatus2 = cudaGetLastError();
	if (cudaGetLastError() != cudaSuccess) fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus2));
	//将数据传回 CPU 端并进行错误检查
	//if (cudaMemcpy(ek, gpu_ek, gpu_ek_size, cudaMemcpyDeviceToHost) != cudaSuccess) printf("cudaMemcpyDeviceToHost failed!\n");
	//if (cudaMemcpy(et, gpu_et, gpu_et_size, cudaMemcpyDeviceToHost) != cudaSuccess) printf("cudaMemcpyDeviceToHost failed!\n");
	//if (cudaMemcpy(lp_ets, gpu_lp_ets, gpu_et_size, cudaMemcpyDeviceToHost) != cudaSuccess) printf("cudaMemcpyDeviceToHost failed!\n");

	//reverse_output_c();

	//show_1d(result, N);  // 测试
	//show_in_file_1d(result, N);  // 测试

	//cudaFree(gpu_eks);//释放显存空间，用 CUDA 接口分配的空间必须用 cudaFree 释放

	//销毁计时器
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}


int main() {
	//freopen("input.txt", "r", stdin);
	//float* temp = new float[N * N];
	//for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) cin >> temp[i * N + j];

	cout << "矩阵大小为" << column_num_c << "，消元子个数为" << ek_num_c << "，被消元行行数为" << et_num_c << endl;


	//-----------------------------------------------------------------
	//float* t_1d = generate_1d(N);
	CUDA_GE(eks_c, ets_c, lp_ets_c);

	//-----------------------------------------------------------------

	//system("pause");
}
#pragma warning(disable : 4996)

#include<iostream>
#include<fstream>
#include<stdlib.h>
#include<ctime>
#include<bitset>
#include<vector>
#include<string>
#include<sstream>

#include<unordered_set>

#include<bitset>

#include<windows.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cuda_runtime_api.h>

#include<immintrin.h>  // AVX

#include<mutex>
#include<condition_variable>

using namespace std;

#define THREAD_NUM 8  // 线程数量
const int BLOCK_SIZE = 1024;


//--------------------------------------函数声明--------------------------------------


int n = 0;  // 矩阵大小
const int k = 1;

int ek_num;  // 非零消元子个数
int et_num;  // 导入被消元行行数

/* 所有数据规模：
1:  130   22    8
2:  254   106   53
3:  562   170   53
4:  1011  539   263
5:  2362  1226  453
6:  3799  2759  1953
7:  8399  6375  4535
8:  23075 18748 14325
9:  39060 23904 14921
10: 43577 39477 54274
11: 85401 5724  756
*/

//------------------------------------数据导入工具------------------------------------
const int column_num_c = 2362;
const int ek_num_c = 1226;  // 非零消元子个数
const int et_num_c = 453;  // 导入被消元行行数

string dir = "C:/Users/CCC/source/repos/GrobnerGE/GrobnerGE/data/t5/";
stringstream ss;


int bit_size = column_num_c / 32 + 1;
class MyBitSet {
public:
	int head;  // 首项
	int* content;

	MyBitSet() {
		head = -1; content = new int[bit_size];
		for (int i = 0; i < bit_size; i++) content[i] = 0;
	}

	// bool operator[](size_t index) {}

	MyBitSet& operator^=(const MyBitSet& b) {  // 默认两个输入bitset长度相同
		for (int i = 0; i < bit_size; i++) content[i] ^= b.content[i];
		for (int i = 0; i < bit_size; i++)
		{
			for (int j = 0; j < 32; j++)
			{
				if ((content[i] & (1 << j)))
				{
					head = i * 32 + j;
					return *this;
				}
			}
		}
		head = -1;
		return *this;
	}

	MyBitSet& my_xor_AVX(const MyBitSet& b) {
		__m256i v_this, v_b;
		int i = 0;
		for (i; i < bit_size - 8; i += 8) {
			v_this = _mm256_loadu_si256((__m256i*) & content[i]);
			v_b = _mm256_loadu_si256((__m256i*) & b.content[i]);
			v_this = _mm256_xor_si256(v_this, v_b);
			_mm256_storeu_si256((__m256i*) & content[i], v_this);
		}
		for (i; i < bit_size; i++)
		{
			content[i] ^= b.content[i];
		}
		for (int i = 0; i < bit_size; i++)
		{
			for (int j = 0; j < 32; j++)
			{
				if ((content[i] & (1 << j)))
				{
					head = i * 32 + j;
					return *this;
				}
			}
		}
		head = -1;
		return *this;
	}


	int test(int index) {
		return content[index / 32] & (1 << (index % 32)) ? 1 : 0;  // 寻址方式
	}

	void set(int index) {  // 置位
		content[index / 32] |= (1 << (index % 32));
	}

	bool any() {
		for (int i = 0; i < bit_size; i++) if (content[i]) return true;
		return false;
	}

private:

};


bitset<column_num_c> eks_c[column_num_c];  // 消元子，开大一些便于检索与升格
bitset<column_num_c> ets_c[et_num_c];

int lp_eks_c[column_num_c];
int lp_ets_c[et_num_c];

int** eks_bits = new int* [column_num_c];
int** ets_bits = new int* [et_num_c];




MyBitSet eks[column_num_c];
MyBitSet ets[et_num_c];



long long head, tail, freq;


//------------------------------------输出调试函数------------------------------------

void reverse_output_MyB()
{
	ofstream outp(dir + "output_MyB.txt");
	for (int i = 0; i < et_num_c; i++)
	{
		for (int j = 0; j < n; j++) if (ets[i].test(j)) outp << n - j - 1 << " ";
		outp << endl;
	}
	outp.close();
}

//------------------------------------数据读取函数------------------------------------

void readData_reverse_MyB()
{  // 倒序读入数据，读入静态位集
	string inek, inet;
	stringstream ss_inek, ss_inet;
	ifstream inElimKey(dir + "elimkey.txt");  // 消元子
	ifstream inElimTar(dir + "elimtar.txt");  // 被消元行
	int inek_loc, p_ek = 0, inet_loc, p_et = 0;  // 用于数据读入
	int lp = -1;
	while (true)  // 读取消元子
	{
		getline(inElimKey, inek);
		ss_inek = stringstream(inek);
		while (ss_inek >> inek)
		{
			inek_loc = stoi(inek);
			if (lp == -1)
			{
				lp = column_num_c - inek_loc - 1;
				eks[lp].head = lp;
			}
			//cout << inek_loc << " ";
			eks[lp].set(column_num_c - inek_loc - 1);

		}
		lp = -1;  p_ek++;
		if (inek.empty()) break;
		//cout << eks_c[p_ek] << endl;
	}
	//cout << "ek_complete" << endl;

	while (true)  // 读取被消元行
	{
		getline(inElimTar, inet);
		ss_inet = stringstream(inet);
		while (ss_inet >> inet)
		{
			inet_loc = stoi(inet);
			if (lp == -1)
			{
				lp = column_num_c - inet_loc - 1;
				ets[p_et].head = lp;

			}
			//cout << inet_loc << " ";
			ets[p_et].set(column_num_c - inet_loc - 1);
		}
		lp = -1;  p_et++;
		if (inet.empty()) break;
		//cout << ets_c[p_et] << endl;
	}
	//cout << "et_complete" << endl;
	inElimKey.close();
	inElimTar.close();
	//cout << "init_complete" << endl;

}

void init_MyB() {
	readData_reverse_MyB();  // 逆序初始化消元子和被消元行阵列
	cout << "init_complete" << endl;
}



//-----------------------------------CUDA算法核函数-----------------------------------

__global__ void upshift_kernel(int i, int** gpu_eks, int** gpu_ets, int* gpu_lp_ets, int et_num_c) {
	//主线程做
	if (blockIdx.x == 0 && threadIdx.x == 0)
	for (int j = 0; j < et_num_c; j++)
	{
		if (i == gpu_lp_ets[j])  // 说明存在对应被消元行
		{
			gpu_eks[i] = gpu_ets[j];
			gpu_lp_ets[j] = -1;
			break;
		}
	}
}


__global__ void elim_kernel(int i, int** gpu_ek, int** gpu_et, int* gpu_lp_ets, int et_num_c, int column_num_c, int bit_size) {
	//int tx = blockDim.x * blockIdx.x + threadIdx.x;
	//if (!tx) data[k * N + k] = 1.0;//对角线元素设为 1
	//int row = k + 1 + blockIdx.x;//每个块负责一行
	//float t;
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockIdx.x;//每个块负责一个被消元行

	bool find = false;
	for (int j = row; j < et_num_c; j += blockDim.x)  // 循环划分并行化
	{
		if (i == gpu_lp_ets[j])  // 说明存在对应被消元行
		{
			//gpu_et[j] ^= gpu_ek[i];

			for (int k = 0; k < bit_size; k++) gpu_et[j][k] ^= gpu_ek[i][k];
			__syncthreads();
			if (threadIdx.x == 0)
			{
				for (int k = 0; k < bit_size; k++)
				{
					for (int l = 0; l < 32; l++)
					{
						if ((gpu_et[j][k] & (1 << l)))
						{
							gpu_lp_ets[j] = k * 32 + l;
							find = true;
							break;
						}
					}
					if (find)
					{
						find ^= find;
						break;
					}
					else gpu_lp_ets[j] = -1;
				}
			}
		}
	}
}



//-------------------------------------展示所有内容-------------------------------------



//------------------------------------CUDA消去算法------------------------------------

void CUDA_GE(bitset<column_num_c>* ek, bitset<column_num_c>* et, int* lp_ets) {

	for (int i = 0; i < column_num_c; i++)
	{	
		eks_bits[i] = new int[bit_size];
		for (int j = 0; j < bit_size; j++)
		{
			eks_bits[i][j] = 0;
		}
	}
	for (int i = 0; i < et_num_c; i++)
	{
		ets_bits[i] = new int[bit_size];
		for (int j = 0; j < bit_size; j++)
		{
			ets_bits[i][j] = 0;
		}
	}
	string inek, inet;
	stringstream ss_inek, ss_inet;
	ifstream inElimKey(dir + "elimkey.txt");  // 消元子
	ifstream inElimTar(dir + "elimtar.txt");  // 被消元行
	int inek_loc, p_ek = 0, inet_loc, p_et = 0;  // 用于数据读入
	int lp = -1;
	while (true)  // 读取消元子
	{
		getline(inElimKey, inek);
		ss_inek = stringstream(inek);
		while (ss_inek >> inek)
		{
			inek_loc = stoi(inek);
			if (lp == -1)
			{
				lp = column_num_c - inek_loc - 1;
				//eks[lp].head = lp;
				lp_eks_c[lp] = lp;
			}
			//cout << inek_loc << " ";
			//eks[lp].set(column_num_c - inek_loc - 1);
			// 拆解set函数
			eks_bits[lp][(column_num_c - inek_loc - 1) / 32] |= (1 << ((column_num_c - inek_loc - 1) % 32));
		}
		lp = -1;  p_ek++;
		if (inek.empty()) break;
		//cout << eks_c[p_ek] << endl;
	}
	//cout << "ek_complete" << endl;

	while (true)  // 读取被消元行
	{
		getline(inElimTar, inet);
		ss_inet = stringstream(inet);
		while (ss_inet >> inet)
		{
			inet_loc = stoi(inet);
			if (lp == -1)
			{
				lp = column_num_c - inet_loc - 1;
				//ets[p_et].head = lp;
				lp_ets_c[p_et] = lp;
			}
			//cout << inet_loc << " ";
			//ets[p_et].set(column_num_c - inet_loc - 1);
			ets_bits[p_et][(column_num_c - inek_loc - 1) / 32] |= (1 << ((column_num_c - inek_loc - 1) % 32));
		}
		lp = -1;  p_et++;
		if (inet.empty()) break;
		//cout << ets_c[p_et] << endl;
	}
	//cout << "et_complete" << endl;
	inElimKey.close();
	inElimTar.close();
	//cout << "init_complete" << endl;

	//show_1d(temp, N);
	cudaError_t ret;//用于错误检查，当 CUDA 接口调用成功会返回 cudaSucess



	int** gpu_ets;
	int** gpu_eks;

	int** gpu_ets_bits = new int* [et_num_c];  // 存放的是指针首地址
	int** gpu_eks_bits = new int* [column_num_c];  // 存放的是指针首地址
	int* gpu_lp_ets;




	int gpu_et_size = et_num_c * sizeof(int*);
	int gpu_ek_size = column_num_c * sizeof(int*);

	int bitset_size = sizeof(int) * bit_size;
	int gpu_lp_ets_size = et_num_c * sizeof(int);

	for (int i = 0; i < et_num; i++)
	{
		int* bits;  // 暂时存放显存地址
		if (cudaMalloc(&bits, gpu_et_size) != cudaSuccess)  printf("cudaMalloc gpudata failed!\n");
		if (cudaMemcpy(bits, ets_bits[i], bitset_size, cudaMemcpyHostToDevice) != cudaSuccess) printf("cudaMemcpyHostToDevice failed!\n");
		gpu_ets_bits[i] = bits;
	}

	if (cudaMalloc(&gpu_ets, gpu_et_size) != cudaSuccess)  printf("cudaMalloc gpudata failed!\n");
	if (cudaMemcpy(gpu_ets, gpu_ets_bits, gpu_et_size, cudaMemcpyHostToDevice) != cudaSuccess) printf("cudaMemcpyHostToDevice failed!\n");

	for (int i = 0; i < column_num_c; i++)
	{
		int* bits;  // 暂时存放显存地址
		if (cudaMalloc(&bits, gpu_ek_size) != cudaSuccess)  printf("cudaMalloc gpudata failed!\n");
		if (cudaMemcpy(bits, eks_bits[i], bitset_size, cudaMemcpyHostToDevice) != cudaSuccess) printf("cudaMemcpyHostToDevice failed!\n");
		gpu_eks_bits[i] = bits;
	}

	if (cudaMalloc(&gpu_eks, gpu_ek_size) != cudaSuccess)  printf("cudaMalloc gpudata failed!\n");
	if (cudaMemcpy(gpu_eks, gpu_eks_bits, gpu_ek_size, cudaMemcpyHostToDevice) != cudaSuccess) printf("cudaMemcpyHostToDevice failed!\n");

	//分配显存空间并且进行错误检查
	if (cudaMalloc(&gpu_lp_ets, gpu_lp_ets_size) != cudaSuccess)  printf("cudaMalloc gpudata failed!\n");
	
	//将数据传输至 GPU 端并进行错误检查
	if (cudaMemcpy(gpu_lp_ets, lp_ets_c, gpu_lp_ets_size, cudaMemcpyHostToDevice) != cudaSuccess) printf("cudaMemcpyHostToDevice failed!\n");

	dim3 dimBlock(BLOCK_SIZE, 1), dimGrid(1, 1); //线程块、线程网格

	cudaEvent_t start, stop;  //计时器
	float elapsedTime = 0.0;
	cudaEventCreate(&start), cudaEventCreate(&stop);
	cudaEventRecord(start, 0);  //开始计时

	cudaError_t exec;
	for (int i = 0; i < column_num_c; i++)  // 取每个消元子，对被消元行进行操作，便于并行化
	{
		if (!(eks_bits[i][i / 32] & (1 << (i % 32))))  // 消元子被逆序初始化时满足“行号” = “首项”的条件
		{
			upshift_kernel << <1, 1 >> > (i, gpu_eks, gpu_ets, gpu_lp_ets, et_num_c);
		}
		cudaDeviceSynchronize();//CPU 与 GPU 之间的同步函数
		exec = cudaGetLastError();
		if (exec != cudaSuccess) printf("upshift_kernel failed, %s\n", cudaGetErrorString(exec));

		elim_kernel << <dimGrid, dimBlock >> > (i, gpu_eks, gpu_ets, gpu_lp_ets, et_num_c, column_num_c, bit_size);//负责消去任务的核函数

		cudaDeviceSynchronize();//CPU 与 GPU 之间的同步函数
		exec = cudaGetLastError();
		if (exec != cudaSuccess) printf("elim_kernel failed, %s\n", cudaGetErrorString(exec));
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);//停止计时
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("CUDA_GE:%f ms\n", elapsedTime);

	cudaError_t cudaStatus2 = cudaGetLastError();
	if (cudaGetLastError() != cudaSuccess) fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus2));
	//将数据传回 CPU 端并进行错误检查
	//if (cudaMemcpy(ek, gpu_ek, gpu_ek_size, cudaMemcpyDeviceToHost) != cudaSuccess) printf("cudaMemcpyDeviceToHost failed!\n");
	//if (cudaMemcpy(et, gpu_et, gpu_et_size, cudaMemcpyDeviceToHost) != cudaSuccess) printf("cudaMemcpyDeviceToHost failed!\n");
	//if (cudaMemcpy(lp_ets, gpu_lp_ets, gpu_et_size, cudaMemcpyDeviceToHost) != cudaSuccess) printf("cudaMemcpyDeviceToHost failed!\n");

	//reverse_output_c();

	//show_1d(result, N);  // 测试
	//show_in_file_1d(result, N);  // 测试

	//cudaFree(gpu_eks);//释放显存空间，用 CUDA 接口分配的空间必须用 cudaFree 释放

	//销毁计时器
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}


int main() {
	//freopen("input.txt", "r", stdin);
	//float* temp = new float[N * N];
	//for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) cin >> temp[i * N + j];

	cout << "矩阵大小为" << column_num_c << "，消元子个数为" << ek_num_c << "，被消元行行数为" << et_num_c << endl;


	//-----------------------------------------------------------------
	//float* t_1d = generate_1d(N);
	CUDA_GE(eks_c, ets_c, lp_ets_c);

	//-----------------------------------------------------------------

	//system("pause");
}
