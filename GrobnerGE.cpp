#include<iostream>
#include<fstream>
#include<stdlib.h>
#include<ctime>
#include<bitset>
#include<vector>
#include<string>
#include<sstream>

#include<unordered_set>

#include<boost/dynamic_bitset.hpp>
#include<bitset>

#include<windows.h>

#include<immintrin.h>  // AVX

#include<pthread.h>
#include<semaphore.h>
#include<mutex>
#include<condition_variable>

using namespace std;
using namespace boost;

#define THREAD_NUM 8  // 线程数量

//--------------------------------------函数声明--------------------------------------

dynamic_bitset<> get_head(size_t index);
void output();
void GrobnerGE();
void db_GrobnerGE();  // 动态位集版本的特殊高斯消去，未来考虑补上稀疏矩阵版本
void readData_bitset();
void readData_reverse_bitset();
void reverse_output();


int n = 0;  // 矩阵大小
const int k = 1;

int ek_num;  // 非零消元子个数
int et_num;  // 导入被消元行行数
vector<dynamic_bitset<>> eks;
vector<int> lp_eks;
dynamic_bitset<>* ets = nullptr;
int* lp_ets = nullptr;

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

文件内部包含：
初代串行算法及部分优化算法
std::bitset下的串行算法及其重排串行算法，及其PThread算法
IO
*/

const int column_num_c = 85401;
const int ek_num_c = 5724;  // 非零消元子个数
const int et_num_c = 756;  // 导入被消元行行数


bitset<column_num_c> eks_c[column_num_c];  // 消元子，开大一些便于检索与升格
bitset<column_num_c> ets_c[et_num_c];

int lp_ets_c[et_num_c];
int lp_eks_c[column_num_c];

long long head, tail, freq;

//------------------------------------数据导入工具------------------------------------

string dir = "C:/Users/CCC/source/repos/GrobnerGE/GrobnerGE/data/t11/";
stringstream ss;

//------------------------------------线程数据结构------------------------------------

typedef struct {
	int t_id;  // 线程 id
	int tasknum;  // 任务数量
}PT_EliminationParam;

typedef struct {
	int t_id; //线程 id
	size_t index;
}PT_GetHeadParam;

//-------------------------------------信号量定义-------------------------------------
sem_t sem_gethead;
sem_t sem_workerstart[THREAD_NUM]; // 每个线程有自己专属的信号量
sem_t sem_workerend[THREAD_NUM];

sem_t sem_leader;
sem_t sem_Divsion[THREAD_NUM - 1];
sem_t sem_Elimination[THREAD_NUM - 1];

//------------------------------------barrier定义-------------------------------------
pthread_barrier_t barrier_gethead;
pthread_barrier_t _Barrier_Elimination;
pthread_barrier_t _Barrier_0_Operation;

//-------------------------------条件变量和互斥锁定义---------------------------------
pthread_mutex_t _mutex;
pthread_cond_t _cond;

//--------------------------------------线程函数--------------------------------------

void* PT_Elimination(void* param) {
	PT_EliminationParam* tt = (PT_EliminationParam*)param;
	int t_id = tt->t_id;
	bool find_flag = false;
	dynamic_bitset<>* temp = nullptr;
	for (int i = t_id; i < et_num; i += THREAD_NUM)
	{
		while (ets_c[i].any())
		{
			find_flag = false;
			size_t index = ets[i].find_first();
			for (int j = 0; j < eks.size(); j++)  // R[lp(E[i])
			{
				if (index == eks[j].find_first())
				{
					ets[i] ^= eks[j];
					find_flag = true;
					break;
				}
			}
			if (!find_flag)
			{
				pthread_mutex_lock(&_mutex);  // 使用互斥锁保证数据间同步
				eks.push_back(ets[i]);
				pthread_mutex_unlock(&_mutex);  // 解锁
				break;
			}
		}
	}
	//pthread_barrier_wait(&barrier_Elimination);
	pthread_exit(nullptr);
	return nullptr;
}


unordered_set<int> ek_set;
void* PT_Elimination_Static(void* param) {
	PT_EliminationParam* tt = (PT_EliminationParam*)param;
	int t_id = tt->t_id;
	for (int i = 0; i < column_num_c; i++)  // 取每个消元子，对被消元行进行操作，便于并行化
	{
		if (!eks_c[i].test(i))  // 消元子被逆序初始化时满足“行号” = “首项”的条件
		{
			pthread_barrier_wait(&_Barrier_Elimination);  // 保证同步
			if (t_id == 0)  // 零号线程负责升格消元子操作
			{
				for (size_t j = 0; j < et_num_c; j++) 
				{
					if (i == lp_ets_c[j])  // 说明存在对应被消元行
					{
						eks_c[i] = ets_c[j];
						lp_ets_c[j] = -1;
						break;
					}
				}
			}
			pthread_barrier_wait(&_Barrier_0_Operation);  // 在0号线程进行完消元子升格操作之后，需要保证数据同步
		}
		for (size_t j = t_id; j < et_num_c; j += THREAD_NUM)  // 循环划分并行化
		{
			if (i == lp_ets_c[j])  // 说明存在对应被消元行
			{
				ets_c[j] ^= eks_c[i];
				for (int k = i; k < column_num_c; k++)
				{
					if (ets_c[j].test(k))
					{
						lp_ets_c[j] = k;
						break;
					}
					if (k == column_num_c - 1) lp_ets_c[j] = -1;
				}
			}
		}
	}
	pthread_exit(nullptr);
	return nullptr;
}

void* PT_Elimination_Static_Block(void* param) {
	PT_EliminationParam* tt = (PT_EliminationParam*)param;
	int t_id = tt->t_id;
	int tasknum = t_id == THREAD_NUM - 1 ? et_num_c % (THREAD_NUM - 1) : et_num_c / (THREAD_NUM - 1);
	for (int i = 0; i < column_num_c; i++)  // 取每个消元子，对被消元行进行操作，便于并行化
	{
		if (!eks_c[i].test(i))  // 消元子被逆序初始化时满足“行号” = “首项”的条件
		{
			pthread_barrier_wait(&_Barrier_Elimination);  // 保证同步
			if (t_id == 0)  // 零号线程负责升格消元子操作
			{
				for (size_t j = 0; j < et_num_c; j++)
				{
					if (i == lp_ets_c[j])  // 说明存在对应被消元行
					{
						eks_c[i] = ets_c[j];
						lp_ets_c[j] = -1;
						break;
					}
				}
			}
			pthread_barrier_wait(&_Barrier_0_Operation);  // 在0号线程进行完消元子升格操作之后，需要保证数据同步
		}
		for (size_t j = t_id * tasknum, k = 0; k < tasknum; j++, k++)  // 块划分并行化
		{
			if (i == lp_ets_c[j])  // 说明存在对应被消元行
			{
				ets_c[j] ^= eks_c[i];
				for (int k = i; k < column_num_c; k++)
				{
					if (ets_c[j].test(k))
					{
						lp_ets_c[j] = k;
						break;
					}
					if (k == column_num_c - 1) lp_ets_c[j] = -1;
				}
			}
		}
	}
	pthread_exit(nullptr);
	return nullptr;
}



void* watch() {
	pthread_mutex_lock(&_mutex);
	while (true)
	{
		pthread_cond_wait(&_cond, &_mutex);
	}
	pthread_mutex_unlock(&_mutex);
	return nullptr;
}

//-------------------------------------PThread算法------------------------------------
void PT_GrobnerGE()
{
	pthread_cond_init(&_cond, nullptr);
	pthread_mutex_init(&_mutex, nullptr);
	pthread_barrier_init(&_Barrier_Elimination, nullptr, THREAD_NUM);
	pthread_barrier_init(&_Barrier_0_Operation, nullptr, THREAD_NUM);
	//int tasknum = eks.size() / THREAD_NUM;
	pthread_t* handles = (pthread_t*)malloc(THREAD_NUM * sizeof(pthread_t));  // 为线程句柄分配内存空间
	PT_EliminationParam* param = (PT_EliminationParam*)malloc(THREAD_NUM * sizeof(PT_EliminationParam));  // 存储线程参数
	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], nullptr, PT_Elimination, (void*)&param[t_id]);
	}

	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		pthread_join(handles[t_id], nullptr);
	}

	for (int i = 0; i < et_num; i++)
	{
		/*
		while (ets[i].any())
		{
			count1++;
			//size_t first = ets[i].find_first();  // lp(E[i])
			temp = get_head(ets[i].find_first());
			//temp = get_head(lp_ets[i]);  // R[lp(E[i])]，即消元子中与被消元行i中首项相同的项
			if (!(temp.size() == 0))
			{
				ets[i] ^= temp;
				//lp_ets[i] = ets[i].find_first();
				count2++;
			}
			else
			{
				count3++;
				eks.push_back(ets[i]);
				break;
			}
		}
		*/
	}
	pthread_mutex_destroy(&_mutex);
	pthread_cond_destroy(&_cond);
	pthread_barrier_destroy(&_Barrier_Elimination);
	pthread_barrier_destroy(&_Barrier_0_Operation);
}

void PT_GrobnerGE_Static()
{
	pthread_barrier_init(&_Barrier_Elimination, nullptr, THREAD_NUM);
	pthread_barrier_init(&_Barrier_0_Operation, nullptr, THREAD_NUM);
	//int tasknum = eks.size() / THREAD_NUM;
	pthread_t* handles = (pthread_t*)malloc(THREAD_NUM * sizeof(pthread_t));  // 为线程句柄分配内存空间
	PT_EliminationParam* param = (PT_EliminationParam*)malloc(THREAD_NUM * sizeof(PT_EliminationParam));  // 存储线程参数
	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], nullptr, PT_Elimination_Static, (void*)&param[t_id]);
	}

	for (int t_id = 0; t_id < THREAD_NUM; t_id++) pthread_join(handles[t_id], nullptr);

	pthread_barrier_destroy(&_Barrier_Elimination);
	pthread_barrier_destroy(&_Barrier_0_Operation);
}

void PT_GrobnerGE_Static_Block()
{
	pthread_barrier_init(&_Barrier_Elimination, nullptr, THREAD_NUM);
	pthread_barrier_init(&_Barrier_0_Operation, nullptr, THREAD_NUM);
	//int tasknum = eks.size() / THREAD_NUM;
	pthread_t* handles = (pthread_t*)malloc(THREAD_NUM * sizeof(pthread_t));  // 为线程句柄分配内存空间
	PT_EliminationParam* param = (PT_EliminationParam*)malloc(THREAD_NUM * sizeof(PT_EliminationParam));  // 存储线程参数
	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], nullptr, PT_Elimination_Static_Block, (void*)&param[t_id]);
	}

	for (int t_id = 0; t_id < THREAD_NUM; t_id++) pthread_join(handles[t_id], nullptr);

	pthread_barrier_destroy(&_Barrier_Elimination);
	pthread_barrier_destroy(&_Barrier_0_Operation);
}


//------------------------------------计算辅助函数------------------------------------

int find_first_bitset(const bitset<column_num_c>& b) {
	for (int i = 0; i < column_num_c; i++) if (b.test(i)) return i;
	return -1;
}

dynamic_bitset<> get_head(size_t index) {  // 获取首项
	for (int i = 0; i < eks.size(); i++)  // R[lp(E[i])
	{
		//cout << index << " " << i << endl;
		if (eks[i].find_first() == index) return eks[i];
	}
	return dynamic_bitset<>(0);
}

dynamic_bitset<> get_head_lp(size_t index) {  // 获取首项
	for (int i = 0; i < eks.size(); i++)  // R[lp(E[i])
	{
		//cout << index << " " << i << endl;
		if (index == lp_eks[i]) return eks[i];
	}
	return dynamic_bitset<>(0);
}

int get_head_lp_c(size_t index) {  // 获取首项
	for (int i = 0; i < column_num_c; i++)
	{
		cout << index << " " << i << endl;
		if (index == lp_eks_c[i]) return i;
		else if (lp_eks_c[i] == -1) break;
	}
	return -1;
}

void xor_static_AVX(bitset<column_num_c>& b1, const bitset<column_num_c>& b2) {  // b1 ^= b2，保证b1和b2等长
	for (size_t _Wpos = 0; _Wpos < column_num_c; ++_Wpos) {
		b1.set(_Wpos, b1._Subscript(_Wpos) ^ b2._Subscript(_Wpos));
	}
	//b1 ^= b2;
}

//--------------------------------计算辅助函数-PThread--------------------------------
void* PT_Rotation_gethead_Inthread(void* param) {
	PT_GetHeadParam* pa = (PT_GetHeadParam*)param;
	int t_id = pa->t_id;
	size_t index = pa->index;
	dynamic_bitset<>* ret = nullptr;
	for (int i = t_id; i < eks.size(); i += THREAD_NUM)  // R[lp(E[i])
	{
		if (index == eks[i].find_first())
		{
			ret = new dynamic_bitset<>(eks[i]);
			break;
		}
	}
	sem_post(&sem_gethead);// 通知 leader, 已完成消去任务
	pthread_exit(ret);
	return ret;
}

dynamic_bitset<> PT_Static_Rotation_get_head(size_t index) {  // 获取首项
	int tasknum = eks.size() / THREAD_NUM;
	pthread_t* handles = (pthread_t*)malloc(THREAD_NUM * sizeof(pthread_t));  // 为线程句柄分配内存空间
	PT_GetHeadParam* param = (PT_GetHeadParam*)malloc(THREAD_NUM * sizeof(PT_GetHeadParam));  // 存储线程参数
	void** ret = new void* [THREAD_NUM];
	sem_init(&sem_gethead, 0, 0);
	if (tasknum < THREAD_NUM)  // 如果一个线程的任务数量还不如线程数量多，那么就直接采用主线程
	{
		for (int i = 0; i < eks.size(); i++)  // R[lp(E[i])
		{
			if (index == eks[i].find_first())
			{
				return eks[i];
			}
		}
		return dynamic_bitset<>(0);
	}
	else {
		for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
			param[t_id].t_id = t_id;
			param[t_id].index = index;
		}
		//创建线程
		for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
			pthread_create(&handles[t_id], nullptr, PT_Rotation_gethead_Inthread, (void*)&param[t_id]);
		}
		for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
			pthread_join(handles[t_id], &ret[t_id]);
		}
		//主线程睡眠（等待所有的工作线程完成此轮任务）
		for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
			sem_wait(&sem_gethead);
		}
		//被唤醒，处理返回值
		for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
			if (ret[t_id]) return *(dynamic_bitset<>*)(ret[t_id]);
		}
		return dynamic_bitset<>(0);
	}

}

//--------------------------------------算法主体--------------------------------------

void GrobnerGE()
{
	dynamic_bitset<> temp = dynamic_bitset<>(n);
	for (int i = 0; i < et_num; i++)
	{
		while (ets[i].any())
		{
			temp = get_head(ets[i].find_first());
			if (!(temp.size() == 0)) ets[i] ^= temp;
			else
			{
				eks.push_back(ets[i]);
				break;
			}
		}
	}
}

void GrobnerGE_LP()
{
	dynamic_bitset<> temp = dynamic_bitset<>(n);
	for (int i = 0; i < et_num; i++)
	{
		while (ets[i].any())
		{
			temp = get_head_lp(lp_ets[i]);  // 对应消元子
			if (temp.size())
			{
				ets[i] ^= temp;
				lp_ets[i] = ets[i].find_first();
			}
			else
			{
				eks.push_back(ets[i]);
				lp_eks.push_back(lp_ets[i]);
				lp_ets[i] = -1;
				break;
			}
		}
	}
}

void GrobnerGE_LP_static()  // 静态位集下的高斯消去
{
	int lp = -1;
	for (int i = 0; i < et_num_c; i++)
	{
		while (ets_c[i].any())
		{
			lp = lp_ets_c[i];
			if (eks_c[lp].test(lp))  // 说明存在对应消元子
			{
				ets_c[i] ^= eks_c[lp];
				lp_ets_c[i] = find_first_bitset(ets_c[i]);
			}
			else
			{
				eks_c[lp] = ets_c[i];
				lp_ets_c[i] = -1;
				break;
			}
		}
	}
}

void GrobnerGE_LP_static_headopt()  // 静态位集下的高斯消去，选取首项进一步优化
{
	int lp = -1;
	for (int i = 0; i < et_num_c; i++)
	{
		while (ets_c[i].any())
		{
			lp = lp_ets_c[i];
			if (eks_c[lp].test(lp))  // 说明存在对应消元子
			{
				ets_c[i] ^= eks_c[lp];
				for (int j = lp; j < column_num_c; j++)
					if (ets_c[i].test(j)) {
						lp_ets_c[i] = j;
						break;
					}
			}
			else
			{
				eks_c[lp] = ets_c[i];
				lp_ets_c[i] = -1;
				break;
			}
		}
	}
}

void GrobnerGE_LP_static_headopt_AVX()  // 静态位集下的高斯消去，选取首项进一步优化，AVX版本
{
	int lp = -1;
	for (int i = 0; i < et_num_c; i++)
	{
		while (ets_c[i].any())
		{
			lp = lp_ets_c[i];
			if (eks_c[lp].test(lp))  // 说明存在对应消元子
			{
				xor_static_AVX(ets_c[i], eks_c[lp]);
				//ets_c[i] ^= eks_c[lp];
				for (int j = lp; j < column_num_c; j++)
					if (ets_c[i].test(j)) {
						lp_ets_c[i] = j;
						break;
					}
			}
			else
			{
				eks_c[lp] = ets_c[i];
				lp_ets_c[i] = -1;
				break;
			}
		}
	}
}

void GrobnerGE_Rearrange() {
	for (int i = 0; i < column_num_c; i++)  // 取每个消元子，对被消元行进行操作，便于并行化
	{
		if (!eks_c[i].test(i))  // 消元子被逆序初始化时满足“行号” = “首项”的条件
		{  // 零号线程负责升格消元子操作
			for (size_t j = 0; j < et_num_c; j++)
			{
				if (i == lp_ets_c[j])  // 说明存在对应被消元行
				{
					eks_c[i] = ets_c[j];
					lp_ets_c[j] = -1;
					break;
				}
			}
		}
		for (size_t j = 0; j < et_num_c; j++)  // 循环划分并行化
		{
			if (i == lp_ets_c[j])  // 说明存在对应被消元行
			{
				ets_c[j] ^= eks_c[i];
				for (int k = i; k < column_num_c; k++)
				{
					if (ets_c[j].test(k))
					{
						lp_ets_c[j] = k;
						break;
					}
					if (k == column_num_c - 1) lp_ets_c[j] = -1;
				}
			}
		}
	}
}

void getheadOpt_GrobnerGE()
{
	dynamic_bitset<> temp = dynamic_bitset<>(n);
	for (int i = 0; i < et_num; i++)
	{
		while (ets[i].any())
		{
			//size_t first = ets[i].find_first();  // lp(E[i])
			temp = PT_Static_Rotation_get_head(ets[i].find_first());  // R[lp(E[i])]，即消元子中与被消元行i中首项相同的项
			if (!(temp.size() == 0))
			{
				ets[i] ^= temp;
			}
			else
			{
				eks.push_back(ets[i]);
				break;
			}
		}
	}
}

void test_GE() {
	bool find_flag = false;  // true说明存在首项
	dynamic_bitset<>* temp = nullptr;
	for (int i = 0; i < et_num; i++)
	{
		while (ets[i].any())
		{
			find_flag = false;
			size_t index = ets[i].find_first();
			for (int j = 0; j < eks.size(); j++)  // R[lp(E[i])
			{
				if (index == eks[j].find_first())
				{
					ets[i] ^= eks[j];
					find_flag = true;
					break;
				}
			}
			if (!find_flag)
			{
				eks.push_back(ets[i]);
				break;
			}
		}
	}
}

void my_xor_AVX(dynamic_bitset<>& db1, dynamic_bitset<>& db2) {

	for (int i = 0; i < db1.size(); i += 8)
	{

	}

}

void GrobnerGE_PT_SET() {
	int lp = -1;
	while (!ek_set.empty())  // 如果还有没被处理过的消元子则继续
	{
		for (size_t i = 0; i < column_num_c; i++)  // 取每个消元子，对被消元行进行处理
		{
			if (eks_c[i].test(i))  // 消元子被逆序初始化时满足“行号” = “首项”的条件
			{
				if (ek_set.find(i) != ek_set.end())
				{
					for (size_t j = 0; j < et_num_c; j++)
					{
						if (i == lp_ets_c[j])  // 说明存在对应被消元行
						{
							ets_c[j] ^= eks_c[i];
							for (int k = i; k < column_num_c; k++)
								if (ets_c[j].test(k)) {
									lp_ets_c[j] = k;
									break;
								}
						}
					}
					ek_set.erase(i);  //消元子被用完丢弃
				}
			}
			else
			{
				for (size_t j = 0; j < et_num_c; j++)
				{
					if (lp_ets_c[j] == i && ek_set.find(i) == ek_set.end())  // 防止消元子重复
					{
						eks_c[i] = ets_c[j];
						lp_ets_c[j] = -1;
						ek_set.insert(i);  // 升格
						break;
					}
				}
			}
		}
	}
}


void GrobnerGE_OMP()
{
	int count1 = 0, count2 = 0, count3 = 0, count4 = 0;
	dynamic_bitset<> temp = dynamic_bitset<>(n);
	for (int i = 0; i < et_num; i++)
	{
		while (ets[i].any())
		{
			count1++;
			//size_t first = ets[i].find_first();  // lp(E[i])
			temp = get_head(ets[i].find_first());
			//temp = get_head(lp_ets[i]);  // R[lp(E[i])]，即消元子中与被消元行i中首项相同的项
			if (!(temp.size() == 0))
			{
				ets[i] ^= temp;
				//lp_ets[i] = ets[i].find_first();
				count2++;
			}
			else
			{
				count3++;
				eks.push_back(ets[i]);
				break;
			}
		}
	}
	cout << count1 << " " << count2 << " " << count3 << endl;
}

//------------------------------------输出调试函数------------------------------------

void output()
{
	ofstream outp(dir + "output.txt");
	for (int i = 0; i < et_num; i++)
	{
		for (int j = 0; j < n; j++) if (ets[i].test(j)) outp << j << " ";
		outp << endl;
	}
	outp.close();
}

void reverse_output()
{
	ofstream outp(dir + "output.txt");
	for (int i = 0; i < et_num; i++)
	{
		for (int j = 0; j < n; j++) if (ets[i].test(j)) outp << n - j - 1 << " ";
		outp << endl;
	}
	outp.close();
}

void reverse_output_c()
{
	ofstream outp(dir + "output_c.txt");
	for (int i = 0; i < et_num_c; i++)
	{
		for (int j = 0; j < n; j++) if (ets_c[i].test(j)) outp << n - j - 1 << " ";
		outp << endl;
	}
	outp.close();
}

//------------------------------------数据读取函数------------------------------------

void readData_bitset()
{
	string inek, inet;
	stringstream ss_inek, ss_inet;
	ifstream inElimKey(dir + "elimkey.txt");  // 消元子
	ifstream inElimTar(dir + "elimtar.txt");  // 被消元行
	int inek_loc, p_ek = 0, inet_loc, p_et = 0;  // 用于数据读入
	while (true)  // 读取消元子
	{
		getline(inElimKey, inek);
		ss_inek = stringstream(inek);
		while (ss_inek >> inek)
		{
			inek_loc = stoi(inek);
			//cout << inek_loc << " ";
			eks[p_ek].set(inek_loc);
		}
		if (inek.empty())
		{
			//cout << "ek_complete" << endl;
			break;
		}
		//cout << eks[p_ek] << endl;
		p_ek++;
	}
	while (true)  // 读取被消元行
	{
		getline(inElimTar, inet);
		ss_inet = stringstream(inet);
		while (ss_inet >> inet)
		{
			inet_loc = stoi(inet);
			//cout << inet_loc << " ";
			ets[p_et].set(inet_loc);
		}
		if (inet.empty())
		{
			//cout << "et_complete" << endl;
			break;
		}
		//cout << ets[p_et] << endl;
		p_et++;
	}
}

void readData_reverse_bitset()
{  // 倒序读入数据
	string inek, inet;
	stringstream ss_inek, ss_inet;
	ifstream inElimKey(dir + "elimkey.txt");  // 消元子
	ifstream inElimTar(dir + "elimtar.txt");  // 被消元行
	int inek_loc, p_ek = 0, inet_loc, p_et = 0;  // 用于数据读入
	while (true)  // 读取消元子
	{
		getline(inElimKey, inek);
		ss_inek = stringstream(inek);
		while (ss_inek >> inek)
		{
			inek_loc = stoi(inek);
			//cout << inek_loc << " ";
			eks[p_ek].set(n - inek_loc - 1);
		}
		if (inek.empty())
		{
			//cout << "ek_complete" << endl;
			break;
		}
		//cout << eks[p_ek] << endl;
		p_ek++;
	}
	while (true)  // 读取被消元行
	{
		getline(inElimTar, inet);
		ss_inet = stringstream(inet);
		while (ss_inet >> inet)
		{
			inet_loc = stoi(inet);
			//cout << inet_loc << " ";
			ets[p_et].set(n - inet_loc - 1);
		}
		if (inet.empty())
		{
			//cout << "et_complete" << endl;
			break;
		}
		//cout << ets[p_et] << endl;
		p_et++;
	}
}

void readData_reverse_bitset_c()
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
			if (lp == -1) lp = n - inek_loc - 1;
			//cout << inek_loc << " ";
			eks_c[lp].set(n - inek_loc - 1);
		}
		lp = -1, p_ek++;
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
				lp = n - inet_loc - 1;
				lp_ets_c[p_et] = lp;
			}
			//cout << inet_loc << " ";
			ets_c[p_et].set(n - inet_loc - 1);
		}
		if (inet.empty()) break;
		lp = -1;
		p_et++;
		//cout << ets_c[p_et] << endl;
	}
	//cout << "et_complete" << endl;
	inElimKey.close();
	inElimTar.close();
	cout << "init_complete" << endl;
}

void readData_reverse_bitset_c_pt()
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
			if (lp == -1) { 
				lp = n - inek_loc - 1; 
				ek_set.insert(lp);
			}
			//cout << inek_loc << " ";
			eks_c[lp].set(n - inek_loc - 1);

		}
		lp = -1, p_ek++;
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
				lp = n - inet_loc - 1;
				lp_ets_c[p_et] = lp;
			}
			//cout << inet_loc << " ";
			ets_c[p_et].set(n - inet_loc - 1);
		}
		if (inet.empty()) break;
		lp = -1;
		p_et++;
		//cout << ets_c[p_et] << endl;
	}
	//cout << "et_complete" << endl;
	inElimKey.close();
	inElimTar.close();
	cout << "init_complete" << endl;

}
void init() {
	eks.clear(); eks.resize(0);
	ets = new dynamic_bitset<>[et_num];
	lp_eks.clear(); lp_eks.resize(0);
	lp_ets = new int[et_num];

	for (int i = 0; i < ek_num; i++) eks.push_back(dynamic_bitset<>(n));

	for (int i = 0; i < et_num; i++) ets[i] = dynamic_bitset<>(n);

	readData_reverse_bitset();  // 逆序初始化消元子和被消元行阵列

	lp_ets = new int[et_num];
	for (int i = 0; i < et_num; i++) lp_ets[i] = ets[i].find_first();//计算被消元行首项
	//cout << lp_ets[i] << " ";
	for (int i = 0; i < ek_num; i++) lp_eks.push_back(eks[i].find_first());
}


void init_c() {
	for (int i = 0; i < column_num_c; i++)  // 初始化，处理上一次算法中的残余数据
	{
		eks_c[i] = *(new bitset<column_num_c>);
		lp_eks_c[i] = -1;
	}
	readData_reverse_bitset_c();  // 逆序初始化消元子和被消元行阵列
	cout << "init_complete" << endl;
}

void init_c_pt() {
	for (int i = 0; i < column_num_c; i++)  // 初始化，处理上一次算法中的残余数据
	{
		eks_c[i] = *(new bitset<column_num_c>);
		lp_eks_c[i] = -1;
	}
	readData_reverse_bitset_c_pt();  // 逆序初始化消元子和被消元行阵列
	cout << "init_complete" << endl;
}

int main() {
	ifstream inParam(dir + "param.txt");
	inParam >> n;  // 导入矩阵大小
	inParam >> ek_num;  // 导入非零消元子个数
	inParam >> et_num;  // 导入被消元行行数
	//静态算法则需要手动更改程序中的全局常量

	cout << "矩阵大小为" << n << "，消元子个数为" << ek_num << "，被消元行行数为" << et_num << endl;

	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	//-----------------------------------------------------------------

	//init();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//GrobnerGE();  // 动态位集存储的矩阵的特殊高斯消去
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "db_GrobnerGE: " << (tail - head) * 1000.0 / freq
		<< "ms" << endl;

	//-----------------------------------------------------------------


	//init();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//GrobnerGE_LP();  // 动态位集存储的矩阵的特殊高斯消去，选取首项部分优化
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "GrobnerGE_LP: " << (tail - head) * 1000.0 / freq
		<< "ms" << endl;
	//reverse_output();

	//-----------------------------------------------------------------


	//init_c();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//GrobnerGE_LP_static();  // 静态位集存储的矩阵的特殊高斯消去，选取首项部分优化
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "GrobnerGE_LP_static: " << (tail - head) * 1000.0 / freq
		<< "ms" << endl;
	//reverse_output_c();


	//-----------------------------------------------------------------

	//init_c();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//GrobnerGE_LP_static_headopt();  // 静态位集存储的矩阵的特殊高斯消去，选取首项部分高度优化
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "GrobnerGE_LP_static_headopt: " << (tail - head) * 1000.0 / freq
		<< "ms" << endl;
	//reverse_output_c();

	//-----------------------------------------------------------------

	//init_c();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//GrobnerGE_LP_static_headopt_AVX();  // 静态位集存储的矩阵的特殊高斯消去，选取首项部分高度优化
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "GrobnerGE_LP_static_headopt_AVX: " << (tail - head) * 1000.0 / freq
		<< "ms" << endl;
	//reverse_output_c();

	//-----------------------------------------------------------------


	//init_c_pt();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//GrobnerGE_PT_SET();  // 静态位集存储的矩阵的特殊高斯消去，选取首项部分高度优化
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "GrobnerGE_PT_SET: " << (tail - head) * 1000.0 / freq
		<< "ms" << endl;
	//reverse_output_c();

	//-----------------------------------------------------------------

	//init_c();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//GrobnerGE_Rearrange();  // GGELSH多线程块划分版本
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "GrobnerGE_Rearrange: " << (tail - head) * 1000.0 / freq
		<< "ms" << endl;
	reverse_output_c();


	//-----------------------------------------------------------------

	//init_c();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	//PT_GrobnerGE_Static();  // GGELSH多线程循环划分版本
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "PT_GrobnerGE_Static: " << (tail - head) * 1000.0 / freq
		<< "ms" << endl;
	//reverse_output_c();

	//-----------------------------------------------------------------

	init_c();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	PT_GrobnerGE_Static_Block();  // GGELSH多线程块划分版本
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "PT_GrobnerGE_Static_Block: " << (tail - head) * 1000.0 / freq
		<< "ms" << endl;
	//reverse_output_c();


}


