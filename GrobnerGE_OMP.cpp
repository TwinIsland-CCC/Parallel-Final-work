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
#include<omp.h>


using namespace std;

#define THREAD_NUM 8  // 线程数量


int n = 0;  // 矩阵大小
const int k = 1;

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


文件内部包含：std::bitset版本串行重排代码的omp算法
*/

const int column_num_c = 43577;
const int ek_num_c = 39477;  // 非零消元子个数
const int et_num_c = 54274;  // 导入被消元行行数


bitset<column_num_c> eks_c[column_num_c];  // 消元子，开大一些便于检索与升格
bitset<column_num_c> ets_c[et_num_c];

int lp_ets_c[et_num_c];
int lp_eks_c[column_num_c];

long long head, tail, freq;

//------------------------------------数据导入工具------------------------------------

string dir = "C:/Users/CCC/source/repos/GrobnerGE/GrobnerGE/data/t10/";
stringstream ss;

//------------------------------------计算辅助函数------------------------------------
void GrobnerGE_OMP()
{
    int i, j, k;
    #pragma omp parallel num_threads(THREAD_NUM), private(i, j, k)
	for (int i = 0; i < column_num_c; i++)  // 取每个消元子，对被消元行进行操作，便于并行化
	{
		if (!eks_c[i].test(i))  // 消元子被逆序初始化时满足“行号” = “首项”的条件
		{
            #pragma omp barrier
            #pragma omp single
            for (size_t j = 0; j < et_num_c; j++) 
            {
                if (i == lp_ets_c[j])  // 说明存在对应被消元行
                {
                    eks_c[i] = ets_c[j];
                    lp_ets_c[j] = -1;
                    break;
                }
            }
            //#pragma omp barrier
		}
        #pragma omp for schedule(guided)
		for (int j = 0; j < et_num_c; j ++)  // 循环划分并行化
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

int find_first_bitset(const bitset<column_num_c>& b) {
	for (int i = 0; i < column_num_c; i++) if (b.test(i)) return i;
	return -1;
}


//------------------------------------输出调试函数------------------------------------

void reverse_output_c()
{
	ofstream outp(dir + "output_c_omp.txt");
	for (int i = 0; i < et_num_c; i++)
	{
		for (int j = 0; j < column_num_c; j++) if (ets_c[i].test(j)) outp << column_num_c - j - 1 << " ";
		outp << endl;
	}
    cout<<"已将结果存储到文件output_c_omp.txt"<<endl;
	outp.close();
}

//------------------------------------数据读取函数------------------------------------

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
			if (lp == -1) lp = column_num_c - inek_loc - 1;
			//cout << inek_loc << " ";
			eks_c[lp].set(column_num_c - inek_loc - 1);
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
				lp = column_num_c - inet_loc - 1;
				lp_ets_c[p_et] = lp;
			}
			//cout << inet_loc << " ";
			ets_c[p_et].set(column_num_c - inet_loc - 1);
		}
		if (inet.empty()) break;
		lp = -1;
		p_et++;
		//cout << ets_c[p_et] << endl;
	}
	//cout << "et_complete" << endl;
	inElimKey.close();
	inElimTar.close();
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


int main() {
	//静态算法则需要手动更改程序中的全局常量

	cout << "矩阵大小为" << column_num_c << "，消元子个数为" << ek_num_c << "，被消元行行数为" << et_num_c << endl;

	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	//-----------------------------------------------------------------

	init_c();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	GrobnerGE_OMP();  // 动态位集存储的矩阵的特殊高斯消去
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "GrobnerGE_OMP: " << (tail - head) * 1000.0 / freq
		<< " ms" << endl;
    reverse_output_c();

	//-----------------------------------------------------------------
}


