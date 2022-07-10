
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <ctime>
#include <bitset>
#include <vector>
#include <string>
#include <sstream>

#include <cmath>
#include <stdio.h>
#include <time.h>
#include <algorithm>

#include<mpi.h>
#include<omp.h>

#include <unordered_set>

#include <bitset>
#include <immintrin.h>
#include <windows.h>

using namespace std;

#define THREAD_NUM 8   // 线程数量
#define PROGRESS_NUM 6 // 进程数量

int n = 0; // 矩阵大小
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
*/


//------------------------------------数据导入工具------------------------------------
const int column_num_c = 130;
const int ek_num_c = 22; // 非零消元子个数
const int et_num_c = 8; // 导入被消元行行数

string dir = "C:/Users/CCC/source/repos/GrobnerGE/GrobnerGE/data/t1/";
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


bitset<column_num_c> eks_c[column_num_c]; // 消元子，开大一些便于检索与升格
bitset<column_num_c> ets_c[et_num_c];

int lp_ets_c[et_num_c];
int lp_eks_c[column_num_c];


MyBitSet eks[column_num_c];
MyBitSet ets[et_num_c];


long long head, tail, freq;

//------------------------------------计算辅助函数------------------------------------

int find_first_bitset(const bitset<column_num_c>& b)
{
    for (int i = 0; i < column_num_c; i++)
        if (b.test(i))
            return i;
    return -1;
}

void GrobnerGE_OMP()
{
    int i, j, k;
#pragma omp parallel num_threads(THREAD_NUM), private(i, j, k)
    for (int i = 0; i < column_num_c; i++) // 取每个消元子，对被消元行进行操作，便于并行化
    {
        if (!eks_c[i].test(i)) // 消元子被逆序初始化时满足“行号” = “首项”的条件
        {
#pragma omp barrier
#pragma omp single
            for (size_t j = 0; j < et_num_c; j++)
            {
                if (i == lp_ets_c[j]) // 说明存在对应被消元行
                {
                    eks_c[i] = ets_c[j];
                    lp_ets_c[j] = -1;
                    break;
                }
            }
            //#pragma omp barrier
        }
#pragma omp for schedule(dynamic)
        for (int j = 0; j < et_num_c; j++) // 循环划分并行化
        {
            if (i == lp_ets_c[j]) // 说明存在对应被消元行
            {
                ets_c[j] ^= eks_c[i];
                for (int k = i; k < column_num_c; k++)
                {
                    if (ets_c[j].test(k))
                    {
                        lp_ets_c[j] = k;
                        break;
                    }
                    if (k == column_num_c - 1)
                        lp_ets_c[j] = -1;
                }
            }
        }
    }
}

//------------------------------------输出调试函数------------------------------------

void reverse_output_c()
{
    ofstream outp(dir + "output_c_mpi.txt");
    for (int i = 0; i < et_num_c; i++)
    {
        for (int j = 0; j < column_num_c; j++)
            if (ets_c[i].test(j))
                outp << column_num_c - j - 1 << " ";
        outp << endl;
    }
    cout << "已将结果存储到文件output_c_mpi.txt" << endl;
    outp.close();
}

//------------------------------------数据读取函数------------------------------------

void readData_reverse_bitset_c()
{ // 倒序读入数据，读入静态位集
    string inek, inet;
    stringstream ss_inek, ss_inet;
    ifstream inElimKey(dir + "elimkey.txt");    // 消元子
    ifstream inElimTar(dir + "elimtar.txt");    // 被消元行
    int inek_loc, p_ek = 0, inet_loc, p_et = 0; // 用于数据读入
    int lp = -1;
    while (true) // 读取消元子
    {
        getline(inElimKey, inek);
        ss_inek = stringstream(inek);
        while (ss_inek >> inek)
        {
            inek_loc = stoi(inek);
            if (lp == -1)
                lp = column_num_c - inek_loc - 1;
            // cout << inek_loc << " ";
            eks_c[lp].set(column_num_c - inek_loc - 1);
        }
        lp = -1, p_ek++;
        if (inek.empty())
            break;
        // cout << eks_c[p_ek] << endl;
    }
    // cout << "ek_complete" << endl;

    while (true) // 读取被消元行
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
            // cout << inet_loc << " ";
            ets_c[p_et].set(column_num_c - inet_loc - 1);
        }
        if (inet.empty())
            break;
        lp = -1;
        p_et++;
        // cout << ets_c[p_et] << endl;
    }
    // cout << "et_complete" << endl;
    inElimKey.close();
    inElimTar.close();
}

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

void reverse_output_MyB()
{
    ofstream outp(dir + "output_MyB.txt");
    for (int i = 0; i < et_num_c; i++)
    {
        for (int j = 0; j < column_num_c; j++) if (ets[i].test(j)) outp << column_num_c - j - 1 << " ";
        outp << endl;
    }
    outp.close();
}


void init_c()
{
    for (int i = 0; i < column_num_c; i++) // 初始化，处理上一次算法中的残余数据
    {
        eks_c[i] = *(new bitset<column_num_c>);
        lp_eks_c[i] = -1;
    }
    readData_reverse_bitset_c(); // 逆序初始化消元子和被消元行阵列
    cout << "init_complete" << endl;
}

void init_MyB() {
    for (int i = 0; i < column_num_c; i++)
    {

    }
    readData_reverse_MyB();  // 逆序初始化消元子和被消元行阵列
    cout << "init_complete" << endl;
}
int main(int argc, char* argv[])
{
    //静态算法则需要手动更改程序中的全局常量

    //cout << "矩阵大小为" << column_num_c << "，消元子个数为" << ek_num_c << "，被消元行行数为" << et_num_c << endl;

    //-----------------------------------------------------------------

    //init_c();

    //GrobnerGE_OMP(); // 动态位集存储的矩阵的特殊高斯消去

    //-----------------------------------------------------------------
    init_MyB();
    int provided;
    //MPI_Init(0, 0);
    MPI_Init_thread(0,0, MPI_THREAD_MULTIPLE, &provided);
    double head, tail;
    head = MPI_Wtime();
    MPI_Request request;
    int rank = 0;
    //MPI_Finalize();
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int upshift = 0;
    int i, j, k;
#pragma omp parallel num_threads(THREAD_NUM), private(i, j, k)
    for (int i = 0; i < column_num_c; i++) // 取每个消元子，对被消元行进行操作，便于并行化
    {
        if (!eks[i].test(i)) // 消元子被逆序初始化时满足“行号” = “首项”的条件
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0) // 零号进程负责升格
            {
//#pragma omp single
                for (size_t j = 0; j < et_num_c; j++)
                {
                    if (i == ets[j].head)  // 说明存在对应被消元行
                    {
                        eks[i] = ets[j];
                        ets[j].head = -1;
                        upshift = 1;
                        for (int j = 1; j < PROGRESS_NUM; j++)
                        {
                            MPI_Send(&upshift, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
                            MPI_Send(&(eks[i].content[0]), bit_size, MPI_INT, j, 1, MPI_COMM_WORLD);
                            MPI_Send(&(ets[i].head), 1, MPI_INT, j, 2, MPI_COMM_WORLD);
                        }
                        break;
                    }
                }
                if (!upshift)
                {
                    for (int j = 1; j < PROGRESS_NUM; j++)
                    {
                        MPI_Send(&upshift, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
                    }
                }
            }
            else 
            { 
                MPI_Recv(&upshift, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            }

            if (rank && upshift)  // 此刻其他进程在接收数据
            {
                MPI_Recv(&(eks[i].content[0]), bit_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&(ets[i].head), 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            upshift = 0;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank) {  // 0号进程以外的进程做消去
            cout <<"rank = "<<rank<< " come2" << endl;
//#pragma omp for
            for (int j = rank - 1; j < et_num_c; j += PROGRESS_NUM - 1) // 循环划分并行化处理被消元行
            {
                if (i == ets[j].head)  // 说明存在对应被消元行
                {
                    ets[j] ^= eks[i];
                    //ets[j].my_xor_AVX(eks[i]);
                    for (int k = i; k < column_num_c; k++)
                    {
                        if (ets[j].test(k))
                        {
                            ets[j].head = k;
                            break;
                        }
                        if (k == column_num_c - 1) ets[j].head = -1;
                    }
                    //将计算后的结果广播回0号进程
                    //cout << "rank = "<< rank <<" sending" << endl;
                    MPI_Send(&(ets[j].content[0]), bit_size, MPI_INT, 0, j, MPI_COMM_WORLD);
                    MPI_Send(&(ets[j].head), 1, MPI_INT, 0, j, MPI_COMM_WORLD);
                }
            }
        }
        else {  // 结果存在0号进程
//#pragma omp for
            for (int j = 0; j < et_num_c; j++) // 处理数据
            {   
                if (i == ets[j].head)  // 说明存在对应被消元行并且子进程执行了操作
                {
                    //获取计算后的结果
                    cout << "receiving:  " << j << endl;
                    MPI_Recv(&(ets[j].content[0]), bit_size, MPI_INT, MPI_ANY_SOURCE, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&(ets[j].head), 1, MPI_INT, MPI_ANY_SOURCE, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
    }
    



    //MPI_Barrier(MPI_COMM_WORLD);


    tail = MPI_Wtime();

    //ofstream outp(dir + "output_MyB_" + to_string(rank) +".txt");
    //if (rank)
    //{
    //    for (int i = rank - 1; i < et_num_c; i += PROGRESS_NUM - 1)
    //    {
    //        for (int j = 0; j < column_num_c; j++) if (ets[i].test(j)) outp << column_num_c - j - 1 << " ";
    //        outp << endl;
    //    }
    //}
    //outp.close();
    

    MPI_Finalize();

    if (!rank)
    {
        cout << "GrobnerGE_MPI: " << (tail - head) * 1000 << " ms" << std::endl;
        //cout << "123" << endl;
        //for (int j = 0; j < column_num_c; j++)
        //{
        //    for (int i = 0; i < bit_size; i++)
        //    {
        //        cout << ets[j].content[i] << " ";
        //    }
        //    cout << endl;
        //}
        reverse_output_MyB();
        // show(n);
    }
    return 0;
}

