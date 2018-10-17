#pragma once
#include "omp.h"
#include "LoadCSV.h"
#include "LayerQL.h"
#include "PooLayerQL.h"
#include "Conv_LayerQL.h"
#include "Sigmoid_LayerQL.h"
#include "Dim_ReduceQL.h"
#include "Fullconnect_LayerQL.h"
#include "Inter_LayerQL.h"
#include <math.h>
#include "Relu_LayerQL.h"
#include "SoftMax_LayerQL.h"
#include "Data_AugmentationQL.h"

namespace tinyDNN
{
	//	99.03%
	void mnist_Conv_T_1()
	{
		LoadCSV::loadCSVTrain();
		LoadCSV::loadCSV_Train_Vector();
		LoadCSV::loadCSVTest();
		LoadCSV::loadCSV_Test_Vector();

		std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(28, 28);
		//						类型			卷积核数			行数			列数				卷积核宽度		卷积核几片		扩充宽度	
		std::shared_ptr<LayerQL<double>> conv_01 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 16, 28, 28, 5, 1, 2);	//卷积层
		std::shared_ptr<Inter_LayerQL<double>> o_01 = in_01 + conv_01;


		std::shared_ptr<LayerQL<double>> rule_01 = std::make_shared<Relu_LayerQL<double>>(Relu_Conv_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_02 = o_01 + rule_01;
		//rule_01->pRelu_k = 0.1;


		std::shared_ptr<LayerQL<double>> pool_01 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 14, 14);	//池化层
		std::shared_ptr<Inter_LayerQL<double>> o_03 = o_02 + pool_01;


		//						类型			卷积核数			行数			列数				卷积核宽度		卷积核几片		扩充宽度	
		std::shared_ptr<LayerQL<double>> conv_02 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 16, 14, 14, 3, 16, 1);	//卷积层
		std::shared_ptr<Inter_LayerQL<double>> o_04 = o_03 + conv_02;


		std::shared_ptr<LayerQL<double>> rule_02 = std::make_shared<Relu_LayerQL<double>>(Relu_Conv_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_05 = o_04 + rule_02;
		//rule_02->pRelu_k = 0.1;


		std::shared_ptr<LayerQL<double>> pool_02 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 7, 7);	//池化层
		std::shared_ptr<Inter_LayerQL<double>> o_06 = o_05 + pool_02;


		std::shared_ptr<LayerQL<double>> dim_reduce_01 = std::make_shared<Dim_ReduceQL<double>>(Dim_Reduce_Layer, 16, 7, 7);	//降维层
		std::shared_ptr<Inter_LayerQL<double>> o_07 = o_06 + dim_reduce_01;


		std::shared_ptr<LayerQL<double>> fullconnect_01 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 16 * 7 * 7, 10);//全连接层
		std::shared_ptr<Inter_LayerQL<double>> o_08 = o_07 + fullconnect_01;


		std::shared_ptr<LayerQL<double>> rule_03 = std::make_shared<Relu_LayerQL<double>>(Relu_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_09 = o_08 + rule_03;
		//rule_03->pRelu_k = 0.1;

		std::shared_ptr<LayerQL<double>> lossLayer_01 = std::make_shared<SoftMax_LayerQL<double>>(SoftMax_Layer);//Loss层
		std::shared_ptr<Inter_LayerQL<double>> o_10 = o_09 + lossLayer_01;

		for (int i = 0; i < 14; i++)
		{
			//	 程序加载初始时间
			DWORD load_time = GetTickCount();

			rule_01->pRelu_k = 0.12;
			rule_02->pRelu_k = 0.12;
			rule_03->pRelu_k = 0.12;

			if (i < 2)
			{
				conv_01->upConv = 0.02;
				conv_02->upConv = 0.02;
				fullconnect_01->upFull = 0.015;
			}
			if (i < 4)
			{
				conv_01->upConv = 0.015;
				conv_02->upConv = 0.015;
				fullconnect_01->upFull = 0.009;
			}
			else if (i < 6)
			{
				conv_01->upConv = 0.008;
				conv_02->upConv = 0.008;
				fullconnect_01->upFull = 0.006;
			}
			//这里有突变,重点关注这个学习率
			else if (i < 8)
			{
				conv_01->upConv = 0.004;
				conv_02->upConv = 0.004;
				fullconnect_01->upFull = 0.003;
			}
			else if (i < 10)
			{
				conv_01->upConv = 0.002;
				conv_02->upConv = 0.002;
				fullconnect_01->upFull = 0.001;
			}
			else if (i < 12)
			{
				conv_01->upConv = 0.001;
				conv_02->upConv = 0.001;
				fullconnect_01->upFull = 0.0005;
			}
			else if (i < 14)
			{
				conv_01->upConv = 0.0005;
				conv_02->upConv = 0.0005;
				fullconnect_01->upFull = 0.0002;
			}
			else if (i < 18)
			{
				conv_01->upConv = 0.00005;
				conv_02->upConv = 0.00005;
				fullconnect_01->upFull = 0.00005;
			}

			//#pragma omp parallel
			for (int j = 0; j < 55000; j++)
			{
				if (j % 10000 == 0) std::cout << i << "::" << j << std::endl;
				//入参
				in_01->forward_Matrix_Vector.clear();
				in_01->forward_Matrix_Vector.push_back(LoadCSV::conv_Input_Vector[j]);
				o_10->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer->backward_Matrix->getMatrixQL().row(j);

				//从头开始进行前向传播
				//#pragma omp parallel
				for (auto k = NetQL<double>::layerQLVector.begin(); k != NetQL<double>::layerQLVector.end(); k++)
				{
					(*k)->calForward();
				}
				//从头开始反向传播 + 权重更新
				//#pragma omp parallel
				for (auto k = NetQL<double>::layerQLVector.rbegin(); k != NetQL<double>::layerQLVector.rend(); k++)
				{
					(*k)->calBackward();
					(*k)->upMatrix();
				}
			}

			double numTotal = 0;
			for (int k = 0; k < 10000; k++)
			{
				//入参
				in_01->forward_Matrix_Vector.clear();
				in_01->forward_Matrix_Vector.push_back(LoadCSV::conv_Input_Vector_T[k]);
				o_10->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer_T->backward_Matrix->getMatrixQL().row(k);

				//前向传播，计算结果
				for (auto k = NetQL<double>::layerQLVector.begin(); k != NetQL<double>::layerQLVector.end(); k++)
				{
					(*k)->calForward();
				}

				//计算得到的最大值位置
				int maxRow, maxColumn;
				lossLayer_01->left_Layer->forward_Matrix->getMatrixQL().maxCoeff(&maxRow, &maxColumn);
				//Lable的最大值位置
				int maxRow_T, maxColumn_T;
				lossLayer_01->right_Layer->backward_Matrix->getMatrixQL().maxCoeff(&maxRow_T, &maxColumn_T);
				//判断是否相等，若相等，则+1
				if (maxColumn == maxColumn_T)
				{
					numTotal++;
				}
			}
			//正确率
			std::cout << numTotal / 10000.00 << std::endl;

			//训练和测试运行时间
			DWORD star_time = GetTickCount();

			//计算运行时间
			std::cout << "这个程序加载时间为：" << (star_time - load_time) << "ms." << std::endl;
		}
	}

	//	99.03%
	void mnist_Conv_T_2()
	{
		LoadCSV::loadCSVTrain();
		LoadCSV::loadCSV_Train_Vector();
		LoadCSV::loadCSVTest();
		LoadCSV::loadCSV_Test_Vector();

		std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(28, 28);
		//						类型			卷积核数			行数			列数				卷积核宽度		卷积核几片		扩充宽度	
		std::shared_ptr<LayerQL<double>> conv_01 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 16, 28, 28, 5, 1, 2);	//卷积层
		std::shared_ptr<Inter_LayerQL<double>> o_01 = in_01 + conv_01;


		std::shared_ptr<LayerQL<double>> rule_01 = std::make_shared<Relu_LayerQL<double>>(Relu_Conv_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_02 = o_01 + rule_01;
		//rule_01->pRelu_k = 0.1;


		std::shared_ptr<LayerQL<double>> pool_01 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 14, 14);	//池化层
		std::shared_ptr<Inter_LayerQL<double>> o_03 = o_02 + pool_01;


		//						类型			卷积核数			行数			列数				卷积核宽度		卷积核几片		扩充宽度	
		std::shared_ptr<LayerQL<double>> conv_02 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 16, 14, 14, 3, 16, 1);	//卷积层
		std::shared_ptr<Inter_LayerQL<double>> o_04 = o_03 + conv_02;


		std::shared_ptr<LayerQL<double>> rule_02 = std::make_shared<Relu_LayerQL<double>>(Relu_Conv_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_05 = o_04 + rule_02;
		//rule_02->pRelu_k = 0.1;


		std::shared_ptr<LayerQL<double>> pool_02 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 7, 7);	//池化层
		std::shared_ptr<Inter_LayerQL<double>> o_06 = o_05 + pool_02;


		std::shared_ptr<LayerQL<double>> dim_reduce_01 = std::make_shared<Dim_ReduceQL<double>>(Dim_Reduce_Layer, 16, 7, 7);	//降维层
		std::shared_ptr<Inter_LayerQL<double>> o_07 = o_06 + dim_reduce_01;


		std::shared_ptr<LayerQL<double>> fullconnect_01 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 16 * 7 * 7, 30);//全连接层
		std::shared_ptr<Inter_LayerQL<double>> o_08 = o_07 + fullconnect_01;


		std::shared_ptr<LayerQL<double>> rule_03 = std::make_shared<Relu_LayerQL<double>>(Relu_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_09 = o_08 + rule_03;
		//rule_03->pRelu_k = 0.1;



		std::shared_ptr<LayerQL<double>> fullconnect_02 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 30, 10);//全连接层
		std::shared_ptr<Inter_LayerQL<double>> o_09_02 = o_09 + fullconnect_02;


		std::shared_ptr<LayerQL<double>> rule_04 = std::make_shared<Relu_LayerQL<double>>(Relu_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_09_03 = o_09_02 + rule_04;
		//rule_03->pRelu_k = 0.1;



		std::shared_ptr<LayerQL<double>> lossLayer_01 = std::make_shared<SoftMax_LayerQL<double>>(SoftMax_Layer);//Loss层
		std::shared_ptr<Inter_LayerQL<double>> o_10 = o_09_03 + lossLayer_01;

		for (int i = 0; i < 18; i++)
		{
			//	 程序加载初始时间
			DWORD load_time = GetTickCount();

			rule_01->pRelu_k = 0.12;
			rule_02->pRelu_k = 0.12;
			rule_03->pRelu_k = 0.12;
			rule_04->pRelu_k = 0.12;

			if (i < 2)
			{
				conv_01->upConv = 0.02;
				conv_02->upConv = 0.02;
				fullconnect_01->upFull = 0.015;
				fullconnect_02->upFull = 0.015;
			}
			if (i < 4)
			{
				conv_01->upConv = 0.015;
				conv_02->upConv = 0.015;
				fullconnect_01->upFull = 0.009;
				fullconnect_02->upFull = 0.009;
			}
			else if (i < 6)
			{
				conv_01->upConv = 0.008;
				conv_02->upConv = 0.008;
				fullconnect_01->upFull = 0.006;
				fullconnect_02->upFull = 0.006;
			}
			//这里有突变,重点关注这个学习率
			else if (i < 8)
			{
				conv_01->upConv = 0.004;
				conv_02->upConv = 0.004;
				fullconnect_01->upFull = 0.003;
				fullconnect_02->upFull = 0.003;
			}
			else if (i < 10)
			{
				conv_01->upConv = 0.002;
				conv_02->upConv = 0.002;
				fullconnect_01->upFull = 0.001;
				fullconnect_02->upFull = 0.001;
			}
			else if (i < 12)
			{
				conv_01->upConv = 0.001;
				conv_02->upConv = 0.001;
				fullconnect_01->upFull = 0.0005;
				fullconnect_02->upFull = 0.0005;
			}
			else if (i < 14)
			{
				conv_01->upConv = 0.0005;
				conv_02->upConv = 0.0005;
				fullconnect_01->upFull = 0.0002;
				fullconnect_02->upFull = 0.0002;
			}
			else if (i < 18)
			{
				conv_01->upConv = 0.00025;
				conv_02->upConv = 0.00025;
				fullconnect_01->upFull = 0.00015;
				fullconnect_02->upFull = 0.00015;
			}

			//#pragma omp parallel
			for (int j = 0; j < 55000; j++)
			{
				if (j % 10000 == 0) std::cout << i << "::" << j << std::endl;
				//入参
				in_01->forward_Matrix_Vector.clear();
				in_01->forward_Matrix_Vector.push_back(LoadCSV::conv_Input_Vector[j]);
				o_10->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer->backward_Matrix->getMatrixQL().row(j);

				//从头开始进行前向传播
				//#pragma omp parallel
				for (auto k = NetQL<double>::layerQLVector.begin(); k != NetQL<double>::layerQLVector.end(); k++)
				{
					(*k)->calForward();
				}
				//从头开始反向传播 + 权重更新
				//#pragma omp parallel
				for (auto k = NetQL<double>::layerQLVector.rbegin(); k != NetQL<double>::layerQLVector.rend(); k++)
				{
					(*k)->calBackward();
					(*k)->upMatrix();
				}
			}

			double numTotal = 0;
			for (int k = 0; k < 10000; k++)
			{
				//入参
				in_01->forward_Matrix_Vector.clear();
				in_01->forward_Matrix_Vector.push_back(LoadCSV::conv_Input_Vector_T[k]);
				o_10->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer_T->backward_Matrix->getMatrixQL().row(k);

				//前向传播，计算结果
				for (auto k = NetQL<double>::layerQLVector.begin(); k != NetQL<double>::layerQLVector.end(); k++)
				{
					(*k)->calForward();
				}

				//计算得到的最大值位置
				int maxRow, maxColumn;
				lossLayer_01->left_Layer->forward_Matrix->getMatrixQL().maxCoeff(&maxRow, &maxColumn);
				//Lable的最大值位置
				int maxRow_T, maxColumn_T;
				lossLayer_01->right_Layer->backward_Matrix->getMatrixQL().maxCoeff(&maxRow_T, &maxColumn_T);
				//判断是否相等，若相等，则+1
				if (maxColumn == maxColumn_T)
				{
					numTotal++;
				}
			}
			//正确率
			std::cout << numTotal / 10000.00 << std::endl;

			//训练和测试运行时间
			DWORD star_time = GetTickCount();

			//计算运行时间
			std::cout << "这个程序加载时间为：" << (star_time - load_time) << "ms." << std::endl;
		}
	}
}