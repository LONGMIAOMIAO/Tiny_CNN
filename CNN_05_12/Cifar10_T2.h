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
	//	66.6%
	void rightValue(std::shared_ptr<Inter_LayerQL<double>> inLayer, std::shared_ptr<Inter_LayerQL<double>> endLayer, std::shared_ptr<LayerQL<double>> lossLayer)
	{
		double numTotal = 0;
		for (int k = 0; k < 10000; k++)
		{
			inLayer->forward_Matrix_Vector = LoadCifar_10::cifar_Input_Vector_T[k];
			endLayer->backward_Matrix->setMatrixQL() = LoadCifar_10::cifar_Out_Lable_T->getMatrixQL().row(k);

			//前向传播，计算结果
			for (auto k = NetQL<double>::layerQLVector.begin(); k != NetQL<double>::layerQLVector.end(); k++)
			{
				(*k)->calForward();
			}

			//计算得到的最大值位置
			int maxRow, maxColumn;
			lossLayer->left_Layer->forward_Matrix->getMatrixQL().maxCoeff(&maxRow, &maxColumn);
			//Lable的最大值位置
			int maxRow_T, maxColumn_T;
			lossLayer->right_Layer->backward_Matrix->getMatrixQL().maxCoeff(&maxRow_T, &maxColumn_T);
			//判断是否相等，若相等，则+1
			if (maxColumn == maxColumn_T)
			{
				numTotal++;
			}
		}
		//正确率
		double accu;
		accu = numTotal / 10000.00;
		std::cout << accu << std::endl;
	}


	//	===========================================						
	void Cifar10_T2_1()
	{
		LoadCifar_10::loadCifar_10_Train();

		std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(32, 32);

		//**********************************************************************数据增强层
		//std::shared_ptr<LayerQL<double>> data_Aumentation_01 = std::make_shared<Data_AugmentationQL<double>>(Data_Augmentation_Layer, 0, 0);
		//std::shared_ptr<Inter_LayerQL<double>> o_00 = in_01 + data_Aumentation_01;

		//						类型			卷积核数			行数			列数				卷积核宽度		卷积核几片		扩充宽度	
		std::shared_ptr<LayerQL<double>> conv_01 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 16, 32, 32, 5, 3, 2);	//卷积层
		std::shared_ptr<Inter_LayerQL<double>> o_02 = in_01 + conv_01;


		std::shared_ptr<LayerQL<double>> sigmoid_01 = std::make_shared<Relu_LayerQL<double>>(Relu_Conv_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_03 = o_02 + sigmoid_01;
		//sigmoid_01->pRelu_k = 0.1;


		std::shared_ptr<LayerQL<double>> pool_02 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 16, 16);	//池化层
		std::shared_ptr<Inter_LayerQL<double>> o_04 = o_03 + pool_02;

		//						类型			卷积核数			行数			列数				卷积核宽度		卷积核几片		扩充宽度	
		std::shared_ptr<LayerQL<double>> conv_02 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 16, 16, 16, 3, 16, 1);	//卷积层
		std::shared_ptr<Inter_LayerQL<double>> o_04_02 = o_04 + conv_02;


		std::shared_ptr<LayerQL<double>> sigmoid_01_02 = std::make_shared<Relu_LayerQL<double>>(Relu_Conv_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_04_03 = o_04_02 + sigmoid_01_02;
		//sigmoid_01_02->pRelu_k = 0.1;


		std::shared_ptr<LayerQL<double>> pool_02_02 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 8, 8);	//池化层
		std::shared_ptr<Inter_LayerQL<double>> o_04_04 = o_04_03 + pool_02_02;


		std::shared_ptr<LayerQL<double>> dim_reduce_01 = std::make_shared<Dim_ReduceQL<double>>(Dim_Reduce_Layer, 16, 8, 8);	//降维层
		std::shared_ptr<Inter_LayerQL<double>> o_05 = o_04_04 + dim_reduce_01;


		std::shared_ptr<LayerQL<double>> fullconnect_01 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 16 * 8 * 8, 50);//全连接层
		std::shared_ptr<Inter_LayerQL<double>> o_06 = o_05 + fullconnect_01;


		std::shared_ptr<LayerQL<double>> sigmoid_02 = std::make_shared<Relu_LayerQL<double>>(Relu_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_07 = o_06 + sigmoid_02;
		//sigmoid_02->pRelu_k = 0.1;

		//===

		std::shared_ptr<LayerQL<double>> fullconnect_02 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 50, 10);//全连接层
		std::shared_ptr<Inter_LayerQL<double>> o_07_2 = o_07 + fullconnect_02;


		std::shared_ptr<LayerQL<double>> sigmoid_03 = std::make_shared<Relu_LayerQL<double>>(Relu_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_07_2_3 = o_07_2 + sigmoid_03;
		//sigmoid_02->pRelu_k = 0.1;


		//===


		std::shared_ptr<LayerQL<double>> lossLayer_01 = std::make_shared<SoftMax_LayerQL<double>>(SoftMax_Layer);//Loss层
		std::shared_ptr<Inter_LayerQL<double>> o_08 = o_07_2_3 + lossLayer_01;

		for (int i = 0; i < 18; i++)
		{
			//	 程序加载初始时间
			DWORD load_time = GetTickCount();

			sigmoid_01->pRelu_k = 0.12;
			sigmoid_01_02->pRelu_k = 0.12;
			sigmoid_02->pRelu_k = 0.12;
			sigmoid_03->pRelu_k = 0.12;

			if (i < 2)
			{
				conv_01->upConv = 0.010;
				conv_02->upConv = 0.010;
				fullconnect_01->upFull = 0.010;
				fullconnect_02->upFull = 0.010;
			}
			if (i < 4)
			{
				conv_01->upConv = 0.008;
				conv_02->upConv = 0.008;
				fullconnect_01->upFull = 0.008;
				fullconnect_02->upFull = 0.008;
			}
			else if (i < 6)
			{
				conv_01->upConv = 0.006;
				conv_02->upConv = 0.006;
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
			else if (i < 16)
			{
				conv_01->upConv = 0.00025;
				conv_02->upConv = 0.00025;
				fullconnect_01->upFull = 0.00015;
				fullconnect_02->upFull = 0.00015;
			}
			else if (i < 18)
			{
				conv_01->upConv = 0.0001;
				conv_02->upConv = 0.0001;
				fullconnect_01->upFull = 0.0001;
				fullconnect_02->upFull = 0.0001;
			}

			//#pragma omp parallel
			for (int j = 0; j < 50000; j++)
			{
				if (j % 10000 == 0) std::cout << i << "::" << j << std::endl;

				in_01->forward_Matrix_Vector = LoadCifar_10::cifar_Input_Vector[j];

				o_08->backward_Matrix->setMatrixQL() = LoadCifar_10::cifar_Out_Lable->getMatrixQL().row(j);

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

			rightValue(in_01, o_08, lossLayer_01);

			//训练和测试运行时间
			DWORD star_time = GetTickCount();

			//计算运行时间
			std::cout << "这个程序加载时间为：" << (star_time - load_time) << "ms." << std::endl;
		}
	}

	//	===========================================
	void Cifar10_T2_2()
	{
		LoadCifar_10::loadCifar_10_Train();

		std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(32, 32);

		//**********************************************************************数据增强层
		//std::shared_ptr<LayerQL<double>> data_Aumentation_01 = std::make_shared<Data_AugmentationQL<double>>(Data_Augmentation_Layer, 0, 0);
		//std::shared_ptr<Inter_LayerQL<double>> o_00 = in_01 + data_Aumentation_01;

		//						类型			卷积核数			行数			列数				卷积核宽度		卷积核几片		扩充宽度	
		std::shared_ptr<LayerQL<double>> conv_01 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 32, 32, 32, 5, 3, 2);	//卷积层
		std::shared_ptr<Inter_LayerQL<double>> o_02 = in_01 + conv_01;


		std::shared_ptr<LayerQL<double>> sigmoid_01 = std::make_shared<Relu_LayerQL<double>>(Relu_Conv_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_03 = o_02 + sigmoid_01;
		//sigmoid_01->pRelu_k = 0.1;


		std::shared_ptr<LayerQL<double>> pool_02 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 16, 16);	//池化层
		std::shared_ptr<Inter_LayerQL<double>> o_04 = o_03 + pool_02;

		//						类型			卷积核数			行数			列数				卷积核宽度		卷积核几片		扩充宽度	
		std::shared_ptr<LayerQL<double>> conv_02 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 32, 16, 16, 3, 32, 1);	//卷积层
		std::shared_ptr<Inter_LayerQL<double>> o_04_02 = o_04 + conv_02;


		std::shared_ptr<LayerQL<double>> sigmoid_01_02 = std::make_shared<Relu_LayerQL<double>>(Relu_Conv_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_04_03 = o_04_02 + sigmoid_01_02;
		//sigmoid_01_02->pRelu_k = 0.1;


		std::shared_ptr<LayerQL<double>> pool_02_02 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 8, 8);	//池化层
		std::shared_ptr<Inter_LayerQL<double>> o_04_04 = o_04_03 + pool_02_02;


		std::shared_ptr<LayerQL<double>> dim_reduce_01 = std::make_shared<Dim_ReduceQL<double>>(Dim_Reduce_Layer, 32, 8, 8);	//降维层
		std::shared_ptr<Inter_LayerQL<double>> o_05 = o_04_04 + dim_reduce_01;


		std::shared_ptr<LayerQL<double>> fullconnect_01 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 32 * 8 * 8, 50);//全连接层
		std::shared_ptr<Inter_LayerQL<double>> o_06 = o_05 + fullconnect_01;


		std::shared_ptr<LayerQL<double>> sigmoid_02 = std::make_shared<Relu_LayerQL<double>>(Relu_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_07 = o_06 + sigmoid_02;
		//sigmoid_02->pRelu_k = 0.1;

		//===

		std::shared_ptr<LayerQL<double>> fullconnect_02 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 50, 10);//全连接层
		std::shared_ptr<Inter_LayerQL<double>> o_07_2 = o_07 + fullconnect_02;


		std::shared_ptr<LayerQL<double>> sigmoid_03 = std::make_shared<Relu_LayerQL<double>>(Relu_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_07_2_3 = o_07_2 + sigmoid_03;
		//sigmoid_02->pRelu_k = 0.1;


		//===


		std::shared_ptr<LayerQL<double>> lossLayer_01 = std::make_shared<SoftMax_LayerQL<double>>(SoftMax_Layer);//Loss层
		std::shared_ptr<Inter_LayerQL<double>> o_08 = o_07_2_3 + lossLayer_01;

		for (int i = 0; i < 18; i++)
		{
			//	 程序加载初始时间
			DWORD load_time = GetTickCount();

			sigmoid_01->pRelu_k = 0.12;
			sigmoid_01_02->pRelu_k = 0.12;
			sigmoid_02->pRelu_k = 0.12;
			sigmoid_03->pRelu_k = 0.12;

			if (i < 2)
			{
				conv_01->upConv = 0.006;
				conv_02->upConv = 0.006;
				fullconnect_01->upFull = 0.006;
				fullconnect_02->upFull = 0.006;
			}
			if (i < 4)
			{
				conv_01->upConv = 0.004;
				conv_02->upConv = 0.004;
				fullconnect_01->upFull = 0.004;
				fullconnect_02->upFull = 0.004;
			}
			else if (i < 6)
			{
				conv_01->upConv = 0.002;
				conv_02->upConv = 0.002;
				fullconnect_01->upFull = 0.002;
				fullconnect_02->upFull = 0.002;
			}
			//这里有突变,重点关注这个学习率
			else if (i < 8)
			{
				conv_01->upConv = 0.001;
				conv_02->upConv = 0.001;
				fullconnect_01->upFull = 0.001;
				fullconnect_02->upFull = 0.001;
			}
			else if (i < 10)
			{
				conv_01->upConv = 0.0008;
				conv_02->upConv = 0.0008;
				fullconnect_01->upFull = 0.0008;
				fullconnect_02->upFull = 0.0008;
			}
			else if (i < 12)
			{
				conv_01->upConv = 0.0006;
				conv_02->upConv = 0.0006;
				fullconnect_01->upFull = 0.0006;
				fullconnect_02->upFull = 0.0006;
			}
			else if (i < 14)
			{
				conv_01->upConv = 0.0004;
				conv_02->upConv = 0.0004;
				fullconnect_01->upFull = 0.0004;
				fullconnect_02->upFull = 0.0004;
			}
			else if (i < 16)
			{
				conv_01->upConv = 0.00025;
				conv_02->upConv = 0.00025;
				fullconnect_01->upFull = 0.00015;
				fullconnect_02->upFull = 0.00015;
			}
			else if (i < 18)
			{
				conv_01->upConv = 0.0001;
				conv_02->upConv = 0.0001;
				fullconnect_01->upFull = 0.0001;
				fullconnect_02->upFull = 0.0001;
			}

			//#pragma omp parallel
			for (int j = 0; j < 50000; j++)
			{
				if (j % 10000 == 0) std::cout << i << "::" << j << std::endl;

				in_01->forward_Matrix_Vector = LoadCifar_10::cifar_Input_Vector[j];

				o_08->backward_Matrix->setMatrixQL() = LoadCifar_10::cifar_Out_Lable->getMatrixQL().row(j);

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

			rightValue(in_01, o_08, lossLayer_01);

			//训练和测试运行时间
			DWORD star_time = GetTickCount();

			//计算运行时间
			std::cout << "这个程序加载时间为：" << (star_time - load_time) << "ms." << std::endl;
		}
	}
}