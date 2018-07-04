#pragma once
#include "LoadCSV.h"
#include "LayerQL.h"
#include "PooLayerQL.h"
#include "Conv_LayerQL.h"
#include "Sigmoid_LayerQL.h"
#include "Dim_ReduceQL.h"
#include "Fullconnect_LayerQL.h"
#include "Inter_LayerQL.h"

namespace tinyDNN
{
	class Mnist_Conv_Test
	{
	public:
		Mnist_Conv_Test()
		{
			//this->mnist_conv_01();
			//this->mnist_conv_02();
			this->mnist_conv_03();
			//this->mnist_conv_04();
		}
		~Mnist_Conv_Test(){}

		//初始加载各层验证是否正确
		void mnist_conv_01()
		{
			LoadCSV::loadCSVTrain();
			LoadCSV::loadCSV_Train_Vector();

			std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(28,28);
			in_01->forward_Matrix_Vector.push_back(LoadCSV::conv_Input_Vector[4]);

			std::shared_ptr<LayerQL<double>> pool_01 = std::make_shared<PooLayerQL<double>>(Pool_Layer,14,14);	//池化层
			std::shared_ptr<Inter_LayerQL<double>> o_01 = in_01 + pool_01;

			//**********************************************************************池化层
			//1111111111111111111111111111111111111111111111111111111111111111111111
			pool_01->calForward();
			std::cout << o_01->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << (o_01->forward_Matrix_Vector[0]->getMatrixQL() * 9).cast<int>() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> conv_01 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 8, 14, 14, 5, 1, 2 );	//卷积层
			std::shared_ptr<Inter_LayerQL<double>> o_02 = o_01 + conv_01;

			//**********************************************************************卷积层
			//2222222222222222222222222222222222222222222222222222222222222222222222
			conv_01->calForward();
			std::cout << o_02->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << "*******" << std::endl;
			std::cout << o_02->forward_Matrix_Vector[7]->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> sigmoid_01 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);	//Sigmoid层
			std::shared_ptr<Inter_LayerQL<double>> o_03 = o_02 + sigmoid_01;

			//**********************************************************************Sigmoid层
			//3333333333333333333333333333333333333333333333333333333333333333333333
			sigmoid_01->calForward(1);
			std::cout << o_03->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << "*******" << std::endl;
			std::cout << o_03->forward_Matrix_Vector[7]->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> pool_02 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 7, 7);	//池化层
			std::shared_ptr<Inter_LayerQL<double>> o_04 = o_03 + pool_02;

			//**********************************************************************池化层
			//4444444444444444444444444444444444444444444444444444444444444444444444
			pool_02->calForward();
			std::cout << o_04->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << "*******" << std::endl;
			std::cout << o_04->forward_Matrix_Vector[7]->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> dim_reduce_01 = std::make_shared<Dim_ReduceQL<double>>(Dim_Reduce_Layer, 8, 7, 7);	//降维层
			std::shared_ptr<Inter_LayerQL<double>> o_05 = o_04 + dim_reduce_01;

			//**********************************************************************降维层
			//55555555555555555555555555555555555555555555555555555555555555555555555
			dim_reduce_01->calForward();
			std::cout << o_05->forward_Matrix->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> fullconnect_01 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 392, 10 );//全连接层
			std::shared_ptr<Inter_LayerQL<double>> o_06 = o_05 + fullconnect_01;

			//**********************************************************************全连接层
			//66666666666666666666666666666666666666666666666666666666666666666666666
			fullconnect_01->calForward();
			std::cout << o_06->forward_Matrix->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> sigmoid_02 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);	//Sigmoid层
			std::shared_ptr<Inter_LayerQL<double>> o_07 = o_06 + sigmoid_02;

			//**********************************************************************Sigmoid层
			//77777777777777777777777777777777777777777777777777777777777777777777777
			sigmoid_02->calForward();
			std::cout << o_07->forward_Matrix->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> lossLayer_01 = std::make_shared<MSE_Loss_LayerQL<double>>(MSE_Loss_Layer);//Loss层
			std::shared_ptr<Inter_LayerQL<double>> o_08 = o_07 + lossLayer_01;

			//**********************************************************************Loss层
			//88888888888888888888888888888888888888888888888888888888888888888888888
			lossLayer_01->calForward();


			o_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer->backward_Matrix->getMatrixQL().row(4);


			for (int i = 0; i < 100; i++)
			{
				//1
				pool_01->calForward();
				//2
				conv_01->calForward();
				//3
				sigmoid_01->calForward(1);
				//4
				pool_02->calForward();
				//5
				dim_reduce_01->calForward();
				//6
				fullconnect_01->calForward();
				//7
				sigmoid_02->calForward();
				//8
				lossLayer_01->calForward();

				//8
				lossLayer_01->calBackward();
				//7
				sigmoid_02->calBackward();
				//6
				fullconnect_01->calBackward();
				//5
				dim_reduce_01->calBackward();
				//4
				pool_02->calBackward();
				//3
				sigmoid_01->calBackward(1);
				//2
				conv_01->calBackward();
				//1
				pool_01->calBackward();

				//6
				fullconnect_01->upMatrix();
				//2
				conv_01->upMatrix();
			}

			std::cout << o_07->forward_Matrix->getMatrixQL() << std::endl;

		}
		//训练两轮 MNIST TRAIN ， 用 MNIST TRAIN 数据验证
		void mnist_conv_02()
		{
			LoadCSV::loadCSVTrain();
			LoadCSV::loadCSV_Train_Vector();

			std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(28, 28);
			in_01->forward_Matrix_Vector.push_back(LoadCSV::conv_Input_Vector[4]);

			std::shared_ptr<LayerQL<double>> pool_01 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 14, 14);	//池化层
			std::shared_ptr<Inter_LayerQL<double>> o_01 = in_01 + pool_01;

			//**********************************************************************池化层
			//1111111111111111111111111111111111111111111111111111111111111111111111
			pool_01->calForward();
			std::cout << o_01->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << (o_01->forward_Matrix_Vector[0]->getMatrixQL() * 9).cast<int>() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> conv_01 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 8, 14, 14, 5, 1, 2);	//卷积层
			std::shared_ptr<Inter_LayerQL<double>> o_02 = o_01 + conv_01;

			//**********************************************************************卷积层
			//2222222222222222222222222222222222222222222222222222222222222222222222
			conv_01->calForward();
			std::cout << o_02->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << "*******" << std::endl;
			std::cout << o_02->forward_Matrix_Vector[7]->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> sigmoid_01 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);	//Sigmoid层
			std::shared_ptr<Inter_LayerQL<double>> o_03 = o_02 + sigmoid_01;

			//**********************************************************************Sigmoid层
			//3333333333333333333333333333333333333333333333333333333333333333333333
			sigmoid_01->calForward(1);
			std::cout << o_03->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << "*******" << std::endl;
			std::cout << o_03->forward_Matrix_Vector[7]->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> pool_02 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 7, 7);	//池化层
			std::shared_ptr<Inter_LayerQL<double>> o_04 = o_03 + pool_02;

			//**********************************************************************池化层
			//4444444444444444444444444444444444444444444444444444444444444444444444
			pool_02->calForward();
			std::cout << o_04->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << "*******" << std::endl;
			std::cout << o_04->forward_Matrix_Vector[7]->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> dim_reduce_01 = std::make_shared<Dim_ReduceQL<double>>(Dim_Reduce_Layer, 8, 7, 7);	//降维层
			std::shared_ptr<Inter_LayerQL<double>> o_05 = o_04 + dim_reduce_01;

			//**********************************************************************降维层
			//55555555555555555555555555555555555555555555555555555555555555555555555
			dim_reduce_01->calForward();
			std::cout << o_05->forward_Matrix->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> fullconnect_01 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 392, 10);//全连接层
			std::shared_ptr<Inter_LayerQL<double>> o_06 = o_05 + fullconnect_01;

			//**********************************************************************全连接层
			//66666666666666666666666666666666666666666666666666666666666666666666666
			fullconnect_01->calForward();
			std::cout << o_06->forward_Matrix->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> sigmoid_02 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);	//Sigmoid层
			std::shared_ptr<Inter_LayerQL<double>> o_07 = o_06 + sigmoid_02;

			//**********************************************************************Sigmoid层
			//77777777777777777777777777777777777777777777777777777777777777777777777
			sigmoid_02->calForward();
			std::cout << o_07->forward_Matrix->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> lossLayer_01 = std::make_shared<MSE_Loss_LayerQL<double>>(MSE_Loss_Layer);//Loss层
			std::shared_ptr<Inter_LayerQL<double>> o_08 = o_07 + lossLayer_01;

			//**********************************************************************Loss层
			//88888888888888888888888888888888888888888888888888888888888888888888888
			lossLayer_01->calForward();

			o_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer->backward_Matrix->getMatrixQL().row(4);

			for (int i = 0; i < 1; i++)
			{
				for (int j = 0; j < 55000; j++)
				{
					//入参
					in_01->forward_Matrix_Vector.clear();
					in_01->forward_Matrix_Vector.push_back(LoadCSV::conv_Input_Vector[j]);

					o_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer->backward_Matrix->getMatrixQL().row(j);

					//1
					pool_01->calForward();
					//2
					conv_01->calForward();
					//3
					sigmoid_01->calForward(1);
					//4
					pool_02->calForward();
					//5
					dim_reduce_01->calForward();
					//6
					fullconnect_01->calForward();
					//7
					sigmoid_02->calForward();
					//8
					lossLayer_01->calForward();

					//8
					lossLayer_01->calBackward();
					//7
					sigmoid_02->calBackward();
					//6
					fullconnect_01->calBackward();
					//5
					dim_reduce_01->calBackward();
					//4
					pool_02->calBackward();
					//3
					sigmoid_01->calBackward(1);
					//2
					conv_01->calBackward();
					//1
					pool_01->calBackward();

					//6
					fullconnect_01->upMatrix();
					//2
					conv_01->upMatrix();
				}
			}

			for (int k = 1000; k < 1100; k++)
			{
				//入参
				in_01->forward_Matrix_Vector.clear();
				in_01->forward_Matrix_Vector.push_back(LoadCSV::conv_Input_Vector[k]);

				o_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer->backward_Matrix->getMatrixQL().row(k);

				//1
				pool_01->calForward();
				//2
				conv_01->calForward();
				//3
				sigmoid_01->calForward(1);
				//4
				pool_02->calForward();
				//5
				dim_reduce_01->calForward();
				//6
				fullconnect_01->calForward();
				//7
				sigmoid_02->calForward();
				//8
				lossLayer_01->calForward();

				//8
				lossLayer_01->calBackward();
				//7
				sigmoid_02->calBackward();
				//6
				fullconnect_01->calBackward();
				//5
				dim_reduce_01->calBackward();
				//4
				pool_02->calBackward();
				//3
				sigmoid_01->calBackward(1);
				//2
				conv_01->calBackward();
				//1
				pool_01->calBackward();

				//6
				fullconnect_01->upMatrix();
				//2
				conv_01->upMatrix();

				std::cout << k + 1 << " : *****************************************" << std::endl;
				std::cout << ( o_07->forward_Matrix->getMatrixQL() * 9 ).cast<int>() << std::endl;
			}
		}
		//训练两轮 MNIST TRAIN ， 用 MNIST TRAIN 数据验证
		void mnist_conv_03()
		{
			LoadCSV::loadCSVTrain();
			LoadCSV::loadCSV_Train_Vector();

			std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(28, 28);
			in_01->forward_Matrix_Vector.push_back(LoadCSV::conv_Input_Vector[4]);

			std::shared_ptr<LayerQL<double>> pool_01 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 14, 14);	//池化层
			std::shared_ptr<Inter_LayerQL<double>> o_01 = in_01 + pool_01;

			//**********************************************************************池化层
			//1111111111111111111111111111111111111111111111111111111111111111111111
			pool_01->calForward();
			std::cout << o_01->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << (o_01->forward_Matrix_Vector[0]->getMatrixQL() * 9).cast<int>() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> conv_01 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 8, 14, 14, 5, 1, 2);	//卷积层
			std::shared_ptr<Inter_LayerQL<double>> o_02 = o_01 + conv_01;

			//**********************************************************************卷积层
			//2222222222222222222222222222222222222222222222222222222222222222222222
			conv_01->calForward();
			std::cout << o_02->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << "*******" << std::endl;
			std::cout << o_02->forward_Matrix_Vector[7]->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> sigmoid_01 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);	//Sigmoid层
			std::shared_ptr<Inter_LayerQL<double>> o_03 = o_02 + sigmoid_01;

			//**********************************************************************Sigmoid层
			//3333333333333333333333333333333333333333333333333333333333333333333333
			sigmoid_01->calForward(1);
			std::cout << o_03->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << "*******" << std::endl;
			std::cout << o_03->forward_Matrix_Vector[7]->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> pool_02 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 7, 7);	//池化层
			std::shared_ptr<Inter_LayerQL<double>> o_04 = o_03 + pool_02;

			//**********************************************************************池化层
			//4444444444444444444444444444444444444444444444444444444444444444444444
			pool_02->calForward();
			std::cout << o_04->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << "*******" << std::endl;
			std::cout << o_04->forward_Matrix_Vector[7]->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> dim_reduce_01 = std::make_shared<Dim_ReduceQL<double>>(Dim_Reduce_Layer, 8, 7, 7);	//降维层
			std::shared_ptr<Inter_LayerQL<double>> o_05 = o_04 + dim_reduce_01;

			//**********************************************************************降维层
			//55555555555555555555555555555555555555555555555555555555555555555555555
			dim_reduce_01->calForward();
			std::cout << o_05->forward_Matrix->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> fullconnect_01 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 392, 10);//全连接层
			std::shared_ptr<Inter_LayerQL<double>> o_06 = o_05 + fullconnect_01;

			//**********************************************************************全连接层
			//66666666666666666666666666666666666666666666666666666666666666666666666
			fullconnect_01->calForward();
			std::cout << o_06->forward_Matrix->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> sigmoid_02 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);	//Sigmoid层
			std::shared_ptr<Inter_LayerQL<double>> o_07 = o_06 + sigmoid_02;

			//**********************************************************************Sigmoid层
			//77777777777777777777777777777777777777777777777777777777777777777777777
			sigmoid_02->calForward();
			std::cout << o_07->forward_Matrix->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> lossLayer_01 = std::make_shared<MSE_Loss_LayerQL<double>>(MSE_Loss_Layer);//Loss层
			std::shared_ptr<Inter_LayerQL<double>> o_08 = o_07 + lossLayer_01;

			//**********************************************************************Loss层
			//88888888888888888888888888888888888888888888888888888888888888888888888
			lossLayer_01->calForward();

			o_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer->backward_Matrix->getMatrixQL().row(4);


			//	 程序加载初始时间
			DWORD load_time = GetTickCount();

			for (int i = 0; i < 20; i++)
			{
				for (int j = 0; j < 55000; j++)
				{
					//入参
					in_01->forward_Matrix_Vector.clear();
					in_01->forward_Matrix_Vector.push_back(LoadCSV::conv_Input_Vector[j]);

					o_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer->backward_Matrix->getMatrixQL().row(j);

					//1
					pool_01->calForward();
					//2
					conv_01->calForward();
					//3
					sigmoid_01->calForward(1);
					//4
					pool_02->calForward();
					//5
					dim_reduce_01->calForward();
					//6
					fullconnect_01->calForward();
					//7
					sigmoid_02->calForward();
					//8
					lossLayer_01->calForward();

					//8
					lossLayer_01->calBackward();
					//7
					sigmoid_02->calBackward();
					//6
					fullconnect_01->calBackward();
					//5
					dim_reduce_01->calBackward();
					//4
					pool_02->calBackward();
					//3
					sigmoid_01->calBackward(1);
					//2
					conv_01->calBackward();
					//1
					pool_01->calBackward();

					//6
					fullconnect_01->upMatrix();
					//2
					conv_01->upMatrix();
				}
			}

			//*******************************************测试
			LoadCSV::loadCSVTest();
			LoadCSV::loadCSV_Test_Vector();

			double numTotal = 0;
			for (int k = 0; k < 10000; k++)
			{
				//入参
				in_01->forward_Matrix_Vector.clear();
				in_01->forward_Matrix_Vector.push_back(LoadCSV::conv_Input_Vector_T[k]);

				o_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer_T->backward_Matrix->getMatrixQL().row(k);

				//1
				pool_01->calForward();		//forward
				//2
				conv_01->calForward();
				//3
				sigmoid_01->calForward(1);
				//4
				pool_02->calForward();
				//5
				dim_reduce_01->calForward();
				//6
				fullconnect_01->calForward();
				//7
				sigmoid_02->calForward();
				//8
				lossLayer_01->calForward();

				////8
				//lossLayer_01->calBackward();	//back
				////7
				//sigmoid_02->calBackward();
				////6
				//fullconnect_01->calBackward();
				////5
				//dim_reduce_01->calBackward();
				////4
				//pool_02->calBackward();
				////3
				//sigmoid_01->calBackward(1);
				////2
				//conv_01->calBackward();
				////1
				//pool_01->calBackward();

				////6
				//fullconnect_01->upMatrix();		//up
				////2
				//conv_01->upMatrix();

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
		//对双卷积层进行计算，验证双卷积的反向传播的正确性
		void mnist_conv_04()
		{
			LoadCSV::loadCSVTrain();
			LoadCSV::loadCSV_Train_Vector();

			std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(28, 28);
			in_01->forward_Matrix_Vector.push_back(LoadCSV::conv_Input_Vector[4]);

			std::shared_ptr<LayerQL<double>> pool_01 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 14, 14);	//池化层
			std::shared_ptr<Inter_LayerQL<double>> o_01 = in_01 + pool_01;

			//**********************************************************************池化层
			//1111111111111111111111111111111111111111111111111111111111111111111111
			pool_01->calForward();
			std::cout << o_01->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << (o_01->forward_Matrix_Vector[0]->getMatrixQL() * 9).cast<int>() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> conv_01 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 8, 14, 14, 5, 1, 2);	//卷积层
			std::shared_ptr<Inter_LayerQL<double>> o_02 = o_01 + conv_01;

			//**********************************************************************卷积层
			//2222222222222222222222222222222222222222222222222222222222222222222222
			conv_01->calForward();
			std::cout << o_02->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << "*******" << std::endl;
			std::cout << o_02->forward_Matrix_Vector[7]->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> sigmoid_01 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);	//Sigmoid层
			std::shared_ptr<Inter_LayerQL<double>> o_03 = o_02 + sigmoid_01;

			//**********************************************************************Sigmoid层
			//3333333333333333333333333333333333333333333333333333333333333333333333
			sigmoid_01->calForward(1);
			std::cout << o_03->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << "*******" << std::endl;
			std::cout << o_03->forward_Matrix_Vector[7]->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> pool_02 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 7, 7);	//池化层
			std::shared_ptr<Inter_LayerQL<double>> o_04 = o_03 + pool_02;

			//**********************************************************************池化层
			//4444444444444444444444444444444444444444444444444444444444444444444444
			pool_02->calForward();
			std::cout << o_04->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << "*******" << std::endl;
			std::cout << o_04->forward_Matrix_Vector[7]->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> conv_02 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 8, 7, 7, 3, 8, 1);	//卷积层
			std::shared_ptr<Inter_LayerQL<double>> o_04_02 = o_04 + conv_02;
			conv_02->calForward();


			std::shared_ptr<LayerQL<double>> dim_reduce_01 = std::make_shared<Dim_ReduceQL<double>>(Dim_Reduce_Layer, 8, 7, 7);	//降维层
			std::shared_ptr<Inter_LayerQL<double>> o_05 = o_04_02 + dim_reduce_01;

			//**********************************************************************降维层
			//55555555555555555555555555555555555555555555555555555555555555555555555
			dim_reduce_01->calForward();
			std::cout << o_05->forward_Matrix->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> fullconnect_01 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 8*7*7, 10);//全连接层
			std::shared_ptr<Inter_LayerQL<double>> o_06 = o_05 + fullconnect_01;

			//**********************************************************************全连接层
			//66666666666666666666666666666666666666666666666666666666666666666666666
			fullconnect_01->calForward();
			std::cout << o_06->forward_Matrix->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> sigmoid_02 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);	//Sigmoid层
			std::shared_ptr<Inter_LayerQL<double>> o_07 = o_06 + sigmoid_02;

			//**********************************************************************Sigmoid层
			//77777777777777777777777777777777777777777777777777777777777777777777777
			sigmoid_02->calForward();
			std::cout << o_07->forward_Matrix->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> lossLayer_01 = std::make_shared<MSE_Loss_LayerQL<double>>(MSE_Loss_Layer);//Loss层
			std::shared_ptr<Inter_LayerQL<double>> o_08 = o_07 + lossLayer_01;

			//**********************************************************************Loss层
			//88888888888888888888888888888888888888888888888888888888888888888888888
			lossLayer_01->calForward();

			o_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer->backward_Matrix->getMatrixQL().row(4);


			//	 程序加载初始时间
			DWORD load_time = GetTickCount();

			for (int i = 0; i < 20; i++)
			{
				for (int j = 0; j < 55000; j++)
				{
					//入参
					in_01->forward_Matrix_Vector.clear();
					in_01->forward_Matrix_Vector.push_back(LoadCSV::conv_Input_Vector[j]);

					o_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer->backward_Matrix->getMatrixQL().row(j);

					//1
					pool_01->calForward();
					//2
					conv_01->calForward();
					//3
					sigmoid_01->calForward(1);
					//4
					pool_02->calForward();
					//***********************************
					conv_02->calForward();
					//5
					dim_reduce_01->calForward();
					//6
					fullconnect_01->calForward();
					//7
					sigmoid_02->calForward();
					//8
					lossLayer_01->calForward();

					//8
					lossLayer_01->calBackward();
					//7
					sigmoid_02->calBackward();
					//6
					fullconnect_01->calBackward();
					//5
					dim_reduce_01->calBackward();
					//***********************************
					conv_02->calBackward();
					//4
					pool_02->calBackward();
					//3
					sigmoid_01->calBackward(1);
					//2
					conv_01->calBackward();
					//1
					pool_01->calBackward();

					//6
					fullconnect_01->upMatrix();
					//2
					conv_01->upMatrix();
				}
			}

			//*******************************************测试
			LoadCSV::loadCSVTest();
			LoadCSV::loadCSV_Test_Vector();

			double numTotal = 0;
			for (int k = 0; k < 10000; k++)
			{
				//入参
				in_01->forward_Matrix_Vector.clear();
				in_01->forward_Matrix_Vector.push_back(LoadCSV::conv_Input_Vector_T[k]);

				o_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer_T->backward_Matrix->getMatrixQL().row(k);

				//1
				pool_01->calForward();		//forward
											//2
				conv_01->calForward();
				//3
				sigmoid_01->calForward(1);
				//4
				pool_02->calForward();
				//***********************************
				conv_02->calForward();
				//5
				dim_reduce_01->calForward();
				//6
				fullconnect_01->calForward();
				//7
				sigmoid_02->calForward();
				//8
				lossLayer_01->calForward();

				////8
				//lossLayer_01->calBackward();	//back
				////7
				//sigmoid_02->calBackward();
				////6
				//fullconnect_01->calBackward();
				////5
				//dim_reduce_01->calBackward();
				////4
				//pool_02->calBackward();
				////3
				//sigmoid_01->calBackward(1);
				////2
				//conv_01->calBackward();
				////1
				//pool_01->calBackward();

				////6
				//fullconnect_01->upMatrix();		//up
				////2
				//conv_01->upMatrix();

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
	};
}