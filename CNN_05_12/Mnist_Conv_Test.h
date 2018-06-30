#pragma once
#include "LoadCSV.h"
#include "LayerQL.h"
#include "PooLayerQL.h"
#include "Conv_LayerQL.h"
#include "Sigmoid_LayerQL.h"
#include "Dim_ReduceQL.h"
#include "Fullconnect_LayerQL.h"

namespace tinyDNN
{
	class Mnist_Conv_Test
	{
	public:
		Mnist_Conv_Test()
		{
			this->mnist_conv_01();
		}
		~Mnist_Conv_Test(){}


		void mnist_conv_01()
		{
			LoadCSV::loadCSVTrain();
			LoadCSV::loadCSV_Train_Vector();

			std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(28,28);
			in_01->forward_Matrix_Vector.push_back(LoadCSV::conv_Input_Vector[3]);

			std::shared_ptr<LayerQL<double>> pool_01 = std::make_shared<PooLayerQL<double>>(Pool_Layer,14,14);	//³Ø»¯²ã
			std::shared_ptr<Inter_LayerQL<double>> o_01 = in_01 + pool_01;

			//**********************************************************************³Ø»¯²ã
			//1111111111111111111111111111111111111111111111111111111111111111111111
			pool_01->calForward();
			std::cout << o_01->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << (o_01->forward_Matrix_Vector[0]->getMatrixQL() * 9).cast<int>() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> conv_01 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 8, 14, 14, 5, 1, 2 );	//¾í»ý²ã
			std::shared_ptr<Inter_LayerQL<double>> o_02 = o_01 + conv_01;

			//**********************************************************************¾í»ý²ã
			//2222222222222222222222222222222222222222222222222222222222222222222222
			conv_01->calForward();
			std::cout << o_02->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << "*******" << std::endl;
			std::cout << o_02->forward_Matrix_Vector[7]->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> sigmoid_01 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);	//Sigmoid²ã
			std::shared_ptr<Inter_LayerQL<double>> o_03 = o_02 + sigmoid_01;

			//**********************************************************************Sigmoid²ã
			//3333333333333333333333333333333333333333333333333333333333333333333333
			sigmoid_01->calForward(1);
			std::cout << o_03->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << "*******" << std::endl;
			std::cout << o_03->forward_Matrix_Vector[7]->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> pool_02 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 7, 7);	//³Ø»¯²ã
			std::shared_ptr<Inter_LayerQL<double>> o_04 = o_03 + pool_02;

			//**********************************************************************³Ø»¯²ã
			//4444444444444444444444444444444444444444444444444444444444444444444444
			pool_02->calForward();
			std::cout << o_04->forward_Matrix_Vector[0]->getMatrixQL() << std::endl;
			std::cout << "*******" << std::endl;
			std::cout << o_04->forward_Matrix_Vector[7]->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> dim_reduce_01 = std::make_shared<Dim_ReduceQL<double>>(Dim_Reduce_Layer, 8, 7, 7);	//½µÎ¬²ã
			std::shared_ptr<Inter_LayerQL<double>> o_05 = o_04 + dim_reduce_01;

			//**********************************************************************½µÎ¬²ã
			//55555555555555555555555555555555555555555555555555555555555555555555555
			dim_reduce_01->calForward();
			std::cout << o_05->forward_Matrix->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> fullconnect_01 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 392, 10 );//È«Á¬½Ó²ã
			std::shared_ptr<Inter_LayerQL<double>> o_06 = o_05 + fullconnect_01;

			//**********************************************************************È«Á¬½Ó²ã
			//66666666666666666666666666666666666666666666666666666666666666666666666
			fullconnect_01->calForward();
			std::cout << o_06->forward_Matrix->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> sigmoid_02 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);	//Sigmoid²ã
			std::shared_ptr<Inter_LayerQL<double>> o_07 = o_06 + sigmoid_02;

			//**********************************************************************Sigmoid²ã
			//77777777777777777777777777777777777777777777777777777777777777777777777
			sigmoid_02->calForward();
			std::cout << o_07->forward_Matrix->getMatrixQL() << std::endl;
			//**********************************************************************

			std::shared_ptr<LayerQL<double>> lossLayer_01 = std::make_shared<MSE_Loss_LayerQL<double>>(MSE_Loss_Layer);//Loss²ã
			std::shared_ptr<Inter_LayerQL<double>> o_08 = o_07 + lossLayer_01;

			//**********************************************************************Loss²ã
			//88888888888888888888888888888888888888888888888888888888888888888888888
			lossLayer_01->calForward();


			o_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer->backward_Matrix->getMatrixQL().row(3);


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
	};
}