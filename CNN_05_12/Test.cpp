#include "Test.h"

namespace tinyDNN
{
	std::shared_ptr<Inter_LayerQL<double>> Test::input_Layer = std::make_shared<Inter_LayerQL<double>>(55000, 784);
	std::shared_ptr<Inter_LayerQL<double>> Test::output_Layer = std::make_shared<Inter_LayerQL<double>>(55000, 10);

	std::shared_ptr<Inter_LayerQL<double>> Test::input_Layer_T = std::make_shared<Inter_LayerQL<double>>(10000, 784);
	std::shared_ptr<Inter_LayerQL<double>> Test::output_Layer_T = std::make_shared<Inter_LayerQL<double>>(10000, 10);

	Test::Test()
	{
		//Full_Layer ForwardTrans
		//this->Fullconnect_Layer_Forward_Test();
		////Full_Layer BackwardTrans
		//this->Fullconnect_Layer_Backward_Test();
		//Full_Layer_update
		//this->Fullconnect_Layer_Update_Test();
		//Full_Layer_update_batch
		//this->Fullconnect_Layer_Update_Batch_Test();

		////Bias_Layer ForwardTrans
		//this->Bias_Layer_Forward_Test();
		////Bias_Layer BackwardTrans
		//this->Bias_Layer_Backward_Test();
		////Bias_Layer update
		//this->Bias_Layer_Update_Test();
		////Bias_Layer_Batch update
		//this->Bias_Layer_Update_Batch_Test();

		//Operator_Layer
		//this->Operator_Test();
		//==========================================
		//Sigmoid_LayerQL ForwardTrans
		//this->Sigmoid_LayerQL_Forward_Test();
		//Sigmoid_LayerQL BackwardTrans
		//this->Sigmoid_LayerQL_Backward_Test();
		//MSE_LOSS_BACKWARD_TEST()
		//this->MSE_Loss_LayerQL_Backward_Test();

		//==========================================
		////测试SGD
		//this->Mnist_Test();
		//测试批量下降
		this->Mnist_Test_02();
	}

	Test::~Test()
	{
	}
	
	void Test::Fullconnect_Layer_Forward_Test()
	{
		//	创建输入智能指针
		std::shared_ptr<Inter_LayerQL<double>> left_Layer = std::make_shared<Inter_LayerQL<double>>(2,3);
		left_Layer->forward_Matrix->setMatrixQL() << 1,2,3,4,5,6;
		std::cout << left_Layer->forward_Matrix->getMatrixQL() << std::endl;
		//	创建第一个全连接层
		std::shared_ptr<LayerQL<double>> fullLayerTest = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 3, 5);
		//	输入与全连接层的相乘
		std::shared_ptr<Inter_LayerQL<double>> out_01 = left_Layer + fullLayerTest;
		//	向前计算全连接
		fullLayerTest->calForward();
		//	输出全连接前后的层
		std::cout << fullLayerTest->left_Layer->forward_Matrix->getMatrixQL() << std::endl;
		std::cout << fullLayerTest->right_Layer->forward_Matrix->getMatrixQL() << std::endl;
		//	创建第二个全连接层
		std::shared_ptr<LayerQL<double>> fullLayerTest_02 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 5, 7);
		std::shared_ptr<Inter_LayerQL<double>> out_02 = out_01 + fullLayerTest_02;
		//	第二个全连接层向前计算
		fullLayerTest_02->calForward();
		//	输出第二个全连接层的计算结果
		std::cout << fullLayerTest_02->left_Layer->forward_Matrix->getMatrixQL() << std::endl;
		std::cout << fullLayerTest_02->right_Layer->forward_Matrix->getMatrixQL() << std::endl;
	}

	void Test::Fullconnect_Layer_Backward_Test()
	{
		std::shared_ptr<Inter_LayerQL<double>> InLayer_01 = std::make_shared<Inter_LayerQL<double>>();

		//	第一层中间层
		std::shared_ptr<LayerQL<double>> fullLayer_01 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer,2,3);
		//	第一层输出
		std::shared_ptr<Inter_LayerQL<double>> InLayer_02 = InLayer_01 + fullLayer_01;
		//	第二层中间层
		std::shared_ptr<LayerQL<double>> fullLayer_02 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer,3,5);
		//	第二层输出
		std::shared_ptr<Inter_LayerQL<double>> InLayer_03 = InLayer_02 + fullLayer_02;
		//	初始化第二层的LOSS
		InLayer_03->backward_Matrix = std::make_unique<MatrixQL<double>>(1, 5);
		InLayer_03->backward_Matrix->setMatrixQL().setConstant(1.0);
		
		std::cout << fullLayer_02->right_Layer->backward_Matrix->getMatrixQL() << std::endl;
		fullLayer_02->calBackward();
		std::cout << fullLayer_02->left_Layer->backward_Matrix->getMatrixQL() << std::endl;		
		
		int mm = 0;

	}

	void Test::Fullconnect_Layer_Update_Test()
	{
		std::shared_ptr<Inter_LayerQL<double>> left_Layer = std::make_shared<Inter_LayerQL<double>>(1, 5);
		left_Layer->forward_Matrix->setMatrixQL() << 1, 2, 3, 4, 5;

		std::shared_ptr<LayerQL<double>> fullLayerTest = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 5, 10);

		std::shared_ptr<Inter_LayerQL<double>> right_Layer = left_Layer + fullLayerTest;

		right_Layer->backward_Matrix->setMatrixQL().resize(1, 10);
		right_Layer->backward_Matrix->setMatrixQL().setConstant(1);
		
		fullLayerTest->upMatrix();

	}

	void Test::Fullconnect_Layer_Update_Batch_Test()
	{
		std::shared_ptr<Inter_LayerQL<double>> left_Layer = std::make_shared<Inter_LayerQL<double>>(3, 5);
		left_Layer->forward_Matrix->setMatrixQL() << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11 ,12, 13, 14, 15;

		std::shared_ptr<LayerQL<double>> fullLayerTest = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 5, 10);

		std::shared_ptr<Inter_LayerQL<double>> right_Layer = left_Layer + fullLayerTest;

		right_Layer->backward_Matrix->setMatrixQL().resize(3, 10);
		right_Layer->backward_Matrix->setMatrixQL().setConstant(1);

		fullLayerTest->upMatrix_batch();
	}


	void Test::Bias_Layer_Forward_Test()
	{
		std::shared_ptr<Inter_LayerQL<double>> left_Layer = std::make_shared<Inter_LayerQL<double>>(2,3);
		left_Layer->forward_Matrix->setMatrixQL().setConstant(1.2);
		std::cout << left_Layer->forward_Matrix->getMatrixQL() << std::endl;

		std::shared_ptr<LayerQL<double>> bias_Layer = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 1, 3, 0.1);
		std::shared_ptr<Inter_LayerQL<double>> out_01 = left_Layer + bias_Layer;

		bias_Layer->calForward();

		std::cout << bias_Layer->left_Layer->forward_Matrix->getMatrixQL() << std::endl;
		std::cout << bias_Layer->right_Layer->forward_Matrix->getMatrixQL() << std::endl;
		//*******************************************************************************************************//

		std::shared_ptr<LayerQL<double>> bias_Layer_02 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 1, 3, 0.1);
		std::shared_ptr<Inter_LayerQL<double>> out_02 = out_01 + bias_Layer_02;

		bias_Layer_02->calForward();

		std::cout << bias_Layer_02->left_Layer->forward_Matrix->getMatrixQL() << std::endl;
		std::cout << bias_Layer_02->right_Layer->forward_Matrix->getMatrixQL() << std::endl;

		//*******************************************************************************************************//

		std::shared_ptr<LayerQL<double>> bias_Layer_03 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 1, 3, 0.1);
		std::shared_ptr<Inter_LayerQL<double>> out_03 = out_02 + bias_Layer_03;

		bias_Layer_03->calForward();

		std::cout << bias_Layer_03->left_Layer->forward_Matrix->getMatrixQL() << std::endl;
		std::cout << bias_Layer_03->right_Layer->forward_Matrix->getMatrixQL() << std::endl;

	}

	void Test::Bias_Layer_Backward_Test()
	{
		std::shared_ptr<Inter_LayerQL<double>> InLayer_01 = std::make_shared<Inter_LayerQL<double>>();
		std::shared_ptr<LayerQL<double>> biasLayer_01 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer,3, 5, 0.1);
		std::shared_ptr<Inter_LayerQL<double>> InLayer_02 = InLayer_01 + biasLayer_01;
		biasLayer_01->right_Layer->backward_Matrix->setMatrixQL().resize(3, 5);
		biasLayer_01->right_Layer->backward_Matrix->setMatrixQL().setOnes();
		std::cout << biasLayer_01->right_Layer->backward_Matrix->getMatrixQL() << std::endl;

		biasLayer_01->calBackward();

		std::cout << biasLayer_01->left_Layer->backward_Matrix->getMatrixQL() << std::endl;

	}

	void Test::Bias_Layer_Update_Test()
	{
		std::shared_ptr<Inter_LayerQL<double>> left_Layer = std::make_shared<Inter_LayerQL<double>>(1, 5);
		left_Layer->forward_Matrix->setMatrixQL().setConstant(1.2);

		std::shared_ptr<LayerQL<double>> bias_Layer = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 1, 5, 0.1);
		std::shared_ptr<Inter_LayerQL<double>> right_01 = left_Layer + bias_Layer;

		right_01->backward_Matrix->setMatrixQL().resize(1, 5);
		right_01->backward_Matrix->setMatrixQL().setConstant(10);

		bias_Layer->upMatrix();

	}

	void Test::Bias_Layer_Update_Batch_Test()
	{
		std::shared_ptr<Inter_LayerQL<double>> left_Layer = std::make_shared<Inter_LayerQL<double>>(3, 5);
		left_Layer->forward_Matrix->setMatrixQL().setConstant(1.2);

		std::shared_ptr<LayerQL<double>> bias_Layer = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 1, 5, 0.1);
		std::shared_ptr<Inter_LayerQL<double>> right_01 = left_Layer + bias_Layer;

		right_01->backward_Matrix->setMatrixQL().resize(3, 5);
		right_01->backward_Matrix->setMatrixQL().setConstant(10);

		bias_Layer->upMatrix_batch();
	}



	void Test::Sigmoid_LayerQL_Forward_Test()
	{
		std::shared_ptr<Inter_LayerQL<double>> intputLayer_01 = std::make_shared<Inter_LayerQL<double>>(2,5);
		intputLayer_01->forward_Matrix->setMatrixQL().setOnes();

		std::shared_ptr<LayerQL<double>> sigmoidLayer_01 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);
		std::shared_ptr<Inter_LayerQL<double>> inputLayer_02 = intputLayer_01 + sigmoidLayer_01;

		sigmoidLayer_01->calForward();

		std::cout << inputLayer_02->forward_Matrix->getMatrixQL() << std::endl;

	}

	void Test::Sigmoid_LayerQL_Backward_Test()
	{
		std::shared_ptr<Inter_LayerQL<double>> intputLayer_01 = std::make_shared<Inter_LayerQL<double>>();

		std::shared_ptr<LayerQL<double>> sigmoidLayer_01 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);
		std::shared_ptr<Inter_LayerQL<double>> inputLayer_02 = intputLayer_01 + sigmoidLayer_01;

		inputLayer_02->forward_Matrix->setMatrixQL().resize(2, 5);
		inputLayer_02->forward_Matrix->setMatrixQL().setConstant(0.5);

		inputLayer_02->backward_Matrix->setMatrixQL().resize(2, 5);
		inputLayer_02->backward_Matrix->setMatrixQL().setConstant(1);
		sigmoidLayer_01->calBackward();

		std::cout << intputLayer_01->backward_Matrix->getMatrixQL() << std::endl;
	}



	void Test::MSE_Loss_LayerQL_Backward_Test()
	{
		using inLayer = std::shared_ptr<Inter_LayerQL<double>>;
		inLayer input_01 = std::make_shared<Inter_LayerQL<double>>(2,5);
		input_01->forward_Matrix->setMatrixQL().setRandom();
		std::cout << input_01->forward_Matrix->getMatrixQL() << std::endl;

		std::shared_ptr<LayerQL<double>> lossLayer_01 = std::make_shared<MSE_Loss_LayerQL<double>>(MSE_Loss_Layer);

		inLayer input_02 = input_01 + lossLayer_01;

		input_02->backward_Matrix->setMatrixQL().resize(2, 5);
		input_02->backward_Matrix->setMatrixQL().setConstant(1.0);
		
		lossLayer_01->calBackward();

		std::cout << input_01->backward_Matrix->getMatrixQL() << std::endl;

	}



	void Test::Operator_Test()
	{


	}

	void Test::Mnist_Test()
	{
		//NetQL<double> layerNet;
		//layerNet.layerQLVector.push_back();
		//NetQL<double>::layerQLVector;

		LoadCSV::loadCSVTrain();
		LoadCSV::loadCSVTest();

		std::shared_ptr<Inter_LayerQL<double>> inLayer_01 = std::make_shared<Inter_LayerQL<double>>(1, 784);


		std::shared_ptr<LayerQL<double>> fullLayer_01 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 784, 100);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_02 = inLayer_01 + fullLayer_01;
		std::shared_ptr<LayerQL<double>> biasLayer_01 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 1, 100, 0.1);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_03 = inLayer_02 + biasLayer_01;
		std::shared_ptr<LayerQL<double>> sigmoidLayer_01 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_04 = inLayer_03 + sigmoidLayer_01;


		std::shared_ptr<LayerQL<double>> fullLayer_02 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 100, 10);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_05 = inLayer_04 + fullLayer_02;
		std::shared_ptr<LayerQL<double>> biasLayer_02 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 1, 10, 0.1);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_06 = inLayer_05 + biasLayer_02;
		std::shared_ptr<LayerQL<double>> sigmoidLayer_02 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_07 = inLayer_06 + sigmoidLayer_02;


		std::shared_ptr<LayerQL<double>> lossLayer_01 = std::make_shared<MSE_Loss_LayerQL<double>>(MSE_Loss_Layer);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_08 = inLayer_07 + lossLayer_01;

		//	 程序加载初始时间
		DWORD load_time = GetTickCount();

		for (int i = 0; i < 2 ; i ++)
		{
			for (int j = 0; j < 55000; j++)
			{
				inLayer_01->forward_Matrix->setMatrixQL() = Test::input_Layer->forward_Matrix->getMatrixQL().row(j);
				inLayer_08->backward_Matrix->setMatrixQL() = Test::output_Layer->backward_Matrix->getMatrixQL().row(j);

				for (auto k = NetQL<double>::layerQLVector.begin(); k != NetQL<double>::layerQLVector.end(); k++)
				{
					(*k)->calForward();
				}

				for (auto k = NetQL<double>::layerQLVector.rbegin(); k != NetQL<double>::layerQLVector.rend(); k++)
				{
					(*k)->calBackward();
					(*k)->upMatrix();
				}
				//for (auto k = NetQL<double>::layerQLVector.begin(); k != NetQL<double>::layerQLVector.end(); k++)
				//{
				//	(*k)->upMatrix();
				//}
				//fullLayer_01->calForward();
				//biasLayer_01->calForward();
				//sigmoidLayer_01->calForward();

				//fullLayer_02->calForward();
				//biasLayer_02->calForward();
				//sigmoidLayer_02->calForward();

				//lossLayer_01->calForward();
				////=======================================
				//lossLayer_01->calBackward();

				//sigmoidLayer_02->calBackward();
				//biasLayer_02->calBackward();
				//fullLayer_02->calBackward();
				//
				//sigmoidLayer_01->calBackward();
				//biasLayer_01->calBackward();
				//fullLayer_01->calBackward();

				////========================================

				//fullLayer_01->upMatrix();
				//biasLayer_01->upMatrix();

				//fullLayer_02->upMatrix();
				//biasLayer_02->upMatrix();
			}
		}

		double numTotal = 0;
		for ( int i = 0; i <10000; i ++ )
		{
			inLayer_01->forward_Matrix->setMatrixQL() = Test::input_Layer_T->forward_Matrix->getMatrixQL().row(i);
			inLayer_08->backward_Matrix->setMatrixQL() = Test::output_Layer_T->backward_Matrix->getMatrixQL().row(i);

			for (auto k = NetQL<double>::layerQLVector.begin(); k != NetQL<double>::layerQLVector.end(); k++)
			{
				(*k)->calForward();
			}
			//fullLayer_01->calForward();
			//biasLayer_01->calForward();
			//sigmoidLayer_01->calForward();

			//fullLayer_02->calForward();
			//biasLayer_02->calForward();
			//sigmoidLayer_02->calForward();

			//lossLayer_01->calForward();

			int maxRow, maxColumn;

			lossLayer_01->left_Layer->forward_Matrix->getMatrixQL().maxCoeff(&maxRow, &maxColumn);

			int maxRow_T, maxColumn_T;
			lossLayer_01->right_Layer->backward_Matrix->getMatrixQL().maxCoeff(&maxRow_T, &maxColumn_T);

			if (maxColumn == maxColumn_T)
			{
				numTotal++;
			}

		}
		std::cout << numTotal / 10000.00 << std::endl;

		//	训练和测试运行时间
		DWORD star_time = GetTickCount();

		std::cout << "这个程序加载时间为：" << (star_time - load_time) << "ms." << std::endl;

		//std::cout << input_Layer->forward_Matrix->getMatrixQL().row(9999) << std::endl;
		//std::cout << input_Layer_T->forward_Matrix->getMatrixQL().row(9999) << std::endl;

		//std::cout << output_Layer->backward_Matrix->getMatrixQL().row(0) << std::endl;
		//std::cout << output_Layer_T->backward_Matrix->getMatrixQL().row(0) << std::endl;
	}

	void Test::Mnist_Test_02()
	{
		LoadCSV::loadCSVTrain();
		LoadCSV::loadCSVTest();

		std::shared_ptr<Inter_LayerQL<double>> inLayer_01 = std::make_shared<Inter_LayerQL<double>>(100, 784);

		std::shared_ptr<LayerQL<double>> fullLayer_01 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 784, 100);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_02 = inLayer_01 + fullLayer_01;
		std::shared_ptr<LayerQL<double>> biasLayer_01 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 1, 100, 0.1);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_03 = inLayer_02 + biasLayer_01;
		std::shared_ptr<LayerQL<double>> sigmoidLayer_01 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_04 = inLayer_03 + sigmoidLayer_01;


		std::shared_ptr<LayerQL<double>> fullLayer_02 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 100, 10);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_05 = inLayer_04 + fullLayer_02;
		std::shared_ptr<LayerQL<double>> biasLayer_02 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 1, 10, 0.1);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_06 = inLayer_05 + biasLayer_02;
		std::shared_ptr<LayerQL<double>> sigmoidLayer_02 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_07 = inLayer_06 + sigmoidLayer_02;


		std::shared_ptr<LayerQL<double>> lossLayer_01 = std::make_shared<MSE_Loss_LayerQL<double>>(MSE_Loss_Layer);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_08 = inLayer_07 + lossLayer_01;

		//	 程序加载初始时间
		DWORD load_time = GetTickCount();

		for (int i = 0; i < 20; i++)
		{
			for (int j = 0; j < 550; j++)
			{
				inLayer_01->forward_Matrix->setMatrixQL() = Test::input_Layer->forward_Matrix->getMatrixQL().block(j*100, 0, 100, 784);
				inLayer_08->backward_Matrix->setMatrixQL() = Test::output_Layer->backward_Matrix->getMatrixQL().block(j * 100, 0, 100, 10);

				for (auto k = NetQL<double>::layerQLVector.begin(); k != NetQL<double>::layerQLVector.end(); k++)
				{
					(*k)->calForward();
				}

				for (auto k = NetQL<double>::layerQLVector.rbegin(); k != NetQL<double>::layerQLVector.rend(); k++)
				{
					(*k)->calBackward();
					(*k)->upMatrix_batch();
				}
			}
		}

		double numTotal = 0;
		for (int i = 0; i < 10000; i++)
		{
			inLayer_01->forward_Matrix->setMatrixQL() = Test::input_Layer_T->forward_Matrix->getMatrixQL().row(i);
			inLayer_08->backward_Matrix->setMatrixQL() = Test::output_Layer_T->backward_Matrix->getMatrixQL().row(i);

			for (auto k = NetQL<double>::layerQLVector.begin(); k != NetQL<double>::layerQLVector.end(); k++)
			{
				(*k)->calForward();
			}

			int maxRow, maxColumn;

			lossLayer_01->left_Layer->forward_Matrix->getMatrixQL().maxCoeff(&maxRow, &maxColumn);

			int maxRow_T, maxColumn_T;
			lossLayer_01->right_Layer->backward_Matrix->getMatrixQL().maxCoeff(&maxRow_T, &maxColumn_T);

			if (maxColumn == maxColumn_T)
			{
				numTotal++;
			}

		}
		std::cout << numTotal / 10000.00 << std::endl;

		//	训练和测试运行时间
		DWORD star_time = GetTickCount();

		std::cout << "这个程序加载时间为：" << (star_time - load_time) << "ms." << std::endl;
	}
}