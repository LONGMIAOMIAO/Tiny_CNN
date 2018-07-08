#include "Test.h"

namespace tinyDNN
{
	////加载MNIST数据集，训练集，55000个
	//std::shared_ptr<Inter_LayerQL<double>> Test::input_Layer = std::make_shared<Inter_LayerQL<double>>(55000, 784);
	//std::shared_ptr<Inter_LayerQL<double>> Test::output_Layer = std::make_shared<Inter_LayerQL<double>>(55000, 10);
	////加载MNIST数据集，测试集，10000个
	//std::shared_ptr<Inter_LayerQL<double>> Test::input_Layer_T = std::make_shared<Inter_LayerQL<double>>(10000, 784);
	//std::shared_ptr<Inter_LayerQL<double>> Test::output_Layer_T = std::make_shared<Inter_LayerQL<double>>(10000, 10);

	Test::Test()
	{
		//Full_Layer ForwardTrans	测试全连接层的向前传播
		//this->Fullconnect_Layer_Forward_Test();
		////Full_Layer BackwardTrans	测试全连接层的向后传播
		//this->Fullconnect_Layer_Backward_Test();
		//Full_Layer_update		测试全连接层的权重更新，这里用的是SGD，每次来一个新数据更新一次
		//this->Fullconnect_Layer_Update_Test();
		//Full_Layer_update_batch	测试全连接层的权重更新，这里用的是MBGD，按照BATCH来更新
		//this->Fullconnect_Layer_Update_Batch_Test();

		//======================================================================================

		////Bias_Layer ForwardTrans		测试Bias层的向前传播
		//this->Bias_Layer_Forward_Test();
		////Bias_Layer BackwardTrans	测试Bias层的向后传播
		//this->Bias_Layer_Backward_Test();
		////Bias_Layer update	测试bias层的b更新，采用SGD
		//this->Bias_Layer_Update_Test();
		////Bias_Layer_Batch update		测试bias层的b更新，采用MBGD
		//this->Bias_Layer_Update_Batch_Test();

		//======================================================================================

		//Operator_Layer	测试友元运算符重载 + ，将中间层和权重等层连接起来
		//this->Operator_Test();

		//======================================================================================

		//Sigmoid_LayerQL ForwardTrans	测试Sigmoid激活函数的向前传播
		//this->Sigmoid_LayerQL_Forward_Test();
		//Sigmoid_LayerQL BackwardTrans	测试Sigmoid激活函数的反向传播
		//this->Sigmoid_LayerQL_Backward_Test();

		//======================================================================================

		//MSE_LOSS_BACKWARD_TEST()	测试Loss层的反向传播
		//this->MSE_Loss_LayerQL_Backward_Test();

		//======================================================================================
		////测试SGD
		this->Mnist_Test();
		////测试批量下降 100一组
		//this->Mnist_Test_02();
		//测试批量下架，10一组
		//this->Mnist_Test_03();

		//======================================================================================
		////测试加载卷积二维图像,训练集
		//LoadCSV::loadCSVTrain();
		//LoadCSV::loadCSV_Train_Vector();
		////测试加载卷积二维图像，测试集
		//LoadCSV::loadCSVTest();
		//LoadCSV::loadCSV_Test_Vector();
		
		//======================================================================================
		////测试Mnist卷积计算
		//this->Mnist_Test_Conv();


	}

	Test::~Test(){}
	
	//============================================================================

	//测试全连接层的前向传播
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
	//测试全连接层的反向传播
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
	//测试全连接层的权重更新，这里用的是SGD，每次来一个新数据更新一次
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
	//测试全连接层的权重更新，这里用的是MBGD，按照BATCH来更新
	void Test::Fullconnect_Layer_Update_Batch_Test()
	{
		std::shared_ptr<Inter_LayerQL<double>> left_Layer = std::make_shared<Inter_LayerQL<double>>(3, 5);
		left_Layer->forward_Matrix->setMatrixQL() << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11 ,12, 13, 14, 15;

		std::shared_ptr<LayerQL<double>> fullLayerTest = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 5, 10);

		std::shared_ptr<Inter_LayerQL<double>> right_Layer = left_Layer + fullLayerTest;

		right_Layer->backward_Matrix->setMatrixQL().resize(3, 10);
		right_Layer->backward_Matrix->setMatrixQL().setConstant(1);

		fullLayerTest->upMatrix_batch(0.35);
	}

	//============================================================================

	//测试Bias层的向前传播
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
	//测试Bias层的向后传播
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
	//测试bias层的b更新，采用SGD
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
	//测试bias层的b更新，采用MBGD
	void Test::Bias_Layer_Update_Batch_Test()
	{
		std::shared_ptr<Inter_LayerQL<double>> left_Layer = std::make_shared<Inter_LayerQL<double>>(3, 5);
		left_Layer->forward_Matrix->setMatrixQL().setConstant(1.2);

		std::shared_ptr<LayerQL<double>> bias_Layer = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 1, 5, 0.1);
		std::shared_ptr<Inter_LayerQL<double>> right_01 = left_Layer + bias_Layer;

		right_01->backward_Matrix->setMatrixQL().resize(3, 5);
		right_01->backward_Matrix->setMatrixQL().setConstant(10);

		bias_Layer->upMatrix_batch(0.35);
	}

	//============================================================================

	//测试Sigmoid层的前向传播
	void Test::Sigmoid_LayerQL_Forward_Test()
	{
		std::shared_ptr<Inter_LayerQL<double>> intputLayer_01 = std::make_shared<Inter_LayerQL<double>>(2,5);
		intputLayer_01->forward_Matrix->setMatrixQL().setOnes();

		std::shared_ptr<LayerQL<double>> sigmoidLayer_01 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);
		std::shared_ptr<Inter_LayerQL<double>> inputLayer_02 = intputLayer_01 + sigmoidLayer_01;

		sigmoidLayer_01->calForward();

		std::cout << inputLayer_02->forward_Matrix->getMatrixQL() << std::endl;

	}
	//测试Sigmoid层的反向传播
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

	//============================================================================

	//测试最小二乘损失函数
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
	
	//============================================================================

	//测试操作运算符
	void Test::Operator_Test(){}

	//============================================================================
	
	void Test::Mnist_Test()
	{
		//加载数据
		LoadCSV::loadCSVTrain();
		LoadCSV::loadCSVTest();

		//制作输入层，1行784列
		std::shared_ptr<Inter_LayerQL<double>> inLayer_01 = std::make_shared<Inter_LayerQL<double>>(1, 784);
		//制作第一个全连接层，784行100列
		std::shared_ptr<LayerQL<double>> fullLayer_01 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 784, 100);
		//合并前两层
		std::shared_ptr<Inter_LayerQL<double>> inLayer_02 = inLayer_01 + fullLayer_01;
		//制作第一个Bias层，1行100列

		std::shared_ptr<LayerQL<double>> biasLayer_01 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 1, 100, 0.1);
		//合并前两层
		std::shared_ptr<Inter_LayerQL<double>> inLayer_03 = inLayer_02 + biasLayer_01;

		//制作第一个sigmoid层
		std::shared_ptr<LayerQL<double>> sigmoidLayer_01 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);
		//合并前两层
		std::shared_ptr<Inter_LayerQL<double>> inLayer_04 = inLayer_03 + sigmoidLayer_01;

		//制作第二个全连接层
		std::shared_ptr<LayerQL<double>> fullLayer_02 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 100, 10);
		//合并前两层
		std::shared_ptr<Inter_LayerQL<double>> inLayer_05 = inLayer_04 + fullLayer_02;
		//制作第二个BIAS层
		std::shared_ptr<LayerQL<double>> biasLayer_02 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 1, 10, 0.1);
		//合并前两层
		std::shared_ptr<Inter_LayerQL<double>> inLayer_06 = inLayer_05 + biasLayer_02;
		//制作第二个SIGMOID层
		std::shared_ptr<LayerQL<double>> sigmoidLayer_02 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);
		//合并前两层
		std::shared_ptr<Inter_LayerQL<double>> inLayer_07 = inLayer_06 + sigmoidLayer_02;

		//制作Loss层
		std::shared_ptr<LayerQL<double>> lossLayer_01 = std::make_shared<MSE_Loss_LayerQL<double>>(MSE_Loss_Layer);
		//输出层
		std::shared_ptr<Inter_LayerQL<double>> inLayer_08 = inLayer_07 + lossLayer_01;

		//	 程序加载初始时间
		DWORD load_time = GetTickCount();

		for (int i = 0; i < 2 ; i ++)
		{
			for (int j = 0; j < 55000; j++)
			{
				//按次序将数据加载进输入层
				inLayer_01->forward_Matrix->setMatrixQL() = LoadCSV::input_Layer->forward_Matrix->getMatrixQL().row(j);
				//按次序将数据加载进输出层
				inLayer_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer->backward_Matrix->getMatrixQL().row(j);
				//从头开始进行前向传播
				for (auto k = NetQL<double>::layerQLVector.begin(); k != NetQL<double>::layerQLVector.end(); k++)
				{
					(*k)->calForward();
				}
				//从头开始反向传播 + 权重更新
				for (auto k = NetQL<double>::layerQLVector.rbegin(); k != NetQL<double>::layerQLVector.rend(); k++)
				{
					(*k)->calBackward();
					(*k)->upMatrix();
				}
			}
		}

		//对正确的数据进行计数
		double numTotal = 0;
		//从第一个开始测试测试集
		for ( int i = 0; i <10000; i ++ )
		{
			//加载输入层和输出层
			inLayer_01->forward_Matrix->setMatrixQL() = LoadCSV::input_Layer_T->forward_Matrix->getMatrixQL().row(i);
			inLayer_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer_T->backward_Matrix->getMatrixQL().row(i);
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

	void Test::Mnist_Test_02()
	{
		//加载数据
		LoadCSV::loadCSVTrain();
		LoadCSV::loadCSVTest();

		//第一个大层，每次训练100行，784列
		std::shared_ptr<Inter_LayerQL<double>> inLayer_01 = std::make_shared<Inter_LayerQL<double>>(100, 784);
		std::shared_ptr<LayerQL<double>> fullLayer_01 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 784, 100);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_02 = inLayer_01 + fullLayer_01;
		std::shared_ptr<LayerQL<double>> biasLayer_01 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 1, 100, 0.1);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_03 = inLayer_02 + biasLayer_01;
		std::shared_ptr<LayerQL<double>> sigmoidLayer_01 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_04 = inLayer_03 + sigmoidLayer_01;

		//第二个大层，每次训练100行，10列
		std::shared_ptr<LayerQL<double>> fullLayer_02 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 100, 10);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_05 = inLayer_04 + fullLayer_02;
		std::shared_ptr<LayerQL<double>> biasLayer_02 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 1, 10, 0.1);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_06 = inLayer_05 + biasLayer_02;
		std::shared_ptr<LayerQL<double>> sigmoidLayer_02 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_07 = inLayer_06 + sigmoidLayer_02;

		//Loss层和输出层
		std::shared_ptr<LayerQL<double>> lossLayer_01 = std::make_shared<MSE_Loss_LayerQL<double>>(MSE_Loss_Layer);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_08 = inLayer_07 + lossLayer_01;

		//程序加载初始时间
		DWORD load_time = GetTickCount();

		//从头开始遍历
		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 550; j++)
			{
				//每次取100个数据加载
				inLayer_01->forward_Matrix->setMatrixQL() = LoadCSV::input_Layer->forward_Matrix->getMatrixQL().block(j*100, 0, 100, 784);
				inLayer_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer->backward_Matrix->getMatrixQL().block(j * 100, 0, 100, 10);
				//前向传播
				for (auto k = NetQL<double>::layerQLVector.begin(); k != NetQL<double>::layerQLVector.end(); k++)
				{
					(*k)->calForward();
				}
				//反向传播
				for (auto k = NetQL<double>::layerQLVector.rbegin(); k != NetQL<double>::layerQLVector.rend(); k++)
				{
					(*k)->calBackward();
					(*k)->upMatrix_batch(0.0035);
				}
			}
		}
		//测试正确率
		double numTotal = 0;
		for (int i = 0; i < 10000; i++)
		{
			//进行测试，每次测试一个
			inLayer_01->forward_Matrix->setMatrixQL() = LoadCSV::input_Layer_T->forward_Matrix->getMatrixQL().row(i);
			inLayer_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer_T->backward_Matrix->getMatrixQL().row(i);
			//前向传播
			for (auto k = NetQL<double>::layerQLVector.begin(); k != NetQL<double>::layerQLVector.end(); k++)
			{
				(*k)->calForward();
			}

			int maxRow, maxColumn;
			lossLayer_01->left_Layer->forward_Matrix->getMatrixQL().maxCoeff(&maxRow, &maxColumn);

			int maxRow_T, maxColumn_T;
			lossLayer_01->right_Layer->backward_Matrix->getMatrixQL().maxCoeff(&maxRow_T, &maxColumn_T);
			//计算正确率
			if (maxColumn == maxColumn_T)
			{
				numTotal++;
			}
		}
		//计算正确率
		std::cout << numTotal / 10000.00 << std::endl;
		//训练和测试运行时间
		DWORD star_time = GetTickCount();
		//计算运行时间
		std::cout << "这个程序加载时间为：" << (star_time - load_time) << "ms." << std::endl;
	}

	void Test::Mnist_Test_03()
	{
		//加载数据
		LoadCSV::loadCSVTrain();
		LoadCSV::loadCSVTest();
		//输入层 10 行 784列
		std::shared_ptr<Inter_LayerQL<double>> inLayer_01 = std::make_shared<Inter_LayerQL<double>>(10, 784);
		//第一大层
		std::shared_ptr<LayerQL<double>> fullLayer_01 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 784, 100);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_02 = inLayer_01 + fullLayer_01;
		std::shared_ptr<LayerQL<double>> biasLayer_01 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 1, 100, 0.1);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_03 = inLayer_02 + biasLayer_01;
		std::shared_ptr<LayerQL<double>> sigmoidLayer_01 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_04 = inLayer_03 + sigmoidLayer_01;
		//第二大层
		std::shared_ptr<LayerQL<double>> fullLayer_02 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 100, 10);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_05 = inLayer_04 + fullLayer_02;
		std::shared_ptr<LayerQL<double>> biasLayer_02 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 1, 10, 0.1);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_06 = inLayer_05 + biasLayer_02;
		std::shared_ptr<LayerQL<double>> sigmoidLayer_02 = std::make_shared<Sigmoid_LayerQL<double>>(Sigmoid_Layer);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_07 = inLayer_06 + sigmoidLayer_02;
		//Loss层
		std::shared_ptr<LayerQL<double>> lossLayer_01 = std::make_shared<MSE_Loss_LayerQL<double>>(MSE_Loss_Layer);
		std::shared_ptr<Inter_LayerQL<double>> inLayer_08 = inLayer_07 + lossLayer_01;

		//程序加载初始时间
		DWORD load_time = GetTickCount();
		//训练开始
		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 5500; j++)
			{
				//每次训练10个
				inLayer_01->forward_Matrix->setMatrixQL() = LoadCSV::input_Layer->forward_Matrix->getMatrixQL().block(j * 10, 0, 10, 784);
				inLayer_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer->backward_Matrix->getMatrixQL().block(j * 10, 0, 10, 10);
				//前向传播
				for (auto k = NetQL<double>::layerQLVector.begin(); k != NetQL<double>::layerQLVector.end(); k++)
				{
					(*k)->calForward();
				}
				//反向传播
				for (auto k = NetQL<double>::layerQLVector.rbegin(); k != NetQL<double>::layerQLVector.rend(); k++)
				{
					(*k)->calBackward();
					(*k)->upMatrix_batch(0.35);
				}
			}
		}
		//计算正确率
		double numTotal = 0;
		for (int i = 0; i < 10000; i++)
		{
			//测试开始
			inLayer_01->forward_Matrix->setMatrixQL() = LoadCSV::input_Layer_T->forward_Matrix->getMatrixQL().row(i);
			inLayer_08->backward_Matrix->setMatrixQL() = LoadCSV::output_Layer_T->backward_Matrix->getMatrixQL().row(i);
			//前向传播
			for (auto k = NetQL<double>::layerQLVector.begin(); k != NetQL<double>::layerQLVector.end(); k++)
			{
				(*k)->calForward();
			}
			
			int maxRow, maxColumn;
			lossLayer_01->left_Layer->forward_Matrix->getMatrixQL().maxCoeff(&maxRow, &maxColumn);

			int maxRow_T, maxColumn_T;
			lossLayer_01->right_Layer->backward_Matrix->getMatrixQL().maxCoeff(&maxRow_T, &maxColumn_T);
			//正确率
			if (maxColumn == maxColumn_T)
			{
				numTotal++;
			}
		}
		//计算正确率
		std::cout << numTotal / 10000.00 << std::endl;
		//	训练和测试运行时间
		DWORD star_time = GetTickCount();
		//输出运行时间
		std::cout << "这个程序加载时间为：" << (star_time - load_time) << "ms." << std::endl;
	}



	void Test::Mnist_Test_Conv()
	{
		//测试加载卷积二维图像,训练集
		LoadCSV::loadCSVTrain();
		LoadCSV::loadCSV_Train_Vector();

		//制作输入层，1行784列
		std::shared_ptr<Inter_LayerQL<double>> inLayer_01 = std::make_shared<Inter_LayerQL<double>>(28, 28);

		for (int i = 0; i < 1; i++)
		{
			for (int j = 999; j < 1000; j++)
			{
				inLayer_01->forward_Matrix = LoadCSV::conv_Input_Vector[j];

				std::cout << (inLayer_01->forward_Matrix->getMatrixQL()*9).cast<int>() << std::endl;
			}
		}

		std::shared_ptr<LayerQL<double>> poolLayer_01 = std::make_shared<PooLayerQL<double>>(Pool_Layer,14,14);
		//这里可以尝试重载赋值运算符
		std::shared_ptr<Inter_LayerQL<double>> out_01 = inLayer_01 + poolLayer_01;
		poolLayer_01->right_Layer->forward_Matrix->setMatrixQL().resize(14, 14);

		poolLayer_01->calForward();

		std::cout << (out_01->forward_Matrix->getMatrixQL() * 9).cast<int>() << std::endl;

		//============================================================================================================

		poolLayer_01->left_Layer->backward_Matrix->setMatrixQL().resize(28, 28);

		poolLayer_01->right_Layer->backward_Matrix = poolLayer_01->right_Layer->forward_Matrix;

		poolLayer_01->calBackward();

		std::cout << (inLayer_01->backward_Matrix->getMatrixQL() * 9).cast<int>() << std::endl;

		//============================================================================================================

		std::cout << "=======================================================================================" << std::endl;
		//测试加载卷积二维图像,训练集
		LoadCSV::loadCSVTest();
		LoadCSV::loadCSV_Test_Vector();

		//制作输入层，1行784列
		std::shared_ptr<Inter_LayerQL<double>> inLayer_01_T = std::make_shared<Inter_LayerQL<double>>(28, 28);

		for (int i = 0; i < 1; i++)
		{
			for (int j = 999; j < 1000; j++)
			{
				inLayer_01_T->forward_Matrix = LoadCSV::conv_Input_Vector_T[j];

				std::cout << (inLayer_01_T->forward_Matrix->getMatrixQL() * 9).cast<int>() << std::endl;
			}
		}
		//===========================================================================================================
	}
}