#include "Test.h"

namespace tinyDNN
{
	Test::Test()
	{
		//Full_Layer ForwardTrans
		//this->Fullconnect_Layer_Forward_Test();
		////Full_Layer BackwardTrans
		//this->Fullconnect_Layer_Backward_Test();
		////Bias_Layer ForwardTrans
		//this->Bias_Layer_Forward_Test();
		////Bias_Layer BackwardTrans
		//this->Bias_Layer_Backward_Test();
		//Operator_Layer
		//this->Operator_Test();
		//=========================================================================================
		//Sigmoid_LayerQL ForwardTrans
		//this->Sigmoid_LayerQL_Forward_Test();
		//Sigmoid_LayerQL BackwardTrans
		//this->Sigmoid_LayerQL_Backward_Test();
		//MSE_LOSS_BACKWARD_TEST()
		this->MSE_Loss_LayerQL_Backward_Test();

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

	void Test::Bias_Layer_Forward_Test()
	{
		std::shared_ptr<Inter_LayerQL<double>> left_Layer = std::make_shared<Inter_LayerQL<double>>(2,3);
		left_Layer->forward_Matrix->setMatrixQL().setConstant(1.2);
		std::cout << left_Layer->forward_Matrix->getMatrixQL() << std::endl;

		std::shared_ptr<LayerQL<double>> bias_Layer = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 2, 3);
		std::shared_ptr<Inter_LayerQL<double>> out_01 = left_Layer + bias_Layer;

		bias_Layer->calForward();

		std::cout << bias_Layer->left_Layer->forward_Matrix->getMatrixQL() << std::endl;
		std::cout << bias_Layer->right_Layer->forward_Matrix->getMatrixQL() << std::endl;
		//*******************************************************************************************************//

		std::shared_ptr<LayerQL<double>> bias_Layer_02 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 2, 3);
		std::shared_ptr<Inter_LayerQL<double>> out_02 = out_01 + bias_Layer_02;

		bias_Layer_02->calForward();

		std::cout << bias_Layer_02->left_Layer->forward_Matrix->getMatrixQL() << std::endl;
		std::cout << bias_Layer_02->right_Layer->forward_Matrix->getMatrixQL() << std::endl;

		//*******************************************************************************************************//

		std::shared_ptr<LayerQL<double>> bias_Layer_03 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer, 2, 3);
		std::shared_ptr<Inter_LayerQL<double>> out_03 = out_02 + bias_Layer_03;

		bias_Layer_03->calForward();

		std::cout << bias_Layer_03->left_Layer->forward_Matrix->getMatrixQL() << std::endl;
		std::cout << bias_Layer_03->right_Layer->forward_Matrix->getMatrixQL() << std::endl;

	}

	void Test::Bias_Layer_Backward_Test()
	{
		std::shared_ptr<Inter_LayerQL<double>> InLayer_01 = std::make_shared<Inter_LayerQL<double>>();
		std::shared_ptr<LayerQL<double>> biasLayer_01 = std::make_shared<Bias_LayerQL<double>>(Bias_Layer,3, 5);
		std::shared_ptr<Inter_LayerQL<double>> InLayer_02 = InLayer_01 + biasLayer_01;
		biasLayer_01->right_Layer->backward_Matrix->setMatrixQL().resize(3, 5);
		biasLayer_01->right_Layer->backward_Matrix->setMatrixQL().setOnes();
		std::cout << biasLayer_01->right_Layer->backward_Matrix->getMatrixQL() << std::endl;

		biasLayer_01->calBackward();

		std::cout << biasLayer_01->left_Layer->backward_Matrix->getMatrixQL() << std::endl;

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
}