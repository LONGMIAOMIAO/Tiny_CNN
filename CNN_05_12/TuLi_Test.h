#pragma once

namespace tinyDNN
{
	void tuli_Conv_1()
	{
		LoadTuLi::load_Tuli();
		LoadTuLi::load_Tuli_T();

		std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(128, 128);

		std::shared_ptr<LayerQL<double>> pool_01 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 64, 64);	//池化层
		std::shared_ptr<Inter_LayerQL<double>> o_01 = in_01 + pool_01;


		//						类型			卷积核数			行数			列数				卷积核宽度		卷积核几片		扩充宽度	
		std::shared_ptr<LayerQL<double>> conv_01 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 32, 64, 64, 7, 1, 3);	//卷积层
		std::shared_ptr<Inter_LayerQL<double>> o_02 = o_01 + conv_01;

		std::shared_ptr<LayerQL<double>> rule_01 = std::make_shared<Relu_LayerQL<double>>(Relu_Conv_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_03 = o_02 + rule_01;

		std::shared_ptr<LayerQL<double>> pool_02 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 32, 32);	//池化层
		std::shared_ptr<Inter_LayerQL<double>> o_04 = o_03 + pool_02;

		//						类型			卷积核数			行数			列数				卷积核宽度		卷积核几片		扩充宽度	
		std::shared_ptr<LayerQL<double>> conv_02 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 32, 32, 32, 5, 32, 2);	//卷积层
		std::shared_ptr<Inter_LayerQL<double>> o_05 = o_04 + conv_02;

		std::shared_ptr<LayerQL<double>> rule_02 = std::make_shared<Relu_LayerQL<double>>(Relu_Conv_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_06 = o_05 + rule_02;

		std::shared_ptr<LayerQL<double>> pool_03 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 16, 16);	//池化层
		std::shared_ptr<Inter_LayerQL<double>> o_07 = o_06 + pool_03;

		//						类型			卷积核数			行数			列数				卷积核宽度		卷积核几片		扩充宽度	
		std::shared_ptr<LayerQL<double>> conv_03 = std::make_shared<Conv_LayerQL<double>>(Conv_Layer, 16, 16, 16, 3, 32, 1);	//卷积层
		std::shared_ptr<Inter_LayerQL<double>> o_08 = o_07 + conv_03;

		std::shared_ptr<LayerQL<double>> rule_03 = std::make_shared<Relu_LayerQL<double>>(Relu_Conv_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_09 = o_08 + rule_03;

		std::shared_ptr<LayerQL<double>> pool_04 = std::make_shared<PooLayerQL<double>>(Pool_Layer, 8, 8);	//池化层
		std::shared_ptr<Inter_LayerQL<double>> o_10 = o_09 + pool_04;

		std::shared_ptr<LayerQL<double>> dim_reduce_01 = std::make_shared<Dim_ReduceQL<double>>(Dim_Reduce_Layer, 16, 8, 8);	//降维层
		std::shared_ptr<Inter_LayerQL<double>> o_11 = o_10 + dim_reduce_01;

		std::shared_ptr<LayerQL<double>> fullconnect_01 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 16 * 8 * 8, 30);//全连接层
		std::shared_ptr<Inter_LayerQL<double>> o_12 = o_11 + fullconnect_01;

		std::shared_ptr<LayerQL<double>> rule_04 = std::make_shared<Relu_LayerQL<double>>(Relu_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_13 = o_12 + rule_04;

		std::shared_ptr<LayerQL<double>> fullconnect_02 = std::make_shared<Fullconnect_LayerQL<double>>(Fullconnect_Layer, 30, 3);//全连接层
		std::shared_ptr<Inter_LayerQL<double>> o_14 = o_13 + fullconnect_02;

		std::shared_ptr<LayerQL<double>> rule_05 = std::make_shared<Relu_LayerQL<double>>(Relu_Layer);	//Relu层
		std::shared_ptr<Inter_LayerQL<double>> o_15 = o_14 + rule_05;

		std::shared_ptr<LayerQL<double>> lossLayer_01 = std::make_shared<SoftMax_LayerQL<double>>(SoftMax_Layer);//Loss层
		std::shared_ptr<Inter_LayerQL<double>> o_16 = o_15 + lossLayer_01;

		rule_01->pRelu_k = 0.12;
		rule_02->pRelu_k = 0.12;
		rule_03->pRelu_k = 0.12;
		rule_04->pRelu_k = 0.12;
		rule_05->pRelu_k = 0.12;

		conv_01->upConv = 0.005;
		conv_02->upConv = 0.005;
		conv_03->upConv = 0.005;
		fullconnect_01->upFull = 0.005;
		fullconnect_02->upFull = 0.005;

		for ( int i = 0; i < 50; i++ )
		{
			std::cout << i << std::endl;
			for ( int j = 1; j < 24; j++ )
			{
				in_01->forward_Matrix_Vector.clear();
				in_01->forward_Matrix_Vector.push_back(LoadTuLi::tuli_Train[j-1]);

				switch (j)
				{
				case 1:
				case 4:
				case 7:
				case 10:
				case 13:
				case 16:
				case 19:
				case 22:
					
					o_16->backward_Matrix->setMatrixQL().resize(1,3);
					o_16->backward_Matrix->setMatrixQL().setZero();
					o_16->backward_Matrix->setMatrixQL()(0, 0) = 1;
					break;

				case 2:
				case 5:
				case 8:
				case 11:
				case 14:
				case 17:
				case 20:
				case 23:

					o_16->backward_Matrix->setMatrixQL().resize(1,3);
					o_16->backward_Matrix->setMatrixQL().setZero();
					o_16->backward_Matrix->setMatrixQL()(0, 1) = 1;
					break;

				case 3:
				case 6:
				case 9:
				case 12:
				case 15:
				case 18:
				case 21:

					o_16->backward_Matrix->setMatrixQL().resize(1, 3);
					o_16->backward_Matrix->setMatrixQL().setZero();
					o_16->backward_Matrix->setMatrixQL()(0, 2) = 1;
					break;

				default:
					break;
				}

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

				if (i > 48)
				{
					std::cout << "i::" << o_16->forward_Matrix->getMatrixQL() << std::endl;
				}
			}
		}

		for (int j = 1; j < 10; j++)
		{
			in_01->forward_Matrix_Vector.clear();
			in_01->forward_Matrix_Vector.push_back(LoadTuLi::tuli_Test[j - 1]);
		
			for (auto k = NetQL<double>::layerQLVector.begin(); k != NetQL<double>::layerQLVector.end(); k++)
			{
				(*k)->calForward();
			}
			std::cout << "j::" << o_16->forward_Matrix->getMatrixQL() << std::endl;
		}
	}
}