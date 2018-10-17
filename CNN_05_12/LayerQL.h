#pragma once
#include "MatrixQL.h"
#include "memory"
#include <iostream>
#include "Inter_LayerQL.h"
namespace tinyDNN 
{
	enum LayerType
	{
		Fullconnect_Layer = 1,
		Bias_Layer = 2,
		Sigmoid_Layer = 3,
		Sigmoid_Conv_Layer = 31,
		Relu_Layer = 9,
		Relu_Conv_Layer = 91,
		MSE_Loss_Layer = 4,
		SoftMax_Layer = 41,
		Pool_Layer = 5,
		Conv_Layer = 6,
		Padding_Layer = 7,
		Dim_Reduce_Layer = 8,
		Data_Augmentation_Layer = 10,
		Bias_Conv_Layer_L = 11
	};

	template <typename Dtype>
	class LayerQL
	{
	public:
		explicit LayerQL( LayerType type ) ;
		virtual ~LayerQL();

		virtual void calForward(int type = 0) const = 0;
		virtual void calBackward(int type = 0) = 0;
		virtual void upMatrix() = 0;
		
		virtual void upMatrix_batch(Dtype upRate) = 0;
		
		friend class Test;
		friend class Mnist_Conv_Test;
		friend void mnist_Conv_T_1();
		friend void mnist_Conv_T_2();
		friend void rightValue(std::shared_ptr<Inter_LayerQL<double>> inLayer, std::shared_ptr<Inter_LayerQL<double>> endLayer, std::shared_ptr<LayerQL<double>> lossLayer);
		//friend void Cifar10_T2_1();

		template <typename Dtype> 
		friend std::shared_ptr<Inter_LayerQL<Dtype>> operator+( std::shared_ptr<Inter_LayerQL<Dtype>>& operLeft,  std::shared_ptr<LayerQL<Dtype>>& operRight);

		double upFull;
		double upConv;
		double pRelu_k;

	protected:
		LayerType layerType;
		std::shared_ptr<Inter_LayerQL<Dtype>> left_Layer;
		std::shared_ptr<Inter_LayerQL<Dtype>> right_Layer;
	};

	//	友元函数重载运算符，用于装载各个Layer
	template <typename Dtype>
	std::shared_ptr<Inter_LayerQL<Dtype>> operator+( std::shared_ptr<Inter_LayerQL<Dtype>>& operLeft,  std::shared_ptr<LayerQL<Dtype>>& operRight)
	{
		std::shared_ptr<Inter_LayerQL<Dtype>> test = std::make_shared<Inter_LayerQL<Dtype>>();

		operRight->left_Layer = operLeft;
		operRight->right_Layer = test;

		NetQL<Dtype>::layerQLVector.push_back(operRight);

		return test;
	}
	//*****************************************************************************************
	template <typename Dtype>
	LayerQL<Dtype>::LayerQL( LayerType type ) : layerType(type)
	{
		std::cout << "Layer Start!" << std::endl;
	}

	template <typename Dtype>
	LayerQL<Dtype>::~LayerQL()
	{
		std::cout << "Layer Over!" << std::endl;
	}
}

