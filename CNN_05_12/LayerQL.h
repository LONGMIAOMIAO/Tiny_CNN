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
		MSE_Loss_Layer = 4,
	};

	template <typename Dtype>
	class LayerQL
	{
	public:
		explicit LayerQL( LayerType type ) ;
		virtual ~LayerQL();

		virtual void calForward() const = 0;
		virtual void calBackward() = 0;
		virtual void upMatrix() = 0;
		
		virtual void upMatrix_batch(Dtype upRate) = 0;
		
		friend class Test;
		template <typename Dtype> 
		friend std::shared_ptr<Inter_LayerQL<Dtype>> operator+( std::shared_ptr<Inter_LayerQL<Dtype>>& operLeft,  std::shared_ptr<LayerQL<Dtype>>& operRight);

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

