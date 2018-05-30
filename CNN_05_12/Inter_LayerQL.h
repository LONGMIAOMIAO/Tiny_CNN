#pragma once
#include "MatrixQL.h"
#include "memory"
#include <iostream>

namespace tinyDNN
{
	template <typename Dtype>
	class LayerQL;

	template <typename Dtype>
	class Inter_LayerQL
	{
	public:
		friend class Test;
		
		template <typename Dtype> 
		friend std::shared_ptr<Inter_LayerQL<Dtype>> operator+(std::shared_ptr<Inter_LayerQL<Dtype>>& operLeft, std::shared_ptr<LayerQL<Dtype>>& operRight);

		explicit Inter_LayerQL(int rowNum = 0, int colNum = 0);
		~Inter_LayerQL();

	public:
		std::unique_ptr<MatrixQL<Dtype>> forward_Matrix;
		std::unique_ptr<MatrixQL<Dtype>> backward_Matrix;
	};

	template <typename Dtype>
	Inter_LayerQL<Dtype>::Inter_LayerQL(int rowNum = 0, int colNum = 0)
	{
		std::cout << "Inter_Layer Start!" << std::endl;
		this->forward_Matrix = std::make_unique<MatrixQL<Dtype>>(rowNum, colNum);
		this->backward_Matrix = std::make_unique<MatrixQL<Dtype>>(rowNum, colNum);
	}

	template <typename Dtype>
	Inter_LayerQL<Dtype>::~Inter_LayerQL()
	{
		std::cout << "Inter_Layer End!" << std::endl;
	}
}
