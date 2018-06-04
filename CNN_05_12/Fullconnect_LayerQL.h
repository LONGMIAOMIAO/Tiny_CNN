#pragma once
#include "LayerQL.h"
#include <random>

namespace tinyDNN
{

	template <typename Dtype> 
	class Fullconnect_LayerQL : public LayerQL<Dtype>
	{
	public:
		friend class Test;
		explicit Fullconnect_LayerQL( LayerType type, int rowNum, int rolNum);
		~Fullconnect_LayerQL() override final;

		void calForward() const override final;
		void calBackward() override final;

		void upMatrix() override final;

	protected:
		std::unique_ptr<MatrixQL<Dtype>> w_MatrixQL;
	};

	//**********************************************************************************************
	template <typename Dtype>
	Fullconnect_LayerQL<Dtype>::Fullconnect_LayerQL(LayerType type, int rowNum, int colNum) : LayerQL(type)
	{
		std::cout << "Fullconnect_LayerQL Start!" << std::endl;
		this->w_MatrixQL = std::make_unique<MatrixQL<Dtype>>(rowNum, colNum);
		//这里后期需要改
		//this->w_MatrixQL->setMatrixQL().setConstant(2.2);
		//this->w_MatrixQL->setMatrixQL().setRandom();

		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<Dtype> normal(0, 0.1);
		for (int i = 0; i < rowNum; i++)
		{
			for (int j = 0; j < colNum; j++)
			{
				//下面随机数可以达到 95.59的正确率 
				//wData(i, j) = GaussRand()*0.1 + 0;
				//下面随机数可以达到 95.86的正确率
				this->w_MatrixQL->setMatrixQL()(i, j) = normal(gen);
			}
		}
	}

	template <typename Dtype>
	Fullconnect_LayerQL<Dtype>::~Fullconnect_LayerQL()
	{
		std::cout << "Fullconnect_LayerQL Over!" << std::endl;
	}

	template <typename Dtype>
	void Fullconnect_LayerQL<Dtype>::calForward() const
	{
		//std::cout << this->w_MatrixQL->getMatrixQL() << std::endl;

		this->right_Layer->forward_Matrix->setMatrixQL() = this->left_Layer->forward_Matrix->getMatrixQL() * this->w_MatrixQL->getMatrixQL();
	}

	template <typename Dtype>
	void Fullconnect_LayerQL<Dtype>::calBackward()
	{
		//std::cout << this->w_MatrixQL->getMatrixQL() << std::endl;

		this->left_Layer->backward_Matrix->setMatrixQL() = this->right_Layer->backward_Matrix->getMatrixQL() * this->w_MatrixQL->getMatrixQL().transpose();
	}

	template <typename Dtype>
	void Fullconnect_LayerQL<Dtype>::upMatrix()
	{
		this->w_MatrixQL->setMatrixQL() = this->w_MatrixQL->getMatrixQL() - 0.15 * (this->left_Layer->forward_Matrix->getMatrixQL().transpose() ) * ( this->right_Layer->backward_Matrix->getMatrixQL() );

		//std::cout << this->w_MatrixQL->getMatrixQL() << std::endl;
	}
}