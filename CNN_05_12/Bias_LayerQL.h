#pragma once
#include "LayerQL.h"
namespace tinyDNN
{
	template <typename Dtype> 
	class Bias_LayerQL : public LayerQL<Dtype>
	{
	public:
		Bias_LayerQL( LayerType type, int rowNum, int colNum, Dtype ranNum);
		~Bias_LayerQL() override final;

		void calForward(int type = 0) const override final;
		void calBackward(int type = 0) override final;
		void upMatrix() override final;
		void upMatrix_batch(Dtype upRate) override final;
	protected:
		std::unique_ptr<MatrixQL<Dtype>> b_MatrixQL;
	};
	//*******************************************************************************************************************************
	template <typename Dtype>
	Bias_LayerQL<Dtype>::Bias_LayerQL(LayerType type, int rowNum, int colNum, Dtype ranNum) : LayerQL(type)
	{
		std::cout << "Bias_LayerQL Start!" << std::endl;
		this->b_MatrixQL = std::make_unique<MatrixQL<Dtype>>( rowNum, colNum);
		//这里后期需要改
		this->b_MatrixQL->setMatrixQL().setConstant(ranNum);
		//this->b_MatrixQL->setMatrixQL().setZero();
	}

	template <typename Dtype>
	Bias_LayerQL<Dtype>::~Bias_LayerQL()
	{
		std::cout << "Bias_LayerQL Over!" << std::endl;
	}

	template <typename Dtype>
	void Bias_LayerQL<Dtype>::calForward(int type = 0) const
	{
		//std::cout << this->b_MatrixQL->getMatrixQL() << std::endl;

		//这里默认b只有一行，结果这里会对b做一个扩展，对每一行都加上一个b
		int rowNum = this->left_Layer->forward_Matrix->getMatrixQL().rows();
		std::unique_ptr<MatrixQL<Dtype>> oMatrix = std::make_unique<MatrixQL<Dtype>>(rowNum, 1);
		oMatrix->setMatrixQL().setOnes();

		this->right_Layer->forward_Matrix->setMatrixQL() = this->left_Layer->forward_Matrix->getMatrixQL() + (oMatrix->getMatrixQL())*(this->b_MatrixQL->getMatrixQL());

		//==========================================================================================

		////这个是单行输入的所用命令,当采用SGD的时候这个会快一些
		//this->right_Layer->forward_Matrix->setMatrixQL() = this->left_Layer->forward_Matrix->getMatrixQL() + (this->b_MatrixQL->getMatrixQL());

	}

	template <typename Dtype>
	void Bias_LayerQL<Dtype>::calBackward(int type = 0)
	{
		//std::cout << this->b_MatrixQL->getMatrixQL() << std::endl;

		//反向传播，直接过去就好
		this->left_Layer->backward_Matrix->setMatrixQL() = this->right_Layer->backward_Matrix->getMatrixQL();
	}

	template <typename Dtype>
	void Bias_LayerQL<Dtype>::upMatrix()
	{
		//std::cout << this->right_Layer->backward_Matrix->getMatrixQL() << std::endl;

		//偏置更新，这里学习率设置为0.5
		this->b_MatrixQL->setMatrixQL() = this->b_MatrixQL->getMatrixQL() - 0.5 * (this->right_Layer->backward_Matrix->getMatrixQL());

		//std::cout << this->b_MatrixQL->getMatrixQL() << std::endl;
	}

	template <typename Dtype>
	void Bias_LayerQL<Dtype>::upMatrix_batch(Dtype upRate)
	{
		//std::cout << this->b_MatrixQL->getMatrixQL() << std::endl;

		//偏置更新，这里需要对反向的按行求和然后求均值
		this->b_MatrixQL->setMatrixQL() = this->b_MatrixQL->getMatrixQL() - 0.1 * this->right_Layer->backward_Matrix->getMatrixQL().colwise().sum();

		//std::cout << this->b_MatrixQL->getMatrixQL() << std::endl;

	}
}