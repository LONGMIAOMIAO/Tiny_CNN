#pragma once
#include "LayerQL.h"

namespace tinyDNN
{
	template <typename Dtype> 
	class Fullconnect_LayerQL : public LayerQL<Dtype>
	{
	public:
		explicit Fullconnect_LayerQL();
		~Fullconnect_LayerQL() final;

		void calForward( std::unique_ptr<MatrixQL<Dtype>>& matrixIn, std::unique_ptr<MatrixQL<Dtype>>& matrixOut) const override final;
		void calBackward(std::unique_ptr<MatrixQL<Dtype>>& matrixIn, std::unique_ptr<MatrixQL<Dtype>>& matrixOut) final;

	protected:
		std::unique_ptr<MatrixQL<Dtype>> wMatrixQL;
	};

	template <typename Dtype>
	Fullconnect_LayerQL<Dtype>::Fullconnect_LayerQL()
	{
		std::cout << "Fullconnect_LayerQL Start!" << std::endl;

		//std::unique_ptr<MatrixQL<Dtype>> iniMatrix(new MatrixQL<Dtype>(3, 3));
		//this->wMatrixQL = std::move(iniMatrix);

		this->wMatrixQL = std::make_unique<MatrixQL<Dtype>>(3, 3);			
		//this->wMatrixQL->setMatrixQL().setOnes();
		this->wMatrixQL->setMatrixQL().setConstant(2);
	}

	template <typename Dtype>
	Fullconnect_LayerQL<Dtype>::~Fullconnect_LayerQL()
	{
		std::cout << "Fullconnect_LayerQL Over!" << std::endl;
	}

	template <typename Dtype>
	void Fullconnect_LayerQL<Dtype>::calForward(std::unique_ptr<MatrixQL<Dtype>>& matrixIn, std::unique_ptr<MatrixQL<Dtype>>& matrixOut) const
	{
		matrixOut->setMatrixQL() = matrixIn->getMatrixQL() * this->wMatrixQL->getMatrixQL();

	}

	template <typename Dtype>
	void Fullconnect_LayerQL<Dtype>::calBackward(std::unique_ptr<MatrixQL<Dtype>>& matrixIn, std::unique_ptr<MatrixQL<Dtype>>& matrixOut)
	{


	}

}