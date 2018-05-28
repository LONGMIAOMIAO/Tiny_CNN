#pragma once
#include "LayerQL.h"

namespace tinyDNN
{


	template <typename Dtype> 
	class Fullconnect_LayerQL : public LayerQL<Dtype>
	{
	public:
		explicit Fullconnect_LayerQL( LayerType type, int rowNum, int rolNum);
		~Fullconnect_LayerQL() override final;

		void calForward( std::unique_ptr<MatrixQL<Dtype>>& feed_Left, std::unique_ptr<MatrixQL<Dtype>>& feed_Right) const override final;
		void calBackward(std::unique_ptr<MatrixQL<Dtype>>& loss_Right, std::unique_ptr<MatrixQL<Dtype>>& loss_Left) override final;



		//std::unique_ptr<LayerQL<Dtype>> operator+(const std::unique_ptr<LayerQL<Dtype>>& operRight) const override final;

	protected:
		std::unique_ptr<MatrixQL<Dtype>> w_MatrixQL;
	};

	//*****************************************************************************************************
	template <typename Dtype>
	Fullconnect_LayerQL<Dtype>::Fullconnect_LayerQL(LayerType type, int rowNum, int colNum) : LayerQL(type)
	{
		std::cout << "Fullconnect_LayerQL Start!" << std::endl;

		//std::unique_ptr<MatrixQL<Dtype>> iniMatrix(new MatrixQL<Dtype>(3, 3));
		//this->wMatrixQL = std::move(iniMatrix);

		this->w_MatrixQL = std::make_unique<MatrixQL<Dtype>>(rowNum, colNum);
		//这里后期需要改
		this->w_MatrixQL->setMatrixQL().setConstant(2.2);
	}

	template <typename Dtype>
	Fullconnect_LayerQL<Dtype>::~Fullconnect_LayerQL()
	{
		std::cout << "Fullconnect_LayerQL Over!" << std::endl;
	}

	template <typename Dtype>
	void Fullconnect_LayerQL<Dtype>::calForward(std::unique_ptr<MatrixQL<Dtype>>& feed_Left, std::unique_ptr<MatrixQL<Dtype>>& feed_Right) const
	{
		feed_Right->setMatrixQL() = feed_Left->getMatrixQL() * this->w_MatrixQL->getMatrixQL();
	}

	template <typename Dtype>
	void Fullconnect_LayerQL<Dtype>::calBackward(std::unique_ptr<MatrixQL<Dtype>>& loss_Right, std::unique_ptr<MatrixQL<Dtype>>& loss_Left)
	{
		loss_Left->setMatrixQL() = loss_Right->getMatrixQL() * this->w_MatrixQL->getMatrixQL().transpose();
	}

	//template <typename Dtype>
	//std::unique_ptr<LayerQL<Dtype>> Fullconnect_LayerQL<Dtype>::operator+(const std::unique_ptr<LayerQL<Dtype>>& operRight) const 
	//{ 
	//	//std::unique_ptr<LayerQL<Dtype>> tt;
	//	//return tt;
	//	return NULL;
	//};

}