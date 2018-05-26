#pragma once
#include "MatrixQL.h"
#include "memory"
#include <iostream>
namespace tinyDNN 
{
	//struct LayerParameter
	//{
	//	int layerType;
	//	int intNum;
	//	int outNum;
	//	int actType;
	//	int initialType;
	//};
	enum LayerType 
	{	
		Fullconnect_Layer = 1,
		Bias_Layer = 2
	};

	template <typename Dtype>
	class LayerQL
	{
	public:
		explicit LayerQL( LayerType type ) ;
		virtual ~LayerQL();

		virtual void calForward( std::unique_ptr<MatrixQL<Dtype>>& feed_Left, std::unique_ptr<MatrixQL<Dtype>>& feed_Right ) const  = 0;
		virtual void calBackward( std::unique_ptr<MatrixQL<Dtype>>& loss_Right, std::unique_ptr<MatrixQL<Dtype>>& loss_Left ) = 0;


	protected:
		LayerType layerType;
	};
	//******************************************************************************************************************************
	template <typename Dtype>
	LayerQL<Dtype>::LayerQL( LayerType type ) : layerType(type)
	{
		std::cout << "Layer Start!" << std::endl;

		////创建一个新的矩阵W指针
		//std::unique_ptr<MatrixQL<Dtype>> iniMatrix(new MatrixQL<Dtype>(3, 3));
		////得到成员矩阵的地址和初始化智能指针的地址
		//std::cout << wMatrixQL.get() << std::endl;
		//std::cout << iniMatrix.get() << std::endl;
		////得到成员矩阵的地址和初始化智能指针自身的地址
		//std::cout << &wMatrixQL << std::endl;
		//std::cout << &iniMatrix << std::endl;

		////分别用reset和std::move两种方法转移智能指针的所有权
		//this->wMatrixQL = std::move(iniMatrix);
		////this->wMatrixQL.reset(iniMatrix.release());
		////再次打印成员矩阵地址和初始矩阵地址，发现两个智能指针所存的地址交换了。
		////说明move是吧两个左值的内容转移了。
		//std::cout << wMatrixQL.get() << std::endl;
		//std::cout << iniMatrix.get() << std::endl;

		//std::cout << &wMatrixQL << std::endl;
		//std::cout << &iniMatrix << std::endl;

		////将成员矩阵随机化
		//this->wMatrixQL->setMatrixQL().setRandom();
		////打印成员矩阵
		//std::cout << this->wMatrixQL->getMatrixQL() << std::endl;
	}

	template <typename Dtype>
	LayerQL<Dtype>::~LayerQL()
	{
		std::cout << "Layer Over!" << std::endl;
	}
}

