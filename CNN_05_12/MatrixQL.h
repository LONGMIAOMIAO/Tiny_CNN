#pragma once
#include <Eigen/Core>
#include<cmath>

namespace tinyDNN {

	//	定义底层Eigen矩阵模板
	template <typename Dtype>
	using MatrixData = typename Eigen::Matrix <Dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

	template <typename Dtype>
	class MatrixQL
	{
		//	定义底层Eigen矩阵模板
		//using MatrixData = Eigen::Matrix <Dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	public:
		//	MatrixQL() = default;
		//	显式调用模板构造函数
		MatrixQL(int rowNum, int colNum);
		//	析构函数
		~MatrixQL();

		const MatrixData<Dtype>& getMatrixQL() const;
		MatrixData<Dtype>& setMatrixQL();

	private:
		//	声明底层矩阵
		MatrixData<Dtype> matrixData;
		//	矩阵行数
		int rowNum;
		//	矩阵列数
		int colNum;
	};

	//	模板构造函数
	template <typename Dtype>
	MatrixQL<Dtype>::MatrixQL(int rowNum, int colNum) : rowNum(rowNum), colNum(colNum)
	{
		//std::cout << "Matrix Start!" << std::endl;
		
		//	将声明的底层库重新设置大小
		this->matrixData.resize(this->rowNum, this->colNum);
	}

	//	模板析构函数
	template <typename Dtype>
	MatrixQL<Dtype>::~MatrixQL()
	{
		//std::cout << "Matrix Over!" << std::endl;
	}

	//	取出底层数据库
	//	注意：这里返回模板类内定义类型的时候，需要加typeName
	template <typename Dtype>
	inline const typename /*MatrixQL<Dtype>::*/MatrixData<Dtype>& MatrixQL<Dtype>::getMatrixQL() const
	{
		return this->matrixData;
	}

	template <typename Dtype>
	inline typename /*MatrixQL<Dtype>::*/MatrixData<Dtype>& MatrixQL<Dtype>::setMatrixQL()
	{
		return this->matrixData;
	}
}