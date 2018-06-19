#pragma once
#include "LayerQL.h"
#include <vector>

namespace tinyDNN
{
	template <typename Dtype>
	class Conv_Kernel
	{
	public:
		explicit Conv_Kernel(int rowNum, int colNum, int kernelWidth, int kernelSize) : rowNum(rowNum), colNum(colNum), kernelWidth(kernelWidth), kernelSize(kernelSize)
		{
			//std::shared_ptr<MatrixQL<double>> test_01 = std::make_shared<MatrixQL<double>>(5, 5);
			//test_01->setMatrixQL().setOnes();
			//std::shared_ptr<MatrixQL<double>> test_02 = std::make_shared<MatrixQL<double>>(5, 5);
			//test_02->setMatrixQL().setOnes();
			//std::shared_ptr<MatrixQL<double>> test_03 = std::make_shared<MatrixQL<double>>(5, 5);
			//test_03->setMatrixQL().setOnes();
			//this->conv_Kernel_Vector.push_back(test_01);
			//this->conv_Kernel_Vector.push_back(test_02);
			//this->conv_Kernel_Vector.push_back(test_03);

			for (int i = 0; i < kernelSize; i++)
			{
				std::shared_ptr<MatrixQL<Dtype>> oneSlice_Kernel = std::make_shared<MatrixQL<Dtype>>(kernelWidth, kernelWidth);
				oneSlice_Kernel->setMatrixQL().setOnes();
				//one_Kernel->setMatrixQL().setRandom();
				//one_Kernel->setMatrixQL().setZero();
				this->conv_Kernel_Vector.push_back(oneSlice_Kernel);
			}
		}

		void conv_CalForward(std::vector<std::shared_ptr<MatrixQL<Dtype>>>& inMatrixVector, std::shared_ptr<MatrixQL<Dtype>>& outMatrix)
		{
			for ( int i = 0; i < kernelSize; i++ )
			{
				//std::cout << inMatrixVector[i]->getMatrixQL() << std::endl;
				//std::cout << conv_Kernel_Vector[i]->getMatrixQL() << std::endl;

				outMatrix->setMatrixQL() = outMatrix->getMatrixQL() + conv_Matrix( inMatrixVector[i], conv_Kernel_Vector[i] )->getMatrixQL();
			}
		}

		std::shared_ptr<MatrixQL<Dtype>> conv_Matrix( std::shared_ptr<MatrixQL<Dtype>>& inMatrixPtr, std::shared_ptr<MatrixQL<Dtype>>& convMatrixPtr )
		{
			std::shared_ptr<MatrixQL<Dtype>> reMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum,colNum);
			for ( int i = 0; i < rowNum; i++ )
			{
				for ( int j = 0; j < colNum; j++ )
				{
					reMatrix->setMatrixQL()(i, j) = (inMatrixPtr->getMatrixQL().block(i, j, kernelWidth, kernelWidth).array() * convMatrixPtr->getMatrixQL().array()).sum();
				}
			}
			return reMatrix;
		}
		
	public:
		std::vector<std::shared_ptr<MatrixQL<Dtype>>> conv_Kernel_Vector;

		//传入矩阵的行和列
		int rowNum;
		int colNum;
		//卷积核的宽度
		int kernelWidth;
		//卷积核的片数
		int kernelSize;
	};


	//==================================================================================================================================================================

	template <typename Dtype> 
	class Conv_LayerQL : public LayerQL<Dtype>
	{
	public:
		explicit Conv_LayerQL(LayerType type, int kernelNum, int rowNum, int colNum, int kernelWidth, int kernelSize ) : LayerQL(type), kernelNum(kernelNum), rowNum(rowNum), colNum(colNum), kernelWidth(kernelWidth), kernelSize(kernelSize)
		{
			std::cout << "Conv_LayerQL Start!" << std::endl;

			for ( int i = 0; i < kernelNum; i++ )
			{
				std::shared_ptr<Conv_Kernel<Dtype>> oneKernel = std::make_shared<Conv_Kernel<Dtype>>(rowNum, colNum, kernelWidth, kernelSize);

				this->conv_Kernel_Vector.push_back( oneKernel );
			}
		}

		~Conv_LayerQL() override final
		{
			std::cout << "Conv_LayerQL Over!" << std::endl;
		}


		void calForward() const override final
		{
			for ( int i = 0; i < kernelNum; i++ )
			{
				std::shared_ptr<MatrixQL<Dtype>> outMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum, colNum);
				outMatrix->setMatrixQL().setZero();

				this->conv_Kernel_Vector[i]->conv_CalForward( this->left_Layer->forward_Matrix_Vector, outMatrix );
				this->right_Layer->forward_Matrix_Vector.push_back( outMatrix );
			}
		}


		void calBackward() override final {};


		void upMatrix() override final {};
		void upMatrix_batch(Dtype upRate) override final {};

	public:
		std::vector<std::shared_ptr<Conv_Kernel<Dtype>>> conv_Kernel_Vector;
		int kernelNum;

		//传入矩阵的行和列
		int rowNum;
		int colNum;
		//卷积核的宽度
		int kernelWidth;
		//卷积核的片数
		int kernelSize;
	};
}