#pragma once
#include "LayerQL.h"
#include <vector>

namespace tinyDNN
{
	template <typename Dtype>
	class Conv_Kernel
	{
	public:
		Conv_Kernel(int rowNum, int colNum, int kernelWidth, int kernelSize, int paddingSize) : rowNum(rowNum), colNum(colNum), kernelWidth(kernelWidth), kernelSize(kernelSize), paddingSize(paddingSize)
		{
			for (int i = 0; i < kernelSize; i++)
			{
				std::shared_ptr<MatrixQL<Dtype>> oneSlice_Kernel = std::make_shared<MatrixQL<Dtype>>(kernelWidth, kernelWidth);
				
				//测试
				////oneSlice_Kernel->setMatrixQL().setOnes();
				//double startNum = 0.1 * (i + 1) ;
				//for ( int p = 0; p < kernelWidth; p++ )
				//{
				//	for ( int q = 0; q < kernelWidth; q++ )
				//	{
				//		oneSlice_Kernel->setMatrixQL()(p, q) = startNum;
				//		startNum = startNum + 0.1 * (i + 1);
				//	}
				//}
				
				////实操
				//oneSlice_Kernel->setMatrixQL().setRandom();

				//实操， 采用高斯随机，正态分布
				std::random_device rd;
				std::mt19937 gen(rd());
				//平均值1，标准差 0.1
				std::normal_distribution<Dtype> normal(0, 0.01);
				for ( int p = 0; p < kernelWidth; p++ )
				{
					for ( int q = 0; q < kernelWidth; q++ )
					{
						oneSlice_Kernel->setMatrixQL()(p, q) = normal(gen);
					}
				}

				//	一个卷积核有i片
				this->conv_Kernel_Vector.push_back(oneSlice_Kernel);
			}
		}
		//	对输入的片集合进行相乘并相加
		void conv_CalForward(std::vector<std::shared_ptr<MatrixQL<Dtype>>>& inMatrixVector, std::shared_ptr<MatrixQL<Dtype>>& outMatrix)
		{

			for ( int i = 0; i < kernelSize; i++ )
			{
				//std::cout << inMatrixVector[i]->getMatrixQL() << std::endl;
				//std::cout << conv_Kernel_Vector[i]->getMatrixQL() << std::endl;

				////更新载入矩阵
				std::shared_ptr<MatrixQL<Dtype>> paddingMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum + 2 * paddingSize, colNum + 2 * paddingSize);
				paddingMatrix->setMatrixQL().setZero();
				paddingMatrix->setMatrixQL().block( paddingSize, paddingSize, rowNum, colNum) = inMatrixVector[i]->getMatrixQL().block(0, 0, rowNum, colNum);

				//对每一个卷积核进行计算
				outMatrix->setMatrixQL() = outMatrix->getMatrixQL() + conv_Matrix( paddingMatrix, conv_Kernel_Vector[i] )->getMatrixQL();
			}
		}
		//对每一张图进行卷积计算
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
		//卷积核的扩充
		int paddingSize;
	};


	//=======================================================================================================================

	template <typename Dtype> 
	class Conv_LayerQL : public LayerQL<Dtype>
	{
	public:
		//							类型			卷积核数			行数			列数				卷积核宽度		卷积核几片		扩充宽度	
		Conv_LayerQL(LayerType type, int kernelNum, int rowNum, int colNum, int kernelWidth, int kernelSize, int paddingSize ) : LayerQL(type), kernelNum(kernelNum), rowNum(rowNum), colNum(colNum), kernelWidth(kernelWidth), kernelSize(kernelSize), paddingSize(paddingSize)
		{
			std::cout << "Conv_LayerQL Start!" << std::endl;

			for ( int i = 0; i < kernelNum; i++ )
			{
				std::shared_ptr<Conv_Kernel<Dtype>> oneKernel = std::make_shared<Conv_Kernel<Dtype>>(rowNum, colNum, kernelWidth, kernelSize, paddingSize);

				this->conv_Kernel_Vector.push_back( oneKernel );
			}
		}

		~Conv_LayerQL() override final
		{
			std::cout << "Conv_LayerQL Over!" << std::endl;
		}

		void calForward(int type = 0) const override final
		{
			//每次向前传播先清理掉集合中的内容再重新插入
			this->right_Layer->forward_Matrix_Vector.clear();

			for ( int i = 0; i < kernelNum; i++ )
			{
				std::shared_ptr<MatrixQL<Dtype>> outMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum, colNum);
				outMatrix->setMatrixQL().setZero();

				this->conv_Kernel_Vector[i]->conv_CalForward( this->left_Layer->forward_Matrix_Vector, outMatrix );
				this->right_Layer->forward_Matrix_Vector.push_back( outMatrix );
			}
		}

		void calBackward(int type = 0) override final
		{
			this->left_Layer->backward_Matrix_Vector.clear();

			for ( int i = 0; i < kernelSize; i++ )
			{
				std::shared_ptr<MatrixQL<Dtype>> matrix_Left = std::make_shared<MatrixQL<Dtype>>( rowNum, colNum );
				matrix_Left->setMatrixQL().setZero();
				this->left_Layer->backward_Matrix_Vector.push_back(matrix_Left);
			}

			for ( int i = 0; i < kernelNum; i++ )
			{
				std::shared_ptr<MatrixQL<Dtype>> paddingMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum + 2 * paddingSize, colNum + 2 * paddingSize);
				paddingMatrix->setMatrixQL().setZero();
				paddingMatrix->setMatrixQL().block(paddingSize, paddingSize, rowNum, colNum) = this->right_Layer->backward_Matrix_Vector[i]->getMatrixQL().block(0, 0, rowNum, colNum);

				for ( int j = 0 ; j < kernelSize; j++ )
				{
					std::shared_ptr<MatrixQL<Dtype>> reMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum, colNum);
					for (int p = 0; p < rowNum; p++)
					{
						for (int q = 0; q < colNum; q++)
						{
							reMatrix->setMatrixQL()(p, q) = (paddingMatrix->getMatrixQL().block(p, q, kernelWidth, kernelWidth).array() * conv_Kernel_Vector[i]->conv_Kernel_Vector[j]->getMatrixQL().reverse().array()).sum();
						}
					}
					this->left_Layer->backward_Matrix_Vector[j]->setMatrixQL() = this->left_Layer->backward_Matrix_Vector[j]->getMatrixQL() + reMatrix->getMatrixQL();
				}
			}
		};

		void upMatrix() override final 
		{
			for ( int i = 0 ; i < kernelNum; i++ )
			{
				for ( int j = 0 ; j < kernelSize; j++ )
				{
					std::shared_ptr<MatrixQL<Dtype>> paddingMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum + 2 * paddingSize, colNum + 2 * paddingSize);
					paddingMatrix->setMatrixQL().setZero();
					paddingMatrix->setMatrixQL().block(paddingSize, paddingSize, rowNum, colNum) = this->left_Layer->forward_Matrix_Vector[j]->getMatrixQL().block(0, 0, rowNum, colNum);

					std::shared_ptr<MatrixQL<Dtype>> upMatrix = std::make_shared<MatrixQL<Dtype>>( kernelWidth, kernelWidth );
					for (int p = 0; p < kernelWidth; p++)
					{
						for (int q = 0; q < kernelWidth; q++)
						{
							upMatrix->setMatrixQL()(p, q) = (paddingMatrix->getMatrixQL().block(p, q, rowNum, colNum).array() *
								this->right_Layer->backward_Matrix_Vector[i]->getMatrixQL().array()).sum();
						}
					}

					//std::cout << "UP值++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
					//std::cout << upMatrix->getMatrixQL() << std::endl;
					//std::cout << "减之前++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
					//std::cout << this->conv_Kernel_Vector[i]->conv_Kernel_Vector[j]->getMatrixQL() << std::endl;

					//*********************************************
					//this->conv_Kernel_Vector[i]->conv_Kernel_Vector[j]->setMatrixQL() = this->conv_Kernel_Vector[i]->conv_Kernel_Vector[j]->getMatrixQL() - 0.5 * upMatrix->getMatrixQL();

					this->conv_Kernel_Vector[i]->conv_Kernel_Vector[j]->setMatrixQL() = this->conv_Kernel_Vector[i]->conv_Kernel_Vector[j]->getMatrixQL() - this->upConv * upMatrix->getMatrixQL();
					//*********************************************

					//std::cout << "减之后++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
					//std::cout << this->conv_Kernel_Vector[i]->conv_Kernel_Vector[j]->getMatrixQL() << std::endl;

				}

			}
		
		};
		void upMatrix_batch(Dtype upRate) override final {};

	public:
		std::vector<std::shared_ptr<Conv_Kernel<Dtype>>> conv_Kernel_Vector;
		//卷积核的个数
		int kernelNum;
		//传入矩阵的行和列
		int rowNum;
		int colNum;
		//卷积核的宽度
		int kernelWidth;
		//卷积核的片数
		int kernelSize;
		//扩充大小
		int paddingSize;
	};
}