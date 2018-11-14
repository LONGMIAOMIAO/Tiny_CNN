#pragma once
#include "LayerQL.h"
namespace tinyDNN
{
	template <typename Dtype>
	class Bias_Conv_Layer : public LayerQL<Dtype>
	{
	public:
		//									   几片			 行数		 列数
		Bias_Conv_Layer( LayerType type, int kernelNum, int rowNum, int colNum ) : LayerQL(type), kernelNum(kernelNum), rowNum(rowNum), colNum(colNum)
		{
			std::cout << "Bias_Conv_Layer Start!" << std::endl;

			b_MatrixQL.reserve( kernelNum );
			for ( int i = 0; i < kernelNum; i++ )
			{
				b_MatrixQL.push_back( std::make_unique<MatrixQL<Dtype>>( rowNum, colNum ) );
				b_MatrixQL.back()->setMatrixQL().setConstant(0.1);
			}
		}

		~Bias_Conv_Layer() override final
		{
			std::cout << "Bias_Conv_Layer Over!" << std::endl;
		}

		void calForward(int type = 0) const override final
		{
			//	每次向前传播先清理掉集合中的内容再重新插入，这里貌似可以优化
			this->right_Layer->forward_Matrix_Vector.clear();

			for (int i = 0; i < kernelNum; i++)
			{
				std::shared_ptr<MatrixQL<Dtype>> outMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum, colNum);
				outMatrix->setMatrixQL().setZero();

				outMatrix->setMatrixQL() = this->left_Layer->forward_Matrix_Vector[i]->getMatrixQL() + b_MatrixQL[i]->getMatrixQL();
				this->right_Layer->forward_Matrix_Vector.push_back(outMatrix);
			}
		}

		void calBackward(int type = 0) override final
		{
			this->left_Layer->backward_Matrix_Vector = this->right_Layer->backward_Matrix_Vector;
		}

		void upMatrix() override final
		{
			for ( int i = 0; i < kernelNum; i++ )
			{
				this->b_MatrixQL[i]->setMatrixQL() = this->b_MatrixQL[i]->getMatrixQL() - 0.1 * this->right_Layer->backward_Matrix_Vector[i]->getMatrixQL();
			}
		}

		void upMatrix_batch(Dtype upRate) override final {};

	public:
		//std::vector<std::shared_ptr<Conv_Kernel<Dtype>>> conv_Kernel_Vector;
		std::vector<std::unique_ptr<MatrixQL<Dtype>>> b_MatrixQL;

		//卷积核的个数
		int kernelNum;
		//传入矩阵的行和列
		int rowNum;
		int colNum;
		//std::unique_ptr<MatrixQL<Dtype>> b_MatrixQL;
	};
}