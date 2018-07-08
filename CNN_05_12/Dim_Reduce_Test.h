#pragma once
#include "Dim_ReduceQL.h"
#include "NetQL.h"
namespace tinyDNN
{
	class Dim_Reduce_Test
	{
	public:
		Dim_Reduce_Test()
		{
			this->cal_ForWard_Test();
			this->cal_BackWard_Test();
		}
		~Dim_Reduce_Test(){}


		void cal_ForWard_Test()
		{
			std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(5, 5);

			double matrix_Num = 1.1;
			for (int i = 0; i < 3; i++)
			{
				std::shared_ptr<MatrixQL<double>> v_01 = std::make_shared<MatrixQL<double>>(5, 5);
				for (int j = 0; j < 5; j++)
				{
					for (int k = 0; k < 5; k++)
					{
						v_01->setMatrixQL()(j, k) = matrix_Num;
						matrix_Num++;
					}
				}
				in_01->forward_Matrix_Vector.push_back(v_01);
			}
			std::shared_ptr<LayerQL<double>> dim_Test = std::make_shared<Dim_ReduceQL<double>>(Dim_Reduce_Layer, 3, 5, 5);

			std::shared_ptr<Inter_LayerQL<double>> o_01 = in_01 + dim_Test;

			int rec = 3;
			while ( rec > 0)
			{
				dim_Test->calForward();
				std::cout << o_01->forward_Matrix->getMatrixQL() << std::endl;

				std::cout << "*******************************************" << std::endl;
				rec--;
			}
		}

		void cal_BackWard_Test()
		{
			std::shared_ptr<Inter_LayerQL<double>> in_01 = std::make_shared<Inter_LayerQL<double>>(5, 5);
			std::shared_ptr<LayerQL<double>> dim_Test = std::make_shared<Dim_ReduceQL<double>>(Dim_Reduce_Layer, 4, 5, 5);
			std::shared_ptr<Inter_LayerQL<double>> o_01 = in_01 + dim_Test;
			
			std::shared_ptr<MatrixQL<double>> r_Matrix = std::make_shared<MatrixQL<double>>(1,100);
			
			double r_Num = 1.2;
			for (int i = 0; i < 100; i++)
			{
				r_Matrix->setMatrixQL()(0, i) = r_Num;
				r_Num++;
			}

			o_01->backward_Matrix = r_Matrix;

			int rec = 3;
			while (rec > 0)
			{
				dim_Test->calBackward();
				std::for_each(in_01->backward_Matrix_Vector.begin(), in_01->backward_Matrix_Vector.end(),
					[](std::shared_ptr<MatrixQL<double>> m_Test)
				{
					std::cout << m_Test->getMatrixQL() << std::endl;

					std::cout << "********************************************" << std::endl;
				}
				);
				std::cout << "------------------------------------------------" << std::endl;
				rec--;
			}
		}
	};
}