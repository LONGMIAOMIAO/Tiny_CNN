#pragma once
#include "MatrixQL.h"
#include "iostream"
#include "memory"
#include "Fullconnect_LayerQL.h"
using namespace tinyDNN;
int main()
{
	//MatrixQL<float> mm(2,2);
	//mm.setMatrixQL().setRandom();
	//std::cout << (mm.getMatrixQL()) << std::endl;
	//std::cout << (mm.setMatrixQL()) << std::endl;
	//std::cout << &(mm.getMatrixQL()) << std::endl;
	//std::cout << &(mm.setMatrixQL()) << std::endl;
	//std::cout << &(mm.getMatrixQL().cast<int>()) << std::endl;


	//MatrixQL<int> nn(2, 2);
	//nn.setMatrixQL().setOnes();
	//std::cout << nn.getMatrixQL() << std::endl;

	//MatrixQL<double> oo(2, 2);
	//oo.setMatrixQL() = mm.getMatrixQL().cast<double>() + nn.getMatrixQL().cast<double>();
	//std::cout << oo.getMatrixQL() << std::endl;

	//std::shared_ptr<int> ptr_01 = std::make_shared<int>(3);
	//std::shared_ptr<int>& ptr_02 = ptr_01;
	//std::cout << ptr_01.use_count() << std::endl;

	//std::unique_ptr<int> ptr_03 = std::make_unique<int>(4);
	//std::unique_ptr<int>& ptr_04 = ptr_03;
	//std::cout << *ptr_04 << std::endl;

	//LayerQL<double> layerTest;
	//Fullconnect_LayerQL<double> layerTest;

	std::unique_ptr<MatrixQL<double>> matrixIn(new MatrixQL<double> (1,3));
	matrixIn->setMatrixQL().setOnes();

	std::unique_ptr<MatrixQL<double>> matrixOut(new MatrixQL<double> (1,3));
	matrixOut->setMatrixQL().setOnes();

	//LayerQL<double>* layerTest = new Fullconnect_LayerQL<double>;

	//std::shared_ptr<LayerQL<double>> layerTest; 
	//layerTest = std::make_shared<Fullconnect_LayerQL<double>>();
	std::unique_ptr<LayerQL<double>> layerTest = std::make_unique<Fullconnect_LayerQL<double>>();

	layerTest->calForward(matrixIn,matrixOut);

	std::cout << matrixOut->getMatrixQL() << std::endl;

	//delete layerTest;
}