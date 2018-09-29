#pragma once
#include "MatrixQL.h"
#include "iostream"
#include "memory"
#include "Fullconnect_LayerQL.h"
#include "Bias_LayerQL.h"
#include "Test.h"
#include "Conv_Test.h"
#include "Pool_Test.h"
#include "Sigmoid_Test.h"
#include "Dim_Reduce_Test.h"
#include "LoadCSV_Test.h"
#include "Mnist_Conv_Test.h"
#include "Relu_LayerQL_Test.h"
#include "SoftMax_Layer_Test.h"
#include "Data_Augmentation_Test.h"
#include "Mnist_Conv_T2.h"
using namespace tinyDNN;
int main()
{
	//Test test;
	//¾í»ı²ã
	//Conv_Test test;
	//Pool_Test pool_Test;
	
	//Sigmoid²ã
	//Sigmoid_Test test;
	
	//½µÎ¬²ã
	//Dim_Reduce_Test test;

	//¼ÓÔØÊı¾İ²ã
	//LoadCSV_Test test;

	//²âÊÔÊı¾İÔöÇ¿²ã
	//Data_Augmentation_Test test;

	////²âÊÔ¾í»ıÍøÂç²ã=======================================================================================
	//Mnist_Conv_Test test;

	//²âÊÔRelu²ã
	//Relu_LayerQL_Test test;

	//²âÊÔSoftmax²ã
	//SoftMax_Layer_Test test;

	mnist_Conv_T_1();

	return 0;
}