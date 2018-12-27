#include "FMM2DTree.hpp"


class userkernel: public kernel {
public:
	userkernel() {
		isTrans		=	true;
		isHomog		=	true;
		isLogHomog	=	false;
		alpha		=	-1.0;
	};
	std::complex<double> getInteraction(const pts2D r1, const pts2D r2, double Kwave_no, double a) {
		double R2	=	(r1.x-r2.x)*(r1.x-r2.x) + (r1.y-r2.y)*(r1.y-r2.y);
		double R	=	sqrt(R2);
		if (R < a) {
			return R/a;
		}
		else {
			return exp(I*Kwave_no*R)/R;
		}
	};
	~userkernel() {};
};


int main(int argc, char* argv[]) {
	int nLevels		=	atoi(argv[1]);
	int nCones_LFR		=	atoi(argv[2]);
	int nChebNodes		=	atoi(argv[3]);
	double L		=	atof(argv[4]);
	double Kwave_no		=	atof(argv[5]);
	int yesToDFMM		=	atoi(argv[6]);
	//double DFMM_A		=	atof(argv[7]);

	double start, end;

	start	=	omp_get_wtime();
	userkernel* mykernel		=	new userkernel();
	FMM2DTree<userkernel>* A	=	new FMM2DTree<userkernel>(mykernel, nLevels, nCones_LFR, nChebNodes, L, Kwave_no, yesToDFMM);

	A->set_Standard_Cheb_Nodes();

	A->createTree();
	
	A->assign_Center_Location();

	A->assign_Tree_Interactions();

	A->assign_Tree_Interactions_LFR();

	A->createCones();

	end		=	omp_get_wtime();
	double timeCreateTree	=	(end-start);

	std::cout << std::endl << "Number of particles is: " << A->N << std::endl;
	std::cout << std::endl << "Time taken to create the tree is: " << timeCreateTree << std::endl;


	start	=	omp_get_wtime();

	A->get_Transfer_Matrix();

	end		=	omp_get_wtime();
	double timeAssemble		=	(end-start);
	std::cout << std::endl << "Time taken to assemble the operators is: " << timeAssemble << std::endl;

	start	=	omp_get_wtime();

	A->assign_Leaf_Charges();

	end		=	omp_get_wtime();
	double timeAssignCharges=	(end-start);
	std::cout << std::endl << "Time taken to assemble the charges is: " << timeAssignCharges << std::endl;

	start	=	omp_get_wtime();
	A->evaluate_All_M2M();
	end		=	omp_get_wtime();
	double timeM2M			=	(end-start);
	std::cout << std::endl << "Time taken for multipole to multipole is: " << timeM2M << std::endl;

	start	=	omp_get_wtime();
	//A->evaluate_HFR_M2L_ACA();

	A->evaluate_All_M2L();
	end		=	omp_get_wtime();
	double timeM2L			=	(end-start);
	std::cout << std::endl << "Time taken for multipole to local is: " << timeM2L << std::endl;

	start	=	omp_get_wtime();
	A->evaluate_All_L2L();
	end		=	omp_get_wtime();
	double timeL2L			=	(end-start);
	std::cout << std::endl << "Time taken for local to local is: " << timeL2L << std::endl;

	start	=	omp_get_wtime();
//	A->evaluate_Leaf();
	end		=	omp_get_wtime();
	double timeLeaf			=	(end-start);
//	std::cout << std::endl << "Time taken for self and neighbors at leaf is: " << timeLeaf << std::endl;

	double totalTime	=	timeCreateTree+timeAssemble+timeAssignCharges+timeAssemble+timeM2M+timeM2L+timeL2L+timeLeaf;
	
	double applyTime	=	timeM2M+timeM2L+timeL2L+timeLeaf;

	std::cout << std::endl << "Total time taken is: " << totalTime << std::endl;

	std::cout << std::endl << "Apply time taken is: " << applyTime << std::endl;

	std::cout << std::endl << "Total Speed in particles per second is: " << A->N/totalTime << std::endl;

	std::cout << std::endl << "Apply Speed in particles per second is: " << A->N/applyTime << std::endl;

	std::cout << std::endl << "Number of particles is: " << A->N << std::endl;

	
	std::cout << std::endl << "Performing Error check..." << std::endl;

	srand(time(NULL));
	int nBox	=	rand()%A->nBoxesPerLevel[nLevels];
	std::cout << std::endl << "Box number is: " << nBox << std::endl;
	std::cout << std::endl << "Box center is: (" << A->tree[nLevels][nBox].center.x << ", " << A->tree[nLevels][nBox].center.y << ");" << std::endl;
	start	=	omp_get_wtime();
	std::cout << std::endl << "Error is: " << A->perform_Error_Check(nBox) << std::endl;
	end		=	omp_get_wtime();
	double errorTime	=	(end-start);

	std::cout << std::endl << "Time taken to compute error is: " << errorTime << std::endl;

	std::cout << std::endl;

}
