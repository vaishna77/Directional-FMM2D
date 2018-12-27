#ifndef _FMM2DTree_HPP__
#define _FMM2DTree_HPP__

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm> 
#include <complex>

#define EIGEN_DONT_PARALLELIZE
using namespace std;
const double PI	=	3.1415926535897932384;
const std::complex<double> I(0.0, 1.0);

struct pts2D {
	double x,y;
};

class FMM2DCone {
public:
	int coneNumber;
	int parentNumber;
	int childrenNumbers[2];
	
	pts2D direction;
	//Directional multipoles and locals of the box in this cone direction
	Eigen::VectorXcd multipoles;
	Eigen::VectorXcd locals;
	FMM2DCone () {
		coneNumber	=	-1;
		parentNumber	=	-1;
		for (int l=0; l<2; ++l) {
			childrenNumbers[l]	=	-1;
		}
	}
};

class FMM2DBox {
public:
	int boxNumber;
	int parentNumber;
	int childrenNumbers[4];
	int neighborNumbers[8];
	int innerNumbers[16];
	int outerNumbers[24];

	std::vector<int > neighborNumbersHFR;
	std::vector<int > InteractionList;
	FMM2DBox () {
		boxNumber		=	-1;
		parentNumber	=	-1;
		for (int l=0; l<4; ++l) {
			childrenNumbers[l]	=	-1;
		}
		for (int l=0; l<8; ++l) {
			neighborNumbers[l]	=	-1;
		}
		for (int l=0; l<16; ++l) {
			innerNumbers[l]		=	-1;
		}
		for (int l=0; l<24; ++l) {
			outerNumbers[l]		=	-1;
		}
	}

	Eigen::VectorXcd multipoles;
	Eigen::VectorXcd locals;

	pts2D center;

	//	The following will be stored only at the leaf nodes
	std::vector<pts2D> chebNodes;
	std::vector<FMM2DCone> ConeTree;
};

class kernel {
public:
	bool isTrans;		//	Checks if the kernel is translation invariant, i.e., the kernel is K(r).
	bool isHomog;		//	Checks if the kernel is homogeneous, i.e., K(r) = r^{alpha}.
	bool isLogHomog;	//	Checks if the kernel is log-homogeneous, i.e., K(r) = log(r^{alpha}).
	double alpha;		//	Degree of homogeneity of the kernel.
	kernel() {};
	~kernel() {};
	//std::complex<double> mycomplex (3.0, 4.0);
	virtual std::complex<double> getInteraction(const pts2D r1, const pts2D r2, double Kwave_no, double a){
		return std::complex<double>(0.0, 0.0);
	};	//	Kernel entry generator
};

template <typename kerneltype>
class FMM2DTree {
public:
	kerneltype* K;
	int nLevels;			//	Number of levels in the tree.
	int nChebNodes;			//	Number of Chebyshev nodes along one direction.
	int rank;				//	Rank of interaction, i.e., rank = nChebNodes*nChebNodes.
	int N;					//	Number of particles.
	double L;				//	Semi-length of the simulation box.
	double smallestBoxSize;	//	This is L/2.0^(nLevels).
	double a;				//	Cut-off for self-interaction. This is less than the length of the smallest box size.
	//const double A = 1.0;
	const double A = 2.0;
	const double B = 1.0;
	double Kwave_no;
	int level_LFR;
	int level_FarField;
	int nCones_LFR;
	int yesToDFMM;

	std::vector<int> nBoxesPerLevel;			//	Number of boxes at each level in the tree.
	std::vector<double> boxRadius;				//	Box radius at each level in the tree assuming the box at the root is [-1,1]^2
	std::vector<double> ConeAperture;
	std::vector<int> nCones;
	std::vector<double> boxHomogRadius;			//	Stores the value of boxRadius^{alpha}
	std::vector<double> boxLogHomogRadius;		//	Stores the value of alpha*log(boxRadius)
	std::vector<std::vector<FMM2DBox> > tree;	//	The tree storing all the information.

	//	Chebyshev nodes
	std::vector<double> standardChebNodes1D;
	std::vector<pts2D> standardChebNodes;
	std::vector<pts2D> standardChebNodesChild;
	std::vector<pts2D> leafChebNodes;

	//	Different Operators
	Eigen::MatrixXcd selfInteraction;		//	Needed only at the leaf level.
	Eigen::MatrixXcd neighborInteraction[8];	//	Neighbor interaction only needed at the leaf level.
	Eigen::MatrixXd M2M[4];					//	Transfer from multipoles of 4 children to multipoles of parent.
	Eigen::MatrixXd L2L[4];					//	Transfer from locals of parent to locals of 4 children.
	Eigen::MatrixXcd M2LInner[16];			//	M2L of inner interactions. This is done on the box [-L,L]^2.
	Eigen::MatrixXcd M2LOuter[24];			//	M2L of outer interactions. This is done on the box [-L,L]^2.

// public:
	FMM2DTree(kerneltype* K, int nLevels, int nCones_LFR, int nChebNodes, double L, double Kwave_no, int yesToDFMM) {
		this->K				=	K;
		this->nLevels			=	nLevels;
		this->nChebNodes		=	nChebNodes;
		this->rank			=	nChebNodes*nChebNodes;
		this->L				=	L;
		this->nCones_LFR		=	nCones_LFR;
		this->yesToDFMM			=	yesToDFMM;
		//this->A				=	A;
		nBoxesPerLevel.push_back(1);
		boxRadius.push_back(L);
		boxHomogRadius.push_back(pow(L,K->alpha));
		boxLogHomogRadius.push_back(K->alpha*log(L));
		for (int k=1; k<=nLevels; ++k) {
			nBoxesPerLevel.push_back(4*nBoxesPerLevel[k-1]);
			boxRadius.push_back(0.5*boxRadius[k-1]);
			boxHomogRadius.push_back(pow(0.5,K->alpha)*boxHomogRadius[k-1]);
			boxLogHomogRadius.push_back(boxLogHomogRadius[k-1]-K->alpha*log(2));
		}
		this->smallestBoxSize	=	boxRadius[nLevels];
		this->a			=	smallestBoxSize;
		this->N			=	rank*nBoxesPerLevel[nLevels];
		this->Kwave_no		=	Kwave_no;
		if (yesToDFMM == 1) {
			if (Kwave_no==0)
				this->level_LFR	=	2;	//actually should be 2; but for checking the accuracy of DFMM i am making it 3; so that HFR code runs even for LFR; if that gives good result it means DFMM code is perfect
			else
				this->level_LFR	=	floor(log(Kwave_no*L/B)/log(2.0));
		}
		else
			this->level_LFR	=	2;
		if (level_LFR < 2) level_LFR = 2;
		cout << "level_LFR: " << level_LFR << endl;
	}

	std::vector<pts2D> shift_Cheb_Nodes(double xShift, double yShift) {
		std::vector<pts2D> shiftedChebNodes;
		for (int k=0; k<rank; ++k) {
			pts2D temp;
			temp.x	=	standardChebNodes[k].x+2*xShift;
			temp.y	=	standardChebNodes[k].y+2*yShift;
			shiftedChebNodes.push_back(temp);
		}
		return shiftedChebNodes;
	}
	
	//	shifted_scaled_cheb_nodes	//	used in evaluating multipoles
	std::vector<pts2D> shift_scale_Cheb_Nodes(double xShift, double yShift, double radius) {
		std::vector<pts2D> shifted_scaled_ChebNodes;
		for (int k=0; k<rank; ++k) {
			pts2D temp;
			temp.x	=	radius*standardChebNodes[k].x+xShift;
			temp.y	=	radius*standardChebNodes[k].y+yShift;
			shifted_scaled_ChebNodes.push_back(temp);
		}
		return shifted_scaled_ChebNodes;
	}

	std::vector<pts2D> shift_Leaf_Cheb_Nodes(double xShift, double yShift) {
		std::vector<pts2D> shiftedChebNodes;
		for (int k=0; k<rank; ++k) {
			pts2D temp;
			temp.x	=	leafChebNodes[k].x+xShift;
			temp.y	=	leafChebNodes[k].y+yShift;
			shiftedChebNodes.push_back(temp);
		}
		return shiftedChebNodes;
	}

	//	get_ChebPoly
	double get_ChebPoly(double x, int n) {
		return cos(n*acos(x));
	}

	//	get_S
	double get_S(double x, double y, int n) {
		double S	=	0.5;
		for (int k=1; k<n; ++k) {
			S+=get_ChebPoly(x,k)*get_ChebPoly(y,k);
		}
		return 2.0/n*S;
	}
	//	set_Standard_Cheb_Nodes
	void set_Standard_Cheb_Nodes() {
		for (int k=0; k<nChebNodes; ++k) {
			standardChebNodes1D.push_back(-cos((k+0.5)/nChebNodes*PI));
		}
		pts2D temp1;
		for (int j=0; j<nChebNodes; ++j) {
			for (int k=0; k<nChebNodes; ++k) {
				temp1.x	=	standardChebNodes1D[k];
				temp1.y	=	standardChebNodes1D[j];
				standardChebNodes.push_back(temp1);
			}
		}
		//	Left Bottom child, i.e., Child 0
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x-0.5;
				temp1.y	=	0.5*temp1.y-0.5;
				standardChebNodesChild.push_back(temp1);
		}
		//	Right Bottom child, i.e., Child 1
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x+0.5;
				temp1.y	=	0.5*temp1.y-0.5;
				standardChebNodesChild.push_back(temp1);
		}
		//	Right Top child, i.e., Child 2
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x+0.5;
				temp1.y	=	0.5*temp1.y+0.5;
				standardChebNodesChild.push_back(temp1);
		}
		//	Left Top child, i.e., Child 3
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x-0.5;
				temp1.y	=	0.5*temp1.y+0.5;
				standardChebNodesChild.push_back(temp1);
		}
	}

	void get_Transfer_Matrix() {
		for (int l=0; l<4; ++l) {
			L2L[l]	=	Eigen::MatrixXd(rank,rank);
			for (int j=0; j<rank; ++j) {
				for (int k=0; k<rank; ++k) {
					L2L[l](j,k)	=	get_S(standardChebNodes[k].x, standardChebNodesChild[j+l*rank].x, nChebNodes)*get_S(standardChebNodes[k].y, standardChebNodesChild[j+l*rank].y, nChebNodes);
				}
			}
		}
		for (int l=0; l<4; ++l) {
			M2M[l]	=	L2L[l].transpose();
		}
	}

	void createTree() {
		//	First create root and add to tree
		FMM2DBox root;
		root.boxNumber		=	0;
		root.parentNumber	=	-1;
		#pragma omp parallel for
		for (int l=0; l<4; ++l) {
			root.childrenNumbers[l]	=	l;
		}
		std::vector<FMM2DBox> rootLevel;
		rootLevel.push_back(root);
		tree.push_back(rootLevel);

		for (int j=1; j<=nLevels; ++j) {
			std::vector<FMM2DBox> level;
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				FMM2DBox box;
				box.boxNumber		=	k;
				box.parentNumber	=	k/4;
				for (int l=0; l<4; ++l) {
					box.childrenNumbers[l]	=	4*k+l;
				}
				level.push_back(box);
			}
			tree.push_back(level);
		}
	}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//	Assigns the interactions for child0 of a box
	void assign_Child0_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k;
		int nN;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  N5  |  N4  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  **  |  N3  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[3]	=	nC+1;
			tree[nL][nC].neighborNumbers[4]	=	nC+2;
			tree[nL][nC].neighborNumbers[5]	=	nC+3;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/****************************/
			/*				   ______	*/
			/*				  |		 |	*/
			/*				  |	 **  |	*/
			/*	 _____________|______|  */
			/*	|	   |	  |			*/
			/*	|  I15 |  N0  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I0  |  I1  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[0];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].neighborNumbers[0]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[15]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 ______			  		*/
			/*	|	   |	  			*/
			/*	|  **  |				*/
			/*	|______|______			*/
			/*	|	   |	  |			*/
			/*	|  N1  |  N2  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I2  |  I3  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].neighborNumbers[2]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].neighborNumbers[1]	=	tree[j][nN].childrenNumbers[3];
			}
		}

		//	Assign children of parent's second neighbor
		{
			/************************************/
			/*	 ______			  				*/
			/*	|	   |	  					*/
			/*	|  **  |						*/
			/*	|______|	   _____________	*/
			/*				  |	     |	    |	*/
			/*				  |  I5  |  O8  |	*/
			/*				  |______|______|	*/
			/*				  |	     |	    |	*/
			/*				  |  I4  |  O7  |	*/
			/*				  |______|______|	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[2];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[4]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[7]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[8]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[5]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's third neighbor
		{
			/************************************/
			/*				   _____________	*/
			/*				  |	     |	    |	*/
			/*				  |  I7  |  O10 |	*/
			/*	 ______		  |______|______|	*/
			/*	|	   |	  |	     |	    |	*/
			/*	|  **  |	  |  I6  |  O9  |	*/
			/*	|______|	  |______|______|	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[3];
			if (nN!=-1) {
				tree[nL][nC].innerNumbers[6]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[9]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[10]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[7]	=	tree[j][nN].childrenNumbers[3];
			}

		}

		//	Assign children of parent's fourth neighbor
		{
			/************************************/
			/*				   _____________	*/
			/*				  |	     |	    |	*/
			/*				  |  O13 |  O12 |	*/
			/*				  |______|______|	*/
			/*				  |	     |	    |	*/
			/*		    	  |  I8  |  O11 |	*/
			/*		    	  |______|______|	*/
			/*									*/
			/*									*/
			/*	 ______							*/
			/*  |      |						*/
			/*  |  **  |						*/
			/*  |______|						*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[4];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[8]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[11]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[12]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[13]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  O15 |  O14 |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  I10 |  I9  |		*/
			/*	|______|______|		*/
			/*						*/
			/*						*/
			/*	 ______				*/
			/*  |	   |			*/
			/*	|  **  |			*/
			/*	|______|			*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[10]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[9]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[14]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[15]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/****************************/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  O17 |  O16 |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I12 |  I11 |			*/
			/*	|______|______|			*/
			/*							*/
			/*							*/
			/*				   ______	*/
			/*  			  |		 |	*/
			/*				  |	 **  |	*/
			/*				  |______|	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[6];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[12]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[11]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[16]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[17]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/****************************/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I13 |  N6  |			*/
			/*	|______|______|______	*/
			/*  |	   |	  |		 |	*/
			/*	|  I14 |  N7  |	 **  |	*/
			/*	|______|______|______|	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[7];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[14]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[7]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].neighborNumbers[6]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[13]	=	tree[j][nN].childrenNumbers[3];				
			}
		}
	}

	//	Assigns the interactions for child1 of a box
	void assign_Child1_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k+1;
		int nN;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  N6  |  N5  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N7  |  **  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[7]	=	nC-1;
			tree[nL][nC].neighborNumbers[5]	=	nC+1;
			tree[nL][nC].neighborNumbers[6]	=	nC+2;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/************************************/
			/*				   		  ______	*/
			/*				  	     |		|	*/
			/*				         |	**  |	*/
			/*	 _____________       |______|  	*/
			/*	|	   |	  |					*/
			/*	|  O22 |  I15 |					*/
			/*	|______|______|					*/
			/*	|	   |	  |					*/
			/*	|  O23 |  I0  |					*/
			/*	|______|______|					*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[0];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[23]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[15]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[22]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 		______		  	*/
			/*		   |	  |			*/
			/*	       |  **  |			*/
			/*	 ______|______|			*/
			/*	|	   |	  |			*/
			/*	|  N0  |  N1  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I1  |  I2  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].neighborNumbers[1]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].neighborNumbers[0]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's second neighbor
		{
			/****************************/
			/*	 ______		  			*/
			/*	|	   |				*/
			/*	|  **  |	  			*/
			/*	|______|_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  N2  |  I5  |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*	       |  I3  |  I4  |	*/
			/*		   |______|______|	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[2];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[4]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[5]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].neighborNumbers[2]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's third neighbor
		{
			/****************************/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  N4  |	 I7	 |  */
			/*	 ______|______|______|	*/
			/*	|	   |	  |	     |	*/
			/*	|  **  |  N3  |  I6  |  */
			/*	|______|______|______| 	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[3];
			if (nN != -1){
				tree[nL][nC].neighborNumbers[3]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[6]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[7]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].neighborNumbers[4]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fourth neighbor
		{
			/****************************/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  O14 |  O13 |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*		   |  I9  |  I8  |	*/
			/*		   |______|______|	*/
			/*				  			*/
			/*				  			*/
			/*	 ______					*/
			/*	|	   |				*/
			/*  |  **  |				*/
			/*  |______|				*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[4];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[9]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[8]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[13]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[14]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  O16 |  O15 |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  I11 |  I10 |		*/
			/*	|______|______|		*/
			/*						*/
			/*						*/
			/*		    ______		*/
			/* 		   |	  |		*/
			/*		   |  **  |		*/
			/*		   |______|		*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[11]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[10]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[15]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[16]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/************************************/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O18 |  O17 |					*/
			/*	|______|______|					*/
			/*	|	   |	  |					*/
			/*	|  O19 |  I12 |					*/
			/*	|______|______|					*/
			/*									*/
			/*									*/
			/*				   		  ______	*/
			/*  			  		 |		|	*/
			/*				  		 |	** 	|	*/
			/*				  		 |______|	*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[6];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[19]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[12]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[17]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[18]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/************************************/
			/*									*/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O20 |  I13 |					*/
			/*	|______|______|		  ______	*/
			/*  |	   |	  |		 |		|	*/
			/*	|  O21 |  I14 |	 	 |	**  |	*/
			/*	|______|______|		 |______|	*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[7];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[21]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[14]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[13]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[20]	=	tree[j][nN].childrenNumbers[3];				
			}
		}
	}

	//	Assigns the interactions for child2 of a box
	void assign_Child2_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k+2;
		int nN;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  N7  |  **  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N0  |  N1  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[0]	=	nC-2;
			tree[nL][nC].neighborNumbers[1]	=	nC-1;
			tree[nL][nC].neighborNumbers[7]	=	nC+1;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/************************************/
			/*				   		  ______	*/
			/*				  	     |		|	*/
			/*				         |	**  |	*/
			/*				         |______|  	*/
			/*									*/
			/*									*/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O23 |  I0  |					*/
			/*	|______|______|					*/
			/*	|	   |	  |					*/
			/*	|  O0  |  O1  |					*/
			/*	|______|______|					*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[0];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[0]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[1]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[23]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 		______		  	*/
			/*		   |	  |			*/
			/*	       |  **  |			*/
			/*	 	   |______|			*/
			/*							*/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I1  |  I2  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  O2  |  O3  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[2]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[3]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's second neighbor
		{
			/****************************/
			/*	 ______		  			*/
			/*	|	   |				*/
			/*	|  **  |	  			*/
			/*	|______|				*/
			/*							*/
			/*							*/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  I3  |  I4  |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*	       |  O4  |  O5  |	*/
			/*		   |______|______|	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[2];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[4]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[5]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[4]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's third neighbor
		{
			/****************************/
			/*	 ____________________	*/
			/*	|	   |	  |	     |	*/
			/*	|  **  |  N3  |	 I6	 |  */
			/*	|______|______|______|	*/
			/*		   |	  |	     |	*/
			/*		   |  N2  |  I5  |  */
			/*		   |______|______| 	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[3];
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[2]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[5]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[6]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].neighborNumbers[3]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fourth neighbor
		{
			/****************************/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  I9  |  I8  |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*		   |  N4  |  I7  |	*/
			/*	 ______|______|______|	*/
			/*	|	   |	  			*/
			/*	|  **  |	  			*/
			/*	|______|	  			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[4];
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[4]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[7]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[8]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[9]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  I11 |  I10 |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N6  |  N5  |		*/
			/*	|______|______|		*/
			/*		   |	  |		*/
			/*		   |  **  |		*/
			/*		   |______|		*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[6]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[5]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[10]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[11]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/************************************/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O19 |  I12 |					*/
			/*	|______|______|					*/
			/*	|	   |	  |					*/
			/*	|  O20 |  I13 |					*/
			/*	|______|______|		  ______	*/
			/*  			  		 |		|	*/
			/*				  		 |	** 	|	*/
			/*				  		 |______|	*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[6];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[20]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[13]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[12]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[19]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/************************************/
			/*									*/
			/*	 _____________		  ______	*/
			/*	|	   |	  |		 |	    |	*/
			/*	|  O21 |  I14 |		 |	**	|	*/
			/*	|______|______|		 |______|	*/
			/*  |	   |	  |		 			*/
			/*	|  O22 |  I15 |	 	 			*/
			/*	|______|______|		 			*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[7];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[22]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[15]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[14]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[21]	=	tree[j][nN].childrenNumbers[3];				
			}
		}
	}

	//	Assigns the interactions for child3 of a box
	void assign_Child3_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k+3;
		int nN;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  **  |  N3  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N1  |  N2  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[1]	=	nC-3;
			tree[nL][nC].neighborNumbers[2]	=	nC-2;
			tree[nL][nC].neighborNumbers[3]	=	nC-1;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/****************************/
			/*				   ______	*/
			/*				  |		 |	*/
			/*				  |	 **  |	*/
			/*				  |______|  */
			/*							*/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I0  |  I1  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  O1  |  O2  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[0];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[1]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[2]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 ______		  			*/
			/*	|	   |				*/
			/*	|  **  |				*/
			/*	|______|				*/
			/*							*/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I2  |  I3  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  O3  |  O4  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[3]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[4]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's second neighbor
		{
			/************************************/
			/*	 ______		  					*/
			/*	|	   |						*/
			/*	|  **  |	  					*/
			/*	|______|						*/
			/*									*/
			/*									*/
			/*				   _____________	*/
			/*		   		  |	     |	    |	*/
			/*		   		  |  I4  |  O7  |	*/
			/*		   		  |______|______|	*/
			/*		   		  |	  	 |	    |	*/
			/*	       		  |  O5  |  O6  |	*/
			/*		   		  |______|______|	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[2];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[5]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[6]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[7]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[4]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's third neighbor
		{
			/************************************/
			/*	 ______		   _____________	*/
			/*	|	   |	  |	     |		|	*/
			/*	|  **  |      |	 I6	 |  O9	|	*/
			/*	|______|	  |______|______|	*/
			/*		   		  |	  	 |	    |	*/
			/*		   		  |  I5  |  O8  |  	*/
			/*		   		  |______|______| 	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[3];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[5]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[8]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[9]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[6]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fourth neighbor
		{
			/************************************/
			/*				   _____________	*/
			/*		   		  |	  	 |	    |	*/
			/*		   		  |  I8  |  O11 |	*/
			/*		   		  |______|______|	*/
			/*		   		  |	  	 |	    |	*/
			/*		   		  |  I7  |  O10 |	*/
			/*	 ______	      |______|______|	*/
			/*	|	   |	  					*/
			/*	|  **  |	  					*/
			/*	|______|	  					*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[4];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[7]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[10]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[11]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[8]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  I10 |  I9  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N5  |  N4  |		*/
			/*	|______|______|		*/
			/*	|	   |			*/
			/*	|  **  |			*/
			/*	|______|			*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[5]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[4]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[9]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[10]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/****************************/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I12 |  I11 |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I13 |  N6  |			*/
			/*	|______|______|______	*/
			/*  			  |		 |	*/
			/*				  |	 **  |	*/
			/*				  |______|	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[6];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[13]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[6]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[11]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[12]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/****************************/
			/*							*/
			/*	 ____________________	*/
			/*	|	   |	  |		 |	*/
			/*	|  I14 |  N7  |	 **	 |	*/
			/*	|______|______|______|	*/
			/*  |	   |	  |		 	*/
			/*	|  I15 |  N0  |	 	 	*/
			/*	|______|______|		 	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[7];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[15]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[0]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].neighborNumbers[7]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[14]	=	tree[j][nN].childrenNumbers[3];				
			}
		}
	}

	//	Assigns the interactions for the children of a box
	void assign_Box_Interactions_LFR(int j, int k) {
		assign_Child0_Interaction(j,k);
		assign_Child1_Interaction(j,k);
		assign_Child2_Interaction(j,k);
		assign_Child3_Interaction(j,k);
	}

	//	Assigns the interactions for the children all boxes at a given level
	void assign_Level_Interactions_LFR(int j) {
		#pragma omp parallel for
		for (int k=0; k<nBoxesPerLevel[j]; ++k) {
			assign_Box_Interactions_LFR(j,k);
		}
	}

	//	Assigns the interactions for the children all boxes in the tree
	void assign_Tree_Interactions_LFR() {
		for (int j=0; j<nLevels; ++j) {
			assign_Level_Interactions_LFR(j);
		}
	}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	double find_distance(int j, int k1, int k2) {
		pts2D r1 = tree[j][k1].center;
		pts2D r2 = tree[j][k2].center;
		return	sqrt((r1.x-r2.x)*(r1.x-r2.x) + (r1.y-r2.y)*(r1.y-r2.y));
	}


	void assign_Child_Interaction(int c, int j, int k) {
	// j: level
	// k: parent box
		//int nBoxesApart	=	Kwave_no*boxRadius[j+1]/A;
		//double distCriterion	=	nBoxesApart*boxRadius[j+1];
		//if (c==0 && j==1 && k==0) {
		//	cout << "nBoxesApart: " << nBoxesApart << endl;
		//}
		int pnn; // parent neighbor number
		for (int n=0; n<tree[j][k].neighborNumbersHFR.size(); ++n) {
			//children of neighbors of parent which are not neighbors to child=IL
			pnn = tree[j][k].neighborNumbersHFR[n];
			for (int nc=0; nc<4; ++nc) { //children of parents neighbors
				double dist	=	find_distance(j+1, 4*k+c, 4*pnn+nc); //j+1, pnn*4+nc //children of parent's neighbor box
									//j+1, k*4+c //current box of interest
				if (j+1 >= level_LFR) { //LFR
					if (dist <= sqrt(8.0)*boxRadius[j+1]) {
						tree[j+1][4*k+c].neighborNumbersHFR.push_back(4*pnn+nc);
					}
					else {
						tree[j+1][4*k+c].InteractionList.push_back(4*pnn+nc);
					}
				}
				else { //HFR
					if (dist <= std::max(sqrt(8.0)*boxRadius[j+1], Kwave_no*boxRadius[j+1]*boxRadius[j+1]/A)) {
					//if (dist <= std::max(sqrt(8.0)*boxRadius[j+1], distCriterion)) {
						tree[j+1][4*k+c].neighborNumbersHFR.push_back(4*pnn+nc);
					}
					else {
						tree[j+1][4*k+c].InteractionList.push_back(4*pnn+nc);
					}
				}				
			}
		}
	}


	//	Assigns the interactions for the children of a box
	void assign_Box_Interactions(int j, int k) {
		for (int c=0; c<4; ++c) {
			assign_Child_Interaction(c,j,k);
		}
	}

	//	Assigns the interactions for the children all boxes at a given level
	void assign_Level_Interactions(int j) {
		//#pragma omp parallel for
		for (int k=0; k<nBoxesPerLevel[j]; ++k) {
			assign_Box_Interactions(j,k);
		}
	}

	//	Assigns the interactions for the children all boxes in the tree
	void assign_Tree_Interactions() {
		//j=0, no neighbors, no IL
		//j=1, no IL, neighbors yes
		//neighbor includes self
		int j = 1;
		for (int c=0; c<4; ++c) {
			for (int n=0; n<4; ++n) {
				tree[j][c].neighborNumbersHFR.push_back(n);

			}
		}
		for (j=1; j<nLevels; ++j) {
			assign_Level_Interactions(j);
		}
/*
		for (j=1; j<=nLevels; ++j) {
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				cout << "j: " << j << "	k: " << k << "	IL: " << tree[j][k].InteractionList.size() << "		NN: " << tree[j][k].neighborNumbers.size() << endl;
				cout << "IL" << endl;
				for (int il=0; il<tree[j][k].InteractionList.size(); ++il) {
					cout << tree[j][k].InteractionList[il] << ",";
				}
				cout << endl;
				cout << "NN: " << endl;
				for (int il=0; il<tree[j][k].neighborNumbers.size(); ++il) {
					cout << tree[j][k].neighborNumbers[il] << ",";
				}
				cout << endl<< endl;
			}
		}
*/
	}

	void createCones() {
	//when to start making cones
		int J;
		ConeAperture.push_back(2.0*PI);// dummy initialisation
		ConeAperture.push_back(2.0*PI);
		nCones.push_back(0);
		nCones.push_back(0);
		for (J=2; J<=nLevels; ++J) {//no cones exist for level 0 and 1
			int k;
			for (k=0; k<nBoxesPerLevel[J]; ++k) {
				if (tree[J][k].InteractionList.size() > 0) { //requirement of cones starts
					break;
				}
			}
			if (k<nBoxesPerLevel[J]) {
				break;
			}
			ConeAperture.push_back(2*PI);
			nCones.push_back(0);
		}
		level_FarField = J;
		cout << "level_FarField: " << level_FarField << endl;
		int j = J-1;
		/*double ConeAperture_double	=	A/Kwave_no/boxRadius[j];
		double nCones_double	=	2*PI/ConeAperture[j];
		nCones[j]	=	pow(2,ceil(log(nCones_double)/log(2)));
		ConeAperture[j]	=	2*PI/nCones[j];
		*/

		//make cones
		//when to stop making cones: whichever happens earlier-- 
		//a) j=level_LFR
		//b) ConeAperture[j] >= A/B
		

		/*
		cout << "nLevels: " << nLevels << endl;
		cout << "level_LFR: " << level_LFR << endl;
		cout << "level_FarField: " << level_FarField << endl;
		cout << "A: " << A << endl;
		cout << "B: " << B << endl;
		*/		

		//ConeAperture[j]	=	PI/8.0/pow(2.0,level_LFR-level_FarField);
		ConeAperture[j]	=	2.0*PI/nCones_LFR/pow(2.0,level_LFR-level_FarField);
		nCones[j]	=	round(2*PI/ConeAperture[j]);
		for (j=J; j<level_LFR; ++j) {
			nCones.push_back(nCones[j-1]/2);
			ConeAperture.push_back(ConeAperture[j-1]*2);
			//cout << "j: " << j << "	nCones: " << nCones[j-1]/2 << "	ConeAperture: " << ConeAperture[j-1]*2 << endl;
			//if (ConeAperture[j] >= A/B) break;
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				for (int c=0; c<nCones[j]; ++c) {
					FMM2DCone cone;
					cone.coneNumber = c;
					cone.parentNumber = cone.coneNumber/2;
					for (int l=0; l<2; ++l) {
						cone.childrenNumbers[l]	=	2*c+l;
					}
					cone.direction.x	=	std::cos(ConeAperture[j]/2.0 + c*ConeAperture[j]);
					cone.direction.y	=	std::sin(ConeAperture[j]/2.0 + c*ConeAperture[j]);
					tree[j][k].ConeTree.push_back(cone);
				}
			}
		}
	}

	
	void assign_Center_Location() {
		int J, K;
		tree[0][0].center.x	=	0.0;
		tree[0][0].center.y	=	0.0;
		for (int j=0; j<nLevels; ++j) {
			J	=	j+1;
			double shift	=	0.5*boxRadius[j];
			#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				K	=	4*k;
				tree[J][K].center.x		=	tree[j][k].center.x-shift;
				tree[J][K+1].center.x	=	tree[j][k].center.x+shift;
				tree[J][K+2].center.x	=	tree[j][k].center.x+shift;
				tree[J][K+3].center.x	=	tree[j][k].center.x-shift;

				tree[J][K].center.y		=	tree[j][k].center.y-shift;
				tree[J][K+1].center.y	=	tree[j][k].center.y-shift;
				tree[J][K+2].center.y	=	tree[j][k].center.y+shift;
				tree[J][K+3].center.y	=	tree[j][k].center.y+shift;
			}
		}
		#pragma omp parallel for
		for (int j=level_FarField; j<=nLevels; ++j) {
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				tree[j][k].chebNodes	=	shift_scale_Cheb_Nodes(tree[j][k].center.x, tree[j][k].center.y, boxRadius[j]);
			}
		}
	}

	void assign_Leaf_Charges() {
		#pragma omp parallel for
		for (int k=0; k<nBoxesPerLevel[nLevels]; ++k) {
			tree[nLevels][k].multipoles	=	0.5*(Eigen::VectorXcd::Ones(rank)+Eigen::VectorXcd::Random(rank));
		}
	}

	double dotProduct(pts2D P1, pts2D P2) {
		return P1.x*P2.x + P1.y*P2.y;
	}


	void evaluate_LFR_M2M() {
		for (int j=nLevels-1; j>=level_LFR; --j) { // LFR
			int J	=	j+1;
			#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				int K	=	4*k;
				tree[j][k].multipoles	=	M2M[0]*tree[J][K].multipoles+M2M[1]*tree[J][K+1].multipoles+M2M[2]*tree[J][K+2].multipoles+M2M[3]*tree[J][K+3].multipoles;
			}
		}
	}
		

	void evaluate_MFR_M2M() {
		// for j < level_LFR tree[j][k].multipoles stops existing; instead we store in tree[j][k].ConeTree.multipoles
		int j	=	level_LFR-1;
		int J	=	j+1;
		for (int k=0; k<nBoxesPerLevel[j]; ++k) {
			int K	=	4*k;
			for (int c=0; c<tree[j][k].ConeTree.size(); ++c) {
				Eigen::VectorXcd leftVec = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
				   leftVec(s) = std::exp(I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[j][k].chebNodes[s]));
				}
				Eigen::VectorXcd rightVec0 = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
					rightVec0(s) = std::exp(-I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[J][K].chebNodes[s]));
				}
				Eigen::VectorXcd rightVec1 = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
					rightVec1(s) = std::exp(-I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[J][K+1].chebNodes[s]));
				}
				Eigen::VectorXcd rightVec2 = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
					rightVec2(s) = std::exp(-I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[J][K+2].chebNodes[s]));
				}
				Eigen::VectorXcd rightVec3 = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
					rightVec3(s) = std::exp(-I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[J][K+3].chebNodes[s]));
				}
				Eigen::MatrixXcd leftMat = leftVec.asDiagonal();
				Eigen::MatrixXcd rightMat0 = rightVec0.asDiagonal();
				Eigen::MatrixXcd rightMat1 = rightVec1.asDiagonal();
				Eigen::MatrixXcd rightMat2 = rightVec2.asDiagonal();
				Eigen::MatrixXcd rightMat3 = rightVec3.asDiagonal();
				tree[j][k].ConeTree[c].multipoles = leftMat*(M2M[0]*rightMat0*tree[J][K].multipoles + M2M[1]*rightMat1*tree[J][K+1].multipoles + M2M[2]*rightMat2*tree[J][K+2].multipoles + M2M[3]*rightMat3*tree[J][K+3].multipoles);
			}
		}
	}


	void evaluate_HFR_M2M() {
		// for j < level_LFR tree[j][k].multipoles stops existing; instead we store in tree[j][k].ConeTree.multipoles
	   for (int j=level_LFR-2; j>=level_FarField; --j) {
		int J	=	j+1;
		for (int k=0; k<nBoxesPerLevel[j]; ++k) {
			int K	=	4*k;
			for (int c=0; c<tree[j][k].ConeTree.size(); ++c) {
				Eigen::VectorXcd leftVec = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
				   leftVec(s) = std::exp(I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[j][k].chebNodes[s]));
				}
				Eigen::VectorXcd rightVec0 = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
					rightVec0(s) = std::exp(-I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[J][K].chebNodes[s]));
				}
				Eigen::VectorXcd rightVec1 = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
					rightVec1(s) = std::exp(-I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[J][K+1].chebNodes[s]));
				}
				Eigen::VectorXcd rightVec2 = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
					rightVec2(s) = std::exp(-I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[J][K+2].chebNodes[s]));
				}
				Eigen::VectorXcd rightVec3 = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
					rightVec3(s) = std::exp(-I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[J][K+3].chebNodes[s]));
				}
				Eigen::MatrixXcd leftMat = leftVec.asDiagonal();
				Eigen::MatrixXcd rightMat0 = rightVec0.asDiagonal();
				Eigen::MatrixXcd rightMat1 = rightVec1.asDiagonal();
				Eigen::MatrixXcd rightMat2 = rightVec2.asDiagonal();
				Eigen::MatrixXcd rightMat3 = rightVec3.asDiagonal();
				int C = c/2;
				tree[j][k].ConeTree[c].multipoles = leftMat*(M2M[0]*rightMat0*tree[J][K].ConeTree[C].multipoles + M2M[1]*rightMat1*tree[J][K+1].ConeTree[C].multipoles + M2M[2]*rightMat2*tree[J][K+2].ConeTree[C].multipoles + M2M[3]*rightMat3*tree[J][K+3].ConeTree[C].multipoles);
			}
		}
	  }
	}


	void evaluate_All_M2M() {
		evaluate_LFR_M2M();
		evaluate_MFR_M2M();
		evaluate_HFR_M2M();
	}
	

	//	Assemble FMM Operators
	void assemble_Operators_FMM(int j) {
		std::vector<pts2D> shiftedChebNodes;
		std::vector<pts2D> BaseBoxChebNodes;
		//	Assigning Base Box Nodes
		for (int k=0; k<rank; ++k) {
			pts2D temp;
			temp.x	=	boxRadius[j]*standardChebNodes[k].x;
			temp.y	=	boxRadius[j]*standardChebNodes[k].y;
			BaseBoxChebNodes.push_back(temp);
		}
		//	Assemble Outer Interactions
		for (int l=0; l<6; ++l) {
			pts2D temp;
			temp.x	=	boxRadius[j]*2*(l-3);
			temp.y	=	boxRadius[j]*2*(-3);
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(temp.x, temp.y, boxRadius[j]);
			obtain_Desired_Operator(BaseBoxChebNodes, shiftedChebNodes, M2LOuter[l]);
		}
		for (int l=0; l<6; ++l) {
			pts2D temp;
			temp.x	=	boxRadius[j]*2*(3);
			temp.y	=	boxRadius[j]*2*(l-3);
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(temp.x, temp.y, boxRadius[j]);
			obtain_Desired_Operator(BaseBoxChebNodes, shiftedChebNodes, M2LOuter[l+6]);
		}
		for (int l=0; l<6; ++l) {
			pts2D temp;
			temp.x	=	boxRadius[j]*2*(3-l);
			temp.y	=	boxRadius[j]*2*(3);
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(temp.x, temp.y, boxRadius[j]);
			obtain_Desired_Operator(BaseBoxChebNodes, shiftedChebNodes, M2LOuter[l+12]);
		}
		for (int l=0; l<6; ++l) {
			pts2D temp;
			temp.x	=	boxRadius[j]*2*(-3);
			temp.y	=	boxRadius[j]*2*(3-l);
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(temp.x, temp.y, boxRadius[j]);
			obtain_Desired_Operator(BaseBoxChebNodes, shiftedChebNodes, M2LOuter[l+18]);
		}
		//	Assemble Inner Interactions
		for (int l=0; l<4; ++l) {
			pts2D temp;
			temp.x	=	boxRadius[j]*2*(l-2);
			temp.y	=	boxRadius[j]*2*(-2);
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(temp.x, temp.y, boxRadius[j]);
			obtain_Desired_Operator(BaseBoxChebNodes, shiftedChebNodes, M2LInner[l]);
		}
		for (int l=0; l<4; ++l) {
			pts2D temp;
			temp.x	=	boxRadius[j]*2*(2);
			temp.y	=	boxRadius[j]*2*(l-2);
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(temp.x, temp.y, boxRadius[j]);
			obtain_Desired_Operator(BaseBoxChebNodes, shiftedChebNodes, M2LInner[l+4]);
		}
		for (int l=0; l<4; ++l) {
			pts2D temp;
			temp.x	=	boxRadius[j]*2*(2-l);
			temp.y	=	boxRadius[j]*2*(2);
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(temp.x, temp.y, boxRadius[j]);
			obtain_Desired_Operator(BaseBoxChebNodes, shiftedChebNodes, M2LInner[l+8]);
		}
		for (int l=0; l<4; ++l) {
			pts2D temp;
			temp.x	=	boxRadius[j]*2*(-2);
			temp.y	=	boxRadius[j]*2*(2-l);
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(temp.x, temp.y, boxRadius[j]);
			obtain_Desired_Operator(BaseBoxChebNodes, shiftedChebNodes, M2LInner[l+12]);
		}
	}
	


	//	Obtain the desired matrix
	void obtain_Desired_Operator(std::vector<pts2D>& ChebNodesX, std::vector<pts2D>& ChebNodesY, Eigen::MatrixXcd& T) {
		T	=	Eigen::MatrixXcd(rank,rank);
		for (int i=0; i<rank; ++i) {
			for (int j=0; j<rank; ++j) {
				T(i,j)	=	K->getInteraction(ChebNodesX[i], ChebNodesY[j], Kwave_no, a);
			}
		}
	}


	void evaluate_HFR_M2L() {
		for (int j=level_FarField; j<=level_LFR-1; ++j) {
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				for (int c=0; c<nCones[j]; ++c) {
					tree[j][k].ConeTree[c].locals	=	Eigen::VectorXcd::Zero(rank);
				}
				for (int l=0; l<tree[j][k].InteractionList.size(); ++l) {
					int b = tree[j][k].InteractionList[l];
					double arg = atan2(tree[j][b].center.y-tree[j][k].center.y, tree[j][b].center.x-tree[j][k].center.x);
					arg = fmod(arg+2*PI, 2*PI);
					int c = int(arg/ConeAperture[j]);
					//if (k==0 && j==level_FarField) {
					//	cout << "nCones[j]: " << nCones[j] << "	arg: " << arg << "	b: " << b << "	c: " << c << endl;
					//}
					
					Eigen::MatrixXcd M2L;
					obtain_Desired_Operator(tree[j][k].chebNodes, tree[j][b].chebNodes, M2L);
					tree[j][k].ConeTree[c].locals+=M2L*tree[j][b].ConeTree[c].multipoles;
				}
			}
		}
	}

	void evaluate_HFR_M2L_ACA() {
		for (int j=level_FarField; j<=level_LFR-1; ++j) {
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				for (int c=0; c<nCones[j]; ++c) {
					tree[j][k].ConeTree[c].locals	=	Eigen::VectorXcd::Zero(rank);
					std::vector<pts2D> ChebyCharges2;
					for (int l=0; l<tree[j][k].InteractionList.size(); ++l) {
						int b = tree[j][k].InteractionList[l];
						double arg = atan2(tree[j][b].center.y-tree[j][k].center.y, tree[j][b].center.x-tree[j][k].center.x);
						arg = fmod(arg+2*PI, 2*PI);
						int b_c = int(arg/ConeAperture[j]);
						if (c==b_c)
							ChebyCharges2.insert( ChebyCharges2.end(), tree[j][b].chebNodes.begin(), tree[j][b].chebNodes.end() );
						
					}
					Eigen::MatrixXcd U;
					Eigen::MatrixXcd V;
					Eigen::VectorXd K12;
					int rankACA;
					double tol_ACA = 1e-7;
					//if (j==level_FarField && k==0 && c==0) {
				   if (ChebyCharges2.size() > 0) {
					ACA(tree[j][k].chebNodes, ChebyCharges2, U, V, K12, tol_ACA, rankACA);
					Eigen::MatrixXcd Kbig	=	Eigen::MatrixXcd::Zero(tree[j][k].chebNodes.size(),ChebyCharges2.size());
					for (int s=0; s<tree[j][k].chebNodes.size(); ++s) {
						for (int d=0; d<ChebyCharges2.size(); ++d) {
							Kbig(s,d)	=	K->getInteraction(tree[j][k].chebNodes[s], ChebyCharges2[d], Kwave_no, a);
						}
					}
					Eigen::MatrixXcd K12Diag = K12.asDiagonal();
				//	cout << "U.rows(): " << U.rows() << "	U.cols(): " << U.cols() << endl;
				//	cout << "K12Diag.rows(): " << K12Diag.rows() << "	K12Diag.cols(): " << K12Diag.cols() << endl;
				//	cout << "V.rows(): " << V.rows() << "	V.cols(): " << V.cols() << endl;
					double errACA = (U*K12Diag*V-Kbig).norm();
					cout << "j: " << j << "	k: " << k << "	c: " << c <<  "	Initial size: " << ChebyCharges2.size() << "	Compressed size: " << rankACA << "	tol_ACA: " << tol_ACA << endl;
				    }
				}
			}
		}
	}

	void evaluate_MFR_M2L() {	
		int j =	level_LFR;
		int J = level_LFR-1;
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				std::vector<int> MFR_IL;
				int K = k/4;
				for (int n=0; n<tree[J][K].neighborNumbersHFR.size(); ++n) {
					int pnn = tree[J][K].neighborNumbersHFR[n];
					for (int nc=0; nc<4; ++nc) {
						double dist	=	find_distance(j, k, 4*pnn+nc); //j+1, pnn*4+nc //children of parent's neighbor box
						int xBoxes = fabs(tree[j][k].center.x-tree[j][4*pnn+nc].center.x)/(2*boxRadius[j]);
						int yBoxes = fabs(tree[j][k].center.y-tree[j][4*pnn+nc].center.y)/(2*boxRadius[j]);
						//if(k==0)
						//	cout << "k: " << 4*pnn+nc << "	xBoxes: " << xBoxes << "	yBoxes: " << yBoxes << " k_center.x: " << tree[j][k].center.x << "	pnn_center.x: " << tree[j][4*pnn+nc].center.x << endl;
						if (xBoxes>3 || yBoxes>3) {					
							MFR_IL.push_back(4*pnn+nc);
							//cout << "k: " << 4*pnn+nc << "	MFR_M2L" << endl;
						}
						/*else 
							if (k==0)
								cout << "k: " << 4*pnn+nc << "	discarded" << endl;
						*/
					}
				}
		
				tree[j][k].locals	=	Eigen::VectorXcd::Zero(rank);
				for (int l=0; l<MFR_IL.size(); ++l) {
					int b = MFR_IL[l];
					Eigen::MatrixXcd M2L;
					obtain_Desired_Operator(tree[j][k].chebNodes, tree[j][b].chebNodes, M2L);
					tree[j][k].locals+=M2L*tree[j][b].multipoles;
				}
				
/*				
			//	cout << endl << "k: " << k << endl;
			//	cout << "MFR_IL: " << MFR_IL.size() << endl;
				for (int l=0; l<MFR_IL.size(); ++l) {
			//		cout << MFR_IL[l] << ",";
				}
				int s=0;
			//	cout << endl << "nInner: " << endl;
				for (int l=0; l<16; ++l) {
						int nInner	=	tree[j][k].innerNumbers[l];
						if (nInner>-1) {
			//				cout << nInner << ",";
							s=s+1;
						}
				}
				int z=0;
			//	cout << endl << "nOuter: " << endl;
				for (int l=0; l<24; ++l) {
						int nOuter	=	tree[j][k].outerNumbers[l];
						if (nOuter>-1) {
			//				cout << nOuter << ",";
							z=z+1;
						}
				}
				if (MFR_IL.size() != s+z)
					cout << "problem" << endl;
*/
			}
					
	}


	void evaluate_LFR_M2L() {
		//for (int j=level_LFR+1; j<=nLevels; ++j) {
		for (int j=level_LFR; j<=nLevels; ++j) {
			assemble_Operators_FMM(j);
			//#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				tree[j][k].locals	=	Eigen::VectorXcd::Zero(rank);
					//	Inner well-separated clusters
					for (int l=0; l<16; ++l) {
						int nInner	=	tree[j][k].innerNumbers[l];
						if (nInner>-1) {
							//if (j==level_LFR && k==0)
							//	cout<< "nInner: " << nInner << endl;
							tree[j][k].locals+=M2LInner[l]*tree[j][nInner].multipoles;
						}
					}
					//	Outer well-separated clusters
					for (int l=0; l<24; ++l) {
						int nOuter	=	tree[j][k].outerNumbers[l];
						if (nOuter>-1) {
							//if (j==level_LFR && k==0)
							//	cout<< "nOuter: " << nOuter << endl;
							tree[j][k].locals+=M2LOuter[l]*tree[j][nOuter].multipoles;
						}
					}
			}
		}
	}


	// ACA
	// ChebyCharges1: Line 1 charge locations (N1)
	// ChebyCharges2: Line 2 charge locations (N2)
	//when the matrix to be approximated is from a kernel which takes ChebyCharges1, ChebyCharges2 as the inputs
	void ACA(std::vector<pts2D> ChebyCharges1, std::vector<pts2D> ChebyCharges2, Eigen::MatrixXcd& U, Eigen::MatrixXcd& V, Eigen::VectorXd& K12, double tol_ACA, int &rankACA) {
		// K12 is a diagonal matrix, so we are storing it as a vector
		int N1	=	ChebyCharges1.size();
		int N2	=	ChebyCharges2.size();
		U	=	Eigen::MatrixXcd(N1,1);
		V	=	Eigen::MatrixXcd(1,N2);
		K12	=	Eigen::VectorXd(1);
		std::ptrdiff_t col_index;
		std::ptrdiff_t row_index;
		Eigen::RowVectorXd row;
		//Eigen::RowVectorXd rowAbs;
		Eigen::VectorXd col;
		//Eigen::VectorXd colAbs;
		Eigen::RowVectorXcd v;
		Eigen::VectorXcd u;
		Eigen::RowVectorXcd row_dumb;
		Eigen::VectorXcd col_dumb;		
		// the matrix to be approximated has dimensions N1xN2
			
		// initialisation
		int k=0;
		Eigen::MatrixXcd R	=	Eigen::MatrixXcd::Zero(N1,N2);
		std::vector<double> row_list;
		std::vector<double> col_list;
		row_index=0;
		row_list.push_back(row_index);
		for (int g=0; g<N2; ++g) {
			R(row_index,g)	=	K->getInteraction(ChebyCharges1[row_index], ChebyCharges2[g], Kwave_no, a);
		}
		//R.row(i)=row;
		row	=	R.row(row_index).cwiseAbs();
		row.maxCoeff(&col_index);
		col_list.push_back(int(col_index));
		v=R.row(row_index)/R(row_index,int(col_index));
		for (int g=0; g<N1; ++g) {
			R(g,col_index)	=	K->getInteraction(ChebyCharges1[g], ChebyCharges2[col_index], Kwave_no, a);
		}
		u	=	R.col(col_index);
		U.col(k)	=	u;
		V.row(k)	=	v;
		K12(k)		=	1.0;
		double normS	=	0.0;
		col_dumb	=	R.col(col_index);
		for (int e=0; e<row_list.size(); ++e) {
			col_dumb(row_list[e])	=	0.0;
		}		
		col	=	col_dumb.cwiseAbs();
		col.maxCoeff(&row_index);
		row_list.push_back(int(row_index));
		// iterations
		while (u.norm()*v.norm() >= tol_ACA*normS) {
		//while (k < rankACA-1) {
			++k;
			for (int g=0; g<N2; ++g) {
				R(row_index,g)	=	K->getInteraction(ChebyCharges1[row_index], ChebyCharges2[g], Kwave_no, a);
			}
			for (int l=0; l<=k-1; ++l){
				R.row(row_index)	-=	U(row_index,l)*V.row(l);
			}
			row_dumb	=	R.row(row_index);
			for (int e=0; e<col_list.size(); ++e) {
				row_dumb(col_list[e])	=	0.0;
			}		
			row	=	row_dumb.cwiseAbs();
			row.maxCoeff(&col_index);
			col_list.push_back(int(col_index));
			v=R.row(row_index)/R(row_index,col_index);
			for (int g=0; g<N1; ++g) {
				R(g,col_index)	=	K->getInteraction(ChebyCharges1[g], ChebyCharges2[col_index], Kwave_no, a);
			}
			for (int l=0; l<=k-1; ++l){
				R.col(col_index)	-=	V(l,col_index)*U.col(l);
			}
	
			u	=	R.col(col_index);
			U.conservativeResize(U.rows(), U.cols()+1);
			U.col(U.cols()-1) = u;
			V.conservativeResize(V.rows()+1, V.cols());
			V.row(V.rows()-1) = v;
			K12.conservativeResize(K12.size()+1);
			
			//U.col(k)	=	u;
			//V.row(k)	=	v;
			K12(k)		=	1.0;
			normS		=	pow(normS,2)+pow(u.norm()*v.norm(),2);
			for (int l=0; l<=k-1; ++l){
				double temp1 = ((U.col(l).adjoint())*u).norm();
				double temp2 = (V.row(l)*v.adjoint()).norm();
				normS	+=	2*temp1*temp2;
			}
			normS	=	sqrt(normS);
			col_dumb	=	R.col(col_index);
			for (int e=0; e<row_list.size(); ++e) {
				col_dumb(row_list[e])	=	0.0;
			}		
			col	=	col_dumb.cwiseAbs();
			col.maxCoeff(&row_index);
			row_list.push_back(int(row_index));
			
		} // end while
		rankACA = k;
		//K12	=	Eigen::VectorXd::Ones(rankACA);


	} // end ACA

/*
	void evaluate_LFR_M2L() {	
		for (int j=level_LFR; j<=nLevels; ++j) {
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				tree[j][k].locals	=	Eigen::VectorXcd::Zero(rank);
				for (int l=0; l<tree[j][k].InteractionList.size(); ++l) {
					int b = tree[j][k].InteractionList[l];
					Eigen::MatrixXcd M2L;
					obtain_Desired_Operator(tree[j][k].chebNodes, tree[j][b].chebNodes, M2L);
					tree[j][k].locals+=M2L*tree[j][b].multipoles;
				}
			}
		}
	}
*/

	void evaluate_All_M2L() {
		evaluate_HFR_M2L();
		evaluate_MFR_M2L();
		evaluate_LFR_M2L();
	}


	void evaluate_HFR_L2L() {
	// for j >= level_FarField interaction lists start existing
	   for (int j=level_FarField; j<=level_LFR-2; ++j) {
		int J	=	j+1;
		for (int k=0; k<nBoxesPerLevel[j]; ++k) {
			int K	=	4*k;
			for (int c=0; c<nCones[j]; ++c) {
				Eigen::VectorXcd rightVec = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
				   rightVec(s) = std::exp(-I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[j][k].chebNodes[s]));
				}
				Eigen::VectorXcd leftVec0 = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
					leftVec0(s) = std::exp(I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[J][K].chebNodes[s]));
				}
				Eigen::VectorXcd leftVec1 = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
					leftVec1(s) = std::exp(I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[J][K+1].chebNodes[s]));
				}
				Eigen::VectorXcd leftVec2 = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
					leftVec2(s) = std::exp(I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[J][K+2].chebNodes[s]));
				}
				Eigen::VectorXcd leftVec3 = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
					leftVec3(s) = std::exp(I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[J][K+3].chebNodes[s]));
				}
				Eigen::MatrixXcd rightMat = rightVec.asDiagonal();
				Eigen::MatrixXcd leftMat0 = leftVec0.asDiagonal();
				Eigen::MatrixXcd leftMat1 = leftVec1.asDiagonal();
				Eigen::MatrixXcd leftMat2 = leftVec2.asDiagonal();
				Eigen::MatrixXcd leftMat3 = leftVec3.asDiagonal();
				int C = c/2;
				tree[J][K].ConeTree[C].locals += leftMat0*(L2L[0]*(rightMat*tree[j][k].ConeTree[c].locals));
				tree[J][K+1].ConeTree[C].locals += leftMat1*(L2L[1]*(rightMat*tree[j][k].ConeTree[c].locals));
				tree[J][K+2].ConeTree[C].locals += leftMat2*(L2L[2]*(rightMat*tree[j][k].ConeTree[c].locals));
				tree[J][K+3].ConeTree[C].locals += leftMat3*(L2L[3]*(rightMat*tree[j][k].ConeTree[c].locals));	
			}
		}
	  }
	}


	void evaluate_MFR_L2L() {
	// for j = level_LFR-1
		int j = level_LFR-1;
		int J	=	j+1;
		for (int k=0; k<nBoxesPerLevel[j]; ++k) {
			int K	=	4*k;
			for (int c=0; c<tree[j][k].ConeTree.size(); ++c) {
				Eigen::VectorXcd rightVec = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
				   rightVec(s) = std::exp(-I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[j][k].chebNodes[s]));
				}
				Eigen::VectorXcd leftVec0 = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
					leftVec0(s) = std::exp(I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[J][K].chebNodes[s]));
				}
				Eigen::VectorXcd leftVec1 = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
					leftVec1(s) = std::exp(I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[J][K+1].chebNodes[s]));
				}
				Eigen::VectorXcd leftVec2 = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
					leftVec2(s) = std::exp(I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[J][K+2].chebNodes[s]));
				}
				Eigen::VectorXcd leftVec3 = Eigen::VectorXcd::Zero(rank);
				for (int s=0; s<rank; ++s) {
					leftVec3(s) = std::exp(I*Kwave_no*dotProduct(tree[j][k].ConeTree[c].direction, tree[J][K+3].chebNodes[s]));
				}
				Eigen::MatrixXcd rightMat = rightVec.asDiagonal();
				Eigen::MatrixXcd leftMat0 = leftVec0.asDiagonal();
				Eigen::MatrixXcd leftMat1 = leftVec1.asDiagonal();
				Eigen::MatrixXcd leftMat2 = leftVec2.asDiagonal();
				Eigen::MatrixXcd leftMat3 = leftVec3.asDiagonal();
				
				tree[J][K].locals += leftMat0*L2L[0]*rightMat*tree[j][k].ConeTree[c].locals;
				tree[J][K+1].locals += leftMat1*L2L[1]*rightMat*tree[j][k].ConeTree[c].locals;
				tree[J][K+2].locals += leftMat2*L2L[2]*rightMat*tree[j][k].ConeTree[c].locals;
				tree[J][K+3].locals += leftMat3*L2L[3]*rightMat*tree[j][k].ConeTree[c].locals;
			}
		}
	}


	void evaluate_LFR_L2L() {
		for (int j=level_LFR; j<nLevels; ++j) {
			int J	=	j+1;
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				int K	=	4*k;
				tree[J][K].locals+=L2L[0]*tree[j][k].locals;
				tree[J][K+1].locals+=L2L[1]*tree[j][k].locals;
				tree[J][K+2].locals+=L2L[2]*tree[j][k].locals;
				tree[J][K+3].locals+=L2L[3]*tree[j][k].locals;
			}
		}
	}


	void evaluate_All_L2L() {
		evaluate_HFR_L2L();
		evaluate_MFR_L2L();
		evaluate_LFR_L2L();
	}

/*	void evaluate_Leaf() {
		if (nLevels <2) {
			#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[nLevels]; ++k) {
				tree[nLevels][k].locals	=	Eigen::VectorXcd::Zero(rank);
			}
		}
		#pragma omp parallel for
		for (int k=0; k<nBoxesPerLevel[nLevels]; ++k) {
			for (int l=0; l<8; ++l) {
				int nNeighbor	=	tree[nLevels][k].neighborNumbers[l];
				if (nNeighbor > -1) {
					tree[nLevels][k].locals+=neighborInteraction[l]*tree[nLevels][nNeighbor].multipoles;
				}
			}
			tree[nLevels][k].locals+=selfInteraction*tree[nLevels][k].multipoles;
		}
	}
*/

	double perform_Error_Check(int nBox) {
		Eigen::VectorXcd potential	=	Eigen::VectorXcd::Zero(rank);
		for (int l1=0; l1<rank; ++l1) {
			for (int k=0; k<nBoxesPerLevel[nLevels]; ++k) {
				//lets compute only farfield
				if (std::find(tree[nLevels][nBox].neighborNumbersHFR.begin(), tree[nLevels][nBox].neighborNumbersHFR.end(), k) != tree[nLevels][nBox].neighborNumbersHFR.end()) continue;
				for (int l2=0; l2<rank; ++l2) {
					potential(l1)+=K->getInteraction(tree[nLevels][nBox].chebNodes[l1], tree[nLevels][k].chebNodes[l2], Kwave_no, a)*tree[nLevels][k].multipoles(l2);
				}
			}
		}
		Eigen::VectorXcd error(rank);
		for (int k=0; k<rank; ++k) {
			error(k)	=	(potential-tree[nLevels][nBox].locals)(k);///potential(k);
		}
		return error.norm()/potential.norm();
	}

};

#endif
