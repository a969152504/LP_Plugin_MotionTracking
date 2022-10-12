#include "SkeletonOpt.h"

#include <math.h>
#include <time.h>

#include <stdio.h>

#include <iostream>
#include <algorithm>

#include <QDebug>

#include <limits>
using namespace std;

SkeletonOpt::SkeletonOpt(int _modelNum, int _vertNum, int _edgeNum)
{
	modelNum=_modelNum;
	vertNum=_vertNum;
	edgeNum=_edgeNum;
    qDebug() << "model:" << modelNum << "vertNum:" << vertNum << "edgeNum:" << edgeNum;

	edges=new EDGE[edgeNum];
	lengthConstrain = new double[edgeNum];
	nodes=new NODE*[modelNum];
	for (int i=0; i<modelNum; i++)
		nodes[i]=new NODE[vertNum];
	result=new NODE[vertNum];

	modelVertexWeight = new double**[modelNum];
	for (int i=0; i<modelNum; i++){
		modelVertexWeight[i] = new double*[vertNum];
		for (int j=0; j<vertNum; j++)
			modelVertexWeight[i][j]=new double[3];
	}
}

SkeletonOpt::~SkeletonOpt(void)
{
	if (edges){
		delete [] edges; edges=0;
	}
	if (nodes){
		for (int i=0; i<modelNum; i++) delete [] nodes[i];
		delete [] nodes; nodes=0;
	}
	if (lengthConstrain){
		delete [] lengthConstrain; lengthConstrain=0;
	}
	if (result){
		delete [] result; result=0;
	}
	if (modelVertexWeight){
		for (int i=0; i<modelNum; i++){
			for (int j=0; j<vertNum; j++)
				delete [] modelVertexWeight[i][j];
			delete [] modelVertexWeight[i];
		}
		delete [] modelVertexWeight; modelVertexWeight=0;
	}
}

bool SkeletonOpt::InsertEdge(int idx, int start, int end){
	if (idx<0 || idx>=edgeNum) return false;
	if (start<0 || start>=vertNum) return false;
	if (end<0 || end>=vertNum) return false;
	EDGE &edge = edges[idx];
	edge.start=start;
	edge.end=end;
	return true;
}

bool SkeletonOpt::InsertNode(int model, int idx, double x, double y, double z){
    if (model<0 || model>=modelNum) return false;
    if (idx<0 || idx>=vertNum) return false;
	NODE &node = nodes[model][idx];
	node.pos[0]=x;
	node.pos[1]=y;
	node.pos[2]=z;
	return true;
}

bool SkeletonOpt::SetLengthConstrain(int idx, double length){
	if (idx<0 || idx>=edgeNum) return false;
	lengthConstrain[idx]=length;
	return true;
}

bool SkeletonOpt::SetInitialVertexPosition(int idx, double x, double y, double z){
	if (idx<0 || idx>=vertNum) return false;
	NODE &node = result[idx];
	node.pos[0]=x;
	node.pos[1]=y;
	node.pos[2]=z;
	return true;
}

bool SkeletonOpt::SetModelVertexWeight(int model, int idx, int dim, double weight){
	if (model<0 || model>=modelNum) return false;
	if (idx<0 || idx>=vertNum) return false;
	if (dim<0 || dim>=3) return false;

	modelVertexWeight[model][idx][dim]=weight;

	return true;
}

bool SkeletonOpt::GetVertexPosition(int idx, double &x, double &y, double &z){
	if (idx<0 || idx>=vertNum) return false;
	NODE &node = result[idx];
	x=node.pos[0];
	y=node.pos[1];
	z=node.pos[2];
	return true;
}

bool SkeletonOpt::DoOptimization(bool doLineSearch){
	int dim=vertNum*3;

	//////////////////////////////////////////////////////////////
	//calculate Lambda
	double *Lambda = new double[dim];
	for (int i=0; i<dim; i++) Lambda[i]=0;

	double *nLength = new double[edgeNum];
	for (int edx=0; edx<edgeNum; edx++){
		EDGE &edge = edges[edx];
		int start = edge.start;
		int end = edge.end;
		NODE &startNode = result[start];
		NODE &endNode = result[end];

		double conLength = lengthConstrain[edx];
		double edgeLength=0;
		double diff[3];
		for (int i=0; i<3; i++){
			double tmp = startNode.pos[i]-endNode.pos[i];
			diff[i]=tmp;
			tmp*=tmp;
			edgeLength+=tmp;
		}
		edgeLength=sqrt(edgeLength);
		nLength[edx]=edgeLength;

		if (edgeLength==0) printf("edgeLength=0!\n");

		double weightP = 2.0 * (1.0 - conLength/edgeLength);
		
		for (int i=0; i<3; i++){
			Lambda[start*3+i]+= weightP * diff[i];
			Lambda[end*3+i]-= weightP * diff[i];
		}
	}

	//Lambda * Lambda^T
	double LLT=0;
	for (int i=0; i<dim; i++)
		LLT += Lambda[i]*Lambda[i];


	//////////////////////////////////////////////////////////////
	//calculate Bx
	double *Bx = new double[dim];
	for (int vdx=0; vdx<vertNum; vdx++){
		NODE &node = result[vdx];
		double diff[3]={0};
		for (int mdx=0; mdx<modelNum; mdx++){
			NODE &targetNode = nodes[mdx][vdx];
			for (int i=0; i<3; i++){
				double weight = modelVertexWeight[mdx][vdx][i];
				diff[i]+= 2 * weight * (targetNode.pos[i]-node.pos[i]);
			}
		}
		for (int i=0; i<3; i++)
			Bx[vdx*3+i]=diff[i];
	}

	double lembda=0;
	if (LLT!=0){
		//Lembda * Bx
		double LBx=0;
		for (int i=0; i<dim; i++)
			LBx += Lambda[i]*Bx[i];

		//////////////////////////////////////////////////////////////
		//b_lembda = -C
		double b_lembda = 0;
		for (int edx=0; edx<edgeNum; edx++){
			double constrain = lengthConstrain[edx];
			double length = nLength[edx];
			double tmp = length-constrain;
			tmp*=tmp;
			b_lembda -= tmp;
		}

		//////////////////////////////////////////////////////////////
		//calculate lembda (H = 2*I)
		double constant = 1.0/(2.0);
		lembda = (constant*LBx - b_lembda)/(constant*LLT);
	}

	//////////////////////////////////////////////////////////////
	//compute delta_x
	double *delta_x = new double[dim];
	for (int i=0; i<dim; i++){
		delta_x[i] = (Bx[i]-Lambda[i]*lembda)/(2.0);
	}

	if (doLineSearch){
		//////////////////////////////////////////////////////////////
		//preform line search for result
		_doLineSearch(delta_x);
	}else{
		//////////////////////////////////////////////////////////////
		//update result
		for (int vdx=0; vdx<vertNum; vdx++){
			for (int i=0; i<3; i++)
				result[vdx].pos[i]+=delta_x[vdx*3+i];
		}
    }

	delete [] delta_x;
	delete [] nLength;
	delete [] Lambda;
	delete [] Bx;
	return true;
}

void SkeletonOpt::Run(int loop,bool linesearch){
	for (int i=0; i<loop; i++){
	        //std::cout << "Run: " << i << std::endl;
		//clock_t time = clock();
		bool linesearch1=true;
		if (i==0) linesearch1=false;
		DoOptimization(linesearch1);
	}
}




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//line search
void SkeletonOpt::_doLineSearch(double *delta)
{
	int n = vertNum*3;

	double* xold, * g, * x, * xi;
	xold = new double[n]; g = new double[n]; 
	x = new double[n]; xi = new double[n];

	for (int i=0; i<vertNum; i++)
		for (int j=0; j<3; j++)
			xold[i*3+j]=result[i].pos[j];

	for (int i=0; i<n; i++){
		xi[i]=delta[i];
	}


	double sum = 0.0;
	for(int i=0;i<n;i++) sum += xold[i]*xold[i];
	const double STPMX = 100.0;
	double stpmax = STPMX*max(sqrt(sum), double(n));
	double fold = _func(xold); _dfunc(xold, g);

	double f; bool check;
	_lineSearch(xold, fold, g, xi, x, f, stpmax, check);

	for (int i=0; i<vertNum; i++)
		for (int j=0; j<3; j++)
			result[i].pos[j] = x[i*3+j];


	delete [] xold; xold=NULL;
	delete [] g; g=NULL;
	delete [] x; x=NULL;
	delete [] xi; xi=NULL;
}

void SkeletonOpt::_lineSearch(double* xold, const double fold, double* g, double* p, 
									 double* x, double& f, const double stpmax, bool& check)
{
	const double ALF=1.0e-4;
	double TOLX = numeric_limits<double>::epsilon();
	int i;
	double a, alam, alam2 = 0.0, alamin, b, disc, f2 = 0.0;
	double rhs1, rhs2, slope, sum, temp, test, tmplam;
	
	int n = vertNum*3;
	check = false;
	sum = 0.0;
	for(i=0;i<n;i++) sum += p[i]*p[i];
	sum=sqrt(sum);
	if(sum>stpmax){
		for(i=0;i<n;i++) p[i] *= stpmax/sum;
	}
	slope = 0.0;
	for(i=0;i<n;i++) slope += g[i]*p[i];
	test = 0.0;
	for(i=0;i<n;i++){
		double temp1 = max(fabs(xold[i]),1.0);
		temp = fabs(p[i])/temp1;
		if(temp>test) test = temp;
	}
	alamin = TOLX/test;
	alam = 1.0;
	for(;;){
		for(i=0;i<n;i++){
			x[i] = xold[i]+alam*p[i];
		}
		f = _func(x);
		if(alam<alamin){
			for(i=0;i<n;i++) x[i]=xold[i];
			check = true;
			return;
		} 
		else if(f<=fold+ALF*alam*slope){
			//double value = fold+ALF*alam*slope;
			return;
		}
		else{
			if(alam==1.0){
				tmplam = -slope/(2.0*(f-fold-slope));
			}
			else{
				rhs1 = f-fold-alam*slope;
				rhs2 = f2-fold-alam2*slope;
				a = (rhs1/(alam*alam)-rhs2/(alam2*alam2))/(alam-alam2);
				b = (-alam2*rhs1/(alam*alam)+alam*rhs2/(alam2*alam2))/(alam-alam2);
				if(a==0.0){
					tmplam = -slope/(2.0*b);
				}
				else{
					disc = b*b-3.0*a*slope;
					if(disc<0.0){
						tmplam = 0.5*alam;
					}
					else if(b<=0.0){
						tmplam = (-b+sqrt(disc))/(3.0*a);
					}
					else{
						tmplam = -slope/(b+sqrt(disc));
					}
				}
				if(tmplam>0.5*alam) tmplam = 0.5*alam;
			}
		}
		alam2 = alam;
		f2 = f;
		alam = max(tmplam,0.1*alam);
	}
}

double SkeletonOpt::_func(double* x)
{
	//compute energy
	double energy = 0;
	for (int edx=0; edx<edgeNum; edx++){
		EDGE &edge = edges[edx];
		int start = edge.start;
		int end = edge.end;

		double conLength = lengthConstrain[edx];
		double edgeLength=0;
		for (int i=0; i<3; i++){
			double tmp = x[start*3+i]-x[end*3+i];
			tmp*=tmp;
			edgeLength+=tmp;
		}
		edgeLength=sqrt(edgeLength);

		double tmp = edgeLength-conLength;
		tmp*=tmp;
		energy += tmp;
	}
	return energy;
}

void SkeletonOpt::_dfunc(double* x, double* df)
{
	double *Lambda = df;
	for (int i=0; i<vertNum*3; i++) Lambda[i]=0;

	for (int edx=0; edx<edgeNum; edx++){
		EDGE &edge = edges[edx];
		int start = edge.start;
		int end = edge.end;


		double conLength = lengthConstrain[edx];
		double edgeLength=0;
		double diff[3];
		for (int i=0; i<3; i++){
			double tmp = x[start*3+i]-x[end*3+i];
			diff[i]=tmp;
			tmp*=tmp;
			edgeLength+=tmp;
		}
		edgeLength=sqrt(edgeLength);

		if (edgeLength==0) printf("(ls)edgeLength=0!\n");

		double weightP = 2.0 * (1.0 - conLength/edgeLength);
		
		for (int i=0; i<3; i++){
			Lambda[start*3+i]+= weightP * diff[i];
			Lambda[end*3+i]-= weightP * diff[i];
		}
	}
}
