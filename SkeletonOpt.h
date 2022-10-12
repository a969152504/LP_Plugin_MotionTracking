#pragma once

struct EDGE{
	int start;
	int end;
};
struct NODE{
	double pos[3];
};

class SkeletonOpt
{
public:
	SkeletonOpt(int _modelNum=2, int _vertNum=32, int _edgeNum=31);
	~SkeletonOpt(void);

	//inputs -- all are needed to set properly before Run
	bool InsertEdge(int idx, int start, int end);
	bool InsertNode(int model, int idx, double x, double y, double z);
	bool SetLengthConstrain(int idx, double length);
	bool SetInitialVertexPosition(int idx, double x, double y, double z);
	bool SetModelVertexWeight(int model, int idx, int dim, double weight);

	//output
	bool GetVertexPosition(int idx, double &x, double &y, double &z);

	//Run
	void Run(int loop=10,bool linesearch = true);


private:
	//optimization
	bool DoOptimization(bool doLineSearch=false);

	//line search
	void _doLineSearch(double *delta);
	void _lineSearch(double* xold, const double fold, double* g, double* p, 
									 double* x, double& f, const double stpmax, bool& check);
	double _func(double* x);
	void _dfunc(double* x, double* df);

private:
    int modelNum = 0;
    int vertNum = 0;
    int edgeNum = 0;

	EDGE *edges;
	NODE **nodes;
	double *lengthConstrain;
	double ***modelVertexWeight;

	NODE *result;
};
