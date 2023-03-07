#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <time.h>

double** make_array(int length, int lt, int lb, int rt, int rb) {
	double** arr = (double**)malloc(sizeof(double*)*length);
	for (size_t i = 0; i < length; i++)
	{
		arr[i] = (double*)calloc(length, sizeof(double));
	}

	arr[0][0] = lt;
	arr[0][length - 1] = rt;
	arr[length - 1][0] = lb;
	arr[length - 1][length - 1] = rb;

	double stepx1 = (arr[0][length - 1] - arr[0][0]) / (length - 1);
	double stepx2 = (arr[length - 1][length - 1] - arr[length - 1][0]) / (length - 1);
	double stepyl = (arr[length - 1][0] - arr[0][0]) / (length - 1);
	double stepyr = (arr[length - 1][length - 1] - arr[0][length - 1]) / (length - 1);

	#pragma acc parallel
	{
	for (size_t i = 1; i < length - 1; i++) arr[0][i] = arr[0][i - 1] + stepx1;
	for (size_t i = 1; i < length - 1; i++) arr[length - 1][i] = arr[length - 1][i - 1] + stepx2;
	for (size_t i = 1; i < length - 1; i++) arr[i][0] = arr[i - 1][0] + stepyl;
	for (size_t i = 1; i < length - 1; i++) arr[i][length - 1] = arr[i - 1][length - 1] + stepyr;
	}

	return arr;
}

int main(int argc, char** argv) { //length iter
	double before = clock();

	int length = atoi(argv[1]);
	int g = 0;
	double error = 1;

	double** arr = make_array(length, 10, 20, 20, 30);
	
	#pragma kernels	
	for (size_t i = 1; i < length-1; i++)
	{
		for (size_t j = 1; j < length-1; j++)
		{
			arr[i][j] = (arr[i + 1][j] + arr[i - 1][j] + arr[i][j - 1] + arr[i][j + 1]) / 4;
	}
													}

	double** anew = make_array(length, 10, 20, 20, 30);

	#pragma acc data  copyin(error, length, anew[0:length][0:length], arr[0:length][0:length])
	while (error > 0.000001 && g < atoi(argv[2])) {
		if (g % 100 == 0) {
			error = 0;
			#pragma acc update device(error)
		}

		#pragma acc data present(arr, anew, error)
		#pragma acc parallel loop independent collapse(2) reduction(max:error)
		for (size_t i = 1; i < length - 1; i++)
		{
			//#pragma acc loop reduction(max:error)
			for (size_t j = 1; j < length - 1; j++)
			{
				anew[i][j] = (arr[i + 1][j] + arr[i - 1][j] + arr[i][j - 1] + arr[i][j + 1]) / 4;
				error = fmax(error, fabs(anew[i][j] - arr[i][j]));
			}
		} 

		if (g % 100 == 0) {
			#pragma acc update host(error)
		}

		double** c = anew;
		anew = arr;
		arr = c;
		g++;
	}
	printf("Last iteration: %d Error: %.6lf\n", g, error);
	free(*arr);
	free(arr);
	free(*anew);
	free(anew);

	double t = clock() - before;
	t /= CLOCKS_PER_SEC;
	printf("%lf\n", t);
	return 0;
}

