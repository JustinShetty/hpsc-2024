#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using std::pow;

const auto NX = 41;
const auto NY = 41;
const auto NT = 500;
const auto NIT = 50;
constexpr auto dx = 2.0 / (NX-1);
constexpr auto dx2 = dx * dx;
constexpr auto dy = 2.0 / (NY-1);
constexpr auto dy2 = dy * dy;
const auto dt = 0.01;
const auto rho = 1.0;
const auto nu = 0.02;

double** cuda_zeros(int m, int n) {
	double** zeros;
	cudaMallocManaged(&zeros, m*sizeof(double*));
	for (int j=0; j<m; j++) {
		cudaMallocManaged(&zeros[j], n*sizeof(double));
		for (int i=0; i<n; i++) zeros[j][i] = 0.0;
	}
	return zeros;
}

void cuda_mat_free(double** mat, int m) {
	for (int j=0; j<m; j++) {
		cudaFree(mat[j]);
	}
	cudaFree(mat);
}

__global__ void kernel(const int total_threads, const int nmax, double** const u, double** const v, double** const p, double** const b) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= total_threads) return;
	auto j = tid / NX;
	auto i = tid % NX;
	if (i == 0 || i == NX-1 || j == 0 || j == NY-1) return;
	auto coordinator_tid = NX + 1;
	// use grid object to synchronize across ALL threads
	auto grid = cooperative_groups::this_grid();

	for(auto n = 0; n < nmax; n++) {
		// b
		b[j][i] = (u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy);
		b[j][i] /= dt;
		b[j][i] -= pow((u[j][i+1] - u[j][i-1]) / (2 * dx), 2);
		b[j][i] -= 2 * (u[j+1][i] - u[j-1][i]) / (2 * dy) * (v[j][i+1] - v[j][i-1]) / (2 * dx);
		b[j][i] -= pow((v[j+1][i] - v[j-1][i]) / (2 * dy), 2);
		b[j][i] *= rho;
		grid.sync();
		continue;

		// p
		for (auto it = 0; it < NIT; it++) {
			double new_p;
			new_p  = dy2 * (p[j][i+1] + p[j][i-1]);
			new_p += dx2 * (p[j+1][i] + p[j-1][i]);
			new_p -= b[j][i] * dx2 * dy2;
			new_p /= 2 * (dx2 + dy2);
			grid.sync();
			p[j][i] = new_p;
			grid.sync();

			if(tid == coordinator_tid) {
				// p[:, -1] = p[:, -2]
				for (auto j = 0; j < NY; j++) p[j][NX-1] = p[j][NX-2];
				// p[0, :] = p[1, :]
				for (auto i = 0; i < NX; i++) p[0][i] = p[1][i];
				// p[:, 0] = p[:, 1]
				for (auto j = 0; j < NY; j++) p[j][0] = p[j][1];
				// p[-1, :] = 0
				for (auto i = 0; i < NX; i++) p[NY-1][i] = 0.0;
			}
			grid.sync();
		}

		// u
		double new_u;
		new_u = u[j][i];
		new_u -= u[j][i] * dt / dx * (u[j][i] - u[j][i-1]);
		new_u -= u[j][i] * dt / dy * (u[j][i] - u[j-1][i]);
		new_u -= dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1]);
		new_u += nu * dt / dx2 * (u[j][i+1] - 2 * u[j][i] + u[j][i-1]);
		new_u += nu * dt / dy2 * (u[j+1][i] - 2 * u[j][i] + u[j-1][i]);
		grid.sync();
		u[j][i] = new_u;
		grid.sync();

		// v
		double new_v;
		new_v = v[j][i];
		new_v -= v[j][i] * dt / dx * (v[j][i] - v[j][i-1]);
		new_v -= v[j][i] * dt / dy * (v[j][i] - v[j-1][i]);
		new_v -= dt / (2 * rho * dy) * (p[j+1][i] - p[j-1][i]);
		new_v += nu * dt / dx2 * (v[j][i+1] - 2 * v[j][i] + v[j][i-1]);
		new_v += nu * dt / dy2 * (v[j+1][i] - 2 * v[j][i] + v[j-1][i]);
		grid.sync();
		v[j][i] = new_v;
		grid.sync();

		if (tid == coordinator_tid){
			// u[0, :]  = 0
			for (auto i = 0; i < NX; i++) u[0][i] = 0.0;
			// u[:, 0]  = 0
			for (auto j = 0; j < NY; j++) u[j][0] = 0.0;
			// u[:, -1] = 0
			for (auto j = 0; j < NY; j++) u[j][NX-1] = 0.0;
			// u[-1, :] = 1
			for (auto i = 0; i < NX; i++) u[NY-1][i] = 1.0;

			// v[0, :]  = 0
			for (auto i = 0; i < NX; i++) v[0][i] = 0.0;
			// v[-1, :] = 0
			for (auto i = 0; i < NX; i++) v[NY-1][i] = 0.0;
			// v[:, 0]  = 0
			for (auto j = 0; j < NY; j++) v[j][0] = 0.0;
			// v[:, -1] = 0
			for (auto j = 0; j < NY; j++) v[j][NX-1] = 0.0;
		}
		grid.sync();
	}
}

int main() {
	auto u = cuda_zeros(NY,NX);
	auto v = cuda_zeros(NY,NX);
	auto p = cuda_zeros(NY,NX);
	auto b = cuda_zeros(NY,NX);

	// total threads (one per element)
	const auto N = NX * NY;
	// max threads per block is 1024
	const auto M = 1024;
	void* args[] = {(void*) &N, (void*) &NT, (void*) &u, (void*) &v, (void*) &p, (void*) &b};
	cudaLaunchCooperativeKernel((void*) kernel, (N+M-1)/M, M, args);
	cudaDeviceSynchronize();

	for (const auto& mat : {u, v, p, b}) {
		cuda_mat_free(mat, NY);
	}
}
