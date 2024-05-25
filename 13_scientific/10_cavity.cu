#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <fstream>

using std::pow;

const auto NX = 5;
const auto NY = 5;
const auto NT = 10;
const auto NIT = 50;
constexpr auto dx = 2.0 / (NX-1);
constexpr auto dx2 = dx * dx;
constexpr auto dy = 2.0 / (NY-1);
constexpr auto dy2 = dy * dy;
const auto dt = 0.01;
const auto rho = 1.0;
const auto nu = 0.02;

class AddressTranslator {
public:
	__host__ __device__ AddressTranslator(const int m_, const int n_) : m(m_), n(n_) {}
	__host__ __device__ int operator()(int j, int i) { return j * n + i; }
private:
	int m, n;
};

__global__ void kernel(
	const int total_threads, 
	double* const u_f2, 
	double* const v_f2,
	double* const b_f2, 
	double* const p_f2
) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= total_threads) return;
	auto j = tid / NX;
	auto i = tid % NX;
	if (i == 0 || i == NX-1 || j == 0 || j == NY-1) return;
	auto coordinator_tid = NX + 1;
	// use grid object to synchronize across ALL threads
	auto grid = cooperative_groups::this_grid();

	AddressTranslator at(NY, NX);

	for(auto n = 0; n < NT; n++) {
		// b
		b_f2[at(j, i)] = (u_f2[at(j, i+1)] - u_f2[at(j, i-1)]) / (2 * dx) + (v_f2[at(j+1, i)] - v_f2[at(j-1, i)]) / (2 * dy);
		b_f2[at(j, i)] /= dt;
		b_f2[at(j, i)] -= pow((u_f2[at(j, i+1)] - u_f2[at(j, i-1)]) / (2 * dx), 2);
		b_f2[at(j, i)] -= 2 * (u_f2[at(j+1, i)] - u_f2[at(j-1, i)]) / (2 * dy) * (v_f2[at(j, i+1)] - v_f2[at(j, i-1)]) / (2 * dx);
		b_f2[at(j, i)] -= pow((v_f2[at(j+1, i)] - v_f2[at(j-1, i)]) / (2 * dy), 2);
		b_f2[at(j, i)] *= rho;
		grid.sync();

		// p
		for (auto it = 0; it < NIT; it++) {
			double new_p;
			new_p = dy2 * (p_f2[at(j, i+1)] + p_f2[at(j, i-1)]);
			new_p += dx2 * (p_f2[at(j+1, i)] + p_f2[at(j-1, i)]);
			new_p -= b_f2[at(j, i)] * dx2 * dy2;
			new_p /= 2 * (dx2 + dy2);
			grid.sync();
			p_f2[at(j, i)] = new_p;
			grid.sync();

			if(tid == coordinator_tid) {
				// p[:, -1] = p[:, -2]
				for (auto j = 0; j < NY; j++) p_f2[at(j, NX-1)] = p_f2[at(j, NX-2)];
				// p[0, :] = p[1, :]
				for (auto i = 0; i < NX; i++) p_f2[at(0,i)] = p_f2[at(1, i)];
				// p[:, 0] = p[:, 1]
				for (auto j = 0; j < NY; j++) p_f2[at(j,0)] = p_f2[at(j,1)];
				// p[-1, :] = 0
				for (auto i = 0; i < NX; i++) p_f2[at(NY-1, i)] = 0.0;
			}
			grid.sync();
		}

		// u
		double new_u;
		new_u = u_f2[at(j, i)];
		new_u -= u_f2[at(j, i)] * dt / dx * (u_f2[at(j, i)] - u_f2[at(j, i-1)]);
		new_u -= u_f2[at(j, i)] * dt / dy * (u_f2[at(j, i)] - u_f2[at(j-1, i)]);
		new_u -= dt / (2 * rho * dy) * (p_f2[at(j, i+1)] - p_f2[at(j, i-1)]);
		new_u += nu * dt / dx2 * (u_f2[at(j, i+1)] - 2 * u_f2[at(j, i)] + u_f2[at(j, i-1)]);
		new_u += nu * dt / dy2 * (u_f2[at(j+1, i)] - 2 * u_f2[at(j, i)] + u_f2[at(j-1, i)]);
		grid.sync();
		u_f2[at(j, i)] = new_u;
		grid.sync();

		// v
		double new_v;
		new_v = v_f2[at(j, i)];
		new_v -= v_f2[at(j, i)] * dt / dx * (v_f2[at(j, i)] - v_f2[at(j, i-1)]);
		new_v -= v_f2[at(j, i)] * dt / dy * (v_f2[at(j, i)] - v_f2[at(j-1, i)]);
		new_v -= dt / (2 * rho * dx) * (p_f2[at(j+1, i)] - p_f2[at(j-1, i)]);
		new_v += nu * dt / dx2 * (v_f2[at(j, i+1)] - 2 * v_f2[at(j, i)] + v_f2[at(j, i-1)]);
		new_v += nu * dt / dy2 * (v_f2[at(j+1, i)] - 2 * v_f2[at(j, i)] + v_f2[at(j-1, i)]);
		grid.sync();
		v_f2[at(j, i)] = new_v;
		grid.sync();

		if (tid == coordinator_tid){
			// u[0, :]  = 0
			for (auto i = 0; i < NX; i++) u_f2[at(0, i)] = 0.0;
			// u[:, 0]  = 0
			for (auto j = 0; j < NY; j++) u_f2[at(j, 0)] = 0.0;
			// u[:, -1] = 0
			for (auto j = 0; j < NY; j++) u_f2[at(j, NX-1)] = 0.0;
			// u[-1, :] = 1
			for (auto i = 0; i < NX; i++) u_f2[at(NY-1, i)] = 1.0;

			// v[0, :]  = 0
			for (auto i = 0; i < NX; i++) v_f2[at(0, i)] = 0.0;
			// v[-1, :] = 0
			for (auto i = 0; i < NX; i++) v_f2[at(NY-1, i)] = 0.0;
			// v[:, 0]  = 0
			for (auto j = 0; j < NY; j++) v_f2[at(j, 0)] = 0.0;
			// v[:, -1] = 0
			for (auto j = 0; j < NY; j++) v_f2[at(j, NX-1)] = 0.0;
		}
		grid.sync();
	}
}

int main() {
	const auto num_elems = NX * NY;
	double* u_f2; cudaMallocManaged(&u_f2, num_elems*sizeof(double));
	double* v_f2; cudaMallocManaged(&v_f2, num_elems*sizeof(double));
	double* b_f2; cudaMallocManaged(&b_f2, num_elems*sizeof(double));
	double* p_f2; cudaMallocManaged(&p_f2, num_elems*sizeof(double));

	// total threads (one per element)
	const auto N = NX * NY;
	// max threads per block is 1024
	const auto M = 1024;
	void* args[] = {(void*) &N, (void*) &u_f2, (void*) &v_f2, (void*) &b_f2, (void*) &p_f2};
	cudaLaunchCooperativeKernel((void*) kernel, (N+M-1)/M, M, args);
	cudaDeviceSynchronize();

	// std::ofstream ufile("u.dat", std::ios::out | std::ios::trunc);
	// std::ofstream vfile("v.dat", std::ios::out | std::ios::trunc);
	// std::ofstream pfile("p.dat", std::ios::out | std::ios::trunc);
	// for (int l = 0 ; l < NT; l++) {
	// 	for (int j = 0; j < NY; j++) {
	// 		for (int i = 0; i < NX; i++) {
	// 			// ufile << u_f2[at(j, i)] << " ";
	// 			// vfile << v_f2[at(j, i)] << " ";
	// 			// pfile << p[j][i] << " ";
	// 		}
	// 	}
	// 	ufile << "\n";
	// 	vfile << "\n";
	// 	pfile << "\n";
	// }
	// ufile.close();
	// vfile.close();
	// pfile.close();

	for (auto& arr : {u_f2, v_f2, b_f2, p_f2}) cudaFree(arr);
}
