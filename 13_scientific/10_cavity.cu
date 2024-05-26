#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <fstream>

#define CUDA_CALL(call)                                                                                \
	do                                                                                                 \
	{                                                                                                  \
		cudaError_t err = call;                                                                        \
		if (err != cudaSuccess)                                                                        \
		{                                                                                              \
			fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
			exit(EXIT_FAILURE);                                                                        \
		}                                                                                              \
	} while (0)

using std::pow;

const auto NX = 41;
const auto NY = 41;
const auto NT = 500;
const auto NIT = 50;
constexpr auto dx = 2.0 / (NX - 1);
constexpr auto dx2 = dx * dx;
constexpr auto dy = 2.0 / (NY - 1);
constexpr auto dy2 = dy * dy;
const auto dt = 0.01;
const auto rho = 1.0;
const auto nu = 0.02;

double **cuda_zeros(int m, int n) {
	double **zeros;
	CUDA_CALL(cudaMallocManaged(&zeros, m * sizeof(double *)));
	for (int j = 0; j < m; j++) {
		CUDA_CALL(cudaMallocManaged(&zeros[j], n * sizeof(double)));
		for (int i = 0; i < n; i++) zeros[j][i] = 0.0;
	}
	return zeros;
}

void cuda_mat_free(double **mat, int m) {
	for (int j = 0; j < m; j++) CUDA_CALL(cudaFree(mat[j]));
	CUDA_CALL(cudaFree(mat));
}

__global__ void kernel(double **u, double **v, double **p, double **b) {
	bool active_thread = true;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= NY * NX) active_thread = false;
	auto j = tid / NX;
	auto i = tid % NX;
	if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1) active_thread = false;
	auto coordinator_tid = NX + 1;
	// grid to synchronize across ALL threads, not just this block
	auto grid = cooperative_groups::this_grid();

	// b
	if (active_thread) {
		b[j][i] = (u[j][i + 1] - u[j][i - 1]) / (2 * dx) + (v[j + 1][i] - v[j - 1][i]) / (2 * dy);
		b[j][i] /= dt;
		b[j][i] -= pow((u[j][i + 1] - u[j][i - 1]) / (2 * dx), 2);
		b[j][i] -= 2 * (u[j + 1][i] - u[j - 1][i]) / (2 * dy) * (v[j][i + 1] - v[j][i - 1]) / (2 * dx);
		b[j][i] -= pow((v[j + 1][i] - v[j - 1][i]) / (2 * dy), 2);
		b[j][i] *= rho;
	}
	grid.sync();

	// p
	for (auto it = 0; it < NIT; it++) {
		double new_p;
		if (active_thread) {
			new_p = dy2 * (p[j][i + 1] + p[j][i - 1]);
			new_p += dx2 * (p[j + 1][i] + p[j - 1][i]);
			new_p -= b[j][i] * dx2 * dy2;
			new_p /= 2 * (dx2 + dy2);
		}
		grid.sync();
		if (active_thread) p[j][i] = new_p;

		// reset boundaries
		if (tid == coordinator_tid) {
			// p[:, -1] = p[:, -2]
			for (auto j = 0; j < NY; j++) p[j][NX - 1] = p[j][NX - 2];
			// p[0, :] = p[1, :]
			for (auto i = 0; i < NX; i++) p[0][i] = p[1][i];
			// p[:, 0] = p[:, 1]
			for (auto j = 0; j < NY; j++) p[j][0] = p[j][1];
			// p[-1, :] = 0
			for (auto i = 0; i < NX; i++) p[NY - 1][i] = 0.0;
		}
		grid.sync();
	}

	// u
	double new_u;
	if (active_thread) {
		new_u = u[j][i];
		new_u -= u[j][i] * dt / dx * (u[j][i] - u[j][i - 1]);
		new_u -= u[j][i] * dt / dy * (u[j][i] - u[j - 1][i]);
		new_u -= dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1]);
		new_u += nu * dt / dx2 * (u[j][i + 1] - 2 * u[j][i] + u[j][i - 1]);
		new_u += nu * dt / dy2 * (u[j + 1][i] - 2 * u[j][i] + u[j - 1][i]);
	}
	grid.sync();
	if (active_thread) u[j][i] = new_u;
	grid.sync();

	// v
	double new_v;
	if (active_thread) {
		new_v = v[j][i];
		new_v -= v[j][i] * dt / dx * (v[j][i] - v[j][i - 1]);
		new_v -= v[j][i] * dt / dy * (v[j][i] - v[j - 1][i]);
		new_v -= dt / (2 * rho * dy) * (p[j + 1][i] - p[j - 1][i]);
		new_v += nu * dt / dx2 * (v[j][i + 1] - 2 * v[j][i] + v[j][i - 1]);
		new_v += nu * dt / dy2 * (v[j + 1][i] - 2 * v[j][i] + v[j - 1][i]);
	}
	grid.sync();
	if (active_thread) v[j][i] = new_v;
	grid.sync();

	// reset boundaries
	if (tid == coordinator_tid) {
		// u[0, :]  = 0
		for (auto i = 0; i < NX; i++) u[0][i] = 0.0;
		// u[:, 0]  = 0
		for (auto j = 0; j < NY; j++) u[j][0] = 0.0;
		// u[:, -1] = 0
		for (auto j = 0; j < NY; j++) u[j][NX - 1] = 0.0;
		// u[-1, :] = 1
		for (auto i = 0; i < NX; i++) u[NY - 1][i] = 1.0;

		// v[0, :]  = 0
		for (auto i = 0; i < NX; i++) v[0][i] = 0.0;
		// v[-1, :] = 0
		for (auto i = 0; i < NX; i++) v[NY - 1][i] = 0.0;
		// v[:, 0]  = 0
		for (auto j = 0; j < NY; j++) v[j][0] = 0.0;
		// v[:, -1] = 0
		for (auto j = 0; j < NY; j++) v[j][NX - 1] = 0.0;
	}
}

int main() {
	auto u = cuda_zeros(NY, NX);
	auto v = cuda_zeros(NY, NX);
	auto p = cuda_zeros(NY, NX);
	auto b = cuda_zeros(NY, NX);

	int dev = 0;
	cudaDeviceProp deviceProp;
	CUDA_CALL(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s\n", deviceProp.name);
	printf("\tMax threads per block %d\n", deviceProp.maxThreadsPerBlock);
	printf("\tMax threads per multiprocessor %d\n", deviceProp.maxThreadsPerMultiProcessor);
	printf("\tMultiprocessor count %d\n", deviceProp.multiProcessorCount);
	printf("\tMax blocks per multiprocessor %d\n", deviceProp.maxBlocksPerMultiProcessor);
	printf("\tMax threads %d\n", deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount);

	// total threads (one per element)
	const auto N = NX * NY;
	const auto tpb = deviceProp.maxThreadsPerBlock;
	const auto num_blocks = (N + tpb - 1) / tpb;
	void *args[] = {(void *)&u, (void *)&v, (void *)&p, (void *)&b};

	std::ofstream ufile("u.dat", std::ios::out | std::ios::trunc);
	std::ofstream vfile("v.dat", std::ios::out | std::ios::trunc);
	std::ofstream pfile("p.dat", std::ios::out | std::ios::trunc);
	for (auto n = 0; n < NT; n++) {
		CUDA_CALL(cudaLaunchCooperativeKernel((void *)kernel, num_blocks, tpb, args));
		CUDA_CALL(cudaDeviceSynchronize());

		for (auto j = 0; j < NY; j++) {
			for (auto i = 0; i < NX; i++) {
				ufile << u[j][i] << " ";
				vfile << v[j][i] << " ";
				pfile << p[j][i] << " ";
			}
		}
		ufile << "\n";
		vfile << "\n";
		pfile << "\n";
	}
	ufile.close();
	vfile.close();
	pfile.close();

	for (const auto &mat : {u, v, p, b}) cuda_mat_free(mat, NY);
}
