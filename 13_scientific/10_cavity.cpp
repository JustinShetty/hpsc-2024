#include <iostream>
#include <vector>
#include <cmath>

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

typedef std::vector<float> FVector;
typedef std::vector<FVector> FMatrix;

FMatrix zeros(int n, int m) {
	return FMatrix(n, FVector(m, 0));
}

void run(int nmax, FMatrix& u, FMatrix& v, FMatrix& p, FMatrix& b) {
	for (auto n = 0; n < nmax; n++) {
		// b
		for (auto j = 1; j < NY - 1; j++) {
			for (auto i = 1; i < NX - 1; i++) {
				b[j][i] = (u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy);
				b[j][i] /= dt;
				b[j][i] -= pow((u[j][i+1] - u[j][i-1]) / (2 * dx), 2);
				b[j][i] -= 2 * (u[j+1][i] - u[j-1][i]) / (2 * dy) * (v[j][i+1] - v[j][i-1]) / (2 * dx);
				b[j][i] -= pow((v[j+1][i] - v[j-1][i]) / (2 * dy), 2);
				b[j][i] *= rho;
			}
		}

		// p
		for (auto it = 0; it < NIT; it++) {
			auto pn = p;
			for (auto j = 1; j < NY - 1; j++) {
				for (auto i = 1; i < NX - 1; i++) {
					p[j][i]  = dy2 * (pn[j][i+1] + pn[j][i-1]);
					p[j][i] += dx2 * (pn[j+1][i] + pn[j-1][i]);
					p[j][i] -= b[j][i] * dx2 * dy2;
					p[j][i] /= 2 * (dx2 + dy2);
				}
			}
			// p[:, -1] = p[:, -2]
			for (auto row : p) row[row.size()-1] = row[row.size()-2];
			// p[0, :] = p[1, :]
			p[0] = p[1];
			// p[:, 0] = p[:, 1]
			for (auto row : p) row[0] = row[1];
			// p[-1, :] = 0
			std::fill(p[p.size()-1].begin(), p[p.size()-1].end(), 0);
		}


		// update u and v
		auto un = u;
		auto vn = v;
		for (auto j = 1; j < NY - 1; j++) {
			for (auto i = 1; i < NX - 1; i++) {
				u[j][i] = un[j][i];
				u[j][i] -= un[j][i] * dt / dx * (un[j][i] - un[j][i-1]);
				u[j][i] -= un[j][i] * dt / dy * (un[j][i] - un[j-1][i]);
				u[j][i] -= dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1]);
				u[j][i] += nu * dt / dx2 * (un[j][i+1] - 2 * un[j][i] + un[j][i-1]);
				u[j][i] += nu * dt / dy2 * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);

				v[j][i] = vn[j][i];
				v[j][i] -= vn[j][i] * dt / dx * (vn[j][i] - vn[j][i-1]);
				v[j][i] -= vn[j][i] * dt / dy * (vn[j][i] - vn[j-1][i]);
				v[j][i] -= dt / (2 * rho * dy) * (p[j+1][i] - p[j-1][i]);
				v[j][i] += nu * dt / dx2 * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1]);
				v[j][i] += nu * dt / dy2 * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
			}
		}


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
}

int main() {
	auto u = zeros(NY,NX);
	auto v = zeros(NY,NX);
	auto p = zeros(NY,NX);
	auto b = zeros(NY,NX);

	run(NT, u, v, p, b);
}
