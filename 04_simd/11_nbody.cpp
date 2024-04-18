#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    // outer loop operands
    auto xi = _mm512_set1_ps(x[i]);
    auto yi = _mm512_set1_ps(y[i]);

    // vectorize positions and masses
    auto xj = _mm512_load_ps(x);
    auto yj = _mm512_load_ps(y);
    auto mj = _mm512_load_ps(m);

    // subtract and square each dimension
    auto rx = _mm512_sub_ps(xi, xj);
    auto rx2 = _mm512_mul_ps(rx, rx);
    auto ry = _mm512_sub_ps(yi, yj);
    auto ry2 = _mm512_mul_ps(ry, ry);

    // compute 1/sqrt(rx^2 + ry^2)
    auto summed_r2 = _mm512_add_ps(rx2, ry2);
    auto r_inv = _mm512_rsqrt14_ps(summed_r2);

    // create mask to exclude same body
    // value taken from MPI lecture slides
    auto zeros = _mm512_set1_ps(0);
    auto small_val = _mm512_set1_ps(1e-15);
    auto mask = _mm512_cmp_ps_mask(summed_r2, small_val, _MM_CMPINT_GT);

    // multiply distance and mass
    auto fxj = _mm512_mul_ps(rx, mj);
    auto fyj = _mm512_mul_ps(ry, mj);
    // multiply by 1/(r^3)
    for(auto i = 0; i < 3; i++) {
	    fxj = _mm512_mul_ps(fxj, r_inv);
            fyj = _mm512_mul_ps(fyj, r_inv);
    }
    // invert forces (we want to do a subtraction reduction) and mask
    auto minus_one = _mm512_set1_ps(-1);
    fxj = _mm512_mul_ps(fxj, minus_one);
    fxj = _mm512_mask_blend_ps(mask, zeros, fxj);
    fyj = _mm512_mul_ps(fyj, minus_one);
    fyj = _mm512_mask_blend_ps(mask, zeros, fyj);
    fx[i] = _mm512_reduce_add_ps(fxj);
    fy[i] = _mm512_reduce_add_ps(fyj);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
