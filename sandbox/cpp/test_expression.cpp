// Auto-generated with Symbolica 1.4.0
// Default build instructions: g++ -shared -O3 -fPIC -ffast-math -funsafe-math-optimizations -march=native test_expression.cpp

#include <iostream>
#include <cmath>

#include <complex>
typedef std::complex<double> Number;
extern "C" unsigned long f_complexf64_get_buffer_len()
{
	return 36;
}


template<typename T>
void f_complexf64_gen(T* params, T* Z, T* out) {
	Z[17] = params[5]+params[8];
	Z[18] = T(173)*Z[17];
	Z[19] = params[3]*Z[18];
	Z[20] = params[7]*T(0, 1);
	Z[21] = params[6]+Z[20];
	Z[22] = params[4]*T(173)*Z[21];
	Z[23] = params[6]*T(-1);
	Z[20] = Z[20]+Z[23];
	Z[24] = params[11]*T(0, 1);
	Z[25] = params[10]+Z[24];
	Z[26] = Z[20]*Z[25];
	Z[27] = params[8]*T(-1);
	Z[27] = params[5]+Z[27];
	Z[28] = params[9]+params[12];
	Z[29] = Z[27]*Z[28];
	Z[26] = Z[26]+Z[29];
	Z[26] = params[1]*Z[26];
	Z[25] = Z[17]*Z[25];
	Z[30] = params[7]*T(0, -1);
	Z[23] = Z[23]+Z[30];
	Z[31] = Z[28]*Z[23];
	Z[25] = Z[25]+Z[31];
	Z[25] = params[2]*Z[25];
	Z[19] = Z[19]+Z[22]+Z[26]+Z[25];
	Z[22] = params[6]+Z[30];
	Z[25] = params[3]*T(173)*Z[22];
	Z[26] = T(173)*Z[27];
	Z[30] = params[4]*Z[26];
	Z[31] = params[11]*T(0, -1);
	Z[32] = params[10]+Z[31];
	Z[33] = Z[27]*Z[32];
	Z[34] = params[12]*T(-1);
	Z[34] = params[9]+Z[34];
	Z[35] = Z[20]*Z[34];
	Z[33] = Z[33]+Z[35];
	Z[33] = params[1]*Z[33];
	Z[32] = Z[23]*Z[32];
	Z[35] = Z[17]*Z[34];
	Z[32] = Z[32]+Z[35];
	Z[32] = params[2]*Z[32];
	Z[25] = Z[25]+Z[30]+Z[33]+Z[32];
	Z[26] = params[1]*Z[26];
	Z[23] = params[2]*T(173)*Z[23];
	Z[30] = params[10]*T(-1);
	Z[31] = Z[31]+Z[30];
	Z[32] = Z[22]*Z[31];
	Z[32] = Z[35]+Z[32];
	Z[32] = params[3]*Z[32];
	Z[27] = Z[27]*Z[31];
	Z[31] = Z[21]*Z[34];
	Z[27] = Z[27]+Z[31];
	Z[27] = params[4]*Z[27];
	Z[23] = Z[26]+Z[23]+Z[32]+Z[27];
	Z[20] = params[1]*T(173)*Z[20];
	Z[18] = params[2]*Z[18];
	Z[24] = Z[24]+Z[30];
	Z[17] = Z[17]*Z[24];
	Z[22] = Z[28]*Z[22];
	Z[17] = Z[17]+Z[22];
	Z[17] = params[3]*Z[17];
	Z[21] = Z[21]*Z[24];
	Z[21] = Z[29]+Z[21];
	Z[21] = params[4]*Z[21];
	Z[17] = Z[20]+Z[18]+Z[17]+Z[21];
	out[0] = Z[19];
	out[1] = Z[25];
	out[2] = Z[23];
	out[3] = Z[17];
	return;
}

extern "C" {
	void f_complexf64(std::complex<double> *params, std::complex<double> *buffer, std::complex<double> *out) {
		f_complexf64_gen(params, buffer, out);
		return;
	}
}
