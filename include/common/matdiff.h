#pragma once
#include "eigen_alias.h"

#include <iostream>
#include "glog/logging.h"

#include <Eigen/Core>
#include <Eigen/Dense>
namespace feh
{
const Eigen::Matrix<FloatType, 9, 3> dTilde = (
    Eigen::Matrix<FloatType, 9, 3>()
        <<
        0, 0, 0,	//--------
        0, 0, 1,	// dv~1/dv
        0, -1, 0,	//--------
        0, 0, -1,    //--------
        0, 0, 0,	// dv~2/dv
        1, 0, 0,	//--------
        0, 1, 0,   //--------
        -1, 0, 0,	// dv~3/dv
        0, 0, 0
).finished();

const Eigen::Matrix<FloatType, 3, 9> dTildeInv = (
    Eigen::Matrix<FloatType, 3, 9>()
        <<
        0,   0,   0,      0,   0,  0.5,      0, -0.5,   0,
        0,   0, -0.5,     0,   0,   0,     0.5,   0,   0,
        0, 0.5,   0,   -0.5,   0,   0,      0,   0,   0
).finished();

// Derivative of the half-trace operator
const Eigen::Matrix<FloatType, 1, 9> dTr2 = ( Eigen::Matrix<FloatType, 1, 9>() << 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5).finished();

const Eigen::Matrix<FloatType, 9, 9> dAt_dA = (
    Eigen::Matrix<FloatType, 9, 9>()
        <<
        1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 1., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0.,
        0., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 1.
).finished();

const Eigen::Matrix<FloatType, 9, 1> dA_ddiagA = (Eigen::Matrix<FloatType, 9, 1>() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).finished();

// /// Derivative function of f=acos(x)
// /// f'(x) = -1/sqrt(1-x^2)
// /// @param x	A real number
// /// @param y	The derivative f'(x)
// FloatType dacosx_dx(FloatType x);

///// Tilde operator
///// V  = [  0 -v3  v2
/////        v3   0 -v1
/////       -v2  v1   0];
///// @param v		A 3-vector
///// @return v_tilde	The skew-symmetric dual of this vector
//const Mat3& tilde(const Vec3& v);
//
///// Tilde-squared operator
///// VV = [ -v2^2-v3^2,      v1*v2,      v1*v3;
/////             v1*v2, -v1^2-v3^2,      v2*v3;
/////             v1*v3,      v2*v3, -v1^2-v2^2];
///// @param v		A 3-vector
///// @return v_tilde	The square of the skew-symmetric dual
//const Mat3& tildeSq(const Vec3 &v);
//
///// Vectorized (row-major) differential of the tilde operator
///// @param v
///// @param dv_tilde_sq
//void dTildeSq(const Vec3 &v, Mat93& dv_tilde_sq);


/// Tensor multiplication of vectorized matrices:  Given matrices
///	dA:=[dA1(:), dA2(:), dA3(:)] (3x3x3->9x3) and B (3x3->9x1)
/// computes [(dAB1)(:), (dAB2)(:), (dAB3)(:)].
/// @param dA	Vectorized (row-major) differential of a function f:v(3x1)|->A(3x3)
/// @param B	Vectorized (row-major) constant 3x3 matrix
/// @param dAB	Vectorized (row-major) differential of the function f:v(3x1)|->AB(3x3)
void dABVectorized(const Mat93 &dA, const Mat3 &B, Mat93 &dAB);

/// Tensor multiplication of vectorized matrices:  Given matrix and vector
///	dA:=[dA1(:), dA2(:), dA3(:)] (3x3x3->9x3) and b (3x1)
/// computes [dA1b, dA2b, dA3b].
/// @param dA	Vectorized (row-major) differential of a function f:v(3x1)|->A(3x3)
/// @param b	Vectorized (row-major) constant 3-vector
/// @param dAb	Vectorized (row-major) differential of the function f:v(3x1)|->Ab(3x1)
void dAbVectorized(const Mat93 &dA, const Vec3 &b, Mat3& dAb);

/// Tensor multiplication of vectorized matrices:  Given matrices
/// A(:) (3x3->9x1) and dB: = [dB1(:), dB2(:), dB3(:)](3x3x3->9x3),
/// computes [(AdB1)(:), (AdB2)(:), (AdB3)(:)].
/// @param A	Vectorized (row-major) constant 3x3 matrix
/// @param dB	Vectorized (row-major) differential of a function f:v(3x1)|->B(3x3)
/// @param AdB	Vectorized (row-major) differential of the function f:v(3x1)|->AB(3x3)
void AdBVectorized(const Mat3 &A, const Mat93 &dB, Mat93 &AdB);

/// Computes the exponential matrix exp(W) and the  matrix differential : d(exp(W)) / dv
/// using Rodrigues' formula:
///       R = I + sin(theta)*W + (1 - cos(theta))W ^ 2
/// where theta = | v | and W is the skew - symmetric form of v / theta
///
/// @param v	Axis-angle vector (3x1)
/// @param dRdv	Vectorized (row-major) differential of the function f:v(3x1)|->rodrigues(v)(3x3)
/// @reference a compact formula for the derivative of a 3-D rotation in exponential coordinates
void RodriguesFwd( const Vec3 &v, Mat3 &R, Mat93 *dRdv=nullptr);


/// Computes the derivative of the matrix logarithm : dv / d(exp(W))
/// using the axis - angle formula :
///   v = (theta / sin(theta)) *[	R(3, 2) - R(2, 3)
///									R(1, 3) - R(3, 1)
///									R(2, 1) - R(1, 2)];
/// where theta = | v | and W is the skew - symmetric form of v / theta
void RodriguesInv(const Mat3 &R, Vec3 &v, Mat39 *dvdR=nullptr);

void dvdvConstrainedLength(const Vec3 &v, FloatType nv_constrained, Mat3& dvdv);

} // feh
