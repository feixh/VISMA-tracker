#include "common/matdiff.h"

namespace feh
{

inline
FloatType dacosx_dx(FloatType x)
{
    return -1 / sqrt(1 - x*x);
}

inline 
Vec9 detensor(const Mat3 &in)
{
    return (Vec9() << in.col(0), in.col(1), in.col(2)).finished();
}

inline
Mat3 tilde(const Vec3& v)
{
    FloatType v1 = v(0), v2 = v(1), v3 = v(2);
    return (Mat3()
        <<
        0, -v3, v2,
        v3, 0, -v1,
        -v2, v1, 0).finished();
}

inline 
Vec3 tildeInv(const Mat3 &R)
{
    return 0.5*(Vec3() << R(2,1) - R(1,2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1)).finished();
}

inline
Mat3 tildeSq(const Vec3 &v)
{
    FloatType v1 = v(0), v2 = v(1), v3 = v(2);
    return (Mat3()
        <<
        -v2*v2 - v3*v3, v1*v2, v1*v3,
        v1*v2, -v1*v1 - v3*v3, v2*v3,
        v1*v3, v2*v3, -v1*v1 - v2*v2).finished();
}

void dTildeSq(const Vec3 &v, Mat93& dv_tilde_sq)
{
    FloatType v1 = v(0), v2 = v(1), v3 = v(2);
    dv_tilde_sq << 0, -2 * v2, -2 * v3,	// ------------
        v2, v1, 0,				// d(v~)^2_1/dv
        v3, 0, v1,				// ------------
        v2, v1, 0,				// ------------
        -2 * v1, 0, -2 * v3,	// d(v~)^2_2/dv
        0, v3, v2,				// ------------
        v3, 0, v1,				// ------------
        0, v3, v2,				// d(v~)^2_3/dv
        -2 * v1, -2 * v2, 0;
}


void dABVectorized(const Mat93 &dA, const Mat3 &B, Mat93& dAB)
{
    for (int l = 0; l < 3; ++l) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                dAB(j * 3 + i, l) = 0;
                for (int k = 0; k < 3; ++k) {
                    dAB(j * 3 + i, l) += dA(k * 3 + i, l) * B(k, j); //, 0);
                }
            }
        }
    }
}


void AdBVectorized(const Mat3 &A, const Mat93 &dB, Mat93& AdB)
{
    for (int l = 0; l < 3; ++l) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                AdB(j * 3 + i, l) = 0;
                for (int k = 0; k < 3; k++) {
                    AdB(j * 3 + i, l) += A(i, k) * dB(j * 3 + k, l);
                }
            }
        }
    }
}


void dAbVectorized(const Mat93 &dA, const Vec3 &b, Mat3& dAb)
{
    for (int l = 0; l < 3; ++l) {
        for (int i = 0; i < 3; ++i) {
            dAb(i, l) = 0;
            for (int k = 0; k < 3; ++k) {
                dAb(i, l) += dA(k * 3 + i, l) * b(k);
            }
        }
    }
}

 void RodriguesFwd(const Vec3 &v, Mat3 &R, Mat93 *dRdv)
 {
     FloatType th = v.norm();
     if (th < 1e-20) {
         R.setIdentity();
         R += tilde(v);
         if (dRdv) *dRdv = dTilde;
         return;
     }
     FloatType cth(std::cos(th));
     FloatType sth(std::sin(th));
     FloatType inv_th(1.0 / th);
     Mat3 vtilde(tilde(v * inv_th));

     // R.setIdentity();
     // R += sth * vtilde + (1-cth) * vtilde * vtilde;

     R = (1-cth)*vtilde;
     R(0, 0) += sth; R(1, 1) += sth; R(2, 2) += sth;
     R *= vtilde;
     R(0, 0) += 1.; R(1, 1) += 1.; R(2, 2) += 1.;

     if (!dRdv) return;
     vtilde *= inv_th;
     Mat3 aa(-R);
     aa(0, 0) += 1.; aa(1, 1) += 1; aa(2, 2) += 1.;
     aa = vtilde*aa;

     *dRdv <<
         detensor((v(0) * vtilde + tilde(aa.col(0))) * R ),
         detensor((v(1) * vtilde + tilde(aa.col(1))) * R ),
         detensor((v(2) * vtilde + tilde(aa.col(2))) * R );
 }


//void RodriguesFwd(const Vec3 &v, Mat3 &R, Mat93 *dRdv)
//{
//    // // Initialize the rotation and its derivative
//    // R.setIdentity();
//
//    // Special zero rotation
//    FloatType th = v.norm();
//    // if (theta*theta < Eigen::NumTraits<FloatType>::epsilon())
//    if (th < 1e-20)
//    {
//        Mat3 tildev(tilde(v));
//        R = tildev; //first order expansion near zero
//        R(0, 0) += 1;
//        R(1, 1) += 1;
//        R(2, 2) += 1;
//        if (dRdv) *dRdv = dTilde;
//        return;
//    }
//    FloatType sth(sin(th));
//    FloatType cth(cos(th));
//
//    // Compute the skew-symmetric representation
//    FloatType inv_th = 1.0 / th;
//    Vec3 w(v * inv_th); // / theta;
//    Mat3 W(tilde(w));
//    Mat3 WW(W * W);
//
//    // -- RODRIGUES' FORMULA
//    R = sth*W + (1 - cth)*WW;
//    R(0, 0) += 1;
//    R(1, 1) += 1;
//    R(2, 2) += 1;
//
//
//    if (!dRdv) return;
//
//    *dRdv = dTilde;  // derivative of the tilde operator
//
//    // -- DERIVATIVE OF RODRIGUES' FORMULA
//    // d(theta)/dv
//    auto wt(w.transpose());
//
//    Mat3 dw_dv(-w*wt*inv_th);
//    dw_dv(0, 0) += inv_th;
//    dw_dv(1, 1) += inv_th;
//    dw_dv(2, 2) += inv_th;
//
//    // d(tilde(w)^2)/dw
//    Mat93 dWW_dw;
//    dTildeSq(w, dWW_dw);
//
//    // Chain rule
//    *dRdv = detensor(W) * cth * wt
//        + sth * *dRdv * dw_dv
//        + detensor(WW) * sth * wt
//        + (1 - cth) * dWW_dw * dw_dv;
//}


void RodriguesInv(const Mat3 &R, Vec3 &v, Mat39 *dvdR)
{
    // Some intermediate values
    FloatType trR(0.5*(R.trace() - 1));
    // WARNING: if trR >= 1 or <= -1, acos(trR) returns NaN, need to handle this -- super hack
    FloatType th(trR >= 1? 0 : acos(trR));
    if (std::isnan(th)) LOG(FATAL) << "theta is NaN, check whether trR == -1 such that acos(trR) == NaN";
    
    // FIXME: handle the case of trR <= -1, c.f.
    // https://github.com/kashif/ceres-solver/blob/master/include/ceres/rotation.h
    
    FloatType sin_th(std::sin(th));
    v = tildeInv(R);
    if (sin_th < 1e-20) 
    {
        if (dvdR) {
            *dvdR = dTildeInv;
        }
        return;
    }

    // v = theta / sin(theta) * tildeInv(R)
    // theta = arccos(0.5*(trace(R)-1))
    FloatType temp(th/sin_th);
    if (dvdR) {
        Eigen::Matrix<FloatType, 1, 9> dtheta_dR(dacosx_dx(trR) * dTr2);
        FloatType dtemp_dtheta = (sin_th - th*std::cos(th)) / sin_th / sin_th;
        *dvdR = temp * dTildeInv +  v * dtemp_dtheta * dtheta_dR;
    }
    v *= temp;
}


// Josh's original with minimal modification
// void RodriguesInv(const Mat3 &R, Vec3 &v, Mat39 *dvdR)
// {
//     using T = FloatType;
//     // Initialize the output values
//     v = Eigen::Matrix<T, 3, 1>();
//     v.setZero();
//     
//     // Some intermediate values
//     T trR = (R.trace() - 1) / 2;
//     T theta = acos(trR);
//     T sth = sin(theta);
//     T tsth = theta/(2*sth);
//     
//     // -- DERIVATIVE OF INVERSE RODRIGUES' FORMULA
//     Eigen::Matrix<T, 3, 9> dvR_dR = 2 * dTildeInv;
//     Eigen::Matrix<T, 3, 1> vR = dvR_dR * Eigen::Map<const Eigen::Matrix<T, 9, 1>>(R.data(), 9, 1);
//     Eigen::Matrix<T, 1, 9> dtheta_dR = dacosx_dx(trR) * dTr2;
//     
//     // Chain Rule
//     if (dvdR) {
//         *dvdR = vR * dtheta_dR * ((0.5 - tsth*trR) /  sth) + tsth * dvR_dR;
//     }
//     
//     // Special zero rotation
//     // should follow conventions in RotationMatrixToAngleAxis from https://github.com/kashif/ceres-solver/blob/master/include/ceres/rotation.h
//     
//     if (sth*sth < Eigen::NumTraits<T>::epsilon())
//     {
//         //Some hacks, should be nicer
//         tsth = 0.5/(1.0-theta*theta/6.0);
//         sth  = 1;
//         if (dvdR) *dvdR = tsth*dvR_dR;
//     }
//     
//     
//     // RODRIGUES' FORMULA
//     vR *= tsth;                                 
//     v(0) = vR(0, 0); v(1) = vR(1, 0);  v(2) = vR(2, 0); // Can't convert matrix to vector
// }



void dvdvConstrainedLength(const Vec3 &v, FloatType nv_constrained, Mat3& dvdv)
{
    FloatType nv = v.norm();
    Vec3 Mv = v / nv;
    dvdv = (nv_constrained / nv) * (Mat3::Identity() - Mv * Mv.transpose());
}


}   // namespace feh

