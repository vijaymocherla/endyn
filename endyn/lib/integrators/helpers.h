// #ifdef EIGEN_USE_MKL
//  #include "mkl_lapacke.h"
//  #include "mkl_lapack.h"
//  #include "mkl.h"
//  #define EIGEN_USE_MKL_ALL
// #else
//  #include "lapacke.h"
// #endif
//  std library headers
// Forcing explicitly to use of MKL as the above conditional doesn't work
 #include "mkl_lapacke.h"
 #include "mkl_lapack.h"
 #include "mkl.h"
 #define EIGEN_USE_MKL_ALL

#include <vector>
#include <complex>
#include <string>
// Eigen headers
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>

using namespace std::complex_literals;


// Use RowMatrixXd instead of MatrixXd to pass a RowMajor array to C++ functions from python
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixList = std::vector<RowMatrixXcd>;
typedef Eigen::SparseMatrix<std::complex<double>> SparseMatrixNcd;
typedef Eigen::SparseVector<std::complex<double>> SparseVectorNcd;
// Declaring the functions
std::tuple<Eigen::VectorXd, Eigen::MatrixXd> diagonalise(Eigen::Ref<RowMatrixXd> matrix);
Eigen::VectorXcd cwiseExpcd(Eigen::VectorXcd array);
Eigen::VectorXd calc_expt(Eigen::VectorXcd yi, int nops, MatrixList ops_list, Eigen::VectorXcd y0);
RowMatrixXd testmul(Eigen::Ref<RowMatrixXd> a, Eigen::Ref<RowMatrixXd> b);
Eigen::VectorXcd linear_solve(Eigen::MatrixXcd A, Eigen::VectorXcd b);
