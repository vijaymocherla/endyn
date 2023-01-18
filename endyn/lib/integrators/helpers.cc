#include "helpers.h"


std::tuple<Eigen::VectorXd, Eigen::MatrixXd> diagonalise(Eigen::Ref<RowMatrixXd> matrix) {
    // Lets call Eigen solver from Eigen to diagonalise matrix     
    // local variables
    int N = matrix.rows(); // The order of the matrix A,  with n >= 0.
    int lda = N;    // The leading dimension of the array A,  LDA >= max(1,N).
    int lwork = N;  // The length of the array work, lwork >= max(1,3*N-1).
    // // extra variables needed for dsyevx
    // int m, info;
    // int il = 1; int iu = N; // specifying the range of eigenvalues
    // double work[lwork*N];
    // double abstol = -1.0; 
    // double vl, vu;
    // int ifail[N];
    Eigen::VectorXd evals(N);
    Eigen::MatrixXd evecs(N,N);
    evecs = matrix;
    // calling the cblas function, pay attention to  'LAPACK_ROW_MAJOR'
    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', lda, &evecs(0, 0), lwork, &evals(0));
    // int info = LAPACKE_dsyevx( LAPACK_ROW_MAJOR, 'V', 'I', 'U', N, &evecs(0, 0), lda,
    //                     vl, vu, il, iu, abstol, &m, &evals(0), work, lwork, ifail );
    // initialize <vector> eigavals and <matrix> eigvecs to pass to <tuple> result
    if ( info == 1) {
        throw std::invalid_argument(" Matrix diagonalisation failed.");
    }
    return std::tuple<Eigen::VectorXd, Eigen::MatrixXd>(evals, evecs.transpose());
}

Eigen::VectorXcd cwiseExpcd(Eigen::VectorXcd array) {
    return array.unaryExpr([](std::complex<double> x) { return exp(x); });
};


RowMatrixXd testmul(Eigen::Ref<RowMatrixXd> a, Eigen::Ref<RowMatrixXd> b) {
    return a*b;
};

// A function to calculate expectation value to print to output file
Eigen::VectorXd calc_expt(Eigen::VectorXcd yi, int nops, MatrixList ops_list, Eigen::VectorXcd y0){
    Eigen::VectorXd result(nops+2);
    result[0] = abs(yi.conjugate().dot(yi)); // norm
    result[1] = abs(yi.conjugate().dot(y0)); // autocorr
    // std::cout<< ops_list[0]<<std::endl;
    for (int i = 0; i < nops; i++) {
        // calculates expectation value of ith operator matrix and stores it at [i+2] position
        result[i+2] = real((yi.conjugate().cwiseProduct(ops_list[i]*yi)).sum()) ;
    }
    return result;
}

// Linear solver system for Crank-Nicholson. 
Eigen::VectorXcd linear_solve(Eigen::MatrixXcd A, Eigen::VectorXcd b) {
    Eigen::BiCGSTAB<Eigen::SparseMatrix<std::complex<double>>> solver;
    Eigen::VectorXcd x = solver.compute(A.sparseView()).solve(b);
    if(solver.info()!=Eigen::Success) {
        // linear solver failed.
        throw std::invalid_argument("linear_solve failed with given arguments.");
    }
    return x;
};
