#pragma once
#include "helpers.h"
// I/O and fmt headers for formatted output.
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <fmt/core.h>
#include <fmt/printf.h>
//  std library headers
#include <vector>
#include <complex>
#include <string>
// Eigen headers
#include <Eigen/Core>
#include <Eigen/Dense>

// typedef struct spiltoperator_input
// {
//     // eigvals;
//     // eigvecs;
//     // time_params;
//     // field_params ;
//     // dpx;
//     // dpy;
//     // dpz;
//     // ncore;
//     // ops_list;
//     // ops_headers;
//     // print_nstep;
//     // outfile;
// };

class SplitOperator {
public:
    SplitOperator();
    static void runPropagator(
        Eigen::VectorXcd evals, 
        RowMatrixXd evecs,
        Eigen::VectorXcd y0,
        std::tuple<double, double, double, int> time_params,
        std::tuple<int, int, int> field_params,
        RowMatrixXd dipole_x,
        RowMatrixXd dipole_y,
        RowMatrixXd dipole_z,
        Eigen::VectorXd field_x,
        Eigen::VectorXd field_y,
        Eigen::VectorXd field_z,
        MatrixList ops_list,
        std::string headers,
        std::string outfilename
    );
};
