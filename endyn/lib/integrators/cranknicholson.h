// cranknicholson.h
#pragma once
#include "helpers.h"
// I/O and fmt headers for formatted output.
#include <stdio.h>
#include <iostream>
#include <fstream>
#define FMT_HEADER_ONLY
#include <fmt/format.h>
#include <fmt/core.h>
#include <fmt/printf.h>
//  std library headers
#include <vector>
#include <complex>
#include <string>
// Eigen headers
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

class CrankNicholson {
public:
    CrankNicholson();
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
