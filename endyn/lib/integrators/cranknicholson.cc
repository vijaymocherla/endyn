#include "cranknicholson.h"

CrankNicholson::CrankNicholson(){};
void CrankNicholson::runPropagator(
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
    std::string outfilename) {    
    // Reading time and field params    
    double ti, tf, dt;
    double fs_to_au = 41.341374575751;
    int switch_fx, switch_fy, switch_fz, print_nstep;
    std::tie (ti, tf, dt, print_nstep) = time_params;
    std::tie (switch_fx, switch_fy, switch_fz) = field_params;
    // Diagonalising dipoles to construct the
    // Unitary transform from dipole eigen -> CI eigen basis.
    if (switch_fx == 1){
        dipole_x = evecs.transpose()*dipole_x*evecs;
    };
    if (switch_fy == 1){
        dipole_y = evecs.transpose()*dipole_y*evecs;
    };
    if (switch_fz == 1){
        dipole_z = evecs.transpose()*dipole_z*evecs;
    }; 
    Eigen::MatrixXcd H0 = evals.asDiagonal();
    int ndim = evals.size();
    Eigen::MatrixXcd IdMat = Eigen::MatrixXd::Identity(ndim, ndim);  
    Eigen::VectorXcd yi_eig, k1, k2, k3, k4;
    Eigen::VectorXd expt_vals;
    // Projecting y0_csf to eigen basis
    yi_eig.noalias() = evecs*y0;
    int tstep=0, nops=ops_list.size(); 
    // Projecting the operators to eigen basis
    for (int i = 0; i < nops; i++) {
        ops_list[i] = evecs.transpose()*ops_list[i]*evecs;
    };
    // writing headers to output file.
    FILE * fout = fopen(outfilename.c_str(), "w"); // output file object fout
    fmt::fprintf(fout, "%20s\t", "time_fs");
    fmt::fprintf(fout, "%20s\t", "norm");
    fmt::fprintf(fout, "%20s\t", "autocorr");
    fmt::fprintf(fout, "%20s\t", headers);
    fmt::fprintf(fout, "\n");
    // writing expectation values at initial time ti.
    expt_vals = calc_expt(y0, nops, ops_list, y0);
    fmt::fprintf(fout, "%20.16f\t", ti/fs_to_au);
    for (int i = 0; i < 2 + nops; i++) {
        fmt::fprintf(fout,"%20.16f\t", expt_vals[i]);
    };
    fmt::fprintf(fout, "\n");
    // starting the propagation loop
    Eigen::MatrixXcd hamiltonian_t;
    while (ti <= tf) {
        for (int i = 0; i <= print_nstep; i++) {
            // adding relevant dipole operators to H0 in CI eigen basis
	    hamiltonian_t = H0;
            if (switch_fx == 1) {
                // H0 -= \mu_x . E_x(t+dt) (note: should be dt/2)
                hamiltonian_t -= (field_x[tstep])*dipole_x;
            }; 
            if (switch_fy == 1) {
                // H0 -= \mu_y . E_y(t+dt) (note: should be dt/2)
                hamiltonian_t -= (field_y[tstep])*dipole_y;
            };
            if (switch_fz == 1) {
                // H0 -= \mu_z . E_z(t+dt) (note: should be dt/2)
                hamiltonian_t -= (field_z[tstep])*dipole_z;
            };
            // the crank-nicholson method goes here;
            // (1 + (1i*dt/2)*hamiltonian_t(dt)) * yi_eig = (1 - (1i*dt/2)*hamiltonian_t(dt)) * yi_eig;
            yi_eig = linear_solve((IdMat + (1i*0.5*dt)*hamiltonian_t), (IdMat - (1i*0.5*dt)*hamiltonian_t)*yi_eig);
            ti += dt;
    	    tstep += 1;
	        if (ti >= tf) {
		        std::cout<<"breaking loop"<<std::endl;     
		        break;
	        };
        };
        // writing expectation values at ti
        expt_vals = calc_expt(yi_eig, nops, ops_list, y0);
        fmt::fprintf(fout, "%20.16f\t", ti/fs_to_au);
        for (int i = 0; i < 2 + nops; i++) {
            fmt::fprintf(fout, "%20.16f\t", expt_vals[i]);
        };
        fmt::fprintf(fout,"\n", "");
    };
    fclose(fout);
    return;
};
