#include "splitoperator.h"

SplitOperator::SplitOperator() {}
void SplitOperator::runPropagator( 
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
    double ti, tf, dt;
    double fs_to_au = 41.341374575751;
    int switch_fx, switch_fy, switch_fz, print_nstep;
    std::tie (ti, tf, dt, print_nstep) = time_params;
    std::tie (switch_fx, switch_fy, switch_fz) = field_params;
    // Diagonalizing some matrices    
    Eigen::MatrixXd v_dx, v_dy, v_dz;
    Eigen::VectorXd u_dx, u_dy, u_dz; 
    Eigen::VectorXd expt_vals;
    // Diagonalising dipoles to construct the
    // Unitary transform from dipole eigen -> CI eigen basis.
    if (switch_fx == 1){
        std::tie (u_dx, v_dx) = diagonalise(dipole_x);
        v_dx = v_dx.transpose()*evecs;
    };
    if (switch_fy == 1){
        std::tie (u_dy, v_dy) = diagonalise(dipole_y);
        v_dy = v_dy.transpose()*evecs;
    };
    if (switch_fz == 1){
        std::tie (u_dz, v_dz) = diagonalise(dipole_z);
        v_dz = v_dz.transpose()*evecs;
    }; 
    Eigen::VectorXcd yi_csf, yi_eig;
    // Projecting y0_csf to eigen basis
    yi_eig.noalias() = evecs*y0;
    int tstep=0, nops=ops_list.size(); 
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
    while (ti <= tf) {
        for (int i = 0; i <= print_nstep; i++) {
    	    // exp(-1i*H0*dt/2). yi
            // yi_eig = cwiseExpcd((-1i*0.5*dt)*evals).cwiseProduct(yi_eig);
            yi_eig = cwiseExpcd((-1i*dt)*evals).cwiseProduct(yi_eig);
            // Exponentiating dipole.field(t) array in dipole eigen basis, 
            // together with basis transform: dipole eigen -> CI eigen.
            if (switch_fx == 1) {
                // W^T .exp(1i*mu0_0x*E_x(t+dt)*dt). W . yi
                yi_eig = v_dx.transpose()*(cwiseExpcd((1i*field_x[tstep]*dt)*u_dx).asDiagonal())*v_dx*yi_eig;
            }; 
            if (switch_fy == 1) {
                // W^T .exp(1i*mu_0y*E_y(t+dt)*dt). W . yi
                yi_eig = v_dy.transpose()*(cwiseExpcd((1i*field_y[tstep]*dt)*u_dy).asDiagonal())*v_dy*yi_eig;
            };
            if (switch_fz == 1) {
                // W^T .exp(1i*mu_0z*E_z(t+dt)*dt). W . yi
                yi_eig = v_dz.transpose()*(cwiseExpcd((1i*field_z[tstep]*dt)*u_dz).asDiagonal())*v_dz*yi_eig;
            };
            // exp(-1i*H0*dt/2). yi
            // yi_eig = cwiseExpcd((-1i*0.5*dt)*evals).cwiseProduct(yi_eig);
            ti += dt;
    	    tstep += 1;
	        if (ti >= tf) {
		        std::cout<<"breaking loop"<<std::endl;     
		        break;
	        };
        };
        // projecting yi_eig back to Position or CSF basis
        yi_csf.noalias() = evecs.transpose()*yi_eig;
        // writing expectation values at ti
        expt_vals = calc_expt(yi_csf, nops, ops_list, y0);
        fmt::fprintf(fout, "%20.16f\t", ti/fs_to_au);
        for (int i = 0; i < 2 + nops; i++) {
            fmt::fprintf(fout, "%20.16f\t", expt_vals[i]);
        };
        fmt::fprintf(fout,"\n", "");
    };
    fclose(fout);
    return;
};
