// Declaring functions
double *comp_hrow_hf(void *mo_eps_in, void *mo_eris_in, int nmo, double scf_energy,
                     void *csfs_in, int *num_csfs, bool *options, int p);
double *comp_hrow_ia(void *mo_eps_in, void *mo_eris_in, int nmo, double scf_energy,
                     void *csfs_in, int *num_csfs, bool *options, int p);
double *comp_hrow_iiaa(void *mo_eps_in, void *mo_eris_in, int nmo, double scf_energy,
                       void *csfs_in, int *num_csfs, bool *options, int p);
double *comp_hrow_iiab(void *mo_eps_in, void *mo_eris_in, int nmo, double scf_energy,
                       void *csfs_in, int *num_csfs, bool *options, int p);
double *comp_hrow_ijaa(void *mo_eps_in, void *mo_eris_in, int nmo, double scf_energy,
                       void *csfs_in, int *num_csfs, bool *options, int p);
double *comp_hrow_ijab_a(void *mo_eps_in, void *mo_eris_in, int nmo, double scf_energy,
                         void *csfs_in, int *num_csfs, bool *options, int p);
double *comp_hrow_ijab_b(void *mo_eps_in, void *mo_eris_in, int nmo, double scf_energy,
                         void *csfs_in, int *num_csfs, bool *options, int p);
double *comp_oeprop_hf(void *mo_oeprop_in, int nmo, double moeprop_trace,
                       void *csfs_in, int *num_csfs, bool *options, int p);
double *comp_oeprop_ia(void *mo_oeprop_in, int nmo, double moeprop_trace,
                       void *csfs_in, int *num_csfs, bool *options, int p);
double *comp_oeprop_iiaa(void *mo_oeprop_in, int nmo, double moeprop_trace,
                         void *csfs_in, int *num_csfs, bool *options, int p);
double *comp_oeprop_iiab(void *mo_oeprop_in, int nmo, double moeprop_trace,
                         void *csfs_in, int *num_csfs, bool *options, int p);
double *comp_oeprop_ijaa(void *mo_oeprop_in, int nmo, double moeprop_trace,
                         void *csfs_in, int *num_csfs, bool *options, int p);
double *comp_oeprop_ijab_a(void *mo_oeprop_in, int nmo, double moeprop_trace,
                           void *csfs_in, int *num_csfs, bool *options, int p);
double *comp_oeprop_ijab_b(void *mo_oeprop_in, int nmo, double moeprop_trace,
                           void *csfs_in, int *num_csfs, bool *options, int p);
