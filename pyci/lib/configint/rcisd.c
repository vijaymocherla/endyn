/* 
 *
 * author: Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
 */

// TO-DO: 
//   - complete parsing routines for functions    
//   - change typedef's for arrays, bools, to NPY_ types   
// 

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <Python.h>
// #include <rcisd.h>
// Declaring math functions
double sqrt(double); 
#define PY_SSIZE_T_CLEAN
// Declaring functions
static PyObject * 
comp_hrow_hf_cfunc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject * 
comp_hrow_ia_cfunc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject * 
comp_hrow_iiaa_cfunc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject * 
comp_hrow_iiab_cfunc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject * 
comp_hrow_ijaa_cfunc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject * 
comp_hrow_ijab_a_cfunc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject * 
comp_hrow_ijab_b_cfunc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject * 
comp_oeprop_hf_cfunc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject * 
comp_oeprop_ia_cfunc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject * 
comp_oeprop_iiaa_cfunc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject * 
comp_oeprop_iiab_cfunc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject * 
comp_oeprop_ijaa_cfunc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject * 
comp_oeprop_ijab_a_cfunc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject * 
comp_oeprop_ijab_b_cfunc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyMethodDef rcisd_methods[] = {
        {"comp_hrow_hf", comp_hrow_hf_cfunc, METH_VARARGS|METH_KEYWORDS, " "},
        {"comp_hrow_ia", comp_hrow_ia_cfunc, METH_VARARGS|METH_KEYWORDS, " "},
        {"comp_hrow_iiaa", comp_hrow_iiaa_cfunc, METH_VARARGS|METH_KEYWORDS, " "},
        {"comp_hrow_iiab", comp_hrow_iiab_cfunc, METH_VARARGS|METH_KEYWORDS, " "},
        {"comp_hrow_ijaa", comp_hrow_ijaa_cfunc, METH_VARARGS|METH_KEYWORDS, " "},
        {"comp_hrow_ijab_a", comp_hrow_ijab_a_cfunc, METH_VARARGS|METH_KEYWORDS, " "},
        {"comp_hrow_ijab_b", comp_hrow_ijab_b_cfunc, METH_VARARGS|METH_KEYWORDS, " "},
        {"comp_oeprop_hf", comp_oeprop_hf_cfunc, METH_VARARGS|METH_KEYWORDS, " "},
        {"comp_oeprop_ia", comp_oeprop_ia_cfunc, METH_VARARGS|METH_KEYWORDS, " "},
        {"comp_oeprop_iiaa", comp_oeprop_iiaa_cfunc, METH_VARARGS|METH_KEYWORDS, " "},
        {"comp_oeprop_iiab", comp_oeprop_iiab_cfunc, METH_VARARGS|METH_KEYWORDS, " "},
        {"comp_oeprop_ijaa", comp_oeprop_ijaa_cfunc, METH_VARARGS|METH_KEYWORDS, " "},
        {"comp_oeprop_ijab_a", comp_oeprop_ijab_a_cfunc, METH_VARARGS|METH_KEYWORDS, " "},
        {"comp_oeprop_ijab_b", comp_oeprop_ijab_b_cfunc, METH_VARARGS|METH_KEYWORDS, " "},
        {NULL, NULL, 0, NULL}    /* Sentinel */
};

static struct PyModuleDef librcisd = {
        PyModuleDef_HEAD_INIT,
        "rcisd",        /* name of module */
        rcisd_doc,      /* module documentation, may be NULL*/
        -1,             /* size of per-interpreter state of the module,
                        or -1 if the module keeps state in global variables. */
        rcisd_methods
};

PyMODINIT_FUNC 
PyInit_rcisd(void)
{
        return PyModule_Create(&librcisd);
};


int
main(int argc, char *argv[])
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    /* Add a built-in module, before Py_Initialize */
    if (PyImport_AppendInittab("spam", PyInit_rcisd) == -1) {
        fprintf(stderr, "Error: could not extend in-built modules table\n");
        exit(1);
    }

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required.
       If this step fails, it will be a fatal error. */
    Py_Initialize();

    /* Optionally import the module; alternatively,
       import can be deferred until the embedded script
       imports it. */
    PyObject *pmodule = PyImport_ImportModule("rcisd");
    if (!pmodule) {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'rcisd'\n");
    }

    PyMem_RawFree(program);
    return 0;
};

static PyObject*
comp_hrow_hf_cfunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
        
        static char *kwlist[] = {"singles", "full_cis", "doubles", 
                                "doubles_iiaa", "doubles_iiab",
                                "doubles_ijaa", "doubles_ijab_A",
                                "doubles_ijab_B", NULL
                                };
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "S|i", kwlist, &))
                return NULL;

        // declaring variables
        int i, j, k, l, a, b, c, d;
        int q  = 0; 
        int n_ia = num_csfs[1]; 
        int n_iiaa = num_csfs[2]; 
        int n_iiab = num_csfs[3]; 
        int n_ijaa = num_csfs[4]; 
        int n_ijab_a = num_csfs[5]; 
        int n_ijab_b = num_csfs[6];
        double E0 = scf_energy;    
        double(*mo_eps) = (double(*))mo_eps_in;
        int ndim = n_ia + n_iiaa + n_iiab + n_ijaa + n_ijab_a + n_ijab_b;   
        double(*mo_eris)[nmo][nmo][nmo] = (double(*)[nmo][nmo][nmo])mo_eris_in;
        double(*csfs)[ndim] = (double(*)[ndim])csfs_in;
        bool singles = options[0]; 
        bool full_cis = options[1]; 
        bool doubles = options[2]; 
        bool doubles_iiaa = options[3]; 
        bool doubles_iiab = options[4]; 
        bool doubles_ijaa = options[5];
        bool doubles_ijab_a = options[6];
        bool doubles_ijab_b = options[7];
        double row[ndim];
        row[0] = E0; 
        q = 1; 
        if (singles){
            n_ia = num_csfs[1];
            q += n_ia;
        }
        if (doubles){
            if (doubles_iiaa){
                for (int idx = q; idx < q + n_iiaa; ++idx){
                    k = csfs[idx][0];
                    l = csfs[idx][1];
                    c = csfs[idx][2];
                    d = csfs[idx][3];
                    row[idx] = mo_eris[k][l][k][l];
                }
            }
            if (doubles_iiab){
                for (int idx = q; idx < q + n_ijaa; ++idx){    
                    k = csfs[idx][0];
                    l = csfs[idx][1];
                    c = csfs[idx][2];
                    d = csfs[idx][3];
                    row[idx] = sqrt(2) * mo_eris[c][k][d][k];
                }
            }
            if (doubles_ijaa){
                for (int idx = q; idx < q + n_ijaa; ++idx){
                     k = csfs[idx][0];
                     l = csfs[idx][1];
                     c = csfs[idx][2];
                     d = csfs[idx][3];
                     row[idx] = sqrt(2) * mo_eris[c][k][c][l];
                }
            }
            if (doubles_ijab_a){
                for (int idx; idx < q + n_ijab_a; ++idx){
                     k = csfs[idx][0];
                     l = csfs[idx][1];
                     c = csfs[idx][2];
                     d = csfs[idx][3];
                     row[idx] = sqrt(3) * (mo_eris[c][k][d][l] - mo_eris[c][l][d][k]);
                }
            }
            if (doubles_ijab_b){
                for (int idx = q; idx < q + n_ijab_b; ++q){
                     k = csfs[idx][0];
                     l = csfs[idx][1];
                     c = csfs[idx][2];
                     d = csfs[idx][3];
                     row[idx] = mo_eris[c][k][d][l] + mo_eris[c][l][d][k];
                }
            }
        }
        return row;
};

static PyObject*
comp_hrow_ia_cfunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
        int i, j, k, l, a, b, c, d;
        int q  = 0; 
        int n_ia = num_csfs[1]; 
        int n_iiaa = num_csfs[2]; 
        int n_iiab = num_csfs[3]; 
        int n_ijaa = num_csfs[4]; 
        int n_ijab_a = num_csfs[5]; 
        int n_ijab_b = num_csfs[6];
        double(*mo_eps) = (double(*))mo_eps_in;
        int ndim = 1 + n_ia + n_iiaa + n_iiab + n_ijaa + n_ijab_a + n_ijab_b;
        double(*mo_eris)[nmo][nmo][nmo] = (double(*)[nmo][nmo][nmo])mo_eris_in;
        double(*csfs)[ndim] = (double(*)[ndim])csfs_in;
        double E0 = scf_energy;
        bool singles = options[0]; 
        bool full_cis = options[1]; 
        bool doubles = options[2]; 
        bool doubles_iiaa = options[3]; 
        bool doubles_iiab = options[4]; 
        bool doubles_ijaa = options[5];
        bool doubles_ijab_a = options[6];
        bool doubles_ijab_b = options[7];
        i = csfs[p][0];
        j = csfs[p][1];
        a = csfs[p][2];
        b = csfs[p][3];
        double row[ndim];
        row[0] = 0.0; 
        q += 1;
        if (singles){
                for (int idx = q; idx < q + n_ia; ++idx){
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((i == k) * (a == c) * (E0 + mo_eps[a] - mo_eps[i]) 
                                 + 2 * mo_eris[a][i][c][k] - mo_eris[c][a][k][i]);
                }
               q += n_ia;
        }
        if (doubles_iiaa){
                for ( int idx = q; idx < q + n_iiaa; ++idx){
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(2) * ((i == k) * mo_eris[c][a][c][i] 
                                            - (a == c) * mo_eris[k][a][k][i]);
                }
        }
        q += n_iiaa;
        if (doubles_iiab){
                for ( int idx = q; idx < q + n_iiab; ++idx){
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((i == k) * (mo_eris[d][a][c][i] + mo_eris[c][a][d][i]) 
                                  - (a == c) * mo_eris[k][d][k][i] 
                                  - (a == d) * mo_eris[k][c][k][i]);
                }
                q += n_iiab;
        }
        if (doubles_ijaa){
                for ( int idx = q; idx < q + n_ijaa; ++idx){
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((i == k) * mo_eris[c][a][c][l] 
                                + (i == l) * mo_eris[c][a][c][k] 
                                - (a == c) * (mo_eris[a][l][k][i] + mo_eris[a][k][l][i]));
                }
                q += n_ijaa;
        }
        if (doubles_ijab_a){
                for ( int idx = q; idx < q + n_ijab_a; ++idx){
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(1.5) * ((i == k) * (mo_eris[a][c][d][l] - mo_eris[a][d][c][l]) 
                                              - (i == l) * (mo_eris[a][c][d][k] - mo_eris[a][d][c][k]) 
                                              + (a == c) * (mo_eris[d][k][l][i] - mo_eris[d][l][k][i]) 
                                              - (a == d) * (mo_eris[c][k][l][i] - mo_eris[c][l][k][i]));
                }
                q += n_ijab_a;
        }
        if (doubles_ijab_b){
                for ( int idx = q; idx < q + n_ijab_b; ++idx){
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(0.5) * ((i == k) * (mo_eris[a][c][d][l] + mo_eris[a][d][c][l]) 
                                              + (i == l) * (mo_eris[a][c][d][k] + mo_eris[a][d][c][k]) 
                                              - (a == c) * (mo_eris[d][k][l][i] + mo_eris[d][l][k][i]) 
                                              - (a == d) * (mo_eris[c][k][l][i] + mo_eris[c][l][k][i]));
                }
                q += n_ijab_b;
        }
        return row; 
};

static PyObject*
comp_hrow_iiaa_cfunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
        int i, j, k, l, a, b, c, d;
        int q  = 0; 
        int n_ia = num_csfs[1]; 
        int n_iiaa = num_csfs[2]; 
        int n_iiab = num_csfs[3]; 
        int n_ijaa = num_csfs[4]; 
        int n_ijab_a = num_csfs[5]; 
        int n_ijab_b = num_csfs[6]; 
        double(*mo_eps) = (double(*))mo_eps_in;
        int ndim = 1 + n_ia + n_iiaa + n_iiab + n_ijaa + n_ijab_a + n_ijab_b;
        double(*mo_eris)[nmo][nmo][nmo] = (double(*)[nmo][nmo][nmo])mo_eris_in;
        double(*csfs)[ndim] = (double(*)[ndim])csfs_in;
        double E0 = scf_energy;
        bool singles = options[0]; 
        bool full_cis = options[1]; 
        bool doubles = options[2]; 
        bool doubles_iiaa = options[3]; 
        bool doubles_iiab = options[4]; 
        bool doubles_ijaa = options[5];
        bool doubles_ijab_a = options[6];
        bool doubles_ijab_b = options[7];
        i = csfs[p][0];
        j = csfs[p][1];
        a = csfs[p][2];
        b = csfs[p][3];
        double row[ndim];
        row[0] = mo_eris[a][i][a][i];
        q += 1;
        if (singles)
        {
                for ( int idx = q; idx < q + n_ia; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(2) * ((k == i) * mo_eris[a][c][a][k] 
                                        - (c == a) * mo_eris[i][c][i][k]);
                }
                q += n_ia;
        }
        if (doubles_iiaa)
        {
                for ( int idx = q; idx < q + n_iiaa; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((i == k) * (a == c) * (E0 - 2 * mo_eps[i] + 2 * mo_eps[a] - 4 * mo_eris[a][a][i][i] + 2 * mo_eris[a][i][a][i]) 
                                + (i == k) * mo_eris[c][a][c][a] 
                                + (a == c) * mo_eris[k][i][k][i]);
                }
                q += n_iiaa;        
        }
        if (doubles_iiab)
        {
                for ( int idx = q; idx < q + n_iiab; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(2) * ((i == k) * (a == c) * (mo_eris[a][i][d][i] - 2 * mo_eris[a][d][i][i]) 
                                + (i == k) * (a == d) * (mo_eris[a][i][c][i] - 2 * mo_eris[a][c][i][i]) 
                                + (i == k) * mo_eris[a][d][a][c]);
                }
                q += n_iiab;
        }

        if (doubles_ijaa)
        {
                for ( int idx = q; idx < q + n_ijaa; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(2) * ((i == k) * (a == c) * (mo_eris[a][i][a][l] - 2 * mo_eris[a][a][l][i]) 
                                        + (i == l) * (a == c) * (mo_eris[a][i][a][k] - 2 * mo_eris[a][a][k][i]) 
                                        + (a == c) * mo_eris[k][i][l][i]);
                }
                q += n_ijaa;
        }
        if (doubles_ijab_a)
        {
                for ( int idx = q; idx < q + n_ijab_a; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(3) * ((i == k) * (a == c) * mo_eris[a][i][d][l] 
                                          - (i == k) * (a == d) * mo_eris[a][i][c][l] 
                                          - (i == l) * (a == c) * mo_eris[a][i][d][k] 
                                          + (i == l) * (a == d) * mo_eris[a][i][c][k]);
                }
        }
        q += n_ijab_a;
        if (doubles_ijab_b)
        {
                for ( int idx = q; idx < q + n_ijab_b; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((i == k) * (a == c) * (mo_eris[a][i][d][l] - 2 * mo_eris[a][d][l][i]) 
                                + (i == k) * (a == d) * (mo_eris[a][i][c][l] - 2 * mo_eris[a][c][l][i]) 
                                + (i == l) * (a == c) * (mo_eris[a][i][d][k] - 2 * mo_eris[a][d][k][i]) 
                                + (i == l) * (a == d) * (mo_eris[a][i][c][k] - 2 * mo_eris[a][c][k][i]));
                }
                q += n_ijab_b;
        }
        return row;
};

static PyObject*
comp_hrow_iiab_cfunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
        int i, j, k, l, a, b, c, d;
        int q  = 0; 
        int n_ia = num_csfs[1]; 
        int n_iiaa = num_csfs[2]; 
        int n_iiab = num_csfs[3]; 
        int n_ijaa = num_csfs[4]; 
        int n_ijab_a = num_csfs[5]; 
        int n_ijab_b = num_csfs[6];
        bool singles = options[0]; 
        bool full_cis = options[1]; 
        bool doubles = options[2]; 
        bool doubles_iiaa = options[3]; 
        bool doubles_iiab = options[4]; 
        bool doubles_ijaa = options[5];
        bool doubles_ijab_a = options[6];
        bool doubles_ijab_b = options[7];
        double E0 = scf_energy;
        double(*mo_eps) = (double(*))mo_eps_in;
        int ndim = 1 + n_ia + n_iiaa + n_iiab + n_ijaa + n_ijab_a + n_ijab_b;
        double(*mo_eris)[nmo][nmo][nmo] = (double(*)[nmo][nmo][nmo])mo_eris_in;
        double(*csfs)[ndim] = (double(*)[ndim])csfs_in;
        i = csfs[p][0];
        j = csfs[p][1];
        a = csfs[p][2];
        b = csfs[p][3];
        double row[ndim];
        row[0] = sqrt(2)*mo_eris[a][i][b][i];
        q += 1;
        if (singles)
        {
                for (int idx = q; idx < q; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((k == i) * (mo_eris[b][c][a][k] + mo_eris[a][c][b][k])
                                - (c == a) * mo_eris[i][b][i][k] 
                                - (c == b) * mo_eris[i][a][i][k]);
                }
                q += n_ia;
        }
        if (doubles_iiaa)
        {
                for ( int idx = q; idx < q + n_iiaa; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(2) * (((k == i) * (c == a) * (mo_eris[c][k][b][k] - 2 * mo_eris[c][b][k][k])) 
                                                + (k == i) * (c == b) * (mo_eris[c][k][a][k] - 2 * mo_eris[c][a][k][k])
                                                + (k == i) * mo_eris[c][b][c][a]);
                }
                q += n_iiaa;
        }
        if (doubles_iiab)
        {
                for ( int idx = q; idx < q; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((i == k) * (a == c) * (b == d) * (E0 - 2.0 * mo_eps[i] + mo_eps[a] + mo_eps[b]) 
                        + (i == k) * (a == c) * (mo_eris[b][i][d][i] - 2.0 * mo_eris[b][d][i][i]) 
                        + (i == k) * (a == d) * (mo_eris[b][i][c][i] - 2.0 * mo_eris[b][c][i][i]) 
                        + (i == k) * (b == c) * (mo_eris[a][i][d][i] - 2.0 * mo_eris[a][d][i][i]) 
                        + (i == k) * (b == d) * (mo_eris[a][i][c][i] - 2.0 * mo_eris[a][c][i][i]) 
                        + (i == k) * (mo_eris[a][c][b][d] + mo_eris[a][d][b][c]) 
                        + (a == c) * (b == d) * (mo_eris[k][i][k][i]));
                }
                q += n_iiab;
        }
        if (doubles_ijaa)
        {
                for ( int idx = q; idx < q + n_ijaa; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((i == k) * (a == c) * (mo_eris[a][l][b][i] - 2 * mo_eris[a][b][l][i]) 
                                + (i == k) * (b == c) * (mo_eris[b][l][a][i] - 2 * mo_eris[b][a][l][i]) 
                                + (i == l) * (a == c) * (mo_eris[a][k][b][i] - 2 * mo_eris[a][b][k][i]) 
                                + (i == l) * (b == c) * (mo_eris[b][k][a][i] - 2 * mo_eris[b][a][k][i]));
                }
                q += n_ijaa;
        }
        if (doubles_ijab_a)
        {
                for ( int idx = q; idx < q + n_ijab_a; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(1.5) * ((i == k) * (a == c) * mo_eris[b][i][d][l] 
                                               - (i == k) * (a == d) * mo_eris[b][i][c][l] 
                                               + (i == k) * (b == c) * mo_eris[a][i][d][l] 
                                               - (i == k) * (b == d) * mo_eris[a][i][c][l] 
                                               - (i == l) * (a == c) * mo_eris[b][i][d][k] 
                                               + (i == l) * (a == d) * mo_eris[b][i][c][k] 
                                               - (i == l) * (b == c) * mo_eris[a][i][d][k] 
                                               + (i == l) * (b == d) * mo_eris[a][i][c][k]);
                }
                q += n_ijab_a;
        }
        if (doubles_ijab_b)
        {
                for ( int idx = q; idx < q + n_ijab_b; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(0.5) * ((i == k) * (a == c) * (mo_eris[b][i][d][l] - 2 * mo_eris[b][d][l][i]) 
                                               + (i == k) * (a == d) * (mo_eris[b][i][c][l] - 2 * mo_eris[b][c][l][i]) 
                                               + (i == k) * (b == c) * (mo_eris[a][i][d][l] - 2 * mo_eris[a][d][l][i]) 
                                               + (i == k) * (b == d) * (mo_eris[a][i][c][l] - 2 * mo_eris[a][c][l][i]) 
                                               + (i == l) * (a == c) * (mo_eris[b][i][d][k] - 2 * mo_eris[b][d][k][i]) 
                                               + (i == l) * (a == d) * (mo_eris[b][i][c][k] - 2 * mo_eris[b][c][k][i])
                                               + (i == l) * (b == c) * (mo_eris[a][i][d][k] - 2 * mo_eris[a][d][k][i]) 
                                               + (i == l) * (b == d) * (mo_eris[a][i][c][k] - 2 * mo_eris[a][c][k][i]) 
                                               + (a == c) * (b == d) * 2 * mo_eris[k][i][l][i]);
                }
                q += n_ijab_b;
        }
        return row;
};

static PyObject*
comp_hrow_ijaa_cfunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
        int i, j, k, l, a, b, c, d;
        int q  = 0; 
        int n_ia = num_csfs[1]; 
        int n_iiaa = num_csfs[2]; 
        int n_iiab = num_csfs[3]; 
        int n_ijaa = num_csfs[4]; 
        int n_ijab_a = num_csfs[5]; 
        int n_ijab_b = num_csfs[6]; 
        double E0 = scf_energy;
        bool singles = options[0]; 
        bool full_cis = options[1]; 
        bool doubles = options[2]; 
        bool doubles_iiaa = options[3]; 
        bool doubles_iiab = options[4]; 
        bool doubles_ijaa = options[5];
        bool doubles_ijab_a = options[6];
        bool doubles_ijab_b = options[7];
        double(*mo_eps) = (double(*))mo_eps_in;
        int ndim = 1 + n_ia + n_iiaa + n_iiab + n_ijaa + n_ijab_a + n_ijab_b;
        double(*mo_eris)[nmo][nmo][nmo] = (double(*)[nmo][nmo][nmo])mo_eris_in;
        double(*csfs)[ndim] = (double(*)[ndim])csfs_in;
        i = csfs[p][0];
        j = csfs[p][1];
        a = csfs[p][2];
        b = csfs[p][3];
        double row[ndim];
        row[0] = sqrt(2) * mo_eris[a][i][a][j];
        q += 1;
        if (singles)
        {
                for ( int idx = q; idx < q; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((k == i) * mo_eris[a][c][a][j] 
                                + (k == j) * mo_eris[a][c][a][i] 
                                + (c == a) * (mo_eris[c][j][i][k] + mo_eris[c][i][j][k]));
                }
                q += n_ia;
        }
        if (doubles_iiaa)
        {
                for ( int idx = q; idx < q + n_iiaa; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(2) * (((k == i) * (c == a) * (mo_eris[c][k][c][j] - 2 * mo_eris[c][c][j][k])) 
                                        + (k == j) * (c == a) * (mo_eris[c][k][c][i] - 2 * mo_eris[c][c][i][k]) 
                                        + (c == a) * mo_eris[i][k][j][k]);
                }
                q += n_iiaa;        
        }
        if (doubles_iiab)
        {
                for ( int idx = q; idx < q + n_iiab; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((k == i) * (c == a) * (mo_eris[c][j][d][k] - 2 * mo_eris[c][d][j][k]) 
                                + (k == i) * (d == a) * (mo_eris[d][j][c][k] - 2 * mo_eris[d][c][j][k]) 
                                + (k == j) * (c == a) * (mo_eris[c][i][d][k] - 2 * mo_eris[c][d][i][k]) 
                                + (k == j) * (d == a) * (mo_eris[d][i][c][k] - 2 * mo_eris[d][c][i][k]));
                }
                q += n_iiab;
        }
        if (doubles_ijaa)
        {
                for ( int idx = q; idx < q + n_ijaa; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((i == k) * (j == l) * (a == c) * (E0 - mo_eps[i] - mo_eps[j] + 2 * mo_eps[a])
                                + (i == k) * (a == c) * (mo_eris[a][l][a][j] - 2 * mo_eris[a][a][l][j]) 
                                + (i == l) * (a == c) * (mo_eris[a][k][a][j] - 2 * mo_eris[a][a][k][j]) 
                                + (j == k) * (a == c) * (mo_eris[a][l][a][i] - 2 * mo_eris[a][a][l][i]) 
                                + (j == l) * (a == c) * (mo_eris[a][k][a][i] - 2 * mo_eris[a][a][k][i]) 
                                + (a == c) * (mo_eris[k][i][l][j] + mo_eris[l][i][k][j]) + (i == k) * (j == l) * (mo_eris[c][a][c][a]));
                }
                q += n_ijaa;
        }
        if (doubles_ijab_a)
        {
                for ( int idx = q; idx < q + n_ijab_a; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(1.5) * ((i == k) * (a == c) * (mo_eris[a][j][d][l]) 
                                            - (i == k) * (a == d) * (mo_eris[a][j][c][l]) 
                                            - (i == l) * (a == c) * (mo_eris[a][j][d][k]) 
                                            + (i == l) * (a == d) * (mo_eris[a][j][c][k]) 
                                            + (j == k) * (a == c) * (mo_eris[a][i][d][l]) 
                                            - (j == k) * (a == d) * (mo_eris[a][i][c][l]) 
                                            - (j == l) * (a == c) * (mo_eris[a][i][d][k]) 
                                            + (j == l) * (a == d) * (mo_eris[a][i][c][k]));
                }
                q += n_ijab_a;
        }
        if (doubles_ijab_b)
        {
                for ( int idx = q; idx < q + n_ijab_b; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(0.5) * ((i == k) * (a == c) * (mo_eris[a][j][d][l] - 2 * mo_eris[a][d][j][l]) 
                                            + (i == k) * (a == d) * (mo_eris[a][j][c][l] - 2 * mo_eris[a][c][j][l]) 
                                            + (i == l) * (a == c) * (mo_eris[a][j][d][k] - 2 * mo_eris[a][d][j][k]) 
                                            + (i == l) * (a == d) * (mo_eris[a][j][c][k] - 2 * mo_eris[a][c][j][k])
                                            + (j == k) * (a == c) * (mo_eris[a][i][d][l] - 2 * mo_eris[a][d][i][l]) 
                                            + (j == k) * (a == d) * (mo_eris[a][i][c][l] - 2 * mo_eris[a][c][i][l]) 
                                            + (j == l) * (a == c) * (mo_eris[a][i][d][k] - 2 * mo_eris[a][d][i][k]) 
                                            + (j == l) * (a == d) * (mo_eris[a][i][c][k] - 2 * mo_eris[a][c][i][k]) 
                                            + (i == k) * (j == l) * 2 * mo_eris[c][a][d][a]);
                }
                q += n_ijab_b;
        }
        return row;
};

static PyObject*
comp_hrow_ijab_a_cfunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
        int i, j, k, l, a, b, c, d;
        int q  = 0; 
        int n_ia = num_csfs[1]; 
        int n_iiaa = num_csfs[2]; 
        int n_iiab = num_csfs[3]; 
        int n_ijaa = num_csfs[4]; 
        int n_ijab_a = num_csfs[5]; 
        int n_ijab_b = num_csfs[6]; 
        double E0 = scf_energy;
        bool singles = options[0]; 
        bool full_cis = options[1]; 
        bool doubles = options[2]; 
        bool doubles_iiaa = options[3]; 
        bool doubles_iiab = options[4]; 
        bool doubles_ijaa = options[5];
        bool doubles_ijab_a = options[6];
        bool doubles_ijab_b = options[7];
        double(*mo_eps) = (double(*))mo_eps_in;
        int ndim = 1 + n_ia + n_iiaa + n_iiab + n_ijaa + n_ijab_a + n_ijab_b;
        double(*mo_eris)[nmo][nmo][nmo] = (double(*)[nmo][nmo][nmo])mo_eris_in;
        double(*csfs)[ndim] = (double(*)[ndim])csfs_in;
        i = csfs[p][0];
        j = csfs[p][1];
        a = csfs[p][2];
        b = csfs[p][3];
        double row[ndim];
        row[0] = sqrt(3) * (mo_eris[a][i][b][j] - mo_eris[a][j][b][i]);
        q += 1;
        if (singles)
        {
                for ( int idx = q; idx < q; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(1.5) * ((k == i) * (mo_eris[c][a][b][j] - mo_eris[c][b][a][j]) 
                                            - (k == j) * (mo_eris[c][a][b][i] - mo_eris[c][b][a][i]) 
                                            + (c == a) * (mo_eris[b][i][j][k] - mo_eris[b][j][i][k]) 
                                            - (c == b) * (mo_eris[a][i][j][k] - mo_eris[a][j][i][k]));
                }
                q += n_ia;
        }
        if (doubles_iiaa)
        {
                for ( int idx = q; idx < q + n_iiaa; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(3) * ((k == i) * (c == a) * mo_eris[c][k][b][j] 
                                          - (k == i) * (c == b) * mo_eris[c][k][a][j] 
                                          - (k == j) * (c == a) * mo_eris[c][k][b][i] 
                                          + (k == j) * (c == b) * mo_eris[c][k][a][i]);
                }
                q += n_iiaa;        
        }
        if (doubles_iiab)
        {
                for ( int idx = q; idx < q + n_iiab; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(1.5) * ((k == i) * (c == a) * mo_eris[d][k][b][j] 
                                            - (k == i) * (c == b) * mo_eris[d][k][a][j] 
                                            + (k == i) * (d == a) * mo_eris[c][k][b][j] 
                                            - (k == i) * (d == b) * mo_eris[c][k][a][j] 
                                            - (k == j) * (c == a) * mo_eris[d][k][b][i] 
                                            + (k == j) * (c == b) * mo_eris[d][k][a][i] 
                                            - (k == j) * (d == a) * mo_eris[c][k][b][i] 
                                            + (k == j) * (d == b) * mo_eris[c][k][a][i]);
                }
                q += n_iiab;
        }
        if (doubles_ijaa)
        {
                for ( int idx = q; idx < q + n_ijaa; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(1.5) * ((k == i) * (c == a) * (mo_eris[c][l][b][j]) 
                                            - (k == i) * (c == b) * (mo_eris[c][l][a][j]) 
                                            - (k == j) * (c == a) * (mo_eris[c][l][b][i]) 
                                            + (k == j) * (c == b) * (mo_eris[c][l][a][i]) 
                                            + (l == i) * (c == a) * (mo_eris[c][k][b][j]) 
                                            - (l == i) * (c == b) * (mo_eris[c][k][a][j]) 
                                            - (l == j) * (c == a) * (mo_eris[c][k][b][i]) 
                                            + (l == j) * (c == b) * (mo_eris[c][k][a][i]));
                }
                q += n_ijaa;
        }
        if (doubles_ijab_a)
        {
                for ( int idx = q; idx < q + n_ijab_a; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((i == k) * (j == l) * (a == c) * (b == d) * (E0 - mo_eps[i] - mo_eps[j] + mo_eps[a] + mo_eps[b]) 
                                + (i == k) * (a == c) * (1.5 * mo_eris[b][j][d][l] - mo_eris[b][d][l][j]) 
                                - (i == k) * (a == d) * (1.5 * mo_eris[b][j][c][l] - mo_eris[b][c][l][j]) 
                                - (i == k) * (b == c) * (1.5 * mo_eris[a][j][d][l] - mo_eris[a][d][l][j]) 
                                + (i == k) * (b == d) * (1.5 * mo_eris[a][j][c][l] - mo_eris[a][c][l][j]) 
                                - (i == l) * (a == c) * (1.5 * mo_eris[b][j][d][k] - mo_eris[b][d][k][j]) 
                                + (i == l) * (a == d) * (1.5 * mo_eris[b][j][c][k] - mo_eris[b][c][k][j]) 
                                + (i == l) * (b == c) * (1.5 * mo_eris[a][j][d][k] - mo_eris[a][d][k][j]) 
                                - (i == l) * (b == d) * (1.5 * mo_eris[a][j][c][k] - mo_eris[a][c][k][j]) 
                                - (j == k) * (a == c) * (1.5 * mo_eris[b][i][d][l] - mo_eris[b][d][l][i]) 
                                + (j == k) * (a == d) * (1.5 * mo_eris[b][i][c][l] - mo_eris[b][c][l][i]) 
                                + (j == k) * (b == c) * (1.5 * mo_eris[a][i][d][l] - mo_eris[a][d][l][i]) 
                                - (j == k) * (b == d) * (1.5 * mo_eris[a][i][c][l] - mo_eris[a][c][l][i])
                                + (j == l) * (a == c) * (1.5 * mo_eris[b][i][d][k] - mo_eris[b][d][k][i]) 
                                - (j == l) * (a == d) * (1.5 * mo_eris[b][i][c][k] - mo_eris[b][c][k][i]) 
                                - (j == l) * (b == c) * (1.5 * mo_eris[a][i][d][k] - mo_eris[a][d][k][i]) 
                                + (j == l) * (b == d) * (1.5 * mo_eris[a][i][c][k] - mo_eris[a][c][k][i]) 
                                + (i == k) * (j == l) * (mo_eris[a][c][d][b] - mo_eris[a][d][c][b]) 
                                + (a == c) * (b == d) * (mo_eris[i][k][l][j] - mo_eris[i][l][k][j]));
                }
                q += n_ijab_a;
        }
        if (doubles_ijab_b)
        {
                for ( int idx = q; idx < q + n_ijab_b; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(0.75) * ((i == k) * (a == c) * (mo_eris[b][j][d][l]) 
                                             + (i == k) * (a == d) * (mo_eris[b][j][c][l]) 
                                             - (i == k) * (b == c) * (mo_eris[a][j][d][l]) 
                                             - (i == k) * (b == d) * (mo_eris[a][j][c][l]) 
                                             + (i == l) * (a == c) * (mo_eris[b][j][d][k]) 
                                             + (i == l) * (a == d) * (mo_eris[b][j][c][k]) 
                                             - (i == l) * (b == c) * (mo_eris[a][j][d][k]) 
                                             - (i == l) * (b == d) * (mo_eris[a][j][c][k]) 
                                             - (j == k) * (a == c) * (mo_eris[b][i][d][l]) 
                                             - (j == k) * (a == d) * (mo_eris[b][i][c][l]) 
                                             + (j == k) * (b == c) * (mo_eris[a][i][d][l]) 
                                             + (j == k) * (b == d) * (mo_eris[a][i][c][l]) 
                                             - (j == l) * (a == c) * (mo_eris[b][i][d][k]) 
                                             - (j == l) * (a == d) * (mo_eris[b][i][c][k])
                                             + (j == l) * (b == c) * (mo_eris[a][i][d][k]) 
                                             + (j == l) * (b == d) * (mo_eris[a][i][c][k]));
                }
                q += n_ijab_b;
        }
        return row;
};

static PyObject*
comp_hrow_ijab_b_cfunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
        int i, j, k, l, a, b, c, d;
        int q  = 0; 
        int n_ia = num_csfs[1]; 
        int n_iiaa = num_csfs[2]; 
        int n_iiab = num_csfs[3]; 
        int n_ijaa = num_csfs[4]; 
        int n_ijab_a = num_csfs[5]; 
        int n_ijab_b = num_csfs[6]; 
        double E0 = scf_energy;
        bool singles = options[0]; 
        bool full_cis = options[1]; 
        bool doubles = options[2]; 
        bool doubles_iiaa = options[3]; 
        bool doubles_iiab = options[4]; 
        bool doubles_ijaa = options[5];
        bool doubles_ijab_a = options[6];
        bool doubles_ijab_b = options[7];
        double(*mo_eps) = (double(*))mo_eps_in;
        int ndim = 1 + n_ia + n_iiaa + n_iiab + n_ijaa + n_ijab_a + n_ijab_b;
        double(*mo_eris)[nmo][nmo][nmo] = (double(*)[nmo][nmo][nmo])mo_eris_in;
        double(*csfs)[ndim] = (double(*)[ndim])csfs_in;
        i = csfs[p][0];
        j = csfs[p][1];
        a = csfs[p][2];
        b = csfs[p][3];
        double row[ndim];
        row[0] = mo_eris[a][i][b][j] - mo_eris[a][j][b][i]; 
        q += 1;
        if (singles)
        {
                for ( int idx = q; idx < q; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(0.5) * ((k == i) * (mo_eris[c][a][b][j] + mo_eris[c][b][a][j]) 
                                              + (k == j) * (mo_eris[c][a][b][i] + mo_eris[c][b][a][i]) 
                                              - (c == a) * (mo_eris[b][i][j][k] + mo_eris[b][j][i][k]) 
                                              - (c == b) * (mo_eris[a][i][j][k] + mo_eris[a][j][i][k]));
                }
                q += n_ia; 
        }
        if (doubles_iiaa)
        {
                for ( int idx = q; idx < q + n_iiaa; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((k == i) * (c == a) * (mo_eris[c][k][b][j] - 2 * mo_eris[c][b][j][k]) 
                                  + (k == i) * (c == b) * (mo_eris[c][k][a][j] - 2 * mo_eris[c][a][j][k]) 
                                  + (k == j) * (c == a) * (mo_eris[c][k][b][i] - 2 * mo_eris[c][b][i][k])
                                  + (k == j) * (c == b) * (mo_eris[c][k][a][i] - 2 * mo_eris[c][a][i][k]));
                }
                q += n_iiaa;        
        }
        if (doubles_iiab)
        {
                for ( int idx = q; idx < q + n_iiab; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(0.5) * ((k == i) * (c == a) * (mo_eris[d][k][b][j] - 2 * mo_eris[d][b][j][k]) 
                                            + (k == i) * (c == b) * (mo_eris[d][k][a][j] - 2 * mo_eris[d][a][j][k]) 
                                            + (k == i) * (d == a) * (mo_eris[c][k][b][j] - 2 * mo_eris[c][b][j][k]) 
                                            + (k == i) * (d == b) * (mo_eris[c][k][a][j] - 2 * mo_eris[c][a][j][k]) 
                                            + (k == j) * (c == a) * (mo_eris[d][k][b][i] - 2 * mo_eris[d][b][i][k]) 
                                            + (k == j) * (c == b) * (mo_eris[d][k][a][i] - 2 * mo_eris[d][a][i][k]) 
                                            + (k == j) * (d == a) * (mo_eris[c][k][b][i] - 2 * mo_eris[c][b][i][k]) 
                                            + (k == j) * (d == b) * (mo_eris[c][k][a][i] - 2 * mo_eris[c][a][i][k]) 
                                            + (c == a) * (d == b) * 2 * mo_eris[i][k][j][k]);
                }
                q += n_iiab;
        }

        if (doubles_ijaa)
        {
                for ( int idx = q; idx < q + n_ijaa; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(0.5) * ((k == i) * (c == a) * (mo_eris[c][l][b][j] - 2 * mo_eris[c][b][l][j]) 
                                            + (k == i) * (c == b) * (mo_eris[c][l][a][j] - 2 * mo_eris[c][a][l][j]) 
                                            + (k == j) * (c == a) * (mo_eris[c][l][b][i] - 2 * mo_eris[c][b][l][i]) 
                                            + (k == j) * (c == b) * (mo_eris[c][l][a][i] - 2 * mo_eris[c][a][l][i]) 
                                            + (l == i) * (c == a) * (mo_eris[c][k][b][j] - 2 * mo_eris[c][b][k][j]) 
                                            + (l == i) * (c == b) * (mo_eris[c][k][a][j] - 2 * mo_eris[c][a][k][j]) 
                                            + (l == j) * (c == a) * (mo_eris[c][k][b][i] - 2 * mo_eris[c][b][k][i]) 
                                            + (l == j) * (c == b) * (mo_eris[c][k][a][i] - 2 * mo_eris[c][a][k][i]) 
                                            + (k == i) * (l == j) * 2 * mo_eris[a][c][b][c]);
                }
                q += n_ijaa;
        }
        if (doubles_ijab_a)
        {
                for ( int idx = q; idx < q + n_ijab_a; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(0.75) * ((k == i) * (c == a) * (mo_eris[d][l][b][j]) 
                                             + (k == i) * (c == b) * (mo_eris[d][l][a][j]) 
                                             - (k == i) * (d == a) * (mo_eris[c][l][b][j]) 
                                             - (k == i) * (d == b) * (mo_eris[c][l][a][j]) 
                                             + (k == j) * (c == a) * (mo_eris[d][l][b][i]) 
                                             + (k == j) * (c == b) * (mo_eris[d][l][a][i]) 
                                             - (k == j) * (d == a) * (mo_eris[c][l][b][i]) 
                                             - (k == j) * (d == b) * (mo_eris[c][l][a][i]) 
                                             - (l == i) * (c == a) * (mo_eris[d][k][b][j]) 
                                             - (l == i) * (c == b) * (mo_eris[d][k][a][j]) 
                                             + (l == i) * (d == a) * (mo_eris[c][k][b][j]) 
                                             + (l == i) * (d == b) * (mo_eris[c][k][a][j]) 
                                             - (l == j) * (c == a) * (mo_eris[d][k][b][i]) 
                                             - (l == j) * (c == b) * (mo_eris[d][k][a][i]) 
                                             + (l == j) * (d == a) * (mo_eris[c][k][b][i]) 
                                             + (l == j) * (d == b) * (mo_eris[c][k][a][i]));
                }
                q += n_ijab_a;
        }
        if (doubles_ijab_b)
        {
                for ( int idx = q; idx < q + n_ijab_b; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((i == k) * (j == l) * (a == c) * (b == d) * (E0 - mo_eps[i] - mo_eps[j] + mo_eps[a] + mo_eps[b]) 
                                + (i == k) * (a == c) * (0.5 * mo_eris[b][j][d][l] - mo_eris[b][d][l][j]) 
                                + (i == k) * (a == d) * (0.5 * mo_eris[b][j][c][l] - mo_eris[b][c][l][j]) 
                                + (i == k) * (b == c) * (0.5 * mo_eris[a][j][d][l] - mo_eris[a][d][l][j]) 
                                + (i == k) * (b == d) * (0.5 * mo_eris[a][j][c][l] - mo_eris[a][c][l][j]) 
                                + (i == l) * (a == c) * (0.5 * mo_eris[b][j][d][k] - mo_eris[b][d][j][k]) 
                                + (i == l) * (a == d) * (0.5 * mo_eris[b][j][c][k] - mo_eris[b][c][k][j]) 
                                + (i == l) * (b == c) * (0.5 * mo_eris[a][j][d][k] - mo_eris[a][d][k][j]) 
                                + (i == l) * (b == d) * (0.5 * mo_eris[a][j][c][k] - mo_eris[a][c][k][j]) 
                                + (j == k) * (a == c) * (0.5 * mo_eris[b][i][d][l] - mo_eris[b][d][l][i]) 
                                + (j == k) * (a == d) * (0.5 * mo_eris[b][i][c][l] - mo_eris[b][c][l][i]) 
                                + (j == k) * (b == c) * (0.5 * mo_eris[a][i][d][l] - mo_eris[a][d][l][i]) 
                                + (j == k) * (b == d) * (0.5 * mo_eris[a][i][c][l] - mo_eris[a][c][l][i]) 
                                + (j == l) * (a == c) * (0.5 * mo_eris[b][i][d][k] - mo_eris[b][d][k][i]) 
                                + (j == l) * (a == d) * (0.5 * mo_eris[b][i][c][k] - mo_eris[b][c][k][i]) 
                                + (j == l) * (b == c) * (0.5 * mo_eris[a][i][d][k] - mo_eris[a][d][k][i]) 
                                + (j == l) * (b == d) * (0.5 * mo_eris[a][i][c][k] - mo_eris[a][c][k][i]) 
                                + (i == k) * (j == l) * (mo_eris[a][c][d][b] + mo_eris[a][d][c][b]) 
                                + (a == c) * (b == d) * (mo_eris[i][k][j][l] + mo_eris[i][l][k][j]));
                }
                q += n_ijab_b;
        }
        return row;
};

static PyObject*
comp_oeprop_hf_cfunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
        int i, j, k, l, a, b, c, d;
        int q  = 0; 
        int n_ia = num_csfs[1]; 
        int n_iiaa = num_csfs[2]; 
        int n_iiab = num_csfs[3]; 
        int n_ijaa = num_csfs[4]; 
        int n_ijab_a = num_csfs[5]; 
        int n_ijab_b = num_csfs[6]; 
        bool singles = options[0]; 
        bool full_cis = options[1]; 
        bool doubles = options[2]; 
        bool doubles_iiaa = options[3]; 
        bool doubles_iiab = options[4]; 
        bool doubles_ijaa = options[5];
        bool doubles_ijab_a = options[6];
        bool doubles_ijab_b = options[7];
        int ndim = 1 + n_ia + n_iiaa + n_iiab + n_ijaa + n_ijab_a + n_ijab_b;
        double(*csfs)[ndim] = (double(*)[ndim])csfs_in;
        double(*mo_oeprop)[nmo] = (double(*)[nmo])mo_oeprop_in;
        i = csfs[p][0];
        j = csfs[p][1];
        a = csfs[p][2];
        b = csfs[p][3];
        double row[ndim];
        row[0] = mo_oeprop_trace;
        if (singles){
                for (int idx = q; idx < q + n_ia; ++idx){
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(2) * mo_oeprop[k][c];
                }
                q += n_ia;
        }
        if (doubles){
                if (doubles_iiaa){
                        q += n_iiaa;
                }
                if (doubles_iiab){
                        q += n_iiab;
                }
                if (doubles_ijaa){
                        for (int idx = q; idx < q + n_ijaa; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = 0.0;
                        }
                        q += n_ijaa;
                }
                if (doubles_ijab_a){
                        for (int idx = q; idx < q + n_ijab_a; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = 0.0;
                        }
                        q += n_ijab_a;
                }
                if (doubles_ijab_b){
                        for (int idx = q; idx < q + n_ijab_b; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = 0.0;
                        }
                        q += n_ijab_b;
                } 
        }
        return row;
};

static PyObject*
comp_oeprop_ia_cfunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
        int i, j, k, l, a, b, c, d;
        int q = 0;
        int n_ia = num_csfs[1];
        int n_iiaa = num_csfs[2];
        int n_iiab = num_csfs[3];
        int n_ijaa = num_csfs[4];
        int n_ijab_a = num_csfs[5];
        int n_ijab_b = num_csfs[6];
        bool singles = options[0];
        bool full_cis = options[1];
        bool doubles = options[2];
        bool doubles_iiaa = options[3];
        bool doubles_iiab = options[4];
        bool doubles_ijaa = options[5];
        bool doubles_ijab_a = options[6];
        bool doubles_ijab_b = options[7];
        int ndim = 1 + n_ia + n_iiaa + n_iiab + n_ijaa + n_ijab_a + n_ijab_b;
        double(*csfs)[ndim] = (double(*)[ndim])csfs_in;
        double(*mo_oeprop)[nmo] = (double(*)[nmo])mo_oeprop_in;
        i = csfs[p][0];
        j = csfs[p][1];
        a = csfs[p][2];
        b = csfs[p][3];
        double row[ndim];
        row[0] = sqrt(2) * mo_oeprop[i][a];
        if (singles)
        {
                for (int idx = q; idx < q + n_ia; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((i == k) * (a == c) * mo_oeprop_trace 
                                  - (a == c) * mo_oeprop[k][i] 
                                  + (i == k) * mo_oeprop[a][c]);
                }
                q += n_ia;
        }
        if (doubles)
        {
                if (doubles_iiaa)
                {
                        for (int idx = q; idx < q + n_iiaa; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = (i == k) * (a == c) * sqrt(2) * mo_oeprop[i][a];
                        }
                        q += n_iiaa;
                }
                if (doubles_iiab)
                {
                        for (int idx = q; idx < q + n_iiab; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = ((i == k) * (a == c) * mo_oeprop[i][d] 
                                          + (i == k) * (a == d) * mo_oeprop[i][c]);
                        }
                        q += n_iiab;
                }
                if (doubles_ijaa)
                {
                        for (int idx = q; idx < q + n_ijaa; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = ((i == k) * (a == c) * mo_oeprop[l][a] 
                                          + (i == l) * (a == c) * mo_oeprop[k][a]);
                        }
                        q += n_ijaa;
                }
                if (doubles_ijab_a)
                {
                        for (int idx = q; idx < q + n_ijab_a; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = sqrt(1.5) * ((i == k) * (a == c) * mo_oeprop[l][d] 
                                                      - (i == k) * (a == d) * mo_oeprop[l][c] 
                                                      - (i == l) * (a == c) * mo_oeprop[k][d] 
                                                      + (i == l) * (a == d) * mo_oeprop[k][c]);
                        }
                        q += n_ijab_a;
                }
                if (doubles_ijab_b)
                {
                        for (int idx = q; idx < q + n_ijab_b; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = sqrt(0.5) * ((i == k) * (a == c) * mo_oeprop[l][d] 
                                                      + (i == k) * (a == d) * mo_oeprop[l][c] 
                                                      + (i == l) * (a == c) * mo_oeprop[k][d] 
                                                      + (i == l) * (a == d) * mo_oeprop[k][c]);
                        }
                        q += n_ijab_b;
                }
        }
        return row;
};

static PyObject*
comp_oeprop_iiaa_cfunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
        int i, j, k, l, a, b, c, d;
        int q = 0;
        int n_ia = num_csfs[1];
        int n_iiaa = num_csfs[2];
        int n_iiab = num_csfs[3];
        int n_ijaa = num_csfs[4];
        int n_ijab_a = num_csfs[5];
        int n_ijab_b = num_csfs[6];
        bool singles = options[0];
        bool full_cis = options[1];
        bool doubles = options[2];
        bool doubles_iiaa = options[3];
        bool doubles_iiab = options[4];
        bool doubles_ijaa = options[5];
        bool doubles_ijab_a = options[6];
        bool doubles_ijab_b = options[7];
        int ndim = 1 + n_ia + n_iiaa + n_iiab + n_ijaa + n_ijab_a + n_ijab_b;
        double(*csfs)[ndim] = (double(*)[ndim])csfs_in;
        double(*mo_oeprop)[nmo] = (double(*)[nmo])mo_oeprop_in;
        i = csfs[p][0];
        j = csfs[p][1];
        a = csfs[p][2];
        b = csfs[p][3];
        double row[ndim];
        row[0] = 0.0;
        if (singles)
        {
                for (int idx = q; idx < q + n_ia; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = (k == i) * (c == a) * sqrt(2) * mo_oeprop[k][c];
                }
                q += n_ia;
        }
        if (doubles)
        {
                if (doubles_iiaa)
                {
                        for (int idx = q; idx < q + n_iiaa; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = (i == k) * (a == c) * (mo_oeprop_trace 
                                                            - 2 * mo_oeprop[i][i] 
                                                            + 2 * mo_oeprop[a][a]);
                        }
                        q += n_iiaa;
                }
                if (doubles_iiab)
                {
                        for (int idx = q; idx < q + n_iiab; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = sqrt(2) * ((i == k) * (a == c) * mo_oeprop[a][d] 
                                                    + (i == k) * (a == d) * mo_oeprop[a][c]);
                        }
                        q += n_iiab;
                }
                if (doubles_ijaa)
                {
                        for (int idx = q; idx < q + n_ijaa; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = - sqrt(2) * ((i == k) * (a == c) * mo_oeprop[l][i] 
                                                      + (i == l) * (a == c) * mo_oeprop[k][i]);
                        }
                        q += n_ijaa;
                }
                if (doubles_ijab_a)
                {
                        for (int idx = q; idx < q + n_ijab_a; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = 0.0;
                        }
                        q += n_ijab_a;
                }
                if (doubles_ijab_b)
                {
                        for (int idx = q; idx < q + n_ijab_b; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = 0.0;
                        }
                        q += n_ijab_b;
                }
        }
        return row;
};

static PyObject*
comp_oeprop_iiab_cfunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
        int i, j, k, l, a, b, c, d;
        int q = 0;
        int n_ia = num_csfs[1];
        int n_iiaa = num_csfs[2];
        int n_iiab = num_csfs[3];
        int n_ijaa = num_csfs[4];
        int n_ijab_a = num_csfs[5];
        int n_ijab_b = num_csfs[6];
        bool singles = options[0];
        bool full_cis = options[1];
        bool doubles = options[2];
        bool doubles_iiaa = options[3];
        bool doubles_iiab = options[4];
        bool doubles_ijaa = options[5];
        bool doubles_ijab_a = options[6];
        bool doubles_ijab_b = options[7];
        int ndim = 1 + n_ia + n_iiaa + n_iiab + n_ijaa + n_ijab_a + n_ijab_b;
        double(*csfs)[ndim] = (double(*)[ndim])csfs_in;
        double(*mo_oeprop)[nmo] = (double(*)[nmo])mo_oeprop_in;
        i = csfs[p][0];
        j = csfs[p][1];
        a = csfs[p][2];
        b = csfs[p][3];
        double row[ndim];
        row[0] = 0.0;
        if (singles)
        {
                for (int idx = q; idx < q + n_ia; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((k == i) * (c == a) * mo_oeprop[k][b] 
                                  + (k == i) * (c == b) * mo_oeprop[k][a]);
                }
                q += n_ia;
        }
        if (doubles)
        {
                if (doubles_iiaa)
                {
                        for (int idx = q; idx < q + n_iiaa; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = sqrt(2) * ((k == i) * (c == a) * mo_oeprop[c][b] 
                                                    + (k == i) * (c == b) * mo_oeprop[c][a]);
                        }
                        q += n_iiaa;
                }
                if (doubles_iiab)
                {
                        for (int idx = q; idx < q + n_iiab; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = ((i == k) * (a == c) * (b == d) * (mo_oeprop_trace - 2 * mo_oeprop[i][i]) 
                                          + (i == k) * (a == c) * mo_oeprop[b][d] 
                                          + (i == k) * (a == d) * mo_oeprop[b][c] 
                                          + (i == k) * (b == c) * mo_oeprop[a][d] 
                                          + (i == k) * (b == d) * mo_oeprop[a][c]);
                        }
                        q += n_iiab;
                }
                if (doubles_ijaa)
                {
                        for (int idx = q; idx < q + n_ijaa; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = 0.0;
                        }
                        q += n_ijaa;
                }
                if (doubles_ijab_a)
                {
                        for (int idx = q; idx < q + n_ijab_a; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = 0.0;
                        }
                        q += n_ijab_a;
                }
                if (doubles_ijab_b)
                {
                        for (int idx = q; idx < q + n_ijab_b; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = -sqrt(2) * ((i == k) * (a == c) * (b == d) * mo_oeprop[l][i] 
                                                     + (i == l) * (a == c) * (b == d) * mo_oeprop[k][i]);
                        }
                        q += n_ijab_b;
                }
        }
        return row;
};

static PyObject*
comp_oeprop_ijaa_cfunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
        int i, j, k, l, a, b, c, d;
        int q = 0;
        int n_ia = num_csfs[1];
        int n_iiaa = num_csfs[2];
        int n_iiab = num_csfs[3];
        int n_ijaa = num_csfs[4];
        int n_ijab_a = num_csfs[5];
        int n_ijab_b = num_csfs[6];
        bool singles = options[0];
        bool full_cis = options[1];
        bool doubles = options[2];
        bool doubles_iiaa = options[3];
        bool doubles_iiab = options[4];
        bool doubles_ijaa = options[5];
        bool doubles_ijab_a = options[6];
        bool doubles_ijab_b = options[7];
        int ndim = 1 + n_ia + n_iiaa + n_iiab + n_ijaa + n_ijab_a + n_ijab_b;
        double(*csfs)[ndim] = (double(*)[ndim])csfs_in;
        double(*mo_oeprop)[nmo] = (double(*)[nmo])mo_oeprop_in;
        i = csfs[p][0];
        j = csfs[p][1];
        a = csfs[p][2];
        b = csfs[p][3];
        double row[ndim];
        row[0] = 0.0; 
        if (singles)
        {
                for (int idx = q; idx < q + n_ia; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = ((k == i) * (c == a) * mo_oeprop[j][c] 
                                  + (k == j) * (c == a) * mo_oeprop[i][c]);
                }
                q += n_ia;
        }
        if (doubles)
        {
                if (doubles_iiaa)
                {
                        for (int idx = q; idx < q + n_iiaa; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = -sqrt(2) * ((k == i) * (c == a) * mo_oeprop[j][k] 
                                                     + (k == j) * (c == a) * mo_oeprop[i][k]);
                        }
                        q += n_iiaa;
                }
                if (doubles_iiab)
                {
                        for (int idx = q; idx < q + n_iiab; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = 0.0;
                        }
                        q += n_iiab;
                }
                if (doubles_ijaa)
                {
                        for (int idx = q; idx < q + n_ijaa; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = ((i == k) * (j == l) * (a == c) * (mo_oeprop_trace + 2 * mo_oeprop[a][a]) 
                                          - (i == k) * (a == c) * mo_oeprop[l][j] 
                                          - (i == l) * (a == c) * mo_oeprop[k][j] 
                                          - (j == k) * (a == c) * mo_oeprop[l][i] 
                                          - (j == l) * (a == c) * mo_oeprop[k][i]);
                        }
                        q += n_ijaa;
                }
                if (doubles_ijab_a)
                {
                        for (int idx = q; idx < q + n_ijab_a; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = 0.0;
                        }
                        q += n_ijab_a;
                }
                if (doubles_ijab_b)
                {
                        for (int idx = q; idx < q + n_ijab_b; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = sqrt(2) * ((i == k) * (j == l) * (a == c) * mo_oeprop[a][d] 
                                                    + (i == k) * (j == l) * (a == d) * mo_oeprop[a][c]);
                        }
                        q += n_ijab_b;
                }
        }
        return row;
};

static PyObject*
comp_oeprop_ijab_a_cfunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
        int i, j, k, l, a, b, c, d;
        int q = 0;
        int n_ia = num_csfs[1];
        int n_iiaa = num_csfs[2];
        int n_iiab = num_csfs[3];
        int n_ijaa = num_csfs[4];
        int n_ijab_a = num_csfs[5];
        int n_ijab_b = num_csfs[6];
        bool singles = options[0];
        bool full_cis = options[1];
        bool doubles = options[2];
        bool doubles_iiaa = options[3];
        bool doubles_iiab = options[4];
        bool doubles_ijaa = options[5];
        bool doubles_ijab_a = options[6];
        bool doubles_ijab_b = options[7];
        int ndim = 1 + n_ia + n_iiaa + n_iiab + n_ijaa + n_ijab_a + n_ijab_b;
        double(*csfs)[ndim] = (double(*)[ndim])csfs_in;
        double(*mo_oeprop)[nmo] = (double(*)[nmo])mo_oeprop_in;
        i = csfs[p][0];
        j = csfs[p][1];
        a = csfs[p][2];
        b = csfs[p][3];
        double row[ndim];
        row[0] = 0.0;
        if (singles)
        {
                for (int idx = q; idx < q + n_ia; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(1.5) * ((k == i) * (c == a) * mo_oeprop[j][b] 
                                              - (k == i) * (c == b) * mo_oeprop[j][a] 
                                              - (k == j) * (c == a) * mo_oeprop[i][b] 
                                              + (k == j) * (c == b) * mo_oeprop[i][a]);
                }
                q += n_ia;
        }
        if (doubles)
        {
                if (doubles_iiaa)
                {
                        for (int idx = q; idx < q + n_iiaa; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = 0.0;
                        }
                        q += n_iiaa;
                }
                if (doubles_iiab)
                {
                        for (int idx = q; idx < q + n_iiab; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = 0.0;
                        }
                        q += n_iiab;
                }
                if (doubles_ijaa)
                {
                        for (int idx = q; idx < q + n_ijaa; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = 0.0;
                        }
                        q += n_ijaa;
                }
                if (doubles_ijab_a)
                {
                        for (int idx = q; idx < q + n_ijab_a; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = ((i == k) * (j == l) * (a == c) * (b == d) * mo_oeprop_trace 
                                          - (i == k) * (a == c) * (b == d) * mo_oeprop[l][j] 
                                          + (i == l) * (a == c) * (b == d) * mo_oeprop[k][j] 
                                          + (j == k) * (a == c) * (b == d) * mo_oeprop[l][i] 
                                          - (j == l) * (a == c) * (b == d) * mo_oeprop[k][i] 
                                          + (i == k) * (j == l) * (a == c) * mo_oeprop[b][d] 
                                          - (i == k) * (j == l) * (a == d) * mo_oeprop[b][c] 
                                          - (i == k) * (j == l) * (b == c) * mo_oeprop[a][d] 
                                          + (i == k) * (j == l) * (b == d) * mo_oeprop[a][c]);
                        }
                        q += n_ijab_a;
                }
                if (doubles_ijab_b)
                {
                        for (int idx = q; idx < q + n_ijab_b; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = 0.0;
                        }
                        q += n_ijab_b;
                }
        }
        return row;
};

static PyObject*
comp_oeprop_ijab_b_cfunc(PyObject *self, PyObject *args, PyObject *kwargs)
{
        int i, j, k, l, a, b, c, d;
        int q = 0;
        int n_ia = num_csfs[1];
        int n_iiaa = num_csfs[2];
        int n_iiab = num_csfs[3];
        int n_ijaa = num_csfs[4];
        int n_ijab_a = num_csfs[5];
        int n_ijab_b = num_csfs[6];
        bool singles = options[0];
        bool full_cis = options[1];
        bool doubles = options[2];
        bool doubles_iiaa = options[3];
        bool doubles_iiab = options[4];
        bool doubles_ijaa = options[5];
        bool doubles_ijab_a = options[6];
        bool doubles_ijab_b = options[7];
        int ndim = 1 + n_ia + n_iiaa + n_iiab + n_ijaa + n_ijab_a + n_ijab_b;
        double(*csfs)[ndim] = (double(*)[ndim])csfs_in;
        double(*mo_oeprop)[nmo] = (double(*)[nmo])mo_oeprop_in;
        i = csfs[p][0];
        j = csfs[p][1];
        a = csfs[p][2];
        b = csfs[p][3];
        double row[ndim];
        row[0] = 0.0;
        if (singles)
        {
                for (int idx = q; idx < q + n_ia; ++idx)
                {
                        k = csfs[idx][0];
                        l = csfs[idx][1];
                        c = csfs[idx][2];
                        d = csfs[idx][3];
                        row[idx] = sqrt(0.5) * ((k == i) * (c == a) * mo_oeprop[j][b] 
                                              + (k == i) * (c == b) * mo_oeprop[j][a] 
                                              + (k == j) * (c == a) * mo_oeprop[i][b] 
                                              + (k == j) * (c == b) * mo_oeprop[i][a]);
                }
                q += n_ia;
        }
        if (doubles)
        {
                if (doubles_iiaa)
                {
                        for (int idx = q; idx < q + n_iiaa; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = 0.0;
                        }
                        q += n_iiaa;
                }
                if (doubles_iiab)
                {
                        for (int idx = q; idx < q + n_iiab; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = -sqrt(2) * ((k == i) * (c == a) * (d == b) * mo_oeprop[j][k] 
                                                     + (k == j) * (c == a) * (d == b) * mo_oeprop[i][k]);
                        }
                        q += n_iiab;
                }
                if (doubles_ijaa)
                {
                        for (int idx = q; idx < q + n_ijaa; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = sqrt(2) * ((k == i) * (l == j) * (c == a) * mo_oeprop[c][b]
                                                    + (k == i) * (l == j) * (c == b) * mo_oeprop[c][a]);
                        }
                        q += n_ijaa;
                }
                if (doubles_ijab_a)
                {
                        for (int idx = q; idx < q + n_ijab_a; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = 0.0;
                        }
                        q += n_ijab_a;
                }
                if (doubles_ijab_b)
                {
                        for (int idx = q; idx < q + n_ijab_b; ++idx)
                        {
                                k = csfs[idx][0];
                                l = csfs[idx][1];
                                c = csfs[idx][2];
                                d = csfs[idx][3];
                                row[idx] = ((i==k)*(j==l)*(a==c)*(b==d)*mo_oeprop_trace
                                          - (i==k)*(a==c)*(b==d)*mo_oeprop[l][j]
                                          - (i==l)*(a==c)*(b==d)*mo_oeprop[k][j]
                                          - (j==k)*(a==c)*(b==d)*mo_oeprop[l][i]
                                          - (j==l)*(a==c)*(b==d)*mo_oeprop[k][i]
                                          + (i==k)*(j==l)*(a==c)*mo_oeprop[b][d]
                                          + (i==k)*(j==l)*(a==d)*mo_oeprop[b][c]
                                          + (i==k)*(j==l)*(b==c)*mo_oeprop[a][d]
                                          + (i==k)*(j==l)*(b==d)*mo_oeprop[a][c]);
                        }
                        q += n_ijab_b;
                }
        }
        return row;
};
