#define PY_SSIZE_T_CLEAN
#include <Python.h>
// Declaring functions
static PyObject * 
comp_hrow_hf(PyObject *self, PyObject *args, PyObject *kwargs);

static PyObject * 
comp_hrow_ia(PyObject *self, PyObject *args, PyObject *kwargs);

static PyObject * 
comp_hrow_iiaa(PyObject *self, PyObject *args, PyObject *kwargs);

static PyObject * 
comp_hrow_iiab(PyObject *self, PyObject *args, PyObject *kwargs);

static PyObject * 
comp_hrow_ijaa(PyObject *self, PyObject *args, PyObject *kwargs);

static PyObject * 
comp_hrow_ijab_a(PyObject *self, PyObject *args, PyObject *kwargs);

static PyObject * 
comp_hrow_ijab_b(PyObject *self, PyObject *args, PyObject *kwargs);

static PyObject * 
comp_oeprop_hf(PyObject *self, PyObject *args, PyObject *kwargs);

static PyObject * 
comp_oeprop_ia(PyObject *self, PyObject *args, PyObject *kwargs);

static PyObject * 
comp_oeprop_iiaa(PyObject *self, PyObject *args, PyObject *kwargs);

static PyObject * 
comp_oeprop_iiab(PyObject *self, PyObject *args, PyObject *kwargs);

static PyObject * 
comp_oeprop_ijaa(PyObject *self, PyObject *args, PyObject *kwargs);

static PyObject * 
comp_oeprop_ijab_a(PyObject *self, PyObject *args, PyObject *kwargs);

static PyObject * 
comp_oeprop_ijab_b(PyObject *self, PyObject *args, PyObject *kwargs);

