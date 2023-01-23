#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

void export_integrators(py::module &m);

PYBIND11_MODULE(_endyn, m){
    m.doc() = "endyn lib python interface";
    export_integrators(m);

}