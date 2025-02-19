# Main build file for pixell, helping with building the fortran,
# c, and cython extensions.
project(
    'pixell',
    ['c', 'fortran', 'cython'],
)

py = import('python').find_installation(pure: false)

# Dependencies
py_dep = py.dependency()
omp_dep = dependency('openmp')

# Libraries
cc = meson.get_compiler('c')
c_m_dep = cc.find_library('m', required: true)

fc = meson.get_compiler('fortran')
fortran_m_dep = fc.find_library('m', required: true)

# Directories
library_install_dir = py.get_install_dir() / 'pixell'

# Includes

# Need to massage these into relative paths to keep meson happy.
# It does not allow for absolute paths.
incdir_numpy = run_command(
    py,
    ['-c', 'import numpy; import os; print(os.path.relpath(numpy.get_include()))'],
    check: true
).stdout().strip()

incdir_f2py = run_command(
    py,
    ['-c', 'import numpy.f2py; import os; print(os.path.relpath(numpy.f2py.get_include()))'],
    check: true
).stdout().strip()

# Build fortran extensions

# Pixell oddity - need to run the makefile (make -C fortran)
# to generate specific fortran files before we can build them.
# TODO: put those commands in this meson file instead.
run_command('make', '-C', 'fortran', check: true)

fortran_include = include_directories(incdir_numpy, incdir_f2py)
add_project_arguments('-Wno-tabs', language : 'fortran')
add_project_arguments('-Wno-conversion', language : 'fortran')

fortran_sources = {
    'fortran/interpol_32.f90': '_interpol_32',
    'fortran/interpol_64.f90': '_interpol_64',
    'fortran/colorize.f90': '_colorize',
    'fortran/array_ops_32.f90': '_array_ops_32',
    'fortran/array_ops_64.f90': '_array_ops_64',
}

foreach source_name, module_name : fortran_sources
    f2py_output = custom_target(
        input: source_name,
        output: [module_name + '-f2pywrappers2.f90', module_name + 'module.c'],
        command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', module_name, '--lower'],
    )

    py.extension_module(
        module_name,
        [source_name, f2py_output],
        incdir_f2py / 'fortranobject.c',
        include_directories: fortran_include,
        dependencies: [py_dep, omp_dep, c_m_dep, fortran_m_dep],
        install: true,
        subdir: 'pixell'
    )
endforeach


# Build c(ython) extensions.

# Before building cython, we must build shared libraries for all of
# the underlying c code that those cython extensions rely on.

helper_sources = {
    'cython/cmisc_core.c': '_cmisc_shared',
    'cython/distances_core.c': '_distances_shared',
    'cython/srcsim_core.c': '_srcsim_shared',
}

linkables = []

foreach source_name, module_name : helper_sources
    linkables += static_library(
        module_name,
        source_name,
        install: true,
        install_dir: library_install_dir,
        include_directories: [incdir_numpy],
        dependencies: [omp_dep, c_m_dep],
    )
endforeach

# Now we can build cython and link our shared libraries to them.

cython_sources = {
    'cython/cmisc.pyx': 'cmisc',
    'cython/distances.pyx': 'distances',
    'cython/srcsim.pyx': 'srcsim',
}

foreach source_name, module_name : cython_sources
    cython_module = py.extension_module(
        module_name,
        source_name,
        include_directories: ['cython', incdir_numpy],
        dependencies: [py_dep, omp_dep, c_m_dep],
        link_with: linkables,
        install: true,
        subdir: 'pixell',
    )
endforeach

# The actual python install itself is left up to a helper build
# script deifned in pixell/
subdir('pixell')
subdir('scripts')
