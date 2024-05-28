project(
  'pixell',
  'c', 'cpp', 'fortran',
  meson_version: '>= 1.1.0',
  default_options: [
    'c_std=c99',
  ],
)

c_compiler = meson.get_compiler('c')
cpp_compiler = meson.get_compiler('cpp')
add_languages('fortran', native: false)
f_compiler = meson.get_compiler('fortran')
system_os = host_machine.system()

# Check compiler is recent enough (see "Toolchain Roadmap" for details)
if system_os == 'windows'
    error('Pixell does not support Windows')
endif

add_project_arguments(['-fopenmp', '-Wno-strict-aliasing', '-g', '-Ofast', '-fPIC'], language: ['c', 'cpp'])
add_project_link_arguments(['-fopenmp', '-g', '-fPIC', '-fno-lto'], language: ['c', 'cpp'])
add_project_arguments(['-O3', '-fPIC'], language: 'fortran')


# https://mesonbuild.com/Python-module.html
py_mod = import('python')
# NOTE: with Meson >=0.64.0 we can add `pure: false` here and remove that line
# everywhere else, see https://github.com/mesonbuild/meson/pull/10783.
py3 = py_mod.find_installation()
py3_dep = py3.dependency()

subdir('pixell')