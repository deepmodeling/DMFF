from distutils.core import setup
from distutils.extension import Extension
import os
import platform

openmm_dir = '@OPENMM_DIR@'
CPPFLOW_DIR = '@CPPFLOW_DIR@'
TENSORFLOW_DIR = '@TENSORFLOW_DIR@'
DMFFPlugin_header_dir = '@DMFFPLUGIN_HEADER_DIR@'
DMFFPlugin_library_dir = '@DMFFPLUGIN_LIBRARY_DIR@'

os.environ["CC"] = "@CMAKE_C_COMPILER@"
os.environ["CXX"] = "@CMAKE_CXX_COMPILER@"

extra_compile_args = ["-std=c++17", "-fPIC"]
extra_link_args = []


# setup extra compile and link arguments on Mac
if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='OpenMMDMFFPlugin._OpenMMDMFFPlugin',
                      sources=['OpenMMDMFFPluginWrapper.cpp'],
                      libraries=['OpenMM', 'OpenMMDMFF'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), os.path.join(CPPFLOW_DIR, 'include'), os.path.join(TENSORFLOW_DIR, 'include'), DMFFPlugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), os.path.join(CPPFLOW_DIR, 'lib'), os.path.join(TENSORFLOW_DIR, 'lib'), DMFFPlugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )



setup(name='OpenMMDMFFPlugin',
      version="@GIT_HASH@",
      ext_modules=[extension],
      packages=['OpenMMDMFFPlugin', "OpenMMDMFFPlugin.tests"],
      package_data={"OpenMMDMFFPlugin":['data/lj_fluid/*.pb', 'data/lj_fluid/variables/variables.index', 'data/lj_fluid/variables/variables.data-00000-of-00001', 'data/lj_fluid_gpu/*.pb', 'data/lj_fluid_gpu/variables/variables.index', 'data/lj_fluid_gpu/variables/variables.data-00000-of-00001', 'data/*.pdb']},
     )
