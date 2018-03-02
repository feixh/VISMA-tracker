from setuptools import setup, Extension
import numpy
import os

os.environ['CC'] = 'g++'

render_module = Extension('_simplerender',
                          ['feh_render/simplerender.cpp', 'feh_render/simplerender.i'],
                          extra_compile_args=['-std=c++11', '-O3'],
                          include_dirs=[numpy.get_include(), '.', 'include', '/usr/include/eigen3', 'thirdparty/glad/include'],
                          library_dirs=['./lib', '/usr/local/lib', '/usr/lib'],
                          libraries=['myrender']
                          )

setup(name='feh_render',
      version='1.0',
      author='Xiaohan Fei',
      packages=['feh_render'],
      ext_modules=[render_module]
      )

# You might also need to add the following libraries to libraries list of render_module
#'GLEW', 'glfw', 'GL', 'X11', 'pthread', 'Xrandr', 'Xi', 'dl', 'Xinerama', 'Xcursor'
