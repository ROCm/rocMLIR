import setuptools
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
import subprocess
import os


def hiputils_install():
  src_dir=os.path.join(os.path.dirname(__file__), 'hiputils')
  subprocess.check_call("make", cwd=src_dir, shell=True)


class CustomInstallCommand(install):
  def run(self):
    install.run(self)
    hiputils_install()


class CustomDevelopCommand(develop):
  def run(self):
    develop.run(self)
    hiputils_install()


class CustomEggInfoCommand(egg_info):
  def run(self):
    egg_info.run(self)
    hiputils_install()


setuptools.setup(
  name="migraphRunner",
  version="0.0.1",
  license="MIT",
  author="Ravil Dorozhinskii",
  author_email="ravil.aviva.com@gmail.com",
  description="migraphx runner for rocMLIR project",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  url="https://github.com/ROCmSoftwarePlatform/rocMLIR",
  python_requires='>=3.5',
  #install_requires=install_requires,
  #include_package_data=True,
  entry_points={
    'console_scripts': [
        'migraphRunner=runner.runner:main',
    ]
  },
  cmdclass={
    'install': CustomInstallCommand,
    'develop': CustomDevelopCommand,
    'egg_info': CustomEggInfoCommand,
  },
)
