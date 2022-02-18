from setuptools import setup, find_packages

install_requires = [
    "numpy",
    "jax_md",
    "openmm",
]
setup(
    name='dmff',
    version='0.0.1',
    author='dptech.net',
    # author_email='hermite@dptech.net',
    description=('DMFF.'),
    url='https://github.com/deepmodeling/DMFF',
    license=None,
    keywords='Differentiable',
    install_requires=install_requires,
    packages=find_packages(),
    zip_safe=False,
    #packages=packages,
    entry_points={},
    include_package_data=True)