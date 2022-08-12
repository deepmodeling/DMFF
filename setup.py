from os import path
import setuptools
import datetime
from setuptools import find_packages

NAME = "dmff"

readme_file = path.join(path.dirname(path.abspath(__file__)), 'README.md')
try:
    from m2r import parse_from_file

    readme = parse_from_file(readme_file)
except ImportError:
    with open(readme_file) as f:
        readme = f.read()

today = datetime.date.today().strftime("%b-%d-%Y")
with open(path.join('dmff', '_date.py'), 'w') as fp:
    fp.write('date = \'%s\'' % today)

install_requires = [
    "numpy>=1.18",
    "jax_md>=0.1.28",
    "openmm",
    "jax>=0.3.7",
    ""
]


def setup(scm=None):
    packages = find_packages(exclude=["tests"])

    setuptools.setup(
        name=NAME,
        use_scm_version=scm,
        setup_requires=['setuptools_scm'],
        author="DeepModeling",
        author_email="windwhisper.yu@gmail.com",
        description="Differentiable Molecular Force Field",
        long_description=readme,
        long_description_content_type="text/markdown",
        url="https://github.com/deepmodeling/DMFF",
        python_requires="~=3.6",
        packages=packages,
        data_files=[],
        package_data={},
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        ],
        keywords='DMFF',
        install_requires=install_requires,
        entry_points={}
    )


try:
    setup(scm={'write_to': 'dmff/_version.py'})
except:
    setup(scm=None)
