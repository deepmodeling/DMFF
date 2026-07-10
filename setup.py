from os import path
import setuptools
import datetime
from setuptools import find_packages

NAME = "dmff"

readme_file = path.join(path.dirname(path.abspath(__file__)), 'README.md')
try:
    from m2r import parse_from_file

    readme = parse_from_file(readme_file)
except (ImportError, AttributeError) as e:
    with open(readme_file) as f:
        readme = f.read()

today = datetime.date.today().strftime("%b-%d-%Y")
with open(path.join('dmff', '_date.py'), 'w') as fp:
    fp.write('date = \'%s\'' % today)

install_requires = [
    "numpy>=1.18",
    "jax>=0.4.1",
    "openmm>=7.6.0",
    "freud-analysis",
    "networkx>=3.0",
    "optax>=0.1.4",
    "pymbar>=4.0.0",
    "tqdm"
]

extras_require = {
    # jaxopt is archived upstream (last release 0.8.5, Nov 2023). Only the QEq
    # module needs it, and dmff/admp/qeq.py already degrades with a warning when
    # it is absent, so it is opt-in rather than a hard dependency.
    "qeq": ["jaxopt>=0.8.0"],
    "docs": [
        "mkdocs>=1.3.0",
        "mkdocs-autorefs>=0.4.1",
        "mkdocs-gen-files>=0.3.4",
        "mkdocs-literate-nav>=0.4.1",
        "mkdocstrings>=0.19.0",
        "mkdocstrings-python>=0.7.0",
        "pygments>=2.12",
    ],
}


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
        python_requires=">=3.9",
        packages=packages,
        data_files=[],
        package_data={},
        classifiers=[
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        ],
        keywords='DMFF',
        install_requires=install_requires,
        extras_require=extras_require,
        entry_points={}
    )


try:
    setup(scm={'write_to': 'dmff/_version.py'})
except:
    setup(scm=None)
