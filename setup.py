from setuptools import setup, find_packages

setup(
    name="samap",
    version="1.0.15",
    description="The SAMap algorithm",
    long_description="The Self-Assembling Manifold Mapping algorithm for mapping single-cell datasets across species.",
    long_description_content_type="text/markdown",
    author="Alexander J. Tarashansky",
    url="https://github.com/atarashansky/SAMap",
    author_email="tarashan@stanford.edu",
    keywords="scrnaseq analysis manifold reconstruction cross-species mapping",
    python_requires=">=3.7",
    install_requires=[
        "sam-algorithm==1.0.2",
        "scipy<1.13.0",
        "numpy==1.23.5",
        "scanpy==1.9.3",
        "hnswlib==0.7.0",
        "dill",
        "numba==0.56.3",
        "h5py==3.8.0",
        "leidenalg",
        "fast-histogram",
        "holoviews-samap"
    ],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
