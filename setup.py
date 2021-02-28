from setuptools import setup, find_packages

setup(
    name="samap",
    version="0.1.5",
    description="The SAMap algorithm",
    long_description="The Self-Assembling Manifold Mapping algorithm for mapping single-cell datasets across species.",
    long_description_content_type="text/markdown",
    author="Alexander J. Tarashansky",
    url="https://github.com/atarashansky/SAMap",
    author_email="tarashan@stanford.edu",
    keywords="scrnaseq analysis manifold reconstruction cross-species mapping",
    python_requires=">=3.7",
    install_requires=[
        "sam-algorithm>=0.8.1",
        "scanpy",
        "hnswlib",
        "dill",
        "h5py<=2.10",
        "leidenalg",
        "umap-learn<0.5.0"
    ],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
