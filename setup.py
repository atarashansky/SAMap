from setuptools import setup, find_packages

setup(
    name="samap",
    version="0.1.1",
    description="The SAMap algorithm",
    long_description="The Self-Assembling Manifold Mapping algorithm for mapping single-cell datasets across species.",
    long_description_content_type="text/markdown",
    author="Alexander J. Tarashansky",
    author_email="tarashan@stanford.edu",
    keywords="scrnaseq analysis manifold reconstruction cross-species mapping",
    python_requires=">=3.6",
    py_modules=["SAMap"],    
    install_requires=[
        "sam-algorithm==0.7.6",
        "scanpy==1.5.1",
        "hnswlib==0.3.4",
        "matplotlib==3.1.3",
        "leidenalg==0.7.0",
    ],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
