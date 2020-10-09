import setuptools


setuptools.setup(
    name="vlin",
    version="0.0.0",
    packages=setuptools.find_packages(),
    install_requires=["cylp", "numpy"],
    python_requires=">=3.7",
    include_package_data=True,
    setup_requires=[],
)
