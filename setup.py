import setuptools


setuptools.setup(
    name="vlin",
    version="0.0.1",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy"],
    python_requires=">=3.6",
    include_package_data=True,
    setup_requires=[],
)
