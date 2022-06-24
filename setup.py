from setuptools import setup, find_packages

setup(
    name='TransferMatrix',
    version='1.0',
    description='Combining methods from solcore and Lmfit for an easy-to-use fitting procedure of thin film reflection contrast data',
    author='Aidan OBeirne',
    author_email='aidanobeirne@me.com',
    url='https://github.com/aidanobeirne/TransferMatrix.git',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['solcore', 'lmfit']
)
