from setuptools import setup, find_packages

setup(
    name='OptiFit',
    version='1.0',
    description='Combining methods from solcore and lmfit for an easy-to-use fitting procedure of thin film reflection contrast data and PL data',
    author='Aidan OBeirne',
    author_email='aidanobeirne@me.com',
    url='https://github.com/aidanobeirne/OptiFit.git',
    packages=['OptiFit'],
    include_package_data=True,
    scripts = ['OptiFit/examples/RC_fit_example.py', 'OptiFit/examples/CompositeModel_fit_example.py'],
    install_requires=['lmfit', 'solcore']
)
