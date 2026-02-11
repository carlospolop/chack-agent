from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='chack-agent',
    version='0.2.0',
    description='Full Chack agent runtime + tools',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Carlos Polop',
    packages=find_packages(include=['chack_agent', 'chack_agent.*', 'chack_tools', 'chack_tools.*']),
    include_package_data=True,
    package_data={
        'chack_agent': ['config/*.yaml'],
    },
    python_requires='>=3.9',
    install_requires=[
        'boto3>=1.28.0',
        'requests>=2.31.0',
        'PyYAML>=6.0.1',
        'pypdf>=4.0.0',
        'openai-agents>=0.7.0',
    ],
    extras_require={
        'openai_agents': ['openai-agents>=0.7.0'],
    },
)
