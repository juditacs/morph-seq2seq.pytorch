from setuptools import setup

setup(name='morph_seq2seq',
      version='0.1',
      description='A simple seq2seq framework in PyTorch',
      url='https://github.com/juditacs/morph-seq2seq.pytorch',
      author='Judit Acs',
      author_email='judit@sch.bme.hu',
      license='MIT',
      packages=['morph_seq2seq'],
      install_requires=[
          'pyyaml',
          'numpy',
          'torch',
          'torchvision',
      ],
      zip_safe=False)
