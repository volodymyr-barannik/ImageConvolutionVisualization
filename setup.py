from setuptools import find_packages, setup

setup(name='ImageConvolutionVisualized',
      version='0.1.8',
      description='Allows to see how convolution works on images. You may define your own convolutions.',
      long_description=open('README.md', 'r').read(),
      long_description_content_type='text/markdown',
      author='Volodymyr Barannik',
      author_email='barannik.volodymyr@gmail.com',
      license='Apache 2.0',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=[
          'numpy',
          'torch>=1.1.0',
          'torchvision>=0.1.0',
          'matplotlib',
          'tensorboard',
      ],
      dependency_links=['https://download.pytorch.org/whl/cu113'])
