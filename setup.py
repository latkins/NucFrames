from distutils.core import setup
setup(name='NucFrames',
      version='0.1',
      package_dir={'NucFrames': 'src'},
      packages=['NucFrames', 'NucFrames.depth_utils', 'NucFrames.distance_utils']
      )
