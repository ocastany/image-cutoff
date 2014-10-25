image-cutoff
============

Spatial Fourier transform and spatial spectrum analysis.
------------

This Python program requires the following modules to be installed: `numpy`, `scipy`, `matplotlib`, `Image`, `colorsys`, `os`.
Demonstration examples are available by running the following command line:

    $ python cutoff.py

For inclusion as a Python module, make sure the file `cutoff.py` is accessible from the `PYTHONPATH`, and use it like in the following example:

    import cutoff
    im = cutoff.load_test_data("lena")
    sp = im.spectrum()
    sp.plot(LOG=True)

Documentation is included in the source code as docstrings.

