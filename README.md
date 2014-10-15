image-cutoff
============

Python program for spatial Fourier transform and spatial spectrum analysis.
------------

The program requires the following Python modules to be installed: `numpy`, `scipy`, `matplotlib`, `Image`, `colorsys`, `os`.
For a demonstration from the commande line, simply run the following command and choose a demonstration example from the list that will be displayed:

    $ python cutoff.py

For inclusion as a Python module, make sure the file `cutoff.py` is accessible from the `PYTHONPATH`, an use it like in the following example:

    import cutoff
    im = cutoff.load_test_data("lena")
    sp = im.spectrum()
    sp.plot(LOG=True)

Documentation is included in the source code as docstrings.
