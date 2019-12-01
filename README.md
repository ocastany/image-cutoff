image-cutoff
============

Spatial Fourier transform and spatial spectrum analysis.
------------

This program is about the spatial spectrum of an image, direct and inverse spatial Fourier transform, contrast, sharpness, azimuthal average of the spectrum, etc.

It is written in Python an requires the following modules: `numpy`, `scipy`, `matplotlib`, `PIL`, `colorsys`, `os`.

For a demonstration, simply run:

    $ python3 cutoff.py

For inclusion as a Python module, make sure the file `cutoff.py` is accessible from the `PYTHONPATH`, and use it like in the following example:

    import cutoff
    im = cutoff.load_test_data("test image")
    sp = im.spectrum()
    sp.plot(LOG=True)

Documentation is included in the source code as docstrings.

The projet can be downloaded with the following command:
    
    $ git clone git://github.com/ocastany/image-cutoff.git

