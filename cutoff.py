#!/usr/bin/python3
# encoding: utf-8

"""
What this program is about...
* Spatial spectrum of an image, direct and inverse spatial Fourier transform 
* RMS value, STD (standard deviation), contrast, sharpness, RMS gradient
* Azimuthal average of the spectrum, for studying radial profile

Note: only grey images are supported

The image and the spectrum are TwoDimensionalDataSet objects. 
* The horizontal and vertical axes are x and y. The coordinates are regularly 
  spaced by steps dx and dy.
* In the frequency domain, the spatial frequencies are called fx and fy.
* The data array is indexed as a[i,j], with 'i' for the x_i value and 'j' for
  the y_j value.

The spatial coordinates (for the x-axis) are 
    x_n = x0 + n*dx, n=0..N-1

The Fourier transform pairs (in the continuous) are
    A(f) = ∫ a(x) exp(-2iπfx) dx
    a(x) = ∫ A(f) exp(+2iπfx) df

In the discretized form, this reads
    A_k = \sum_n a_n exp(-2iπ f_k x_n) dx
    a_n = \sum_k A_k exp(+2iπ f_k x_n) df

where the coordinates of the discrete frequencies are
    f_k = k / (N dx) = [0, 1, ..., N-1] / (N dx)    if 'center' = 'high'
    f_k = [-(N//2), ..., (N-1)//2] / (N dx)         if 'center' = 'low'

Note: 'high' frequencies are around N//2 + q N, q \in \Z
      'low' frequencies are around 0 + q N, q \in \Z

We have df = 1 / (N dx), or dx = 1 / (N df).

Surfaces are S = Nx dx Ny dy and S' = 1 / (dx dy)

[More explanation should be added]
"""

from PIL import Image, ImageFilter
import numpy, numpy.fft, numpy.linalg
from numpy import pi, exp, log10, newaxis, sqrt, inf
import scipy.special, scipy.misc, scipy.signal
from scipy import special, misc, signal
import matplotlib.colors
import matplotlib.pyplot as pyplot
import colorsys
import os

#############################################################################
# Class for 2D data sets

class TwoDimensionalDataSet:
    """Two-dimensional data set."""

    data = None     # Array containing the 2D data
    shape = None    # Shape of the data (Nx, Ny)

    x = None        # Array for the x coordinate (column vector)
    x_ = None       # Flattened copy of x (ravel)
    Nx = None       # Size of x
    x0 = None       # First value of x
    dx = None       # Increment of x
    x1 = None       # Last value of x (x[-1])

    y = None        # Array for the y coordinate (row vector)
    y_ = None       # Flattened copy of y (ravel)
    Ny = None       # Size of y
    y0 = None       # First value of y
    dy = None       # Increment of y
    y1 = None       # Last value of y (y[-1])

    name = ""       # Name of the TwoDimensionalDataSet
    type = 'original'   # 'original' or 'spectrum', influence how the data 
                        # is plot.

    interpolation = 'nearest'   # Type of interpolation used in function
                                # imshow() to plot the data. See show().

    image = None        # Image produced by imshow()
    _LOG_plot = False   # Whether a log plot should be displayed (log10).

    def __init__(self, data, x=None, y=None, **kwargs):
        """Creates a TwoDimensionalDataSet object.
        
        'data' : Value of the data, array of shape (Nx, Ny).
        'x' : Array of Nx regularly spaced and increasing values. If none
              is provided, an array of integers starting from 0 is built.
        'y' : ...

        'kwargs' : Keyword arguments are passed to the configure() method.
                   (see the corresponding documentation for more details)

        Notes : 
        * Attribute 'data' may be modified directly if desired.
        * Attribute 'image' points to the image produced by imshow()
        """
        # Name and type of the data set
        self.configure(**kwargs)

        # Store the supplied data field...
        if data.dtype.kind in ['f', 'c']:   # We want to work with float type
            self.data = data                # in order to avoid overflow errors
        else:
            self.data = data.astype(float)
        self.shape = data.shape
        (self.Nx, self.Ny) = (Nx, Ny) = data.shape

        # Store the x and y axis coordinates
        if x is None:
            self.x = numpy.arange(Nx).reshape(Nx, 1)
        elif isinstance(x, numpy.ndarray):
            if numpy.size(x) != Nx:
                raise ValueError("'x' size does not match 'data' size.")
            self.x = x.reshape(-1, 1)
        else:
            raise TypeError("'x' should be an array.")

        if y is None:
            self.y = numpy.arange(Ny).reshape(1, Ny)
        elif isinstance(y, numpy.ndarray):
            self.y = y.reshape(1, -1)
            if numpy.size(y) != Ny:
                raise ValueError("'y' size does not match 'data' size.")
        else:
            raise TypeError("'y' should be an array.")

        self.x = self.x.astype(float)
        self.y = self.y.astype(float)

        # Store coordinate information
        (x_, y_) = (self.x.ravel(), self.y.ravel())
        (self.x_, self.y_) = (x_, y_)
        (self.x0, self.y0) = (x_[0], y_[0])
        (self.x1, self.y1) = (x_[-1], y_[-1])
        (dx,dy) = (x_[1]-x_[0], y_[1]-y_[0])
        self.dx = (self.x1 + dx - self.x0) / Nx     # Trick to limit rounding
        self.dy = (self.y1 + dy - self.y0) / Ny     # error.
        
    def configure(self, **kwargs): 
        """Configure the parameters given as keyworkds in 'kwargs'.

        'type' : 
            * 'original' => axes will be named 'x' and 'y'
            * 'spectrum' => axes will be named 'fx' and 'fy'

        'name' : Name of the data set

        'interpolation' : Type of interpolation in the plots, for example 
            'nearest'. See the pyplot.imshow() documentation.
        """
        if 'type' in kwargs:
            self.type = kwargs['type']
        if 'name' in kwargs:
            self.name = kwargs['name']
        if 'interpolation' in kwargs:
            self.interpolation = kwargs['interpolation']

    def get_config(self):
        """Returns a dictionary of configuration keywords."""
        d = {}
        d['type'] = self.type
        d['name'] = self.name
        d['interpolation'] = self.interpolation
        return d

    def copy(self):
        """Create and return a copy of this object."""
        copy = TwoDimensionalDataSet(self.data, self.x, self.y, 
                                     **self.get_config())
        return copy

    def _coordinates_to_indices(self, x=None, y=None):
        """Returns index tuple (i,j) from (x,y) coordinates."""
        if x is None:
            i = None
        else:
            i = int(round((x-self.x0) / self.dx))

        if y is None:
            j = None
        else:
            j = int(round((y-self.y0) / self.dy))

        return (i,j)

    def _indices_to_coordinates(self, i=None, j=None):
        """Return coordinates (x,y) from (i,j) indices."""
        if i is None:
            x = None
        else:
            x = self.x_[i]
        if j is None:
            y = None
        else:
            y = self.y_[j]
        return (x,y)


    ###############################################################"
    # Plotting

    # For the graphical representation we use pyplot.imshow(). By default this
    # function draws the data like a matrix, with the i index vertical and 
    # starting above, and the j index horizontal. For our purpose, in order to 
    # draw the data with the appropriate orientation, we transpose and change 
    # the origin to 'lower': ax.imshow(a.T, origin='lower')

    def plot(self, AXES=True, COLORBAR=True, LOG=None, 
             vmin=None, vmax=None, vmin_LOG=-15, Smax=0.3, 
             **kwargs):
        """Plot the two-dimensional data set.
       
        'AXES' : Boolean, display axes passing through (0,0).
        'COLORBAR' : Display a color bar scale.
        
        'LOG' : Boolean. If True, the magnitude 'r' of the data is changed to
                'r = log10(r)', while keeping the same complex argument.
                If value is None, attribute _LOG_plot is considered.
        
        'vmin', 'vmax' : Define the magnitude range for the full color range. 
            The colors for points with a magnitude outside the range will be
            truncated to the low or high limits. If not supplied, the min
            and max of the magnitude 'r' will be used.

        'vmin_LOG' : This parameter is considered for log plots and may be 
            either None or a real number. In the latter case, it is used for 
            'vmin' if 'vmin' is not defined. 
            This is useful for where some points have a very small magnitude
            (for example, a point with r = 0 leads to min(log10(r)) = -inf).

        'Smax' : Maximum saturation of the colors of the plot. For example, a
            value of Smax = 0 will give a gray plot (as if the complex argument
            was ignored).
        
        'kwargs' : keywords are passed to imshow(). 
        """
        # Prepare the decoration
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        if self.type == 'original':
            (x_text, y_text) = ("$x$", "$y$")
        elif self.type == 'spectrum':
            (x_text, y_text) = ("$f_x$", "$f_y$")
        ax.set_xlabel(x_text)
        ax.set_ylabel(y_text)
        title = self.name

        if AXES:
            ax.axhline(linestyle='--')
            ax.axvline(linestyle='--')

        if LOG is not None:
            self._LOG_plot = LOG         # Store choice for next time
        else:
            LOG = self._LOG_plot         # Use stored choice

        # Prepare the values 
        r = numpy.abs(self.data)        # Magnitude of the complex number
        arg = numpy.angle(self.data)    # Argument of the complex number

        if LOG is True:
            title += " (log10)"
            r = log10(r)    # array may contain '-inf' values
       
        if vmin is None:
            vmin = r.min()
            if LOG and vmin_LOG and vmin < vmin_LOG:
                vmin = vmin_LOG
        if vmax is None:
            vmax = r.max()

        RGB_map = self._colorize(r, arg, vmin, vmax, Smax)
            
        image = ax.imshow(RGB_map, 
                    origin='lower',
                    vmin=vmin, vmax=vmax,
                    cmap = pyplot.cm.Greys_r,
                    interpolation=self.interpolation,
                    extent=(self.x0-self.dx/2, self.x1+self.dx/2, 
                            self.y0-self.dy/2, self.y1+self.dy/2),
                    **kwargs)
        self.image = image

        if COLORBAR:
            # Colorbar for the 'image' mappable
            self.colorbar = fig.colorbar(image)
        
        # Final decoration of the plot
        ax.set_title(title)
        ax.format_coord = self._format_coord
        image.format_cursor_data = self._format_cursor_data
        fig.show()      # Finally display the image
        return image

## Note : The font used in the status bar can be changed. 
#  
#    label = fig.canvas.toolbar._message_label
#    label.configure(font="TkFixedFont 8")

    def update_plot(self, image=None,
                    vmin=None, vmax=None, vmin_LOG=-15, Smax=0.3):
        """Update a plot in place, with identical XY scale and decorations.

        'image' : Image object to update. If not given, attribute 'image' 
                  is used.
       
        Decorations of the plot remain the same: title, axis labels, etc.
        """
        # Prepare the values 
        r = numpy.abs(self.data)        # Magnitude of the complex number
        arg = numpy.angle(self.data)    # Argument of the complex number

        LOG = self._LOG_plot
        if LOG:
            r = log10(r)    # array may contain '-inf' values
       
        if vmin is None:
            vmin = r.min()
            if LOG and vmin_LOG and vmin < vmin_LOG:
                vmin = vmin_LOG
        if vmax is None:
            vmax = r.max()

        RGB_map = self._colorize(r, arg, vmin, vmax, Smax)
        
        if image is None:
            image = self.image

        image.set_data(RGB_map)
        image.set_clim(vmin=vmin, vmax=vmax)
        image.figure.show()
        # vérifier si c'est bien le moyen le plus rapide de
        # réafficher l'image

    def _colorize(self, r, arg, rmin=None, rmax=None, Smax=1.0):
        """Colorize a map according to magnitude 'r' and argument 'arg'.
        
        Typically used to represent complex numbers 'r exp(i arg)'.
        
        'r' : Magnitude array, real values, shape MxN
        'rmin', 'rmax' : Define clipping values for the magnitude range.
            By default, the range is determined from the min and max of 'r'.

        'arg' : Argument array, real value, same shape as 'r'
                The argument represents an angle in radians.

        Smax' : Maximum saturation of the colors in the plot.
                For example, Smax=0 will give a gray plot.

        Returns : RGB map, shape NxMx3
        """
        if rmax is None:
            rmax = r.max()
        else:
            r[r>rmax] = rmax

        if rmin is None:
            rmin = r.min()
        else:
            r[r<rmin] = rmin
    
        V = (r - rmin) / (rmax - rmin)  # V in range [0, 1]

        a = arg / (2*pi)        
        a = (a + 0.5) % 1 - 0.5 # a in range [-0.5, 0.5[
        S = Smax * 2 * abs(a)   # S in range [0, Smax], maximum for |arg| = pi
        H = a + 0.5             # H in range [0,1[
        # arg=0 -> a=0, H=0.5, cyan
        # arg=pi -> a=-0.5, H=0, red

        HSV = numpy.array((H,S,V))  # Shape 3xMxN
        HSV = HSV.swapaxes(0,2)     # Shape NxMx3
        RGB = matplotlib.colors.hsv_to_rgb(HSV)
        return RGB

    # For overriding "self.ax.format_coord()"
    def _format_coord(self, x, y):
        # String for x and y
        (x, y) = (float(x), float(y))
        xy_str = "(x, y) = ({:8.2e}, {:8.2e})".format(x,y)
        # String for the data value
        (i,j) = self._coordinates_to_indices(x, y)
        if 0 <= i < self.Nx and 0 <= j < self.Ny:
            z = self.data[i,j]
            z_str = "  -> {:9.1e} + {:9.1e} j".format(z.real, z.imag)
        else:
            z_str = ""
        return xy_str + z_str

    # For overriding "self.image.format_cursor_data()"
    def _format_cursor_data(self, data):
        return "" 


    ###############################################################
    # Transformations acting on a data set

    def power(self):
        """Calculates the power array of the data: P = |data|²
        
        Returns : TwoDimensionalDataSet(|data|², x, y)
        """
        return TwoDimensionalDataSet(
            numpy.abs(self.data)**2, self.x, self.y, 
            **dict(self.get_config(), name="Power(" + self.name + ")"))

    def sqrt(self):
        """Calculates the square root of the data: P = sqrt(data)
        
        Returns : TwoDimensionalDataSet(sqrt(data), x, y)
        """
        return TwoDimensionalDataSet(
            numpy.sqrt(self.data), self.x, self.y, 
            **dict(self.get_config(), name="Sqrt(" + self.name + ")"))

    def grad(self, direction):
        """Calculates the gradient along 'direction'.
        
        'direction' : 'x' or 'y'

        Returns : TwoDimensionalDataSet(∇a, x[:-1], y[:-1])
        """
        if direction == 'x':
            grad = numpy.diff(self.data, axis=0)[:,:-1] / self.dx
        elif direction == 'y':
            grad = numpy.diff(self.data, axis=1)[:-1,:] / self.dy
        else:
            raise ValueError("Choice is only 'x' or 'y'.")
        return TwoDimensionalDataSet(
            grad, self.x_[:-1], self.y_[:-1], 
            **dict(self.get_config(), 
                name="Grad_" + direction + "(" + self.name + ")"))
 
    def __neg__(self):
        """Returns the negative of the data set"""
        return TwoDimensionalDataSet(
            -self.data, self.x, self.y, 
            **dict(self.get_config(), 
                name="-(" + self.name + ")"))
    
    ###############################################################
    # Cropping

    def crop(self, indices=None, coordinates=None):
        """Extract the requested rectangle.

        Provide either 'indices' or 'coordinates':
        'indices' = (i1,j1,i2,j2) integers
        'coordinates' = (x1,y1,x2,y2)

        '1' is the bottom left corner, '2' is the top right corner.
        """
        if coordinates is not None:
            (i1, j1) = self._coordinates_to_indices(coordinates[:2])
            (i2, j2) = self._coordinates_to_indices(coordinates[2:])
        if indices is not None:
            (i1, j1, i2, j2) = indices
        data = self.data[i1:i2, j1:j2]
        (x, y) = (self.x_[i1:i2], self.y_[j1:j2])
        return TwoDimensionalDataSet(data, x, y, **self.get_config())

    def crop_center(self, i, j):
        """Extract a rectangle from the center.

        'i, j' : integers defining the size of the rectangle
        """
        (ic, jc) = (self.Nx//2, self.Ny//2)
        (i1, j1) = (ic - i//2, jc - j//2)
        (i2, j2) = (i1 + i, j1 + j)
        return self.crop(indices=(i1,j1,i2,j2))

    ###############################################################
    # Filtering

    def filter(self, mask):
        """Filters dataset filtered with a 'mask' array.
        
        'mask' : boolean array with same dimension as the dataset. Value is 
                 kept for True and nulled for False.
        Reminder : inverting a boolean array 'm' is simply done with '-m'.

        Returns : filtered dataset
        """
        if mask.shape != self.data.shape:
            raise ValueError("Incorrect shape of mask.")
        data = self.data * mask
        return TwoDimensionalDataSet(data, self.x, self.y, **self.get_config())
   
    def mask_empty(self):
        """Returns an empty mask array."""
        return numpy.zeros_like(self.data, dtype=bool)

    def mask_point(self, indices=None, coordinates=None, ref='center'):
        """Returns a mask array for a single point.
        
        Provide either 'indices' or 'coordinates':
        'indices' = (i,j) integers
        'coordinates' = (x,y)
        'ref' : 'center'/'origin', tells which reference is used with indices.
        """
        mask = numpy.zeros_like(self.data, dtype=bool)
        if coordinates is not None:
            (i,j) = self._coordinates_to_indices(*coordinates)
            mask[i,j] = True
        if indices is not None:
            (i,j) = indices
            if ref is 'center':
                mask[self.Nx//2 + i, self.Ny//2 + j] = True
            else:
                mask[i,j] = True
        return mask


    def mask_point_pair(self, i, j):
        """Returns a mask array for a pair of opposite points.
        
        'i, j' : Couple of integers. The points of the pair will be
                   (Nx//2 + i, Ny//2 + j) and (Nx//2 - i, Ny //2 - j)
        """
        mask = numpy.zeros_like(self.data, dtype=bool)
        mask[self.Nx//2 + i, self.Ny//2 + j] = True
        mask[self.Nx//2 - i, self.Ny//2 - j] = True
        return mask

    def mask_rectangle(self, indices=None, coordinates=None,
                       random=False, ff=0.5):
        """Returns a mask array for the specified rectangle.
       
        Provide either 'indices' or 'coordinates':
        'indices' = (i1,j1,i2,j2) integers
        'coordinates' = (x1,y1,x2,y2)

        '1' is the bottom left corner, '2' is the top right corner.

        'random' : If True, the rectangle will be filled randomly according
                   to the filling factor 'ff'.
        """
        mask = numpy.zeros_like(self.data, dtype=bool)
        if coordinates is not None:
            (i1, j1) = self._coordinates_to_indices(coordinates[:2])
            (i2, j2) = self._coordinates_to_indices(coordinates[2:])
        if indices is not None:
            (i1, j1, i2, j2) = indices
        if random:
            mask[i1:i2,j1:j2] = numpy.random.rand(i2-i1, j2-j1) < ff 
        else:
            mask[i1:i2,j1:j2] = True
        return mask

    def mask_rectangle_center(self, i, j, random=False, ff=0.5):
        """Returns a mask array for a centered rectangle.

        'i, j' : Couple of integers, size of the rectangle
        """
        (ic, jc) = (self.Nx//2, self.Ny//2)
        (i1, j1) = (ic - i//2, jc - j//2)
        (i2, j2) = (i1 + i, j1 + j)
        return self.mask_rectangle(indices=(i1,j1,i2,j2), 
                                   random=random, ff=ff)
       

    ###############################################################
    # Transformations of one data set (with another)

    def _coordinates_match(self, a):
        """Check if coordinates match.
        
        Check if the coordinates are the same as the coordinates
        of the TwoDimensionalDataSet 'a'.

        Returns : boolean
        """
        if numpy.any(a.x - self.x) or numpy.any(a.y - self.y):
            return False
        else:
            return True

    def __add__(self, a):
        """Add two data sets of same dimension"""
        if not self._coordinates_match(a):
            raise ValueError("Coordinates do not match")
        return TwoDimensionalDataSet(
            self.data + a.data, 
            self.x, self.y, 
            **dict(self.get_config(), name="Sum"))
 
    def __sub__(self, a):
        """Substract two data sets of same dimension"""
        if not self._coordinates_match(a):
            raise ValueError("Coordinates do not match")
        return TwoDimensionalDataSet(
            self.data - a.data, 
            self.x, self.y, 
            **dict(self.get_config(), name="Sub"))
    
    
    ###############################################################"
    # Extraction of values 

    def sum(self):
        """Calculates the integral of the data over the xy plane.

        Returns : ∫ a dx dy = data.sum() * dx * dy
        
        Example of use: 
        Parseval's theorem reads "a.power().sum() == A.power().sum()", 
        if 'A' is the Fourier transform of 'a'.
        """
        return self.data.sum() * self.dx * self.dy

    def surface(self):
        """Area of the surface supporting the data.
        
        Returns : S = ∫ dx dy = (Nx * dx) * (Ny * dy)
        """
        S = self.Nx * self.dx * self.Ny * self.dy
        return S


    ###############################################################"
    # Fourier calculations

    def spectrum(self, center='low'):
        """Computes the spatial spectrum of the 2D array data.
        
        Definition of the transformation:
            A(f) = ∫ a(x) exp(-2iπfx) dx
        (written for a single dimension, 'a' is the original) 
        The data is discretized on points x_n = x0 + n*dx, n=0..N-1
            A_k = \sum_n a_n exp(-2iπ f_k x_n) dx
        For f_k = k / (N dx), k=0..N-1, we have:
            A_k = exp(-2iπ f_k x0) * \sum_n a_n exp(-2iπ n k / N) * dx

        The discrete frequencies f_k are defined by k modulo N. In other
        words, frequencies f_k and f_{k-N} are equivalent. The highest 
        frequency of the range is f_{N//2}.
        
        The range k = [0, ..., N-1] has the high frequencies in the middle
        of the range. The range k = [-(N//2), ..., (N-1)//2] is obtained by
        a shift of -(N//2) and brings the low frequencies in the middle.

        'center' : 'low' or 'high'. Determines the order of the frequency
            coordinates so that 'low' or 'high' frequencies are in the middle 
            of the (fx, fy) vectors. More precisely, we have:
            f_k = [-(N//2), ..., (N-1)//2] / (N dx)    for 'center' = 'low'
            f_k = [0, 1, ..., N-1] / (N dx)            for 'center' = 'high'
   
        Return : TwoDimensionalDataSet(A, fx, fy), where 'A' is the spatial 
            spectrum and fx and fy are the associated spatial frequencies.
        """
        (Nx, Ny) = self.shape
        (x0, y0) = (self.x0, self.y0)
        (dx, dy) = (self.dx, self.dy)

        # Compute the discrete spatial frequency spectrum
        a = self.data
        A = numpy.fft.fft2(a)
        # Get the Fourier frequencies
        # (f_unit is x_unit^-1, for example pixel^-1 if x is a pixel number)
        if center == 'low':
            # Center the spectrum and the frequencies.
            (fx, fy) = (numpy.fft.fftfreq(Nx, dx), numpy.fft.fftfreq(Ny, dy))
            (fx, fy) = (numpy.fft.fftshift(fx), numpy.fft.fftshift(fy))
            A = numpy.fft.fftshift(A)
    
        if center == 'high':
            # High frenquencies are in the center of the map. No shift.
            (fx, fy) = (numpy.arange(Nx, dtype=float) / (Nx*dx),
                        numpy.arange(Ny, dtype=float) / (Ny*dy))
       
        # Shape the frequency vectors
        (fx, fy) = (fx[:,numpy.newaxis], fy[numpy.newaxis,:])
    
        # Take the coordinates into account
        A0 = exp(-2j*pi*(fx*x0+fy*y0)) * A * dx * dy
   
        result = TwoDimensionalDataSet(A0, fx, fy, **self.get_config())
        result.configure(type='spectrum')
        return result

    def Fourier_coeff(self, indices=None, coordinates=None):
        """Returns Fourier coefficient for frequency (fx,fy).
       
        Provide either 'indices' or 'coordinates':
        'indices' = (i,j) integers, interpreted as f_i = i / (Nx dx) 
        'coordinates' = (fx,fy) frequencies

        The formulation assumes that this object is the original image.
        
        The result is one point of the array returned by the spectrum() method,
        but is calculated differently. 

        Returns : A_k = \sum_{n=0}^{N-1} a_n exp(-2iπ f_k x_n) dx
        """
        (x, y) = (self.x, self.y)
        (dx, dy) = (self.dx, self.dy)

        if indices is not None:
            (i,j) = indices
            fx = i / float(self.Nx * dx)
            fy = j / float(self.Ny * dy)
        if coordinates is not None:
            (fx,fy) = coordinates

        A_fx_fy = numpy.sum(self.data * exp(-2j*pi*(fx*x+fy*y)) * dx * dy)
        return A_fx_fy
  

    def spectrum_inverse(self, x0=0.0, y0=0.0):
        """Computes the original image from the spatial spectrum.
      
        Definition of the inverse transformation:
            a(x) = ∫ A(f) exp(+2iπfx) df
        (written for a single dimension, 'A' is the spatial spectrum)
        The data is discretized on points f_k = k * df
            a_n = \sum_k A_k exp(+2iπ f_k x_n) df
        For x_n = x0 + n / (N df), n=0..N-1, we have:
            a_n = \sum_k A_k exp(+2iπ f_k x0) exp(+2iπ k n / N) df

        (x0, y0) : Optional value for the start of the original image.

        Return : TwoDimensionalDataSet(a, x, y), where 'a' is the original 
            spectrum and x and y are the positions.
        """
        (Nx, Ny) = self.shape
        (fx, fy) = (self.x, self.y)
        (fx_, fy_) = (self.x_, self.y_)
        (df_x, df_y) = (self.dx, self.dy)

        # Correct for the origin point
        A = self.data
        A0 = A * exp(+2j*pi*(fx*x0+fy*y0))

        # Reorder the frequencies if low frequencies are in the center
        if fx_[Nx//2] == 0.0 and fy_[Ny//2] == 0.0:
            A0 = numpy.fft.ifftshift(A0)
            (fx, fy) = (numpy.fft.ifftshift(fx), numpy.fft.ifftshift(fy))

        # Invert the spectrum
        a = numpy.fft.ifft2(A0) * Nx * df_x * Ny * df_y

        # Create the position coordinates
        (x, y) = (x0 + numpy.arange(Nx, dtype=float) / (Nx*df_x),
                  y0 + numpy.arange(Ny, dtype=float) / (Ny*df_y))
    
        # Shape the coordinate vectors
        (x, y) = (x[:,numpy.newaxis], y[numpy.newaxis,:])
      
        result = TwoDimensionalDataSet(a, x, y, **self.get_config())
        result.configure(type='original')
        return result

    def Fourier_inverse_coeff(self, indices=None, coordinates=None):
        """Returns the inverse Fourier coefficient for position (x,y).

        Provide either 'indices' or 'coordinates':
        'indices' = (i,j) integers, interpreted as x_i = i dx = i / (N df)
        'coordinates' = (x,y)

        The formulation assumes that this object 'A' is the spatial frequency 
        spectrum.
        
        Returns : a_n = \sum_{k=0}^{N-1} A_k exp(+2 i pi n k / N) df
                  a   = \sum A_k exp(+2 i pi x f_k) df
        (formula written for 1D instead of 2D)
        """
        (Nx, Ny) = self.shape
        (fx, fy) = (self.x, self.y)
        (df_x, df_y) = (self.dx, self.dy)
        A = self.data

        if indices is not None:
            (i,j) = indices
            x = i / float(self.Nx * df_x)
            y = j / float(self.Ny * df_y)
        if coordinates is not None:
            (x,y) = coordinates

        a_x_y = numpy.sum(A * exp(+2j*pi*(fx*x+fy*y)) * df_x * df_y)
        return a_x_y


    ###############################################################"
    # Statistics

    def RMS_value(self):
        """Calculates the RMS value of the dataset.
        
        Returns : sqrt{<|a|²>} = sqrt{(∫ |a|² dx dy) / (∫ dx dy)}
        Examples :
            For a chessboard +D/-D : RMS_value = |D|
            For a chessboard  D/0:   RMS_value = |D|/sqrt(2)
        Note : 
            RMS_value = sqrt(a.power().sum() / a.surface())
        Parseval theorm implies :
            RMS_value(A) = RMS_value(a) * sqrt(Nx Ny) * dx dy
        """
        a = self.data
        a_RMS =  numpy.linalg.norm(a, 'fro') / sqrt(self.Nx*self.Ny)
        return a_RMS
    
    def STD_value(self):
        """Calculates the standard deviation of the dataset.

        Returns : sqrt{<|a - <a>|²>} = sqrt{(∫ |a - <a>|² dx dy) / (∫ dx dy)}
        """
        a = self.data
        a_STD = numpy.linalg.norm(a - a.mean(), 'fro') / sqrt(self.Nx*self.Ny)
        return a_STD

    def RMS_gradient_value(self):
        """Calculates the RMS value of the spatial gradient.
        
        Returns : sqrt{<|∇a|²>} = sqrt{(∫ |∇a|² dx dy) / (∫ dx dy)}
        The unit of the returned value is 'data_unit / coordinate_unit'.
        Example :
            For a chessboard +D/-D, returns sqrt(2)*2*D
            For a chessboard D/0, returns sqrt(2)*D
        Note : This is equal to 
          sqrt((a.grad('x').power() + a.grad('y').power()).sum() / a.surface())
        """
        # Should be identical to
        # numpy.hypot(a.grad('x').RMS_value(), a.grad('y').RMS_value())
        (Nx, Ny) = (self.Nx, self.Ny)
        (dx, dy) = (self.dx, self.dy)
        grad_x = numpy.diff(self.data, axis=0) / dx
        grad_y = numpy.diff(self.data, axis=1) / dy
        grad_x_RMS = numpy.linalg.norm(grad_x, 'fro') / sqrt((Nx-1) * Ny)
        grad_y_RMS = numpy.linalg.norm(grad_y, 'fro') / sqrt(Nx * (Ny-1))
        grad_RMS = numpy.hypot(grad_x_RMS, grad_y_RMS)
        return grad_RMS

    def contrast(self):
        """Calculates the average contrast of the data.
       
        The contrast calculated here is based on the variance of the data. 
        If should give a value near 1 (maybe higher) for a highly contrasted 
        image.
       
        Returns : sqrt{<|a - <a>|²>} / |<a>|
        The returned value has no unit. 
        """
        a = self.data
        a_avg = a.mean()
        a_STD = numpy.linalg.norm(a - a_avg, 'fro') / sqrt(self.Nx*self.Ny)
        contrast = a_STD / abs(a_avg)
        return contrast

    def sharpness(self, ref='std'):
        """Sharpness of the data.
        
        This sharpness function is calculated from the RMS gradient.
        Keywords: 'sharpness', 'resolution', 'acutance'. 
        
        A sharpness value of 1 is returned for a chessboard of type 'ref',
        where possible values for 'ref' are:
            'D/0'  : chessboard with values D and 0
            '+D/-D': chessboard with values +D and -D
            'std'  : the standard deviation will be used as a reference 
                     (returns a contrast of 1 for both chessboard types)

        Returns : sharpness value = RMS_gradient_value() / ref_value
        The unit of the returned value is '1/coordinate_unit'

        ref_value is linked to the RMS of STD value of the image.

        Example : For pixels with independant random value, the result is 0.5
        """
        if ref == 'D/0':
            a_ref = 2 * self.RMS_value()
        elif ref == '+D/-D':
            a_ref = sqrt(2) * 2 * self.RMS_value()
        elif ref == 'std':
            a_ref = sqrt(2) * 2 * self.STD_value()
        else:
            raise ValueError("Argument not understood")

        return self.RMS_gradient_value() / a_ref

    def match_reference(self, indices=None, coordinates=None, ref='chess D/0'):
        """Calculates the overlap ratio between the data and a reference 'ref'.
        
        Provide either 'indices' or 'coordinates':
        'indices' = (i,j) integers, such that f_i = i / (Nx dx)
        'coordinates' = (fx,fy) frequencies

        If neither parameter is provided, we take fx = 1/(2 dx)

        'ref' : The ratio is 1 if the data corresponds to the specified 
            reference:
            * 'chess +D/-D' : chessboard with alternating + and - values
            * 'chess D/0' : chessboard aternating between a value and 0
            * 'sine +D/-D' : alternating sine, with 0 average
            * 'sine D/0' : sine between a value and 0
        """
        (dx, dy) = (self.dx, self.dy)

        if indices is not None:
            (i,j) = indices
            fx = i / float(self.Nx * dx)
            fy = j / float(self.Ny * dy)
        elif coordinates is not None:
            (fx,fy) = coordinates
        else:
            (fx,fy) = (1./(2*dx), 1./(2*dy))

        if ref == 'chess +D/-D':
            a_ref = self.RMS_value()
        elif ref == 'chess D/0':
            a_ref = self.RMS_value() / sqrt(2)
        elif ref == 'sine +D/-D':
            a_ref = self.RMS_value() / sqrt(2)
        elif ref == 'sine D/0':
            a_ref = self.RMS_value() / sqrt(6)
        else:
            raise ValueError("Reference map does not exist: " + ref)

        A_fx_fy_avg = self.Fourier_coeff(coordinates=(fx,fy)) / self.surface()

        return numpy.abs(A_fx_fy_avg) / a_ref


    ###############################################################"
    # Azimuthal average

    def distance(self, origin='middle', *args):
        """Array of distances from the specified origin to every data point.
        
        'origin' : point from which the distance is calculated 
            'zero'   => distance from (x=0, y=0)
            'middle'   => distance from middle of the image (x[Nx//2], y[Ny//2])
            'indices'    => distance from position (x[args[0]], y[args[1]])
            'coordinates'  => distance from position (x,y) = (args[0], args[1])

        Returns : array of the same shape as self.data

        Note : this is really the 2D distance, no modulo is used.
        """
        if origin == 'zero':
            (x_origin, y_origin) = (0.0, 0.0)
        elif origin == 'middle':
            (x_origin, y_origin) = (self.x_[self.Nx//2], self.y_[self.Ny//2])
        elif origin == 'indices':
            (x_origin, y_origin) = (self.x_[args[0]], self.y_[args[1]])
        elif origin == 'coordinates':
            (x_origin, y_origin) = (args[0], args[1])
        else:
            raise ValueError("Keyword not understood.")
        return numpy.hypot(self.x - x_origin, self.y - y_origin)


    def azimuthal_average(self, r='zero', r_bins='circumscribed'):
        """Azimuthal average of the data set according to distance 'r'.
 
        'r' : Distance used for the average. The following may be specified:
            * A keyword 'middle' or 'zero' indicating the origin taken for 
              the calculation of the distance. Will be passed to the distance()
              function.
            * A tuple of parameters that will be sent to the distance() function
              to obtain the distance of every point.
              Example : r = ('coordinates', 300, 300)
            * A one-dimensional array of the same shape as the data, for
              example the output of the distance() function.
        
        'r_bins' : Definition of the 'r' bins. The following may be specified: 
            * A one-dimensional array of increasing values defining bins for 
              the 'r' value in which the data will be averaged.
            * A float 'dr' implies that the range of 'r' will be divided into
              bins of equal length 'dr'.
            * An integer M implies that the range of 'r' will be divided into
              M bins of equal length.
            * A keyword 'circumscribed' implies that the bins will be searched
              based on the coordinates so that the distance never exceeds 
              existing coordinates.
    
        Returns : RadialProfile(r_avg, D_avg), where r_avg = r_bins[:-1] and 
            'D_avg' has the same length. D_avg[i] is the averaged value of 
            the data for all points for which 'r' in the interval 
            [ r_bins[i], r_bins[i+1] ) (the bound is included for the last bin).
        """
        ####
        # Set up the distance...
        if isinstance(r, numpy.ndarray):
            pass    # nothing to do if the distance is given as parameter
        elif isinstance(r, tuple):
            r = self.distance(*r)   # sent to distance() method
        elif isinstance(r, str):
            r = self.distance(r)    # sent to distance() method
        else:
            raise("Argument 'r' was not understood.")

        ####
        # Set up the bins...
        
        # When the bins are not specified explicitly, find the index of the
        # min distance, and the min/max distance.
        if not isinstance(r_bins, numpy.ndarray):
            idx_flat = r.argmin()
            (argmin_x, argmin_y) = numpy.unravel_index(idx_flat, r.shape)
            r_min = r[argmin_x, argmin_y]
            r_max = r.max()
        
        # Create the bins as requested
        if isinstance(r_bins, numpy.ndarray):
            pass    # Will use 'r_bins' as it is provided
        elif r_bins == 'circumscribed':
            if (argmin_x == 0 or argmin_y == 0 or argmin_x == self.Nx-1 or 
                argmin_y == self.Ny-1):
                print("Center out of range. Using full range.")
                r_bins = numpy.arange(r_min, r_max + self.dx, self.dx)
            else:
                (r_left, r_right) = (numpy.min(r[0,:]), numpy.min(r[-1,:]))
                (r_bottom, r_top) = (numpy.min(r[:,0]), numpy.min(r[:,-1]))
                r_circ = min(r_left, r_right, r_bottom, r_top)
                if (r_circ < self.dx + self.dy):
                    print("Center too close to the side. Using full range.")
                    r_bins = numpy.arange(r_min, r_max + self.dx, self.dx)
                else:
                    dxy = min(self.dx, self.dy)
                    r_bins = numpy.arange(r_min, r_circ + dxy, dxy)
        elif isinstance(r_bins, float):
            step = r_bins
            r_bins = numpy.arange(r_min, r_max + step, step)
        elif isinstance(r_bins, int):
            step = (r_max - r_min) / r_bins
            r_bins = numpy.arange(r_min, r_max + step, step)
        else:
            raise ValueError("Argument 'r_bins' not understood.")

        ####
        # Check validity of the bins...
        if numpy.diff(r_bins < 0).any():
            raise AttributeError("Bins must increase monotonically.")
        # Flat view of the arrays:
        (r, data) = (r.ravel(), self.data.ravel())

        ####
        # Compute the average...
        data_sum = numpy.histogram(r, r_bins, weights=data)[0]
        r_sum = numpy.histogram(r, r_bins, weights=r)[0]
        r_nb = numpy.histogram(r, r_bins)[0]

        # It is possible to write a quicker calculation that avoids multiple 
        # calls to histogram(). See file my 'histogram.py'. However, the code
        # is longer and not used here.

        data_avg = data_sum / r_nb
        r_avg = r_sum / r_nb
        dS = r_nb * self.dx * self.dy

        if self.type == 'original':
            xlabel = "Distance $r$"
        elif self.type == 'spectrum':
            xlabel = "Distance $f$"
        profile = RadialProfile(r_avg, data_avg, xlabel=xlabel, name=self.name)
        profile.dS = dS
        return profile

    ###############################################################
    # Utilities

    def frequency_plot(self, center='high', **kwarg):
        """Plot the azimuthal RMS value of the spectrum of an image.
        
        Since a spectrum lives on a rectangular lattice, a radial average is 
        only meaningful when it is circumscribed to a lattice cell.

        'center' = 'high' or 'low', lattice center for the azimuthal average
        'kwargs' are passed to the RadialProfile.plot() method

        Calls method self.spectrum(center).power().azimuthal_average('middle')
        """
        power_avg = self.spectrum(center).power().azimuthal_average('middle')
        power_avg.xlabel += " (from " + center + " frequencies)"
        RMS = power_avg.sqrt()
        RMS.plot(**kwarg)
        return RMS


    ###############################################################"
    # Junk

    def correlate(self, image, max=10):
        """Correlation of this TwoDimensionalDataSet with another 'image'.
        
        NOT FINISHED
        'max' : Maximum distance in X and Y for which the correlation is
                calculated. A margin of length 'max' is removed around 
                the given 'image'.
        """
        # C'est super lent de faire une corrélation... il faudrait réduire
        # la taille pour approcher la bonne valeur. Ajouter par exemple un 
        # paramètre de réduction r=4.
        if isinstance(image, TwoDimensionalDataSet):
            image = image.data
        # Pas terminé
        image = image[max:-max, max:-max]
        corr = scipy.signal.correlate2d(self.data, image, mode='valid')
        px_range = numpy.arange(-max,max)
        return TwoDimensionalDataSet(corr, x=px_range, y=px_range)

    def PIL_smooth(self):
        """Smoothes an image (uses Python Imaging Library)
        
        Uses ImageFilter.SMOOTH_MORE from the Python Imaging Library (PIL).
        """
        # Rudimentaire mais peut être utile
        data = 255 * numpy.abs(self.data) / self.data.max()
        im = Image.fromarray(data)
        im = im.convert('L')
        im = im.filter(ImageFilter.SMOOTH_MORE)
        a = numpy.asarray(im)
        return TwoDimensionalDataSet(a, self.x, self.y, **self.get_config())


#############################################################################
# Class for radial profiles

class RadialProfile:
    """Class returned by TwoDimensionalDataSet.azimuthal_average()

    Describes a radial profile g(r).
    """

    name = ""       # Name for that data
    r = None        # Radius (one-dimensional array)
    dS = None       # Mesure of the surface to integrate
    g = None        # Data (same shape as r)
    xlabel = "$r"   # Labels for the plot
    ylabel = "Azimuthal average" 
    ax = None       # Axis where the plot was drawn

    def __init__(self, r, g, **kwargs):
        """Creates a radial profile g(r)

        'r' : Radius (one-dimensional array)
        'g' : Data (same shape as r)

        'kwargs' : are passed to the configure() method
        """
        (self.g, self.r) = (g, r)
        self.configure(**kwargs)
    
    def configure(self, **kwargs): 
        """Configure the parameters given as keyworkds in 'kwargs'.

        'name' : Name of the radial plot

        'xlabel'/'ylabel' : Label for the plot

        """
        if 'name' in kwargs:
            self.name = kwargs['name']
        if 'xlabel' in kwargs:
            self.xlabel = kwargs['xlabel']
        if 'ylabel' in kwargs:
            self.ylabel = kwargs['ylabel']

    def get_config(self):
        """Returns a dictionary of configuration keywords."""
        d = {}
        d['name'] = self.name
        d['xlabel'] = self.xlabel
        d['ylabel'] = self.ylabel
        return d

    def sum(self, use_measure=True):
        """Radial integral: \int 2 pi r g(r) dr."""
        (r, g, dS) = (self.r, self.g, self.dS)
        if self.dS is None or use_measure is False:
            return numpy.trapz(2*pi*r*g, r)
        else:
            return (g * dS).sum()

    def sqrt(self):
        """Returns a radial profile with sqrt(data)"""
        return RadialProfile(
            self.r,
            numpy.sqrt(self.g),
            **dict(self.get_config(), name="Sqrt(" + self.name + ")"))
 
    def square(self):
        """Returns a radial profile with square(data)"""
        return RadialProfile(
            self.r,
            self.g**2,
            **dict(self.get_config(), name="(" + self.name + ")^2"))
 
    def plot(self, LOGY=True, *args, **kwargs):
        """Plot radial profile"""
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        self.ax = ax
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.name)
        if 'marker' not in kwargs:
            kwargs['marker'] = '.'
        if LOGY:
            ax.semilogy(self.r, self.g, *args, **kwargs)
        else:
            ax.plot(self.r, self.g, *args, **kwargs)
        fig.show()

        
#############################################################################
#############################################################################
# Examples

def load_test_data(TYPE='gaussian', PARAM=None, OUTPUT_FILE_NAME=None):
    """Test image
    
    'TYPE' : 'gaussian', 'circular wave', 'plane wave', 'test image', 
             'chessboard', 'y-lines'
    'PARAM' : Dimension for that type (see details in the source code)
    'OUTPUT_FILE_NAME' : Filename for saving the image
    'PLOT' : If a plot should be displayed
    
    Return : tuple (x,y,a), where 'a' is the data test
    """
    (Nx, Ny) = (640, 480)               # defaults for most of the examples
    (x_center, y_center) = (300, 300)
    (x,y) = numpy.ogrid[0:Nx, 0:Ny]
    r = numpy.hypot(x - x_center, y - y_center)
    if TYPE == 'gaussian':
        # Normalized gaussian (integral of the surface equal to 1):
        # gauss.sum() = 1
        # Spectrum is A(f) = exp(-1/2 (2 pi r_gaussian)² (fx²+fy²)) * 
        # exp(-2j pi (fx xc + fy yc))
        if PARAM:
            r_gaussian = PARAM
        else:
            r_gaussian = 10
        a = 1./(2*pi*r_gaussian**2)*exp(-1./2*(r/r_gaussian)**2)
    elif TYPE == 'circular wave':
        # Bessel function of order 0
        if PARAM:
            f = PARAM
        else:
            f = 0.1
        a = scipy.special.jn(0, 2*pi*f*r)
    elif TYPE == 'plane wave':
        # Plane cosine
        if PARAM:
            (fx, fy) = PARAM
        else:
            (fx, fy) = (0.08, 0.04)
        a = numpy.cos(2*pi*(fx*x+fy*y))
    elif TYPE == 'test image':
        # Test image from SciPy
        a = scipy.misc.face().sum(-1)/3   # convert to greyscale
        a = a[::-1]
        a = a.T
        (Nx, Ny) = a.shape
        (x, y) = (numpy.arange(Nx), numpy.arange(Ny))
    elif TYPE == 'chessboard':
        # Chessboard
        if PARAM is not None:
            (Nx, Ny) = PARAM
        else:
            (Nx, Ny) = (21, 21)
        (x,y) = numpy.ogrid[0:Nx, 0:Ny]
        a = 10 * (-1)**((x+y)%2)
    elif TYPE == 'y-lines':
        # Lines parallel to the y axis
        if PARAM is not None:
            (Nx, Ny) = PARAM
        else:
            (Nx, Ny) = (21, 21)
        (x,y) = numpy.ogrid[0:Nx, 0:Ny]
        a = 10 * (-1)**(x%2) + numpy.zeros_like(y)
    else:
        raise ValueError("TYPE of test data not recognized.")

    if OUTPUT_FILE_NAME is not None:
        a_image = 255 * a.T / f.max()
        i = Image.fromarray(a_image).convert('L')
        i.save(OUTPUT_FILE_NAME)

    return TwoDimensionalDataSet(a,x,y, name=TYPE)

   

def load_image(filename, mode='gray', PLOT=False):
    """Load an image and return the corresponding data (a,x,y).
    
    'mode' : 'gray' converts the image to 8 bit gray scale
             'R', 'G' or 'B' take that component from the image
    """
    # Get the source image
    im = Image.open(filename)
    if PLOT:
        im.show()

    im = im.transpose(Image.ROTATE_270)
    bands = im.getbands()
    if mode == 'gray':
        im = im.convert('L')    # 8-bit gray scale
    if mode in ['R', 'G', 'B']:
        idx = bands.index(mode)
        im = im.split()[idx]

    a = numpy.asarray(im)
    (Nx, Ny) = a.shape
    (x,y) = numpy.ogrid[0:Nx, 0:Ny]
    return TwoDimensionalDataSet(a,x,y, name=os.path.basename(filename))


##############################################################################
# The following is run when the file is not imported as a module.
# It shows different functionalities of the program.


if __name__ == "__main__":
    print("""Demonstration examples:
    1) Test image: spectrum and frequency plot
    2) Gaussian: numerical validation
    3) Spectral filtering examples
    q) Quit """)
    choice = input("Choice: ")
    
    if choice == '1':
        image = load_test_data('test image')
        image.plot()
        sp = image.spectrum()
        # Spectrum plot
        sp.plot(LOG=True)

        # Contour calculation example :
        # (image.grad('x').power() + image.grad('y').power()).sqrt().plot()

        # Azimuthal average
        # image.frequency_plot('high')
        fp = image.frequency_plot('low')

    elif choice == '2':
        r_g = 15
        gauss = load_test_data(TYPE='gaussian', PARAM=r_g)
        gauss.plot()
        sp = gauss.spectrum()
        (x_c, y_c) = (300, 300) # Center of the gaussian image
        
        # Check the Fourier transform analytically
        print("\nThree FT calculations should give the same result:")
        (fx_c, fy_c) = (320, 240)   # Index of the centrer of the spectrum
        (i, j) = (310, 235)         # Index near the center of the spectrum
        (fx, fy) = (sp.x_[i], sp.y_[j])
        c1 = sp.data[i,j]
        print("sp.data[i,j] =               " + str(c1))
        c2 = gauss.Fourier_coeff(coordinates=(fx,fy))
        print("gauss.Fourier_coeff(fx,fy) = " + str(c2))
        c3 = exp(-1./2*(2*pi*r_g)**2 * (fx**2+fy**2)) * \
             exp(-2j*pi*(fx*x_c + fy*y_c))
        print("Analytically :               " + str(c3))

        # Plot the RMS analytically
        print("\nFrequency plot: " +
              "calculated and analytic values are superimposed.")
        az_avg = gauss.frequency_plot('low', LOGY=False, linestyle="")
        f = az_avg.r
        az_avg.ax.plot(f, exp(-1./2*(2*pi*r_g)**2*f**2), 'g')
        az_avg.ax.figure.show()

        # Check the RMS value of the spectrum
        print("\nFour ways to calculate the spectrum RMS average:")
        c1 = 1./(2*sqrt(pi)*r_g)
        print("* Analytically                               : " + str(c1))
        c2 = gauss.RMS_value()*sqrt(gauss.Nx * gauss.Ny)
        print("* with gauss.RMS_value()                     : " + str(c2))
        c3 = sp.RMS_value()
        print("* with sp.RMS_value()                        : " + str(c3))
        profile = sp.power().azimuthal_average('middle')
        c4 = sqrt(profile.sum() / sp.surface())
        print("* with sp.power().azimuthal_average('middle'): " + str(c4))

    elif choice == '3':
        im = load_test_data('test image')
        image = im.plot(vmin=0, vmax=255, COLORBAR=False)
        sp = im.spectrum()
        text = image.axes.text(0,0,"")

        # List of sharpness - filling couples:
        sf = [(1, 30), (3, 30), (5, 30),
              (5, 50), (5, 70), (5, 80), (5, 90), (5, 100),
              (7, 100), (10, 100), (20, 100), (40, 100), (100, 100),
              (100, 90), (100, 80), (100, 60), (100, 40), (100, 30), 
              (100, 20), (100, 10), (100, 4), (100, 1)]
        
        for (n, (sharpness, filling)) in enumerate(sf):
            input("Press a key for next image (sharpness " + 
                  "{:3d}%, filling {:3d}%)".format(sharpness, filling))
            (i, j) = (im.Nx*sharpness/100, im.Ny*sharpness/100)
            ma = sp.mask_rectangle_center(i, j, random=True, ff=filling/100.)
            im_filtered = sp.filter(ma).spectrum_inverse()
            im_filtered.update_plot(image, vmin=0, Smax=1.0)
            text.set_text("Sharpness {:3d}%, ".format(sharpness) + 
                          "filling {:3d}%".format(filling))
            # image.figure.savefig("im_filtered_{:02d}".format(n))
            # $ convert -delay 50 -loop 0 im_filtered_*.png animated.gif
            # $ animate animated.gif    or   $ firefox animated.gif
            
    input("\nEnd of the demonstration.")

