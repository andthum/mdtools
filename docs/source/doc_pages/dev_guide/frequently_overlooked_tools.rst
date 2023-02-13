.. _frequently-overlooked-tools-label:

Frequently overlooked tools
===========================

This page serves as a collection of functions, classes and other objects
from the Python modules MDTools is built on and that we believe are
quite useful for the project but often overlooked or not valued
appropriately.  The idea behind this collection is that if you are
facing a problem and do not know how to solve it or think there must be
a better solution, you can take a look at this page and might find a
function, class or whatever that you have not thought of or even did not
know and that solves (or at least helps you solving) your problem.

Feel free to add functions, classes and other objects that you believe
are quite useful for the project but not well-known.  When adding
objects, maintain alphabetical order and write a few sentences why you
think the object is useful for MDTools developers.  A short example
showing the power of the promoted object is welcome.

.. contents:: Site contents
    :depth: 2
    :local:


Numpy
-----

:meth:`numpy.ufunc.at`

    Perform fast inplace operations on
    :class:`arrays <numpy.ndarray>` at specified indices.

    Example

    >>> a = np.zeros(4)
    >>> ix = np.array([0, 0, 2])
    >>> np.add.at(a, ix, 1)
    >>> a
    array([2., 0., 1., 0.])

    Contrarily:

    >>> b = np.zeros(4)
    >>> b[ix] += 1
    >>> b
    array([1., 0., 1., 0.])

:meth:`numpy.ufunc.reduce`

    Apply the `universal function`_ :class:`~numpy.ufunc` along one axis
    of an :class:`array <numpy.ndarray>`.

:meth:`numpy.ufunc.reduceat`

    Perform a local :meth:`~numpy.ufunc.reduce` at specified slices.
    See :func:`mdtools.structure.contact_matrix` for an example.

.. _universal function: https://numpy.org/doc/stable/reference/ufuncs.html
