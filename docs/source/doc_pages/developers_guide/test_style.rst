.. _test-style-label:

Test style
==========

.. warning::

    At the moment there are no unittests implemented.  However,
    unittests are very important to ensure that the code is working
    properly. Without unittests we cannot leave the alpha state of
    development.  We plan to implement extensive test suites using
    |pytest|.

Besides unittests, implement plausibility tests in your code.  For
instance, if you know that the return value of your function must be
between ``0`` and ``1`` (e.g. because it is a probability), check
whether this is indeed the case before returning the value.  If the test
is computationally expensive (either because it is very heavy on its own
or because it is or might be executed thousands of times in a loop),
wrap it in a debug statement so that the user can decide whether to use
this additional layer of safety or not.
