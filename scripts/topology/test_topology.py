"""Test functions in scripts with pytest"""

# Third-party libraries
import fourier_to_ryckaert_bellemans as ftrb
import numpy as np
import pytest
import ryckaert_bellemans_to_fourier as rbtf


def test_f_to_rb():
    """
    Test the conversion of Fourier dihedral coefficients to
    Ryckaert-Bellemans dihedral coefficients.
    """
    rb = ftrb.calc_rb_coeff([1.1, -2.2, 3.3, -4.4])
    assert len(rb) == 6
    assert np.allclose(rb, [0.0, 4.4, -15.4, -6.6, 17.6, 0.0])

    with pytest.raises(ValueError):
        ftrb.calc_rb_coeff([1.1, -2.2, 3.3, -4.4, 1.0])
    with pytest.raises(ValueError):
        ftrb.calc_rb_coeff([1.1, -2.2, 3.3])


def test_rb_to_f():
    """
    Test the conversion of Ryckaert-Bellemans dihedral coefficients to
    Fourier dihedral coefficients
    """
    f = rbtf.calc_f_coeff([0.0, 4.4, -15.4, -6.6, 17.6, 0.0])
    assert len(f) == 4
    assert np.allclose(f, [1.1, -2.2, 3.3, -4.4])

    with pytest.raises(ValueError):
        rbtf.calc_f_coeff([0.0, 4.4, -15.4, -6.6, 17.6, 0.0, 1.0])
    with pytest.raises(ValueError):
        rbtf.calc_f_coeff([0.0, 4.4, -15.4, -6.6, 17.6])
