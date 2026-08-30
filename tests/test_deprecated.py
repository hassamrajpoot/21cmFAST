"""
Tests for deprecated parameters and APIs.

This module consolidates all deprecation warning tests in one place.
Each deprecated parameter should have:
1. A test verifying the deprecation warning fires correctly.
2. A test decorated with @deprecation.fail_if_not_removed that will
   fail when the removed_in version is reached, reminding developers
   to clean up the deprecated code.

When a parameter is removed in v5, remove its tests from this module.
"""

import deprecation
import pytest
from astropy import units as un

from py21cmfast import lightconers as lcn
from py21cmfast.wrapper.inputs import (
    AstroOptions,
    InputParameters,
    MatterOptions,
)

# ── AstroParams ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fix_vcb_avg", [True, False])
@pytest.mark.filterwarnings(
    "ignore:^USE_MINI_HALOS is False but V_CB_MODEL:UserWarning"
)
def test_fix_vcb_avg_deprecated_warning(fix_vcb_avg):
    """Test that using FIX_VCB_AVG shows deprecation warning."""
    v_cb_model = "AVG-DEBUG" if fix_vcb_avg else "NONE"
    with pytest.warns(deprecation.DeprecatedWarning, match="FIX_VCB_AVG is deprecated"):
        inputs = InputParameters(
            random_seed=1,
            astro_options=AstroOptions(FIX_VCB_AVG=fix_vcb_avg),
            matter_options=MatterOptions(V_CB_MODEL=v_cb_model),
        )
    assert v_cb_model == inputs.matter_options.V_CB_MODEL
    assert fix_vcb_avg == inputs.astro_options.FIX_VCB_AVG


@deprecation.fail_if_not_removed
def test_fix_vcb_avg_is_removed():
    """Fails when removed_in version is reached, reminding you to delete FIX_VCB_AVG."""
    InputParameters(
        random_seed=1,
        astro_options=AstroOptions(FIX_VCB_AVG=True),
        matter_options=MatterOptions(V_CB_MODEL="AVG-DEBUG"),
    )


# ── Lightconers ───────────────────────────────────────────────────────────────


def test_with_equal_cdist_slices_deprecated_warning():
    """Test that with_equal_cdist_slices raises a deprecation warning."""
    with pytest.warns(
        deprecation.DeprecatedWarning, match="with_equal_cdist_slices is deprecated"
    ):
        lcn.RectilinearLightconer.with_equal_cdist_slices(
            min_redshift=6.0,
            max_redshift=7.0,
            resolution=2 * un.Mpc,
        )


@deprecation.fail_if_not_removed
def test_with_equal_cdist_slices_is_removed():
    """Fails when removed_in version is reached, reminding you to delete with_equal_cdist_slices."""
    lcn.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=6.0,
        max_redshift=7.0,
        resolution=2 * un.Mpc,
    )
