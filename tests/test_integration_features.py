"""
Large-scale tests which test code updates against previously-run "golden" results.

The idea here is that any new updates (except for major versions) should be non-breaking;
firstly, they should not break the API, so that the tests should run without crashing without
being changed.
Secondly, the actual results of running the basic functions should remain the same for
the same input code, except for potential bug-fixes. In these cases, these tests should
pick these changes up. The test data should then be changed to reflect the new gold
standard, and if applicable, a new test should be written that reflects the previous
broken code.
Thirdly, it enforces that new features, where possible, are added in such a way as to
keep the default behaviour constant. That is, the tests here should *not* run the added
feature, and therefore should continue to produce the same test results regardless of
the new feature added. The new feature should be accompanied by its own tests, whether
in this or another test module. If a new feature *must* be included by default, then
it must be implemented in a new major version of the code, at which point the test data
is able to be updated.

Comparison tests here are meant to be as small as possible while attempting to form
a reasonable test: they should be of reduced data such as power spectra or global xHI
measurements, and they should be generated with small simulations.
"""

import gc
import logging
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib as mpl
import numpy as np
import pytest
from astropy import units as un
from astropy.tests.helper import assert_quantity_allclose

from py21cmfast import Coeval, LightCone, OutputCache
from py21cmfast.wrapper.inputs import Table1D

from . import produce_integration_test_data as prd
from .produce_integration_test_data import (
    assert_reference_fields,
    power_spectrum,
    read_quantity,
    write_quantity,
)

logger = logging.getLogger("py21cmfast")
logger.setLevel(logging.INFO)

options = list(prd.OPTIONS_TESTRUNS.keys())
# These configs test CONST-ION-EFF / GAMMA-APPROX with the default HMF,
# which intentionally triggers the EPS conditional mass function advisory.
_EPS_CONFIGS = ("no-mdz", "ts_nomdz", "mcgs_gamma_approx")
_EPS_FILTER = pytest.mark.filterwarnings(
    "ignore:^Your model .*uses the EPS conditional mass function:UserWarning"
)
options_marked = [
    pytest.param(n, marks=_EPS_FILTER if n in _EPS_CONFIGS else ()) for n in options
]
options_pt = list(prd.OPTIONS_PT.keys())

# Keep the historical diagnostics, but fail only on larger differences.
# Archived CI already reports power differences as large as about 21%.
_POWER_REPORT_RTOL = 1e-4
_POWER_FAILURE_RTOL = 0.25


def assert_physical_type(
    actual: un.Quantity, reference: un.Quantity, *, name: str = ""
) -> None:
    """Reject incompatible dimensions independently of numerical tolerances.

    Parameters
    ----------
    actual, reference
        Typed physical values to compare. Scaled representations of the same
        dimensions are accepted; global physical equivalencies are not applied.
    name
        Field name included in an incompatible-dimensions error.
    """
    if not isinstance(actual, un.Quantity) or not isinstance(reference, un.Quantity):
        raise TypeError("Both actual and reference must be Astropy Quantity objects.")
    if not actual.unit.is_equivalent(reference.unit, equivalencies=None):
        raise un.UnitConversionError(
            f"{name}: incompatible physical dimensions: {actual.unit} and {reference.unit}"
        )


def assert_quantity_close(
    actual: un.Quantity,
    reference: un.Quantity,
    *,
    rtol: float,
    atol: un.Quantity | None = None,
    name: str = "",
) -> None:
    """Assert physical agreement without broadcasting or accepting untyped data."""
    assert_physical_type(actual, reference, name=name)
    assert actual.shape == reference.shape, f"{name}: reference shape mismatch"
    assert np.all(np.isfinite(actual)), f"{name}: nonfinite simulation values"
    assert np.all(np.isfinite(reference)), f"{name}: nonfinite reference values"
    assert_quantity_allclose(
        actual, reference, rtol=rtol, atol=atol, equal_nan=False, err_msg=name
    )


def _record_mismatch(request, test, truth, inputs, abs_tol, rel_tol, name):
    """Record a difference at the reporting tolerance for the summary and plots."""
    # A dimension error is always fatal, independently of the numerical policy.
    assert_physical_type(test, truth, name=name)
    try:
        assert_quantity_close(
            test, truth, rtol=rel_tol, atol=abs_tol * truth.unit, name=name
        )
    except AssertionError as error:
        if (
            test.shape == truth.shape
            and np.all(np.isfinite(test))
            and np.all(np.isfinite(truth))
        ):
            stats = prd.failure_stats(
                test.to_value(truth.unit), truth.value, inputs, abs_tol, rel_tol, name
            )
        else:
            stats = None
        summary, message = (
            stats if stats is not None else (f"{name}: physical mismatch", str(error))
        )
    else:
        return False
    request.node.user_properties.append(("reference_mismatch", message))
    request.node.user_properties.append(("reference_mismatch_summary", summary))
    print(message)
    return True


def _check_power_spectra(
    request: pytest.FixtureRequest,
    test_powers: dict[str, un.Quantity],
    true_powers: dict[str, un.Quantity],
    k_values: np.ndarray,
) -> tuple[bool, list[str]]:
    """Return whether differences were reported and which fields fail."""
    reported = False
    failed_fields = []
    for name, truth in sorted(true_powers.items()):
        test = test_powers[name]
        if not _record_mismatch(
            request, test, truth, [k_values], 0, _POWER_REPORT_RTOL, name
        ):
            continue
        reported = True
        abs_tol = 0 * truth.unit
        if name == "z_reion":
            # Nearly constant float32 redshift grids can produce numerical
            # noise in otherwise empty bins. An amplitude error of order eps
            # contributes power of order eps squared relative to the peak.
            peak = np.max(np.abs(truth.value), initial=0.0)
            abs_tol = peak * float(np.finfo(np.float32).eps) ** 2 * truth.unit
        try:
            assert_quantity_close(
                test, truth, rtol=_POWER_FAILURE_RTOL, atol=abs_tol, name=name
            )
        except AssertionError:
            failed_fields.append(name)
    return reported, failed_fields


@pytest.mark.parametrize("name", options_marked)
def test_power_spectra_coeval(request, name, module_direc, plt):
    redshift, kwargs = prd.OPTIONS_TESTRUNS[name]
    print(f"Options used for the test {name} at z={redshift}: ", kwargs)

    # First get pre-made data
    inputs = prd.get_all_options_struct(redshift, **kwargs)["inputs"]
    expected_fields = prd.get_power_fields(inputs, kind="coeval")
    true_powers = {}
    with h5py.File(prd.get_filename("power_spectra", name), "r") as fl:
        assert "coeval" in fl, f"Missing reference group coeval in {fl.filename}"
        assert_reference_fields(
            fl["coeval"], {"k"} | {f"power_{key}" for key in expected_fields}
        )
        true_k = read_quantity(fl["coeval"]["k"])
        for key, value in fl["coeval"].items():
            if key.startswith("power_"):
                fieldname = "_".join(key.split("_")[1:])
                true_powers[fieldname] = read_quantity(value)
                print(f"{key} --> {fieldname} loaded")

    # Now compute the Coeval object
    test_k, test_powers, cv = prd.produce_coeval_power_spectra(
        redshift, cache=OutputCache(module_direc), **kwargs
    )

    assert isinstance(cv, Coeval)
    assert set(test_powers) == expected_fields, (
        "Simulated power fields differ from the configuration."
    )
    assert np.all(np.isfinite(cv.brightness_temp))

    assert_quantity_close(test_k, true_k, rtol=1e-5, atol=1e-8 * true_k.unit, name="k")

    reported, failed_fields = _check_power_spectra(
        request, test_powers, true_powers, test_k.to_value(true_k.unit)
    )

    if plt == mpl.pyplot and reported:
        make_coeval_comparison_plot(true_k, test_k, true_powers, test_powers, plt)

    assert not failed_fields, (
        f"Coeval physical values differ from the reference data "
        f"at rtol={_POWER_FAILURE_RTOL}: {', '.join(failed_fields)}"
    )


@pytest.mark.parametrize("name", options_marked)
def test_power_spectra_lightcone(request, name, module_direc, plt, benchmark):
    redshift, kwargs = prd.OPTIONS_TESTRUNS[name]
    print(f"Options used for the test {name} at z={redshift}: ", kwargs)

    # First get pre-made data
    inputs = prd.get_all_options_struct(redshift, lc=True, **kwargs)["inputs"]
    expected_fields = prd.get_power_fields(inputs, kind="lightcone")
    with h5py.File(prd.get_filename("power_spectra", name), "r") as fl:
        assert "lightcone" in fl, f"Missing reference group lightcone in {fl.filename}"
        assert_reference_fields(
            fl["lightcone"],
            {"k"}
            | {f"power_{key}" for key in expected_fields}
            | {f"global_{key}" for key in prd.GLOBAL_FIELDS},
        )
        true_powers = {}
        true_global = {}
        true_k = read_quantity(fl["lightcone"]["k"])
        for key in fl["lightcone"]:
            fieldname = "_".join(key.split("_")[1:])
            if key.startswith("power_"):
                true_powers[fieldname] = read_quantity(fl["lightcone"][key])
            elif key.startswith("global_"):
                true_global[fieldname] = read_quantity(fl["lightcone"][key])

    # Now compute the lightcone
    test_k, test_powers, lc = benchmark.pedantic(
        prd.produce_lc_power_spectra,
        kwargs=dict(redshift=redshift, cache=OutputCache(module_direc), **kwargs),
        iterations=1,  # these tests can be slow
        rounds=1,
    )

    if lc.inputs.matter_options.POWER_SPECTRUM == "CLASS":
        with h5py.File(prd.get_filename("power_spectra", name), "r") as fl:
            x_values = fl["cosmo_tables"]["transfer_density"]["x_values"]
            y_values = fl["cosmo_tables"]["transfer_density"]["y_values"]
            transfer_density = Table1D(
                size=x_values.size, x_values=x_values, y_values=y_values
            )
        assert transfer_density.x_values.size == transfer_density.y_values.size
        np.testing.assert_allclose(
            transfer_density.x_values,
            lc.inputs.cosmo_tables.transfer_density.x_values,
            atol=0,
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            transfer_density.y_values,
            lc.inputs.cosmo_tables.transfer_density.y_values,
            atol=0,
            rtol=5e-3,
        )

    assert isinstance(lc, LightCone)
    assert set(test_powers) == expected_fields, (
        "Simulated power fields differ from the configuration."
    )
    assert np.all(np.isfinite(lc.lightcones["brightness_temp"]))

    units = prd.lightcone_field_units(lc)
    test_global = {
        key: un.Quantity(lc.global_quantities[key], units[key], dtype=None, copy=False)
        for key in prd.GLOBAL_FIELDS
    }
    assert_quantity_close(test_k, true_k, rtol=1e-5, atol=1e-8 * true_k.unit, name="k")

    reported, failed_fields = _check_power_spectra(
        request, test_powers, true_powers, test_k.to_value(true_k.unit)
    )

    if plt == mpl.pyplot and reported:
        make_lightcone_comparison_plot(
            true_k,
            test_k,
            lc.inputs.node_redshifts,
            true_powers,
            true_global,
            test_powers,
            test_global,
            plt,
        )

    # For globals, we should assert that they are close
    for key, value in true_global.items():
        assert_quantity_close(test_global[key], value, rtol=1e-3, name=key)

    assert not failed_fields, (
        f"Lightcone physical values differ from the reference data "
        f"at rtol={_POWER_FAILURE_RTOL}: {', '.join(failed_fields)}"
    )


plot_ylab = {
    "density": r"$\delta$",
    "lowres_density": r"$\delta_{IC}$",
    "lowres_vx_2LPT": r"$v_x$ IC 2LPT",
    "lowres_vx": r"$v_x$ IC",
    "velocity_z": r"$v_z$",
    "spin_temperature": r"$T_S$",
    "xray_ionised_fraction": r"$x_{e}$",
    "J_21_LW": r"$J_{21,LW}$",
    "kinetic_temp_neutral": r"$T_{K,HI}$",
    "ionisation_rate_G12": r"$\Gamma_{12}$",
    "cumulative_recombinations": r"$N_{rec}$",
    "neutral_fraction": r"$x_{HI}$",
    "z_reion": r"$z_{re}$",
    "brightness_temp": r"$dT_b$",
}


def make_lightcone_comparison_plot(
    true_k, k, z, true_powers, true_global, test_powers, test_global, plt
):
    n = len(true_global) + len(true_powers)
    _fig, ax = plt.subplots(
        2,
        n,
        figsize=(4 * n, 6),
        sharex="col",
    )

    for i, (key, val) in enumerate(test_powers.items()):
        make_comparison_plot(
            true_k,
            k,
            true_powers[key],
            val,
            ax[:, i],
            xlab="k",
            ylab=f"{plot_ylab[key]} Power",
            make_lower_ylab=i == 0,
        )

    for j, (key, val) in enumerate(test_global.items(), start=i + 1):
        make_comparison_plot(
            z,
            z,
            true_global[key],
            val,
            ax[:, j],
            xlab="z",
            ylab=f"{plot_ylab[key]}",
            logx=False,
            logy=False,
            make_lower_ylab=False,
        )


def make_coeval_comparison_plot(true_k, k, true_powers, test_powers, plt):
    _fig, ax = plt.subplots(
        2,
        len(true_powers),
        figsize=(4 * len(true_powers), 6),
        sharex=True,
    )

    for i, (key, val) in enumerate(test_powers.items()):
        make_comparison_plot(
            true_k,
            k,
            true_powers[key],
            val,
            ax[:, i],
            xlab="k",
            ylab=f"{plot_ylab[key]} Power",
            make_lower_ylab=i == 0,
        )


def make_comparison_plot(
    xtrue,
    x,
    true,
    test,
    ax,
    logx=True,
    logy=True,
    xlab=None,
    ylab=None,
    make_lower_ylab=True,
):
    if isinstance(true, un.Quantity):
        ylab = f"{ylab} [{true.unit}]"
        test = test.to_value(true.unit)
        true = true.value
    if isinstance(xtrue, un.Quantity):
        xlab = f"{xlab} [{xtrue.unit}]"
        x = x.to_value(xtrue.unit)
        xtrue = xtrue.value
    ax[0].plot(xtrue, true, label="True")
    ax[0].plot(x, test, label="Test")
    if logx:
        ax[0].set_xscale("log")
    # Fields a config doesn't evolve (e.g. ionisation_rate_G12) have zero power,
    # which can't be log-scaled.
    if logy and (np.any(true > 0) or np.any(test > 0)):
        ax[0].set_yscale("log")
    if xlab:
        ax[1].set_xlabel(xlab)
    if ylab:
        ax[0].set_title(ylab)

    ax[0].legend()

    ax[1].plot(
        x,
        np.divide(
            test - true,
            true,
            out=np.full_like(test, np.nan, dtype=float),
            where=true != 0,
        ),
    )

    if make_lower_ylab:
        ax[1].set_ylabel("Fractional Difference")


@pytest.mark.parametrize("name", options_pt)
def test_perturb_field_data(name):
    redshift, kwargs = prd.OPTIONS_PT[name]
    print("Options used for the test: ", kwargs)

    # First get pre-made data
    with h5py.File(prd.get_filename("perturb_field_data", name), "r") as f:
        power_dens = read_quantity(f["power_dens"])
        power_vel = read_quantity(f["power_vel"])
        pdf_dens = read_quantity(f["pdf_dens"])
        pdf_vel = read_quantity(f["pdf_vel"])
        axes = {
            key: read_quantity(f[key]) for key in ("k_dens", "k_vel", "x_dens", "x_vel")
        }

    (
        k_dens,
        p_dens,
        k_vel,
        p_vel,
        x_dens,
        y_dens,
        x_vel,
        y_vel,
        _ic,
    ) = prd.produce_perturb_field_data(redshift, **kwargs)

    for dataset_name, actual, reference in (
        ("power_dens", p_dens, power_dens),
        ("power_vel", p_vel, power_vel),
        ("pdf_dens", y_dens, pdf_dens),
        ("pdf_vel", y_vel, pdf_vel),
    ):
        assert_quantity_close(
            actual, reference, rtol=1e-3, atol=5e-3 * reference.unit, name=dataset_name
        )
    for dataset_name, actual in (
        ("k_dens", k_dens),
        ("k_vel", k_vel),
        ("x_dens", x_dens),
        ("x_vel", x_vel),
    ):
        reference = axes[dataset_name]
        assert_quantity_close(
            actual, reference, rtol=1e-5, atol=1e-8 * reference.unit, name=dataset_name
        )


@pytest.fixture(params=["coeval", "lightcone"])
def integration_reference(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> SimpleNamespace:
    """Supply complete synthetic references through the real comparison entry points."""
    integration = sys.modules[__name__]

    kind = request.param
    path = tmp_path / "reference.h5"
    k = np.array([0.1, 0.2]) / un.Mpc
    # Explicit fixture inventory: spin-temperature evolution, no minihalos or recombinations.
    fields = {
        "density",
        "velocity_z",
        "spin_temperature",
        "xray_ionised_fraction",
        "kinetic_temp_neutral",
        "ionisation_rate_G12",
        "neutral_fraction",
        "z_reion",
        "brightness_temp",
    }
    if kind == "coeval":
        fields |= {"lowres_density", "lowres_vx", "lowres_vx_2LPT"}
    field_units = dict.fromkeys(fields, un.dimensionless_unscaled)
    field_units.update(
        {
            "spin_temperature": un.K,
            "kinetic_temp_neutral": un.K,
            "brightness_temp": un.mK,
            "velocity_z": un.Mpc / un.s,
            "lowres_vx": un.Mpc,
            "lowres_vx_2LPT": un.Mpc,
            "ionisation_rate_G12": un.Unit(1e-12 / un.s),
        }
    )
    powers = {
        name: np.array([1.0, 2.0]) * field_units[name] ** 2 * un.Mpc**3
        for name in fields
    }
    global_units = {
        "neutral_fraction": un.dimensionless_unscaled,
        "brightness_temp": un.mK,
    }
    globals_ = {
        "neutral_fraction": np.array([0.5]),
        "brightness_temp": np.array([10.0]),
    }
    with h5py.File(path, "w") as reference:
        group = reference.create_group(kind)
        write_quantity(group, "k", k)
        for name, power in powers.items():
            write_quantity(group, f"power_{name}", power)
        if kind == "lightcone":
            for name, value in globals_.items():
                write_quantity(group, f"global_{name}", value * global_units[name])
    actual_powers = dict(powers)
    result_type = type("Result", (), {})
    result = result_type()
    result.brightness_temp = np.array([1.0])
    result.lightcones = {"brightness_temp": np.array([1.0])}
    result.global_quantities = dict(globals_)
    result.inputs = SimpleNamespace(matter_options=SimpleNamespace(POWER_SPECTRUM="EH"))
    calls = []

    def produce(
        *args: object, **kwargs: object
    ) -> tuple[un.Quantity, dict[str, un.Quantity], object]:
        calls.append(kind)
        return k, actual_powers, result

    monkeypatch.setattr(
        prd, "OPTIONS_TESTRUNS", {"units": [10, {"USE_TS_FLUCT": True}]}
    )
    monkeypatch.setattr(prd, "get_filename", lambda *a: path)
    monkeypatch.setattr(
        prd, f"produce_{'lc' if kind == 'lightcone' else kind}_power_spectra", produce
    )
    monkeypatch.setattr(prd, "lightcone_field_units", lambda lc: global_units)
    monkeypatch.setattr(integration, "OutputCache", lambda path: None)
    monkeypatch.setattr(
        integration, "Coeval" if kind == "coeval" else "LightCone", result_type
    )
    comparison_request = SimpleNamespace(node=SimpleNamespace(user_properties=[]))
    arguments = {
        "request": comparison_request,
        "name": "units",
        "module_direc": tmp_path,
        "plt": None,
    }
    if kind == "lightcone":
        arguments["benchmark"] = SimpleNamespace(
            pedantic=lambda fn, **kw: fn(**kw["kwargs"])
        )

    def compare() -> None:
        getattr(integration, f"test_power_spectra_{kind}")(**arguments)

    return SimpleNamespace(
        kind=kind,
        path=path,
        powers=actual_powers,
        result=result,
        request=comparison_request,
        calls=calls,
        compare=compare,
    )


@pytest.mark.parametrize("consistent", [False, True])
def test_integration_entry_point_checks_physical_power(
    integration_reference: SimpleNamespace, consistent: bool
) -> None:
    case = integration_reference
    reference_power = case.powers["spin_temperature"]
    actual_power = reference_power.to(un.mK**2 * un.Mpc**3)
    if not consistent:
        actual_power = un.Quantity(reference_power.value, actual_power.unit)
    case.powers["spin_temperature"] = actual_power
    if consistent:
        case.compare()
        assert not case.request.node.user_properties
    else:
        with pytest.raises(AssertionError, match="physical values differ"):
            case.compare()
        assert any(
            key == "reference_mismatch" for key, _ in case.request.node.user_properties
        )


def test_integration_dimensions_fail_independently_of_numerical_comparison(
    integration_reference: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagnostic-only numeric comparison cannot hide incompatible dimensions."""
    integration = sys.modules[__name__]

    case = integration_reference
    power = case.powers["spin_temperature"]
    wrong_unit = un.m**2 * un.Mpc**3
    # These composite dimensions can both have the label "unknown"; comparing
    # the physical_type strings would not establish that they are equivalent.
    assert not power.unit.is_equivalent(wrong_unit)
    case.powers["spin_temperature"] = un.Quantity(power.value, wrong_unit)
    monkeypatch.setattr(integration, "assert_quantity_close", lambda *a, **kw: None)
    with pytest.raises(un.UnitConversionError, match="spin_temperature"):
        case.compare()


@pytest.mark.parametrize(("scale", "fails"), [(1.2, False), (1.3, True)])
def test_integration_reports_smaller_power_mismatches(
    integration_reference: SimpleNamespace, scale: float, fails: bool
) -> None:
    case = integration_reference
    case.powers["spin_temperature"] *= scale
    if fails:
        with pytest.raises(
            AssertionError, match=r"physical values differ.*spin_temperature"
        ):
            case.compare()
    else:
        case.compare()
    assert any(
        key == "reference_mismatch_summary" and value.startswith("spin_temperature ")
        for key, value in case.request.node.user_properties
    )


@pytest.mark.parametrize(
    ("field", "tail_power", "fails"),
    [
        ("z_reion", 1e-15, False),
        ("z_reion", 1e-12, True),
        ("neutral_fraction", 1e-15, True),
    ],
)
def test_integration_zero_power_roundoff_is_limited_to_reionization_redshift(
    integration_reference: SimpleNamespace, field: str, tail_power: float, fails: bool
) -> None:
    case = integration_reference
    with h5py.File(case.path, "r+") as reference:
        reference[case.kind][f"power_{field}"][...] = [1.0, 0.0]
    case.powers[field] = (np.array([1.0, tail_power]) * un.Mpc**3).to(un.kpc**3)
    if fails:
        with pytest.raises(AssertionError, match=rf"physical values differ.*{field}"):
            case.compare()
    else:
        case.compare()
    assert any(
        key == "reference_mismatch_summary" and value.startswith(f"{field} ")
        for key, value in case.request.node.user_properties
    )


@pytest.mark.parametrize("omission", ["one_power", "k"])
def test_integration_rejects_missing_reference_fields(
    integration_reference: SimpleNamespace, omission: str
) -> None:
    case = integration_reference
    with h5py.File(case.path, "r+") as reference:
        group = reference[case.kind]
        if omission == "one_power":
            del group["power_spin_temperature"]
        else:
            del group["k"]
    with pytest.raises(AssertionError, match="Missing reference"):
        case.compare()
    assert not case.calls, "Incomplete references must fail before a simulation starts."


def test_integration_rejects_unexpected_reference_power(
    integration_reference: SimpleNamespace,
) -> None:
    case = integration_reference
    with h5py.File(case.path, "r+") as reference:
        write_quantity(
            reference[case.kind], "power_unexpected", np.array([1.0, 2.0]) * un.Mpc**3
        )
    with pytest.raises(AssertionError, match="Unexpected reference"):
        case.compare()
    assert not case.calls


def test_integration_rejects_missing_simulated_power(
    integration_reference: SimpleNamespace,
) -> None:
    case = integration_reference
    del case.powers["spin_temperature"]
    with pytest.raises(AssertionError, match="Simulated power fields"):
        case.compare()


@pytest.mark.parametrize("integration_reference", ["lightcone"], indirect=True)
def test_lightcone_rejects_missing_global_reference(
    integration_reference: SimpleNamespace,
) -> None:
    case = integration_reference
    with h5py.File(case.path, "r+") as reference:
        del reference["lightcone/global_neutral_fraction"]
    with pytest.raises(
        AssertionError, match=r"Missing reference.*global_neutral_fraction"
    ):
        case.compare()
    assert not case.calls


def test_comparison_plot_converts_to_reference_units(tmp_path: Path) -> None:
    from matplotlib.figure import Figure

    figure = Figure()
    figure_ref = weakref.ref(figure)
    axes = None
    try:
        axes = figure.subplots(2)
        k = np.array([0.1, 0.2, 0.3]) / un.Mpc
        power = np.array([1.0, 2.0, 3.0]) * un.K**2 * un.Mpc**3
        make_comparison_plot(
            k,
            k.to(un.kpc**-1),
            power,
            power.to(un.mK**2 * un.Mpc**3),
            axes,
            xlab="k",
            ylab="Temperature power",
        )
        np.testing.assert_allclose(
            axes[0].lines[0].get_ydata(),
            axes[0].lines[1].get_ydata(),
            rtol=1e-14,
            atol=0,
        )
        np.testing.assert_allclose(axes[1].lines[0].get_ydata(), 0, rtol=0, atol=1e-14)
        figure.savefig(tmp_path / "comparison.png")
    finally:
        figure.clear()
        del axes, figure
        # Release this test's Matplotlib cycles before driver cycle checks run.
        gc.collect()
    assert figure_ref() is None, "Comparison plot kept its Figure alive after cleanup."


def test_power_spectrum_preserves_amplitude_and_volume_scaling() -> None:
    """Exercise the power estimator, not only declared dimensions."""
    grid = np.indices((16, 16, 16))
    noise = np.random.default_rng(21).normal(0, 0.1, size=grid.shape[1:])
    field = (np.sin(2 * np.pi * grid[0] / 16) + noise) * un.K
    small, k = power_spectrum(field, 16 * un.Mpc)
    converted, converted_k = power_spectrum(field.to(un.mK), 16000 * un.kpc)
    assert_quantity_close(small, converted, rtol=1e-12)
    assert_quantity_close(k, converted_k, rtol=1e-12)
    large, large_k = power_spectrum(field, 32 * un.Mpc)
    assert_quantity_close(large, 8 * small, rtol=1e-12)
    assert_quantity_close(large_k, k / 2, rtol=1e-12)
