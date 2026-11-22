"""Unit tests for output structures."""

import pickle

import attrs
import numpy as np
import pytest
from astropy import units as un

from py21cmfast import (
    InitialConditions,  # An example of an output struct
    InputParameters,
    OutputCache,
    compute_initial_conditions,
    config,
    determine_halo_catalog,
    perturb_halo_catalog,
)
from py21cmfast.wrapper import outputs as ox


@pytest.fixture
def init(default_input_struct: InputParameters):
    return InitialConditions.new(inputs=default_input_struct)


@pytest.fixture(scope="module")
def ic_with_halos(default_input_struct, cache: OutputCache):
    return compute_initial_conditions(
        inputs=default_input_struct.evolve_input_structs(SOURCE_MODEL="CHMF-SAMPLER"),
        write=True,
        cache=cache,
    )


@pytest.fixture
def halo_cat(ic_with_halos: InitialConditions, default_input_struct: InputParameters):
    return determine_halo_catalog(
        redshift=10.0,
        initial_conditions=ic_with_halos,
        inputs=default_input_struct.evolve_input_structs(SOURCE_MODEL="CHMF-SAMPLER"),
    )


@pytest.fixture
def pert_halo_cat(ic_with_halos: InitialConditions, halo_cat: ox.HaloCatalog):
    return perturb_halo_catalog(
        initial_conditions=ic_with_halos,
        halo_catalog=halo_cat,
    )


def test_different_seeds(
    init: InitialConditions,
    default_input_struct: InputParameters,
):
    ic2 = InitialConditions.new(
        inputs=default_input_struct.clone(
            random_seed=default_input_struct.random_seed + 1
        )
    )

    assert init is not ic2
    assert init != ic2

    # make sure we didn't inadvertantly set the random seed while doing any of this
    assert init.random_seed == default_input_struct.random_seed


def test_pickleability(default_input_struct: InputParameters):
    ic_ = InitialConditions.new(inputs=default_input_struct)
    s = pickle.dumps(ic_)

    ic2 = pickle.loads(s)
    assert repr(ic_) == repr(ic2)


def test_reading_purged(ic: InitialConditions):
    lowres_density = ic.get("lowres_density")

    # Remove it from memory
    ic.purge()

    assert not ic.arrays["lowres_density"].state.computed_in_mem
    assert ic.arrays["lowres_density"].state.on_disk

    # But we can still get it.
    lowres_density_2 = ic.get("lowres_density")

    assert ic.arrays["lowres_density"].state.on_disk
    assert ic.arrays["lowres_density"].state.computed_in_mem

    assert np.allclose(lowres_density_2, lowres_density)

    ic.load_all()


def test_direct_array_access_after_purge_loads_transparently(ic: InitialConditions):
    """Plain attribute access resolves a purged array, and caches it back (see #565)."""
    expected = ic.get("lowres_density")

    ic.purge()
    assert not ic.arrays["lowres_density"].state.computed_in_mem

    assert ic.lowres_density.mean() == pytest.approx(expected.mean())
    assert np.allclose(ic.lowres_density, expected)

    # Cached back onto the struct as a side effect of the access above.
    assert ic.arrays["lowres_density"].state.computed_in_mem

    ic.load_all()


def test_direct_array_access_gives_a_real_ndarray(ic: InitialConditions):
    """The attribute must behave as an array in *every* respect, not just some.

    Half duck-typing was the original problem: methods worked but operators didn't.
    """
    density = ic.lowres_density

    assert isinstance(density, np.ndarray)
    assert np.all((density - density) == 0.0)
    assert (density**2).mean() >= 0.0
    # `Array == <scalar>` used to silently evaluate to False.
    assert np.all(np.zeros_like(density) == 0.0)
    assert density[0].shape == density.shape[1:]
    assert len(density) == density.shape[0]


def test_direct_array_access_after_purge_respects_no_cache_config(
    ic: InitialConditions,
):
    """Direct access still resolves the value with auto-caching disabled.

    But it must not repopulate memory on the struct.
    """
    expected = ic.get("lowres_density")
    ic.purge()

    with config.use(CACHE_ARRAYS_ON_ACCESS=False):
        assert ic.lowres_density.mean() == pytest.approx(expected.mean())
        assert not ic.arrays["lowres_density"].state.computed_in_mem

    ic.load_all()


def test_repr_of_purged_struct_does_not_load_it(ic: InitialConditions):
    """Merely inspecting a purged struct (e.g. in a REPL) must stay cheap.

    The `Array` lives in a private attrs field, so the generated `__repr__` never
    goes through the descriptor and so never touches the disk.
    """
    ic.purge()
    assert not ic.arrays["lowres_density"].state.computed_in_mem

    _ = repr(ic)
    _ = repr(ic.arrays["lowres_density"])

    assert not ic.arrays["lowres_density"].state.computed_in_mem
    assert ic.arrays["lowres_density"]._value is None

    ic.load_all()


def test_has(ic: InitialConditions):
    """`has()` distinguishes "no data yet" from "this field does not exist"."""
    assert ic.has("lowres_density")
    assert not ic.has("lowres_vcb")  # not produced by these inputs

    ic.purge()
    # Still available: it is on disk, and reading the attribute will load it.
    assert ic.has("lowres_density")
    ic.load_all()

    fresh = ox.InitialConditions.new(inputs=ic.inputs)
    assert not fresh.has("lowres_density")
    with pytest.raises(ValueError, match="not on disk and not initialized"):
        _ = fresh.lowres_density


@pytest.mark.parametrize("struct", list(ox._ALL_OUTPUT_STRUCTS.values()))
def test_all_fields_exist(struct: ox.OutputStruct):
    cstruct = ox.StructWrapper(struct.__name__)

    this = attrs.fields_dict(struct)

    # Ensure that all fields in the cstruct are also defined on this class. Array
    # fields are declared privately and exposed under the public name by a descriptor.
    for name in cstruct.pointer_fields:
        assert name in struct._array_field_names
        assert this[f"_{name}"].type == ox.Array

    for name in cstruct.primitive_fields:
        assert name in this


def test_halocatalogs(default_input_struct_lc: InputParameters):
    """Ensure that the halo catalogs can be made."""
    # First let's define buffer_size
    inputs = default_input_struct_lc.evolve_input_structs(SOURCE_MODEL="CHMF-SAMPLER")
    halo_cat = ox.HaloCatalog.new(redshift=0.0, inputs=inputs, buffer_size=1)
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=inputs, buffer_size=1
    )
    assert isinstance(halo_cat, ox.HaloCatalog)
    assert isinstance(pert_halo_cat, ox.PerturbedHaloCatalog)

    # Now let's not define buffer_size, it should default to None
    halo_cat = ox.HaloCatalog.new(
        redshift=0.0,
        inputs=inputs,  # buffer_size = None
    )
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0,
        inputs=inputs,  # buffer_size = None
    )
    assert isinstance(halo_cat, ox.HaloCatalog)
    assert isinstance(pert_halo_cat, ox.PerturbedHaloCatalog)

    # Now let's define dummy=True
    halo_cat = ox.HaloCatalog.new(redshift=0.0, inputs=inputs, dummy=True)
    pert_halo_cat = ox.PerturbedHaloCatalog.new(redshift=0.0, inputs=inputs, dummy=True)
    assert isinstance(halo_cat, ox.HaloCatalog)
    assert isinstance(pert_halo_cat, ox.PerturbedHaloCatalog)
    assert halo_cat.buffer_size == 0
    assert pert_halo_cat.buffer_size == 0


# NOTE: These do not test every field, but does test every conditional in the
#   OutputStruct constructors, a better approach would probably be to have a
#   comprehensive list of {"field_name": {"flag": value}} conditions for the fields
#   in the output module which is checked in the constructors
def test_optional_field_ic(default_input_struct_lc: InputParameters):
    """Ensure that the correct InitialConditions fields are set based on the parameters."""
    ic = ox.InitialConditions.new(inputs=default_input_struct_lc)
    assert "lowres_vx" in ic.arrays
    assert "lowres_vx_2LPT" in ic.arrays
    assert "hires_vx" not in ic.arrays
    assert "hires_vx_2LPT" in ic.arrays  # Python requires it, check the C
    assert "lowres_vcb" not in ic.arrays

    ic = ox.InitialConditions.new(
        inputs=default_input_struct_lc.evolve_input_structs(
            PERTURB_ALGORITHM="ZELDOVICH"
        )
    )
    assert "lowres_vy" in ic.arrays
    assert "lowres_vy_2LPT" not in ic.arrays
    assert "hires_vy" not in ic.arrays
    assert "hires_vy_2LPT" not in ic.arrays

    ic = ox.InitialConditions.new(
        inputs=default_input_struct_lc.evolve_input_structs(PERTURB_ON_HIGH_RES=True)
    )
    assert "lowres_vz" not in ic.arrays
    assert "lowres_vz_2LPT" not in ic.arrays
    assert "hires_vz" in ic.arrays
    assert "hires_vz_2LPT" in ic.arrays

    ic = ox.InitialConditions.new(
        inputs=default_input_struct_lc.evolve_input_structs(
            V_CB_MODEL="FLUCTS",
            POWER_SPECTRUM="CLASS",
            USE_MCGS=True,
            USE_TS_FLUCT=True,
            RECOMB_MODEL="inhomogeneous",
            R_BUBBLE_MAX=50.0,
            M_TURN_STELLAR_FEEDBACK=5.0,
            K_MAX_FOR_CLASS=1.0,
        )
    )

    assert "lowres_vx" in ic.arrays
    assert "lowres_vx_2LPT" in ic.arrays
    assert "hires_vx" not in ic.arrays
    assert "hires_vx_2LPT" in ic.arrays
    assert "lowres_vcb" in ic.arrays


def test_optional_field_perturb(default_input_struct_lc: InputParameters):
    """Ensure that the correct PerturbedField fields are set based on the parameters."""
    pt = ox.PerturbedField.new(redshift=0.0, inputs=default_input_struct_lc)
    assert "density" in pt.arrays
    assert "velocity_z" in pt.arrays
    assert "velocity_x" in pt.arrays
    assert "velocity_y" in pt.arrays

    pt = ox.PerturbedField.new(
        redshift=0.0,
        inputs=default_input_struct_lc.evolve_input_structs(KEEP_3D_VELOCITIES=False),
    )
    assert "density" in pt.arrays
    assert "velocity_z" in pt.arrays
    assert "velocity_x" not in pt.arrays
    assert "velocity_y" not in pt.arrays


def test_optional_field_perturbed_halocat(default_input_struct_lc: InputParameters):
    """Ensure that the correct EmissivityFields fields are set based on the parameters."""
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=default_input_struct_lc, buffer_size=1
    )
    assert "halo_masses" in pert_halo_cat.arrays
    assert "halo_coords" in pert_halo_cat.arrays
    assert "halo_masses" in pert_halo_cat.arrays
    assert "halo_coords" in pert_halo_cat.arrays
    assert "stellar_masses_acg" in pert_halo_cat.arrays
    assert "n_ion" in pert_halo_cat.arrays
    assert "xray_luminosity" not in pert_halo_cat.arrays
    assert "fesc_weighted_sfr" not in pert_halo_cat.arrays
    assert "stellar_masses_mcg" not in pert_halo_cat.arrays
    assert "sfr_mcg" not in pert_halo_cat.arrays

    inputs = default_input_struct_lc.evolve_input_structs(USE_TS_FLUCT=True)
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=inputs, buffer_size=1
    )
    assert "xray_luminosity" in pert_halo_cat.arrays
    inputs = inputs.evolve_input_structs(
        RECOMB_MODEL="inhomogeneous", R_BUBBLE_MAX=50.0
    )
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=inputs, buffer_size=1
    )
    assert "fesc_weighted_sfr" in pert_halo_cat.arrays
    inputs = inputs.evolve_input_structs(
        USE_MCGS=True,
        V_CB_MODEL="FLUCTS",
        POWER_SPECTRUM="CLASS",
        M_TURN_STELLAR_FEEDBACK=5.0,
    )
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=inputs, buffer_size=1
    )
    assert "stellar_masses_mcg" in pert_halo_cat.arrays
    assert "sfr_mcg" in pert_halo_cat.arrays


def test_optional_emissivity_fields(default_input_struct_lc: InputParameters):
    """Ensure that the correct EmissivityFields fields are set based on the parameters."""
    emissivity_fields = ox.EmissivityFields.new(
        redshift=0.0, inputs=default_input_struct_lc
    )
    assert "halo_number" not in emissivity_fields.arrays
    assert "halo_mass_density" not in emissivity_fields.arrays
    assert "stellar_mass_density_acg" not in emissivity_fields.arrays
    assert "stellar_mass_density_mcg" not in emissivity_fields.arrays
    assert "sfrd_acg" not in emissivity_fields.arrays
    assert "sfrd_mcg" not in emissivity_fields.arrays
    assert "xray_emissivity" not in emissivity_fields.arrays
    assert "fesc_weighted_sfrd" not in emissivity_fields.arrays
    assert "n_ion" in emissivity_fields.arrays

    with config.use(EXTRA_EMISSIVITY_FIELDS=True):
        emissivity_fields = ox.EmissivityFields.new(
            redshift=0.0, inputs=default_input_struct_lc
        )
        assert "halo_mass_density" in emissivity_fields.arrays
        assert "halo_number" in emissivity_fields.arrays
        assert "stellar_mass_density_acg" in emissivity_fields.arrays
        assert "stellar_mass_density_mcg" not in emissivity_fields.arrays

        emissivity_fields = ox.EmissivityFields.new(
            redshift=0.0,
            inputs=default_input_struct_lc.evolve_input_structs(
                USE_TS_FLUCT=True,
                RECOMB_MODEL="inhomogeneous",
                R_BUBBLE_MAX=50.0,
                USE_MCGS=True,
                V_CB_MODEL="FLUCTS",
                POWER_SPECTRUM="CLASS",
                M_TURN_STELLAR_FEEDBACK=5.0,
            ),
        )
        assert "stellar_mass_density_mcg" in emissivity_fields.arrays

    inputs = default_input_struct_lc.evolve_input_structs(
        RECOMB_MODEL="inhomogeneous", R_BUBBLE_MAX=50.0, SOURCE_MODEL="L-INTEGRAL"
    )
    emissivity_fields = ox.EmissivityFields.new(redshift=0.0, inputs=inputs)
    assert "fesc_weighted_sfrd" in emissivity_fields.arrays

    inputs = inputs.evolve_input_structs(USE_TS_FLUCT=True)
    emissivity_fields = ox.EmissivityFields.new(redshift=0.0, inputs=inputs)
    assert "sfrd_acg" in emissivity_fields.arrays
    assert "xray_emissivity" in emissivity_fields.arrays

    inputs = inputs.evolve_input_structs(
        USE_MCGS=True,
        V_CB_MODEL="FLUCTS",
        POWER_SPECTRUM="CLASS",
        M_TURN_STELLAR_FEEDBACK=5.0,
    )
    emissivity_fields = ox.EmissivityFields.new(redshift=0.0, inputs=inputs)
    assert "sfrd_mcg" in emissivity_fields.arrays


def test_optional_setup_radiation_fields(default_input_struct_lc: InputParameters):
    """Ensure that the correct fields of RadiationFieldsSetup are set based on the parameters."""
    rfs = ox.RadiationFieldsSetup.new(redshift=0.0, inputs=default_input_struct_lc)
    assert "filtered_sfrd_acg_for_lya" in rfs.arrays
    assert "filtered_xray_emissivity" in rfs.arrays
    assert "filtered_sfrd_mcg_for_lya" not in rfs.arrays

    inputs = default_input_struct_lc.evolve_input_structs(
        USE_TS_FLUCT=True,
        USE_MCGS=True,
        V_CB_MODEL="FLUCTS",
        POWER_SPECTRUM="CLASS",
        RECOMB_MODEL="inhomogeneous",
        R_BUBBLE_MAX=50.0,
        M_TURN_STELLAR_FEEDBACK=5.0,
    )
    rfs = ox.RadiationFieldsSetup.new(redshift=0.0, inputs=inputs)
    assert "filtered_sfrd_mcg_for_lya" in rfs.arrays


def test_optional_field_ts(default_input_struct_lc: InputParameters):
    """Ensure that the correct TsBox fields are set based on the parameters."""
    ts = ox.TsBox.new(redshift=0.0, inputs=default_input_struct_lc)
    assert "spin_temperature" in ts.arrays
    assert "xray_ionised_fraction" in ts.arrays
    assert "kinetic_temp_neutral" in ts.arrays
    assert "J_21_LW" not in ts.arrays

    inputs = default_input_struct_lc.evolve_input_structs(
        USE_TS_FLUCT=True,
        RECOMB_MODEL="inhomogeneous",
        R_BUBBLE_MAX=50.0,
        USE_MCGS=True,
        V_CB_MODEL="FLUCTS",
        POWER_SPECTRUM="CLASS",
        M_TURN_STELLAR_FEEDBACK=5.0,
    )
    ts = ox.TsBox.new(redshift=0.0, inputs=inputs)
    assert "J_21_LW" in ts.arrays


def test_optional_field_ion(default_input_struct_lc: InputParameters):
    """Ensure that the correct IonizedBox fields are set based on the parameters."""
    ion = ox.IonizedBox.new(redshift=0.0, inputs=default_input_struct_lc)
    assert "neutral_fraction" in ion.arrays
    assert "nion_conditional_filtered_mcg" not in ion.arrays
    assert "cumulative_recombinations" not in ion.arrays

    inputs = default_input_struct_lc.evolve_input_structs(
        RECOMB_MODEL="inhomogeneous",
        R_BUBBLE_MAX=50.0,
    )
    ion = ox.IonizedBox.new(redshift=0.0, inputs=inputs)
    assert "cumulative_recombinations" in ion.arrays

    inputs = inputs.evolve_input_structs(
        USE_TS_FLUCT=True,
        USE_MCGS=True,
        V_CB_MODEL="FLUCTS",
        POWER_SPECTRUM="CLASS",
        M_TURN_STELLAR_FEEDBACK=5.0,
    )
    ion = ox.IonizedBox.new(redshift=0.0, inputs=inputs)
    assert "nion_conditional_filtered_mcg" in ion.arrays


@pytest.fixture(scope="module", params=["E-INTEGRAL", "CHMF-SAMPLER"])
def field_metadata_inputs(
    default_input_struct_lc: InputParameters, request: pytest.FixtureRequest
) -> InputParameters:
    """Enable all optional fields, with two mutually exclusive source grids."""
    return default_input_struct_lc.evolve_input_structs(
        HII_DIM=2,
        DIM=4,
        USE_TS_FLUCT=True,
        USE_MCGS=True,
        V_CB_MODEL="AVG-DEBUG",
        M_TURN_STELLAR_FEEDBACK=5.0,
        RECOMB_MODEL=(
            "homogeneous" if request.param == "E-INTEGRAL" else "inhomogeneous"
        ),
        CELL_RECOMB=request.param == "E-INTEGRAL",
        R_BUBBLE_MAX=50.0,
        MINIMIZE_MEMORY=False,
        USE_LYA_HEATING=request.param == "CHMF-SAMPLER",
        USE_X_RAY_HEATING=True,
        LYA_MULTIPLE_SCATTERING=True,
        SOURCE_MODEL=request.param,
    )


def _check_field_metadata(
    box: ox.OutputStruct,
    expected: dict[str, un.UnitBase],
    *,
    optional_fields: dict[str, bool],
) -> None:
    """Check the field inventory and physical dimensions, allowing equivalent units."""
    expected_names = {name for name in expected if optional_fields.get(name, True)}
    assert set(box.arrays) == expected_names
    for name, array in box.arrays.items():
        assert isinstance(expected[name], un.UnitBase), name
        assert array.unit.is_equivalent(expected[name], equivalencies=None), name


@pytest.mark.parametrize("high_resolution", [False, True])
def test_initial_conditions_field_metadata(
    default_input_struct_lc: InputParameters,
    high_resolution: bool,
) -> None:
    """Distinguish the initial displacement fields from evolved velocities."""
    # Fluctuating relative velocities require the minihalo thermal/recombination model.
    inputs = default_input_struct_lc.evolve_input_structs(
        PERTURB_ON_HIGH_RES=high_resolution,
        PERTURB_ALGORITHM="2LPT",
        V_CB_MODEL="FLUCTS",
        POWER_SPECTRUM="CLASS",
        USE_MCGS=True,
        USE_TS_FLUCT=True,
        RECOMB_MODEL="inhomogeneous",
        R_BUBBLE_MAX=50.0,
        M_TURN_STELLAR_FEEDBACK=5.0,
    )
    expected = {
        "lowres_density": un.dimensionless_unscaled,
        "hires_density": un.dimensionless_unscaled,
        "lowres_vcb": un.m / un.s,
    }
    optional_fields = {}
    for resolution in ("lowres", "hires"):
        for axis in "xyz":
            name = f"{resolution}_v{axis}"
            expected[name] = un.m
            optional_fields[name] = (resolution == "hires") == high_resolution
            expected[name + "_2LPT"] = un.m
            optional_fields[name + "_2LPT"] = (
                resolution == "hires" or not high_resolution
            )
    _check_field_metadata(
        ox.InitialConditions.new(inputs=inputs),
        expected,
        optional_fields=optional_fields,
    )


def test_perturbed_field_metadata(
    default_input_struct: InputParameters,
) -> None:
    """Check density and the velocity components enabled for each geometry."""
    inputs = default_input_struct.evolve_input_structs(
        HII_DIM=2, DIM=4, KEEP_3D_VELOCITIES=True
    )
    _check_field_metadata(
        ox.PerturbedField.new(inputs=inputs, redshift=10.0),
        {
            "density": un.dimensionless_unscaled,
            **{f"velocity_{axis}": un.m / un.s for axis in "xyz"},
        },
        optional_fields={},
    )


def test_brightness_temperature_field_metadata(
    default_input_struct_lc: InputParameters,
) -> None:
    """Check temperature dimensions and the optional optical-depth field."""
    inputs = default_input_struct_lc.evolve_input_structs(USE_TS_FLUCT=True)
    _check_field_metadata(
        ox.BrightnessTemp.new(inputs=inputs, redshift=10.0),
        {
            "brightness_temp": un.K,
            "tau_21": un.dimensionless_unscaled,
        },
        optional_fields={},
    )


def test_ts_box_field_metadata(field_metadata_inputs: InputParameters) -> None:
    """Describe TsBox fields using their physical dimensions and meanings."""
    box = ox.TsBox.new(inputs=field_metadata_inputs, redshift=10.0)
    _check_field_metadata(
        box,
        {
            "spin_temperature": un.K,
            "xray_ionised_fraction": un.dimensionless_unscaled,
            "kinetic_temp_neutral": un.K,
            "J_21_LW": un.erg / (un.s * un.cm**2 * un.Hz * un.sr),
        },
        optional_fields={"J_21_LW": field_metadata_inputs.astro_options.USE_MCGS},
    )


def test_ionized_box_field_metadata(field_metadata_inputs: InputParameters) -> None:
    """Describe IonizedBox fields across optional-field configurations."""
    box = ox.IonizedBox.new(inputs=field_metadata_inputs, redshift=10.0)
    matter = field_metadata_inputs.matter_options
    astro = field_metadata_inputs.astro_options
    _check_field_metadata(
        box,
        {
            "neutral_fraction": un.dimensionless_unscaled,
            "ionisation_rate_G12": un.s**-1,
            "z_reion": un.dimensionless_unscaled,
            "mean_free_path": un.Mpc,
            "cumulative_recombinations": un.dimensionless_unscaled,
            "kinetic_temperature": un.K,
            "nion_conditional_filtered_acg": un.dimensionless_unscaled,
            "nion_conditional_filtered_mcg": un.dimensionless_unscaled,
        },
        optional_fields={
            "kinetic_temperature": not matter.MINIMIZE_MEMORY,
            "mean_free_path": (
                not matter.MINIMIZE_MEMORY and astro.RECOMB_MODEL != "none"
            ),
            "cumulative_recombinations": astro.RECOMB_MODEL != "none",
            "nion_conditional_filtered_acg": not matter.lagrangian_source_grid,
            "nion_conditional_filtered_mcg": (
                not matter.lagrangian_source_grid and astro.USE_MCGS
            ),
        },
    )


def test_halo_catalog_field_metadata(default_input_struct: InputParameters) -> None:
    """Separate dimensional halo properties from dimensionless Gaussian variates."""
    _check_field_metadata(
        ox.HaloCatalog.new(inputs=default_input_struct, redshift=10.0, buffer_size=3),
        {
            "halo_masses": un.kg,
            "halo_coords": un.m,
            "star_rng": un.dimensionless_unscaled,
            "sfr_rng": un.dimensionless_unscaled,
            "xray_rng": un.dimensionless_unscaled,
        },
        optional_fields={},
    )


def test_perturbed_halo_catalog_field_metadata(
    field_metadata_inputs: InputParameters,
) -> None:
    """Check halo properties before deposition divides them by cell volume."""
    inputs = field_metadata_inputs
    _check_field_metadata(
        ox.PerturbedHaloCatalog.new(inputs=inputs, redshift=10.0, buffer_size=3),
        {
            "halo_masses": un.kg,
            "halo_coords": un.m,
            "stellar_masses_acg": un.kg,
            "stellar_masses_mcg": un.kg,
            "sfr_acg": un.kg / un.s,
            "sfr_mcg": un.kg / un.s,
            "n_ion": un.m**3,
            "xray_luminosity": un.erg / un.s,
            "fesc_weighted_sfr": un.kg / un.s,
        },
        optional_fields={
            "stellar_masses_mcg": inputs.astro_options.USE_MCGS,
            "sfr_mcg": inputs.astro_options.USE_MCGS,
            "xray_luminosity": inputs.astro_options.USE_TS_FLUCT,
            "fesc_weighted_sfr": inputs.astro_options.RECOMB_MODEL != "none",
        },
    )


def test_emissivity_fields_field_metadata(
    field_metadata_inputs: InputParameters,
) -> None:
    """Check deposited counts, densities, and the normalized ionizing budget."""
    inputs = field_metadata_inputs
    with config.use(EXTRA_EMISSIVITY_FIELDS=True):
        box = ox.EmissivityFields.new(inputs=inputs, redshift=10.0)
    _check_field_metadata(
        box,
        {
            "halo_number": un.dimensionless_unscaled,
            "halo_mass_density": un.kg / un.m**3,
            "stellar_mass_density_acg": un.kg / un.m**3,
            "stellar_mass_density_mcg": un.kg / un.m**3,
            "sfrd_acg": un.kg / (un.s * un.m**3),
            "sfrd_mcg": un.kg / (un.s * un.m**3),
            "xray_emissivity": un.erg / (un.s * un.m**3),
            "n_ion": un.dimensionless_unscaled,
            "fesc_weighted_sfrd": un.kg / (un.s * un.m**3),
        },
        optional_fields={
            "stellar_mass_density_mcg": inputs.astro_options.USE_MCGS,
            "sfrd_acg": inputs.astro_options.USE_TS_FLUCT,
            "sfrd_mcg": inputs.astro_options.USE_TS_FLUCT
            and inputs.astro_options.USE_MCGS,
            "xray_emissivity": inputs.astro_options.USE_TS_FLUCT,
            "fesc_weighted_sfrd": inputs.astro_options.RECOMB_MODEL != "none"
            and inputs.matter_options.lagrangian_source_grid,
        },
    )


def test_radiation_fields_setup_field_metadata(
    field_metadata_inputs: InputParameters,
) -> None:
    """Check frequency integrals, shell coordinates, and interpolation helpers."""
    inputs = field_metadata_inputs
    astro = inputs.astro_options
    expected = {
        "R_values": un.m,
        "zpp_avg": un.dimensionless_unscaled,
        "zpp_edges": un.dimensionless_unscaled,
        "inverse_diff": un.dimensionless_unscaled,
        "m_xHII_low_box": un.dimensionless_unscaled,
        "inverse_val_box": un.dimensionless_unscaled,
        "ave_log10_MturnLW": un.dimensionless_unscaled,
        "filtered_xray_emissivity": un.erg / (un.s * un.m**3),
    }
    optional = {"ave_log10_MturnLW": astro.USE_MCGS}
    for integral, dimension in {
        "heat": un.erg * un.cm**2 * un.Hz,
        "ion": un.cm**2 * un.Hz,
        "lya": un.cm**3 / un.sr,
    }.items():
        expected[f"freq_int_{integral}_tbl"] = dimension
        expected[f"freq_int_{integral}_tbl_diff"] = dimension
    for population in ("acg", "mcg"):
        for band in ("lya", "lw"):
            name = f"filtered_sfrd_{population}_for_{band}"
            expected[name] = un.kg / (un.s * un.m**3)
            optional[name] = (
                population == "acg" or astro.USE_MCGS
                if band == "lya"
                else astro.USE_MCGS and astro.LYA_MULTIPLE_SCATTERING
            )
        for channel in ("continuum", "injected", "continuum_injected"):
            name = f"lya_flux_{channel}_prefactor_{population}"
            expected[name] = un.Hz ** (-1)
            optional[name] = (population == "acg" or astro.USE_MCGS) and (
                (channel != "continuum_injected" and astro.USE_LYA_HEATING)
                or (channel == "continuum_injected" and (not astro.USE_LYA_HEATING))
            )
        name = f"lyw_flux_prefactor_{population}"
        expected[name] = un.dimensionless_unscaled
        optional[name] = astro.USE_MCGS
    _check_field_metadata(
        ox.RadiationFieldsSetup.new(inputs=inputs, redshift=10.0),
        expected,
        optional_fields=optional,
    )


def test_radiation_fields_field_metadata(
    field_metadata_inputs: InputParameters,
) -> None:
    """Distinguish temperature/ionization rates and photon-number/energy intensity."""
    inputs = field_metadata_inputs
    astro = inputs.astro_options
    number_intensity = un.cm ** (-2) * un.s ** (-1) * un.Hz ** (-1) * un.sr ** (-1)
    _check_field_metadata(
        ox.RadiationFields.new(inputs=inputs, redshift=10.0),
        {
            "xray_heating_rate": un.K / un.s,
            "xray_ionization_rate": un.s ** (-1),
            "xray_lya_flux": number_intensity,
            "lya_flux_continuum": number_intensity,
            "lya_flux_injected": number_intensity,
            "lya_flux_continuum_injected": number_intensity,
            "lyw_flux": un.erg / (un.s * un.cm**2 * un.Hz * un.sr),
        },
        optional_fields={
            "xray_heating_rate": astro.USE_X_RAY_HEATING,
            "lyw_flux": astro.USE_MCGS,
            "lya_flux_continuum": astro.USE_LYA_HEATING,
            "lya_flux_injected": astro.USE_LYA_HEATING,
            "lya_flux_continuum_injected": not astro.USE_LYA_HEATING,
        },
    )


def test_optional_field_bt(default_input_struct_lc: InputParameters):
    """Ensure that the correct BrightnessTemp fields are set based on the parameters."""
    bt = ox.BrightnessTemp.new(redshift=0.0, inputs=default_input_struct_lc)
    assert "brightness_temp" in bt.arrays
    assert "tau_21" not in bt.arrays

    inputs = default_input_struct_lc.evolve_input_structs(USE_TS_FLUCT=True)
    bt = ox.BrightnessTemp.new(redshift=0.0, inputs=inputs)
    assert "tau_21" in bt.arrays


@pytest.mark.parametrize(
    ("struct", "param"),
    [
        pytest.param(kls, param, id=f"{name}-{param}")
        for name, kls in ox._ALL_OUTPUT_STRUCTS.items()
        for param in ox._struct_params(kls.compute.__wrapped__)
    ],
)
def test_compute_rejects_none_struct_inputs(struct: type[ox.OutputStruct], param: str):
    """Test that every compute() raises a clear error if a struct input is None."""
    with pytest.raises(
        ValueError, match=rf"Cannot compute {struct.__name__} .* {param} \("
    ):
        struct.dummy().compute(**{param: None})


@pytest.mark.parametrize("struct", list(ox.OutputStructZ.__subclasses__()))
def test_bad_required_array(default_input_struct, struct):
    # no struct takes this input
    inputs = default_input_struct.evolve_input_structs(SOURCE_MODEL="CHMF-SAMPLER")
    bt = ox.BrightnessTemp.new(redshift=10.0, inputs=inputs)
    kwargs = {"inputs": inputs, "redshift": 10.0}
    if struct is ox.PerturbedHaloCatalog:
        kwargs["buffer_size"] = 1
    output = struct.new(**kwargs)

    with pytest.raises((ValueError, TypeError), match="is not an input required for"):
        _ = output.get_required_input_arrays(bt)


def test_halocatalog_iteration(halo_cat: ox.HaloCatalog):
    """Test HaloCatalog iteration, len, and indexing."""
    # Test len
    assert len(halo_cat) == halo_cat.n_halos

    # Test iteration and indexing
    halo_list = []
    for halo in halo_cat:
        halo_list.append(halo)
        assert isinstance(halo, ox.Halo)
        assert halo.mass is not None
        assert halo.coords is not None
        assert halo.star_rng is not None
        assert halo.sfr_rng is not None
        assert halo.xray_rng is not None
        assert halo.redshift is not None

    assert len(halo_list) == halo_cat.n_halos

    # Test indexing
    first_halo = halo_cat[0]
    assert isinstance(first_halo, ox.Halo)
    assert first_halo.mass == halo_list[0].mass
    assert np.all(first_halo.coords == halo_list[0].coords)


def test_perturbed_halocatalog_iteration(pert_halo_cat: ox.PerturbedHaloCatalog):
    """Test PerturbedHaloCatalog iteration, len, and indexing."""
    # Test len
    assert len(pert_halo_cat) == pert_halo_cat.n_halos

    # Test iteration and indexing
    halo_list = []
    for halo in pert_halo_cat:
        halo_list.append(halo)
        assert isinstance(halo, ox.Halo)
        assert halo.mass is not None
        assert halo.coords is not None
        assert halo.redshift is not None

    assert len(halo_list) == pert_halo_cat.n_halos

    # Test indexing
    first_halo = pert_halo_cat[0]
    assert isinstance(first_halo, ox.Halo)
    assert first_halo.mass == halo_list[0].mass
    assert np.all(first_halo.coords == halo_list[0].coords)


def test_bad_indices_for_halocatalog(
    halo_cat: ox.HaloCatalog, pert_halo_cat: ox.PerturbedHaloCatalog
):
    """Test bad indices for halo catalog and perturbed halo catalog."""
    with pytest.raises(IndexError, match=f"Halo index {halo_cat.n_halos} out of range"):
        halo_cat[halo_cat.n_halos]
    with pytest.raises(IndexError, match="Halo index -1 out of range"):
        halo_cat[-1]
    with pytest.raises(
        IndexError, match=f"Halo index {pert_halo_cat.n_halos} out of range"
    ):
        pert_halo_cat[pert_halo_cat.n_halos]
    with pytest.raises(IndexError, match="Halo index -1 out of range"):
        pert_halo_cat[-1]
