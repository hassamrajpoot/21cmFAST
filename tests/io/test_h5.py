"""Tests of the HDF5 read/write functionality."""

from pathlib import Path

import h5py
import numpy as np
import pytest

import py21cmfast as p21c
from py21cmfast import InputParameters
from py21cmfast._templates import list_templates
from py21cmfast.io import h5
from py21cmfast.wrapper import outputs as ox


class TestHDF5ToDict:
    """Tests of the hdf5_to_dict function."""

    def test_empty(self, tmp_path: Path):
        """Test reading/writing an empty dict."""
        pth = tmp_path / "tmp.h5"
        with h5py.File(pth, "w") as fl:
            fl.create_group("group")

        with h5py.File(pth, "r") as fl:
            out = h5.hdf5_to_dict(fl["group"])

        assert out == {}

    def test_only_attrs(self, tmp_path: Path):
        """Test reading only from  the attrs."""
        pth = tmp_path / "tmp.h5"
        dct = {
            "int": 1,
            "str": "hi",
            "float": 3.14,
        }

        with h5py.File(pth, "w") as fl:
            for k, v in dct.items():
                fl.attrs[k] = v

        with h5py.File(pth, "r") as fl:
            out = h5.hdf5_to_dict(fl)

        assert out == dct

    def test_with_dataset(self, tmp_path):
        """Test a case with a dataset, not only attrs."""
        pth = tmp_path / "tmp.h5"

        with h5py.File(pth, "w") as fl:
            fl["dataset"] = np.zeros(10)

        with h5py.File(pth, "r") as fl:
            out = h5.hdf5_to_dict(fl)

        assert np.allclose(out["dataset"], np.zeros(10))

    def test_recursive(self, tmp_path: Path):
        """Test a recursive set of groups and datasets."""
        pth = tmp_path / "tmp.h5"
        with h5py.File(pth, "w") as fl:
            fl["dataset1"] = np.zeros(10)
            grp = fl.create_group("group")
            grp["dataset2"] = np.ones(10)

        with h5py.File(pth, "r") as fl:
            out = h5.hdf5_to_dict(fl)

        assert np.allclose(out["dataset1"], np.zeros(10))
        assert np.allclose(out["group"]["dataset2"], np.ones(10))


_ALL_TEMPLATE_NAMES = [t["name"] for t in list_templates()]


# Marks for the advisories that some templates trigger (to be removed with #778).
def _h5_roundtrip_marks(model, size):
    marks = []
    if model == "Qin20":
        marks.append(
            pytest.mark.filterwarnings(
                "ignore:^USE_MCGS needs a non-trivial V_CB_MODEL:UserWarning"
            )
        )
    if model in ("const-zeta", "Munoz21"):
        marks.append(
            pytest.mark.filterwarnings(
                "ignore:^Your model .*uses the EPS conditional mass function:UserWarning"
            )
        )
    if model in (
        "latest",
        "minihalos",
        "latest-discrete",
        "minihalos-discrete",
        "Park19",
        "fixed-halos",
        "Qin20",
        "Munoz21",
    ) and size in ("size-tiny", "size-small"):
        marks.append(
            pytest.mark.filterwarnings(
                "ignore:^You are setting R_BUBBLE_MAX != 50 when RECOMB_MODEL:UserWarning"
            )
        )
    return marks


class TestInputsIO:
    """Tests of reading and writing InputParameters to HDF5."""

    def setup_class(self):
        """Set up the class."""
        self.defaults = InputParameters(random_seed=0)

    def test_appending(self, tmp_path: Path):
        """Test that writing to an existing file doesn't wipe out existing info."""
        pth = tmp_path / "tmp.h5"

        with h5py.File(pth, "w") as fl:
            fl["sentinel"] = 1
            h5._write_inputs_to_group(self.defaults, pth)
            assert "sentinel" in fl

    def test_not_closing_group(self, tmp_path):
        """Test that reading from an open Hdpy.File doesn't close it."""
        pth = tmp_path / "tmp.h5"

        with h5py.File(pth, "w") as fl:
            # Make sure this is not overwritten
            fl["sentinel"] = 2

            h5._write_inputs_to_group(self.defaults, fl)
            fl["new"] = 1  # can write because it's not closed.

        with h5py.File(pth, "r") as fl:
            assert "new" in fl
            assert "InputParameters" in fl
            assert "sentinel" in fl

    @pytest.mark.parametrize(
        "inputs",
        [
            pytest.param((t, s), marks=_h5_roundtrip_marks(t, s))
            for t in _ALL_TEMPLATE_NAMES
            if not t.startswith("size-")
            for s in _ALL_TEMPLATE_NAMES
            if s.startswith("size-")
        ],
    )
    def test_roundtrip(
        self, tmp_path: Path, inputs: InputParameters | str | tuple[str]
    ):
        """Test that writing inputs to file then reading them again gets the same answer."""
        if not isinstance(inputs, InputParameters):
            inputs = InputParameters.from_template(inputs, random_seed=0)

        pth = tmp_path / "tmp.h5"
        h5._write_inputs_to_group(inputs, pth)
        new = h5.read_inputs(pth)
        assert new == inputs

    def test_roundtrip_without_precomputed_cosmo_tables(self, tmp_path, monkeypatch):
        """Cosmo tables are omitted in cache files when not precomputed."""
        inputs = InputParameters(random_seed=0).evolve_input_structs(
            POWER_SPECTRUM="CLASS"
        )
        pth = tmp_path / "tmp.h5"
        h5._write_inputs_to_group(inputs, pth)

        with h5py.File(pth, "r") as fl:
            assert "cosmo_tables" not in fl["InputParameters"]

        new = h5.read_inputs(pth)
        with pytest.raises(AttributeError, match="has no attribute 'cosmo_tables'"):
            object.__getattribute__(new, "cosmo_tables")

        def _raise_recompute(*args, **kwargs):
            raise RuntimeError("recompute")

        monkeypatch.setattr("py21cmfast.wrapper.inputs.run_classy", _raise_recompute)
        # Access after deserialization without cached tables should invoke computation.
        with pytest.raises(RuntimeError, match="recompute"):
            _ = new.cosmo_tables

    def test_roundtrip_with_precomputed_cosmo_tables(self, tmp_path, monkeypatch):
        """Cached cosmo tables in file are reused on read."""
        inputs = InputParameters(random_seed=0).evolve_input_structs(
            POWER_SPECTRUM="CLASS", K_MAX_FOR_CLASS=1.0
        )
        _ = inputs.cosmo_tables

        pth = tmp_path / "tmp.h5"
        h5._write_inputs_to_group(inputs, pth)

        with h5py.File(pth, "r") as fl:
            assert "cosmo_tables" in fl["InputParameters"]

        new = h5.read_inputs(pth)
        assert object.__getattribute__(new, "cosmo_tables") is not None

        def _raise_recompute(*args, **kwargs):
            raise RuntimeError("recompute")

        monkeypatch.setattr("py21cmfast.wrapper.inputs.run_classy", _raise_recompute)
        assert new.cosmo_tables == inputs.cosmo_tables


@pytest.fixture
def metadata_cache(
    default_input_struct_lc: InputParameters, tmp_path: Path
) -> tuple[ox.IonizedBox, Path]:
    """A small real cache with dimensional, scaled, and dimensionless fields."""
    inputs = default_input_struct_lc.evolve_input_structs(HII_DIM=2, DIM=4)
    box = ox.IonizedBox.new(inputs=inputs, redshift=10.0)
    for name, array in box.arrays.items():
        setattr(box, name, array.with_value(np.full(array.shape, 2, dtype=array.dtype)))
    for name in box._struct.primitive_fields:
        if getattr(box, name) is None:
            setattr(box, name, 0.0)
    path = tmp_path / "output.h5"
    h5.write_output_to_hdf5(box, path)
    return box, path


def _stamp_pre_metadata_version(file: h5py.File, box: ox.OutputStruct) -> None:
    """Identify files written before array metadata was introduced."""
    for name in ("OutputFields", "InputParameters"):
        file[box._name][name].attrs["21cmFAST-version"] = "4.3.0"


@pytest.mark.filterwarnings(
    "ignore::py21cmfast.io._compat_framework.FormatDeprecationWarning"
)
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("cache_on_access", [False, True])
def test_output_metadata_roundtrip(
    metadata_cache: tuple[ox.IonizedBox, Path], legacy: bool, cache_on_access: bool
) -> None:
    box, path = metadata_cache
    with h5py.File(path, "r+") as file:
        for name, dataset in file[box._name]["OutputFields"].items():
            assert dataset.attrs["unit"] == box.arrays[name].unit.to_string()
            assert dataset.attrs["description"] == box.arrays[name].description
            if legacy:
                del dataset.attrs["unit"]
                del dataset.attrs["description"]
        if legacy:
            _stamp_pre_metadata_version(file, box)
    restored = h5.read_output_struct(path)
    for name, array in restored.arrays.items():
        assert array.unit == box.arrays[name].unit
        assert array.description == box.arrays[name].description
        assert array._value is None
        assert array.state.on_disk
        with p21c.config.use(CACHE_ARRAYS_ON_ACCESS=cache_on_access):
            quantity = array.get_quantity()
            np.testing.assert_array_equal(quantity.value, 2)
            assert quantity.dtype == box.arrays[name].dtype
        assert array.state.computed_in_mem == cache_on_access


@pytest.mark.parametrize(
    ("stored_unit", "stored_value", "native_value"),
    [
        ("1 / s", 2e-12, 2.0),
        ("1e-12 s-1", 2.0, 2.0),
        ("1e28 / s", 1e-35, 1e5),
    ],
    ids=["scaled", "equivalent-spelling", "float32-intermediate"],
)
def test_cache_metadata_converts_values_to_native_units(
    metadata_cache: tuple[ox.IonizedBox, Path],
    stored_unit: str,
    stored_value: float,
    native_value: float,
) -> None:
    box, path = metadata_cache
    with h5py.File(path, "r+") as file:
        dataset = file[box._name]["OutputFields"]["ionisation_rate_G12"]
        dataset[...] = stored_value
        dataset.attrs["unit"] = stored_unit
        dataset.attrs["description"] = "Stored photoionization rate."
    restored = h5.read_output_struct(path)
    array = restored.arrays["ionisation_rate_G12"]
    assert array.unit == box.arrays["ionisation_rate_G12"].unit
    assert array.description == "Stored photoionization rate."
    if stored_unit == "1e-12 s-1":
        assert array.state.on_disk
        assert array._value is None
        assert array.cache_backend is not None
    assert restored.ionisation_rate_G12.dtype == array.dtype
    assert np.isfinite(restored.ionisation_rate_G12).all()
    np.testing.assert_allclose(
        restored.ionisation_rate_G12, native_value, rtol=1e-6, atol=0
    )


@pytest.mark.parametrize("spectral_equivalency", [False, True])
def test_cache_rejects_incompatible_unit_metadata(
    metadata_cache: tuple[ox.IonizedBox, Path], spectral_equivalency: bool
) -> None:
    from astropy import units as un

    box, path = metadata_cache
    with h5py.File(path, "r+") as file:
        dataset = file[box._name]["OutputFields"]["ionisation_rate_G12"]
        dataset[...] = 500
        dataset.attrs["unit"] = "nm"
    equivalencies = un.spectral() if spectral_equivalency else []
    with (
        un.set_enabled_equivalencies(equivalencies),
        pytest.raises(un.UnitConversionError),
    ):
        h5.read_output_struct(path)


@pytest.mark.parametrize("existing_file", [False, True])
def test_cache_writer_rejects_unknown_legacy_metadata(
    metadata_cache: tuple[ox.IonizedBox, Path], tmp_path: Path, existing_file: bool
) -> None:
    from py21cmfast.wrapper.arrays import Array

    box, _ = metadata_cache
    state = dict(box.arrays["neutral_fraction"].__dict__)
    state.pop("unit")
    legacy = object.__new__(Array)
    legacy.__setstate__(state)
    box.neutral_fraction = legacy
    target = metadata_cache[1] if existing_file else tmp_path / "unknown-metadata.h5"
    before = target.read_bytes() if existing_file else None
    with pytest.raises(ValueError, match="without unit and description"):
        h5.write_output_to_hdf5(box, target)
    if existing_file:
        assert target.read_bytes() == before
    else:
        assert not target.exists()


@pytest.mark.parametrize("source_version", ["4.4.0", "5.0.0"])
@pytest.mark.filterwarnings(
    "ignore::py21cmfast.io._compat_framework.FormatDeprecationWarning"
)
def test_cache_from_metadata_format_requires_metadata(
    metadata_cache: tuple[ox.IonizedBox, Path],
    monkeypatch: pytest.MonkeyPatch,
    source_version: str,
) -> None:
    from packaging.version import Version

    from py21cmfast.io import _compat
    from py21cmfast.io import _compat_framework as framework

    box, path = metadata_cache
    with h5py.File(path, "r+") as file:
        fields = file[box._name]["OutputFields"]
        fields.attrs["21cmFAST-version"] = source_version
        file[box._name]["InputParameters"].attrs["21cmFAST-version"] = source_version
        del fields["neutral_fraction"].attrs["unit"]
    monkeypatch.setattr(framework, "CURRENT_VERSION", Version("5.0.0"))
    monkeypatch.setattr(
        _compat,
        "FORMAT_HISTORY",
        (*_compat.FORMAT_HISTORY, framework.FormatRelease("4.4.0", "4.5.0", [])),
    )
    with pytest.raises(ValueError, match="missing unit or description"):
        h5.read_output_struct(path)


@pytest.mark.filterwarnings(
    "ignore::py21cmfast.io._compat_framework.FormatDeprecationWarning"
)
def test_migration_writes_recovered_array_metadata(
    metadata_cache: tuple[ox.IonizedBox, Path], tmp_path: Path
) -> None:
    from py21cmfast.io import migrate

    box, path = metadata_cache
    with h5py.File(path, "r+") as file:
        _stamp_pre_metadata_version(file, box)
        for dataset in file[box._name]["OutputFields"].values():
            del dataset.attrs["unit"]
            del dataset.attrs["description"]
    cache = p21c.OutputCache(tmp_path / "migrated")
    result = migrate.migrate_cache_file(path, cache)
    assert result.status == "migrated", result.message
    with h5py.File(result.destination, "r") as file:
        for name, dataset in file[box._name]["OutputFields"].items():
            assert dataset.attrs["unit"] == box.arrays[name].unit.to_string()
            assert dataset.attrs["description"] == box.arrays[name].description
            np.testing.assert_array_equal(dataset[...], 2)

    assert migrate.migrate_cache_file(result.destination, cache).status == "up-to-date"
    with h5py.File(result.destination, "r+") as file:
        _stamp_pre_metadata_version(file, box)
        for dataset in file[box._name]["OutputFields"].values():
            del dataset.attrs["unit"]
            del dataset.attrs["description"]
    before = result.destination.read_bytes()
    same_location = migrate.migrate_cache_file(result.destination, cache)
    assert same_location.status == "exists"
    assert "different destination" in same_location.message
    assert result.destination.read_bytes() == before


class TestLoadHighLevelSimulation:
    """Tests of the `load_high_level_simulation` convenience reader."""

    def test_lightcone(self, lc, tmp_path: Path):
        """A lightcone file should come back as a LightCone."""
        pth = tmp_path / "lightcone.h5"
        lc.save(pth)

        assert isinstance(h5.load_high_level_simulation(pth), p21c.LightCone)

    def test_global_evolution(self, default_input_struct_lc, tmp_path: Path):
        """A global-evolution file should come back as a GlobalEvolution."""
        pth = tmp_path / "global.h5"
        p21c.GlobalEvolution(
            inputs=default_input_struct_lc,
            quantities={
                "brightness_temp": np.zeros(len(default_input_struct_lc.node_redshifts))
            },
        ).save(pth)

        assert isinstance(h5.load_high_level_simulation(pth), p21c.GlobalEvolution)

    def test_unrecognized_file(self, tmp_path: Path):
        """Anything else should raise a helpful error."""
        pth = tmp_path / "not-a-simulation.h5"
        with h5py.File(pth, "w") as fl:
            fl.attrs["something_else"] = True

        with pytest.raises(ValueError, match="not a recognized 21cmFAST output"):
            h5.load_high_level_simulation(pth)
