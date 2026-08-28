"""
Compute single physical fields.

These functions are high-level wrappers around C-functions that compute 3D fields, for
example initial conditions, perturbed fields and ionization fields.
"""

import logging
import warnings

import numpy as np
from astropy import constants
from astropy import units as un
from astropy.cosmology import z_at_value

from ..wrapper.arrays import Array
from ..wrapper.inputs import InputParameters
from ..wrapper.outputs import (
    BrightnessTemp,
    HaloBox,
    HaloCatalog,
    InitialConditions,
    IonizedBox,
    PerturbedField,
    PerturbedHaloCatalog,
    RadiationFields,
    TsBox,
)
from ._global_initialization import init_c_state
from ._param_config import (
    check_output_consistency,
    single_field_func,
)

logger = logging.getLogger(__name__)

update_rad_fields_mode = {
    "setup": 0,
    "eval": 1,
    "cleanup": 2,
}


@single_field_func
@init_c_state(ps=True)
def compute_initial_conditions(
    *,
    inputs: InputParameters,
    initial_density: np.ndarray | float | None = None,
) -> InitialConditions:
    r"""
    Compute initial conditions.

    Parameters
    ----------
    inputs
        The InputParameters instance defining the run.
    initial_density: np.ndarray or float, optional
        A realization of the density field on the high resolution grid.
        This input can also be used to determine the global density field,
        in case we have a single cell in the box.
    regenerate : bool, optional
        Whether to force regeneration of data, even if matching cached data is found.
    cache
        An OutputCache object defining how to read cached boxes.
    write
        A boolean specifying whether we need to cache the box.

    Returns
    -------
    :class:`~InitialConditions`
    """
    # Initialize memory for the boxes that will be returned.
    ics = InitialConditions.new(inputs=inputs)

    if inputs.simulation_options.HII_DIM == 1 and inputs.simulation_options.DIM == 1:
        # If we have only one cell (could happen if we do a global evolution), we don't really need to compute the ics
        shape = (1, 1, 1)
        required_arrays = PerturbedField.new(
            redshift=0, inputs=inputs
        ).get_required_input_arrays(ics)

        # Set the arrays to zero, or according to initial_density if the arrays are density fields
        for array in required_arrays:
            if (initial_density is not None) and (
                array in ["hires_density", "lowres_density"]
            ):
                value = initial_density
            else:
                value = 0.0
            setattr(
                ics,
                array,
                Array(shape=shape, dtype=np.float32)
                .initialize()
                .with_value(val=value * np.ones(shape)),
            )
        return ics
    else:
        if initial_density is not None:
            if np.abs(initial_density.mean() > 1e-3):
                warnings.warn(
                    f"Your initial_density has mean {initial_density.mean()}. "
                    + "Make sure you know what you are doing.",
                    stacklevel=2,
                )
            shape = ics.hires_density.shape
            if initial_density.shape != shape:
                raise ValueError(
                    "The shape of your high resolution initial_density is not consistent with inputs!"
                    + f" According to inputs, initial_density must be of shape {shape}, got {initial_density.shape}."
                )

            ics.hires_density = (
                Array(shape=shape, dtype=np.float32)
                .initialize()
                ._with_value_not_computed(val=initial_density)
            )
        return ics.compute()


@single_field_func
@init_c_state(broadcast_inputs=True)
def perturb_field(
    *,
    redshift: float,
    inputs: InputParameters | None = None,
    initial_conditions: InitialConditions,
) -> PerturbedField:
    r"""
    Compute a perturbed field at a given redshift.

    Parameters
    ----------
    redshift : float
        The redshift at which to compute the perturbed field.
    initial_conditions : :class:`~InitialConditions` instance
        The initial conditions.

    Returns
    -------
    :class:`~PerturbedField`

    Other Parameters
    ----------------
    regenerate, write, cache:
        See docs of :func:`initial_conditions` for more information.

    Examples
    --------
    >>> initial_conditions = compute_initial_conditions()
    >>> field7 = perturb_field(7.0, initial_conditions)
    >>> field8 = perturb_field(8.0, initial_conditions)

    The user and cosmo parameter structures are by default inferred from the
    ``initial_conditions``.
    """
    # Initialize perturbed boxes.
    fields = PerturbedField.new(redshift=redshift, inputs=inputs)

    # Run the C Code
    return fields.compute(ics=initial_conditions)


@single_field_func
@init_c_state(sigma=True)
def determine_halo_catalog(
    *,
    redshift: float,
    inputs: InputParameters | None = None,
    initial_conditions: InitialConditions,
    descendant_halos: HaloCatalog | None = None,
) -> HaloCatalog:
    r"""
    Find a halo list, given a redshift.

    Parameters
    ----------
    redshift : float
        The redshift at which to determine the halo list.
    initial_conditions : :class:`~InitialConditions` instance
        The initial conditions fields (density, velocity).
    descendant_halos : :class:`~HaloCatalog` instance, optional
        The halos that form the descendants (i.e. lower redshift) of those computed by
        this function. If this is not provided, we generate the initial stochastic halos
        directly in this function (and progenitors can then be determined by these).

    Returns
    -------
    :class:`~HaloCatalog`

    Other Parameters
    ----------------
    regenerate, write, cache:
        See docs of :func:`initial_conditions` for more information.
    """
    if inputs.matter_options.HMF != "ST":
        warnings.warn(
            "DexM Halofinder sses a fit to the Sheth-Tormen mass function."
            "With HMF!=1 the Halos from DexM will not be from the same mass function",
            stacklevel=2,
        )

    if descendant_halos is None:
        descendant_halos = HaloCatalog.dummy()

    # Initialize halo list boxes.
    halo_catalog = HaloCatalog.new(
        redshift=redshift,
        desc_redshift=descendant_halos.redshift,
        inputs=inputs,
    )

    # Run the C Code
    halo_catalog.compute(
        ics=initial_conditions,
        descendant_halos=descendant_halos,
    )

    halo_catalog.trim_to_n_halos()
    return halo_catalog


@single_field_func
@init_c_state(broadcast_inputs=True)
def perturb_halo_catalog(
    *,
    initial_conditions: InitialConditions,
    inputs: InputParameters | None = None,
    previous_spin_temp: TsBox | None = None,
    previous_ionize_box: IonizedBox | None = None,
    halo_catalog: HaloCatalog,
) -> PerturbedHaloCatalog:
    r"""
    Given a halo list, perturb the halos for a given redshift.

    Parameters
    ----------
    initial_conditions : :class:`~InitialConditions`
        The initial conditions of the run. The user and cosmo params
        as well as the random seed will be set from this object.
    halo_catalog: :class: `~HaloCatalog`
        The halo catalogue in Lagrangian space to be perturbed.

    Returns
    -------
    :class:`~PerturbedHaloCatalog`

    Other Parameters
    ----------------
    regenerate, write, direc:
        See docs of :func:`initial_conditions` for more information.

    Examples
    --------
    Fill this in once finalised

    """
    hbuffer_size = (
        halo_catalog.n_halos if halo_catalog.n_halos else halo_catalog.buffer_size
    )
    redshift = halo_catalog.redshift

    # Initialize halo list boxes.
    fields = PerturbedHaloCatalog.new(
        redshift=redshift,
        buffer_size=hbuffer_size,
        inputs=inputs,
    )
    if previous_spin_temp is None:
        if (
            redshift >= inputs.simulation_options.Z_HEAT_MAX
            or not inputs.astro_options.USE_MINI_HALOS
        ):
            # Dummy spin temp is OK since we're above Z_HEAT_MAX
            previous_spin_temp = TsBox.dummy()
        else:
            raise ValueError("Below Z_HEAT_MAX you must specify the previous_spin_temp")

    if previous_ionize_box is None:
        if (
            redshift >= inputs.simulation_options.Z_HEAT_MAX
            or not inputs.astro_options.USE_MINI_HALOS
        ):
            # Dummy ionize box is OK since we're above Z_HEAT_MAX
            previous_ionize_box = IonizedBox.dummy()
        else:
            raise ValueError(
                "Below Z_HEAT_MAX you must specify the previous_ionize_box"
            )

    # Run the C Code
    return fields.compute(
        ics=initial_conditions,
        halo_catalog=halo_catalog,
        previous_spin_temp=previous_spin_temp,
        previous_ionize_box=previous_ionize_box,
    )


@single_field_func
@init_c_state(sigma=True)
def compute_halo_grid(
    *,
    redshift: float,
    initial_conditions: InitialConditions | None = None,
    inputs: InputParameters | None = None,
    halo_catalog: HaloCatalog | None = None,
    perturbed_field: PerturbedField | None = None,
    previous_spin_temp: TsBox | None = None,
    previous_ionize_box: IonizedBox | None = None,
) -> HaloBox:
    r"""
    Compute grids of halo properties from a catalogue.

    At the moment this simply produces halo masses, stellar masses and SFR on a grid of
    HII_DIM. In the future this will compute properties such as emissivities which will
    be passed directly into ionize_box etc. instead of the catalogue.

    Parameters
    ----------
    initial_conditions : :class:`~InitialConditions`, optional
        The initial conditions of the run. Becomes relevant only for Lagrangian source models,
        or alternatively, if the user uses a Eulerian source model with USE_MINI_HALOS==True
        and V_CB_MODEL = "FLUCTS".
    inputs : :class:`~InputParameters`, optional
        The input parameters specifying the run.
    halo_catalog: :class:`~HaloCatalog`, optional
        This contains all the dark matter haloes obtained if using a discrete halo model.
        This is a list of halo masses and coordinates for the dark matter halos.
    perturbed_field: :class:`~PerturbedField`, optional
        The perturbed field at the current redshift. Becomes relevant only for Eulerian source models.
    previous_spin_temp : :class:`TsBox`, optional
        The previous spin temperature box. Used for feedback when USE_MINI_HALOS==True
    previous_ionize_box: :class:`IonizedBox` or None
        An at the last timestep. Used for feedback when USE_MINI_HALOS==True

    Returns
    -------
    :class:`~HaloBox` :
        An object containing the halo box data.

    Other Parameters
    ----------------
    regenerate, write, cache:
        See docs of :func:`initial_conditions` for more information.
    """
    box = HaloBox.new(redshift=redshift, inputs=inputs)

    if halo_catalog is None:
        if inputs.matter_options.has_discrete_halos:
            raise ValueError(
                f"You must provide halo_catalog for SOURCE_MODEL = {inputs.matter_options.SOURCE_MODEL}"
            )
        else:
            halo_catalog = HaloCatalog.dummy()

    if initial_conditions is None:
        if inputs.matter_options.lagrangian_source_grid:
            raise ValueError(
                f"You must provide initial_conditions for SOURCE_MODEL = {inputs.matter_options.SOURCE_MODEL}"
            )
        elif (
            inputs.matter_options.SOURCE_MODEL == "E-INTEGRAL"
            and inputs.astro_options.USE_MINI_HALOS
            and inputs.matter_options.V_CB_MODEL == "FLUCTS"
        ):
            raise ValueError(
                "You must provide initial_conditions for SOURCE_MODEL = E- INTEGRAL, "
                "USE_MINI_HALOS = True and V_CB_MODEL = FLUCTS"
            )
        else:
            initial_conditions = InitialConditions.dummy()

    if perturbed_field is None:
        if not inputs.matter_options.lagrangian_source_grid:
            raise ValueError(
                f"You must provide perturbed_field for SOURCE_MODEL = {inputs.matter_options.SOURCE_MODEL}"
            )
        else:
            perturbed_field = PerturbedField.dummy()

    # NOTE: due to the order, we use the previous spin temp here, like spin_temperature,
    #       but UNLIKE ionize_box, which uses the current box
    # TODO: think about the inconsistency here
    # NOTE: if USE_MINI_HALOS is TRUE, so is USE_TS_FLUCT and RECOMB_MODEL != "none"
    if previous_spin_temp is None:
        if (
            redshift >= inputs.simulation_options.Z_HEAT_MAX
            or not inputs.astro_options.USE_MINI_HALOS
        ):
            # Dummy spin temp is OK since we're above Z_HEAT_MAX
            previous_spin_temp = TsBox.dummy()
        else:
            raise ValueError("Below Z_HEAT_MAX you must specify the previous_spin_temp")

    if previous_ionize_box is None:
        if (
            redshift >= inputs.simulation_options.Z_HEAT_MAX
            or not inputs.astro_options.USE_MINI_HALOS
        ):
            # Dummy ionize box is OK since we're above Z_HEAT_MAX
            previous_ionize_box = IonizedBox.dummy()
        else:
            raise ValueError(
                "Below Z_HEAT_MAX you must specify the previous_ionize_box"
            )

    return box.compute(
        initial_conditions=initial_conditions,
        perturbed_field=perturbed_field,
        halo_catalog=halo_catalog,
        previous_ionize_box=previous_ionize_box,
        previous_spin_temp=previous_spin_temp,
    )


# TODO: make this more general and probably combine with the lightcone interp function
# TODO: remove the need_c argument, this is currently required because we call this function once for just computing the history of
# log10_Mcrit_MCG_ave - see other comment about this in need_c.
def interp_halo_boxes(
    halo_boxes: list[HaloBox],
    interp_fields: list[str],
    redshift: float,
    need_c: bool,
) -> HaloBox:
    """
    Interpolate HaloBox history to the desired redshift.

    Photon conservation & Xray sources require halo boxes at redshifts
    that are not equal to the current redshift, and may be between redshift steps.
    So we need a function to interpolate between two halo boxes.
    We assume here that z_arr is strictly INCERASING

    Parameters
    ----------
    halo_boxes : list of HaloBox instances
        The halobox history to be interpolated
    interp_fields: list[str]
        The properties of the haloboxes to be interpolated
    redshift : float
        The desired redshift of interpolation
    need_c : bool
        Whether we need to compute the radiation field boxes in the C code or not.

    Returns
    -------
    :class:`~HaloBox` :
        An object containing the halo box data
    """
    inputs = halo_boxes[0].inputs
    z_halos = [box.redshift for box in halo_boxes]
    if not np.all(np.diff(z_halos) > 0):
        raise ValueError("halo_boxes must be in ascending order of redshift")

    if redshift > z_halos[-1] or redshift < z_halos[0]:
        raise ValueError(f"Invalid z_target {redshift} for redshift array {z_halos}")

    # If we do global evolution, no need to do that
    if inputs.simulation_options.HII_DIM > 1 and need_c:
        arr_fields = [f for f in interp_fields if f in halo_boxes[0].arrays]
        computed = [box.ensure_arrays_computed(*arr_fields) for box in halo_boxes]
        if not all(computed):
            raise ValueError("Some of the HaloBox fields required are not computed")

    idx_prog = np.searchsorted(z_halos, redshift, side="left")

    if idx_prog == 0 or idx_prog == len(z_halos):
        logger.debug(f"redshift {redshift} beyond limits, {z_halos[0], z_halos[-1]}")
        raise ValueError

    z_prog = z_halos[idx_prog]
    idx_desc = idx_prog - 1

    z_desc = z_halos[idx_desc]
    interp_param = (redshift - z_desc) / (z_prog - z_desc)

    # If we do global evolution, no need to do that
    if inputs.simulation_options.HII_DIM > 1 and need_c:
        # I set the box redshift to be the stored one so it is read properly into the ionize box
        # for the xray source it doesn't matter, also since it is not _compute()'d, it won't be cached
        check_output_consistency(
            dict(
                zip(
                    [f"box-{i}" for i in range(len(halo_boxes))],
                    halo_boxes,
                    strict=True,
                )
            )
        )
    hbox_out = HaloBox.new(redshift=redshift, inputs=inputs)

    # initialise the memory
    if need_c:
        hbox_out._init_arrays()

    # interpolate halo boxes in gridded SFR
    hbox_prog = halo_boxes[idx_prog]
    hbox_desc = halo_boxes[idx_desc]

    for field in interp_fields:
        field_desc = hbox_desc.get(field)
        field_prog = hbox_prog.get(field)
        interp_field = np.zeros_like(field_desc)
        interp_field[...] = (1 - interp_param) * field_desc + interp_param * field_prog
        hbox_out.set(field, interp_field)

    return hbox_out


# TODO: I gave this function an initializer decorator since it can call the C code, and if some curious users will try to
# call it directly they will run into segfaul. Also here, I set the argument to be sigma=True, but this is an "overkill"
# and should be set to broadcast_inputs=True once https://github.com/21cmfast/21cmFAST/issues/668 is fixed.
# TODO: perturbed_field and initial_conditions should be removed from the function's signature once
# https://github.com/21cmfast/21cmFAST/issues/668 is fixed.
@init_c_state(sigma=True)
def _interpolate_and_evaluate_radiation_fields(
    redshift: float,
    inputs: InputParameters,
    hboxes: list[HaloBox],
    interp_fields: list[str],
    R_range: np.ndarray,
    zpp_avg: np.ndarray,
    z_max: float,
    need_c: bool,
    radiation_fields: RadiationFields,
    perturbed_field: PerturbedField | None = None,
    previous_spin_temp: TsBox | None = None,
    initial_conditions: InitialConditions | None = None,
    cleanup: bool | None = None,
    R_star: float | None = None,
) -> RadiationFields:
    """
    Interpolate the halo boxes and evaluate the radiation fields.

    Parameters
    ----------
    redshift: float
        The redshift at which to evaluate the radiation fields.
    inputs: InputParameters
        The input parameters specifying the run.
    hboxes: list of HaloBox
        The halo boxes to interpolate.
    interp_fields : list[str]
        The fields to interpolate and evaluate.
    R_range: np.ndarray
        The range of shell's radii to be integrated in order to evaluate the radiation fields.
    zpp_avg: np.ndarray
        The average redshifts for each shell.
    z_max: float
        The maximum redshift for the integration.
        If a shell crosses this redshift, its contribution to the radiation fields is ignored.
    need_c: bool
        Whether to evaluate the 3D boxes via the C code.
    previous_spin_temp: :class:`~TsBox`
        The spin temperature box at the previous redshift. Becomes relevant only when redshift < Z_HEAT_MAX.
    radiation_fields: :class:`~RadiationFields`
        An object containing the radiation fields, before they had been computed.
    R_star: float | None, optional
        The comoving diffusion scale in the case of Lyman alpha multiple scattering.
        Becomes relevant only when `LYA_MULTIPLE_SCATTERING` and `need_c` are set to True.

    Returns
    -------
    :class:`~RadiationFields` :
        An object containing the radiation fields, after they had been computed.
    """
    for i in range(inputs.astro_params.N_STEP_TS):
        R_inner = R_range[i - 1].to("Mpc").value if i > 0 else 0
        R_outer = R_range[i].to("Mpc").value

        if zpp_avg[i] >= z_max:
            radiation_fields.filtered_sfr.value[...] = 0
            radiation_fields.filtered_xray.value[...] = 0
            if inputs.astro_options.USE_MINI_HALOS:
                if not need_c:
                    # If the shell is beyond z_max, we compute the mean log10_Mcrit_MCG
                    # under the assumption of zero LW flux a constant v_cb, and no reionization feedback
                    from ..wrapper import cfuncs

                    mturn_MCG = cfuncs.get_molecular_cooling_threshold_with_feedbacks(
                        inputs=inputs,
                        redshifts=zpp_avg[i],
                        J_LW_21=0.0,
                        v_cb=inputs.cosmo_tables.V_CB_AVG,
                    )
                    radiation_fields.mean_log10_Mcrit_LW.value[i] = np.log10(
                        np.max([mturn_MCG, inputs.astro_params.M_TURN_STELLAR_FEEDBACK])
                    )
                radiation_fields.filtered_sfr_mini.value[...] = 0
                if inputs.astro_options.LYA_MULTIPLE_SCATTERING:
                    radiation_fields.filtered_sfr_lw.value[...] = 0
                    radiation_fields.filtered_sfr_mini_lw.value[...] = 0
            logger.debug(f"ignoring Radius {i} which is above Z_HEAT_MAX")
            continue

        hbox_interp = interp_halo_boxes(
            halo_boxes=hboxes[::-1],
            interp_fields=interp_fields,
            redshift=zpp_avg[i],
            need_c=need_c,
        )

        if need_c:
            radiation_fields = radiation_fields.compute(
                redshift=redshift,
                halobox=hbox_interp,
                R_inner=R_inner,
                R_outer=R_outer,
                R_ct=i,
                R_star=R_star.to("Mpc").value,
                perturbed_field=perturbed_field,
                previous_spin_temp=previous_spin_temp,
                initial_conditions=initial_conditions,
                mode=update_rad_fields_mode["eval"],
                cleanup=cleanup,
                allow_already_computed=True,
            )
        else:
            radiation_fields.mean_log10_Mcrit_LW.value[i] = (
                hbox_interp.log10_Mcrit_MCG_ave
            )

    return radiation_fields


# NOTE: the current implementation of this box is very hacky, since I have trouble figuring out a way to _compute()
#   over multiple redshifts in a nice way using this wrapper.
# TODO: if we move some code to jax or similar I think this would be one of the first candidates (just filling out some filtered grids)
# TODO: I changed the argument of the initializer below to sigma=True (instead of broadcast_inputs=True). This is because we now call
# setup_radiation_fields() in the C code that corresponds to this function. However, sigma is required only in the old Eulerian source models.
# This should be changed back once https://github.com/21cmfast/21cmFAST/issues/668 is fixed.
@single_field_func
@init_c_state(sigma=True)
def compute_radiation_fields(
    *,
    hboxes: list[HaloBox],
    redshift: float,
    previous_ionize_box: IonizedBox | None = None,
    perturbed_field: PerturbedField | None = None,
    previous_spin_temp: TsBox | None = None,
    initial_conditions: InitialConditions | None = None,
    cleanup: bool = True,
) -> RadiationFields:
    r"""
    Compute the radiation fields, given the past emissivity fields.

    This will filter over the emissivity history in annuli, computing the contribution to the
    radiation fields.

    Parameters
    ----------
    hboxes: Sequence of :class:`~HaloBox` instances
        This contains the list of Halobox instances which are used to create this source field
    previous_ionize_box: :class:`IonizedBox` or None
        An ionized box at higher redshift. This is only used if `LYA_MULTIPLE_SCATTERING` is true.
    previous_spin_temp: :class:`TsBox` or None
        The spin temperature box at the previous redshift. Becomes relevant only when redshift < Z_HEAT_MAX.

    Returns
    -------
    :class:`~RadiationFields` :
        An object containing x ray heating, ionisation, and lyman alpha rates.

    Other Parameters
    ----------------
    regenerate, write, cache:
        See docs of :func:`initial_conditions` for more information.
    """
    z_halos = [hb.redshift for hb in hboxes]
    inputs = hboxes[0].inputs

    # Initialize halo list boxes.
    radiation_fields = RadiationFields.new(redshift=redshift, inputs=inputs)

    # set minimum R at cell size
    l_factor = (4 * np.pi / 3.0) ** (-1 / 3)
    if inputs.simulation_options.HII_DIM == 1:
        # If HII_DIM=1 (happens when we run_global_evolution), we take a typical cell size of 1.5Mpc,
        # just to for setting the z'' array (note that filtering won't be done on a box with a single cell)
        R_min = 1.5 * l_factor
    else:
        R_min = (
            inputs.simulation_options.BOX_LEN
            / inputs.simulation_options.HII_DIM
            * l_factor
        )
    z_max = min(max(z_halos), inputs.simulation_options.Z_HEAT_MAX)

    # now we need to find the closest halo box to the redshift of the shell
    cosmo_ap = inputs.cosmo_params.cosmo
    cmd_zp = cosmo_ap.comoving_distance(redshift)
    R_steps = np.arange(0, inputs.astro_params.N_STEP_TS)
    R_factor = (inputs.astro_params.R_MAX_TS / R_min) ** (
        R_steps / inputs.astro_params.N_STEP_TS
    )
    R_range = un.Mpc * R_min * R_factor
    cmd_edges = cmd_zp + R_range  # comoving distance edges
    # Get the edges of the shells
    zmin = z_at_value(cosmo_ap.comoving_distance, cmd_edges.min()).value
    zmax = z_at_value(cosmo_ap.comoving_distance, cmd_edges.max()).value
    zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 100)
    dgrid = cosmo_ap.comoving_distance(zgrid)
    zpp_edges = np.interp(cmd_edges.value, dgrid.value, zgrid)
    # the `average` redshift of the shell is the average of the
    # inner and outer redshifts (following the C code)
    zpp_avg = zpp_edges - np.diff(np.insert(zpp_edges, 0, redshift)) / 2

    # Initialize the memory, since we may not go through the C code
    radiation_fields._init_arrays()

    # Let's figure out if we really need to go through the C code
    sfr_allzero = np.all([np.all(hbox.get("halo_sfr") == 0) for hbox in hboxes])
    lowest_shell_above_zmax = zpp_avg.min() >= z_max
    need_c = not (sfr_allzero or lowest_shell_above_zmax)

    if need_c:
        # Compute the comoving diffusion scale in the case of Lyman alpha multiple scattering
        if inputs.astro_options.LYA_MULTIPLE_SCATTERING:
            # TODO: In principle, the diffusion scale varies locally but for simplicty, we consider the global ionization value.
            # See https://github.com/21cmfast/21cmFAST/issues/606.
            if previous_ionize_box is None:
                x_HI = 1.0
            else:
                x_HI = previous_ionize_box.neutral_fraction.value.mean()
            A_alpha = 6.25e8 * un.Hz
            nu_Lya = 2.46606727e15 * un.Hz
            n_H_z0 = (
                (1.0 - inputs.cosmo_params.Y_He)
                * inputs.cosmo_params.cosmo.critical_density(0)
                * inputs.cosmo_params.OMb
                / constants.m_p
            )
            # Eq. (24) in arxiv: 2601.14360
            R_star = (
                3.0 * constants.c**4 * A_alpha**2 * n_H_z0 * x_HI * (1.0 + redshift)
            )
            R_star /= (
                32.0
                * np.pi**3
                * nu_Lya**4
                * inputs.cosmo_params.cosmo.H0**2
                * inputs.cosmo_params.OMm
            )
        else:
            R_star = 0.0 * un.Mpc

        interp_fields = ["halo_sfr", "halo_xray"]
        if inputs.astro_options.USE_MINI_HALOS:
            interp_fields += ["halo_sfr_mini"]

        # Get log10_Mcrit_MCG_ave for each shell
        # TODO: The reason why this field is evaluated separately is because it is already required in setup_radiation_fields() in the C code,
        # as it sets rad_setup->ave_log10_MturnLW. This array is mostly used for the old Eulerian source model calculations, which will be fixed
        # in the future (see https://github.com/21cmfast/21cmFAST/issues/668), but it is still needed for the Lagrangian source model as well
        # (specifically, in global_reion_properties) for two purposese:
        #   (1) For computing the global Nion, which is used for the NO_LIGHT condition. Here however, note that only the first entry of
        #       ave_log10_MturnLW is needed, namely for that computation we care about only the CURRENT global turnover mass, not its history.
        #       This usage therefore does not require the full global array as we compute below.
        #   (2) For computing the lower limit of the frequency integral (it is used in nu_tau_one_MINI, which is called by fill_freqint_tables).
        #       Here, we ought to have the full history of the global turnover mass, since that lower limit depends on the frequency in which
        #       the X-ray optical depth is unity. In order to compute the X-ray optical depth, we need to integrate over the history of the global
        #       neutral volume filling factor, which is currently approximated by the global Nion (the code does something like x_HI = 1 - Nion/(1-x_e)).
        #       Hence, the full history of the global turnover mass is needed in order to detemrine the history of x_HI, which is integrated in order to
        #       get the X-ray optical depth, which is required for setting the lower limit for the frequency integral. In that context, note that the code
        #       actually uses the global turnover mass at zpp (the redshift that corresponds to the shell), and not at zhat (the dummy integration variable
        #       in the X-ray optical depth integral), which I believe is a MISTAKE/undocmented approximation. Anyway, the necessity for the full history of
        #       the global turnover mass for setting the lower limit (as well evaluating the global turnover mass at zpp) should be fixed when addressing
        #       https://github.com/21cmfast/21cmFAST/issues/659, where the global x_HI at zpp is taken from its history, as was evaluated by the reionization
        #       code.
        if inputs.astro_options.USE_MINI_HALOS:
            radiation_fields = _interpolate_and_evaluate_radiation_fields(
                redshift=redshift,
                inputs=inputs,
                hboxes=hboxes,
                interp_fields=["log10_Mcrit_MCG_ave"],
                R_range=R_range,
                zpp_avg=zpp_avg,
                z_max=z_max,
                need_c=False,  # We don't want to use the C code at this stage
                radiation_fields=radiation_fields,
            )

        # Initialize the radiation fields in the C code (with mode="setup")
        # TODO: this is a bit hacky. A better solution would be to have RadiationFieldsSetup that we have in the C code as an
        #       OutputStructZ, but I suggeste to wait with this solution until https://github.com/21cmfast/21cmFAST/issues/668 is
        #       fixed, as RadiationFieldsSetup would become much simpler
        radiation_fields.compute(
            redshift=redshift,
            halobox=hboxes[0],  # dummy
            R_inner=0.0,  # dummy
            R_outer=0.0,  # dummy
            R_ct=0,  # dummy
            R_star=0.0,  # dummy
            perturbed_field=perturbed_field,
            previous_spin_temp=previous_spin_temp,
            initial_conditions=initial_conditions,
            mode=update_rad_fields_mode["setup"],
            cleanup=cleanup,
            allow_already_computed=True,
        )

        # Interpolate the halo boxes and evaluate the radiation fields
        radiation_fields = _interpolate_and_evaluate_radiation_fields(
            redshift=redshift,
            inputs=inputs,
            hboxes=hboxes,
            interp_fields=interp_fields,
            R_range=R_range,
            zpp_avg=zpp_avg,
            z_max=z_max,
            R_star=R_star,
            need_c=True,  # Now we call the C function
            perturbed_field=perturbed_field,
            previous_spin_temp=previous_spin_temp,
            initial_conditions=initial_conditions,
            cleanup=cleanup,
            radiation_fields=radiation_fields,
        )

        # Cleanup the radiation fields in the C code (with mode="cleanup"). This also multiplies the radiation fields by appropriate
        # constants to make them physically meaningful quantities.
        # TODO: this is a bit hacky. A better solution would be to have RadiationFieldsSetup that we have in the C code as an
        #       OutputStructZ, but I suggeste to wait with this solution until https://github.com/21cmfast/21cmFAST/issues/668 is
        #       fixed, as RadiationFieldsSetup would become much simpler
        radiation_fields.compute(
            redshift=redshift,
            halobox=hboxes[0],  # dummy
            R_inner=0.0,  # dummy
            R_outer=0.0,  # dummy
            R_ct=0,  # dummy
            R_star=0.0,  # dummy
            perturbed_field=perturbed_field,
            previous_spin_temp=previous_spin_temp,
            initial_conditions=initial_conditions,
            mode=update_rad_fields_mode["cleanup"],
            cleanup=cleanup,
            allow_already_computed=True,
        )
    else:
        # Sometimes we don't compute at all
        # (if the first zpp > z_max or there are no halos at max R)
        # in which case the array is not marked as computed
        for name, array in radiation_fields.arrays.items():
            setattr(radiation_fields, name, array.computed())

    return radiation_fields


@single_field_func
@init_c_state(sigma=True, heat=True)
def compute_spin_temperature(
    *,
    initial_conditions: InitialConditions,
    perturbed_field: PerturbedField,
    inputs: InputParameters | None = None,
    radiation_fields: RadiationFields,
    previous_spin_temp: TsBox | None = None,
    cleanup: bool = False,
) -> TsBox:
    r"""
    Compute spin temperature boxes at a given redshift.

    See the notes below for how the spin temperature field is evolved through redshift.

    Parameters
    ----------
    initial_conditions : :class:`~InitialConditions`
        The initial conditions
    inputs : :class:`~InputParameters`
        The input parameters specifying the run. Since this will be the first box
        to use the astro params/flags when SOURCE_MODEL='E-INTEGRAL' and USE_TS_FLUCT=True.
    perturbed_field : :class:`~PerturbedField`, optional
        If given, this field will be used, otherwise it will be generated. To be generated,
        either `initial_conditions` and `redshift` must be given, or `simulation_options`, `cosmo_params` and
        `redshift`. By default, this will be generated at the same redshift as the spin temperature
        box. The redshift of perturb field is allowed to be different than `redshift`. If so, it
        will be interpolated to the correct redshift, which can provide a speedup compared to
        actually computing it at the desired redshift.
    radiation_fields : :class:`RadiationFields`
        This input specifies radiation fields, i.e. X-ray heating rate, photoionization rate, and Lyman-alpha flux.
    previous_spin_temp : :class:`TsBox` or None
        The previous spin temperature box. Needed when we are beyond the first snapshot

    Returns
    -------
    :class:`~TsBox`
        An object containing the spin temperature box data.

    Other Parameters
    ----------------
    regenerate, write, cache:
        See docs of :func:`initial_conditions` for more information.
    """
    redshift = perturbed_field.redshift

    if redshift >= inputs.simulation_options.Z_HEAT_MAX:
        previous_spin_temp = TsBox.dummy()

    # Set up the box without computing anything.
    box = TsBox.new(
        redshift=redshift,
        inputs=inputs,
    )

    # Run the C Code
    return box.compute(
        cleanup=cleanup,
        perturbed_field=perturbed_field,
        radiation_fields=radiation_fields,
        prev_spin_temp=previous_spin_temp,
        ics=initial_conditions,
    )


@single_field_func
@init_c_state(sigma=True, heat=True, recomb=True)
def compute_ionization_field(
    *,
    perturbed_field: PerturbedField,
    initial_conditions: InitialConditions,
    inputs: InputParameters | None = None,
    previous_perturbed_field: PerturbedField | None = None,
    previous_ionized_box: IonizedBox | None = None,
    spin_temp: TsBox | None = None,
    halobox: HaloBox | None = None,
) -> IonizedBox:
    r"""
    Compute an ionized box at a given redshift.

    This function has various options for how the evolution of the ionization is
    computed (if at all). See the Notes below for details.

    Parameters
    ----------
    initial_conditions : :class:`~InitialConditions` instance
        The initial conditions.
    inputs : :class:`~InputParameters`
        The input parameters specifying the run. Since this may be the first box
        to use the astro params/flags, it is needed when we have not computed a TsBox or HaloBox.
    perturbed_field : :class:`~PerturbedField`
        The perturbed density field.
    previous_perturbed_field : :class:`~PerturbedField`, optional
        An perturbed field at higher redshift. This is only used if USE_MINI_HALOS is included.
    previous_ionize_box: :class:`IonizedBox` or None
        An ionized box at higher redshift. This is only used if `RECOMB_MODEL != "none"` and/or `USE_TS_FLUCT`
        is true. If either of these are true, and this is not given, then it will be assumed that
        this is the "first box", i.e. that it can be populated accurately without knowing source
        statistics.
    spin_temp: :class:`TsBox` or None, optional
        A spin-temperature box, only required if `USE_TS_FLUCT` is True. If None, will try to read
        in a spin temp box at the current redshift, and failing that will try to automatically
        create one, using the previous ionized box redshift as the previous spin temperature
        redshift.
    halobox: :class:`~HaloBox` or None, optional
        If passed, this contains all the dark matter haloes obtained if using the
        lagrangian source models. These are grids containing summed halo properties
        such as ionizing emissivity.

    Returns
    -------
    :class:`~IonizedBox` :
        An object containing the ionized box data.

    Notes
    -----
    Typically, the ionization field at any redshift is dependent on the evolution of xHI up until
    that redshift, which necessitates providing a previous ionization field to define the current
    one. If neither the spin temperature field, nor inhomogeneous recombinations (specified in
    flag options) are used, no evolution needs to be done. If the redshift is beyond
    Z_HEAT_MAX, previous fields are not required either.
    """
    redshift = perturbed_field.redshift

    if redshift >= inputs.simulation_options.Z_HEAT_MAX:
        # Previous boxes must be "initial"
        previous_ionized_box = IonizedBox.initial(inputs=inputs)
        previous_perturbed_field = PerturbedField.initial(inputs=inputs)

    if inputs.evolution_required:
        if previous_ionized_box is None:
            raise ValueError(
                "You need to provide a previous ionized box when redshift < Z_HEAT_MAX."
            )
        if previous_perturbed_field is None:
            raise ValueError(
                "You need to provide a previous perturbed field when redshift < Z_HEAT_MAX."
            )
    else:
        if previous_ionized_box is None:
            previous_ionized_box = IonizedBox.initial(inputs=inputs)
        if previous_perturbed_field is None:
            previous_perturbed_field = PerturbedField.initial(inputs=inputs)

    if inputs.simulation_options.HII_DIM > 1:
        box = IonizedBox.new(inputs=inputs, redshift=redshift)

        if not inputs.matter_options.lagrangian_source_grid:
            # Construct an empty halo field to pass in to the function.
            halobox = HaloBox.dummy()
        elif halobox is None:
            raise ValueError(
                f"A HaloBox must be provided for SOURCE_MODEL={inputs.matter_options.SOURCE_MODEL}"
            )

        # Set empty spin temp box if necessary.
        if not inputs.astro_options.USE_TS_FLUCT:
            spin_temp = TsBox.dummy()
        elif spin_temp is None:
            raise ValueError("No spin temperature box given but USE_TS_FLUCT=True")

        # Run the C Code
        box.compute(
            perturbed_field=perturbed_field,
            prev_perturbed_field=previous_perturbed_field,
            prev_ionize_box=previous_ionized_box,
            spin_temp=spin_temp,
            halobox=halobox,
            ics=initial_conditions,
        )
    else:
        # If we have only one cell (could happen if we do a global evolution), we set the neutral fraction
        # according to the global evolution

        from .global_evolution import (
            compute_global_reionization_at_z,  # This is imported here to prevent circular import
        )

        box = compute_global_reionization_at_z(
            redshift=redshift,
            inputs=inputs,
            previous_ionized_box=previous_ionized_box,
            spin_temp=spin_temp,
        )

    # There is a need to purge the "initial" boxes since they were exposed to C
    if previous_ionized_box.initial:
        previous_ionized_box.purge(force=True)
    if previous_perturbed_field.initial:
        previous_perturbed_field.purge(force=True)

    return box


@single_field_func
@init_c_state(broadcast_inputs=True)
def brightness_temperature(
    *,
    ionized_box: IonizedBox,
    perturbed_field: PerturbedField,
    spin_temp: TsBox | None = None,
) -> BrightnessTemp:
    r"""
    Compute a coeval brightness temperature box.

    Parameters
    ----------
    ionized_box: :class:`IonizedBox`
        A pre-computed ionized box.
    perturbed_field: :class:`PerturbedField`
        A pre-computed perturbed field at the same redshift as `ionized_box`.
    spin_temp: :class:`TsBox`, optional
        A pre-computed spin temperature, at the same redshift as the other boxes.

    Returns
    -------
    :class:`BrightnessTemp` instance.
    """
    redshift = ionized_box.redshift
    inputs = ionized_box.inputs

    if spin_temp is None:
        if inputs.astro_options.USE_TS_FLUCT:
            raise ValueError(
                "You have USE_TS_FLUCT=True, but have not provided a spin_temp!"
            )
        else:
            spin_temp = TsBox.dummy()

    box = BrightnessTemp.new(redshift=redshift, inputs=inputs)

    return box.compute(
        spin_temp=spin_temp,
        ionized_box=ionized_box,
        perturbed_field=perturbed_field,
    )
