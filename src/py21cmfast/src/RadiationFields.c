/*
Module for computing the radiation fields in 21cmFAST.
This includes X-ray heating rate, photoionization rate, and Lyman-alpha flux.
*/
#include "RadiationFields.h"

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "cexcept.h"
#include "cosmology.h"
#include "debugging.h"
#include "dft.h"
#include "elec_interp.h"
#include "exceptions.h"
#include "filtering.h"
#include "heating_helper_progs.h"
#include "hmf.h"
#include "indexing.h"
#include "interp_tables.h"
#include "logger.h"
#include "thermochem.h"

// global pointer to the radiation fields setup struct. This is useful since we visit this module
// several times per snapshot
RadiationFieldsSetup *rad_setup = NULL;

// Global arrays which have yet to be moved to structs
// R x box arrays
float **delNL0, **log10_Mcrit_LW;

// arrays for R-dependent prefactors
double *lya_flux_continuum_injected_prefactor, *lya_flux_continuum_injected_prefactor_MINI;
double *lyw_flux_prefactor, *lyw_flux_prefactor_MINI;
double *lya_flux_continuum_prefactor, *lya_flux_injected_prefactor;
double *lya_flux_continuum_prefactor_MINI, *lya_flux_injected_prefactor_MINI;

// boxes to hold stellar fraction integrals (Fcoll or SFRD)
float *del_fcoll_Rct, *del_fcoll_Rct_MINI;

// x_e interpolation boxes / arrays (not a RGI)
float *inverse_val_box;
int *m_xHII_low_box;
float *inverse_diff;

// interpolation tables for the heating/ionisation integrals
double **freq_int_heat_tbl, **freq_int_ion_tbl, **freq_int_lya_tbl, **freq_int_heat_tbl_diff;
double **freq_int_ion_tbl_diff, **freq_int_lya_tbl_diff;

// R-dependent arrays which are set once
double *R_values, *dzpp_list, *dtdz_list, *zpp_growth, *zpp_for_evolve_list, *zpp_edge;
double *sigma_min, *sigma_max, *M_max_R, *M_min_R;
double *min_densities, *max_densities;

// Arrays which specify the Radii, distances, redshifts of each shell
//   They will have a global instance since they are reused a lot
//   However it is worth considering passing them into functions instead
//  struct radii_spec{
//      double *R_values; //Radii edge of each shell
//      double *zpp_edge; //redshift of the shell edge
//      double *zpp_cen; //middle redshift of cell (z_inner + z_outer)/2
//      double *dzpp_list; //redshift difference between inner and outer edge
//      double *dtdz_list; //dtdz at zpp_cen
//      double *zpp_growth; //linear growth factor D(z) at zpp_cen
//  }
//  struct radii_spec r_s;

bool TsInterpArraysInitialised = false;

void alloc_global_arrays() {
    int i;
    // z-edges
    zpp_for_evolve_list = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    zpp_growth = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    zpp_edge = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    dzpp_list = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    dtdz_list = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    R_values = calloc(astro_params_global->N_STEP_TS, sizeof(double));

    sigma_min = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    sigma_max = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    M_min_R = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    M_max_R = calloc(astro_params_global->N_STEP_TS, sizeof(double));

    // frequency integral tables
    freq_int_heat_tbl = (double **)calloc(x_int_NXHII, sizeof(double *));
    freq_int_ion_tbl = (double **)calloc(x_int_NXHII, sizeof(double *));
    freq_int_lya_tbl = (double **)calloc(x_int_NXHII, sizeof(double *));
    freq_int_heat_tbl_diff = (double **)calloc(x_int_NXHII, sizeof(double *));
    freq_int_ion_tbl_diff = (double **)calloc(x_int_NXHII, sizeof(double *));
    freq_int_lya_tbl_diff = (double **)calloc(x_int_NXHII, sizeof(double *));
    for (i = 0; i < x_int_NXHII; i++) {
        freq_int_heat_tbl[i] = (double *)calloc(astro_params_global->N_STEP_TS, sizeof(double));
        freq_int_ion_tbl[i] = (double *)calloc(astro_params_global->N_STEP_TS, sizeof(double));
        freq_int_lya_tbl[i] = (double *)calloc(astro_params_global->N_STEP_TS, sizeof(double));
        freq_int_heat_tbl_diff[i] =
            (double *)calloc(astro_params_global->N_STEP_TS, sizeof(double));
        freq_int_ion_tbl_diff[i] = (double *)calloc(astro_params_global->N_STEP_TS, sizeof(double));
        freq_int_lya_tbl_diff[i] = (double *)calloc(astro_params_global->N_STEP_TS, sizeof(double));
    }
    inverse_diff = (float *)calloc(x_int_NXHII, sizeof(float));

    // spectral stuff
    if (astro_options_global->USE_LYA_HEATING) {
        lya_flux_continuum_prefactor = calloc(astro_params_global->N_STEP_TS, sizeof(double));
        lya_flux_injected_prefactor = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    } else {
        lya_flux_continuum_injected_prefactor =
            calloc(astro_params_global->N_STEP_TS, sizeof(double));
    }

    if (astro_options_global->USE_MINI_HALOS) {
        lyw_flux_prefactor = calloc(astro_params_global->N_STEP_TS, sizeof(double));
        lyw_flux_prefactor_MINI = calloc(astro_params_global->N_STEP_TS, sizeof(double));
        if (astro_options_global->USE_LYA_HEATING) {
            lya_flux_continuum_prefactor_MINI =
                calloc(astro_params_global->N_STEP_TS, sizeof(double));
            lya_flux_injected_prefactor_MINI =
                calloc(astro_params_global->N_STEP_TS, sizeof(double));
        } else {
            lya_flux_continuum_injected_prefactor_MINI =
                calloc(astro_params_global->N_STEP_TS, sizeof(double));
        }
    }

    // Nonhalo stuff
    if (source_model_uses_eulerian_grids(matter_options_global->SOURCE_MODEL)) {
        int num_R_boxes =
            matter_options_global->MINIMIZE_MEMORY ? 1 : astro_params_global->N_STEP_TS;

        delNL0 = (float **)calloc(num_R_boxes, sizeof(float *));
        for (i = 0; i < num_R_boxes; i++) {
            delNL0[i] = (float *)calloc(HII_TOT_NUM_PIXELS, sizeof(float));
        }
        if (astro_options_global->USE_MINI_HALOS) {
            log10_Mcrit_LW = (float **)calloc(num_R_boxes, sizeof(float *));
            for (i = 0; i < num_R_boxes; i++) {
                log10_Mcrit_LW[i] = (float *)calloc(HII_TOT_NUM_PIXELS, sizeof(float));
            }
        }

        del_fcoll_Rct = calloc(HII_TOT_NUM_PIXELS, sizeof(float));
        if (astro_options_global->USE_MINI_HALOS) {
            del_fcoll_Rct_MINI = calloc(HII_TOT_NUM_PIXELS, sizeof(float));
        }

        min_densities = calloc(astro_params_global->N_STEP_TS, sizeof(double));
        max_densities = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    }

    // helpers for the interpolation
    // NOTE: The frequency integrals are tables regardless of the flag
    m_xHII_low_box = (int *)calloc(HII_TOT_NUM_PIXELS, sizeof(int));
    inverse_val_box = (float *)calloc(HII_TOT_NUM_PIXELS, sizeof(float));

    TsInterpArraysInitialised = true;
}

void free_ts_global_arrays() {
    int i;
    // frequency integrals
    for (i = 0; i < x_int_NXHII; i++) {
        free(freq_int_heat_tbl[i]);
        free(freq_int_ion_tbl[i]);
        free(freq_int_lya_tbl[i]);
        free(freq_int_heat_tbl_diff[i]);
        free(freq_int_ion_tbl_diff[i]);
        free(freq_int_lya_tbl_diff[i]);
    }
    free(freq_int_heat_tbl);
    free(freq_int_ion_tbl);
    free(freq_int_lya_tbl);
    free(freq_int_heat_tbl_diff);
    free(freq_int_ion_tbl_diff);
    free(freq_int_lya_tbl_diff);
    free(inverse_diff);

    // z- edges
    free(zpp_growth);
    free(zpp_edge);
    free(zpp_for_evolve_list);
    free(dzpp_list);
    free(dtdz_list);
    free(R_values);

    free(sigma_min);
    free(sigma_max);
    free(M_min_R);
    free(M_max_R);

    // spectral
    if (astro_options_global->USE_LYA_HEATING) {
        free(lya_flux_continuum_prefactor);
        free(lya_flux_injected_prefactor);
    } else {
        free(lya_flux_continuum_injected_prefactor);
    }
    if (astro_options_global->USE_MINI_HALOS) {
        free(lyw_flux_prefactor);
        free(lyw_flux_prefactor_MINI);
        if (astro_options_global->USE_LYA_HEATING) {
            free(lya_flux_injected_prefactor_MINI);
            free(lya_flux_continuum_prefactor_MINI);
        } else {
            free(lya_flux_continuum_injected_prefactor_MINI);
        }
    }

    // interpolation helpers
    free(m_xHII_low_box);
    free(inverse_val_box);

    // interp tables
    if (source_model_uses_eulerian_grids(matter_options_global->SOURCE_MODEL)) {
        int num_R_boxes =
            matter_options_global->MINIMIZE_MEMORY ? 1 : astro_params_global->N_STEP_TS;

        for (i = 0; i < num_R_boxes; i++) {
            free(delNL0[i]);
        }
        free(delNL0);

        if (astro_options_global->USE_MINI_HALOS) {
            for (i = 0; i < num_R_boxes; i++) {
                free(log10_Mcrit_LW[i]);
            }
            free(log10_Mcrit_LW);
        }

        free(del_fcoll_Rct);
        if (astro_options_global->USE_MINI_HALOS) {
            free(del_fcoll_Rct_MINI);
        }

        free(min_densities);
        free(max_densities);
    }

    TsInterpArraysInitialised = false;
}

// This function should construct all the tables which depend on R
void setup_z_edges(double zp) {
    double R, R_factor;
    double zpp, prev_zpp, prev_R;
    double dzpp_for_evolve;
    int R_ct;

    if (simulation_options_global->HII_DIM == 1) {
        // If HII_DIM=1 (happens when we run_global_evolution), we take a typical cell size
        // of 1.5Mpc, just to for setting the z'' array (note that filtering won't be done on a box
        // with a single cell)
        R = physconst.l_factor * 1.5;
    } else {
        R = physconst.l_factor * simulation_options_global->BOX_LEN /
            (float)simulation_options_global->HII_DIM;
    }
    R_factor = pow(astro_params_global->R_MAX_TS / R, 1 / ((float)astro_params_global->N_STEP_TS));

    for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
        R_values[R_ct] = R;
        if (R_ct == 0) {
            prev_zpp = zp;
            prev_R = 0;
        } else {
            prev_zpp = zpp_edge[R_ct - 1];
            prev_R = R_values[R_ct - 1];
        }

        // cell size
        zpp_edge[R_ct] =
            prev_zpp - (R_values[R_ct] - prev_R) * physconst.cm_per_Mpc / drdz(prev_zpp);
        // average redshift value of shell: z'' + 0.5 * dz''
        zpp = (zpp_edge[R_ct] + prev_zpp) * 0.5;

        zpp_for_evolve_list[R_ct] = zpp;
        if (R_ct == 0) {
            dzpp_for_evolve = zp - zpp_edge[0];
        } else {
            dzpp_for_evolve = zpp_edge[R_ct - 1] - zpp_edge[R_ct];
        }
        zpp_growth[R_ct] = dicke(zpp);      // growth factors
        dzpp_list[R_ct] = dzpp_for_evolve;  // z bin width
        dtdz_list[R_ct] = dtdz(zpp);        // dt/dz''

        M_min_R[R_ct] = minimum_source_mass(zpp_for_evolve_list[R_ct], true);
        M_max_R[R_ct] = RtoM(R_values[R_ct]);

        R *= R_factor;
    }
    LOG_DEBUG("%d steps R range [%.2e,%.2e] z range [%.2f,%.2f]", R_ct, R_values[0],
              R_values[R_ct - 1], zp, zpp_edge[R_ct - 1]);
}

void calculate_spectral_factors(double zp) {
    double nuprime;
    bool first_radii = true, first_zero = true;
    double trial_zpp;
    int counter, ii;
    int n_pts_radii = 1000;
    double weight = 0.;
    int R_ct, n_ct;
    double zpp, zpp_integrand;

    double sum_lyn_val, sum_lyn_val_MINI;
    double sum_lyLW_val, sum_lyLW_val_MINI;
    double sum_lynto2_val, sum_lynto2_val_MINI;
    double sum_ly2_val, sum_ly2_val_MINI;
    // technically don't need to initialise since it is only used in R_ct > 1 conditional
    double sum_lyn_prev = 0., sum_lyn_prev_MINI = 0.;
    double sum_ly2_prev = 0., sum_ly2_prev_MINI = 0.;
    double sum_lynto2_prev = 0., sum_lynto2_prev_MINI = 0.;
    double prev_zpp = 0;
    for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
        zpp = zpp_for_evolve_list[R_ct];
        // We need to set up prefactors for how much of Lyman-N radiation is recycled to Lyman-alpha
        sum_lyLW_val = 0.;
        sum_lyLW_val_MINI = 0.;
        sum_lynto2_val = 0.;
        sum_lynto2_val_MINI = 0.;
        sum_ly2_val = 0.;
        sum_ly2_val_MINI = 0.;

        // in case we use LYA_HEATING, we separate the ==2 and >2 cases
        nuprime = nu_n(2) * (1. + zpp) / (1. + zp);
        if (zpp < zmax(zp, 2)) {
            sum_ly2_val = frecycle(2) * spectral_emissivity(nuprime, 0, 2);
            if (astro_options_global->USE_MINI_HALOS) {
                sum_ly2_val_MINI = frecycle(2) * spectral_emissivity(nuprime, 0, 3);

                if (nuprime < physconst.nu_LW_thresh / physconst.nu_ion_HI)
                    nuprime = physconst.nu_LW_thresh / physconst.nu_ion_HI;
                // NOTE: are we comparing nuprime at z' and z'' correctly here?
                //   currently: emitted frequency >= received frequency of next n
                if (nuprime >= nu_n(2 + 1)) continue;

                sum_lyLW_val +=
                    (1. - astro_params_global->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 2);
                sum_lyLW_val_MINI +=
                    (1. - astro_params_global->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 3);
            }
        }

        for (n_ct = NSPEC_MAX; n_ct >= 3; n_ct--) {
            if (zpp > zmax(zp, n_ct)) continue;

            nuprime = nu_n(n_ct) * (1 + zpp) / (1.0 + zp);
            sum_lynto2_val += frecycle(n_ct) * spectral_emissivity(nuprime, 0, 2);
            if (astro_options_global->USE_MINI_HALOS) {
                sum_lynto2_val_MINI += frecycle(n_ct) * spectral_emissivity(nuprime, 0, 3);

                if (nuprime < physconst.nu_LW_thresh / physconst.nu_ion_HI)
                    nuprime = physconst.nu_LW_thresh / physconst.nu_ion_HI;
                if (nuprime >= nu_n(n_ct + 1)) continue;
                sum_lyLW_val +=
                    (1. - astro_params_global->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 2);
                sum_lyLW_val_MINI +=
                    (1. - astro_params_global->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 3);
            }
        }
        sum_lyn_val = sum_ly2_val + sum_lynto2_val;
        sum_lyn_val_MINI = sum_ly2_val_MINI + sum_lynto2_val_MINI;

        // At the edge of the redshift limit, part of the shell will still contain a contribution
        //   This loop approximates the volume which contains the contribution
        //   and multiplies this by the previous shell's value.
        // This should probably be done separately for each line, since they all face the same issue
        //   It could also be avoided by approximating the integral within each bin better
        //   than taking the value at the bin centre
        if (R_ct > 1 && sum_lyn_val == 0.0 && sum_lyn_prev > 0. && first_radii) {
            for (ii = 0; ii < n_pts_radii; ii++) {
                trial_zpp = prev_zpp + (zpp - prev_zpp) * (float)ii / ((float)n_pts_radii - 1.);
                counter = 0;
                for (n_ct = NSPEC_MAX; n_ct >= 2; n_ct--) {
                    if (trial_zpp > zmax(zp, n_ct)) continue;
                    counter += 1;
                }
                // This is the first sub-radius which has no contribution
                // Use this distance to weigh contribution at previous R
                if (counter == 0 && first_zero) {
                    first_zero = false;
                    weight = (float)ii / (float)n_pts_radii;
                }
            }
            sum_lyn_val = weight * sum_lyn_prev;
            sum_ly2_val = weight * sum_ly2_prev;
            sum_lynto2_val = weight * sum_lynto2_prev;
            if (astro_options_global->USE_MINI_HALOS) {
                sum_lyn_val_MINI = weight * sum_lyn_prev_MINI;
                sum_ly2_val_MINI = weight * sum_ly2_prev_MINI;
                sum_lynto2_val_MINI = weight * sum_lynto2_prev_MINI;
            }
            first_radii = false;
        }
        zpp_integrand = (pow(1 + zp, 2) * (1 + zpp));

        if (astro_options_global->USE_LYA_HEATING) {
            lya_flux_continuum_prefactor[R_ct] = zpp_integrand * sum_ly2_val;
            lya_flux_injected_prefactor[R_ct] = zpp_integrand * sum_lynto2_val;
            LOG_ULTRA_DEBUG("cont %.2e inj %.2e", lya_flux_continuum_prefactor[R_ct],
                            lya_flux_injected_prefactor[R_ct]);
        } else {
            lya_flux_continuum_injected_prefactor[R_ct] = zpp_integrand * sum_lyn_val;
            LOG_ULTRA_DEBUG("z: %.2e R: %.2e int %.2e starlya: %.4e", zpp, R_values[R_ct],
                            zpp_integrand, lya_flux_continuum_injected_prefactor[R_ct]);
        }
        if (astro_options_global->USE_MINI_HALOS) {
            lyw_flux_prefactor[R_ct] = zpp_integrand * sum_lyLW_val;
            lyw_flux_prefactor_MINI[R_ct] = zpp_integrand * sum_lyLW_val_MINI;
            LOG_ULTRA_DEBUG("LW: %.2e LWmini: %.2e", lyw_flux_prefactor[R_ct],
                            lyw_flux_prefactor_MINI[R_ct]);
            if (astro_options_global->USE_LYA_HEATING) {
                lya_flux_continuum_prefactor_MINI[R_ct] = zpp_integrand * sum_ly2_val_MINI;
                lya_flux_injected_prefactor_MINI[R_ct] = zpp_integrand * sum_lynto2_val_MINI;
                LOG_ULTRA_DEBUG("cont mini %.2e inj mini %.2e",
                                lya_flux_continuum_prefactor_MINI[R_ct],
                                lya_flux_injected_prefactor_MINI[R_ct]);
            } else {
                lya_flux_continuum_injected_prefactor_MINI[R_ct] = zpp_integrand * sum_lyn_val_MINI;
                LOG_ULTRA_DEBUG("starmini: %.2e", lya_flux_continuum_injected_prefactor_MINI[R_ct]);
            }
        }

        sum_lyn_prev = sum_lyn_val;
        sum_lyn_prev_MINI = sum_lyn_val_MINI;
        sum_ly2_prev = sum_ly2_val;
        sum_ly2_prev_MINI = sum_ly2_val_MINI;
        sum_lynto2_prev = sum_lynto2_val;
        sum_lynto2_prev_MINI = sum_lynto2_val_MINI;
        prev_zpp = zpp;
    }
}

// fill fftwf boxes, do the r2c transform and normalise
void prepare_filter_boxes(double redshift, float *input_dens, float *input_vcb, float *input_j21,
                          fftwf_complex *output_dens, fftwf_complex *output_LW,
                          ScalingConstants *sc) {
    int i, j, k;
    index_huge ct, index_f;
    double curr_vcb, curr_j21, M_buf;
    int box_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                      HII_D_PARA};

// NOTE: Meraxes just applies a pointer cast box = (fftwf_complex *) input. Figure out why this
// works,
//       They pad the input by a factor of 2 to cover the complex part, but from the type I thought
//       it would be stored [(r,c),(r,c)...] Not [(r,r,r,r....),(c,c,c....)] so the alignment should
//       be wrong, right?
#pragma omp parallel for private(i, j, k, ct, index_f) \
    num_threads(simulation_options_global->N_THREADS) collapse(3)
    for (i = 0; i < box_dim[0]; i++) {
        for (j = 0; j < box_dim[1]; j++) {
            for (k = 0; k < box_dim[2]; k++) {
                ct = grid_index_general(i, j, k, box_dim);
                index_f = grid_index_fftw_r(i, j, k, box_dim);
                *((float *)output_dens + index_f) = input_dens[ct];
            }
        }
    }
    // Transform unfiltered box to k-space to prepare for filtering
    dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                 HII_D_PARA, simulation_options_global->N_THREADS, output_dens);
#pragma omp parallel for num_threads(simulation_options_global->N_THREADS)
    for (ct = 0; ct < HII_KSPACE_NUM_PIXELS; ct++) {
        output_dens[ct] /= (float)HII_TOT_NUM_PIXELS;
    }

    if (astro_options_global->USE_MINI_HALOS) {
        curr_vcb = sc->vcb_const;
#pragma omp parallel for firstprivate(curr_vcb) private(i, j, k, curr_j21, M_buf, ct, index_f) \
    num_threads(simulation_options_global->N_THREADS) collapse(3)
        for (i = 0; i < box_dim[0]; i++) {
            for (j = 0; j < box_dim[1]; j++) {
                for (k = 0; k < box_dim[2]; k++) {
                    ct = grid_index_general(i, j, k, box_dim);
                    index_f = grid_index_fftw_r(i, j, k, box_dim);
                    if (matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS) {
                        curr_vcb = input_vcb[ct];
                    }
                    curr_j21 = input_j21[ct];
                    // NOTE: we don't use reionization_feedback here, I assume it wouldn't do much
                    // but it's inconsistent
                    M_buf =
                        molecular_cooling_threshold_with_feedbacks(redshift, curr_j21, curr_vcb);
                    M_buf = fmax(M_buf, astro_params_global->M_TURN_STELLAR_FEEDBACK);
                    *((float *)output_LW + index_f) = log10(M_buf);
                }
            }
        }
        // Transform unfiltered box to k-space to prepare for filtering
        dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                     HII_D_PARA, simulation_options_global->N_THREADS, output_LW);
#pragma omp parallel for num_threads(simulation_options_global->N_THREADS)
        for (ct = 0; ct < HII_KSPACE_NUM_PIXELS; ct++) {
            output_LW[ct] /= (float)HII_TOT_NUM_PIXELS;
        }
    }
}

// fill a box[R_ct][box_ct] array for use in TS by filtering on different scales and storing results
void fill_Rbox_table(float **result, fftwf_complex *unfiltered_box, double *R_array, int n_R,
                     double min_value, double const_factor, double *min_arr, double *average_arr,
                     double *max_arr) {
    // allocate table/grid memory
    int i, j, k, R_ct;
    double R;
    double ave_buffer, min_out_R, max_out_R;
    int box_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                      HII_D_PARA};

    fftwf_complex *box =
        (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
    // Smooth the density field, at the same time store the minimum and maximum densities for their
    // usage in the interpolation tables
    LOG_ULTRA_DEBUG("db0");
    for (R_ct = 0; R_ct < n_R; R_ct++) {
        R = R_array[R_ct];
        ave_buffer = 0;
        min_out_R = 1e20;
        max_out_R = -1e20;
        // copy over unfiltered box
        memcpy(box, unfiltered_box, sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
        LOG_ULTRA_DEBUG("db1 %d", R_ct);

        // don't filter on cell size
        if (R > physconst.l_factor *
                    (simulation_options_global->BOX_LEN / simulation_options_global->HII_DIM)) {
            filter_box(box, box_dim, astro_options_global->HEAT_FILTER, R, 0., 0.);
        }

        LOG_ULTRA_DEBUG("db2 %d", R_ct);

        // now fft back to real space
        dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                     HII_D_PARA, simulation_options_global->N_THREADS, box);

        LOG_ULTRA_DEBUG("db3 %d", R_ct);
        // copy over the values
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS)
        {
            float curr;
            index_huge index_r, index_f;
#pragma omp for reduction(+ : ave_buffer) reduction(max : max_out_R) reduction(min : min_out_R)
            for (i = 0; i < box_dim[0]; i++) {
                for (j = 0; j < box_dim[1]; j++) {
                    for (k = 0; k < box_dim[2]; k++) {
                        index_r = grid_index_general(i, j, k, box_dim);
                        index_f = grid_index_fftw_r(i, j, k, box_dim);
                        curr = *((float *)box + index_f);

                        // NOTE: Min value is on the grid BEFORE constant factor
                        //  correct for aliasing in the filtering step
                        if (curr < min_value) {
                            curr = min_value;
                        }

                        // constant factors (i.e linear extrapolation to z=0 for dens.)
                        curr = curr * const_factor;

                        ave_buffer += curr;
                        if (curr < min_out_R) min_out_R = curr;
                        if (curr > max_out_R) max_out_R = curr;
                        result[R_ct][index_r] = curr;
                    }
                }
            }
        }
        average_arr[R_ct] = ave_buffer / HII_TOT_NUM_PIXELS;
        min_arr[R_ct] = min_out_R;
        max_arr[R_ct] = max_out_R;
        LOG_ULTRA_DEBUG("db4 %d", R_ct);
    }
    LOG_ULTRA_DEBUG("db5");
    fftwf_free(box);
}

// construct the [x_e][R_ct] tables
// NOTE: these have always been interpolation tables in x_e, regardless of flags
// NOTE: Frequency integrals are based on PREVIOUS x_e_ave
//   The x_e tables are not regular, hence the precomputation of indices/interp points
void fill_freqint_tables(double zp, double x_e_ave, double filling_factor_of_HI_zp,
                         double *log10_Mcrit_LW_ave, int R_mm, ScalingConstants *sc) {
    double lower_int_limit;
    int x_e_ct, R_ct;
    int R_start, R_end;
    // if we minimize mem these arrays are filled one by one
    if (matter_options_global->MINIMIZE_MEMORY &&
        source_model_uses_eulerian_grids(matter_options_global->SOURCE_MODEL)) {
        R_start = R_mm;
        R_end = R_mm + 1;
    } else {
        R_start = 0;
        R_end = astro_params_global->N_STEP_TS;
    }
#pragma omp parallel private(R_ct, x_e_ct, lower_int_limit) \
    num_threads(simulation_options_global -> N_THREADS)
    {
#pragma omp for
        // In TauX we integrate Nion from zpp to zp using the LW turnover mass at zp (predending its
        // at zpp)
        //   Calculated from the average smoothed zp grid (from previous LW field) at radius R
        // NOTE: The one difference currently between the halobox and density field options is the
        // weighting of the average
        //   density -> volume weighted cell average || halo -> halo weighted average
        for (R_ct = R_start; R_ct < R_end; R_ct++) {
            // TODO: At the moment, inhomogeneous reionization feedback cannot be accounted in
            // SpinTemperatureBox.c,
            //      see https://github.com/21cmfast/21cmFAST/issues/470. Thus, we use the
            //      homogeneous (feedback-free) ACG turnover mass. It is important to remember to
            //      fix this when issue #470 is fixed!
            if (astro_options_global->USE_MINI_HALOS) {
                lower_int_limit = fmax(
                    nu_tau_one_MINI(zp, zpp_for_evolve_list[R_ct], x_e_ave, filling_factor_of_HI_zp,
                                    log10(sc->mturn_acg_homogeneous), log10_Mcrit_LW_ave[R_ct], sc),
                    (astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz);
            } else {
                lower_int_limit =
                    fmax(nu_tau_one(zp, zpp_for_evolve_list[R_ct], x_e_ave, filling_factor_of_HI_zp,
                                    log10(sc->mturn_acg_homogeneous), sc),
                         (astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz);
            }
            // set up frequency integral table for later interpolation for the cell's x_e value
            for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++) {
                freq_int_heat_tbl[x_e_ct][R_ct] =
                    integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 0);
                freq_int_ion_tbl[x_e_ct][R_ct] =
                    integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 1);
                freq_int_lya_tbl[x_e_ct][R_ct] =
                    integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 2);

                // we store these to avoid calculating them in the box_ct loop
                if (x_e_ct > 0) {
                    freq_int_heat_tbl_diff[x_e_ct - 1][R_ct] =
                        freq_int_heat_tbl[x_e_ct][R_ct] - freq_int_heat_tbl[x_e_ct - 1][R_ct];
                    freq_int_ion_tbl_diff[x_e_ct - 1][R_ct] =
                        freq_int_ion_tbl[x_e_ct][R_ct] - freq_int_ion_tbl[x_e_ct - 1][R_ct];
                    freq_int_lya_tbl_diff[x_e_ct - 1][R_ct] =
                        freq_int_lya_tbl[x_e_ct][R_ct] - freq_int_lya_tbl[x_e_ct - 1][R_ct];
                }
            }
            LOG_ULTRA_DEBUG("Nu Integrals || R_ct %d R %.2e zpp %.2f nu_min %.2e", R_ct,
                            R_values[R_ct], zpp_for_evolve_list[R_ct], lower_int_limit);
            LOG_ULTRA_DEBUG("heat[x_e=0] %.2e ion[x_e=0] %.2e lya[x_e=0] %.2e",
                            freq_int_heat_tbl[0][R_ct], freq_int_ion_tbl[0][R_ct],
                            freq_int_lya_tbl[0][R_ct]);
        }
// separating the inverse diff loop to prevent a race on different R_ct (shouldn't matter)
#pragma omp for
        for (x_e_ct = 0; x_e_ct < x_int_NXHII - 1; x_e_ct++) {
            inverse_diff[x_e_ct] = 1. / (x_int_XHII[x_e_ct + 1] - x_int_XHII[x_e_ct]);
        }
    }

    for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
        for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++) {
            if (isfinite(freq_int_heat_tbl[x_e_ct][R_ct]) == 0 ||
                isfinite(freq_int_ion_tbl[x_e_ct][R_ct]) == 0 ||
                isfinite(freq_int_lya_tbl[x_e_ct][R_ct]) == 0) {
                LOG_ERROR("One of the frequency interpolation tables has an infinity or a NaN");
                //                        Throw(ParameterError);
                Throw(TableGenerationError);
            }
        }
    }
}

// calculate the global properties used for making the frequency integrals,
//   used for filling factor, ST_OVER_PS, and NO_LIGHT
int global_reion_properties(double zp, RadiationFieldsSetup *rad_setup) {
    int R_ct;
    double sum_nion = 0, sum_nion_mini = 0;
    double zpp;

    // For a lot of global evolution, this code uses Nion_general. We can replace this with the halo
    // field at the same snapshot, but the nu integrals go from zp to zpp to find the tau = 1
    // barrier so it needs the QHII in a range [zp,zpp]. I want to replace this whole thing with a
    // global history struct but I will need to change the Tau function chain.
    double determine_zpp_max, determine_zpp_min;

    // at z', we need a differenc constant struct
    ScalingConstants sc;
    set_scaling_constants(zp, &sc, false);

    if (uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)) {
        determine_zpp_min = zp * 0.999;
        // NOTE: must be called after setup_z_edges for this line
        determine_zpp_max = zpp_for_evolve_list[astro_params_global->N_STEP_TS - 1] * 1.001;

        // We need the tables for the frequency integrals & mean fixing
        // NOTE: These global tables confuse me, we do ~400 (x50 for mini) integrals to build the
        // table, despite only having
        //   ~100 redshifts. The benefit of interpolating here would only matter if we keep the same
        //   table over subsequent snapshots, which we don't seem to do. The Nion table is used in
        //   nu_tau_one a lot but I think there's a better way to do that
        if (source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL)) {
            /* initialise interpolation of the mean collapse fraction for global reionization.*/
            initialise_Nion_Ts_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max,
                                      &sc);

            initialise_SFRD_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max,
                                   &sc);
        } else {
            init_FcollTable(determine_zpp_min, determine_zpp_max, true);
        }
    }

    // For consistency between halo and non-halo based, the NO_LIGHT and filling_factor_zp
    //   are based on the expected global Nion. as mentioned above it would be nice to
    //   change this to a saved reionisation/sfrd history from previous snapshots
    // TODO: At the moment, inhomogeneous reionization feedback cannot be accounted in
    // SpinTemperatureBox.c,
    //      see https://github.com/21cmfast/21cmFAST/issues/470. Thus, we use the homogeneous
    //      (feedback-free) ACG turnover mass. It is important to remember to fix this when issue
    //      #470 is fixed!
    sum_nion = EvaluateNionTs(zp, log10(sc.mturn_acg_homogeneous), &sc);
    if (astro_options_global->USE_MINI_HALOS) {
        sum_nion_mini = EvaluateNionTs_MINI(zp, log10(sc.mturn_acg_homogeneous),
                                            rad_setup->ave_log10_MturnLW[0], &sc);
    }

    LOG_DEBUG("nion zp = %.3e (%.3e MINI)", sum_nion, sum_nion_mini);

    double ION_EFF_FACTOR, ION_EFF_FACTOR_MINI = 0.;
    if (source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL)) {
        ION_EFF_FACTOR = astro_params_global->F_STAR10 * astro_params_global->F_ESC10 *
                         astro_params_global->POP2_ION;
        ION_EFF_FACTOR_MINI = astro_params_global->F_STAR7_MINI * astro_params_global->F_ESC7_MINI *
                              astro_params_global->POP3_ION;
    } else {
        // no mini-halos when SOURCE_MODE is mass independent (constant ionization efficiency)
        ION_EFF_FACTOR = astro_params_global->HII_EFF_FACTOR;
    }

    // NOTE: only used without MASS_DEPENDENT_ZETA
    rad_setup->Q_HI_zp = 1 - (ION_EFF_FACTOR * sum_nion + ION_EFF_FACTOR_MINI * sum_nion_mini) /
                                 (1.0 - rad_setup->x_e_ave_p);

    // Initialise freq tables & prefactors (x_e by R tables)
    if (source_model_uses_lagrangian_grids(matter_options_global->SOURCE_MODEL) ||
        !matter_options_global->MINIMIZE_MEMORY) {
        if (source_model_uses_eulerian_grids(matter_options_global->SOURCE_MODEL)) {
            // Now global SFRD at (R_ct) for the mean fixing
            for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
                zpp = zpp_for_evolve_list[R_ct];
                // TODO: At the moment, inhomogeneous reionization feedback cannot be accounted in
                // SpinTemperatureBox.c,
                //      see https://github.com/21cmfast/21cmFAST/issues/470. Thus, we use the
                //      homogeneous (feedback-free) ACG turnover mass. It is important to remember
                //      to fix this when issue #470 is fixed!
                rad_setup->mean_sfr_zpp[R_ct] =
                    EvaluateSFRD(zpp, log10(sc.mturn_acg_homogeneous), &sc);
                if (astro_options_global->USE_MINI_HALOS) {
                    rad_setup->mean_sfr_zpp_mini[R_ct] =
                        EvaluateSFRD_MINI(zpp, log10(sc.mturn_acg_homogeneous),
                                          rad_setup->ave_log10_MturnLW[R_ct], &sc);
                }
            }
        }
        fill_freqint_tables(zp, rad_setup->x_e_ave_p, rad_setup->Q_HI_zp,
                            rad_setup->ave_log10_MturnLW, 0, &sc);
    }

    return sum_nion + sum_nion_mini > 1e-15 ? 0 : 1;  // NO_LIGHT returned
}

void calculate_sfrd_from_grid(int R_ct, float *dens_R_grid, float *Mcrit_R_grid, float *sfrd_grid,
                              float *sfrd_grid_mini, double *ave_sfrd, double *ave_sfrd_mini,
                              ScalingConstants *sc) {
    double ave_sfrd_buf = 0;
    double ave_sfrd_buf_mini = 0;
    if (astro_options_global->INTEGRATION_METHOD_ATOMIC == INTEGRATION_METHOD_GAUSS_LEGENDRE ||
        (astro_options_global->USE_MINI_HALOS &&
         astro_options_global->INTEGRATION_METHOD_MINI == INTEGRATION_METHOD_GAUSS_LEGENDRE))
        initialise_GL(log(M_min_R[R_ct]), log(M_max_R[R_ct]));

    if (uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)) {
        if (matter_options_global->SOURCE_MODEL == SOURCE_MODEL_E_INTEGRAL) {
            initialise_SFRD_Conditional_table(zpp_for_evolve_list[R_ct],
                                              min_densities[R_ct] * zpp_growth[R_ct],
                                              max_densities[R_ct] * zpp_growth[R_ct] * 1.001,
                                              M_min_R[R_ct], M_max_R[R_ct], M_max_R[R_ct], sc);
        } else if (matter_options_global->SOURCE_MODEL == SOURCE_MODEL_CONST_ION_EFF) {
            initialise_FgtrM_delta_table(
                min_densities[R_ct] * zpp_growth[R_ct], max_densities[R_ct] * zpp_growth[R_ct],
                zpp_for_evolve_list[R_ct], zpp_growth[R_ct], sigma_min[R_ct], sigma_max[R_ct]);
        } else {
            LOG_ERROR("Source model %d is trying to calculate SFRD from grid, something went wrong",
                      matter_options_global->SOURCE_MODEL);
            Throw(ValueError);
        }
    }

#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
    {
        index_huge box_ct;
        double curr_dens;
        double curr_mcrit = 0.;
        double fcoll, dfcoll;
        double fcoll_MINI = 0;

#pragma omp for reduction(+ : ave_sfrd_buf, ave_sfrd_buf_mini)
        for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
            curr_dens = dens_R_grid[box_ct] * zpp_growth[R_ct];
            if (astro_options_global->USE_MINI_HALOS) curr_mcrit = Mcrit_R_grid[box_ct];

            if (matter_options_global->SOURCE_MODEL == SOURCE_MODEL_E_INTEGRAL) {
                // TODO: we use below the homogeneous ACG turnover mass, because we don't have
                // the inhomogeneous reionization feedback in this module! (see
                // https://github.com/21cmfast/21cmFAST/issues/470)
                fcoll = EvaluateSFRD_Conditional(curr_dens, log10(sc->mturn_acg_homogeneous),
                                                 zpp_growth[R_ct], M_min_R[R_ct], M_max_R[R_ct],
                                                 M_max_R[R_ct], sigma_max[R_ct], sc);
                sfrd_grid[box_ct] = (1. + curr_dens) * fcoll;

                if (astro_options_global->USE_MINI_HALOS) {
                    fcoll_MINI = EvaluateSFRD_Conditional_MINI(
                        curr_dens, log10(sc->mturn_acg_homogeneous), curr_mcrit, zpp_growth[R_ct],
                        M_min_R[R_ct], M_max_R[R_ct], M_max_R[R_ct], sigma_max[R_ct], sc);
                    sfrd_grid_mini[box_ct] = (1. + curr_dens) * fcoll_MINI;
                }
            } else {
                fcoll = EvaluateFcoll_delta(curr_dens, zpp_growth[R_ct], sigma_min[R_ct],
                                            sigma_max[R_ct]);
                dfcoll = EvaluatedFcolldz(curr_dens, zpp_for_evolve_list[R_ct], sigma_min[R_ct],
                                          sigma_max[R_ct]);
                sfrd_grid[box_ct] = (1. + curr_dens) * dfcoll;
            }
            ave_sfrd_buf += fcoll;
            ave_sfrd_buf_mini += fcoll_MINI;
        }
    }
    *ave_sfrd = ave_sfrd_buf / HII_TOT_NUM_PIXELS;
    *ave_sfrd_mini = ave_sfrd_buf_mini / HII_TOT_NUM_PIXELS;

    // These functions check for allocation
    free_conditional_tables();
}

/*
    This function calculates calculates R-indpendent quantities that are useful for the computation
   of the radiation fields (x-ray heating rate, photoionization rate, lyman alpha flux, etc.). This
   is done by setting the fields in the input rad_setup.
*/
void setup_radiation_fields(float redshift, float perturbed_field_redshift,
                            RadiationFields *radiation_fields, RadiationFieldsSetup *rad_setup,
                            PerturbedField *perturbed_field, TsBox *previous_spin_temp,
                            InitialConditions *ini_boxes) {
    int R_ct;
    index_huge box_ct;

    // allocate the global arrays we always use
    if (!TsInterpArraysInitialised) {
        alloc_global_arrays();
    }
    if (astro_options_global->USE_MINI_HALOS) {
        rad_setup->ave_log10_MturnLW = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    }
    if (source_model_uses_eulerian_grids(matter_options_global->SOURCE_MODEL)) {
        rad_setup->delta_unfiltered =
            (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
        rad_setup->ave_dens = calloc(astro_params_global->N_STEP_TS, sizeof(double));
        rad_setup->mean_sfr_zpp = calloc(astro_params_global->N_STEP_TS, sizeof(double));
        if (astro_options_global->USE_MINI_HALOS) {
            rad_setup->log10_Mcrit_LW_unfiltered =
                (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
            rad_setup->min_log10_MturnLW = calloc(astro_params_global->N_STEP_TS, sizeof(double));
            rad_setup->max_log10_MturnLW = calloc(astro_params_global->N_STEP_TS, sizeof(double));
            rad_setup->mean_sfr_zpp_mini = calloc(astro_params_global->N_STEP_TS, sizeof(double));
        }
    }

    // setup the R_ct 1D arrays
    setup_z_edges(redshift);

    calculate_spectral_factors(redshift);

    // Fill the R_ct,box_ct fields
    // Since we use the average Mturn for the global tables this must be done first
    // NOTE: The filtered Mturn for the previous snapshot is used for Fcoll at ALL zpp
    //   regardless of distance from current reshift, this also goes for the averages
    if (source_model_uses_eulerian_grids(matter_options_global->SOURCE_MODEL)) {
        double log10_Mcrit_limit;
        // now that we have the sigma table we can assign the sigma arrays
        for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
            sigma_min[R_ct] = EvaluateSigma(log(M_min_R[R_ct]));
            sigma_max[R_ct] = EvaluateSigma(log(M_max_R[R_ct]));
        }
        set_scaling_constants(redshift, &rad_setup->sc, false);
        rad_setup->inverse_growth_factor_z = 1. / dicke(perturbed_field_redshift);
        // copy over to FFTW, do the forward FFTs and apply constants
        LOG_ULTRA_DEBUG("Starting unfiltered boxes.");
        prepare_filter_boxes(redshift, perturbed_field->density, ini_boxes->lowres_vcb,
                             previous_spin_temp->J_21_LW, rad_setup->delta_unfiltered,
                             rad_setup->log10_Mcrit_LW_unfiltered, &rad_setup->sc);
        LOG_ULTRA_DEBUG("Prepared unfiltered boxes.");
        // fill the filtered boxes if we are storing them all
        if (!matter_options_global->MINIMIZE_MEMORY) {
            fill_Rbox_table(delNL0, rad_setup->delta_unfiltered, R_values,
                            astro_params_global->N_STEP_TS, -1, rad_setup->inverse_growth_factor_z,
                            min_densities, rad_setup->ave_dens, max_densities);
            LOG_ULTRA_DEBUG("Filled density filtered boxes.");
            if (astro_options_global->USE_MINI_HALOS) {
                // NOTE: we are using previous_zp LW threshold for all zpp, inconsistent with
                // the halo model minimum turnover NOTE: should be zpp_max?
                log10_Mcrit_limit =
                    log10(molecular_cooling_threshold_with_feedbacks(redshift, 0., 0.));
                fill_Rbox_table(log10_Mcrit_LW, rad_setup->log10_Mcrit_LW_unfiltered, R_values,
                                astro_params_global->N_STEP_TS, log10_Mcrit_limit, 1,
                                rad_setup->min_log10_MturnLW, rad_setup->ave_log10_MturnLW,
                                rad_setup->max_log10_MturnLW);
            }
        } else {
            // we still need the average Mturn at R_ct==0 for NO_LIGHT
            // TODO: Remove this and come up with a better way to get NO_LIGHT
            if (astro_options_global->USE_MINI_HALOS) {
                fill_Rbox_table(log10_Mcrit_LW, rad_setup->log10_Mcrit_LW_unfiltered,
                                &(R_values[0]), 1, 0, 1, &rad_setup->min_log10_MturnLW[0],
                                &rad_setup->ave_log10_MturnLW[0], &rad_setup->max_log10_MturnLW[0]);
            }
        }
        LOG_DEBUG("Constructed filtered boxes.");
    } else if (astro_options_global->USE_MINI_HALOS) {
        for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
            rad_setup->ave_log10_MturnLW[R_ct] = radiation_fields->mean_log10_Mcrit_LW[R_ct];
        }
    }

    double x_e_ave_p = 0.0;
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
    {
#pragma omp for reduction(+ : x_e_ave_p)
        for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
            x_e_ave_p += previous_spin_temp->xray_ionised_fraction[box_ct];
        }
    }
    rad_setup->x_e_ave_p = x_e_ave_p / (float)HII_TOT_NUM_PIXELS;
    LOG_DEBUG("Prev Box: x_e_ave %.3e", rad_setup->x_e_ave_p);

    // this should initialise and use the global tables (given box average turnovers)
    //   and use them to give: Filling factor at zp (only used for !MASS_DEPENDENT_ZETA to get
    //   ion_eff) global SFRD at each filter radius (numerator of ST_over_PS factor)

    rad_setup->NO_LIGHT = global_reion_properties(redshift, rad_setup);

#pragma omp parallel private(box_ct) num_threads(simulation_options_global -> N_THREADS)
    {
        float xHII_call;
#pragma omp for
        for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
            xHII_call = previous_spin_temp->xray_ionised_fraction[box_ct];
            // Check if ionized fraction is within boundaries; if not, adjust to be within
            if (xHII_call > x_int_XHII[x_int_NXHII - 1] * 0.999) {
                xHII_call = x_int_XHII[x_int_NXHII - 1] * 0.999;
            } else if (xHII_call < x_int_XHII[0]) {
                xHII_call = 1.001 * x_int_XHII[0];
            }
            // these are the index and interpolation term, moved outside the R loop and stored
            // to not calculate them R times
            m_xHII_low_box[box_ct] = locate_xHII_index(xHII_call);
            inverse_val_box[box_ct] = (xHII_call - x_int_XHII[m_xHII_low_box[box_ct]]) *
                                      inverse_diff[m_xHII_low_box[box_ct]];
        }
    }
}

/*
    This function helps to calculate the radiation fields (x-ray heating rate, photoionization rate,
   lyman alpha flux, etc.). The radiation fields are all given by an integral over the past source
   emissivities. Numerically, this integral is evaluated via the trapezoidal rule, which is done by
   summing over the contributions from each redshift shell. Thus, this function calculates the
   contribution from a single redshift shell (zpp) and adds it to the total radiation fields. This
   is done by filling the arrays in RadiationFields.
*/
void accumulate_radiation_shell(float redshift, RadiationFieldsSetup *rad_setup,
                                RadiationFields *radiation_fields, int R_ct) {
    index_huge box_ct;
    double z_edge_factor, dzpp_for_evolve, zpp, xray_R_factor;
    double lya_flux_continuum_prefactor_mini = 0., lya_flux_injected_prefactor_mini = 0.,
           lya_flux_continuum_injected_prefactor_mini = 0.;
    double ave_fcoll, ave_fcoll_MINI;
    double avg_fix_term = 1.;
    double avg_fix_term_MINI = 1.;
    int R_index;
    float *delta_box_input;
    float *Mcrit_box_input = NULL;  // may be unused

    dzpp_for_evolve = dzpp_list[R_ct];
    zpp = zpp_for_evolve_list[R_ct];
    // dtdz'' dz'' -> dR for the radius sum (c included in constants)
    if (matter_options_global->SOURCE_MODEL == SOURCE_MODEL_CONST_ION_EFF)
        z_edge_factor = dzpp_for_evolve;  // uses dfcoll/dz
    else if (matter_options_global->SOURCE_MODEL == SOURCE_MODEL_E_INTEGRAL)
        z_edge_factor =
            fabs(dzpp_for_evolve * dtdz_list[R_ct]) * hubble(zpp) / astro_params_global->t_STAR;
    else
        z_edge_factor = fabs(dzpp_for_evolve * dtdz_list[R_ct]);

    xray_R_factor = pow(1 + zpp, -(astro_params_global->X_RAY_SPEC_INDEX));

    // index for grids. For Eulerian grid source models (<2), we can use a single
    // filter radius at a time, if MINIMIZE_MEMORY=True
    if (source_model_uses_eulerian_grids(matter_options_global->SOURCE_MODEL) &&
        matter_options_global->MINIMIZE_MEMORY) {
        R_index = 0;
    } else {
        R_index = R_ct;
    }

    if (source_model_uses_eulerian_grids(matter_options_global->SOURCE_MODEL)) {
        set_scaling_constants(zpp, &rad_setup->sc, false);
        if (matter_options_global->MINIMIZE_MEMORY) {
            // we call the filtering functions once here per R
            // This unnecessarily allocates and frees a fftwf box every time but surely
            // that's not a bottleneck
            fill_Rbox_table(delNL0, rad_setup->delta_unfiltered, &(R_values[R_ct]), 1, -1,
                            rad_setup->inverse_growth_factor_z, &min_densities[R_ct],
                            &rad_setup->ave_dens[R_ct], &max_densities[R_ct]);
            if (astro_options_global->USE_MINI_HALOS) {
                fill_Rbox_table(log10_Mcrit_LW, rad_setup->log10_Mcrit_LW_unfiltered,
                                &(R_values[R_ct]), 1, 0, 1, &rad_setup->min_log10_MturnLW[R_ct],
                                &rad_setup->ave_log10_MturnLW[R_ct],
                                &rad_setup->max_log10_MturnLW[R_ct]);
            }
            // get the global things we missed before
            // TODO: At the moment, inhomogeneous reionization feedback cannot be
            // accounted in SpinTemperatureBox.c,
            //      see https://github.com/21cmfast/21cmFAST/issues/470. Thus, we use
            //      the homogeneous (feedback-free) ACG turnover mass. It is important
            //      to remember to fix this when issue #470 is fixed!
            rad_setup->mean_sfr_zpp[R_ct] =
                EvaluateSFRD(zpp_for_evolve_list[R_ct], log10(rad_setup->sc.mturn_acg_homogeneous),
                             &rad_setup->sc);
            if (astro_options_global->USE_MINI_HALOS) {
                rad_setup->mean_sfr_zpp_mini[R_ct] = EvaluateSFRD_MINI(
                    zpp_for_evolve_list[R_ct], log10(rad_setup->sc.mturn_acg_homogeneous),
                    rad_setup->ave_log10_MturnLW[R_ct], &rad_setup->sc);
            }
            // fill one row of the interp tables
            fill_freqint_tables(redshift, rad_setup->x_e_ave_p, rad_setup->Q_HI_zp,
                                rad_setup->ave_log10_MturnLW, R_ct, &rad_setup->sc);
        }
        // set input pointers (doing things this way helps with flag flexibility)
        delta_box_input = delNL0[R_index];
        if (astro_options_global->USE_MINI_HALOS) {
            Mcrit_box_input = log10_Mcrit_LW[R_index];
        }
        calculate_sfrd_from_grid(R_ct, delta_box_input, Mcrit_box_input, del_fcoll_Rct,
                                 del_fcoll_Rct_MINI, &ave_fcoll, &ave_fcoll_MINI, &rad_setup->sc);
        avg_fix_term = rad_setup->mean_sfr_zpp[R_ct] / ave_fcoll;
        if (astro_options_global->USE_MINI_HALOS)
            avg_fix_term_MINI = rad_setup->mean_sfr_zpp_mini[R_ct] / ave_fcoll_MINI;

#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
        ScalingConstants sc_sfrd;
        sc_sfrd = evolve_scaling_constants_sfr(&sc);
        LOG_SUPER_DEBUG(
            "z %6.2f ave sfrd val %.3e global %.3e (int %.3e) Mmin %.3e ratio %.4e "
            "z_edge "
            "%.4e",
            zpp_for_evolve_list[R_ct], ave_fcoll, rad_setup->mean_sfr_zpp[R_ct],
            Nion_General(zpp_for_evolve_list[R_ct], log(M_min_R[R_ct]), log(M_MAX_INTEGRAL),
                         sc_sfrd.mturn_acg_homogeneous, &sc_sfrd),
            M_min_R[R_ct], avg_fix_term, z_edge_factor);
        if (astro_options_global->USE_MINI_HALOS) {
            LOG_SUPER_DEBUG(
                "MINI sfrd val %.3e global %.3e (int %.3e) ratio %.3e log10McritLW "
                "%.3e",
                ave_fcoll_MINI, mean_sfr_zpp_mini[R_ct],
                Nion_General_MINI(zpp_for_evolve_list[R_ct], log(M_min_R[R_ct]),
                                  log(M_MAX_INTEGRAL), sc_sfrd.mturn_acg_homogeneous,
                                  pow(10., ave_log10_MturnLW[R_ct]), &sc_sfrd),
                avg_fix_term_MINI, ave_log10_MturnLW[R_ct]);
        }
#endif
    }

    // minihalo factors should be separated since they may not be allocated
    if (astro_options_global->USE_MINI_HALOS) {
        if (astro_options_global->USE_LYA_HEATING) {
            lya_flux_continuum_prefactor_mini = lya_flux_continuum_prefactor_MINI[R_ct];
            lya_flux_injected_prefactor_mini = lya_flux_injected_prefactor_MINI[R_ct];
        } else {
            lya_flux_continuum_injected_prefactor_mini =
                lya_flux_continuum_injected_prefactor_MINI[R_ct];
        }
    }

// NOTE: The ionisation box has a final delta dependence of (1+delta_source)/(1+delta_absorber)
//   But here it's just (1+delta_source). This is for photon conservation.
//   If we assume attenuation at mean density as we do in nu_tau_one(), we HAVE to assume mean
//   density absorption otherwise we do not conserve photons
#pragma omp parallel private(box_ct) num_threads(simulation_options_global -> N_THREADS)
    {
        // private variables
        int xidx;
        double ival;
        double freq_int_heat, freq_int_ion, freq_int_lya;
        double sfr_term, xray_sfr;
        double sfr_term_mini = 0;
        double sfr_term_lw, sfr_term_mini_lw;
#pragma omp for
        for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
            // sum each R contribution together
            // The dxdt boxes exist for two reasons. Firstly it allows the
            // MINIMIZE_MEMORY to work (replaces ~40*NUM_PIXELS with ~4-16*NUM_PIXELS),
            //   as the FFT is done in the R-loop.
            // Secondly, it is *likely* faster to fill these boxes, and sum with a outer
            // R loop than an inner one.

            if (source_model_uses_lagrangian_grids(matter_options_global->SOURCE_MODEL)) {
                sfr_term = radiation_fields->filtered_sfr[box_ct] * z_edge_factor;
                // Minihalos and s->yr conversion are already included here
                xray_sfr =
                    radiation_fields->filtered_xray[box_ct] * z_edge_factor * xray_R_factor * 1e38;
                if (astro_options_global->USE_MINI_HALOS &&
                    astro_options_global->LYA_MULTIPLE_SCATTERING) {
                    sfr_term_lw = radiation_fields->filtered_sfr_lw[box_ct] * z_edge_factor;
                } else {
                    sfr_term_lw = sfr_term;
                }
            } else {
                // NOTE: for SOURCE_MODEL==0 F_STAR10 is still used for constant
                // stellar fraction
                sfr_term = del_fcoll_Rct[box_ct] * z_edge_factor * avg_fix_term *
                           astro_params_global->F_STAR10;
                xray_sfr = sfr_term * astro_params_global->L_X * xray_R_factor * physconst.s_per_yr;
            }
            if (astro_options_global->USE_MINI_HALOS) {
                if (source_model_uses_lagrangian_grids(matter_options_global->SOURCE_MODEL)) {
                    sfr_term_mini = radiation_fields->filtered_sfr_mini[box_ct] * z_edge_factor;
                    if (astro_options_global->LYA_MULTIPLE_SCATTERING) {
                        sfr_term_mini_lw =
                            radiation_fields->filtered_sfr_mini_lw[box_ct] * z_edge_factor;
                    } else {
                        sfr_term_mini_lw = sfr_term_mini;
                    }
                } else {
                    sfr_term_mini = del_fcoll_Rct_MINI[box_ct] * z_edge_factor * avg_fix_term_MINI *
                                    astro_params_global->F_STAR7_MINI;
                    xray_sfr += sfr_term_mini * astro_params_global->L_X_MINI * xray_R_factor *
                                physconst.s_per_yr;
                    sfr_term_lw = sfr_term;
                    sfr_term_mini_lw = sfr_term_mini;
                }
            }

            // Evaluate the frequency integrals for this shell (R_ct) and cell (box_ct) via
            // linear interpolation
            xidx = m_xHII_low_box[box_ct];
            ival = inverse_val_box[box_ct];
            freq_int_heat =
                freq_int_heat_tbl_diff[xidx][R_ct] * ival + freq_int_heat_tbl[xidx][R_ct];
            freq_int_ion = freq_int_ion_tbl_diff[xidx][R_ct] * ival + freq_int_ion_tbl[xidx][R_ct];
            freq_int_lya = freq_int_lya_tbl_diff[xidx][R_ct] * ival + freq_int_lya_tbl[xidx][R_ct];

            // Evaluate the radiation fields by adding the contribution from this shell
            // (R_ct) This implements trapezoidal integration over the shells
            if (astro_options_global->USE_X_RAY_HEATING) {
                radiation_fields->xray_heating_rate[box_ct] += xray_sfr * freq_int_heat;
            }
            radiation_fields->xray_ionization_rate[box_ct] += xray_sfr * freq_int_ion;
            radiation_fields->xray_lya_flux[box_ct] += xray_sfr * freq_int_lya;
            if (astro_options_global->USE_MINI_HALOS) {
                radiation_fields->lyw_flux[box_ct] +=
                    sfr_term_lw * lyw_flux_prefactor[R_ct] +
                    sfr_term_mini_lw * lyw_flux_prefactor_MINI[R_ct];
            }
            if (astro_options_global->USE_LYA_HEATING) {
                radiation_fields->lya_flux_continuum[box_ct] +=
                    sfr_term * lya_flux_continuum_prefactor[R_ct] +
                    sfr_term_mini * lya_flux_continuum_prefactor_mini;
                radiation_fields->lya_flux_injected[box_ct] +=
                    sfr_term * lya_flux_injected_prefactor[R_ct] +
                    sfr_term_mini * lya_flux_injected_prefactor_mini;
            } else {
                radiation_fields->lya_flux_continuum_injected[box_ct] +=
                    sfr_term * lya_flux_continuum_injected_prefactor[R_ct] +
                    sfr_term_mini * lya_flux_continuum_injected_prefactor_mini;
            }

            // I cannot check the integral if we are using the halo field since delNL0
            // (filtered density) is not calculated
#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
            if (box_ct == 0 &&
                source_model_uses_eulerian_grids(matter_options_global->SOURCE_MODEL)) {
                double integral_db;
                if (matter_options_global->SOURCE_MODEL == SOURCE_MODEL_E_INTEGRAL) {
                    integral_db =
                        Nion_ConditionalM(zpp_growth[R_ct], log(M_min_R[R_ct]), log(M_max_R[R_ct]),
                                          M_max_R[R_ct], sigma_max[R_ct],
                                          delNL0[R_index][box_ct] * zpp_growth[R_ct],
                                          sc_sfrd.mturn_acg_homogeneous, &sc_sfrd,
                                          astro_options_global->INTEGRATION_METHOD_ATOMIC) *
                        z_edge_factor * (1 + delNL0[R_index][box_ct] * zpp_growth[R_ct]) *
                        avg_fix_term * astro_params_global->F_STAR10;
                } else {
                    integral_db =
                        dfcoll_dz(zpp_for_evolve_list[R_ct], sigma_min[R_ct],
                                  delNL0[R_index][box_ct] * zpp_growth[R_ct], sigma_max[R_ct]) *
                        z_edge_factor * (1 + delNL0[R_index][box_ct] * zpp_growth[R_ct]) *
                        avg_fix_term * astro_params_global->F_STAR10;
                }

                LOG_SUPER_DEBUG("Cell 0: R=%.1f (%.3f) | SFR %.4e | integral %.4e | delta %.4e",
                                R_values[R_ct], zpp_for_evolve_list[R_ct], sfr_term, integral_db,
                                delNL0[R_index][box_ct]);
                if (astro_options_global->USE_MINI_HALOS)
                    LOG_SUPER_DEBUG(
                        "MINI SFR %.4e | integral %.4e", sfr_term_mini,
                        Nion_ConditionalM_MINI(
                            zpp_growth[R_ct], log(M_min_R[R_ct]), log(M_max_R[R_ct]),
                            log(M_max_R[R_ct]), sigma_max[R_ct],
                            delNL0[R_index][box_ct] * zpp_growth[R_ct],
                            sc_sfrd.mturn_acg_homogeneous, pow(10, log10_Mcrit_LW[R_ct][box_ct]),
                            &sc_sfrd, astro_options_global->INTEGRATION_METHOD_MINI) *
                            z_edge_factor * (1 + delNL0[R_index][box_ct] * zpp_growth[R_ct]) *
                            avg_fix_term_MINI * astro_params_global->F_STAR7_MINI);

                if (astro_options_global->USE_X_RAY_HEATING) {
                    LOG_SUPER_DEBUG(
                        "xh %.2e | xi %.2e | xl %.2e",
                        radiation_fields->xray_heating_rate[box_ct] / astro_params_global->L_X,
                        radiation_fields->xray_ionization_rate[box_ct] / astro_params_global->L_X,
                        radiation_fields->xray_lya_flux[box_ct] / astro_params_global->L_X);
                } else {
                    LOG_SUPER_DEBUG(
                        "xi %.2e | xl %.2e",
                        radiation_fields->xray_ionization_rate[box_ct] / astro_params_global->L_X,
                        radiation_fields->xray_lya_flux[box_ct] / astro_params_global->L_X);
                }

                if (astro_options_global->USE_LYA_HEATING) {
                    LOG_SUPER_DEBUG("ct %.2e | ij %.2e",
                                    radiation_fields->lya_flux_continuum[box_ct],
                                    radiation_fields->lya_flux_injected[box_ct]);
                } else {
                    LOG_SUPER_DEBUG("sl %.2e",
                                    radiation_fields->lya_flux_continuum_injected[box_ct]);
                }
            }
#endif
        }  // end of box_ct loop
    }  // end of pragma loop
}

/*
    This function multiplies the radiation fields by the appropriate constants to convert them to
   physical meaningful quantities. The radiation fields are all given by an integral over the past
   source emissivities. Numerically, this integral is evaluated via the trapezoidal rule, which is
   done by summing over the contributions from each redshift shell. For efficiency, the constants
   are not included in the integral, but rather multiplied at the end (outside the integral). This
   function does that multiplication.
*/
void multiply_radiation_fields_by_constants(float redshift, RadiationFields *radiation_fields,
                                            float perturbed_field_redshift,
                                            PerturbedField *perturbed_field,
                                            TsBox *previous_spin_temp) {
    double luminosity_converstion_factor, xray_prefactor, volunit_inv, Nb_zp, lya_star_prefactor;
    double growth_factor_z, growth_factor_zp, inverse_growth_factor_z;

    if (fabs(astro_params_global->X_RAY_SPEC_INDEX - 1.0) < 1e-6) {
        luminosity_converstion_factor =
            (astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz *
            log(astro_params_global->NU_X_BAND_MAX / (astro_params_global->NU_X_THRESH));
        luminosity_converstion_factor = 1. / luminosity_converstion_factor;
    } else {
        luminosity_converstion_factor =
            pow((astro_params_global->NU_X_BAND_MAX) * physconst.eV_to_Hz,
                1. - (astro_params_global->X_RAY_SPEC_INDEX)) -
            pow((astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz,
                1. - (astro_params_global->X_RAY_SPEC_INDEX));
        luminosity_converstion_factor = 1. / luminosity_converstion_factor;
        luminosity_converstion_factor *=
            pow((astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz,
                -(astro_params_global->X_RAY_SPEC_INDEX)) *
            (1 - (astro_params_global->X_RAY_SPEC_INDEX));
    }
    // Finally, convert to the correct units. physconst.eV_to_Hz*physconst.h_p as only want to
    // divide by eV -> erg (owing to the definition of Luminosity)
    luminosity_converstion_factor /= (physconst.h_p);

    // for halos, we just want the SFR -> X-ray part
    // NOTE: compared to Mesinger+11: (1+zpp)^2 (1+zp) -> (1+zp)^3
    //(1+z)^3 is here because we don't want it in the
    // star lya (already in zpp integrand)
    xray_prefactor = luminosity_converstion_factor /
                     ((astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz) * physconst.c_cms *
                     pow(1 + redshift, astro_params_global->X_RAY_SPEC_INDEX + 3);
    Nb_zp = N_b0 * (1 + redshift) * (1 + redshift) * (1 + redshift);
    // converts SFR density -> stellar baryon density + prefactors
    lya_star_prefactor = physconst.c_cms / (4.0 * M_PI) * physconst.Msun / physconst.m_p *
                         (1 - 0.75 * cosmo_params_global->Y_He);

    growth_factor_z = dicke(perturbed_field_redshift);
    inverse_growth_factor_z = 1. / growth_factor_z;
    growth_factor_zp = dicke(redshift);

    // converts the grid emissivity unit to per cm-3
    if (source_model_uses_lagrangian_grids(matter_options_global->SOURCE_MODEL)) {
        volunit_inv = pow(physconst.cm_per_Mpc, -3);
    } else {
        volunit_inv = cosmo_params_global->OMb * RHOcrit * pow(physconst.cm_per_Mpc, -3);
    }

    index_huge box_ct;
#pragma omp parallel private(box_ct) num_threads(simulation_options_global -> N_THREADS)
    {
        double curr_delta, prev_xe;
#pragma omp for
        for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
            curr_delta =
                perturbed_field->density[box_ct] * growth_factor_zp * inverse_growth_factor_z;
            prev_xe = previous_spin_temp->xray_ionised_fraction[box_ct];
            if (astro_options_global->USE_X_RAY_HEATING) {
                radiation_fields->xray_heating_rate[box_ct] *=
                    xray_prefactor * volunit_inv * 2.0 / 3.0 / physconst.k_B / (1.0 + prev_xe);
                ;
            }
            radiation_fields->xray_ionization_rate[box_ct] *= xray_prefactor * volunit_inv;
            radiation_fields->xray_lya_flux[box_ct] *=
                xray_prefactor * volunit_inv * Nb_zp * (1 + curr_delta);
            if (astro_options_global->USE_MINI_HALOS) {
                radiation_fields->lyw_flux[box_ct] *=
                    lya_star_prefactor * volunit_inv * physconst.h_p * 1e21;
            }
            if (astro_options_global->USE_LYA_HEATING) {
                radiation_fields->lya_flux_continuum[box_ct] *= lya_star_prefactor * volunit_inv;
                radiation_fields->lya_flux_injected[box_ct] *= lya_star_prefactor * volunit_inv;
            } else {
                radiation_fields->lya_flux_continuum_injected[box_ct] *=
                    lya_star_prefactor * volunit_inv;
            }
        }
    }
}

void free_rad_setup(RadiationFieldsSetup *rad_setup, short cleanup) {
    if (astro_options_global->USE_MINI_HALOS) {
        free(rad_setup->ave_log10_MturnLW);
    }
    if (source_model_uses_eulerian_grids(matter_options_global->SOURCE_MODEL)) {
        fftwf_free(rad_setup->delta_unfiltered);
        free(rad_setup->ave_dens);
        free(rad_setup->mean_sfr_zpp);
        if (astro_options_global->USE_MINI_HALOS) {
            fftwf_free(rad_setup->log10_Mcrit_LW_unfiltered);
            free(rad_setup->min_log10_MturnLW);
            free(rad_setup->max_log10_MturnLW);
            free(rad_setup->mean_sfr_zpp_mini);
        }
    }
    free(rad_setup);

    // we definitely don't need these tables anymore
    // Having these free's here instead of after global_reion_properties just for
    // MINIMIZE_MEMORY is not ideal,
    //    but the log10Mturn average is needed
    free_global_tables();

    if (source_model_uses_eulerian_grids(matter_options_global->SOURCE_MODEL)) {
        fftwf_forget_wisdom();
        fftwf_cleanup_threads();
        fftwf_cleanup();
    }
    if (cleanup) {
        free_ts_global_arrays();
    }
}

// NOTE: I've moved this to a function to help in simplicity, it is not clear whether it is faster
//   to do all of one radii at once (more clustered FFT and larger thread blocks) or all of one box
//   (better memory locality)
// TODO: filter speed tests
void one_annular_filter(float *input_box, float *output_box, double R_inner, double R_outer,
                        double R_star, int filter_type, double *u_avg, double *f_avg) {
    int i, j, k;
    index_huge ct;
    double unfiltered_avg = 0;
    double filtered_avg = 0;
    int box_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                      HII_D_PARA};

    fftwf_complex *unfiltered_box =
        (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
    fftwf_complex *filtered_box =
        (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS) \
    reduction(+ : unfiltered_avg)
    {
        float curr_val;
        index_huge index_r, index_f;
#pragma omp for
        for (i = 0; i < box_dim[0]; i++) {
            for (j = 0; j < box_dim[1]; j++) {
                for (k = 0; k < box_dim[2]; k++) {
                    index_r = grid_index_general(i, j, k, box_dim);
                    index_f = grid_index_fftw_r(i, j, k, box_dim);
                    curr_val = input_box[index_r];
                    *((float *)unfiltered_box + index_f) = curr_val;
                    unfiltered_avg += curr_val;
                }
            }
        }
    }
    // No need to filter the box if we only have one cell!
    if (simulation_options_global->HII_DIM > 1) {
        // Transform unfiltered box to k-space to prepare for filtering
        // this would normally only be done once but we're using a different redshift for each R now
        dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                     HII_D_PARA, simulation_options_global->N_THREADS, unfiltered_box);

// remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from real space to k-space
// Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
        {
#pragma omp for
            for (ct = 0; ct < HII_KSPACE_NUM_PIXELS; ct++) {
                unfiltered_box[ct] /= (float)HII_TOT_NUM_PIXELS;
            }
        }

        // Smooth the density field, at the same time store the minimum and maximum densities for
        // their usage in the interpolation tables copy over unfiltered box
        memcpy(filtered_box, unfiltered_box, sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

        // Don't filter on the cell scale
        if (R_inner > 0) {
            filter_box(filtered_box, box_dim, filter_type, R_inner, R_outer, R_star);
        }

        // now fft back to real space
        dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                     HII_D_PARA, simulation_options_global->N_THREADS, filtered_box);
    }
// copy over the values
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS) \
    reduction(+ : filtered_avg)
    {
        float curr_val;
        index_huge index_f, index_r;
#pragma omp for
        for (i = 0; i < box_dim[0]; i++) {
            for (j = 0; j < box_dim[1]; j++) {
                for (k = 0; k < box_dim[2]; k++) {
                    index_r = grid_index_general(i, j, k, box_dim);
                    index_f = grid_index_fftw_r(i, j, k, box_dim);
                    if (simulation_options_global->HII_DIM > 1) {
                        curr_val = *((float *)filtered_box + index_f);
                    } else {  // Just take the unfiltered box/cell if HII_DIM = 1
                        curr_val = *((float *)unfiltered_box + index_f);
                    }
                    // correct for aliasing in the filtering step
                    if (curr_val < 0.) curr_val = 0.;

                    output_box[index_r] = curr_val;
                    filtered_avg += curr_val;
                }
            }
        }
    }

    unfiltered_avg /= HII_TOT_NUM_PIXELS;
    filtered_avg /= HII_TOT_NUM_PIXELS;

    *u_avg = unfiltered_avg;
    *f_avg = filtered_avg;

    fftwf_free(filtered_box);
    fftwf_free(unfiltered_box);
}

int UpdateRadiationFields(float redshift, HaloBox *halobox, double R_inner, double R_outer,
                          int R_ct, double R_star, short mode, short cleanup,
                          float perturbed_field_redshift, PerturbedField *perturbed_field,
                          TsBox *previous_spin_temp, InitialConditions *ini_boxes,
                          RadiationFields *radiation_fields) {
    int status, filter_type;
    Try {
        // If the redshift is above Z_HEAT_MAX, we skip the calculation of the radiation fields and
        // return early. Note that setup_radiation_fields below requires previous_spin_temp, and it
        // must be evaluated at least once for redshift > Z_HEAT_MAX, (as the highest node redshift
        // cannot be below Z_HEAT_MAX) so by the time redshift < Z_HEAT_MAX we should have a valid
        // previous_spin_temp to pass to setup_radiation_fields. I think this is already protected
        // at the python level, but it's still good to have this guard here as well.
        if (redshift >= simulation_options_global->Z_HEAT_MAX) {
            LOG_DEBUG("Redshift %.3f is above Z_HEAT_MAX %.3f, skipping radiation field setup.",
                      redshift, simulation_options_global->Z_HEAT_MAX);
            return 0;
        }

        // We need to do some setup for the radiation fields, before we integrate contributions from
        // shells.
        if (mode == UPDATE_RADIATION_FIELDS_SETUP) {
            rad_setup = malloc(sizeof(RadiationFieldsSetup));
            setup_radiation_fields(redshift, perturbed_field_redshift, radiation_fields, rad_setup,
                                   perturbed_field, previous_spin_temp, ini_boxes);
            radiation_fields->Q_HI = rad_setup->Q_HI_zp;
        }

        // Multiply the radiation fields by constants and free the rad_setup struct
        else if (mode == UPDATE_RADIATION_FIELDS_CLEANUP) {
            if (!rad_setup->NO_LIGHT) {
                multiply_radiation_fields_by_constants(redshift, radiation_fields,
                                                       perturbed_field_redshift, perturbed_field,
                                                       previous_spin_temp);
            }
            free_rad_setup(rad_setup, cleanup);
            rad_setup = NULL;
        }

        // If we are in the evaluation mode, we calculate the contribution from this shell to the
        // radiation fields
        else if (mode == UPDATE_RADIATION_FIELDS_EVAL) {
            // If there are no stars, skip the calculation below
            if (!rad_setup->NO_LIGHT) {
                filter_type = astro_options_global->LYA_MULTIPLE_SCATTERING
                                  ? FILTER_SPHERICAL_SHELL_MULTIPLE_SCATTERING
                                  : FILTER_SPHERICAL_SHELL_STRAIGHT_LINE;

                // only print once, since this is called for every R
                if (R_ct == 0) LOG_DEBUG("starting RadiationFields");

                double sfr_avg, fsfr_avg, sfr_avg_mini = 0., fsfr_avg_mini = 0.;
                double xray_avg, fxray_avg;
                one_annular_filter(halobox->halo_sfr, radiation_fields->filtered_sfr, R_inner,
                                   R_outer, R_star, filter_type, &sfr_avg, &fsfr_avg);
                one_annular_filter(halobox->halo_xray, radiation_fields->filtered_xray, R_inner,
                                   R_outer, R_star, FILTER_SPHERICAL_SHELL_STRAIGHT_LINE, &xray_avg,
                                   &fxray_avg);
                if (astro_options_global->USE_MINI_HALOS) {
                    one_annular_filter(halobox->halo_sfr_mini, radiation_fields->filtered_sfr_mini,
                                       R_inner, R_outer, R_star, filter_type, &sfr_avg_mini,
                                       &fsfr_avg_mini);
                    // In case of multiple scattering and mini-halos, we need to filter the SFRD
                    // fields again for the the LW feedback, as these photons travel in straight
                    // lines
                    if (astro_options_global->LYA_MULTIPLE_SCATTERING) {
                        one_annular_filter(
                            halobox->halo_sfr, radiation_fields->filtered_sfr_lw, R_inner, R_outer,
                            R_star, FILTER_SPHERICAL_SHELL_STRAIGHT_LINE, &sfr_avg, &fsfr_avg);
                        one_annular_filter(halobox->halo_sfr_mini,
                                           radiation_fields->filtered_sfr_mini_lw, R_inner, R_outer,
                                           R_star, FILTER_SPHERICAL_SHELL_STRAIGHT_LINE,
                                           &sfr_avg_mini, &fsfr_avg_mini);
                    }
                }

                if (R_ct == astro_params_global->N_STEP_TS - 1)
                    LOG_DEBUG("finished RadiationFields");

                LOG_SUPER_DEBUG(
                    "R = [%8.3f - %8.3f] | mean filtered sfr  = %10.3e unfiltered %10.3e", R_inner,
                    R_outer, fsfr_avg, sfr_avg);
                LOG_ULTRA_DEBUG("mean filtered xray = %10.3e unfiltered %10.3e", fxray_avg,
                                xray_avg);
                if (astro_options_global->USE_MINI_HALOS) {
                    LOG_SUPER_DEBUG(
                        "MINI: filtered sfr %10.3e unfiltered %10.3e log10_Mcrit_LW = %10.3e",
                        fsfr_avg_mini, sfr_avg_mini, radiation_fields->mean_log10_Mcrit_LW[R_ct]);
                }

                // Given the filtered emissivities, we accumulate the contribution of this shell to
                // the radiation fields
                accumulate_radiation_shell(redshift, rad_setup, radiation_fields, R_ct);

                // free fftwf only if we have a full box (with more than one cell)
                // TODO: Should we call the following at every shell, or only after
                // UpdateRadiationFields is no longer called in this snapshot?
                if (simulation_options_global->HII_DIM > 1) {
                    fftwf_forget_wisdom();
                    fftwf_cleanup_threads();
                    fftwf_cleanup();
                }
            }
        } else {
            LOG_ERROR("Invalid mode %d passed to UpdateRadiationFields", mode);
            Throw(ValueError);
        }
    }  // End of try
    Catch(status) { return (status); }
    return (0);
}
