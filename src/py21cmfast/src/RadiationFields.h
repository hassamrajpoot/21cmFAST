#ifndef _RADIATIONFIELDS_H
#define _RADIATIONFIELDS_H

#include <complex.h>
#include <fftw3.h>

#include "OutputStructs.h"
#include "scaling_relations.h"

#define UPDATE_RADIATION_FIELDS_SETUP 0
#define UPDATE_RADIATION_FIELDS_EVAL 1
#define UPDATE_RADIATION_FIELDS_CLEANUP 2

int UpdateRadiationFields(float redshift, HaloBox *halobox, double R_inner, double R_outer,
                          int R_ct, double R_star, short mode, short cleanup,
                          float perturbed_field_redshift, PerturbedField *perturbed_field,
                          TsBox *previous_spin_temp, InitialConditions *ini_boxes,
                          RadiationFields *radiation_fields);

typedef struct RadiationFieldsSetup {
    double *ave_log10_MturnLW;
    double inverse_growth_factor_z;
    double x_e_ave_p;
    double Q_HI_zp;
    int NO_LIGHT;
    // TODO: remove these fields below when https://github.com/21cmfast/21cmFAST/issues/668 is fixed
    // (they are needed only for the Eulerian source model)
    ScalingConstants sc;
    fftwf_complex *delta_unfiltered;
    fftwf_complex *log10_Mcrit_LW_unfiltered;
    double *ave_dens;
    double *min_log10_MturnLW;
    double *max_log10_MturnLW;
    double *mean_sfr_zpp;
    double *mean_sfr_zpp_mini;
} RadiationFieldsSetup;

void setup_radiation_fields(float redshift, float perturbed_field_redshift,
                            RadiationFields *radiation_fields, RadiationFieldsSetup *rad_setup,
                            PerturbedField *perturbed_field, TsBox *previous_spin_temp,
                            InitialConditions *ini_boxes);

void accumulate_radiation_shell(float redshift, RadiationFieldsSetup *rad_setup,
                                RadiationFields *radiation_fields, int R_ct);

void multiply_radiation_fields_by_constants(float redshift, RadiationFields *radiation_fields,
                                            float perturbed_field_redshift,
                                            PerturbedField *perturbed_field,
                                            TsBox *previous_spin_temp);

void free_rad_setup(RadiationFieldsSetup *rad_setup, short cleanup);

#endif
