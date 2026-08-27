#ifndef ROCK_TOSA_CUSTOM_OPS_H
#define ROCK_TOSA_CUSTOM_OPS_H

// Header to define attributes used by TOSA operations representing rocMLIR
// semantics.

#define ROCK_CUSTOMOP_DOMAIN_NAME "rocmlir"
#define ROCK_ATTR_NO_SIGNED_ZEROS "rock.no_signed_zeros"

// For consistency, this should match ConvOpBwdDataType in RockAttrDefs.td
#define ROCK_CUSTOMOP_CONV_BWD_DATA "conv_bwd_data"
#define ROCK_CUSTOMOP_UNSIGNED_DIV "unsigned_div"
#define ROCK_CUSTOMOP_UNSIGNED_MAX "unsigned_max"
#define ROCK_CUSTOMOP_UNSIGNED_CAST "unsigned_cast"
#define ROCK_CUSTOMOP_EXPAND_STRIDES "expand_strides"

#endif // ROCK_TOSA_CUSTOM_OPS_H
