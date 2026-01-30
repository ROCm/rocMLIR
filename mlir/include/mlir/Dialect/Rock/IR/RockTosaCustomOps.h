#ifndef ROCK_TOSA_CUSTOM_OPS_H
#define ROCK_TOSA_CUSTOM_OPS_H

// Header to define "domain_name" and "operator_name"s attributes for
// tosa::CustomOp representing rocMLIR operations.

#define ROCK_CUSTOMOP_DOMAIN_NAME "rocmlir"

// For consistency, this should match ConvOpBwdDataType in RockAttrDefs.td
#define ROCK_CUSTOMOP_CONV_BWD_DATA "conv_bwd_data"
#define ROCK_CUSTOMOP_CONV_BWD_WEIGHT "conv_bwd_weight"
#define ROCK_CUSTOMOP_UNSIGNED_DIV "unsigned_div"
#define ROCK_CUSTOMOP_UNSIGNED_CAST "unsigned_cast"
#define ROCK_CUSTOMOP_DEREF "deref"

#endif // ROCK_TOSA_CUSTOM_OPS_H
