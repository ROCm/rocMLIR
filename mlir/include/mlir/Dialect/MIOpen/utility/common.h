#ifndef COMMON_H
#define COMMON_H

// The ArgumentFields keep track of differences between conv operations
struct ArgumentFields {
  int gridwiseGemmArgumentPosition[3];
  StringRef gemmTargetCharName[3];
};

template <typename T, typename U>
void affixGridwiseGemmAttributes(T &convOp, U &gop, OpBuilder &b);

#endif
