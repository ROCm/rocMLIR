#ifndef COMMON_H
#define COMMON_H

// The ArgumentFields keep track of differences between conv operations
struct ArgumentFields {
  int gridwiseGemmArgumentPosition[3];
  StringRef gemmTargetCharName[3];
};

void affixGridwiseGemmAttributes(Operation *convOp, Operation *gop,
                                 OpBuilder &b);
#endif
