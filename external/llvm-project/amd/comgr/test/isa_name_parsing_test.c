/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void parseIsaName(amd_comgr_action_info_t DataAction, const char *IsaName,
                  amd_comgr_status_t ExpectedStatus) {
  amd_comgr_status_t TrueStatus =
      amd_comgr_action_info_set_isa_name(DataAction, IsaName);
  if (TrueStatus != ExpectedStatus) {
    amd_comgr_status_t Status;
    const char *TrueStatusString, *ExpectedStatusString;
    Status = amd_comgr_status_string(TrueStatus, &TrueStatusString);
    checkError(Status, "amd_comgr_status_string");
    Status = amd_comgr_status_string(ExpectedStatus, &ExpectedStatusString);
    checkError(Status, "amd_comgr_status_string");
    printf("Parsing \"%s\" resulted in \"%s\"; expected \"%s\"\n", IsaName,
           TrueStatusString, ExpectedStatusString);
    exit(1);
  }
}

int main(int argc, char *argv[]) {
  amd_comgr_status_t Status;
  amd_comgr_action_info_t dataAction;

  Status = amd_comgr_create_action_info(&dataAction);
  checkError(Status, "amd_comgr_create_action_info");

#define PARSE_VALID_ISA_NAME(name)                                             \
  parseIsaName(dataAction, name, AMD_COMGR_STATUS_SUCCESS)
#define PARSE_INVALID_ISA_NAME(name)                                           \
  parseIsaName(dataAction, name, AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)

  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx803");
  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx801:xnack+");
  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx801:xnack-");
  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx908:sramecc+");
  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx908:sramecc-");
  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx908:xnack+:sramecc+");
  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx908:xnack-:sramecc+");
  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx908:xnack-:sramecc-");

  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx1010:xnack+");
  PARSE_VALID_ISA_NAME("");
  PARSE_VALID_ISA_NAME(NULL);

  PARSE_INVALID_ISA_NAME("amdgcn-amd-amdhsa--gfx801:xnack+:sramecc+");
  PARSE_INVALID_ISA_NAME("amdgcn-amd-amdhsa--gfx803:::");
  PARSE_INVALID_ISA_NAME("amdgcn-amd-amdhsa-opencl-gfx803");
  PARSE_INVALID_ISA_NAME("amdgcn-amd-amdhsa-gfx803");
  PARSE_INVALID_ISA_NAME("gfx803");
  PARSE_INVALID_ISA_NAME(" amdgcn-amd-amdhsa--gfx803");
  PARSE_INVALID_ISA_NAME(" amdgcn-amd-amdhsa--gfx803 ");
  PARSE_INVALID_ISA_NAME("amdgcn-amd-amdhsa--gfx803 ");
  PARSE_INVALID_ISA_NAME("   amdgcn-amd-amdhsa--gfx803  ");

  Status = amd_comgr_destroy_action_info(dataAction);
  checkError(Status, "amd_comgr_destroy_action_info");
}
