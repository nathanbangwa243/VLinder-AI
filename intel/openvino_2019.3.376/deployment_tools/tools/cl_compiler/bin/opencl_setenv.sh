#!/bin/bash

# ---------------------------------------------------------------------------
#  Copyright (C) 2019 Intel Ireland Ltd. All rights reserved
#  ---------------------------------------------------------------------------
source_name=$( readlink -f ${BASH_SOURCE})
SHAVE_CL_INSTALL=${source_name%/*}/..
echo $SHAVE_CL_INSTALL
export SHAVE_LDSCRIPT_DIR="$SHAVE_CL_INSTALL/ldscripts/"
export SHAVE_MA2X8XLIBS_DIR="$SHAVE_CL_INSTALL/lib"
export SHAVE_MOVIASM_DIR="$SHAVE_CL_INSTALL/bin"
export SHAVE_MYRIAD_LD_DIR="$SHAVE_CL_INSTALL/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$SHAVE_CL_INSTALL/lib"
