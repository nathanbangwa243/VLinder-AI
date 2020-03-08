#!/bin/bash

# Copyright (2014-2018) Intel Corporation All Rights Reserved.
#
# The source code, information and material ("Material") contained
# herein is owned by Intel Corporation or its suppliers or licensors,
# and title to such Material remains with Intel Corporation or its
# suppliers or licensors. The Material contains proprietary information
# of Intel or its suppliers and licensors. The Material is protected by
# worldwide copyright laws and treaty provisions. No part of the
# Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed or disclosed in any way
# without Intel's prior express written permission. No license under any
# patent, copyright or other intellectual property rights in the
# Material is granted to or conferred upon you, either expressly, by
# implication, inducement, estoppel or otherwise. Any license under such
# intellectual property rights must be express and approved by Intel in
# writing.
#
# Unless otherwise agreed by Intel in writing, you may not remove or alter
# this notice or any other notice embedded in Materials by Intel or Intel's
# suppliers or licensors in any way.

# Set Bash color
ECHO_PREFIX_INFO="\033[1;32;40mINFO...\033[0;0m"
ECHO_PREFIX_ERROR="\033[1;31;40mError...\033[0;0m"

# Try command  for test command result.
function try_command {
    "$@"
    status=$?
    if [ $status -ne 0 ]; then
        echo -e $ECHO_PREFIX_ERROR "ERROR with \"$@\", Return status $status."
        exit $status
    fi
    return $status
}


# This script must be run as root
if [[ $EUID -ne 0 ]]; then
    echo -e $ECHO_PREFIX_ERROR "This script must be run as root!" 1>&2
    exit 1
fi


#detect system arch.
ULONG_MASK=`getconf ULONG_MAX`
if [ $ULONG_MASK == 18446744073709551615 ]; then
    SYSARCH=64
else
    echo -e $ECHO_PREFIX_ERROR "This package does not support 32-bit system.\n"
    exit 1
fi


try_command lsb_release -si > /dev/null

# dectect OS version
LINUX_DISTRO=`lsb_release -si`

if [ "$LINUX_DISTRO" == "CentOS" ]; then
    LINUX_DISTRO="CentOS"
else if [ "$LINUX_DISTRO" == "RedHatEnterpriseServer" ]; then
    LINUX_DISTRO="RHEL-Server"
# else
#    echo -e $ECHO_PREFIX_ERROR "The package is majorly for RHEL/CentOS, directly install on $LINUX_DISTRO may be not functional. Do you still want to insall?\n"
#    read -p "press 'y' to confirm, otherwise cancelled. (Suggest to check this script and the package content for decision.)" install_force
#    if [ "$install_force" == "y" ]; then
#        echo -e $ECHO_PREFIX_INFO "Ok, will continue."
#    else
#        echo -e $ECHO_PREFIX_INFO "The installation is cancelled."
#        echo -e $ECHO_PREFIX_INFO "The umd driver within the package may still be functional on $LINUX_DISTRO, as long as gcc/glibc version meet the requirement.\n"
#        exit 1
#    fi
fi
fi
echo -e $ECHO_PREFIX_INFO "Install on $LINUX_DISTRO ..."


MEDIASDK_DIR=opt/intel/mediasdk

if [ ! -d $MEDIASDK_DIR ] && [ ! -d usr/lib64 ]; then
    echo -e $ECHO_PREFIX_ERROR "Cannot find installation content directory!"
    exit 1
fi

# Install MSDK
if [ -d $MEDIASDK_DIR ]; then
    try_command rm -fr /$MEDIASDK_DIR
    try_command mkdir -p /$MEDIASDK_DIR
    try_command cp -dfr $MEDIASDK_DIR/* /$MEDIASDK_DIR
    echo -e $ECHO_PREFIX_INFO "MediaSDK installed successfully in /$MEDIASDK_DIR!"
else
    echo -e $ECHO_PREFIX_ERROR "MediaSDK missed in this package!"
fi

# Install MDF Runtime
MDF_DIR=opt/intel/common/mdf
if [ -d $MDF_DIR ]; then
    try_command rm -fr /$MDF_DIR
    try_command mkdir -p /$MDF_DIR
    try_command cp -dfr $MDF_DIR/* /$MDF_DIR
    echo -e $ECHO_PREFIX_INFO "MDF Runtime installed successfully in /$MDF_DIR!"
else
    echo -e $ECHO_PREFIX_ERROR "MDF Runtime missed in this package!"
fi

echo -e $ECHO_PREFIX_INFO "Installing Config files..."
if [ -f etc/profile.d/intel-mediasdk.sh ]; then
    try_command cp -f etc/profile.d/intel-mediasdk.* /etc/profile.d/
    echo -e  $ECHO_PREFIX_INFO "The LIBVA_DRIVERS_PATH/LIBVA_DRIVER_NAME will be exported through /etc/profile.d/intel-mediasdk.(c)sh for intel media solution. Please reboot to make it effective."
fi
if [ -f etc/profile.d/intel-mediasdk-devel.sh ]; then
    try_command cp -f etc/profile.d/intel-mediasdk-devel.* /etc/profile.d/
fi

# !!! IMPORTANT !!!
# keep below as the final step for UMD installation to avoid
#    'ldconfig'
# missing anything
if [ -d etc/ld.so.conf.d ]; then
    try_command mkdir -p /etc/ld.so.conf.d
    try_command cp -dfr etc/ld.so.conf.d/* /etc/ld.so.conf.d/
    echo -e $ECHO_PREFIX_INFO "Calling ldconfig after all user-space drivers and config files are in place..."
    try_command ldconfig
    echo -e $ECHO_PREFIX_INFO "Calling to ldconfig is done."
fi

echo -e $ECHO_PREFIX_INFO "Package installation Done. Please Reboot."
