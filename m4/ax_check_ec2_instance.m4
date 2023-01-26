# -*- Autoconf -*-
#
# Copyright (c) 2023      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# $1 -> action if is EC2 isntance
# $2 -> action if not EC2 instance
AC_DEFUN([AX_CHECK_EC2_INSTANCE],[
  ec2_instance_found="no"

  AC_MSG_CHECKING([if running on EC2 instance])

  # PVM Instance
  AS_IF([test "${ec2_instance_found}" = "no"],
        [AS_ECHO(["Looking for PVM instance"]) >& AS_MESSAGE_LOG_FD
         ec2_instance_tmp=`cat /sys/hypervisor/uuid 2>&1`
	 ec2_instance_rc=$?
         AS_ECHO(["hypervisor_uuid is ${ec2_instance_tmp}"]) >& AS_MESSAGE_LOG_FD
         AS_IF([test ${ec2_instance_rc} -ne 0],
               [ec2_instance_found="no"],
               [echo ${ec2_instance_tmp} | grep -q '^ec2'
                AS_IF([test $? -eq 0], [ec2_instance_found="yes"], [ec2_instance_found="no"])])])

  # Getting the instance id out of HVM requires root privs, which
  # we're not going to do.  But on all the pre-nitro HVM instances I
  # checked, the bios_version had amazon in the string, so use that.
  AS_IF([test "${ec2_instance_found}" = "no"],
        [AS_ECHO(["Looking for HVM instance"]) >& AS_MESSAGE_LOG_FD
         ec2_instance_tmp=`cat /sys/devices/virtual/dmi/id/bios_version 2>&1`
	 ec2_instance_rc=$?
         AS_ECHO(["bios_version is ${ec2_instance_tmp}"]) >& AS_MESSAGE_LOG_FD
         AS_IF([test ${ec2_instance_rc} -ne 0],
               [ec2_instance_found="no"],
               [echo ${ec2_instance_tmp} | grep -q '.*amazon$'
                AS_IF([test $? -eq 0], [ec2_instance_found="yes"], [ec2_instance_found="no"])])])

  # Nitro Instances: Look at DMI board_assett_tag, which should be an
  # EC2 instance id.  Support both old 8 character and new 17
  # character formats.
  AS_IF([test "${ec2_instance_found}" = "no"],
        [AS_ECHO(["Looking for Nitro instance"]) >& AS_MESSAGE_LOG_FD
         ec2_instance_tmp=`cat /sys/devices/virtual/dmi/id/board_asset_tag` >& /dev/null
	 ec2_instance_rc=$?
         AS_ECHO(["board_asset is ${ec2_instance_tmp}"]) >& AS_MESSAGE_LOG_FD
         AS_IF([test ${ec2_instance_rc} -ne 0],
               [ec2_instance_found="no"],
               [dnl The extra escaping on the line below is so that
		dnl Autoconf doesn't eat the brackets in the regex.
                [echo ${ec2_instance_tmp} | grep -q 'i-\([^\s]\{8\}\|[^\s]\{17\}\)$']
                AS_IF([test $? -eq 0], [ec2_instance_found="yes"], [ec2_instance_found="no"])])])

  AC_MSG_RESULT([${ec2_instance_found}])
  AS_IF([test "${ec2_instance_found}" = "yes"], [$1], [$2])

  AS_UNSET([ec2_instance_tmp])
  AS_UNSET([ec2_instance_found])
  AS_UNSET([ec2_instance_rc])
])
