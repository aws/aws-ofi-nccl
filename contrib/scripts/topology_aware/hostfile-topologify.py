#!/usr/bin/env python3
#
# Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# DESCRIPTION:
#   This script optimizes the ordering of compute nodes in a hostfile based on their
#   network topology. It takes a hostfile containing a list of hostnames (one per line)
#   and reorders them so that adjoining ranks are as close as possible in the network
#   topology. This optimization can significantly improve performance for communication-
#   intensive workloads by minimizing network latency between adjacent ranks.
#
# USAGE:
#   python3 hostfile-topologify.py --input <input_hostfile> [--output <output_hostfile>]
#
# ARGUMENTS:
#   --input  : Path to the input hostfile containing hostnames (one per line)
#   --output : Path to write the topology-aware hostfile (default: stdout)


import boto3
import argparse
import sys
import socket
import time

pagination_count = 64
max_retries = 5

def generate_topology_csv(input_file, output_file):
    ec2_client = boto3.client('ec2', None)

    done = False

    network_to_hostname = {}

    while not done:
        ip_to_hostname = {}
        instanceid_to_hostname = {}

        # translate hostname to private ip, since PCluster uses custom
        # hostnames that the EC2 control plane doesn't see.
        for i in range(pagination_count):
            hostname = input_file.readline()
            if not hostname:
                done = True
                break
            hostname = hostname.strip()

            ip = None
            for i in range(max_retries):
                try:
                    ip = socket.gethostbyname(socket.getfqdn(hostname))
                except:
                    time.sleep(1)
                else:
                    break
            if ip == None:
                print("Error getting ip address for %s" % (hostname))
                sys.exit(1)

            ip_to_hostname[ip] = hostname

        if len(ip_to_hostname.keys()) == 0:
            break

        # build instanceid -> hostname map by describing all the ips
        # and matching ip to instance id, then translating through
        # ip_to_hostname.
        pagination_done = False
        next_token = ""
        while not pagination_done:
            response = ec2_client.describe_instances(
                Filters=[
                    {
                        'Name': 'network-interface.addresses.private-ip-address',
                        'Values': list(ip_to_hostname.keys())
                    }
                ],
                MaxResults=pagination_count,
                NextToken=next_token)

            if 'NextToken' in response:
                next_token = response['NextToken']
            else:
                pagination_done = True

            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instanceid = instance['InstanceId']
                    for network_interface in instance['NetworkInterfaces']:
                        private_ip = network_interface['PrivateIpAddress']
                        if private_ip in ip_to_hostname:
                            instanceid_to_hostname[instanceid] = ip_to_hostname[private_ip]

        pagination_done = False
        next_token = ""
        while not pagination_done:
            response = ec2_client.describe_instance_topology(
                InstanceIds=list(instanceid_to_hostname.keys()),
                NextToken=next_token)

            if 'NextToken' in response:
                next_token = response['NextToken']
            else:
                pagination_done = True

            for instance in response['Instances']:
                instanceid = instance['InstanceId']

                # NetworkNodes[2] (layer iii) is the bottom layer
                # connected to the instance. NetworkNode[1] (layer ii)
                # is one layer upper than NetworkNodes[2].
                l2_node = instance['NetworkNodes'][1]
                l3_node = instance['NetworkNodes'][2]

                if network_to_hostname.get(l2_node) == None:
                    network_to_hostname[l2_node] = {}
                if network_to_hostname[l2_node].get(l3_node) == None:
                    network_to_hostname[l2_node][l3_node] = []
                network_to_hostname[l2_node][l3_node].append(
                    instanceid_to_hostname[instanceid])

    for l2 in network_to_hostname:
        for l3 in network_to_hostname[l2]:
            for hostname in network_to_hostname[l2][l3]:
                output_file.write("%s\n" % (hostname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate placement information in CSV formation",
    )
    parser.add_argument(
        "--output",
        help="Output file to write (default: stdout)",
        default=None
    )
    parser.add_argument(
        "--input",
        help="input hostfile",
        required=True,
        default=None
    )

    args = parser.parse_args()

    if args.output != None:
        output_file_handle = open(args.output, "w")
    else:
        output_file_handle = sys.stdout

    input_file_handle = open(args.input, "r")

    generate_topology_csv(input_file_handle, output_file_handle)

    input_file_handle.close()
    if args.output != None:
        output_file_handle.close()
