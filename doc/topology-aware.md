# Topology-Aware Scheduling and Rank Mapping

## Topology-Aware Scheduling

Topology-aware scheduling enables schedulers to intelligently select hosts from a large pool based on their physical network topology. By allocating jobs to hosts within the same network zone or to physically proximate nodes, topology-aware scheduling significantly reduces network latency, overall network traffic, and cross-zone data transfer cost. This optimization is especially beneficial for applications using NCCL and aws-ofi-nccl libraries for their communication.

### Slurm

AWS provides [ec2-topology-aware-for-slurm](https://github.com/aws-samples/ec2-topology-aware-for-slurm/tree/main), a tool that helps with configuring [Slurm workload manager](https://slurm.schedmd.com/topology.html) on AWS ParallelCluster to be aware of the Amazon Elastic Compute Cloud (EC2) instance network topology. This tool queries the [Amazon EC2 Instance Topology API](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeInstanceTopology.html) to generate a topology configuration file, named **topology.conf**, which accurately describes the network topology of EC2 instances within the AWS ParallelCluster. With this configuration, Slurm can make topology-aware scheduling decisions, scheduling jobs on closely connected instances to minimize network latency and maximize throughput in high-performance computing (HPC) environments. For a deeper understanding of Amazon EC2 instance topology, please refer to the [EC2 User Guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-ec2-instance-topology-works.html).

### Amazon EKS

Amazon Elastic Kubernetes Service (EKS) also supports topology-aware scheduling through its native integration with EC2 instance topology. When worker nodes are launched in an EKS cluster, Amazon EKS automatically discovers and exposes their network topology information as node labels. Users can leverage these topology labels by creating custom UserData scripts that query this information, and then use it to influence pod placement decisions in their preferred scheduler.

Here is a sample UserData script for reference:
```
#!/usr/bin/env python3

import argparse
import json
import re
import sys
from kubernetes import client, config

INSTANCE_ID_PATTERN = r'i-[0-9a-f]*'

def generate_topology_from_node_labels(output_file):
    config.load_kube_config()
    kube_client_v1 = client.CoreV1Api()
    nodes = kube_client_v1.list_node()

    output = {'Nodes': []}
    for node in nodes.items:
        node_info = {}
        node_info['Name'] = node.metadata.name

        match = re.search(INSTANCE_ID_PATTERN, node.spec.provider_id)
        node_info['InstanceId'] = match.group()

        topology_labels = {
            key: value for key, value in node.metadata.labels.items()
            if key.startswith('topology.')
        }
        node_info['Topology'] = topology_labels

        output['Nodes'].append(node_info)

    json.dump(output, output_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Kubernetes cluster topology in JSON formation",
    )
    parser.add_argument(
        "--output",
        help="Output file to write (default: stdout)",
        default=None
    )
    args = parser.parse_args()

    if args.output != None:
        file_handle = open(args.output, "w")
    else:
        file_handle = sys.stdout

    generate_topology_from_node_labels(file_handle)

    file_handle.close()
```

And the output will be like:
```
{
    "Nodes": [
        {
            "Name": "ip-192-168-0-1.us-west-2.compute.internal",
            "InstanceId": "i-1111111111example",
            "Topology": {
                "topology.k8s.aws/network-node-layer-1": "nn-1111111111example",
                "topology.k8s.aws/network-node-layer-2": "nn-2222222222example",
                "topology.k8s.aws/network-node-layer-3": "nn-3333333333example",
                "topology.k8s.aws/zone-id": "usw2-az2",
                "topology.kubernetes.io/region": "us-west-2",
                "topology.kubernetes.io/zone": "us-west-2a"
            }
        },
        {
            "Name": "ip-192-168-50-1.us-west-2.compute.internal",
            "InstanceId": "i-2222222222example",
            "Topology": {
                "topology.k8s.aws/network-node-layer-1": "nn-4444444444example",
                "topology.k8s.aws/network-node-layer-2": "nn-5555555555example",
                "topology.k8s.aws/network-node-layer-3": "nn-6666666666example",
                "topology.k8s.aws/zone-id": "usw2-az1",
                "topology.kubernetes.io/region": "us-west-2",
                "topology.kubernetes.io/zone": "us-west-2b"
            }
        },
        {
            "Name": "ip-192-168-51-1.us-west-2.compute.internal",
            "InstanceId": "i-3333333333example",
            "Topology": {
                "topology.k8s.aws/network-node-layer-1": "nn-4444444444example",
                "topology.k8s.aws/network-node-layer-2": "nn-7777777777example",
                "topology.k8s.aws/network-node-layer-3": "nn-8888888888example",
                "topology.k8s.aws/zone-id": "usw2-az1",
                "topology.kubernetes.io/region": "us-west-2",
                "topology.kubernetes.io/zone": "us-west-2b"
            }
        }
    ]
}
```

## Topology-Aware Rank Mapping

Topology-aware rank mapping is a performance optimization technique that aligns the logical arrangement of processes (ranks) with the physical topology of the underlying hardware architecture. This mapping strategy aims to minimize communication overhead by placing processes that frequently communicate with each other physically close on the system's hardware, thereby reducing network latency.

The aws-ofi-nccl library includes a topology optimization tool called **hostfile-topologify.py**, which sorts hosts based on their network topology and helps to achieve topology-aware rank mapping. The process involves three key steps: First, provide a basic hostfile including all the hostnames as input to the **hostfile-topologify.py** script. Next, the **hostfile-topologify.py** script sorts hostnames in the hostfile based on their network topology, optimizing the placement so that processes with adjacent ranks are positioned as close as possible within the network infrastructure. Finally, topo_rmap hostfile is generated that maps each rank to its optimal host location. This optimized hostfile can then be used with the Open MPI `mpirun` command to launch jobs with topology-aware process placement. This procedure can be implemented through the following script:

```
# Note: this sample script is expected to be run inside of a Slurm (e.g. salloc/sbatch/srun) job

HOSTFILE=$(mktemp)
HOSTFILE_TOPO=$(mktemp)
HOSTFILE_TOPO_RMAP=$(mktemp)
scontrol show hostnames | tee ${HOSTFILE}
AWS_DEFAULT_REGION=<your-region>
<path-to-aws-ofi-nccl-library>/contrib/scripts/topology_aware/hostfile-topologify.py --input ${HOSTFILE} --output ${HOSTFILE_TOPO}
(for i in $(cat ${HOSTFILE_TOPO}) ; do seq 1 ${SLURM_NTASKS_PER_NODE} | xargs -i -- echo $i ; done) > ${HOSTFILE_TOPO_RMAP}
```

When executing Open MPI `mpirun` command, it can leverage the topo_rmap hostfile generated in the previous step. By using *`--mca rmaps seq`* option in Open MPI v4.1 or *`--map-by seq`* option in Open MPI v5.0, mpirun specifies arbitrary mappings to use the "sequential mapper," which reads the hostfile line by line, assigning processes to nodes in whatever order the hostfile specifies.

Here is an example for Open MPI v4.1 version. For further details, please refer to [mpirun v4.1 documentation](https://www.open-mpi.org/doc/v4.1/man1/mpirun.1.php#sect12).
```
mpirun \
    -N ${SLURM_NTASKS_PER_NODE} \
    --hostfile ${HOSTFILE_TOPO_RMAP} \
    --mca rmaps seq \
    bash -c <your-application-executable>
```

Here is an example for Open MPI v5.0 version. For further details, please refer to [mpirun v5.0 documentation](https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man1/mpirun.1.html#mapping-ranking-and-binding-oh-my).
```
mpirun \
    --hostfile ${HOSTFILE_TOPO_RMAP} \
    --map-by seq \
    bash -c <your-application-executable>
```