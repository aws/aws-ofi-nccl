# Hostfile Topologify

A utility script that optimizes the ordering of compute nodes in a hostfile based on their network topology in AWS EC2. This optimization can significantly improve performance for communication-intensive workloads by minimizing network latency between adjacent ranks.

## Prerequisites

- Required Python packages (install via `pip install -r requirements.txt`):
  - boto3
- AWS credentials with EC2 describe permissions configured:
  
  ### AWS Credentials Setup
  
  **Option 1: Using AWS CLI configuration**:
     ```
     aws configure
     ```
     Enter ```AWS Access Key ID``` and ```Secret Access Key``` of your IAM role along with region and output format.
  
  **Option 2: Using environment variables**:
     ```
     export AWS_ACCESS_KEY_ID=<your_access_key>
     export AWS_SECRET_ACCESS_KEY=<your_secret_key>
     export AWS_SESSION_TOKEN=<your_session_token>
     ```
     Copy the above environment variables of your IAM role and paste them into your terminal. Then set ```AWS_DEFAULT_REGION``` environment variable.
     ```
     export AWS_DEFAULT_REGION=<your_default_region>
     ```
  
  **Option3: Using EC2 instance profile** (recommended for EC2 instances):

     If running on an EC2 instance, attach an IAM role with the required permissions.
  
  ### Required IAM Permissions
  
  Your IAM user or role needs at least the following permissions:
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "ec2:DescribeInstances",
          "ec2:DescribeInstanceTopology"
        ],
        "Resource": "*"
      }
    ]
  }
  ```
  You can add the permissions by adding the ```arn:aws:iam::aws:policy/AmazonEC2ReadOnlyAccess``` managed policy to the IAM user or role.


## Usage

1. Create an input hostfile with one hostname per line. For example, this can be generated using a Slurm command ```scontrol``` inside of a Slurm salloc/sbatch job:
   ```
   scontrol show hostname $SLURM_NODELIST > hostfile
   ```

2. Create a Python Virtual Environment and install dependencies:
   ```
   python3 -m venv env
   source env/bin/activate

   pip3 install -r requirements.txt
   ```

3. Run the ```hostfile-topologify.py``` Python script to generate topology-aware hostfile:
   ```
   python3 hostfile-topologify.py --input hostfile [--output hostfile_topo_aware]
   ```

   If `--output` is not specified, the topology-aware hostfile will be printed to stdout.

4. Exit the Python Virtual Environment:
   ```
   deactivate
   ```