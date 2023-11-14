# EFA Cheatsheet

## 1. Settings via environment variables

For optimized performance, you may need to set additional environment variables depending on the
versions of your libfabric.

<table>
   <thead>
      <th>Setting</th>
      <th>Description</th>
   </thead>
   <tr>
      <td><code>FI_EFA_USE_HUGE_PAGE=0</code></td>
      <td>Set to 0 when you see <code>os.fork()</code> causes <code>OSError: Cannot allocate memory</code>.
      Typically caused by multi-process PyTorch data loaders. Disabling huge page
      causes minor performance hit, but it's needed to prevent fork fails due to the operating
      system running out of huge pages.</td>
   </tr>
   <tr>
      <td><code>FI_EFA_FORK_SAFE=1</code></td>
      <td>Not needed anymore.<p>It used to be needed for <code>kernel<5.15</code> (see
      <a href=https://github.com/ofiwg/libfabric/pull/9112>this</a>). However, all reasonably recent
      versions of the plugin (since at least v1.2, probably even older) always set this flag for
      supported versions of Libfabric, regardless of kernel version.</p>
      </td>
   </tr>
   <tr>
      <td><code>FI_EFA_USE_DEVICE_RDMA=1</code></td>
      <td>Do not set for libfabric>=1.18.0 and aws-ofi-nccl>=1.7.0. It's not harmful to set it on
      p4/p5 instances with the newer software, but you just don't have to set it.</td>
   </tr>
   <tr>
      <td><code>FI_EFA_ENABLE_SHM_TRANSFER=1</code></td>
      <td>Not needed.<p>Libfabric will disable SHM transfer when the application sets <code>FI_OPT_CUDA_API_PERMITTED</code>
      to false, which this plugin does (see discussion
      <a href="https://github.com/aws/aws-ofi-nccl/pull/287#discussion_r1362937281">here</a>.)</p>
      </td>
   </tr>
   <tr>
      <td><code>FI_PROVIDER=efa</code></td>
      <td>Use for aws-ofi-nccl<=1.5.0 AND
      <a href=https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-instance-types>EFA-enabled GPU instances</a>.
      </td>
   </tr>
   <tr>
      <td><code>NCCL_PROTO=simple</code></td>
      <td>Use for aws-ofi-nccl<=1.5.0 AND <a href=https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-instance-types>EFA-enabled GPU instances</a>
      </td>
   </tr>
   <tr>
      <td><code>NCCL_SOCKET_NTHREADS</code></td>
      <td>Not applicable for EFA.</td>
   </tr>
   <tr>
      <td><code>NCCL_NSOCKS_PERTHREAD</code></td>
      <td>Not applicable for EFA.</td>
   </tr>
   <tr>
      <td><code>NCCL_MIN_CHANNELS=xxx</code></td>
      <td>Recommend to leave it out to use the default. For e.g., on p4d/p4de, the number of
      channels should be 8, which is the minimum for a 4-NIC platform. The reduction message is
      split by number of GPUs in the job, then the number of channels, so having more channels than
      necessary causes smaller messages which causes EFA to be starved for data.</td>
   </tr>
   <tr>
      <td><code>NCCL_BUFFSIZE=xxx</code></td>
      <td>Recommend to leave it out to use the default.</td>
   </tr>
   <tr>
      <td><code>RDMAV_FORK_SAFE=1</code></td>
      <td>Do not use. This is a RDMA-core environment variable. Prefer <code>FI_EFA_FORK_SAFE</code>
      (if it still makes sense for your Linux kernel version). The two look the same, but actually
      behaves very differently, especially on newer kernels, where <code>RDMAV_FORK_SAFE=1</code>
      can break things.</td>
   </tr>
   <tr>
      <td><code>RDMAV_*</code></td>
      <td>Do not use.</td>
   </tr>
   <tr>
      <td>NCCL version</td>
      <td>Recommend one of the stable releases.</td>
   </tr>
</table>

## 2. Sample Presets

The following table shows the environment variables you need to set for common library versions.

As of this writing, only [p4 and p5 instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-limits)
support both EFA and NVIDIA GPUDirect Remote Direct Memory Access (RDMA).

<table>
   <thead>
      <th>Situation</th>
      <th>Actions</th>
   </thead>

   <tr>
      <td>p5.48xlarge</td>
      <td>
         • cuda>=12.0<br>
         • nccl>=2.18.0 (recommend at least 2.18.5)<br>
         • aws-ofi-nccl>=1.7.2 (recommend at least 1.7.3)<br>
         • efa-installer>=1.29.0 (to avoid nccl>=2.19.0 raising libfabric errors)
      </td>
   </tr>
   <tr>
      <td>Memory allocation errors</td>
      <td><pre>export FI_EFA_USE_HUGE_PAGE=0</pre></td>
   </tr>
   <tr>
      <td>
         • libfabric>=1.18.0<br>
         • aws-ofi-nccl>=1.7.0</td>
      <td>N/A</td>
   </tr>
   <tr>
      <td>
         • aws-ofi-nccl>=1.6.0,<1.7.0<br>
         • p4/p5 instances</td>
      <td><pre>export FI_EFA_USE_DEVICE_RDMA=1</pre></td>
   </tr>
   <tr>
      <td>
         • aws-ofi-nccl>=1.6.0,<1.7.0<br>
         • EFA instances without RDMA</td>
      <td>N/A</td>
   </tr>
   <tr>
      <td>
         • aws-ofi-nccl<=1.5.0<br>
         • p4/p5 instances</td>
      <td>
<pre>
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export NCCL_PROTO=simple
</pre>
      </td>
   <tr>
      <td>
         • aws-ofi-nccl<=1.5.0<br>
         • EFA instances without RDMA</td>
      <td>
<pre>
export FI_PROVIDER=efa
export NCCL_PROTO=simple
</pre>
      </td>
   </tr>
</table>
