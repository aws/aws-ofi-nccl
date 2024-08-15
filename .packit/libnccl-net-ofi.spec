# Whether to build with cuda support. Default: on if neuron
%if "%{with_cuda}" == "1" && "%{with_neuron}" == "1"
%{error:Neuron and CUDA must not be enabled together}
%endif

%if "%{with_cuda}" == "0" && "%{with_neuron}" == "0"
%{error:One of Neuron or CUDA must be enabled}
%endif

%if "%{with_cuda}" == "1"
%{!?target: %global target nccl}
%endif
%if "%{with_neuron}" == "1"
%{!?target: %global target nccom}
%endif

%global pname_base lib%{!?with_neuron:nccl}%{?with_neuron:nccom}-net-ofi
%global pname %{pname_base}%{?with_platform_aws:-aws}

%if "%{with_platform_aws}"
%global _prefix /opt/amazon/%{pname_base}
%endif

# (CUDA only) what toolkit package to declare a build dependency on. Default: 12-6
%{!?_cuda_toolkit_version: %global _cuda_toolkit_version 12-6}

Name:           %{pname}
Version:        null
Release:        0%{dist}
Summary:        NCCL + libfabric compatibility layer
License:        Apache-2.0
URL:            https://github.com/aws/aws-ofi-nccl
Source0:        null
%if "%{_vendor}" == "debbuild"
Group:          devel
%else
Group:          Development/Tools%{?suse_version:/Building}
BuildRequires:  hwloc-devel
BuildRequires:  make
BuildRequires:  gcc
BuildRequires:  gcc-c++
%if "%{with_platform_aws}"
BuildRequires:  libfabric-aws-devel
Requires:       libfabric-aws
%else
BuildRequires:  libfabric1-devel
Requires:       libfabric
%endif
%if "%{with_cuda}" == "1"
BuildRequires:  cuda-cudart-devel-%{_cuda_toolkit_version}
%endif
%endif
Requires:       hwloc

%description
This is a plugin which lets EC2 developers use libfabric as network provider
while running NCCL applications.


%prep
%setup
%build
%configure \
    --prefix="%{_prefix}" \
    --disable-tests \
    --with-mpi=no \
%if "%{with_cuda}" == "1"
    --with-cuda=/usr/local/cuda-12 \
    --enable-neuron=no \
%else
    --with-cuda=no \
    --enable-neuron=yes \
%endif
%if "%{with_platform_aws}" == "1"
    --enable-platform-aws \
    --with-libfabric=/opt/amazon/efa
%else
    --disable-platform-aws
%endif
%make_build


%install
%make_install
find %{buildroot} -name '*.la' -exec rm -f {} ';'
%ldconfig_scriptlets


%files
%{_libdir}/*.so
%{_datadir}/aws-ofi-nccl/xml/*.xml
%license LICENSE NOTICE
%doc


%changelog
* Thu Aug 08 2024 Nicholas Sielicki <nslick@amazon.com>
Initial Package
