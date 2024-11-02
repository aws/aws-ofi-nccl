#!/usr/bin/env rpm

%global _prefix /opt/amazon/ofi-nccl

Version:         @VERSION@
Release:         1%{?dist}
Source0:         @TARBALL@
Prefix:   /opt/amazon/ofi-nccl
Name:     libnccl-ofi
URL:      https://github.com/aws/aws-ofi-nccl
Summary:  NCCL + libfabric compatibility layer
License:  Apache License 2.0
BuildRequires: autoconf, autoconf-archive, automake, libtool, hwloc-devel
AutoProv: yes

%description
This is a plugin which lets EC2 developers use libfabric as network provider
while running NCCL applications.


%prep
%setup -n aws-ofi-nccl-%{version}
%build
autoreconf -ivf
%configure --disable-picky-compiler \
           --sysconfdir=%{_prefix}/conf \
           --with-cuda=/usr/local/cuda \
           --enable-cudart-dynamic \
           --with-libfabric=/opt/amazon/efa \
           --disable-werror \
           --disable-tests \
           --enable-platform-aws
%make_build

%install
%make_install
mkdir -p %{buildroot}/etc/ld.so.conf.d && \
    echo "%{_libdir}" >> %{buildroot}/etc/ld.so.conf.d/100_ofinccl.conf
find %{buildroot} -name '*.la' -exec rm -f {} ';'
%ldconfig_scriptlets

%files
%{_libdir}/*.so
%{_datadir}/aws-ofi-nccl/xml/*.xml
%{_sysconfdir}/ld.so.conf.d/*.conf
%license LICENSE NOTICE
%doc
