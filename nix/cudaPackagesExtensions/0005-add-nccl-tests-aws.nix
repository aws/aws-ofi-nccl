{ replaceDependency }:
ffinal: pprev: {
  nccl-tests-aws = replaceDependency {
    drv = ffinal.nccl-tests;
    oldDependency = ffinal.nccl;
    newDependency = ffinal.ncclAws;
  };
}
