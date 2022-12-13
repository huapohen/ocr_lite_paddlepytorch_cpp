set(CMAKE_SYSTEM_NAME QNX)

set(arch gcc_ntoaarch64)

SET(CMAKE_C_COMPILER "$ENV{QNX_HOST}/usr/bin/aarch64-unknown-nto-qnx7.0.0-gcc")
SET(CMAKE_CXX_COMPILER "$ENV{QNX_HOST}/usr/bin/aarch64-unknown-nto-qnx7.0.0-g++")

set(CMAKE_C_COMPILER_TARGET ${arch})
set(CMAKE_CXX_COMPILER_TARGET ${arch})

set(CMAKE_SYSROOT $ENV{QNX_TARGET})
