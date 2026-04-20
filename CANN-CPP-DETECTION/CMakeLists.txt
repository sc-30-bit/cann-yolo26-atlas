cmake_minimum_required(VERSION 3.10)
project(yolo26_acl_cpp_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(ASCEND_BASE /usr/local/Ascend/ascend-toolkit/latest)
set(ASCEND_INC ${ASCEND_BASE}/include)
set(ASCEND_LIB_CANDIDATES
    ${ASCEND_BASE}/aarch64-linux/lib64
    ${ASCEND_BASE}/runtime/lib64
)

find_library(ASCENDCL_LIB
    NAMES ascendcl
    PATHS ${ASCEND_LIB_CANDIDATES}
    NO_DEFAULT_PATH
)

if(NOT ASCENDCL_LIB)
    message(FATAL_ERROR "Cannot find libascendcl.so. Please check Ascend toolkit path.")
endif()

message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "ASCENDCL_LIB: ${ASCENDCL_LIB}")

add_executable(yolo26_acl_demo main.cpp)

target_include_directories(yolo26_acl_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ASCEND_INC}
)

target_link_libraries(yolo26_acl_demo PRIVATE
    ${OpenCV_LIBS}
    ${ASCENDCL_LIB}
)

set_target_properties(yolo26_acl_demo PROPERTIES
    BUILD_RPATH "${ASCEND_BASE}/aarch64-linux/lib64;${ASCEND_BASE}/runtime/lib64"
)