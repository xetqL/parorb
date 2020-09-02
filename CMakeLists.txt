cmake_minimum_required(VERSION 3.17)
project(orb)

find_package(MPI REQUIRED)
find_package(Boost 1.65.0 REQUIRED)
set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(parmedian)
include_directories(yalbb/includes)
include_directories(${BOOST_INCLUDE_DIRS})
add_executable(orb main.cpp)

target_link_libraries(orb PUBLIC ${CMAKE_CURRENT_LIST_DIR}/yalbb/lib/libyalbb.a ${BOOST_LIBRARIES})
