project(UnscentedKF)

cmake_minimum_required (VERSION 3.5)

#add_definitions(-std=c++0x)
add_definitions(-Wall)  # show WARNINGS all, so you can then fix them immediately!

# BDD-Style Tests
set(sources src/test_tools.cpp)
#set(sources src/test_ukf.cpp)

# SDC Term 2 Project 2 - Unscented Kalman Filters
#set(sources src/ukf.cpp src/main.cpp src/tools.cpp)

add_executable(UnscentedKF ${sources})
