project(UnscentedKF)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++0x)
#add_definitions(-Wall)  # show WARNINGS all, so you can then fix them immediately!

# BDD-Style Tests
#set(sources src/test_tools.cpp)
#set(sources src/test_ukf.cpp src/ukf.cpp src/tools.cpp)

# SDC Term 2 Project 2 - Unscented Kalman Filters
set(sources src/main.cpp src/ukf.cpp src/tools.cpp)

add_executable(UnscentedKF ${sources})
