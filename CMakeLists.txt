cmake_minimum_required(VERSION 3.13)
set(program_target "neural_network")
project(${program_target} VERSION 1.0.0 LANGUAGES CXX)

# Define program target
add_executable(${program_target})
target_include_directories(${program_target} PUBLIC include)

# Add source directories
add_subdirectory(src)

# Add cmath library
target_link_libraries(${program_target} m)
