cmake_minimum_required(VERSION 3.8)
project(cam_ball_balancer_cpp)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)  # -O3: el punto de este paquete es la velocidad
endif()
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(cam_ball_balancer_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio calib3d aruco)

# Nucleo sin ROS (lo usan el nodo, el benchmark y los tests).
add_library(vision_core STATIC src/vision_core.cpp src/camera.cpp)
target_include_directories(vision_core PUBLIC include)
target_link_libraries(vision_core PUBLIC ${OpenCV_LIBS})
set_target_properties(vision_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(vision_core PUBLIC Threads::Threads)

add_executable(vision_node src/vision_node.cpp)
target_link_libraries(vision_node vision_core)
ament_target_dependencies(vision_node rclcpp geometry_msgs sensor_msgs tf2_ros visualization_msgs ament_index_cpp cam_ball_balancer_msgs)

add_executable(bench_vision src/bench_vision.cpp)
target_link_libraries(bench_vision vision_core)
ament_target_dependencies(bench_vision ament_index_cpp)

install(TARGETS vision_node bench_vision DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_vision_core test/test_vision_core.cpp)
  target_link_libraries(test_vision_core vision_core)
endif()

ament_package()
