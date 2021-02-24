find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS}) 

add_executable(sql_parser sql_parser.cpp main.cpp)

target_link_libraries(sql_parser ${Boost_LIBRARIES})