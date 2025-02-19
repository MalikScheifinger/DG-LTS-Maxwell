#ifndef TOOLS_NUMBER_SPACES_H_
#define TOOLS_NUMBER_SPACES_H_

#include <iostream>
#include <vector>

namespace MaxwellProblem::Tools {

/**
 * @brief Generates numbers equally spaced between start and stop
 * 
 * start = 0.1
 * stop = 0.01
 * nums = 4
 * ret = [0.1, 0.07, 0.04, 0.01]
 * 
 * @param start
 * @param stop 
 * @param nums
 * @param endpoint
 * @return std::vector<double> 
 */
std::vector<double> lin_spaced(const double start, const double stop, const int nums, const bool endpoint=true);

/**
 * @brief Generates numbers equally spaced in log space between start and stop
 * 
 * start = 0.1
 * stop = 0.01
 * nums = 4
 * ret = [0.1, 0.0464159, 0.0215443, 0.01]
 * 
 * @param start 
 * @param stop 
 * @param nums
 * @param endpoint
 * @return std::vector<double> 
 */
std::vector<double> log_spaced(const double start, const double stop, const int nums, const bool endpoint=true);

/**
 * @brief Generates numbers equally spaced in log2 space between start and stop
 * 
 * start = 0.1
 * stop = 0.0125
 * nums = 4
 * ret = [0.1, 0.05, 0.025, 0.0125]
 * 
 * @param start 
 * @param stop 
 * @param nums 
 * @param endpoint
 * @return std::vector<double> 
 */
std::vector<double> log2_spaced(const double start, const double stop, const int nums, const bool endpoint=true);
}// namespace MaxwellProblem::Tools



#endif//TOOLS_NUMBER_SPACES_H_