/*
 MIT License
 
 Copyright (c) 2018 Yuki Koyama
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#ifndef rand_util_hpp
#define rand_util_hpp

#include <vector>
#include <string>

namespace randutil
{
    /// \brief Generate a random number from a standard uniform distribution.
    double GenNumFromUniformDist();
    
    /// \brief Generate a random number from a standard normal distribution.
    double GenNumFromNormalDist();
    
    /// \brief Generate a random number from a uniform integer distribution.
    /// \param lower_bound The minimum value that is possible to be generated.
    /// \param upper_bound The maximum value that is possible to be generated.
    int GenNumFromUniformIntDist(int lower_bound, int upper_bound);
    
    /// \brief Generate a set of distinct random numbers from a uniform integer distribution.
    /// \param lower_bound The minimum value that is possible to be generated.
    /// \param upper_bound The maximum value that is possible to be generated.
    /// \param num The number of numbers to be generated.
    /// \details This function uses a greedy algorithm that assumes lower_bound - upper_bound >> num.
    ///          Otherwise, this function gets unreasonably slow.
    std::vector<int> GenDistinctNumsFromUniformIntDist(int lower_bound, int upper_bound, int num);
    
    /// \brief Generate a string consisting of random characters.
    /// \param length The length of the resulting string.
    std::string GenRandomString(int length = 16);
}

#endif // rand_util_hpp
