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

#include <rand-util.hpp>
#include <random>
#include <cassert>

namespace
{
    std::random_device seed;
    std::default_random_engine engine(seed());
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::normal_distribution<double> normal_dist(0.0, 1.0);
}

namespace randutil
{
    double GenNumFromUniformDist()
    {
        return uniform_dist(engine);
    }
    
    double GenNumFromNormalDist()
    {
        return normal_dist(engine);
    }
    
    int GenNumFromUniformIntDist(int lower_bound, int upper_bound)
    {
        assert(upper_bound - lower_bound >= 0 && "Invalid args");
        
        std::uniform_int_distribution<int> uniform_int_dist(lower_bound, upper_bound);
        return uniform_int_dist(engine);
    }
    
    std::vector<int> GenDistinctNumsFromUniformIntDist(int lower_bound, int upper_bound, int num)
    {
        assert(num >= 0 && upper_bound - lower_bound > num && "Invalid args");
        
        std::vector<int> results;
        for (int i = 0; i < num; ++ i)
        {
            while (true)
            {
                const int candidate_num = GenNumFromUniformIntDist(lower_bound, upper_bound);
                
                bool duplicated = false;
                for (const int& num : results)
                {
                    if (num == candidate_num) { duplicated = true; break; }
                }
                
                if (!duplicated) { results.push_back(candidate_num); break; }
            }
        }
        
        return results;
    }
    
    std::string GenRandomString(int length)
    {
        constexpr char char_set[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        
        std::string result = "";
        
        for (int i = 0; i < length; ++ i)
        {
            result += char_set[GenNumFromUniformIntDist(0, 61)];
        }
        
        return result;
    }
}
