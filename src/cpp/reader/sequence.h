#pragma once

#include <cstdlib>
namespace bfiocpp{

class Seq
{
    private:
        long start_index_, stop_index_, step_;
    public:
        inline Seq(const long start, const long  stop, const long  step=1):start_index_(start), stop_index_(stop), step_(step){} 
        inline long Start()  const  {return start_index_;}
        inline long Stop()  const {return stop_index_;}
        inline long Step()  const {return step_;}
};
} //ns bfiocpp