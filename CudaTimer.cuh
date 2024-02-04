#include <cuda.h>
#include <cuda_runtime.h>


class CudaTimer{

    public:
      CudaTimer();
      ~CudaTimer();
      void start_timer();
      void stop_timer();
      float get_time();

    private:
      float time;
      cudaEvent_t start, stop;

};
