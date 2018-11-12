#include <dmlc/omp.h>
#include <perflab/build_hist.h>
#include <omp.h>
#include <vector>
#include <cstring>
#include <dmlc/timer.h>
#if defined(__INTEL_COMPILER)
    #include <immintrin.h>
#endif

namespace perflab {

#if defined(__INTEL_COMPILER)
    void BuildHist_Blocking512_AVX512(const std::vector<GradientPair>& gpair, const std::vector<size_t>& instance_set,
                         const GHistIndexMatrix& gmat, std::vector<GHistEntry>* hist, GHistBuilder* builder)
    {
        const auto nthread = static_cast<dmlc::omp_uint>(builder->nthread_);
        const size_t nrows = instance_set.size();

        const size_t* row_ptr =  gmat.row_ptr.data();
        const size_t* rid = instance_set.data();
        const float* pgh = (float*)gpair.data();
        const uint32_t* index = gmat.index.data();
        float* data = (float*)builder->data_.data();

        char* isInit = (char*)_mm_malloc(nthread, 64);
        memset(isInit, '\0', nthread);

        const size_t sizeOfBlock = 512;
        size_t nBlocks = nrows/sizeOfBlock;
        nBlocks += !!(nrows - nBlocks*sizeOfBlock);

        #pragma omp parallel for num_threads(nthread) schedule(guided)
        for (size_t iBlock = 0; iBlock < nBlocks; iBlock++)
        {
            dmlc::omp_uint tid = omp_get_thread_num();
            const size_t off = 2*tid * builder->nbin_;
            float* data_local = data + off;
            auto* data_local_hist = builder->data_.data() + tid * builder->nbin_;

            if (!isInit[tid])
            {
                memset(data_local, '\0', 2*sizeof(float)*builder->nbin_);
                isInit[tid] = true;
            }
            const size_t iStart = iBlock*sizeOfBlock;
            const size_t iEnd = (((iBlock+1)*sizeOfBlock > nrows) ? nrows : iStart + sizeOfBlock);

            for(size_t i = iStart; i < iEnd; ++i)
            {
                size_t st = row_ptr[rid[i]];
                size_t end = row_ptr[rid[i]+1];
                __m512 gh = _mm512_castpd_ps(_mm512_set1_pd(*((double*)(pgh + 2*rid[i]))));
                size_t j = st;
                for (; j < end-7; j+=8)
                {
                    __m256i idx = _mm256_loadu_epi32((index + j));
                    __m512 hist = _mm512_castpd_ps(_mm512_i32gather_pd(idx, data_local, 8));
                    hist        = _mm512_add_ps(hist, gh);
                   _mm512_i32scatter_pd(data_local, idx, _mm512_castps_pd(hist), 8);
                }

                const auto ghStruct =  gpair[rid[i]];

                for (; j < end; ++j)
                {
                    data_local_hist[index[j]].Add(ghStruct);
                }
            }
        }
        size_t nWorkedBins = 1;
        for(size_t i = 0; i < nthread; ++i)
        {
            if (isInit[i] && nWorkedBins < i+1) nWorkedBins=i+1;
        }
        _mm_free(isInit);

        const uint32_t nbin = builder->nbin_;
        float* histData = (float*)hist->data();
        const size_t align = ((64 - ((size_t)(histData) & 63))&63) / sizeof(float);

        if(nWorkedBins == 1)
        {
            memcpy(histData, data, 2*sizeof(float)*nbin);
        }
        else if (2*nbin > 16 + align)
        {
            for(size_t i = 0; i < align; i++)
            {
                histData[i] = data[i];
                for(size_t iB = 1; iB < nWorkedBins; ++iB)
                    histData[i] += data[2*iB*nbin+i];
            }
            const size_t size = (2*nbin - align);
            const size_t sizeOfBlock = 1024;
            size_t nBlocks = size/sizeOfBlock;
            nBlocks += !!(nrows - nBlocks*sizeOfBlock);

            #pragma omp parallel for num_threads(nthread) schedule(dynamic)
            for (size_t iBlock = 0; iBlock < nBlocks; iBlock++)
            {
                const int iStart = iBlock*sizeOfBlock;
                const int iEnd = (((iBlock+1)*sizeOfBlock > size) ? size : iStart + sizeOfBlock);
                int i = iStart;
                for(; i < iEnd-15; i+=16)
                {
                    __m512 sum = _mm512_load_ps(data + i);
                    for(size_t iB = 1; iB < nWorkedBins; ++iB)
                    {
                        __m512 adder = _mm512_load_ps(data + 2*iB*nbin + i);
                        sum = _mm512_add_ps(sum, adder);
                    }
                    _mm512_store_ps(histData+i, sum);
                }
                for(; i < iEnd; i++)
                {
                    histData[i] = data[i];
                    for(size_t iB = 1; iB < nWorkedBins; ++iB)
                        histData[i] += data[2*iB*nbin+i];
                }
            }
        }
        else
        {
            for(size_t i = 0; i < 2*nbin; i++)
            {
                histData[i] = data[i];
                for(size_t iB = 1; iB < nWorkedBins; ++iB)
                    histData[i] += data[2*iB*nbin+i];
            }
        }
    }
#endif

void BuildHist_Blocking512(const std::vector<GradientPair>& gpair,
                                                         const std::vector<size_t>& instance_set,
                                                         const GHistIndexMatrix& gmat,
                                                         std::vector<GHistEntry>* hist, GHistBuilder* builder)
{
    const auto nthread = static_cast<dmlc::omp_uint>(builder->nthread_);
    const size_t nrows = instance_set.size();

    const size_t* row_ptr =  gmat.row_ptr.data();
    const size_t* rid = instance_set.data();
    const float* pgh = (float*)gpair.data();
    const uint32_t* index = gmat.index.data();
    float* data = (float*)builder->data_.data();

    char* isInit = (char*)malloc(nthread);
    memset(isInit, '\0', nthread);

    const size_t sizeOfBlock = 512;
    size_t nBlocks = nrows/sizeOfBlock;
    nBlocks += !!(nrows - nBlocks*sizeOfBlock);

    #pragma omp parallel for num_threads(nthread) schedule(guided)
    for (size_t iBlock = 0; iBlock < nBlocks; iBlock++)
    {
        dmlc::omp_uint tid = omp_get_thread_num();
        const size_t off = 2*tid * builder->nbin_;
        float* data_local = data + off;
        auto* data_local_hist = builder->data_.data() + tid * builder->nbin_;

        if (!isInit[tid])
        {
            memset(data_local, '\0', 2*sizeof(float)*builder->nbin_);
            isInit[tid] = true;
        }
        const size_t iStart = iBlock*sizeOfBlock;
        const size_t iEnd = (((iBlock+1)*sizeOfBlock > nrows) ? nrows : iStart + sizeOfBlock);
        for(size_t i = iStart; i < iEnd; ++i)
        {
            const size_t iColStart = row_ptr[rid[i]];
            const size_t iColEnd = row_ptr[rid[i]+1];
            const auto gh = gpair[rid[i]];

            for (size_t j = iColStart; j < iColEnd; ++j)
            {
                data_local_hist[index[j]].Add(gh);
            }
        }
    }
    size_t nWorkedBins = 1;
    for(size_t i = 0; i < nthread; ++i)
    {
        if (isInit[i] && nWorkedBins < i+1) nWorkedBins=i+1;
    }
    free(isInit);

    const uint32_t nbin = builder->nbin_;
    float* histData = (float*)hist->data();

    if(nWorkedBins == 1)
    {
        memcpy(histData, data, 2*sizeof(float)*nbin);
    }
    else
    {
        const size_t size = (2*nbin);
        const size_t sizeOfBlock = 1024;
        size_t nBlocks = size/sizeOfBlock;
        nBlocks += !!(nrows - nBlocks*sizeOfBlock);

        #pragma omp parallel for num_threads(nthread) schedule(dynamic)
        for (size_t iBlock = 0; iBlock < nBlocks; iBlock++)
        {
            const int iStart = iBlock*sizeOfBlock;
            const int iEnd = (((iBlock+1)*sizeOfBlock > size) ? size : iStart + sizeOfBlock);

            for(int i = iStart; i < iEnd; i++)
            {
                histData[i] = data[i];
                for(size_t iB = 1; iB < nWorkedBins; ++iB)
                    histData[i] += data[2*iB*nbin+i];
            }
        }
    }
}

void BuildHistXGBoost(const std::vector<GradientPair>& gpair,
                                                         const std::vector<size_t>& instance_set,
                                                         const GHistIndexMatrix& gmat,
                                                         std::vector<GHistEntry>* hist, GHistBuilder* builder)
{
    std::fill(builder->data_.begin(), builder->data_.end(), GHistEntry());
    constexpr int kUnroll = 8;  // loop unrolling factor
    const auto nthread = static_cast<dmlc::omp_uint>(builder->nthread_);
    const size_t nrows = instance_set.size();
    const size_t rest = nrows % kUnroll;

    #pragma omp parallel for num_threads(nthread) schedule(guided)
    for (dmlc::omp_uint i = 0; i < nrows - rest; i += kUnroll) {
      const dmlc::omp_uint tid = omp_get_thread_num();
      const size_t off = tid * builder->nbin_;
      size_t rid[kUnroll];
      size_t ibegin[kUnroll];
      size_t iend[kUnroll];
      GradientPair stat[kUnroll];
      for (int k = 0; k < kUnroll; ++k) {
        rid[k] = instance_set[i + k];
      }
      for (int k = 0; k < kUnroll; ++k) {
        ibegin[k] = gmat.row_ptr[rid[k]];
        iend[k] = gmat.row_ptr[rid[k] + 1];
      }
      for (int k = 0; k < kUnroll; ++k) {
        stat[k] = gpair[rid[k]];
      }
      for (int k = 0; k < kUnroll; ++k) {
        for (size_t j = ibegin[k]; j < iend[k]; ++j) {
          const uint32_t bin = gmat.index[j];
          builder->data_[off + bin].Add(stat[k]);
        }
      }
    }
    for (size_t i = nrows - rest; i < nrows; ++i) {
      const size_t rid = instance_set[i];
      const size_t ibegin = gmat.row_ptr[rid];
      const size_t iend = gmat.row_ptr[rid + 1];
      const GradientPair stat = gpair[rid];
      for (size_t j = ibegin; j < iend; ++j) {
        const uint32_t bin = gmat.index[j];
        builder->data_[bin].Add(stat);
      }
    }

    /* reduction */
    const uint32_t nbin = builder->nbin_;
    #pragma omp parallel for num_threads(nthread) schedule(static)
    for (dmlc::omp_uint bin_id = 0; bin_id < dmlc::omp_uint(nbin); ++bin_id) {
      for (dmlc::omp_uint tid = 0; tid < nthread; ++tid) {
        (*hist)[bin_id].Add(builder->data_[tid * builder->nbin_ + bin_id]);
      }
    }
}


void GHistBuilder::BuildHist(const std::vector<GradientPair>& gpair,
                                                         const std::vector<size_t>& instance_set,
                                                         const GHistIndexMatrix& gmat,
                                                         std::vector<GHistEntry>* hist)
{
    auto t1 = dmlc::GetTime();
    BuildHistXGBoost(gpair, instance_set, gmat, hist, this);
    auto t2 = dmlc::GetTime();
    LOG(INFO) << "XGBoost = " << (t2-t1);

    auto t3 = dmlc::GetTime();
    BuildHist_Blocking512(gpair, instance_set, gmat, hist, this);
    auto t4 = dmlc::GetTime();
    LOG(INFO) << "Blocking by 512 = " << (t4-t3);

    #if defined(__INTEL_COMPILER)
        auto t7 = dmlc::GetTime();
        BuildHist_Blocking512_AVX512(gpair, instance_set, gmat, hist, this);
        auto t8 = dmlc::GetTime();
        LOG(INFO) << "Blocking by 512 + AVX512 = " << (t8-t7);
    #endif
}

}  // namespace perflab
