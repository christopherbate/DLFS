#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include <fstream>
#include <queue>
#include <tuple>

#include "LocalSource.hpp"
#include "ExampleSource.hpp"
#include "ImageLoader.hpp"
#include "../tensor/TensorList.hpp"

#define DATALOADER_PIPELINE_SIZE 5

namespace DLFS
{

typedef std::tuple<Tensor<uint8_t>, Tensor<float>, Tensor<float>> ObjDetExampleBatch;
typedef std::array<float, 4> BBoxArray;

class DataLoader
{
public:
    DataLoader(const std::string &examples_path,
               const std::string &local_data_dir);
    ~DataLoader();

    void GetNextBatch();

    inline void SetBatchSize(unsigned int batchSize)
    {
        m_batchSize = batchSize;
        m_totalBatches = m_exampleSource.GetNumExamples() / m_batchSize;
    }

    inline unsigned int GetBatchSize()
    {
        return m_batchSize;
    }

    inline int GetLength()
    {
        return m_totalBatches;
    }

    inline void Summary()
    {
        LOG.INFO() << "DATA LOADER:";
        LOG.INFO() << " Curr. Batch: " << m_batchIndex << "/" << m_totalBatches;
    }

    inline void SetPrefetch(unsigned int prefetchLimit)
    {
        m_prefetchLimit = prefetchLimit;
    }

private:
    unsigned int m_batchSize{1};

    // the total number of non-fractional batches
    int m_totalBatches;

    // The current batch number within the total dataset of batches
    int m_batchIndex;

    // The current total example index in the dataset
    int m_exampleIndex;

    // The limit on the size of m_batchesReady
    unsigned int m_prefetchLimit{1};

    LocalSource m_dataSource;
    ExampleSource m_exampleSource;
    ImageLoader m_imgLoader;

    // The set of batches which are already loaded.
    std::queue<ObjDetExampleBatch> m_batchesReady;
};
} // namespace DLFS

#endif