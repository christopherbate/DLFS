/**
 *  DataLoader owns a Source (def: LocalSource), ExampleSource (def: flatbuffer file), 
 *      and other pre-processing loading objects (ImageLoader)
 *  User declares a dataloader by specifying a location for the source of objects (usually images),
 *  and also specifies a flatbuffer file that contains the annotations.
 * 
 *  Currently data is loaded on demand, but in the future, the loader should run autonomously,
 *  using query times to adjust the amount of pipelining (simple feedback loop / PID)
 * 
 *  - Handles pipelining the batches
 *  - TODO: Multi-threadded implementation
 *  - TODO: Add augmentation interfaces and options
*/
#pragma once

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

typedef std::tuple<TensorPtr<uint8_t>, TensorPtr<float>, TensorPtr<uint16_t>> ObjDetExampleBatch;
typedef std::array<float, 4> BBoxArray;

class DataLoader
{
public:
    DataLoader(const std::string &examples_path);
    ~DataLoader();
    
    void SetFileSource(const std::string &data_src)
    {
        m_useFileSource = true;
        m_dataSource.SetDirectory(data_src);
    }

    void GetNextBatch();
    
    inline void SetUseJpegDecoder(bool val){
        m_useJpegDecoder = val;
    }

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
    bool m_useFileSource{false};
    bool m_useJpegDecoder{false};

    // The set of batches which are already loaded.
    std::queue<ObjDetExampleBatch> m_batchesReady;
};
} // namespace DLFS