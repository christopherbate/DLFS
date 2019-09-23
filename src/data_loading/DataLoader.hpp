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

#include <mutex>
#include <string>
#include <vector>
#include <fstream>
#include <queue>
#include <thread>
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
    DataLoader(const std::string &dataset_path = "");
    ~DataLoader();

    /**
     * LoadDataset
     * Loads a dataset flatbuffer db from a specific path
     */
    void LoadDataset(const std::string &dataset_path)
    {
        m_exampleSource.Init(dataset_path);
        m_batchIndex = m_exampleIndex = 0;
        m_totalBatches = m_exampleSource.GetNumExamples() / m_batchSize;

        LOG.INFO() << "Dataset: " << dataset_path << ", " << m_exampleSource.GetNumExamples()
                   << " examples found";
    }

    /**
     * Set data source path 
     * 
     * Tells the m_dataSource where to find files.
     * 
     * The data source is only used if this is specified, otherwise we look for imags 
     * in the datasets db.
     */
    void SetDataSourcePath(const std::string &data_src_path)
    {
        if(!m_dataSource.get()){
            m_dataSource = std::make_unique<LocalSource>();
        }
        m_dataSource->SetPath(data_src_path);
    }

    /**
     * Loads a single batch into the queue
     */
    void RunOnce();

    inline void SetBatchSize(unsigned int batchSize)
    {
        m_batchSize = batchSize;
        m_totalBatches = m_exampleSource.GetNumExamples() / m_batchSize;
    }

    inline unsigned int GetBatchSize()
    {
        return m_batchSize;
    }

    /**
     * Returns total number of BATCHES
     */
    inline int GetLength()
    {
        return m_totalBatches;
    }

    /**
     * Prints log messages for 
     * current batch, and statistics (time per batch, queue size, etc.)
     */
    inline void Summary()
    {
        LOG.INFO() << "DATA LOADER:";
        LOG.INFO() << "--Curr. Batch: " << m_batchIndex << "/" << m_totalBatches;
        LOG.INFO() << "--Queued Batches: " << m_batchesReady.size();
    }

    /**
     * Prefetch Queue size
     */
    inline void SetPrefetch(unsigned int prefetchLimit)
    {
        m_prefetchLimit = prefetchLimit;
    }

    /**
     * Returns next batch
     */
    ObjDetExampleBatch DequeBatch()
    {
        ObjDetExampleBatch batch = m_batchesReady.front();
        m_batchesReady.pop();
        return batch;
    }

    /**
     * Controls Image Decoding
     */
    void EnableJpegDecoder()
    {
        m_useJpegDecoder = true;
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

    ExampleSource m_exampleSource;
    std::mutex m_exSrcMutex;

    ImageLoader m_imgLoader;
    bool m_useJpegDecoder{false};

    std::unique_ptr<DataSource> m_dataSource{nullptr};

    /**
     * Threading
     */

    // The set of batches which are already loaded.
    std::queue<ObjDetExampleBatch> m_batchesReady;
};
} // namespace DLFS