#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include <fstream>
#include <tuple>

#include "LocalSource.hpp"
#include "ExampleSource.hpp"
#include "ImageLoader.hpp"
#include "../tensor/TensorList.hpp"

#define DATALOADER_PIPELINE_SIZE 5

namespace DLFS
{

typedef std::tuple<Tensor, Tensor, Tensor> ObjDetExampleBatch;
typedef std::array<float, 4> BBoxArray;

class DataLoader
{
public:
    DataLoader(const std::string &examples_path,
               const std::string &local_data_dir,
               unsigned int batchSize = 1);
    ~DataLoader();

    int GetLength();
    void GetNextBatch();

    void Summary();    

private:
    unsigned int m_batchSize;
    int m_length;
    int m_batchIndex;
    int m_exampleIndex;
    LocalSource m_dataSource;
    ExampleSource m_exampleSource;
    ImageLoader m_imgLoader;

    std::array<ObjDetExampleBatch, DATALOADER_PIPELINE_SIZE> m_exampleBatches;
    unsigned int m_currPipelineBatch;
};
} // namespace DLFS

#endif 