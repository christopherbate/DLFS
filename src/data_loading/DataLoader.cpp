#include "DataLoader.hpp"
#include <iostream>

using namespace DLFS;
using namespace std;

DataLoader::DataLoader(const std::string &examples_path, const std::string &local_data_dir,
                       unsigned int batchSize)
    : m_batchSize(batchSize), m_dataSource(local_data_dir),
      m_imgLoader(batchSize)
{
    m_exampleSource.init(examples_path);
    m_batchIndex = m_exampleIndex = 0;
    m_length = m_exampleSource.GetNumExamples() / m_batchSize;
    m_currPipelineBatch = 0;
}

DataLoader::~DataLoader()
{
}

void DataLoader::GetNextBatch()
{
    // retrieve next available tensor
    assert(m_currPipelineBatch < DATALOADER_PIPELINE_SIZE);
    Tensor &imgBatchTensor = std::get<0>(m_exampleBatches[m_currPipelineBatch]);
    Tensor &bboxBatchTensor = std::get<1>(m_exampleBatches[m_currPipelineBatch]);
    Tensor &catIdBatchTensor = std::get<2>(m_exampleBatches[m_currPipelineBatch]);
    m_currPipelineBatch = (m_currPipelineBatch + 1) % DATALOADER_PIPELINE_SIZE;

    // Load examples and image binaries
    vector<vector<uint8_t>> imgBufs(m_batchSize);
    vector<vector<BBoxArray>> boxes(m_batchSize);
    vector<vector<uint32_t>> catIds(m_batchSize);
    TensorShapeList boxShape(m_batchSize);
    for (auto i = 0; i < m_batchSize; i++)
    {
        const Example *ex = m_exampleSource.GetExample(m_exampleIndex);

        m_dataSource.getBlob(ex->file_name()->str(), imgBufs[i]);

        unsigned int numAnn = ex->annotations()->Length();
        boxShape.SetShape(i, {1, (int)numAnn, 4, 1});
        m_exampleIndex = (m_exampleIndex + 1) % m_exampleSource.GetNumExamples();
        for (auto ann : *ex->annotations())
        {
            catIds[i].push_back(ann->cat_id());
            boxes[i].push_back({ann->bbox()->y1(), ann->bbox()->x1(),
                                ann->bbox()->y2(), ann->bbox()->x2()});
        }
    }

    // Allocate the label tensors.
    auto annMaxDims = boxShape.FindMaxDims();
    bboxBatchTensor.SetShape(m_batchSize, annMaxDims[1], 4, 1);
    bboxBatchTensor.SetPitch(4);
    catIdBatchTensor.SetShape(m_batchIndex, annMaxDims[1], 1, 1);
    catIdBatchTensor.SetPitch(4);
    bboxBatchTensor.AllocateIfNecessary();
    catIdBatchTensor.AllocateIfNecessary();

    unsigned int idx = 0;
    for (auto devPtr : bboxBatchTensor.GetIterablePointersOverBatch())
    {
        checkCudaErrors(cudaMemcpy(devPtr, boxes[idx].data(),
                                   boxes[idx].size()*sizeof(float), cudaMemcpyHostToDevice));
        idx++;                                   
    }
    
    idx = 0;
    for (auto devPtr : catIdBatchTensor.GetIterablePointersOverBatch())
    {
        checkCudaErrors(cudaMemcpy(devPtr, catIds[idx].data(),
                                   catIds[idx].size()*sizeof(float), cudaMemcpyHostToDevice));
        idx++;                                   
    }

    // Decode Jpegs.
    m_imgLoader.DecodeJPEG(imgBufs, imgBatchTensor);

    // Increment the batch index.
    m_batchIndex = (m_batchIndex + 1) % m_length;
}

int DataLoader::GetLength()
{
    return m_length;
}

void DataLoader::Summary()
{
    cout << "DATA LOADER:";
    cout << " Batch Sz: " << m_batchSize;
    cout << " Num Ex: " << m_exampleSource.GetNumExamples();
    cout << " Curr. Batch: " << m_batchIndex << "/" << m_length << endl;
}