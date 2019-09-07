#include <iostream>
#include <memory>

#include "DataLoader.hpp"
#include "Logging.hpp"

using namespace DLFS;
using namespace std;

DataLoader::DataLoader(const std::string &examples_path, const std::string &local_data_dir)
    : m_dataSource(local_data_dir)
{
    m_exampleSource.Init(examples_path);
    m_batchIndex = m_exampleIndex = 0;
    m_totalBatches = m_exampleSource.GetNumExamples() / m_batchSize;

    LOG.INFO() << "ExampleSource " << examples_path << ", " << m_exampleSource.GetNumExamples()
               << " examples";
}

DataLoader::~DataLoader()
{
}

void DataLoader::GetNextBatch()
{
    // New data
    TensorPtr<uint8_t> imgBatchTensor = std::make_shared<Tensor<uint8_t>>("ImageBatchTensor");
    TensorPtr<float> bboxBatchTensor = std::make_shared<Tensor<float>>("BboxBatchTensor");
    TensorPtr<float> catIdBatchTensor = std::make_shared<Tensor<float>>("CatIdBatchTensor");    

    // Load examples and image binaries
    vector<vector<uint8_t>> imgBufs(m_batchSize);
    vector<vector<BBoxArray>> boxes(m_batchSize);
    vector<vector<uint32_t>> catIds(m_batchSize);

    TensorShapeList boxShape(m_batchSize);
    for (unsigned int i = 0; i < m_batchSize; i++)
    {
        const Example *ex = m_exampleSource.GetExample(m_exampleIndex);

        m_dataSource.getBlob(ex->file_name()->str(), imgBufs[i]);

        LOG.DEBUG() << "Loaded image from: " << ex->file_name()->str();

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

    bboxBatchTensor->SetShape(m_batchSize, annMaxDims[1], 4, 1);
    catIdBatchTensor->SetShape(m_batchSize, annMaxDims[1], 1, 1);

    bboxBatchTensor->AllocateIfNecessary();
    catIdBatchTensor->AllocateIfNecessary();

    LOG.DEBUG() << "Batch BBOX shape " << bboxBatchTensor->PrintShape();

    unsigned int idx = 0;
    for (auto devPtr : bboxBatchTensor->GetIterablePointersOverBatch())
    {
        LOG.DEBUG() << "Bbox Tensor: Copying " << boxes[idx].size() * sizeof(float) << "bytes to" << uint64_t(devPtr);
        checkCudaErrors(cudaMemcpy(devPtr, boxes[idx].data(),
                                   boxes[idx].size() * sizeof(float), cudaMemcpyHostToDevice));
        idx++;
    }

    idx = 0;
    for (auto devPtr : catIdBatchTensor->GetIterablePointersOverBatch())
    {
        LOG.DEBUG() << "CatId Tensor: Copying " << boxes[idx].size() * sizeof(float) << "bytes to" << uint64_t(devPtr);
        checkCudaErrors(cudaMemcpy(devPtr, catIds[idx].data(),
                                   catIds[idx].size() * sizeof(float), cudaMemcpyHostToDevice));
        idx++;
    }

    checkCudaErrors(cudaDeviceSynchronize());

    // Decode Jpegs.
    m_imgLoader.DecodeJPEG(imgBufs, imgBatchTensor);

    // Push into queue
    m_batchesReady.emplace(std::make_tuple(imgBatchTensor, bboxBatchTensor, catIdBatchTensor));
}