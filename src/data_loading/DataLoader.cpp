#include <iostream>
#include <memory>

#include "DataLoader.hpp"
#include "Logging.hpp"

using namespace DLFS;
using namespace std;

DataLoader::DataLoader(const std::string &examples_path)
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
    TensorPtr<uint16_t> catIdBatchTensor = std::make_shared<Tensor<uint16_t>>("CatIdBatchTensor");

    // Load examples and image binaries
    vector<vector<uint8_t>> imgBufs(m_batchSize);
    vector<vector<BBoxArray>> boxes(m_batchSize);
    vector<vector<uint32_t>> catIds(m_batchSize);
    vector<uint32_t> widths(m_batchSize);
    vector<uint32_t> heights(m_batchSize);

    TensorShapeList boxShape(m_batchSize);
    for (unsigned int i = 0; i < m_batchSize; i++)
    {
        const Example *ex = m_exampleSource.GetExample(m_exampleIndex);

        // Check if the example has an image array, if so, load it.
        if (ex->image() != NULL)
        {            
            imgBufs[i].assign(ex->image()->begin(), ex->image()->end());
        }
        else
        {
            if(!m_useFileSource){
                throw std::runtime_error("DataLoader was not configured with a DataSource, but .db does not contain images.");
            }            
            m_dataSource.GetBlob(ex->file_name()->str(), imgBufs[i]);
        }

        unsigned int numAnn = ex->annotations()->Length();

        widths[i] = ex->width();
        heights[i] = ex->height();

        LOG.DEBUG() << "Image " << ex->id() << " has " << numAnn << " annotations";

        boxShape.SetShape(i, {1, (int)numAnn, 4, 1});

        m_exampleIndex = (m_exampleIndex + 1) % m_exampleSource.GetNumExamples();

        for (auto ann : *ex->annotations())
        {            
            // Add the cat id.
            catIds[i].push_back((uint16_t)ann->cat_id());
            LOG.DEBUG() << "Found ann with cat id: " << ann->cat_id();

            if (ann->bbox())
            {
                // TODO: Should add option to throw if no bbox
                boxes[i].push_back({ann->bbox()->y1(), ann->bbox()->x1(),
                                    ann->bbox()->y2(), ann->bbox()->x2()});
            }
        }
    }

    // Allocate the label tensors.
    auto annMaxDims = boxShape.FindMaxDims();

    bboxBatchTensor->SetShape(m_batchSize, annMaxDims[1], 4, 1);
    catIdBatchTensor->SetShape(m_batchSize, annMaxDims[1], 1, 1);

    bboxBatchTensor->AllocateIfNecessary();
    catIdBatchTensor->AllocateIfNecessary();
    
    LOG.DEBUG() << "Building batch bbox tensor: " << bboxBatchTensor->PrintShape();
    unsigned int idx = 0;
    for (auto devPtr : bboxBatchTensor->GetIterablePointersOverBatch())
    {        
        checkCudaErrors(cudaMemcpy(devPtr, boxes[idx].data(),
                                   boxes[idx].size() * sizeof(float), cudaMemcpyHostToDevice));
        idx++;
    }

    idx = 0;
    LOG.DEBUG() << "Building batch catId tensor: " << catIdBatchTensor->PrintShape();
    for (auto devPtr : catIdBatchTensor->GetIterablePointersOverBatch())
    {        
        checkCudaErrors(cudaMemcpy(devPtr, catIds[idx].data(),
                                   catIds[idx].size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        idx++;
    }

    // Load the images onto the device buffer.
    // either decode jpegs or directly copy
    if(m_useJpegDecoder){
        m_imgLoader.DecodeJPEG(imgBufs, imgBatchTensor);
    } else {        
        uint32_t maxWidth = 0;
        uint32_t maxHeight = 0;
        for(unsigned i = 0; i < widths.size(); i++){
            maxWidth = std::max(maxWidth, widths[i]);
            maxHeight = std::max(maxHeight, heights[i]);
        }
        LOG.DEBUG() << "Allocating image with shape " << maxHeight << " " <<  maxWidth;
        imgBatchTensor->SetShape(m_batchSize, maxWidth, maxHeight, 1);        
        imgBatchTensor->AllocateIfNecessary(); 
        imgBatchTensor->CopyBatchBuffersToDevice(imgBufs);
    }

    checkCudaErrors(cudaDeviceSynchronize());    

    // Push into queue
    m_batchesReady.emplace(std::make_tuple(imgBatchTensor, bboxBatchTensor, catIdBatchTensor));
}