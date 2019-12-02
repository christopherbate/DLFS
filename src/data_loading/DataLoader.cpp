#include <iostream>
#include <memory>

#include "DataLoader.hpp"
#include "Logging.hpp"

using namespace DLFS;
using namespace std;

DataLoader::DataLoader(const std::string &dataset_path) {
    if (dataset_path.size() > 0) {
        LoadDataset(dataset_path);
    }
}

DataLoader::~DataLoader() {}

void DataLoader::RunOnce() {
    // New data
    TensorPtr<uint8_t> imgBatchTensor =
        std::make_shared<Tensor<uint8_t>>("ImageBatchTensor");
    TensorPtr<float> bboxBatchTensor =
        std::make_shared<Tensor<float>>("BboxBatchTensor");
    TensorPtr<uint32_t> catIdBatchTensor =
        std::make_shared<Tensor<uint32_t>>("CatIdBatchTensor");

    // Load examples and image binaries
    vector<vector<uint8_t>> imgBufs(m_batchSize);
    vector<vector<BBoxArray>> boxes(m_batchSize);
    vector<vector<uint32_t>> catIds(m_batchSize);
    vector<uint32_t> widths(m_batchSize);
    vector<uint32_t> heights(m_batchSize);
    TensorShapeList boxShape(m_batchSize);
    uint32_t maxWidth = 0;
    uint32_t maxHeight = 0;

    for (unsigned int i = 0; i < m_batchSize; i++) {
        LOG.DEBUG() << "Loading example " << i;
        const Example *ex = nullptr;
        {
            std::lock_guard<std::mutex> lk(m_exSrcMutex);
            ex = m_exampleSource.GetExample(m_exampleIndex);
            m_exampleIndex =
                (m_exampleIndex + 1) % m_exampleSource.GetNumExamples();
        }

        LOG.DEBUG() << "Loading image " << i;

        // Check if the example has an image array, if so, load it.
        if (ex->image() != NULL) {
            imgBufs[i].assign(ex->image()->begin(), ex->image()->end());
        } else if (m_dataSource.get() != nullptr) {
            m_dataSource->GetBlob(ex->file_name()->str(), imgBufs[i]);
        } else {
            throw std::runtime_error(
                "Example did not have an image, and no data source specified.");
        }

        unsigned int numAnn = ex->annotations()->Length();

        widths[i] = ex->width();
        heights[i] = ex->height();
        maxWidth = std::max(maxWidth, widths[i]);
        maxHeight = std::max(maxHeight, heights[i]);

        LOG.DEBUG() << "Image " << ex->id() << " has " << numAnn
                    << " annotations";

        boxShape.SetShape(i, {1, (int)numAnn, 4, 1});

        for (auto ann : *ex->annotations()) {
            // Add the cat id.
            catIds[i].push_back((uint32_t)ann->cat_id());
            LOG.DEBUG() << "Found ann with cat id: " << ann->cat_id();

            if (ann->bbox()) {
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

    LOG.DEBUG() << "Building batch bbox tensor: "
                << bboxBatchTensor->PrintShape();
    unsigned int idx = 0;
    for (auto devPtr : bboxBatchTensor->GetIterablePointersOverBatch()) {
        checkCudaErrors(cudaMemcpy(devPtr, boxes[idx].data(),
                                   boxes[idx].size() * sizeof(boxes[idx][0]),
                                   cudaMemcpyHostToDevice));
        idx++;
    }

    idx = 0;
    LOG.DEBUG() << "Building batch catId tensor: "
                << catIdBatchTensor->PrintShape();
    for (auto devPtr : catIdBatchTensor->GetIterablePointersOverBatch()) {
        checkCudaErrors(cudaMemcpy(devPtr, catIds[idx].data(),
                                   catIds[idx].size() * sizeof(catIds[idx][0]),
                                   cudaMemcpyHostToDevice));
        idx++;
    }

    // Load the images onto the device buffer.
    // either decode jpegs or directly copy
    if (m_useJpegDecoder) {
        m_imgLoader.BatchDecodeJPEG(imgBufs, imgBatchTensor);
    } else {
        LOG.DEBUG() << "Allocating image with shape " << maxHeight << " "
                    << maxWidth;
        imgBatchTensor->SetShape(m_batchSize, maxWidth, maxHeight, 1);
        imgBatchTensor->AllocateIfNecessary();
        imgBatchTensor->CopyBatchBuffersToDevice(imgBufs);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    // Push into queue
    m_batchesReady.emplace(
        std::make_tuple(imgBatchTensor, bboxBatchTensor, catIdBatchTensor));

    m_batchIndex = (m_batchIndex + 1) % m_totalBatches;
}