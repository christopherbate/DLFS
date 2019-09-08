/**
 * This is a command line utility that converts a COCO annotations JSON file
 * into a flatbuffer with the schema defined in src/data_loading/dataset.fbs
 * 
 * Usage:
 * convert_coco [path to anns.json] [path to output .anns]
 */
#include "../data_loading/DataSource.hpp"
#include "../data_loading/dataset_generated.h"

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cassert>
#include <exception>
#include <unordered_map>

using namespace std;
using namespace DLFS;

const string usage = "convert_mnist [mnist_image_file] [mnist label file] [out file]";

unsigned int Tick()
{
    static chrono::time_point<chrono::steady_clock> last_time = chrono::steady_clock::now();

    auto time_now = chrono::steady_clock::now();

    unsigned int elapsed = chrono::duration_cast<chrono::milliseconds>(time_now - last_time).count();

    last_time = time_now;

    return elapsed;
}

void ReadHeader(std::ifstream &stream,
                std::vector<uint32_t> &header_prealloc)
{
    // Read the number of examples in images.
    stream.read((char *)header_prealloc.data(), header_prealloc.size() * sizeof(uint32_t));
    for (auto &it : header_prealloc)
    {
        it = __builtin_bswap32(it);
        cout << "FileHeader: " << it << endl;
    }
}

void ReadImages(std::ifstream &stream_imgs,
                std::ifstream &stream_labels,
                std::vector<std::vector<uint8_t>> &buffer_imgs,
                std::vector<uint8_t> &labels_buffer,
                unsigned int num_images)
{
    // Seek past the header of 4 ints.
    stream_imgs.seekg(4 * 4);
    const size_t img_size = 28 * 28;
    buffer_imgs.resize(num_images);
    for (unsigned int i = 0; i < num_images; i++)
    {
        buffer_imgs[i].resize(img_size);
        stream_imgs.read((char *)buffer_imgs[i].data(), img_size);
        for (auto &it : buffer_imgs[i])
        {
            it = __builtin_bswap32(it);
        }
        if (!stream_imgs)
        {
            throw std::runtime_error("Failed to read image file.");
        }
    }

    stream_labels.seekg(2 * 4);
    labels_buffer.resize(num_images);
    stream_labels.read((char *)labels_buffer.data(), labels_buffer.size());    
    if (stream_labels.gcount() != (unsigned)labels_buffer.size())
    {
        throw std::runtime_error("Short read: only " + to_string(stream_labels.gcount()) + " labels.");
    }
    for(auto &it : labels_buffer){
        if(it >= 10)
        {
            cout << unsigned(it) << " was detected, labels should be 0-9" << endl;
            throw std::runtime_error("Out of bounds.");
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        cout << usage << endl;
        return 0;
    }

    Tick();

    string input_path_img = argv[1];
    string input_path_label = argv[2];
    string output_path = argv[3];

    vector<uint32_t> img_file_header(4);
    vector<uint32_t> label_file_header(2);
    vector<vector<uint8_t>> image_buffers;
    vector<uint8_t> label_buffer;

    // read in the json file.
    ifstream infile_img(input_path_img, ifstream::in | ifstream::binary);
    if (!infile_img.is_open())
    {
        throw std::runtime_error("Could not open image file.");
    }

    ifstream infile_label(input_path_label, ifstream::in | ifstream::binary);
    if (!infile_label.is_open())
    {
        throw std::runtime_error("Could not open label file.");
    }

    ReadHeader(infile_img, img_file_header);
    ReadHeader(infile_label, label_file_header);

    assert(img_file_header[1] == label_file_header[1]);

    ReadImages(infile_img, infile_label, image_buffers, label_buffer,
               img_file_header[1]);

    cout << "Read images in " << Tick() << " msec." << endl;

    // Create the builder
    flatbuffers::FlatBufferBuilder builder(1024);

    // Build categories.
    vector<string> cat_names(10);
    for (unsigned int i = 0; i < 10; i++)
    {
        cat_names.push_back(to_string(i));
    }

    vector<flatbuffers::Offset<Category>> cat_vector;
    for (unsigned i = 0; i < 10; i++)
    {
        auto fb_cat_name = builder.CreateString(cat_names[i]);
        uint16_t cat_id = i;
        auto cat = CreateCategory(builder, fb_cat_name, cat_id);
        cat_vector.push_back(cat);
    }
    auto fb_cats = builder.CreateVectorOfSortedTables(&cat_vector);

    cout << "Serialized categories in " << Tick() << " msec." << endl;

    // Build annotations map
    unordered_map<unsigned int, vector<flatbuffers::Offset<Annotation>>> imageIdToAnns;
    for (unsigned i = 0; i < img_file_header[1]; i++)
    {
        uint64_t imageId = i;
        uint64_t annId = i;
        uint16_t catId = label_buffer[i];
        auto ann = CreateAnnotation(builder, NULL, catId, annId, imageId);
        imageIdToAnns[imageId].push_back(ann);
    }
    cout << "Built image toannotation map in " << Tick() << " msec." << endl;

    // Build images
    vector<flatbuffers::Offset<Example>> ex_vector;
    for (unsigned i = 0; i < img_file_header[1]; i++)
    {
        uint64_t imgId = i;
        auto fb_anns = builder.CreateVector(imageIdToAnns[imgId]);
        auto img_vector = builder.CreateVector(image_buffers[i]);
        auto ex = CreateExample(builder, 0, imgId, fb_anns, img_vector, 28, 28);
        ex_vector.push_back(ex);
    }
    auto fb_examples = builder.CreateVectorOfSortedTables(&ex_vector);

    cout << "Serialized images in " << Tick() << " msec." << endl;

    DatasetBuilder datasetBuilder(builder);
    datasetBuilder.add_id(0);
    datasetBuilder.add_examples(fb_examples);
    datasetBuilder.add_categories(fb_cats);
    auto dataset = datasetBuilder.Finish();
    builder.Finish(dataset);

    ofstream outfile(output_path, ofstream::out | ofstream::binary);
    outfile.write((const char *)builder.GetBufferPointer(), builder.GetSize());

    cout << "Finished serializing and writing dataset. Size "
         << builder.GetSize() << " in " << Tick() << " msec." << endl;

    return 0;
}