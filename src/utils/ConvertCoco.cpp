/**
 * This is a command line utility that converts a COCO annotations JSON file
 * into a flatbuffer with the schema defined in src/data_loading/dataset.fbs
 * 
 * Usage:
 * convert_coco [path to anns.json] [path to output .anns]
 */
#include "../data_loading/DataSource.hpp"
#include "../data_loading/dataset_generated.h"
#include "external/json.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;
using namespace DLFS;

const string usage = "convert_coco [path/to/anns.json] [path/to/output.anns]";

BoundingBox convertToBBOX(int x, int y, int w, int h)
{
    BoundingBox bbox(y, x, y + h, x + w);
    return bbox;
}

unsigned int Tick()
{
    static chrono::time_point<chrono::steady_clock> last_time = chrono::steady_clock::now();

    auto time_now = chrono::steady_clock::now();

    unsigned int elapsed = chrono::duration_cast<chrono::milliseconds>(time_now - last_time).count();

    last_time = time_now;

    return elapsed;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cout << usage << endl;
        return 0;
    }

    Tick();

    string input_path = argv[1];
    string output_path = argv[2];

    // read in the json file.
    ifstream infile(input_path);
    json input_json;
    infile >> input_json;

    cout << "Loaded JSON dataset in " << Tick() << " msec." << endl;

    // Create the builder
    flatbuffers::FlatBufferBuilder builder(1024);

    // Build categories.
    auto categories = input_json["categories"];
    vector<flatbuffers::Offset<Category>> cat_vector;
    for (auto it = categories.begin(); it != categories.end(); it++)
    {
        string cat_name = (*it)["name"];
        auto fb_cat_name = builder.CreateString(cat_name);
        uint16_t cat_id = (*it)["id"];

        auto cat = CreateCategory(builder, fb_cat_name, cat_id);
        cat_vector.push_back(cat);
    }
    auto fb_cats = builder.CreateVectorOfSortedTables(&cat_vector);

    cout << "Serialized categories in " << Tick() << " msec." << endl;

    // Build annotations map
    auto anns = input_json["annotations"];
    unordered_map<unsigned int, vector<flatbuffers::Offset<Annotation>>> imageIdToAnns;
    for (auto it = anns.begin(); it != anns.end(); it++)
    {
        uint64_t imageId = (*it)["image_id"];
        uint64_t annId = (*it)["id"];
        auto cocoBbox = (*it)["bbox"];
        uint16_t catId = (*it)["category_id"];
        auto bbox = convertToBBOX(cocoBbox[0], cocoBbox[1], cocoBbox[2],
                                  cocoBbox[3]);
        auto ann = CreateAnnotation(builder, &bbox, catId, annId, imageId);

        imageIdToAnns[imageId].push_back(ann);
    }

    cout << "Built image toannotation map in " << Tick() << " msec." << endl;

    // Build images
    auto imgs = input_json["images"];
    vector<flatbuffers::Offset<Example>> ex_vector;
    for (auto it = imgs.begin(); it != imgs.end(); it++)
    {
        uint64_t imgId = (*it)["id"];
        string file_name = (*it)["file_name"];
        auto fb_file_name = builder.CreateString(file_name);
        auto fb_anns = builder.CreateVector(imageIdToAnns[imgId]);

        auto img = CreateExample(builder, fb_file_name, imgId, fb_anns);
        ex_vector.push_back(img);
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