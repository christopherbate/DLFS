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

#include <cerrno>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
using json = nlohmann::json;
using namespace DLFS;

const string usage = "convert_coco [path/to/anns.json] [path/to/output.anns]";

BoundingBox convertToBBOX(int x, int y, int w, int h) {
    BoundingBox bbox(y, x, y + h, x + w);
    return bbox;
}

unsigned int Tick() {
    static chrono::time_point<chrono::steady_clock> last_time =
        chrono::steady_clock::now();

    auto time_now = chrono::steady_clock::now();

    unsigned int elapsed =
        chrono::duration_cast<chrono::milliseconds>(time_now - last_time)
            .count();

    last_time = time_now;

    return elapsed;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        cout << usage << endl;
        return 0;
    }

    Tick();

    string input_path = argv[1];
    string output_path = argv[2];

    // read in the json file.
    ifstream infile(input_path);
    if (!infile.is_open()) {
        cerr << "Failed to open input file. " << errno << endl;
        return -1;
    }

    json input_json;
    infile >> input_json;

    cout << "Loaded JSON dataset in " << Tick() << " msec." << endl;

    // Create the builder
    flatbuffers::FlatBufferBuilder builder(1024);

    // Forward declare the data structures we will need.
    unordered_map<unsigned int, vector<flatbuffers::Offset<Annotation>>>
        imageIdToAnns; /*imageId to Annotations*/
    unordered_map<unsigned int, set<unsigned int>>
        catIdToExampleIdSet; /*catId to unique Example Ids*/
    unordered_map<uint64_t, uint64_t> imgIdRemap;
    vector<flatbuffers::Offset<Example>> ex_vector;   /* Example list */
    vector<flatbuffers::Offset<Category>> cat_vector; /* Cateogry List */

    // Build annotations map
    auto anns = input_json["annotations"];
    for (auto it = anns.begin(); it != anns.end(); it++) {
        uint64_t imageId = (*it)["image_id"];
        uint64_t annId = (*it)["id"];
        auto cocoBbox = (*it)["bbox"];
        uint16_t catId = (*it)["category_id"];

        auto bbox =
            convertToBBOX(cocoBbox[0], cocoBbox[1], cocoBbox[2], cocoBbox[3]);
        float area = (bbox.x2() - bbox.x1()) * (bbox.y2() - bbox.y1());
        auto ann =
            CreateAnnotation(builder, &bbox, catId, annId, imageId, area);

        imageIdToAnns[imageId].push_back(ann);
        catIdToExampleIdSet[catId].insert(imageId);
    }

    cout << "Built image to annotation map in " << Tick() << " msec." << endl;

    // Build images
    auto imgs = input_json["images"];
    unsigned int skip_count = 0;
    unsigned int total_count = 0;
    unsigned int idx = 0;
    for (auto it = imgs.begin(); it != imgs.end(); it++) {
        uint64_t imgId = (*it)["id"];
        if (imageIdToAnns[imgId].size() == 0) {
            cout << "WARNING: skipping imageId " << imgId
                 << " as it has no annotations." << endl;
            skip_count++;
            continue;
        }
        string file_name = (*it)["file_name"];
        auto fb_file_name = builder.CreateString(file_name);
        auto fb_anns = builder.CreateVector(imageIdToAnns[imgId]);

        auto img = CreateExample(builder, fb_file_name, imgId, fb_anns, 0,
                                 (*it)["width"], (*it)["height"], idx);
        imgIdRemap[imgId] = idx;

        ex_vector.push_back(img);
        idx++;
        total_count++;
    }
    auto fb_examples = builder.CreateVectorOfSortedTables(&ex_vector);

    // Build categories.
    auto categories = input_json["categories"];
    for (auto it = categories.begin(); it != categories.end(); it++) {
        string cat_name = (*it)["name"];
        auto fb_cat_name = builder.CreateString(cat_name);
        uint16_t cat_id = (*it)["id"];

        // Create the example id vector
        std::vector<uint64_t> img_ids;
        cout << "Category " << cat_id << " ids: ";
        for (auto set_it = catIdToExampleIdSet[cat_id].begin();
             set_it != catIdToExampleIdSet[cat_id].end(); set_it++) {
            img_ids.push_back(imgIdRemap[*set_it]);
            cout << imgIdRemap[*set_it] << " ";
        }
        cout << endl;

        auto fb_cat_ex = builder.CreateVector(img_ids);
        auto cat = CreateCategory(builder, fb_cat_name, cat_id, fb_cat_ex,
                                  img_ids.size(), img_ids.size());
        cat_vector.push_back(cat);
    }
    auto fb_cats = builder.CreateVectorOfSortedTables(&cat_vector);

    cout << "Serialized categories in " << Tick() << " msec." << endl;

    cout << "Serialized images in " << Tick() << " msec." << endl;

    cout << "Total images : " << total_count << " Skipped images " << skip_count
         << endl;

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