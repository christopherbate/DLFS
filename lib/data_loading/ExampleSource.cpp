#include <fstream>
#include <iostream>

#include "../Logging.hpp"
#include "ExampleSource.hpp"
#include "dataset_generated.h"
#include "lib/utils/Timer.hpp"

using namespace DLFS;
using namespace std;

ExampleSource::ExampleSource() { m_buffer = NULL; }

ExampleSource::~ExampleSource() {
    if (m_buffer) {
        delete[] m_buffer;
        m_buffer = NULL;
    }
}

void ExampleSource::Init(const string &path) {
    ifstream infile(path, ifstream::in | ifstream::binary);

    if (!infile.is_open()) {
        throw std::runtime_error("Failed to open annotations db " + path);
    }

    Timer timer;
    timer.tick();

    infile.seekg(0, std::ios::end);
    int length = infile.tellg();
    infile.seekg(0, std::ios::beg);
    m_buffer = new char[length];
    infile.read(m_buffer, length);
    infile.close();

    m_dataset = GetMutableDataset(m_buffer);

    m_numExamples = m_dataset->examples()->Length();

    LOG.INFO() << "Loaded serialized datset in " << timer.tick() << " msec.";
    LOG.INFO() << "Num Examples: " << m_numExamples;
}

const Example *ExampleSource::GetExample(unsigned int idx) {
    return m_dataset->examples()->Get(idx);
}