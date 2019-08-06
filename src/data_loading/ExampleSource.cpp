#include "ExampleSource.hpp"
#include "dataset_generated.h"
#include "../utils/Timer.hpp"

#include <iostream>
#include <fstream>

using namespace DLFS;
using namespace std;

ExampleSource::ExampleSource()
{
    m_buffer = NULL;
}

ExampleSource::~ExampleSource()
{
    if (m_buffer)
    {
        delete[] m_buffer;
        m_buffer = NULL;
    }
}

void ExampleSource::init(const string &path)
{
    ifstream infile(path, ifstream::in | ifstream::binary);
    
    if(!infile.is_open()){
        throw std::runtime_error("Failed to open annotations db.");
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

    cout << "Loaded datset in " << timer.tick() << " msec."
         << "\n";
    cout << "Num Examples: " << m_numExamples << endl;
}

const Example *ExampleSource::GetExample(unsigned int idx)
{    
    return m_dataset->examples()->Get(idx);
}