#include "AnnotationSource.hpp"
#include "annotations_generated.h"
#include "../utils/Timer.hpp"

#include <iostream>
#include <fstream>

using namespace DLFS;
using namespace std;

void AnnotationSource::init(const string &path)
{
    ifstream infile(path, ifstream::in | ifstream::binary);

    Timer timer;
    timer.tick();

    infile.seekg(0, std::ios::end);
    int length = infile.tellg();
    infile.seekg(0, std::ios::beg);
    char *data = new char[length];
    infile.read(data, length);
    infile.close();

    m_dataset = GetMutableDataset(data);        

    m_numExamples = m_dataset->examples()->Length();

    cout << "Loaded datset in " << timer.tick() << " msec." << "\n";
    cout << "Num Examples: " << m_numExamples << endl;
}

const Example* AnnotationSource::GetExample(unsigned int idx)
{    
    return m_dataset->examples()->Get(idx);
}