#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include <fstream>

#include "DataSource.hpp"
#include "AnnotationSource.hpp"
namespace DLFS{
class DataLoader
{
public: 
    DataLoader();
    ~DataLoader();

    int get_length();
    int get_batch();

private:
    int m_batchSize;
    int m_length;
    int m_batchIndex;
    DataSource *m_dataSource;
    AnnotationSource m_annSource;
};
}

#endif