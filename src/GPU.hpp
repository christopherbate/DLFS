#ifndef GPU_H
#define GPU_H

namespace DLFS
{

class GPU
{
public:
    struct Properties
    {
        int deviceNum = 0;
    };

private:
    Properties m_props;

public:
    GPU();
    ~GPU();
};

}

#endif