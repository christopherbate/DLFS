# automatically generated by the FlatBuffers compiler, do not modify

# namespace: DLFS

import flatbuffers

class Category(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsCategory(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Category()
        x.Init(buf, n + offset)
        return x

    # Category
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Category
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Category
    def Id(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 0

def CategoryStart(builder): builder.StartObject(2)
def CategoryAddName(builder, name): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)
def CategoryAddId(builder, id): builder.PrependUint16Slot(1, id, 0)
def CategoryEnd(builder): return builder.EndObject()
