import awkward  as ak
import vector


def Px(obj_p4):
    return obj_p4.px

def Py(obj_p4):
    return obj_p4.py

def Pz(obj_p4):
    return obj_p4.pz

def E(obj_p4):
    return obj_p4.E

def GetNumPyArray(awk_arr, tot_length, i):
    return ak.to_numpy(ak.fill_none(ak.pad_none(awk_arr[:, :tot_length], tot_length), 0.0))[:, i]