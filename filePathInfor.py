import pandas as pd


def get_filePath(dtaType,para,simu):
    if dtaType == 'single_nor':
        odir = '../single_nor/nor' + str(para) + '_' + str(simu)
    elif dtaType == 'dual_nor':
        odir = '../dual_nor/nor' + str(para) + '_' + str(simu)
    return odir