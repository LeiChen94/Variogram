import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew
from scipy.optimize import curve_fit
from math import exp,sqrt,e
import powerlaw
import random
from sklearn.metrics import r2_score
import time
import filePathInfor

#h-scatterplot

def fun_gaus(x,a,b,c):
    para = [-pow((item-b),2)/2/pow(c,2) for item in x]
    return [a*exp(item) for item in para]
def model_gaus(x,c0,c1,a):
    return [c0+c1*(1-exp(-pow(item/a,2))) for item in x]
def model_exp(x,c0,c1,a):
    return [c0+c1*(1-exp(-(item/a))) for item in x]
def model_spher(x,c0,c1,a):
    return [c0+c1*(1.5*item/a - 0.5*pow((item/a),3)) for item in x]
def model_poly(x,a,b,c):
    return [ a*pow(item,2)+b*item+c for item in x]


def save_result(v,filepath):
    f = open(filename,'wb')
    pickle.dump(v,f)
    f.close()

#半变异函数参数
nuggetList = []
rangeList = []
kList = []
sillList = []

maxLag = 500     # half of study range
sigmaList = range(150,151,20)
cellSize = range(10,85,5)
simulateList = range(11,13)
sampleNumList = range(400000,400001,10000)

nuggetList = []
rangeList = []

dtaType = 'single_nor'
for para in sigmaList:
    for simu in simulateList:
        start = time.time()
        c0_list = []
        c1_list = []
        k_gaus_list = []
        k_poly_list = []
        k_spher_list = []
        r2_list = []
        for csize in cellSize:
            ## 1 数据预处理(对数变换、求均值）
            odir = filePathInfor.get_filePath(dtaType,para,simu)
            df = pd.read_csv(odir+'/nor_varidf'+str(csize)+'.csv')
            pt_df = pd.read_csv(odir+'/xy_dta.csv')
            sampleNum = len(pt_df)
            df = df[df['vari'] > 0]
            #df['vari'] = df['vari']/pow(cellsize/1000,4)   #/K^4
            tor = csize*0.4
            gap = int(tor*2)
            skewraw_list = []
            skewlog_list = []
            laglist = []
            lenlist = []
            semilog_list = []
            semiraw_list = []
            mean_list =[]
            median_list = []
            boxdta_list = []
            boxlabel_list =[]
            count = -1
            gapnum = int((maxLag - csize)/(csize*0.8))
            for lag in range(csize,maxLag,gap):
                count += 1
                dta = df[(df['lags'] > lag - tor) & (df['lags'] < lag + tor)]
                lendta = len(dta)
                rawdta = dta['vari']
                logdta = np.log(rawdta)

                if(len(dta)<30):
                    print(lendta,lag)
                    break
                else:
                    laglist.append(lag/1000)
                    semiraw = round(np.mean(rawdta), 2) / 2
                    semiraw_list.append(semiraw)
                    semilog = round(np.mean(logdta), 2) / 2
                    semilog_list.append(semilog)
                    skew_raw = skew(rawdta)
                    skew_log = round(skew(logdta),2)
                    lenlist.append(lendta)
                    skewraw_list.append(skew_raw)
                    skewlog_list.append(skew_log)

            # 2 模型拟合_高斯模型
            p_fit1,pcov1 = curve_fit(model_gaus,laglist,semilog_list)
            c0,c1,a = p_fit1.tolist()
            y1 = model_gaus(laglist,*p_fit1)
            k1 = round(c0/(c0+c1),2)
            k_gaus_list.append(k1)
            c0_list.append(c0)
            c1_list.append(c1)
            r2 = round(r2_score(y1,semilog_list),3)
            r2_list.append(r2)

        #3 保存计算结果
        res_dict = {'cellsize':cellSize,'c0':c0_list,'c1':c1_list,'k':k_gaus_list,'r2':r2_list}
        res_df = pd.DataFrame(res_dict)
        res_df.to_csv(odir+'/k_gaus_'+str(para)+'_'+str(simu)+'.csv',index = None)
        end = time.time()
        consume = end - start


        print('sigma = '+str(para)+',simulate = '+str(simu)+',TimeConsumed = '+str(consume)+'\n')
        plt.scatter(cellSize, k_gaus_list)
        plt.plot(cellSize,k_gaus_list, label='gaus')
        if dtaType == 'single_nor':
            title = 'single_sigma = '+str(para)+',sampleNum = '+str(sampleNum)
        elif dtaType == 'dual_nor':
            title = 'dual_sigma = '+str(para)+',sampleNum = '+str(sampleNum/2)
        plt.title(title)
        plt.show()


m = 1