import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

sigmaList = range(260,381,20)
simulateList = range(1,6)
sampleNumList = [20000]#[500000,300000,100000,50000,30000,10000,5000,3000]  #,1000]

for sampleNum in sampleNumList:
    kdf = pd.DataFrame(index=range(10, 85, 5))
    for i in simulateList:
        # filePath = '../dual_nor/k_gaus/k_gaus_'+str(sampleNum)+'_'+str(i)+'.csv'
        filePath = '../dual_nor/k_gaus/k_gaus_150_' + str(i) + '.csv'
        # fr = open(filePath,'rb')
        # dta = pickle.load(fr)
        dta = pd.read_csv(filePath)
        csizeList = dta['cellsize']
        columnName = 'simualte'+str(i)
        kdf[columnName] = dta['k'].tolist()
        m = 1
        # nuggetList = dta[1]
        # k_List = dta[3]
    #     plt.scatter(csizeList,kList,label='simulate ' + str(i))
    #     plt.plot(csizeList,kList)cellSize
    cellSize = range(10,85,5)
    kdf = kdf.iloc[:15,:]
    dta_low = kdf.apply(lambda x:x.min(),axis = 1)
    dta_up = kdf.apply(lambda  x:x.max(),axis = 1)
    dta_mean = kdf.apply(lambda x:x.mean(),axis = 1)
    fsize = 18
    plt.plot(cellSize,dta_low,c = 'gray',alpha = 0.2)
    plt.plot(cellSize,dta_up,c= 'gray',alpha = 0.2)
    plt.scatter(cellSize,dta_mean,c = 'r',label = 'para = '+str(sampleNum))
    plt.plot(cellSize,dta_mean,c = 'r')
    plt.fill_between(cellSize,dta_low,dta_up,facecolor = 'gray',alpha = 0.2)
    plt.xlabel('cellsize',fontsize = fsize)
    plt.ylabel(r'$c_0/(c_0+c_1)$',fontsize = fsize)
    plt.legend()
    plt.show()
m = 1

# #boxplot
# # labels = ['300','400','500','600','700','800','900','1000']
# # plt.boxplot(kdf,labels = labels)
# # fsize = 18
# # plt.xlabel('cellsize',fontsize = fsize)
# # plt.ylabel(r'$c_0/(c_0+c_1)$',fontsize = fsize)
# # plt.show()
# m = 1

def draw_confidence(func,xdata,ydata,nstd):
    #curve fit [with only y-error]
    popt,pcov=curve_fit(func,xdata=xdata,ydata=ydata)
    perr = np.sqrt(np.diag(pcov))

    # print fit parameters and 1-sigma estimates
    print('fit parameter '+str(nstd)+' - sigma error')
    print('———————————–')
    for i in range(len(popt)):
        print(str(popt[i]) +' +- '+str(perr[i]))
    ydata_fit=np.array(func(xdata, *popt))
    R2=r2_score(y_true=ydata,y_pred=ydata_fit)
    print (R2)
    # prepare confidence level curves
    nstd = nstd  # to draw nstd-sigma intervals
    popt_up = popt + nstd * perr
    popt_dw = popt - nstd * perr

    xstep=np.linspace(min(xdata),max(xdata),1000)
    fit = np.array(func(xstep, *popt))
    fit_up = func(xstep, *popt_up)
    fit_dw = func(xstep, *popt_dw)

    # ax=plt.gca()
    # plt.plot(xdata,fit, 'r', lw = 2, label ='best fit curve')
    # plt.plot(xdata, ydata, 'k–', lw = 2, label ='True curve')
    # ax.fill_between(xdata,fit_up,fit_dw,alpha=.25,label=str(nstd)+'-sigma interval')

    return xstep,fit,fit_up,fit_dw