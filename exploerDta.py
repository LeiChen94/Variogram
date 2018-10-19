import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew
from scipy.optimize import curve_fit
from math import exp,sqrt,e
import powerlaw
import random
from sklearn.metrics import r2_score
import pickle

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
def model_pure_nugget_effect(x,c0):
    return [c0 for item in x]


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
sigma = 80
cellSize = range(25,30,5)
simulateList = range(1,2)
nuggetList = []
rangeList = []

for simu in simulateList:
    k_gaus_list = []
    k_poly_list = []
    k_spher_list = []
    for cellsize in cellSize:
        # odir = '../random/random_' + str(simu)
        odir = '../dual_nor/nor' + str(sigma) + '_' + str(simu)
        # odir = '../block_dta/block' + '_' + str(simu)
        # df = pd.read_csv(odir + '/block_varidf' + str(cellsize) + '.csv')
        df = pd.read_csv(odir+'/nor_varidf'+str(cellsize)+'.csv')
        df = df[df['vari'] > 0]
        #df['vari'] = df['vari']/pow(cellsize/1000,4)   #/K^4
        tor = cellsize*0.4
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
        gapnum = int((maxLag - cellsize)/(cellsize*0.8))
        for lag in range(cellsize,maxLag,gap):
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

            # mean = round(np.mean(logdta),2)
            # median = round(np.median(logdta),2)
            # mean_list.append(mean)
            # median_list.append(median)

            #数据拟合判断
            # fit = powerlaw.Fit(rawdta,discrete=True)
            # R, p = fit.distribution_compare('lognormal', 'power_law',normalized_ratio = True)
            # print(R, p)
            # bins,pdf = fit.pdf()
            # xlist = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
            # plt.scatter(xlist,pdf)
            # fig = fit.plot_pdf(linewidth=3, label='Empirical data',original_data = True)
            # fit.power_law.plot_pdf(ax=fig, color='r', linestyle='--', label='Powerlaw fit')
            # fit.lognormal.plot_pdf(ax=fig, color='g', linestyle='--', label='Lognormal fit')
            # plt.title('Cellsize='+str(cellsize)+',Lag='+str(lag)+',R='+str(round(R,4)))
            # plt.legend()
            # plt.show()
            # print(fit.xmin)  # 幂律指数
            # print(fit.xmax)  # 噪声可能取的最小值

            #plt.hist(dta['vari'],50,label = 'rawdta');
            # plt.hist(logdta,50,label = 'lag='+ str(lag)+',skew='+str(skew_log)+';'+str(mean)+';'+str(median))
            # plt.legend()
            # plt.show()
        #     if (count % 4 == 0):
        #         boxlabel_list.append(lag/1000)
        #         boxdta_list.append(logdta)
        #         # plt.boxplot((rawdta,logdta),labels = ('rawdta','logdta'))
        # red_square = dict(markerfacecolor='r', marker='s')
        # plt.boxplot(boxdta_list,labels = boxlabel_list)
        # fsize = 14
        # plt.ylim(0,50)
        # plt.xlabel('lag(km)',fontsize = fsize)
        # plt.ylabel('squared difference', fontsize=fsize)
        # plt.show()
        #均值和中位数对比
        # plt.scatter(laglist,mean_list,label = 'mean',c = 'blue',alpha = 0.7);
        # plt.scatter(laglist,median_list,label = 'median',c = 'red',alpha = 0.7)
        # fsize = 14
        # plt.xlabel('lag(km)',fontsize = fsize)
        # plt.ylabel('semivariance', fontsize=fsize)
        # plt.legend()
        # plt.show()

        #绘制偏度分布图
        # fig = plt.figure()
        # left, bottom, width, height = 0.12, 0.12, 0.8, 0.8
        # ax1 = fig.add_axes([left, bottom, width, height])
        # fsize = 14
        # ax1.scatter(laglist,skewraw_list,label = 'rawDta',c = 'blue',alpha = 0.7)
        # ax1.scatter(laglist,skewlog_list,label = 'logDta',c = 'red',alpha = 0.7)
        # ax1.set_xlabel('Lag(km)',fontsize = fsize)
        # ax1.set_ylabel('Skewness',fontsize = fsize)
        # ax1.legend(loc = [0.04,0.22])
        # fsize = 12
        # left, bottom, width, height = 0.5, 0.28, 0.36, 0.36
        # ax2 = fig.add_axes([left, bottom, width, height])
        # ax2.scatter(laglist,skewlog_list,label = 'logDta',c = 'red',alpha = 0.7,s = 10)
        # ax2.set_xlabel('Lag(km)',fontsize = fsize)
        # ax2.set_ylabel('Skewness',fontsize = fsize)
        # ax2.set_title('Skewness of logDta',fontsize = fsize)
        # plt.show()


        #高斯模型
       
        p_fit1,pcov1 = curve_fit(model_gaus,laglist,semilog_list)
        c0,c1,a = p_fit1.tolist()
        y1 = model_gaus(laglist,*p_fit1)
        k1 = round(c0/(c0+c1),2)
        k_gaus_list.append(k1)
        print(len(laglist),len(semilog_list))

        # 多项式拟合
        p_fit2, pcov2 = curve_fit(model_poly, laglist, semilog_list)
        a, b, c = p_fit2.tolist()
        y2 = model_poly(laglist, *p_fit2)
        c0 = c
        c1 = -b**2/4/a
        k2 =  round(c0 / (c0+c1),2)
        k_poly_list.append(k2)

        # 球面模型
        p_fit3, pcov3 = curve_fit(model_spher, laglist, semilog_list)
        c0, c1, a = p_fit3.tolist()
        y3 = model_spher(laglist, *p_fit3)
        k3 = round(c0 / (c0+c1),2)
        k_spher_list.append(k3)
        '''
        #纯块金值模型
        p_fit4, pcov4 = curve_fit(model_pure_nugget_effect, laglist, semilog_list)
        y4= model_pure_nugget_effect(laglist, *p_fit4)
        '''
        ### draw the semivariogram
        dta_list = semilog_list
        R2_gaus = round(r2_score(y1, dta_list), 3)
        R2_poly = round(r2_score(y2, dta_list), 3)
        R2_spher = round(r2_score(y3, dta_list), 3)
        '''
        R2_pure = round(r2_score(y4, dta_list), 3)
        '''
        dta_list = semilog_list
        plt.scatter(laglist,dta_list,alpha = 0.8,label = str(cellsize)) #+',$R^2$='+str(R2_gaus)) #+',nugget='+ str(nugget)+',k ='+str(k)+',x='+str(x)) #+',r2='+str(R2)+',cou='+str(len(laglist)))
        plt.plot(laglist, y1,label = str(R2_gaus)+','+str(k1))
        plt.plot(laglist, y2,label = str(R2_poly)+','+str(k2))
        # plt.plot(laglist,y3,label = str(R2_spher)+','+str(k3))
        '''
        plt.plot(laglist, y4, label=str(R2_pure))
        '''
    #绘制k与cellsize关系图
    # plt.scatter(cellSize, k_gaus_list)
    # plt.plot(cellSize,k_gaus_list, label='gaus')
    # plt.plot(cellSize, k_poly_list, label='ploy')
    # plt.plot(cellSize, k_spher_list, label='spher')
    # res = np.vstack((cellSize,np.array(nuggetList),np.array(kList)))

##3 图标名称、坐标标记等
#3.1 半方差图图标等
fsize = 16
plt.tick_params(labelsize = 12)
plt.xlabel("Lag distance(km)",fontsize = fsize )
if (dta_list == semilog_list) :
    title = "(log_nor)"
    ylabel = '(log)'
else:
    title = "(raw_nor)"
    ylabel = ""
plt.ylabel(r"Semivariance"+ylabel, fontsize=fsize)
plt.title("The semivariogram"+title,fontsize = fsize)
plt.legend()
plt.show()

#3.2 k与cellsize关系图
# fsize = 18
# plt.xlabel('cellsize',fontsize = fsize)   #plt.ylabel('range '+ r'$\alpha$',fontsize = fsize)
# plt.ylabel(r'$c_0/(c_0+c_1)$',fontsize = fsize)
# plt.title('random',fontsize = fsize)
# plt.legend()
# plt.show()

m = 1