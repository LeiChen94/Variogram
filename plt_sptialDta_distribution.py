import pandas as pd
from pyproj import Proj, transform
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import numpy as np
import math


# artificialdta
gridLeft = 0
gridBottom = 0

def proj_trans(lon, lat):
    p1 = Proj(init='epsg:4326')  # 地理坐标系WGS1984
    p2 = Proj(init='epsg:32650')  # 投影坐标WGS_1984_UTM_Zone_50N
    lon_val = lon.values
    lat_val = lat.values
    x1, y1 = p1(lon_val, lat_val)
    x2, y2 = transform(p1, p2, x1, y1, radians=True)
    return x2, y2

def grid_confirm(cellSize, gridNum, x, y):
    # 以网格左下角为原点，划分格网序号;
    xidArr = np.ceil((x - gridLeft) / cellSize);
    yidArr = np.ceil((y - gridBottom) / cellSize);
    outIndex = np.array([], dtype=np.bool)
    # x,y(1,gridNum = 30)
    for i in range(0, len(xidArr)):
        if (xidArr[i] < 1) | (xidArr[i] > gridNum):
            outIndex = np.append(outIndex, True)
        else:
            outIndex = np.append(outIndex, False)
    for j in range(0, len(yidArr)):
        if (yidArr[j] < 1) | (yidArr[j] > gridNum):
            outIndex[j] = True

    grid_id = (yidArr - 1) * gridNum + xidArr - 1
    # 标记出界点
    grid_id[outIndex] = -1
    # grid_id(0,gridNum*gridNum-1 = 899)
    totalNum = gridNum * gridNum - 1
    grid_id[(grid_id < 0) | (grid_id > totalNum)] = -1
    grid_id = grid_id.astype(np.int)
    return grid_id

def grid_inf(cellSize, gridNum):
    gridid_arr = np.arange(gridNum * gridNum)
    xid_arr = gridid_arr % gridNum
    yid_arr = gridid_arr // gridNum
    return gridid_arr, xid_arr, yid_arr  # 格网序列号（自然编码），格网行列号（地理编码）

# make dta
muXList = [500 , 18000 , 24000]
muYList = [500 , 10000 , 26000]
sigma = 300
sampleNum = 10000

for simu in range(1,2):
    #指定不同的聚合尺度
    x = np.round(np.random.normal(muXList[0], sigma, sampleNum), 0)
    y = np.round(np.random.normal(muYList[0], sigma, sampleNum), 0)
    dta = pd.DataFrame(columns = ['x','y','checkin_num'])
    dta['x'] = x
    dta['y'] = y
    dta['checkin_num'] = 1
    # dta.to_csv('../single_nor_test/normal'+str(sigma)+'_'+str(simu)+'/xy_dta.csv',index = None)

    norm = colors.Normalize(0,0)
    fig,axs = plt.subplots(2,4)
    axs = axs.flatten()

    for i,csize in enumerate(range(40,5,-5)):
        gridNum = int(1000/ csize)
        gridpos = grid_inf(csize,gridNum)
        df = pd.DataFrame()
        df['checkin_num'] = dta['checkin_num']
        #gridinf格网行列号信息
        gridinf = pd.DataFrame(index = gridpos[0],columns = ['gridx','gridy'])
        gridinf['gridx'] = gridpos[1]
        gridinf['gridy'] = gridpos[2]

        df['gridId'] = grid_confirm(csize,gridNum,x,y)
        df['num'] = 1
        checkin_cluster = df['checkin_num'].groupby(df['gridId']).sum()
        poi_cluster = df['num'].groupby(df['gridId']).sum()

        #sdd_df空间分布数据
        sdd_df = pd.concat([checkin_cluster,poi_cluster,gridinf],axis = 1).fillna(0)
        sdd_df.rename(columns = {'checkin_num':'checkin_sum','num':'poi_sum'},inplace = True)
        if (sdd_df.index[0] == -1):
            sdd_df.drop(sdd_df.index[0], inplace=True)  # 去掉网格id为-1的统计记录

        #绘制空间分布热力图
        checkin_arr = np.array(sdd_df['checkin_sum'])
        if i == 0:
            vmax= max(checkin_arr)
            vmin = 0
            norm = colors.Normalize(vmin,vmax)
        checkin_matrix = checkin_arr.reshape(gridNum,gridNum)
        checkin_sort = sorted(sdd_df.checkin_sum , reverse = True)
        ax = axs[i]
        im = ax.imshow(checkin_matrix , plt.cm.Reds ,norm = norm)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('cellsize = '+str(csize))

plt.subplots_adjust(bottom = .05,top = 0.99,hspace =.002)
# fig.colorbar(im, ax=axs, fraction=.1)
# pos_cbar = fig.add_axes([0.95,0.2,0.075,0.6])
# cb=plt.colorbar(im,ax = axs[1])
fig.tight_layout()
# plt.subplots_adjust(top = 1,left = 0.5)
plt.show()
m = 1