
import numpy as np
import pandas as pd
from datetime import timedelta
import time
from epiweeks import Week
import pdb
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error


# ew202201 remove
death_remove = []
hosp_remove = [] 
increase_death_interval_high = []
increase_death_interval_low = []

regions_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
            'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
            'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE',
            'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
            'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
            'VA', 'WA', 'WV', 'WI', 'WY', 'US']

def check_region_data(datapath,region,target_name,ew):
    df = pd.read_csv(datapath, header=0)
    df = df[(df['region']==region)]
    if df.size == 0:
        print('region ', region, ' is missing!')
        return False
    if target_name == 'hosp':  # next starts at 1
        y = df.loc[:,'cdc_hospitalized'].to_numpy()
    elif target_name == 'death':
        y = df.loc[:,'death_jhu_incidence'].to_numpy()
    elif target_name == 'flu_hospitalizations':
        y = df.loc[:,'flu_hospitalizations'].to_numpy()
    if y.sum()==0.:
        print('region ', region, ' is all zeros part!')
        return False
    return True


# get cumulative
def get_cumsum_region(datafile,region,target_name,ew):
    df = pd.read_csv(datafile, header=0)
    df = df[(df['region']==region)]
    if target_name=='death':
        cum = df.loc[:,'cdc_flu_hosp'].sum()
    elif target_name=='hosp':
        cum = None
        # raise Exception('not implemented')
        # cum = df.loc[:,'hospitalizedIncrease'].sum()
    else:
        print('error', region,target_name)
        time.sleep(2)
    return cum

def get_predictions_from_pkl(next,res_path,region,week_current=None):
    """ reads from pkl, returns predictions for a region as a list"""
    if week_current is None:
        week_current = int(str(Week.thisweek(system="CDC") - 1)[-2:])
        week_current = str(week_current)
    if len(week_current)==1:
        week_current = '0'+week_current
    path=res_path+'flu_deploy_week_' + week_current +'_feats_' + str(next) + '_predictions.pkl' # normal

    if not os.path.exists(path):
        print(path)
        return None
    predictions = []
    
    with open(path, 'rb') as f:
        data_pickle = pickle.load(f)

    idx = regions_list.index(region)
    predictions = data_pickle[:,idx]
    return predictions

# set the start week for the visualization
ew_vis_start='202152'

def visualize_region(target_name,region,predictions,datafile,opt,_min=None,_max=None,ew=19,suffix='',daily=False,fig_path='./figures/',show_rmse=False):
    """
        @param target_name: 'death' or 'hosp'
        @param predictions: [next1, next2, ...]
        @param datafile: e.g. "./data/merged-data3.csv"
        @param opt: 'inc' or 'cum'   NOTE: it converts inc to cumulatives
        @param _min: percentile for 5% - e.g. [min1,min2] for next2 pred  
        @param _max: percentile for 95% - e.g. [max1,max2] for next2 pred 
        @param daily: boolean

            
        ##### IMPORTANT !!!!!!!!!!!!!!!!!!!!
        @param ew: the week before the start of prediction, like ew =19, so the first week of prediction is 20, but the ground truth maybe more than 20
        NOTE:
        @param ew: making it epiweek obj
    """
    
    overlap= False
    df = pd.read_csv(datafile, header=0)
    df = df[df['region']==region]
    def convert(x):
        return Week.fromstring(str(x), system="CDC")
    df['epiweek'] = df.loc[:,'epiweek'].apply(convert)
    ## The end week of the groud turth
    end_week = df['epiweek'].max()
    ## The length of predictions
    len_pred = len(predictions)

    ## determine whether rmse is needed
    ## then establish the variables in need
    ## overlap_pred is the instance prediction from the model overlapped with ground truth in the same time interval
    ## overlap_inc is the instance ground truth in the same time inveral with overlap_pred
    # if end_week - ew !=0:
    if end_week != ew:
        overlap= True
        #overlap_pred = predictions[:(end_week-ew)]
        df_overlap = df[(df.loc[:,'epiweek'] <= (end_week)) & (df.loc[:,'epiweek'] > ew)].copy() 
        if target_name =='hosp':
            # overlap_inc = df_overlap.loc[:,'hospitalizedIncrease'].to_numpy()
            overlap_inc = df_overlap.loc[:,'cdc_hospitalized'].to_numpy()
        elif target_name == 'death' or target_name=='cum_death':
            overlap_inc = df_overlap.loc[:,'death_jhu_incidence'].to_numpy()

    cum_25 = df[(df.loc[:,'epiweek'] <= Week.fromstring('202040'))].loc[:,'death_jhu_incidence'].sum()
    df = df[(df.loc[:,'epiweek'] <= ew) & (df.loc[:,'epiweek'] >= Week.fromstring(ew_vis_start))]   ## data only from 10 to ew
    epiweeks = list(df.loc[:,'epiweek'].to_numpy())
    # days=list(df.loc[:,'date'].to_numpy())
    days=list(df.loc[:,'epiweek'].to_numpy())
    days=list(range(1,len(days)+1))
    if target_name == 'hosp':  # next starts at 1
        # inc = df.loc[:,'hospitalizedIncrease'].to_numpy()
        inc = df.loc[:,'cdc_hospitalized'].to_numpy()
        title_txt = 'Hospitalizations'
    elif target_name == 'death' or target_name=='cum_death':
        # inc = df.loc[:,'deathIncrease'].to_numpy()
        inc = df.loc[:,'death_jhu_incidence'].to_numpy()
        title_txt = 'Mortality'
    elif target_name == 'flu hosp':
        inc = df.loc[:,'cdc_flu_hosp'].to_numpy()
        title_txt = 'Flu Hosp'

    if opt=='inc':
        y=inc
        if overlap ==True:
            y_overlap = overlap_inc
        label='Incidence'
    elif opt=='cum':
        """
            Hack to fix cumulative: add what is was before ew202025
        """
        cum = [cum_25]
        
        for i in range(len(inc)-1):
            cum.append(inc[i+1]+cum[-1])
        
        y=cum
        if overlap==True:
            overlap_inc[0] += y[-1]
            y_overlap  = np.cumsum(overlap_inc, dtype=np.float64)
        label='Cumulative'
        _min[0] = _min[0]+cum[-1]
        _min[1:] = [_min[i]+y[-1]+sum(predictions[:i]) for i in range(1,len(_min))]
        _max[0] = _max[0]+y[-1]
        _max[1:] = [_max[i]+y[-1]+sum(predictions[:i]) for i in range(1,len(_max))]
        predictions[0] = predictions[0] + cum[-1]
        predictions[1:] = [sum(predictions[:i+1]) for i in range(1,len_pred)]
        

    ## weeks of predictions: like from 10 to 18 is the range of ground truth, 19 to 21 is the range of predictions

    ## overlap_pred is the instance prediction from the model overlapped with ground truth in the same time interval
    if overlap==True:
        overlap_pred = predictions[:(end_week-ew)]
        overlap_pred_weeks = list(range(ew+1, end_week+1))
    pred_weeks = [epiweeks[-1]+w for w in range(1,1+len_pred)]
    pred_days = [days[-1]+w for w in range(1,1+len_pred)]

    ## Calculate the RMSE
    if overlap ==True:  
        RMSE = []
        for index in range(1,len(overlap_pred)+1):
            RMSE.append(np.sqrt(mean_squared_error([overlap_pred[index-1]], [y_overlap[index-1]])))
        y_overlap = y_overlap.tolist()
        red_x= [epiweeks[-1]] + overlap_pred_weeks
        red_y = [y[-1]] + y_overlap

    if daily:
        plt.plot(days,y,'b',label='Ground truth data from JHU',linestyle='-')
        plt.plot([days[-1]]+pred_days,[y[-1]]+predictions,linestyle='-', marker='o', markersize=1, linewidth=1, color='r',label='Associated predictions')
    else:
        epiweeks = [str(e)[-2:] for e in epiweeks]
        pred_weeks = [str(e)[-2:] for e in pred_weeks]
        ## Plot ground truth data first
        plt.plot(epiweeks,y,'b',label='Ground truth data from JHU',linestyle='-')
        ## The predictions starts from the last week of ground truth
        plt.plot([epiweeks[-1]]+pred_weeks,[y[-1]]+predictions,linestyle='--', marker='o', color='r',label='Associated predictions')
    ## Plot the overlap data
    
    if overlap==True:
        plt.plot(red_x,red_y,linestyle='-', color='m', marker= "^",label=' Associated Ground Truth to Compare')

        if show_rmse:
            ##Plot RMSE 
            y_max = np.max(predictions)
            tick = y_max/10
            for  index,value in enumerate(RMSE): 
                plt.text( 0.85* end_week, 0.5*y_max - tick*index, 'rmse'+ str(index+1)+ ': '+str(round(value,2)), size=12)

    if daily:
        plt.xlabel('day')
    else:
        plt.xlabel('epidemic week')
    plt.ylabel(label+' '+target_name+' counts')

    #print(y)
    if _min is not None and _max is not None:  

        _min.insert(0,y[-1])
        _max.insert(0,y[-1])
        if daily:
            plt.fill_between([days[-1]]+pred_days, _min, _max, alpha = 0.25, label='95% Confidence Interval')
        else:
            plt.fill_between([epiweeks[-1]]+pred_weeks, _min, _max, alpha = 0.25, label='95% Confidence Interval')
        # plt.fill_between(pred_weeks, _min, _max, alpha = 0.25, label='95% Confidence Interval')
    # plt.xscale([0,])
    plt.legend(loc='upper left')
    plt.gca().set_ylim(bottom=0)
    if opt=='inc':
        if region=='US':
            plt.title('US Incidence '+title_txt)  
        else:
            plt.title(region+' '+label)  
        plt.savefig(fig_path+region+'_'+target_name+'_'+'ew'+str(ew)+suffix+'.png')
        print('inc predictions >>>>>',predictions)
    else:
        if region=='US':
            plt.title('US Cumulative '+title_txt)  
        else:
            plt.title(region+' '+label)   
        plt.savefig(fig_path+region+'_cum'+target_name+'_'+'ew'+str(ew)+suffix+'.png')
        print('cum predictions >>>>>',predictions)
    
    plt.close()

def parse(region,ew,target_name,suffix,daily,write_submission,visualize,data_ew=None,res_path='./results-flu/',sub_path='./submissions-flu/'):
    """
        @param write_submission: bool
        @param visualize: bool
        @param data_ew: int, this is needed to use the most recent data file
                    if None, it takes the values of ew
    """
    if data_ew is None:
        data_ew=ew  
    if daily:
        k_ahead=30 # changed from 28 to 30 on nov14 as requested by CDC (only needed for training)
        datafile='./data/daily/covidcast/delphi_'+str(data_ew)+'.csv'
    else:
        k_ahead=4
        datafile='./data/weekly/weeklydata/'+str(data_ew)+'.csv'
    # pdb.set_trace()
    if not check_region_data(datafile,region,target_name,ew): # checks if region is there in our dataset
        return 0    

    # prev_cum = get_cumsum_region(datafile,region,target_name,ew)
    print(region)
    point_preds = []
    lower_bounds_preds = []
    upper_bounds_preds = []
    for nextk in range(1,k_ahead+1):
        predictions = get_predictions_from_pkl(nextk,res_path,region,str(ew.week))
        
        if predictions is None:
            continue
        quantile_cuts = [0.01, 0.025] + list(np.arange(0.05, 0.95+0.05, 0.05,dtype=float)) + [0.975, 0.99]
        new_predictions = []
        for pred in predictions:
            if pred < 0:
                pred = 0
            new_predictions.append(pred)
        predictions = new_predictions

        quantiles = np.quantile(predictions, quantile_cuts)
        df = pd.read_csv(datafile, header=0)
        df = df[(df['region']==region)]
        # add to list
        lower_bounds_preds.append(quantiles[1])
        upper_bounds_preds.append(quantiles[-2])
        point_preds.append(np.mean(predictions))

        suffix_=suffix
        if write_submission:
            # >>>>>> 
            team='GT'
            model='FluFNP-raw'
            date=ew.enddate() + timedelta(days=2)
            datex=date
            date=date.strftime("%Y-%m-%d") 
            
            sub_file=sub_path+date+'-'+team+'-'+model+'.csv'
            if not os.path.exists(sub_file):
                f = open(sub_file,'w+')
                f.write('forecast_date,target,target_end_date,location,type,quantile,value\n')
                f.close()

            f = open(sub_file,'a+')
            # first target is inmmediate saturday
            target_end_date = datex + timedelta(days=5) + timedelta(days=7)*(nextk-1)  # good for weekly pred
            location_fips=df.loc[:,'fips'].iloc[-1]
            location_fips=str(location_fips)
            if len(location_fips)==1:
                location_fips = '0'+location_fips 

            if region in hosp_remove:
                suffix_=suffix+'_rm'
                continue
            f.write(str(datex)+','+str(nextk)+' wk ahead inc '+target_name+','+str(target_end_date)+','+location_fips+','+'point'+','+'NA'+','+"{:.2f}".format(np.mean(predictions))+'\n')
            for q_c, q_v in zip(quantile_cuts, quantiles):
                f.write(str(datex)+','+str(nextk)+' wk ahead inc '+target_name+','+str(target_end_date)+','+location_fips+','+'quantile'+','+"{:.4f}".format(q_c)+','+"{:.4f}".format(q_v)+'\n')

        
    suffix_=suffix
    if visualize:
        figpath=f'./figures-flu/ew{ew}/raw/'
        if not os.path.exists(figpath):
            os.makedirs(figpath)
        print('==='+target_name+' '+region+'===')
        # pdb.set_trace()
        visualize_region(target_name,region,point_preds,datafile,'inc',lower_bounds_preds.copy(),upper_bounds_preds.copy(),ew,suffix_,daily,figpath)

    
if __name__ == "__main__":
    
    PLOT=False
    # WRITE_SUBMISSION_FILE=False
    WRITE_SUBMISSION_FILE=True
    ew= Week.thisweek(system="CDC")
    print(ew)
    import pdb

    target_name='flu_hospitalizations'
    daily=False
    suffix='M1_10_vEW'+str(ew) # for the predictions files
    print(suffix)

    for region in regions_list:
        parse(region,ew,target_name,suffix,daily,WRITE_SUBMISSION_FILE,PLOT)

    """ to generate past ones """

    # for ew in ['01','02','03','04','05','06','07','08','09']:
    # for ew in ['10','11','12']:
    #     ew=Week.fromstring('2022'+str(ew))
    #     for region in temp_regions:
    #         parse(region,ew,target_name,suffix,daily,WRITE_SUBMISSION_FILE,PLOT)

    quit()
