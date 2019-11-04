# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 23:49:21 2019

@author: falli
"""
import time
import requests
import os,shutil
import numpy as np
import pandas as pd
from math import isnan
from astropy.io import fits
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.time import TimeDelta,Time
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from photutils.centroids import fit_2dgaussian
from astropy.visualization import HistEqStretch
from skimage.feature import register_translation,match_template
from PyAstronomy.pyasl.asl.astroTimeLegacy import helio_jd
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture, CircularAnnulus, aperture_photometry


def delfile(file_name):
    if os.path.exists(file_name):
        if os.path.isfile(file_name):
            os.remove(file_name)
        if os.path.isdir(file_name):
            shutil.rmtree(file_name,True)
            
def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
def imgshow(data):
    fig = plt.figure(figsize=(20,20));fig.add_subplot(111)
    norm = ImageNormalize(stretch=HistEqStretch(data))
    plt.imshow(data, cmap='Greys', origin='lower', norm=norm)
    plt.show()
    
def imageshow(data,positions,aper=[8,12,20],rim_size=50):
    min_x=int(np.min(positions.T[0]))-rim_size;max_x=int(np.max(positions.T[0]))+rim_size
    min_y=int(np.min(positions.T[1]))-rim_size;max_y=int(np.max(positions.T[1]))+rim_size
    data=data[min_y:max_y,min_x:max_x]
    positions=positions-np.array([min_x,min_y])
    apertures = CircularAperture(positions, r=aper[0])
    annulus_apertures = CircularAnnulus(positions, r_in=aper[1], r_out=aper[2])          
    fig = plt.figure(figsize=(20,20));fig.add_subplot(111)
    apertures.plot(color='blue',lw=2,alpha=1)
    annulus_apertures.plot(color='red',lw=2,alpha=0.5)
    norm = ImageNormalize(stretch=HistEqStretch(data))
    plt.imshow(data, cmap='Greys', origin='lower', norm=norm)
    plt.show()

def middle_obs_time(utc,exptime,fmt):
    if fmt=='isot':
        return (Time(utc, format='isot', scale='utc')+TimeDelta(int(exptime)/2, format='sec'))
    if fmt=='jd':
        return (Time(utc, format='jd', scale='utc')+TimeDelta(int(exptime)/2, format='sec'))

def fine_center(data,positions,box_size=10):
    positions=positions.astype(int)
    fine_positions=[];fine_fwmh=[];fine_snr=[]
    for i in range(len(positions)):
        data_find=data[(positions[i][1]-box_size):(positions[i][1]+box_size),(positions[i][0]-box_size):(positions[i][0]+box_size)]
        try:
            print('Star_%u is fitting...'%(i+1))
            gaussian_fit=fit_2dgaussian(data_find)
        except:
            print('ERROR:fine_center')
            return [-1],[-1],[-1]
        fine_positions.append(np.array([gaussian_fit.x_mean.value,gaussian_fit.y_mean.value]))
        fine_fwmh.append(np.max([gaussian_fit.x_stddev.value,gaussian_fit.y_stddev.value])*2)
        fine_snr.append(gaussian_fit.amplitude/gaussian_fit.constant)
    return np.array(fine_positions)+positions-box_size,np.array(fine_fwmh),np.array(fine_snr)

def split_radec(loc,sep=' '):
    if ':' in loc:
        loc=loc.replace(':','')
    if loc[0] in ['+','-']:
        return loc[0:3]+sep+loc[3:5]+sep+loc[5:]
    else:
        return loc[0:2]+sep+loc[2:4]+sep+loc[4:]
    
def check_images(data_out,bin=1,fits_all=None,pho_all=1):
    images_table=pd.DataFrame(columns=('File_NAME','OBJ_NAME','RA','DEC','EXPTIME','FILTER','IMAGETYP','DATE-OBS','xbin','ybin'))
    if fits_all==None:
        fits_all=[fits_image for fits_image in os.listdir(os.getcwd()) if ('.fit' in fits_image)]
    for i in range(len(fits_all)):
        with fits.open(fits_all[i]) as hdu:
            print(fits_all[i],hdu[0].header['OBJECT'],hdu[0].header['XBINNING'])
            if hdu[0].header['IMAGETYP']=='Bias Frame' and hdu[0].header['XBINNING']==bin:
                images_table.loc[i]=[fits_all[i],hdu[0].header['OBJECT'],'','',hdu[0].header['EXPTIME'],'','bias',\
                                 hdu[0].header['DATE-OBS'],hdu[0].header['XBINNING'],hdu[0].header['YBINNING']]
            elif hdu[0].header['IMAGETYP']=='Flat Field' and hdu[0].header['XBINNING']==bin:
                images_table.loc[i]=[fits_all[i],hdu[0].header['OBJECT'],'','',hdu[0].header['EXPTIME'],hdu[0].header['FILTER'],'flat',\
                                 hdu[0].header['DATE-OBS'],hdu[0].header['XBINNING'],hdu[0].header['YBINNING']] 
            elif hdu[0].header['IMAGETYP']=='Light Frame' and hdu[0].header['XBINNING']==bin:
                images_table.loc[i]=[fits_all[i],hdu[0].header['OBJECT'],'','',hdu[0].header['EXPTIME'],hdu[0].header['FILTER'],'light',\
                                 hdu[0].header['DATE-OBS'],hdu[0].header['XBINNING'],hdu[0].header['YBINNING']] 
    if (set(images_table['FILTER'][images_table['IMAGETYP']=='light']) & set(images_table['FILTER'][images_table['IMAGETYP']=='flat']))==\
        set(images_table['FILTER'][images_table['IMAGETYP']=='light']):
        images_table.to_csv(os.path.join(data_out,'images_infs.csv'),index=False,header=True)
        if not pho_all:
            0/0
        return images_table
    elif input('the flat frames is not complete, type "y" to conttinue.\n').lower()=='y':
        check_images()
    else:
        0/0
    return             
             
def combine_images(images_table,trim_size,data_out):
    master_table=pd.DataFrame(columns=('File_NAME','FILTER','IMAGETYP','DATA'))
    #combine bias frames
    bias_data=[];
    for bias_frame in images_table['File_NAME'][images_table['IMAGETYP']=='bias']:
        with fits.open(bias_frame) as hdu:
            bias_data.append(hdu[0].data[trim_size[0][0]:trim_size[0][1],trim_size[1][0]:trim_size[1][1]].astype(float))
    master_bias=np.median(np.array(bias_data),axis=0);
    master_bias_file=os.path.join(data_out,'MasterBias.fits')
    delfile(master_bias_file);fits.PrimaryHDU(master_bias).writeto(master_bias_file);
    master_table.loc['bias']=['MasterBias.fits','','bias',master_bias];
    #combine flat frames according to filter
    list_filter=list(set(images_table['FILTER'][images_table['IMAGETYP']=='light']));
    for i in range(len(list_filter)):
        flat_data=[];
        for flat_frame in images_table['File_NAME'][(images_table['IMAGETYP']=='flat') & (images_table['FILTER']==list_filter[i])]:
            with fits.open(flat_frame) as hdu:
                flat_trim=hdu[0].data[trim_size[0][0]:trim_size[0][1],trim_size[1][0]:trim_size[1][1]].astype(float)-master_bias
                flat_trim_scale=flat_trim/np.median(flat_trim)
                flat_data.append(flat_trim_scale)
        master_flat=np.median(np.array(flat_data),axis=0)/np.mean(np.array(flat_data));
        master_flat_file=os.path.join(data_out,'MasterFlat_'+list_filter[i]+'.fits')
        delfile(master_flat_file);fits.PrimaryHDU(master_flat).writeto(master_flat_file);
        master_table.loc[list_filter[i]]=['MasterFlat_'+list_filter[i]+'.fits',list_filter[i],'flat',master_flat];
    return master_table,list_filter

def preprocess(obj_image,images_table,master_table,trim_size,data_out):
    new_obj_image=os.path.join(data_out,'FBT_'+obj_image);delfile(new_obj_image);
    shutil.copyfile(obj_image,new_obj_image)
    with fits.open(new_obj_image,mode='update') as hdu:
        hdu[0].data=(hdu[0].data[trim_size[0][0]:trim_size[0][1],trim_size[1][0]:trim_size[1][1]].astype(float)-master_table.loc['bias']['DATA'])/master_table.loc[images_table['FILTER'][images_table['File_NAME']==obj_image]]['DATA'].values[0]
        hdu[0].header.set('Trim', 'T');hdu[0].header.set('Bias',master_table.loc['bias']['File_NAME']);hdu[0].header.set('Flat',master_table.loc[images_table['FILTER'][images_table['File_NAME']==obj_image]]['File_NAME'].values[0]);
        hdu.flush();
    return new_obj_image,hdu[0].header,hdu[0].data

def ref2image(data,positions,aper,fig_name):
    apertures = CircularAperture(positions, r=aper[0])
    annulus_apertures = CircularAnnulus(positions, r_in=aper[1], r_out=aper[2])          
    fig = plt.figure(figsize=(20,20));fig.add_subplot(111)
    apertures.plot(color='blue',lw=2,alpha=1)
    annulus_apertures.plot(color='red',lw=2,alpha=0.5)
    norm = ImageNormalize(stretch=HistEqStretch(data))
    plt.imshow(data, cmap='Greys', origin='lower', norm=norm)
    for i in range(len(positions)):
        plt.text(positions[i][0]+10,positions[i][1]+10,str(i+1),fontdict={'size':'50','color':'blue'})
    plt.axis('off')
    plt.savefig(fig_name,dpi=150)
    plt.show()
    
def image_shift0(fbt_data,ref_data,percent_fit=0.5,precision=100):
    min_size=int(len(ref_data)/2*(1-percent_fit));max_size=int(len(ref_data)/2*(1+percent_fit))
    src_image=fbt_data[min_size:max_size,min_size:max_size]
    target_image=ref_data[min_size:max_size,min_size:max_size]
    shift,_,_=register_translation(src_image, target_image, upsample_factor=precision, space='real')
    return np.array([shift[1],shift[0]])

def image_shift1(fbt_data,ref_data,percent_fit=0.2,precision=100):
    min_size=int(len(ref_data)/2*(1-percent_fit));max_size=int(len(ref_data)/2*(1+percent_fit))
    src_image=fbt_data
    target_image=ref_data[min_size:max_size,min_size:max_size]
    result=match_template(src_image,target_image)
    MN=len(src_image)-len(target_image)+1
    shift=np.array([np.argmax(result)//MN,np.argmax(result)%MN])-np.array([min_size,min_size])
    return np.array(shift)

def iis_good_image(data,positions):
    _,_,snr=fine_center(data,positions)
    if -1 in snr:
        return 0
    print('SNR=',snr)
    if np.min(snr) > 0:
        return 1
    else:
        return 0

def is_good_image(data,positions):
    _,_,snr=fine_center(data,positions)
    if -1 in snr:
        return 0
    print('SNR=',snr)
    return 1
    
def aper_phot(data,header,positions,aper,data_out,gain=1,rdnoise=0):
    exptime=header['EXPTIME']
    apertures = CircularAperture(positions, r=aper[0])
    annulus_apertures = CircularAnnulus(positions, r_in=aper[1], r_out=aper[2])
    annulus_masks = annulus_apertures.to_mask(method='center')

    bkg_median = [];
    try:
      for mask in annulus_masks:
        annulus_data = mask.multiply(data)
        annulus_data_1d = annulus_data[mask.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
    except:
      return None
    bkg_intensity = np.array(bkg_median)*apertures.area()*gain
    phot = aperture_photometry(data, apertures)
    phot['signal_intensity']=phot['aperture_sum']*gain - bkg_intensity 
    phot['SNR']=phot['signal_intensity']/(phot['signal_intensity']+bkg_intensity+rdnoise**2)**0.5
    phot['magnitude'] = -2.5*np.log10(phot['signal_intensity']/exptime)+24
    phot['delta_magnitude']= 2.5*np.log10(1+1/phot['SNR'])

    for col in phot.colnames:
        phot[col].info.format = '%.4f'  # for consistent table output

    imageshow(data,positions,aper=aper)
    return phot

def is_nan(array):
    for i in range(len(array)):
        if isnan(array[i]):
            return 1

def utc2bjd(jd_utc, ra, dec, raunits='degrees'):
    """
    [revised from barycorr.py]
    Query the web interface for utc2bjd.pro and compute the barycentric
    Julian Date for each value in jd_utc.

    See also: http://astroutils.astronomy.ohio-state.edu/time/utc2bjd.html
    :param jd_utc: Julian date (UTC)
    :param ra: RA (J2000) [deg/hours]
    :param dec: Dec (J2000) [deg]
    :param raunits: Unit of the RA value: 'degrees' (default) or 'hours'
    :return: BJD(TDB) at ~20 ms accuracy (observer at geocenter)
    """

    # Check if there are multiple values of jd_utc
    if not isinstance(jd_utc, (list, np.ndarray)):
        jd_utc = [jd_utc]

    # Prepare GET parameters
    params = {
        'JDS': ','.join(map(repr, jd_utc)),
        'RA': ra,
        'DEC': dec,
        'RAUNITS': raunits,
        'FUNCTION': 'utc2bjd',
    }

    # Query the web server
    return query_webserver('http://astroutils.astronomy.ohio-state.edu/time/convert.php',params,len(jd_utc))

def query_webserver(server_url, params, expected_length):
    '''
    [revised from barycorr.py]
    '''
    r = requests.get(server_url, params=params)
    # Convert multiline string output to numpy float array
    result = [float(x) for x in r.text.splitlines() if len(x) > 0]
    if expected_length == 1:
        return result[0]
    else:
        return np.array(result)

def split_convert(jd_utc,ra,dec):
    max_num=300;
    if len(jd_utc)<=max_num:
        bjd=utc2bjd(jd_utc,ra,dec)
    else:
        bjd=np.array([])
        for i in range(len(jd_utc)//max_num):
            bjd=np.hstack((bjd,utc2bjd(jd_utc[i*max_num:(i+1)*max_num],ra,dec)))
        bjd=np.hstack((bjd,utc2bjd(jd_utc[(i+1)*max_num:],ra,dec)))
    return bjd

def pro_photo(object_list,ref_image_list,ref_pixel_coo_list,radec_list,gain,rdnoise,aper_enlarge,trim_size,data_in,data_out,pho_all=1):
    images_table=check_images(data_out,bin=1,pho_all=1)                  #检查相关文件flat,bias是不是齐全，并将mubiaow文件信息存入images_table
    master_table,_=combine_images(images_table,trim_size,data_out)       #生成master文件，并存入master_table
    if len(object_list)==len(ref_image_list) and len(object_list)==len(ref_pixel_coo_list):
        data_res_out=os.path.join(data_out,'pho_results');mkdir(data_res_out) #新建存放所有目标结果的目录
        for k in range(len(object_list)):
            object_name=object_list[k];ref_image=ref_image_list[k];ref_coo=np.array(ref_pixel_coo_list[k])-np.array([trim_size[0][0],trim_size[1][0]]);
            data_res_out_object=os.path.join(data_res_out,object_name+'_'+time.strftime("%Y%m%d_%H%M%S", time.localtime()));os.makedirs(data_res_out_object)        #新建存放每个目标结果的目录
            pho_res=pd.DataFrame(columns=('UTC','Mag','EXPTIME','FILTER','File_NAME','HJD','Err','Index','x_center','y_center','aper1','aper2','aper3','SNR'))
            ref_name,ref_header,ref_data=preprocess(ref_image,images_table,master_table,trim_size,data_out);
            ref_coo,ref_fwmh,_=fine_center(ref_data,ref_coo,box_size=10)
            aper=np.median(ref_fwmh)*aper_enlarge;print(object_name,':  FWMH= ',ref_fwmh)
            ref2image(ref_data,ref_coo,aper,fig_name=os.path.join(data_res_out_object,object_name+'.jpg'))
            image_pho_all=list(images_table['File_NAME'][images_table['OBJ_NAME']==object_name])
            filter_pho_all=list(set(images_table['FILTER'][images_table['OBJ_NAME']==object_name]))
            
            for i in range(len(image_pho_all)):
                image_pho=image_pho_all[i];print(image_pho)
                fbt_name,fbt_header,fbt_data=preprocess(image_pho,images_table,master_table,trim_size,data_out);
                targe_coo=ref_coo+image_shift0(fbt_data,ref_data,percent_fit=0.6,precision=100);
                if is_good_image(fbt_data,targe_coo):
                    phot_table=aper_phot(fbt_data,fbt_header,targe_coo,aper=aper,data_out=data_out,gain=gain,rdnoise=rdnoise)
                    try:
                        if is_nan(phot_table['magnitude']):
                            continue
                    except:
                        continue
                else:
                    continue

                if len(phot_table)==len(ref_coo):
                    for j in range(len(phot_table)):
                        pho_res.loc[str(i)+'_'+str(j)]=[\
                            middle_obs_time(fbt_header['DATE-OBS'],fbt_header['EXPTIME'],'isot'),\
                            round(phot_table['magnitude'][j],3),fbt_header['EXPTIME'],fbt_header['FILTER'],\
                            image_pho,middle_obs_time(fbt_header['JD'],fbt_header['EXPTIME'],'jd'),\
                            round(phot_table['delta_magnitude'][j],3),str(j+1),\
                            round(phot_table['xcenter'][j].value,2),round(phot_table['ycenter'][j].value,2),\
                            round(aper[0],1),round(aper[1],1),round(aper[2],1),round(phot_table['SNR'][j],2)]  
            radec = SkyCoord(radec_list[k], unit=(u.hourangle, u.deg))
            try:
                pho_res['JD']=pho_res['HJD']
                pho_res['BJD']=split_convert(np.array(pho_res['JD']),radec.ra.value,radec.dec.value)
                pho_res['HJD']=np.array([helio_jd(jd-2400000,radec.ra.value,radec.dec.value)+2400000 for jd in pho_res['JD']])
            except:
                pass
            for filter_pho in filter_pho_all:
                pho_res[pho_res['FILTER']==filter_pho].to_csv(os.path.join(data_res_out_object,object_name+'_'+filter_pho+str(len(ref_coo))+'.dat'),index=False,header=False,sep=" ")
                pho_res[pho_res['FILTER']==filter_pho].to_csv(os.path.join(data_res_out_object,object_name+'_'+filter_pho+str(len(ref_coo))+'.csv'),index=False,header=True)


#########################################主程序部分#############################################################################################################################
data_in='C:\\Users\\falli\\Desktop\\Project\\Photutils\\data70';os.chdir(data_in)
data_out='C:\\Users\\falli\\Desktop\\Project\\Photutils\\phres';

#----------------------------------------测光目标设置---------------------------------------------------------------------------------------
object_list=['20191018NSVS6550']                       #测光目标
ref_image_list=['20191018NSVS6550-020V.fit']            #对应的参考图
#pixel_coo_star1=[[636,659],[894,338],[604,402],[561,456],[348,876]] #对应的参考图上的目标位置,多个则分别命名为pixel_coo_star2,pixel_coo_star3....
#pixel_coo_star2=[[678,580],[632,530],[624,362],[667,373],[749,350]]
pixel_coo_star3=[[1002,1044],[653,1043],[1348,1375],[1124,986]]
ref_pixel_coo_list=[pixel_coo_star3]  

radec_star1='02 20 50.9 +33 20 46.6'
radec_list=[radec_star1] 
#----------------------------------------测光参数设置---------------------------------------------------------------------------------------
#Set some properties of the instrument
gain=1.5;
rdnoise=8.5;
#Photometric parameters
aper_enlarge=np.array([3,5,6])   #孔径测光参数
trim_size=np.array([[500,1600],[500,1600]]) #裁图尺寸

if __name__ == "__main__":
    pro_photo(object_list,ref_image_list,ref_pixel_coo_list,radec_list,gain,rdnoise,aper_enlarge,trim_size,data_in,data_out,pho_all=1);

'''
pho_all=
    0:只写入image_infs
    1:写入并处理
'''












