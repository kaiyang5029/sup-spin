from copy import deepcopy
import cv2
from cv2 import sqrt
import numpy as np
import os,re,csv
import sys
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
from dataclasses import dataclass
from skimage import data, filters, measure, morphology
import time
from scipy import ndimage,stats
'''
LTEM image analysis. 
First load the image. 
Then apply median filter and gaussian filter to remove noise.
Then apply threshold to get the binary image. 
Then use skimage.measure to get the number of domains and the number of stripe domains. 
Then use the circularity to get the number of stripe domains.
'''

sys.path.append(r"C:\Users\jsche\Documents\GitHub\sup-spin\mumax3")

@dataclass
class InputOptions:
    title: str = r'skyrjmion_counting.txt'
    plot_path: str = r'C:\Users\jsche\Documents\GitHub\sup-spin\mumax3\Stray field resources\skyrmion_outlines'
    extension: str = '.tif'
    median_widow: int = 7
    offset_light : float = -3500
    offset_dark : float = -3500


class Domains(object):
    '''
    # Create Domains object.
    '''
    sk_circularity_threshold_min: float = 0.6
    sk_circularity_threshold_max: float = 0.8
    sk_circularity_threshold_num: int = 10

    def __init__(self,bin_imag : np.ndarray,intensity_image = None) -> None:
        self.bin_image: np.ndarray = bin_imag 
        self.intensity_image= intensity_image
        pass
    @property
    def labels(self):
        if not hasattr(self, '_label'):
            self._label :np.ndarray = measure.label(self.bin_image)
        return self._label

    @property
    def props(self):
        if not hasattr(self, '_props'):
            self._props = measure.regionprops(self.labels,intensity_image= self.intensity_image)
        return self._props

    @property
    def  domains_circularities(self):
        if not hasattr(self, '_domains_circularities'):
            self._domains_circularities = np.array([4 * np.pi * domain.area / domain.perimeter ** 2 for domain in self.props])
        return self._domains_circularities
    @property
    def num_sks(self):
        if not hasattr(self, '_num_sks'):
            sk_circularity_thresholds = np.linspace(self.sk_circularity_threshold_min, self.sk_circularity_threshold_max, self.sk_circularity_threshold_num)
            self._num_sks = [np.count_nonzero(self.domains_circularities > sk_circularity_threshold) for sk_circularity_threshold in sk_circularity_thresholds]
        return self._num_sks

    @property
    def num_sk(self):
        return np.mean(self.num_sks)

    @property
    def num_domains(self):
        if not hasattr(self, '_num_domains'):
            self._num_domains = len(self.props)
        return self._num_domains

    @property
    def num_stripes(self):
        if not hasattr(self, '_num_stripes'):
            self._num_domains = self.num_domains - self.num_sk
        return self._num_domains


class LTEM_image_analysis(object):
    Input = InputOptions()
    sk_circularity_threshold_min: float = 0.7
    sk_circularity_threshold_max: float = 1.5
    sk_circularity_threshold_num: int = 10
    def __init__(self,file) -> None:
        self.load_imag(file)




    def load_imag (self,file :str):
        '''
        load imag from a file pass then extract the field
        '''
        self.LTEM_img = cv2.imread(file,cv2.IMREAD_ANYDEPTH)
        self.file_name =  file.split('/')[-1]
        self.file_path = file.removesuffix(self.file_name)
		# extract field
        re_search = re.search(r'\d+\s*(\+?-?\d+)\s*Oe', self.file_name )
        if re_search:
			# try Oe
            self.field = float(re_search.group(1))
        else:
			# convert from mT to Oe
            re_search = re.search(r'\d+\s*(\+?-?\d+)\s*mT', self.file_name )
            if re_search:
                    self.field = float(re_search.group(1))  * 10.0
            else:
                    # try just a number, and pray that it is in mT
                re_search = re.search(r'\d+\s*(\+?-?\d+)', self.file_name )
                if re_search:
                    self.field = float(re_search.group(1))
                else:
                    print('Failed to read field value from %s. Assuming zero field.'%self.file_name)

        
    def pretreat (self):
        '''
        Pretreat Image. Apply median filter then gaussian filter. then subtract by the mean value to make sure the center pixel is zero intensity.
        '''

        time_start = time.time()
        self.pretreat_img = ndimage.median_filter(self.LTEM_img, size= self.Input.median_widow)
        self.pretreat_img = ndimage.gaussian_filter(self.pretreat_img,5,0)
        self.pretreat_img = np.array(self.pretreat_img,dtype= np.float64)
        self.pretreat_img = self.pretreat_img - np.mean(self.pretreat_img)
        self.plot_no_axis_gray_image(self.pretreat_img ,'inter_check_pretreat_%.2f_Oe.tif'%self.field)
        time_end = time.time()
        print('Pretreat Ends. Cost %f s'%(time_end - time_start))


    def threshold_local(self):
        '''
        Seperate the LTEM image into dark and light parts. Apply isodata filter to binarize the image.
        '''

        time_start = time.time()
        light_imag = deepcopy (self.pretreat_img)
        dark_imag = -1* deepcopy (self.pretreat_img)

        #light_imag[light_imag < 0] = 0
        #dark_imag[dark_imag < 0] = 0

        #print(np.sort(light_imag.ravel())[-1])
        #plt.hist(self.pretreat_img.ravel(), bins=2500)
        #plt.show()
        #print(threshold)
        #threshold = filters.threshold_isodata(light_imag)
        #threshold = np.sort(light_imag.ravel())[-500]/2
        threshold = filters.threshold_local(light_imag, 201, offset= self.Input.offset_light)
        #print(threshold)
        threshold_image = light_imag > threshold 
        threshold_image = morphology.remove_small_objects(threshold_image, 200)
        threshold_image = morphology.remove_small_holes(threshold_image, 200)
        self.Light_domains = Domains(threshold_image,self.pretreat_img)
        self.plot_no_axis_gray_image(threshold_image,'inter_check_light_%.2f_Oe.tif'%self.field)

        #threshold = filters.threshold_isodata(dark_imag)
        #print(threshold)
        threshold = filters.threshold_local(dark_imag, 201, offset= self.Input.offset_dark)
        #threshold = np.sort(dark_imag.ravel())[-500]/2
        threshold_image = dark_imag > threshold 
        threshold_image = morphology.remove_small_objects(threshold_image, 200)
        threshold_image = morphology.remove_small_holes(threshold_image, 200)
        self.dark_domains = Domains(threshold_image,-1* self.pretreat_img)
        time_end = time.time()
        self.plot_no_axis_gray_image(threshold_image,'inter_check_dark_%.2f_Oe.tif'%self.field)
        
        print('Binarized the image. Cost %.2f s'%(time_end -time_start))
    def make_domains_pair(self,ord_y,ord_x,ang_span = np.pi/4, max_dia = 40):
        '''
        Make the light and dark domains into light-dark domain pairs. Calculating the distance between the pairs.
        '''

        #ord_y,ord_x = self.cal_max_corr(self.pretreat_img)
        time_start = time.time()
        
        
        deg = np.arctan2 (ord_y,ord_x)
        
        pairs = []
        print('Tilt direction %f (deg)'%(deg/np.pi*180))
        for index in range(1, self.Light_domains.labels.max()):
            l_y,l_x = self.Light_domains.props[index].centroid_weighted
            index_l = []
            distance_l = []
            for d_index in range(1, self.dark_domains.labels.max()):
                d_y,d_x = self.dark_domains.props[d_index].centroid_weighted
                #print(d_y,d_x)
                ddeg = np.arctan2((d_y - l_y),(d_x - l_x)) - deg
                ddeg = (ddeg + np.pi) % (2*np.pi) - np.pi
                if abs(ddeg) < ang_span :
                    index_l.append(d_index)
                    distance_l.append(np.linalg.norm((d_y - l_y,d_x - l_x)))
            distance_l = np.array(distance_l)

            if len(distance_l) and np.min(distance_l) < max_dia:
                pairs.append((index,index_l[distance_l.argmin()]))
        self.domain_pairs = pairs
        time_end = time.time()
        print('Paired domains. Cost %.2f s'%(time_end -time_start))
    def conv(self):
        '''
        # Not in use
        Idealy will do convolution based on the correlation result. Provide the LTEM image with merged dark-light pairs.
        '''
        self.pretreat()
        ord_y,ord_x = self.cal_max_corr(self.pretreat_img)
        #test# ord_y,ord_x = 60,20
        #print(ord_y,ord_x )
        gauss_kernel = -1*self.gkernel(origin =(ord_y,ord_x )) + self.gkernel(origin = (-1*ord_y,-1*ord_x) )

        time_start = time.time()
        conv_imag = ndimage.convolve(self.pretreat_img,gauss_kernel,mode= 'reflect')
        time_end = time.time()

        print('Convolution finished. Cost %f s'%(time_end - time_start))

        #plt.imshow(conv_imag,'binary')
        #plt.show()

    def cal_max_corr (self,img : np.ndarray, cal_len = 30,reduce_num = 2):
        '''
        Calculate the correlation of the image. Will provide the direction of the tilting.
        '''

        img = deepcopy(img)
        light_imag = deepcopy (img)
        dark_imag = -1* deepcopy (img)
        light_imag[light_imag < 0] = 0
        dark_imag[dark_imag < 0] = 0
        time_start = time.time()
        light_imag = measure.block_reduce(light_imag, (reduce_num,reduce_num), np.mean)
        dark_imag = measure.block_reduce(dark_imag, (reduce_num,reduce_num), np.mean)
        #plt.imshow(img,'binary')
        #plt.show()
        
        corr_arr = np.zeros((2*cal_len,2*cal_len))
        ord_list = []
        for i in np.arange ( -cal_len,cal_len):
            for j in np.arange ( -cal_len,cal_len):
                
                corr = np.sum(np.multiply(dark_imag,np.roll(light_imag,(j,i),axis= (1,0))))
                
                #plt.imshow(np.roll(img2,j,axis= 1))
                #plt.show()
                #print(corr)
                corr_arr[i,j] = corr
                ord_list.append((i,j))
        #self.plot_no_axis_gray_image(corr_arr,'inter_check_corr_%.2f_Oe.tif'%self.field)
        time_end = time.time()
        print('Correlation map constructed. Cost %f s'%(  time_end - time_start))
        #plt.imshow(corr_arr)
        #plt.show()
        
        #print(np.unravel_index(corr_arr.argmin(),(2*cal_len,2*cal_len)))
        corr_y, corr_x = np.unravel_index(corr_arr.argmax(),(2*cal_len,2*cal_len))
        if corr_y >= cal_len:
            corr_y = corr_y - 2*cal_len
        if corr_x >= cal_len:
            corr_x = corr_x - 2*cal_len
        
        return corr_y*reduce_num,corr_x*reduce_num

    def gkernel(self, origin = (0,0), kernlen = 61 , nsig = 10):
        
        """Returns a 2D Gaussian kernel."""

        x = np.linspace(-nsig, nsig, kernlen+1)
        kern1dx = np.diff(stats.norm.cdf(x,loc = origin[1]*nsig/kernlen))
        kern1dy = np.diff(stats.norm.cdf(x,loc = origin[0]*nsig/kernlen))
        kern2d = np.outer(kern1dy, kern1dx)
        return kern2d/kern2d.sum()

    @property
    def pairs_circularities(self):
        if not hasattr(self,'domain_pairs'):
            raise(ValueError,'No Pair defined')

        if not hasattr(self,'_pairs_circularities'):
            self._pairs_circularities = []

            for index in range(len(self.domain_pairs)):
                pair = self.domain_pairs[index]
                l_y,l_x = self.Light_domains.props[pair[0]].centroid_weighted
                d_y,d_x = self.dark_domains.props[pair[1]].centroid_weighted
                pair_distance =max( np.linalg.norm((d_y - l_y,d_x - l_x)) , (self.Light_domains.props[pair[0]].axis_minor_length+self.dark_domains.props[pair[1]].axis_minor_length)/2)
                side_length = (self.Light_domains.props[pair[0]].axis_major_length+self.dark_domains.props[pair[1]].axis_major_length)/2
                self._pairs_circularities.append(side_length/pair_distance)
            self._pairs_circularities = np.array(self._pairs_circularities)
        return self._pairs_circularities

    @property
    def num_sks(self):
        if not hasattr(self, '_num_sks'):
            sk_circularity_thresholds = np.linspace(self.sk_circularity_threshold_min, self.sk_circularity_threshold_max, self.sk_circularity_threshold_num)
            self._num_sks = [np.count_nonzero(np.logical_and((self.pairs_circularities > 1/(1+sk_circularity_threshold)), ( self.pairs_circularities < 1+ sk_circularity_threshold ))) for sk_circularity_threshold in sk_circularity_thresholds]
        return self._num_sks

    @property
    def num_sk(self):
        return np.mean(self.num_sks)
    @property
    def num_sk_std(self):
        return np.std(self.num_sks)
    
    @property
    def num_domains(self):
        if not hasattr(self, '_num_domains'):
            self._num_domains = len(self.domain_pairs)
        return self._num_domains


    def plot_sks_inter_check(self):
        path = self.Input.plot_path
        img = self.LTEM_img
        name = 'inter_check_SKs_counting_%.2f_Oe.tif'%self.field
        isExists=os.path.exists(path)
        if not isExists:
            os.makedirs(path) 
        fig, ax = plt.subplots()
        ax.imshow(img,'gray')
        ax.axis('off')
        sk_circularity_threshold = (self.sk_circularity_threshold_min + self.sk_circularity_threshold_max)/2
        for index, circularity in enumerate(self.pairs_circularities):
            pair = self.domain_pairs[index]
            l_y,l_x = self.Light_domains.props[pair[0]].centroid_weighted
            d_y,d_x = self.dark_domains.props[pair[1]].centroid_weighted
            #ax.scatter(l_x,l_y, c= 'g',marker= '+',s = 1,linewidths = 0.2)
            #ax.scatter(d_x,d_y, c= 'g',marker= '+',s = 1,linewidths = 0.2)
            if (circularity > 1/(1+ sk_circularity_threshold))and ( circularity < 1 + sk_circularity_threshold):
                ax.scatter((l_x+d_x)/2,(l_y+d_y)/2, c= 'r',marker= '+',s = 2,linewidths = 0.2)
            else:
                pass
                #ax.scatter((l_x+d_x)/2,(l_y+d_y)/2, c= 'b',marker= '+',s = 2,linewidths = 0.2)
        plt.savefig(os.path.join(path,name), dpi= 500, facecolor='w', edgecolor='w',
        orientation='portrait',
        transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def plot_no_axis_gray_image(self,img,name):
        path = self.Input.plot_path
        isExists=os.path.exists(path)
        if not isExists:
            os.makedirs(path) 
        fig, ax = plt.subplots()
        ax.imshow(img,'gray')
        ax.axis('off')
        plt.savefig(os.path.join(path,name), dpi= 500, facecolor='w', edgecolor='w',
        orientation='portrait',
        transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def excute(self):
        self.pretreat()
        self.threshold_local()
        ord_y,ord_x = self.cal_max_corr(self.pretreat_img)
        self.make_domains_pair(ord_y,ord_x,max_dia= 35)
        self.plot_sks_inter_check()
        #print(self.pairs_circularities)
        print('Counting finished in field : %.2f Oe. SKs %d in %d.'%(self.field, self.num_sk, self.num_domains))
        print('----------------------------------------------------------')


def save_file_(headers, data, file_name,save_dir):
    
    
    def mkdir(path):

        path=path.strip()
        path=path.rstrip("\\")
    
        isExists=os.path.exists(path)
    
        if not isExists:
            os.makedirs(path) 
            return True
        else:
            return False
    mkdir(save_dir)
    f = open(save_dir+'\\'+file_name,'w',newline="")
    if isinstance(data,dict):
        headers = []
        n = []
        for name in data:
            headers.append(name)
            headers.append('SD')
            n.extend(data[name])
        #print(headers )
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerow(n)
    
    
    else :
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(data)
    
def main():
    ipt = InputOptions()
    root = tk.Tk()
    input_files = filedialog.askopenfiles(title='Select files')
		# close the Tkinter window
    root.destroy()
    if not input_files:
        print('no selected')
        return None

    save_data = []

    if not os.path.exists(ipt.plot_path):
        os.makedirs(ipt.plot_path)

    for file in input_files:
        a = LTEM_image_analysis(file.name)
        a.excute()
        save_data.append([a.field,a.num_sk,a.num_sk_std,a.num_domains])
    save_data.sort(key= lambda x: x[0])
    save_file_(['name','num_sks','num_sks_std','num_domains'],save_data,time.strftime("%H%M%S", time.localtime())+ipt.title,ipt.plot_path)

if __name__ == '__main__':
    main()

