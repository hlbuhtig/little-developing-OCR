import os,cv2
import numpy as np
def wh1(r,i):
    i1=r.find('>',i)
    i2=r.find('<',i1+1)
    return r[i1+1:i2],i2+1

def wh2(r,i):
    i1=r.find('"',i)
    i2=r.find('"',i1+1)
    return r[i1+1:i2],i2+1

def getlist(dirr,make_mask=0):
    f=open(dirr+'/segmentation.xml','r',encoding='UTF-8')
    
    r=f.read()
    
    '''
    q=0
    while(1):
        q+=1
        #print(q)
        r=f.readline()
        if(q>8000):
            print(q,r)
        #print(q)
    '''
    di={'d':[],'f':[],'m':[],'t':[]}
    #picture path, segmentations, mask path, segmentation record number

    i=0
    while(1):
        i=r.find('<imageName>',i)
        
        if(i<0):
            break
        
        name,i=wh1(r,i)
        di['d'].append(dirr+name)
        di['m'].append(dirr+name+'_mask.jpg')
        #print(dirr+name)
        i_copy=i

        ii=r.find('<imageName>',i)
        if(ii<0):
            ii=len(r)
        fr=len(di['f'])
        while(1):
            i=r.find('<taggedRectangle ',i)
            if(i<0 or i>ii):
                break
            x,i=wh2(r,i)
            y,i=wh2(r,i)
            w,i=wh2(r,i)
            h,i=wh2(r,i)
            rtg=[float(x),float(y),float(w),float(h)]
            rtg= [int(x) for x in rtg]
            di['f'].append(rtg)
        to=len(di['f'])
        di['t'].append([fr,to])
        #print(to-fr)
        i=i_copy
        
        if(make_mask):
            #print(name)
            rx,ii=wh2(r,i)
            ry,ii=wh2(r,ii)
            rx=int(rx)
            ry=int(ry)
            mm=np.zeros((rx,ry))
            for q in range(fr,to):
                #print(q,di['f'][q])
                w=di['f'][q][2]
                h=di['f'][q][3]
                #print(w,h)
                for w in range(w):
                    
                    for e in range(h):
                        x=di['f'][q][0]+w
                        y=di['f'][q][1]+e
                        if(x>rx-1):
                            x=rx-1
                        if(y>ry-1):
                            y=ry-1
                        mm[x][y]=255
            
            cv2.imwrite(dirr+name+'_mask.jpg',mm.T)
            

    print(len(di['d']),len(di['f']),len(di['t']))
    
    f.close()
    return di

if __name__ == '__main__':
    dirr='../../data/'
    make_mask=0
    getlist(dirr+'/SceneTrialTrain/',make_mask)
    getlist(dirr+'/SceneTrialTest/',make_mask)



