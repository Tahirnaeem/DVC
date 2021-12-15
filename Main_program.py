import cv2 as cv
import os,sys
import tkinter as tk
from threading import Thread
import threading
from time import sleep

import numpy as np
import torch
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from torchvision import datasets, models, transforms
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from Crypto.Cipher import AES
import torch.nn.functional as F1
import io 
from tkinter import messagebox

import copy
from contextlib import suppress



 
global std_w, std_h, img,window_loop,faces
canvas_width = 600
canvas_height =480

img_width = 600
img_height =480
std_h=  768   #135x63
std_w= 1024

starting_scale= 0.8
scale_interval= 0.2

mean=[0.485, 0.456, 0.406]                              
std=[0.229, 0.224, 0.225]

size_limit = 2000000
size_lock =0
 


#This creates the main window of an application
window = tk.Tk()
window.title("DeepVisualCounter v0.1")
window.geometry("700x500")

application_path=None
# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)

print("current path ", application_path)

window.iconbitmap(application_path+'/Untitled.ico')



canvas = Canvas(window, 
          width=canvas_width, 
          height=canvas_height  ) #(0,0,300,500),scrollregion=canvas.bbox("all")



###############################################################################################
def get_file():
   global img
   global std_w, std_h
   file = filedialog.askopenfile(parent=window,mode='rb',title='Choose a file' ,filetypes = (("jpeg files","*.jpg"),("image files","*.jpg")))
   if not file:
     return 0
   if os.stat(file.name).st_size > size_limit and size_lock == 1 :
     print( "file size exceeding ", os.stat(file.name).st_size ) 
     messagebox.showinfo("Warning" , "Sorry, This file is too large !  This version only supports files with size upto "+repr(int(size_limit/1000000)) +" MB. Contact dvc@tahirnaeem.net if you want to process larger files. ")
     return 0
   #print (file.name )
   #print( "file size exceeding ", os.stat(file.name).st_size ) 

   display_wait( " Loading image.... ","green",0)
   #print("displayed white image")
   
   detectAndDisplay(file)
 
def findXCenter(canvas, item):  
  coords = canvas.bbox(item)
  xOffset = (350 / 2) - ((coords[2] - coords[0]) / 2)
  return xOffset

def donothing():
   filewin = Toplevel(window)
   button = Button(filewin, text="Do nothing button")
   button.pack()

def about():
   About = Toplevel(window)
   About.iconbitmap(application_path+'/Untitled.ico')
   About.resizable(0,0)

   
   canvas = Canvas(About,      width=350,        height=80  ) 
   textID= canvas.create_text(0,0,fill="black" ,font="Helvetica 8 ", anchor="nw",
    text="\nCopy Rights 2019 Tahir Naeem \ncheck www.tahirnaeem.net for updates \nQuestions/Feedback : dvc@tahirnaeem.net   " ) #len(faces)
   xOffset = findXCenter(canvas, textID)
   canvas.move(textID, xOffset, 0)


   canvas.pack()
 

#################################################################################################
# Model 

class CNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]        
        self.frontend  = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self,x):
        x = self.frontend(x)
        x = F1.relu(self.backend(x))
        x = F1.relu(self.output_layer(x)) 
        return x            
                
def make_layers(cfg, in_channels = 3,ks=3, batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2,dilation=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=ks, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            #print (layers)
    return nn.Sequential(*layers)          
    
################################################################################################
menubar = Menu(window)
filemenu = Menu(menubar, tearoff=0)
#filemenu.add_command(label="New", command=donothing)
filemenu.add_command(label="Open", command=get_file)
#filemenu.add_command(label="Save", command=donothing)
#filemenu.add_command(label="Save as...", command=donothing)
#filemenu.add_command(label="Close", command=donothing)

menubar.add_cascade(label="File", menu=filemenu)


helpmenu = Menu(menubar, tearoff=0)
#helpmenu.add_command(label="Help Index", command=donothing)
helpmenu.add_command(label="About...", command=about)
menubar.add_cascade(label="Help", menu=helpmenu)

window.config(menu=menubar)


window.configure(background='black')

hbar=Scrollbar(window,orient=HORIZONTAL)
hbar.pack(side=BOTTOM,fill=X)
hbar.config(command=canvas.xview)
vbar=Scrollbar(window,orient=VERTICAL)
vbar.pack(side=RIGHT,fill=Y)
vbar.config(command=canvas.yview)

window_loop=1

def openModel():      

  key = 'test-key' # 'd123s567z9abcdef'
  IV = 16 * '\x00'         # Initialization vector: discussed later
  mode = AES.MODE_CFB
  encryptor = AES.new(key, mode, IV=IV)
  


  final_file = open(application_path+"/params.pt", "rb")
  xen_text = final_file.read()
  plaintext = encryptor.decrypt(xen_text)
  plaintext = plaintext[16:]
  final_file.close()

  #end = open("DC/hello.pt", "wb")
  #end.write(plaintext)
  #end.close()

  return plaintext

def closeModel(xen_text):
  end = open("DC/hello.pt", "wb")
  end.write(xen_text)
  end.close()


def load_model():
  global dl_completion
  try:
    model = CNet()
    
  except:
      print("Memory error... Cant load Program")
      display_wait( "Memory error... Cant load Program","red",0)
  try:
  


    print("load starting.." )
    plaintext= openModel()
    
    buffer = io.BytesIO(plaintext)
    
    state_dict = torch.load( buffer , map_location=lambda storage, loc: storage)

    print("state starting..")
    model.load_state_dict(state_dict)
    #print(type(plaintext))
    #model.load_state_dict(plaintext)
    print("loading ok")
    model.eval()

    #closeModel(xen_text)

  except:
      print("missing files in installation \n or no space left in drive \n Or program has no write permissions..\n please re download the program")
      display_wait( "missing files in installation \n or no space left in drive\n Or program has no write permissions.. \n please re download the program","red",0)
      dl_completion=-1
      return 0
  return model 

 

 
 
def load_image( infilename ) :
    #img = Image.open( infilename )
    #img.load()
    #data = np.asarray( img, dtype="uint8" )
    data =plt.imread(infilename)
    
    return data


def rescale_image(im, resize_ratio):    
    h1,w1 = int(im.shape[0]*resize_ratio), int(im.shape[1]*resize_ratio)
    if h1 ==0 or w1 ==0:
      print ("error while resizing ",im.size, h1,w1,resize_ratio)
    rszd_img = cv.resize(im, dsize=( int(round(w1)), int(round(h1))), interpolation=cv.INTER_AREA   )  
    return rszd_img 
  
# takes y, x, h , w returns coordinates inform of x, y, x_lower,  y_lower

def devideImage(img , std_h,std_w):
    #print(img.ndim, img.shape)
    height, width=img.shape[0], img.shape[1]
    c=0
    if img.ndim == 3:
      c=3
    else:
      c=1
    #img= to_rgb2(img)
    outlist =[]
    n_yloop= math.ceil( height/std_h)
    n_xloop= math.ceil( width/std_w)
    #print("image to devide:", img.shape,n_yloop, n_xloop)

    y_start ,y_end,x_start, x_end= (0,)*4
    y_end=std_h
    x_end=std_w

    x_short=0
    y_short=0
    
    ipart_counter=0
    Lbl_coordinates_Boundry ={}

    for x_index  in range(0,n_xloop):
        for y_index in range(0,n_yloop):
          if c ==3:
            new_img=np.zeros((std_h, std_w, c), dtype='uint8')
          else:
            new_img=np.zeros((std_h, std_w ), dtype='uint8')
          #print( "sss",c,x_short )
          img_part=img[y_start:y_end,x_start: x_end] 
          
          Lbl_coordinates_Boundry[ipart_counter ] =  (y_start,y_end,x_start,x_end)  
          ipart_counter =ipart_counter+1

          
          y_real= std_h-y_short
          x_real= std_w-x_short
          th,tw= img_part.shape[0],img_part.shape[1]

          if y_real > th:
            #print("y_real,th :",y_real , th)
            y_real=th

          if x_real > tw:
            #print("y_real,th :",x_real , tw)        
            x_real=tw    

          new_img[0:y_real, 0:x_real]=  img_part
          outlist.append(new_img)
          #print(new_img.shape)
          #x_short=0
          y_short=0

          if y_end == height or y_end > height:

                #print("column change")
                y_start=0
                y_end=std_h
                if x_start+std_w < width :
                  x_start =x_start+ std_w

                if x_end+std_w < width :
                  x_end =x_end+ std_w
                else:
                  x_short= std_w- (width-x_end) 
                  x_end=width 

          else:

            if y_start + std_h < height: # normal case
              y_start=y_start+ std_h

            if y_end + std_h > height:
              #print("dddd",y_end, y_short)
              y_short=std_h- (height-y_end) 
              y_end = height


            else: # normal case
              y_end= y_end + std_h

    return outlist,Lbl_coordinates_Boundry #(y_start,y_end,x_start,x_end)



def to_rgb2(im):
    # as 1, but we use broadcasting in one line
    h, w = im.shape[0], im.shape[1]
    ret = np.empty((h, w, 3), dtype=np.uint8)
    ret[:, :, :] = im[:, :, np.newaxis]
    return ret          


def normalize_image(im):
  im=F.to_tensor(im)
  im= F.normalize(im, mean,std)
  return im


def get_pred(img_arr, std_h,std_w):
  global model 
  
  scale= starting_scale
  interval = scale_interval
  max_count=0
  res=0
  
  final_outputMaps=[]
  final_scale_dict={}
  final_scale=1
  total_scale_rounds= round((1-starting_scale)/interval, 0 )+1
  scale_complete= 0

  for i in range(0,50):
    img_copy= copy.deepcopy(img_arr) 
    rs_im= rescale_image(img_copy,scale)
    #print(" prcoessing scale : ", scale )
    output_list=[]
    temp_max_count=0
    scale_complete= i/total_scale_rounds

    

    #print("total Rounds",total_scale_rounds,(1-scale)/interval)
   # print ( rs_im.shape[0] , std_h , rs_im.shape[1] , std_w )
    if rs_im.shape[0] > std_h or rs_im.shape[1] > std_w: # image or its this scale is greater than std size so devide it and process
      input_list,coord_dict= devideImage(rs_im , std_h,std_w) #(y_start,y_end,x_start,x_end)
      #print("input list: ", input_list[0].shape, input_list[1].shape)
      print(scale, coord_dict)
    
      output_list=[]
      
      #print(" deviding : ", scale,len(input_list ))
      for k in range(0,len(input_list)):
        #print(" processing part # : ", k, len(input_list) , rs_im.shape ,     )
        y_start,y_end,x_start,x_end= coord_dict[k]
        img = input_list[k]
        img= img[0:(y_end-y_start), 0:(x_end-x_start)]

        input_tensor= normalize_image(input_list[k])
        input_tensor= torch.unsqueeze(input_tensor,0)
        try:
          output= model(input_tensor)
          #_,_,h,w=input_tensor.size()
          #output= torch.randn( 1,1,int(h)/8, int(w)/8)    #
       
      
        except:
          display_wait( "Not enough RAM available.. ","red",0)
          dl_completion=-1
          return 0 

        output_np= output[0].cpu().transpose( 0,2).transpose( 0,1).data.numpy()
        #print(" processing part # : ", k, len(input_list) , img.shape , output_np.shape,coord_dict[k]    )
        output_list.append(output_np)
        output_sum = torch.sum(output.data.cpu()).item()
        del output, input_tensor
        
        temp_max_count +=output_sum

        
        patch_complete= (k+1)/len(input_list)       

        complete = 100*((scale_complete+ (patch_complete/total_scale_rounds))  )
        
        #print(patch_complete, scale_complete,total_scale_rounds,i )
        display_wait( "Analysing Image...."+repr(int(complete))+" % Complete","green",0)
        #print(" processing part # : ",k,  temp_max_count ,coord_dict[i] )
      #print("output ",scale, temp_max_count )
      if temp_max_count > max_count:  
        max_count=temp_max_count
        final_outputMaps= output_list
        final_scale_dict= coord_dict
        final_scale=scale
        #print("saving parts scale :", scale)
        #print("saving parts dict :", coord_dict)
        #print("maxcount: ",max_count)

    else:  # image or its this scale is not that large so process it at once. 
      #print(" prcessing whole img at scale  : ", scale, max_count )
      input_tensor= normalize_image(rs_im)
      input_tensor= torch.unsqueeze(input_tensor,0)
      #_,_,h,w=input_tensor.size()
      output= model(input_tensor) #torch.randn( 1,1,int(h)/8, int(w)/8)    
      
      output_np= output[0].cpu().transpose( 0,2).transpose( 0,1).data.numpy()
      output_sum = torch.sum(output.data.cpu()).item()
      temp_max_count +=output_sum
      #print(" prcessing whole img at scale  : ", scale )

      if temp_max_count > max_count:
        max_count=temp_max_count
        final_outputMaps=[]
        final_outputMaps.append(output_np)
        #print("max output size ; ",output_np.shape, scale ,temp_max_count,rs_im.shape)
        final_scale_dict= {}
        final_scale=scale   
        #print("saving larger scale :", scale)
        #print("saving larger dict :", coord_dict)

    scale +=interval
    if rs_im.shape[0] < std_h and rs_im.shape[1] < std_w:
      display_wait( "Analysing Image...."+repr(int(scale_complete*100))+" % Complete","green",0)
    if scale > 1.09: # if large than oringal size or a bit larger, stop
      break
  if max_count <1:
    return img_arr,int(max_count)

 # predictions with max count complete... now construct output image:
  #print(" processing complete maxcount  : ", scale ) 
  result_img_copy= copy.deepcopy(img_arr) 
  #print("final scale before recalc", final_scale)
  final_scale=1/final_scale #(1-final_scale)+1

  #print("final scale after recalc", final_scale)

  img_total_scale= (final_scale)*8 # only to be used with coordinates coming out of small patches
  rest=0
  if len(final_scale_dict) > 0: # it was devided image
    
    for i in range(0, len(final_outputMaps)):
      #rescaled_map = rescale_image(final_outputMaps[i], 1/final_scale)
      #print("#############################",i,len(final_outputMaps) )
      #img_total_scale= (1/final_scale)*8
      output_np=final_outputMaps[i] # to bring them in RGB scale
      output_np= np.squeeze(output_np)
      y_start,y_end,x_start,x_end= final_scale_dict[i]
      #print( output_np.shape, final_scale_dict[i] )
      color= None
      if i == 0:
        color= (3,3,3)
      if i == 1:
        color= (253,0,0)

      if i == 2:
        color= (0,253,0)
        
      if i == 3:
        color= (0,0,245) 

      
      #cv.rectangle(result_img_copy,(int(x_start*final_scale), int(y_start*final_scale)),( int(x_end*final_scale), int(y_end*final_scale)),color,3)
      #font = cv.FONT_HERSHEY_SIMPLEX
      #cv.putText(result_img_copy, repr(i), (int(x_start*final_scale), int(y_start*final_scale)) , font, 3, (255, 0, 0), 2, cv.LINE_AA)

      #cv.rectangle(result_img_copy,(int(x_start), int(y_start)),( int(x_end), int(y_end)),color,3)
      #print("printing part ",i,final_scale, img_total_scale,x_start,y_start,x_end,y_end)
      color= (0,0,245)

      for ix,iy in np.ndindex(output_np.shape):
          #print(output_np[ix,iy], int(output_np[ix,iy]*100))
          if output_np[ix,iy] > 0.01:
            a=1
            cv.circle(result_img_copy,(int(round(x_start*final_scale,0))+ int(round(iy*img_total_scale,0)),int(round((y_start*final_scale),0))+ int(round(ix*img_total_scale,0))), int(5+10*output_np[ix,iy]), color, -1)
          else:
            rest +=output_np[ix,iy]
  else:
    rest=0
    #print("--------------------------------------",final_scale)
    color= (0,0,245)
    img_total_scale= (final_scale)*8
    output_np=final_outputMaps[0] # to bring them in RGB scale
    output_np= np.squeeze(output_np)
    #print("final scale",final_scale  )
    for ix,iy in np.ndindex(output_np.shape):
        #print(output_np[ix,iy], int(output_np[ix,iy]*100))
        if output_np[ix,iy] > 0.01:
          #print(iy,img_total_scale,iy*img_total_scale,output_np.shape  )
          cv.circle(result_img_copy,( int(iy*img_total_scale),int(ix*img_total_scale)), int(5+10*output_np[ix,iy]), color, -1)
        else:
          rest +=output_np[ix,iy]

 
  img_copy= copy.deepcopy(img_arr) 
  result_image = cv.addWeighted(img_copy,0.3,result_img_copy,0.7,0)

  return result_image,int(max_count)
 

def opencvDetect(img):
  global faces
  fname= application_path+'/haarcascade_frontalface_alt2.xml'
  if not os.path.isfile(fname):
    print("haar file doesnt exist")
    print ( os.getcwd() )
  #print(img.shape)

  face_cascade = cv.CascadeClassifier(fname)
  #print("obj ", face_cascade)
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  #return faces

def dlDetect(img, std_h, std_w):


  global  model ,result_image, dl_completion, max_count

  model= load_model()
  if dl_completion ==-1:
    return -1  
  #try:
  result_image,max_count = get_pred(img, std_h,std_w) 
  #except Exception as e:
  #  display_wait("sorry, an error occured while analysing the image ....","red",0)

  dl_completion=1
  #return body_pt_list
 
 

def add_faces(img, faces, color_arr):
  #print(faces)
  for (x,y,w,h) in faces:
      #print(x,y,w,h,fcounter)        
      img= cv.rectangle(img, (x,y) ,( x+w,y+h   ),  color_arr, 2)
      #print("adding face")
  return img



def detectAndDisplay(file):
  global model,faces,dl_completion,im, max_count,result_image
  im=None
  try:
    im=load_image(file)
  except:
    display_wait( "Cant load image.. bad file perhaps","red",0)
    print("Cant load image.. bad file perhaps")
    return 0
  img=  copy.deepcopy(im) 
  img2=  copy.deepcopy(im) 
  dl_completion=0

  display_wait( "Analysing the image, please wait.....","green",0)

  faces_thread = Thread(target = opencvDetect, args = (img, ))
  dl_thread = Thread(target = dlDetect, args = (img2, std_h, std_w, ))
  faces_thread.daemon = True  
  dl_thread.daemon = True 
  faces_thread.start()
  dl_thread.start()
 
  #faces=  opencvDetect(img)  
  #pts_list= dlDetect(im, std_h, std_w)
  
  
  faces_thread.join()
  #print("face detection completed......",im.shape)
  add_faces(img, faces, (255,0,0))

  
  while dl_completion ==0:
    sleep(1) 
    try:
      window.update()
      print("....... ")
    except:
      print("ooops.. ok to go now")
      sys.exit()

    

  dl_thread.join()
  if dl_completion == -1:
    return 0 

  #print("DL detection completed......")   
  result_image = cv.addWeighted(img,0.3,result_image,0.7,0)
  #print("detection addition completed......", im.shape)
  canvas=display_img(result_image,1, max_count)

def addsliders():
  
  slider_canvas = Canvas(canvas, 
          width=80, 
          height=400 )           

  hbar=Scrollbar(window,orient=HORIZONTAL)
  hbar.pack(side=BOTTOM,fill=X)
  hbar.config(command=canvas.xview)
  vbar=Scrollbar(window,orient=VERTICAL)
  vbar.pack(side=RIGHT,fill=Y)
  vbar.config(command=canvas.yview)

  canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
  canvas.pack(side=TOP,expand=True,fill=BOTH)
  slider_canvas.pack(expand=True, side=RIGHT, anchor="ne")


  scale_slider = Scale(slider_canvas, from_=-40, to=42, label="Scale       " ,command=scale_update , width=20 , length=200 )
  sens_slider = Scale(slider_canvas, from_=-40, to=42 ,label="Sensitivity", command=sens_update, width=20 , length=200 )
  button1 = Button (slider_canvas, text ="Update", command = UpdateImage)


def display_wait(text_msg=" Waiting.... ", text_color="green",  loopflag=0):
  #addsliders()
  

  img = np.zeros([window.winfo_height(),window.winfo_width(),3],dtype=np.uint8)
  img.fill(190) # or img[:] = 255

  img= Image.fromarray(img)
  ph = ImageTk.PhotoImage(img)  
  #window.configure(background='grey')

  canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
  canvas.pack(side=TOP,expand=True,fill=BOTH)
  canvas.delete("all")
  canvas.configure(scrollregion = (0,0,img.width,img.height) )
 
  canvas.create_image(20,20, anchor=NW, image=ph)
 

  textID=canvas.create_text(int(canvas_width/2),int(canvas_height/2),fill=text_color , anchor="nw" ,font="Helvetica 14 italic bold",text=text_msg ) #len(faces)
  xOffset = findXCenter(canvas, textID)
  canvas.move(textID, xOffset, 0)

  print("Canv waiting.. ")
  ctr= 50
  while (loopflag >ctr and loopflag ==0) or loopflag==1:
    window.update()
    #print("update...")
    if ctr > -10:
      ctr -=1
  print("Canv update ended.. ")
  #window.mainloop()
  return canvas


def display_img(img, loopflag, max_count):
  #addsliders()
  img= Image.fromarray(img)
  ph = ImageTk.PhotoImage(img)  
  #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
  #img = ImageTk.PhotoImage(Image.open(path))

  canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
  canvas.pack(side=TOP,expand=True,fill=BOTH)


  #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
  #panel = tk.Label(window, image = ph)
  canvas.delete("all")
  
  canvas.configure(scrollregion = (0,0,img.width,img.height) )
 
  canvas.create_image(20,20, anchor=NW, image=ph)
  #slider_canvas.create_image(400,20, anchor=NW, image=ph)
  #canvas.create_rectangle(45,12, 70, 30, outline='green')
  canvas.create_text(120,12,fill="blue",font="Helvetica 10 italic bold",text="Total persons Count: "+repr(max_count)) #len(faces)
  canvas.create_text(240,12,fill="red",font="Helvetica 10 italic bold",text=", Faces: "+repr(len(faces))) #
  #scale_slider.pack()  
  #sens_slider.pack()
  #button1.pack()
  #canvas.pack(side=TOP,expand=True,fill=BOTH)
  print("Canv update starting.. ")
  ctr= 50
  while (loopflag >ctr and loopflag ==0) or loopflag==1:
    
    try:
        window.update()
    except TclError:
        #messagebox.showinfo('Info', 'Application terminated')
        print("ok to quit, dont worry.")
        return

    #print("update...")
    if ctr > -10:
      ctr -=1
  print("Canv update ended.. ")
  #window.mainloop()
  return canvas
  ##








#panel.place(relheight=.095,relwidth=0.25,relx=0.7,rely=0.03)

#The Pack geometry manager packs widgets in rows or columns.
#panel.pack(side = "left", fill = "both", expand = "yes")

#Start the GUI

 
window.mainloop()



